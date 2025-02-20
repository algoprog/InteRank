import os
import re
import json
import torch
import time
import concurrent.futures
from typing import List, Tuple

CACHE_PATH = 'hf_cache'
os.environ['HF_TOKEN'] = 'your-hf-token'
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH
os.environ['HF_HOME'] = CACHE_PATH
os.environ['HF_DATASETS_CACHE'] = CACHE_PATH
os.environ['TORCH_HOME'] = CACHE_PATH

os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

from transformers import StoppingCriteria, StoppingCriteriaList
from torch import LongTensor, FloatTensor

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from openai import OpenAI

DEEPINFRA_CLIENT = OpenAI(api_key="your-api-key", 
                          base_url="https://api.deepinfra.com/v1/openai")
OPENAI_CLIENT = OpenAI(api_key="your-api-key")


def llm_api(prompt, model="meta-llama/Llama-3.3-70B-Instruct-Turbo", client=DEEPINFRA_CLIENT, temperature=0.0, max_retries=2):
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=16384,
                temperature=temperature,
                messages=[{'role': 'user', 'content': prompt}])
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in {1} seconds...")
            time.sleep(1)
    return None

QUERY_PROMPT_TEMPLATE = "Based on the following question, generate a list of up to 10 search queries that would help find answers. Each query should be specific and focused on different aspects of the question. Format the output as one query per line. Question: {question}"

RELEVANCE_PROMPT_TEMPLATE = "Explain whether the following document is relevant or not to the given question.{relevance} Then end your response with a relevance label (0: irrelevant, 1: partially relevant, 2: relevant) and the symbol '##'. Question: {question}\nDocument: {document}"
RELEVANCE_PROMPT_TEMPLATE_API = "Explain whether the following document is relevant or not to the given question.{relevance} Then end your response with a relevance label (0: irrelevant, 1: partially relevant, 2: relevant) and the symbol '##'. Your response should have this format: '<explanation>\nRelevance Label: <label> ##'. Question: {question}\nDocument: {document}"
RELEVANCE_LABEL_ONLY_TEMPLATE = "Rate the relevance of the following document to the given question using these labels (0: irrelevant, 1: partially relevant, 2: relevant).{relevance} Your response should have this format: 'Relevance Label: <label> ##'. Question: {question}\nDocument: {document}"

class RewardWeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute r³ * -log p(response|prompt) for each example
        Higher rewards mean we want to maximize p(response) more strongly
        """
        # Get reward from inputs
        if isinstance(inputs, dict):
            rewards = inputs.pop('reward')
        else:
            rewards = inputs.data.pop('reward')
        
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=model.device)
        
        # Calculate r³ scaling factor [batch_size]
        reward_scale = torch.pow(rewards, 3)
        
        # Forward pass to get logits
        outputs = model(**inputs)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
            if logits is None:
                raise ValueError("The model did not return logits")
        else:
            logits = outputs[0]
        
        # Get logits and shift them
        logits = logits[:, :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
        labels = inputs['labels'][:, 1:].contiguous()    # [batch_size, seq_len-1]
        
        # Calculate loss per example
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))  # [batch_size * seq_len]
        
        # Reshape and mask losses
        losses = losses.view(labels.size(0), -1)  # [batch_size, seq_len]
        mask = (labels != -100).float()
        losses = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [batch_size]
        
        # Apply reward scaling and take mean, accounting for batch size
        weighted_loss = (reward_scale * losses)
        if self.args.average_tokens_across_devices:
            # If averaging across devices, sum losses and divide by total number of items
            weighted_loss = weighted_loss.sum() / num_items_in_batch if num_items_in_batch else weighted_loss.mean()
        else:
            # Otherwise just take mean per device
            weighted_loss = weighted_loss.mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss

class InstructRank:
    def __init__(self, base_model="meta-llama/Llama-3.1-8B-Instruct", new_model=None, mode="train", explain=True, relevance=''):
        self.base_model = base_model
        self.new_model = new_model
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.mode = mode
        self.explain = explain  # New flag to control explanation generation
        self.relevance = ' ' + relevance

        if base_model != 'api':
            self.load_model()
            self.load_tokenizer()

    def load_dataset(self, file_path, max_entries=None, max_length=512):
        if self.mode == "train_rewarded":
            print("loading reward dataset...")
            examples = []
            rewards = []
            
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if max_entries and len(examples) >= max_entries:
                        break
                        
                    try:
                        d = json.loads(line)
                        # Normalize reward to [0, 1] range if it isn't already
                        reward = float(d['reward'])
                            
                        # Create chat messages
                        messages = [{
                            'role': 'user',
                            'content': d['prompt']
                        }, {
                            'role': 'assistant',
                            'content': d['response']
                        }]
                        
                        # Tokenize the text with padding and truncation
                        encoded = self.tokenizer(
                            self.tokenizer.apply_chat_template(messages, tokenize=False) + self.tokenizer.eos_token,
                            max_length=max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        )
                        
                        # Add labels for computing loss properly
                        labels = encoded['input_ids'].clone()
                        # Mask prompt tokens with -100
                        prompt_len = len(self.tokenizer(messages[0]['content'])['input_ids'])
                        labels[0, :prompt_len] = -100
                        
                        examples.append({
                            'input_ids': encoded['input_ids'][0],
                            'attention_mask': encoded['attention_mask'][0],
                            'labels': labels[0]
                        })
                        rewards.append(reward)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
            
            dataset = Dataset.from_dict({
                'input_ids': [ex['input_ids'] for ex in examples],
                'attention_mask': [ex['attention_mask'] for ex in examples],
                'labels': [ex['labels'] for ex in examples],
                'reward': rewards
            })
            
            print(f"reward dataset loaded with {len(dataset)} examples")
            print(f"reward stats - min: {min(rewards):.3f}, max: {max(rewards):.3f}, mean: {sum(rewards)/len(rewards):.3f}")
            return dataset
        else:
            print("loading dataset...")
            examples = []
            entries_processed = 0

            def truncate_text(text, max_words=4000):
                words = text.split()
                return ' '.join(words[:max_words])

            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if max_entries and entries_processed >= max_entries:
                        break
                    
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {line_num}: {e}")
                        continue
                    
                    # Skip if required fields are missing
                    if not all(key in d.get("question", {}) for key in ["title", "body"]):
                        print(f"Missing required question fields on line {line_num}")
                        continue

                    if "generated_queries" not in d:
                        print(f"Missing generated_queries on line {line_num}")
                        continue

                    entries_processed += 1
                    
                    # Create query generation example - enumerate queries
                    query_messages = [{
                        'role': 'user', 
                        'content': QUERY_PROMPT_TEMPLATE.format(
                            question=d["question"]["title"] + " " + d["question"]["body"]
                        )}, 
                        {'role': 'assistant', 
                         'content': "\n".join(f"{i+1}. {query}" for i, query in enumerate(d["generated_queries"])) + " ##"}
                    ]
                    
                    # Tokenize the text
                    query_text = self.tokenizer.apply_chat_template(query_messages, tokenize=False) + self.tokenizer.eos_token
                    encoded = self.tokenizer(
                        query_text,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )
                    examples.append({
                        'input_ids': encoded['input_ids'][0],
                        'attention_mask': encoded['attention_mask'][0],
                        'labels': encoded['input_ids'][0].clone()
                    })
                    
                    # Create relevance examples for each link
                    for link in d.get("links", []):
                        if "label" in link:
                            relevance_messages = [{
                                'role': 'user',
                                'content': (RELEVANCE_PROMPT_TEMPLATE if self.explain else RELEVANCE_LABEL_ONLY_TEMPLATE).format(
                                    question=d["question"]["title"] + " " + d["question"]["body"],
                                    document=truncate_text(link["text"])
                                )},
                                {'role': 'assistant',
                                 'content': (f"{link['explanation']}\n\nRelevance Label: {link['label']} ##" 
                                           if self.explain and 'explanation' in link
                                           else f"Relevance Label: {link['label']} ##")}
                            ]
                            
                            # Tokenize the text
                            relevance_text = self.tokenizer.apply_chat_template(relevance_messages, tokenize=False) + self.tokenizer.eos_token
                            encoded = self.tokenizer(
                                relevance_text,
                                max_length=max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt"
                            )
                            examples.append({
                                'input_ids': encoded['input_ids'][0],
                                'attention_mask': encoded['attention_mask'][0],
                                'labels': encoded['input_ids'][0].clone()
                            })
                    
                    # Handle other_document similarly
                    if "other_document" in d and "label" in d["other_document"]:
                        relevance_messages = [{
                            'role': 'user',
                            'content': (RELEVANCE_PROMPT_TEMPLATE if self.explain else RELEVANCE_LABEL_ONLY_TEMPLATE).format(
                                question=d["question"]["title"] + " " + d["question"]["body"],
                                document=truncate_text(d["other_document"]["text"])
                            )},
                            {'role': 'assistant',
                             'content': (f"{d['other_document']['explanation']}\n\nRelevance Label: {d['other_document']['label']} ##"
                                       if self.explain and 'explanation' in d["other_document"]
                                       else f"Relevance Label: {d['other_document']['label']} ##")}
                        ]
                        
                        # Tokenize the text
                        relevance_text = self.tokenizer.apply_chat_template(relevance_messages, tokenize=False) + self.tokenizer.eos_token
                        encoded = self.tokenizer(
                            relevance_text,
                            max_length=max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        )
                        examples.append({
                            'input_ids': encoded['input_ids'][0],
                            'attention_mask': encoded['attention_mask'][0],
                            'labels': encoded['input_ids'][0].clone()
                        })
            
            dataset = Dataset.from_dict({
                'input_ids': [ex['input_ids'] for ex in examples],
                'attention_mask': [ex['attention_mask'] for ex in examples],
                'labels': [ex['labels'] for ex in examples]
            })
            print("dataset loaded")
            
            return dataset

    def load_model(self):
        print("loading model...")

        if self.mode in ["train", "train_rewarded"]:
            # Add device specification
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                torch_dtype="auto",
                trust_remote_code=True,
                cache_dir=CACHE_PATH,
                device_map="auto"
            )
            model.config.use_cache = True
            model.config.pretraining_tp = 1
            model.gradient_checkpointing_enable()
        else:
            #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            model = LLM(model=self.base_model, 
                        enable_lora=True, 
                        download_dir=CACHE_PATH, 
                        dtype=torch.bfloat16,
                        #dtype=torch.float16,
                        gpu_memory_utilization=0.7,
                        #tensor_parallel_size=1,
                        #max_num_seqs=1,
                        max_lora_rank=64, 
                        max_model_len=16384,
                        enable_prefix_caching=True
                        )
        
        self.model = model
        print("model loaded")
    
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True, cache_dir=CACHE_PATH)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
        self.tokenizer = tokenizer

    def prepare_model_for_training(self, alpha, rank):
        self.model = prepare_model_for_kbit_training(self.model)
        peft_config = LoraConfig(
            lora_alpha=alpha,
            lora_dropout=0.1,
            r=rank,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        return peft_config

    def train(self, dataset, peft_config, alpha, rank, batch_size=4, output_dir=None):
        acc_steps = 16
        training_args = TrainingArguments(
            output_dir=output_dir or "tmp_trainer",
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=acc_steps,
            optim="adamw_8bit",
            learning_rate=2e-4,
            weight_decay=0.001,
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            logging_steps=1,
            save_steps=100,
            report_to="wandb",
            fp16=False,
            bf16=True,
            group_by_length=True,
            lr_scheduler_type="constant",
            remove_unused_columns=False,
            no_cuda=False
        )

        # Use custom trainer class for reward training mode
        trainer_class = RewardWeightedSFTTrainer if self.mode == "train_rewarded" else SFTTrainer
        
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config
        )

        trainer.train()
        batch_size = batch_size * acc_steps
        model_name = f"{self.new_model}_alpha{alpha}_rank{rank}_batch{batch_size}"
        trainer.model.save_pretrained(model_name)
        trainer.tokenizer.save_pretrained(model_name)

    def build_prompt(self, text, task="query"):
        if task == "query":
            return self.tokenizer.apply_chat_template([{
                'role': 'user',
                'content': QUERY_PROMPT_TEMPLATE.format(**text)
            }], tokenize=False)
        else:  # relevance task
            if not self.new_model:
                if self.explain:
                    template = RELEVANCE_PROMPT_TEMPLATE_API
                else:
                    template = RELEVANCE_LABEL_ONLY_TEMPLATE
            else:
                if self.explain:
                    template = RELEVANCE_PROMPT_TEMPLATE
                else:
                    template = RELEVANCE_LABEL_ONLY_TEMPLATE
            
            return self.tokenizer.apply_chat_template([{
                'role': 'user',
                'content': template.format(**text)
            }], tokenize=False)

    def generate_queries(self, questions, temperature=0):
        prompts = [self.build_prompt({"question": question}, task="query") for question in questions]
        responses = self._generate_responses_vllm(prompts, temperature)
        
        # Parse the responses to extract only enumerated lines
        parsed_queries = []
        for response in responses:
            # Remove the '##' terminator (with or without space)
            response = re.sub(r'\s*##', '', response)
            # Remove quotation marks
            response = re.sub(r'"', '', response)
            # Split into lines and only keep lines that start with a number followed by period
            queries = []
            for line in response.strip().split('\n'):
                # Match lines that start with a number followed by period and space
                if re.match(r'^\d+\.\s+', line):
                    # Extract the query part after the enumeration
                    query = re.sub(r'^\d+\.\s+', '', line).strip()
                    if query:  # Only add non-empty queries
                        queries.append(query)
            parsed_queries.append(queries)
        
        return parsed_queries

    def explain_relevance(self, questions, documents, temperature=0):
        # Get the first 8_000 words of the document
        documents = [document[:8_000] for document in documents]

        prompts = [self.build_prompt({"question": question, "document": document, "relevance": self.relevance}, task="relevance") 
                   for question, document in zip(questions, documents)]
        responses = self._generate_responses_vllm(prompts, temperature)
        
        explanations_labels = []
        for i, response in enumerate(responses):
            try:
                if self.explain:
                    # Split by 'Relevance label:' or 'Relevance Label:' (case insensitive)
                    parts = re.split(r'relevance\s+label:', response, flags=re.IGNORECASE)
                    
                    if len(parts) != 2:
                        # Try to infer label from the explanation text if no explicit label
                        explanation = response.split('##')[0].strip()
                        if explanation.startswith('model\n'):
                            explanation = explanation[6:].strip()
                        
                        # Infer label based on keywords in the first 30 words
                        first_sentence = " ".join(explanation.lower().split()[:30])
                        if 'partially relevant' in first_sentence:
                            label = 1
                        elif 'relevant' in first_sentence and 'not relevant' not in first_sentence and 'irrelevant' not in first_sentence:
                            label = 2
                        else:
                            label = 0
                        
                        explanations_labels.append((explanation, label))
                        continue
                    
                    explanation = parts[0].strip()
                    if explanation.startswith('model\n'):
                        explanation = explanation[6:].strip()
                    
                    try:
                        # Clean up the label part and extract the number
                        label_part = parts[1].split('##')[0].strip()
                        label = int(re.search(r'\d+', label_part).group())
                        if label not in [0, 1, 2]:
                            raise ValueError("Invalid label value")
                        explanations_labels.append((explanation, label))
                    except (ValueError, AttributeError, IndexError):
                        print(f"Warning: Invalid label format for index {i}")
                        with open("errors.jsonl", "a") as f:
                            error_log = {
                                "prompt": prompts[i],
                                "response": response,
                                "error": "Invalid label format"
                            }
                            f.write(json.dumps(error_log) + "\n")
                        explanations_labels.append((explanation, 0))
                else:
                    # For label-only mode
                    try:
                        label_text = response.strip().split('Relevance Label:')[1].split('##')[0].strip().split()[-1]
                        label = int(label_text)
                        if label not in [0, 1, 2]:
                            raise ValueError("Invalid label value")
                        explanations_labels.append(("", label))
                    except (ValueError, IndexError):
                        print(f"Warning: Invalid label format for index {i}")
                        # Log the error
                        with open("errors.jsonl", "a") as f:
                            error_log = {
                                "prompt": prompts[i],
                                "response": response,
                                "error": "Invalid label format"
                            }
                            f.write(json.dumps(error_log) + "\n")
                        explanations_labels.append(("", 0))
            except Exception as e:
                print(f"Warning: Unexpected error processing response {i}: {str(e)}")
                # Log the error
                with open("errors.jsonl", "a") as f:
                    error_log = {
                        "prompt": prompts[i],
                        "response": response,
                        "error": str(e)
                    }
                    f.write(json.dumps(error_log) + "\n")
                explanations_labels.append(("", 0))
        
        return explanations_labels

    def explain_relevance_api(self, questions: List[str], documents: List[str], temperature=0, max_workers=50) -> List[Tuple[str, int]]:
        """Similar to explain_relevance but uses API endpoint instead of local model with parallel processing"""
        # Get the first 8_000 words of the document
        documents = [document[:8_000] for document in documents]
        
        def process_single_pair(args) -> Tuple[str, int]:
            question, document = args
            template = RELEVANCE_PROMPT_TEMPLATE_API if self.explain else RELEVANCE_LABEL_ONLY_TEMPLATE
            prompt = template.format(question=question, document=document, relevance=self.relevance)
            
            response = llm_api(prompt, temperature=temperature)
            if not response:
                print("Warning: API returned no response")
                return ("", 0)
            
            try:
                if self.explain:
                    # Split by 'Relevance label:' or 'Relevance Label:' (case insensitive)
                    parts = re.split(r'relevance\s+label:', response, flags=re.IGNORECASE)
                    
                    if len(parts) != 2:
                        # Try to infer label from the explanation text if no explicit label
                        explanation = response.split('##')[0].strip()
                        
                        # Infer label based on keywords in the first 30 words
                        first_sentence = " ".join(explanation.lower().split()[:30])
                        if 'partially relevant' in first_sentence:
                            label = 1
                        elif 'relevant' in first_sentence and 'not relevant' not in first_sentence and 'irrelevant' not in first_sentence:
                            label = 2
                        else:
                            label = 0
                        
                        return (explanation, label)
                    
                    explanation = parts[0].strip()
                    
                    try:
                        # Clean up the label part and extract the number
                        label_part = parts[1].split('##')[0].strip()
                        label = int(re.search(r'\d+', label_part).group())
                        if label not in [0, 1, 2]:
                            raise ValueError("Invalid label value")
                        return (explanation, label)
                    except (ValueError, AttributeError, IndexError):
                        print(f"Warning: Invalid label format in API response")
                        return (explanation, 0)
                else:
                    # For label-only mode
                    try:
                        label_text = response.strip().split('Relevance Label:')[1].split('##')[0].strip().split()[-1]
                        label = int(label_text)
                        if label not in [0, 1, 2]:
                            raise ValueError("Invalid label value")
                        return ("", label)
                    except (ValueError, IndexError):
                        print(f"Warning: Invalid label format in API response")
                        return ("", 0)
            except Exception as e:
                print(f"Warning: Unexpected error processing API response: {str(e)}")
                return ("", 0)

        # Process pairs in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create list of question-document pairs
            pairs = list(zip(questions, documents))
            # Submit all pairs for processing
            future_to_pair = {executor.submit(process_single_pair, pair): pair for pair in pairs}
            
            # Collect results in order
            results = []
            for future in concurrent.futures.as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    result = future.result()
                    results.append((pair, result))
                except Exception as e:
                    print(f"Error processing pair {pair}: {str(e)}")
                    results.append((pair, ("", 0)))
            
            # Sort results back into original order
            results.sort(key=lambda x: pairs.index(x[0]))
            return [r[1] for r in results]

    def _generate_responses(self, prompts):
        input_ids = self.tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True).to(self.model.device)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("##")
        ]
        attention_mask = input_ids['input_ids'] != self.tokenizer.pad_token_id

        outputs = self.model.generate(
            input_ids['input_ids'],
            tokenizer=self.tokenizer,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            eos_token_id=terminators,
            stop_strings=[" ##", "<|eot_id|>"],
            do_sample=False
        )
        
        responses = [self.tokenizer.decode(output[input_ids['input_ids'].shape[-1]:], skip_special_tokens=True).replace("assistant\n\n", "", 1) 
                     for output in outputs]
        return responses

    def _generate_responses_vllm(self, prompts, temperature=0):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=1024,
            stop=[" ##", "<|eot_id|>"]
        )

        if self.new_model:
            outputs = self.model.generate(prompts, sampling_params, lora_request=LoRARequest("my_adapter", 1, self.new_model))
        else:
            outputs = self.model.generate(prompts, sampling_params)
        
        responses = [output.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "") 
                     for output in outputs]
        return responses

    def annotate_pairs(self, file_path, output_path, batch_size=64, start_index=0, max_entries=None, include_groundtruth=False, k=1, temperature=0):
        print("Processing query-document pairs...")
        pairs = []
        entries_processed = 0
        entries_skipped = 0

        def truncate_text(text, max_words=3_000):
            words = text.split()
            return ' '.join(words[:max_words])

        # Collect all query-document pairs
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip entries until we reach start_index
                if entries_skipped < start_index:
                    entries_skipped += 1
                    continue
                
                if max_entries and entries_processed >= max_entries:
                    break
                
                try:
                    d = json.loads(line)
                    question = d["question"]["title"] + " " + d["question"]["body"]
                    
                    # Process links
                    for link in d.get("links", []):
                        pair = {
                            "query": question,
                            "document": truncate_text(link["text"])
                        }
                        if include_groundtruth and "label" in link:
                            pair["groundtruth"] = {
                                "explanation": link.get("explanation", ""),
                                "label": link["label"]
                            }
                        pairs.append(pair)
                    
                    # Process other_document if present
                    if "other_document" in d:
                        pair = {
                            "query": question,
                            "document": truncate_text(d["other_document"]["text"])
                        }
                        if include_groundtruth and "label" in d["other_document"]:
                            pair["groundtruth"] = {
                                "explanation": d["other_document"].get("explanation", ""),
                                "label": d["other_document"]["label"]
                            }
                        pairs.append(pair)
                    
                    entries_processed += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue

        # Process in batches
        print(f"Total pairs to process: {len(pairs)}")
        with open(output_path, 'w') as out_f:
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                questions = [pair["query"] for pair in batch]
                documents = [pair["document"] for pair in batch]
                
                print(f"Processing batch {i//batch_size + 1}/{(len(pairs)-1)//batch_size + 1}")
                
                # Get k predictions for each pair
                all_predictions = []
                for _ in range(k):
                    explanations_labels = self.explain_relevance(questions, documents, temperature)
                    all_predictions.append(explanations_labels)
                
                # Write results
                for idx, pair in enumerate(batch):
                    output = {
                        "query": pair["query"],
                        "document": pair["document"],
                    }
                    
                    # Format predictions based on k
                    if k == 1:
                        explanation, label = all_predictions[0][idx]
                        output["prediction"] = {
                            "label": label
                        }
                        if explanation:  # Only include explanation if not empty
                            output["prediction"]["explanation"] = explanation
                    else:
                        output["predictions"] = []
                        for pred in all_predictions:
                            prediction = {
                                "label": pred[idx][1]
                            }
                            if pred[idx][0]:  # Only include explanation if not empty
                                prediction["explanation"] = pred[idx][0]
                            output["predictions"].append(prediction)
                    
                    # Include ground truth if available
                    if "groundtruth" in pair:
                        output["groundtruth"] = pair["groundtruth"]
                    
                    out_f.write(json.dumps(output) + "\n")

    def generate_query_sets(self, file_path, output_path, batch_size=64, start_index=0, max_entries=None, include_groundtruth=False, k=1, temperature=0):
        print("Processing questions for query generation...")
        questions = []
        entries_processed = 0
        entries_skipped = 0

        # Collect all questions
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip entries until we reach start_index
                if entries_skipped < start_index:
                    entries_skipped += 1
                    continue
                
                if max_entries and entries_processed >= max_entries:
                    break
                
                try:
                    d = json.loads(line)
                    question = d["question"]["title"] + " " + d["question"]["body"]
                    question_data = {
                        "question": question,
                    }
                    if include_groundtruth:
                        question_data["groundtruth_queries"] = d.get("generated_queries", [])
                    questions.append(question_data)
                    entries_processed += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue

        # Process in batches
        print(f"Total questions to process: {len(questions)}")
        with open(output_path, 'w') as out_f:
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                batch_questions = [q["question"] for q in batch]
                
                print(f"Processing batch {i//batch_size + 1}/{(len(questions)-1)//batch_size + 1}")
                
                # Get k sets of queries for each question
                all_predictions = []
                for _ in range(k):
                    query_sets = self.generate_queries(batch_questions, temperature)
                    # Filter out sets with less than 10 queries
                    query_sets = [queries if len(queries) >= 10 else None for queries in query_sets]
                    all_predictions.append(query_sets)
                
                # Write results
                for idx, question_data in enumerate(batch):
                    output = {
                        "question": question_data["question"],
                    }
                    
                    # Include ground truth if requested and available
                    if include_groundtruth and "groundtruth_queries" in question_data:
                        output["groundtruth_queries"] = question_data["groundtruth_queries"]
                    
                    # Format predictions based on k
                    if k == 1:
                        if all_predictions[0][idx] is not None:  # Only write if we have 10+ queries
                            output["generated_queries"] = all_predictions[0][idx]
                            out_f.write(json.dumps(output) + "\n")
                    else:
                        valid_sets = [pred[idx] for pred in all_predictions if pred[idx] is not None]
                        if valid_sets:  # Only write if we have at least one valid set
                            output["generated_query_sets"] = valid_sets
                            out_f.write(json.dumps(output) + "\n")
                    
                    # Remove this duplicate section
                    # if include_groundtruth and "groundtruth_queries" in question_data:
                    #     output["groundtruth_queries"] = question_data["groundtruth_queries"]

if __name__ == '__main__':
    # Usage example
    base_model = "meta-llama/Llama-3.2-3B-Instruct" # meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.1-8B-Instruct, google/gemma-2-2b-it, Qwen/Qwen2.5-3B-Instruct
    new_model = "../llama32-3b-ranker-es"
    output_dir = "../results_3b_es"

    ranker = InstructRank(base_model=base_model, new_model=new_model, mode="train", explain=False)
    dataset = ranker.load_dataset('../data/train.jsonl', max_entries=20_000, max_length=8192)
    
    alpha = 16
    rank = 64
    batch_size = 1
    
    peft_config = ranker.prepare_model_for_training(alpha, rank)
    ranker.train(dataset, peft_config, alpha, rank, batch_size, output_dir)

    # # Inference example
    # ranker = InstructRank(base_model="api",#"google/gemma-2-2b-it", 
    #                       new_model="../results_gemma2_2b_label_only/checkpoint-4650", 
    #                       mode="inference",
    #                       explain=True)
    # print("inference...")

    # # ranker.annotate_pairs('../data/train.jsonl', 'explanations_3b_2_rl1.jsonl', 
    # #                       batch_size=64, start_index=5000, max_entries=5000, include_groundtruth=True, k=8, temperature=1.0)
    
    # # Load 100 examples from train.jsonl
    # examples = []
    # with open('../data/train.jsonl', 'r') as f:
    #     for i, line in enumerate(f):
    #         if i >= 100:  # Stop after 100 examples
    #             break
    #         data = json.loads(line)
    #         examples.append({
    #             'query': data['question']['title'] + " " + data['question']['body'],
    #             'document': data['links'][0]['text'] if data.get('links') else data['other_document']['text']
    #         })
    
    # questions = [ex['query'] for ex in examples]
    # documents = [ex['document'] for ex in examples]
    
    # # print("Generated queries:")
    # # queries = ranker.generate_queries(questions)
    # # with open("queries_temp.jsonl", "w+") as f:
    # #     for i, query in enumerate(queries):
    # #         f.write(json.dumps({"question": questions[i], "generated_queries": query}) + "\n")
    
    # print("\nRelevance explanations:")

    # t1 = time.time()
    # explanations_labels = ranker.explain_relevance_api(questions, documents)
    # t2 = time.time()
    # print(f"Time taken: {t2 - t1} seconds")

    # # write docs with explanations_labels to a jsonl file
    # with open("explanations_temp.jsonl", "w+") as f:
    #     for i, (explanation, label) in enumerate(explanations_labels):
    #         f.write(json.dumps({"question": questions[i], "document": documents[i], "explanation": explanation, "label": label}) + "\n")

