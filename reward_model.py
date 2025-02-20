import os
import re
import json
import torch
import time

CACHE_PATH = 'hf_cache'
os.environ['HF_TOKEN'] = 'your-hf-token'
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH
os.environ['HF_HOME'] = CACHE_PATH
os.environ['HF_DATASETS_CACHE'] = CACHE_PATH
os.environ['TORCH_HOME'] = CACHE_PATH

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

from transformers import StoppingCriteria, StoppingCriteriaList
from torch import LongTensor, FloatTensor

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


ASSESSMENT_PROMPT_TEMPLATE = """You are an expert at evaluating document relevance annotations. Given a query, document, and human annotation (including explanation and relevance label), assess the quality of the annotation using the following scoring system and output your evaluation in JSON format.

SCORING CRITERIA (Maximum 5 points):

1. Relevance Label Accuracy (0-2 points)
- 2 points: Label perfectly matches document-query relationship
- 1 point: Label is off by one level
- 0 points: Label is off by two levels

2. Explanation Quality (0-2 points)
- Key connections identified (+0.5)
- Relationships accurately described (+0.5)
- No unsupported claims (+0.5)
- Clear logical reasoning (+0.5)

3. Completeness (0-1 point)
- All major relevance signals discussed (+0.5)
- Thorough analysis without redundancy (+0.5)

QUERY: {query}
DOCUMENT: {document}
EXPLANATION: {explanation}
RELEVANCE LABEL: {label}"""

class RewardModel:
    def __init__(self, base_model="meta-llama/Llama-3.1-8B-Instruct", new_model=None, mode="train"):
        self.base_model = base_model
        self.new_model = new_model
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.mode = mode

        self.load_model()
        self.load_tokenizer()

    def load_dataset(self, file_path, max_entries=None):
        print("loading dataset...")
        examples = []
        entries_processed = 0

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if max_entries and entries_processed >= max_entries:
                    break
                
                try:
                    d = json.loads(line)
                    entries_processed += 1

                    # Create training example from prediction
                    if "prediction" in d and "annotation" in d["prediction"]:
                        assessment_messages = [{
                            'role': 'user',
                            'content': ASSESSMENT_PROMPT_TEMPLATE.format(
                                query=d["query"],
                                document=d["document"],
                                explanation=d["prediction"]["explanation"],
                                label=d["prediction"]["label"]
                            )
                        }, {
                            'role': 'assistant',
                            'content': json.dumps(d["prediction"]["annotation"])
                        }]
                        examples.append(self.tokenizer.apply_chat_template(assessment_messages, tokenize=False) + self.tokenizer.eos_token)

                    # Create training example from groundtruth
                    if "groundtruth" in d and "annotation" in d["groundtruth"]:
                        assessment_messages = [{
                            'role': 'user',
                            'content': ASSESSMENT_PROMPT_TEMPLATE.format(
                                query=d["query"],
                                document=d["document"],
                                explanation=d["groundtruth"]["explanation"],
                                label=d["groundtruth"]["label"]
                            )
                        }, {
                            'role': 'assistant',
                            'content': json.dumps(d["groundtruth"]["annotation"])
                        }]
                        examples.append(self.tokenizer.apply_chat_template(assessment_messages, tokenize=False) + self.tokenizer.eos_token)

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue

        dataset = Dataset.from_dict({'text': examples})
        print(f"dataset loaded with {len(examples)} examples")
        return dataset

    def load_model(self):
        print("loading model...")

        if self.mode == "train":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                #attn_implementation="flash_attention_2",
                trust_remote_code=True,
                cache_dir=CACHE_PATH
            )
            model.config.use_cache = True
            model.config.pretraining_tp = 1
            model.gradient_checkpointing_enable()
        else:
            model = LLM(model=self.base_model, 
                        enable_lora=True, 
                        download_dir=CACHE_PATH, 
                        max_lora_rank=64, 
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

    def train(self, dataset, peft_config, alpha, rank, batch_size, output_dir):
        acc_steps = 8
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=acc_steps,
            optim="adamw_8bit",
            save_steps=100,
            logging_steps=1,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="wandb"
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=peft_config,
            max_seq_length=8192,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=False,
        )

        trainer.train()
        batch_size = batch_size * acc_steps
        model_name = f"{self.new_model}_alpha{alpha}_rank{rank}_batch{batch_size}"
        trainer.model.save_pretrained(model_name)
        trainer.tokenizer.save_pretrained(model_name)

    def assess_explanation(self, query, document, explanation, label):
        """Generate assessment for a single explanation"""
        prompt = self.tokenizer.apply_chat_template([{
            'role': 'user',
            'content': ASSESSMENT_PROMPT_TEMPLATE.format(
                query=query,
                document=document,
                explanation=explanation,
                label=label
            )
        }], tokenize=False)
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            stop=[" ##", "<|eot_id|>"]
        )

        output = self.model.generate([prompt], sampling_params, lora_request=LoRARequest("my_adapter", 1, self.new_model))[0]
        response = output.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response}")
            return None

    def batch_assess_explanations(self, queries, documents, explanations, labels, batch_size=32):
        """Generate assessments for multiple explanations in batches"""
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_explanations = explanations[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            prompts = [
                self.tokenizer.apply_chat_template([{
                    'role': 'user',
                    'content': ASSESSMENT_PROMPT_TEMPLATE.format(
                        query=q,
                        document=d,
                        explanation=e,
                        label=l
                    )
                }], tokenize=False)
                for q, d, e, l in zip(batch_queries, batch_docs, batch_explanations, batch_labels)
            ]
            
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1024,
                stop=[" ##", "<|eot_id|>"]
            )

            outputs = self.model.generate(prompts, sampling_params, lora_request=LoRARequest("my_adapter", 1, self.new_model))
            responses = [output.outputs[0].text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "") 
                        for output in outputs]
            
            for response in responses:
                try:
                    results.append(json.loads(response))
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON response: {response}")
                    results.append(None)
        
        return results

if __name__ == '__main__':
    # For training
    base_model = "meta-llama/Llama-3.1-8B-Instruct"
    new_model = "../results_reward-8b/checkpoint-1400"
    output_dir = "../results_reward-8b"

    # # Initialize and train
    # ranker = RewardModel(base_model=base_model, new_model=new_model, mode="train")
    # dataset = ranker.load_dataset('explanations_annotated.jsonl')
    
    # alpha = 16
    # rank = 64
    # batch_size = 2
    
    # peft_config = ranker.prepare_model_for_training(alpha, rank)
    # ranker.train(dataset, peft_config, alpha, rank, batch_size, output_dir)

    # For inference
    ranker = RewardModel(base_model=base_model, new_model=new_model, mode="inference")
    
    # # # Single assessment
    # # assessment = ranker.assess_explanation(
    # #     query="What is...",
    # #     document="The document states...",
    # #     explanation="This document is relevant because...",
    # #     label=2
    # # )

    with open("explanation.json", "r") as f:
        data = json.load(f)
        queries = [data["query"]]
        documents = [data["document"]]
        explanations = [data["prediction"]["explanation"]]
        labels = [data["prediction"]["label"]]

    # Batch assessment
    assessments = ranker.batch_assess_explanations(
        queries=queries,
        documents=documents,
        explanations=explanations,
        labels=labels,
        batch_size=32
    )

    open("assessment.json", "w").write(json.dumps(assessments, indent=4))

