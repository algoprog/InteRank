import os
import json
import torch
import math
import argparse

CACHE_PATH = 'hf_cache'
os.environ['HF_TOKEN'] = 'your-hf-token'
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH
os.environ['HF_HOME'] = CACHE_PATH
os.environ['HF_DATASETS_CACHE'] = CACHE_PATH
os.environ['TORCH_HOME'] = CACHE_PATH

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict

RELEVANCE_PROMPT_TEMPLATE_TRAIN = "Explain whether the following document is relevant or not to the given question. Then end your response with a relevance label (0: irrelevant, 1: partially relevant, 2: relevant) and the symbol '##'. Question: {question}\nDocument: {document}"
RELEVANCE_PROMPT_TEMPLATE = "Explain whether the following document is relevant or not to the given question. Then end your response with a relevance label (0: irrelevant, does not provide any useful information related to the question, 1: partially relevant, provides some context related to the question but does not provide answer to the specific question, 2: relevant, provides useful information to answer the question) and the symbol '##'. Question: {question}\nDocument: {document}"
RESPONSE_TEMPLATE = "{explanation}\n\nRelevance Label: {label} ##"
QUERY_PROMPT_TEMPLATE = "Based on the following question, generate a list of up to 10 search queries that would help find answers. Each query should be specific and focused on different aspects of the question. Format the output as one query per line and end your response with the symbol '##'. Question: {question}"
QUERY_PROMPT_TEMPLATE_TRAIN = "Based on the following question, generate a list of up to 10 search queries that would help find answers. Each query should be specific and focused on different aspects of the question. Format the output as one query per line. Question: {question}"

def load_jsonl(file_path, max_entries=None, start_index=0):
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < start_index:
                continue
            if max_entries is not None and len(data) >= max_entries:
                break
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def format_conversation(question: str, document: str, explanation: str, label: int) -> List[Dict]:
    user_prompt = RELEVANCE_PROMPT_TEMPLATE.format(
        question=question,
        document=document
    )
    assistant_response = RESPONSE_TEMPLATE.format(
        explanation=explanation,
        label=label
    )
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_response}
    ]

def get_reward_scores_batch(conversations: List[List[Dict]], rm_model, rm_tokenizer, device: str) -> List[float]:
    # Tokenize all conversations in the batch
    conv_tokenized = [
        rm_tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt")
        for conv in conversations
    ]
    
    # Pad to max length in batch
    max_length = max(conv.size(1) for conv in conv_tokenized)
    padded_convs = torch.stack([
        torch.nn.functional.pad(conv.squeeze(0), (0, max_length - conv.size(1)), value=rm_tokenizer.pad_token_id)
        for conv in conv_tokenized
    ]).to(device)
    
    # Get reward scores
    with torch.no_grad():
        scores = rm_model(padded_convs).logits.squeeze(-1)
    
    return scores.tolist()

def process_batch(batch_items: List[tuple], rm_model, rm_tokenizer, device: str) -> List[float]:
    conversations = [
        format_conversation(query, document, explanation, label)
        for query, document, explanation, label in batch_items
    ]
    return get_reward_scores_batch(conversations, rm_model, rm_tokenizer, device)

def save_batch_jsonl(data, file_path, mode='a'):
    with open(file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            f.flush()

def create_preference_pairs(item):
    """Create preference pair between highest and lowest scoring explanations."""
    pairs = []
    query = item["query"]
    document = item["document"]
    
    # Collect all explanations with scores
    all_explanations = [
        (pred["explanation"], pred["label"], pred["score"])
        for pred in item["predictions"]
    ]
    gt = item["groundtruth"]
    all_explanations.append((gt["explanation"], gt["label"], gt["score"]))
    
    # Find highest and lowest scoring explanations
    best_exp = max(all_explanations, key=lambda x: x[2])
    worst_exp = min(all_explanations, key=lambda x: x[2])
    
    # Only create pair if they're different
    if best_exp != worst_exp:
        prompt = RELEVANCE_PROMPT_TEMPLATE_TRAIN.format(
            question=query,
            document=document
        )
        chosen = RESPONSE_TEMPLATE.format(
            explanation=best_exp[0],
            label=best_exp[1]
        )
        rejected = RESPONSE_TEMPLATE.format(
            explanation=worst_exp[0],
            label=worst_exp[1]
        )
        
        pair = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": best_exp[2],
            "rejected_score": worst_exp[2]
        }
        pairs.append(pair)
    
    return pairs

def create_query_preference_pairs(item):
    """Create preference pair between highest and lowest scoring query sets."""
    pairs = []
    question = item["question"]
    
    # Collect all query sets with scores
    all_query_sets = list(zip(item["generated_query_sets"], item["query_set_scores"]))
    if "groundtruth_queries" in item and "groundtruth_score" in item:
        all_query_sets.append((item["groundtruth_queries"], item["groundtruth_score"]))
    
    # Find highest and lowest scoring query sets
    best_queries = max(all_query_sets, key=lambda x: x[1])
    worst_queries = min(all_query_sets, key=lambda x: x[1])
    
    # Only create pair if they're different
    if best_queries[0] != worst_queries[0]:
        chosen = "\n".join(f"{i+1}. {query}" for i, query in enumerate(best_queries[0])) + " ##"
        rejected = "\n".join(f"{i+1}. {query}" for i, query in enumerate(worst_queries[0])) + " ##"
        
        pair = {
            "prompt": QUERY_PROMPT_TEMPLATE_TRAIN.format(question=question),
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": best_queries[1],
            "rejected_score": worst_queries[1]
        }
        pairs.append(pair)
    
    return pairs

def setup_scoring(args):
    """Setup common scoring configuration and models."""
    config = {
        "device": "cuda:0",
        "model_name": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        "batch_size": 2,
        "max_entries": 5000
    }
    
    print("Loading model and tokenizer...")
    rm = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map=config["device"],
        num_labels=1
    ).half()
    rm_tokenizer = AutoTokenizer.from_pretrained(config["model_name"], model_max_length=8096)
    
    return config, rm, rm_tokenizer

def prepare_query_items(data):
    """Prepare items for query scoring."""
    all_items = []
    for item in data:
        question = item["question"]
        
        # Process generated query sets
        for query_set in item["generated_query_sets"]:
            query_text = "\n".join(f"{i+1}. {query}" for i, query in enumerate(query_set)) + " ##"
            all_items.append((question, query_text, ("pred", item, {"queries": query_set})))
        
        # Process groundtruth queries
        if "groundtruth_queries" in item:
            query_text = "\n".join(f"{i+1}. {query}" for i, query in enumerate(item["groundtruth_queries"])) + " ##"
            all_items.append((question, query_text, ("gt", item, {"queries": item["groundtruth_queries"]})))
    
    return all_items

def prepare_explanation_items(data):
    """Prepare items for explanation scoring."""
    all_items = []
    for item in data:
        query = item["query"]
        document = item["document"]
        
        for pred in item["predictions"]:
            if "explanation" in pred:
                all_items.append((
                    query, document, pred["explanation"], pred["label"], 
                    ("pred", item, pred)
                ))
        
        all_items.append((
            query, document, 
            item["groundtruth"]["explanation"], 
            item["groundtruth"]["label"],
            ("gt", item, None)
        ))
    return all_items

def process_scoring_batch(batch, rm, rm_tokenizer, device, is_query_mode=False):
    """Process a batch of items for scoring."""
    conversations = []
    for item in batch:
        if is_query_mode:
            question, queries, _ = item
            queries = "\n".join(f"{query}" for query in queries)
            conv = [
                {"role": "user", "content": QUERY_PROMPT_TEMPLATE.format(question=question)},
                {"role": "assistant", "content": queries}
            ]
        else:
            query, document, explanation, label, _ = item
            conv = format_conversation(query, document, explanation, label)
        conversations.append(conv)
    
    return get_reward_scores_batch(conversations, rm, rm_tokenizer, device)

def load_processed_ids(output_file):
    """Load IDs of items that have already been processed."""
    processed_ids = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Use a tuple of question title and body as a unique identifier
                if "question" in item:  # For query scoring
                    id_tuple = (item["question"]["title"], item["question"]["body"])
                else:  # For explanation scoring
                    id_tuple = (item["query"], item["document"])
                processed_ids.add(id_tuple)
    return processed_ids

def score_items(all_items, config, rm, rm_tokenizer, output_file, is_query_mode):
    """Score items in batches and save results."""
    if not os.path.exists(output_file):
        open(output_file, 'w').close()
    
    processed_ids = load_processed_ids(output_file)
    
    # Filter out already processed items
    filtered_items = []
    for item in all_items:
        metadata = item[-1]
        _, item_obj, _ = metadata
        if is_query_mode:
            id_tuple = (item_obj["question"])
        else:
            id_tuple = (item_obj["query"], item_obj["document"])
        
        if id_tuple not in processed_ids:
            filtered_items.append(item)
    
    all_items = filtered_items
    print(f"Processing {len(all_items)} new items...")
    
    if not all_items:
        print("No new items to process.")
        return

    num_batches = math.ceil(len(all_items) / config["batch_size"])
    processed_items = {}
    
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * config["batch_size"]
        end_idx = min((batch_idx + 1) * config["batch_size"], len(all_items))
        batch = all_items[start_idx:end_idx]
        
        scores = process_scoring_batch(batch, rm, rm_tokenizer, config["device"], is_query_mode)
        
        completed_items = []
        for item, score in zip(batch, scores):
            metadata = item[-1]
            item_type, item_obj, pred = metadata
            item_id = id(item_obj)
            
            if item_id not in processed_items:
                processed_items[item_id] = {
                    "item": item_obj,
                    "pending": (len(item_obj.get("generated_query_sets", [])) + 
                              (1 if "groundtruth_queries" in item_obj else 0)) if is_query_mode 
                              else (len(item_obj["predictions"]) + 1)
                }
            
            if item_type == "pred":
                if is_query_mode:
                    # For query mode, store just the score in the list
                    if "query_set_scores" not in item_obj:
                        item_obj["query_set_scores"] = []
                    item_obj["query_set_scores"].append(score)
                else:
                    pred["score"] = score
            else:  # gt
                if is_query_mode:
                    item_obj["groundtruth_score"] = score
                else:
                    item_obj["groundtruth"]["score"] = score
                    
            processed_items[item_id]["pending"] -= 1
            
            if processed_items[item_id]["pending"] == 0:
                completed_items.append(processed_items[item_id]["item"])
                del processed_items[item_id]
        
        if completed_items:
            save_batch_jsonl(completed_items, output_file)

def create_single_items(item, min_reward=0.0, boosted=False):
    """Create single items from scored explanations."""
    singles = []
    query = item["query"]
    document = item["document"]
    
    # Collect all explanations with scores
    all_scores = [pred["score"] for pred in item["predictions"]]
    all_scores.append(item["groundtruth"]["score"])
    
    # Calculate min and max for normalization
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score

    # Calculate misclassification probability if boosted
    if boosted:
        gt_label = item["groundtruth"]["label"]
        diff_label_count = sum(1 for pred in item["predictions"] if pred["label"] != gt_label)
        misclass_prob = diff_label_count / len(item["predictions"]) if item["predictions"] else 0

    prompt = RELEVANCE_PROMPT_TEMPLATE_TRAIN.format(
        question=query,
        document=document
    )
    
    # Process predictions
    for pred in item["predictions"]:
        # Normalize score to [0, 1]
        normalized_score = (pred["score"] - min_score) / score_range if score_range != 0 else 1.0
        
        # Only add items with normalized reward >= min_reward
        if normalized_score >= min_reward:
            # Apply boosting if enabled
            if boosted:
                normalized_score *= misclass_prob
            if normalized_score > 0:
                response = RESPONSE_TEMPLATE.format(
                    explanation=pred["explanation"],
                    label=pred["label"]
                )
                singles.append({
                    "prompt": prompt,
                    "response": response,
                    "reward": normalized_score,
                    "misclass_prob": misclass_prob
                })
    
    # Process groundtruth
    gt = item["groundtruth"]
    normalized_score = (gt["score"] - min_score) / score_range if score_range != 0 else 1.0
    
    if normalized_score >= min_reward:
        if boosted:
            normalized_score *= misclass_prob
        
        if normalized_score > 0:
            singles.append({
                "prompt": prompt,
                "response": RESPONSE_TEMPLATE.format(
                explanation=gt["explanation"],
                label=gt["label"]
            ),
            "reward": normalized_score,
            "misclass_prob": misclass_prob
        })
    
    return singles

def create_single_query_items(item, min_reward=0.0, boosted=False):
    """Create single items from scored query sets."""
    singles = []
    question = item["question"]
    
    # Collect all scores
    all_scores = item["query_set_scores"]
    if "groundtruth_score" in item:
        all_scores = all_scores + [item["groundtruth_score"]]
    
    # Calculate min and max for normalization
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score
    
    prompt = QUERY_PROMPT_TEMPLATE_TRAIN.format(question=question)
    
    # Process generated query sets
    for queries, score in zip(item["generated_query_sets"], item["query_set_scores"]):
        # Normalize score to [0, 1]
        normalized_score = (score - min_score) / score_range if score_range != 0 else 1.0
        
        # Only add items with normalized reward >= min_reward
        if normalized_score >= min_reward:
            if normalized_score > 0:
                response = "\n".join(f"{i+1}. {query}" for i, query in enumerate(queries)) + " ##"
                singles.append({
                    "prompt": prompt,
                    "response": response,
                    "reward": normalized_score
                })
    
    # Process groundtruth if available
    if "groundtruth_queries" in item and "groundtruth_score" in item:
        normalized_score = (item["groundtruth_score"] - min_score) / score_range if score_range != 0 else 1.0
            
        if normalized_score >= min_reward:            
            if normalized_score > 0:
                response = "\n".join(f"{i+1}. {query}" for i, query in enumerate(item["groundtruth_queries"])) + " ##"
                singles.append({
                    "prompt": prompt,
                    "response": response,
                    "reward": normalized_score
                })
    
    return singles

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["score", "pair", "score_query", "pair_query", "single", "single_query"], 
                      default="single_query",
                      help="Mode to run: 'score' for scoring explanations, 'pair' for creating preference pairs, "
                           "'score_query' for scoring query generation, 'pair_query' for creating query preference pairs, "
                           "'single' for creating single items with normalized scores, 'single_query' for creating single query items")
    parser.add_argument("--input", default="queries_scored.jsonl", help="Input file path")
    parser.add_argument("--output", default="queries_rewards_boosted.jsonl", help="Output file path")
    parser.add_argument("--start-index", type=int, default=0, help="Start processing from this index in the input file")
    parser.add_argument("--max-entries", type=int, default=None, help="Max entries to process")
    parser.add_argument("--min-reward", type=float, default=0.85, 
                       help="Minimum normalized reward threshold for including items (default: 0.0)")
    parser.add_argument("--boosted", action="store_true", 
                       help="Whether to boost rewards based on misclassification probability")
    args = parser.parse_args()

    is_query_mode = args.mode == "score_query"

    if args.mode in ["score", "score_query"]:
        config, rm, rm_tokenizer = setup_scoring(args)
        data = load_jsonl(args.input, args.max_entries, args.start_index)
        
        all_items = (prepare_query_items(data) if is_query_mode 
                    else prepare_explanation_items(data))
        
        score_items(all_items, config, rm, rm_tokenizer, args.output, 
                   is_query_mode=is_query_mode)
        
        print(f"Results saved to {args.output}")
    elif args.mode == "single":
        print(f"Loading scored explanations from {args.input}...")
        data = load_jsonl(args.input)
        
        print("Creating single items...")
        all_items = []
        for item in tqdm(data):
            items = create_single_items(item, args.min_reward, args.boosted)
            all_items.extend(items)
        
        print(f"Saving {len(all_items)} single items to {args.output}...")
        save_jsonl(all_items, args.output)
        print("Done!")
    elif args.mode == "single_query":
        print(f"Loading scored queries from {args.input}...")
        data = load_jsonl(args.input)
        
        print("Creating single items for queries...")
        all_items = []
        for item in tqdm(data):
            items = create_single_query_items(item, args.min_reward, args.boosted)
            all_items.extend(items)
        
        print(f"Saving {len(all_items)} single items to {args.output}...")
        save_jsonl(all_items, args.output)
        print("Done!")
    elif args.mode == "pair_query":
        print(f"Loading scored queries from {args.input}...")
        data = load_jsonl(args.input)
        
        print("Creating preference pairs for queries...")
        all_pairs = []
        for item in tqdm(data):
            pairs = create_query_preference_pairs(item)
            all_pairs.extend(pairs)
        
        print(f"Saving {len(all_pairs)} preference pairs to {args.output}...")
        save_jsonl(all_pairs, args.output)
        print("Done!")
    else:  # pair mode for explanations
        print(f"Loading scored explanations from {args.input}...")
        data = load_jsonl(args.input)
        
        print("Creating preference pairs...")
        all_pairs = []
        for item in tqdm(data):
            pairs = create_preference_pairs(item)
            all_pairs.extend(pairs)
        
        print(f"Saving {len(all_pairs)} preference pairs to {args.output}...")
        save_jsonl(all_pairs, args.output)
        print("Done!")

if __name__ == "__main__":
    main()