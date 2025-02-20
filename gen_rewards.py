import time
import json
import random

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


DEEPINFRA_CLIENT = OpenAI(api_key="your-api-key", 
                          base_url="https://api.deepinfra.com/v1/openai")
OPENAI_CLIENT = OpenAI(api_key="your-api-key")


def llm_api(prompt, model="gpt-4o", client=OPENAI_CLIENT, temperature=0.0, max_retries=2):
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

def generate_annotation(query, document, explanation, label):
    prompt = f"""
    You are an expert at evaluating document relevance annotations. Given a query, document, and human annotation (including explanation and relevance label), assess the quality of the annotation using the following scoring system and output your evaluation in JSON format.

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

    Output your evaluation in this JSON format, nothing else in your response:

    {{
        "label_assessment": {{
            "agrees_with_label": boolean,
            "points_awarded": float,
            "explanation": string
        }},
        "explanation_quality": {{
            "points_awarded": float,
            "strengths": [
                {{
                    "description": string,
                    "points": float
                }}
            ]
        }},
        "completeness": {{
            "points_awarded": float,
            "coverage_assessment": string,
            "analysis_assessment": string
        }},
        "final_score": float
    }}

    QUERY: {query}
    DOCUMENT: {document}
    EXPLANATION: {explanation}
    RELEVANCE LABEL: {label}
    """
    response = llm_api(prompt)
    response = response.replace("```json", "").replace("```", "")
    response = json.loads(response)

    return response

def process_annotations(input_file, output_file):
    # First, read already processed queries from output file
    processed_queries = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                processed_queries.add(item['query'])
    except FileNotFoundError:
        pass  # Output file doesn't exist yet

    # Open output file in append mode instead of write mode
    with open(output_file, 'a') as outfile:
        # Read input JSONL file line by line
        with open(input_file, 'r') as f:
            data = [json.loads(line) for line in f]
        
        # Filter out already processed items
        unprocessed_data = [item for item in data if item['query'] not in processed_queries]
        print(f"Skipping {len(data) - len(unprocessed_data)} already processed items")
        print(f"Processing {len(unprocessed_data)} remaining items")

        with ThreadPoolExecutor(max_workers=32) as executor:
            # Submit both prediction and groundtruth annotations as separate tasks
            futures = {}
            for item in unprocessed_data:
                # Submit prediction annotation
                pred_future = executor.submit(
                    generate_annotation, 
                    item['query'], 
                    item['document'],
                    item['prediction']['explanation'], 
                    item['prediction']['label']
                )
                # Submit groundtruth annotation
                gt_future = executor.submit(
                    generate_annotation, 
                    item['query'], 
                    item['document'],
                    item['groundtruth']['explanation'], 
                    item['groundtruth']['label']
                )
                futures[pred_future] = ('prediction', item)
                futures[gt_future] = ('groundtruth', item)

            # Process results as they complete
            completed_items = {}
            for future in as_completed(futures):
                annotation_type, item = futures[future]
                try:
                    annotation = future.result()
                    
                    # Initialize dict for this item if not exists
                    item_key = (item['query'], item['document'])
                    if item_key not in completed_items:
                        completed_items[item_key] = item.copy()
                    
                    # Store the annotation
                    completed_items[item_key][annotation_type]['annotation'] = annotation
                    
                    # If both annotations are done, write to file
                    current_item = completed_items[item_key]
                    if ('annotation' in current_item['prediction'] and 
                        'annotation' in current_item['groundtruth']):
                        outfile.write(json.dumps(current_item) + '\n')
                        outfile.flush()
                        del completed_items[item_key]
                        
                except Exception as e:
                    print(f"Error processing {annotation_type} annotation: {e}")

    return True

def evaluate_preference(prompt, chosen, rejected):
    eval_prompt = f"""
    You are an expert at evaluating the quality of responses. Given a prompt and two possible responses (chosen and rejected), 
    determine which response is better and explain your reasoning. Output your evaluation in JSON format.

    PROMPT: {prompt}
    
    RESPONSE A (CHOSEN):
    {chosen}
    
    RESPONSE B (REJECTED):
    {rejected}

    Output your evaluation in this JSON format, nothing else in your response:
    {{
        "explanation": string explaining which response is better and why,
        "selection": "chosen" or "rejected"
    }}
    """
    response = llm_api(eval_prompt)
    response = response.replace("```json", "").replace("```", "")
    return json.loads(response)

def process_preference_pairs(input_file, output_file, sample_size=100):
    # Read already processed items
    processed_prompts = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                processed_prompts.add(item['prompt'])
    except FileNotFoundError:
        pass

    # Read and sample input data
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Filter unprocessed items and sample
    unprocessed_data = [item for item in data if item['prompt'] not in processed_prompts]
    sample_data = random.sample(unprocessed_data, min(sample_size, len(unprocessed_data)))
    
    print(f"Processing {len(sample_data)} samples")

    # Process samples
    with open(output_file, 'a') as outfile:
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for item in sample_data:
                future = executor.submit(
                    evaluate_preference,
                    item['prompt'],
                    item['chosen'],
                    item['rejected']
                )
                futures.append((future, item))

            for future, item in futures:
                try:
                    evaluation = future.result()
                    output_item = {
                        'prompt': item['prompt'],
                        'chosen': item['chosen'],
                        'rejected': item['rejected'],
                        'evaluation': evaluation
                    }
                    outfile.write(json.dumps(output_item) + '\n')
                    outfile.flush()
                except Exception as e:
                    print(f"Error processing preference pair: {e}")

    return True

# Update main block
if __name__ == "__main__":
    # Choose which mode to run
    mode = "preferences"  # or "annotations"
    
    if mode == "preferences":
        processed_data = process_preference_pairs(
            'explanations_preference_pairs_v2.jsonl',
            'explanations_preference_pairs_evaluated_gpt4o.jsonl'
        )
    else:
        processed_data = process_annotations(
            'explanations.jsonl',
            'explanations_annotated.jsonl'
        )