import json
import argparse
import time
import os
import openai
from typing import List, Dict, Any, Tuple

# ====== Global Configuration Section ======
API_KEY = ""  
BASE_URL = "https://aihubmix.com/v1"
MODEL = "gpt-5.2"
MAX_TOKENS = 2048
REQUEST_DELAY = 0.5  # Request interval in seconds
# =========================================

def create_client():
    """Create an OpenAI-compatible client."""
    return openai.OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL.strip()
    )

def call_gpt52_api(client: openai.OpenAI, prompt: str) -> Tuple[str, float]:
    """Call the GPT endpoint and return content plus latency."""
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=MAX_TOKENS,
        )
        duration = time.time() - start_time
        return response.choices[0].message.content.strip(), duration
    except Exception as e:
        print(f"API request failed: {str(e)}")
        return "", time.time() - start_time

def build_prompt(item: Dict) -> str:
    """Construct prompt with solving ideas and restriction rules"""
    question = item.get("question", "")
    idea = item.get("thought", "")
    # Priority rule: use `grade_cautions` when available.
    limitations_list = item.get("grade_cautions", [])
    limitations_str = "\n".join([f"- {limit}" for limit in limitations_list]) if limitations_list else "None"

    # The prompt body remains in Chinese to preserve the original experiment setting.
    return f"""
请根据我的解题思路解决以下问题：
{question}

我的解题思路如下：
{idea}

你的解答过程中严禁出现以下内容或方法：
{limitations_str}

请注意，如果我的解题思路有明显的错误，请纠正后再解答。
"""

def load_checkpoint(checkpoint_path: str) -> int:
    """Load the saved processing checkpoint."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f).get("processed_count", 0)
        except:
            return 0
    return 0

def save_checkpoint(checkpoint_path: str, processed_count: int):
    """Save the current processing checkpoint."""
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({"processed_count": processed_count}, f)

def save_results(output_path: str, results: List[Dict]):
    """Save processed dataset results."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def process_dataset(input_path: str, output_path: str, checkpoint_path: str):
    """Run the dataset processing pipeline."""
    client = create_client()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data if isinstance(data, list) else [data]
    total_items = len(items)
    
    processed_count = load_checkpoint(checkpoint_path)
    
    # Initialize the result list or resume from an unfinished output file.
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except:
            results = items.copy()
    else:
        results = items.copy()

    total_duration = 0.0
    successful_count = 0
    
    print(f"🚀 Start processing problems, total: {total_items} | Skipped finished items: {processed_count}")
    
    for idx in range(processed_count, total_items):
        item = items[idx]
        prompt = build_prompt(item)

        print(f"[{idx+1}/{total_items}] Processing ID: {item.get('id', 'N/A')}", end=" ")
        
        solution, duration = call_gpt52_api(client, prompt)
        
        # Only store the generated answer in the `solution` field.
        results[idx]["solution"] = solution
        
        if solution:
            successful_count += 1
            total_duration += duration
            print(f"✅ Time cost: {duration:.2f}s")
        else:
            print(f"❌ Failed")
        
        # Save output and checkpoint after each item.
        save_results(output_path, results)
        processed_count += 1
        save_checkpoint(checkpoint_path, processed_count)
        
        if idx < total_items - 1:
            time.sleep(REQUEST_DELAY)
    
    # Calculate average inference time and save summary statistics.
    avg_duration = total_duration / successful_count if successful_count > 0 else 0
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({"average_time_seconds": avg_duration}, f, ensure_ascii=False, indent=2)

    print(f"\nAll tasks completed! Average time cost: {avg_duration:.2f}s | Statistics saved to: {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path of input file")
    parser.add_argument("--output", required=True, help="Path of output file")
    args = parser.parse_args()
    
    ckpt_path = args.output.replace('.json', '_checkpoint.json')
    process_dataset(args.input, args.output, ckpt_path)
