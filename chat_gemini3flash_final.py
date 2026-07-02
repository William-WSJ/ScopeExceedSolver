import json
import argparse
import time
import os
import openai
from typing import List, Dict, Any, Tuple

# ====== Global Configuration Section (Set all parameters here) ======
API_KEY = ""  # Replace with your actual API key
BASE_URL = "https://aihubmix.com/v1"
MODEL = "gemini-3-flash-preview"
MAX_TOKENS = 1024
REQUEST_DELAY = 0.5  # Request interval in seconds
# ====== End of Global Configuration Section ======

def create_client():
    """Create OpenAI client with global configuration"""
    return openai.OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL.strip()
    )

def call_gemini_api(client: openai.OpenAI, prompt: str) -> Tuple[str, float]:
    """Call Gemini API to get answer and print response in real time"""
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=MAX_TOKENS,
        )
        content = response.choices[0].message.content
        print(f"\n[Gemini Response]:\n{content}")
        duration = time.time() - start_time
        return content, duration
    except Exception as e:
        print(f"API request failed: {str(e)}")
        return "", time.time() - start_time

def build_thought_limitations_prompt(item: Dict) -> str:
    """Construct prompt based on question, reasoning and restrictions"""
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
    """Load processing checkpoint"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f).get("processed_count", 0)
        except:
            return 0
    return 0

def save_checkpoint(checkpoint_path: str, processed_count: int):
    """Save processing checkpoint"""
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({"processed_count": processed_count}, f)

def save_results(output_path: str, results: List[Dict]):
    """Save output results to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def process_dataset(input_path: str, output_path: str, checkpoint_path: str):
    """Core logic for dataset processing"""
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
        for item in results:
            item["solution"] = ""

    total_duration = 0.0
    successful_count = 0
    
    print(f"🚀 Start processing tasks (Model: {MODEL}) | Total items: {total_items} | Start index: {processed_count}")
    
    for idx in range(processed_count, total_items):
        item = items[idx]
        prompt = build_thought_limitations_prompt(item)
        
        print(f"\n[{idx+1}/{total_items}] Processing question ID: {item.get('id', 'N/A')}")
        
        solution, duration = call_gemini_api(client, prompt)
        
        # Only write generated content to the `solution` field.
        results[idx]["solution"] = solution
        
        if solution:
            successful_count += 1
            total_duration += duration
            print(f"✅ Success | Time cost: {duration:.2f} seconds")
        else:
            print(f"❌ Failed")
        
        # Save results and checkpoint after each item.
        save_results(output_path, results)
        processed_count += 1
        save_checkpoint(checkpoint_path, processed_count)
        
        if idx < total_items - 1:
            time.sleep(REQUEST_DELAY)
    
    # Calculate summary statistics.
    avg_duration = total_duration / successful_count if successful_count > 0 else 0
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({"average_time_seconds": avg_duration}, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"✅ Processing completed! Results saved to: {output_path}")
    print(f"⏱️ Average latency: {avg_duration:.4f} seconds per question")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with Gemini (Reasoning + Restrictions)")
    parser.add_argument("--input", type=str, required=True, help="File path of input JSON")
    parser.add_argument("--output", type=str, required=True, help="File path of output JSON")
    
    args = parser.parse_args()
    
    checkpoint_file = args.output.replace('.json', '_checkpoint.json')
    
    process_dataset(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=checkpoint_file
    )
