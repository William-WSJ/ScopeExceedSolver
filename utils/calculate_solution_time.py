import json
import requests
import argparse
import time
import os
from typing import Dict, Any, Tuple, List, Union

def build_prompt(question: str, idea: str, limitations: List[str]) -> str:
    """
    Construct complete prompt with predefined template
    """
    limitations_str = "\n".join([f"- {limit}" for limit in limitations])
    return f"""
请根据我的解题思路解决以下问题：
{question}

我的解题思路如下：
{idea}

你的解答过程中严禁出现以下内容或方法：
{limitations_str}

请注意，如果我的解题思路有明显的错误，请纠正后再解答。
"""

def call_deepseek_api(
    prompt: str, 
    api_key: str, 
    base_url: str = "https://api.siliconflow.cn", 
    model: str = "deepseek-ai/DeepSeek-V3",
    max_tokens: int = 2048
) -> Tuple[str, float]:
    """
    Invoke DeepSeek-V3 API to generate solution
    Return: (response text, request duration in seconds)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "thinking_budget": 4096,
        "min_p": 0.05,
        "stop": None,
        "temperature": 0.0,  # Ensure deterministic output
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60  # Request timeout: 60 seconds
        )
        duration = time.time() - start_time
        
        if response.status_code != 200:
            print(f"API Error ({response.status_code}): {response.text}")
            return "", duration
            
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"Request failed: {str(e)}")
        return "", duration

def process_dataset(
    input_path: str, 
    output_path: str, 
    idea_key: str, 
    api_key: str, 
    model: str,
    max_tokens: int
) -> float:
    """
    Process full dataset and calculate average request latency
    """
    # Load dataset file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Standardize data to list format
    items = data if isinstance(data, list) else [data]
    
    results = []
    total_duration = 0.0
    successful_count = 0
    total_items = len(items)
    
    print(f"🚀 Start processing {total_items} questions (Model: {model})")
    print(f"💡 Idea field key: '{idea_key}' | ⚠️ Restriction field key: 'cautions'")
    
    for idx, item in enumerate(items):
        # Check required fields existence
        if idea_key not in item:
            print(f"❌ Skip ID {item.get('id', 'N/A')}: Missing field '{idea_key}'")
            continue
            
        if "grade_cautions" not in item:
            print(f"❌ Skip ID {item.get('id', 'N/A')}: Missing field 'cautions'")
            continue
            
        if "question" not in item:
            print(f"❌ Skip ID {item.get('id', 'N/A')}: Missing field 'question'")
            continue
        
        # Assemble prompt content
        prompt = build_prompt(
            question=item["question"],
            idea=item[idea_key],
            limitations=item["cautions"]
        )
        
        print(f"\n[{idx+1}/{total_items}] Processing Question ID: {item.get('id', 'N/A')}")
        print(f"❓ Question preview: {item['question'][:50]}{'...' if len(item['question']) > 50 else ''}")
        
        # Send API request
        solution, duration = call_deepseek_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens
        )
        
        # Record request output
        result = {
            "id": item.get("id"),
            "question": item["question"],
            "idea_used": item[idea_key],
            "limitations_used": item["cautions"],
            "solution": solution,
            "duration_seconds": duration,
            "success": bool(solution)
        }
        results.append(result)
        
        if solution:
            successful_count += 1
            total_duration += duration
            print(f"✅ Success | Latency: {duration:.4f} seconds")
            print(f"📝 Solution preview: {solution[:100]}{'...' if len(solution) > 100 else ''}")
        else:
            print(f"❌ Failed | Latency: {duration:.4f} seconds")
    
    # Calculate average latency
    avg_duration = total_duration / successful_count if successful_count > 0 else 0
    
    # Assemble output data with metadata
    output_data = {
        "metadata": {
            "input_file": input_path,
            "idea_key": idea_key,
            "model": model,
            "max_tokens": max_tokens,
            "total_items": total_items,
            "successful_items": successful_count,
            "failed_items": total_items - successful_count,
            "average_duration_seconds": avg_duration,
            "api_endpoint": "https://api.siliconflow.cn/v1/chat/completions"
        },
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"✅ All tasks finished!")
    print(f"📊 Total samples: {total_items}")
    print(f"✅ Succeeded: {successful_count} | ❌ Failed: {total_items - successful_count}")
    print(f"⏱️ Average latency: {avg_duration:.4f} seconds per question")
    print(f"💾 Output saved to: {output_path}")
    print(f"{'='*50}")
    
    return avg_duration

def get_api_key() -> str:
    """Read API key from environment variable or user input"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        api_key = input("Please input your SiliconFlow API key: ").strip()
    return api_key

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate DeepSeek-V3 API inference latency via SiliconFlow")
    parser.add_argument("--input", type=str, required=True, help="File path of input JSON dataset")
    parser.add_argument("--output", type=str, required=True, help="File path of output result JSON")
    parser.add_argument("--idea-key", type=str, default="thought", 
                        help="Key name of reasoning field in JSON (default: thought)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3",
                        help="Target model name (default: deepseek-ai/DeepSeek-V3)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Maximum generation token limit (default: 2048)")
    
    args = parser.parse_args()
    
    # Acquire API key
    api_key = get_api_key()
    if not api_key:
        print("❌ Error: No valid API key provided")
        exit(1)
    
    print(f"🔧 Runtime Config: Model={args.model} | Max Tokens={args.max_tokens} | Idea Field={args.idea_key}")
    
    # Start dataset processing pipeline
    avg_time = process_dataset(
        input_path=args.input,
        output_path=args.output,
        idea_key=args.idea_key,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens
    )