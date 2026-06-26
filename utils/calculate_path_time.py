import json
import requests
import argparse
import time  # Import time module

def call_llama_factory_api(question: str, base_url: str = "http://127.0.0.1:8000", max_tokens: int = 2048) -> tuple:
    """
    Invoke the OpenAI-compatible API of LLaMA-Factory to get model output.
    Return: (response_content, duration_in_seconds)
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama",
        "messages": [
            {"role": "system", "content": "请给出下列问题的解题思路"},
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }

    start_time = time.time()  # Record start timestamp
    
    try:
        response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        duration = time.time() - start_time  # Calculate request latency
        return content, duration
    except Exception as e:
        duration = time.time() - start_time  # Record latency even when request fails
        print(f"Error calling API for question: {question}\nError: {e}")
        return "", duration

def process_json_file(input_path: str, output_path: str, base_url: str = "http://127.0.0.1:8000"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Compatible with single JSON object or JSON array format
    if isinstance(data, dict):
        items = [data]
        is_single = True
    else:
        items = data
        is_single = False

    updated_items = []
    total_duration = 0.0
    successful_calls = 0
    
    for item in items:
        question = item["question"]
        print(f"Processing ID: {item.get('id', 'N/A')} ...")
        thought, duration = call_llama_factory_api(question, base_url=base_url, max_tokens=2048)
        
        # Only count successful responses into statistics
        if thought:
            total_duration += duration
            successful_calls += 1
            item["thought"] = thought
            updated_items.append(item)
            print(f"  ✓ Completed in {duration:.4f} seconds")
        else:
            print(f"  ✗ Failed after {duration:.4f} seconds")

    # Calculate average inference latency
    if successful_calls > 0:
        avg_duration = total_duration / successful_calls
        print(f"\n✅ Successfully processed {successful_calls}/{len(items)} items")
        print(f"⏱️  Average generation time: {avg_duration:.4f} seconds per question")
    else:
        print("\n❌ No items were successfully processed")
        avg_duration = 0.0

    # Write processed data to file
    if updated_items:  # Only save file if there are valid processed samples
        with open(output_path, "w", encoding="utf-8") as f:
            if is_single:
                json.dump(updated_items[0], f, ensure_ascii=False, indent=2)
            else:
                json.dump(updated_items, f, ensure_ascii=False, indent=2)
        print(f"💾 Output saved to {output_path}")
    else:
        print("⚠️  No output file generated due to all processing failures")

    # Return average latency for subsequent analysis
    return avg_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call LLaMA-Factory API to append 'thought' field to JSON data and calculate average inference time.")
    parser.add_argument("--input", type=str, required=True, help="File path of input JSON")
    parser.add_argument("--output", type=str, required=True, help="File path of output JSON")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base URL of LLaMA-Factory API service")

    args = parser.parse_args()
    
    # Run dataset processing pipeline and get average latency
    avg_time = process_json_file(args.input, args.output, base_url=args.url)
    
    # Save latency metrics to separate statistics file (optional feature)
    stats_file = args.output.replace(".json", "_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "average_generation_time_seconds": avg_time,
            "input_file": args.input,
            "output_file": args.output,
            "api_url": args.url,
            "total_items_processed": len(json.load(open(args.input, "r", encoding="utf-8"))),
            "successful_items": len(json.load(open(args.output, "r", encoding="utf-8"))) if avg_time > 0 else 0
        }, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Statistics saved to {stats_file}")