import json
import requests
import argparse
import time  # 导入时间模块

def call_llama_factory_api(question: str, base_url: str = "http://127.0.0.1:8000", max_tokens: int = 2048) -> tuple:
    """
    调用 LLaMA-Factory 的 OpenAI 兼容 API 获取模型输出。
    返回: (response_content, duration_in_seconds)
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

    start_time = time.time()  # 记录开始时间
    
    try:
        response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        duration = time.time() - start_time  # 计算耗时
        return content, duration
    except Exception as e:
        duration = time.time() - start_time  # 即使出错也记录耗时
        print(f"Error calling API for question: {question}\nError: {e}")
        return "", duration

def process_json_file(input_path: str, output_path: str, base_url: str = "http://127.0.0.1:8000"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持单个对象或列表
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
        
        # 只有成功获取到thought才计入统计
        if thought:
            total_duration += duration
            successful_calls += 1
            item["thought"] = thought
            updated_items.append(item)
            print(f"  ✓ Completed in {duration:.4f} seconds")
        else:
            print(f"  ✗ Failed after {duration:.4f} seconds")

    # 计算平均耗时
    if successful_calls > 0:
        avg_duration = total_duration / successful_calls
        print(f"\n✅ Successfully processed {successful_calls}/{len(items)} items")
        print(f"⏱️  Average generation time: {avg_duration:.4f} seconds per question")
    else:
        print("\n❌ No items were successfully processed")
        avg_duration = 0.0

    # 写回
    if updated_items:  # 只有成功处理的项目才写入
        with open(output_path, "w", encoding="utf-8") as f:
            if is_single:
                json.dump(updated_items[0], f, ensure_ascii=False, indent=2)
            else:
                json.dump(updated_items, f, ensure_ascii=False, indent=2)
        print(f"💾 Output saved to {output_path}")
    else:
        print("⚠️  No output file generated due to processing failures")

    # 返回平均耗时供进一步分析使用
    return avg_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call LLaMA-Factory API to add 'thought' field to JSON and measure average generation time.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="LLaMA-Factory API base URL")

    args = parser.parse_args()
    
    # 执行处理并获取平均耗时
    avg_time = process_json_file(args.input, args.output, base_url=args.url)
    
    # 将平均耗时保存到单独的统计文件（可选）
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