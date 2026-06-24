import json
import argparse
import time
import os
import openai
from typing import List, Dict, Any, Tuple

# ====== 全局配置区 ======
API_KEY = ""  
BASE_URL = "https://aihubmix.com/v1"
MODEL = "gpt-5.2"
MAX_TOKENS = 2048
REQUEST_DELAY = 0.5  # 请求间隔时间(秒)
# ========================

def create_client():
    return openai.OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL.strip()
    )

def call_gpt52_api(client: openai.OpenAI, prompt: str) -> Tuple[str, float]:
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
        print(f"API 请求失败: {str(e)}")
        return "", time.time() - start_time

def build_prompt(item: Dict) -> str:
    """构建带思路和限制条件的 Prompt"""
    question = item.get("question", "")
    idea = item.get("thought", "")
    # 优先级：cautions > grade_cautions
    limitations_list = item.get("grade_cautions", [])
    limitations_str = "\n".join([f"- {limit}" for limit in limitations_list]) if limitations_list else "无"

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
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f).get("processed_count", 0)
        except:
            return 0
    return 0

def save_checkpoint(checkpoint_path: str, processed_count: int):
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({"processed_count": processed_count}, f)

def save_results(output_path: str, results: List[Dict]):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def process_dataset(input_path: str, output_path: str, checkpoint_path: str):
    client = create_client()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data if isinstance(data, list) else [data]
    total_items = len(items)
    
    processed_count = load_checkpoint(checkpoint_path)
    
    # 初始化或加载已有结果
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
    
    print(f"🚀 开始处理题目，总计: {total_items} | 已跳过: {processed_count}")
    
    for idx in range(processed_count, total_items):
        item = items[idx]
        prompt = build_prompt(item)

        print(f"[{idx+1}/{total_items}] 处理 ID: {item.get('id', 'N/A')}", end=" ")
        
        solution, duration = call_gpt52_api(client, prompt)
        
        # 只记录 solution
        results[idx]["solution"] = solution
        
        if solution:
            successful_count += 1
            total_duration += duration
            print(f"✅ 耗时: {duration:.2f}s")
        else:
            print(f"❌ 失败")
        
        # 实时保存结果与断点
        save_results(output_path, results)
        processed_count += 1
        save_checkpoint(checkpoint_path, processed_count)
        
        if idx < total_items - 1:
            time.sleep(REQUEST_DELAY)
    
    # 计算平均时间并保存统计
    avg_duration = total_duration / successful_count if successful_count > 0 else 0
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({"average_time_seconds": avg_duration}, f, ensure_ascii=False, indent=2)

    print(f"\n任务完成！平均耗时: {avg_duration:.2f}s | 统计已保存至: {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--output", required=True, help="输出文件路径")
    args = parser.parse_args()
    
    ckpt_path = args.output.replace('.json', '_checkpoint.json')
    process_dataset(args.input, args.output, ckpt_path)