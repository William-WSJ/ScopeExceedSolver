import json
import argparse
import time
import os
import openai
from typing import List, Dict, Any, Tuple

# ====== 全局配置区 ======
API_KEY = ""  # 已替换为您的 API 密钥
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
MAX_TOKENS = 2048
REQUEST_DELAY = 0.5  # 请求间隔时间(秒)
# ====== 全局配置区结束 ======

def create_client():
    """创建 OpenAI 客户端"""
    return openai.OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

def call_model_api(client: openai.OpenAI, prompt: str) -> Tuple[str, float]:
    """
    调用 API 获取解答，并返回完整文本和总耗时
    """
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        content = response.choices[0].message.content.strip()
        # 实时打印输出，方便观察过程
        print(f"\n[模型回复]:\n{content}")
        
        duration = time.time() - start_time
        return content, duration

    except Exception as e:
        print(f"\nAPI请求失败: {str(e)}")
        return "", time.time() - start_time

def build_thought_limitations_prompt(item: Dict) -> str:
    """构建结合思路、问题和限制条件的 Prompt"""
    question = item.get("question", "")
    idea = item.get("thought", "")
    
    # 优先级：cautions > grade_cautions
    limitations_list = item.get("grade_cautions", [])
    limitations_str = "\n".join([f"- {limit}" for limit in limitations_list]) if limitations_list else "无"

    # 严格遵循用户要求的提示语结构
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
    """加载 checkpoint"""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f).get("processed_count", 0)
        except:
            return 0
    return 0

def save_checkpoint(checkpoint_path: str, processed_count: int):
    """保存 checkpoint"""
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({"processed_count": processed_count}, f)

def save_results(output_path: str, results: List[Dict]):
    """保存结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def process_dataset(input_path: str, output_path: str, checkpoint_path: str):
    """核心处理逻辑"""
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
        for item in results:
            item["solution"] = ""

    total_duration = 0.0
    successful_count = 0
    
    print(f"🚀 开始处理题目 (模型: {MODEL}) | 总计: {total_items} | 起始位置: {processed_count}")

    for idx in range(processed_count, total_items):
        item = items[idx]
        prompt = build_thought_limitations_prompt(item)
        
        print(f"\n[{idx+1}/{total_items}] 处理问题 ID: {item.get('id', 'N/A')}")
        
        solution, duration = call_model_api(client, prompt)
        
        # 只记录 solution 字段
        results[idx]["solution"] = solution
        
        if solution:
            successful_count += 1
            total_duration += duration
            print(f"✅ 成功 | 耗时: {duration:.2f} 秒")
        else:
            print(f"❌ 失败")
        
        # 实时保存
        save_results(output_path, results)
        processed_count += 1
        save_checkpoint(checkpoint_path, processed_count)
        
        if idx < total_items - 1:
            time.sleep(REQUEST_DELAY)
    
    # 保存统计报告
    avg_duration = total_duration / successful_count if successful_count > 0 else 0
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({"average_time_seconds": avg_duration}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}\n✅ 处理完成！\n⏱️ 平均耗时: {avg_duration:.4f} 秒/题\n💾 统计已保存至: {stats_path}\n{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用指定模型处理数据集 (Thought + Limitations)")
    parser.add_argument("--input", type=str, required=True, help="输入JSON路径")
    parser.add_argument("--output", type=str, required=True, help="输出JSON路径")
    
    args = parser.parse_args()
    
    checkpoint_file = args.output.replace('.json', '_checkpoint.json')
    
    process_dataset(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=checkpoint_file
    )