import json
import requests
import argparse
import time
import os
from typing import Dict, Any, Tuple, List, Union

def build_prompt(question: str, idea: str, limitations: List[str]) -> str:
    """
    根据提供的模板构建完整的提示语
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
    调用 DeepSeek-V3 API 获取解答
    返回: (response_content, duration_in_seconds)
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
        "temperature": 0.0,  # 保持确定性输出
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
            timeout=60  # 60秒超时
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
    处理整个数据集，统计平均耗时
    """
    # 读取数据集
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 确保是列表格式
    items = data if isinstance(data, list) else [data]
    
    results = []
    total_duration = 0.0
    successful_count = 0
    total_items = len(items)
    
    print(f"🚀 开始处理 {total_items} 个问题 (使用模型: {model})")
    print(f"💡 Idea 字段: '{idea_key}' | ⚠️ 限制条件字段: 'cautions'")
    
    for idx, item in enumerate(items):
        # 验证必要字段
        if idea_key not in item:
            print(f"❌ 跳过 ID {item.get('id', 'N/A')}: 缺少 '{idea_key}' 字段")
            continue
            
        if "grade_cautions" not in item:
            print(f"❌ 跳过 ID {item.get('id', 'N/A')}: 缺少 'cautions' 字段")
            continue
            
        if "question" not in item:
            print(f"❌ 跳过 ID {item.get('id', 'N/A')}: 缺少 'question' 字段")
            continue
        
        # 构建提示语
        prompt = build_prompt(
            question=item["question"],
            idea=item[idea_key],
            limitations=item["cautions"]
        )
        
        print(f"\n[{idx+1}/{total_items}] 处理问题 ID: {item.get('id', 'N/A')}")
        print(f"❓ 问题: {item['question'][:50]}{'...' if len(item['question']) > 50 else ''}")
        
        # 调用 API
        solution, duration = call_deepseek_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens
        )
        
        # 记录结果
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
            print(f"✅ 成功 | 耗时: {duration:.4f} 秒")
            print(f"📝 解答: {solution[:100]}{'...' if len(solution) > 100 else ''}")
        else:
            print(f"❌ 失败 | 耗时: {duration:.4f} 秒")
    
    # 计算平均耗时
    avg_duration = total_duration / successful_count if successful_count > 0 else 0
    
    # 保存结果
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
    print(f"✅ 处理完成!")
    print(f"📊 总样本数: {total_items}")
    print(f"✅ 成功: {successful_count} | ❌ 失败: {total_items - successful_count}")
    print(f"⏱️ 平均耗时: {avg_duration:.4f} 秒/题")
    print(f"💾 结果已保存至: {output_path}")
    print(f"{'='*50}")
    
    return avg_duration

def get_api_key() -> str:
    """从环境变量或输入获取API密钥"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        api_key = input("请输入 SiliconFlow API 密钥: ").strip()
    return api_key

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 DeepSeek-V3 API 解题耗时 (SiliconFlow)")
    parser.add_argument("--input", type=str, required=True, help="输入JSON数据集路径")
    parser.add_argument("--output", type=str, required=True, help="输出结果JSON路径")
    parser.add_argument("--idea-key", type=str, default="thought", 
                        help="JSON中idea字段的键名 (默认: thought)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3",
                        help="使用的模型名称 (默认: deepseek-ai/DeepSeek-V3)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="最大生成token数 (默认: 2048)")
    
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = get_api_key()
    if not api_key:
        print("❌ 错误: 未提供API密钥")
        exit(1)
    
    print(f"🔧 配置: 模型={args.model} | 最大token={args.max_tokens} | Idea字段={args.idea_key}")
    
    # 处理数据集
    avg_time = process_dataset(
        input_path=args.input,
        output_path=args.output,
        idea_key=args.idea_key,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens
    )