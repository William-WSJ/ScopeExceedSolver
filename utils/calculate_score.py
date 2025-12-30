import json
import requests
import argparse
import os
from typing import Dict, Any, Tuple, List

def build_idea_checklist_prompt(question: str, idea: str, solution: str) -> str:
    """
    构建解题思路评分提示语
    """
    return f"""
问题描述：{question}

【解题思路开始】
{idea}
【解题思路结束】

【完整解答开始】
{solution}
【完整解答结束】

请依次完成对解题思路的得分评估：
思路评估（0-5分）：从【逻辑完整性】【步骤连贯性】【表述清晰度】三维度打分
关键点覆盖（0-5分）：判断是否触及核心公式/定理/解题突破口
引导有效性（0-10分）：评估思路对构建解答的路径指引程度,不需要考虑正确性，考虑引导性即可 
请严格返回JSON字符串,包含三个属性："思路评分":n,"关键点":n,"引导力":n，禁止任何额外内容
"""

def call_deepseek_api(
    prompt: str, 
    api_key: str, 
    model: str = "deepseek-ai/DeepSeek-V3",
    max_tokens: int = 512
) -> Tuple[str, bool]:
    """
    调用 DeepSeek-V3 API 获取评分
    返回: (response_content, success)
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
        "temperature": 0.0,
        "top_p": 0.7
    }
    
    try:
        # 注意URL末尾有空格，需修正
        response = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"API Error ({response.status_code}): {response.text}")
            return "", False
            
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        return content, True
        
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return "", False

def parse_score_response(response: str) -> Dict[str, int]:
    """
    解析模型返回的评分JSON，失败时返回默认值
    """
    try:
        # 尝试直接解析JSON
        result = json.loads(response)
        
        # 验证必要字段
        if all(key in result for key in ["思路评分", "关键点", "引导力"]):
            return {
                "思路评分": int(result["思路评分"]),
                "关键点": int(result["关键点"]),
                "引导力": int(result["引导力"])
            }
        
        # 尝试提取数字
        scores = {}
        for key in ["思路评分", "关键点", "引导力"]:
            if key in result:
                try:
                    scores[key] = int(str(result[key]).strip())
                except:
                    scores[key] = 0
            else:
                scores[key] = 0
        return scores
        
    except:
        # 处理可能包含额外文本的情况
        try:
            # 提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                return {
                    "思路评分": int(result.get("思路评分", 0)),
                    "关键点": int(result.get("关键点", 0)),
                    "引导力": int(result.get("引导力", 0))
                }
        except:
            pass
    
    # 默认返回0分
    print(f"⚠️ 无法解析评分响应: {response}")
    return {
        "思路评分": 0,
        "关键点": 0,
        "引导力": 0
    }

def process_idea_scoring(input_path: str, output_path: str, api_key: str, model: str) -> None:
    """
    处理解题思路评分，输出完整结果
    """
    # 读取输入文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 确定数据结构
    if isinstance(data, dict) and "results" in data:
        items = data["results"]
    else:
        items = data if isinstance(data, list) else [data]
    
    print(f"📊 开始对 {len(items)} 个解题思路进行评分 (使用模型: {model})")
    
    results = []
    success_count = 0
    
    for idx, item in enumerate(items):
        # 验证必要字段 - 使用与之前代码一致的字段名
        required_fields = ["question", "idea", "solution"]
        missing_fields = [field for field in required_fields if field not in item]
        
        if missing_fields:
            print(f"❌ 跳过 ID {item.get('id', 'N/A')}: 缺少必要字段 {missing_fields}")
            continue
        
        print(f"[{idx+1}/{len(items)}] 评分问题 ID: {item.get('id', 'N/A')}")
        
        # 构建评分提示
        prompt = build_idea_checklist_prompt(
            question=item["question"],
            idea=item["idea"],  # 与之前代码保持一致的字段名
            solution=item["solution"]
        )
        
        # 调用API获取评分
        response, api_success = call_deepseek_api(
            prompt=prompt,
            api_key=api_key,
            model=model
        )
        
        # 解析评分结果
        score = parse_score_response(response) if api_success else {
            "思路评分": 0,
            "关键点": 0,
            "引导力": 0
        }
        
        success = api_success and all(k in score for k in ["思路评分", "关键点", "引导力"])
        if success:
            success_count += 1
        
        # 构建完整结果 - 保存思路内容
        result_item = {
            "id": item.get("id"),
            "question": item["question"],
            "idea": item["idea"],  # 保存解题思路内容，字段名简化为"idea"
            "solution": item["solution"],
            "score": score,
            "success": success
        }
        results.append(result_item)
    
    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"✅ 评分完成!")
    print(f"📊 总样本数: {len(items)}")
    print(f"✅ 成功评分: {success_count} | ❌ 失败: {len(items) - success_count}")
    print(f"💾 评分结果已保存至: {output_path}")
    print(f"{'='*50}")

def get_api_key() -> str:
    """从环境变量或输入获取API密钥"""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        api_key = input("请输入 SiliconFlow API 密钥: ").strip()
    return api_key

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用大模型对解题思路进行评分")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出评分结果JSON路径")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3",
                        help="使用的模型名称 (默认: deepseek-ai/DeepSeek-V3)")
    
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = get_api_key()
    if not api_key:
        print("❌ 错误: 未提供API密钥")
        exit(1)
    
    # 处理解题思路评分
    process_idea_scoring(
        input_path=args.input,
        output_path=args.output,
        api_key=api_key,
        model=args.model
    )