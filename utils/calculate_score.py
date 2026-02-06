import json
import requests
import argparse
import os
from typing import Dict, Any, Tuple, List

def build_idea_checklist_prompt(question: str, thought: str, solution: str, acc: bool) -> str:
    """
    全量 22 子项 Prompt：
    1. 严格映射 LaTeX 22 个子项的文字描述。
    2. 引入 acc 字段辅助判断 solution 正确性。
    3. 明确以 solution 为基准评估 thought。
    """
    acc_status = "正确" if acc else "错误"
    
    return f"""你是一名极其严谨的数学教育评估专家。
请结合【问题】与提供的【完整解答】及其【正确性标识】，对【解题思路 (thought)】进行深度评估。

### 【评估基准】
- **问题 (Question)**: {question}
- **完整解答 (Solution)**: {solution} (参考基准)
- **解答正确性 (acc)**: {acc} (标识该 solution 事实上是{acc_status}的)
- **待评估思路 (Thought)**: {thought} (评估对象)

---

### 【22 项评分标准清单】

#### 维度一：思路评分 (Logical & Structural, 0-5分)
1) 推理完整性：推理步骤是否形成完整闭环。
2) 逻辑一致性：前提与结论是否保持一致。
3) 严密性：是否存在逻辑跳跃或矛盾。
4) 规则性：是否遵循基本逻辑规则。
5) 认知逻辑：步骤序列是否符合认知逻辑。
6) 自然过渡：相邻步骤间过渡是否自然。
7) 精简性：是否存在冗余或重复步骤。
8) 层次性：整体结构是否具有层次性。
9) 规范性：数学符号使用是否规范。
10) 术语清晰：关键术语是否定义清晰。
11) 表达准确：语言表达是否简洁准确。
12) 推导依据：公式推导过程是否清晰。

#### 维度二：关键点覆盖 (Key Point Coverage, 0-5分)
13) 核心识别：是否识别题目核心考查点。
14) 公式应用：是否应用关键公式/定理。
15) 突破路径：是否找到解题突破口。
16) 全局性：关键点是否贯穿解题全过程。

#### 维度三：引导力有效性 (Guidance Effectiveness, 0-10分)
17) 方向明确：解题方向是否清晰明确。
18) 分步引导：是否提供分步引导。
19) 框架构建：是否建立可操作的解题框架。
20) 可操作性：引导内容是否具有实际操作性（不考虑正确性）。

#### 维度四：解答正确性 (Solution Correctness, 0-10分)
21) 答案正确：最终答案是否正确。
22) 步骤正确：解题步骤是否正确。

---

### 【输出要求】
请以 JSON 格式返回评分结果。禁止任何额外文字。格式如下：
{{
  "思路评分": 整数(0-5),
  "关键点": 整数(0-5),
  "引导力": 整数(0-10),
  "正确性": 整数(0-10)
}}"""

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