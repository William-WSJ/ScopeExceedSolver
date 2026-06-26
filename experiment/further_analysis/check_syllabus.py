import json
from openai import OpenAI

# Initialize Ollama client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

INPUT_FILE = "/home/wangsijin/final/syllabus_qwen-7b_full_gpt5_2.json"
OUTPUT_FILE = "syllabus_qwen-7b_full_gpt5_2.json"

def check_answer(question, answer, solution):
    """Judge whether answers are consistent, only return True/False strictly"""
    prompt = f"""
请帮我判断这个问题的答案是否与我给出的答案一致

问题：{question}
正确答案：{answer}
我的解答：{solution}

注意：
- 正确答案和我的答案可能在形式上不一样，你需要仔细辨别后再给出回答；
- 如果我的解答没有明确给出答案或者解答中出现乱码等与题目无关的内容且影响最终判断，则返回False。
请不要返回任何其他内容，只返回True或者False即可。True表示答案一致，False表示答案不一致。
"""
    try:
        response = client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            timeout=30.0
        )
        result = response.choices[0].message.content.strip()
        return "True" in result  # Compatibility processing: return true if output contains True
    except:
        return False

def check_exceeds_scope(solution, limitations):
    """
    Judge whether the solution exceeds scope based on solution and limitation list
    Adopt Few-shot examples to improve model judgment accuracy
    """
    if not limitations:
        return False, "无限制列表，默认不超纲"

    # Format limitation list
    limitations_str = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(limitations)])

    prompt = f"""你是一个严格的逻辑审查员。请判断提供的“解答过程”是否涉及了“限制列表”中的任何内容。

### 限制列表：
{limitations_str}

### 判定示例（Few-shot）：

示例 1：
限制列表：1. 出现乘除法； 2. 使用小数
解答过程：3 + 3 + 3 = 9
判定：{{"exceeds": false, "reason": "解答中仅使用了加法，未涉及限制列表中的乘除法或小数。"}}

示例 2：
限制列表：1. 使用方程或字母设未知数； 2. 结果使用分数
解答过程：设一共有 x 支笔，x - 2 = 8，所以 x = 10。
判定：{{"exceeds": true, "reason": "违反了规则1：出现了字母 x 和设方程的思维。"}}

示例 3：
限制列表：1. 出现乘除法； 2. 出现题目中没有的字母
解答过程：2 * 5 = 10，答：一共10个。
判定：{{"exceeds": true, "reason": "违反了规则1：解答过程中使用了乘法符号 '*'。"}}

示例 4：
限制列表：1. 使用负数； 2. 使用绝对值
解答过程：100 - 90 = 10。
判定：{{"exceeds": false, "reason": "解答符合要求，未涉及负数或绝对值。"}}

### 待审核任务：
解答过程：{solution}

请仅返回 JSON 格式：
{{
  "exceeds": true/false,
  "reason": "简述理由，若超纲请指明违反了第几条"
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[
                {"role": "system", "content": "你是一个只输出 JSON 格式的自动化合规检查工具。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        res = json.loads(response.choices[0].message.content)
        return res.get("exceeds", False), res.get("reason", "")
    except:
        return True, "判定过程异常，默认设为超纲"

# 1. Load JSON file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Process items one by one
for i, item in enumerate(data):
    # --- Logic A: Answer correctness check ---
    # if not all(k in item for k in ["question", "answer", "solution"]):
    #     item["acc"] = False
    # else:
    #     item["acc"] = check_answer(item["question"], item["answer"], item["solution"])

    # --- Logic B: Out-of-scope check ---
    # Prioritize cautions field, fall back to grade_cautions if missing
    current_limitations = item.get("cautions")
    solution_text = item.get("solution", "")
    
    exceeds, reason = check_exceeds_scope(solution_text, current_limitations)
    item["exceeds_scope"] = exceeds
    item["exceeds_reason"] = reason

    # Print real-time processing progress
    scope_status = "❌ Out of Scope" if exceeds else "✅ Compliant"
    # acc_status = "✔ Correct" if item["acc"] else "✘ Incorrect"
    print(f"[{i+1}/{len(data)}] ID: {item.get('id')} | Answer Match: {item.get('acc')} | Status: {scope_status} | Reason: {reason}")

# 3. Save processed results
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nProcessing completed! Results saved to {OUTPUT_FILE}")
print(f"Out-of-scope statistics: {sum(1 for item in data if item.get('exceeds_scope', False))}/{len(data)}")