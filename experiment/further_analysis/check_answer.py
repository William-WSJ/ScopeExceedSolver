import json
from openai import OpenAI

# 初始化Ollama客户端
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

INPUT_FILE = "/home/wangsijin/with_idea_limitations/syllabus_qwen-7b_full_gemini3.json"
OUTPUT_FILE = "syllabus_qwen-7b_full_gemini3.json"

def check_answer(question, answer, solution):
    """判断答案是否一致，严格只返回True/False"""
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
        return result == "True"  # 严格匹配"True"字符串
    except:
        return False  # 任何错误都返回False

# 1. 读取JSON文件
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 逐条处理
for item in data:
    # 跳过无效数据
    if not all(k in item for k in ["question", "answer", "solution"]):
        item["acc"] = False
        continue
        
    # 获取判断结果
    item["acc"] = check_answer(
        item["question"],
        item["answer"],
        item["solution"]
    )
    print(f"问题ID: {item.get('id', 'N/A')} - 答案一致: {item['acc']}")

# 3. 保存结果（覆盖原文件或另存）
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"处理完成！结果已保存至 questions_with_acc.json")
print(f"统计: {sum(1 for item in data if item.get('acc', False))}/{len(data)} 条答案一致")