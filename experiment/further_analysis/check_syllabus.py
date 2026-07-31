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
Please help me determine if the answer to this question is consistent with the answer I provided.

        Question: {question}\n
        Correct Answer: {answer}\n
        My Answer: {solution}

        Note:
        - The correct answer and my answer may differ in form, so you need to carefully distinguish them before giving your response;
        - If my solution does not clearly provide an answer or contains garbled content unrelated to the question that affects the final judgment, please return False.
        
        Please do not return any other content, only return True or False. True indicates the answers are consistent, False indicates they are inconsistent.

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
        return False, "null"

    # Format limitation list
    limitations_str = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(limitations)])

    prompt = f"""Please help me check if there are any scope exceedances in my solution process. The possible scope exceedances are listed below.
            Solution Process: {solution}\n
            Scope Exceedance List: {limitations}\n
    
            Requirements:
            - Please only return True or False. True indicates that the solution process contains content or methods from the scope exceedance list, and False indicates that it does not.
            - Please carefully inspect my solution process; any presence of content from the out-of-scope list constitutes a violation.
            - If my solution fails to provide a clear answer, or contains garbled/irrelevant content that impedes the final judgment, please return True.
    
            I emphasize once again: do not return any other content. Only return True or False, as defined above.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        res = json.loads(response.choices[0].message.content)
        return res.get("exceeds", False), res.get("reason", "")
    except:
        return True, "null"

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