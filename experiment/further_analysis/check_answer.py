import json
from openai import OpenAI

# Initialize Ollama client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

INPUT_FILE = "/home/wangsijin/with_idea_limitations/syllabus_qwen-7b_full_gemini3.json"
OUTPUT_FILE = "syllabus_qwen-7b_full_gemini3.json"

def check_answer(question, answer, solution):
    """Judge whether two answers match, only return strict True/False"""
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
        return result == "True"  # Strict exact match for string "True"
    except:
        return False  # Return False for any exception

# 1. Load JSON file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Process each record one by one
for item in data:
    # Skip invalid records missing required fields
    if not all(k in item for k in ["question", "answer", "solution"]):
        item["acc"] = False
        continue
        
    # Get matching judgment result
    item["acc"] = check_answer(
        item["question"],
        item["answer"],
        item["solution"]
    )
    print(f"Question ID: {item.get('id', 'N/A')} - Answer Matched: {item['acc']}")

# 3. Save output results (overwrite or save to new file)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Processing completed! Results saved to questions_with_acc.json")
print(f"Statistics: {sum(1 for item in data if item.get('acc', False))}/{len(data)} records with matching answers")