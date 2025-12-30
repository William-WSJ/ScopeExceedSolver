import json
import requests
import argparse

def call_llama_factory_api(question: str, base_url: str = "http://127.0.0.1:8000", max_tokens: int = 2048) -> str:
    """
    调用 LLaMA-Factory 的 OpenAI 兼容 API 获取模型输出。
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama",  # 模型名可任意，LLaMA-Factory 通常忽略此字段
        "messages": [
            {"role": "system", "content": "请给出下列问题的解题思路"},
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # 保证确定性输出（可按需调整）
        "stream": False
    }

    try:
        response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling API for question: {question}\nError: {e}")
        return ""

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
    for item in items:
        question = item["question"]
        print(f"Processing ID: {item.get('id', 'N/A')} ...")
        thought = call_llama_factory_api(question, base_url=base_url, max_tokens=2048)
        item["thought"] = thought
        updated_items.append(item)

    # 写回
    with open(output_path, "w", encoding="utf-8") as f:
        if is_single:
            json.dump(updated_items[0], f, ensure_ascii=False, indent=2)
        else:
            json.dump(updated_items, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call LLaMA-Factory API to add 'thought' field to JSON.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="LLaMA-Factory API base URL")

    args = parser.parse_args()
    process_json_file(args.input, args.output, base_url=args.url)