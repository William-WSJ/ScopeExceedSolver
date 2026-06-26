import json
import requests
import argparse

def call_llama_factory_api(question: str, base_url: str = "http://127.0.0.1:8000", max_tokens: int = 2048) -> str:
    """
    Call LLaMA-Factory OpenAI-compatible API to get model response.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama",  # Model name can be arbitrary, this field is usually ignored by LLaMA-Factory
        "messages": [
            {"role": "system", "content": "请给出下列问题的解题思路"},  # Inference instruction for the model, recommended to keep it simple
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Ensure deterministic output (adjust as needed)
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

    # Support single JSON object or JSON array
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

    # Write results to output file
    with open(output_path, "w", encoding="utf-8") as f:
        if is_single:
            json.dump(updated_items[0], f, ensure_ascii=False, indent=2)
        else:
            json.dump(updated_items, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call LLaMA-Factory API to append 'thought' field into JSON data.")
    parser.add_argument("--input", type=str, required=True, help="File path of input JSON")
    parser.add_argument("--output", type=str, required=True, help="File path of output JSON")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base URL of LLaMA-Factory API service")

    args = parser.parse_args()
    process_json_file(args.input, args.output, base_url=args.url)