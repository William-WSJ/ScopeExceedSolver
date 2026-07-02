import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
EXPERIMENT_DIR = PROJECT_ROOT / "experiment"

def read_test_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def read_checkpoint(mode):
    checkpoint_file = f"checkpoint_{mode}.txt"
    if not os.path.exists(checkpoint_file):
        return {"count": 0, "correct": 0, "relevant": 0}
    
    with open(checkpoint_file, "r") as f:
        data = f.read().strip().split(',')
        count, correct, relevant = int(data[0]), int(data[1]), int(data[2])
        return {"count": count, "correct": correct, "relevant": relevant}

def write_checkpoint(mode, count, correct, relevant):
    checkpoint_file = f"checkpoint_{mode}.txt"
    with open(checkpoint_file, "w") as f:
        f.write(f"{count},{correct},{relevant}")

# Batch inference helper: convert a JSON dataset into JSONL requests.
def convert_to_jsonl(input_file, output_file):
    def prepare_content(data):
        content_template = (
            "请根据我的解题思路解决以下问题：\n"
            "{question}\n"
            "我的解题思路如下：\n"
            "{idea}\n"
            "请注意，如果我的解题思路有明显的错误，请纠正后再解答。"
        )

        # Merge list elements into a single string.
        def join_list_to_string(lst):
            return '\n'.join(lst)

        # Build the final prompt content.
        content = content_template.format(
            question=data["question"],
            idea=data['thoughts_mini_finetuned'],
            # answer=data["answer"],
            # solution=data["solution"],
            # limitations=join_list_to_string(data.get("grade_cautions", []))
        )
        return content

    def create_json_obj(data):
        content = prepare_content(data)
        return {
            "custom_id": f"request-{data['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "deepseek-v3",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}
                ],
                "max_tokens": 8192
            }
        }

    # Read the input file.
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_list = json.load(infile)

    # Wrap a single object into a list if needed.
    if not isinstance(data_list, list):
        data_list = [data_list]

    # Write one JSON object per line.
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for data in data_list:
            json_obj = create_json_obj(data)
            json_obj["custom_id"] = f"request-{data['id']}"
            json_line = json.dumps(json_obj, ensure_ascii=False)
            outfile.write(json_line + '\n')

def extract():
    dataset_path = DATASET_DIR / "syllabus_check_2395_latest.json"
    output_path = EXPERIMENT_DIR / "further_analysis" / "new_data.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        # Filter items belonging to the selected knowledge points.
        if item["knowledge"] in {"鸡兔同笼问题", "同增同减问题", "倍的概念及其应用"}:
            new_data.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    extract()
