import json
import os
import pandas as pd

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

# Batch inference: convert json file to jsonl format
def convert_to_jsonl(input_file, output_file):
    def prepare_content(data):
        content_template = (
            "请根据我的解题思路解决以下问题：\n"
            "{question}\n"
            "我的解题思路如下：\n"
            "{idea}\n"
            "请注意，如果我的解题思路有明显的错误，请纠正后再解答。"
        )

        # Merge all elements in the list into one string
        def join_list_to_string(lst):
            return '\n'.join(lst)

        # Generate the final content text
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

    # Read input file
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_list = json.load(infile)

    # Wrap single object into a list if input is not an array
    if not isinstance(data_list, list):
        data_list = [data_list]

    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for data in data_list:  # Numbering starts from 1
            # print(data)
            json_obj = create_json_obj(data)
            json_obj["custom_id"] = f"request-{data['id']}"  # Update custom_id field
            json_line = json.dumps(json_obj, ensure_ascii=False)
            outfile.write(json_line + '\n')
            # break

def extract():
    with open("dataset/syllabus_check_2395_latest.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        # Filter items with specified knowledge points
        if item["knowledge"] == "鸡兔同笼问题" or item["knowledge"] == "同增同减问题" or item["knowledge"] == "倍的概念及其应用":
            new_data.append(item)
    # Save filtered dataset to target path
    with open("experiment/further_analysis/new_data.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
extract()