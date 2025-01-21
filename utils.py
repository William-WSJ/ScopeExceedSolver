import json
import os

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

def convert_to_jsonl(input_file, output_file):
    def prepare_content(data):
        content_template = (
            "请根据我的解题思路解决以下问题：\n\n"
            "{question}\n\n"
            "我的解题思路如下：\n\n"
            "{thoughts}\n\n"
            "你的解答过程中严禁出现以下内容或方法：\n\n"
            "{cautions}\n\n"
            "请注意：\n"
            "1. 如果我的解题思路有明显的错误，请纠正后再解答。\n"
            "2. 如果我没有指明严禁出现的方法，请遵循第一条规则解答。"
        )

        # 将列表中的元素合并成字符串
        def join_list_to_string(lst):
            return '\n'.join(lst)

        # 构建最终的content文本
        content = content_template.format(
            question=data["question"],
            thoughts=data["thoughts"],
            cautions=join_list_to_string(data.get("grade_cautions", []))
        )
        return content

    def create_json_obj(data):
        content = prepare_content(data)
        return {
            "custom_id": f"request-{data['id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}
                ],
                "max_tokens": 8192
            }
        }

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_list = json.load(infile)

    # 检查是否是列表，如果不是则包装成列表
    if not isinstance(data_list, list):
        data_list = [data_list]

    # 打开输出文件准备写入
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for index, data in enumerate(data_list, start=1):  # 从1开始编号
            json_obj = create_json_obj(data)
            json_obj["custom_id"] = f"request-{index}"  # 更新custom_id
            json_line = json.dumps(json_obj, ensure_ascii=False)
            outfile.write(json_line + '\n')

# 使用示例
# convert_to_jsonl('merged_output_set_4298.json', 'output.jsonl')