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


# Batch inference convert json to jsonl
def convert_to_jsonl(input_file, output_file):
    def prepare_content(data):
        content_template = (
            "请根据我的解题思路解决以下问题：\n"
            "{question}\n"
            "我的解题思路如下：\n"
            "{idea}\n"
            # "你的解答过程中严禁出现以下内容或方法：\n"
            # "{limitations}\n"
            "请注意，如果我的解题思路有明显的错误，请纠正后再解答。"
        )

        # 将列表中的元素合并成字符串
        def join_list_to_string(lst):
            return '\n'.join(lst)

        # 构建最终的content文本
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

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_list = json.load(infile)

    # 检查是否是列表，如果不是则包装成列表
    if not isinstance(data_list, list):
        data_list = [data_list]

    # 打开输出文件准备写入
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for data in data_list:  # 从1开始编号
            # print(data)
            json_obj = create_json_obj(data)
            json_obj["custom_id"] = f"request-{data['id']}"  # 更新custom_id
            json_line = json.dumps(json_obj, ensure_ascii=False)
            outfile.write(json_line + '\n')
            # break

# Calculate idea scores
def calculate_average_scores(merged_file):
    # 初始化计数器和总分
    total_scores = {
        "思路评分": 0,
        "关键点": 0,
        "引导力": 0,
        "正确性": 0
    }
    count = 0
    
    # 读取合并后的文件
    with open(merged_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # 遍历每条记录
        for item in data:
            if 'evaluation' in item:
                count += 1
                # 遍历evaluation中的每个评分项
                for score_dict in item['evaluation']:
                    for key, value in score_dict.items():
                        total_scores[key] += value
    
    # 计算并打印平均分
    print("评分统计结果：")
    print("-" * 20)
    for key, total in total_scores.items():
        if count > 0:
            average = total / count
            print(f"{key}: {average:.2f} {total}")
    print(f"统计样本数: {count}")

def count_high_grade_osr(file_path: str = '') -> None:
    """
    分别统计指定文件中三年级和四年级中osr为true的样本数量
    
    Args:
        file_path (str): 要统计的JSON文件路径
    """
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 统计符合条件的样本
        count_grade3 = 0
        count_grade4 = 0
        total_grade3 = 0
        total_grade4 = 0
        
        for item in data:
            grade = item.get('grade')
            if grade == 3:
                total_grade3 += 1
                if item.get('osr') == "True":
                    count_grade3 += 1
            elif grade == 4:
                total_grade4 += 1
                if item.get('osr') == "True":
                    count_grade4 += 1
                
        # 打印结果
        print(f"文件 {file_path} 中：")
        print("\n三年级统计：")
        print(f"grade为3且osr为true的样本数量：{count_grade3}")
        print(f"三年级总样本数量：{total_grade3}")
        if total_grade3 > 0:
            print(f"占比：{count_grade3/total_grade3*100:.2f}%")
            
        print("\n四年级统计：")
        print(f"grade为4且osr为true的样本数量：{count_grade4}")
        print(f"四年级总样本数量：{total_grade4}")
        if total_grade4 > 0:
            print(f"占比：{count_grade4/total_grade4*100:.2f}%")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 不是有效的JSON格式")
    except Exception as e:
        print(f"处理过程中出现错误：{str(e)}")

# count_high_grade_osr() 