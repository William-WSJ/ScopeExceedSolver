import pandas as pd
import json

def analyze_math_data(json_list):
    """
    针对特定格式JSON进行3-4年级知识点分析
    """
    rows = []
    for file_path in json_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 假设你的JSON是列表格式，如果是一行一个JSON对象需改用 lines=True
            data = json.load(f)
            for item in data:
                # 提取核心字段
                rows.append({
                    'grade': str(item.get('grade', '')),
                    'knowledge': item.get('knowledge', 'Unknown'),
                    'acc': 1 if item.get('acc') is True else 0,
                    'osr': 1 if item.get('exceeds_scope') is True else 0
                })

    df = pd.DataFrame(rows)

    # 1. 筛选 3-4 年级
    df_filtered = df[df['grade'].isin(['3', '4'])].copy()

    # 2. 按知识点聚合
    # 计算题目数量、准确率均值、超纲率均值
    knowledge_stats = df_filtered.groupby('knowledge').agg(
        Sample_Size=('acc', 'count'),
        Accuracy=('acc', 'mean'),
        OSR_Rate=('osr', 'mean')
    )

    # 3. 格式化百分比
    knowledge_stats['Accuracy'] = (knowledge_stats['Accuracy'] * 100).round(2)
    knowledge_stats['OSR_Rate'] = (knowledge_stats['OSR_Rate'] * 100).round(2)

    # 4. 排序：优先看超纲最严重的知识点
    return knowledge_stats.sort_values(by='OSR_Rate', ascending=False)

# 使用方式
# json_list = ["./syllabus_4o-mini_finetuned_deepseek_v3_2_subset.json",
#              "./syllabus_deepseek-7b_full_deepseek_v3_2_subset.json",
#              "./syllabus_mimo-7b_full_deepseek_v3_2_subset.json",
#              "./syllabus_qwen-7b_full_deepseek_v3_2_subset.json"]
json_list = ["../baseline-result/syllabus_baseline_deepseek_v3_2_direct.json"]
# json_list = ["../baseline-result/syllabus_baseline_deepseek_v3_2_with_limitations.json"]
stats = analyze_math_data(json_list)
print(stats)