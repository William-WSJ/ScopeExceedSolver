import json
import pandas as pd

def filter_target_knowledge_data(json_list, output_file="targeted_analysis_data.json"):
    # 定义你选定的 5 个核心知识点
    target_knowledge = [
        "鸡兔同笼问题",
        "积的变化规律",
        "商的变化规律",
        "同增同减问题",
        "倍的概念及其应用"
    ]
    
    filtered_results = []
    
    for file_path in json_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 兼容处理：确保 data 是列表
                if isinstance(data, dict):
                    data = [data]
                
                for item in data:
                    # 只有当知识点在目标列表中，且年级为 3 或 4 时保留
                    if item.get('knowledge') in target_knowledge and str(item.get('grade')) in ['3', '4']:
                        filtered_results.append(item)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # 输出筛选后的数据统计
    print(f"筛选完成！总共提取样本数: {len(filtered_results)}")
    
    # 保存为新的 JSON 文件，方便后续实验调用
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
        
    return filtered_results

# 使用示例
json_list = ["./syllabus_baseline_deepseek_v3_2_direct.json"]
target_data = filter_target_knowledge_data(json_list, output_file="syllabus_baseline_deepseek_v3_2_subset.json")