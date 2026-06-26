import json
import pandas as pd

def filter_target_knowledge_data(json_list, output_file="targeted_analysis_data.json"):
    # Define 5 core knowledge points you selected
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
                # Compatibility handling: convert single dict to list
                if isinstance(data, dict):
                    data = [data]
                
                for item in data:
                    # Keep samples only if knowledge point matches target list and grade is 3 or 4
                    if item.get('knowledge') in target_knowledge and str(item.get('grade')) in ['3', '4']:
                        filtered_results.append(item)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # Print statistics after filtering
    print(f"Filtering completed! Total extracted samples: {len(filtered_results)}")
    
    # Save filtered data to new JSON file for subsequent experiments
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
        
    return filtered_results

# Usage example
json_list = ["./syllabus_baseline_deepseek_v3_2_direct.json"]
target_data = filter_target_knowledge_data(json_list, output_file="syllabus_baseline_deepseek_v3_2_subset.json")