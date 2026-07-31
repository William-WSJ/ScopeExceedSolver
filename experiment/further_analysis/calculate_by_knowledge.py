import pandas as pd
import json

def analyze_math_data(json_list):
    rows = []
    for file_path in json_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            
            data = json.load(f)
            for item in data:
                
                rows.append({
                    'grade': str(item.get('grade', '')),
                    'knowledge': item.get('knowledge', 'Unknown'),
                    'acc': 1 if item.get('acc') is True else 0,
                    'osr': 1 if item.get('exceeds_scope') is True else 0
                })

    df = pd.DataFrame(rows)

    df_filtered = df[df['grade'].isin(['3', '4'])].copy()

    knowledge_stats = df_filtered.groupby('knowledge').agg(
        Sample_Size=('acc', 'count'),
        Accuracy=('acc', 'mean'),
        OSR_Rate=('osr', 'mean')
    )

    knowledge_stats['Accuracy'] = (knowledge_stats['Accuracy'] * 100).round(2)
    knowledge_stats['OSR_Rate'] = (knowledge_stats['OSR_Rate'] * 100).round(2)

    return knowledge_stats.sort_values(by='OSR_Rate', ascending=False)

json_list = ["../baseline-result/syllabus_baseline_deepseek_v3_2_direct.json"]
# json_list = ["../baseline-result/syllabus_baseline_deepseek_v3_2_with_limitations.json"]
stats = analyze_math_data(json_list)
print(stats)