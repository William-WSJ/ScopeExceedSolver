import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

def calculate_osr_by_grade(json_path: str, output_csv: str = None) -> pd.DataFrame:
    """
    从JSON文件中按年级统计超出范围率（OSR）
    
    Args:
        json_path (str): JSON文件路径（应为包含题目记录的列表）
        output_csv (str, optional): 可选，将结果保存为CSV文件的路径
    
    Returns:
        pd.DataFrame: 包含统计结果的DataFrame，列包括：
            - grade: 年级
            - total_count: 该年级总样本数
            - exceeds_scope_count: exceeds_scope=True 的样本数
            - osr_rate: 超出范围率（小数形式，如0.15表示15%）
            - osr_percentage: 超出范围率（百分比字符串，如"15.00%"）
    """
    # 1. 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保是列表格式
        if not isinstance(data, list):
            raise ValueError("JSON文件内容应为列表格式（包含多个题目记录）")
    except Exception as e:
        raise RuntimeError(f"读取JSON文件失败: {e}")
    
    # 2. 按年级统计
    stats = defaultdict(lambda: {'total': 0, 'exceeds': 0})
    
    for idx, record in enumerate(data):
        grade = int(record.get('grade'))
        if grade is None:
            print(f"警告: 第 {idx} 条记录缺少 'grade' 字段，已跳过")
            continue
        
        stats[grade]['total'] += 1
        if record.get('exceeds_scope') is True:
            stats[grade]['exceeds'] += 1
    
    # 3. 构建结果
    results = []
    for grade in sorted(stats.keys()):
        total = stats[grade]['total']
        exceeds = stats[grade]['exceeds']
        osr_rate = exceeds / total if total > 0 else 0.0
        results.append({
            'grade': grade,
            'total_count': total,
            'exceeds_scope_count': exceeds,
            'osr_rate': round(osr_rate, 4),
            'osr_percentage': f"{osr_rate * 100:.2f}%"
        })
    
    # 4. 转换为DataFrame
    df = pd.DataFrame(results).sort_values('grade').reset_index(drop=True)
    
    # 5. 可选：保存为CSV
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✓ 统计结果已保存至: {output_csv}")
    
    return df

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 请将下面的路径替换为您的实际JSON文件路径
    JSON_FILE_PATH = "/root/autodl-tmp/syllabus_qwen-7b_full_deepseek_v3_2.json"      # ← 修改此处
    OUTPUT_CSV_PATH = "osr_by_grade.csv"            # 可选：结果保存路径
    
    try:
        # 执行统计
        result_df = calculate_osr_by_grade(
            json_path=JSON_FILE_PATH,
            # output_csv=OUTPUT_CSV_PATH  # 如不需要保存，可设为 None 或删除此参数
        )
        
        # 打印结果
        print("\n=== 按年级统计的超出范围率（OSR） ===")
        print(result_df.to_string(index=False))
        print("\n说明:")
        print("- osr_rate: 超出范围率（小数形式）")
        print("- osr_percentage: 超出范围率（百分比形式）")
        print("- OSR = (exceeds_scope=True 的题目数 / 该年级总题目数) × 100%")
        
    except Exception as e:
        print(f"❌ 执行出错: {e}")