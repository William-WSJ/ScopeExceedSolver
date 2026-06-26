import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

def calculate_osr_by_grade(json_path: str, output_csv: str = None) -> pd.DataFrame:
    """
    Calculate Out-of-Scope Rate (OSR) grouped by grade from JSON file
    
    Args:
        json_path (str): Path to JSON file (should be a list of problem records)
        output_csv (str, optional): Optional path to export statistics as CSV file
    
    Returns:
        pd.DataFrame: DataFrame containing statistical results with columns:
            - grade: Student grade level
            - total_count: Total samples in this grade
            - exceeds_scope_count: Number of samples marked exceeds_scope=True
            - osr_rate: Raw out-of-scope rate (decimal, e.g. 0.15 stands for 15%)
            - osr_percentage: Formatted percentage string (e.g. "15.00%")
    """
    # 1. Load JSON file
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Enforce list data format
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of problem records")
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON file: {e}")
    
    # 2. Aggregate statistics grouped by grade
    stats = defaultdict(lambda: {'total': 0, 'exceeds': 0})
    
    for idx, record in enumerate(data):
        grade = int(record.get('grade'))
        if grade is None:
            print(f"Warning: Record No.{idx} missing 'grade' field, skipped")
            continue
        
        stats[grade]['total'] += 1
        if record.get('exceeds_scope') is True:
            stats[grade]['exceeds'] += 1
    
    # 3. Compile output records
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
    
    # 4. Convert list to DataFrame
    df = pd.DataFrame(results).sort_values('grade').reset_index(drop=True)
    
    # 5. Optional export to CSV
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✓ Statistics saved to: {output_csv}")
    
    return df

# ==================== Usage Example ====================
if __name__ == "__main__":
    # Replace the path below with your actual JSON file path
    JSON_FILE_PATH = "/root/autodl-tmp/syllabus_qwen-7b_full_deepseek_v3_2.json"      # ← Modify this line
    OUTPUT_CSV_PATH = "osr_by_grade.csv"            # Optional export destination
    
    try:
        # Run statistical calculation
        result_df = calculate_osr_by_grade(
            json_path=JSON_FILE_PATH,
            # output_csv=OUTPUT_CSV_PATH  # Remove or set to None if export is unnecessary
        )
        
        # Print statistical table
        print("\n=== Out-of-Scope Rate (OSR) Statistics Grouped by Grade ===")
        print(result_df.to_string(index=False))
        print("\nExplanation:")
        print("- osr_rate: Raw out-of-scope rate (decimal format)")
        print("- osr_percentage: Out-of-scope rate (percentage format)")
        print("- OSR = (Number of exceeds_scope=True records / Total records of grade) × 100%")
        
    except Exception as e:
        print(f"❌ Execution error: {e}")