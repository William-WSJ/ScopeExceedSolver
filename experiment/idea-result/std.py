import numpy as np
import json
from collections import defaultdict

# ===================== Core Configuration (Only modify this line) =====================
JSON_FILE_PATH = "./syllabus_qwen-7b_full.json"  # Replace with absolute/relative path of your JSON file
# JSON_FILE_PATH = "./syllabus_baseline_gpt_5_2_direct.json"  # Replace with absolute/relative path of your JSON file
# ======================================================================================

# Step 1: Load JSON file (supports 2 common formats: JSON array / one JSON object per line)
def load_json_data(file_path):
    """Load JSON file and return sample list"""
    data_list = []
    try:
        # Format 1: Complete JSON array ([{}, {}, ...])
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except json.JSONDecodeError:
        # Format 2: One JSON object per line (commonly used for large files)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))
    return data_list

# Load dataset
data_list = load_json_data(JSON_FILE_PATH)
print(f"Successfully loaded {len(data_list)} sample records\n")

# Step 2: Group data by grade, convert string boolean acc/osr to numeric values (True=1, False=0)
grade_groups = defaultdict(lambda: {"acc": [], "osr": [], "count": 0})  # Store data grouped by grade
for sample in data_list:
    # Extract core fields (handle missing fields to avoid runtime errors)
    grade = int(sample.get("grade", None))
    acc_str = str(sample.get("acc", "false"))
    osr_str = str(sample.get("exceeds_scope", "false"))

    if grade is None:
        continue  # Skip samples without grade labels
    
    # Convert boolean string to numeric: "True" -> 1, "False" -> 0
    acc_val = 1 if acc_str.strip() == "True" else 0
    osr_val = 1 if osr_str.strip() == "True" else 0
    
    # Append values to corresponding grade group
    grade_groups[grade]["acc"].append(acc_val)
    grade_groups[grade]["osr"].append(osr_val)
    grade_groups[grade]["count"] += 1  # Count samples of current grade

# Step 3: Calculate mean, standard deviation and 95% confidence interval for each grade
grade_stats = {}
print("="*80)
print("Statistical Results by Grade (Mean + Std Dev + 95% Confidence Interval)")
print("="*80)
print(f"{'Grade':<6}{'SampleNum':<8}{'Acc Mean(%)':<12}{'Acc Std(%)':<15}{'Acc 95%CI(%)':<20}{'OSR Mean(%)':<12}{'OSR Std(%)':<15}{'OSR 95%CI(%)':<20}")
print("-"*80)

# Z-score for 95% confidence interval (1.96 for large sample size)
Z_VALUE = 1.96

for grade in sorted(grade_groups.keys()):
    # Basic dataset of current grade
    acc_list = grade_groups[grade]["acc"]
    osr_list = grade_groups[grade]["osr"]
    n = grade_groups[grade]["count"]
    
    # Calculate mean value (Accuracy / Out-of-Scope Rate)
    acc_mean = np.mean(acc_list)
    osr_mean = np.mean(osr_list)
    
    # Sample standard deviation (ddof=1 for unbiased estimation)
    acc_std = np.std(acc_list, ddof=1)
    osr_std = np.std(osr_list, ddof=1)
    
    # Compute 95% confidence interval: mean ± Z*(std/sqrt(n))
    # Confidence interval for Accuracy
    acc_se = acc_std / np.sqrt(n)  # Standard Error
    acc_ci_lower = acc_mean - Z_VALUE * acc_se
    acc_ci_upper = acc_mean + Z_VALUE * acc_se
    # Confidence interval for OSR
    osr_se = osr_std / np.sqrt(n)
    osr_ci_lower = osr_mean - Z_VALUE * osr_se
    osr_ci_upper = osr_mean + Z_VALUE * osr_se
    
    # Save raw statistics for overall weighted calculation later
    grade_stats[grade] = {
        "count": n,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "acc_ci": (acc_ci_lower, acc_ci_upper),
        "osr_mean": osr_mean,
        "osr_std": osr_std,
        "osr_ci": (osr_ci_lower, osr_ci_upper)
    }
    
    # Print formatted results (convert to percentage, 2 decimal places)
    print(f"{grade:<6}{n:<8}"
          f"{acc_mean*100:<12.2f}{acc_std*100:<15.2f}"
          f"[{acc_ci_lower*100:.2f}, {acc_ci_upper*100:.2f}]"
          f"{osr_mean*100:<12.2f}{osr_std*100:<15.2f}"
          f"[{osr_ci_lower*100:.2f}, {osr_ci_upper*100:.2f}]")

# Step 4: Compute overall weighted mean, weighted std dev and 95% confidence interval (weighted by sample count)
print("="*80)
print("Overall Statistical Results (Sample-Size Weighted, Core Metric for Paper)")
print("="*80)

# Total number of all samples
total_n = sum([grade_stats[g]["count"] for g in grade_stats.keys()])

# 1. Weighted mean (weight = sample count per grade)
acc_weighted_mean = sum([grade_stats[g]["acc_mean"] * grade_stats[g]["count"] for g in grade_stats.keys()]) / total_n
osr_weighted_mean = sum([grade_stats[g]["osr_mean"] * grade_stats[g]["count"] for g in grade_stats.keys()]) / total_n

# 2. Weighted standard deviation (weight = sample count, consistent with your previous output)
acc_weighted_std = sum([grade_stats[g]["acc_std"] * grade_stats[g]["count"] for g in grade_stats.keys()]) / total_n
osr_weighted_std = sum([grade_stats[g]["osr_std"] * grade_stats[g]["count"] for g in grade_stats.keys()]) / total_n

# 3. Overall 95% confidence interval
acc_total_se = acc_weighted_std / np.sqrt(total_n)
acc_total_ci_lower = acc_weighted_mean - Z_VALUE * acc_total_se
acc_total_ci_upper = acc_weighted_mean + Z_VALUE * acc_total_se

osr_total_se = osr_weighted_std / np.sqrt(total_n)
osr_total_ci_lower = osr_weighted_mean - Z_VALUE * osr_total_se
osr_total_ci_upper = osr_weighted_mean + Z_VALUE * osr_total_se

# Print overall statistics
print(f"Total Samples: {total_n}")
print("\n【Accuracy】")
print(f"  Weighted Mean: {acc_weighted_mean*100:.2f}%")
print(f"  Weighted Sample Std Dev (ddof=1): {acc_weighted_std*100:.2f}%")
print(f"  95% Confidence Interval: [{acc_total_ci_lower*100:.2f}%, {acc_total_ci_upper*100:.2f}%]")
print(f"  Margin of Error: ±{Z_VALUE * acc_total_se*100:.2f}%")

print("\n【OSR (Out-of-Scope Rate)】")
print(f"  Weighted Mean: {osr_weighted_mean*100:.2f}%")
print(f"  Weighted Sample Std Dev (ddof=1): {osr_weighted_std*100:.2f}%")
print(f"  95% Confidence Interval: [{osr_total_ci_lower*100:.2f}%, {osr_total_ci_upper*100:.2f}%]")
print(f"  Margin of Error: ±{Z_VALUE * osr_total_se*100:.2f}%")