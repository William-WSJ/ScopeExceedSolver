import numpy as np
import json
from collections import defaultdict

# ===================== 核心配置（仅需修改这一行） =====================
JSON_FILE_PATH = "./response_syllabus_5.json"  # 替换为你的JSON文件绝对/相对路径
# =====================================================================

# 步骤1：读取JSON文件（兼容2种常见格式：JSON数组 / 每行1个JSON对象）
def load_json_data(file_path):
    """读取JSON文件，返回样本列表"""
    data_list = []
    try:
        # 格式1：完整JSON数组（[{}, {}, ...]）
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except json.JSONDecodeError:
        # 格式2：每行1个JSON对象（常见于大文件）
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))
    return data_list

# 加载数据
data_list = load_json_data(JSON_FILE_PATH)
print(f"成功读取 {len(data_list)} 条样本数据\n")

# 步骤2：按年级分组，将acc/osr（字符串布尔值）转为数值（True=1，False=0）
grade_groups = defaultdict(lambda: {"acc": [], "osr": [], "count": 0})  # 按年级存储数据
for sample in data_list:
    # 提取核心字段（兼容字段缺失的情况，避免报错）
    grade = int(sample.get("grade", None))
    acc_str = sample.get("acc", "False")
    osr_str = sample.get("osr", "False")
    
    if grade is None:
        continue  # 跳过无年级标注的样本
    
    # 布尔值转数值："True"→1，"False"→0
    acc_val = 1 if acc_str.strip() == "True" else 0
    osr_val = 1 if osr_str.strip() == "True" else 0
    
    # 按年级存入列表
    grade_groups[grade]["acc"].append(acc_val)
    grade_groups[grade]["osr"].append(osr_val)
    grade_groups[grade]["count"] += 1  # 该年级样本数

# 步骤3：计算各年级的均值、标准差、95%置信区间
grade_stats = {}
print("="*80)
print("各年级统计结果（均值+标准差+95%置信区间）")
print("="*80)
print(f"{'年级':<6}{'样本数':<8}{'Acc均值(%)':<12}{'Acc标准差(%)':<15}{'Acc 95%CI(%)':<20}{'OSR均值(%)':<12}{'OSR标准差(%)':<15}{'OSR 95%CI(%)':<20}")
print("-"*80)

# 95%置信区间的z值（大样本用1.96）
Z_VALUE = 1.96

for grade in sorted(grade_groups.keys()):
    # 基础数据
    acc_list = grade_groups[grade]["acc"]
    osr_list = grade_groups[grade]["osr"]
    n = grade_groups[grade]["count"]
    
    # 均值（正确率/超纲率）
    acc_mean = np.mean(acc_list)
    osr_mean = np.mean(osr_list)
    
    # 样本标准差（ddof=1）
    acc_std = np.std(acc_list, ddof=1)
    osr_std = np.std(osr_list, ddof=1)
    
    # 计算95%置信区间：均值 ± Z*(标准差/√n)
    # Acc置信区间
    acc_se = acc_std / np.sqrt(n)  # 标准误
    acc_ci_lower = acc_mean - Z_VALUE * acc_se
    acc_ci_upper = acc_mean + Z_VALUE * acc_se
    # OSR置信区间
    osr_se = osr_std / np.sqrt(n)
    osr_ci_lower = osr_mean - Z_VALUE * osr_se
    osr_ci_upper = osr_mean + Z_VALUE * osr_se
    
    # 存储结果（保留原始数值，方便后续计算整体）
    grade_stats[grade] = {
        "count": n,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "acc_ci": (acc_ci_lower, acc_ci_upper),
        "osr_mean": osr_mean,
        "osr_std": osr_std,
        "osr_ci": (osr_ci_lower, osr_ci_upper)
    }
    
    # 打印（转为百分比，保留2位小数）
    print(f"{grade:<6}{n:<8}"
          f"{acc_mean*100:<12.2f}{acc_std*100:<15.2f}"
          f"[{acc_ci_lower*100:.2f}, {acc_ci_upper*100:.2f}]"
          f"{osr_mean*100:<12.2f}{osr_std*100:<15.2f}"
          f"[{osr_ci_lower*100:.2f}, {osr_ci_upper*100:.2f}]")

# 步骤4：计算整体（样本量加权）的均值、标准差、95%置信区间
print("="*80)
print("整体统计结果（样本量加权，论文核心使用）")
print("="*80)

# 总样本量
total_n = sum([grade_stats[g]["count"] for g in grade_stats.keys()])

# 1. 加权均值（样本量为权重）
acc_weighted_mean = sum([grade_stats[g]["acc_mean"] * grade_stats[g]["count"] for g in grade_stats.keys()]) / total_n
osr_weighted_mean = sum([grade_stats[g]["osr_mean"] * grade_stats[g]["count"] for g in grade_stats.keys()]) / total_n

# 2. 加权标准差（样本量为权重，即你之前的结果）
acc_weighted_std = sum([grade_stats[g]["acc_std"] * grade_stats[g]["count"] for g in grade_stats.keys()]) / total_n
osr_weighted_std = sum([grade_stats[g]["osr_std"] * grade_stats[g]["count"] for g in grade_stats.keys()]) / total_n

# 3. 整体95%置信区间
acc_total_se = acc_weighted_std / np.sqrt(total_n)
acc_total_ci_lower = acc_weighted_mean - Z_VALUE * acc_total_se
acc_total_ci_upper = acc_weighted_mean + Z_VALUE * acc_total_se

osr_total_se = osr_weighted_std / np.sqrt(total_n)
osr_total_ci_lower = osr_weighted_mean - Z_VALUE * osr_total_se
osr_total_ci_upper = osr_weighted_mean + Z_VALUE * osr_total_se

# 输出整体结果
print(f"总样本量：{total_n}")
print("\n【Acc】")
print(f"  加权均值：{acc_weighted_mean*100:.2f}%")
print(f"  加权样本标准差（ddof=1）：{acc_weighted_std*100:.2f}%")
print(f"  95%置信区间：[{acc_total_ci_lower*100:.2f}%, {acc_total_ci_upper*100:.2f}%]")
print(f"  边际误差：±{Z_VALUE * acc_total_se*100:.2f}%")

print("\n【OSR】")
print(f"  加权均值：{osr_weighted_mean*100:.2f}%")
print(f"  加权样本标准差（ddof=1）：{osr_weighted_std*100:.2f}%")
print(f"  95%置信区间：[{osr_total_ci_lower*100:.2f}%, {osr_total_ci_upper*100:.2f}%]")
print(f"  边际误差：±{Z_VALUE * osr_total_se*100:.2f}%")