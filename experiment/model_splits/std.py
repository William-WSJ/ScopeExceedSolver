import numpy as np
import json

# ===================== 核心配置（仅需修改这一行） =====================
JSON_FILE_PATH = "./result_data_deepseek_7b_lora.json"  # 替换为你的JSON文件路径
# =====================================================================

# 步骤1：读取JSON文件（兼容2种格式：JSON数组 / 每行1个JSON对象）
def load_json_data(file_path):
    """读取JSON文件，返回样本列表"""
    data_list = []
    try:
        # 格式1：完整JSON数组（[{}, {}, ...]）
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except json.JSONDecodeError:
        # 格式2：每行1个JSON对象（大文件常见格式）
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))
    return data_list

# 加载数据
data_list = load_json_data(JSON_FILE_PATH)
total_samples = len(data_list)
print(f"✅ 成功读取 {total_samples} 条样本数据\n")

# 步骤2：提取所有样本的acc/osr值（布尔值转数值：True=1，False=0）
acc_list = []  # 存储所有样本的acc值（1=正确，0=错误）
osr_list = []  # 存储所有样本的osr值（1=超纲，0=不超纲）

for sample in data_list:
    # 提取acc/osr字段（兼容字段缺失，默认False）
    acc = sample.get("acc", False)
    osr = sample.get("osr", False)
    
    # 布尔值转数值（兼容布尔类型/字符串类型的True/False）
    acc_val = 1 if acc is True or str(acc).strip() == "True" else 0
    osr_val = 1 if osr is True or str(osr).strip() == "True" else 0
    
    acc_list.append(acc_val)
    osr_list.append(osr_val)

# 步骤3：计算整体的均值、标准差、95%置信区间
# 95%置信区间的z值（大样本用1.96）
Z_VALUE = 1.96
# 总样本量
n = len(acc_list)  # acc和osr样本量一致

# ========== Acc（正确率）统计 ==========
acc_mean = np.mean(acc_list)  # 均值（正确率）
acc_std = np.std(acc_list, ddof=1)  # 样本标准差（ddof=1，无偏估计）
acc_se = acc_std / np.sqrt(n)  # 标准误
# 95%置信区间：均值 ± Z*(标准差/√n)
acc_ci_lower = acc_mean - Z_VALUE * acc_se
acc_ci_upper = acc_mean + Z_VALUE * acc_se

# ========== OSR（超纲率）统计 ==========
osr_mean = np.mean(osr_list)
osr_std = np.std(osr_list, ddof=1)
osr_se = osr_std / np.sqrt(n)
osr_ci_lower = osr_mean - Z_VALUE * osr_se
osr_ci_upper = osr_mean + Z_VALUE * osr_se

# 步骤4：输出结果（转为百分比，保留2位小数，适配论文表格）
print("="*80)
print("📊 整体样本统计结果（均值+标准差+95%置信区间）")
print("="*80)
print(f"总样本量：{n}")

print("\n【Acc（正确率）】")
print(f"  均值：{acc_mean*100:.2f}%")
print(f"  样本标准差（ddof=1）：{acc_std*100:.2f}%")
print(f"  95%置信区间：[{acc_ci_lower*100:.2f}%, {acc_ci_upper*100:.2f}%]")
print(f"  边际误差：±{Z_VALUE * acc_se*100:.2f}%")

print("\n【OSR（超纲率）】")
print(f"  均值：{osr_mean*100:.2f}%")
print(f"  样本标准差（ddof=1）：{osr_std*100:.2f}%")
print(f"  95%置信区间：[{osr_ci_lower*100:.2f}%, {osr_ci_upper*100:.2f}%]")
print(f"  边际误差：±{Z_VALUE * osr_se*100:.2f}%")