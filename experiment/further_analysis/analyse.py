import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

def extract_json_scores(raw_text):
    """
    强力提取器：从 raw_model_response 字符串中定位并解析 JSON 分数。
    """
    if not raw_text or not isinstance(raw_text, str):
        return None
    try:
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(raw_text[start:end])
    except:
        pass
    return None

def analyze_reliability(file_path: Path):
    """
    分析 acc 字段与模型给出的 '正确性' 分数的相关性。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    acc_list = []      # 客观标签 (1=True, 0=False)
    score_list = []    # 模型评估的 '正确性' 分数 (0-10)
    
    for item in data:
        # 1. 优先尝试从已有的 model_scores 取，没有则从 raw_model_response 提取
        scores = item.get("model_scores")
        if not scores:
            scores = extract_json_scores(item.get("raw_model_response", ""))
            
        # 2. 只有拿到有效分数且包含 '正确性' 字段时才记录
        if scores and "正确性" in scores:
            acc_val = 1 if item.get("acc") is True else 0
            score_val = scores["正确性"]
            
            acc_list.append(acc_val)
            score_list.append(score_val)
    
    if len(acc_list) < 5:
        print(f"--- 文件 {file_path.name} ---")
        print("跳过：有效数据量不足（需至少5条含有分数的记录）。")
        return

    # 计算 Pearson 相关系数
    corr, p_value = pearsonr(acc_list, score_list)
    
    # 分组计算均值
    scores_true = [s for a, s in zip(acc_list, score_list) if a == 1]
    scores_false = [s for a, s in zip(acc_list, score_list) if a == 0]
    
    mean_true = np.mean(scores_true) if scores_true else 0
    mean_false = np.mean(scores_false) if scores_false else 0
    diff = mean_true - mean_false
    
    # 打印报告
    print(f"\n" + "="*50)
    print(f"📊 审计文件: {file_path.name}")
    print(f"有效样本量: {len(acc_list)}")
    print(f"Pearson 相关系数 (r): {corr:.4f}")
    print(f"acc=True (正确组) 平均分: {mean_true:.2f}")
    print(f"acc=False (错误组) 平均分: {mean_false:.2f}")
    print(f"【关键指标】均值分差: {diff:.2f}")
    
    # 自动化判定
    # 相关性 > 0.7 且分差 > 4.0 被认为是黄金判别标准
    if corr > 0.7 and diff > 4.0:
        status = "🌟 高可信度 (模型逻辑清晰，数据极具参考价值)"
    elif corr > 0.5 and diff > 2.5:
        status = "✅ 中等可信 (模型基本能分清对错，数据可用)"
    else:
        status = "❌ 低可信度 (判分分布随机，建议检查模型是否进入幻觉)"
        
    print(f"最终判定结论: {status}")
    print("="*50)

# ---------------------------------------------------------
# 执行区域
# ---------------------------------------------------------
output_dir = Path("/root/autodl-tmp/process")

if not output_dir.exists():
    print(f"错误：找不到输出目录 {output_dir}")
else:
    # 寻找所有处理过的 json 文件
    target_files = sorted(list(output_dir.glob("*_process.json")))
    if not target_files:
        print("未发现以 _process.json 结尾的结果文件。")
    else:
        print(f"开始分析共 {len(target_files)} 个文件...")
        for f in target_files:
            analyze_reliability(f)