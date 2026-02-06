import os
import json
import re
import pandas as pd
from pathlib import Path
def calculate_score_averages(model_splits_dir="../experiment/further_analysis/propagation"):
    """
    针对SPS实验数据统计思路得分，并应用排名转换。
    """
    # 1. 定义模型及其对应的文件名后缀
    # 文件名格式：syllabus_{suffix}_deepseek_v3_2_subset_process_prop_analysed.json
    model_configs = {
        "GPT_4O_MINI": "4o-mini_finetuned",
        "DEEPSEEK_7B_LORA": "deepseek-7b_lora",
        "DEEPSEEK_7B_FULL": "deepseek-7b_full",
        "MIMO_7B_LORA": "mimo-7b_lora",
        "MIMO_7B_FULL": "mimo-7b_full",
        "QWEN_7B_LORA": "qwen-7b_lora",
        "QWEN_7B_FULL": "qwen-7b_full"
    }

    results = {}
    raw_scores_summary = []

    print("=" * 65)
    print(f"{'Model Name':<20} | {'Idea':>6} | {'Key':>6} | {'Lead':>6} | {'Corr':>6} | {'Count':>6}")
    print("-" * 65)

    for nick_name, suffix in model_configs.items():
        filename = f"syllabus_{suffix}_deepseek_v3_2_subset_process_prop_analysed.json"
        filepath = os.path.join(model_splits_dir, filename)

        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_scores = {"思路评分": 0.0, "关键点": 0.0, "引导力": 0.0, "正确性": 0.0}
        valid_count = 0

        for item in data:
            # 统一从 raw_model_response 提取，因为这是你的思路评价原始输出
            raw_text = item.get("raw_model_response", "")
            if not raw_text:
                continue

            try:
                # 提取 JSON 块
                start = raw_text.find('{')
                end = raw_text.rfind('}') + 1
                if start != -1 and end != 0:
                    score_json = json.loads(raw_text[start:end])
                    
                    # 字段映射（处理可能出现的细微键名差异）
                    mapping = {
                        "思路评分": ["思路评分"],
                        "关键点": ["关键点"],
                        "引导力": ["引导力", "引导性"],
                        "正确性": ["正确性"]
                    }
                    
                    found_in_item = False
                    for std_key, alt_keys in mapping.items():
                        for k in alt_keys:
                            if k in score_json:
                                total_scores[std_key] += float(score_json[k])
                                found_in_item = True
                                break
                    if found_in_item:
                        valid_count += 1
            except:
                continue

        # 计算均值
        row = {"Model": nick_name, "Count": valid_count}
        if valid_count > 0:
            for k in total_scores:
                avg = total_scores[k] / valid_count
                total_scores[k] = avg
                row[k] = round(avg, 3)
            
            print(f"{nick_name:<20} | {row['思路评分']:>6.2f} | {row['关键点']:>6.2f} | {row['引导力']:>6.2f} | {row['正确性']:>6.2f} | {valid_count:>6}")
        else:
            for k in total_scores: row[k] = 0.0
            print(f"{nick_name:<20} | 数据缺失")

        results[nick_name] = total_scores
        raw_scores_summary.append(row)

    # --- 排名转换逻辑 ---
    df = pd.DataFrame(raw_scores_summary)
    
    def rank_score_transform(series):
        """标准排名转换：9 - (rank - 1)"""
        # 降序排名，method='average' 处理并列
        ranks = series.rank(ascending=False, method='average')
        return (9 - (ranks - 1)).round(1)

    ranked_df = pd.DataFrame()
    ranked_df["Model"] = df["Model"]
    
    dimensions = ["思路评分", "关键点", "引导力", "正确性"]
    for dim in dimensions:
        ranked_df[dim] = rank_score_transform(df[dim])

    # 计算 Overall 并生成最终排名分数
    df["Overall_Raw"] = df[dimensions].mean(axis=1)
    ranked_df["Overall"] = rank_score_transform(df["Overall_Raw"])

    print("\n" + "=" * 65)
    print("各模型排名转换得分 (Rank-based Scores)")
    print("=" * 65)
    print(ranked_df.to_string(index=False))
    
    return results
    """
    统计model_splits目录中所有模型的原始得分，并应用排名转换算法
    将原始得分转换为排名分数（3-9分）
    
    Args:
        model_splits_dir (str): model_splits目录路径
    
    Returns:
        DataFrame: 包含各模型原始平均分和排名分数的表格
    """
    # 确保目录存在
    if not os.path.exists(model_splits_dir):
        raise ValueError(f"目录不存在: {model_splits_dir}")
    
    # 定义所有模型文件
    model_files = {
        "Qwen_7B_FULL": "result_data_qwen_7b_full.json",
        "Qwen_7B_LORA": "result_data_qwen_7b_lora.json", 
        "Qwen_1.5B_FULL": "result_data_qwen_1_5b_full.json",
        "Qwen_1.5B_LORA": "result_data_qwen_1_5b_lora.json",
        "Deepseek_7B_FULL": "result_data_deepseek_7b_full.json",
        "Deepseek_7B_LORA": "result_data_deepseek_7b_lora.json",
        "GPT_4o_MINI": "result_data_gpt_4o_mini.json"
    }
    
    print("=" * 60)
    print("各模型原始评分统计")
    print("=" * 60)
    
    # 存储原始平均分
    raw_scores = []
    scores_dict = {}
    
    # 1. 计算每个模型的原始平均分
    for model_name, filename in model_files.items():
        filepath = os.path.join(model_splits_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\n{model_name}: 文件不存在 ({filename})")
            # 存储为0分便于后续处理
            raw_scores.append({
                "Model": model_name,
                "IdeaScore": 0.0,
                "KeyPoints": 0.0,
                "Guidance": 0.0,
                "Correctness": 0.0,
                "Overall": 0.0
            })
            continue
        
        # 读取数据文件
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"文件解析错误: {filepath} - {str(e)}")
                continue
        
        # 检查是否是列表
        if not isinstance(data, list):
            data = [data]
        
        # 初始化分数统计
        scores = {
            "IdeaScore": 0,
            "KeyPoints": 0,
            "Guidance": 0,
            "Correctness": 0,
            "Overall": 0
        }
        count = 0
        
        # 确定score字段名
        score_field = "score"
        
        # 统计评分
        for item in data:
            if score_field in item and item[score_field] is not None:
                try:
                    score_text = item[score_field]
                    
                    # 如果已经是字典格式，直接使用
                    if isinstance(score_text, dict):
                        score_json = score_text
                    else:
                        # 使用正则表达式提取JSON部分
                        json_match = re.search(r'\{.*\}', score_text, re.DOTALL)
                        if json_match:
                            score_json = json.loads(json_match.group())
                        else:
                            continue
                    
                    # 检查并提取各维度分数
                    idea_score = score_json.get("思路评分", score_json.get("ideaRating", 0))
                    key_points = score_json.get("关键点", score_json.get("keyPoints", 0))
                    guidance = score_json.get("引导性", score_json.get("leadership", 0))
                    correctness = score_json.get("正确性", 0)
                    
                    # 累加分数
                    scores["IdeaScore"] += idea_score
                    scores["KeyPoints"] += key_points
                    scores["Guidance"] += guidance
                    scores["Correctness"] += correctness
                    
                    # 计算综合得分
                    avg = (idea_score + key_points + guidance + correctness) / 4
                    scores["Overall"] += avg
                    
                    count += 1
                        
                except (KeyError, TypeError) as e:
                    print(f"警告：{model_name} 项目ID {item.get('id', 'N/A')} 的score字段解析失败: {e}")
                    continue
        
        # 计算平均分
        if count > 0:
            model_scores = {
                "Model": model_name,
                "IdeaScore": round(scores["IdeaScore"] / count, 3),
                "KeyPoints": round(scores["KeyPoints"] / count, 3),
                "Guidance": round(scores["Guidance"] / count, 3),
                "Correctness": round(scores["Correctness"] / count, 3),
                "Overall": round(scores["Overall"] / count, 3),
                "Count": count
            }
        else:
            model_scores = {
                "Model": model_name,
                "IdeaScore": 0.0,
                "KeyPoints": 0.0,
                "Guidance": 0.0,
                "Correctness": 0.0,
                "Overall": 0.0,
                "Count": 0
            }
        
        # 打印统计结果
        print(f"\n{model_name} 评分统计：")
        print("-" * 30)
        print(f"思路评分: {model_scores['IdeaScore']:.3f}")
        print(f"关键点: {model_scores['KeyPoints']:.3f}")
        print(f"引导性: {model_scores['Guidance']:.3f}")
        print(f"正确性: {model_scores['Correctness']:.3f}")
        print(f"综合得分: {model_scores['Overall']:.3f}")
        print(f"样本数: {count}")
        
        raw_scores.append(model_scores)
        scores_dict[model_name] = model_scores
    
    # 转换为DataFrame
    raw_df = pd.DataFrame(raw_scores)
    
    # 2. 应用排名转换算法
    print("\n" + "=" * 60)
    print("各模型排名转换分数")
    print("=" * 60)
    
    # 排名转换函数
    def convert_to_rank_score(column_series):
        # 按值降序排序并记录索引
        sorted_series = column_series.sort_values(ascending=False)
        
        # 计算排名（处理并列情况）
        ranks = {}
        seen = {}
        
        # 分组相同值
        for idx, value in sorted_series.items():
            if value not in seen:
                seen[value] = []
            seen[value].append(idx)
        
        # 计算平均排名
        current_rank = 1
        for value, indices in seen.items():
            # 计算平均排名
            avg_rank = current_rank + (len(indices) - 1) / 2
            for idx in indices:
                ranks[idx] = avg_rank
            current_rank += len(indices)
        
        # 转换为分数 (9 - (rank-1))
        scores = []
        for idx in column_series.index:
            rank = ranks.get(idx, 0)
            if rank == 0:
                scores.append(0)
            else:
                score = 9 - (rank - 1)
                scores.append(round(score, 1))
        
        return pd.Series(scores, index=column_series.index)
    
    # 创建转换后的DataFrame
    ranked_df = pd.DataFrame()
    ranked_df["Model"] = raw_df["Model"]
    
    # 对每个评分维度应用转换
    for column in ["IdeaScore", "KeyPoints", "Guidance", "Correctness", "Overall"]:
        ranked_df[column] = convert_to_rank_score(raw_df[column])
    
    # 打印结果
    print(ranked_df.to_string(index=False))
    
    # 同时返回原始分数和转换后分数
    return raw_df, ranked_df
# 示例调用
score = calculate_score_averages()