def calculate_score_averages(model_splits_dir="experiment/model_splits"):
    """
    统计model_splits目录中所有模型的score字段平均得分，并应用排名转换
    
    Args:
        model_splits_dir (str): model_splits目录路径
    
    Returns:
        dict: 包含各模型平均得分的字典
    """
    import re
    import pandas as pd
    
    # 定义所有模型文件（包括GPT-4o-mini）
    model_files = {
        "QWEN_7B_FULL": "result_data_qwen_7b_full.json",
        "QWEN_7B_LORA": "result_data_qwen_7b_lora.json", 
        "QWEN_1_5B_FULL": "result_data_qwen_1_5b_full.json",
        "QWEN_1_5B_LORA": "result_data_qwen_1_5b_lora.json",
        "DEEPSEEK_7B_FULL": "result_data_deepseek_7b_full.json",
        "DEEPSEEK_7B_LORA": "result_data_deepseek_7b_lora.json",
        "GPT_4O_MINI": "result_data_gpt_4o_mini.json"
    }
    
    print("=" * 60)
    print("各模型评分统计")
    print("=" * 60)
    
    results = {}
    raw_scores = []
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(model_splits_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\n{model_name}: 文件不存在 ({filename})")
            results[model_name] = {
                "count": 0,
                "思路评分": 0.0,
                "关键点": 0.0,
                "引导性": 0.0,
                "正确性": 0.0
            }
            raw_scores.append({
                "Model": model_name,
                "思路评分": 0.0,
                "关键点": 0.0,
                "引导性": 0.0,
                "正确性": 0.0
            })
            continue
        
        # 读取数据文件
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查是否是列表
        if not isinstance(data, list):
            data = [data]
        
        # 初始化分数统计
        total_scores = {
            "思路评分": 0,
            "关键点": 0,
            "引导性": 0,
            "正确性": 0
        }
        count = 0
        
        # 确定score字段名
        if model_name == "GPT_4O_MINI":
            score_field = "score_gpt_4o_mini"
        elif model_name == "QWEN_7B_FULL":
            score_field = "score_qwen_7b_full"
        else:
            score_field = f"score"
        
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
                    
                    # 累加各项分数，处理可能的字段名映射
                    score_mapping = {
                        "思路评分": ["思路评分", "ideaRating"],
                        "关键点": ["关键点", "keyPoints"], 
                        "引导性": ["引导性", "引导力", "leadership"],
                        "正确性": ["正确性", "correctness"]
                    }
                    
                    # 处理每个维度的分数
                    dimensions_found = 0
                    for standard_key, possible_keys in score_mapping.items():
                        for key in possible_keys:
                            if key in score_json:
                                total_scores[standard_key] += score_json[key]
                                dimensions_found += 1
                                break
                    
                    if dimensions_found > 0:
                        count += 1
                        
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"警告：{model_name} 项目ID {item.get('id', 'N/A')} 的score字段解析失败: {e}")
                    continue
        
        # 计算并输出平均分
        print(f"\n{model_name} 原始评分统计：")
        print("-" * 30)
        
        model_results = {"count": count}
        model_raw = {"Model": model_name}
        
        for key, total in total_scores.items():
            if count > 0:
                average = total / count
                print(f"  {key}: {average:.2f}")
                model_results[key] = average
                model_raw[key] = average
            else:
                print(f"  {key}: 0.00")
                model_results[key] = 0.0
                model_raw[key] = 0.0
        
        print(f"  样本数: {count}")
        results[model_name] = model_results
        raw_scores.append(model_raw)
    
    # 转换为DataFrame
    raw_df = pd.DataFrame(raw_scores)
    
    # 应用排名转换算法
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
    for column in ["思路评分", "关键点", "引导性", "正确性"]:
        ranked_df[column] = convert_to_rank_score(raw_df[column])
    
    # 计算Overall排名分数
    raw_df["Overall"] = raw_df[["思路评分", "关键点", "引导性", "正确性"]].mean(axis=1)
    ranked_df["Overall"] = convert_to_rank_score(raw_df["Overall"])
    
    # 打印结果
    print(ranked_df.to_string(index=False))
    
    print("\n" + "=" * 60)
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
# score = calculate_score_averages()