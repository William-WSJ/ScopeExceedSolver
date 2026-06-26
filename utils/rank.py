import os
import json
import re
import pandas as pd
from pathlib import Path

def calculate_score_averages(model_splits_dir="../experiment/further_analysis/propagation"):
    """
    Calculate reasoning metric scores for SPS experiment data and apply rank transformation.
    """
    # 1. Define models and their corresponding filename suffixes
    # File naming format: syllabus_{suffix}_deepseek_v3_2_subset_process_prop_analysed.json
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

        # Use English keys for statistics storage, Chinese only as source file aliases
        total_scores = {"idea_score": 0.0, "key_points": 0.0, "guidance": 0.0, "correctness": 0.0}
        valid_count = 0

        for item in data:
            # Extract content uniformly from raw_model_response, which stores raw evaluation output
            raw_text = item.get("raw_model_response", "")
            if not raw_text:
                continue

            try:
                # Extract embedded JSON snippet
                start = raw_text.find('{')
                end = raw_text.rfind('}') + 1
                if start != -1 and end != 0:
                    score_json = json.loads(raw_text[start:end])
                    
                    # Field mapping: English standard key -> list of original Chinese field names in source file
                    mapping = {
                        "idea_score": ["思路评分"],
                        "key_points": ["关键点"],
                        "guidance": ["引导力", "引导性"],
                        "correctness": ["正确性"]
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
            except Exception:
                continue

        # Calculate average values
        row = {"Model": nick_name, "Count": valid_count}
        if valid_count > 0:
            for k in total_scores:
                avg = total_scores[k] / valid_count
                total_scores[k] = avg
                row[k] = round(avg, 3)
            
            # Print still uses original Chinese source keys for alignment with raw data display
            print(f"{nick_name:<20} | {row['idea_score']:>6.2f} | {row['key_points']:>6.2f} | {row['guidance']:>6.2f} | {row['correctness']:>6.2f} | {valid_count:>6}")
        else:
            for k in total_scores:
                row[k] = 0.0
            print(f"{nick_name:<20} | Missing Data")

        results[nick_name] = total_scores
        raw_scores_summary.append(row)

    # --- Rank Transformation Pipeline ---
    df = pd.DataFrame(raw_scores_summary)
    
    def rank_score_transform(series):
        """Standard rank transformation formula: 9 - (rank - 1)"""
        # Descending sort ranking, average method to resolve tied values
        ranks = series.rank(ascending=False, method='average')
        return (9 - (ranks - 1)).round(1)

    ranked_df = pd.DataFrame()
    ranked_df["Model"] = df["Model"]
    
    # Use unified English dimension identifiers
    dimensions = ["idea_score", "key_points", "guidance", "correctness"]
    for dim in dimensions:
        ranked_df[dim] = rank_score_transform(df[dim])

    # Compute overall composite score and generate final rank-based metrics
    df["Overall_Raw"] = df[dimensions].mean(axis=1)
    ranked_df["Overall"] = rank_score_transform(df["Overall_Raw"])

    print("\n" + "=" * 65)
    print("Rank-Converted Scores of All Models (Rank-based Scores)")
    print("=" * 65)
    print(ranked_df.to_string(index=False))
    
    return results

# Sample function invocation
score = calculate_score_averages()