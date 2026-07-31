import json
from pathlib import Path

import pandas as pd


def calculate_score_averages(model_splits_dir: str = "../experiment/further_analysis/propagation"):
    """Calculate archived path-quality scores and apply rank transformation."""
    model_configs = {
        "GPT_4O_MINI": "4o-mini_finetuned",
        "DEEPSEEK_7B_LORA": "deepseek-7b_lora",
        "DEEPSEEK_7B_FULL": "deepseek-7b_full",
        "MIMO_7B_LORA": "mimo-7b_lora",
        "MIMO_7B_FULL": "mimo-7b_full",
        "QWEN_7B_LORA": "qwen-7b_lora",
        "QWEN_7B_FULL": "qwen-7b_full",
    }

    propagation_dir = Path(model_splits_dir)
    results = {}
    raw_scores_summary = []

    print("=" * 65)
    print(f"{'Model Name':<20} | {'Idea':>6} | {'Key':>6} | {'Lead':>6} | {'Corr':>6} | {'Count':>6}")
    print("-" * 65)

    for model_name, suffix in model_configs.items():
        filename = f"syllabus_{suffix}_deepseek_v3_2_subset_process_prop_analysed.json"
        filepath = propagation_dir / filename

        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        totals = {"idea_score": 0.0, "key_points": 0.0, "guidance": 0.0, "correctness": 0.0}
        valid_count = 0

        for item in data:
            raw_text = item.get("raw_model_response", "")
            if not raw_text:
                continue

            try:
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                if start != -1 and end != 0:
                    score_json = json.loads(raw_text[start:end])
                    mapping = {
                        "idea_score": ["idea-score"],
                        "key_points": ["key-points"],
                        "guidance": ["guidance"],
                        "correctness": ["correctness"],
                    }

                    found = False
                    for standard_key, aliases in mapping.items():
                        for alias in aliases:
                            if alias in score_json:
                                totals[standard_key] += float(score_json[alias])
                                found = True
                                break
                    if found:
                        valid_count += 1
            except Exception:
                continue

        row = {"Model": model_name, "Count": valid_count}
        if valid_count > 0:
            for key in totals:
                average = totals[key] / valid_count
                totals[key] = average
                row[key] = round(average, 3)
            print(
                f"{model_name:<20} | {row['idea_score']:>6.2f} | {row['key_points']:>6.2f} | "
                f"{row['guidance']:>6.2f} | {row['correctness']:>6.2f} | {valid_count:>6}"
            )
        else:
            for key in totals:
                row[key] = 0.0
            print(f"{model_name:<20} | Missing data")

        results[model_name] = totals
        raw_scores_summary.append(row)

    df = pd.DataFrame(raw_scores_summary)

    def rank_score_transform(series):
        """Apply the rank transformation formula `9 - (rank - 1)`."""
        ranks = series.rank(ascending=False, method="average")
        return (9 - (ranks - 1)).round(1)

    ranked_df = pd.DataFrame()
    ranked_df["Model"] = df["Model"]

    dimensions = ["idea_score", "key_points", "guidance", "correctness"]
    for dimension in dimensions:
        ranked_df[dimension] = rank_score_transform(df[dimension])

    df["Overall_Raw"] = df[dimensions].mean(axis=1)
    ranked_df["Overall"] = rank_score_transform(df["Overall_Raw"])

    print("\n" + "=" * 65)
    print("Rank-converted scores of all models")
    print("=" * 65)
    print(ranked_df.to_string(index=False))

    return results


score = calculate_score_averages()
