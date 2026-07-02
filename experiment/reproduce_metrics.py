import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiment"


@dataclass
class MetricSummary:
    count: int
    accuracy_mean: Optional[float]
    accuracy_margin: Optional[float]
    osr_mean: Optional[float]
    osr_margin: Optional[float]


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def sample_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


Z_VALUE = 1.96


def load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def summarize_result_file(path: Path) -> MetricSummary:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list-like result data in {path}")

    grade_groups = defaultdict(lambda: {"acc": [], "osr": [], "count": 0})
    for item in data:
        grade = item.get("grade")
        if grade is None:
            continue
        grade = int(grade)
        acc_value = 1.0 if parse_bool(item.get("acc", False)) else 0.0
        osr_value = 1.0 if parse_bool(item.get("exceeds_scope", False)) else 0.0
        grade_groups[grade]["acc"].append(acc_value)
        grade_groups[grade]["osr"].append(osr_value)
        grade_groups[grade]["count"] += 1

    total_n = sum(group["count"] for group in grade_groups.values())
    if total_n == 0:
        return MetricSummary(
            count=len(data),
            accuracy_mean=None,
            accuracy_margin=None,
            osr_mean=None,
            osr_margin=None,
        )

    acc_weighted_mean = 0.0
    osr_weighted_mean = 0.0
    acc_weighted_std = 0.0
    osr_weighted_std = 0.0
    for group in grade_groups.values():
        count = group["count"]
        acc_mean = sum(group["acc"]) / count
        osr_mean = sum(group["osr"]) / count
        acc_std = sample_std(group["acc"])
        osr_std = sample_std(group["osr"])
        acc_weighted_mean += acc_mean * count
        osr_weighted_mean += osr_mean * count
        acc_weighted_std += acc_std * count
        osr_weighted_std += osr_std * count

    acc_weighted_mean /= total_n
    osr_weighted_mean /= total_n
    acc_weighted_std /= total_n
    osr_weighted_std /= total_n

    acc_margin = Z_VALUE * (acc_weighted_std / (total_n ** 0.5))
    osr_margin = Z_VALUE * (osr_weighted_std / (total_n ** 0.5))

    return MetricSummary(
        count=len(data),
        accuracy_mean=acc_weighted_mean,
        accuracy_margin=acc_margin,
        osr_mean=osr_weighted_mean,
        osr_margin=osr_margin,
    )


def read_stats_file(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_correctness_score(raw_model_response: str) -> Optional[int]:
    if not raw_model_response:
        return None
    start = raw_model_response.find("{")
    end = raw_model_response.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(raw_model_response[start:end])
    except json.JSONDecodeError:
        return None
    for key in ("正确性", "correctness"):
        if key in payload:
            try:
                return int(payload[key])
            except Exception:
                return None
    return None


def summarize_error_propagation(path: Path) -> Tuple[int, Dict[str, Dict[str, float]]]:
    data = load_json(path)
    stats: Dict[str, Dict[str, int]] = {}
    total = 0
    for item in data:
        score = parse_correctness_score(item.get("raw_model_response", ""))
        if score is None or score >= 10:
            continue
        error_type = item.get("path_error_type")
        if not error_type or error_type == "Correct":
            continue
        stats.setdefault(error_type, {"count": 0, "FR": 0, "CER": 0, "MER": 0})
        stats[error_type]["count"] += 1
        total += 1
        propagation_type = item.get("propagation_type")
        if propagation_type in {"FR", "CER", "MER"}:
            stats[error_type][propagation_type] += 1

    normalized: Dict[str, Dict[str, float]] = {}
    for error_type, values in stats.items():
        count = values["count"]
        normalized[error_type] = {
            "count": count,
            "proportion": (count / total) if total else 0.0,
            "FR": (values["FR"] / count) if count else 0.0,
            "CER": (values["CER"] / count) if count else 0.0,
            "MER": (values["MER"] / count) if count else 0.0,
        }
    return total, normalized


def summarize_rank_inputs(directory: Path) -> List[Dict[str, float]]:
    model_configs = {
        "GPT-4o mini finetuned": "4o-mini_finetuned",
        "Deepseek-Instruct LoRA": "deepseek-7b_lora",
        "Deepseek-Instruct full": "deepseek-7b_full",
        "Mimo-RL LoRA": "mimo-7b_lora",
        "Mimo-RL full": "mimo-7b_full",
        "Qwen-Instruct LoRA": "qwen-7b_lora",
        "Qwen-Instruct full": "qwen-7b_full",
    }
    rows = []
    for name, suffix in model_configs.items():
        path = directory / f"syllabus_{suffix}_deepseek_v3_2_subset_process_prop_analysed.json"
        if not path.exists():
            continue
        data = load_json(path)
        totals = {"idea_score": 0.0, "key_points": 0.0, "guidance": 0.0, "correctness": 0.0}
        valid = 0
        for item in data:
            raw = item.get("raw_model_response", "")
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end <= start:
                continue
            try:
                payload = json.loads(raw[start:end])
            except json.JSONDecodeError:
                continue
            mapping = {
                "idea_score": ["思路评分"],
                "key_points": ["关键点"],
                "guidance": ["引导力", "引导性"],
                "correctness": ["正确性"],
            }
            found = False
            for key, aliases in mapping.items():
                for alias in aliases:
                    if alias in payload:
                        totals[key] += float(payload[alias])
                        found = True
                        break
            if found:
                valid += 1
        if valid:
            rows.append({"model": name, **{k: totals[k] / valid for k in totals}, "count": valid})
    return rows


def format_metric(summary: MetricSummary) -> str:
    if summary.accuracy_mean is None or summary.osr_mean is None:
        return f"n={summary.count}, metrics unavailable (missing acc/exceeds_scope fields)"
    return (
        f"n={summary.count}, accuracy={summary.accuracy_mean*100:.2f}±{summary.accuracy_margin*100:.2f}%, "
        f"osr={summary.osr_mean*100:.2f}±{summary.osr_margin*100:.2f}%"
    )


def report_main_tables() -> None:
    mappings = {
        "Baseline SyllabusCheck": {
            "GPT-5.2 direct": EXPERIMENT_ROOT / "baseline-result" / "syllabus_baseline_gpt_5_2_direct.json",
            "GPT-5.2 constrained": EXPERIMENT_ROOT / "baseline-result" / "syllabus_baseline_gpt_5_2_with_limitations.json",
            "Deepseek-V3.2 direct": EXPERIMENT_ROOT / "baseline-result" / "syllabus_baseline_deepseek_v3_2_direct.json",
            "Deepseek-V3.2 constrained": EXPERIMENT_ROOT / "baseline-result" / "syllabus_baseline_deepseek_v3_2_with_limitations.json",
            "Gemini-3-flash direct": EXPERIMENT_ROOT / "baseline-result" / "syllabus_baseline_gemini_3_flash_direct.json",
            "Gemini-3-flash constrained": EXPERIMENT_ROOT / "baseline-result" / "syllabus_baseline_gemini_3_flash_with_limitations.json",
        },
        "Baseline CMATH": {
            "GPT-5.2 direct": EXPERIMENT_ROOT / "baseline-result" / "cmath_baseline_gpt_5_2_direct.json",
            "GPT-5.2 constrained": EXPERIMENT_ROOT / "baseline-result" / "cmath_baseline_gpt_5_2_with_limitations.json",
            "Deepseek-V3.2 direct": EXPERIMENT_ROOT / "baseline-result" / "cmath_baseline_deepseek_v3_2_direct.json",
            "Deepseek-V3.2 constrained": EXPERIMENT_ROOT / "baseline-result" / "cmath_baseline_deepseek_v3_2_with_limitations.json",
            "Gemini-3-flash direct": EXPERIMENT_ROOT / "baseline-result" / "cmath_baseline_gemini_3_flash_direct.json",
            "Gemini-3-flash constrained": EXPERIMENT_ROOT / "baseline-result" / "cmath_baseline_gemini_3_flash_with_limitations.json",
        },
        "Idea-only SyllabusCheck": {
            "Qwen base": EXPERIMENT_ROOT / "idea-result" / "syllabus_qwen-7b_base.json",
            "Qwen finetuned": EXPERIMENT_ROOT / "idea-result" / "syllabus_qwen-7b_full.json",
            "Deepseek base": EXPERIMENT_ROOT / "idea-result" / "syllabus_deepseek-7b_base.json",
            "Deepseek finetuned": EXPERIMENT_ROOT / "idea-result" / "syllabus_deepseek-7b_full.json",
            "Mimo base": EXPERIMENT_ROOT / "idea-result" / "syllabus_mimo-7b_base.json",
            "Mimo finetuned": EXPERIMENT_ROOT / "idea-result" / "syllabus_mimo-7b_full.json",
            "GPT-4o mini base": EXPERIMENT_ROOT / "idea-result" / "syllabus_4o-mini_base.json",
            "GPT-4o mini finetuned": EXPERIMENT_ROOT / "idea-result" / "syllabus_4o-mini_finetuned.json",
        },
        "Idea-only CMATH": {
            "Qwen base": EXPERIMENT_ROOT / "idea-result" / "cmath_qwen-7b_base.json",
            "Qwen finetuned": EXPERIMENT_ROOT / "idea-result" / "cmath_qwen-7b_full.json",
            "Deepseek base": EXPERIMENT_ROOT / "idea-result" / "cmath_deepseek-7b_base.json",
            "Deepseek finetuned": EXPERIMENT_ROOT / "idea-result" / "cmath_deepseek-7b_full.json",
            "Mimo base": EXPERIMENT_ROOT / "idea-result" / "cmath_mimo-7b_base.json",
            "Mimo finetuned": EXPERIMENT_ROOT / "idea-result" / "cmath_mimo-7b_full.json",
            "GPT-4o mini base": EXPERIMENT_ROOT / "idea-result" / "cmath_4o-mini_base.json",
            "GPT-4o mini finetuned": EXPERIMENT_ROOT / "idea-result" / "cmath_4o-mini_finetuned.json",
        },
        "Idea+Constraint SyllabusCheck": {
            "Qwen -> GPT-5.2": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_qwen-7b_full_gpt5_2.json",
            "Deepseek -> GPT-5.2": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_deepseek-7b_full_gpt5_2.json",
            "GPT-4o mini -> GPT-5.2": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_4o-mini_finetuned_gpt5_2.json",
            "Qwen -> Deepseek-V3.2": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_qwen-7b_full_deepseek_v3_2.json",
            "Deepseek -> Deepseek-V3.2": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_deepseek-7b_full_deepseek_v3_2.json",
            "GPT-4o mini -> Deepseek-V3.2": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_4o-mini_finetuned_deepseek_v3_2.json",
            "Qwen -> Gemini-3-flash": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_qwen-7b_full_gemini3.json",
            "Deepseek -> Gemini-3-flash": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_deepseek-7b_full_gemini3.json",
            "GPT-4o mini -> Gemini-3-flash": EXPERIMENT_ROOT / "idea-limitation-result" / "syllabus_4o-mini_finetuned_gemini3.json",
        },
        "Idea+Constraint CMATH": {
            "Qwen -> GPT-5.2": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_qwen-7b_full_gpt5_2.json",
            "Deepseek -> GPT-5.2": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_deepseek-7b_full_gpt5_2.json",
            "GPT-4o mini -> GPT-5.2": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_4o-mini_finetuned_gpt5_2.json",
            "Qwen -> Deepseek-V3.2": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_qwen-7b_full_deepseek_v3_2.json",
            "Deepseek -> Deepseek-V3.2": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_deepseek-7b_full_deepseek_v3_2.json",
            "GPT-4o mini -> Deepseek-V3.2": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_4o-mini_finetuned_deepseek_v3_2.json",
            "Qwen -> Gemini-3-flash": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_qwen-7b_full_gemini3.json",
            "Deepseek -> Gemini-3-flash": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_deepseek-7b_full_gemini3.json",
            "GPT-4o mini -> Gemini-3-flash": EXPERIMENT_ROOT / "idea-limitation-result" / "cmath_4o-mini_finetuned_gemini3.json",
        },
        "Further Analysis Subset": {
            "Baseline": EXPERIMENT_ROOT / "further_analysis" / "syllabus_baseline_deepseek_v3_2_subset.json",
            "Qwen-Instruct LoRA": EXPERIMENT_ROOT / "further_analysis" / "subset" / "syllabus_qwen-7b_lora_deepseek_v3_2_subset.json",
            "Qwen-Instruct full": EXPERIMENT_ROOT / "further_analysis" / "subset" / "syllabus_qwen-7b_full_deepseek_v3_2_subset.json",
            "Deepseek-Instruct LoRA": EXPERIMENT_ROOT / "further_analysis" / "subset" / "syllabus_deepseek-7b_lora_deepseek_v3_2_subset.json",
            "Deepseek-Instruct full": EXPERIMENT_ROOT / "further_analysis" / "subset" / "syllabus_deepseek-7b_full_deepseek_v3_2_subset.json",
            "Mimo-RL LoRA": EXPERIMENT_ROOT / "further_analysis" / "subset" / "syllabus_mimo-7b_lora_deepseek_v3_2_subset.json",
            "Mimo-RL full": EXPERIMENT_ROOT / "further_analysis" / "subset" / "syllabus_mimo-7b_full_deepseek_v3_2_subset.json",
            "GPT-4o mini finetuned": EXPERIMENT_ROOT / "further_analysis" / "subset" / "syllabus_4o-mini_finetuned_deepseek_v3_2_subset.json",
        },
    }

    for section, files in mappings.items():
        print(f"\n=== {section} ===")
        for label, path in files.items():
            print(f"- {label}: {format_metric(summarize_result_file(path))}")


def report_further_analysis() -> None:
    print("\n=== Further Analysis Timing Files ===")
    stats_dir = EXPERIMENT_ROOT / "further_analysis" / "subset"
    for path in sorted(stats_dir.glob("*_stats.json")):
        payload = read_stats_file(path)
        avg_time = payload.get("average_time_seconds")
        print(f"- {path.name}: average_time_seconds={avg_time}")

    print("\n=== Error Propagation ===")
    propagation_dir = EXPERIMENT_ROOT / "further_analysis" / "propagation"
    grand_total = 0
    effective_total = 0
    merged: Dict[str, Dict[str, float]] = {}
    for path in sorted(propagation_dir.glob("*.json")):
        total, stats = summarize_error_propagation(path)
        grand_total += total
        for error_type, values in stats.items():
            slot = merged.setdefault(error_type, {"count": 0, "FR": 0, "CER": 0, "MER": 0})
            slot["count"] += values["count"]
            slot["FR"] += values["FR"] * values["count"]
            slot["CER"] += values["CER"] * values["count"]
            slot["MER"] += values["MER"] * values["count"]
    effective_total = sum(values["count"] for error_type, values in merged.items() if error_type != "Unknown/Correct")
    print(f"- total scanned samples counted from archived propagation files: {grand_total}")
    print(f"- effective error samples used in the manuscript table (excluding Unknown/Correct): {effective_total}")
    for error_type, values in merged.items():
        count = values["count"]
        if not count:
            continue
        proportion_denominator = grand_total if error_type == "Unknown/Correct" else effective_total
        print(
            f"- {error_type}: count={count}, proportion={count / proportion_denominator * 100:.1f}%, "
            f"FR={values['FR'] / count * 100:.1f}%, CER={values['CER'] / count * 100:.1f}%, MER={values['MER'] / count * 100:.1f}%"
        )

    print("\n=== Rank Input Summary ===")
    rows = summarize_rank_inputs(EXPERIMENT_ROOT / "further_analysis" / "propagation")
    for row in rows:
        print(
            f"- {row['model']}: IdeaScore={row['idea_score']:.3f}, KeyPoints={row['key_points']:.3f}, "
            f"Guidance={row['guidance']:.3f}, Correctness={row['correctness']:.3f}, Count={row['count']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce summary metrics from archived ScopeExceedSolver result files.")
    parser.add_argument(
        "--section",
        choices=["tables", "further", "all"],
        default="all",
        help="Which metric group to report.",
    )
    args = parser.parse_args()

    if args.section in {"tables", "all"}:
        report_main_tables()
    if args.section in {"further", "all"}:
        report_further_analysis()


if __name__ == "__main__":
    main()
