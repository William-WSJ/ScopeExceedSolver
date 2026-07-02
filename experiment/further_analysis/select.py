import json
from typing import Iterable, List


def filter_target_knowledge_data(json_list: Iterable[str], output_file: str = "targeted_analysis_data.json") -> List[dict]:
    """Extract the Grade 3-4 target-knowledge subset used in further analysis."""
    target_knowledge = [
        "鸡兔同笼问题",
        "积的变化规律",
        "商的变化规律",
        "同增同减问题",
        "倍的概念及其应用",
    ]

    filtered_results = []

    for file_path in json_list:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]

                for item in data:
                    if item.get("knowledge") in target_knowledge and str(item.get("grade")) in ["3", "4"]:
                        filtered_results.append(item)
        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")

    print(f"Filtering completed. Total extracted samples: {len(filtered_results)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)

    return filtered_results


json_list = ["./syllabus_baseline_deepseek_v3_2_direct.json"]
target_data = filter_target_knowledge_data(json_list, output_file="syllabus_baseline_deepseek_v3_2_subset.json")
