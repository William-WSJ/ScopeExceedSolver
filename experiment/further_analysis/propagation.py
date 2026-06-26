import json
import time
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List, Optional

# ==================== 配置区 ====================
BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_PATH = "/root/autodl-tmp/models/openai-mirror/gpt-oss-20b"

INPUT_DIR = "./process"      # 评分结果文件夹
OUTPUT_DIR = "./propagation" # 传播分析结果文件夹
TIMEOUT = 120.0
MAX_RETRIES = 3
# ==============================================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- 提示语模板 ---

PATH_ERROR_PROMPT_TEMPLATE = """
任务：分析小学数学题的"思路"，判断其是否存在错误。若无错误请输出0，若有错误请输出对应的类型编号。

错误类型定义：
0. 思路正确：思路逻辑清晰，无任何错误
1. 逻辑不完整：思路缺少关键环节，无法完整引导解题
2. 步骤跳跃：解题步骤间存在大跨度跳跃，不符合学生认知水平
3. 分析错误：对问题条件或要求的分析存在错误
4. 解答错误：思路本身包含错误的解法或计算
5. 未知错误：不属于以上任何类型

题目：{question}
思路：{thought}
错误类型编号："""

EXECUTION_ERROR_PROPAGATION_PROMPT_TEMPLATE = """
任务：判断以下小学数学题的"解答结果"对"思路"中错误的处理情况，仅输出传播类型编号（1/2/3）。
传播类型定义：
1. 延续错误：解答完全遵循了思路中的错误，没有纠正 (CER)
2. 修正错误：解答识别并修正了思路中的错误 (FR)
3. 误判错误：解答在思路正确的情况下引入了新的错误 (MER)

题目：{question}
思路：{thought}
解答结果：{solve}
传播类型编号："""

# --- 核心辅助函数 ---

def parse_correctness_score(raw_response: str) -> int:
    """从 raw_model_response 字符串中解析出 '正确性' 分数"""
    try:
        if not raw_response: return 10
        # 寻找 JSON 块
        start = raw_response.find('{')
        end = raw_response.rfind('}') + 1
        if start != -1 and end != 0:
            data = json.loads(raw_response[start:end])
            return int(data.get("正确性", 10))
    except:
        pass
    return 10 # 解析失败默认给10，跳过分析以保安全

def call_model_for_code(prompt: str) -> str:
    """调用模型并提取输出中的第一个数字编号"""
    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
            stream=False,
            timeout=TIMEOUT
        )
        # ==================== 格式化输出区 ====================
        print("\n" + "="*30 + " API RESPONSE DEBUG " + "="*30)
        # 使用 model_dump_json 格式化打印整个响应对象
        print(response.model_dump_json(indent=2))
        print("="*80)
        msg = response.choices[0].message
        content = msg.content or getattr(msg, 'reasoning_content', "")
        if not content: return "5"
        
        # 提取第一个数字
        for char in content.strip():
            if char.isdigit(): return char
        return "5"
    except:
        return "5"

# --- 业务逻辑 ---

def process_file(file_path: Path):
    output_path = Path(OUTPUT_DIR) / f"{file_path.stem}_prop_analysed.json"
    print(f"\n📂 正在分析文件: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    processed_results = []
    # 局部统计初始化
    stats = {
        "Logical Incompleteness": {"count": 0, "FR": 0, "CER": 0, "MER": 0},
        "Step Skipping": {"count": 0, "FR": 0, "CER": 0, "MER": 0},
        "Analysis Error": {"count": 0, "FR": 0, "CER": 0, "MER": 0},
        "Solution Error": {"count": 0, "FR": 0, "CER": 0, "MER": 0}
    }
    error_found_count = 0

    for item in tqdm(items):
        # 1. 判断是否需要进行错误分析
        raw_res = item.get("raw_model_response", "")
        correctness_score = parse_correctness_score(raw_res)
        
        # 逻辑：只要正确性评分 < 10，就判定为存在缺陷
        if correctness_score >= 10:
            item["path_error_type"] = "Correct"
            processed_results.append(item)
            continue

        # 2. 判定思路错误类型 (Path Error)
        err_code = call_model_for_code(PATH_ERROR_PROMPT_TEMPLATE.format(
            question=item.get("question", ""),
            thought=item.get("thought", "")
        ))
        
        error_map = {
            "1": "Logical Incompleteness", 
            "2": "Step Skipping", 
            "3": "Analysis Error", 
            "4": "Solution Error"
        }
        err_type = error_map.get(err_code)

        if err_type:
            error_found_count += 1
            stats[err_type]["count"] += 1
            
            # 3. 判定传播情况 (Propagation)
            prop_code = call_model_for_code(EXECUTION_ERROR_PROPAGATION_PROMPT_TEMPLATE.format(
                question=item.get("question", ""),
                thought=item.get("thought", ""),
                solve=item.get("solution", "")
            ))
            
            prop_map = {"1": "CER", "2": "FR", "3": "MER"}
            prop_type = prop_map.get(prop_code, "CER") # 默认归类为延续错误
            
            stats[err_type][prop_type] += 1
            item["path_error_type"] = err_type
            item["propagation_type"] = prop_type
        else:
            item["path_error_type"] = "Unknown/Correct"

        processed_results.append(item)

    # 保存单个分析结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, ensure_ascii=False, indent=2)
    
    return stats, error_found_count

def print_latex_summary(grand_stats, grand_total):
    """输出符合要求的 LaTeX 表格代码"""
    print("\n" + "="*25 + " LaTeX TABLE START " + "="*25)
    print(r"\begin{table*}[!htbp]")
    print(r"  \centering")
    print(r"  \caption{Error propagation analysis results (n=" + str(grand_total) + r").}")
    print(r"  \rowcolors{3}{gray!20}{white}")
    print(r"  \begin{tabular}{cccccc}")
    print(r"    \toprule")
    print(r"    Error Type & Sample & Proportion & FR & CER & MER \\")
    print(r"    \midrule")
    
    for name, data in grand_stats.items():
        count = data["count"]
        if grand_total == 0 or count == 0:
            print(f"    {name} & {count} & 0.0\% & 0.0\% & 0.0\% & 0.0\% \\\\")
            continue
            
        prop = (count / grand_total) * 100
        fr = (data["FR"] / count) * 100
        cer = (data["CER"] / count) * 100
        mer = (data["MER"] / count) * 100
        print(f"    {name} & {count} & {prop:.1f}\\% & {fr:.1f}\\% & {cer:.1f}\\% & {mer:.1f}\\% \\\\")
        
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table*}")
    print("="*60)

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 全局汇总统计
    grand_stats = {
        "Logical Incompleteness": {"count": 0, "FR": 0, "CER": 0, "MER": 0},
        "Step Skipping": {"count": 0, "FR": 0, "CER": 0, "MER": 0},
        "Analysis Error": {"count": 0, "FR": 0, "CER": 0, "MER": 0},
        "Solution Error": {"count": 0, "FR": 0, "CER": 0, "MER": 0}
    }
    grand_total_errors = 0

    json_files = sorted([f for f in Path(INPUT_DIR).glob("*.json") if "_propagation" not in f.name])
    
    if not json_files:
        print(f"未在 {INPUT_DIR} 找到 JSON 文件")
        return

    for f in json_files:
        f_stats, f_count = process_file(f)
        grand_total_errors += f_count
        for k in grand_stats:
            for field in ["count", "FR", "CER", "MER"]:
                grand_stats[k][field] += f_stats[k][field]

    # 输出最终的 LaTeX 表格
    print_latex_summary(grand_stats, grand_total_errors)

if __name__ == "__main__":
    main()