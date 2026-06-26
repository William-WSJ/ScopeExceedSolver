import json
import time
import os
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Optional

# ==================== Configuration Area ====================
BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_PATH = "/root/autodl-tmp/models/openai-mirror/gpt-oss-20b"

INPUT_DIR = "/root/autodl-tmp/subset"    # Folder path for pending JSON files
OUTPUT_DIR = "/root/autodl-tmp/process"  # Folder to store output results
TIMEOUT = 120.0                
MAX_RETRIES = 3               
# ============================================================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def build_idea_checklist_prompt(question: str, thought: str, solution: str, acc: bool) -> str:
    """
    Full prompt covering all 22 sub-items:
    1. Strictly map text descriptions of the 22 LaTeX evaluation sub-items.
    2. Utilize the acc field to assist judging the correctness of the solution.
    3. Evaluate the thought based on the solution as reference standard.
    """
    acc_status = "正确" if acc else "错误"
    
    return f"""你是一名极其严谨的数学教育评估专家。
请结合【问题】与提供的【完整解答】及其【正确性标识】，对【解题思路 (thought)】进行深度评估。

### 【评估基准】
- **问题 (Question)**: {question}
- **完整解答 (Solution)**: {solution} (参考基准)
- **解答正确性 (acc)**: {acc} (标识该 solution 事实上是{acc_status}的)
- **待评估思路 (Thought)**: {thought} (评估对象)

---

### 【22 项评分标准清单】

#### 维度一：思路评分 (Logical & Structural, 0-5分)
1) 推理完整性：推理步骤是否形成完整闭环。
2) 逻辑一致性：前提与结论是否保持一致。
3) 严密性：是否存在逻辑跳跃或矛盾。
4) 规则性：是否遵循基本逻辑规则。
5) 认知逻辑：步骤序列是否符合认知逻辑。
6) 自然过渡：相邻步骤间过渡是否自然。
7) 精简性：是否存在冗余或重复步骤。
8) 层次性：整体结构是否具有层次性。
9) 规范性：数学符号使用是否规范。
10) 术语清晰：关键术语是否定义清晰。
11) 表达准确：语言表达是否简洁准确。
12) 推导依据：公式推导过程是否清晰。

#### 维度二：关键点覆盖 (Key Point Coverage, 0-5分)
13) 核心识别：是否识别题目核心考查点。
14) 公式应用：是否应用关键公式/定理。
15) 突破路径：是否找到解题突破口。
16) 全局性：关键点是否贯穿解题全过程。

#### 维度三：引导力有效性 (Guidance Effectiveness, 0-10分)
17) 方向明确：解题方向是否清晰明确。
18) 分步引导：是否提供分步引导。
19) 框架构建：是否建立可操作的解题框架。
20) 可操作性：引导内容是否具有实际操作性（不考虑正确性）。

#### 维度四：解答正确性 (Solution Correctness, 0-10分)
21) 答案正确：最终答案是否正确。
22) 步骤正确：解题步骤是否正确。

---

### 【输出要求】
请以 JSON 格式返回评分结果。禁止任何额外文字。格式如下：
{{
  "思路评分": 整数(0-5),
  "关键点": 整数(0-5),
  "引导力": 整数(0-10),
  "正确性": 整数(0-10)
}}"""

def call_model_full(prompt: str) -> str:
    """Robust non-streaming request function: handle NoneType and print complete response"""
    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
            stream=False,
            timeout=TIMEOUT
        )
        # ==================== Format Print Zone ====================
        print("\n" + "="*30 + " API RESPONSE DEBUG " + "="*30)
        # Use model_dump_json to format and print full response object
        print(response.model_dump_json(indent=2))
        print("="*80)
        # ==========================================================
        content = response.choices[0].message.content
        # print(content)
        if content is None:
            return ""
        # Print raw model output in real time
        print(f"\n[Model Response]:\n{content}")
        return content.strip()
    except Exception as e:
        print(f"\n   ⚠️ API request exception occurred: {e}")
        return ""

def process_file(file_path: Path):
    """Process single JSON file sequentially: save raw model output directly"""
    output_path = Path(OUTPUT_DIR) / f"{file_path.stem}_process.json"
    print(f"\n📂 Start processing file: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    results = []
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] ID: {item.get('id')} ", end="", flush=True)
        
        raw_ans = ""
        # 3-time retry mechanism
        for attempt in range(1, MAX_RETRIES + 1):
            prompt = build_idea_checklist_prompt(
                question=item.get('question', ''),
                thought=item.get('thought', ''),
                solution=item.get('solution', ''),
                acc=item.get('acc', True)
            )
            raw_ans = call_model_full(prompt)
            
            if raw_ans:
                print("-> OK", end=" ")
                break
            else:
                print(f"-> R{attempt}", end=" ")
                time.sleep(1)

        # Deep copy original data and append raw model response field
        item_copy = item.copy()
        item_copy["raw_model_response"] = raw_ans
        results.append(item_copy)
        print() 

    # Save full dataset immediately after finishing one file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ File {file_path.name} processed and saved successfully.")

def main():
    # Initialize directory paths
    in_p = Path(INPUT_DIR)
    out_p = Path(OUTPUT_DIR)
    out_p.mkdir(parents=True, exist_ok=True)
    
    # Sequential traversal: a.json -> b.json -> c.json
    json_files = sorted([f for f in in_p.glob("*.json") if "_process" not in f.name])
    
    if not json_files:
        print(f"No JSON files found under directory {INPUT_DIR}.")
        return

    for f in json_files:
        process_file(f)

    print("\n" + "="*50 + "\n🎊 All files processing completed!\n" + "="*50)

if __name__ == "__main__":
    main()