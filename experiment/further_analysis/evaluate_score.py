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
    
    return f"""Here is the academic translation of the prompt, meticulously aligned with the descriptions provided in your LaTeX table to ensure consistency in your evaluation metrics:

Please conduct an in-depth evaluation of the [Solution Strategy (Thought)] based on the [Question], the provided [Complete Solution], and its [Correctness Indicator].

### [Evaluation Baseline]

* **Question**: {question}
* **Complete Solution**: {solution} (Reference baseline)
* **Solution Correctness (acc)**: {acc} (Indicates that this solution is factually {acc_status})
* **Strategy to be Evaluated (Thought)**: {thought} (Evaluation object)

---

### [22-Item Scoring Checklist]

#### Dimension 1: Strategy Evaluation (Logical & Structural, 0-5 points)

*Logical Completeness:*

1. **Reasoning Completeness**: Whether the reasoning steps form a complete closed loop.
2. **Logical Consistency**: The consistency between the premises and conclusion.
3. **Rigor**: Presence of any logical jumps or contradictions.
4. **Regularity**: Whether it follows basic logical rules.

*Step Coherence:*
5) **Cognitive Logic**: Whether the step sequence conforms to cognitive logic.
6) **Natural Transition**: Whether the transition between adjacent steps is natural.
7) **Conciseness**: Whether there are redundant or duplicate steps.
8) **Hierarchical Structure**: Whether the overall structure is hierarchical.

*Clarity of Expression:*
9) **Standardization**: Whether the use of mathematical symbols is standardized.
10) **Terminology Clarity**: Whether the key terms are defined clearly.
11) **Expression Accuracy**: Whether the language expressions are concise and accurate.
12) **Derivation Basis**: Whether the process of formula derivation is clear.

#### Dimension 2: Key Point Coverage (0-5 points)

13. **Core Identification**: Whether to identify the core test points.
14. **Formula Application**: Whether to apply key formula theorems.
15. **Breakthrough Path**: Whether to find a breakthrough in solving the problem.
16. **Global Scope**: Whether the key points run through the whole course.

#### Dimension 3: Guidance Effectiveness (0-10 points)

17. **Directional Clarity**: Whether the direction of problem-solving is clear?
18. **Step-by-Step Guidance**: Whether there is a step-by-step guide?
19. **Framework Construction**: Whether to establish a problem-solving framework.
20. **Operability**: Whether it is operable (without considering correctness).

#### Dimension 4: Solution Correctness (0-10 points)

21. **Answer Correctness**: Whether the answer is correct.
22. **Step Correctness**: Whether the solution step is correct.
"""

def call_model_full(prompt: str) -> str:
    """Call the evaluator model and return the raw response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
            stream=False,
            timeout=TIMEOUT
        )
        # ==================== Debug output ====================
        print("\n" + "="*30 + " API RESPONSE DEBUG " + "="*30)
        # Print the full response object for debugging.
        print(response.model_dump_json(indent=2))
        print("="*80)
        # ==========================================================
        content = response.choices[0].message.content
        # print(content)
        if content is None:
            return ""
        # Print the raw model output for debugging.
        print(f"\n[Model Response]:\n{content}")
        return content.strip()
    except Exception as e:
        print(f"\n   ⚠️ API request exception occurred: {e}")
        return ""

def process_file(file_path: Path):
    """Process one subset file and save the raw scoring output."""
    output_path = Path(OUTPUT_DIR) / f"{file_path.stem}_process.json"
    print(f"\n📂 Start processing file: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    results = []
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] ID: {item.get('id')} ", end="", flush=True)
        
        raw_ans = ""
        # Retry up to three times for each sample.
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

        # Copy the original item and append the raw model response.
        item_copy = item.copy()
        item_copy["raw_model_response"] = raw_ans
        results.append(item_copy)
        print() 

    # Save the full scored file after processing completes.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Finished and saved: {output_path.name}")

def main():
    # Initialize input and output directory paths.
    in_p = Path(INPUT_DIR)
    out_p = Path(OUTPUT_DIR)
    out_p.mkdir(parents=True, exist_ok=True)
    
    # Process files sequentially in sorted order.
    json_files = sorted([f for f in in_p.glob("*.json") if "_process" not in f.name])
    
    if not json_files:
        print(f"No JSON files found under {INPUT_DIR}.")
        return

    for f in json_files:
        process_file(f)

    print("\n" + "="*50 + "\n🎊 All files processing completed!\n" + "="*50)

if __name__ == "__main__":
    main()