import json
from tqdm import tqdm
import time
from openai import OpenAI
import sys

client = OpenAI(
    base_url="http://localhost:11434/v1",  # 本地 Ollama API
    api_key="ollama"  # 随便填个 key
)

def chat_stream(prompt):
    stream = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True  # 启用流式响应
    )

    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            # print(content, end="", flush=True)  # 实时输出，不换行
            full_response += content
    # print()  # 最后换行
    return full_response

PATH_ERROR_PROMPT_TEMPLATE = """
任务：判断以下小学数学题的"思路"属于哪种错误类型，仅输出错误类型编号（1/2/3/4/5），无需额外解释。
错误类型定义：
1. 逻辑不完整：思路缺少关键环节，无法完整引导解题
2. 步骤跳跃：解题步骤间存在大跨度跳跃，不符合学生认知水平
3. 分析错误：对问题条件或要求的分析存在错误
4. 解答错误：思路本身包含错误的解法或计算
5. 未知错误：不属于以上任何类型

题目：{question}
思路：{thought}
错误类型编号：
"""

EXECUTION_ERROR_PROPAGATION_PROMPT_TEMPLATE = """
任务：判断以下小学数学题的"解答结果"对"思路"中错误的处理情况，仅输出传播类型编号（1/2/3），无需额外解释。
传播类型定义：
1. 延续错误：解答完全遵循了思路中的错误，没有纠正
2. 修正错误：解答识别并修正了思路中的错误
3. 误判错误：解答在思路正确的情况下引入了新的错误

题目：{question}
思路：{thought}
解答结果：{solve}
传播类型编号：
"""

def analyze_path_errors_with_template(json_file_path):
    """使用模板分析路径生成错误类型"""
    # 加载数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保数据是列表格式
    if isinstance(data, dict) and 'data' in data:
        samples = data['data']
    elif isinstance(data, list):
        samples = data
    else:
        print("数据格式不支持")
        return []
    
    print(f"开始分析 {len(samples)} 条样本的思路错误类型")
    
    results = []
    
    for sample in tqdm(samples, desc="分析思路错误类型"):
        # 构造提示
        prompt = PATH_ERROR_PROMPT_TEMPLATE.format(
            question=sample["question"],
            thought=sample["thoughts_qwen_7b_full"]
        )
        
        # 调用您的chat_stream函数
        error_type_code = chat_stream(prompt).strip()
        
        # 将错误类型编号映射为具体类型
        error_types = {
            "1": "逻辑不完整",
            "2": "步骤跳跃",
            "3": "分析错误",
            "4": "解答错误",
            "5": "未知错误"
        }
        
        # 处理可能的无效响应
        error_type = error_types.get(error_type_code, "未知错误")
        sample["path_error_type"] = error_type
        sample["path_error_code"] = error_type_code if error_type_code in error_types else "5"
        results.append(sample)
        
        # 避免API调用过于频繁
        time.sleep(0.3)
    
    return results

def analyze_error_propagation_with_template(samples_with_errors):
    """分析错误传播类型"""
    results = []
    
    # 过滤出有错误的样本（不包括"未知错误"）
    error_samples = [s for s in samples_with_errors 
                    if s.get("path_error_code") in ["1", "2", "3", "4"]]
    
    print(f"开始分析 {len(error_samples)} 条有错误样本的传播情况")
    
    for sample in tqdm(error_samples, desc="分析错误传播类型"):
        # 构造提示
        prompt = EXECUTION_ERROR_PROPAGATION_PROMPT_TEMPLATE.format(
            question=sample["question"],
            thought=sample["thoughts_qwen_7b_full"],
            solve=sample["solve_have_grade_cautions_qwen_7b_full"]
        )
        
        # 调用您的chat_stream函数
        propagation_code = chat_stream(prompt).strip()
        
        # 将传播类型编号映射为具体类型
        propagation_types = {
            "1": "延续错误",
            "2": "修正错误",
            "3": "误判错误"
        }
        
        # 处理可能的无效响应
        propagation_type = propagation_types.get(propagation_code, "未知传播")
        sample["propagation_type"] = propagation_type
        sample["propagation_code"] = propagation_code if propagation_code in propagation_types else "3"
        results.append(sample)
        
        # 避免API调用过于频繁
        time.sleep(0.3)
    
    return results

def generate_error_propagation_report(annotated_samples):
    """生成错误传播分析报告（保持不变）"""
    # 按错误类型分组
    error_groups = {
        "逻辑不完整": [],
        "步骤跳跃": [],
        "分析错误": [],
        "解答错误": []
    }
    
    for sample in annotated_samples:
        if sample.get("path_error_type") in error_groups:
            error_groups[sample["path_error_type"]].append(sample)
    
    # 生成统计报告
    report = {
        "summary": {
            "total_samples": len(annotated_samples),
            "error_distribution": {},
            "propagation_summary": {}
        },
        "details": {}
    }
    
    # 总体错误分布
    for error_type in error_groups:
        count = len(error_groups[error_type])
        if count > 0:
            report["summary"]["error_distribution"][error_type] = {
                "count": count,
                "percentage": f"{count/len(annotated_samples)*100:.1f}%"
            }
    
    # 每种错误类型的传播分析
    for error_type, samples in error_groups.items():
        if not samples:
            continue
            
        # 统计传播类型分布
        propagation_counts = {"延续错误": 0, "修正错误": 0, "误判错误": 0}
        for sample in samples:
            if sample.get("propagation_type") in propagation_counts:
                propagation_counts[sample["propagation_type"]] += 1
        
        # 计算比例
        total = len(samples)
        propagation_rates = {
            pt: f"{count/total:.1%}" for pt, count in propagation_counts.items()
        }
        
        report["details"][error_type] = {
            "sample_count": total,
            "propagation_distribution": propagation_counts,
            "propagation_rates": propagation_rates
        }
        
        # 添加到总体摘要
        report["summary"]["propagation_summary"][error_type] = {
            "fix_rate": f"{propagation_counts['修正错误']/total:.1%}",
            "continue_rate": f"{propagation_counts['延续错误']/total:.1%}"
        }
    
    return report

# 主执行流程
if __name__ == "__main__":
    # 1. 配置输入输出路径
    INPUT_JSON_PATH = "/home/wangsijin/model_splits/result_data_qwen_7b_full.json"  # 替换为您的JSON文件路径
    OUTPUT_PREFIX = "error_analysis_results_qwen7b"
    
    # 2. 标注路径生成错误类型
    annotated_samples = analyze_path_errors_with_template(INPUT_JSON_PATH)
    
    # 3. 保存中间结果
    with open("samples_with_path_errors.json", "w", encoding="utf-8") as f:
        json.dump(annotated_samples, f, ensure_ascii=False, indent=2)
    
    # 4. 分析错误传播
    propagation_samples = analyze_error_propagation_with_template(annotated_samples)
    
    # 5. 生成报告
    report = generate_error_propagation_report(propagation_samples)
    
    # 6. 保存最终结果
    with open(f"{OUTPUT_PREFIX}_annotated.json", "w", encoding="utf-8") as f:
        json.dump(propagation_samples, f, ensure_ascii=False, indent=2)
    
    with open(f"{OUTPUT_PREFIX}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n错误传播分析完成！关键发现:")
    print(f"- 总样本数: {report['summary']['total_samples']}")
    print("- 错误类型分布:")
    for error_type, stats in report['summary']['error_distribution'].items():
        print(f"  • {error_type}: {stats['count']}条 ({stats['percentage']})")
    
    print("\n- 错误传播情况:")
    for error_type, stats in report['summary']['propagation_summary'].items():
        print(f"  • {error_type}: 修正率 {stats['fix_rate']}, 延续率 {stats['continue_rate']}")