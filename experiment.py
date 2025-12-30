import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.resolve()
sys.path.append(str(project_root))

import argparse
import os
import json
from typing import List, Dict, Any, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from prompt import (
    direct_answer_prompt,
    direct_answer_with_limitations_prompt,
    generate_idea_prompt,
    generate_idea_with_limitations_prompt,
    answer_check_prompt,
    exceeds_scope_check_prompt
)
from utils import read_test_json

# Set your OpenAI API Key here
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

def process_direct_answer(item: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """Process a single item using direct answer prompt without limitations"""
    solution_chain = direct_answer_prompt | llm | parser
    answer_check_chain = answer_check_prompt | llm | parser
    exceeds_scope_check_chain = exceeds_scope_check_prompt | llm | parser

    # Generate solution
    solution = solution_chain.invoke({"question": item["question"]})
    
    # Check correctness
    correct_response = answer_check_chain.invoke({
        "question": item["question"],
        "answer": item["answer"],
        "solution": solution
    }).strip()
    correct = (correct_response == "True")
    
    # Check relevance to limitations
    relevant_response = exceeds_scope_check_chain.invoke({
        "solution": solution,
        "limitations": item["cautions"]
    }).strip()
    relevant = (relevant_response == "True")
    
    return solution, correct, relevant

def process_direct_answer_with_limitations(item: Dict[str, Any]) -> Tuple[str, bool, bool]:
    """Process a single item using direct answer prompt with limitations"""
    solution_chain = direct_answer_with_limitations_prompt | llm | parser
    answer_check_chain = answer_check_prompt | llm | parser
    exceeds_scope_check_chain = exceeds_scope_check_prompt | llm | parser

    # Generate solution with limitations
    solution = solution_chain.invoke({
        "question": item["question"],
        "limitations": item["grade_cautions"]
    })
    
    # Check correctness
    correct_response = answer_check_chain.invoke({
        "question": item["question"],
        "answer": item["answer"],
        "solution": solution
    }).strip()
    correct = (correct_response == "True")
    
    # Check relevance to limitations
    relevant_response = exceeds_scope_check_chain.invoke({
        "solution": solution,
        "limitations": item["cautions"]
    }).strip()
    relevant = (relevant_response == "True")
    
    return solution, correct, relevant

def process_generate_idea(item: Dict[str, Any], idea_key: str) -> Tuple[str, bool, bool]:
    """Process a single item using generated idea without limitations"""
    solution_chain = generate_idea_prompt | llm | parser
    answer_check_chain = answer_check_prompt | llm | parser
    exceeds_scope_check_chain = exceeds_scope_check_prompt | llm | parser

    # Get idea from specified key
    if idea_key not in item:
        available_keys = ", ".join(item.keys())
        raise ValueError(f"Idea key '{idea_key}' not found in item. Available keys: {available_keys}")
    
    idea = item[idea_key]
    
    # Generate solution using idea
    solution = solution_chain.invoke({
        "question": item["question"],
        "idea": idea
    })
    
    # Check correctness
    correct_response = answer_check_chain.invoke({
        "question": item["question"],
        "answer": item["answer"],
        "solution": solution
    }).strip()
    correct = (correct_response == "True")
    
    # Check relevance to limitations
    relevant_response = exceeds_scope_check_chain.invoke({
        "solution": solution,
        "limitations": item["cautions"]
    }).strip()
    relevant = (relevant_response == "True")
    
    return solution, correct, relevant, idea  # Return the idea content

def process_generate_idea_with_limitations(item: Dict[str, Any], idea_key: str) -> Tuple[str, bool, bool]:
    """Process a single item using generated idea with limitations"""
    solution_chain = generate_idea_with_limitations_prompt | llm | parser
    answer_check_chain = answer_check_prompt | llm | parser
    exceeds_scope_check_chain = exceeds_scope_check_prompt | llm | parser

    # Get idea from specified key
    if idea_key not in item:
        available_keys = ", ".join(item.keys())
        raise ValueError(f"Idea key '{idea_key}' not found in item. Available keys: {available_keys}")
    
    idea = item[idea_key]
    
    # Generate solution using idea and limitations
    solution = solution_chain.invoke({
        "question": item["question"],
        "idea": idea,
        "limitations": item["cautions"]
    })
    
    # Check correctness
    correct_response = answer_check_chain.invoke({
        "question": item["question"],
        "answer": item["answer"],
        "solution": solution
    }).strip()
    correct = (correct_response == "True")
    
    # Check relevance to limitations
    relevant_response = exceeds_scope_check_chain.invoke({
        "solution": solution,
        "limitations": item["cautions"]
    }).strip()
    relevant = (relevant_response == "True")
    
    return solution, correct, relevant, idea  # Return the idea content

def run_direct_answer(dataset_path: str, output_path: str):
    """Run direct answer experiment on entire dataset"""
    dataset = read_test_json(dataset_path)
    results = []
    correct_count = 0
    relevant_count = 0
    
    print(f"Processing {len(dataset)} items with direct answer prompt...")
    
    for idx, item in enumerate(dataset):
        solution, correct, relevant = process_direct_answer(item)
        
        results.append({
            "id": item.get("id"),
            "question": item["question"],
            "solution": solution,
            "correct": correct,
            "relevant": relevant
        })
        
        if correct:
            correct_count += 1
        if relevant:
            relevant_count += 1
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == len(dataset) - 1:
            print(f"Processed {idx+1}/{len(dataset)} items")
    
    # Calculate metrics
    total = len(dataset)
    accuracy = correct_count / total if total > 0 else 0
    relevant_rate = relevant_count / total if total > 0 else 0
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy,
                "relevant_rate": relevant_rate,
                "total_items": total,
                "correct_items": correct_count,
                "relevant_items": relevant_count
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDirect Answer Results:")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total})")
    print(f"Relevant Rate: {relevant_rate:.4f} ({relevant_count}/{total})")
    print(f"Results saved to {output_path}")

def run_direct_answer_with_limitations(dataset_path: str, output_path: str):
    """Run direct answer with limitations experiment on entire dataset"""
    dataset = read_test_json(dataset_path)
    results = []
    correct_count = 0
    relevant_count = 0
    
    print(f"Processing {len(dataset)} items with direct answer with limitations prompt...")
    
    for idx, item in enumerate(dataset):
        solution, correct, relevant = process_direct_answer_with_limitations(item)
        
        results.append({
            "id": item.get("id"),
            "question": item["question"],
            "solution": solution,
            "correct": correct,
            "relevant": relevant
        })
        
        if correct:
            correct_count += 1
        if relevant:
            relevant_count += 1
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == len(dataset) - 1:
            print(f"Processed {idx+1}/{len(dataset)} items")
    
    # Calculate metrics
    total = len(dataset)
    accuracy = correct_count / total if total > 0 else 0
    relevant_rate = relevant_count / total if total > 0 else 0
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy,
                "relevant_rate": relevant_rate,
                "total_items": total,
                "correct_items": correct_count,
                "relevant_items": relevant_count
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDirect Answer with Limitations Results:")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total})")
    print(f"Relevant Rate: {relevant_rate:.4f} ({relevant_count}/{total})")
    print(f"Results saved to {output_path}")

def run_generate_idea(dataset_path: str, output_path: str, idea_key: str):
    """Run generate idea experiment on entire dataset"""
    dataset = read_test_json(dataset_path)
    results = []
    correct_count = 0
    relevant_count = 0
    
    print(f"Processing {len(dataset)} items with generate idea prompt (idea key: '{idea_key}')...")
    
    for idx, item in enumerate(dataset):
        solution, correct, relevant, idea_content = process_generate_idea(item, idea_key)
        
        results.append({
            "id": item.get("id"),
            "question": item["question"],
            "solution": solution,
            "correct": correct,
            "relevant": relevant,
            "idea_key": idea_key,
            "idea_content": idea_content  # Save the actual idea content
        })
        
        if correct:
            correct_count += 1
        if relevant:
            relevant_count += 1
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == len(dataset) - 1:
            print(f"Processed {idx+1}/{len(dataset)} items")
    
    # Calculate metrics
    total = len(dataset)
    accuracy = correct_count / total if total > 0 else 0
    relevant_rate = relevant_count / total if total > 0 else 0
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy,
                "relevant_rate": relevant_rate,
                "total_items": total,
                "correct_items": correct_count,
                "relevant_items": relevant_count,
                "idea_key": idea_key
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nGenerate Idea Results:")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total})")
    print(f"Relevant Rate: {relevant_rate:.4f} ({relevant_count}/{total})")
    print(f"Results saved to {output_path}")

def run_generate_idea_with_limitations(dataset_path: str, output_path: str, idea_key: str):
    """Run generate idea with limitations experiment on entire dataset"""
    dataset = read_test_json(dataset_path)
    results = []
    correct_count = 0
    relevant_count = 0
    
    print(f"Processing {len(dataset)} items with generate idea with limitations prompt (idea key: '{idea_key}')...")
    
    for idx, item in enumerate(dataset):
        solution, correct, relevant, idea_content = process_generate_idea_with_limitations(item, idea_key)
        
        results.append({
            "id": item.get("id"),
            "question": item["question"],
            "solution": solution,
            "correct": correct,
            "relevant": relevant,
            "idea_key": idea_key,
            "idea_content": idea_content  # Save the actual idea content
        })
        
        if correct:
            correct_count += 1
        if relevant:
            relevant_count += 1
        
        # Progress indicator
        if (idx + 1) % 10 == 0 or idx == len(dataset) - 1:
            print(f"Processed {idx+1}/{len(dataset)} items")
    
    # Calculate metrics
    total = len(dataset)
    accuracy = correct_count / total if total > 0 else 0
    relevant_rate = relevant_count / total if total > 0 else 0
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "accuracy": accuracy,
                "relevant_rate": relevant_rate,
                "total_items": total,
                "correct_items": correct_count,
                "relevant_items": relevant_count,
                "idea_key": idea_key
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nGenerate Idea with Limitations Results:")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total})")
    print(f"Relevant Rate: {relevant_rate:.4f} ({relevant_count}/{total})")
    print(f"Results saved to {output_path}")

def get_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser"""
    parser = argparse.ArgumentParser(description="Run experiments with different prompting strategies")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input JSON dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save results JSON")
    
    subparsers = parser.add_subparsers(dest='prompt_type', required=True, help='Prompting strategy to use')
    
    # Direct answer (no limitations)
    subparsers.add_parser('direct_answer', help='Use direct answer prompt without limitations')
    
    # Direct answer with limitations
    subparsers.add_parser('direct_answer_with_limitations', help='Use direct answer prompt with grade limitations')
    
    # Generate idea (no limitations)
    parser_idea = subparsers.add_parser('generate_idea', help='Use generated idea without limitations')
    parser_idea.add_argument('--idea-key', type=str, default='thought', 
                            help='Key name for idea field in JSON (default: thought)')
    
    # Generate idea with limitations
    parser_idea_lim = subparsers.add_parser('generate_idea_with_limitations', help='Use generated idea with limitations')
    parser_idea_lim.add_argument('--idea-key', type=str, default='thought', 
                                 help='Key name for idea field in JSON (default: thought)')
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Dispatch to the appropriate function
    if args.prompt_type == 'direct_answer':
        run_direct_answer(args.dataset, args.output)
    
    elif args.prompt_type == 'direct_answer_with_limitations':
        run_direct_answer_with_limitations(args.dataset, args.output)
    
    elif args.prompt_type == 'generate_idea':
        run_generate_idea(args.dataset, args.output, args.idea_key)
    
    elif args.prompt_type == 'generate_idea_with_limitations':
        run_generate_idea_with_limitations(args.dataset, args.output, args.idea_key)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()