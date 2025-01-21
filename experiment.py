import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.resolve()
sys.path.append(str(project_root))

import argparse
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from prompt import direct_answer_prompt, direct_answer_with_limitations_prompt, generate_idea_prompt, generate_idea_with_limitations_prompt, answer_check_prompt, exceeds_scope_check_prompt
from global_variable import MY_DATASET, OPEN_DATASET, TEST
from utils import read_test_json, read_checkpoint, write_checkpoint

# Enter your OpenAI API Key here
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

def getargparse():
    parser = argparse.ArgumentParser(description="Control the command of experiments")
    subparsers = parser.add_subparsers(dest='prompt', help='Choose the prompt to execute.')

    # direct_answer
    parser_direct_answer = subparsers.add_parser('direct_answer_prompt', help='Direct answer')
    
    # direct_answer_with_limitations
    parser_direct_answer_with_limitations = subparsers.add_parser('direct_answer_with_limitations_prompt', help='Direct answer with given limitations')
    
    # generate_idea
    parser_generate_idea = subparsers.add_parser('generate_idea_prompt', help='Generate idea')
    parser_generate_idea.add_argument('--model', choices=['gpt', 'qwen'], required=True, help='Model to use for generating ideas')
    parser_generate_idea.add_argument('--finetuned', action='store_true', help='Use a fine-tuned model')

    # generate_idea_with_limitations
    parser_generate_idea_with_limitations = subparsers.add_parser('generate_idea_with_limitations_prompt', help='Generate idea with limitations')
    parser_generate_idea_with_limitations.add_argument('--model', choices=['gpt', 'qwen'], required=True, help='Model to use for generating ideas')
    parser_generate_idea_with_limitations.add_argument('--finetuned', action='store_true', help='Use a fine-tuned model')

    return parser.parse_args()

def solution_direct_answer(item):
    solution_chain = direct_answer_prompt | llm | parser
    answer_check_chain = answer_check_prompt | llm | parser
    exceeds_scope_check_chain = exceeds_scope_check_prompt | llm | parser

    solution = solution_chain.invoke({
        "question": item["question"]
    })
    correct = answer_check_chain.invoke({
        "question": item["question"],
        "answer": item["answer"],
        "solution": solution
    })
    relevant = exceeds_scope_check_chain.invoke({
        "solution": solution,
        "limitations": item["cautions"]
    })
    
    return correct, relevant

def solution_direct_answer_with_limitations(item):
    solution_chain = direct_answer_with_limitations_prompt | llm | parser
    answer_check_chain = answer_check_prompt | llm | parser
    exceeds_scope_check_chain = exceeds_scope_check_prompt | llm | parser

    solution = solution_chain.invoke({
        "question": item["question"],
        "limitations": item["cautions"]
    })
    correct = answer_check_chain.invoke({
        "question": item["question"],
        "answer": item["answer"],
        "solution": solution
    })
    relevant = exceeds_scope_check_chain.invoke({
        "solution": solution,
        "limitations": item["cautions"]
    })
    
    return correct, relevant

def solution_generate_idea(item, model, finetuned):
    solution_chain = generate_idea_prompt | llm | parser
    answer_check_chain = answer_check_prompt | llm | parser
    exceeds_scope_check_chain = exceeds_scope_check_prompt | llm | parser

    solution = solution_chain.invoke({
        "question": item["question"],
        "idea": item[f"{model}_{'finetuned_' if finetuned else ''}idea"]
    })
    correct = answer_check_chain.invoke({
        "question": item["question"],
        "answer": item["answer"],
        "solution": solution
    })
    relevant = exceeds_scope_check_chain.invoke({
        "solution": solution,
        "limitations": item["cautions"]
    })
    
    return correct, relevant

def solution_generate_idea_with_limitations(item, model, finetuned):
    solution_chain = generate_idea_with_limitations_prompt | llm | parser
    answer_check_chain = answer_check_prompt | llm | parser
    exceeds_scope_check_chain = exceeds_scope_check_prompt | llm | parser

    solution = solution_chain.invoke({
        "question": item["question"],
        "idea": item[f"{model}_{'finetuned_' if finetuned else ''}idea"],
        "limitations": item["cautions"]
    })
    correct = answer_check_chain.invoke({
        "question": item["question"],
        "answer": item["answer"],
        "solution": solution
    })
    relevant = exceeds_scope_check_chain.invoke({
        "solution": solution,
        "limitations": item["cautions"]
    })
    
    return correct, relevant

def direct_answer():
    dataset = read_test_json(TEST)
    checkpoint = read_checkpoint("direct_answer")
    start_index = checkpoint["count"]

    for idx, item in enumerate(dataset[start_index:], start=start_index):
        correct, relevant = solution_direct_answer(item)

        checkpoint["count"] = idx + 1
        checkpoint["correct"] += correct
        checkpoint["relevant"] += relevant
        write_checkpoint("direct_answer_with_limitations", checkpoint["count"], checkpoint["correct"], checkpoint["relevant"])
    
    accuracy = checkpoint["correct"] / checkpoint["count"] if checkpoint["count"] > 0 else 0
    relevant_rate = checkpoint["relevant"] / checkpoint["count"] if checkpoint["count"] > 0 else 0
    print(f"Direct Answer Accuracy: {accuracy}")
    print(f"Direct Answer Relevant Rate: {relevant_rate}")

def direct_answer_with_limitations():
    dataset = read_test_json(TEST)
    checkpoint = read_checkpoint("direct_answer_with_limitations")
    start_index = checkpoint["count"]

    for idx, item in enumerate(dataset[start_index:], start=start_index):
        correct, relevant = solution_direct_answer_with_limitations(item)

        checkpoint["count"] = idx + 1
        checkpoint["correct"] += correct
        checkpoint["relevant"] += relevant
        write_checkpoint("direct_answer_with_limitations", checkpoint["count"], checkpoint["correct"], checkpoint["relevant"])
    
    accuracy = checkpoint["correct"] / checkpoint["count"] if checkpoint["count"] > 0 else 0
    relevant_rate = checkpoint["relevant"] / checkpoint["count"] if checkpoint["count"] > 0 else 0
    print(f"Direct Answer with Limitations Accuracy: {accuracy}")
    print(f"Direct Answer with Limitations Relevant Rate: {relevant_rate}")

def generate_idea(model, finetuned):
    dataset = read_test_json(TEST)
    checkpoint = read_checkpoint("generate_idea")
    start_index = checkpoint["count"]

    for idx, item in enumerate(dataset[start_index:], start=start_index):
        correct, relevant = solution_generate_idea(item, model, finetuned)

        checkpoint["count"] = idx + 1
        checkpoint["correct"] += correct
        checkpoint["relevant"] += relevant
        write_checkpoint("generate_idea", checkpoint["count"], checkpoint["correct"], checkpoint["relevant"])
    
    accuracy = checkpoint["correct"] / checkpoint["count"] if checkpoint["count"] > 0 else 0
    relevant_rate = checkpoint["relevant"] / checkpoint["count"] if checkpoint["count"] > 0 else 0
    print(f"Generate Idea Accuracy: {accuracy}")
    print(f"Generate Idea Relevant Rate: {relevant_rate}")

def generate_idea_with_limitations(model, finetuned):
    dataset = read_test_json(TEST)
    checkpoint = read_checkpoint("generate_idea_with_limitations")
    start_index = checkpoint["count"]

    for idx, item in enumerate(dataset[start_index:], start=start_index):
        correct, relevant = solution_generate_idea_with_limitations(item, model, finetuned)

        checkpoint["count"] = idx + 1
        checkpoint["correct"] += correct
        checkpoint["relevant"] += relevant
        write_checkpoint("generate_idea_with_limitations", checkpoint["count"], checkpoint["correct"], checkpoint["relevant"])
    
    accuracy = checkpoint["correct"] / checkpoint["count"] if checkpoint["count"] > 0 else 0
    relevant_rate = checkpoint["relevant"] / checkpoint["count"] if checkpoint["count"] > 0 else 0
    print(f"Generate Idea with Limitations Accuracy: {accuracy}")
    print(f"Generate Idea with Limitations Relevant Rate: {relevant_rate}")

def main():
    args = getargparse()

    if args.prompt == 'direct_answer_prompt':
        func = direct_answer
        params = []
    elif args.prompt == 'direct_answer_with_limitations_prompt':
        func = direct_answer_with_limitations
        params = []
    elif args.prompt == 'generate_idea_prompt':
        func = generate_idea
        params = [args.model, args.finetuned]
    elif args.prompt == 'generate_idea_with_limitations_prompt':
        func = generate_idea_with_limitations
        params = [args.model, args.finetuned]
    else:
        print("Unknown prompt provided.")
        return

    func(*params)

if __name__ == "__main__":
    main()