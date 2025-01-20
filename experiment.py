import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.resolve()
sys.path.append(str(project_root))

import argparse

from prompt import direct_answer_prompt, direct_answer_with_limitations_prompt, generate_idea_and_solution_prompt, generate_idea_with_limitations_and_solution_prompt
from global_variable import MY_DATASET, OPEN_DATASET

def getargparse():
    parser = argparse.ArgumentParser(description="Control the command of experiments")
    subparsers = parser.add_subparsers(dest='prompt', help='Choose the prompt to execute.')

    # direct_answer
    parser_direct_answer = subparsers.add_parser('direct_answer_prompt', help='Direct answer')
    
    # direct_answer_with_limitations
    parser_direct_answer_with_limitations = subparsers.add_parser('direct_answer_with_limitations_prompt', help='Direct answer with given limitations')
    
    # generate_idea_and_solution
    parser_generate_idea_and_solution = subparsers.add_parser('generate_idea_and_solution_prompt', help='Generate idea and answer')
    parser_generate_idea_and_solution.add_argument('--model', choices=['gpt', 'qwen'], required=True, help='Model to use for generating ideas')
    parser_generate_idea_and_solution.add_argument('--finetuned', action='store_true', help='Use a fine-tuned model')

    # generate_idea_with_limitations_and_solution
    parser_generate_idea_with_limitations = subparsers.add_parser('generate_idea_with_limitations_and_solution_prompt', help='Generate idea with limitations and answer')
    parser_generate_idea_with_limitations.add_argument('--model', choices=['gpt', 'qwen'], required=True, help='Model to use for generating ideas')
    parser_generate_idea_with_limitations.add_argument('--finetuned', action='store_true', help='Use a fine-tuned model')

    return parser.parse_args()

def direct_answer():
    print("Executing direct answer...")

def direct_answer_with_limitations():
    print("Executing direct answer with given limitations...")

def generate_idea_and_solution(model, finetuned):
    print(f"Generating idea and answering using {model}{' (fine-tuned)' if finetuned else ''}...")

def generate_idea_with_limitations_and_solution(model, finetuned):
    print(f"Generating idea with limitations and answering using {model}{' (fine-tuned)' if finetuned else ''}...")

def main():
    args = getargparse()

    if args.prompt == 'direct_answer_prompt':
        func = direct_answer
        params = []
    elif args.prompt == 'direct_answer_with_limitations_prompt':
        func = direct_answer_with_limitations
        params = []
    elif args.prompt == 'generate_idea_and_solution_prompt':
        func = generate_idea_and_solution
        params = [args.model, args.finetuned]
    elif args.prompt == 'generate_idea_with_limitations_and_solution_prompt':
        func = generate_idea_with_limitations_and_solution
        params = [args.model, args.finetuned]
    else:
        print("Unknown prompt provided.")
        return

    func(*params)

if __name__ == "__main__":
    main()