# ScopeExceedSolver

ScopeExceedSolver is the code and data repository for our study on Staged problem solving improves curriculum alignment of large language models for primary mathematics education.

## Overview

This repository supports three tasks:

1. Evaluate answer accuracy on primary-school math datasets.
2. Evaluate whether generated solutions exceed grade-level syllabus constraints.
3. Test whether staged problem-solving paths improve both accuracy and curriculum alignment.

## Repository Structure

```text
ScopeExceedSolver/
├── README.md
├── experiment.py
├── generate_path.py
├── chat_dsv32_final.py
├── chat_gemini3flash_final.py
├── chat_gpt52_final.py
├── batch_openai.py
├── prompt.py
├── utils.py
├── global_variable.py
├── dataset/
├── model/
├── experiment/
└── utils/
```

## Datasets

- `dataset/syllabus_check_2395_latest.json`: SyllabusCheckDataset evaluation set.
- `dataset/cmath_label_latest.json`: CMATH evaluation set.
- `dataset/README.md`: dataset notes and sample format.

Common fields used by the scripts include `question`, `answer`, `grade`, `knowledge`, `acc`, `exceeds_scope`, `cautions`, and `grade_cautions`.

## Reproducibility

From the project root:

```bash
python experiment/reproduce_metrics.py --section tables
python experiment/reproduce_metrics.py --section further
python utils/rank.py
```

These commands read archived JSON outputs already stored in the repository and do not call external APIs.

## Main Result Tables

### Baselines on SyllabusCheckDataset

| Path Execution Agent | Path Generation Agent | Constraints | Accuracy | OSR |
| --- | --- | --- | --- | --- |
| GPT-5.2 | - | × | 88.98±1.24% | 23.26±1.48% |
| GPT-5.2 | - | ✓ | 87.18±1.31% | 7.43±0.94% |
| DeepSeek-V3.2 | - | × | 93.03±0.95% | 35.20±1.75% |
| DeepSeek-V3.2 | - | ✓ | 92.69±1.02% | 21.42±1.44% |
| Gemini-3-flash | - | × | 95.37±0.83% | 20.00±1.41% |
| Gemini-3-flash | - | ✓ | 94.91±0.87% | 5.68±0.83% |

### Baselines on CMATH

| Path Execution Agent | Path Generation Agent | Constraints | Accuracy | OSR |
| --- | --- | --- | --- | --- |
| GPT-5.2 | - | × | 95.17±1.66% | 8.67±1.72% |
| GPT-5.2 | - | ✓ | 95.00±1.67% | 3.83±1.12% |
| DeepSeek-V3.2 | - | × | 96.50±1.43% | 14.17±2.05% |
| DeepSeek-V3.2 | - | ✓ | 96.17±1.47% | 6.83±1.55% |
| Gemini-3-flash | - | × | 96.83±1.37% | 7.33±1.55% |
| Gemini-3-flash | - | ✓ | 96.50±1.44% | 4.50±1.18% |

### Path-Only Experiments on SyllabusCheckDataset

| Path Generation Agent | Constraints | Accuracy | OSR |
| --- | --- | --- | --- |
| Qwen2.5-Math-Instruct | × | 92.82±1.01% | 27.77±1.56% |
| Qwen2.5-Math-Instruct finetuned | × | 93.36±0.97% | 23.17±1.48% |
| DeepSeek-Math-Instruct | × | 92.73±1.01% | 28.89±1.57% |
| DeepSeek-Math-Instruct finetuned | × | 94.28±0.91% | 25.76±1.52% |
| Mimo-RL | × | 92.19±1.05% | 27.97±1.56% |
| Mimo-RL finetuned | × | 94.03±0.93% | 24.38±1.50% |
| GPT-4o mini | × | 92.36±1.04% | 26.81±1.55% |
| GPT-4o mini finetuned | × | 93.70±0.95% | 25.09±1.51% |

### Focused Further Analysis on the Grade 3–4 Subset

| Strategy | Accuracy | OSR |
| --- | --- | --- |
| Baseline | 95.59±2.43% | 82.35±4.30% |
| Qwen-Instruct LoRA | 94.12±2.79% | 62.13±5.72% |
| Qwen-Instruct full | 95.96±2.30% | 65.07±5.65% |
| DeepSeek-Instruct LoRA | 93.75±2.87% | 66.54±5.50% |
| DeepSeek-Instruct full | 96.32±2.22% | 63.60±5.65% |
| Mimo-RL LoRA | 92.65±3.09% | 58.82±5.75% |
| Mimo-RL full | 93.38±2.94% | 65.44±5.51% |
| GPT-4o mini finetuned | 96.32±2.24% | 64.34±5.57% |

## Experiment Directory Mapping

- `experiment/baseline-result/`: direct-answer baselines and constrained baselines.
- `experiment/idea-result/`: path-only experiments.
- `experiment/idea-limitation-result/`: path-plus-constraint experiments.
- `experiment/further_analysis/`: focused Grade 3–4 subset analysis.
- `experiment/reproduce_metrics.py`: summary script for archived result files.
