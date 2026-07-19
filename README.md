## Title

ScopeExceedSolver is the code and data repository for our study on staged problem solving for improving curriculum alignment of large language models in primary mathematics education.

Repository URL: `https://github.com/William-WSJ/ScopeExceedSolver`

## Description

This repository contains the datasets, archived experiment outputs, prompt templates, and analysis scripts used in our experiments. It supports three core tasks:

1. Evaluate answer accuracy on primary-school math datasets.
2. Evaluate whether generated solutions exceed grade-level syllabus constraints.
3. Test whether staged problem-solving paths improve both accuracy and curriculum alignment.

The repository is intended both as a reproducibility package and as a reference implementation of the evaluation pipeline described in the manuscript.

## Dataset Information

The `dataset/` directory stores the datasets used in the experiments.

- `dataset/syllabus_check_2395_latest.json`: evaluation set for curriculum-alignment and out-of-syllabus analysis on SyllabusCheckDataset.
- `dataset/cmath_label_latest.json`: evaluation set for CMATH experiments.
- `dataset/train.json`: training data used for path-generation model training or fine-tuning.
- `dataset/README.md`: dataset notes, provenance, and an example training-record format.

Common fields used across the scripts include `question`, `answer`, `grade`, `knowledge`, `acc`, `exceeds_scope`, `cautions`, `grade_cautions`, and `thought`.

External dataset release pages:

- CMATH: `https://modelscope.cn/datasets/wangsijin/cmath_label_latest`
- SyllabusCheck: `https://modelscope.cn/datasets/wangsijin/SyllabusCheck`
- Idea-Generator-Training-Dataset: `https://modelscope.cn/datasets/wangsijin/Idea-Generator-Training-Dataset`

## Code Information

The main code and analysis components are:

- `experiment.py`: main experiment driver for direct-answer, constrained-answer, path-conditioned, and path-conditioned-with-constraints settings.
- `generate_path.py`: appends path-generation outputs (`thought` fields) to input JSON files by calling an OpenAI-compatible endpoint.
- `prompt.py`: prompt templates for generation, answer checking, and out-of-syllabus checking.
- `utils.py`: shared helpers for loading JSON, checkpointing, batch formatting, and metric-related utilities.
- `experiment/reproduce_metrics.py`: reproduces the tables and summary statistics from archived JSON outputs already included in the repository.
- `utils/rank.py`: summarizes ranking-style analyses from the further-analysis artifacts.
- `chat_dsv32_final.py`, `chat_gemini3flash_final.py`, `chat_gpt52_final.py`, `batch_openai.py`, and `aihubmix.py`: model-specific or batch-inference helper scripts retained for the original experiments.

The repository also includes archived outputs in `experiment/`, released model references in `model/`, and figure assets in `figures/`.

## Repository Structure

```text
ScopeExceedSolver/
├── README.md
├── requirements.txt
├── experiment.py
├── generate_path.py
├── prompt.py
├── utils.py
├── batch_openai.py
├── chat_dsv32_final.py
├── chat_gemini3flash_final.py
├── chat_gpt52_final.py
├── dataset/
├── experiment/
├── figures/
├── model/
└── utils/
```

## Reproducibility

This repository provides both the algorithms and the code needed to reproduce the reported summary results.

- The archived JSON outputs required for the manuscript tables are already stored under `experiment/`.
- The summary scripts read those archived outputs directly and do not require new API calls.
- The path-generation workflow is provided in `generate_path.py` for users who want to reproduce the staged pipeline on their own deployments.
- The evaluation workflow is provided in `experiment.py` for users who want to rerun model-based experiments.

For a lightweight reproducibility check that does not call external APIs, run:

```bash
python experiment/reproduce_metrics.py --section tables
python experiment/reproduce_metrics.py --section further
python utils/rank.py
```

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Current dependencies:

- `openai`
- `langchain-core`
- `langchain-openai`
- `pandas`
- `requests`

Python 3.9+ is recommended.

## Usage Instructions

### 1. Reproduce the archived manuscript tables

From the project root:

```bash
python experiment/reproduce_metrics.py --section tables
python experiment/reproduce_metrics.py --section further
python utils/rank.py
```

These commands only read repository files and summarize the saved results.

### 2. Generate staged solution paths

Deploy a path-generation model behind an OpenAI-compatible endpoint, then run:

```bash
python generate_path.py \
  --input dataset/syllabus_check_2395_latest.json \
  --output experiment/path/generated_thoughts.json \
  --url http://127.0.0.1:8000
```

This script reads each item, queries the serving endpoint, and appends a `thought` field to the output JSON.

### 3. Run evaluation experiments

`experiment.py` contains the main evaluation logic for:

- direct answer generation,
- direct answer generation with syllabus limitations,
- answer generation conditioned on a generated path, and
- answer generation conditioned on both a path and syllabus limitations.

Before using `experiment.py`, configure the target model/API credentials in the script or adapt it to your runtime environment.

### 4. Inspect archived results

The archived outputs are grouped as follows:

- `experiment/baseline-result/`: direct-answer baselines and constrained baselines.
- `experiment/idea-result/`: path-only experiments.
- `experiment/idea-limitation-result/`: path-plus-constraint experiments.
- `experiment/further_analysis/`: focused Grade 3-4 subset analysis.
- `experiment/path/`: generated path files.

## Methodology

The repository implements the following workflow:

1. Start from evaluation items in `dataset/`.
2. Optionally generate an intermediate solution path (`thought`) with a path-generation model using `generate_path.py`.
3. Use an execution model to produce a final solution, either directly or conditioned on the generated path.
4. Optionally provide grade-level restriction text (`cautions` or `grade_cautions`) to discourage out-of-syllabus reasoning.
5. Check answer correctness with the answer-check prompt in `prompt.py`.
6. Check out-of-syllabus behavior with the scope-check prompt in `prompt.py`.
7. Aggregate saved JSON outputs into manuscript-ready summaries with `experiment/reproduce_metrics.py` and `utils/rank.py`.

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

### Focused Further Analysis on the Grade 3-4 Subset

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

## Citations

If you use this repository, datasets, or archived outputs in academic work, please cite the corresponding manuscript and repository release.

Repository citation target:

- `https://github.com/William-WSJ/ScopeExceedSolver`

Dataset release pages are listed in the Dataset Information section above.

## License and Contribution Guidelines

No standalone license file is currently included in this repository. Please contact the repository owner for reuse questions that are not already covered by the manuscript or dataset/model release pages.

Contributions should preserve the current data format and archived-result layout. For substantial changes, include a short note describing:

- which dataset or result files are affected,
- whether any metrics are recomputed,
- whether external API behavior changes, and
- whether README or dataset/model notes need updating.
