# Further Analysis

This folder stores the focused Grade 3–4 subset analysis used in the manuscript.

## Contents

- `syllabus_baseline_deepseek_v3_2_subset.json`: archived baseline subset used for focused analysis.
- `subset/`: archived subset outputs and timing statistics for different path-generation strategies.
- `process/`: path-evaluation outputs with score JSON.
- `process_multi_rounds/`: repeated scoring rounds.
- `propagation/`: archived path-error and error-propagation annotations.
- `calculate_by_knowledge.py`, `calculate_grade_osr.py`: auxiliary analysis scripts.
- `evaluate_score.py`, `propagation.py`: legacy analysis scripts.

## Quick Reproduction

From the project root:

```bash
python experiment/reproduce_metrics.py --section further
python utils/rank.py
```

## Focused Subset Results

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

## Timing Summary

| Strategy | Average Path Execution Time |
| --- | --- |
| GPT-4o mini finetuned | 14.75s |
| DeepSeek-Instruct full | 13.09s |
| DeepSeek-Instruct LoRA | 13.27s |
| Mimo-RL full | 14.70s |
| Mimo-RL LoRA | 12.89s |
| Qwen-Instruct full | 13.63s |
| Qwen-Instruct LoRA | 13.37s |

## Error Propagation Summary

Effective error samples exclude `Unknown/Correct` and use `n=596`.

| Error Type | Sample | Proportion | FR | CER | MER |
| --- | --- | --- | --- | --- | --- |
| Logical Incompleteness | 132 | 22.1% | 50.0% | 40.2% | 9.8% |
| Analysis Error | 354 | 59.4% | 80.8% | 17.5% | 1.7% |
| Solution Error | 110 | 18.5% | 86.4% | 10.9% | 2.7% |

## Notes

- `subset/*_stats.json` stores timing summaries.
- `process/*.json` and `propagation/*.json` are the sources for path-score and error-propagation analysis.
