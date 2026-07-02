# Experiment Directory

This directory stores the archived outputs and summary scripts for all reported experiments.

## Contents

- `baseline-result/`: direct-answer baselines and constrained baselines.
- `idea-result/`: path-only experiments.
- `idea-limitation-result/`: path-plus-constraint experiments.
- `further_analysis/`: focused Grade 3–4 subset analysis, including timing, path scoring, and error propagation.
- `path/`: generated path files.
- `reproduce_metrics.py`: unified metric summary script for archived result files.

## Quick Reproduction

From the project root:

```bash
python experiment/reproduce_metrics.py --section tables
python experiment/reproduce_metrics.py --section further
python utils/rank.py
```

## Result Summary

### Baseline SyllabusCheck

| Setting | Accuracy | OSR |
| --- | --- | --- |
| GPT-5.2 direct | 88.98±1.24% | 23.26±1.48% |
| GPT-5.2 constrained | 87.18±1.31% | 7.43±0.94% |
| DeepSeek-V3.2 direct | 93.03±0.95% | 35.20±1.75% |
| DeepSeek-V3.2 constrained | 92.69±1.02% | 21.42±1.44% |
| Gemini-3-flash direct | 95.37±0.83% | 20.00±1.41% |
| Gemini-3-flash constrained | 94.91±0.87% | 5.68±0.83% |

### Baseline CMATH

| Setting | Accuracy | OSR |
| --- | --- | --- |
| GPT-5.2 direct | 95.17±1.66% | 8.67±1.72% |
| GPT-5.2 constrained | 95.00±1.67% | 3.83±1.12% |
| DeepSeek-V3.2 direct | 96.50±1.43% | 14.17±2.05% |
| DeepSeek-V3.2 constrained | 96.17±1.47% | 6.83±1.55% |
| Gemini-3-flash direct | 96.83±1.37% | 7.33±1.55% |
| Gemini-3-flash constrained | 96.50±1.44% | 4.50±1.18% |

### Further Analysis Subset

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

## Mapping to Archived Files

- `tab:sps-scd-1-4` -> `baseline-result/syllabus_baseline_*.json`
- `tab:sps-cmath-1-4` -> `baseline-result/cmath_baseline_*.json`
- `tab:sps-scd-5-12` -> `idea-result/syllabus_*.json`
- `tab:sps-cmath-5-12` -> `idea-result/cmath_*.json`
- `tab:sps-syllabus-*` -> `idea-limitation-result/syllabus_*.json`
- `tab:sps-cmath-*` -> `idea-limitation-result/cmath_*.json`
- `tab:result-acc-osr` -> `further_analysis/syllabus_baseline_deepseek_v3_2_subset.json` and `further_analysis/subset/*_subset.json`
- `tab:result-time` -> `further_analysis/subset/*_stats.json`
- `tab:error-propagation` -> `further_analysis/propagation/*.json`

## Notes

- `baseline-result/std.py` and other directory-level `std.py` scripts provide manuscript-style summaries for specific result groups.
- `further_analysis/syllabus_baseline_deepseek_v3_2_subset.json` already contains preserved baseline judgments and should not be recomputed.
