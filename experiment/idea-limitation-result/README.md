# Idea + Limitation Results

This folder stores experiments where the execution model receives both:

1. a generated solution path (`thought`), and
2. an explicit curriculum limitation prompt.

The prompt templates are defined in `prompt.py` and in the API scripts that preserve the original Chinese task setting.

## Naming Convention

File names follow this pattern:

```text
{dataset}_{path-generator}_{path-executor}.json
```

Examples:

- `syllabus_qwen-7b_full_gpt5_2.json`
- `cmath_deepseek-7b_full_gemini3.json`

## Reproduction

```bash
python experiment/reproduce_metrics.py --section tables
```

These files support the manuscript tables where both staged reasoning and explicit curriculum constraints are used.
