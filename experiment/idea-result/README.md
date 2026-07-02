# Idea-Only Results

This folder stores experiments where the execution model receives a generated solution path (`thought`) but no explicit curriculum limitation prompt.

## Naming Convention

- `syllabus_*.json`: results on `SyllabusCheckDataset`
- `cmath_*.json`: results on `CMATH`
- `*_base.json`: unfine-tuned path generator
- `*_full.json` or `*_finetuned.json`: fine-tuned path generator

## Stored Fields

These result files already include:

- `thought`
- `solution`
- `acc`
- `exceeds_scope`
- `exceeds_reason`

## Reproduction

```bash
python experiment/reproduce_metrics.py --section tables
```

The outputs correspond to the manuscript tables for path-generation-agent experiments without additional answer constraints.
