# Baseline Results

This folder stores the baseline outputs reported in the manuscript.

## File Groups

- `syllabus_baseline_*`: results on `SyllabusCheckDataset`
- `cmath_baseline_*`: results on `CMATH`
- `*_direct.json`: direct-answer baseline
- `*_with_limitations.json`: constrained baseline

## Reproducing Table Values

Most files already contain:

- `acc`: final-answer correctness flag
- `exceeds_scope`: out-of-syllabus flag

You can summarize them with:

```bash
python experiment/reproduce_metrics.py --section tables
```

For the grade-wise mean, standard deviation, and 95% confidence interval reported as `mean ± margin of error`, you can run:

```bash
cd experiment/baseline-result
python std.py
```

The default configuration in `std.py` targets `syllabus_baseline_deepseek_v3_2_direct.json`, which corresponds to the `Deepseek-V3.2` direct baseline row in the manuscript's SyllabusCheckDataset table.
