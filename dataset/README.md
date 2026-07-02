# Dataset Directory

This directory stores the datasets used in the ScopeExceedSolver experiments.

## Included Files

- `cmath_label_latest.json`  
  Evaluation dataset used for experiments on CMATH.

- `syllabus_check_2395_latest.json`  
  Evaluation dataset used for curriculum-alignment and out-of-syllabus analysis on SyllabusCheckDataset.

- `train.json`  
  Training dataset file used for path-generation model training.

## Dataset Sources

The evaluation datasets are also available from ModelScope:

- CMATH: `https://modelscope.cn/datasets/wangsijin/cmath_label_latest`
- SyllabusCheck: `https://modelscope.cn/datasets/wangsijin/SyllabusCheck`

The training dataset is available from ModelScope here:

- Idea-Generator-Training-Dataset: `https://modelscope.cn/datasets/wangsijin/Idea-Generator-Training-Dataset`

## Training Data Note

The file `train.json` in this directory is the training dataset used in this repository.

The ModelScope dataset link above is the full released training-data source. The JSON example below is included only to illustrate the data format and field structure.

## Training Sample Format

Example record in JSON format:

```json
[
  {
    "instruction": "请给出下列问题的解题思路",
    "input": "用1立方厘米的小正方体摆一个棱长5厘米的正方体，需要$\\underline{\\hspace{5em}}$个．摆成的正方体的底面积是$\\underline{\\hspace{5em}}$平方厘米．",
    "output": "解题思路： 1. **确定小正方体的数量：** - 首先，理解一个立方体的体积计算公式为 $V = a^3$，其中 $a$ 是立方体的棱长。 - 题目要求用1立方厘米的小正方体（即每个棱长为1厘米）拼成一个大正方体。 - 大正方体的棱长给定为5厘米，因此根据体积公式，可以确定需要多少个小正方体来构成这个大正方体。 2. **计算正方体底面积：** - 底面积是一个正方形的面积，计算公式为 $A = a^2$，其中 $a$ 是正方形的边长。 - 题目中，正方体的棱长为5厘米，因此底面是一个边长为5厘米的正方形。 - 利用正方形面积计算公式，可以求出正方体的底面积。 据此思路解答。"
  }
]
```

## Usage Notes

- Use `cmath_label_latest.json` and `syllabus_check_2395_latest.json` for evaluation.
- Use `train.json` for training or fine-tuning path-generation models.
- The sample shown above is for format illustration only; the actual training corpus is stored in `train.json` and released through the ModelScope dataset link.
