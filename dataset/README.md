This folder contains all evaluation dataset for experiments. The dataset is downloaded from [here]
CMATH: (https://modelscope.cn/datasets/wangsijin/cmath_label_latest).
SyllabusCheck: (https://modelscope.cn/datasets/wangsijin/SyllabusCheck).

The full training dataset will be provided if needed.

Here is a training set example in JSON format:

```json
[
  {
    "instruction": "请给出下列问题的解题思路",
    "input": "用1立方厘米的小正方体摆一个棱长5厘米的正方体，需要$\\underline{\\hspace{5em}}$个．摆成的正方体的底面积是$\\underline{\\hspace{5em}}$平方厘米．",
    "output": "解题思路： 1. **确定小正方体的数量：** - 首先，理解一个立方体的体积计算公式为 $V = a^3$，其中 $a$ 是立方体的棱长。 - 题目要求用1立方厘米的小正方体（即每个棱长为1厘米）拼成一个大正方体。 - 大正方体的棱长给定为5厘米，因此根据体积公式，可以确定需要多少个小正方体来构成这个大正方体。 2. **计算正方体底面积：** - 底面积是一个正方形的面积，计算公式为 $A = a^2$，其中 $a$ 是正方形的边长。 - 题目中，正方体的棱长为5厘米，因此底面是一个边长为5厘米的正方形。 - 利用正方形面积计算公式，可以求出正方体的底面积。 据此思路解答。"
  }
]
