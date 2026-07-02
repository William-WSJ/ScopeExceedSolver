# Model Directory

This directory documents the fine-tuned path-generation models used in the ScopeExceedSolver experiments.

## Released Model Links

### Qwen2.5-Math-7B-Instruct

- Full fine-tuned model: `https://modelscope.cn/models/wangsijin/Qwen2.5-Math-Idea-Generator`
- LoRA fine-tuned model: `https://modelscope.cn/models/wangsijin/Qwen2.5-Math-Idea-Generator-7B-lora`

### Deepseek-Math-7B-Instruct

- Full fine-tuned model: `https://modelscope.cn/models/wangsijin/Deepseek-Math-Idea-Generator-7B-full`
- LoRA fine-tuned model: `https://modelscope.cn/models/wangsijin/Deepseek-Math-Idea-Generator-7B-lora`

### Mimo-7B-RL

- Full fine-tuned model: `https://modelscope.cn/models/wangsijin/Mimo-Idea-Generator-7B-full`
- LoRA fine-tuned model: `https://modelscope.cn/models/wangsijin/Mimo-Idea-Generator-7B-lora`

## How These Models Are Used

These models are used as path-generation agents in the staged problem-solving pipeline. They generate intermediate solution paths (`thought` fields), which are then passed to downstream execution models for final answer generation.

## Deployment Notes

After downloading the model files, deploy them through an OpenAI-compatible serving stack. In this repository, `generate_path.py` is the reference calling script.

Common deployment choices include:

- `LLaMA-Factory`
- `Ollama`
- `vLLM`

## Reference Deployment Workflow

1. Download a fine-tuned model from one of the ModelScope links above.
2. Serve the model through an OpenAI-compatible endpoint.
3. Run `generate_path.py` to append `thought` fields to the evaluation JSON.

Example:

```bash
python generate_path.py \
  --input dataset/syllabus_check_2395_latest.json \
  --output experiment/path/generated_thoughts.json \
  --url http://127.0.0.1:8000
```

## Additional Reference

For LLaMA-Factory deployment details, see the official repository:

- `https://github.com/hiyouga/LlamaFactory`
