```markdown
# Scope Exceed Solver

This repository contains scripts for testing, evaluating, and generating solutions for math problems using various AI models.

## Features

- Test results generation using DeepSeek-v3.2
- Custom dataset evaluation
- Solution path generation
- Experimental results analysis
- Error propagation analysis

## Quick Start

### 1. Testing Results

To run tests on DeepSeek-v3.2:

```bash
python chat_dsv32_final.py --input /path/to/test/set --output /path/to/output/file
```

### 2. Evaluating Custom Datasets

For custom dataset evaluation, modify the prompt building function in the scripts. The prompt content is located in `prompt.py`.

### 3. Generating Solution Paths

To generate solution paths:
1. Start the model service using Llama Factory
2. Run the generation script:

```bash
python generate_path.py --input /path/to/dataset --output /path/to/output/file
```

Note: GPT-4o-mini is a closed-source model. Contact the author if needed.

### 4. Experimental Results

All experimental results are stored in the `experiment/` folder. Run `std.py` in each subfolder to reproduce the data (remember to update file paths).

### 5. Further Analysis

The `experiment/further_analysis/` folder contains:
- Selected datasets
- Statistical data
- Experimental result files
- Related code in the `utils/` folder

## Configuration

Make sure to configure the following in your environment:
- Local Ollama API endpoint
- Model settings
- File paths for input/output

## Requirements

- Python 3.x
- Required packages (see requirements.txt)
- Ollama for local model serving
- Llama Factory for model deployment

## License

[Add your license information here]
```