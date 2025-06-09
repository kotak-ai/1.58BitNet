# 1.58 BitNet

This repository provides an implementation of the 1.58 BitNet LLaMA model. The project aims to reduce memory usage through ternary quantisation while keeping the training workflow close to standard LLaMA models.

## Requirements

- `torch`
- `transformers`
- `datasets` (for dataset loading)
- `safetensors`

Install the dependencies with:

```bash
pip install torch transformers datasets safetensors
```

## Creating a Model

Use `new-model-architecture-creation.py` to generate a blank ternary model. It prompts for the desired parameter count and saves the result in the current directory.

```bash
python new-model-architecture-creation.py
```

## Cross‑Entropy Training

`trainingv2.py` performs standard CE training on tokenised text datasets. The dataset may be a `.txt`, `.json` or `.jsonl` file.

```bash
python trainingv2.py --dataset path/to/data.jsonl --model_path path/to/model --output_dir ce_out --iters 1000
```

## GRPO Training

`grpo_train.py` implements Grouped Response Policy Optimisation (GRPO). The dataset must contain pairs of queries and answers, formatted as JSON or JSONL with records of the form:

```json
{"query": "...", "answer": "..."}
```

A reward model can optionally be supplied via `--reward_model`; otherwise an F1 based reward is used.

```bash
python grpo_train.py --dataset qa.jsonl --model_path path/to/model --output_dir grpo_out --steps 1000
```

## Hardware and Example Commands

Training large models is memory intensive. The scripts are tested on Apple MPS with CPU fallbacks but work on any hardware that PyTorch supports. Expect to require tens of gigabytes of RAM for multi‑billion parameter models.

Example command for a small CE run:

```bash
python trainingv2.py --dataset data/train.jsonl --model_path llama_750m --output_dir ce_model --iters 10000 --batch_size 8
```

Example command for GRPO with a reward model:

```bash
python grpo_train.py --dataset qa.jsonl --model_path llama_750m --reward_model rm.ckpt --output_dir grpo_model
```

## Evaluation

`evaluation.py` compares a CE fine‑tuned model to a GRPO model using QA F1 score.

```bash
python evaluation.py --dataset qa.jsonl --ce_model ce_model --grpo_model grpo_model
```

