# 1.58 BitNet

This repository provides an implementation of the 1.58 BitNet LLaMA model. The project aims to reduce memory usage through ternary quantisation while keeping the training workflow close to standard LLaMA models.

## Requirements and Setup

The code depends only on a few common packages:

- `torch`
- `transformers`
- `datasets` (for dataset loading)
- `safetensors`

Install them with:

```bash
pip install torch transformers datasets safetensors
```

The training scripts are tested on Apple MPS hardware but will run on any
PyTorch device (CPU or CUDA).  Multi‑billion parameter models may require tens of
gigabytes of RAM.

## Creating a Model

Use `new-model-architecture-creation.py` to generate a blank ternary model.  The
script asks for the desired parameter count and writes the model to a directory
named `llama_<params>_ternary_quantized_optimized`.  Passing `--e` enables an
experimental quantisation mode.

```bash
python new-model-architecture-creation.py
```

## Cross‑Entropy Training

`trainingv2.py` performs standard CE training on tokenised text datasets.  The
file may be plain text or a JSON/JSONL file containing a `text` field.  Each
line or record is treated as one training example.

```bash
python trainingv2.py --dataset path/to/data.jsonl --model_path path/to/model --output_dir ce_out --iters 1000
```

## GRPO Training

`grpo_train.py` implements Grouped Response Policy Optimisation (GRPO).  The
dataset should be JSON or JSONL with one object per record:

```json
{"query": "...", "answer": "..."}
```

During training candidate answers are generated for each query and scored.  If
`--reward_model PATH` is provided the score comes from a saved
`RewardModel`; otherwise the robust F1 reward from `qa_reward` is used.

Optional features:

- `--config FILE` &ndash; JSON file with argument defaults (e.g. `guiding_prompt`).
- `--two_layer` &ndash; enable the two stage trainer with self-correction.
- `--csv_log LOG.csv` &ndash; append metrics to a CSV file. When `--two_layer` is
  active the log also includes up to three `corrected_n` columns with corrected
  answers for inspection.
- `--resume CKPT` &ndash; resume training from a checkpoint created with
  `save_checkpoint`.

Example `config.json` providing defaults:

```json
{
  "lr": 0.0005,
  "group_size": 4,
  "guiding_prompt": "Review and correct the answer:"
}
```
`guiding_prompt` can be either the prompt text itself or a path to a file
containing the text.

```bash
python grpo_train.py --dataset qa.jsonl --model_path path/to/model \
    --output_dir grpo_out --steps 1000 --csv_log metrics.csv
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

## Two-Layer Self-Correction

Passing `--two_layer` enables a second GRPO pass that attempts to refine the first answer. The second pass concatenates the query, the original answer and the text from `--guiding_prompt` (separated by the tokenizer's `sep_token_id` or `eos_token_id`) before generating the correction.

```bash
python grpo_train.py --dataset qa.jsonl --model_path llama_750m --reward_model rm.ckpt --output_dir grpo_model --two_layer --guiding_prompt "Review and correct the answer:"
```

## Evaluation

`evaluation.py` compares a CE fine‑tuned model to a GRPO model using the same QA
F1 reward as training.  The dataset format matches the GRPO training set
(`{ "query": ..., "answer": ... }`).

```bash
python evaluation.py --dataset qa.jsonl --ce_model ce_model --grpo_model grpo_model
```


## Reward Model Examples

Two reference implementations are provided for scoring generated answers:

- `simple_reward_model.py` – a minimal linear classifier trained on a small
  labelled dataset.
- `reward_model.py` – a more expressive Transformer based scorer supporting
  contrastive training and saving/loading checkpoints.

Run the demo for the simple model with:

```bash
python simple_reward_model.py
```

The demo prints higher scores for correct answers and can be extended to create
custom reward models for GRPO training.

## Running the Tests

Unit tests cover the data utilities, GRPO trainer, reward models and evaluation
code.  Run them with:

```bash
pytest
```

All tests should pass; one test is skipped when WordNet data is unavailable.
