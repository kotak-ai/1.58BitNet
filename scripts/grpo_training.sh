#!/bin/bash
# Example GRPO training with default hyperparameters on a QA dataset.
# Replace DATA_PATH, MODEL_PATH and REWARD_MODEL as needed.
DATA_PATH="qa.jsonl"
MODEL_PATH="llama_750m"
REWARD_MODEL="rm.ckpt"  # path to RewardModel checkpoint
OUTPUT_DIR="grpo_model"
python grpo_train.py \
    --dataset "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --reward_model "$REWARD_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --steps 1000
