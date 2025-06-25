#!/bin/bash
# Example GRPO training mixing two reward models.
# Replace DATA_PATH, MODEL_PATH and reward checkpoint paths as needed.
DATA_PATH="qa.jsonl"
MODEL_PATH="llama_750m"
REWARD_MODEL1="rm1.ckpt"
REWARD_MODEL2="rm2.ckpt"
OUTPUT_DIR="grpo_two_rewards"
python grpo_train.py \
    --dataset "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --reward_model "$REWARD_MODEL1" "$REWARD_MODEL2" \
    --reward_weights 0.7 0.3 \
    --rule_weight 0.5 \
    --output_dir "$OUTPUT_DIR"
