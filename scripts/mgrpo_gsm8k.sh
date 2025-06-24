#!/bin/bash
# Run MGRPO on the GSM8K dataset using paper hyperparameters.
MODEL_PATH="llama_750m"
REWARD_MODEL="rm.ckpt"
OUTPUT_DIR="mgrpo_gsm8k"
python scripts/run_mgrpo_reasoning.py gsm8k \
    --model_path "$MODEL_PATH" \
    --reward_model "$REWARD_MODEL" \
    --output_dir "$OUTPUT_DIR"
