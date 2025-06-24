#!/bin/bash
# Run MGRPO on the MATH benchmark using paper hyperparameters.
MODEL_PATH="llama_750m"
REWARD_MODEL="rm.ckpt"
OUTPUT_DIR="mgrpo_math"
python scripts/run_mgrpo_reasoning.py math \
    --model_path "$MODEL_PATH" \
    --reward_model "$REWARD_MODEL" \
    --output_dir "$OUTPUT_DIR"
