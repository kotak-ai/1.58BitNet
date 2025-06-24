#!/bin/bash
# Run MGRPO on the OlympiadBench dataset using paper hyperparameters.
MODEL_PATH="llama_750m"
REWARD_MODEL="rm.ckpt"
OUTPUT_DIR="mgrpo_olympiad"
python scripts/run_mgrpo_reasoning.py olympiadbench \
    --model_path "$MODEL_PATH" \
    --reward_model "$REWARD_MODEL" \
    --output_dir "$OUTPUT_DIR"
