#!/bin/bash
# Example CE training using default hyperparameters on a text dataset.
# Replace DATA_PATH with the path to your dataset and MODEL_PATH with the base model.
DATA_PATH="data/train.jsonl"
MODEL_PATH="llama_750m"
OUTPUT_DIR="ce_model"
CKPT="$OUTPUT_DIR/ce.ckpt"
python trainingv2.py \
    --dataset "$DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --iters 10000 \
    --batch_size 8 \
    --resume "$CKPT" \
    --save_interval 1000

# To resume training after interruption run the same command again:
# python trainingv2.py \
#     --dataset "$DATA_PATH" \
#     --model_path "$MODEL_PATH" \
#     --output_dir "$OUTPUT_DIR" \
#     --iters 10000 \
#     --batch_size 8 \
#     --resume "$CKPT" \
#     --save_interval 1000
