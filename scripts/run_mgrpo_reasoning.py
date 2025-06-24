#!/usr/bin/env python3
"""Run Multi-layer GRPO on reasoning benchmarks.

This wrapper loads the selected dataset, writes a temporary JSONL file in the
format expected by ``grpo_train.py`` and launches training with the hyper-
parameters from ``scripts/paper_config.json``.
"""
import argparse
import json
import subprocess
import tempfile
from grpo_data import (
    load_math_dataset,
    load_gsm8k_dataset,
    load_minerva_math_dataset,
    load_olympiadbench_dataset,
)

DATASET_LOADERS = {
    "math": load_math_dataset,
    "gsm8k": load_gsm8k_dataset,
    "minerva": load_minerva_math_dataset,
    "olympiadbench": load_olympiadbench_dataset,
}

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MGRPO on a reasoning benchmark")
    parser.add_argument(
        "dataset", choices=DATASET_LOADERS.keys(), help="Benchmark to train on"
    )
    parser.add_argument("--model_path", default="llama_750m", help="Pretrained model")
    parser.add_argument("--reward_model", required=True, help="RewardModel checkpoint")
    parser.add_argument("--output_dir", default="mgrpo_model", help="Directory for the trained model")
    parser.add_argument(
        "--config",
        default="scripts/paper_config.json",
        help="JSON config with hyperparameters",
    )
    parser.add_argument(
        "--guiding_prompt",
        default="Review and correct the answer:",
        help="Prompt used during the second pass",
    )
    args = parser.parse_args(argv)

    data = DATASET_LOADERS[args.dataset]()
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for rec in data:
            json.dump({"query": rec["query"], "answer": rec["answer"]}, f)
            f.write("\n")
        dataset_path = f.name

    cmd = [
        "python",
        "grpo_train.py",
        "--dataset",
        dataset_path,
        "--model_path",
        args.model_path,
        "--reward_model",
        args.reward_model,
        "--output_dir",
        args.output_dir,
        "--two_layer",
        "--guiding_prompt",
        args.guiding_prompt,
        "--config",
        args.config,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
