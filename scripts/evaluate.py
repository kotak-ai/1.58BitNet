#!/usr/bin/env python3
"""Run evaluation on the official reasoning datasets and print a table."""
import pandas as pd
from evaluation import run

CE_MODEL = "ce_model"      # path to CE model
GRPO_MODEL = "grpo_model"  # path to GRPO model

def main():
    datasets = ["math", "gsm8k", "minerva", "olympiadbench"]
    rows = []
    for name in datasets:
        metrics = run(name, CE_MODEL, GRPO_MODEL, task="reasoning", two_layer=True)
        m = metrics["grpo"]
        rows.append({
            "Dataset": name,
            "Acc.@t1": m["accuracy_t1"],
            "Acc.@t2": m["accuracy_t2"],
            "Delta_i->c": m["delta_i2c"],
            "Delta_c->i": m["delta_c2i"],
        })
    df = pd.DataFrame(rows)
    df["Acc.@t1"] *= 100
    df["Acc.@t2"] *= 100
    df["Delta_i->c"] *= 100
    df["Delta_c->i"] *= 100
    print(df.to_string(index=False, formatters={col: lambda x: f"{x:.1f}" for col in df.columns[1:]}))

if __name__ == "__main__":
    main()
