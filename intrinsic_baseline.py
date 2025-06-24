import argparse
from transformers import AutoTokenizer
from llama_model import LlamaModel
from grpo_data import (
    load_qa_dataset,
    load_math_dataset,
    load_gsm8k_dataset,
    load_minerva_math_dataset,
    load_olympiadbench_dataset,
)
from evaluation import evaluate_model, evaluate_reasoning_model


def load_dataset(name: str):
    name_l = name.lower()
    if name_l == "math":
        return load_math_dataset()
    if name_l == "gsm8k":
        return load_gsm8k_dataset()
    if name_l == "minerva":
        return load_minerva_math_dataset()
    if name_l == "olympiadbench":
        return load_olympiadbench_dataset()
    return load_qa_dataset(name)


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the intrinsic self-correction baseline",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--task",
        choices=["qa", "reasoning"],
        default="qa",
        help="Dataset type: 'qa' uses F1 reward, 'reasoning' uses accuracy",
    )
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument(
        "--guiding_prompt",
        type=str,
        default="Review and correct the answer:",
        help="Prompt appended during the second pass",
    )
    parser.add_argument(
        "--second_max_length",
        type=int,
        default=20,
        help="Tokens to generate for the correction",
    )
    return parser


def run(
    dataset: str,
    model: str,
    *,
    task: str = "qa",
    max_length: int = 20,
    guiding_prompt: str = "Review and correct the answer:",
    second_max_length: int = 20,
) -> dict:
    data = load_dataset(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = LlamaModel.load_pretrained(model)
    if task == "qa":
        score = evaluate_model(
            model_obj,
            tokenizer,
            data,
            max_length,
            two_layer=True,
            guiding_prompt=guiding_prompt,
            second_max_length=second_max_length,
        )
        print(f"Intrinsic self-correction F1: {score:.4f}")
        return {"f1": score}
    metrics = evaluate_reasoning_model(
        model_obj,
        tokenizer,
        data,
        max_length,
        two_layer=True,
        guiding_prompt=guiding_prompt,
        second_max_length=second_max_length,
    )
    metrics_to_show = [
        "accuracy_t1",
        "accuracy_t1_prime",
        "accuracy_t2",
        "delta_t1_t2",
        "delta_t1p_t2",
        "delta_i2c",
        "delta_c2i",
    ]
    header = f"{'Metric':<18}{'Value':>12}"
    print(header)
    print("-" * len(header))
    for key in metrics_to_show:
        val = metrics.get(key)
        val_str = f"{val:.4f}" if val is not None else "-"
        print(f"{key:<18}{val_str:>12}")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    run(
        args.dataset,
        args.model,
        task=args.task,
        max_length=args.max_length,
        guiding_prompt=args.guiding_prompt,
        second_max_length=args.second_max_length,
    )


if __name__ == "__main__":
    main()
