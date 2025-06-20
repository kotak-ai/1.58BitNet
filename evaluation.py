import argparse
import torch
from llama_model import LlamaModel
from grpo_data import (
    load_qa_dataset,
    construct_second_pass_input,
)
from reward_utils import qa_reward, accuracy_reward


def generate_response(
    model, tokenizer, query: str, max_length: int, return_tokens: bool = False
) -> str | tuple[torch.Tensor, str]:
    tokens = tokenizer.encode(query, return_tensors="pt")
    out = model.generate(tokens, max_length=tokens.size(1) + max_length, do_sample=False)
    resp_tokens = out[0, tokens.size(1):]
    text = tokenizer.decode(resp_tokens.tolist())
    if return_tokens:
        return resp_tokens, text
    return text


def evaluate_model(
    model,
    tokenizer,
    dataset,
    max_length: int,
    *,
    two_layer: bool = False,
    guiding_prompt: str = "Review and correct the answer:",
    second_max_length: int = 20,
) -> float:
    scores = []
    if two_layer:
        guidance_tokens = torch.tensor(
            tokenizer.encode(guiding_prompt, add_special_tokens=False),
            dtype=torch.long,
        )
        sep = getattr(tokenizer, "sep_token_id", None)
        if sep is None:
            sep = getattr(tokenizer, "eos_token_id", 0)
        sep = int(sep)
    for sample in dataset:
        if two_layer:
            query_tokens = tokenizer.encode(sample["query"], add_special_tokens=False)
            inp = torch.tensor([query_tokens], dtype=torch.long)
            out = model.generate(inp, max_length=len(query_tokens) + max_length, do_sample=False)
            first_resp = out[0, len(query_tokens):]
            q_tokens = torch.tensor(query_tokens, dtype=torch.long)
            sec_inp, sec_len = construct_second_pass_input(
                q_tokens,
                first_resp,
                guidance_tokens,
                sep,
            )
            gen = model.generate(
                sec_inp.unsqueeze(0),
                max_length=sec_len + second_max_length,
                do_sample=False,
            )
            final_resp = gen[0, sec_len:]
            resp_text = tokenizer.decode(final_resp.tolist())
        else:
            resp_text = generate_response(model, tokenizer, sample["query"], max_length)
        scores.append(qa_reward(resp_text, sample["answer"]))
    return sum(scores) / len(scores)


def evaluate_reasoning_model(
    model,
    tokenizer,
    dataset,
    max_length: int,
    *,
    two_layer: bool = False,
    guiding_prompt: str = "Review and correct the answer:",
    second_max_length: int = 20,
) -> dict:
    """Evaluate ``model`` on a reasoning dataset using accuracy metrics."""

    correct_first = 0
    correct_second = 0
    changed_ic = 0
    changed_ci = 0
    if two_layer:
        guidance_tokens = torch.tensor(
            tokenizer.encode(guiding_prompt, add_special_tokens=False),
            dtype=torch.long,
        )
        sep = getattr(tokenizer, "sep_token_id", None)
        if sep is None:
            sep = getattr(tokenizer, "eos_token_id", 0)
        sep = int(sep)
    for sample in dataset:
        query_tokens = tokenizer.encode(sample["query"], add_special_tokens=False)
        inp = torch.tensor([query_tokens], dtype=torch.long)
        out = model.generate(inp, max_length=len(query_tokens) + max_length, do_sample=False)
        first_resp = out[0, len(query_tokens):]
        first_text = tokenizer.decode(first_resp.tolist())
        first_ok = bool(accuracy_reward(first_text, sample["answer"]))
        if two_layer:
            q_tokens = torch.tensor(query_tokens, dtype=torch.long)
            sec_inp, sec_len = construct_second_pass_input(
                q_tokens,
                first_resp,
                guidance_tokens,
                sep,
            )
            gen = model.generate(
                sec_inp.unsqueeze(0),
                max_length=sec_len + second_max_length,
                do_sample=False,
            )
            final_resp = gen[0, sec_len:]
            final_text = tokenizer.decode(final_resp.tolist())
        else:
            final_text = first_text
        second_ok = bool(accuracy_reward(final_text, sample["answer"]))

        correct_first += first_ok
        correct_second += second_ok
        if not first_ok and second_ok:
            changed_ic += 1
        if first_ok and not second_ok:
            changed_ci += 1

    N = len(dataset)
    return {
        "accuracy_t1": correct_first / N,
        "accuracy_t2": correct_second / N,
        "delta_i2c": changed_ic / N,
        "delta_c2i": changed_ci / N,
    }


def get_arg_parser() -> argparse.ArgumentParser:
    """Return an argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Evaluate GRPO vs CE models")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ce_model", type=str, required=True)
    parser.add_argument("--grpo_model", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument(
        "--task",
        choices=["qa", "reasoning"],
        default="qa",
        help="Type of dataset: 'qa' uses F1 reward, 'reasoning' uses accuracy",
    )
    parser.add_argument(
        "--two_layer",
        action="store_true",
        help="Generate a corrected answer using a second pass",
    )
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
    ce_model: str,
    grpo_model: str,
    *,
    max_length: int = 20,
    task: str = "qa",
    two_layer: bool = False,
    guiding_prompt: str = "Review and correct the answer:",
    second_max_length: int = 20,
) -> dict:
    """Evaluate the CE and GRPO models and return the metrics."""
    if dataset.lower() == "math":
        from grpo_data import load_math_dataset
        data = load_math_dataset()
    elif dataset.lower() == "gsm8k":
        from grpo_data import load_gsm8k_dataset
        data = load_gsm8k_dataset()
    elif dataset.lower() == "minerva":
        from grpo_data import load_minerva_math_dataset
        data = load_minerva_math_dataset()
    elif dataset.lower() == "olympiadbench":
        from grpo_data import load_olympiadbench_dataset
        data = load_olympiadbench_dataset()
    else:
        data = load_qa_dataset(dataset)
    from transformers import AutoTokenizer
    ce_tok = AutoTokenizer.from_pretrained(ce_model)
    grpo_tok = AutoTokenizer.from_pretrained(grpo_model)
    ce = LlamaModel.load_pretrained(ce_model)
    grpo = LlamaModel.load_pretrained(grpo_model)

    if task == "qa":
        ce_score = evaluate_model(
            ce,
            ce_tok,
            data,
            max_length,
            two_layer=two_layer,
            guiding_prompt=guiding_prompt,
            second_max_length=second_max_length,
        )
        grpo_score = evaluate_model(
            grpo,
            grpo_tok,
            data,
            max_length,
            two_layer=two_layer,
            guiding_prompt=guiding_prompt,
            second_max_length=second_max_length,
        )
        print(f"CE model F1: {ce_score:.4f}")
        print(f"GRPO model F1: {grpo_score:.4f}")
        return {"ce_f1": ce_score, "grpo_f1": grpo_score}
    ce_metrics = evaluate_reasoning_model(
        ce,
        ce_tok,
        data,
        max_length,
        two_layer=two_layer,
        guiding_prompt=guiding_prompt,
        second_max_length=second_max_length,
    )
    grpo_metrics = evaluate_reasoning_model(
        grpo,
        grpo_tok,
        data,
        max_length,
        two_layer=two_layer,
        guiding_prompt=guiding_prompt,
        second_max_length=second_max_length,
    )
    print("CE model metrics:")
    for k, v in ce_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("GRPO model metrics:")
    for k, v in grpo_metrics.items():
        print(f"  {k}: {v:.4f}")
    return {"ce": ce_metrics, "grpo": grpo_metrics}


def main(argv: list[str] | None = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)

    run(
        args.dataset,
        args.ce_model,
        args.grpo_model,
        max_length=args.max_length,
        task=args.task,
        two_layer=args.two_layer,
        guiding_prompt=args.guiding_prompt,
        second_max_length=args.second_max_length,
    )


if __name__ == "__main__":
    main()
