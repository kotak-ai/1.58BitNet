import argparse
import torch
from llama_model import LlamaModel
from grpo_data import load_qa_dataset, construct_second_pass_input
from reward_utils import qa_reward


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO vs CE models")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ce_model", type=str, required=True)
    parser.add_argument("--grpo_model", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=20)
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
    args = parser.parse_args()

    data = load_qa_dataset(args.dataset)
    from transformers import AutoTokenizer
    ce_tok = AutoTokenizer.from_pretrained(args.ce_model)
    grpo_tok = AutoTokenizer.from_pretrained(args.grpo_model)
    ce_model = LlamaModel.load_pretrained(args.ce_model)
    grpo_model = LlamaModel.load_pretrained(args.grpo_model)

    ce_score = evaluate_model(
        ce_model,
        ce_tok,
        data,
        args.max_length,
        two_layer=args.two_layer,
        guiding_prompt=args.guiding_prompt,
        second_max_length=args.second_max_length,
    )
    grpo_score = evaluate_model(
        grpo_model,
        grpo_tok,
        data,
        args.max_length,
        two_layer=args.two_layer,
        guiding_prompt=args.guiding_prompt,
        second_max_length=args.second_max_length,
    )

    print(f"CE model F1: {ce_score:.4f}")
    print(f"GRPO model F1: {grpo_score:.4f}")


if __name__ == "__main__":
    main()
