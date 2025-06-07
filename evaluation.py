import argparse
from llama_model import LlamaModel
from transformers import AutoTokenizer
from grpo_data import load_qa_dataset
from reward_utils import qa_reward


def generate_response(model, tokenizer, query: str, max_length: int) -> str:
    tokens = tokenizer.encode(query, return_tensors="pt")
    out = model.generate(tokens, max_length=tokens.size(1) + max_length, do_sample=False)
    resp_tokens = out[0, tokens.size(1):].tolist()
    return tokenizer.decode(resp_tokens)


def evaluate_model(model, tokenizer, dataset, max_length: int) -> float:
    scores = []
    for sample in dataset:
        resp = generate_response(model, tokenizer, sample["query"], max_length)
        scores.append(qa_reward(resp, sample["answer"]))
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO vs CE models")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ce_model", type=str, required=True)
    parser.add_argument("--grpo_model", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=20)
    args = parser.parse_args()

    data = load_qa_dataset(args.dataset)
    ce_tok = AutoTokenizer.from_pretrained(args.ce_model)
    grpo_tok = AutoTokenizer.from_pretrained(args.grpo_model)
    ce_model = LlamaModel.load_pretrained(args.ce_model)
    grpo_model = LlamaModel.load_pretrained(args.grpo_model)

    ce_score = evaluate_model(ce_model, ce_tok, data, args.max_length)
    grpo_score = evaluate_model(grpo_model, grpo_tok, data, args.max_length)

    print(f"CE model F1: {ce_score:.4f}")
    print(f"GRPO model F1: {grpo_score:.4f}")


if __name__ == "__main__":
    main()
