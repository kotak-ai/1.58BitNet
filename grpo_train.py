import argparse
import random
import torch
from transformers import AutoTokenizer
from llama_model import LlamaModel
from grpo import GRPOTrainer
from grpo_data import load_qa_dataset, build_grpo_batch



def main():
    parser = argparse.ArgumentParser(description="GRPO training loop")
    parser.add_argument("--dataset", type=str, required=True, help="Path to JSON or JSONL dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Directory with pretrained model")
    parser.add_argument("--output_dir", type=str, default="grpo_model", help="Where to save the trained model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    args = parser.parse_args()

    dataset = load_qa_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaModel.load_pretrained(args.model_path)
    ref_model = LlamaModel.load_pretrained(args.model_path)
    trainer = GRPOTrainer(model, ref_model, clip_eps=args.clip_eps, beta=args.beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        batch = random.sample(dataset, args.batch_size)
        q, r, l, rew = build_grpo_batch(batch, tokenizer, model, args.group_size, args.max_length)
        loss = trainer.step(q, r, l, rew, optimizer)
        if step % 10 == 0:
            print(f"Step {step}: loss {loss.item():.4f}")

    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
