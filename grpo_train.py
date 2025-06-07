import argparse
import json
import random
import torch
from transformers import AutoTokenizer
from llama_model import LlamaModel
from grpo import GRPOTrainer
from grpo_data import load_qa_dataset, build_grpo_batch
from reward_utils import qa_reward


def load_dataset(path):
    """Load a dataset of {"query":..., "answer":...}. Supports JSON and JSONL."""
    data = []
    if path.endswith(".jsonl"):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                data.append({'query': obj['query'], 'answer': obj['answer']})
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            for obj in json.load(f):
                data.append({'query': obj['query'], 'answer': obj['answer']})
    else:
        raise ValueError('Unsupported dataset format')
    return data


def reward_fn(generated: str, reference: str) -> float:
    """F1-based reward for QA datasets."""
    return qa_reward(generated, reference)


def pad_sequences(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        lengths[i] = len(s)
    return tensor, lengths


def prepare_batch(samples, tokenizer, model, group_size, max_length):
    q_tokens = [tokenizer.encode(s['query'], add_special_tokens=False) for s in samples]
    answers = [s['answer'] for s in samples]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    queries, _ = pad_sequences(q_tokens, pad_id)

    B = len(samples)
    responses = []
    lengths = []
    rewards = []
    for i in range(B):
        q = q_tokens[i]
        inp = torch.tensor([q], dtype=torch.long)
        grp_resp = []
        grp_len = []
        grp_rew = []
        for _ in range(group_size):
            out = model.generate(inp, max_length=len(q) + max_length, do_sample=True)
            resp = out[0, len(q):].tolist()
            grp_resp.append(resp)
            grp_len.append(len(resp))
            gen_text = tokenizer.decode(resp)
            grp_rew.append(reward_fn(gen_text, answers[i]))
        responses.append(grp_resp)
        lengths.append(grp_len)
        rewards.append(grp_rew)

    max_resp_len = max(max(l) for l in lengths)
    resp_tensor = torch.full((B, group_size, max_resp_len), pad_id, dtype=torch.long)
    len_tensor = torch.zeros(B, group_size, dtype=torch.long)
    for b in range(B):
        for g in range(group_size):
            seq = responses[b][g]
            resp_tensor[b, g, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            len_tensor[b, g] = len(seq)
    reward_tensor = torch.tensor(rewards, dtype=torch.float)
    return queries, resp_tensor, len_tensor, reward_tensor

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
