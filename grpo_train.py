import argparse
import json
import random
import logging
import csv
import os
import torch
from llama_model import LlamaModel
from grpo import GRPOTrainer, MultiLayerGRPOTrainer
from grpo_data import load_qa_dataset, build_grpo_batch, f1_reward
from reward_utils import qa_reward
from reward_model import RewardModel

try:  # progress bar is optional
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm may not be installed
    tqdm = None


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



def pad_sequences(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        lengths[i] = len(s)
    return tensor, lengths


def prepare_batch(samples, tokenizer, model, group_size, max_length, reward_fn=f1_reward):
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
            grp_rew.append(reward_fn(gen_text, answers[i], samples[i]["query"]))
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


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str) -> None:
    """Save model/optimizer states and step to ``path``."""
    torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    """Load model/optimizer states from ``path`` and return training step."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    return int(ckpt.get("step", 0))


def get_arg_parser() -> argparse.ArgumentParser:
    """Return an ArgumentParser with all CLI options."""
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
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--two_layer", action="store_true", help="Use MultiLayer GRPO")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument(
        "--reward_model",
        type=str,
        default=None,
        help="Path to RewardModel checkpoint (use F1 reward if not set)",
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between logging metrics")
    parser.add_argument(
        "--csv_log",
        type=str,
        default=None,
        help="Optional CSV log file (includes a few corrected samples if two_layer is used)",
    )
    parser.add_argument(
        "--guiding_prompt",
        type=str,
        default="Review and correct the answer:",
        help="Guiding prompt for the second GRPO layer",
    )
    parser.add_argument("--progress", action="store_true", help="Show progress bar if tqdm is available")
    return parser


def update_args_with_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Apply configuration from ``args.config`` to ``args`` and parser defaults."""
    if not getattr(args, "config", None):
        return
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "guiding_prompt" in cfg:
        gp = cfg["guiding_prompt"]
        if isinstance(gp, str) and os.path.isfile(gp):
            with open(gp, "r", encoding="utf-8") as pf:
                cfg["guiding_prompt"] = pf.read().strip()
    for key, value in cfg.items():
        old_default = parser.get_default(key)
        parser.set_defaults(**{key: value})
        if hasattr(args, key) and getattr(args, key) == old_default:
            setattr(args, key, value)


def simple_improvement_verifier(new_reward: float, old_reward: float, threshold: float = 0.05) -> bool:
    """Return True if ``new_reward`` exceeds ``old_reward`` by ``threshold``."""
    return (new_reward - old_reward) > threshold

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    update_args_with_config(args, parser)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
    )

    csv_writer = None
    NUM_LOG_TEXT = 3
    if args.csv_log:
        fieldnames = ["step", "loss", "mean_reward", "kl"]
        if args.two_layer:
            fieldnames.append("improvement_rate")
            fieldnames.extend([f"corrected_{i+1}" for i in range(NUM_LOG_TEXT)])
        write_header = not os.path.exists(args.csv_log)
        csv_file = open(args.csv_log, "a", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            csv_writer.writeheader()

    dataset = load_qa_dataset(args.dataset)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.reward_model:
        reward_model = RewardModel.load(args.reward_model, tokenizer)
        def reward_fn(gen: str, ref: str, query: str) -> float:
            return reward_model.score(query, gen)
    else:
        def reward_fn(gen: str, ref: str, query: str) -> float:
            return qa_reward(gen, ref)
    model = LlamaModel.load_pretrained(args.model_path)
    ref_model = LlamaModel.load_pretrained(args.model_path)
    if args.two_layer:
        answers_holder = {"answers": []}

        def second_layer_reward(text: str) -> float:
            return max(qa_reward(text, a) for a in answers_holder["answers"])

        trainer = MultiLayerGRPOTrainer(
            model,
            ref_model,
            second_layer_reward,
            tokenizer,
            guiding_prompt=args.guiding_prompt,
            clip_eps=args.clip_eps,
            beta=args.beta,
            verifier=simple_improvement_verifier,
        )
    else:
        trainer = GRPOTrainer(model, ref_model, clip_eps=args.clip_eps, beta=args.beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, args.resume)
        logging.info("Resumed from step %d", start_step)

    iterator = range(start_step, args.steps)
    if args.progress and tqdm is not None:
        iterator = tqdm(iterator, total=args.steps, initial=start_step)

    for step in iterator:
        batch = random.sample(dataset, args.batch_size)
        q, r, l, rew = build_grpo_batch(
            batch,
            tokenizer,
            model,
            args.group_size,
            args.max_length,
            reward_fn=reward_fn,
        )
        mean_reward = float(rew.mean())

        with torch.no_grad():
            B, G, L = r.shape
            flat_r = r.view(B * G, L)
            flat_l = l.view(B * G)
            logits_p = model(flat_r)
            logits_ref = ref_model(flat_r)
            if args.two_layer:
                log_fn = trainer.layer1._log_probs
            else:
                log_fn = trainer._log_probs
            lp = log_fn(logits_p, flat_r)
            lr_ref = log_fn(logits_ref, flat_r)
            mask = torch.arange(L, device=flat_r.device).unsqueeze(0) < flat_l.unsqueeze(1)
            kl = (torch.exp(lp) * (lp - lr_ref)) * mask
            kl_div = kl.sum() / mask.sum()

        corrected_texts = []
        if args.two_layer:
            answers_holder["answers"] = [s["answer"] for s in batch]
            log_n = NUM_LOG_TEXT if csv_writer else 0
            res = trainer.train_batch(q, r, l, rew, optimizer, log_texts=log_n)
            if log_n:
                loss, rate, corrected_texts = res
            else:
                loss, rate = res
        else:
            loss = trainer.step(q, r, l, rew, optimizer)
            rate = None

        metrics = {
            "step": step,
            "loss": loss.item(),
            "mean_reward": mean_reward,
            "kl": kl_div.item(),
        }
        if rate is not None:
            metrics["improvement_rate"] = rate
        if args.two_layer:
            for i in range(NUM_LOG_TEXT):
                key = f"corrected_{i+1}"
                if i < len(corrected_texts):
                    metrics[key] = corrected_texts[i]
                else:
                    metrics[key] = ""

        if step % args.log_interval == 0:
            msg = (
                f"Step {step}: loss {metrics['loss']:.4f}, "
                f"reward {metrics['mean_reward']:.4f}, kl {metrics['kl']:.4f}"
            )
            if rate is not None:
                msg += f", improvement rate {rate:.2f}"
            logging.info(msg)
            if csv_writer:
                csv_writer.writerow(metrics)

    model.save_pretrained(args.output_dir)
    if args.resume:
        save_checkpoint(model, optimizer, args.steps, args.resume)
    if csv_writer:
        csv_file.close()


if __name__ == "__main__":
    main()
