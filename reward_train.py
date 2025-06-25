import argparse
import json
import os
import random
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer

from reward_model import RewardModel
from training_utils import save_checkpoint, load_checkpoint

try:  # optional progress bar
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm may be missing
    tqdm = None


def load_labelled_dataset(path: str) -> List[Dict[str, object]]:
    """Load a dataset of ``{"query":..., "answer":..., "label":...}``."""
    data: List[Dict[str, object]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                data.append({
                    "query": obj["query"],
                    "answer": obj["answer"],
                    "label": float(obj["label"]),
                })
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            for obj in json.load(f):
                data.append({
                    "query": obj["query"],
                    "answer": obj["answer"],
                    "label": float(obj["label"]),
                })
    else:
        raise ValueError("Unsupported dataset format")
    return data


def load_pair_dataset(path: str) -> List[Dict[str, str]]:
    """Load a dataset of ``{"query":..., "positive":..., "negative":...}``."""
    data: List[Dict[str, str]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                data.append({
                    "query": obj["query"],
                    "positive": obj["positive"],
                    "negative": obj["negative"],
                })
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            for obj in json.load(f):
                data.append({
                    "query": obj["query"],
                    "positive": obj["positive"],
                    "negative": obj["negative"],
                })
    else:
        raise ValueError("Unsupported dataset format")
    return data


def batch_iterator(dataset: List[Dict[str, object]], batch_size: int):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        batch = [dataset[j] for j in batch_idx]
        yield batch


def train(
    model: RewardModel,
    tokenizer,
    dataset: Optional[List[Dict[str, object]]] = None,
    *,
    pairs: Optional[List[Dict[str, str]]] = None,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-4,
    resume: str | None = None,
    save_interval: int = 0,
    progress: bool = False,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_step = 0
    if resume and os.path.exists(resume):
        start_step = load_checkpoint(model, optimizer, resume)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    step = start_step
    iterator = range(epochs)
    if progress and tqdm is not None:
        iterator = tqdm(iterator, total=epochs)
    for _ in iterator:
        if pairs is not None:
            for batch in batch_iterator(pairs, batch_size):
                queries = [
                    tokenizer.encode(it["query"], add_special_tokens=False)
                    for it in batch
                ]
                positives = [
                    tokenizer.encode(it["positive"], add_special_tokens=False)
                    for it in batch
                ]
                negatives = [
                    tokenizer.encode(it["negative"], add_special_tokens=False)
                    for it in batch
                ]
                loss = model.contrastive_loss(queries, positives, negatives)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if resume and save_interval and step % save_interval == 0:
                    save_checkpoint(model, optimizer, step, resume)
        elif dataset is not None:
            for batch in batch_iterator(dataset, batch_size):
                scores = []
                labels = []
                for item in batch:
                    q_ids = torch.tensor(
                        tokenizer.encode(item["query"], add_special_tokens=False),
                        dtype=torch.long,
                    )
                    a_ids = torch.tensor(
                        tokenizer.encode(item["answer"], add_special_tokens=False),
                        dtype=torch.long,
                    )
                    score = model._score_ids(q_ids, a_ids)
                    scores.append(score)
                    labels.append(float(item["label"]))
                logits = torch.stack(scores)
                tgt = torch.tensor(labels, dtype=torch.float32)
                loss = loss_fn(logits, tgt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if resume and save_interval and step % save_interval == 0:
                    save_checkpoint(model, optimizer, step, resume)
        else:
            raise ValueError("No training data provided")
    if resume:
        save_checkpoint(model, optimizer, step, resume)


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a RewardModel")
    parser.add_argument(
        "--dataset",
        help="JSON or JSONL dataset with 'query', 'answer' and 'label' fields",
    )
    parser.add_argument(
        "--pairs",
        help="JSON or JSONL dataset with 'query', 'positive' and 'negative' fields",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to pretrained tokenizer (HF format)",
    )
    parser.add_argument(
        "--output",
        default="reward_model.pt",
        help="Where to save the trained model state_dict",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file")
    parser.add_argument("--save_interval", type=int, default=0, help="Steps between checkpoint saves")
    parser.add_argument("--progress", action="store_true", help="Show progress bar if tqdm is available")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    if not args.dataset and not args.pairs:
        parser.error("one of --dataset or --pairs is required")

    dataset = load_labelled_dataset(args.dataset) if args.dataset else None
    pairs = load_pair_dataset(args.pairs) if args.pairs else None
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = RewardModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_length=args.max_length,
        tokenizer=tokenizer,
    )
    train(
        model,
        tokenizer,
        dataset,
        pairs=pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
        save_interval=args.save_interval,
        progress=args.progress,
    )
    model.save(args.output)


if __name__ == "__main__":
    main()
