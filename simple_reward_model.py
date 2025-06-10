import torch
import torch.nn as nn
from typing import List, Dict

class SimpleTokenizer:
    """Very basic whitespace tokenizer that builds a vocabulary on the fly."""
    def __init__(self) -> None:
        self.vocab = {"<pad>": 0}

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for word in text.lower().split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            tokens.append(self.vocab[word])
        return tokens

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def build_tokenizer(dataset: List[Dict[str, str]]) -> SimpleTokenizer:
    """Create a tokenizer using all text in the dataset."""
    tok = SimpleTokenizer()
    for item in dataset:
        tok.encode(item["query"])
        tok.encode(item["answer"])
    return tok


class SimpleRewardModel(nn.Module):
    """Tiny classifier that scores question/answer pairs."""
    def __init__(self, vocab_size: int, embed_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim * 2, 1)

    def forward(self, q_ids: torch.Tensor, a_ids: torch.Tensor) -> torch.Tensor:
        q_emb = self.embed(q_ids).mean(dim=1)
        a_emb = self.embed(a_ids).mean(dim=1)
        x = torch.cat([q_emb, a_emb], dim=-1)
        return self.classifier(x).squeeze(-1)


def train_reward_model(
    model: SimpleRewardModel,
    tokenizer: SimpleTokenizer,
    dataset: List[Dict[str, object]],
    epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    """Train the model on a small QA classification dataset."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for sample in dataset:
            q_ids = torch.tensor([tokenizer.encode(sample["query"])], dtype=torch.long)
            a_ids = torch.tensor([tokenizer.encode(sample["answer"])], dtype=torch.long)
            label = torch.tensor([sample["label"]], dtype=torch.float32)
            opt.zero_grad()
            out = model(q_ids, a_ids)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, label)
            loss.backward()
            opt.step()


def score(model: SimpleRewardModel, tokenizer: SimpleTokenizer, query: str, answer: str) -> float:
    """Return a reward score in the range 0-1 for a QA pair."""
    q_ids = torch.tensor([tokenizer.encode(query)], dtype=torch.long)
    a_ids = torch.tensor([tokenizer.encode(answer)], dtype=torch.long)
    with torch.no_grad():
        logits = model(q_ids, a_ids)
        prob = torch.sigmoid(logits)[0]
    return float(prob)


def _demo() -> None:
    data = [
        {"query": "what is the capital of france", "answer": "paris", "label": 1.0},
        {"query": "what is the capital of france", "answer": "london", "label": 0.0},
    ]
    tok = build_tokenizer(data)
    model = SimpleRewardModel(tok.vocab_size)
    train_reward_model(model, tok, data, epochs=100, lr=0.1)
    good = score(model, tok, "what is the capital of france", "paris")
    bad = score(model, tok, "what is the capital of france", "london")
    print(f"Paris score: {good:.3f}")
    print(f"London score: {bad:.3f}")


if __name__ == "__main__":
    _demo()
