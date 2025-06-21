import torch
import torch.nn as nn
from typing import Iterable, Sequence, Optional, Callable

class RewardModel(nn.Module):
    """Transformer-based scorer for query/response pairs.

    This model is intentionally more complex than the previous lightweight
    version.  It uses positional and segment embeddings together with a
    deep Transformer encoder to provide robust scoring for query/response
    pairs.  It is designed for contrastive training where correct
    responses should obtain higher scores than incorrect ones.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 256,
        dropout: float = 0.1,
        tokenizer: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_length, hidden_size)
        self.seg_embed = nn.Embedding(2, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        sep = getattr(tokenizer, "sep_token_id", None)
        if sep is None:
            sep = getattr(tokenizer, "eos_token_id", 0)
        self.sep_id = int(sep)
        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_ids: Optional[torch.Tensor] = None,
        seg_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        if pos_ids is not None:
            x = x + self.pos_embed(pos_ids)
        if seg_ids is not None:
            x = x + self.seg_embed(seg_ids)
        x = self.dropout(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)

    def _score_ids(self, query_ids: torch.Tensor, resp_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.cat([
            query_ids,
            torch.tensor([self.sep_id], dtype=torch.long),
            resp_ids,
        ])
        if ids.size(0) > self.max_length:
            ids = ids[: self.max_length]
        pos = torch.arange(len(ids), dtype=torch.long)
        seg = torch.cat([
            torch.zeros(len(query_ids), dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.ones(len(resp_ids), dtype=torch.long),
        ])[: len(ids)]
        return self.forward(ids.unsqueeze(0), pos.unsqueeze(0), seg.unsqueeze(0))[0]

    def score(self, query: str, response: str) -> float:
        if self.tokenizer is None:
            raise ValueError("RewardModel requires a tokenizer")
        q_ids = torch.tensor(
            self.tokenizer.encode(query, add_special_tokens=False), dtype=torch.long
        )
        r_ids = torch.tensor(
            self.tokenizer.encode(response, add_special_tokens=False),
            dtype=torch.long,
        )
        with torch.no_grad():
            val = self._score_ids(q_ids, r_ids)
        return float(val)

    def contrastive_loss(
        self,
        queries: Sequence[Sequence[int]],
        positives: Sequence[Sequence[int]],
        negatives: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Return logistic contrastive loss for batches of token ids."""
        device = next(self.parameters()).device
        loss = 0.0
        for q, p, n in zip(queries, positives, negatives):
            q_ids = torch.tensor(q, dtype=torch.long, device=device)
            p_ids = torch.tensor(p, dtype=torch.long, device=device)
            n_ids = torch.tensor(n, dtype=torch.long, device=device)
            pos = self._score_ids(q_ids, p_ids)
            neg = self._score_ids(q_ids, n_ids)
            loss = loss + torch.nn.functional.softplus(-(pos - neg))
        return loss / len(queries)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, tokenizer, **kwargs) -> "RewardModel":
        vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))
        model = cls(vocab_size=vocab_size, tokenizer=tokenizer, **kwargs)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model


def load_reward_models(
    paths: Sequence[str],
    tokenizer,
    weights: Sequence[float] | None = None,
) -> Callable[[str, str, str], float]:
    """Return a scoring function combining multiple reward models.

    Parameters
    ----------
    paths : list[str]
        Checkpoints to load with :meth:`RewardModel.load`.
    tokenizer : object
        Tokenizer passed to ``RewardModel.load``.
    weights : list[float], optional
        Relative weights for each model. Defaults to equal weighting.
    """

    models = [RewardModel.load(p, tokenizer) for p in paths]
    if weights is None:
        weights = [1.0] * len(models)
    if len(weights) != len(models):
        raise ValueError("weights must match number of reward models")
    total = float(sum(weights))
    norm = [float(w) / total for w in weights]

    def score_fn(generated: str, reference: str, query: str) -> float:
        score = 0.0
        for m, w in zip(models, norm):
            score += w * m.score(query, generated)
        return score

    return score_fn
