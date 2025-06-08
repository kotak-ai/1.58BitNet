"""Reward utilities for GRPO training on QA tasks."""

from collections import Counter
import re
from typing import Set
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.stem import PorterStemmer

try:
    from nltk.corpus import wordnet as wn
    _WN_AVAILABLE = True
except Exception:  # pragma: no cover - wordnet may be missing
    _WN_AVAILABLE = False

_STEM = PorterStemmer()

# minimal synonym fallback when WordNet is unavailable
_SYNONYMS = {
    _STEM.stem("car"): {_STEM.stem("automobile"), _STEM.stem("auto")},
    _STEM.stem("automobile"): {_STEM.stem("car"), _STEM.stem("auto")},
}

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text

def _tokenize(text: str) -> Set[str]:
    tokens = [_STEM.stem(t) for t in _normalize(text).split()]
    expanded: Set[str] = set(tokens)
    if _WN_AVAILABLE:
        for t in list(expanded):
            try:
                syns = wn.synsets(t)
            except LookupError:  # corpus not downloaded
                syns = []
            for syn in syns:
                for lemma in syn.lemma_names():
                    expanded.add(_STEM.stem(lemma.lower()))
    # manual synonyms fallback
    for t in list(expanded):
        if t in _SYNONYMS:
            expanded.update(_STEM.stem(s) for s in _SYNONYMS[t])
    return expanded


def f1_score(prediction: str, ground_truth: str) -> float:
    """Robust F1 that uses stemming and optional WordNet synonyms."""
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(ground_truth)
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    num_same = len(pred_tokens & gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def qa_reward(generated: str, reference: str) -> float:
    """Return a robust F1 reward for QA tasks."""
    return f1_score(generated, reference)


class RewardModelScorer:
    """Use a sequence classification model to score responses."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score(self, query: str, response: str) -> float:
        inp = self.tokenizer(query, response, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inp).logits
        probs = torch.softmax(logits, dim=-1)
        if probs.size(-1) == 1:
            return float(torch.sigmoid(logits)[0, 0])
        return float(probs[0, -1])
