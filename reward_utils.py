"""Reward utilities for GRPO training on QA tasks."""

from collections import Counter
import re
from typing import Set
import torch
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:  # pragma: no cover - transformers may be missing
    AutoTokenizer = AutoModelForSequenceClassification = None  # type: ignore[misc]
from nltk.stem import PorterStemmer

# Attempt to provide WordNet access via nltk first and fall back to the
# optional ``wn`` library when available.  ``_get_synsets`` is defined to
# return an iterable of synsets for a token or an empty list when no WordNet
# data is accessible.
try:
    from nltk.corpus import wordnet as _nltk_wn
    try:
        _nltk_wn.synsets("test")  # verify corpus exists
        _WN_AVAILABLE = True
        def _get_synsets(token: str):
            return _nltk_wn.synsets(token)
    except LookupError:
        _WN_AVAILABLE = False
except Exception:  # pragma: no cover - nltk may be missing
    _WN_AVAILABLE = False

if not _WN_AVAILABLE:
    try:
        import wn as _wnlib  # type: ignore
        try:
            _wnlex = _wnlib.Wordnet("omw-en")
            _wnlex.synsets("test")  # verify lexicon exists
            _WN_AVAILABLE = True
            def _get_synsets(token: str):
                return _wnlex.synsets(token)
        except Exception:
            _WN_AVAILABLE = False
    except Exception:  # pragma: no cover - wn package missing
        _WN_AVAILABLE = False

if not _WN_AVAILABLE:  # pragma: no cover - executed when WordNet missing
    def _get_synsets(token: str):
        return []

_STEM = PorterStemmer()


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
            for syn in _get_synsets(t):
                for lemma in syn.lemma_names():
                    expanded.add(_STEM.stem(lemma.lower()))
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
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required for RewardModelScorer")
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
