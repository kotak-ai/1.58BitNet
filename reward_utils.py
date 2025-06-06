import re
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_reward(generated: str, reference: str) -> float:
    """F1-based reward for QA tasks."""
    return f1_score(generated, reference)


class RewardModelScorer:
    """Use a sequence classification model to score responses."""

    def __init__(self, model_name: str):
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
