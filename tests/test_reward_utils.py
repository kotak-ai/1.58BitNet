import unittest
import torch
from unittest import mock
import reward_utils
from reward_utils import (
    qa_reward,
    _WN_AVAILABLE,
    reasoning_token_f1,
    step_correctness,
    RewardModelScorer,
)


class RewardUtilsTest(unittest.TestCase):
    def test_exact_match(self):
        pred = "Paris is the capital of France."
        ref = "Paris is the capital of France"
        self.assertAlmostEqual(qa_reward(pred, ref), 1.0)

    def test_partial_match(self):
        pred = "Paris"
        ref = "Paris is the capital of France"
        val = qa_reward(pred, ref)
        self.assertGreater(val, 0.0)
        self.assertLess(val, 1.0)

    @unittest.skipUnless(_WN_AVAILABLE, "WordNet not available")
    def test_synonym(self):
        self.assertGreater(qa_reward("car", "automobile"), 0.0)

    def test_stemming(self):
        self.assertAlmostEqual(qa_reward("running", "run"), 1.0)

    def test_reasoning_token_f1(self):
        pred = "<think>step one</think>42"
        ref_reason = "step one"
        self.assertAlmostEqual(reasoning_token_f1(pred, ref_reason), 1.0)

    def test_step_correctness(self):
        pred = "<think>step one. step two</think>42"
        ref = "step one. step two"
        self.assertAlmostEqual(step_correctness(pred, ref), 1.0)

    def test_reward_model_scorer_dense(self):
        class DummyTok:
            sep_token_id = 0

            def __call__(self, q, r, return_tensors=None):
                ids = list(range(1, len(q) + 1)) + [0] + list(range(1, len(r) + 1))
                return {"input_ids": torch.tensor([ids])}

            def encode(self, text, add_special_tokens=False):
                return list(range(len(text)))

        class DummyOut:
            def __init__(self, logits):
                self.logits = logits

        class DummyModel:
            def eval(self):
                pass

            def __call__(self, **kwargs):
                L = kwargs["input_ids"].size(1)
                logits = torch.arange(1, L + 1, dtype=torch.float).view(1, L, 1)
                return DummyOut(logits)

        with mock.patch.object(reward_utils, "AutoTokenizer", create=True) as at:
            with mock.patch.object(reward_utils, "AutoModelForSequenceClassification", create=True) as am:
                at.from_pretrained.return_value = DummyTok()
                am.from_pretrained.return_value = DummyModel()
                scorer = RewardModelScorer("dummy")
                dense = scorer.score("a", "bc", dense=True)
                self.assertEqual(dense.numel(), 2)
                self.assertLess(dense[0], dense[1])

if __name__ == "__main__":
    unittest.main()
