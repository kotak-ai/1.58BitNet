import unittest
import torch
from reward_model import RewardModel, load_reward_models
from unittest import mock

class DummyTokenizer:
    vocab_size = 10
    sep_token_id = 9
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 10 for c in text]

class RewardModelTest(unittest.TestCase):
    def test_score_type(self):
        tok = DummyTokenizer()
        model = RewardModel(vocab_size=tok.vocab_size, tokenizer=tok)
        val = model.score("hi", "there")
        self.assertIsInstance(val, float)

    def test_dense_score(self):
        tok = DummyTokenizer()
        model = RewardModel(vocab_size=tok.vocab_size, tokenizer=tok)
        seq = model.score("hi", "abc", dense=True)
        self.assertEqual(seq.numel(), 3)
        self.assertIsInstance(seq, torch.Tensor)

    def test_load_reward_models_weighted_sum(self):
        tok = DummyTokenizer()

        class DummyRM:
            def __init__(self, val):
                self.val = val

            def score(self, query: str, resp: str) -> float:
                return self.val

        with mock.patch.object(RewardModel, "load", side_effect=[DummyRM(0.2), DummyRM(0.8)]):
            fn = load_reward_models(["a.pt", "b.pt"], tok, weights=[1, 3])
        val = fn("gen", "ref", "q")
        self.assertAlmostEqual(val, (0.2 * 1 + 0.8 * 3) / 4)

if __name__ == '__main__':
    unittest.main()
