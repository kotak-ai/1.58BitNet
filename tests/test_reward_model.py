import unittest
import torch
from reward_model import RewardModel

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

if __name__ == '__main__':
    unittest.main()
