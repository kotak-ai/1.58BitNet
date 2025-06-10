import unittest
from simple_reward_model import (
    SimpleTokenizer,
    SimpleRewardModel,
    train_reward_model,
    score,
)


class SimpleRewardModelTest(unittest.TestCase):
    def test_training_improves_score(self):
        data = [
            {"query": "hi", "answer": "there", "label": 1.0},
            {"query": "hi", "answer": "wrong", "label": 0.0},
        ]
        tok = SimpleTokenizer()
        for item in data:
            tok.encode(item["query"])
            tok.encode(item["answer"])
        model = SimpleRewardModel(tok.vocab_size)
        train_reward_model(model, tok, data, epochs=50, lr=0.1)
        pos = score(model, tok, "hi", "there")
        neg = score(model, tok, "hi", "wrong")
        self.assertGreater(pos, neg)


if __name__ == "__main__":
    unittest.main()
