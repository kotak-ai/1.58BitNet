import os
import unittest
from unittest import mock
import torch
import reward_train

class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 5 for c in text]
    vocab_size = 10

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.called = 0
    def contrastive_loss(self, q, p, n):
        self.called += 1
        return self.w.sum()
    def save(self, path):
        pass

class RewardTrainTest(unittest.TestCase):
    def test_load_pair_dataset(self):
        path = os.path.join(os.path.dirname(__file__), 'data', 'contrastive_pairs.jsonl')
        data = reward_train.load_pair_dataset(path)
        self.assertEqual(len(data), 2)
        self.assertEqual(set(data[0].keys()), {'query', 'positive', 'negative'})

    def test_contrastive_training_used(self):
        path = os.path.join(os.path.dirname(__file__), 'data', 'contrastive_pairs.jsonl')
        pairs = reward_train.load_pair_dataset(path)
        tok = DummyTokenizer()
        model = DummyModel()
        reward_train.train(model, tok, dataset=None, pairs=pairs, epochs=1, batch_size=1)
        self.assertGreaterEqual(model.called, 1)

if __name__ == '__main__':
    unittest.main()
