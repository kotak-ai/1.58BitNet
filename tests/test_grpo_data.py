import unittest
import torch

from grpo_data import build_grpo_batch

class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 10 + 2 for c in text.lower()]

    def decode(self, tokens):
        return " ".join(str(t) for t in tokens)

class DummyModel(torch.nn.Module):
    def generate(self, inp, max_length, do_sample=True):
        B = inp.size(0)
        L = max_length - inp.size(1)
        return torch.randint(2, 9, (B, inp.size(1) + L))

class GRPODataTest(unittest.TestCase):
    def test_batch_shapes(self):
        data = [{'query': 'hi', 'answer': 'hello'}]
        tok = DummyTokenizer()
        model = DummyModel()
        q, r, l, rew = build_grpo_batch(data, tok, model, group_size=2, max_length=3)
        self.assertEqual(q.size(0), 1)
        self.assertEqual(r.shape[:2], (1,2))
        self.assertEqual(l.shape, (1,2))
        self.assertEqual(rew.shape, (1,2))

if __name__ == '__main__':
    unittest.main()
