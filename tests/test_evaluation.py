import unittest
import torch
from evaluation import evaluate_model

class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    _map = {'a':2,'b':3,'x':4,'y':5}
    _rev = {v:k for k,v in _map.items()}
    def encode(self, text, return_tensors=None, add_special_tokens=False):
        ids = [self._map[text]]
        if return_tensors == 'pt':
            return torch.tensor([ids])
        return ids
    def decode(self, tokens):
        return ''.join(self._rev[t] for t in tokens)

class DummyModel(torch.nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping
    def generate(self, inp, max_length, do_sample=False):
        token = inp[0,0].item()
        out_tok = [self.mapping.get(token, 4)]
        return torch.tensor([inp[0].tolist() + out_tok])

class EvalTest(unittest.TestCase):
    def test_compare(self):
        data = [{'query':'a','answer':'x'}, {'query':'b','answer':'y'}]
        tok = DummyTokenizer()
        ce_model = DummyModel({2:4,3:4})
        grpo_model = DummyModel({2:4,3:5})
        ce_score = evaluate_model(ce_model, tok, data, 1)
        grpo_score = evaluate_model(grpo_model, tok, data, 1)
        self.assertLess(ce_score, grpo_score)

if __name__ == '__main__':
    unittest.main()
