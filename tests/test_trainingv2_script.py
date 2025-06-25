import os
import shutil
import tempfile
import unittest
import torch
from unittest import mock


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 10

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 5 + 2 for c in text]

class DummyConfig:
    def __init__(self, **kwargs):
        self.vocab_size = 10
        self.hidden_size = 4
        self.num_hidden_layers = 1
        self.num_attention_heads = 1
        self.intermediate_size = 4
        self.pretraining_tp = 1
        self.max_position_embeddings = 8
        self.__dict__.update(kwargs)

    @classmethod
    def from_pretrained(cls, path):
        return cls()

class DummyModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.config = config

    def forward(self, x, attention_mask=None, cos=None, sin=None):
        emb = self.embed(x)
        return self.lm_head(emb)

    def generate(self, inp, max_length, do_sample=True):
        B, L = inp.size()
        gen = torch.zeros(B, max_length - L, dtype=torch.long)
        return torch.cat([inp, gen], dim=1)

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)

import sys
import types

transformers = types.SimpleNamespace(AutoTokenizer=DummyTokenizer, LlamaConfig=DummyConfig)
sys.modules.setdefault('transformers', transformers)
sys.modules.setdefault('safetensors.torch', types.SimpleNamespace(load_file=lambda *a, **k: None))
import trainingv2

class ScriptCheckpointTest(unittest.TestCase):
    def test_save_and_resume(self):
        tmp = tempfile.mkdtemp()
        try:
            data_path = os.path.join(tmp, "data.jsonl")
            with open(data_path, "w") as f:
                f.write('{"text": "ab"}\n{"text": "bc"}\n')
            ckpt = os.path.join(tmp, "ckpt.pt")
            parser = trainingv2.get_arg_parser()
            args = parser.parse_args([
                '--dataset', data_path,
                '--model_path', tmp,
                '--output_dir', tmp,
                '--iters', '2',
                '--batch_size', '1',
                '--resume', ckpt,
                '--save_interval', '1',
            ])
            orig_iter = trainingv2.iterate_batches
            def _iter_batches(*a, **k):
                for b in orig_iter(*a, **k):
                    yield tuple(t.long() for t in b)

            with mock.patch.object(trainingv2, 'LlamaConfig', DummyConfig), \
                 mock.patch.object(trainingv2, 'AutoTokenizer', DummyTokenizer), \
                 mock.patch.object(trainingv2, 'LlamaModel', DummyModel), \
                 mock.patch.object(trainingv2, 'iterate_batches', _iter_batches):
                trainingv2.run(args)
            self.assertEqual(torch.load(ckpt)['step'], 2)

            args2 = parser.parse_args([
                '--dataset', data_path,
                '--model_path', tmp,
                '--output_dir', tmp,
                '--iters', '4',
                '--batch_size', '1',
                '--resume', ckpt,
                '--save_interval', '1',
            ])
            with mock.patch.object(trainingv2, 'LlamaConfig', DummyConfig), \
                 mock.patch.object(trainingv2, 'AutoTokenizer', DummyTokenizer), \
                 mock.patch.object(trainingv2, 'LlamaModel', DummyModel), \
                 mock.patch.object(trainingv2, 'iterate_batches', _iter_batches):
                trainingv2.run(args2)
            self.assertEqual(torch.load(ckpt)['step'], 4)
        finally:
            shutil.rmtree(tmp)


class EvalHookTest(unittest.TestCase):
    def test_run_eval_invokes_evaluation(self):
        tmp = tempfile.mkdtemp()
        try:
            data_path = os.path.join(tmp, "data.jsonl")
            with open(data_path, "w") as f:
                f.write('{"text": "ab"}\n')
            parser = trainingv2.get_arg_parser()
            args = parser.parse_args([
                '--dataset', data_path,
                '--model_path', tmp,
                '--output_dir', tmp,
                '--iters', '1',
                '--batch_size', '1',
                '--run_eval',
                '--eval_dataset', data_path,
            ])
            def _iter_batches(*a, **k):
                yield (torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([1]))
            with mock.patch.object(trainingv2, 'LlamaConfig', DummyConfig), \
                 mock.patch.object(trainingv2, 'AutoTokenizer', DummyTokenizer), \
                 mock.patch.object(trainingv2, 'LlamaModel', DummyModel), \
                 mock.patch.object(trainingv2, 'iterate_batches', _iter_batches), \
                 mock.patch('trainingv2.evaluation.run') as eval_run:
                trainingv2.run(args)
                eval_run.assert_called_once()
        finally:
            shutil.rmtree(tmp)

if __name__ == '__main__':
    unittest.main()
