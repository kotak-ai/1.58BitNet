import os
import json
import shutil
import tempfile
import unittest
import torch

import importlib

TRANS_AVAILABLE = importlib.util.find_spec("transformers") is not None

if TRANS_AVAILABLE:
    import inference
    import llama_model
    import quantized_model_io as qio

class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def save_pretrained(self, path):
        with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
            json.dump({}, f)

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        ids = [ord(c) % 10 + 2 for c in text.lower()]
        if return_tensors == 'pt':
            return torch.tensor([ids])
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        return ' '.join(str(int(t)) for t in tokens)

class DummyConfig:
    def __init__(self, **kwargs):
        self.vocab_size = 8
        self.hidden_size = 4
        self.num_hidden_layers = 1
        self.num_attention_heads = 1
        self.intermediate_size = 4
        self.pretraining_tp = 1
        self.max_position_embeddings = 8
        self.layer_norm_eps = 1e-5
        self.bos_token_id = None
        self.eos_token_id = None
        self.__dict__.update(kwargs)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            return cls(**json.load(f))

    def save_pretrained(self, path):
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.__dict__, f)

@unittest.skipUnless(TRANS_AVAILABLE, "transformers not available")
class InferenceIntegrationTest(unittest.TestCase):
    def test_run_with_quantized_model(self):
        tmp = tempfile.mkdtemp()
        try:
            orig_cfg = llama_model.LlamaConfig
            orig_tok = llama_model.AutoTokenizer
            orig_save = llama_model.save_file
            orig_load = llama_model.load_file
            orig_qsave = qio.save_file
            orig_qload = qio.load_file
            orig_itok = inference.AutoTokenizer
            orig_imod = inference.LlamaModel

            llama_model.LlamaConfig = DummyConfig
            llama_model.AutoTokenizer = DummyTokenizer
            llama_model.save_file = torch.save
            llama_model.load_file = torch.load
            qio.save_file = torch.save
            qio.load_file = torch.load
            inference.AutoTokenizer = DummyTokenizer
            inference.LlamaModel = llama_model.LlamaModel

            config = DummyConfig()
            model = llama_model.LlamaModel(config)
            model.save_pretrained(tmp)

            out = inference.run(tmp, ['hi'], max_length=2)
            self.assertIsInstance(out[0], str)
        finally:
            shutil.rmtree(tmp)
            llama_model.LlamaConfig = orig_cfg
            llama_model.AutoTokenizer = orig_tok
            llama_model.save_file = orig_save
            llama_model.load_file = orig_load
            qio.save_file = orig_qsave
            qio.load_file = orig_qload
            inference.AutoTokenizer = orig_itok
            inference.LlamaModel = orig_imod

if __name__ == '__main__':
    unittest.main()
