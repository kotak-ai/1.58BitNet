import os
import json
import shutil
import tempfile
import torch
import unittest

from quantization_utils import pack_quantized_tensor, unpack_quantized_tensor, quantize_tensor
import llama_model

class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()
    def save_pretrained(self, path):
        with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
            json.dump({}, f)

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

class PackUtilsTest(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        torch.manual_seed(0)
        t = torch.randint(-1, 2, (10, 7), dtype=torch.int8)
        packed, shape = pack_quantized_tensor(t)
        out = unpack_quantized_tensor(packed, shape)
        self.assertTrue(torch.equal(t, out))

    def test_model_save_load_roundtrip(self):
        tmp = tempfile.mkdtemp()
        try:
            orig_LlamaConfig = llama_model.LlamaConfig
            orig_Tokenizer = llama_model.AutoTokenizer
            orig_save_file = llama_model.save_file
            orig_load_file = llama_model.load_file
            import quantized_model_io as qio
            orig_qsave = qio.save_file
            orig_qload = qio.load_file
            llama_model.LlamaConfig = DummyConfig
            llama_model.AutoTokenizer = DummyTokenizer
            llama_model.save_file = torch.save
            llama_model.load_file = torch.load
            qio.save_file = torch.save
            qio.load_file = torch.load

            config = DummyConfig()
            model = llama_model.LlamaModel(config)
            model.save_pretrained(tmp)
            loaded = llama_model.LlamaModel.load_pretrained(tmp)
            for name, param in model.state_dict().items():
                loaded_param = loaded.state_dict()[name]
                if torch.is_floating_point(param) and param.numel() > 1:
                    q = quantize_tensor(param)
                    self.assertTrue(torch.equal(loaded_param, q.float()))
                else:
                    self.assertTrue(torch.equal(loaded_param, param))
        finally:
            shutil.rmtree(tmp)
            llama_model.LlamaConfig = orig_LlamaConfig
            llama_model.AutoTokenizer = orig_Tokenizer
            llama_model.save_file = orig_save_file
            llama_model.load_file = orig_load_file
            qio.save_file = orig_qsave
            qio.load_file = orig_qload

if __name__ == '__main__':
    unittest.main()
