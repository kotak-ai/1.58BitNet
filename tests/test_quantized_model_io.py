import os
import shutil
import tempfile
import torch
import unittest

import quantized_model_io as qio
from quantization_utils import quantize_tensor

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)

class QuantizedIOTest(unittest.TestCase):
    def test_save_load_roundtrip(self):
        tmp = tempfile.mkdtemp()
        try:
            orig_save = qio.save_file
            orig_load = qio.load_file
            qio.save_file = torch.save
            qio.load_file = torch.load

            model = DummyModel()
            qio.save_quantized_model(model, tmp)
            loaded = DummyModel()
            qio.load_quantized_model(loaded, tmp)

            for name, param in model.state_dict().items():
                q = quantize_tensor(param)
                self.assertTrue(torch.equal(loaded.state_dict()[name], q.float()))
        finally:
            shutil.rmtree(tmp)
            qio.save_file = orig_save
            qio.load_file = orig_load

if __name__ == '__main__':
    unittest.main()
