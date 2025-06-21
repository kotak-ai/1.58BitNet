import unittest
import torch
import torch.nn as nn
from llama_model import BitLinear
from quantization_utils import RMSNorm

class BitLinearComparisonTest(unittest.TestCase):
    def test_quantized_close_to_float(self):
        torch.manual_seed(0)
        lin = nn.Linear(4, 3, bias=False)
        bitlin = BitLinear(4, 3, bias=False)
        bitlin.weight.data.copy_(lin.weight.data)
        x = torch.randn(5, 4)

        ref = nn.functional.linear(RMSNorm(x), lin.weight)
        out = bitlin(x)

        rel_err = torch.mean(torch.abs(out - ref)) / torch.mean(torch.abs(ref))
        self.assertLess(rel_err.item(), 0.6)

if __name__ == '__main__':
    unittest.main()
