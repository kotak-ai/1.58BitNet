import torch
import unittest
from quantization_utils import act_quant_4bit, weight_quant

class QuantizationTest(unittest.TestCase):
    def test_activation_quant_4bit_range(self):
        x = torch.randn(2, 4)
        q = act_quant_4bit(x)
        self.assertTrue(q.max() <= 7)
        self.assertTrue(q.min() >= -8)

    def test_weight_quant_1bit(self):
        w = torch.randn(10, 5)
        q = weight_quant(w)
        self.assertTrue(q.unique().numel() <= 3)

if __name__ == '__main__':
    unittest.main()
