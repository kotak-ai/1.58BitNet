import torch
import unittest
from quantization_utils import act_quant_4bit, weight_quant, quantize_tensor_1_58bit

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

    def test_quantize_tensor_1_58bit_values(self):
        torch.manual_seed(0)
        x = torch.randn(5, 5)
        q, scale = quantize_tensor_1_58bit(x)
        vals = set(q.unique().tolist())
        self.assertTrue(vals.issubset({-1, 0, 1}))

    def test_quantize_tensor_1_58bit_reconstruction(self):
        torch.manual_seed(1)
        x = torch.randn(8, 8)
        q, scale = quantize_tensor_1_58bit(x)
        recon = q.float() * scale
        rel_err = torch.mean(torch.abs(x - recon)) / torch.mean(torch.abs(x))
        self.assertLess(rel_err.item(), 0.5)

if __name__ == '__main__':
    unittest.main()
