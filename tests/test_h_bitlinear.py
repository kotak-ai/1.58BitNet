import torch
import unittest
from h_bitlinear import HBitLinear, convert_linear_to_hbitlinear
import torch.nn as nn

class HBitLinearTest(unittest.TestCase):
    def test_forward_shape(self):
        layer = HBitLinear(4, 2, bias=False)
        inp = torch.randn(3, 4)
        out = layer(inp)
        self.assertEqual(out.shape, (3, 2))

    def test_numerical_close(self):
        torch.manual_seed(0)
        lin = nn.Linear(3, 2)
        x = torch.randn(4, 3)
        ref = lin(x)
        hlin = convert_linear_to_hbitlinear(lin)
        out = hlin(x)
        rel_err = torch.mean(torch.abs(out - ref)) / torch.mean(torch.abs(ref))
        self.assertLess(rel_err.item(), 0.75)

if __name__ == '__main__':
    unittest.main()
