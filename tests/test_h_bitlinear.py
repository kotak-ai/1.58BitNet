import torch
import unittest
from h_bitlinear import HBitLinear

class HBitLinearTest(unittest.TestCase):
    def test_forward_shape(self):
        layer = HBitLinear(4, 2, bias=False)
        inp = torch.randn(3, 4)
        out = layer(inp)
        self.assertEqual(out.shape, (3, 2))

if __name__ == '__main__':
    unittest.main()
