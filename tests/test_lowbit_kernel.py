import unittest
import torch
from quantization_utils import gemm_lowbit

class LowBitKernelTest(unittest.TestCase):
    def test_matches_matmul(self):
        torch.manual_seed(0)
        x = torch.randint(-2, 3, (8, 6), dtype=torch.int8)
        w = torch.randint(-2, 3, (4, 6), dtype=torch.int8)
        out = gemm_lowbit(x, w)
        ref = (x.to(torch.float32) @ w.to(torch.float32).t())
        self.assertTrue(torch.equal(out, ref))

if __name__ == "__main__":
    unittest.main()
