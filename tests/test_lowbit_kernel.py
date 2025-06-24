import unittest
import torch
from quantization_utils import gemm_lowbit, pack_ternary

class LowBitKernelTest(unittest.TestCase):
    def test_matches_matmul(self):
        torch.manual_seed(0)
        x = torch.randint(-2, 3, (8, 6), dtype=torch.int8)
        w = torch.randint(-2, 3, (4, 6), dtype=torch.int8)
        out = gemm_lowbit(x, w)
        ref = (x.to(torch.float32) @ w.to(torch.float32).t())
        self.assertTrue(torch.equal(out, ref))

    def test_packed_matches_unpacked(self):
        torch.manual_seed(0)
        x = torch.randint(-8, 8, (6, 5), dtype=torch.int8)
        w = torch.randint(-1, 2, (4, 5), dtype=torch.int8)
        packed = pack_ternary(w)
        out_packed = gemm_lowbit(x, packed, w.shape)
        out_ref = gemm_lowbit(x, w)
        self.assertTrue(torch.equal(out_packed, out_ref))

    def test_packed_cuda_mps(self):
        for device in ("cuda", "mps"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            if device == "mps" and not torch.backends.mps.is_available():
                continue
            torch.manual_seed(0)
            x = torch.randint(-8, 8, (6, 5), dtype=torch.int8, device=device)
            w = torch.randint(-1, 2, (4, 5), dtype=torch.int8, device=device)
            packed = pack_ternary(w)
            out_packed = gemm_lowbit(x, packed, w.shape)
            out_ref = gemm_lowbit(x, w)
            self.assertTrue(torch.equal(out_packed.cpu(), out_ref.cpu()))

if __name__ == "__main__":
    unittest.main()
