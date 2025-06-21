import math
import torch
import unittest
import quantized_model_io as qio

class SizeReportingTest(unittest.TestCase):
    def test_total_size_estimation(self):
        model = torch.nn.Linear(4, 3)
        state = {"model.lin.weight": model.weight, "model.lin.bias": model.bias}
        qsd, wm, meta = qio.quantize_state_dict(state)
        expected = 0
        for t in state.values():
            expected += math.ceil(t.numel() * 1.58 / 8) + t.dim() * 4
        self.assertEqual(meta["total_size"], expected)

if __name__ == "__main__":
    unittest.main()
