import os
import torch
import unittest
from training_utils import save_checkpoint, load_checkpoint

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1))

class CheckpointTest(unittest.TestCase):
    def test_round_trip(self):
        model = DummyModel()
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        orig = model.w.clone()
        optim.step()
        save_checkpoint(model, optim, 5, "ckpt.pt")
        new_model = DummyModel()
        new_optim = torch.optim.SGD(new_model.parameters(), lr=0.1)
        step = load_checkpoint(new_model, new_optim, "ckpt.pt")
        self.assertEqual(step, 5)
        self.assertTrue(torch.allclose(model.w, new_model.w))
        os.remove("ckpt.pt")

if __name__ == "__main__":
    unittest.main()
