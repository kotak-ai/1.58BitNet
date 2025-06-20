import os
import torch
import unittest
from training_utils import save_checkpoint, load_checkpoint

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1))

    def forward(self):
        return self.w

class ResumeBehaviourTest(unittest.TestCase):
    def test_resume_updates(self):
        torch.manual_seed(0)
        model_full = SimpleModel()
        optim_full = torch.optim.SGD(model_full.parameters(), lr=0.1)

        torch.manual_seed(0)
        model_resume = SimpleModel()
        optim_resume = torch.optim.SGD(model_resume.parameters(), lr=0.1)

        # Train full model for 4 steps
        for _ in range(4):
            loss = (model_full() ** 2).sum()
            loss.backward()
            optim_full.step()
            optim_full.zero_grad()

        # Train resume model for 2 steps then save
        for _ in range(2):
            loss = (model_resume() ** 2).sum()
            loss.backward()
            optim_resume.step()
            optim_resume.zero_grad()
        save_checkpoint(model_resume, optim_resume, 2, 'tmp.ckpt')

        # Load and continue for 2 more steps
        cont_model = SimpleModel()
        cont_optim = torch.optim.SGD(cont_model.parameters(), lr=0.1)
        step = load_checkpoint(cont_model, cont_optim, 'tmp.ckpt')
        self.assertEqual(step, 2)
        for _ in range(2,4):
            loss = (cont_model() ** 2).sum()
            loss.backward()
            cont_optim.step()
            cont_optim.zero_grad()

        os.remove('tmp.ckpt')
        self.assertTrue(torch.allclose(model_full.w, cont_model.w))

if __name__ == '__main__':
    unittest.main()
