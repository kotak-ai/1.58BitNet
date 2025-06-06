import torch
import unittest
from grpo import GRPOTrainer, MultiLayerGRPOTrainer

class DummyModel(torch.nn.Module):
    def __init__(self, vocab=10):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.linear = torch.nn.Linear(8, vocab)
    def forward(self, x):
        emb = self.embed(x)
        return self.linear(emb)

def simple_verifier(resp: torch.Tensor) -> bool:
    return int(resp[-1]) % 2 == 0

class GRPOTest(unittest.TestCase):
    def test_single_step(self):
        model = DummyModel()
        ref = DummyModel()
        trainer = GRPOTrainer(model, ref)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        queries = torch.randint(0, 10, (2, 3))
        responses = torch.randint(0, 10, (2, 2, 4))
        lengths = torch.tensor([[4,4],[4,4]])
        rewards = torch.tensor([[1.0, 0.0],[0.0,1.0]])
        loss = trainer.step(queries, responses, lengths, rewards, optim)
        self.assertIsInstance(loss.item(), float)

    def test_multilayer(self):
        model = DummyModel()
        ref = DummyModel()
        trainer = MultiLayerGRPOTrainer(model, ref, simple_verifier)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        queries = torch.randint(0, 10, (1, 3))
        responses = torch.randint(0, 10, (1, 2, 4))
        lengths = torch.tensor([[4,4]])
        rewards = torch.tensor([[0.0,1.0]])
        loss, rate = trainer.train_batch(queries, responses, lengths, rewards, optim)
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

if __name__ == '__main__':
    unittest.main()
