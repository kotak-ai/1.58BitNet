import torch
import unittest
from grpo import GRPOTrainer, MultiLayerGRPOTrainer


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 10 + 2 for c in text.lower()]

    def decode(self, tokens):
        return " ".join(str(int(t)) for t in tokens)

class DummyModel(torch.nn.Module):
    def __init__(self, vocab=10):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.linear = torch.nn.Linear(8, vocab)
    def forward(self, x):
        emb = self.embed(x)
        return self.linear(emb)

    def generate(self, inp, max_length, do_sample=True):
        B = inp.size(0)
        L = max_length - inp.size(1)
        gen = torch.randint(0, self.linear.out_features, (B, L))
        return torch.cat([inp, gen], dim=1)

def simple_reward(text: str) -> float:
    last = int(text.split()[-1])
    return float(last % 2 == 0)

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
        tok = DummyTokenizer()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="fix",
        )
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        queries = torch.randint(0, 10, (1, 3))
        responses = torch.randint(0, 10, (1, 2, 4))
        lengths = torch.tensor([[4,4]])
        rewards = torch.tensor([[0.0,1.0]])
        loss, rate = trainer.train_batch(queries, responses, lengths, rewards, optim)
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_generation_and_correction(self):
        class RecordingModel(DummyModel):
            def __init__(self):
                super().__init__()
                self.calls = []

            def generate(self, inp, max_length, do_sample=True):
                self.calls.append(inp.clone())
                B = inp.size(0)
                L = max_length - inp.size(1)
                gen = torch.full((B, L), 4, dtype=torch.long)
                return torch.cat([inp, gen], dim=1)

        tok = DummyTokenizer()
        model = RecordingModel()
        ref = DummyModel()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="fix",
        )

        # replace step methods to avoid heavy computation and capture rewards
        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        captured = {}

        def layer2_step(q, r, l, rewards, opt):
            captured["rewards"] = rewards.clone()
            return torch.tensor(0.0)

        trainer.layer2.step = layer2_step

        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        queries = torch.tensor([[2, 3]], dtype=torch.long)
        responses = torch.tensor([[[4, 5, 6, 7]]], dtype=torch.long)
        lengths = torch.tensor([[4]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)

        loss, rate = trainer.train_batch(queries, responses, lengths, rewards, optim)
        self.assertIsInstance(loss.item(), float)
        self.assertEqual(rate, 1.0)

        expected_inp = torch.cat(
            [trainer.guidance_tokens, queries[0], responses[0, 0]], dim=0
        )
        self.assertTrue(torch.equal(model.calls[0][0], expected_inp))
        self.assertTrue(torch.equal(captured["rewards"], torch.tensor([[1.0]])))

    def test_log_texts(self):
        model = DummyModel()
        ref = DummyModel()
        tok = DummyTokenizer()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="fix",
        )
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        queries = torch.randint(0, 10, (1, 3))
        responses = torch.randint(0, 10, (1, 2, 4))
        lengths = torch.tensor([[4, 4]])
        rewards = torch.tensor([[0.0, 1.0]])
        loss, rate, texts = trainer.train_batch(
            queries,
            responses,
            lengths,
            rewards,
            optim,
            log_texts=2,
        )
        self.assertEqual(len(texts), 2)

    def test_layer2_old_model_updated(self):
        model = DummyModel()
        ref = DummyModel()
        tok = DummyTokenizer()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="fix",
        )
        optim = torch.optim.SGD(model.parameters(), lr=0.01)
        queries = torch.randint(0, 10, (1, 3))
        responses = torch.randint(0, 10, (1, 2, 4))
        lengths = torch.tensor([[4, 4]])
        rewards = torch.tensor([[0.0, 1.0]])

        init_params = [p.clone() for p in model.parameters()]
        called = {"ok": False}

        def layer2_step(q, r, l, rew, opt):
            for p1, p2 in zip(trainer.layer2.model.parameters(), trainer.layer2.old_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
            called["ok"] = True
            return torch.tensor(0.0)

        trainer.layer2.step = layer2_step
        trainer.train_batch(queries, responses, lengths, rewards, optim)
        self.assertTrue(called["ok"])
        changed = any(
            not torch.allclose(p, q) for p, q in zip(model.parameters(), init_params)
        )
        self.assertTrue(changed)

if __name__ == '__main__':
    unittest.main()
