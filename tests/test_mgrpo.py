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
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
        responses = torch.randint(0, 10, (1, 2, 4))
        lengths = torch.tensor([[4,4]])
        rewards = torch.tensor([[0.0,1.0]])
        loss, rate = trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
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
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
        responses = torch.tensor([[[4, 5, 6, 7]]], dtype=torch.long)
        lengths = torch.tensor([[4]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)

        loss, rate = trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        self.assertIsInstance(loss.item(), float)
        self.assertEqual(rate, 1.0)

        expected_inp = torch.cat(
            [
                queries[0],
                responses[0, 0],
                torch.tensor([trainer.sep_id], dtype=torch.long),
                trainer.guidance_tokens,
            ],
            dim=0,
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
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
        responses = torch.randint(0, 10, (1, 2, 4))
        lengths = torch.tensor([[4, 4]])
        rewards = torch.tensor([[0.0, 1.0]])
        loss, rate, texts = trainer.train_batch(
            queries,
            ql,
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
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
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
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        self.assertTrue(called["ok"])
        changed = any(
            not torch.allclose(p, q) for p, q in zip(model.parameters(), init_params)
        )
        self.assertTrue(changed)

    def test_verifier_allows_improvement(self):
        class RecordingModel(DummyModel):
            def generate(self, inp, max_length, do_sample=True):
                B = inp.size(0)
                L = max_length - inp.size(1)
                gen = torch.full((B, L), 3, dtype=torch.long)
                return torch.cat([inp, gen], dim=1)

        tok = DummyTokenizer()
        model = RecordingModel()
        ref = DummyModel()

        def reward_fn(text: str) -> float:
            last = int(text.split()[-1])
            return float(last) / 10.0

        def verifier(new: float, old: float) -> bool:
            return (new - old) > 0.15

        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            reward_fn,
            tok,
            guiding_prompt="fix",
            verifier=verifier,
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        captured = {}

        def layer2_step(q, r, l, rewards, opt):
            captured["rewards"] = rewards.clone()
            return torch.tensor(0.0)

        trainer.layer2.step = layer2_step

        optim = torch.optim.SGD(model.parameters(), lr=0.0)
        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.1]], dtype=torch.float)

        _, rate = trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        self.assertEqual(rate, 1.0)
        self.assertTrue(torch.allclose(captured["rewards"], torch.tensor([[0.3]])))

    def test_deterministic_corrections(self):
        class RecordingModel(DummyModel):
            def __init__(self):
                super().__init__()
                self.calls = []

            def generate(self, inp, max_length, do_sample=True):
                self.calls.append(inp.clone())
                B = inp.size(0)
                L = max_length - inp.size(1)
                gen = torch.randint(0, self.linear.out_features, (B, L))
                return torch.cat([inp, gen], dim=1)

        tok = DummyTokenizer()
        model = RecordingModel()
        ref = DummyModel()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="guide",
            second_max_length=2,
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        captured = {}

        def layer2_step(q, r, l, rewards, opt):
            captured["tokens"] = r.clone()
            return torch.tensor(0.0)

        trainer.layer2.step = layer2_step

        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        torch.manual_seed(42)
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        first = captured["tokens"].clone()

        torch.manual_seed(42)
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        second = captured["tokens"].clone()

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.size(-1), trainer.second_max_length)

if __name__ == '__main__':
    unittest.main()
