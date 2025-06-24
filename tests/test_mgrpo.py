import torch
import unittest
import random
from grpo import GRPOTrainer, MultiLayerGRPOTrainer
from grpo_train import parse_guiding_prompts, dynamic_verifier
from grpo_data import construct_second_pass_input


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
        class RecordingModel(DummyModel):
            def generate(self, inp, max_length, do_sample=True):
                B = inp.size(0)
                L = max_length - inp.size(1)
                gen = torch.full((B, L), 4, dtype=torch.long)
                return torch.cat([inp, gen], dim=1)

        model = RecordingModel()
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

        def layer2_step(q, r, l, rewards, opt, advantages=None):
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

        expected_inp, _ = construct_second_pass_input(
            tok,
            queries[0],
            responses[0, 0],
            trainer.guidance_tokens[0],
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
        self.assertLessEqual(len(texts), 2)

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

        def layer2_step(q, r, l, rew, opt, advantages=None):
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

        def layer2_step(q, r, l, rewards, opt, advantages=None):
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

    def test_dynamic_verifier_rejects_incorrect(self):
        class WrongModel(DummyModel):
            def generate(self, inp, max_length, do_sample=True):
                B = inp.size(0)
                L = max_length - inp.size(1)
                gen = torch.full((B, L), 2, dtype=torch.long)
                return torch.cat([inp, gen], dim=1)

        tok = DummyTokenizer()
        model = WrongModel()
        ref = DummyModel()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="fix",
            verifier=dynamic_verifier,
        )
        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        called = {"ok": False}

        def layer2_step(*args, **kwargs):
            called["ok"] = True
            return torch.tensor(0.0)

        trainer.layer2.step = layer2_step

        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), 2, dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        _, rate = trainer.train_batch(
            queries,
            ql,
            responses,
            lengths,
            rewards,
            optim,
            references=["3"],
        )
        self.assertEqual(rate, 0.0)
        self.assertFalse(called["ok"])

    def test_dynamic_verifier_accepts_correct(self):
        class CorrectModel(DummyModel):
            def generate(self, inp, max_length, do_sample=True):
                B = inp.size(0)
                L = max_length - inp.size(1)
                gen = torch.full((B, L), 3, dtype=torch.long)
                return torch.cat([inp, gen], dim=1)

        tok = DummyTokenizer()
        model = CorrectModel()
        ref = DummyModel()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="fix",
            verifier=dynamic_verifier,
        )
        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        called = {"ok": False}

        def layer2_step(*args, **kwargs):
            called["ok"] = True
            return torch.tensor(0.0)

        trainer.layer2.step = layer2_step

        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), 2, dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        _, rate = trainer.train_batch(
            queries,
            ql,
            responses,
            lengths,
            rewards,
            optim,
            references=["3"],
        )
        self.assertEqual(rate, 1.0)
        self.assertTrue(called["ok"])

    def test_deterministic_corrections(self):
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
            guiding_prompt="guide",
            second_max_length=2,
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        captured = {}

        def layer2_step(q, r, l, rewards, opt, advantages=None):
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

        trainer.correction_buffer.clear()

        torch.manual_seed(42)
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        second = captured["tokens"].clone()

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.size(-1), trainer.second_max_length)

    def test_random_guiding_prompt_selection(self):
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
        prompts = ["fix", "check"]
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt=prompts,
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        trainer.layer2.step = lambda *args, **kwargs: torch.tensor(0.0)
        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), 2, dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        random.seed(0)
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        call1 = model.calls[-1][0]

        random.seed(1)
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        call2 = model.calls[-1][0]

        enc = [
            torch.tensor(tok.encode(p, add_special_tokens=False), dtype=torch.long)
            for p in prompts
        ]

        def identify(call):
            for e, name in zip(enc, prompts):
                for i in range(call.size(0) - e.numel() + 1):
                    if torch.equal(call[i : i + e.numel()], e):
                        return name
            return None

        # at least ensure different prompts are chosen
        self.assertNotEqual(call1.tolist(), call2.tolist())

    def test_random_guiding_prompt_selection_file(self):
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
        with open("file_prompts.txt", "w", encoding="utf-8") as f:
            f.write("fix\ncheck\n")
        prompts = parse_guiding_prompts("file_prompts.txt")
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt=prompts,
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        trainer.layer2.step = lambda *args, **kwargs: torch.tensor(0.0)
        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), 2, dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        random.seed(0)
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        call1 = model.calls[-1][0]

        random.seed(1)
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        call2 = model.calls[-1][0]

        self.assertNotEqual(call1.tolist(), call2.tolist())

    def test_guiding_prompt_probabilities(self):
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
        prompts = ["fix", "check"]
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt=prompts,
            prompt_probs=[1.0, 0.0],
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        trainer.layer2.step = lambda *args, **kwargs: torch.tensor(0.0)
        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), 2, dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        call = model.calls[-1][0]
        enc = torch.tensor(tok.encode("fix", add_special_tokens=False), dtype=torch.long)
        self.assertTrue(any(torch.equal(call[i : i + enc.numel()], enc) for i in range(call.size(0) - enc.numel() + 1)))

    def test_guiding_prompt_schedule(self):
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
        prompts = ["fix", "check"]
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt=prompts,
            prompt_schedule=[1, 0],
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        trainer.layer2.step = lambda *args, **kwargs: torch.tensor(0.0)
        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), 2, dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        first_call = model.calls[-1][0]
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        second_call = model.calls[-1][0]

        expected_first, _ = construct_second_pass_input(
            tok,
            queries[0, : ql[0]],
            responses[0, 0, : lengths[0, 0]],
            torch.tensor(tok.encode(prompts[1], add_special_tokens=False), dtype=torch.long),
        )
        expected_second, _ = construct_second_pass_input(
            tok,
            queries[0, : ql[0]],
            responses[0, 0, : lengths[0, 0]],
            torch.tensor(tok.encode(prompts[0], add_special_tokens=False), dtype=torch.long),
        )

        self.assertTrue(torch.equal(first_call[: expected_first.numel()], expected_first))
        self.assertTrue(torch.equal(second_call[: expected_second.numel()], expected_second))

    def test_augmentation_size_multiple_corrections(self):
        class RecordingModel(DummyModel):
            def generate(self, inp, max_length, do_sample=True):
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
            augmentation_size=2,
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        captured = {}

        def layer2_step(q, r, l, rewards, opt, advantages=None):
            captured["shapes"] = (r.shape, l.shape, rewards.shape, advantages.shape)
            captured["rate"] = advantages.size(0)
            return torch.tensor(0.0)

        trainer.layer2.step = layer2_step

        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        loss, rate = trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        self.assertEqual(rate, 1.0)
        self.assertEqual(captured["shapes"], ((2, 1, trainer.second_max_length), (2, 1), (2, 1), (2, 1)))

    def test_augmentation_size_updates_model(self):
        tok = DummyTokenizer()
        model = DummyModel()
        ref = DummyModel()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="fix",
            augmentation_size=2,
            second_max_length=2,
        )

        # disable the first layer for speed
        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
        responses = torch.tensor([[[4, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)

        captured = {}
        orig_step = trainer.layer2.step

        def wrapped_step(q, r, l, rewards_, opt, advantages=None):
            captured["shape"] = r.shape
            before = [p.clone() for p in trainer.layer2.model.parameters()]
            loss = orig_step(q, r, l, rewards_, opt, advantages)
            for p1, p2 in zip(trainer.layer2.model.parameters(), trainer.layer2.old_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
            changed = any(not torch.allclose(b, a) for b, a in zip(before, trainer.layer2.model.parameters()))
            captured["changed"] = changed
            return loss

        trainer.layer2.step = wrapped_step

        init_params = [p.clone() for p in model.parameters()]

        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)

        self.assertEqual(captured["shape"], (2, 1, trainer.second_max_length))
        self.assertTrue(captured["changed"])
        overall_changed = any(
            not torch.allclose(p, q) for p, q in zip(model.parameters(), init_params)
        )
        self.assertTrue(overall_changed)

    def test_correction_buffer_reused(self):
        class RecordingModel(DummyModel):
            def generate(self, inp, max_length, do_sample=True):
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
            second_max_length=2,
        )

        trainer.layer1.step = lambda *args, **kwargs: torch.tensor(0.0)
        captured = {}

        def layer2_step(q, r, l, rewards_, opt, advantages=None):
            captured.setdefault("sizes", []).append(r.size(0))
            return torch.tensor(0.0)

        trainer.layer2.step = layer2_step

        queries = torch.tensor([[2, 3]], dtype=torch.long)
        ql = torch.full((1,), queries.size(1), dtype=torch.long)
        responses = torch.tensor([[[1, 5]]], dtype=torch.long)
        lengths = torch.tensor([[2]], dtype=torch.long)
        rewards = torch.tensor([[0.0]], dtype=torch.float)
        optim = torch.optim.SGD(model.parameters(), lr=0.0)

        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)
        trainer.train_batch(queries, ql, responses, lengths, rewards, optim)

        self.assertEqual(captured["sizes"][0], 1)
        self.assertEqual(captured["sizes"][1], 2)

if __name__ == '__main__':
    unittest.main()
