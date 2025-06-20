import unittest
import torch

from grpo import GRPOTrainer, MultiLayerGRPOTrainer
from grpo_data import pad_sequences


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 5) + 2 for c in text.lower()]

    def decode(self, tokens):
        return " ".join(str(int(t)) for t in tokens)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab=10):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 4)
        self.linear = torch.nn.Linear(4, vocab)

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


class GRPOIntegrationTest(unittest.TestCase):
    def test_training_progress(self):
        torch.manual_seed(4)
        data = [
            {"query": "hi", "answer": "hello"},
            {"query": "bye", "answer": "goodbye"},
        ]
        tok = DummyTokenizer()
        q_tokens = [tok.encode(d["query"]) for d in data]
        a_tokens = [tok.encode(d["answer"]) for d in data]
        pad_id = tok.pad_token_id
        queries = pad_sequences(q_tokens, pad_id)
        q_lens = torch.tensor([len(q) for q in q_tokens], dtype=torch.long)
        max_resp = max(len(a) for a in a_tokens)
        B, G = len(data), 2
        responses = torch.full((B, G, max_resp), pad_id, dtype=torch.long)
        lengths = torch.full((B, G), max_resp, dtype=torch.long)
        rewards = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        for i, ans in enumerate(a_tokens):
            responses[i, 1, :len(ans)] = torch.tensor(ans)
            responses[i, 0, :len(ans)] = torch.randint(2, 7, (len(ans),))
        model = DummyModel()
        ref = DummyModel()
        trainer = GRPOTrainer(model, ref)
        eval_optim = torch.optim.SGD(model.parameters(), lr=0.0)
        train_optim = torch.optim.SGD(model.parameters(), lr=0.05)
        trainer.old_model.load_state_dict(model.state_dict())
        init_loss = trainer.step(queries, responses, lengths, rewards, eval_optim).item()
        trainer.old_model.load_state_dict(model.state_dict())
        init_params = [p.clone() for p in model.parameters()]
        for _ in range(5):
            trainer.step(queries, responses, lengths, rewards, train_optim)
        trainer.old_model.load_state_dict(model.state_dict())
        final_loss = trainer.step(queries, responses, lengths, rewards, eval_optim).item()
        self.assertLess(final_loss, init_loss)
        changed = any(not torch.allclose(p, q) for p, q in zip(model.parameters(), init_params))
        self.assertTrue(changed)

    def test_two_layer_training_loop(self):
        torch.manual_seed(4)
        data = [
            {"query": "hi", "answer": "hello"},
            {"query": "bye", "answer": "goodbye"},
        ]
        tok = DummyTokenizer()
        q_tokens = [tok.encode(d["query"]) for d in data]
        a_tokens = [tok.encode(d["answer"]) for d in data]
        pad_id = tok.pad_token_id
        queries = pad_sequences(q_tokens, pad_id)
        q_lens = torch.tensor([len(q) for q in q_tokens], dtype=torch.long)
        max_resp = max(len(a) for a in a_tokens)
        B, G = len(data), 2
        responses = torch.full((B, G, max_resp), pad_id, dtype=torch.long)
        lengths = torch.full((B, G), max_resp, dtype=torch.long)
        rewards = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        for i, ans in enumerate(a_tokens):
            responses[i, 1, : len(ans)] = torch.tensor(ans)
            responses[i, 0, : len(ans)] = torch.randint(2, 7, (len(ans),))
        model = DummyModel()
        ref = DummyModel()
        trainer = MultiLayerGRPOTrainer(
            model,
            ref,
            simple_reward,
            tok,
            guiding_prompt="fix",
        )
        optim = torch.optim.SGD(model.parameters(), lr=0.05)
        init_params = [p.clone() for p in model.parameters()]
        rates = []
        for _ in range(3):
            _, rate = trainer.train_batch(queries, q_lens, responses, lengths, rewards, optim)
            rates.append(rate)
        changed = any(not torch.allclose(p, q) for p, q in zip(model.parameters(), init_params))
        self.assertTrue(changed)
        for r in rates:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)


if __name__ == "__main__":
    unittest.main()
