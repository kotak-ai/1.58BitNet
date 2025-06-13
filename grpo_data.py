import json
import random
import torch
from typing import List, Dict, Callable
from reward_utils import qa_reward


def f1_reward(generated: str, reference: str, query: str) -> float:
    """Wrapper that ignores ``query`` when computing F1."""
    return qa_reward(generated, reference)


def load_qa_dataset(path: str) -> List[Dict[str, str]]:
    """Load a QA dataset where each record has 'query' and 'answer'."""
    data = []
    if path.endswith(".jsonl"):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                data.append({'query': obj['query'], 'answer': obj['answer']})
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            for obj in json.load(f):
                data.append({'query': obj['query'], 'answer': obj['answer']})
    else:
        raise ValueError('Unsupported dataset format')
    return data


def pad_sequences(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return tensor


def build_grpo_batch(
    samples: List[Dict[str, str]],
    tokenizer,
    model,
    group_size: int,
    max_length: int,
    reward_fn: Callable[[str, str, str], float] = f1_reward,
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """Generate candidate responses and compute rewards."""
    q_tokens = [tokenizer.encode(s['query'], add_special_tokens=False) for s in samples]
    answers = [s['answer'] for s in samples]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    queries = pad_sequences(q_tokens, pad_id)

    B = len(samples)
    responses = []
    lengths = []
    rewards = []
    for i in range(B):
        q = q_tokens[i]
        inp = torch.tensor([q], dtype=torch.long)
        grp_resp = []
        grp_len = []
        grp_rew = []
        for _ in range(group_size):
            out = model.generate(inp, max_length=len(q) + max_length, do_sample=True)
            resp = out[0, len(q):].tolist()
            grp_resp.append(resp)
            grp_len.append(len(resp))
            gen_text = tokenizer.decode(resp)
            grp_rew.append(reward_fn(gen_text, answers[i], samples[i]["query"]))
        responses.append(grp_resp)
        lengths.append(grp_len)
        rewards.append(grp_rew)

    max_resp_len = max(max(l) for l in lengths)
    resp_tensor = torch.full((B, group_size, max_resp_len), pad_id, dtype=torch.long)
    len_tensor = torch.zeros(B, group_size, dtype=torch.long)
    for b in range(B):
        for g in range(group_size):
            seq = responses[b][g]
            resp_tensor[b, g, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            len_tensor[b, g] = len(seq)
    reward_tensor = torch.tensor(rewards, dtype=torch.float)
    return queries, resp_tensor, len_tensor, reward_tensor


def construct_second_pass_input(
    query_tokens: torch.Tensor,
    output_tokens: torch.Tensor,
    guidance_tokens: torch.Tensor,
) -> (torch.Tensor, int):
    """Combine guidance, query, and first-layer output tokens."""
    combined = torch.cat([guidance_tokens.view(-1), query_tokens.view(-1), output_tokens.view(-1)])
    return combined, combined.numel()

