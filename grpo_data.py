import json
import random
import torch
from typing import List, Dict, Callable, Optional
from reward_utils import qa_reward


def _load_split_dataset(name: str, split: str, path: Optional[str] = None):
    """Return dataset ``split`` from ``path`` or ``name`` with friendly errors."""
    from datasets import load_dataset

    ds_id = path or name
    try:
        return load_dataset(ds_id, split=split)
    except Exception as exc:
        msg = (
            f"Failed to load dataset '{ds_id}'. "
            "Ensure the path is correct and the dataset is available locally "
            "or via the Hugging Face hub."
        )
        raise RuntimeError(msg) from exc


def build_layer1_prompt(query: str, system_prompt: str | None = None) -> str:
    """Return the text prompt for the first GRPO layer."""
    if system_prompt is None:
        system_prompt = (
            "You are an AI assistant. A conversation between User and Assistant.\n"
            "The User asks a question, and the Assistant solves it step-by-step.\n"
            "The Assistant must first output a detailed step-by-step reasoning process enclosed within "
            "<think></think> tags. After the </think> tag, the Assistant must provide the final answer "
            "based on the reasoning."
        )
    parts = []
    parts.append(f"<|im_start|>system\n{system_prompt}\n<|im_end|>")
    parts.append(f"<|im_start|>user\n{query}\n<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)

def build_layer2_prompt(
    query: str,
    answer: str,
    guidance: str,
    system_prompt: str | None = None,
) -> str:
    """Return the text prompt for the second GRPO layer."""
    if system_prompt is None:
        system_prompt = (
            "You are a helpful AI assistant tasked with reviewing and correcting solutions.\n"
            "The User will provide a problem and an attempted solution. Your job is to identify any errors "
            "and provide a corrected solution if needed. Always show your reasoning process."
        )
    parts = []
    parts.append(f"<|im_start|>system\n{system_prompt}\n<|im_end|>")
    parts.append(f"<|im_start|>user\n{query}\n<|im_start|>assistant\n<think>{answer}</think>\n{guidance}\n<|im_end|>")
    parts.append("<|im_start|>assistant")
    return "\n".join(parts)


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
                rec = {'query': obj['query'], 'answer': obj['answer']}
                if 'reasoning' in obj:
                    rec['reasoning'] = obj['reasoning']
                data.append(rec)
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            for obj in json.load(f):
                rec = {'query': obj['query'], 'answer': obj['answer']}
                if 'reasoning' in obj:
                    rec['reasoning'] = obj['reasoning']
                data.append(rec)
    else:
        raise ValueError('Unsupported dataset format')
    return data


def load_math_dataset(split: str = "test[:500]", path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load the MATH benchmark via the :mod:`datasets` library."""
    ds = _load_split_dataset("hendrycks/math", split, path)
    return [{"query": x["problem"], "answer": x["solution"]} for x in ds]


def load_gsm8k_dataset(split: str = "test", path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load the GSM8K dataset."""
    ds = _load_split_dataset("openai/gsm8k", split, path)
    return [{"query": x["question"], "answer": x["answer"]} for x in ds]


def load_minerva_math_dataset(split: str = "test", path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load the Minerva Math dataset used in the paper."""
    ds = _load_split_dataset("knoveleng/Minerva-Math", split, path)
    return [{"query": x["problem"], "answer": x["solution"]} for x in ds]


def load_olympiadbench_dataset(split: str = "test", path: Optional[str] = None) -> List[Dict[str, str]]:
    """Load the OlympiadBench dataset."""
    ds = _load_split_dataset("lmms-lab/OlympiadBench", split, path)
    return [{"query": x["problem"], "answer": x["solution"]} for x in ds]


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
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """Generate candidate responses and compute rewards."""
    q_tokens = [tokenizer.encode(s['query'], add_special_tokens=False) for s in samples]
    answers = [s['answer'] for s in samples]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    queries = pad_sequences(q_tokens, pad_id)
    q_lens = torch.tensor([len(q) for q in q_tokens], dtype=torch.long)

    B = len(samples)
    responses = []
    lengths = []
    rewards = []
    for i in range(B):
        prompt = build_layer1_prompt(samples[i]["query"])
        inp_tok = tokenizer.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([inp_tok], dtype=torch.long)
        grp_resp = []
        grp_len = []
        grp_rew = []
        for _ in range(group_size):
            out = model.generate(inp, max_length=len(inp_tok) + max_length, do_sample=True)
            resp = out[0, len(inp_tok):].tolist()
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
    return queries, q_lens, resp_tensor, len_tensor, reward_tensor

def construct_second_pass_input(
    tokenizer,
    query_tokens: torch.Tensor,
    output_tokens: torch.Tensor,
    guidance_tokens: torch.Tensor,
    system_prompt: str | None = None,
) -> (torch.Tensor, int):
    """Return tokens for the layer-2 prompt based on ``query_tokens`` and ``output_tokens``."""
    query_text = tokenizer.decode(query_tokens.tolist())
    output_text = tokenizer.decode(output_tokens.tolist())
    guidance_text = tokenizer.decode(guidance_tokens.tolist())
    
    # Use Layer 2 specific system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a helpful AI assistant tasked with reviewing and correcting solutions.\n"
            "The User will provide a problem or a question, and an attempted solution. Your job is to identify any errors "
            "and provide a corrected solution if needed. Always show your reasoning process."
        )
    
    prompt = build_layer2_prompt(query_text, output_text, guidance_text, system_prompt)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long), len(ids)
