import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable

import copy
from grpo_data import construct_second_pass_input

class GRPOTrainer:
    """Implements the single-layer Group Relative Policy Optimization algorithm."""
    def __init__(self, model: nn.Module, ref_model: nn.Module, clip_eps: float = 0.2, beta: float = 0.01):
        self.model = model
        self.ref_model = ref_model
        self.clip_eps = clip_eps
        self.beta = beta
        self.old_model = copy.deepcopy(model)
        self.old_model.load_state_dict(model.state_dict())

    def _log_probs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def grpo_objective(self, logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        # advantages may be per token or per sequence
        if adv.dim() == 1:
            adv = adv.unsqueeze(1).expand_as(logp)
        obj = torch.minimum(ratio * adv, clipped * adv)
        kl = (torch.exp(logp) * (logp - ref_logp)).sum(-1)
        return obj.mean(dim=1) - self.beta * kl

    def step(self, queries: torch.Tensor, responses: torch.Tensor, lengths: torch.Tensor, rewards: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Performs a single GRPO policy update.

        Args:
            queries: [B, Lq] tokens for queries.
            responses: [B, G, L] response tokens.
            lengths: [B, G] lengths for responses.
            rewards: [B, G] scalar rewards for each response.
        Returns:
            Loss tensor.
        """
        B, G, L = responses.shape
        with torch.no_grad():
            baseline = rewards.mean(dim=1, keepdim=True)
            adv = rewards - baseline
        responses_flat = responses.view(B * G, L)
        lengths_flat = lengths.view(B * G)
        logits = self.model(responses_flat)
        old_logits = self.old_model(responses_flat)
        ref_logits = self.ref_model(responses_flat)
        logp = self._log_probs(logits, responses_flat)
        old_logp = self._log_probs(old_logits, responses_flat)
        ref_logp = self._log_probs(ref_logits, responses_flat)
        adv_flat = adv.view(B * G)
        obj = self.grpo_objective(logp, old_logp, adv_flat, ref_logp)
        loss = -torch.mean(obj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update old model
        self.old_model.load_state_dict(self.model.state_dict())
        return loss.detach()

class MultiLayerGRPOTrainer:
    """Two-layer GRPO with self-correction.

    Parameters
    ----------
    reward_fn : Callable[[str], float]
        Function used to score corrected responses produced by the second
        layer.  It receives the decoded response text and returns a scalar
        reward value.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[str], float],
        tokenizer,
        guiding_prompt: str,
        clip_eps: float = 0.2,
        beta: float = 0.01,
        verifier: Callable[[float, float], bool] | None = None,
    ):
        self.layer1 = GRPOTrainer(model, ref_model, clip_eps, beta)
        self.layer2 = GRPOTrainer(model, ref_model, clip_eps, beta)
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.guidance_tokens = torch.tensor(
            tokenizer.encode(guiding_prompt, add_special_tokens=False),
            dtype=torch.long,
        )
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0)
        self.pad_id = pad_id
        self.verifier = verifier

    def train_batch(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        lengths: torch.Tensor,
        rewards: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        log_texts: int = 0,
    ) -> Tuple[torch.Tensor, float] | Tuple[torch.Tensor, float, list[str]]:
        """Train using two GRPO passes and measure improvement.

        Returns a tuple of the combined loss from both layers and the
        fraction of responses where the second pass achieved a higher
        reward than the first.
        """
        B, G, L = responses.shape
        loss1 = self.layer1.step(queries, responses, lengths, rewards, optimizer)
        # attempt self-correction using the second layer
        corrected = []
        corrected_len = []
        corrected_rewards = []
        corrected_queries = []
        log_text_list: list[str] = []
        success = 0
        for b in range(B):
            for g in range(G):
                resp = responses[b, g, : lengths[b, g]]
                inp, inp_len = construct_second_pass_input(
                    queries[b], resp, self.guidance_tokens
                )
                with torch.no_grad():
                    gen = self.layer2.model.generate(
                        inp.unsqueeze(0),
                        max_length=inp_len + L,
                        do_sample=True,
                    )
                new_resp = gen[0, inp_len:]
                text = self.tokenizer.decode(new_resp.tolist())
                reward_val = float(self.reward_fn(text))
                # count improvement relative to the original reward using verifier
                base_reward = float(rewards[b, g])
                if self.verifier is None:
                    improved = reward_val > base_reward
                else:
                    improved = bool(self.verifier(reward_val, base_reward))
                success += int(improved)
                corrected.append(new_resp)
                corrected_len.append(new_resp.numel())
                corrected_rewards.append(reward_val)
                corrected_queries.append(queries[b])
                if len(log_text_list) < log_texts:
                    log_text_list.append(text)
        if not corrected:
            if log_texts:
                return loss1, 0.0, log_text_list
            return loss1, 0.0
        max_len = max(corrected_len)
        corr_tensor = torch.full(
            (len(corrected), 1, max_len), self.pad_id, dtype=torch.long
        )
        for i, seq in enumerate(corrected):
            corr_tensor[i, 0, : seq.numel()] = seq
        corr_len = torch.tensor(corrected_len, dtype=torch.long).unsqueeze(1)
        corr_rewards = torch.tensor(corrected_rewards, dtype=torch.float).unsqueeze(1)
        corr_queries = torch.stack(corrected_queries)
        loss2 = self.layer2.step(corr_queries, corr_tensor, corr_len, corr_rewards, optimizer)
        if log_texts:
            return loss1 + loss2, float(success) / (B * G), log_text_list
        return loss1 + loss2, float(success) / (B * G)
