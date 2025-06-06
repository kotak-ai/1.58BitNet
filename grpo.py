import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

import copy

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
    """Two-layer GRPO with self-correction."""
    def __init__(self, model: nn.Module, ref_model: nn.Module, verifier, clip_eps: float = 0.2, beta: float = 0.01):
        self.layer1 = GRPOTrainer(model, ref_model, clip_eps, beta)
        self.layer2 = GRPOTrainer(model, ref_model, clip_eps, beta)
        self.verifier = verifier

    def train_batch(self, queries: torch.Tensor, responses: torch.Tensor, lengths: torch.Tensor, rewards: torch.Tensor, optimizer: torch.optim.Optimizer) -> Tuple[torch.Tensor, float]:
        B, G, L = responses.shape
        loss1 = self.layer1.step(queries, responses, lengths, rewards, optimizer)
        # construct second layer inputs
        corrected = []
        corrected_len = []
        corrected_rewards = []
        for b in range(B):
            for g in range(G):
                resp = responses[b, g]
                if self.verifier(resp):
                    corrected.append(resp)
                    corrected_len.append(lengths[b, g])
                    corrected_rewards.append(1.0)
                else:
                    # attempt correction by appending guiding token 0
                    new_resp = torch.cat([queries[b], resp])[:L]
                    if self.verifier(new_resp):
                        corrected.append(new_resp)
                        corrected_len.append(min(len(new_resp), L))
                        corrected_rewards.append(1.0)
        if not corrected:
            return loss1, 0.0
        corr_tensor = torch.stack(corrected)
        corr_len = torch.tensor(corrected_len, dtype=torch.long)
        corr_rewards = torch.tensor(corrected_rewards, dtype=torch.float)
        corr_tensor = corr_tensor.unsqueeze(1)
        corr_len = corr_len.unsqueeze(1)
        corr_rewards = corr_rewards.unsqueeze(1)
        loss2 = self.layer2.step(queries[:corr_tensor.size(0)], corr_tensor, corr_len, corr_rewards, optimizer)
        return loss1 + loss2, float(len(corrected)) / (B * G)
