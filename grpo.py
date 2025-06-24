import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable, Sequence

import copy
import random
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

    def grpo_objective(
        self,
        logp: torch.Tensor,
        old_logp: torch.Tensor,
        adv: torch.Tensor,
        ref_logp: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the clipped policy objective with KL penalty."""

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        if adv.dim() == 1:
            adv = adv.unsqueeze(1).expand_as(logp)
        obj = torch.minimum(ratio * adv, clipped * adv)
        obj = (obj * mask).sum(dim=1) / mask.sum(dim=1)
        kl = (torch.exp(logp) * (logp - ref_logp) * mask).sum(dim=1) / mask.sum(dim=1)
        return obj - self.beta * kl

    def step(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
        lengths: torch.Tensor,
        rewards: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        advantages: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            if advantages is None:
                baseline = rewards.mean(dim=1, keepdim=True)
                adv = rewards - baseline
            else:
                adv = advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        responses_flat = responses.view(B * G, L)
        lengths_flat = lengths.view(B * G)
        logits = self.model(responses_flat)
        old_logits = self.old_model(responses_flat)
        ref_logits = self.ref_model(responses_flat)
        logp = self._log_probs(logits, responses_flat)
        old_logp = self._log_probs(old_logits, responses_flat)
        ref_logp = self._log_probs(ref_logits, responses_flat)
        adv_flat = adv.view(B * G)
        mask = (torch.arange(L, device=responses.device).unsqueeze(0) < lengths_flat.unsqueeze(1)).float()
        obj = self.grpo_objective(logp, old_logp, adv_flat, ref_logp, mask)
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
    guiding_prompt : str or list[str]
        One or more prompts appended during the second pass.  If multiple
        prompts are provided a random one is chosen for each correction.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[str], float],
        tokenizer,
        guiding_prompt: str | list[str],
        clip_eps: float = 0.2,
        beta: float = 0.01,
        verifier: Callable[[float, float], bool] | None = None,
        second_max_length: int = 20,
         augmentation_size: int = 1,
    ):
        self.layer1 = GRPOTrainer(model, ref_model, clip_eps, beta)
        self.layer2 = GRPOTrainer(model, ref_model, clip_eps, beta)
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(guiding_prompt, str):
            guiding_prompt = [guiding_prompt]
        self.guidance_tokens = [
            torch.tensor(
                tokenizer.encode(p, add_special_tokens=False),
                dtype=torch.long,
            )
            for p in guiding_prompt
        ]
        sep = getattr(tokenizer, "sep_token_id", None)
        if sep is None:
            sep = getattr(tokenizer, "eos_token_id", 0)
        self.sep_id = int(sep)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0)
        self.pad_id = pad_id
        self.verifier = verifier
        self.second_max_length = second_max_length
        self.augmentation_size = augmentation_size
        # store successful corrections to reuse in future iterations
        self.correction_buffer: list[tuple[torch.Tensor, torch.Tensor, int, float, float]] = []

    def train_batch(
        self,
        queries: torch.Tensor,
        query_lengths: torch.Tensor,
        responses: torch.Tensor,
        lengths: torch.Tensor,
        rewards: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        log_texts: int = 0,
        references: Sequence[str] | None = None,
    ) -> Tuple[torch.Tensor, float] | Tuple[torch.Tensor, float, list[str]]:
        """Train using two GRPO passes and measure improvement.

        Returns a tuple of the combined loss from both layers and the
        fraction of responses where the second pass achieved a higher
        reward than the first.
        """
        B, G, L = responses.shape
        loss1 = self.layer1.step(queries, responses, lengths, rewards, optimizer)
        # ensure the second layer uses the updated policy from the first step
        self.layer2.old_model.load_state_dict(self.layer2.model.state_dict())
        # attempt self-correction using the second layer
        corrected = []
        corrected_len = []
        corrected_rewards = []
        corrected_adv = []
        corrected_queries = []
        new_buffer_entries: list[tuple[torch.Tensor, torch.Tensor, int, float, float]] = []
        log_text_list: list[str] = []
        success = 0
        total_attempts = 0
        
        for b in range(B):
            q_tokens = queries[b, : query_lengths[b]]
            for g in range(G):
                resp = responses[b, g, : lengths[b, g]]
                base_reward = float(rewards[b, g])
                for _ in range(self.augmentation_size):
                    guidance = random.choice(self.guidance_tokens)
                    inp, inp_len = construct_second_pass_input(
                        self.tokenizer,
                        q_tokens,
                        resp,
                        guidance,
                    )
                    with torch.no_grad():
                        gen = self.layer2.model.generate(
                            inp.unsqueeze(0),
                            max_length=inp_len + self.second_max_length,
                            do_sample=True,
                        )
                    new_resp = gen[0, inp_len:]
                    text = self.tokenizer.decode(new_resp.tolist())
                    reward_val = float(self.reward_fn(text))
                    if self.verifier is None:
                        improved = reward_val > base_reward
                    else:
                        ref = references[b] if references is not None else None
                        try:
                            improved = bool(
                                self.verifier(
                                    reward_val,
                                    base_reward,
                                    text,
                                    ref,
                                )
                            )
                        except TypeError:
                            # Backwards compatibility with two-argument verifiers
                            improved = bool(self.verifier(reward_val, base_reward))
                    if improved:
                        success += 1
                        if len(log_text_list) < log_texts:
                            log_text_list.append(text)
                        new_buffer_entries.append(
                            (
                                queries[b].clone(),
                                new_resp.clone(),
                                new_resp.numel(),
                                reward_val,
                                reward_val - base_reward,
                            )
                        )
                    corrected.append(new_resp)
                    corrected_len.append(new_resp.numel())
                    corrected_rewards.append(reward_val)
                    corrected_adv.append(reward_val - base_reward)
                    corrected_queries.append(queries[b])
                    total_attempts += 1
                        
        # combine stored corrections from previous iterations with the new ones
        buf_q = [e[0] for e in self.correction_buffer]
        buf_r = [e[1] for e in self.correction_buffer]
        buf_l = [e[2] for e in self.correction_buffer]
        buf_rewards = [e[3] for e in self.correction_buffer]
        buf_adv = [e[4] for e in self.correction_buffer]

        all_r = buf_r + corrected
        all_len = buf_l + corrected_len
        all_rewards = buf_rewards + corrected_rewards
        all_adv = buf_adv + corrected_adv
        all_queries = buf_q + corrected_queries

        max_len = max(all_len)
        corr_tensor = torch.full((len(all_r), 1, max_len), self.pad_id, dtype=torch.long)
        for i, seq in enumerate(all_r):
            corr_tensor[i, 0, : seq.numel()] = seq
        corr_len = torch.tensor(all_len, dtype=torch.long).unsqueeze(1)
        corr_rewards = torch.tensor(all_rewards, dtype=torch.float).unsqueeze(1)
        corr_adv = torch.tensor(all_adv, dtype=torch.float).unsqueeze(1)
        corr_queries = torch.stack(all_queries)
        loss2 = self.layer2.step(
            corr_queries,
            corr_tensor,
            corr_len,
            corr_rewards,
            optimizer,
            advantages=corr_adv,
        )
        # store successful corrections for the next iteration
        self.correction_buffer.extend(new_buffer_entries)

        denom = B * G * self.augmentation_size
        if log_texts:
            return loss1 + loss2, float(success) / denom, log_text_list
        return loss1 + loss2, float(success) / denom
