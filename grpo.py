import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable, Sequence

import copy
import random
from grpo_data import construct_second_pass_input
from custom_gradient_checkpointing import custom_checkpoint


class GRPOTrainer:
    """Implements the single-layer Group Relative Policy Optimization algorithm.

    This trainer uses PPO-style clipped objectives with a KL penalty against
    a reference model to train language models using group-relative rewards.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        clip_eps: float = 0.2,
        beta: float = 0.01,
        *,
        update_old_every: int = 4,
        grad_checkpoint: bool = False,
    ):
        """Initialize the GRPO trainer.

        Args:
            model: The policy model to train
            ref_model: Reference model for KL penalty (frozen)
            clip_eps: PPO clipping epsilon
            beta: KL penalty coefficient
            update_old_every: Steps between old policy updates (for proper PPO)
            grad_checkpoint: Whether to use gradient checkpointing
        """
        self.model = model
        self.ref_model = ref_model
        self.clip_eps = clip_eps
        self.beta = beta
        self.grad_checkpoint = grad_checkpoint
        self.update_old_every = update_old_every
        self._step_count = 0

        # Old model for importance sampling ratio
        self.old_model = copy.deepcopy(model)
        self.old_model.load_state_dict(model.state_dict())
        for param in self.old_model.parameters():
            param.requires_grad = False
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _log_probs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities of actions given logits."""
        logp = F.log_softmax(logits, dim=-1)
        return logp.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    def _compute_attention_mask(self, lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        """Create attention mask from sequence lengths."""
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask.float()

    def grpo_objective(
        self,
        logp: torch.Tensor,
        old_logp: torch.Tensor,
        adv: torch.Tensor,
        ref_logp: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the clipped policy objective with KL penalty.

        Args:
            logp: Log probabilities from current policy [B*G, L]
            old_logp: Log probabilities from old policy [B*G, L]
            adv: Advantages [B*G] or [B*G, L] for dense rewards
            ref_logp: Log probabilities from reference policy [B*G, L]
            mask: Sequence mask [B*G, L]

        Returns:
            Per-sample objective values [B*G]
        """
        # Compute importance sampling ratio
        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

        # Expand advantages if scalar
        if adv.dim() == 1:
            adv = adv.unsqueeze(1).expand_as(logp)

        # PPO clipped objective
        obj = torch.minimum(ratio * adv, clipped * adv)
        obj = (obj * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # KL divergence penalty (sample-based estimate)
        kl = ((logp - ref_logp) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

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
            queries: [B, Lq] tokens for queries (unused but kept for API compatibility)
            responses: [B, G, L] response tokens
            lengths: [B, G] lengths for responses
            rewards: [B, G] scalar rewards for each response or [B, G, L] dense per-token rewards
            optimizer: Optimizer for model parameters
            advantages: Optional pre-computed advantages [B, G] or [B, G, L]

        Returns:
            Loss tensor.
        """
        B, G, L = responses.shape
        dense = rewards.dim() == 3
        device = responses.device

        # Compute advantages from rewards if not provided
        with torch.no_grad():
            if advantages is None:
                baseline = rewards.mean(dim=1, keepdim=True)
                adv = rewards - baseline
            else:
                adv = advantages
            # Normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten for batch processing
        responses_flat = responses.view(B * G, L)
        lengths_flat = lengths.view(B * G)

        # Create proper attention mask
        attention_mask = self._compute_attention_mask(lengths_flat, L, device)

        # Forward pass through all models with attention mask
        if self.grad_checkpoint:
            logits = custom_checkpoint(
                lambda x, m: self.model(x, attention_mask=m),
                responses_flat, attention_mask
            )
        else:
            logits = self.model(responses_flat, attention_mask=attention_mask)

        with torch.no_grad():
            old_logits = self.old_model(responses_flat, attention_mask=attention_mask)
            ref_logits = self.ref_model(responses_flat, attention_mask=attention_mask)

        # Compute log probabilities
        logp = self._log_probs(logits, responses_flat)
        old_logp = self._log_probs(old_logits, responses_flat)
        ref_logp = self._log_probs(ref_logits, responses_flat)

        # Prepare advantages
        adv_flat = adv.view(B * G, L) if dense else adv.view(B * G)

        # Compute objective and loss
        obj = self.grpo_objective(logp, old_logp, adv_flat, ref_logp, attention_mask)
        loss = -torch.mean(obj)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update old model periodically (not every step for proper PPO)
        self._step_count += 1
        if self._step_count % self.update_old_every == 0:
            self.old_model.load_state_dict(self.model.state_dict())

        return loss.detach()

    def sync_old_model(self):
        """Manually synchronize old model with current model."""
        self.old_model.load_state_dict(self.model.state_dict())


class MultiLayerGRPOTrainer:
    """Two-layer GRPO with self-correction.

    This trainer implements a two-pass approach where:
    1. First layer generates initial responses
    2. Second layer attempts to correct/improve responses using guiding prompts

    Parameters
    ----------
    reward_fn : Callable[[str], float]
        Function used to score corrected responses produced by the second layer.
    guiding_prompt : str or list[str]
        One or more prompts appended during the second pass.
    prompt_probs : list[float], optional
        Probabilities for selecting guiding prompts.
    prompt_schedule : list[int], optional
        Fixed sequence of prompt indices to use.
    max_buffer_size : int
        Maximum number of corrections to store in buffer.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[str], float],
        tokenizer,
        guiding_prompt: str | list[str],
        *,
        prompt_probs: Sequence[float] | None = None,
        prompt_schedule: Sequence[int] | None = None,
        clip_eps: float = 0.2,
        beta: float = 0.01,
        verifier: Callable[[float, float], bool] | None = None,
        second_max_length: int = 20,
        augmentation_size: int = 1,
        grad_checkpoint: bool = False,
        update_old_every: int = 4,
        max_buffer_size: int = 1000,
    ):
        self.layer1 = GRPOTrainer(
            model, ref_model, clip_eps, beta,
            update_old_every=update_old_every, grad_checkpoint=grad_checkpoint
        )
        self.layer2 = GRPOTrainer(
            model, ref_model, clip_eps, beta,
            update_old_every=update_old_every, grad_checkpoint=grad_checkpoint
        )
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer

        # Process guiding prompts
        if isinstance(guiding_prompt, str):
            guiding_prompt = [guiding_prompt]
        self.guidance_tokens = [
            torch.tensor(
                tokenizer.encode(p, add_special_tokens=False),
                dtype=torch.long,
            )
            for p in guiding_prompt
        ]

        # Prompt selection probabilities
        if prompt_probs is not None:
            if len(prompt_probs) != len(self.guidance_tokens):
                raise ValueError("prompt_probs must match number of guiding prompts")
            total = float(sum(prompt_probs))
            if total <= 0:
                raise ValueError("prompt_probs sum must be positive")
            self.prompt_probs = [float(p) / total for p in prompt_probs]
        else:
            self.prompt_probs = None

        # Prompt schedule
        if prompt_schedule is not None:
            if not all(0 <= i < len(self.guidance_tokens) for i in prompt_schedule):
                raise ValueError("prompt_schedule indices out of range")
            self.prompt_schedule = list(prompt_schedule)
        else:
            self.prompt_schedule = None
        self._schedule_idx = 0

        # Token IDs
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
        self.max_buffer_size = max_buffer_size

        # Bounded correction buffer
        self.correction_buffer: list[tuple[torch.Tensor, torch.Tensor, int, float, float]] = []

    def _select_prompt_idx(self) -> int:
        """Select a guiding prompt index based on configuration."""
        if self.prompt_schedule is not None:
            idx = self.prompt_schedule[self._schedule_idx % len(self.prompt_schedule)]
            self._schedule_idx += 1
        elif self.prompt_probs is not None:
            idx = random.choices(range(len(self.guidance_tokens)), weights=self.prompt_probs)[0]
        else:
            idx = random.randrange(len(self.guidance_tokens))
        return idx

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
        fraction of responses where the second pass achieved a higher reward.
        """
        B, G, L = responses.shape
        device = responses.device

        # First layer training
        loss1 = self.layer1.step(queries, responses, lengths, rewards, optimizer)

        # Sync layer2's old model with updated policy
        self.layer2.old_model.load_state_dict(self.layer2.model.state_dict())

        # Generate corrections using second layer
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
                if rewards.dim() == 3:
                    base_reward = float(rewards[b, g, : lengths[b, g]].mean())
                else:
                    base_reward = float(rewards[b, g])

                for _ in range(self.augmentation_size):
                    idx = self._select_prompt_idx()
                    guidance = self.guidance_tokens[idx]

                    inp, inp_len = construct_second_pass_input(
                        self.tokenizer, q_tokens, resp, guidance
                    )

                    with torch.no_grad():
                        gen = self.layer2.model.generate(
                            inp.unsqueeze(0).to(device),
                            max_length=inp_len + self.second_max_length,
                            do_sample=True,
                        )
                    new_resp = gen[0, inp_len:]
                    text = self.tokenizer.decode(new_resp.tolist())
                    query_text = self.tokenizer.decode(q_tokens.tolist())
                    ref_text = references[b] if references is not None else None

                    # Compute reward for correction
                    try:
                        reward_val = float(self.reward_fn(text, ref_text, query_text))
                    except TypeError:
                        reward_val = float(self.reward_fn(text))

                    # Check if correction is an improvement
                    if self.verifier is None:
                        store = True
                        improved = reward_val > base_reward
                    else:
                        ref = references[b] if references is not None else None
                        try:
                            improved = bool(self.verifier(reward_val, base_reward, text, ref))
                        except TypeError:
                            improved = bool(self.verifier(reward_val, base_reward))
                        store = improved

                    if improved:
                        success += 1
                        if len(log_text_list) < log_texts:
                            log_text_list.append(text)
                        new_buffer_entries.append((
                            queries[b].clone(),
                            new_resp.clone(),
                            new_resp.numel(),
                            reward_val,
                            reward_val - base_reward,
                        ))

                    if store:
                        corrected.append(new_resp)
                        corrected_len.append(new_resp.numel())
                        corrected_rewards.append(reward_val)
                        corrected_adv.append(reward_val - base_reward)
                        corrected_queries.append(queries[b])

                    total_attempts += 1

        # Combine with buffered corrections (bounded)
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

        # Train layer 2 on corrections
        if all_r:
            max_len = max(all_len)
            corr_tensor = torch.full((len(all_r), 1, max_len), self.pad_id, dtype=torch.long, device=device)
            for i, seq in enumerate(all_r):
                corr_tensor[i, 0, : seq.numel()] = seq.to(device)
            corr_len = torch.tensor(all_len, dtype=torch.long, device=device).unsqueeze(1)
            corr_rewards = torch.tensor(all_rewards, dtype=torch.float, device=device).unsqueeze(1)
            corr_adv = torch.tensor(all_adv, dtype=torch.float, device=device).unsqueeze(1)
            corr_queries = torch.stack([q.to(device) for q in all_queries])

            loss2 = self.layer2.step(
                corr_queries,
                corr_tensor,
                corr_len,
                corr_rewards,
                optimizer,
                advantages=corr_adv,
            )
        else:
            loss2 = torch.tensor(0.0, device=device)

        # Update buffer with new entries (bounded)
        self.correction_buffer.extend(new_buffer_entries)
        if len(self.correction_buffer) > self.max_buffer_size:
            # Keep most recent entries
            self.correction_buffer = self.correction_buffer[-self.max_buffer_size:]

        denom = B * G * self.augmentation_size
        if log_texts:
            return loss1 + loss2, float(success) / denom, log_text_list
        return loss1 + loss2, float(success) / denom

    def clear_buffer(self):
        """Clear the correction buffer."""
        self.correction_buffer.clear()
