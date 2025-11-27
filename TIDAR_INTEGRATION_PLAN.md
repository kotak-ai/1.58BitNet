# TiDAR Integration Plan for 1.58BitNet

## Executive Summary

This document provides a comprehensive technical plan to integrate **TiDAR (Think in Diffusion, Talk in Autoregression)** from [arXiv:2511.08923](https://arxiv.org/abs/2511.08923) into the existing 1.58BitNet framework. TiDAR enables 4.71x-5.91x faster inference by combining:
- **Diffusion-based parallel drafting** ("thinking") with bidirectional attention
- **Autoregressive verification** ("talking") with causal attention

This is an **additive integration** - all existing functionality (1.58-bit quantization, GRPO training, multi-layer self-correction) remains intact.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [New Files to Create](#2-new-files-to-create)
3. [Modifications to Existing Files](#3-modifications-to-existing-files)
4. [Detailed Implementation Specifications](#4-detailed-implementation-specifications)
5. [Training Pipeline](#5-training-pipeline)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Integration with Existing Features](#7-integration-with-existing-features)
8. [Testing Strategy](#8-testing-strategy)
9. [Hyperparameters and Configuration](#9-hyperparameters-and-configuration)

---

## 1. Architecture Overview

### 1.1 Core Concept

TiDAR partitions each generation step into three token sections:

```
[PREFIX TOKENS] [VERIFICATION TOKENS] [DRAFT TOKENS]
     (causal)        (causal)         (bidirectional)
```

| Section | Attention Type | Purpose |
|---------|----------------|---------|
| Prefix | Causal (left-to-right) | Context from prompt + accepted tokens |
| Verification | Causal | Tokens drafted in previous step, being verified |
| Draft | Bidirectional (within block) | New tokens being pre-drafted via diffusion |

### 1.2 Key Mathematical Formulations

**Autoregressive Distribution:**
```
p_AR(x; θ) = ∏_i p_θ(x_i | x_{<i})
```

**Diffusion (Marginal) Distribution:**
```
p_Diff(x; θ) = E_{x̃~q(·|x)} ∏_i p_θ(x_i | x̃)
```

**Combined Loss:**
```
L_TiDAR(θ) = [1/(1+α)] × {
    Σ_{i=1}^{S-1} [α/(S-1) × L_AR(x_i, x_{i+1}; θ)] +
    Σ_{i=1}^{S-1} [1/(S-1) × L_Diff([MASK], x_i; θ)]
}
```

Where:
- `α ∈ [0,1]`: Loss balancing factor (default: 1.0)
- `S`: Sequence length
- `L_AR`: Cross-entropy with label shift (next-token prediction)
- `L_Diff`: Cross-entropy without label shift (token reconstruction)

---

## 2. New Files to Create

### 2.1 `tidar_attention.py` - Hybrid Attention Mask Utilities

```python
# Purpose: Create and manage hybrid causal/bidirectional attention masks
# Dependencies: torch

Key Functions:
├── create_tidar_mask(seq_len, prefix_len, verify_len, draft_len, device)
│   Returns: [seq_len, seq_len] attention mask tensor
│
├── create_training_mask(seq_len, device)
│   Returns: Mask for training (prefix causal, last block bidirectional)
│
├── TiDARMaskCache
│   Class for efficient mask slicing during inference
│   Methods:
│   ├── __init__(max_seq_len, block_size)
│   ├── get_mask(prefix_len, verify_len, draft_len)
│   └── reset()
```

### 2.2 `tidar_model.py` - TiDAR-Enhanced LLaMA Model

```python
# Purpose: LLaMA model with TiDAR hybrid attention support
# Dependencies: llama_model.py, tidar_attention.py

Classes:
├── TiDARLlamaAttention(LlamaAttention)
│   Modifications:
│   ├── forward() accepts attention_mask with hybrid pattern
│   └── Supports both causal and bidirectional modes
│
├── TiDARLlamaDecoderLayer(LlamaDecoderLayer)
│   Uses TiDARLlamaAttention
│
├── TiDARLlamaModel(LlamaModel)
│   Modifications:
│   ├── forward() with mode='ar'|'diff'|'hybrid'
│   ├── forward_hybrid() for combined AR+diffusion pass
│   ├── generate_tidar() for TiDAR inference
│   └── MASK_TOKEN_ID class attribute
```

### 2.3 `tidar_training.py` - TiDAR Training Loop

```python
# Purpose: Training with dual AR+diffusion losses
# Dependencies: tidar_model.py, trainingv2.py

Functions:
├── create_tidar_training_batch(batch, mask_token_id)
│   Doubles sequence: [original] + [fully_masked]
│   Returns: (input_ids, ar_labels, diff_labels, attention_mask)
│
├── tidar_loss(model, inputs, ar_labels, diff_labels, attention_mask, alpha)
│   Computes combined TiDAR loss
│   Returns: (total_loss, ar_loss, diff_loss)
│
├── TiDARTrainer
│   Class for TiDAR training
│   Methods:
│   ├── __init__(model, tokenizer, config)
│   ├── train_step(batch, optimizer)
│   ├── train_epoch(dataloader, optimizer)
│   └── save_checkpoint(path)

Key Parameters:
├── alpha: float = 1.0  # AR/diffusion loss balance
├── mask_ratio: float = 1.0  # For training, always 1.0 (full mask)
└── block_size: int = 16  # Tokens per draft block
```

### 2.4 `tidar_inference.py` - TiDAR Generation Engine

```python
# Purpose: Fast inference with diffusion drafting + AR verification
# Dependencies: tidar_model.py, tidar_attention.py

Classes:
├── TiDARGenerator
│   Methods:
│   ├── __init__(model, tokenizer, block_size, beta)
│   ├── generate(prompt, max_length, temperature)
│   ├── _draft_tokens(prefix_kv, num_tokens)
│   ├── _verify_tokens(drafted, prefix_kv)
│   ├── _rejection_sample(ar_probs, diff_probs, drafted)
│   └── _aggregate_logits(ar_logits, diff_logits, beta)
│
├── TiDARKVCache
│   Efficient KV cache for hybrid attention
│   Methods:
│   ├── __init__(num_layers, max_seq_len)
│   ├── update(layer_idx, key, value, positions)
│   ├── get(layer_idx, positions)
│   └── evict(positions_to_remove)

Key Parameters:
├── block_size: int = 16  # Tokens to draft per step
├── beta: float = 1.0  # Logit aggregation (1=trust AR, 0=trust diffusion)
└── max_drafts: int = 5  # Maximum draft attempts before fallback
```

### 2.5 `tidar_config.py` - Configuration Dataclass

```python
# Purpose: Central configuration for TiDAR parameters
# Dependencies: dataclasses

@dataclass
class TiDARConfig:
    # Model
    base_model_path: str
    mask_token_id: int = -1  # Auto-detect if -1

    # Training
    alpha: float = 1.0  # Loss balance (1.0 = equal weight)
    learning_rate: float = 1e-5
    min_learning_rate: float = 3e-6
    warmup_ratio: float = 0.01
    max_seq_length: int = 4096
    batch_size: int = 4
    grad_accum_steps: int = 1

    # Inference
    block_size: int = 16
    beta: float = 1.0
    max_draft_attempts: int = 5

    # Quantization (integration with 1.58-bit)
    use_quantization: bool = True
    use_ste: bool = True
```

---

## 3. Modifications to Existing Files

### 3.1 `llama_model.py`

**Changes Required:**

```python
# Line ~160: LlamaAttention.forward()
# ADD: Support for 4D attention masks (hybrid pattern)

def forward(self, hidden_states, attention_mask, cos, sin,
            attention_mode='causal'):  # NEW PARAMETER
    # ... existing code ...

    # MODIFY: Attention mask handling
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            # TiDAR hybrid mask: [batch, 1, seq, seq]
            # Use directly without modification
            attention_bias = attention_mask
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
            attention_bias = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
        # ... rest of existing logic
```

### 3.2 `quantization_utils.py`

**No changes required** - STE functions already support TiDAR's gradient flow needs.

### 3.3 `h_bitlinear.py`

**No changes required** - Quantized linear layers work with TiDAR.

### 3.4 `trainingv2.py`

**Changes Required:**

```python
# ADD: Import TiDAR components
from tidar_training import TiDARTrainer, create_tidar_training_batch

# ADD: New argument in get_arg_parser()
parser.add_argument("--tidar", action="store_true",
                    help="Enable TiDAR training mode")
parser.add_argument("--tidar_alpha", type=float, default=1.0,
                    help="TiDAR loss balance factor")
parser.add_argument("--tidar_block_size", type=int, default=16,
                    help="TiDAR draft block size")

# MODIFY: run() function to branch on --tidar flag
def run(args):
    if args.tidar:
        return run_tidar_training(args)
    else:
        # ... existing training code ...
```

### 3.5 `grpo.py`

**Changes Required (for TiDAR+GRPO integration):**

```python
# ADD: TiDAR-aware GRPO trainer
class TiDARGRPOTrainer(GRPOTrainer):
    """GRPO trainer that uses TiDAR for faster response generation."""

    def __init__(self, model, ref_model, tidar_generator, ...):
        super().__init__(model, ref_model, ...)
        self.tidar_gen = tidar_generator

    def generate_responses(self, queries, num_samples, max_length):
        """Use TiDAR for fast parallel response generation."""
        # Use diffusion drafting for faster sampling
        return self.tidar_gen.generate_batch(queries, num_samples, max_length)
```

---

## 4. Detailed Implementation Specifications

### 4.1 Hybrid Attention Mask Construction

```python
def create_tidar_mask(
    seq_len: int,
    prefix_len: int,
    verify_len: int,
    draft_len: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create TiDAR hybrid attention mask.

    Structure:
    - Prefix tokens (0:prefix_len): Causal attention
    - Verify tokens (prefix_len:prefix_len+verify_len): Causal attention
    - Draft tokens (prefix_len+verify_len:): Bidirectional within block

    Returns:
        mask: [1, 1, seq_len, seq_len] where 1=attend, 0=mask
    """
    mask = torch.zeros(seq_len, seq_len, device=device)

    # Prefix: causal (lower triangular)
    causal_end = prefix_len + verify_len
    for i in range(causal_end):
        mask[i, :i+1] = 1.0

    # Draft: bidirectional within block, causal to prefix
    draft_start = prefix_len + verify_len
    for i in range(draft_start, seq_len):
        # Can attend to all prefix+verify tokens
        mask[i, :causal_end] = 1.0
        # Can attend to all draft tokens (bidirectional)
        mask[i, draft_start:] = 1.0

    return mask.unsqueeze(0).unsqueeze(0)
```

### 4.2 Training Data Preparation

```python
def create_tidar_training_batch(
    input_ids: torch.Tensor,  # [batch, seq_len]
    mask_token_id: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare batch for TiDAR training.

    Doubles sequence: [original tokens] + [MASK tokens]

    Returns:
        - input_ids: [batch, 2*seq_len] with masks appended
        - ar_labels: [batch, seq_len] for AR loss (shifted)
        - diff_labels: [batch, seq_len] for diffusion loss (aligned)
        - attention_mask: [batch, 1, 2*seq_len, 2*seq_len] hybrid mask
    """
    batch_size, seq_len = input_ids.shape

    # Create doubled input: [tokens] + [MASK repeated]
    mask_section = torch.full((batch_size, seq_len), mask_token_id,
                              dtype=torch.long, device=device)
    doubled_input = torch.cat([input_ids, mask_section], dim=1)

    # AR labels: next-token prediction on first half
    ar_labels = input_ids[:, 1:].clone()  # Shifted by 1
    ar_labels = F.pad(ar_labels, (0, 1), value=-100)  # Pad end

    # Diffusion labels: token reconstruction on second half
    diff_labels = input_ids.clone()  # Aligned (no shift)

    # Create hybrid attention mask
    # First half: causal, Second half: bidirectional
    attention_mask = create_training_mask(2 * seq_len, device)
    attention_mask = attention_mask.expand(batch_size, -1, -1, -1)

    return doubled_input, ar_labels, diff_labels, attention_mask


def create_training_mask(doubled_seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Training mask: prefix causal, last block bidirectional.
    """
    seq_len = doubled_seq_len // 2
    mask = torch.zeros(doubled_seq_len, doubled_seq_len, device=device)

    # First half: standard causal
    for i in range(seq_len):
        mask[i, :i+1] = 1.0

    # Second half: bidirectional within, causal to first half
    for i in range(seq_len, doubled_seq_len):
        mask[i, :seq_len] = 1.0  # Attend to all of first half
        mask[i, seq_len:] = 1.0  # Attend to all of second half (bidirectional)

    return mask.unsqueeze(0).unsqueeze(0)
```

### 4.3 Loss Computation

```python
def tidar_loss(
    model: nn.Module,
    input_ids: torch.Tensor,      # [batch, 2*seq_len]
    ar_labels: torch.Tensor,      # [batch, seq_len]
    diff_labels: torch.Tensor,    # [batch, seq_len]
    attention_mask: torch.Tensor, # [batch, 1, 2*seq_len, 2*seq_len]
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute TiDAR combined loss.

    L_TiDAR = [1/(1+α)] × {α × L_AR + L_Diff}

    Returns:
        (total_loss, ar_loss, diff_loss)
    """
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1] // 2

    # Forward pass through model
    logits = model(input_ids, attention_mask=attention_mask)
    # logits: [batch, 2*seq_len, vocab_size]

    # Split logits
    ar_logits = logits[:, :seq_len-1, :]   # First half, shifted for NTP
    diff_logits = logits[:, seq_len:, :]   # Second half, aligned

    # AR Loss (next-token prediction)
    ar_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    ar_loss = ar_loss_fn(
        ar_logits.reshape(-1, ar_logits.size(-1)),
        ar_labels[:, :-1].reshape(-1)
    )

    # Diffusion Loss (token reconstruction)
    diff_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    diff_loss = diff_loss_fn(
        diff_logits.reshape(-1, diff_logits.size(-1)),
        diff_labels.reshape(-1)
    )

    # Combined loss with balancing
    total_loss = (1.0 / (1.0 + alpha)) * (alpha * ar_loss + diff_loss)

    return total_loss, ar_loss, diff_loss
```

### 4.4 Rejection Sampling for Verification

```python
def rejection_sample(
    drafted_tokens: torch.Tensor,  # [num_drafts]
    ar_probs: torch.Tensor,        # [num_drafts, vocab_size]
    diff_probs: torch.Tensor,      # [num_drafts, vocab_size]
) -> Tuple[torch.Tensor, int]:
    """
    Verify drafted tokens using rejection sampling.

    For each position i:
        1. Compute acceptance probability:
           p_accept = min(1, p_AR(draft[i]) / max(p_Diff(draft[i]), p_AR(draft[i])))
        2. Sample u ~ Uniform(0, 1)
        3. If u <= p_accept: accept and continue
           Else: reject and stop (return accepted prefix)

    Returns:
        (accepted_tokens, num_accepted)
    """
    num_drafts = drafted_tokens.shape[0]
    accepted = []

    for i in range(num_drafts):
        token = drafted_tokens[i].item()
        p_ar = ar_probs[i, token].item()
        p_diff = diff_probs[i, token].item()

        # Acceptance probability
        p_accept = min(1.0, p_ar / max(p_diff, p_ar, 1e-10))

        # Sample
        u = torch.rand(1).item()
        if u <= p_accept:
            accepted.append(token)
        else:
            break

    if accepted:
        return torch.tensor(accepted, dtype=drafted_tokens.dtype), len(accepted)
    else:
        return torch.tensor([], dtype=drafted_tokens.dtype), 0
```

### 4.5 TiDAR Generation Loop

```python
class TiDARGenerator:
    def __init__(
        self,
        model: TiDARLlamaModel,
        tokenizer,
        block_size: int = 16,
        beta: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.beta = beta
        self.mask_cache = TiDARMaskCache(4096, block_size)
        self.kv_cache = TiDARKVCache(model.config.num_hidden_layers, 4096)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate text using TiDAR hybrid inference.

        Each step:
        1. Verify previous drafts using AR (rejection sampling)
        2. Pre-draft next block using diffusion
        3. All in single forward pass
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        prefix = input_ids[0].tolist()
        drafted = []
        generated = []

        while len(generated) < max_new_tokens:
            # Construct input for this step
            # [prefix tokens] + [drafted tokens] + [MASK tokens for new draft]
            verify_tokens = drafted if drafted else []
            mask_tokens = [self.model.MASK_TOKEN_ID] * self.block_size

            step_input = torch.tensor(
                [prefix + verify_tokens + mask_tokens],
                dtype=torch.long, device=device
            )

            # Get hybrid attention mask
            mask = self.mask_cache.get_mask(
                len(prefix), len(verify_tokens), self.block_size
            )

            # Single forward pass
            logits = self.model(step_input, attention_mask=mask)

            # Extract AR logits (verification section)
            # Extract diffusion logits (draft section)
            verify_start = len(prefix)
            verify_end = verify_start + len(verify_tokens)
            draft_start = verify_end

            ar_logits = logits[0, verify_start:verify_end, :]
            diff_logits = logits[0, draft_start:, :]

            # Temperature scaling
            ar_probs = F.softmax(ar_logits / temperature, dim=-1)
            diff_probs = F.softmax(diff_logits / temperature, dim=-1)

            # Rejection sampling on verification tokens
            if verify_tokens:
                verify_tensor = torch.tensor(verify_tokens, device=device)
                accepted, num_accepted = rejection_sample(
                    verify_tensor, ar_probs, diff_probs[:len(verify_tokens)]
                )

                # Add accepted tokens to generated output
                generated.extend(accepted.tolist())
                prefix.extend(accepted.tolist())

                # If not all accepted, sample one corrected token
                if num_accepted < len(verify_tokens):
                    # Sample from AR distribution at rejection point
                    corrected = torch.multinomial(ar_probs[num_accepted], 1)
                    generated.append(corrected.item())
                    prefix.append(corrected.item())

            # Draft new tokens from diffusion
            drafted = []
            for i in range(self.block_size):
                # Aggregate logits (optional, default to AR)
                if self.beta == 1.0:
                    probs = diff_probs[i]
                else:
                    combined = self.beta * ar_logits[i] + (1 - self.beta) * diff_logits[i]
                    probs = F.softmax(combined / temperature, dim=-1)

                token = torch.multinomial(probs, 1).item()
                drafted.append(token)

                # Stop if EOS
                if token == self.tokenizer.eos_token_id:
                    break

            # Check for EOS in generated
            if self.tokenizer.eos_token_id in generated:
                break

        return self.tokenizer.decode(generated)
```

---

## 5. Training Pipeline

### 5.1 Training Script: `train_tidar.py`

```python
#!/usr/bin/env python3
"""TiDAR Training Script for 1.58BitNet"""

import argparse
import torch
from transformers import AutoTokenizer, LlamaConfig
from tidar_model import TiDARLlamaModel
from tidar_training import TiDARTrainer
from tidar_config import TiDARConfig
from device_utils import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", default="tidar_model")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=4096)
    args = parser.parse_args()

    device = get_device()

    # Load model and tokenizer
    config = LlamaConfig.from_pretrained(args.model_path)
    model = TiDARLlamaModel(config)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Ensure mask token exists
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        model.resize_token_embeddings(len(tokenizer))

    model.MASK_TOKEN_ID = tokenizer.mask_token_id

    # Create trainer
    tidar_config = TiDARConfig(
        base_model_path=args.model_path,
        alpha=args.alpha,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_seq_length=args.max_length,
    )

    trainer = TiDARTrainer(model, tokenizer, tidar_config)

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Train
    trainer.train(dataset, args.iters)

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
```

### 5.2 Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `alpha` | 1.0 | Equal weight AR and diffusion losses |
| `learning_rate` | 1e-5 | From paper |
| `min_lr` | 3e-6 | Cosine decay minimum |
| `warmup_ratio` | 0.01 | 1% of total steps |
| `batch_size` | 4-8 | Per-GPU, adjust for VRAM |
| `max_seq_length` | 4096 | Standard context length |
| `precision` | bfloat16 | Mixed precision |
| `grad_accum_steps` | 1-4 | Effective batch size scaling |

---

## 6. Inference Pipeline

### 6.1 Generation Script: `generate_tidar.py`

```python
#!/usr/bin/env python3
"""TiDAR Fast Generation for 1.58BitNet"""

import argparse
import torch
from transformers import AutoTokenizer
from tidar_model import TiDARLlamaModel
from tidar_inference import TiDARGenerator
from device_utils import get_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    device = get_device()

    # Load model
    model = TiDARLlamaModel.load_pretrained(args.model_path)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Create generator
    generator = TiDARGenerator(
        model, tokenizer,
        block_size=args.block_size,
        beta=args.beta,
    )

    # Generate
    output = generator.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(output)

if __name__ == "__main__":
    main()
```

### 6.2 Throughput Expectations

Based on paper results:
| Model Size | AR Baseline | TiDAR | Speedup |
|------------|-------------|-------|---------|
| 1.5B | 1.0x | 4.71x | 4.71x |
| 8B | 1.0x | 5.91x | 5.91x |

With 1.58-bit quantization + TiDAR:
- **Memory**: 20x reduction from quantization
- **Speed**: 4-6x from TiDAR parallel drafting
- **Combined**: Run 7B models at >100 tokens/sec on consumer GPU

---

## 7. Integration with Existing Features

### 7.1 1.58-Bit Quantization Compatibility

TiDAR works directly with quantized layers:
- `HBitLinear` and `BitLinear` layers unchanged
- STE gradient flow works for both AR and diffusion losses
- Weight quantization applies to all projections equally

```python
class TiDARLlamaModel(LlamaModel):
    def __init__(self, config, linear_cls=HBitLinear, ...):
        super().__init__(config, linear_cls=linear_cls)
        # TiDAR inherits quantization from base class
```

### 7.2 GRPO Integration

TiDAR can accelerate GRPO response generation:

```python
class TiDARGRPOTrainer(MultiLayerGRPOTrainer):
    """GRPO with TiDAR-accelerated generation."""

    def __init__(self, model, ref_model, tidar_generator, ...):
        super().__init__(model, ref_model, ...)
        self.tidar_gen = tidar_generator

    def generate_responses(self, queries, num_samples):
        """Use TiDAR for 4-6x faster response generation."""
        responses = []
        for q in queries:
            for _ in range(num_samples):
                resp = self.tidar_gen.generate(q, max_new_tokens=100)
                responses.append(resp)
        return responses
```

### 7.3 Multi-Layer Self-Correction

TiDAR can be used in the second pass of multi-layer GRPO:

```python
# In train_batch():
# Use TiDAR for faster correction generation
with torch.no_grad():
    corrected = self.tidar_gen.generate(
        second_pass_prompt,
        max_new_tokens=self.second_max_length,
    )
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/test_tidar_attention.py
def test_create_tidar_mask():
    mask = create_tidar_mask(20, 10, 4, 6, 'cpu')
    assert mask.shape == (1, 1, 20, 20)
    # Verify causal region is lower triangular
    assert mask[0, 0, 5, 6] == 0  # Future masked in causal
    assert mask[0, 0, 5, 4] == 1  # Past visible in causal
    # Verify bidirectional region
    assert mask[0, 0, 16, 18] == 1  # Draft can see draft


# tests/test_tidar_loss.py
def test_tidar_loss_computation():
    model = TiDARLlamaModel(test_config)
    input_ids = torch.randint(0, 1000, (2, 100))
    doubled, ar_labels, diff_labels, mask = create_tidar_training_batch(
        input_ids, mask_token_id=1000, device='cpu'
    )
    total, ar, diff = tidar_loss(model, doubled, ar_labels, diff_labels, mask)
    assert total.requires_grad
    assert ar.item() > 0
    assert diff.item() > 0


# tests/test_rejection_sampling.py
def test_rejection_sample_accepts_high_prob():
    drafted = torch.tensor([5, 10, 15])
    ar_probs = torch.zeros(3, 100)
    ar_probs[:, [5, 10, 15]] = 0.9  # High AR prob for drafted tokens
    diff_probs = ar_probs.clone()

    accepted, num = rejection_sample(drafted, ar_probs, diff_probs)
    assert num == 3  # All should be accepted
```

### 8.2 Integration Tests

```python
# tests/test_tidar_generation.py
def test_tidar_generates_coherent_text():
    model = TiDARLlamaModel.from_pretrained(test_model_path)
    tokenizer = AutoTokenizer.from_pretrained(test_model_path)
    generator = TiDARGenerator(model, tokenizer)

    output = generator.generate("The quick brown fox", max_new_tokens=20)
    assert len(output) > 0
    assert isinstance(output, str)


def test_tidar_faster_than_ar():
    # Benchmark comparison
    import time

    # AR generation
    start = time.time()
    ar_output = model.generate(prompt, max_length=100)
    ar_time = time.time() - start

    # TiDAR generation
    start = time.time()
    tidar_output = generator.generate(prompt, max_new_tokens=100)
    tidar_time = time.time() - start

    assert tidar_time < ar_time * 0.5  # At least 2x faster
```

---

## 9. Hyperparameters and Configuration

### 9.1 Default Configuration

```python
TIDAR_DEFAULT_CONFIG = {
    # Training
    "alpha": 1.0,              # Loss balance (equal weight)
    "learning_rate": 1e-5,     # From paper
    "min_lr": 3e-6,            # Cosine decay minimum
    "warmup_ratio": 0.01,      # 1% warmup
    "weight_decay": 0.1,       # AdamW decay
    "batch_size": 4,           # Per-GPU
    "max_seq_length": 4096,    # Context length
    "grad_accum_steps": 1,     # Gradient accumulation

    # Inference
    "block_size": 16,          # Tokens per draft
    "beta": 1.0,               # Trust AR (1) vs diffusion (0)
    "temperature": 1.0,        # Sampling temperature
    "max_draft_attempts": 5,   # Before AR fallback

    # Architecture
    "use_quantization": True,  # 1.58-bit weights
    "use_ste": True,           # Straight-through estimator
    "precision": "bfloat16",   # Mixed precision
}
```

### 9.2 Tuning Guide

| Parameter | Effect of Increase | Recommended Range |
|-----------|-------------------|-------------------|
| `alpha` | More AR focus | 0.5 - 2.0 |
| `block_size` | More parallel tokens, higher rejection risk | 8 - 32 |
| `beta` | Trust AR over diffusion | 0.5 - 1.0 |
| `temperature` | More diverse outputs | 0.7 - 1.2 |

---

## 10. File Structure After Integration

```
1.58BitNet/
├── llama_model.py          # [MODIFIED] 4D attention mask support
├── h_bitlinear.py          # [UNCHANGED]
├── quantization_utils.py   # [UNCHANGED]
├── grpo.py                 # [MODIFIED] TiDARGRPOTrainer added
├── trainingv2.py           # [MODIFIED] --tidar flag added
│
├── tidar/                  # [NEW DIRECTORY]
│   ├── __init__.py
│   ├── tidar_attention.py  # Hybrid attention masks
│   ├── tidar_model.py      # TiDARLlamaModel
│   ├── tidar_training.py   # Training loop
│   ├── tidar_inference.py  # Generation engine
│   └── tidar_config.py     # Configuration
│
├── train_tidar.py          # [NEW] Training script
├── generate_tidar.py       # [NEW] Generation script
│
└── tests/
    ├── test_tidar_attention.py
    ├── test_tidar_loss.py
    ├── test_tidar_generation.py
    └── test_tidar_grpo.py
```

---

## References

- **TiDAR Paper**: [arXiv:2511.08923](https://arxiv.org/abs/2511.08923)
- **Project Page**: [tidarlm.github.io](https://tidarlm.github.io/)
- **HuggingFace**: [papers/2511.08923](https://huggingface.co/papers/2511.08923)

---

## Summary

This integration plan adds TiDAR capabilities to 1.58BitNet while preserving all existing functionality:

| Feature | Status |
|---------|--------|
| 1.58-bit quantization | Preserved, works with TiDAR |
| GRPO training | Preserved, enhanced with TiDAR speedup |
| Multi-layer self-correction | Preserved, can use TiDAR generation |
| Standard training | Preserved, --tidar flag adds TiDAR mode |

**Expected Benefits:**
- **4-6x faster inference** from parallel drafting
- **Combined with 20x memory reduction** from quantization
- **7B models on consumer GPUs** at interactive speeds
