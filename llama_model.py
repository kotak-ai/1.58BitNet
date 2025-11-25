import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import LlamaConfig, AutoTokenizer
except Exception:  # pragma: no cover - transformers may be missing
    LlamaConfig = AutoTokenizer = None  # type: ignore[misc]

from quantization_utils import (
    quantize_tensor,
    activation_quant,
    weight_quant,
    activation_norm_quant,
    gemm_lowbit,
    kv_cache_quant,
    act_quant_8bit,
    act_quant_4bit,
    quantize_tensor_1_58bit,
    pack_quantized_tensor,
    unpack_quantized_tensor,
    ste_round_clamp,
)
from device_utils import get_device

try:
    from safetensors.torch import save_file, load_file
except Exception:  # pragma: no cover - safetensors may be missing
    save_file = load_file = None  # type: ignore[misc]

import os
import json
import shutil
from tqdm import tqdm
import numpy as np
import time
from custom_gradient_checkpointing import custom_checkpoint
from h_bitlinear import HBitLinear


def RMSNorm(x, eps=1e-6):
    """RMS Normalization function."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RMSNormLayer(nn.Module):
    """RMS Normalization as a layer (LLaMA style)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class QuantizedEmbedding(nn.Module):
    """Embedding layer with optional quantization using STE."""

    def __init__(self, num_embeddings, embedding_dim, experiment=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)
        self.eps = 1e-5
        self.experiment = experiment
        self.weight_scale = None

    def forward(self, input):
        if self.experiment:
            # Use STE for gradient flow through quantization
            q, scale = quantize_tensor_1_58bit(self.weight, self.eps)
            self.weight_scale = scale
            quantized_weight = q.float() * scale
        else:
            # Standard embedding without quantization during training
            quantized_weight = self.weight
        return F.embedding(input, quantized_weight)


class BitLinear(nn.Linear):
    """Linear layer with ternary quantization using STE for gradient flow."""

    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5
        self.weight_scale = None

    def forward(self, x):
        x_float = x.to(torch.float32)
        x_norm = RMSNorm(x_float)
        x_quant = activation_quant(x_norm)

        # Quantize weights using STE (no .detach()!)
        # This allows gradients to flow through quantization
        w_scale = self.weight.abs().mean().clamp(min=self.eps)
        w_q = ste_round_clamp(self.weight / w_scale, -1, 1)
        w_quant = w_q * w_scale
        self.weight_scale = w_scale

        y = F.linear(x_quant, w_quant, self.bias)
        return y


def rotate_half(x):
    """Rotate half the hidden dims for rotary position embeddings."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def compute_rotary_embeddings(seq_length, head_dim, device, base=10000.0):
    """Compute correct rotary position embeddings.

    Returns cos and sin tensors of shape [seq_length, head_dim].
    """
    # Compute inverse frequencies for half the head dimension
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))

    # Compute position indices
    position_ids = torch.arange(seq_length, dtype=torch.float32, device=device)

    # Compute angles: [seq_length, half_dim]
    freqs = torch.outer(position_ids, inv_freq)

    # Create full cos/sin by repeating for both halves of head_dim
    # This ensures all positions in the head dimension are filled
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_heads, seq_len, head_dim]
        cos: Cosine tensor of shape [seq_len, head_dim]
        sin: Sine tensor of shape [seq_len, head_dim]

    Returns:
        Rotated q and k tensors.
    """
    # Add dimensions for batch and num_heads: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-head attention with proper head dimension reshape."""

    def __init__(self, config, linear_cls=HBitLinear):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.q_proj = linear_cls(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = linear_cls(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = linear_cls(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = linear_cls(config.hidden_size, config.hidden_size, bias=False)

        self.pretraining_tp = getattr(config, "pretraining_tp", 1)

    def forward(self, hidden_states, attention_mask, cos, sin):
        batch_size, seq_length, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim] for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings (after reshape!)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache quantization only during inference (not training)
        if not self.training:
            key_states = kv_cache_quant(key_states, training=False)
            value_states = kv_cache_quant(value_states, training=False)

        # Compute attention scores: [batch, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for num_heads dimension if needed
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)

            # Convert mask to attention bias (0 -> 0, 1 -> -inf for masked positions)
            # Assume mask is 1 for attended positions, 0 for masked
            attention_mask = attention_mask.to(dtype=attention_scores.dtype)
            attention_scores = attention_scores + (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min

        # Softmax and apply to values
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Compute attention output: [batch, num_heads, seq_len, head_dim]
        attention_output = torch.matmul(attention_probs, value_states)

        # Reshape back to [batch, seq_len, hidden_size]
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        # Output projection
        attention_output = self.o_proj(attention_output)

        return attention_output


class LlamaMLP(nn.Module):
    """LLaMA MLP with SwiGLU activation."""

    def __init__(self, config, linear_cls=HBitLinear):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = linear_cls(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = linear_cls(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = linear_cls(config.hidden_size, config.intermediate_size, bias=False)

        self.pretraining_tp = getattr(config, "pretraining_tp", 1)

    def forward(self, hidden_states):
        if self.pretraining_tp > 1:
            slice_size = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_size, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_size, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_size, dim=1)

            gate_proj = torch.cat(
                [F.linear(hidden_states, gate_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(hidden_states, up_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )

            # SwiGLU activation
            intermediate_states = F.silu(gate_proj) * up_proj
            intermediate_states = act_quant_8bit(intermediate_states)

            intermediate_slices = intermediate_states.split(slice_size, dim=2)
            down_proj = sum(
                F.linear(intermediate_slices[i], down_proj_slices[i])
                for i in range(self.pretraining_tp)
            )
        else:
            # Standard forward pass with SwiGLU
            gate_proj = self.gate_proj(hidden_states)
            up_proj = self.up_proj(hidden_states)

            # SwiGLU: silu(gate) * up
            hidden_gelu = F.silu(gate_proj) * up_proj
            hidden_gelu = act_quant_8bit(hidden_gelu)

            down_proj = self.down_proj(hidden_gelu)

        down_proj = act_quant_4bit(down_proj)
        return down_proj


class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config, experiment=False, linear_cls=HBitLinear):
        super().__init__()
        self.self_attn = LlamaAttention(config, linear_cls=linear_cls)
        self.mlp = LlamaMLP(config, linear_cls=linear_cls)

        # Use RMSNorm like original LLaMA
        self.input_layernorm = RMSNormLayer(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-5))
        self.post_attention_layernorm = RMSNormLayer(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-5))

        self.experiment = experiment

    def forward(self, hidden_states, attention_mask, cos, sin):
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention with gradient checkpointing
        hidden_states = custom_checkpoint(
            self.self_attn, hidden_states, attention_mask, cos, sin
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = custom_checkpoint(self.mlp, hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    """LLaMA model with 1.58-bit quantization."""

    def __init__(self, config, experiment=False, linear_cls=HBitLinear):
        super().__init__()
        self.config = config

        self.embed_tokens = QuantizedEmbedding(
            config.vocab_size, config.hidden_size, experiment=experiment
        )

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, experiment=experiment, linear_cls=linear_cls)
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNormLayer(config.hidden_size, eps=getattr(config, "rms_norm_eps", 1e-5))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Move to best available device
        device = get_device()
        self.to(device)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len] or [batch, seq_len, seq_len]
            **kwargs: Additional arguments (cos, sin for pre-computed RoPE)

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        hidden_states = self.embed_tokens(input_ids)

        # Get or compute rotary embeddings
        cos = kwargs.get("cos", None)
        sin = kwargs.get("sin", None)

        if cos is None or sin is None:
            seq_length = input_ids.size(1)
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            cos, sin = compute_rotary_embeddings(seq_length, head_dim, input_ids.device)

        # Process through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, cos, sin)

        # Final normalization and LM head
        hidden_states = self.norm(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        return lm_logits

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=20,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ):
        """Simple autoregressive generation returning token ids."""
        device = self.lm_head.weight.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        generated = input_ids

        for _ in range(max_length):
            logits = self.forward(generated, attention_mask)
            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                if top_k is not None and top_k > 0:
                    k = min(int(top_k), next_token_logits.size(-1))
                    values, indices = torch.topk(next_token_logits, k)
                    logits_mask = torch.full_like(next_token_logits, float("-inf"))
                    logits_mask.scatter_(-1, indices, values)
                    next_token_logits = logits_mask

                if top_p is not None and 0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_logits[sorted_indices_to_remove] = float("-inf")
                    logits_mask = torch.full_like(next_token_logits, float("-inf"))
                    logits_mask.scatter_(-1, sorted_indices, sorted_logits)
                    next_token_logits = logits_mask

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if (
                hasattr(self.config, "eos_token_id")
                and self.config.eos_token_id is not None
                and (next_token == self.config.eos_token_id).all()
            ):
                break

            if attention_mask is not None:
                new_mask = torch.ones(
                    (attention_mask.size(0), 1),
                    device=device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        return generated

    @classmethod
    def load_pretrained(cls, model_path, linear_cls=HBitLinear):
        """Load a pretrained model from a directory."""
        if LlamaConfig is None:
            raise ImportError("transformers is required to load pretrained models")
        from quantized_model_io import load_quantized_model

        config = LlamaConfig.from_pretrained(model_path)
        model = cls(config, linear_cls=linear_cls)
        load_quantized_model(model, model_path)
        return model

    def save_pretrained(self, save_directory):
        """Save the model and config to a directory."""
        if AutoTokenizer is None:
            raise ImportError("transformers is required to save pretrained models")
        from quantized_model_io import save_quantized_model

        os.makedirs(save_directory, exist_ok=True)

        # Update config
        self.config.hidden_size = self.embed_tokens.embedding_dim
        self.config.num_attention_heads = self.config.hidden_size // self.layers[0].self_attn.head_dim
        self.config.num_hidden_layers = len(self.layers)
        self.config.intermediate_size = self.layers[0].mlp.intermediate_size
        self.config.vocab_size = self.embed_tokens.num_embeddings

        if hasattr(self.config, "num_key_value_heads"):
            self.config.num_key_value_heads = self.config.num_attention_heads

        self.config.use_cache = True
        self.config.tie_word_embeddings = False
        self.config.model_type = "llama"
        self.config.torch_dtype = str(self.embed_tokens.weight.dtype).split(".")[-1]

        # Save config
        self.config.save_pretrained(save_directory)

        # Try to save tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("DeepInfra/Llama-2-70b-chat-tokenizer")
            tokenizer.save_pretrained(save_directory)
        except Exception:
            pass  # Tokenizer download may fail

        # Save model weights
        save_quantized_model(self, save_directory)

    def save_sharded_safetensors(self, output_path, shard_size=9 * 1024 * 1024 * 1024):
        """Save model in sharded safetensors format."""
        if save_file is None:
            raise ImportError("safetensors is required to save sharded weights")

        state_dict = self.state_dict()
        num_shards = math.ceil(
            sum(v.numel() * v.element_size() for v in state_dict.values()) / shard_size
        )

        os.makedirs(output_path, exist_ok=True)

        shard_id = 1
        shard_state_dict = {}
        shard_size_bytes = 0

        for key, value in state_dict.items():
            shard_state_dict[key] = value
            shard_size_bytes += value.numel() * value.element_size()

            if shard_size_bytes >= shard_size:
                shard_file = os.path.join(
                    output_path, f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors"
                )
                save_file(shard_state_dict, shard_file)
                print(f"Saved shard {shard_id} at: {shard_file}")

                shard_id += 1
                shard_state_dict = {}
                shard_size_bytes = 0

        if shard_state_dict:
            shard_file = os.path.join(
                output_path, f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors"
            )
            save_file(shard_state_dict, shard_file)
            print(f"Saved shard {shard_id} at: {shard_file}")

    def create_additional_files(self, save_directory, model_path, state_dicts, num_shards):
        """Create index and generation config files."""
        weight_map = {}
        total_size = 0
        for shard_id, state_dict in enumerate(state_dicts, start=1):
            shard_file = f"model-{shard_id:05d}-of-{num_shards:05d}.safetensors"
            for key in state_dict.keys():
                weight_map[key] = shard_file
            for v in state_dict.values():
                total_size += math.ceil(v.numel() * 1.58 / 8) + v.dim() * 4

        index_data = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=4)

        generation_config = {
            "max_length": 4096,
            "min_length": 0,
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "num_return_sequences": 1,
        }
        with open(os.path.join(save_directory, "generation_config.json"), "w") as f:
            json.dump(generation_config, f, indent=4)
