import torch
import numpy as np


# === Straight-Through Estimator (STE) Functions ===

class _STERound(torch.autograd.Function):
    """Straight-Through Estimator for rounding.

    Forward pass: applies rounding
    Backward pass: passes gradients through unchanged (identity)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _STEClamp(torch.autograd.Function):
    """Straight-Through Estimator for clamping.

    Forward pass: applies clamping
    Backward pass: passes gradients through unchanged (identity)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        return torch.clamp(x, min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradient estimator."""
    return _STERound.apply(x)


def ste_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp with straight-through gradient estimator."""
    return _STEClamp.apply(x, min_val, max_val)


def ste_round_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Round and clamp with straight-through gradient estimator."""
    return ste_clamp(ste_round(x), min_val, max_val)


# === Normalization Functions ===

def RMSNorm(x, eps: float = 1e-6):
    """Compute RMS normalization used before quantization."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


# === Activation Quantization Functions ===

def activation_quant(x):
    """Quantize activations to int8 range with STE and dequantize back."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = ste_round_clamp(x * scale, -128, 127) / scale
    return y


def weight_quant(w):
    """Quantize weights to ternary (-1, 0, 1) with STE and dequantize back."""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = ste_round_clamp(w * scale, -1, 1) / scale
    return u


def activation_norm_quant(x):
    """Normalize and quantize activations, returning quantized values and scale."""
    x = RMSNorm(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = ste_round_clamp(x * scale, -128, 127)
    return y, scale


def act_quant_8bit(x):
    """Quantize activations to int8 range with STE and dequantize back.

    This function quantizes to [-128, 127] range and then dequantizes
    back to the original scale to preserve magnitude for downstream operations.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = ste_round_clamp(x * scale, -128, 127)
    return y / scale  # Dequantize back to original scale


def act_quant_4bit(x):
    """Quantize activations to int4 range with STE and dequantize back.

    This function quantizes to [-8, 7] range and then dequantizes
    back to the original scale to preserve magnitude for downstream operations.
    """
    scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = ste_round_clamp(x * scale, -8, 7)
    return y / scale  # Dequantize back to original scale


def act_quant_8bit_raw(x):
    """Quantize activations to int8 range, returning raw quantized values.

    Use this when you need the actual integer values (e.g., for integer GEMM).
    Returns tuple of (quantized_tensor, scale) so caller can dequantize.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = ste_round_clamp(x * scale, -128, 127)
    return y, scale


def act_quant_4bit_raw(x):
    """Quantize activations to int4 range, returning raw quantized values.

    Use this when you need the actual integer values (e.g., for integer GEMM).
    Returns tuple of (quantized_tensor, scale) so caller can dequantize.
    """
    scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = ste_round_clamp(x * scale, -8, 7)
    return y, scale


# === Low-Bit Matrix Multiplication ===

class _LowBitMatMul(torch.autograd.Function):
    """Low-bit matrix multiply supporting CPU, CUDA and MPS."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, w)

        # Perform integer GEMM on all devices.  PyTorch supports int32 matmul on
        # CPU and CUDA.  On MPS fall back to float computation after casting.
        if x.device.type == "mps":
            out = (x.to(torch.int32) @ w.to(torch.int32).t()).to(torch.float32)
        else:
            out = (x.to(torch.int32) @ w.to(torch.int32).t()).to(torch.float32)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = grad_w = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ w.to(torch.float32)
        if ctx.needs_input_grad[1]:
            grad_w = grad_output.t() @ x.to(torch.float32)

        return grad_x, grad_w


class _PackedLowBitMatMul(torch.autograd.Function):
    """Matrix multiply with packed 1.58-bit weights."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_w: torch.ByteTensor, shape: torch.Tensor) -> torch.Tensor:
        w = unpack_ternary(packed_w.to(x.device), tuple(shape.tolist()))
        ctx.save_for_backward(x, w)
        out = (x.to(torch.int32) @ w.to(torch.int32).t()).to(torch.float32)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = grad_w = grad_shape = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ w.to(torch.float32)
        return grad_x, None, None


def gemm_lowbit(x: torch.Tensor, w: torch.Tensor, weight_shape=None) -> torch.Tensor:
    """Matrix multiply for int8 tensors with float32 accumulation.

    This function works on CPU, CUDA and MPS by dispatching to a custom
    :class:`torch.autograd.Function`. When ``w`` is a packed uint8 tensor the
    original ``weight_shape`` must be provided and the multiplication is
    performed without explicit unpacking on the host.
    """

    if x.dtype != torch.int8:
        raise TypeError("gemm_lowbit expects int8 inputs")

    if w.dtype == torch.int8:
        if weight_shape is not None:
            raise TypeError("weight_shape should be None for int8 weights")
        return _LowBitMatMul.apply(x, w)
    elif w.dtype == torch.uint8:
        if weight_shape is None:
            raise TypeError("weight_shape required for packed weights")
        if not isinstance(weight_shape, torch.Tensor):
            weight_shape = torch.tensor(weight_shape, dtype=torch.int32, device=x.device)
        else:
            weight_shape = weight_shape.to(torch.int32).to(x.device)
        return _PackedLowBitMatMul.apply(x, w, weight_shape)
    else:
        raise TypeError("Unsupported weight dtype")


# === Tensor Quantization Functions ===

def quantize_tensor(x: torch.Tensor, eps: float = 1e-5):
    """Quantize tensor to ternary values (-1, 0, 1) using mean absolute scale."""
    gamma = x.abs().mean()
    quantized_x = ste_round_clamp(x / (gamma + eps), -1, 1).to(torch.int8)
    return quantized_x


def quantize_tensor_1_58bit(x: torch.Tensor, eps: float = 1e-5):
    """Ternary quantization with a mean absolute scale and STE.

    Returns the quantized tensor and the scale used for reconstruction.
    Uses Straight-Through Estimator to allow gradient flow during training.
    """
    scale = x.abs().mean().clamp(min=eps)
    q = ste_round_clamp(x / scale, -1, 1).to(torch.int8)
    return q, scale


def quantize_tensor_1_58bit_no_ste(x: torch.Tensor, eps: float = 1e-5):
    """Ternary quantization without STE (for inference only).

    Returns the quantized tensor and the scale used for reconstruction.
    This version uses regular round/clamp and should only be used during inference.
    """
    scale = x.abs().mean().clamp(min=eps)
    q = torch.round(x / scale).clamp_(-1, 1).to(torch.int8)
    return q, scale


# === KV Cache Quantization ===

def kv_cache_quant(x, training: bool = False):
    """Quantize key/value cache to 4-bit range.

    Args:
        x: Input tensor to quantize
        training: If True, uses STE and returns dequantized values.
                  If False, returns raw quantized values for inference.

    Returns:
        Quantized (and dequantized if training) tensor.
    """
    scale = 15.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    if training:
        # During training: use STE and dequantize to preserve gradients
        y = ste_round_clamp(x * scale, -16, 15)
        return y / scale
    else:
        # During inference: return quantized values
        y = torch.round(x * scale).clamp_(-16, 15)
        return y


def kv_cache_quant_with_scale(x):
    """Quantize key/value cache and return both quantized values and scale.

    Useful when you need to store the scale separately for later dequantization.
    """
    scale = 15.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = ste_round_clamp(x * scale, -16, 15)
    return y, scale


# === Ternary Packing Utilities ===

_TER_MULTS = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)


def pack_ternary(t: torch.Tensor) -> torch.ByteTensor:
    """Pack a tensor with values in ``{-1, 0, 1}`` into base-3 bytes.

    Five ternary values are stored per byte giving approximately 1.6 bits per
    value.  The original shape is not stored and must be provided when
    unpacking.
    """

    if t.dtype != torch.int8:
        raise TypeError("pack_ternary expects int8 input")

    flat = (t + 1).view(-1).to(torch.int32)
    pad = (-flat.numel()) % 5
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    flat = flat.view(-1, 5)
    factors = _TER_MULTS.to(flat.device)
    packed = (flat * factors).sum(dim=1).to(torch.uint8)
    return packed


def unpack_ternary(packed: torch.ByteTensor, shape) -> torch.Tensor:
    """Reverse :func:`pack_ternary` returning an ``int8`` tensor."""

    if packed.dtype != torch.uint8:
        raise TypeError("unpack_ternary expects uint8 input")

    code = packed.to(torch.int32)
    digits = []
    for _ in range(5):
        digits.append(code % 3)
        code //= 3
    digits = torch.stack(digits, dim=1)
    out = digits.view(-1)[: int(np.prod(shape))].to(torch.int8) - 1
    return out.view(shape)


def pack_quantized_tensor(t: torch.Tensor):
    """Pack a quantized int8 tensor and return the packed data with shape."""

    if t.dtype != torch.int8:
        raise TypeError("pack_quantized_tensor expects int8 input")

    packed = pack_ternary(t)
    shape = torch.tensor(t.shape, dtype=torch.int32)
    return packed, shape


def unpack_quantized_tensor(packed: torch.ByteTensor, shape: torch.Tensor) -> torch.Tensor:
    """Unpack data produced by :func:`pack_quantized_tensor`."""

    if packed.dtype != torch.uint8:
        raise TypeError("unpack_quantized_tensor expects uint8 data")
    if shape.dtype not in (torch.int32, torch.int64):
        raise TypeError("shape tensor must be int32 or int64")

    return unpack_ternary(packed, tuple(shape.tolist()))
