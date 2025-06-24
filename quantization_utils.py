import torch
import numpy as np


def RMSNorm(x, eps: float = 1e-6):
    """Compute RMS normalization used before quantization."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

def activation_norm_quant(x):
    x = RMSNorm(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale

def act_quant_8bit(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y

def act_quant_4bit(x):
    scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-8, 7)
    return y

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

def quantize_tensor(x: torch.Tensor, eps: float = 1e-5):
    gamma = x.abs().mean()
    quantized_x = torch.clamp(torch.round(x / (gamma + eps)), -1, 1).to(torch.int8)
    return quantized_x

def quantize_tensor_1_58bit(x: torch.Tensor, eps: float = 1e-5):
    """Ternary quantization with a mean absolute scale.

    Returns the quantized tensor and the scale used for reconstruction.
    """
    scale = x.abs().mean().clamp(min=eps)
    q = torch.round(x / scale).clamp_(-1, 1).to(torch.int8)
    return q, scale

def kv_cache_quant(x):
    scale = 15.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-16, 15)
    return y

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
