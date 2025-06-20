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


def gemm_lowbit(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Matrix multiply for int8 tensors with float32 accumulation.

    This function works on CPU, CUDA and MPS by dispatching to a custom
    :class:`torch.autograd.Function`.
    """

    if x.dtype != torch.int8 or w.dtype != torch.int8:
        raise TypeError("gemm_lowbit expects int8 inputs")
    return _LowBitMatMul.apply(x, w)

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

#def pack_quantized_tensor(quantized_tensor: torch.Tensor):
#    padded_length = (quantized_tensor.numel() + 4) // 5 * 5
#    padded_tensor = torch.full((padded_length,), -1, dtype=torch.int8)
#    padded_tensor[:quantized_tensor.numel()] = quantized_tensor.reshape(-1)
#    reshaped_tensor = padded_tensor.view(-1, 5)
#
#    unsigned_tensor = reshaped_tensor + 1
#    packed_data = torch.zeros(reshaped_tensor.size(0), dtype=torch.uint8)
#    shifts = torch.arange(0, 10, 2, dtype=torch.uint8)
#    for i in range(5):
#        packed_data |= unsigned_tensor[:, i] << shifts[i]
#
#    return packed_data

#def unpack_quantized_tensor(packed_data: torch.Tensor, original_shape):
#    unpacked_data = torch.zeros(packed_data.size(0), 5, dtype=torch.int8)
#    shifts = torch.arange(0, 10, 2, dtype=torch.uint8)
#    for i in range(5):
#        unpacked_data[:, i] = (packed_data >> shifts[i]) & 3
#
#    ternary_tensor = unpacked_data - 1
#    original_numel = np.prod(original_shape)
#    return ternary_tensor.view(-1)[:original_numel].view(original_shape)
