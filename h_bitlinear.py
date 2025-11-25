import math
import torch
import torch.nn as nn
from quantization_utils import (
    ste_round_clamp,
    quantize_tensor_1_58bit,
    quantize_tensor_1_58bit_no_ste,
    gemm_lowbit,
    pack_ternary,
    unpack_ternary,
)


def hadamard(n):
    """Generate Hadamard matrix of size n x n (n must be power of 2)."""
    if n == 1:
        return torch.tensor([[1.0]], dtype=torch.float32)
    H = hadamard(n // 2)
    return torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)


class HBitLinear(nn.Module):
    """Linear layer with Hadamard transform and 1.58-bit quantized weights.

    This layer keeps weights as trainable parameters during training and uses
    Straight-Through Estimator (STE) for gradient flow through quantization.

    For inference, weights can be packed into a compact ternary format.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = 1e-5

        # Trainable weight parameter (NOT deleted!)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Hadamard matrix for power-of-2 input dimensions
        if (in_features & (in_features - 1)) == 0:
            self.register_buffer(
                "hadamard_matrix",
                hadamard(in_features),
                persistent=False,
            )
        else:
            self.hadamard_matrix = None

        # Buffers for packed weights (used only during inference after pack())
        self.register_buffer("packed_weight", None)
        self.register_buffer("weight_scale", None)
        self.register_buffer(
            "weight_shape",
            torch.tensor([out_features, in_features], dtype=torch.int32),
        )

        self._is_packed = False

    def pack(self):
        """Pack weights for inference. Call this before saving or inference."""
        if self._is_packed:
            return

        with torch.no_grad():
            q, scale = quantize_tensor_1_58bit_no_ste(self.weight.data, self.eps)
            self.packed_weight = pack_ternary(q)
            self.weight_scale = scale

        self._is_packed = True

    def unpack(self) -> torch.Tensor:
        """Unpack weights from ternary format."""
        if self.packed_weight is None:
            raise ValueError("Weights are not packed")
        return unpack_ternary(self.packed_weight, tuple(self.weight_shape.tolist()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization.

        During training: uses STE for gradient flow
        During inference: uses packed weights if available
        """
        # Apply Hadamard transform if available
        if self.hadamard_matrix is not None:
            x = torch.matmul(x, self.hadamard_matrix.to(x.device)) / math.sqrt(self.in_features)

        if self.training or not self._is_packed:
            # Training mode: use STE for gradients through quantization
            # Quantize activations
            a_scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
            x_q = ste_round_clamp(x * a_scale, -8, 7)

            # Quantize weights with STE
            w_scale = self.weight.abs().mean().clamp(min=self.eps)
            w_q = ste_round_clamp(self.weight / w_scale, -1, 1)

            # Matrix multiply and rescale
            out = torch.nn.functional.linear(x_q, w_q)
            out = out * w_scale / a_scale

        else:
            # Inference mode with packed weights
            a_scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
            x_q = (x * a_scale).round().clamp_(-8, 7).to(torch.int8)

            out_int = gemm_lowbit(x_q, self.packed_weight, self.weight_shape)
            out = out_int.to(torch.float32) * self.weight_scale / a_scale

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, packed={self._is_packed}"
        )


def convert_linear_to_hbitlinear(linear: nn.Linear) -> HBitLinear:
    """Create an :class:`HBitLinear` from a trained :class:`torch.nn.Linear`.

    The weights are copied but NOT packed - call pack() explicitly if needed.
    """
    layer = HBitLinear(
        linear.in_features, linear.out_features, bias=linear.bias is not None
    )
    with torch.no_grad():
        layer.weight.copy_(linear.weight)
        if linear.bias is not None:
            layer.bias.copy_(linear.bias)
    return layer


def pack_model_weights(model: nn.Module) -> None:
    """Pack all HBitLinear weights in a model for inference."""
    for module in model.modules():
        if isinstance(module, HBitLinear):
            module.pack()


def unpack_model_weights(model: nn.Module) -> None:
    """Unpack all HBitLinear weights in a model (restore trainable weights)."""
    for module in model.modules():
        if isinstance(module, HBitLinear):
            if module._is_packed and module.packed_weight is not None:
                with torch.no_grad():
                    unpacked = module.unpack().float() * module.weight_scale
                    module.weight.data.copy_(unpacked)
                module._is_packed = False
