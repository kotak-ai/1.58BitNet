def hadamard(n):
    if n == 1: return torch.tensor([[1.]], dtype=torch.float32)
    H = hadamard(n//2)
    return torch.cat([torch.cat([H,H],dim=1),torch.cat([H,-H],dim=1)],dim=0)

import math
import torch
import torch.nn as nn
from quantization_utils import (
    act_quant_4bit,
    quantize_tensor_1_58bit,
    gemm_lowbit,
    pack_ternary,
    unpack_ternary,
)

class HBitLinear(nn.Linear):
    """Linear layer with Hadamard transform and packed 1.58-bit weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        if (in_features & (in_features - 1)) == 0:
            self.register_buffer(
                "hadamard",
                torch.tensor(hadamard(in_features), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.hadamard = None
        self.eps = 1e-5

        self.register_buffer("packed_weight", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("weight_scale", torch.tensor(1.0))

        self.pack()

    def pack(self):
        q, scale = quantize_tensor_1_58bit(self.weight, self.eps)
        self.packed_weight = pack_ternary(q)
        self.weight_scale = scale
        del self.weight
        self.register_parameter("weight", None)

    def unpack(self) -> torch.Tensor:
        return unpack_ternary(self.packed_weight, (self.out_features, self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.hadamard is not None:
            x = torch.matmul(x, self.hadamard) / math.sqrt(self.in_features)

        a_scale = 7.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
        x_q = (x * a_scale).round().clamp_(-8, 7).to(torch.int8)

        w_q = self.unpack()
        out_int = gemm_lowbit(x_q, w_q)
        out = out_int.to(torch.float32) * self.weight_scale / a_scale
        if self.bias is not None:
            out += self.bias
        return out


def convert_linear_to_hbitlinear(linear: nn.Linear) -> HBitLinear:
    """Create an :class:`HBitLinear` from a trained :class:`torch.nn.Linear`."""

    layer = HBitLinear(linear.in_features, linear.out_features, bias=linear.bias is not None)
    if linear.bias is not None:
        layer.bias.data.copy_(linear.bias.data)
    # Temporarily assign weight for packing
    layer.register_parameter("weight", nn.Parameter(linear.weight.detach().clone()))
    layer.pack()
    return layer
