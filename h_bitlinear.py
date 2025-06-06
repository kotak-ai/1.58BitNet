def hadamard(n):
    if n == 1: return torch.tensor([[1.]], dtype=torch.float32)
    H = hadamard(n//2)
    return torch.cat([torch.cat([H,H],dim=1),torch.cat([H,-H],dim=1)],dim=0)

import math
import torch
import torch.nn as nn
from quantization_utils import act_quant_4bit, weight_quant

class HBitLinear(nn.Linear):
    """Linear layer with Hadamard transform for activations and 1-bit weights."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        # Precompute Hadamard matrix for input dimension if power of two
        if (in_features & (in_features - 1)) == 0:
            self.register_buffer('hadamard', torch.tensor(hadamard(in_features), dtype=torch.float32), persistent=False)
        else:
            self.hadamard = None
        self.eps = 1e-5

    def forward(self, x):
        if self.hadamard is not None:
            # Apply Hadamard transform
            x = torch.matmul(x, self.hadamard) / math.sqrt(self.in_features)
        # Quantize activations to 4-bit
        x = act_quant_4bit(x)
        # Quantize weights to 1-bit
        w = weight_quant(self.weight)
        return nn.functional.linear(x, w, self.bias)
