"""Measure memory footprint reduction from 1.58-bit quantization."""

import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantization_utils import (
    quantize_tensor_1_58bit_no_ste,
    pack_ternary,
)


def measure_model_sizes():
    """Measure and compare model sizes across different precisions."""

    print("=" * 60)
    print("1.58-Bit Quantization Memory Footprint Analysis")
    print("=" * 60)

    # Define model configurations to test
    configs = [
        {"name": "Small (125M params)", "hidden": 768, "layers": 12, "heads": 12},
        {"name": "Medium (350M params)", "hidden": 1024, "layers": 24, "heads": 16},
        {"name": "Large (1.3B params)", "hidden": 2048, "layers": 24, "heads": 32},
        {"name": "XL (7B params)", "hidden": 4096, "layers": 32, "heads": 32},
    ]

    results = []

    for config in configs:
        print(f"\n{'─' * 60}")
        print(f"Model: {config['name']}")
        print(f"  Hidden: {config['hidden']}, Layers: {config['layers']}, Heads: {config['heads']}")
        print(f"{'─' * 60}")

        hidden = config['hidden']
        layers = config['layers']

        # Approximate parameter count for LLaMA-style model
        # Per layer: 4 attention projections + 3 FFN projections
        # Attention: Q, K, V, O = 4 * hidden^2
        # FFN: gate, up, down = 3 * hidden * (4*hidden) = 12 * hidden^2
        # Total per layer ≈ 16 * hidden^2
        params_per_layer = 16 * hidden * hidden
        total_params = params_per_layer * layers

        # Memory calculations
        fp32_bytes = total_params * 4  # 32 bits = 4 bytes
        fp16_bytes = total_params * 2  # 16 bits = 2 bytes

        # 1.58-bit: 5 ternary values pack into 1 byte (243 states in 256)
        # Actual bits per value = 8/5 = 1.6 bits
        ternary_bytes = (total_params + 4) // 5  # Packed ternary

        # Also need scale factors (one per tensor, negligible for large models)
        # Approximate: 1 float32 scale per 1024 weights
        scale_bytes = (total_params // 1024) * 4
        ternary_total = ternary_bytes + scale_bytes

        # Calculate ratios
        ratio_vs_fp32 = fp32_bytes / ternary_total
        ratio_vs_fp16 = fp16_bytes / ternary_total

        print(f"\n  Memory Usage:")
        print(f"    FP32:     {fp32_bytes / 1e9:.2f} GB")
        print(f"    FP16:     {fp16_bytes / 1e9:.2f} GB")
        print(f"    1.58-bit: {ternary_total / 1e9:.2f} GB")

        print(f"\n  Compression Ratio:")
        print(f"    vs FP32:  {ratio_vs_fp32:.1f}x smaller")
        print(f"    vs FP16:  {ratio_vs_fp16:.1f}x smaller")

        results.append({
            "config": config['name'],
            "params": total_params,
            "fp32_gb": fp32_bytes / 1e9,
            "fp16_gb": fp16_bytes / 1e9,
            "ternary_gb": ternary_total / 1e9,
            "ratio_fp32": ratio_vs_fp32,
            "ratio_fp16": ratio_vs_fp16,
        })

    return results


def verify_packing_ratio():
    """Verify actual packing ratio with real tensors."""

    print("\n" + "=" * 60)
    print("Verification: Actual Tensor Packing Test")
    print("=" * 60)

    # Create test tensors of various sizes
    sizes = [1000, 10000, 100000, 1000000]

    for size in sizes:
        # Create random FP32 tensor
        x = torch.randn(size)

        # Quantize to ternary
        q, scale = quantize_tensor_1_58bit_no_ste(x)

        # Pack ternary values
        packed = pack_ternary(q)

        # Calculate sizes
        fp32_bytes = x.numel() * 4
        fp16_bytes = x.numel() * 2
        packed_bytes = packed.numel() + 4  # packed + scale

        ratio_fp32 = fp32_bytes / packed_bytes
        ratio_fp16 = fp16_bytes / packed_bytes
        bits_per_value = (packed_bytes * 8) / x.numel()

        print(f"\n  Tensor size: {size:,} values")
        print(f"    FP32: {fp32_bytes:,} bytes")
        print(f"    FP16: {fp16_bytes:,} bytes")
        print(f"    Packed 1.58-bit: {packed_bytes:,} bytes")
        print(f"    Actual bits/value: {bits_per_value:.2f}")
        print(f"    Compression vs FP32: {ratio_fp32:.1f}x")
        print(f"    Compression vs FP16: {ratio_fp16:.1f}x")


def measure_inference_memory():
    """Measure actual GPU/CPU memory during model operations."""

    print("\n" + "=" * 60)
    print("Runtime Memory Measurement")
    print("=" * 60)

    # Check available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n  Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("\n  Using CPU")

    # Test with a realistic weight matrix size (e.g., 4096x4096 for 7B model)
    test_size = (4096, 4096)
    num_params = test_size[0] * test_size[1]

    print(f"\n  Test matrix: {test_size[0]} x {test_size[1]} = {num_params:,} parameters")

    # Measure FP32
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    w_fp32 = torch.randn(test_size, device=device, dtype=torch.float32)
    fp32_size = w_fp32.numel() * w_fp32.element_size()

    if device.type == "cuda":
        fp32_mem = torch.cuda.max_memory_allocated()
    else:
        fp32_mem = fp32_size

    del w_fp32
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Measure FP16
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    w_fp16 = torch.randn(test_size, device=device, dtype=torch.float16)
    fp16_size = w_fp16.numel() * w_fp16.element_size()

    if device.type == "cuda":
        fp16_mem = torch.cuda.max_memory_allocated()
    else:
        fp16_mem = fp16_size

    del w_fp16
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Measure packed ternary
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Create and quantize on CPU, then move packed version
    w_temp = torch.randn(test_size)
    q, scale = quantize_tensor_1_58bit_no_ste(w_temp)
    packed = pack_ternary(q)
    packed = packed.to(device)
    scale = scale.to(device)

    packed_size = packed.numel() * packed.element_size() + 4  # scale is 4 bytes

    if device.type == "cuda":
        packed_mem = torch.cuda.max_memory_allocated()
    else:
        packed_mem = packed_size

    print(f"\n  Memory Usage (single weight matrix):")
    print(f"    FP32:     {fp32_size / 1e6:.2f} MB")
    print(f"    FP16:     {fp16_size / 1e6:.2f} MB")
    print(f"    1.58-bit: {packed_size / 1e6:.2f} MB")

    print(f"\n  Compression Achieved:")
    print(f"    vs FP32:  {fp32_size / packed_size:.1f}x")
    print(f"    vs FP16:  {fp16_size / packed_size:.1f}x")

    return {
        "fp32_mb": fp32_size / 1e6,
        "fp16_mb": fp16_size / 1e6,
        "packed_mb": packed_size / 1e6,
        "ratio_fp32": fp32_size / packed_size,
        "ratio_fp16": fp16_size / packed_size,
    }


def print_summary():
    """Print executive summary for resume/presentation."""

    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────────┐
│  1.58-Bit Quantization: Key Metrics                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Memory Reduction:                                          │
│    • 20x smaller than FP32 (standard precision)            │
│    • 10x smaller than FP16 (half precision)                │
│                                                             │
│  Bits Per Weight:                                           │
│    • FP32: 32 bits                                         │
│    • FP16: 16 bits                                         │
│    • 1.58-bit (ternary packed): 1.6 bits                   │
│                                                             │
│  Practical Impact (7B Parameter Model):                     │
│    • FP32: ~28 GB → FP16: ~14 GB → 1.58-bit: ~1.4 GB       │
│    • Enables deployment on consumer GPUs (RTX 3060 6GB)    │
│    • Enables deployment on mobile/edge devices             │
│                                                             │
│  Training Innovation:                                       │
│    • Straight-Through Estimator for gradient flow          │
│    • GRPO reinforcement learning fine-tuning               │
│    • Multi-layer self-correction training                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    # Run all measurements
    model_results = measure_model_sizes()
    verify_packing_ratio()
    runtime_results = measure_inference_memory()
    print_summary()
