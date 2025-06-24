import os
import json
import math
import torch

try:
    from safetensors.torch import save_file, load_file
except Exception:  # pragma: no cover - optional dependency
    save_file = load_file = None  # type: ignore[misc]

from quantization_utils import quantize_tensor, pack_quantized_tensor, unpack_quantized_tensor


def quantize_state_dict(state_dict: dict[str, torch.Tensor]):
    """Quantise a model ``state_dict`` using ternary packing.

    Returns the packed tensors, a weight map and metadata containing the total
    byte size.
    """
    quantized: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    total_size = 0

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        if param.dtype == torch.uint8 or "weight_scale" in name or param.numel() == 1:
            quantized[name] = param
            size = param.numel() * param.element_size()
            total_size += size
            weight_map[name] = "model.safetensors"
            continue

        if param.dtype == torch.int8:
            packed, shape = pack_quantized_tensor(param)
        else:
            q = quantize_tensor(param)
            packed, shape = pack_quantized_tensor(q)

        quantized[name] = packed
        quantized[name + ".shape"] = shape
        numel = param.numel()
        size = math.ceil(numel * 1.58 / 8) + shape.numel() * shape.element_size()
        total_size += size
        weight_map[name] = "model.safetensors"
        weight_map[name + ".shape"] = "model.safetensors"

    metadata = {"total_size": total_size}
    return quantized, weight_map, metadata


def save_quantized_model(model: torch.nn.Module, save_directory: str) -> None:
    """Save ``model`` weights quantised to ternary format at ``save_directory``."""
    if save_file is None:
        raise ImportError("safetensors is required to save quantized models")

    os.makedirs(save_directory, exist_ok=True)
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    qsd, weight_map, meta = quantize_state_dict(state_dict)

    with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": meta, "weight_map": weight_map}, f, indent=2)

    save_file(qsd, os.path.join(save_directory, "model.safetensors"))


def load_quantized_model(model: torch.nn.Module, model_path: str) -> None:
    """Load ternary quantised weights from ``model_path`` into ``model``."""
    if load_file is None:
        raise ImportError("safetensors is required to load quantized models")

    sd = load_file(os.path.join(model_path, "model.safetensors"))
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if key.endswith(".shape"):
            continue
        shape_key = key + ".shape"
        if shape_key in sd:
            value = unpack_quantized_tensor(value, sd[shape_key])
        out[key.replace("model.", "")] = value

    model.load_state_dict(out)

