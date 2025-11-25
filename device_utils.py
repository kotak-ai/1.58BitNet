"""Device detection and management utilities."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device: The optimal device for computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name() -> str:
    """Return a human-readable name for the current device."""
    device = get_device()
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


def move_to_device(model: torch.nn.Module, device: torch.device = None) -> torch.nn.Module:
    """Move a model to the specified device or best available.

    Args:
        model: PyTorch model to move
        device: Target device (uses get_device() if None)

    Returns:
        The model on the target device
    """
    if device is None:
        device = get_device()
    return model.to(device)


def ensure_same_device(*tensors) -> torch.device:
    """Ensure all tensors are on the same device, returning that device.

    Raises ValueError if tensors are on different devices.
    """
    devices = set(t.device for t in tensors if t is not None)
    if len(devices) > 1:
        raise ValueError(f"Tensors are on different devices: {devices}")
    return devices.pop() if devices else get_device()


def log_device_info():
    """Print information about the available compute devices."""
    print(f"Selected device: {get_device_name()}")
    print(f"  Device type: {get_device().type}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  MPS (Metal Performance Shaders) is available")
