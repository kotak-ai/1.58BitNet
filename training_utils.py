import torch
import math


def cosine_lr_wd(step: int, total_steps: int, base_lr: float, base_wd: float, warmup_ratio: float = 0.03):
    """Return learning rate and weight decay for ``step`` using cosine schedule."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
        wd = base_wd
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        wd = base_wd * (1 - progress)
    return lr, max(wd, 0.0)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: str) -> None:
    """Save model and optimizer state along with current step."""
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
    }, path)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> int:
    """Load states from ``path`` and return stored step."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    return int(ckpt.get("step", 0))
