import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Return the total number of parameters in ``model``."""
    return sum(p.numel() for p in model.parameters())


def verify_parameter_count(actual: int, requested: int, tolerance_ratio: float = 0.01) -> None:
    """Ensure ``actual`` matches ``requested`` within ``tolerance_ratio``."""
    diff = abs(actual - requested)
    tolerance = requested * tolerance_ratio
    print(f"Requested parameters: {requested:,}; actual: {actual:,}")
    if diff > tolerance:
        raise ValueError(
            f"Parameter mismatch exceeds {tolerance_ratio * 100:.2f}% tolerance"
        )

