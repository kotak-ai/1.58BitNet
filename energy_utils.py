import numpy as np


def compute_energy_metrics(energy_level: float, energy_capacity: float,
                           energy_harvested: float, energy_consumed: float,
                           thermal_load: float, thermal_limit: float):
    """Calculate EVS, TRI and SHE metrics.

    Args:
        energy_level: current stored energy.
        energy_capacity: maximum energy capacity.
        energy_harvested: energy gained this step.
        energy_consumed: energy spent this step.
        thermal_load: current temperature.
        thermal_limit: maximum safe temperature.

    Returns:
        tuple: (evs, tri, she)
    """
    # Energetic Viability Score (0..1)
    evs = (energy_level + energy_harvested - energy_consumed) / (energy_capacity + 1e-5)
    evs = float(np.clip(evs, 0.0, 1.0))

    # Thermal Resilience Index (0..1)
    tri = 1.0 - thermal_load / (thermal_limit + 1e-5)
    tri = float(np.clip(tri, 0.0, 1.0))

    # Survival Horizon Expectation (time steps until depletion)
    delta = energy_consumed - energy_harvested
    if delta <= 0:
        she = float('inf')
    else:
        she = energy_level / (delta + 1e-5)
    return evs, tri, she


def energy_utility(evs: float, tri: float, she: float,
                   weights=(0.5, 0.3, 0.2), horizon_norm: float = 10.0):
    """Aggregate metrics into an energy-based utility value."""
    she_term = np.tanh((0 if she == float('inf') else she) / horizon_norm)
    return weights[0] * evs + weights[1] * tri + weights[2] * she_term
