"""Preliminary, monotonic PHES capital-cost model.

The model is intended for relative design comparison, not as a supplier quotation.
PV cost is excluded because the application assumes an existing PV system.
"""

from __future__ import annotations

from src.constants import PIPE_LENGTH_TO_HEAD_RATIO


def _tiered_reservoir_base_cost(volume_m3: float) -> float:
    """Return a monotonic base cost for one reservoir.

    Marginal rates decrease with scale. Unlike the previous implementation,
    crossing a volume threshold can never reduce total cost.
    """
    if volume_m3 < 0:
        raise ValueError("Reservoir volume cannot be negative.")

    remaining = float(volume_m3)
    cost = 0.0
    tiers = [
        (100.0, 8000.0),
        (200.0, 6500.0),
        (200.0, 5500.0),
        (float("inf"), 4500.0),
    ]
    for width, marginal_rate in tiers:
        used = min(remaining, width)
        cost += used * marginal_rate
        remaining -= used
        if remaining <= 0:
            break
    return cost


def _pipe_cost_per_m(pipe_diameter_m: float) -> float:
    """Approximate diameter-sensitive pipe cost per metre.

    The coefficient is provisional and must later be calibrated using local
    quotations. The important modelling correction is that larger pipes now
    cost more, preventing the optimizer from selecting maximum diameter for free.
    """
    if pipe_diameter_m <= 0:
        raise ValueError("Pipe diameter must be positive.")
    reference_diameter = 0.10
    reference_cost_per_m = 1500.0
    return reference_cost_per_m * (pipe_diameter_m / reference_diameter) ** 1.5


def calculate_capital_cost(
    volume_m3: float,
    head_m: float,
    pipe_diameter_m: float,
    pump_power_kw: float,
    turbine_power_kw: float,
    pv_kwp: float,
    upper_type: str,
    lower_type: str,
):
    """Estimate PHES capital cost in LKR.

    ``volume_m3`` is the combined gross capacity of both reservoirs. Each
    reservoir is assumed to provide half of that total installed capacity.
    ``pv_kwp`` is retained for API compatibility but PV cost is excluded.
    """
    del pv_kwp  # PV is explicitly outside this PHES-only cost model.

    if volume_m3 <= 0 or head_m <= 0:
        raise ValueError("Volume and head must be positive.")
    if pump_power_kw <= 0 or turbine_power_kw <= 0:
        raise ValueError("Pump and turbine power must be positive.")

    upper_volume = volume_m3 / 2.0
    lower_volume = volume_m3 / 2.0

    cost_factors = {
        "new_tank": 1.0,
        "excavated": 0.5,
        "pond": 0.3,
        "river": 0.2,
    }
    upper_factor = cost_factors.get(upper_type, 1.0)
    lower_factor = cost_factors.get(lower_type, 1.0)

    reservoir_cost = (
        _tiered_reservoir_base_cost(upper_volume) * upper_factor
        + _tiered_reservoir_base_cost(lower_volume) * lower_factor
    )

    # Provisional equipment rates retained from the original project.
    pump_cost = pump_power_kw * 8000.0
    turbine_cost = turbine_power_kw * 15000.0

    # One reversible/bidirectional penstock is assumed, consistent with the
    # hydraulic model. Diameter now affects cost.
    pipe_length_m = head_m * PIPE_LENGTH_TO_HEAD_RATIO
    pipe_cost = pipe_length_m * _pipe_cost_per_m(pipe_diameter_m)

    bos_cost = (pump_cost + turbine_cost + pipe_cost) * 0.20
    equipment_cost = reservoir_cost + pump_cost + turbine_cost + pipe_cost + bos_cost
    installation_cost = equipment_cost * 0.20
    total = equipment_cost + installation_cost

    return {
        "total_lkr": total,
        "total_usd": total / 300.0,
        "breakdown": {
            "reservoir_lkr": reservoir_cost,
            "pump_lkr": pump_cost,
            "turbine_lkr": turbine_cost,
            "pipe_lkr": pipe_cost,
            "bos_lkr": bos_cost,
            "installation_lkr": installation_cost,
        },
        "upper_volume_m3": upper_volume,
        "lower_volume_m3": lower_volume,
        "total_volume_m3": volume_m3,
        "pipe_length_m": pipe_length_m,
        "cost_model_note": "Preliminary comparative estimate; local quotations required.",
    }
