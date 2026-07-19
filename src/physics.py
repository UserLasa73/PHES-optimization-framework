"""Core physics functions for the PHES simulator."""

from __future__ import annotations

import math

from src.constants import (
    GRAVITY,
    KINEMATIC_VISCOSITY,
    MAX_PIPE_VELOCITY_MPS,
    MINOR_LOSS_FRACTION,
    PIPE_LENGTH_TO_HEAD_RATIO,
    PUMP_EFFICIENCY,
    TURBINE_EFFICIENCY,
    WATER_DENSITY,
)


def calculate_water_velocity(flow_rate_m3s: float, pipe_diameter_m: float) -> float:
    if flow_rate_m3s <= 0 or pipe_diameter_m <= 0:
        return 0.0
    area = math.pi * (pipe_diameter_m / 2.0) ** 2
    return flow_rate_m3s / area


def calculate_darcy_weisbach_loss(
    flow_rate_m3s: float,
    diameter_m: float,
    length_m: float,
    roughness_m: float = 0.00015,
) -> float:
    """Calculate major head loss using Darcy-Weisbach and Swamee-Jain."""
    if flow_rate_m3s <= 0 or diameter_m <= 0 or length_m <= 0:
        return 0.0

    velocity = calculate_water_velocity(flow_rate_m3s, diameter_m)
    reynolds = velocity * diameter_m / KINEMATIC_VISCOSITY
    if reynolds <= 0:
        return 0.0

    if reynolds < 2000:
        friction_factor = 64.0 / reynolds
    elif reynolds < 4000:
        friction_factor = 0.03
    else:
        term = roughness_m / (3.7 * diameter_m) + 5.74 / (reynolds**0.9)
        friction_factor = 0.25 / (math.log10(term) ** 2)

    return friction_factor * (length_m / diameter_m) * (velocity**2) / (2.0 * GRAVITY)


def calculate_total_head_loss(
    flow_rate_m3s: float,
    pipe_diameter_m: float,
    gross_head_m: float,
    roughness_m: float = 0.00015,
) -> float:
    if flow_rate_m3s <= 0:
        return 0.0
    pipe_length_m = gross_head_m * PIPE_LENGTH_TO_HEAD_RATIO
    major = calculate_darcy_weisbach_loss(
        flow_rate_m3s, pipe_diameter_m, pipe_length_m, roughness_m
    )
    return major * (1.0 + MINOR_LOSS_FRACTION)


def calculate_pump_power_from_flow(
    flow_rate_m3s: float,
    total_dynamic_head_m: float,
    pump_efficiency: float = PUMP_EFFICIENCY,
) -> float:
    """Electrical pump input power in kW for a given flow and total head."""
    if flow_rate_m3s <= 0 or total_dynamic_head_m <= 0 or pump_efficiency <= 0:
        return 0.0
    return (
        WATER_DENSITY * GRAVITY * total_dynamic_head_m * flow_rate_m3s
        / pump_efficiency
        / 1000.0
    )


def calculate_turbine_power(
    flow_rate_m3s: float,
    net_head_m: float,
    turbine_efficiency: float = TURBINE_EFFICIENCY,
) -> float:
    """Electrical turbine output power in kW."""
    if flow_rate_m3s <= 0 or net_head_m <= 0 or turbine_efficiency <= 0:
        return 0.0
    return (
        WATER_DENSITY
        * GRAVITY
        * net_head_m
        * flow_rate_m3s
        * turbine_efficiency
        / 1000.0
    )


def calculate_pump_flow_rate(
    pump_power_kw: float,
    gross_head_m: float,
    pipe_diameter_m: float | None = None,
    roughness_m: float = 0.00015,
    max_velocity_mps: float = MAX_PIPE_VELOCITY_MPS,
) -> float:
    """Solve pump flow under power, friction, and velocity constraints.

    Pumping must overcome ``gross_head + head_loss``. A bisection solution is
    inexpensive and avoids the original sign error that subtracted friction.
    """
    if pump_power_kw <= 0 or gross_head_m <= 0:
        return 0.0

    no_loss_upper = (
        pump_power_kw * 1000.0 * PUMP_EFFICIENCY
        / (WATER_DENSITY * GRAVITY * gross_head_m)
    )
    if pipe_diameter_m is None:
        return max(0.0, no_loss_upper)

    area = math.pi * (pipe_diameter_m / 2.0) ** 2
    upper = min(no_loss_upper, area * max_velocity_mps)
    lower = 0.0

    for _ in range(28):
        mid = (lower + upper) / 2.0
        loss = calculate_total_head_loss(mid, pipe_diameter_m, gross_head_m, roughness_m)
        required_kw = calculate_pump_power_from_flow(mid, gross_head_m + loss)
        if required_kw <= pump_power_kw:
            lower = mid
        else:
            upper = mid
    return lower


def calculate_turbine_flow_for_power(
    target_power_kw: float,
    gross_head_m: float,
    pipe_diameter_m: float,
    roughness_m: float = 0.00015,
    max_velocity_mps: float = MAX_PIPE_VELOCITY_MPS,
) -> float:
    """Return the minimum flow that approximately meets a requested output.

    The search is performed on the increasing branch of the turbine power curve.
    If the requested output is not hydraulically attainable, the flow that gives
    the maximum attainable power is returned.
    """
    if target_power_kw <= 0 or gross_head_m <= 0 or pipe_diameter_m <= 0:
        return 0.0

    area = math.pi * (pipe_diameter_m / 2.0) ** 2
    q_velocity = area * max_velocity_mps

    def output_kw(q: float) -> float:
        loss = calculate_total_head_loss(q, pipe_diameter_m, gross_head_m, roughness_m)
        return calculate_turbine_power(q, gross_head_m - loss)

    # Locate the peak on a small deterministic grid, then bisect the rising branch.
    grid_n = 32
    grid = [q_velocity * i / grid_n for i in range(grid_n + 1)]
    powers = [output_kw(q) for q in grid]
    peak_index = max(range(len(powers)), key=powers.__getitem__)
    peak_q = grid[peak_index]
    peak_power = powers[peak_index]
    if peak_power <= 0:
        return 0.0
    if target_power_kw >= peak_power:
        return peak_q

    lower = 0.0
    upper = peak_q
    for _ in range(28):
        mid = (lower + upper) / 2.0
        if output_kw(mid) < target_power_kw:
            lower = mid
        else:
            upper = mid
    return upper


def check_velocity_limit(
    flow_rate_m3s: float,
    pipe_diameter_m: float,
    max_velocity: float = MAX_PIPE_VELOCITY_MPS,
) -> bool:
    return calculate_water_velocity(flow_rate_m3s, pipe_diameter_m) <= max_velocity


def estimate_reservoir_surface_area(gross_capacity_m3: float, depth_m: float = 3.0) -> float:
    if gross_capacity_m3 <= 0 or depth_m <= 0:
        return 0.0
    return gross_capacity_m3 / depth_m


def calculate_evaporation_loss(
    surface_area_m2: float,
    evaporation_rate_mm_month: float,
    hours: float,
) -> float:
    """Calculate water loss in m³ from a fixed reservoir surface area."""
    if surface_area_m2 <= 0 or evaporation_rate_mm_month <= 0 or hours <= 0:
        return 0.0
    evaporation_m_per_hour = (evaporation_rate_mm_month / 1000.0) / (30.44 * 24.0)
    return surface_area_m2 * evaporation_m_per_hour * hours


def get_seepage_loss_factor(reservoir_type: str) -> float:
    """Monthly fractional seepage assumption used by the preliminary model."""
    return {
        "new_tank": 0.00,
        "excavated": 0.05,
        "pond": 0.10,
        "river": 0.20,
    }.get(reservoir_type, 0.00)


def calculate_stored_energy(
    usable_water_volume_m3: float,
    head_m: float,
    turbine_efficiency: float = TURBINE_EFFICIENCY,
) -> float:
    """Usable electrical energy obtainable from stored water, in kWh."""
    if usable_water_volume_m3 <= 0 or head_m <= 0:
        return 0.0
    joules = WATER_DENSITY * GRAVITY * head_m * usable_water_volume_m3
    return joules * turbine_efficiency / 3_600_000.0


def calculate_round_trip_efficiency(
    pumped_energy_kwh: float,
    recovered_or_stored_energy_kwh: float,
) -> float:
    """Return efficiency without silently clipping physically invalid values."""
    if pumped_energy_kwh <= 0:
        return 0.0
    return 100.0 * recovered_or_stored_energy_kwh / pumped_energy_kwh


def calculate_autonomy_days(stored_energy_kwh: float, daily_load_kwh: float) -> float:
    if daily_load_kwh <= 0:
        return 0.0
    return stored_energy_kwh / daily_load_kwh
