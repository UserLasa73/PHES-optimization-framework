"""Shared helpers for ML and physics NSGA-II optimizers."""

from __future__ import annotations

import math

from deap import tools


def extract_nondominated(population):
    """Return the true first non-dominated front from feasible individuals."""
    feasible = [
        individual
        for individual in population
        if individual.fitness.valid
        and len(individual.fitness.values) == 2
        and math.isfinite(individual.fitness.values[0])
        and math.isfinite(individual.fitness.values[1])
        and individual.fitness.values[0] > -1.0e5
        and individual.fitness.values[1] < 1.0e11
    ]
    if not feasible:
        return []
    return tools.sortNondominated(
        feasible, len(feasible), first_front_only=True
    )[0]


def front_to_records(front):
    records = []
    seen = set()
    for individual in front:
        key = tuple(round(float(value), 8) for value in individual)
        if key in seen:
            continue
        seen.add(key)
        efficiency, cost = individual.fitness.values
        records.append(
            {
                "volume_m3": float(individual[0]),
                "head_m": float(individual[1]),
                "pipe_diameter_m": float(individual[2]),
                "pump_power_kw": float(individual[3]),
                "turbine_power_kw": float(individual[4]),
                "efficiency": float(efficiency),
                "cost": float(cost),
            }
        )
    return sorted(records, key=lambda row: (row["cost"], -row["efficiency"]))


def select_compromise_design(records):
    """Select a balanced point by normalized distance to the ideal point."""
    if not records:
        return None
    efficiencies = [row["efficiency"] for row in records]
    costs = [row["cost"] for row in records]
    eff_min, eff_max = min(efficiencies), max(efficiencies)
    cost_min, cost_max = min(costs), max(costs)

    def score(row):
        eff_gap = (
            (eff_max - row["efficiency"]) / (eff_max - eff_min)
            if eff_max > eff_min
            else 0.0
        )
        cost_gap = (
            (row["cost"] - cost_min) / (cost_max - cost_min)
            if cost_max > cost_min
            else 0.0
        )
        return (eff_gap**2 + cost_gap**2) ** 0.5

    return min(records, key=score)
