"""Re-evaluate exported ML Pareto solutions with the physics simulator.

Usage:
    python -m scripts.validate_pareto path/to/phes_pareto_designs.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator import PumpedHydroSimulator
from src.solar_data_loader import fetch_load_data, fetch_solar_data
from src.user_inputs import UserInputs

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

REQUIRED_COLUMNS = {
    "volume_m3",
    "head_m",
    "pipe_diameter_m",
    "pump_power_kw",
    "turbine_power_kw",
    "efficiency",
    "cost",
    "location",
    "latitude",
    "longitude",
    "pv_kwp",
    "daily_energy_kwh",
    "required_autonomy_days",
    "reservoir_type",
    "evaporation_rate_mm_month",
    "pipe_roughness_m",
}


def _scenario_key(row):
    return tuple(
        row[name]
        for name in [
            "location",
            "latitude",
            "longitude",
            "pv_kwp",
            "daily_energy_kwh",
            "required_autonomy_days",
            "reservoir_type",
            "evaporation_rate_mm_month",
            "pipe_roughness_m",
        ]
    )


def _make_user(row):
    user = UserInputs()
    user.location = str(row["location"])
    user.latitude = float(row["latitude"])
    user.longitude = float(row["longitude"])
    user.pv_kwp = float(row["pv_kwp"])
    user.daily_energy_kwh = float(row["daily_energy_kwh"])
    user.autonomy_days = float(row["required_autonomy_days"])
    user.upper_reservoir_type = str(row["reservoir_type"])
    user.lower_reservoir_type = str(row["reservoir_type"])
    user.evaporation_rate_mm_month = float(row["evaporation_rate_mm_month"])
    user.pipe_roughness_m = float(row["pipe_roughness_m"])
    user.demand_spike_factor = 1.0
    user.has_grid_backup = False
    user.year = 2021
    user.tilt_angle = 10.0
    user.azimuth_angle = 180.0
    return user


def validate(input_path: Path, output_path: Path):
    frame = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    profile_cache = {}
    rows = []
    for index, row in frame.iterrows():
        key = _scenario_key(row)
        if key not in profile_cache:
            user = _make_user(row)
            profile_cache[key] = (
                user,
                fetch_solar_data(user),
                fetch_load_data(user),
            )
        user, solar, load = profile_cache[key]
        design = {
            "volume_m3": float(row["volume_m3"]),
            "head_m": float(row["head_m"]),
            "pipe_diameter_m": float(row["pipe_diameter_m"]),
            "pump_power_kw": float(row["pump_power_kw"]),
            "turbine_power_kw": float(row["turbine_power_kw"]),
        }
        metrics = PumpedHydroSimulator(user, design).simulate(solar, load)["metrics"]
        predicted_efficiency = float(row["efficiency"])
        predicted_cost = float(row["cost"])
        physics_efficiency = float(metrics["efficiency_percent"])
        physics_cost = float(metrics["capital_cost_lkr"])
        physics_autonomy = float(metrics["autonomy_days"])

        rows.append(
            {
                "source_row": int(index),
                **design,
                "ml_efficiency_percent": predicted_efficiency,
                "physics_efficiency_percent": physics_efficiency,
                "efficiency_absolute_error_pp": abs(
                    predicted_efficiency - physics_efficiency
                ),
                "ml_cost_lkr": predicted_cost,
                "physics_cost_lkr": physics_cost,
                "cost_absolute_error_lkr": abs(predicted_cost - physics_cost),
                "physics_autonomy_days": physics_autonomy,
                "autonomy_requirement_met": physics_autonomy >= user.autonomy_days,
                "physics_load_served_ratio": metrics["load_served_ratio"],
                "physics_valid": metrics["is_physically_valid"],
                "water_balance_residual_m3": metrics["water_balance_residual_m3"],
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)

    valid = result[result["physics_valid"]]
    summary = {
        "input_rows": int(len(result)),
        "physically_valid_rows": int(len(valid)),
        "mean_efficiency_absolute_error_pp": float(
            result["efficiency_absolute_error_pp"].mean()
        ),
        "median_efficiency_absolute_error_pp": float(
            result["efficiency_absolute_error_pp"].median()
        ),
        "max_efficiency_absolute_error_pp": float(
            result["efficiency_absolute_error_pp"].max()
        ),
        "autonomy_feasible_fraction": float(
            result["autonomy_requirement_met"].mean()
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Detailed validation: {output_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIR / "pareto_physics_validation.csv",
    )
    args = parser.parse_args()
    validate(args.input_csv, args.output)


if __name__ == "__main__":
    main()
