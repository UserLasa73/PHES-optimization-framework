"""Generate a leakage-resistant physics-simulation dataset for the surrogate."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.model_features import CODE_TO_RESERVOIR_TYPE, TRAINING_BOUNDS
from src.simulator import PumpedHydroSimulator
from src.solar_data_loader import fetch_load_data, fetch_solar_data
from src.user_inputs import UserInputs

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "training_data_all_inputs.csv"
DIAGNOSTIC_PATH = DATA_DIR / "training_data_diagnostics.csv"

LOCATIONS = [
    {
        "name": "Vavuniya",
        "lat": 8.7542,
        "lon": 80.4982,
    },
    {
        "name": "Colombo",
        "lat": 6.9271,
        "lon": 79.8612,
    },
    {
        "name": "Jaffna",
        "lat": 9.6615,
        "lon": 80.0255,
    },
]

SAMPLES_PER_LOCATION = 1000
BASE_SEED = 42

SAMPLED_VARIABLES = [
    "volume_m3",
    "head_m",
    "pipe_diameter_m",
    "pump_power_kw",
    "turbine_power_kw",
    "pv_kwp",
    "daily_energy_kwh",
    "evaporation_rate_mm_month",
    "reservoir_type_code",
]


def generate_lhs_samples(n_samples: int, seed: int) -> pd.DataFrame:
    """Generate an independent LHS design for one location."""
    sampler = qmc.LatinHypercube(d=len(SAMPLED_VARIABLES), seed=seed)
    unit = sampler.random(n=n_samples)
    scaled = np.zeros_like(unit)

    # Use log scaling only for broad positive engineering ranges.
    log_scaled = {"volume_m3", "pump_power_kw", "turbine_power_kw"}
    for index, name in enumerate(SAMPLED_VARIABLES):
        low, high = TRAINING_BOUNDS[name]
        if name in log_scaled:
            scaled[:, index] = 10 ** (
                np.log10(low) + unit[:, index] * (np.log10(high) - np.log10(low))
            )
        elif name == "reservoir_type_code":
            scaled[:, index] = np.floor(low + unit[:, index] * (high - low + 1))
            scaled[:, index] = np.clip(scaled[:, index], low, high)
        else:
            scaled[:, index] = low + unit[:, index] * (high - low)

    frame = pd.DataFrame(scaled, columns=SAMPLED_VARIABLES)
    frame["reservoir_type_code"] = frame["reservoir_type_code"].astype(int)
    return frame


def _base_user(location: dict) -> UserInputs:
    user = UserInputs()
    user.location = location["name"]
    user.latitude = location["lat"]
    user.longitude = location["lon"]
    user.year = 2023
    user.tilt_angle = 10.0
    user.azimuth_angle = 180.0
    user.autonomy_days = 0.0
    user.demand_spike_factor = 1.0
    user.has_grid_backup = False
    user.random_seed = BASE_SEED
    return user


def generate_dataset() -> pd.DataFrame:
    total = SAMPLES_PER_LOCATION * len(LOCATIONS)
    print(f"Generating {total} simulations across {len(LOCATIONS)} locations...")
    started = time.perf_counter()
    rows = []

    for location_index, location in enumerate(LOCATIONS):
        print(f"[{location_index + 1}/{len(LOCATIONS)}] {location['name']}")
        samples = generate_lhs_samples(
            SAMPLES_PER_LOCATION, seed=BASE_SEED + location_index
        )
        user = _base_user(location)

        # PV output scales linearly with installed kWp, so calculate the site's
        # one-kWp profile only once. This is both correct and much faster.
        solar_per_kwp = np.asarray(fetch_solar_data(user, capacity_kwp=1.0), dtype=float)
        annual_yield = float(solar_per_kwp.sum())

        for sample_index, sample in samples.iterrows():
            user.pv_kwp = float(sample["pv_kwp"])
            user.daily_energy_kwh = float(sample["daily_energy_kwh"])
            user.evaporation_rate_mm_month = float(
                sample["evaporation_rate_mm_month"]
            )
            reservoir_code = int(sample["reservoir_type_code"])
            reservoir_type = CODE_TO_RESERVOIR_TYPE[reservoir_code]
            user.upper_reservoir_type = reservoir_type
            user.lower_reservoir_type = reservoir_type

            solar = (solar_per_kwp * user.pv_kwp).tolist()
            load = fetch_load_data(user)
            design = {
                "volume_m3": float(sample["volume_m3"]),
                "head_m": float(sample["head_m"]),
                "pipe_diameter_m": float(sample["pipe_diameter_m"]),
                "pump_power_kw": float(sample["pump_power_kw"]),
                "turbine_power_kw": float(sample["turbine_power_kw"]),
            }
            metrics = PumpedHydroSimulator(user, design).simulate(solar, load)["metrics"]

            rows.append(
                {
                    **design,
                    "pv_kwp": user.pv_kwp,
                    "daily_energy_kwh": user.daily_energy_kwh,
                    "evaporation_rate_mm_month": user.evaporation_rate_mm_month,
                    "reservoir_type_code": reservoir_code,
                    "latitude": user.latitude,
                    "longitude": user.longitude,
                    "annual_solar_yield_kwh_per_kwp": annual_yield,
                    "location": user.location,
                    "design_id": f"{location_index:02d}-{sample_index:04d}",
                    "efficiency": metrics["efficiency_percent"],
                    "realized_efficiency": metrics["realized_efficiency_percent"],
                    "cost": metrics["capital_cost_lkr"],
                    "autonomy": metrics["autonomy_days"],
                    "load_served_ratio": metrics["load_served_ratio"],
                    "pumped": metrics["total_pumped_kwh"],
                    "generated": metrics["total_generated_kwh"],
                    "unmet": metrics["total_unmet_kwh"],
                    "curtailed": metrics["total_curtailed_kwh"],
                    "water_balance_residual_m3": metrics[
                        "water_balance_residual_m3"
                    ],
                    "is_physically_valid": metrics["is_physically_valid"],
                }
            )

            if (sample_index + 1) % 100 == 0:
                print(f"  completed {sample_index + 1}/{SAMPLES_PER_LOCATION}")

    full = pd.DataFrame(rows)
    full.to_csv(DIAGNOSTIC_PATH, index=False)

    valid = full[
        full["is_physically_valid"]
        & np.isfinite(full["efficiency"])
        & np.isfinite(full["autonomy"])
        & (full["efficiency"] >= 0.0)
        & (full["efficiency"] <= 100.0)
        & (full["pumped"] > 0.0)
    ].copy()

    training_columns = [
        "volume_m3",
        "head_m",
        "pipe_diameter_m",
        "pump_power_kw",
        "turbine_power_kw",
        "pv_kwp",
        "daily_energy_kwh",
        "evaporation_rate_mm_month",
        "reservoir_type_code",
        "latitude",
        "longitude",
        "annual_solar_yield_kwh_per_kwp",
        "location",
        "design_id",
        "efficiency",
        "cost",
        "autonomy",
        "load_served_ratio",
    ]
    valid[training_columns].to_csv(OUTPUT_PATH, index=False)

    elapsed = time.perf_counter() - started
    print(f"Saved {len(valid)} valid rows to {OUTPUT_PATH}")
    print(f"Diagnostics saved to {DIAGNOSTIC_PATH}")
    print(f"Elapsed: {elapsed / 60.0:.1f} minutes")
    return valid[training_columns]


if __name__ == "__main__":
    dataset = generate_dataset()
    print(dataset.describe(include="all").to_string())
