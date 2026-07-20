"""Generate the final evidence for Thesis Tables 4.8, 4.9 and 4.10.

This script performs:
1. ML NSGA-II optimization for Vavuniya, Colombo and Jaffna.
2. Exact same-design physics validation.
3. Selection of four Vavuniya Pareto representatives.
4. Physics verification of the Colombo and Jaffna ML Pareto designs.
5. CSV, JSON and LaTeX-ready terminal output.

It does NOT run the slow physics-based NSGA-II optimizer.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from optimization.common import select_compromise_design
from optimization.optimization import (
    extract_pareto_front,
    run_optimization,
)
from src.model_features import (
    RESERVOIR_TYPE_TO_CODE,
    SURROGATE_FEATURES,
)
from src.simulator import PumpedHydroSimulator
from src.solar_data_loader import (
    annual_solar_yield_per_kwp,
    fetch_load_data,
    fetch_solar_data,
)
from src.user_inputs import UserInputs


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "thesis_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROOT / "models"


# ============================================================================
# FINAL CASE-STUDY DEFINITIONS
# ============================================================================

SCENARIOS = {
    "Vavuniya": {
        "case_label": "Residential case",
        "latitude": 8.7542,
        "longitude": 80.4982,
        "pv_kwp": 20.0,
        "daily_energy_kwh": 20.0,
        "autonomy_days": 0.50,
        "budget_lkr": 4_000_000.0,
        "reservoir_type": "new_tank",
        "max_volume_m3": 800.0,
        "evaporation_rate_mm_month": 50.0,
        "pipe_roughness_m": 0.00015,
    },
    "Colombo": {
        "case_label": "Small-commercial case",
        "latitude": 6.9271,
        "longitude": 79.8612,
        "pv_kwp": 30.0,
        "daily_energy_kwh": 50.0,
        "autonomy_days": 0.40,
        "budget_lkr": 8_000_000.0,
        "reservoir_type": "new_tank",
        "max_volume_m3": 800.0,
        "evaporation_rate_mm_month": 50.0,
        "pipe_roughness_m": 0.00015,
    },
    "Jaffna": {
        "case_label": "Community/island proxy case",
        "latitude": 9.6615,
        "longitude": 80.0255,
        "pv_kwp": 25.0,
        "daily_energy_kwh": 30.0,
        "autonomy_days": 0.75,
        "budget_lkr": 8_000_000.0,
        "reservoir_type": "new_tank",
        "max_volume_m3": 800.0,
        "evaporation_rate_mm_month": 50.0,
        "pipe_roughness_m": 0.00015,
    },
}


DESIGN_COLUMNS = [
    "volume_m3",
    "head_m",
    "pipe_diameter_m",
    "pump_power_kw",
    "turbine_power_kw",
]


# ============================================================================
# LOAD THE AUTONOMY MODEL
# ============================================================================

stored_features = joblib.load(MODEL_DIR / "feature_names.pkl")

if list(stored_features) != list(SURROGATE_FEATURES):
    raise ValueError(
        "The saved model feature list does not match the corrected "
        "12-feature surrogate schema."
    )

AUTONOMY_MODEL = joblib.load(
    MODEL_DIR / "xgboost_autonomy.pkl"
)


# ============================================================================
# USER AND FEATURE HELPERS
# ============================================================================

def make_user(location_name: str, scenario: dict) -> UserInputs:
    """Create a UserInputs instance for one final case study."""

    user = UserInputs()

    user.location = location_name
    user.latitude = float(scenario["latitude"])
    user.longitude = float(scenario["longitude"])

    user.pv_kwp = float(scenario["pv_kwp"])
    user.daily_energy_kwh = float(
        scenario["daily_energy_kwh"]
    )
    user.autonomy_days = float(
        scenario["autonomy_days"]
    )

    user.upper_reservoir_type = str(
        scenario["reservoir_type"]
    )
    user.lower_reservoir_type = str(
        scenario["reservoir_type"]
    )

    user.max_volume_m3 = float(
        scenario["max_volume_m3"]
    )
    user.budget_lkr = float(
        scenario["budget_lkr"]
    )

    user.evaporation_rate_mm_month = float(
        scenario["evaporation_rate_mm_month"]
    )
    user.pipe_roughness_m = float(
        scenario["pipe_roughness_m"]
    )

    user.tilt_angle = 10.0
    user.azimuth_angle = 180.0
    user.year = 2023

    user.demand_spike_factor = 1.0
    user.has_grid_backup = False
    user.load_csv_path = None

    return user


def build_feature_array(
    design: dict,
    user: UserInputs,
    annual_yield: float,
) -> np.ndarray:
    """Construct the corrected 12-feature model input."""

    values = {
        "volume_m3": float(design["volume_m3"]),
        "head_m": float(design["head_m"]),
        "pipe_diameter_m": float(
            design["pipe_diameter_m"]
        ),
        "pump_power_kw": float(
            design["pump_power_kw"]
        ),
        "turbine_power_kw": float(
            design["turbine_power_kw"]
        ),
        "pv_kwp": float(user.pv_kwp),
        "daily_energy_kwh": float(
            user.daily_energy_kwh
        ),
        "evaporation_rate_mm_month": float(
            user.evaporation_rate_mm_month
        ),
        "reservoir_type_code": float(
            RESERVOIR_TYPE_TO_CODE[
                user.upper_reservoir_type
            ]
        ),
        "latitude": float(user.latitude),
        "longitude": float(user.longitude),
        "annual_solar_yield_kwh_per_kwp": float(
            annual_yield
        ),
    }

    return np.asarray(
        [[values[name] for name in SURROGATE_FEATURES]],
        dtype=float,
    )


def predict_ml_autonomy(
    design: dict,
    user: UserInputs,
    annual_yield: float,
) -> float:
    """Predict autonomy for one exact design."""

    features = build_feature_array(
        design,
        user,
        annual_yield,
    )

    return float(
        AUTONOMY_MODEL.predict(features)[0]
    )


# ============================================================================
# ML OPTIMIZATION
# ============================================================================

def run_ml_case(
    location_name: str,
    scenario: dict,
):
    """Run ML NSGA-II and return its true Pareto front."""

    user = make_user(location_name, scenario)

    print("\n" + "=" * 100)
    print(f"RUNNING ML CASE: {location_name}")
    print("=" * 100)

    start = time.perf_counter()

    population = run_optimization(user)
    records = extract_pareto_front(population)

    ml_runtime = time.perf_counter() - start

    if not records:
        raise RuntimeError(
            f"No ML Pareto solutions found for {location_name}."
        )

    frame = pd.DataFrame(records)
    frame = frame.sort_values(
        ["cost", "efficiency"],
        ascending=[True, False],
    ).reset_index(drop=True)

    annual_yield = annual_solar_yield_per_kwp(user)

    frame["ml_autonomy_days"] = [
        predict_ml_autonomy(
            row,
            user,
            annual_yield,
        )
        for row in frame.to_dict("records")
    ]

    frame.insert(
        0,
        "front_index",
        range(len(frame)),
    )

    frame["location"] = location_name
    frame["pv_kwp"] = user.pv_kwp
    frame["daily_energy_kwh"] = (
        user.daily_energy_kwh
    )
    frame["required_autonomy_days"] = (
        user.autonomy_days
    )
    frame["budget_lkr"] = user.budget_lkr
    frame["reservoir_type"] = (
        user.upper_reservoir_type
    )

    pareto_path = (
        OUTPUT_DIR
        / f"{location_name.lower()}_ml_pareto.csv"
    )

    frame.to_csv(
        pareto_path,
        index=False,
    )

    print(
        f"ML Pareto alternatives: {len(frame)}"
    )
    print(
        f"ML optimization runtime: {ml_runtime:.4f} s"
    )
    print(
        f"Saved ML Pareto front: {pareto_path}"
    )

    return {
        "user": user,
        "records": records,
        "frame": frame,
        "annual_yield": annual_yield,
        "ml_runtime_s": ml_runtime,
        "pareto_path": pareto_path,
    }


# ============================================================================
# DESIGN-SELECTION HELPERS
# ============================================================================

def find_matching_index(
    frame: pd.DataFrame,
    design: dict,
) -> int:
    """Find the row corresponding to a selected design."""

    distance = np.zeros(len(frame), dtype=float)

    scales = {
        "volume_m3": 780.0,
        "head_m": 40.0,
        "pipe_diameter_m": 0.30,
        "pump_power_kw": 28.0,
        "turbine_power_kw": 23.0,
    }

    for column in DESIGN_COLUMNS:
        difference = (
            frame[column].astype(float)
            - float(design[column])
        )
        distance += (
            difference / scales[column]
        ) ** 2

    return int(distance.argmin())


def first_unused(
    ordered_indices: list[int],
    used_indices: set[int],
) -> int:
    """Return the first unselected Pareto index."""

    for index in ordered_indices:
        if int(index) not in used_indices:
            return int(index)

    return int(ordered_indices[0])


def select_vavuniya_representatives(
    frame: pd.DataFrame,
    records: list[dict],
) -> OrderedDict:
    """Select four unique representative ML Pareto designs."""

    compromise = select_compromise_design(records)

    compromise_index = find_matching_index(
        frame,
        compromise,
    )

    used = {compromise_index}

    cost_order = list(
        frame.sort_values(
            "cost",
            ascending=True,
        ).index
    )

    low_cost_index = first_unused(
        cost_order,
        used,
    )
    used.add(low_cost_index)

    efficiency_order = list(
        frame.sort_values(
            "efficiency",
            ascending=False,
        ).index
    )

    high_efficiency_index = first_unused(
        efficiency_order,
        used,
    )
    used.add(high_efficiency_index)

    middle_position = (
        len(cost_order) - 1
    ) / 2.0

    middle_order = sorted(
        cost_order,
        key=lambda index: abs(
            cost_order.index(index)
            - middle_position
        ),
    )

    middle_index = first_unused(
        middle_order,
        used,
    )

    return OrderedDict(
        [
            (
                "ML compromise",
                compromise_index,
            ),
            (
                "Low-cost Pareto design",
                low_cost_index,
            ),
            (
                "Middle Pareto design",
                middle_index,
            ),
            (
                "High-efficiency Pareto design",
                high_efficiency_index,
            ),
        ]
    )


def compromise_ranked_indices(
    frame: pd.DataFrame,
) -> list[int]:
    """Rank ML Pareto designs by distance to the ideal objective point."""

    efficiency_min = float(
        frame["efficiency"].min()
    )
    efficiency_max = float(
        frame["efficiency"].max()
    )
    cost_min = float(frame["cost"].min())
    cost_max = float(frame["cost"].max())

    if efficiency_max > efficiency_min:
        efficiency_gap = (
            efficiency_max
            - frame["efficiency"]
        ) / (
            efficiency_max
            - efficiency_min
        )
    else:
        efficiency_gap = pd.Series(
            np.zeros(len(frame)),
            index=frame.index,
        )

    if cost_max > cost_min:
        cost_gap = (
            frame["cost"]
            - cost_min
        ) / (
            cost_max
            - cost_min
        )
    else:
        cost_gap = pd.Series(
            np.zeros(len(frame)),
            index=frame.index,
        )

    score = np.sqrt(
        efficiency_gap**2
        + cost_gap**2
    )

    return [
        int(index)
        for index in score.sort_values().index
    ]


# ============================================================================
# PHYSICS EVALUATION
# ============================================================================

def load_physics_profiles(user: UserInputs):
    """Load solar and load profiles once for a case."""

    start = time.perf_counter()

    solar = fetch_solar_data(user)
    load = fetch_load_data(user)

    runtime = time.perf_counter() - start

    return solar, load, runtime


def evaluate_same_design(
    row: pd.Series,
    user: UserInputs,
    solar: list[float],
    load: list[float],
) -> dict:
    """Evaluate one exact ML Pareto design using the physics simulator."""

    design = {
        column: float(row[column])
        for column in DESIGN_COLUMNS
    }

    start = time.perf_counter()

    metrics = PumpedHydroSimulator(
        user,
        design,
    ).simulate(
        solar,
        load,
    )["metrics"]

    simulation_runtime = (
        time.perf_counter() - start
    )

    ml_efficiency = float(
        row["efficiency"]
    )
    ml_autonomy = float(
        row["ml_autonomy_days"]
    )

    physics_efficiency = float(
        metrics["efficiency_percent"]
    )
    physics_autonomy = float(
        metrics["autonomy_days"]
    )
    physics_cost = float(
        metrics["capital_cost_lkr"]
    )

    budget_met = (
        user.budget_lkr is None
        or physics_cost
        <= float(user.budget_lkr)
    )

    physics_feasible = bool(
        metrics["is_physically_valid"]
        and physics_efficiency >= 70.0
        and physics_autonomy
        >= user.autonomy_days
        and budget_met
    )

    return {
        **design,
        "ml_efficiency_percent": ml_efficiency,
        "physics_efficiency_percent": (
            physics_efficiency
        ),
        "efficiency_absolute_error_pp": abs(
            ml_efficiency
            - physics_efficiency
        ),
        "ml_autonomy_days": ml_autonomy,
        "physics_autonomy_days": (
            physics_autonomy
        ),
        "autonomy_absolute_error_days": abs(
            ml_autonomy
            - physics_autonomy
        ),
        "estimated_cost_lkr": float(
            row["cost"]
        ),
        "physics_cost_lkr": physics_cost,
        "physics_load_served_ratio": float(
            metrics["load_served_ratio"]
        ),
        "physics_valid": bool(
            metrics["is_physically_valid"]
        ),
        "autonomy_requirement_met": bool(
            physics_autonomy
            >= user.autonomy_days
        ),
        "budget_requirement_met": bool(
            budget_met
        ),
        "physics_feasible": physics_feasible,
        "water_balance_residual_m3": float(
            metrics[
                "water_balance_residual_m3"
            ]
        ),
        "single_physics_simulation_runtime_s": (
            simulation_runtime
        ),
        "front_index": int(
            row["front_index"]
        ),
    }


# ============================================================================
# TABLE 4.8 — VAVUNIYA SAME-DESIGN VALIDATION
# ============================================================================

def generate_table_4_8(vavuniya_result: dict):
    """Generate four same-design Vavuniya validation rows."""

    user = vavuniya_result["user"]
    frame = vavuniya_result["frame"]
    records = vavuniya_result["records"]

    solar, load, profile_runtime = (
        load_physics_profiles(user)
    )

    selected_indices = (
        select_vavuniya_representatives(
            frame,
            records,
        )
    )

    rows = []

    for label, index in selected_indices.items():
        row = frame.loc[index]

        result = evaluate_same_design(
            row,
            user,
            solar,
            load,
        )

        result["design_label"] = label
        rows.append(result)

    output = pd.DataFrame(rows)

    column_order = [
        "design_label",
        "front_index",
        "volume_m3",
        "head_m",
        "pipe_diameter_m",
        "pump_power_kw",
        "turbine_power_kw",
        "ml_efficiency_percent",
        "physics_efficiency_percent",
        "efficiency_absolute_error_pp",
        "ml_autonomy_days",
        "physics_autonomy_days",
        "autonomy_absolute_error_days",
        "estimated_cost_lkr",
        "physics_load_served_ratio",
        "physics_valid",
        "autonomy_requirement_met",
        "physics_feasible",
        "water_balance_residual_m3",
        "single_physics_simulation_runtime_s",
    ]

    output = output[column_order]

    output_path = (
        OUTPUT_DIR
        / "table_4_8_same_design_validation.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 100)
    print("TABLE 4.8 — SAME-DESIGN VALIDATION")
    print("=" * 100)

    print(
        output[
            [
                "design_label",
                "ml_efficiency_percent",
                "physics_efficiency_percent",
                "ml_autonomy_days",
                "physics_autonomy_days",
                "physics_feasible",
            ]
        ].to_string(index=False)
    )

    print("\nLATEX ROWS FOR TABLE 4.8")
    print("-" * 100)

    for _, row in output.iterrows():
        print(
            f"{row['design_label']} & "
            f"{row['ml_efficiency_percent']:.4f} & "
            f"{row['physics_efficiency_percent']:.4f} & "
            f"{row['ml_autonomy_days']:.4f} & "
            f"{row['physics_autonomy_days']:.4f} \\\\"
        )

    print(
        f"\nProfile preparation runtime: "
        f"{profile_runtime:.4f} s"
    )
    print(f"Saved: {output_path}")

    return output


# ============================================================================
# TABLES 4.9 AND 4.10 — COLOMBO AND JAFFNA
# ============================================================================

def generate_case_table(
    location_name: str,
    scenario: dict,
    ml_result: dict,
    table_number: str,
):
    """Generate one location case-study result.

    Designs are considered in ML compromise order. The first design that also
    satisfies the constraints under the annual physics simulator is selected.
    """

    user = ml_result["user"]
    frame = ml_result["frame"]

    solar, load, profile_runtime = (
        load_physics_profiles(user)
    )

    ordered_indices = compromise_ranked_indices(
        frame
    )

    selected_result = None
    checked_results = []

    for index in ordered_indices:
        result = evaluate_same_design(
            frame.loc[index],
            user,
            solar,
            load,
        )

        checked_results.append(result)

        if result["physics_feasible"]:
            selected_result = result
            break

    if selected_result is None:
        selected_result = checked_results[0]

        print(
            f"\nWARNING: No fully physics-feasible ML Pareto design "
            f"was found for {location_name}."
        )

    total_simulation_runtime = sum(
        item[
            "single_physics_simulation_runtime_s"
        ]
        for item in checked_results
    )

    case_result = {
        "table_number": table_number,
        "location": location_name,
        "case_label": scenario["case_label"],
        "inputs": {
            "pv_kwp": float(user.pv_kwp),
            "daily_energy_kwh": float(
                user.daily_energy_kwh
            ),
            "required_autonomy_days": float(
                user.autonomy_days
            ),
            "budget_lkr": float(
                user.budget_lkr
            ),
            "reservoir_type": (
                user.upper_reservoir_type
            ),
            "maximum_combined_volume_m3": float(
                user.max_volume_m3
            ),
            "evaporation_rate_mm_month": float(
                user.evaporation_rate_mm_month
            ),
            "pipe_roughness_m": float(
                user.pipe_roughness_m
            ),
        },
        "ml_pareto_count": int(len(frame)),
        "ml_optimization_runtime_s": float(
            ml_result["ml_runtime_s"]
        ),
        "profile_preparation_runtime_s": float(
            profile_runtime
        ),
        "physics_candidates_checked": int(
            len(checked_results)
        ),
        "total_case_physics_validation_runtime_s": float(
            profile_runtime
            + total_simulation_runtime
        ),
        "selection_method": (
            "First physics-feasible design in "
            "ML compromise ranking"
        ),
        "selected_design": selected_result,
    }

    output_path = (
        OUTPUT_DIR
        / f"table_{table_number.replace('.', '_')}_"
        f"{location_name.lower()}_case.json"
    )

    output_path.write_text(
        json.dumps(
            case_result,
            indent=2,
        ),
        encoding="utf-8",
    )

    design = selected_result

    print("\n" + "=" * 100)
    print(
        f"TABLE {table_number} — "
        f"{location_name.upper()} CASE STUDY"
    )
    print("=" * 100)

    print(
        f"Case type:                  "
        f"{scenario['case_label']}"
    )
    print(
        f"PV capacity:                "
        f"{user.pv_kwp:.2f} kWp"
    )
    print(
        f"Daily demand:               "
        f"{user.daily_energy_kwh:.2f} kWh/day"
    )
    print(
        f"Required autonomy:          "
        f"{user.autonomy_days:.2f} days"
    )
    print(
        f"Budget:                     "
        f"LKR {user.budget_lkr:,.0f}"
    )
    print(
        f"ML efficiency:              "
        f"{design['ml_efficiency_percent']:.4f}%"
    )
    print(
        f"Physics efficiency:         "
        f"{design['physics_efficiency_percent']:.4f}%"
    )
    print(
        f"ML autonomy:                "
        f"{design['ml_autonomy_days']:.4f} days"
    )
    print(
        f"Physics autonomy:           "
        f"{design['physics_autonomy_days']:.4f} days"
    )
    print(
        f"Combined reservoir volume:  "
        f"{design['volume_m3']:.2f} m3"
    )
    print(
        f"Head:                       "
        f"{design['head_m']:.2f} m"
    )
    print(
        f"Pipe diameter:              "
        f"{design['pipe_diameter_m']:.4f} m"
    )
    print(
        f"Pump power:                 "
        f"{design['pump_power_kw']:.2f} kW"
    )
    print(
        f"Turbine power:              "
        f"{design['turbine_power_kw']:.2f} kW"
    )
    print(
        f"Estimated cost:             "
        f"LKR {design['estimated_cost_lkr']:,.2f}"
    )
    print(
        f"Load served ratio:          "
        f"{design['physics_load_served_ratio']:.4f}"
    )
    print(
        f"Physics feasible:           "
        f"{design['physics_feasible']}"
    )
    print(
        f"ML runtime:                 "
        f"{ml_result['ml_runtime_s']:.4f} s"
    )
    print(
        f"Selected physics runtime:   "
        f"{design['single_physics_simulation_runtime_s']:.4f} s"
    )
    print(
        f"Physics candidates checked: "
        f"{len(checked_results)}"
    )

    print(
        f"\nLATEX ROWS FOR TABLE {table_number}"
    )
    print("-" * 100)

    latex_rows = [
        (
            "Case type",
            scenario["case_label"],
        ),
        (
            "PV capacity",
            f"{user.pv_kwp:.2f} kWp",
        ),
        (
            "Daily demand",
            f"{user.daily_energy_kwh:.2f} kWh/day",
        ),
        (
            "Required autonomy",
            f"{user.autonomy_days:.2f} days",
        ),
        (
            "Budget",
            f"LKR {user.budget_lkr:,.0f}",
        ),
        (
            "ML efficiency",
            f"{design['ml_efficiency_percent']:.4f}\\%",
        ),
        (
            "Physics efficiency for the same design",
            f"{design['physics_efficiency_percent']:.4f}\\%",
        ),
        (
            "ML autonomy",
            f"{design['ml_autonomy_days']:.4f} days",
        ),
        (
            "Physics autonomy for the same design",
            f"{design['physics_autonomy_days']:.4f} days",
        ),
        (
            "Combined volume",
            f"{design['volume_m3']:.2f} m$^3$",
        ),
        (
            "Gross head",
            f"{design['head_m']:.2f} m",
        ),
        (
            "Pipe diameter",
            f"{design['pipe_diameter_m']:.4f} m",
        ),
        (
            "Pump / turbine rating",
            (
                f"{design['pump_power_kw']:.2f} / "
                f"{design['turbine_power_kw']:.2f} kW"
            ),
        ),
        (
            "Estimated cost",
            f"LKR {design['estimated_cost_lkr']:,.0f}",
        ),
        (
            "Physics load-served ratio",
            f"{design['physics_load_served_ratio']:.4f}",
        ),
        (
            "ML optimization runtime",
            f"{ml_result['ml_runtime_s']:.4f} s",
        ),
        (
            "Single physics-simulation runtime",
            (
                f"{design['single_physics_simulation_runtime_s']:.4f} s"
            ),
        ),
    ]

    for item, value in latex_rows:
        print(f"{item} & {value} \\\\")

    print(f"\nSaved: {output_path}")

    return case_result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "#" * 100)
    print("FINAL THESIS VALIDATION AND CASE-STUDY RUN")
    print("#" * 100)

    print(
        "\nThis script runs ML NSGA-II searches and exact "
        "same-design annual physics checks."
    )
    print(
        "It does not run the slow physics NSGA-II optimizer."
    )

    case_runs = {}

    for location_name, scenario in SCENARIOS.items():
        case_runs[location_name] = run_ml_case(
            location_name,
            scenario,
        )

    table_4_8 = generate_table_4_8(
        case_runs["Vavuniya"]
    )

    table_4_9 = generate_case_table(
        "Colombo",
        SCENARIOS["Colombo"],
        case_runs["Colombo"],
        "4.9",
    )

    table_4_10 = generate_case_table(
        "Jaffna",
        SCENARIOS["Jaffna"],
        case_runs["Jaffna"],
        "4.10",
    )

    summary = {
        "table_4_8_rows": int(
            len(table_4_8)
        ),
        "table_4_9": table_4_9,
        "table_4_10": table_4_10,
    }

    summary_path = (
        OUTPUT_DIR
        / "final_thesis_results_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "#" * 100)
    print("FINAL THESIS OUTPUT GENERATION COMPLETED")
    print("#" * 100)

    print(f"\nAll files are saved in:\n{OUTPUT_DIR}")

    print("\nRequired files:")
    print(
        "1. table_4_8_same_design_validation.csv"
    )
    print(
        "2. table_4_9_colombo_case.json"
    )
    print(
        "3. table_4_10_jaffna_case.json"
    )
    print(
        "4. final_thesis_results_summary.json"
    )


if __name__ == "__main__":
    main()