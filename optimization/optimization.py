"""NSGA-II optimization using trained XGBoost surrogate models."""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
from deap import algorithms, base, creator, tools

from optimization.common import extract_nondominated, front_to_records
from src.cost_model import calculate_capital_cost
from src.model_features import (
    RESERVOIR_TYPE_TO_CODE,
    SURROGATE_FEATURES,
    TRAINING_BOUNDS,
)
from src.solar_data_loader import annual_solar_yield_per_kwp
from src.user_inputs import UserInputs

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

POPULATION_SIZE = 100
N_GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.2
MIN_EFFICIENCY = 70.0
RANDOM_SEED = 42

CURRENT_USER = UserInputs()
CURRENT_SOLAR_YIELD = None
MODEL_EFF = None
MODEL_AUTO = None


def get_bounds(user=None):
    user = user or CURRENT_USER
    maximum_volume = min(float(user.max_volume_m3), TRAINING_BOUNDS["volume_m3"][1])
    if maximum_volume < TRAINING_BOUNDS["volume_m3"][0]:
        raise ValueError("Maximum volume is below the trained minimum volume.")
    return {
        "volume_m3": (TRAINING_BOUNDS["volume_m3"][0], maximum_volume),
        "head_m": TRAINING_BOUNDS["head_m"],
        "pipe_diameter_m": TRAINING_BOUNDS["pipe_diameter_m"],
        "pump_power_kw": TRAINING_BOUNDS["pump_power_kw"],
        "turbine_power_kw": TRAINING_BOUNDS["turbine_power_kw"],
    }


def _validate_user_domain(user):
    checks = {
        "pv_kwp": float(user.pv_kwp),
        "daily_energy_kwh": float(user.daily_energy_kwh),
        "evaporation_rate_mm_month": float(user.evaporation_rate_mm_month),
    }
    violations = []
    for name, value in checks.items():
        low, high = TRAINING_BOUNDS[name]
        if not low <= value <= high:
            violations.append(f"{name}={value} outside [{low}, {high}]")
    if violations:
        raise ValueError(
            "ML mode cannot extrapolate beyond its training domain: "
            + "; ".join(violations)
            + ". Use Physics mode or regenerate/retrain with wider bounds."
        )


def _load_models():
    global MODEL_EFF, MODEL_AUTO
    feature_path = MODEL_DIR / "feature_names.pkl"
    efficiency_path = MODEL_DIR / "xgboost_efficiency.pkl"
    autonomy_path = MODEL_DIR / "xgboost_autonomy.pkl"
    for path in (feature_path, efficiency_path, autonomy_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {path}. Regenerate the dataset and retrain."
            )
    stored_features = joblib.load(feature_path)
    if list(stored_features) != list(SURROGATE_FEATURES):
        raise ValueError(
            "The repository still contains legacy 9-feature models. Run "
            "`python -m scripts.generate_dataset` and then "
            "`python -m scripts.train_surrogate` to create corrected models."
        )
    MODEL_EFF = joblib.load(efficiency_path)
    MODEL_AUTO = joblib.load(autonomy_path)


def _feature_vector(individual):
    user = CURRENT_USER
    values = {
        "volume_m3": float(individual[0]),
        "head_m": float(individual[1]),
        "pipe_diameter_m": float(individual[2]),
        "pump_power_kw": float(individual[3]),
        "turbine_power_kw": float(individual[4]),
        "pv_kwp": float(user.pv_kwp),
        "daily_energy_kwh": float(user.daily_energy_kwh),
        "evaporation_rate_mm_month": float(user.evaporation_rate_mm_month),
        "reservoir_type_code": RESERVOIR_TYPE_TO_CODE.get(
            user.upper_reservoir_type, 0
        ),
        "latitude": float(user.latitude),
        "longitude": float(user.longitude),
        "annual_solar_yield_kwh_per_kwp": float(CURRENT_SOLAR_YIELD),
    }
    return np.asarray([[values[name] for name in SURROGATE_FEATURES]], dtype=float)


def evaluate(individual):
    X = _feature_vector(individual)
    efficiency = float(MODEL_EFF.predict(X)[0])
    autonomy = float(MODEL_AUTO.predict(X)[0])
    cost = float(
        calculate_capital_cost(
            individual[0],
            individual[1],
            individual[2],
            individual[3],
            individual[4],
            CURRENT_USER.pv_kwp,
            CURRENT_USER.upper_reservoir_type,
            CURRENT_USER.lower_reservoir_type,
        )["total_lkr"]
    )

    feasible = (
        np.isfinite(efficiency)
        and np.isfinite(autonomy)
        and efficiency >= MIN_EFFICIENCY
        and autonomy >= CURRENT_USER.autonomy_days
        and (
            CURRENT_USER.budget_lkr is None
            or cost <= float(CURRENT_USER.budget_lkr)
        )
    )
    if not feasible:
        # With weights (maximize efficiency, minimize cost), this point is
        # dominated by every feasible design and cannot be rewarded.
        return -1.0e6, 1.0e12
    return efficiency, cost


def setup_deap(user=None):
    user = user or CURRENT_USER
    bounds = get_bounds(user)

    if not hasattr(creator, "FitnessPHESML"):
        creator.create("FitnessPHESML", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "IndividualPHESML"):
        creator.create("IndividualPHESML", list, fitness=creator.FitnessPHESML)

    toolbox = base.Toolbox()
    toolbox.register("attr_volume", random.uniform, *bounds["volume_m3"])
    toolbox.register("attr_head", random.uniform, *bounds["head_m"])
    toolbox.register("attr_pipe", random.uniform, *bounds["pipe_diameter_m"])
    toolbox.register("attr_pump", random.uniform, *bounds["pump_power_kw"])
    toolbox.register("attr_turbine", random.uniform, *bounds["turbine_power_kw"])
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.IndividualPHESML,
        (
            toolbox.attr_volume,
            toolbox.attr_head,
            toolbox.attr_pipe,
            toolbox.attr_pump,
            toolbox.attr_turbine,
        ),
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    low = [bounds[name][0] for name in bounds]
    high = [bounds[name][1] for name in bounds]
    toolbox.register(
        "mate", tools.cxSimulatedBinaryBounded, low=low, up=high, eta=20.0
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=low,
        up=high,
        eta=20.0,
        indpb=0.1,
    )
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)
    return toolbox


def run_optimization(user=None):
    global CURRENT_USER, CURRENT_SOLAR_YIELD
    CURRENT_USER = user or UserInputs()
    _validate_user_domain(CURRENT_USER)
    _load_models()
    CURRENT_SOLAR_YIELD = annual_solar_yield_per_kwp(CURRENT_USER)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    toolbox = setup_deap(CURRENT_USER)
    population = toolbox.population(n=POPULATION_SIZE)
    population, _ = algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=POPULATION_SIZE,
        lambda_=POPULATION_SIZE,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=N_GENERATIONS,
        verbose=False,
    )
    return population


def extract_pareto_front(population):
    return front_to_records(extract_nondominated(population))
