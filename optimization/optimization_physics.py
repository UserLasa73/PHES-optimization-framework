"""NSGA-II optimization using the corrected hourly physics simulator."""

from __future__ import annotations

import random

import numpy as np
from deap import algorithms, base, creator, tools

from optimization.common import extract_nondominated, front_to_records
from src.model_features import TRAINING_BOUNDS
from src.simulator import PumpedHydroSimulator
from src.solar_data_loader import fetch_load_data, fetch_solar_data
from src.user_inputs import UserInputs

POPULATION_SIZE = 100
N_GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.2
MIN_EFFICIENCY = 70.0
RANDOM_SEED = 42

CURRENT_USER = UserInputs()
CURRENT_SOLAR_DATA = None
CURRENT_LOAD_DATA = None
EVALUATION_CACHE = {}


def get_bounds(user=None):
    user = user or CURRENT_USER
    maximum_volume = min(float(user.max_volume_m3), TRAINING_BOUNDS["volume_m3"][1])
    return {
        "volume_m3": (TRAINING_BOUNDS["volume_m3"][0], maximum_volume),
        "head_m": TRAINING_BOUNDS["head_m"],
        "pipe_diameter_m": TRAINING_BOUNDS["pipe_diameter_m"],
        "pump_power_kw": TRAINING_BOUNDS["pump_power_kw"],
        "turbine_power_kw": TRAINING_BOUNDS["turbine_power_kw"],
    }


def evaluate(individual):
    key = tuple(round(float(value), 7) for value in individual)
    cached = EVALUATION_CACHE.get(key)
    if cached is not None:
        return cached

    design = {
        "volume_m3": float(individual[0]),
        "head_m": float(individual[1]),
        "pipe_diameter_m": float(individual[2]),
        "pump_power_kw": float(individual[3]),
        "turbine_power_kw": float(individual[4]),
    }
    metrics = PumpedHydroSimulator(CURRENT_USER, design).simulate(
        CURRENT_SOLAR_DATA, CURRENT_LOAD_DATA
    )["metrics"]
    efficiency = float(metrics["efficiency_percent"])
    autonomy = float(metrics["autonomy_days"])
    cost = float(metrics["capital_cost_lkr"])

    feasible = (
        metrics["is_physically_valid"]
        and efficiency >= MIN_EFFICIENCY
        and autonomy >= CURRENT_USER.autonomy_days
        and (
            CURRENT_USER.budget_lkr is None
            or cost <= float(CURRENT_USER.budget_lkr)
        )
    )
    result = (efficiency, cost) if feasible else (-1.0e6, 1.0e12)
    EVALUATION_CACHE[key] = result
    return result


def setup_deap(user=None):
    user = user or CURRENT_USER
    bounds = get_bounds(user)
    if bounds["volume_m3"][1] < bounds["volume_m3"][0]:
        raise ValueError("Maximum volume is below the lower design bound.")

    if not hasattr(creator, "FitnessPHESPhysics"):
        creator.create("FitnessPHESPhysics", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "IndividualPHESPhysics"):
        creator.create(
            "IndividualPHESPhysics", list, fitness=creator.FitnessPHESPhysics
        )

    toolbox = base.Toolbox()
    toolbox.register("attr_volume", random.uniform, *bounds["volume_m3"])
    toolbox.register("attr_head", random.uniform, *bounds["head_m"])
    toolbox.register("attr_pipe", random.uniform, *bounds["pipe_diameter_m"])
    toolbox.register("attr_pump", random.uniform, *bounds["pump_power_kw"])
    toolbox.register("attr_turbine", random.uniform, *bounds["turbine_power_kw"])
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.IndividualPHESPhysics,
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


def run_optimization_physics(user=None):
    global CURRENT_USER, CURRENT_SOLAR_DATA, CURRENT_LOAD_DATA, EVALUATION_CACHE
    CURRENT_USER = user or UserInputs()

    # Critical speed correction: profiles are independent of the candidate design
    # and are now fetched once per optimization run, not once per individual.
    CURRENT_SOLAR_DATA = fetch_solar_data(CURRENT_USER)
    CURRENT_LOAD_DATA = fetch_load_data(CURRENT_USER)
    EVALUATION_CACHE = {}

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


def extract_pareto_front_physics(population):
    return front_to_records(extract_nondominated(population))
