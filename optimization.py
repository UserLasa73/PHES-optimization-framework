"""
optimization.py
NSGA-II Multi-Objective Optimization for PHES Design.
"""

import numpy as np
import pandas as pd
import random
import joblib
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings('ignore')

from user_inputs import UserInputs
from cost_model import calculate_capital_cost


# ============================================================================
# CONFIGURATION (Default - overridden by app)
# ============================================================================

DEFAULT_USER = UserInputs()
DEFAULT_USER.latitude = 8.9
DEFAULT_USER.longitude = 79.9
DEFAULT_USER.pv_kwp = 30.0
DEFAULT_USER.tilt_angle = 10.0
DEFAULT_USER.azimuth_angle = 0.0
DEFAULT_USER.daily_energy_kwh = 50.0
DEFAULT_USER.upper_reservoir_type = "new_tank"
DEFAULT_USER.lower_reservoir_type = "new_tank"
DEFAULT_USER.autonomy_days = 2.0
DEFAULT_USER.evaporation_rate_mm_month = 50.0
DEFAULT_USER.demand_spike_factor = 1.0
DEFAULT_USER.has_grid_backup = False
DEFAULT_USER.pipe_roughness_m = 0.00015
# NEW: User volume constraint (only max needed)
DEFAULT_USER.max_volume_m3 = 800  # Default matches original hard-coded bound
DEFAULT_USER.budget_lkr = None

CURRENT_USER = DEFAULT_USER

# ===== CONSTANTS =====
POPULATION_SIZE = 100
N_GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.2

MIN_EFFICIENCY = 0.0

# ============================================================================
# GET DYNAMIC BOUNDS
# ============================================================================

def get_bounds(user=None):
    """Get optimization bounds based on user inputs."""
    
    if user is None:
        user = CURRENT_USER
    
    bounds = {
        'volume_m3': (20, user.max_volume_m3),  # ← Min is fixed at 20 (from training data)
        'head_m': (5, 45),
        'pipe_diameter_m': (0.05, 0.35),
        'pump_power_kw': (2, 30),
        'turbine_power_kw': (2, 25)
    }
    
    return bounds


# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading surrogate models...")
model_eff = joblib.load('models/xgboost_efficiency.pkl')
model_auto = joblib.load('models/xgboost_autonomy.pkl')
print("Models loaded.")

RES_TYPE_MAP = {'new_tank': 0, 'excavated': 1, 'pond': 2, 'river': 3}

# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def evaluate(individual):
    """Evaluate a design using the current user."""
    
    user = CURRENT_USER
    res_code = RES_TYPE_MAP.get(user.upper_reservoir_type, 0)
    
    X = np.array([[
        individual[0], individual[1], individual[2], individual[3], individual[4],
        user.pv_kwp,
        user.daily_energy_kwh,
        user.evaporation_rate_mm_month,
        res_code
    ]])
    
    efficiency = model_eff.predict(X)[0]
    autonomy = model_auto.predict(X)[0]
    
    cost_dict = calculate_capital_cost(
        individual[0], individual[1], individual[2],
        individual[3], individual[4],
        user.pv_kwp,
        user.upper_reservoir_type,
        user.lower_reservoir_type
    )
    cost = cost_dict['total_lkr']
    
    # ===== HARD CONSTRAINTS =====
    if efficiency < 70.0:
        return [1000.0, 100000000.0]
    
    if autonomy < user.autonomy_days:
        return [1000.0, 100000000.0]
    
    if user.budget_lkr is not None and cost > user.budget_lkr:
        return [1000.0, 100000000.0]  # Reject only if budget is set

    return [-efficiency, cost]


# ============================================================================
# SETUP DEAP (UPDATED TO USE DYNAMIC BOUNDS)
# ============================================================================

def setup_deap(user=None):
    """Setup DEAP with dynamic bounds from user."""
    
    if user is None:
        user = CURRENT_USER
    
    # Get dynamic bounds
    bounds = get_bounds(user)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    # Use bounds from user input
    toolbox.register("attr_volume", random.uniform, 
                     bounds['volume_m3'][0], bounds['volume_m3'][1])
    toolbox.register("attr_head", random.uniform, 
                     bounds['head_m'][0], bounds['head_m'][1])
    toolbox.register("attr_pipe", random.uniform, 
                     bounds['pipe_diameter_m'][0], bounds['pipe_diameter_m'][1])
    toolbox.register("attr_pump", random.uniform, 
                     bounds['pump_power_kw'][0], bounds['pump_power_kw'][1])
    toolbox.register("attr_turbine", random.uniform, 
                     bounds['turbine_power_kw'][0], bounds['turbine_power_kw'][1])
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_volume, toolbox.attr_head, toolbox.attr_pipe,
                      toolbox.attr_pump, toolbox.attr_turbine), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[bounds['volume_m3'][0], bounds['head_m'][0], 
                          bounds['pipe_diameter_m'][0], bounds['pump_power_kw'][0],
                          bounds['turbine_power_kw'][0]],
                     up=[bounds['volume_m3'][1], bounds['head_m'][1], 
                         bounds['pipe_diameter_m'][1], bounds['pump_power_kw'][1],
                         bounds['turbine_power_kw'][1]],
                     eta=20.0)
    
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[bounds['volume_m3'][0], bounds['head_m'][0], 
                          bounds['pipe_diameter_m'][0], bounds['pump_power_kw'][0],
                          bounds['turbine_power_kw'][0]],
                     up=[bounds['volume_m3'][1], bounds['head_m'][1], 
                         bounds['pipe_diameter_m'][1], bounds['pump_power_kw'][1],
                         bounds['turbine_power_kw'][1]],
                     eta=20.0, indpb=0.1)
    
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)
    
    return toolbox


# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

def run_optimization(user=None):
    """Run NSGA-II optimization with the given user."""
    
    global CURRENT_USER
    
    if user is not None:
        CURRENT_USER = user
    else:
        CURRENT_USER = DEFAULT_USER
    
    print("=" * 70)
    print("NSGA-II OPTIMIZATION")
    print("=" * 70)
    print(f"Reservoir Type: {CURRENT_USER.upper_reservoir_type}")
    print(f"PV Capacity: {CURRENT_USER.pv_kwp} kWp")
    print(f"Autonomy: >= {CURRENT_USER.autonomy_days} days")
    print(f"Max Volume: {CURRENT_USER.max_volume_m3} m³")
    print("=" * 70)
    
    toolbox = setup_deap(user=CURRENT_USER)
    population = toolbox.population(n=POPULATION_SIZE)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    print("\nRunning optimization...")
    population, logbook = algorithms.eaMuPlusLambda(
        population, toolbox, mu=POPULATION_SIZE, lambda_=POPULATION_SIZE,
        cxpb=CX_PROB, mutpb=MUT_PROB, ngen=N_GENERATIONS,
        stats=stats, verbose=False
    )
    
    return population


# ============================================================================
# EXTRACT PARETO FRONT
# ============================================================================

def extract_pareto_front(population):
    pareto_front = []
    for ind in population:
        if ind.fitness.values[0] < 1000:
            pareto_front.append({
                'volume_m3': ind[0],
                'head_m': ind[1],
                'pipe_diameter_m': ind[2],
                'pump_power_kw': ind[3],
                'turbine_power_kw': ind[4],
                'efficiency': -ind.fitness.values[0],
                'cost': ind.fitness.values[1]
            })
    return sorted(pareto_front, key=lambda x: x['efficiency'], reverse=True)