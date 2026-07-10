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

CURRENT_USER = DEFAULT_USER

# ===== BOUNDS =====
BOUNDS = {
    'volume_m3': (20, 300),        # ← Max 300 m³ (realistic for homes)
    'head_m': (5, 30),             # ← Typical home hill height
    'pipe_diameter_m': (0.05, 0.25),
    'pump_power_kw': (2, 15),
    'turbine_power_kw': (1, 10)
}

POPULATION_SIZE = 100
N_GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.2

MIN_EFFICIENCY = 0.0

# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading surrogate models...")
model_eff = joblib.load('models/xgboost_efficiency.pkl')
model_auto = joblib.load('models/xgboost_autonomy.pkl')
print("Models loaded.")

RES_TYPE_MAP = {'new_tank': 0, 'excavated': 1, 'pond': 2, 'river': 3}

# ===== INSERT DIAGNOSTIC HERE =====
print("\n" + "=" * 70)
print("DIAGNOSTIC: Testing model predictions")
print("=" * 70)

test_X = np.array([[1000, 30, 0.25, 15, 10, 30, 50, 50, 1]])
eff = model_eff.predict(test_X)[0]
auto = model_auto.predict(test_X)[0]
print(f"Volume=1000, Head=30, Excavated → Eff={eff:.1f}%, Auto={auto:.2f} days")

test_X2 = np.array([[400, 25, 0.25, 10, 8, 30, 50, 50, 1]])
eff2 = model_eff.predict(test_X2)[0]
auto2 = model_auto.predict(test_X2)[0]
print(f"Volume=400, Head=25, Excavated → Eff={eff2:.1f}%, Auto={auto2:.2f} days")

print("=" * 70)
print("")


# In optimization.py
test_X = np.array([[1000, 30, 0.25, 15, 10, 30, 50, 50, 1]])
eff = model_eff.predict(test_X)[0]
auto = model_auto.predict(test_X)[0]
print(f"Volume=1000, Excavated → Eff={eff:.1f}%, Auto={auto:.2f} days")

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
    
    # Constraints
    # ===== SOFT CONSTRAINTS (PENALTY, NOT REJECTION) =====
    penalty = 0.0
    
    # Penalty for low efficiency (instead of rejecting)
    if efficiency < 80.0:
        penalty += (80.0 - efficiency) * 1000  # Add penalty
    
    # Penalty for low autonomy (instead of rejecting)
    if autonomy < user.autonomy_days:
        penalty += (user.autonomy_days - autonomy) * 100000  # Add penalty
    
    # ===== OBJECTIVE WITH PENALTY =====
    # This way, designs that FAIL constraints still appear, 
    # but they have a penalty added to their cost
    adjusted_cost = cost + penalty
    
    return [-efficiency, adjusted_cost]


# ============================================================================
# SETUP DEAP
# ============================================================================

def setup_deap():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    toolbox.register("attr_volume", random.uniform, BOUNDS['volume_m3'][0], BOUNDS['volume_m3'][1])
    toolbox.register("attr_head", random.uniform, BOUNDS['head_m'][0], BOUNDS['head_m'][1])
    toolbox.register("attr_pipe", random.uniform, BOUNDS['pipe_diameter_m'][0], BOUNDS['pipe_diameter_m'][1])
    toolbox.register("attr_pump", random.uniform, BOUNDS['pump_power_kw'][0], BOUNDS['pump_power_kw'][1])
    toolbox.register("attr_turbine", random.uniform, BOUNDS['turbine_power_kw'][0], BOUNDS['turbine_power_kw'][1])
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_volume, toolbox.attr_head, toolbox.attr_pipe,
                      toolbox.attr_pump, toolbox.attr_turbine), n=1)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[BOUNDS['volume_m3'][0], BOUNDS['head_m'][0], 
                          BOUNDS['pipe_diameter_m'][0], BOUNDS['pump_power_kw'][0],
                          BOUNDS['turbine_power_kw'][0]],
                     up=[BOUNDS['volume_m3'][1], BOUNDS['head_m'][1], 
                         BOUNDS['pipe_diameter_m'][1], BOUNDS['pump_power_kw'][1],
                         BOUNDS['turbine_power_kw'][1]],
                     eta=20.0)
    
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[BOUNDS['volume_m3'][0], BOUNDS['head_m'][0], 
                          BOUNDS['pipe_diameter_m'][0], BOUNDS['pump_power_kw'][0],
                          BOUNDS['turbine_power_kw'][0]],
                     up=[BOUNDS['volume_m3'][1], BOUNDS['head_m'][1], 
                         BOUNDS['pipe_diameter_m'][1], BOUNDS['pump_power_kw'][1],
                         BOUNDS['turbine_power_kw'][1]],
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
    print("=" * 70)
    
    toolbox = setup_deap()
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
