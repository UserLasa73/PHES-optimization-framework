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
# CONFIGURATION
# ============================================================================

USER = UserInputs()
USER.latitude = 8.9
USER.longitude = 79.9
USER.pv_kwp = 30.0
USER.tilt_angle = 10.0
USER.azimuth_angle = 0.0
USER.daily_energy_kwh = 50.0
USER.upper_reservoir_type = "new_tank"
USER.lower_reservoir_type = "new_tank"
USER.autonomy_days = 0.0
USER.evaporation_rate_mm_month = 50.0
USER.demand_spike_factor = 1.0
USER.has_grid_backup = False
USER.pipe_roughness_m = 0.00015

# ===== EXPANDED BOUNDS =====
BOUNDS = {
    'volume_m3': (50, 200),      # ← Bigger range
    'head_m': (10, 50),            # ← Bigger range
    'pipe_diameter_m': (0.1, 0.5),
    'pump_power_kw': (3, 30),      # ← Bigger range
    'turbine_power_kw': (2, 20)    # ← Bigger range
}

POPULATION_SIZE = 100
N_GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.2


# ============================================================================
# LOAD MODELS
# ============================================================================

print("Loading surrogate models...")
model_eff = joblib.load('models/xgboost_efficiency.pkl')
model_auto = joblib.load('models/xgboost_autonomy.pkl')
print("Models loaded.")

RES_TYPE_MAP = {'new_tank': 0, 'excavated': 1, 'pond': 2, 'river': 3}
RES_TYPE_CODE = RES_TYPE_MAP.get(USER.upper_reservoir_type, 0)


# ============================================================================
# DIAGNOSTIC
# ============================================================================

print("\nDIAGNOSTIC: Testing random designs...")
auto_values = []
for i in range(20):
    test_ind = [
        random.uniform(BOUNDS['volume_m3'][0], BOUNDS['volume_m3'][1]),
        random.uniform(BOUNDS['head_m'][0], BOUNDS['head_m'][1]),
        random.uniform(BOUNDS['pipe_diameter_m'][0], BOUNDS['pipe_diameter_m'][1]),
        random.uniform(BOUNDS['pump_power_kw'][0], BOUNDS['pump_power_kw'][1]),
        random.uniform(BOUNDS['turbine_power_kw'][0], BOUNDS['turbine_power_kw'][1])
    ]
    X = np.array([[
        test_ind[0], test_ind[1], test_ind[2], test_ind[3], test_ind[4],
        USER.pv_kwp, USER.daily_energy_kwh, USER.evaporation_rate_mm_month,
        RES_TYPE_CODE
    ]])
    auto = model_auto.predict(X)[0]
    auto_values.append(auto)
    print(f"  Volume={test_ind[0]:.0f}, Head={test_ind[1]:.1f} → Autonomy={auto:.2f} days")

print(f"\n  Max autonomy found: {max(auto_values):.2f} days")
print(f"  Required: {USER.autonomy_days} days")


# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def evaluate(individual):
    """Evaluate a design."""
    
    X = np.array([[
        individual[0], individual[1], individual[2], individual[3], individual[4],
        USER.pv_kwp, USER.daily_energy_kwh, USER.evaporation_rate_mm_month,
        RES_TYPE_CODE
    ]])
    
    efficiency = model_eff.predict(X)[0]
    autonomy = model_auto.predict(X)[0]
    
    # Cost
    design = {
        'volume_m3': individual[0],
        'head_m': individual[1],
        'pipe_diameter_m': individual[2],
        'pump_power_kw': individual[3],
        'turbine_power_kw': individual[4]
    }
    
    cost_dict = calculate_capital_cost(
        design['volume_m3'], design['head_m'], design['pipe_diameter_m'],
        design['pump_power_kw'], design['turbine_power_kw'],
        USER.pv_kwp, USER.upper_reservoir_type, USER.lower_reservoir_type
    )
    cost = cost_dict['total_lkr']
    
    # ===== CONSTRAINTS =====
    # 1. Autonomy
    if autonomy < USER.autonomy_days:  # 2.0 days
        return [1000.0, 100000000.0]
    
    # 2. Efficiency (NEW)
    if efficiency < 80.0:
        return [1000.0, 100000000.0]
    
    # Valid design
    return [-efficiency, cost]


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
# RUN
# ============================================================================

def run_optimization():
    print("=" * 70)
    print("NSGA-II OPTIMIZATION")
    print("=" * 70)
    print(f"Population: {POPULATION_SIZE}")
    print(f"Generations: {N_GENERATIONS}")
    print(f"Autonomy Constraint: >= {USER.autonomy_days} days")
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
        stats=stats, verbose=True
    )
    
    return population

# ============================================================================
# EXTRACT PARETO FRONT
# ============================================================================

def extract_pareto_front(population):
    """
    Extract Pareto front designs from NSGA-II population.
    
    Args:
        population: DEAP population of individuals
    
    Returns:
        list of dicts with design parameters and objectives
    """
    pareto_front = []
    
    for ind in population:
        if ind.fitness.values[0] < 1000:  # Valid design (not penalized)
            pareto_front.append({
                'volume_m3': ind[0],
                'head_m': ind[1],
                'pipe_diameter_m': ind[2],
                'pump_power_kw': ind[3],
                'turbine_power_kw': ind[4],
                'efficiency': -ind.fitness.values[0],
                'cost': ind.fitness.values[1]
            })
    
    # Sort by efficiency (descending)
    pareto_front = sorted(pareto_front, key=lambda x: x['efficiency'], reverse=True)
    
    return pareto_front


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    population = run_optimization()
    
    # Extract valid designs
    valid = []
    for ind in population:
        if ind.fitness.values[0] < 1000:
            valid.append({
                'volume_m3': ind[0],
                'head_m': ind[1],
                'pipe_diameter_m': ind[2],
                'pump_power_kw': ind[3],
                'turbine_power_kw': ind[4],
                'efficiency': -ind.fitness.values[0],
                'cost': ind.fitness.values[1]
            })
    
    if valid:
        valid = sorted(valid, key=lambda x: x['efficiency'], reverse=True)
        print("\n" + "=" * 70)
        print(f"VALID DESIGNS FOUND: {len(valid)}")
        print("=" * 70)
        for i, d in enumerate(valid[:10]):
            print(f"{i+1}: Vol={d['volume_m3']:.0f}, Head={d['head_m']:.1f}, Eff={d['efficiency']:.1f}%, Cost={d['cost']:,.0f} LKR")
    else:
        print("\n NO VALID DESIGNS FOUND. Try increasing bounds.")