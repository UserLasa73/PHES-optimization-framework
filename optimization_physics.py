"""
optimization_physics.py
NSGA-II Multi-Objective Optimization using REAL Physics Simulator.
This is a SEPARATE file - does NOT affect the ML optimizer.
"""

import numpy as np
import random
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings('ignore')

from user_inputs import UserInputs
from cost_model import calculate_capital_cost
from simulator import PumpedHydroSimulator
from solar_data_loader import fetch_solar_data, fetch_load_data


# ============================================================================
# CONFIGURATION
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
DEFAULT_USER.max_volume_m3 = 800  # Default matches current bound 800

CURRENT_USER = DEFAULT_USER

# ===== BOUNDS =====
def get_bounds(user=None):
    if user is None:
        user = CURRENT_USER
    
    bounds = {
        'volume_m3': (20, user.max_volume_m3),
        'head_m': (5, 45),
        'pipe_diameter_m': (0.05, 0.35),
        'pump_power_kw': (2, 30),
        'turbine_power_kw': (2, 25)
    }
    return bounds

POPULATION_SIZE = 100
N_GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.2


# ============================================================================
# FITNESS FUNCTION (Uses REAL Physics Simulator)
# ============================================================================

def evaluate(individual):
    """Evaluate a design using the REAL Physics Simulator."""
    
    user = CURRENT_USER
    
    # Extract design parameters
    design = {
        'volume_m3': individual[0],
        'head_m': individual[1],
        'pipe_diameter_m': individual[2],
        'pump_power_kw': individual[3],
        'turbine_power_kw': individual[4]
    }
    
    # ===== COST =====
    cost_dict = calculate_capital_cost(
        individual[0], individual[1], individual[2],
        individual[3], individual[4],
        user.pv_kwp,
        user.upper_reservoir_type,
        user.lower_reservoir_type
    )
    cost = cost_dict['total_lkr']
    
    # ===== EFFICIENCY & AUTONOMY (REAL SIMULATOR) =====
    solar_data = fetch_solar_data(user)
    load_data = fetch_load_data(user)
    
    sim = PumpedHydroSimulator(user, design)
    results = sim.simulate(solar_data, load_data)
    metrics = results['metrics']
    
    efficiency = metrics['efficiency_percent']
    autonomy = metrics['autonomy_days']
    
    # ===== SOFT CONSTRAINTS (PENALTIES) =====
    penalty = 0.0
    
    # Penalty for low efficiency
    if efficiency < 80.0:
        penalty += (80.0 - efficiency) * 1000
    
    # Penalty for low autonomy
    if autonomy < user.autonomy_days:
        penalty += (user.autonomy_days - autonomy) * 100000
    
    #Penalty for exceeding max volume
    if individual[0] > user.max_volume_m3:
        penalty += (individual[0] - user.max_volume_m3) * 100000
    
    adjusted_cost = cost + penalty
    
    return [-efficiency, adjusted_cost]


# ============================================================================
# SETUP DEAP
# ============================================================================

def setup_deap(user=None):
    if user is None:
        user = CURRENT_USER
    
    bounds = get_bounds(user)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    
    toolbox.register("attr_volume", random.uniform, bounds['volume_m3'][0], bounds['volume_m3'][1])
    toolbox.register("attr_head", random.uniform, bounds['head_m'][0], bounds['head_m'][1])
    toolbox.register("attr_pipe", random.uniform, bounds['pipe_diameter_m'][0], bounds['pipe_diameter_m'][1])
    toolbox.register("attr_pump", random.uniform, bounds['pump_power_kw'][0], bounds['pump_power_kw'][1])
    toolbox.register("attr_turbine", random.uniform, bounds['turbine_power_kw'][0], bounds['turbine_power_kw'][1])
    
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
# RUN OPTIMIZATION (Physics Mode)
# ============================================================================

def run_optimization_physics(user=None):
    """Run NSGA-II optimization using REAL Physics Simulator."""
    
    global CURRENT_USER
    
    if user is not None:
        CURRENT_USER = user
    else:
        CURRENT_USER = DEFAULT_USER
    
    print("=" * 70)
    print("NSGA-II OPTIMIZATION (PHYSICS SIMULATOR)")
    print("=" * 70)
    print(f"Reservoir Type: {CURRENT_USER.upper_reservoir_type}")
    print(f"PV Capacity: {CURRENT_USER.pv_kwp} kWp")
    print(f"Autonomy: >= {CURRENT_USER.autonomy_days} days")
    print(f"Max Volume: {CURRENT_USER.max_volume_m3} m3")  # Add this line
    print("=" * 70)
    print("  WARNING: Physics simulator mode is SLOW.")
    print(f"   {POPULATION_SIZE * N_GENERATIONS} evaluations")
    print(f"   Estimated time: ~{(POPULATION_SIZE * N_GENERATIONS * 0.5) / 60:.0f} minutes")
    print("=" * 70)
    
    toolbox = setup_deap(user=CURRENT_USER)  # Pass user here
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
    
    print("Optimization complete!")
    return population


# ============================================================================
# EXTRACT PARETO FRONT
# ============================================================================

def extract_pareto_front_physics(population):
    """Extract Pareto front designs from population."""
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