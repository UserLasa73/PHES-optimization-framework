"""
generate_dataset.py
Generate training data for MULTIPLE Sri Lankan locations with ALL user inputs.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import time
import warnings
warnings.filterwarnings('ignore')

from user_inputs import UserInputs
from simulator import PumpedHydroSimulator
from solar_data_loader import fetch_solar_data, fetch_load_data


# ============================================================================
# CONFIGURATION
# ============================================================================

# Multiple locations across Sri Lanka
LOCATIONS = [
    {'name': 'Vavuniya', 'lat': 8.9, 'lon': 79.9},
    {'name': 'Colombo', 'lat': 6.9, 'lon': 79.9},
    {'name': 'Jaffna', 'lat': 9.7, 'lon': 80.0},
    {'name': 'Kandy', 'lat': 7.3, 'lon': 80.6},
    {'name': 'Galle', 'lat': 6.0, 'lon': 80.2},
    {'name': 'Trincomalee', 'lat': 8.6, 'lon': 81.2},
    {'name': 'Batticaloa', 'lat': 7.7, 'lon': 81.7},
    {'name': 'Anuradhapura', 'lat': 8.3, 'lon': 80.4},
]

SAMPLES_PER_LOCATION = 500  # 500 × 8 = 4000 samples

# Reservoir type mapping
RES_TYPE_MAP = {
    0: 'new_tank',
    1: 'excavated',
    2: 'pond',
    3: 'river'
}

# Base fixed user parameters (location will change per location)
BASE_USER = UserInputs()
BASE_USER.year = 2025
BASE_USER.tilt_angle = 10.0
BASE_USER.azimuth_angle = 0.0
BASE_USER.autonomy_days = 2.0
BASE_USER.demand_spike_factor = 1.0
BASE_USER.has_grid_backup = False
BASE_USER.pipe_roughness_m = 0.00015
BASE_USER.upper_reservoir_type = "new_tank"
BASE_USER.lower_reservoir_type = "new_tank"


# ============================================================================
# PARAMETER BOUNDS (ALL inputs + design variables)
# ============================================================================

BOUNDS = {
    'volume_m3': (50, 2000),        # Small to large
    'head_m': (5, 50),              # Very low to very high
    'pipe_diameter_m': (0.05, 0.5), # Tiny to huge
    'pump_power_kw': (2, 50),       # Small to large pump
    'turbine_power_kw': (2, 40),    # Small to large turbine
    'pv_kwp': (5, 50),
    'daily_energy_kwh': (10, 100),
    'evaporation_rate_mm_month': (30, 80),
    'reservoir_type_code': (0, 3)
}

VARIABLE_NAMES = list(BOUNDS.keys())
VARIABLE_RANGES = [BOUNDS[name] for name in VARIABLE_NAMES]
N_VARIABLES = len(VARIABLE_NAMES)


# ============================================================================
# FUNCTIONS
# ============================================================================

def generate_lhs_samples(n_samples, n_vars, bounds):
    """Generate Latin Hypercube samples using scipy."""
    sampler = qmc.LatinHypercube(d=n_vars)
    samples = sampler.random(n=n_samples)
    
    scaled = np.zeros_like(samples)
    for i, (low, high) in enumerate(bounds):
        if i == 8:  # reservoir_type_code (integer)
            scaled[:, i] = np.floor(low + samples[:, i] * (high - low + 1))
            scaled[:, i] = np.clip(scaled[:, i], low, high)
        else:
            scaled[:, i] = low + samples[:, i] * (high - low)
    
    return scaled


def run_single_simulation(user, design, solar_data, load_data):
    """Run one simulation and return metrics with ALL inputs."""
    sim = PumpedHydroSimulator(user, design)
    result = sim.simulate(solar_data, load_data)
    metrics = result['metrics']
    
    return {
        # INPUTS (features)
        'volume_m3': design['volume_m3'],
        'head_m': design['head_m'],
        'pipe_diameter_m': design['pipe_diameter_m'],
        'pump_power_kw': design['pump_power_kw'],
        'turbine_power_kw': design['turbine_power_kw'],
        'pv_kwp': user.pv_kwp,
        'daily_energy_kwh': user.daily_energy_kwh,
        'evaporation_rate_mm_month': user.evaporation_rate_mm_month,
        'reservoir_type_code': user.reservoir_type_code,
        'reservoir_type': user.upper_reservoir_type,
        'location': user.location,
        'latitude': user.latitude,
        'longitude': user.longitude,
        
        # OUTPUTS (targets)
        'efficiency': metrics['efficiency_percent'],
        'cost': metrics['capital_cost_lkr'],
        'autonomy': metrics['autonomy_days'],
        'autonomy_met': 1 if metrics['autonomy_met'] else 0,
        'pumped': metrics['total_pumped_kwh'],
        'generated': metrics['total_generated_kwh'],
        'unmet': metrics['total_unmet_kwh'],
        'curtailed': metrics['total_curtailed_kwh']
    }


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_dataset():
    """Generate dataset for all locations with all inputs."""
    
    print("=" * 70)
    print("MULTI-LOCATION DATASET GENERATION (All Inputs)")
    print("=" * 70)
    print(f"Locations: {len(LOCATIONS)}")
    print(f"Samples per location: {SAMPLES_PER_LOCATION}")
    print(f"Total samples: {len(LOCATIONS) * SAMPLES_PER_LOCATION}")
    print(f"Variables: {VARIABLE_NAMES}")
    print("=" * 70)
    
    # Generate LHS designs (same for all locations)
    print(f"\n Generating {SAMPLES_PER_LOCATION} LHS designs...")
    lhs_samples = generate_lhs_samples(SAMPLES_PER_LOCATION, N_VARIABLES, VARIABLE_RANGES)
    
    all_results = []
    
    for loc_idx, loc in enumerate(LOCATIONS):
        print(f"\n" + "-" * 70)
        print(f" Location: {loc['name']} ({loc['lat']}°N, {loc['lon']}°E)")
        print("-" * 70)
        
        # Create user for this location (base + location)
        user = UserInputs()
        user.year = BASE_USER.year
        user.tilt_angle = BASE_USER.tilt_angle
        user.azimuth_angle = BASE_USER.azimuth_angle
        user.autonomy_days = BASE_USER.autonomy_days
        user.demand_spike_factor = BASE_USER.demand_spike_factor
        user.has_grid_backup = BASE_USER.has_grid_backup
        user.pipe_roughness_m = BASE_USER.pipe_roughness_m
        user.location = loc['name']
        user.latitude = loc['lat']
        user.longitude = loc['lon']
        
        # Fetch solar data for this location
        print(f"   Fetching solar data...")
        solar_data = fetch_solar_data(user)
        load_data = fetch_load_data(user)
        
        print(f"   Solar: {sum(solar_data):.0f} kWh/year")
        print(f"   Running {SAMPLES_PER_LOCATION} simulations...")
        
        start_time = time.time()
        
        for i in range(SAMPLES_PER_LOCATION):
            # Create design (design variables only)
            design = {
                'volume_m3': lhs_samples[i, 0],
                'head_m': lhs_samples[i, 1],
                'pipe_diameter_m': lhs_samples[i, 2],
                'pump_power_kw': lhs_samples[i, 3],
                'turbine_power_kw': lhs_samples[i, 4]
            }
            
            # Set user inputs (varied across samples)
            user.pv_kwp = lhs_samples[i, 5]
            user.daily_energy_kwh = lhs_samples[i, 6]
            user.evaporation_rate_mm_month = lhs_samples[i, 7]
            res_code = int(lhs_samples[i, 8])
            res_type = RES_TYPE_MAP[res_code]
            user.upper_reservoir_type = res_type
            user.lower_reservoir_type = res_type
            user.reservoir_type_code = res_code
            
            # Run simulation
            result = run_single_simulation(user, design, solar_data, load_data)
            all_results.append(result)
            
            # Progress
            if (i + 1) % 50 == 0:
                print(f"   Completed {i+1}/{SAMPLES_PER_LOCATION}")
        
        elapsed = time.time() - start_time
        print(f"   Done in {elapsed:.1f} seconds")
    
    # ===== CREATE DATAFRAME =====
    print("\n" + "=" * 70)
    print("SAVING DATASET")
    print("=" * 70)
    
    df = pd.DataFrame(all_results)
    df = df[(df['efficiency'] >= 0) & (df['efficiency'] <= 100)]
    
    print(f"Total samples: {len(df)}")
    print(f"Locations: {df['location'].unique().tolist()}")
    
    # Summary by location
    print("\nSummary by location:")
    print(df.groupby('location').agg({
        'efficiency': ['mean', 'min', 'max'],
        'cost': ['mean', 'min', 'max'],
        'autonomy': ['mean']
    }).round(2))
    
    # Summary by reservoir type
    print("\nSummary by reservoir type:")
    print(df.groupby('reservoir_type').agg({
        'efficiency': ['mean', 'min', 'max'],
        'cost': ['mean', 'min', 'max']
    }).round(2))
    
    # Save
    df.to_csv('training_data_all_inputs.csv', index=False)
    print(f"\n Saved to: training_data_all_inputs.csv")
    
    return df


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    df = generate_dataset()
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nFeature columns:", [c for c in df.columns if c not in ['efficiency', 'cost', 'autonomy', 'autonomy_met', 'pumped', 'generated', 'unmet', 'curtailed']])