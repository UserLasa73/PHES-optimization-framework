"""
generate_dataset.py
Generate full-spectrum training data for XGBoost surrogate models.
Covers all 8 Sri Lankan locations from the proposal.
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

SAMPLES_PER_LOCATION = 400   # 400 x 8 = 3,200 total samples

# ===== RESERVOIR TYPE MAPPING =====
RES_TYPE_MAP = {
    0: 'new_tank',
    1: 'excavated',
    2: 'pond',
    3: 'river'
}

# ===== BASE USER =====
BASE_USER = UserInputs()
BASE_USER.year = 2025
BASE_USER.tilt_angle = 10.0
BASE_USER.azimuth_angle = 0.0
BASE_USER.autonomy_days = 2.0
BASE_USER.demand_spike_factor = 1.0
BASE_USER.has_grid_backup = False
BASE_USER.pipe_roughness_m = 0.00015

# ============================================================================
# BOUNDS
# ============================================================================

BOUNDS = {
    'volume_m3': (20, 800),
    'head_m': (5, 45),
    'pipe_diameter_m': (0.05, 0.35),
    'pump_power_kw': (2, 30),
    'turbine_power_kw': (2, 25),
    'pv_kwp': (5, 30),
    'daily_energy_kwh': (10, 50),
    'evaporation_rate_mm_month': (30, 80),
    'reservoir_type_code': (0, 3)
}

VARIABLE_NAMES = list(BOUNDS.keys())
VARIABLE_RANGES = [BOUNDS[name] for name in VARIABLE_NAMES]
N_VARIABLES = len(VARIABLE_NAMES)


def generate_lhs_samples(n_samples):
    """Generate LHS samples with log scaling."""
    sampler = qmc.LatinHypercube(d=N_VARIABLES)
    samples = sampler.random(n=n_samples)
    
    log_params = [0, 2, 3, 4]
    
    scaled = np.zeros_like(samples)
    for i, (low, high) in enumerate(VARIABLE_RANGES):
        if i in log_params:
            log_low = np.log10(max(low, 0.001))
            log_high = np.log10(high)
            log_values = log_low + samples[:, i] * (log_high - log_low)
            scaled[:, i] = 10 ** log_values
        elif i == 8:
            scaled[:, i] = np.floor(low + samples[:, i] * (high - low + 1))
            scaled[:, i] = np.clip(scaled[:, i], low, high)
        else:
            scaled[:, i] = low + samples[:, i] * (high - low)
    
    return scaled


def run_single_simulation(user, design, solar_data, load_data):
    """Run one simulation and return metrics."""
    sim = PumpedHydroSimulator(user, design)
    result = sim.simulate(solar_data, load_data)
    metrics = result['metrics']
    
    return {
        'volume_m3': design['volume_m3'],
        'head_m': design['head_m'],
        'pipe_diameter_m': design['pipe_diameter_m'],
        'pump_power_kw': design['pump_power_kw'],
        'turbine_power_kw': design['turbine_power_kw'],
        'pv_kwp': user.pv_kwp,
        'daily_energy_kwh': user.daily_energy_kwh,
        'evaporation_rate_mm_month': user.evaporation_rate_mm_month,
        'reservoir_type_code': user.reservoir_type_code,
        'efficiency': metrics['efficiency_percent'],
        'cost': metrics['capital_cost_lkr'],
        'autonomy': metrics['autonomy_days'],
        'pumped': metrics['total_pumped_kwh'],
        'generated': metrics['total_generated_kwh'],
        'unmet': metrics['total_unmet_kwh'],
        'curtailed': metrics['total_curtailed_kwh'],
        'reservoir_type': user.upper_reservoir_type,
        'location': user.location
    }


def generate_dataset():
    """Generate dataset for all 8 locations."""
    
    total_samples = SAMPLES_PER_LOCATION * len(LOCATIONS)
    
    print("=" * 70)
    print("FULL-SPECTRUM DATASET GENERATION")
    print("=" * 70)
    print(f"Locations: {len(LOCATIONS)} (from proposal)")
    for loc in LOCATIONS:
        print(f"  - {loc['name']} ({loc['lat']}N, {loc['lon']}E)")
    print(f"Samples per location: {SAMPLES_PER_LOCATION}")
    print(f"Total samples: {total_samples}")
    print("=" * 70)
    print("NOTE: No filters applied. Bad, average, and good designs all included.")
    print("=" * 70)
    
    lhs_samples = generate_lhs_samples(SAMPLES_PER_LOCATION)
    
    all_results = []
    start_time = time.time()
    
    for loc_idx, loc in enumerate(LOCATIONS):
        print(f"\nLocation {loc_idx+1}/{len(LOCATIONS)}: {loc['name']}")
        
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
        
        solar_data = fetch_solar_data(user)
        load_data = fetch_load_data(user)
        
        print(f"  Running {SAMPLES_PER_LOCATION} simulations...")
        
        for i in range(SAMPLES_PER_LOCATION):
            design = {
                'volume_m3': lhs_samples[i, 0],
                'head_m': lhs_samples[i, 1],
                'pipe_diameter_m': lhs_samples[i, 2],
                'pump_power_kw': lhs_samples[i, 3],
                'turbine_power_kw': lhs_samples[i, 4]
            }
            
            user.pv_kwp = lhs_samples[i, 5]
            user.daily_energy_kwh = lhs_samples[i, 6]
            user.evaporation_rate_mm_month = lhs_samples[i, 7]
            res_code = int(lhs_samples[i, 8])
            user.upper_reservoir_type = RES_TYPE_MAP[res_code]
            user.lower_reservoir_type = RES_TYPE_MAP[res_code]
            user.reservoir_type_code = res_code
            
            result = run_single_simulation(user, design, solar_data, load_data)
            all_results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"    Completed {i+1}/{SAMPLES_PER_LOCATION}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    
    # ===== CREATE DATAFRAME =====
    df_full = pd.DataFrame(all_results)
    print(f"\nTotal rows generated: {len(df_full)}")
    
    # ===== CLEAN: ONLY 12 COLUMNS FOR TRAINING =====
    df_clean = df_full[[
        'volume_m3', 'head_m', 'pipe_diameter_m',
        'pump_power_kw', 'turbine_power_kw',
        'pv_kwp', 'daily_energy_kwh',
        'evaporation_rate_mm_month', 'reservoir_type_code',
        'efficiency', 'cost', 'autonomy'
    ]]
    
    df_clean = df_clean[(df_clean['efficiency'] >= 0) & (df_clean['efficiency'] <= 100)]
    print(f"Valid rows: {len(df_clean)}")
    
    # ===== STATISTICS =====
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\nEfficiency:")
    print(f"  Mean:  {df_clean['efficiency'].mean():.1f}%")
    print(f"  Min:   {df_clean['efficiency'].min():.1f}%")
    print(f"  Max:   {df_clean['efficiency'].max():.1f}%")
    print(f"  Std:   {df_clean['efficiency'].std():.1f}%")
    
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    df_clean['eff_range'] = pd.cut(df_clean['efficiency'], bins=bins, labels=labels, right=False)
    counts = df_clean['eff_range'].value_counts().sort_index()
    
    print("\nEfficiency Range:")
    for label, count in counts.items():
        print(f"  {label}: {count} ({count/len(df_clean)*100:.1f}%)")
    
    print(f"\nAutonomy:")
    print(f"  Mean:  {df_clean['autonomy'].mean():.2f} days")
    print(f"  Min:   {df_clean['autonomy'].min():.2f} days")
    print(f"  Max:   {df_clean['autonomy'].max():.2f} days")
    
    print(f"\nCost:")
    print(f"  Mean:  {df_clean['cost'].mean():,.0f} LKR")
    print(f"  Min:   {df_clean['cost'].min():,.0f} LKR")
    print(f"  Max:   {df_clean['cost'].max():,.0f} LKR")
    
    # ===== SAVE =====
    df_clean.to_csv('training_data_all_inputs.csv', index=False)
    print(f"\nSaved: training_data_all_inputs.csv")
    print(f"Columns: {list(df_clean.columns)}")
    print("=" * 70)
    
    return df_clean


if __name__ == "__main__":
    df = generate_dataset()
    print("\nFirst 5 rows:")
    print(df.head().to_string())