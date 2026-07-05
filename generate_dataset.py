"""
generate_dataset.py
Generate training data for XGBoost surrogate models.
Uses Latin Hypercube Sampling (LHS) via scipy.
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

N_SAMPLES = 2000

USER = UserInputs()
USER.latitude = 8.9
USER.longitude = 79.9
USER.pv_kwp = 30.0
USER.tilt_angle = 10.0
USER.azimuth_angle = 0.0
USER.daily_energy_kwh = 50.0
USER.upper_reservoir_type = "new_tank"
USER.lower_reservoir_type = "new_tank"
USER.autonomy_days = 2.0
USER.evaporation_rate_mm_month = 50.0
USER.demand_spike_factor = 1.0
USER.has_grid_backup = False
USER.pipe_roughness_m = 0.00015

# ============================================================================
# PARAMETER BOUNDS
# ============================================================================

BOUNDS = {
    'volume_m3': (50, 500),
    'head_m': (10, 30),
    'pipe_diameter_m': (0.1, 0.3),
    'pump_power_kw': (3, 15),
    'turbine_power_kw': (2, 10)
}

VARIABLE_NAMES = list(BOUNDS.keys())
VARIABLE_RANGES = [BOUNDS[name] for name in VARIABLE_NAMES]
N_VARIABLES = len(VARIABLE_NAMES)


# ============================================================================
# LHS FUNCTION (Using scipy)
# ============================================================================

def generate_lhs_samples(n_samples, n_vars, bounds):
    """Generate Latin Hypercube samples using scipy."""
    sampler = qmc.LatinHypercube(d=n_vars)
    samples = sampler.random(n=n_samples)
    
    # Scale to bounds
    scaled = np.zeros_like(samples)
    for i, (low, high) in enumerate(bounds):
        scaled[:, i] = low + samples[:, i] * (high - low)
    
    return scaled


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_dataset(n_samples=N_SAMPLES):
    """Generate dataset using LHS."""
    
    print("=" * 70)
    print("DATASET GENERATION")
    print("=" * 70)
    print(f"Target samples: {n_samples}")
    print(f"Variables: {VARIABLE_NAMES}")
    print(f"PV Capacity: {USER.pv_kwp} kWp")
    print(f"Daily Load: {USER.daily_energy_kwh} kWh")
    print("=" * 70)
    
    # Step 1: Get solar/load data
    print("\n Fetching solar and load data...")
    solar_data = fetch_solar_data(USER)
    load_data = fetch_load_data(USER)
    print(f"   Solar: {sum(solar_data):.0f} kWh/year")
    print(f"   Load:  {sum(load_data):.0f} kWh/year")
    
    # Step 2: Generate LHS samples
    print(f"\n Generating {n_samples} LHS samples...")
    scaled_samples = generate_lhs_samples(n_samples, N_VARIABLES, VARIABLE_RANGES)
    print(f"   Shape: {scaled_samples.shape}")
    
    # Step 3: Run simulations
    print(f"\n Running {n_samples} simulations...")
    results = []
    start_time = time.time()
    
    for i in range(n_samples):
        # Create design
        design = {}
        for j, name in enumerate(VARIABLE_NAMES):
            design[name] = scaled_samples[i, j]
        
        # Run simulation
        sim = PumpedHydroSimulator(USER, design)
        sim_result = sim.simulate(solar_data, load_data)
        metrics = sim_result['metrics']
        
        # Store
        results.append({
            'volume_m3': design['volume_m3'],
            'head_m': design['head_m'],
            'pipe_diameter_m': design['pipe_diameter_m'],
            'pump_power_kw': design['pump_power_kw'],
            'turbine_power_kw': design['turbine_power_kw'],
            'efficiency': metrics['efficiency_percent'],
            'cost': metrics['capital_cost_lkr'],
            'autonomy': metrics['autonomy_days'],
            'autonomy_met': 1 if metrics['autonomy_met'] else 0,
            'pumped': metrics['total_pumped_kwh'],
            'generated': metrics['total_generated_kwh'],
            'unmet': metrics['total_unmet_kwh'],
            'curtailed': metrics['total_curtailed_kwh']
        })
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"   Completed {i+1}/{n_samples} samples")
    
    elapsed = time.time() - start_time
    print(f"\n Completed in {elapsed:.1f} seconds")
    
    # Step 4: Create DataFrame
    df = pd.DataFrame(results)
    df = df[(df['efficiency'] >= 0) & (df['efficiency'] <= 100)]
    
    # Step 5: Summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nEfficiency: Mean={df['efficiency'].mean():.1f}%, Min={df['efficiency'].min():.1f}%, Max={df['efficiency'].max():.1f}%")
    print(f"Cost: Mean={df['cost'].mean():,.0f} LKR, Min={df['cost'].min():,.0f} LKR, Max={df['cost'].max():,.0f} LKR")
    print(f"Autonomy: Mean={df['autonomy'].mean():.2f} days")
    
    valid = df[df['autonomy_met'] == 1]
    print(f"Valid designs (autonomy >= 2 days): {len(valid)} ({len(valid)/len(df)*100:.1f}%)")
    
    # Step 6: Save
    df.to_csv('training_data_2000_samples.csv', index=False)
    print(f"\n Saved to: training_data_2000_samples.csv")
    
    return df


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    df = generate_dataset(N_SAMPLES)
    print("\nFirst 5 rows:")
    print(df.head().to_string())