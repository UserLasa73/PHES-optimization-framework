"""
generate_dataset.py
Generate training data for MULTIPLE Sri Lankan locations.
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

SAMPLES_PER_LOCATION = 300  # 300 × 8 = 2400 samples
TOTAL_SAMPLES = SAMPLES_PER_LOCATION * len(LOCATIONS)

# Fixed user inputs (except location)
USER = UserInputs()
USER.year = 2025
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
# FUNCTIONS
# ============================================================================

def generate_lhs_samples(n_samples, n_vars, bounds):
    """Generate Latin Hypercube samples using scipy."""
    sampler = qmc.LatinHypercube(d=n_vars)
    samples = sampler.random(n=n_samples)
    
    scaled = np.zeros_like(samples)
    for i, (low, high) in enumerate(bounds):
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
        'latitude': user.latitude,
        'longitude': user.longitude,
        'location': user.location,
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
    """Generate dataset for all locations."""
    
    print("=" * 70)
    print("MULTI-LOCATION DATASET GENERATION")
    print("=" * 70)
    print(f"Locations: {len(LOCATIONS)}")
    print(f"Samples per location: {SAMPLES_PER_LOCATION}")
    print(f"Total samples: {TOTAL_SAMPLES}")
    print("=" * 70)
    
    # Generate LHS designs (same for all locations)
    print(f"\n Generating {SAMPLES_PER_LOCATION} LHS designs...")
    lhs_samples = generate_lhs_samples(SAMPLES_PER_LOCATION, N_VARIABLES, VARIABLE_RANGES)
    
    all_results = []
    
    for loc_idx, loc in enumerate(LOCATIONS):
        print(f"\n" + "-" * 70)
        print(f" Location: {loc['name']} ({loc['lat']}°N, {loc['lon']}°E)")
        print("-" * 70)
        
        # Update user with this location
        USER.location = loc['name']
        USER.latitude = loc['lat']
        USER.longitude = loc['lon']
        
        # Fetch solar data for this location
        print(f"   Fetching solar data...")
        solar_data = fetch_solar_data(USER)
        load_data = fetch_load_data(USER)
        
        print(f"   Solar: {sum(solar_data):.0f} kWh/year")
        print(f"   Running {SAMPLES_PER_LOCATION} simulations...")
        
        start_time = time.time()
        
        for i in range(SAMPLES_PER_LOCATION):
            # Create design
            design = {}
            for j, name in enumerate(VARIABLE_NAMES):
                design[name] = lhs_samples[i, j]
            
            # Run simulation
            result = run_single_simulation(USER, design, solar_data, load_data)
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
    
    # Save
    df.to_csv('training_data_multi_location.csv', index=False)
    print(f"\n Saved to: training_data_multi_location.csv")
    
    return df


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    df = generate_dataset()
    print("\nFirst 5 rows:")
    print(df.head().to_string())