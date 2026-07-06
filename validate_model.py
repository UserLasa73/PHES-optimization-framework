"""
validate_model.py
Compare XGBoost surrogate predictions against physics simulator.
"""

import numpy as np
import pandas as pd
import joblib
from user_inputs import UserInputs
from simulator import PumpedHydroSimulator
from solar_data_loader import fetch_solar_data, fetch_load_data


def test_design(design, user):
    """
    Test a single design with BOTH surrogate and simulator.
    
    Args:
        design: dict with volume_m3, head_m, pipe_diameter_m, pump_power_kw, turbine_power_kw
        user: UserInputs object
    
    Returns:
        dict with surrogate and simulator results
    """
    
    # ===== SURROGATE PREDICTIONS =====
    model_eff = joblib.load('models/xgboost_efficiency.pkl')
    model_cost = joblib.load('models/xgboost_cost.pkl')
    
    X = np.array([[
        design['volume_m3'],
        design['head_m'],
        design['pipe_diameter_m'],
        design['pump_power_kw'],
        design['turbine_power_kw']
    ]])
    
    eff_pred = model_eff.predict(X)[0]
    cost_pred = model_cost.predict(X)[0]
    
    # ===== SIMULATOR PREDICTIONS =====
    sim = PumpedHydroSimulator(user, design)
    
    # Get solar and load data
    solar_data = fetch_solar_data(user)
    load_data = fetch_load_data(user)
    
    results = sim.simulate(solar_data, load_data)
    metrics = results['metrics']
    
    eff_sim = metrics['efficiency_percent']
    cost_sim = metrics['capital_cost_lkr']
    autonomy_sim = metrics['autonomy_days']
    
    return {
        'surrogate': {
            'efficiency': eff_pred,
            'cost': cost_pred
        },
        'simulator': {
            'efficiency': eff_sim,
            'cost': cost_sim,
            'autonomy': autonomy_sim
        },
        'design': design
    }


def print_comparison(results):
    """Print comparison between surrogate and simulator."""
    
    print("=" * 70)
    print("SURROGATE VS SIMULATOR COMPARISON")
    print("=" * 70)
    
    print("\nDesign Parameters:")
    for key, value in results['design'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 70)
    print("EFFICIENCY")
    print("-" * 70)
    surrogate = results['surrogate']['efficiency']
    simulator = results['simulator']['efficiency']
    error = abs(surrogate - simulator)
    error_pct = (error / simulator) * 100 if simulator > 0 else 0
    
    print(f"  Surrogate:  {surrogate:.2f}%")
    print(f"  Simulator:  {simulator:.2f}%")
    print(f"  Difference: {error:.2f}% ({error_pct:.1f}% error)")
    
    print("\n" + "-" * 70)
    print("COST")
    print("-" * 70)
    surrogate = results['surrogate']['cost']
    simulator = results['simulator']['cost']
    error = abs(surrogate - simulator)
    error_pct = (error / simulator) * 100 if simulator > 0 else 0
    
    print(f"  Surrogate:  {surrogate:,.0f} LKR")
    print(f"  Simulator:  {simulator:,.0f} LKR")
    print(f"  Difference: {error:,.0f} LKR ({error_pct:.1f}% error)")
    
    print("\n" + "-" * 70)
    print("AUTONOMY (Simulator only)")
    print("-" * 70)
    autonomy = results['simulator']['autonomy']
    print(f"  Simulator:  {autonomy:.2f} days")
    
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    if error_pct < 5:
        print("  PASS: Error < 5%")
    else:
        print(f"  WARNING: Error = {error_pct:.1f}% (target: < 5%)")


if __name__ == "__main__":
    
    # ===== USER INPUTS =====
    user = UserInputs()
    user.latitude = 8.9
    user.longitude = 79.9
    user.pv_kwp = 30.0
    user.tilt_angle = 10.0
    user.azimuth_angle = 0.0
    user.daily_energy_kwh = 50.0
    user.upper_reservoir_type = "new_tank"
    user.lower_reservoir_type = "new_tank"
    user.autonomy_days = 2.0
    user.evaporation_rate_mm_month = 50.0
    user.demand_spike_factor = 1.0
    user.has_grid_backup = False
    user.pipe_roughness_m = 0.00015
    
    # ===== TEST DESIGN =====
    design = {
        'volume_m3': 200.0,
        'head_m': 20.0,
        'pipe_diameter_m': 0.25,
        'pump_power_kw': 20.0,
        'turbine_power_kw': 15.0
    }
    
    # ===== RUN COMPARISON =====
    results = test_design(design, user)
    print_comparison(results)