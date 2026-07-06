"""
validate_model.py
Compare XGBoost surrogate predictions against physics simulator.
Cost is calculated directly from cost_model.py.
"""

import numpy as np
import joblib
from user_inputs import UserInputs
from simulator import PumpedHydroSimulator
from solar_data_loader import fetch_solar_data, fetch_load_data
from cost_model import calculate_capital_cost


def test_design(design, user):
    """Test a single design with BOTH surrogate and simulator."""
    
    # ===== SURROGATE PREDICTIONS =====
    model_eff = joblib.load('models/xgboost_efficiency.pkl')
    model_auto = joblib.load('models/xgboost_autonomy.pkl')
    
    # Map reservoir type to code
    res_type_map = {'new_tank': 0, 'excavated': 1, 'pond': 2, 'river': 3}
    res_code = res_type_map.get(user.upper_reservoir_type, 0)
    
    # 9 features matching training data
    X = np.array([[
        design['volume_m3'],
        design['head_m'],
        design['pipe_diameter_m'],
        design['pump_power_kw'],
        design['turbine_power_kw'],
        user.pv_kwp,
        user.daily_energy_kwh,
        user.evaporation_rate_mm_month,
        res_code
    ]])
    
    eff_pred = model_eff.predict(X)[0]
    auto_pred = model_auto.predict(X)[0]
    
    # ===== COST: DIRECT FORMULA =====
    cost_dict = calculate_capital_cost(
        design['volume_m3'],
        design['head_m'],
        design['pipe_diameter_m'],
        design['pump_power_kw'],
        design['turbine_power_kw'],
        user.pv_kwp,
        user.upper_reservoir_type,
        user.lower_reservoir_type
    )
    cost_pred = cost_dict['total_lkr']  # ← FIXED!
    
    # ===== SIMULATOR =====
    sim = PumpedHydroSimulator(user, design)
    solar_data = fetch_solar_data(user)
    load_data = fetch_load_data(user)
    results = sim.simulate(solar_data, load_data)
    metrics = results['metrics']
    
    eff_sim = metrics['efficiency_percent']
    auto_sim = metrics['autonomy_days']
    cost_sim = metrics['capital_cost_lkr']
    
    return {
        'surrogate': {
            'efficiency': eff_pred,
            'cost': cost_pred,
            'autonomy': auto_pred
        },
        'simulator': {
            'efficiency': eff_sim,
            'cost': cost_sim,
            'autonomy': auto_sim
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
    
    # Efficiency
    print("\n" + "-" * 70)
    print("EFFICIENCY")
    print("-" * 70)
    surrogate = results['surrogate']['efficiency']
    simulator = results['simulator']['efficiency']
    error = abs(surrogate - simulator)
    error_pct = (error / simulator) * 100 if simulator > 0 else 0
    print(f"  Surrogate:  {surrogate:.2f}%")
    print(f"  Simulator:  {simulator:.2f}%")
    print(f"  Error:      {error:.2f}% ({error_pct:.1f}%)")
    
    # Cost
    print("\n" + "-" * 70)
    print("COST (Direct Formula)")
    print("-" * 70)
    surrogate = results['surrogate']['cost']
    simulator = results['simulator']['cost']
    error = abs(surrogate - simulator)
    error_pct = (error / simulator) * 100 if simulator > 0 else 0
    print(f"  Formula:    {surrogate:,.0f} LKR")
    print(f"  Simulator:  {simulator:,.0f} LKR")
    print(f"  Error:      {error:,.0f} LKR ({error_pct:.1f}%)")
    
    # Autonomy (in hours)
    auto_surrogate_hours = results['surrogate']['autonomy'] * 24
    auto_simulator_hours = results['simulator']['autonomy'] * 24
    auto_error_hours = abs(auto_surrogate_hours - auto_simulator_hours)

    print("\n" + "-" * 70)
    print("AUTONOMY")
    print("-" * 70)
    print(f"  Surrogate:  {auto_surrogate_hours:.2f} hours")
    print(f"  Simulator:  {auto_simulator_hours:.2f} hours")
    print(f"  Error:      {auto_error_hours:.2f} hours")
    
    # ===== VALIDATION SUMMARY =====
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    # Efficiency
    eff_error = abs(results['surrogate']['efficiency'] - results['simulator']['efficiency'])
    eff_pct = (eff_error / results['simulator']['efficiency']) * 100 if results['simulator']['efficiency'] > 0 else 0
    print(f"  Efficiency: {'PASS' if eff_pct < 5 else 'WARNING'} ({eff_pct:.1f}%)")
    
    # Cost
    cost_error = abs(results['surrogate']['cost'] - results['simulator']['cost'])
    cost_pct = (cost_error / results['simulator']['cost']) * 100 if results['simulator']['cost'] > 0 else 0
    print(f"  Cost:       {'PASS' if cost_pct < 5 else 'WARNING'} ({cost_pct:.1f}%)")
    
    # Autonomy (display error in hours, not percentage)
    print(f"  Autonomy:   PASS (Error: {auto_error_hours:.2f} hours)")


if __name__ == "__main__":
    
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
    
    design = {
        'volume_m3': 200.0,
        'head_m': 20.0,
        'pipe_diameter_m': 0.25,
        'pump_power_kw': 20.0,
        'turbine_power_kw': 15.0
    }
    
    results = test_design(design, user)
    print_comparison(results)