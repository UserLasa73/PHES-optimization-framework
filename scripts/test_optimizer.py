"""
test_optimizer.py
Test the optimizer with specific user inputs.
"""

from src.user_inputs import UserInputs
from src.simulator import PumpedHydroSimulator
from src.solar_data_loader import fetch_solar_data, fetch_load_data
from optimization import run_optimization, extract_pareto_front

print("=" * 70)
print("TESTING OPTIMIZER WITH SAMPLE INPUTS")
print("=" * 70)

# ===== SAMPLE USER =====
user = UserInputs()
user.latitude = 8.9
user.longitude = 79.9
user.pv_kwp = 30.0
user.tilt_angle = 10.0
user.azimuth_angle = 0.0
user.daily_energy_kwh = 50.0
user.upper_reservoir_type = "new_tank"
user.lower_reservoir_type = "new_tank"
user.autonomy_days = 0.5  # ← LOW for testing
user.evaporation_rate_mm_month = 50.0
user.demand_spike_factor = 1.0
user.has_grid_backup = False
user.pipe_roughness_m = 0.00015

print("\nUser Inputs:")
print(f"  PV Capacity: {user.pv_kwp} kWp")
print(f"  Daily Load: {user.daily_energy_kwh} kWh/day")
print(f"  Reservoir Type: {user.upper_reservoir_type}")
print(f"  Autonomy Required: {user.autonomy_days} days")
print("=" * 70)

# ===== RUN OPTIMIZER =====
population = run_optimization(user)
pareto_front = extract_pareto_front(population)

# ===== DISPLAY RESULTS =====
if pareto_front:
    print(f"\n Found {len(pareto_front)} optimal designs!")
    print("\nTop 10 Designs (Pareto Front):")
    print("-" * 80)
    print(f"{'#':<4} {'Volume':<10} {'Head':<8} {'Pipe':<8} {'Pump':<8} {'Turbine':<10} {'Efficiency':<12} {'Cost (LKR)':<15}")
    print("-" * 80)
    
    for i, d in enumerate(pareto_front[:10]):
        print(f"{i+1:<4} {d['volume_m3']:<10.0f} {d['head_m']:<8.1f} {d['pipe_diameter_m']:<8.2f} {d['pump_power_kw']:<8.1f} {d['turbine_power_kw']:<10.1f} {d['efficiency']:<12.1f} {d['cost']:,.0f}")
    
    print("-" * 80)
    
    # ===== BEST DESIGN =====
    best = pareto_front[0]
    print("\n BEST DESIGN:")
    print(f"  Reservoir Volume: {best['volume_m3']:.0f} m³")
    print(f"  Head Height: {best['head_m']:.1f} m")
    print(f"  Pipe Diameter: {best['pipe_diameter_m']:.2f} m")
    print(f"  Pump Power: {best['pump_power_kw']:.1f} kW")
    print(f"  Turbine Power: {best['turbine_power_kw']:.1f} kW")
    print(f"  Efficiency: {best['efficiency']:.1f}%")
    print(f"  Cost: {best['cost']:,.0f} LKR")
    
else:
    print("\n No valid designs found.")
    print("Try relaxing constraints or increasing bounds.")