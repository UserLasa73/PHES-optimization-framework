"""
test_simulator.py
Quick test to verify simulator works with your code.
"""

import numpy as np
from user_inputs import UserInputs
from simulator import PumpedHydroSimulator
from solar_data_loader import fetch_solar_data, fetch_load_data

# ===== 1. CREATE USER INPUTS =====
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

# ===== 2. CREATE A DESIGN TO TEST =====
design = {
    'head_m': 25.0,
    'volume_m3': 5000.0,
    'pipe_diameter_m': 0.25,
    'pump_power_kw': 20.0,
    'turbine_power_kw': 15.0
}

# ===== 3. FETCH DATA (FIXED) =====
print("=" * 60)
print("FETCHING DATA")
print("=" * 60)

# ✅ FIXED: Pass user object, not individual values
solar_data = fetch_solar_data(user)  # ← ONE argument

# ✅ FIXED: Use fetch_load_data with user object
load_data = fetch_load_data(user)    # ← ONE argument

# ===== 4. VERIFY =====
print(f"Solar length: {len(solar_data)} hours")
print(f"Load length:  {len(load_data)} hours")

# Trim if needed
if len(solar_data) > 8760:
    print(f"⚠️ Solar has {len(solar_data)} hours, trimming to 8760")
    solar_data = solar_data[:8760]
if len(load_data) > 8760:
    print(f"⚠️ Load has {len(load_data)} hours, trimming to 8760")
    load_data = load_data[:8760]

assert len(solar_data) == 8760, f"Solar is {len(solar_data)}, expected 8760"
assert len(load_data) == 8760, f"Load is {len(load_data)}, expected 8760"
print("✅ Both solar and load data have 8760 hours!")

print(f"\nTotal Solar: {sum(solar_data):.0f} kWh/year")
print(f"Total Load:  {sum(load_data):.0f} kWh/year")
print(f"PV Size: {user.pv_kwp} kWp")
print(f"Head: {design['head_m']} m")
print(f"Volume: {design['volume_m3']} m³")
print(f"Reservoir Types: {user.upper_reservoir_type} / {user.lower_reservoir_type}")

# ===== 5. RUN SIMULATION =====
print("\n🔄 Running simulation...")
sim = PumpedHydroSimulator(user, design)
results = sim.simulate(solar_data, load_data)

# ===== 6. DISPLAY RESULTS =====
metrics = results['metrics']

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Efficiency: {metrics['efficiency_percent']:.1f}%")
print(f"Autonomy: {metrics['autonomy_days']:.1f} days (needed: {user.autonomy_days} days)")
print(f"Autonomy Met: {'✅ YES' if metrics['autonomy_met'] else '❌ NO'}")
print(f"Capital Cost: {metrics['capital_cost_lkr']:,.0f} LKR")
print(f"\nEnergy Summary:")
print(f"  Pumped: {metrics['total_pumped_kwh']:.0f} kWh")
print(f"  Generated: {metrics['total_generated_kwh']:.0f} kWh")
print(f"  Unmet Load: {metrics['total_unmet_kwh']:.0f} kWh")
print(f"  Curtailed: {metrics['total_curtailed_kwh']:.0f} kWh")
print(f"  Grid Used: {metrics['grid_used_kwh']:.0f} kWh")
print(f"\nFinal Reservoir Levels:")
print(f"  Upper: {(sim.upper_volume / sim.max_upper * 100):.1f}%")
print(f"  Lower: {(sim.lower_volume / sim.max_lower * 100):.1f}%")

# ===== 7. SHOW FIRST 24 HOURS =====
print("\n" + "=" * 60)
print("FIRST 24 HOURS")
print("=" * 60)
print(f"{'Hour':<6} {'Solar':<8} {'Load':<8} {'Upper%':<8} {'State':<15}")
print("-" * 50)

for i in range(24):
    h = sim.history
    state = h['state'][i][:15] if len(h['state'][i]) > 15 else h['state'][i]
    upper_pct = (h['upper_volume'][i] / sim.max_upper) * 100
    print(f"{i:<6} {h['solar_power'][i]:<8.1f} {h['load_power'][i]:<8.1f} {upper_pct:<8.1f} {state:<15}")