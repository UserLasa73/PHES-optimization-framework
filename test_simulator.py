"""
test_simulator_debug.py
Quick test to check if simulator is correctly calculating losses.
"""

from user_inputs import UserInputs
from simulator import PumpedHydroSimulator
from solar_data_loader import fetch_solar_data, fetch_load_data

# ===== CREATE USER =====
user = UserInputs()
user.latitude = 8.9
user.longitude = 79.9
user.pv_kwp = 10.0
user.daily_energy_kwh = 20.0
user.upper_reservoir_type = "new_tank"
user.lower_reservoir_type = "new_tank"
user.evaporation_rate_mm_month = 50.0
user.pipe_roughness_m = 0.00015
user.autonomy_days = 2.0

# ===== DESIGN THAT GAVE 100% EFFICIENCY =====
design = {
    'volume_m3': 800,
    'head_m': 44.4,
    'pipe_diameter_m': 0.339,
    'pump_power_kw': 4.5,
    'turbine_power_kw': 11.1
}

print("=" * 70)
print("DEBUG: SINGLE SIMULATION TEST")
print("=" * 70)
print(f"Design: Volume={design['volume_m3']} m3, Head={design['head_m']} m")
print(f"Pump={design['pump_power_kw']} kW, Turbine={design['turbine_power_kw']} kW")
print("=" * 70)

# ===== GET DATA =====
print("\nFetching solar and load data...")
solar_data = fetch_solar_data(user)
load_data = fetch_load_data(user)

# ===== RUN SIMULATOR =====
print("Running simulation...")
sim = PumpedHydroSimulator(user, design)
results = sim.simulate(solar_data, load_data)
metrics = results['metrics']

# ===== PRINT RESULTS =====
print("\n" + "=" * 70)
print("SIMULATION RESULTS")
print("=" * 70)
print(f"Total Pumped:      {metrics['total_pumped_kwh']:.1f} kWh")
print(f"Total Generated:   {metrics['total_generated_kwh']:.1f} kWh")
print(f"Efficiency:        {metrics['efficiency_percent']:.1f}%")
print(f"Unmet Load:        {metrics['total_unmet_kwh']:.1f} kWh")
print(f"Curtailed Energy:  {metrics['total_curtailed_kwh']:.1f} kWh")
print(f"Final Upper:       {metrics.get('upper_reservoir_type', 'N/A')}") # Not in metrics

# Check if tank ever had water
history = results['history']
max_upper = max(history['upper_volume']) if history['upper_volume'] else 0
min_upper = min(history['upper_volume']) if history['upper_volume'] else 0

print(f"\nUpper Reservoir:")
print(f"  Max Volume: {max_upper:.1f} m3")
print(f"  Min Volume: {min_upper:.1f} m3")
print(f"  Max Upper %: {(max_upper / sim.max_upper) * 100:.1f}%")

# ===== CHECK IF EFFICIENCY IS CORRECT =====
manual_efficiency = (metrics['total_generated_kwh'] / metrics['total_pumped_kwh']) * 100 if metrics['total_pumped_kwh'] > 0 else 0
print(f"\nManual Efficiency: {manual_efficiency:.1f}%")
print(f"Reported Efficiency: {metrics['efficiency_percent']:.1f}%")

if manual_efficiency == metrics['efficiency_percent']:
    print("Efficiency calculation is consistent.")
else:
    print("Efficiency calculation is INCONSISTENT!")

# ===== CHECK FOR 100% EFFICIENCY =====
if metrics['efficiency_percent'] > 95:
    print("\nWARNING: Efficiency > 95% is physically impossible!")
    print("The simulator is NOT applying losses correctly.")
    print("Possible causes:")
    print("  1. Pump efficiency is not being applied")
    print("  2. Turbine efficiency is not being applied")
    print("  3. Friction losses are being ignored")
    print("  4. total_pumped and total_generated are being recorded incorrectly")
else:
    print("\nEfficiency is realistic (< 85%).")