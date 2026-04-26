# main.py
from src.physics import simulate_one_hour
from src.config import DEFAULT_DESIGN

# Simulation state
current_soc = 1000.0  # Starting with 1000m3
solar_input = 30.0    # 30 kW
load_demand = 10.0    # 10 kW
total_vol = 5000.0

# Run 1 hour
new_soc, status = simulate_one_hour(DEFAULT_DESIGN, current_soc, solar_input, load_demand,total_vol)

print(f"--- 1-Hour Simulation Result ---")
print(f"Status: {status}")
print(f"Initial SOC: {current_soc} m3")
print(f"Final SOC:   {new_soc:.2f} m3")