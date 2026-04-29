import csv
from src.physics import simulate_one_hour
from src.config import DEFAULT_DESIGN

# 1. State variables (Initial conditions)
v_upper = 1000.0
v_lower = 4000.0

# Store these to calculate the energy balance at the end of year
initial_v_upper = v_upper

total_pumped = 0.0
total_generated = 0.0

# 2. run for 1 year solar data
with open('data/solar_data.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Convert strings from CSV to floats
        solar = float(row['Solar_Input'])
        load = float(row['Load_Demand'])
        
        # Run simulation
        v_upper, v_lower, status, pumpedEnergy, generatedEnergy = simulate_one_hour(
            DEFAULT_DESIGN, v_upper, v_lower, solar, load
        )

        print(f"Status: {status}, Upper: {v_upper:.2f}, Lower: {v_lower:.2f}, pumped Energy: {pumpedEnergy:.2f}Kwh, generated Energy: {generatedEnergy:.2f}Kwh")

        total_pumped += pumpedEnergy
        total_generated += generatedEnergy

# 3. Find Energy Balance after the loop finishes
def calculate_potential(vol, h_gross):
    # Energy in kWh = (Volume * rho * g * H) / (3600 * 1000)
    return (vol * 1000 * 9.81 * h_gross) / 3600000

final_energy = calculate_potential(v_upper, DEFAULT_DESIGN['h_gross'])
initial_energy = calculate_potential(initial_v_upper, DEFAULT_DESIGN['h_gross'])

#Net energy balance closer to Zero is the better. (No wasted energy)
energy_balance = final_energy - initial_energy

efficiency = (total_generated / total_pumped) * 100

print(f"--- 8760-Hour Simulation Result ---")
print(f"pumped Energy: {total_pumped:.2f}Kwh")
print(f"generated Energy: {total_generated:.2f}Kwh")
print(f"Total Round Trip Efficiency: {efficiency:.2f}%")
print(f"Net Energy Balance (Final - Initial): {energy_balance:.2f} kWh")
