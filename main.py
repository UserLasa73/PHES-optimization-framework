import csv
from src.physics import simulate_one_hour
from src.config import DEFAULT_DESIGN

# 1. State variables (Initial conditions)
v_upper = 1000.0
v_lower = 4000.0

total_pumped = 0.0
total_generated = 0.0

#run for 1 year solar data
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


efficiency = (total_generated / total_pumped) * 100

print(f"--- 8760-Hour Simulation Result ---")
#print(f"Status: {status}, Upper: {v_upper:.2f}, Lower: {v_lower:.2f}")
print(f"pumped Energy: {total_pumped:.2f}Kwh")
print(f"generated Energy: {total_generated:.2f}Kwh")
print(f"Total Round Trip Efficiency: {efficiency:.2f}%")
