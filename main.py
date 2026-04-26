# main.py
from src.physics import simulate_one_hour
from src.config import DEFAULT_DESIGN

total_pumped = 0.0
total_generated = 0.0

#for 1 year
for t in range(8760):
    #for 1 hour
    v_upper, v_lower, status, pumpedEnergy, generatedEnergy = simulate_one_hour(DEFAULT_DESIGN)
    
    total_pumped += pumpedEnergy
    total_generated += generatedEnergy

efficiency = (total_generated / total_pumped) * 100

print(f"--- 1-Hour Simulation Result ---")
print(f"Status: {status}, Upper: {v_upper:.2f}, Lower: {v_lower:.2f}")
print(f"pumped Energy: {pumpedEnergy:.2f}Kwh")
print(f"generated Energy: {generatedEnergy:.2f}Kwh")
print(f"Total Round Trip Efficiency: {efficiency:.2f}%")
