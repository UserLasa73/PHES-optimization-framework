import csv
from src.physics import simulate_one_hour
from src.config import DEFAULT_DESIGN, PHYSICS_PARAMS

# 1. State variables (Initial conditions)
v_upper = 1000.0
v_lower = 4000.0

# Store these to calculate the energy balance at the end of year
initial_v_upper = v_upper
initial_v_lower = v_lower

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


def print_technical_summary(design, params, total_pumped, total_gen, balance, upperVol, lowerVol):
    print("\n" + "="*45)
    print("   PHES SYSTEM TECHNICAL SPECIFICATIONS")
    print("="*45)
    print(f"Infrastructure Specs:")
    print(f"  - Gross Head (H):       {design['h_gross']} m")
    print(f"  - Pipe Diameter (D):    {design['d_pipe']} m")
    print(f"  - Reservoir Max Vol:    {design['v_max']} m3")
    print(f"  - Reservoir Surface area:    {design['surface_area']} m2")
    print(f"  - initial Upper Reservoir Vol:    {upperVol:.2f} m3")
    print(f"  - initial Lower Reservoir Vol:    {lowerVol:.2f} m3")
    print("-"*45)
    print(f"Operational Parameters:")
    print(f"  - Pump Efficiency:      {params['pump_efficiency']*100:.1f}%")
    print(f"  - Turbine Efficiency:   {params['turbine_efficiency']*100:.1f}%")
    print(f"  - Pipe Friction (f):    {params['pipe_friction_coeff']}")
    print(f"  - Seepage/Evap Rates:   {params['seepage_rate_per_hour']*100:.3f}% / {params['evap_rate_mm_day']} mm/day")
    print("-"*45)
    print(f"Annual Performance Audit:")
    print(f"  - Total Pumped:         {total_pumped:.2f} kWh")
    print(f"  - Total Generated:      {total_gen:.2f} kWh")
    print(f"  - Round-Trip Efficiency:{(total_gen/total_pumped)*100:.2f}%" if total_pumped > 0 else "0%")
    print(f"  - Net Energy Balance:   {balance:.2f} kWh")
    print("="*45 + "\n")


# Print the report!
print_technical_summary(DEFAULT_DESIGN, PHYSICS_PARAMS, total_pumped, total_generated, energy_balance,initial_v_upper,initial_v_lower)