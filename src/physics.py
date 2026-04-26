# The core simulator (equations, hourly flows)# src/physics.py
from .config import PHYSICS_PARAMS

def simulate_one_hour(design, current_soc, solar_input, load_demand,total_vol):
    """
    Simulates 1 hour of PHES operation.
    """
    # 1. Calculate Evaporation Loss (m3/h)
    evap_loss = (design['surface_area'] * PHYSICS_PARAMS['evap_rate_mm_day']) / 1000 / 24
    
    # 2. Update SOC with evaporation
    new_soc = max(design['v_dead'], current_soc - evap_loss)
    
    # 3. Simple Logic: If solar > load, PUMP. If solar < load, GENERATE.
    net_power = solar_input - load_demand

    # 4. Track the lower reservoir
    v_upper = current_soc
    v_lower = total_vol - v_upper
    
    # 5. Safety Check: Does the lower reservoir have enough water to pump?
    if net_power > 0 and v_lower < design['v_dead']:
        status = "Blocked: Lower Reservoir Empty"
        return v_upper, status # Cannot pump
    
    # A. Pumping Mode
    if net_power > 0 and new_soc < design['v_max']:
        flow = min(net_power / (PHYSICS_PARAMS['rho'] * PHYSICS_PARAMS['g'] * design['h_gross']), 
                   design['v_max'] - new_soc)
        new_soc += flow
        status = "Pumping"
        
    # B. Generation Mode
    elif net_power < 0 and new_soc > design['v_dead']:
        flow = min(abs(net_power) / (PHYSICS_PARAMS['rho'] * PHYSICS_PARAMS['g'] * design['h_gross']), 
                   new_soc - design['v_dead'])
        new_soc -= flow
        status = "Generating"
        
    else:
        status = "Idle"
        


    return new_soc, status