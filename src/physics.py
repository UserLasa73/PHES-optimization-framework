from .config import PHYSICS_PARAMS
import math

def simulate_one_hour(design,v_upper, v_lower, solar_input, load_demand):
    """
    Simulates 1 hour of PHES operation using explicit reservoir volumes.
    """
    # 1. Calculate Evaporation Loss from the upper reservoir
    # Evap depends on surface area (m3/h)
    evap_loss = (v_upper * PHYSICS_PARAMS['evap_rate_mm_day']) / 1000 / 24
    
    # Apply evaporation (cannot go below dead storage)
    v_upper = max(design['v_dead'], v_upper - evap_loss)
    # The lower reservoir gains the evaporated water implicitly 
    # (or we consider it lost from the system - let's assume it's lost from v_upper)

    #2.Apply Seepage Loss (Before anything else) The water that leaks out is gone from the upper reservoir
    seepage = v_upper * PHYSICS_PARAMS['seepage_rate_per_hour']
    v_upper -= seepage


    #3. Net Power calculation
    net_power = (solar_input) - (load_demand) #in Kw

    # Calculate Cross-sectional Area
    area = math.pi * (PHYSICS_PARAMS['d_pipe'] / 2)**2
    
    energy_pumped = 0.0
    energy_generated = 0.0

    status = "Idle"
    
    # 4. Pumping Mode (Solar excess)
    if net_power > PHYSICS_PARAMS['min_pump_threshold_kw']:
        if v_lower > design['v_dead'] and v_upper < design['v_max']:
            # Max flow constrained by pump power and available space in upper reservoir
            max_flow_power_per_second= net_power / (PHYSICS_PARAMS['rho'] * PHYSICS_PARAMS['g'] * design['h_gross'])
            max_flow_power_per_hour=max_flow_power_per_second*3600
            max_flow_space = design['v_max'] - v_upper
            flow = min(max_flow_power_per_hour, max_flow_space, v_lower - design['v_dead'])
            
            # Friction Calculation
            velocity = (flow / 3600) / area
            h_f = PHYSICS_PARAMS['pipe_friction_coeff'] * (PHYSICS_PARAMS['pipe_length'] / PHYSICS_PARAMS['d_pipe']) * (velocity**2 / (2 * PHYSICS_PARAMS['g']))
            
            # Apply Friction to Pumping Cost
            effective_h = design['h_gross'] + h_f
            energy_pumped = (PHYSICS_PARAMS['rho'] * PHYSICS_PARAMS['g'] * flow * effective_h) / (1000 * PHYSICS_PARAMS['pump_efficiency'])


            v_upper += flow
            v_lower -= flow
            status = "Pumping"
            
    # 5. Generation Mode (Solar deficit)
    elif net_power < -PHYSICS_PARAMS['min_gen_threshold_kw']: #net deficit is a negative number when demand is higher than solar load
        if v_upper > design['v_dead']:
            # Max flow constrained by turbine power and available water in upper reservoir
            max_flow_power_per_second = abs(net_power) / (PHYSICS_PARAMS['rho'] * PHYSICS_PARAMS['g'] * design['h_gross'])
            flow = min(max_flow_power_per_second * 3600, v_upper - design['v_dead'])
            
            # Friction Calculation
            velocity = (flow / 3600) / area
            h_f = PHYSICS_PARAMS['pipe_friction_coeff'] * (PHYSICS_PARAMS['pipe_length'] / PHYSICS_PARAMS['d_pipe']) * (velocity**2 / (2 * PHYSICS_PARAMS['g']))
            
            # Apply Friction to Generation Gain
            effective_h = max(0, design['h_gross'] - h_f) # Don't allow negative head
            # Energy gained = (rho*g*Q*H_eff) * efficiency
            energy_generated = (PHYSICS_PARAMS['rho'] * PHYSICS_PARAMS['g'] * flow * effective_h * PHYSICS_PARAMS['turbine_efficiency']) / 1000

            v_upper -= flow
            v_lower += flow
            status = "Generating"
    
            
    return v_upper, v_lower, status, energy_pumped, energy_generated