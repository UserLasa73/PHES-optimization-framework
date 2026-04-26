from .config import PHYSICS_PARAMS

def simulate_one_hour(design):
    """
    Simulates 1 hour of PHES operation using explicit reservoir volumes.
    """
    # 1. Calculate Evaporation Loss from the upper reservoir
    # Evap depends on surface area (m3/h)
    evap_loss = (design['v_upper'] * PHYSICS_PARAMS['evap_rate_mm_day']) / 1000 / 24
    
    # Apply evaporation (cannot go below dead storage)
    design['v_upper'] = max(design['v_dead'], design['v_upper'] - evap_loss)
    # The lower reservoir gains the evaporated water implicitly 
    # (or we consider it lost from the system - let's assume it's lost from design['v_upper'])

    # 2. Net Power calculation

    net_power = (design['solar_input']*1000) - (design['load_demand']*1000) #in watts
    
    status = "Idle"
    
    # 3. Pumping Mode (Solar excess)
    if net_power > 0:
        if design['v_lower'] > design['v_dead'] and design['v_upper'] < design['v_max']:
            # Max flow constrained by pump power and available space in upper reservoir
            max_flow_power_per_second= net_power / (PHYSICS_PARAMS['rho'] * PHYSICS_PARAMS['g'] * design['h_gross'])
            max_flow_power_per_hour=max_flow_power_per_second*3600
            max_flow_space = design['v_max'] - design['v_upper']
            flow = min(max_flow_power_per_hour, max_flow_space, design['v_lower'] - design['v_dead'])
            
            design['v_upper'] += flow
            design['v_lower'] -= flow
            status = "Pumping"
            
    # 4. Generation Mode (Solar deficit)
    elif net_power < 0:
        if design['v_upper'] > design['v_dead']:
            # Max flow constrained by turbine power and available water in upper reservoir
            max_flow_power_per_second = abs(net_power) / (PHYSICS_PARAMS['rho'] * PHYSICS_PARAMS['g'] * design['h_gross'])
            max_flow_power_per_hour=max_flow_power_per_second*3600
            max_flow_water = design['v_upper'] - design['v_dead']
            flow = min(max_flow_power_per_hour, max_flow_water)
            
            design['v_upper'] -= flow
            design['v_lower'] += flow
            status = "Generating"
            
    return design['v_upper'], design['v_lower'], status