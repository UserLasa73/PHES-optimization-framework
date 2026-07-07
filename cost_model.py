"""
cost_model.py
Cost estimation for PHES system only (PV excluded - user already has panels).
"""

def calculate_capital_cost(volume_m3, head_m, pipe_diameter_m, 
                           pump_power_kw, turbine_power_kw, pv_kwp,
                           upper_type, lower_type):
    """
    Calculate total capital cost for PHES system only.
    PV cost is EXCLUDED (user already has solar panels).
    """
    
    # ===== RESERVOIR COST =====
    upper_volume = volume_m3 * 0.5
    lower_volume = volume_m3 * 0.5
    
    if volume_m3 <= 100:
        cost_per_m3 = 8000
    elif volume_m3 <= 500:
        cost_per_m3 = 5000
    elif volume_m3 <= 2000:
        cost_per_m3 = 3500
    else:
        cost_per_m3 = 2500
    
    cost_factors = {
        "new_tank": 1.0,
        "excavated": 0.5,
        "pond": 0.3,
        "river": 0.2
    }
    
    upper_factor = cost_factors.get(upper_type, 1.0)
    lower_factor = cost_factors.get(lower_type, 1.0)
    
    reservoir_cost = (upper_volume * cost_per_m3 * upper_factor) + \
                     (lower_volume * cost_per_m3 * lower_factor)
    
    # ===== PUMP COST =====
    pump_cost = pump_power_kw * 8000
    
    # ===== TURBINE COST =====
    turbine_cost = turbine_power_kw * 15000
    
    # ===== PIPE COST =====
    pipe_length = head_m * 2.5
    pipe_cost = pipe_length * 2 * 1500
    
    # ===== CONTROLS (BOS) =====
    bos_cost = (pump_cost + turbine_cost + pipe_cost) * 0.20
    
    # ===== INSTALLATION =====
    equipment = reservoir_cost + pump_cost + turbine_cost + pipe_cost + bos_cost
    installation = equipment * 0.20
    
    # ===== TOTAL =====
    total = equipment + installation
    
    return {
        'total_lkr': total,
        'total_usd': total / 300.0,
        'breakdown': {
            'reservoir_lkr': reservoir_cost,
            'pump_lkr': pump_cost,
            'turbine_lkr': turbine_cost,
            'pipe_lkr': pipe_cost,
            'bos_lkr': bos_cost,
            'installation_lkr': installation
        },
        'upper_volume_m3': upper_volume,
        'lower_volume_m3': lower_volume,
        'total_volume_m3': volume_m3
    }