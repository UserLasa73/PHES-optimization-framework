"""
cost_model.py
Cost estimation for small-scale PHES systems.
Based on 2025 Sri Lankan market prices.
"""

def calculate_capital_cost(volume_m3, head_m, pipe_diameter_m, 
                           pump_power_kw, turbine_power_kw, pv_kwp,
                           upper_type, lower_type):
    """
    Calculate total capital cost for small-scale PHES.
    
    Args:
        volume_m3: Total water volume (m³)
        head_m: Head height (m)
        pipe_diameter_m: Pipe diameter (m)
        pump_power_kw: Pump power (kW)
        turbine_power_kw: Turbine power (kW)
        pv_kwp: PV capacity (kWp)
        upper_type: Upper reservoir type
        lower_type: Lower reservoir type
    
    Returns:
        dict: Cost breakdown and total
    """
    
    # ===== RESERVOIR COST =====
    # Small systems cost more per m³
    if volume_m3 <= 100:
        cost_per_m3 = 8000
    elif volume_m3 <= 500:
        cost_per_m3 = 5000
    elif volume_m3 <= 2000:
        cost_per_m3 = 3500
    else:
        cost_per_m3 = 2500
    
    # Cost factors by type
    cost_factors = {
        "new_tank": 1.0,
        "excavated": 0.5,
        "pond": 0.3,
        "river": 0.2
    }
    
    upper_factor = cost_factors.get(upper_type, 1.0)
    lower_factor = cost_factors.get(lower_type, 1.0)
    
    reservoir_cost = (volume_m3 * cost_per_m3 * upper_factor) + \
                     (volume_m3 * cost_per_m3 * lower_factor)
    
    # ===== PUMP COST =====
    if pump_power_kw <= 10:
        pump_cost_per_kw = 8000
    else:
        pump_cost_per_kw = 6000
    pump_cost = pump_power_kw * pump_cost_per_kw
    
    # ===== TURBINE COST =====
    if turbine_power_kw <= 10:
        turbine_cost_per_kw = 15000
    else:
        turbine_cost_per_kw = 12000
    turbine_cost = turbine_power_kw * turbine_cost_per_kw
    
    # ===== PIPE COST =====
    if pipe_diameter_m <= 0.15:
        pipe_cost_per_m = 800
    else:
        pipe_cost_per_m = 1500
    
    pipe_length = head_m * 2.5  # Estimate
    pipe_cost = pipe_length * 2 * pipe_cost_per_m  # Two pipes
    
    # ===== SOLAR PV COST =====
    pv_cost = pv_kwp * 90000
    
    # ===== BALANCE OF SYSTEM =====
    bos_cost = (pump_cost + turbine_cost + pipe_cost) * 0.20
    
    # ===== INSTALLATION =====
    equipment = reservoir_cost + pump_cost + turbine_cost + pipe_cost + pv_cost + bos_cost
    installation = equipment * 0.20
    
    # ===== TOTAL =====
    total = equipment + installation
    
    return {
        'total_lkr': total,
        'total_usd': total / 300,  # Approximate exchange rate
        'breakdown': {
            'reservoir_lkr': reservoir_cost,
            'pump_lkr': pump_cost,
            'turbine_lkr': turbine_cost,
            'pipe_lkr': pipe_cost,
            'pv_lkr': pv_cost,
            'bos_lkr': bos_cost,
            'installation_lkr': installation
        }
    }


def get_seepage_loss(volume_m3, reservoir_type):
    """
    Calculate seepage loss based on reservoir type.
    """
    seepage_factors = {
        "new_tank": 0.00,
        "excavated": 0.05,
        "pond": 0.10,
        "river": 0.20
    }
    factor = seepage_factors.get(reservoir_type, 0.00)
    return volume_m3 * factor


def get_reservoir_cost_factor(reservoir_type):
    """
    Get cost multiplier for reservoir type.
    """
    cost_factors = {
        "new_tank": 1.0,
        "excavated": 0.5,
        "pond": 0.3,
        "river": 0.2
    }
    return cost_factors.get(reservoir_type, 1.0)