# physics.py
# All physics calculations for the pumped hydro system

import math
import numpy as np
from constants import *

def pump_flow_rate(pump_power_kw, head_m, efficiency=PUMP_EFFICIENCY):
    """
    Calculate water flow rate from pump power
    Inputs:
        pump_power_kw: pump power in kilowatts
        head_m: head height in meters
        efficiency: pump efficiency (default 0.85)
    Returns:
        flow_rate: cubic meters per second
    """
    power_w = pump_power_kw * 1000  # Convert kW to Watts
    flow_rate = (power_w * efficiency) / (WATER_DENSITY * GRAVITY * head_m)
    return flow_rate

def turbine_power(flow_rate_m3s, head_m, efficiency=TURBINE_EFFICIENCY):
    """
    Calculate electrical power from water flow
    Inputs:
        flow_rate_m3s: water flow in cubic meters per second
        head_m: head height in meters
        efficiency: turbine efficiency (default 0.90)
    Returns:
        power_kw: electrical power in kilowatts
    """
    power_w = WATER_DENSITY * GRAVITY * head_m * flow_rate_m3s * efficiency
    power_kw = power_w / 1000  # Convert Watts to kW
    return power_kw

def darcy_weisbach_loss(flow_rate_m3s, diameter_m, length_m, roughness_m=PIPE_ROUGHNESS):
    """
    Calculate head loss due to pipe friction
    Inputs:
        flow_rate_m3s: water flow in cubic meters per second
        diameter_m: pipe diameter in meters
        length_m: pipe length in meters
        roughness_m: pipe roughness in meters
    Returns:
        head_loss: head loss in meters
    """
    # Calculate pipe cross-sectional area
    area = math.pi * (diameter_m / 2) ** 2
    
    # Calculate water velocity
    if area <= 0:
        return 0
    velocity = flow_rate_m3s / area
    
    # Calculate Reynolds number
    Re = velocity * diameter_m / KINEMATIC_VISCOSITY
    
    # If flow is very slow, no significant losses
    if Re < 2000:
        return 0
    
    # Swamee-Jain approximation for friction factor
    # (simpler than Colebrook-White equation)
    f = 0.25 / (math.log10(roughness_m / (3.7 * diameter_m) + 5.74 / (Re ** 0.9))) ** 2
    
    # Darcy-Weisbach equation for head loss
    head_loss = f * (length_m / diameter_m) * (velocity ** 2) / (2 * GRAVITY)
    
    return head_loss

def calculate_capex(volume_m3, head_m, pipe_diameter_m, pump_power_kw, turbine_power_kw, pv_kwp):
    """
    Calculate estimated capital cost in LKR
    Simple parametric cost model
    """
    # Reservoir cost: LKR 1000 per cubic meter
    reservoir_cost = volume_m3 * 1000
    
    # Pump cost: LKR 1000 per kW
    pump_cost = pump_power_kw * 1000
    
    # Turbine cost: LKR 1000 per kW
    turbine_cost = turbine_power_kw * 1000
    
    # Pipe cost: LKR 500 per meter (2x for up and down)
    pipe_length = head_m * 3  # Estimate pipe length as 3x head
    pipe_cost = pipe_length * 2 * 500  # 2 pipes (up and down)
    
    # PV cost: LKR 100,000 per kWp
    pv_cost = pv_kwp * 100000
    
    # Civil works: 30% of equipment cost
    equipment_cost = reservoir_cost + pump_cost + turbine_cost + pipe_cost + pv_cost
    civil_cost = equipment_cost * 0.30
    
    total_cost = equipment_cost + civil_cost
    
    return total_cost