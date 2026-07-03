# physics.py
# All physics calculations for the pumped hydro system

import math
from constants import *

def water_flow_from_pump(power_kw, height_m):
    """How much water can a pump move? (m³ per second)"""
    power_w = power_kw * 1000
    flow = (power_w * PUMP_EFFICIENCY) / (WATER_DENSITY * GRAVITY * height_m)
    return flow

def power_from_water_flow(flow_m3s, height_m):
    """How much electricity can we make? (kW)"""
    power_w = WATER_DENSITY * GRAVITY * height_m * flow_m3s * TURBINE_EFFICIENCY
    return power_w / 1000

def friction_loss(flow_m3s, pipe_m, length_m):
    """How much energy is lost to friction?"""
    if flow_m3s <= 0:
        return 0
    
    area = math.pi * (pipe_m/2) ** 2
    velocity = flow_m3s / area
    
    # Simplified friction factor
    f = 0.02  # Rough estimate
    
    loss = f * (length_m/pipe_m) * (velocity**2) / (2 * GRAVITY)
    return loss