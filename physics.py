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