"""
physics.py
Core physics calculations for Pumped Hydro Energy Storage system.
All formulas used by the simulator.
"""

import math
from constants import *

# ============================================================================
# PUMP AND TURBINE CALCULATIONS
# ============================================================================

def calculate_pump_flow_rate(pump_power_kw: float, head_m: float) -> float:
    """
    Calculate water flow rate from pump power.
    
    Formula: Q = (P × η_pump) / (ρ × g × H)
    
    Where:
        Q = Flow rate (m³/s)
        P = Pump power (Watts)
        η_pump = Pump efficiency (85%)
        ρ = Water density (1000 kg/m³)
        g = Gravity (9.81 m/s²)
        H = Head (m)
    
    Args:
        pump_power_kw: Pump power in kilowatts
        head_m: Total head in meters
    
    Returns:
        Flow rate in m³/s
    """
    if head_m <= 0:
        return 0.0
    
    power_watts = pump_power_kw * 1000.0
    flow_rate = (power_watts * PUMP_EFFICIENCY) / (WATER_DENSITY * GRAVITY * head_m)
    
    return max(0.0, flow_rate)

def calculate_turbine_power(flow_rate_m3s: float, head_m: float) -> float:
    """
    Calculate electrical power from water flow.
    
    Formula: P = ρ × g × H × Q × η_turbine
    
    Where:
        P = Power (Watts)
        ρ = Water density (1000 kg/m³)
        g = Gravity (9.81 m/s²)
        H = Head (m)
        Q = Flow rate (m³/s)
        η_turbine = Turbine efficiency (90%)
    
    Args:
        flow_rate_m3s: Water flow in m³/s
        head_m: Total head in meters
    
    Returns:
        Power in kilowatts
    """
    if flow_rate_m3s <= 0 or head_m <= 0:
        return 0.0
    
    power_watts = WATER_DENSITY * GRAVITY * head_m * flow_rate_m3s * TURBINE_EFFICIENCY
    power_kw = power_watts / 1000.0
    
    return max(0.0, power_kw)

def calculate_pump_power_from_flow(flow_rate_m3s: float, head_m: float) -> float:
    """
    Calculate pump power required for a given flow rate.
    
    Formula: P = (ρ × g × H × Q) / η_pump
    
    Args:
        flow_rate_m3s: Flow rate in m³/s
        head_m: Total head in meters
    
    Returns:
        Power in kilowatts
    """
    if flow_rate_m3s <= 0 or head_m <= 0:
        return 0.0
    
    power_watts = (WATER_DENSITY * GRAVITY * head_m * flow_rate_m3s) / PUMP_EFFICIENCY
    power_kw = power_watts / 1000.0
    
    return max(0.0, power_kw)



