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



# ============================================================================
# PIPE FRICTION CALCULATIONS
# ============================================================================

def calculate_darcy_weisbach_loss(flow_rate_m3s: float, 
                                   diameter_m: float, 
                                   length_m: float,
                                   roughness_m: float = 0.00015) -> float:
    """
    Calculate head loss due to pipe friction using Darcy-Weisbach equation.
    
    Formula: h_f = f × (L/D) × (V²/2g)
    
    Where:
        h_f = Head loss (m)
        f = Friction factor (dimensionless)
        L = Pipe length (m)
        D = Pipe diameter (m)
        V = Flow velocity (m/s)
        g = Gravity (m/s²)
    
    Args:
        flow_rate_m3s: Flow rate in m³/s
        diameter_m: Pipe internal diameter in meters
        length_m: Pipe length in meters
        roughness_m: Pipe wall roughness in meters (default: 0.00015 for steel)
    
    Returns:
        Head loss in meters
    """
    if flow_rate_m3s <= 0 or diameter_m <= 0 or length_m <= 0:
        return 0.0
    
    # Cross-sectional area: A = π × (D/2)²
    area = math.pi * (diameter_m / 2.0) ** 2.0
    
    # Flow velocity: V = Q / A
    velocity = flow_rate_m3s / area
    
    # Reynolds number: Re = (V × D) / ν
    reynolds = velocity * diameter_m / KINEMATIC_VISCOSITY
    
    # Calculate friction factor
    if reynolds < 2000:
        # Laminar flow: f = 64/Re
        friction_factor = 64.0 / reynolds
    elif reynolds < 4000:
        # Transition zone - use conservative estimate
        friction_factor = 0.03
    else:
        # Turbulent flow - Swamee-Jain approximation
        # f = 0.25 / [log10(ε/(3.7D) + 5.74/Re^0.9)]²
        try:
            friction_factor = 0.25 / (
                math.log10(roughness_m / (3.7 * diameter_m) + 5.74 / (reynolds ** 0.9))
            ) ** 2.0
        except (ValueError, ZeroDivisionError):
            # Fallback for numerical issues
            friction_factor = 0.02
    
    # Darcy-Weisbach: h_f = f × (L/D) × (V²/2g)
    head_loss = friction_factor * (length_m / diameter_m) * (velocity ** 2.0) / (2.0 * GRAVITY)
    
    return max(0.0, head_loss)
