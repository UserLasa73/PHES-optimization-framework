"""
physics.py
Core physics calculations for Pumped Hydro Energy Storage system.
All formulas used by the simulator.
"""

import math
from src.constants import *

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


def calculate_total_head_loss(flow_rate_m3s: float,
                               pipe_diameter_m: float,
                               head_m: float,
                               roughness_m: float = 0.00015) -> float:
    """
    Calculate total head loss including major (friction) and minor losses.
    
    Minor losses (bends, valves, etc.) are estimated as 10% of major loss.
    
    Args:
        flow_rate_m3s: Flow rate in m³/s
        pipe_diameter_m: Pipe diameter in meters
        head_m: Gross head in meters
        roughness_m: Pipe roughness in meters
    
    Returns:
        Total head loss in meters
    """
    if flow_rate_m3s <= 0:
        return 0.0
    
    # Estimate pipe length (3x head for sloping terrain)
    pipe_length = head_m * 3.0
    
    # Major losses (friction)
    major_loss = calculate_darcy_weisbach_loss(
        flow_rate_m3s, 
        pipe_diameter_m, 
        pipe_length, 
        roughness_m
    )
    
    # Minor losses (10% of major loss for bends, valves, etc.)
    minor_loss = major_loss * 0.10
    
    return major_loss + minor_loss


# ============================================================================
# RESERVOIR CALCULATIONS
# ============================================================================

def estimate_reservoir_surface_area(volume_m3: float, depth_m: float = 3.0) -> float:
    """
    Estimate reservoir surface area from volume and depth.
    
    Assumes cylindrical shape: Volume = Area × Depth
    Area = Volume / Depth
    
    Args:
        volume_m3: Reservoir volume in m³
        depth_m: Reservoir depth in meters (default: 3m)
    
    Returns:
        Surface area in m²
    """
    if volume_m3 <= 0 or depth_m <= 0:
        return 0.0
    
    return volume_m3 / depth_m


def calculate_evaporation_loss(volume_m3: float,
                                surface_area_m2: float,
                                evaporation_rate_mm_month: float,
                                hours: int) -> float:
    """
    Calculate water loss due to evaporation.
    
    Args:
        volume_m3: Current water volume in m³
        surface_area_m2: Reservoir surface area in m²
        evaporation_rate_mm_month: Evaporation rate in mm/month
        hours: Number of hours in the period
    
    Returns:
        Water volume lost in m³
    """
    if volume_m3 <= 0 or surface_area_m2 <= 0:
        return 0.0
    
    # Convert mm/month to m/hour
    days = hours / 24.0
    months = days / 30.44  # Average month length
    
    evap_m_per_month = evaporation_rate_mm_month / 1000.0
    evap_m_per_hour = evap_m_per_month / (30.44 * 24.0)
    
    # Volume lost = surface_area × evaporation_depth
    volume_lost = surface_area_m2 * evap_m_per_hour * hours
    
    # Can't lose more than 50% of available water
    return min(volume_lost, volume_m3 * 0.50)


# ============================================================================
# ENERGY CALCULATIONS
# ============================================================================

def calculate_stored_energy(water_volume_m3: float, head_m: float) -> float:
    """
    Calculate potential energy stored in upper reservoir.
    
    Formula: E = ρ × g × H × V × η_turbine
    
    Args:
        water_volume_m3: Water volume in upper reservoir (m³)
        head_m: Head height (m)
    
    Returns:
        Stored energy in kWh
    """
    if water_volume_m3 <= 0 or head_m <= 0:
        return 0.0
    
    energy_joules = WATER_DENSITY * GRAVITY * head_m * water_volume_m3 * TURBINE_EFFICIENCY
    energy_kwh = energy_joules / 1000.0 / 3600.0  # Joules → kWh
    
    return energy_kwh


def calculate_round_trip_efficiency(pumped_energy_kwh: float, 
                                     generated_energy_kwh: float) -> float:
    """
    Calculate round-trip efficiency.
    
    Formula: η_rt = (E_generated / E_pumped) × 100
    
    Args:
        pumped_energy_kwh: Total energy used for pumping (kWh)
        generated_energy_kwh: Total energy generated (kWh)
    
    Returns:
        Efficiency percentage (0-100)
    """
    if pumped_energy_kwh <= 0:
        return 0.0
    
    efficiency = (generated_energy_kwh / pumped_energy_kwh) * 100.0
    
    return min(100.0, max(0.0, efficiency))

# ============================================================================
# SEEPAGE CALCULATIONS
# ============================================================================

def get_seepage_loss_factor(reservoir_type: str) -> float:
    """
    Get seepage loss factor for reservoir type.
    
    Args:
        reservoir_type: 'new_tank', 'excavated', 'pond', or 'river'
    
    Returns:
        Monthly seepage loss as fraction (0.0 to 1.0)
    """
    seepage_factors = {
        "new_tank": 0.00,      # 0% per month
        "excavated": 0.05,     # 5% per month
        "pond": 0.10,          # 10% per month
        "river": 0.20          # 20% per month
    }
    
    return seepage_factors.get(reservoir_type, 0.00)


def apply_seepage_loss(volume_m3: float, reservoir_type: str) -> float:
    """
    Apply monthly seepage loss to reservoir volume.
    
    Args:
        volume_m3: Current water volume
        reservoir_type: Type of reservoir
    
    Returns:
        Water volume after seepage loss
    """
    factor = get_seepage_loss_factor(reservoir_type)
    loss = volume_m3 * factor
    return max(0.0, volume_m3 - loss)

# ============================================================================
# COST CALCULATIONS
# ============================================================================

def get_reservoir_cost_factor(reservoir_type: str) -> float:
    """
    Get cost factor for reservoir type.
    
    Args:
        reservoir_type: 'new_tank', 'excavated', 'pond', or 'river'
    
    Returns:
        Cost multiplier (1.0 = base cost)
    """
    cost_factors = {
        "new_tank": 1.0,       # Full cost (concrete tank)
        "excavated": 0.6,      # 60% of full cost
        "pond": 0.3,           # 30% of full cost
        "river": 0.1           # 10% of full cost (natural river)
    }
    
    return cost_factors.get(reservoir_type, 1.0)


def calculate_water_velocity(flow_rate_m3s: float, pipe_diameter_m: float) -> float:
    """
    Calculate water velocity in pipe.
    
    Formula: V = Q / A = Q / (π × (D/2)²)
    
    Args:
        flow_rate_m3s: Flow rate in m³/s
        pipe_diameter_m: Pipe diameter in meters
    
    Returns:
        Velocity in m/s
    """
    if flow_rate_m3s <= 0 or pipe_diameter_m <= 0:
        return 0.0
    
    area = math.pi * (pipe_diameter_m / 2.0) ** 2.0
    velocity = flow_rate_m3s / area
    
    return velocity


def check_velocity_limit(flow_rate_m3s: float, pipe_diameter_m: float, 
                          max_velocity: float = 5.0) -> bool:
    """
    Check if water velocity exceeds maximum allowable.
    
    Args:
        flow_rate_m3s: Flow rate in m³/s
        pipe_diameter_m: Pipe diameter in meters
        max_velocity: Maximum allowable velocity (m/s) (default: 5.0)
    
    Returns:
        True if velocity is within limit
    """
    velocity = calculate_water_velocity(flow_rate_m3s, pipe_diameter_m)
    return velocity <= max_velocity

# ============================================================================
# AUTONOMY CALCULATIONS
# ============================================================================

def calculate_autonomy_days(stored_energy_kwh: float, 
                             daily_load_kwh: float) -> float:
    """
    Calculate autonomy in days.
    
    Args:
        stored_energy_kwh: Energy stored in upper reservoir (kWh)
        daily_load_kwh: Average daily load (kWh/day)
    
    Returns:
        Autonomy in days
    """
    if daily_load_kwh <= 0:
        return 0.0
    
    return stored_energy_kwh / daily_load_kwh

