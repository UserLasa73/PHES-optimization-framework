# src/config.py

# This makes it easy to edit parameters in one place without touching the physics code
PHYSICS_PARAMS = {
    "rho": 1000.0,          # Water density (kg/m3)
    "g": 9.81,              # Gravity (m/s2)
    "pipe_friction_coeff": 0.02,  # Darcy friction factor
    "pipe_length": 100.0,         # meters
    "d_pipe": 0.3,                # diameter
    "pump_efficiency": 0.75,      # 75% efficient
    "turbine_efficiency": 0.85,   # 85% efficient
    "pipe_roughness": 0.0015, # mm (PVC)
    "evap_rate_mm_day": 5.0,  # mm/day
    "seepage_rate_per_hour": 0.0001, # e.g., 0.01% loss per hour
    "min_pump_threshold_kw": 5.0,
    "min_gen_threshold_kw": 5.0,
}

SYSTEM_CONSTANTS = {
    "total_water_volume": 5000.0, # The fixed total volume of the system (m3)
}

# The initial "Design" the optimizer might test
DEFAULT_DESIGN = {
    "v_max": 5000.0,
    "v_dead": 250.0,
    "h_gross": 20.0,
    "d_pipe": 0.3,
    "surface_area": 500.0
}