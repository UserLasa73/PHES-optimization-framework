# src/config.py

# This makes it easy to edit parameters in one place without touching the physics code
PHYSICS_PARAMS = {
    "rho": 1000.0,          # Water density (kg/m3)
    "g": 9.81,              # Gravity (m/s2)
    "pipe_roughness": 0.0015, # mm (PVC)
    "evap_rate_mm_day": 5.0,  # mm/day
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
    "surface_area": 500.0,
    "v_upper":1000.0, 
    "v_lower":4000.0, 
    "solar_input":30.0, #per hour
    "load_demand":20.0
}