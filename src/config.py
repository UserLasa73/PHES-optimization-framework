# src/config.py

# This makes it easy to edit parameters in one place without touching the physics code
PHYSICS_PARAMS = {
    "rho": 1000.0,          # Water density (kg/m3)
    "g": 9.81,              # Gravity (m/s2)
    "pipe_roughness": 0.0015, # mm (PVC)
    "evap_rate_mm_day": 5.0,  # mm/day
}

# The initial "Design" the optimizer might test
DEFAULT_DESIGN = {
    "v_max": 5000.0,
    "v_dead": 250.0,
    "h_gross": 20.0,
    "d_pipe": 0.3,
    "p_pump": 20.0,
    "p_turb": 20.0,
    "surface_area": 500.0
}