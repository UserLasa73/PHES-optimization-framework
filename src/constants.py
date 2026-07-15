# constants.py
# All physical constants used in the simulation

# Physical constants
WATER_DENSITY = 1000          # kg/m³ (density of water)
GRAVITY = 9.81                # m/s² (gravity)
KINEMATIC_VISCOSITY = 1e-6    # m²/s (water at 20°C)
SECONDS_PER_HOUR = 3600       # seconds in 1 hour

# Efficiency values (typical for PHES systems)
PUMP_EFFICIENCY = 0.85        # 85% pump efficiency
TURBINE_EFFICIENCY = 0.90     # 90% turbine efficiency  
MOTOR_EFFICIENCY = 0.95       # 95% motor efficiency

# Pipe parameters
PIPE_ROUGHNESS = 0.00015      # m (steel pipe roughness)

# Loss parameters
MONTHLY_EVAPORATION = 0.03    # 3% evaporation per month

# Minimum water level (can't go below this)
MIN_WATER_PERCENT = 0.10      # 10%
MAX_WATER_PERCENT = 0.95      # 95% (leave some space)

# SEEPAGE LOSSES (simple)
SEEPAGE_LOSS = {
    "new_tank": 0.0,      # 0% per month
    "excavated": 0.05,    # 5% per month
    "pond": 0.10,         # 10% per month
    "river": 0.20         # 20% per month
}