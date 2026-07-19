"""Physical and numerical constants used by the PHES model."""

# Physical constants
WATER_DENSITY = 1000.0          # kg/m³
GRAVITY = 9.81                  # m/s²
KINEMATIC_VISCOSITY = 1.0e-6    # m²/s, water near 20 °C
SECONDS_PER_HOUR = 3600.0
TIME_STEP_HOURS = 1.0

# Component efficiencies (constant-efficiency baseline model)
PUMP_EFFICIENCY = 0.85
TURBINE_EFFICIENCY = 0.90

# Reservoir operating limits, applied to each reservoir's gross capacity
MIN_WATER_FRACTION = 0.05
MAX_WATER_FRACTION = 0.95

# Hydraulic assumptions
DEFAULT_PIPE_ROUGHNESS_M = 0.00015
PIPE_LENGTH_TO_HEAD_RATIO = 3.0
MINOR_LOSS_FRACTION = 0.10
MAX_PIPE_VELOCITY_MPS = 5.0

# Reproducibility
DEFAULT_RANDOM_SEED = 42
