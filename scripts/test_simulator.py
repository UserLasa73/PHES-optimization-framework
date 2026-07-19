"""Small reproducible simulator smoke test.

Run from the repository root with:
    python -m scripts.test_simulator
"""

from src.simulator import PumpedHydroSimulator
from src.user_inputs import UserInputs

user = UserInputs()
user.evaporation_rate_mm_month = 0.0

design = {
    "volume_m3": 200.0,
    "head_m": 20.0,
    "pipe_diameter_m": 0.30,
    "pump_power_kw": 10.0,
    "turbine_power_kw": 10.0,
}

solar = [20.0, 0.0]
load = [0.0, 20.0]
metrics = PumpedHydroSimulator(user, design).simulate(solar, load)["metrics"]
for key in [
    "efficiency_percent",
    "realized_efficiency_percent",
    "total_pumped_kwh",
    "total_generated_kwh",
    "total_unmet_kwh",
    "water_balance_residual_m3",
    "is_physically_valid",
]:
    print(f"{key}: {metrics[key]}")
