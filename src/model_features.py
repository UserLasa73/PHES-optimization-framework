"""Shared surrogate feature schema and supported training domain."""

SURROGATE_FEATURES = [
    "volume_m3",
    "head_m",
    "pipe_diameter_m",
    "pump_power_kw",
    "turbine_power_kw",
    "pv_kwp",
    "daily_energy_kwh",
    "evaporation_rate_mm_month",
    "reservoir_type_code",
    "latitude",
    "longitude",
    "annual_solar_yield_kwh_per_kwp",
]

# Initial validated domain for the fixed model. The Streamlit ML mode should not
# extrapolate beyond these ranges unless a new dataset/model is trained.
TRAINING_BOUNDS = {
    "volume_m3": (20.0, 800.0),
    "head_m": (5.0, 45.0),
    "pipe_diameter_m": (0.05, 0.35),
    "pump_power_kw": (2.0, 30.0),
    "turbine_power_kw": (2.0, 25.0),
    "pv_kwp": (5.0, 30.0),
    "daily_energy_kwh": (10.0, 50.0),
    "evaporation_rate_mm_month": (30.0, 80.0),
    "reservoir_type_code": (0, 3),
}

RESERVOIR_TYPE_TO_CODE = {
    "new_tank": 0,
    "excavated": 1,
    "pond": 2,
    "river": 3,
}
CODE_TO_RESERVOIR_TYPE = {value: key for key, value in RESERVOIR_TYPE_TO_CODE.items()}
