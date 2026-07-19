"""Historical PVGIS solar data and deterministic load-profile utilities.

The annual solar profile is read from locally cached PVGIS-ERA5 files for
Vavuniya, Colombo, and Jaffna. The cached timestamps must already be converted
to Asia/Colombo local time and contain exactly 8,760 hourly records for 2023.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


ROOT = Path(__file__).resolve().parents[1]
WEATHER_DIR = ROOT / "data" / "weather"
HISTORICAL_YEAR = 2023

WEATHER_FILES = {
    "vavuniya": WEATHER_DIR / "vavuniya_2023_pvgis.csv",
    "colombo": WEATHER_DIR / "colombo_2023_pvgis.csv",
    "jaffna": WEATHER_DIR / "jaffna_2023_pvgis.csv",
}

REQUIRED_WEATHER_COLUMNS = {
    "poa_direct",
    "poa_sky_diffuse",
    "poa_ground_diffuse",
    "temp_air",
    "wind_speed",
}


def _normalise_location_name(value: str) -> str:
    return str(value).strip().lower()


def load_historical_weather(user) -> pd.DataFrame:
    """Load one cached 2023 PVGIS weather profile for the selected location."""
    location_key = _normalise_location_name(getattr(user, "location", ""))

    if location_key not in WEATHER_FILES:
        supported = ", ".join(name.title() for name in WEATHER_FILES)
        raise ValueError(
            f"Historical weather is available only for: {supported}. "
            f"Received location={getattr(user, 'location', None)!r}."
        )

    path = WEATHER_FILES[location_key]
    if not path.exists():
        raise FileNotFoundError(
            f"Historical weather file not found: {path}. "
            "Run: python -m scripts.download_weather_2023"
        )

    weather = pd.read_csv(path)
    if "timestamp_local" not in weather.columns:
        raise ValueError(
            f"{path.name} must contain a 'timestamp_local' column."
        )

    missing = REQUIRED_WEATHER_COLUMNS.difference(weather.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    weather["timestamp_local"] = pd.to_datetime(
        weather["timestamp_local"], errors="raise"
    )
    weather = weather.set_index("timestamp_local").sort_index()

    if len(weather) != 8760:
        raise ValueError(
            f"{path.name} must contain exactly 8760 rows; received {len(weather)}."
        )

    if not np.all(weather.index.year == HISTORICAL_YEAR):
        raise ValueError(
            f"{path.name} contains timestamps outside local year {HISTORICAL_YEAR}."
        )

    for column in REQUIRED_WEATHER_COLUMNS:
        weather[column] = pd.to_numeric(weather[column], errors="raise")

    return weather


def fetch_solar_data(user, capacity_kwp: float | None = None):
    """Calculate hourly PV output from cached historical PVGIS weather.

    PVGIS supplied the plane-of-array irradiance for a fixed 10-degree,
    south-facing surface. A simple temperature correction and an 85% aggregate
    system derating factor are applied. Output is limited to the installed PV
    capacity and returned as 8,760 hourly kW values.
    """
    pv_kwp = float(user.pv_kwp if capacity_kwp is None else capacity_kwp)
    if pv_kwp <= 0:
        raise ValueError("PV capacity must be positive.")

    weather = load_historical_weather(user)

    poa_global = (
        weather["poa_direct"]
        + weather["poa_sky_diffuse"]
        + weather["poa_ground_diffuse"]
    ).clip(lower=0.0)

    # Estimate module-cell temperature using the Faiman model.
    cell_temperature = pvlib.temperature.faiman(
        poa_global=poa_global,
        temp_air=weather["temp_air"],
        wind_speed=weather["wind_speed"].clip(lower=0.0),
    )

    # Typical crystalline-silicon power temperature coefficient: -0.4%/degree C.
    temperature_factor = 1.0 - 0.004 * (cell_temperature - 25.0)
    temperature_factor = temperature_factor.clip(lower=0.70, upper=1.10)

    aggregate_derating = 0.85
    solar_kw = (
        (poa_global / 1000.0)
        * pv_kwp
        * temperature_factor
        * aggregate_derating
    )

    # Treat installed kWp as the maximum AC-side output for this preliminary model.
    solar_kw = solar_kw.clip(lower=0.0, upper=pv_kwp)
    return solar_kw.to_numpy(dtype=float).tolist()


def annual_solar_yield_per_kwp(user) -> float:
    """Return annual historical PV energy yield in kWh per installed kWp."""
    return float(sum(fetch_solar_data(user, capacity_kwp=1.0)))


def _load_shape_24h() -> np.ndarray:
    """Normalized residential/farm-style daily shape with morning/evening peaks."""
    shape = np.array(
        [
            0.55, 0.50, 0.48, 0.48, 0.52, 0.70,
            1.25, 1.45, 1.20, 0.85, 0.78, 0.76,
            0.80, 0.82, 0.78, 0.80, 0.95, 1.20,
            1.65, 1.85, 1.75, 1.45, 1.05, 0.75,
        ],
        dtype=float,
    )
    return shape / shape.sum()


def generate_load_profile(user):
    """Create a deterministic 8760-hour profile with exact daily energy."""
    daily_energy = float(user.daily_energy_kwh)
    if daily_energy <= 0:
        raise ValueError("Daily energy demand must be positive.")

    daily = _load_shape_24h() * daily_energy
    profile = np.tile(daily, 365)

    spike_factor = float(getattr(user, "demand_spike_factor", 1.0))
    if spike_factor > 1.0:
        rng = np.random.default_rng(int(getattr(user, "random_seed", 42)))
        spike_count = max(1, int(len(profile) * 0.01))
        indices = rng.choice(len(profile), size=spike_count, replace=False)
        profile[indices] *= spike_factor

    return profile.astype(float).tolist()


def fetch_load_data(user):
    """Read an uploaded hourly load CSV or generate the default annual profile."""
    csv_path = getattr(user, "load_csv_path", None)
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Load profile not found: {path}")
        frame = pd.read_csv(path)
        if "load_kw" not in frame.columns:
            raise ValueError("Load CSV must contain a 'load_kw' column.")
        load = frame["load_kw"].astype(float).to_numpy()
        if len(load) != 8760:
            raise ValueError("Load CSV must contain exactly 8760 hourly values.")
        if np.any(load < 0):
            raise ValueError("Load values cannot be negative.")
        return load.tolist()

    return generate_load_profile(user)
