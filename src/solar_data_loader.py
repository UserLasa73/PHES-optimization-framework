"""Solar and load profile generation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location


def _non_leap_hourly_index(year: int, timezone: str):
    """Return exactly 8760 hourly timestamps for a non-leap simulation year."""
    start = pd.Timestamp(year=year, month=1, day=1, tz=timezone)
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz=timezone)
    times = pd.date_range(start, end, freq="h", inclusive="left")
    if len(times) == 8784:  # leap year: remove 29 February
        times = times[~((times.month == 2) & (times.day == 29))]
    if len(times) != 8760:
        raise ValueError(f"Expected 8760 hourly timestamps, received {len(times)}.")
    return times


def fetch_solar_data(user, capacity_kwp: float | None = None):
    """Generate a reproducible clear-sky PV profile for the selected site.

    This remains a clear-sky baseline rather than measured weather. The thesis
    and interface must describe it accordingly until historical/TMY data are added.
    """
    latitude = float(user.latitude)
    longitude = float(user.longitude)
    pv_kwp = float(user.pv_kwp if capacity_kwp is None else capacity_kwp)
    tilt = float(user.tilt_angle)
    azimuth = float(user.azimuth_angle)
    year = int(getattr(user, "year", 2021))

    location = Location(latitude, longitude, tz="Asia/Colombo")
    times = _non_leap_hourly_index(year, location.tz)
    solar_position = location.get_solarposition(times)
    clearsky = location.get_clearsky(times)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )["poa_global"]

    # Simple system derating retained as a documented baseline assumption.
    solar_kw = ((poa / 1000.0) * pv_kwp * 0.85).clip(lower=0.0)
    return solar_kw.to_numpy(dtype=float).tolist()


def annual_solar_yield_per_kwp(user) -> float:
    """Annual clear-sky energy yield (kWh per installed kWp)."""
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
    """Create a deterministic 8760-hour profile with exact daily energy.

    The previous implementation used a base load and then added peaks without
    renormalizing, so the annual energy exceeded the user's stated demand. This
    implementation guarantees each normal day sums to ``daily_energy_kwh``.
    """
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
    csv_path = getattr(user, "load_csv_path", None)
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Load profile not found: {path}")
        df = pd.read_csv(path)
        if "load_kw" not in df.columns:
            raise ValueError("Load CSV must contain a 'load_kw' column.")
        load = df["load_kw"].astype(float).to_numpy()
        if len(load) != 8760:
            raise ValueError("Load CSV must contain exactly 8760 hourly values.")
        if np.any(load < 0):
            raise ValueError("Load values cannot be negative.")
        return load.tolist()

    return generate_load_profile(user)
