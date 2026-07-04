"""
solar_data_loader.py
Fetches solar and load data using user inputs.
"""

import pvlib
import pandas as pd
import numpy as np
from pvlib.location import Location


def fetch_solar_data(user):
    """
    Fetch hourly solar data using PVlib from UserInputs object.
    
    Args:
        user: UserInputs object containing latitude, longitude, pv_kwp, etc.
    
    Returns:
        List of hourly solar power in kW (8760 hours)
    """
    # Get values from user object
    latitude = user.latitude
    longitude = user.longitude
    pv_kwp = user.pv_kwp
    tilt = user.tilt_angle
    azimuth = user.azimuth_angle
    year = getattr(user, 'year', 2021)  # Default 2021 if not set
    
    # Create location
    location = Location(latitude, longitude, tz='Asia/Colombo')
    
    # Create time range (non-leap year = 8760 hours)
    start = pd.Timestamp(f'{year}-01-01 00:00:00', tz=location.tz)
    times = pd.date_range(start=start, periods=8760, freq='h')
    
    # Get solar position
    solar_position = location.get_solarposition(times)
    
    # Get clearsky GHI
    clearsky = location.get_clearsky(times)
    ghi = clearsky['ghi']  # W/m²
    
    # Simple PV model
    solar_kw = (ghi / 1000) * pv_kwp * 0.85  # 85% derating
    
    # Only during daytime (elevation > 0)
    solar_kw = solar_kw * (solar_position['elevation'] > 0)
    
    return solar_kw.tolist()


def fetch_load_data(user):
    """
    Get load data from UserInputs object.
    
    Args:
        user: UserInputs object with daily_energy_kwh, load_csv_path, etc.
    
    Returns:
        List of hourly load in kW (8760 hours)
    """
    daily_kwh = user.daily_energy_kwh
    spike_factor = user.demand_spike_factor
    csv_path = getattr(user, 'load_csv_path', None)
    
    if csv_path:
        df = pd.read_csv(csv_path)
        if 'load_kw' in df.columns:
            load = df['load_kw'].tolist()
            # Ensure 8760 hours
            if len(load) > 8760:
                load = load[:8760]
            elif len(load) < 8760:
                while len(load) < 8760:
                    load.append(load[-1])  # Pad
            if spike_factor > 1.0:
                load = apply_spikes(load, spike_factor)
            return load
    
    # Generate from daily average
    return generate_load_profile(user)


def generate_load_profile(user):
    """Generate realistic load profile from UserInputs."""
    hours = 8760
    hour_of_day = np.arange(hours) % 24
    
    daily_kwh = user.daily_energy_kwh
    spike_factor = user.demand_spike_factor
    
    base = daily_kwh / 24.0
    load = np.ones(hours) * base
    
    # Morning peak (6-8 am)
    morning = (hour_of_day >= 6) & (hour_of_day <= 8)
    load[morning] = base * 1.8
    
    # Evening peak (6-10 pm)
    evening = (hour_of_day >= 18) & (hour_of_day <= 22)
    load[evening] = base * 2.2
    
    # Random variation
    load = load * np.random.uniform(0.85, 1.15, hours)
    
    # Apply spikes
    if spike_factor > 1.0:
        load = apply_spikes(load, spike_factor)
    
    return load.tolist()


def apply_spikes(load, spike_factor):
    """Apply demand spikes to load profile."""
    load = np.array(load)
    spike_hours = np.random.choice(len(load), size=int(len(load)*0.01), replace=False)
    load[spike_hours] = load[spike_hours] * spike_factor
    return load.tolist()