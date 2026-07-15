"""
solar_data_loader.py
Fetches solar and load data using user inputs.
Uses pvlib with proper tilt/azimuth modeling.
"""

import pvlib
import pandas as pd
import numpy as np
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, FixedMount
from pvlib.modelchain import ModelChain

from src.user_inputs import UserInputs


def fetch_solar_data(user):
    """Fetch solar data with tilt/azimuth correction."""
    
    latitude = user.latitude
    longitude = user.longitude
    pv_kwp = user.pv_kwp
    tilt = user.tilt_angle
    azimuth = user.azimuth_angle
    
    location = Location(latitude, longitude, tz='Asia/Colombo')
    
    # Create time range
    times = pd.date_range('2021-01-01', periods=8760, freq='h', tz=location.tz)
    
    # Get solar position
    solar_position = location.get_solarposition(times)
    
    # Get GHI
    clearsky = location.get_clearsky(times)
    ghi = clearsky['ghi']
    
    # Get DHI and DNI
    dhi = clearsky['dhi']
    dni = clearsky['dni']
    
    # Calculate POA irradiance (this considers tilt and azimuth!)
    from pvlib.irradiance import get_total_irradiance
    poa_irradiance = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=dni,
        ghi=ghi,
        dhi=dhi
    )
    
    # POA irradiance in W/m²
    poa = poa_irradiance['poa_global']
    
    # Convert to power: (POA / 1000) * PV capacity * derating
    solar_kw = (poa / 1000) * pv_kwp * 0.85
    
    # Only positive values
    solar_kw = solar_kw.clip(lower=0)
    
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
    spike_factor = getattr(user, 'demand_spike_factor', 1.0)
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
                    load.append(load[-1])  # Pad with last value
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
    spike_factor = getattr(user, 'demand_spike_factor', 1.0)
    
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


# ===== FOR TESTING =====
if __name__ == "__main__":
    # Test the solar data fetcher
    from user_inputs import UserInputs
    
    user = UserInputs()
    user.latitude = 8.9
    user.longitude = 79.9
    user.pv_kwp = 30.0
    user.tilt_angle = 10.0
    user.azimuth_angle = 0.0
    
    solar = fetch_solar_data(user)
    print(f"Solar data length: {len(solar)} hours")
    print(f"Total solar: {sum(solar):.0f} kWh/year")
    print(f"First 10 values: {solar[:10]}")
    print(f"Max solar: {max(solar):.2f} kW")