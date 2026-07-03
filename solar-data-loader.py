import pvlib
import pandas as pd
import numpy as np
from pvlib.location import Location

def fetch_solar_data(latitude, longitude, pv_kwp, tilt=10, azimuth=0, year=2020):
    """
    Fetch hourly solar data using PVlib.
    
    Args:
        latitude: Site latitude
        longitude: Site longitude
        pv_kwp: PV capacity in kWp
        tilt: Panel tilt angle (degrees)
        azimuth: Panel azimuth (degrees from north)
        year: Year for data
    
    Returns:
        List of hourly solar power in kW
    """
    # Create location
    location = Location(latitude, longitude, tz='Asia/Colombo')
    
    # Create time range
    times = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', 
                          freq='H', tz=location.tz)
    
    # Get solar position
    solar_position = location.get_solarposition(times)
    
    # Get clearsky GHI
    clearsky = location.get_clearsky(times)
    ghi = clearsky['ghi']  # W/m²
    
    # Simple PV model (you can improve this)
    # POA irradiance (simplified)
    solar_kw = (ghi / 1000) * pv_kwp * 0.85  # 85% derating
    
    # Only during daytime
    solar_kw = solar_kw * (solar_position['elevation'] > 0)
    
    return solar_kw.tolist()

def fetch_load_data(daily_kwh, csv_path=None, spike_factor=1.0):
    """
    Get load data from CSV or generate profile.
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        if 'load_kw' in df.columns:
            load = df['load_kw'].tolist()
            if spike_factor > 1.0:
                # Apply spikes
                load = apply_spikes(load, spike_factor)
            return load
    
    # Generate from daily average
    return generate_load_profile(daily_kwh, spike_factor)

def generate_load_profile(daily_kwh, spike_factor=1.0):
    """Generate realistic load profile."""
    hours = 8760
    hour_of_day = np.arange(hours) % 24
    
    base = daily_kwh / 24.0
    load = np.ones(hours) * base
    
    # Morning peak
    morning = (hour_of_day >= 6) & (hour_of_day <= 8)
    load[morning] = base * 1.8
    
    # Evening peak
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
    import numpy as np
    load = np.array(load)
    spike_hours = np.random.choice(len(load), size=int(len(load)*0.01), replace=False)
    load[spike_hours] = load[spike_hours] * spike_factor
    return load.tolist()