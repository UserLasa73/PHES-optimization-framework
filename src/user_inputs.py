# user_inputs.py
# All user inputs as defined in proposal

class UserInputs:
    def __init__(self):
        # Site
        self.location = ""
        self.latitude = 8.9
        self.longitude = 79.9
        
        # Solar PV
        self.pv_kwp = 30.0
        self.tilt_angle = 10.0
        self.azimuth_angle = 0.0
        self.year = 2021 #shouldn't be a leap year!
        
        # Load
        self.daily_energy_kwh = 50.0
        self.load_csv_path = None  # Optional
        
        # Physical
        self.head_m = None  # None = auto-optimize (10-30m)
        self.upper_reservoir_type = "new_tank"
        self.lower_reservoir_type = "new_tank"
        
        # Requirements
        self.autonomy_days = 2.0
        self.budget_lkr = 2000000  # None = no budget constraint
        self.max_volume_m3 = 800  # Default matches current bound 800
        
        # Advanced
        self.evaporation_rate_mm_month = 50.0  # Sri Lanka avg
        self.pipe_roughness_m = 0.00015
        self.demand_spike_factor = 1.0  # 1.0 = no spikes
        self.has_grid_backup = False