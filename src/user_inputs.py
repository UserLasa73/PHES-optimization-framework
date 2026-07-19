"""Container for user and site inputs used by the PHES framework."""

from src.constants import DEFAULT_PIPE_ROUGHNESS_M, DEFAULT_RANDOM_SEED


class UserInputs:
    def __init__(self):
        # Site
        self.location = "Vavuniya"
        self.latitude = 8.9
        self.longitude = 79.9
        self.year = 2021  # non-leap year for 8760-hour simulations

        # Solar PV
        self.pv_kwp = 20.0
        self.tilt_angle = 10.0
        # pvlib convention: 180° is south-facing, 0° is north-facing.
        self.azimuth_angle = 180.0

        # Load
        self.daily_energy_kwh = 20.0
        self.load_csv_path = None

        # Reservoir configuration
        # volume_m3 in a design means TOTAL installed reservoir capacity:
        # upper gross capacity + lower gross capacity.
        self.upper_reservoir_type = "new_tank"
        self.lower_reservoir_type = "new_tank"

        # Requirements
        self.autonomy_days = 0.5
        self.budget_lkr = None
        self.max_volume_m3 = 800.0

        # Advanced assumptions
        self.evaporation_rate_mm_month = 50.0
        self.pipe_roughness_m = DEFAULT_PIPE_ROUGHNESS_M
        self.demand_spike_factor = 1.0
        self.has_grid_backup = False
        self.random_seed = DEFAULT_RANDOM_SEED
