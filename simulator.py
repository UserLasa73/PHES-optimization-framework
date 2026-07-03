# simulator.py
# Complete simulator with ALL proposal features

import numpy as np
from constants import *
from user_inputs import UserInputs

class PumpedHydroSimulator:
    def __init__(self, user_inputs, design_params):
        """
        user_inputs: UserInputs object (fixed)
        design_params: Dict being optimized
        """

        self.user = user_inputs
        self.design = design_params
        
        # Extract design parameters
        self.head = design_params['head_m']
        self.volume = design_params['volume_m3']
        self.pipe_diam = design_params['pipe_diameter_m']
        self.pump_power = design_params['pump_power_kw']
        self.turbine_power = design_params['turbine_power_kw']

        # Reservoir limits
        self.max_upper = self.volume * 0.95
        self.max_lower = self.volume * 0.95
        self.min_upper = self.volume * 0.05
        self.min_lower = self.volume * 0.05
        
        # Get seepage losses from reservoir types
        self.upper_seepage = SEEPAGE_LOSS.get(user_inputs.upper_reservoir_type, 0.0)
        self.lower_seepage = SEEPAGE_LOSS.get(user_inputs.lower_reservoir_type, 0.0)
        
        # Initial state (70% full)
        self.upper_volume = self.max_upper * 0.70
        self.lower_volume = self.max_lower * 0.70
        
        # Tracking
        self.hour = 0
        self.total_pumped = 0.0
        self.total_generated = 0.0
        self.total_unmet = 0.0
        self.total_curtailed = 0.0
        self.grid_used = 0.0  # Track grid usage

        self.history = {
            'upper_volume': [], 'lower_volume': [],
            'pumped_energy': [], 'generated_energy': [],
            'unmet_load': [], 'curtailed_energy': [],
            'grid_used': [], 'state': []
        }

    
    