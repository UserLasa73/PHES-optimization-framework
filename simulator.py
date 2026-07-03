"""
simulator.py
Complete simulator with ALL proposal features.
Uses physics.py for all calculations.
"""

import numpy as np
from constants import *
from physics import *


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
        
        # Get seepage losses from reservoir types (using physics.py)
        self.upper_seepage = get_seepage_loss_factor(user_inputs.upper_reservoir_type)
        self.lower_seepage = get_seepage_loss_factor(user_inputs.lower_reservoir_type)
        
        # Initial state (70% full)
        self.upper_volume = self.max_upper * 0.70
        self.lower_volume = self.max_lower * 0.70
        
        # Tracking
        self.hour = 0
        self.total_pumped = 0.0
        self.total_generated = 0.0
        self.total_unmet = 0.0
        self.total_curtailed = 0.0
        self.grid_used = 0.0
        
        self.history = {
            'upper_volume': [], 'lower_volume': [],
            'pumped_energy': [], 'generated_energy': [],
            'unmet_load': [], 'curtailed_energy': [],
            'grid_used': [], 'state': [],
            'solar_power': [], 'load_power': [], 'net_power': []
        }

    def simulate_hour(self, solar_kw, load_kw):
        """Simulate ONE hour with all logic"""
        net = solar_kw - load_kw
        
        # Apply demand spike (sudden increase)
        if self.user.demand_spike_factor > 1.0:
            if np.random.random() < 0.01:
                load_kw = load_kw * self.user.demand_spike_factor
                net = solar_kw - load_kw
        
        pumped = 0.0
        generated = 0.0
        unmet = 0.0
        curtailed = 0.0
        grid = 0.0
        state = 'idle'
        
        # --- PUMPING (excess power) ---
        if net > 0:
            excess = net
            space_upper = self.max_upper - self.upper_volume
            water_lower = self.lower_volume - self.min_lower
            
            if space_upper > 0 and water_lower > 0:
                # USE physics.py
                max_flow = calculate_pump_flow_rate(self.pump_power, self.head)
                max_pump_vol = max_flow * SECONDS_PER_HOUR
                
                pump_vol = min(max_pump_vol, space_upper, water_lower)
                
                if pump_vol > 0:
                    # Move water UP
                    self.upper_volume += pump_vol
                    self.lower_volume -= pump_vol
                    
                    # Energy used (USE physics.py)
                    power_used = calculate_pump_power_from_flow(
                        pump_vol / SECONDS_PER_HOUR, self.head
                    )
                    pumped = min(power_used, excess)
                    curtailed = max(0, excess - pumped)
                    state = 'pumping'
            else:
                curtailed = excess
                state = 'idle_full' if space_upper <= 0 else 'idle_empty'

                # --- GENERATING (deficit) ---
        elif net < 0:
            deficit = -net
            water_upper = self.upper_volume - self.min_upper
            space_lower = self.max_lower - self.lower_volume
            
            # Check grid backup
            if self.user.has_grid_backup and water_upper <= 0:
                grid = deficit
                state = 'grid_backup'
                self.grid_used += grid
            
            elif water_upper > 0 and space_lower > 0:
                # Calculate max turbine flow (USE physics.py)
                max_flow = self.turbine_power * 1000.0 / (
                    WATER_DENSITY * GRAVITY * self.head * TURBINE_EFFICIENCY
                )
                max_water_use = max_flow * SECONDS_PER_HOUR
                
                water_to_use = min(max_water_use, water_upper, space_lower)
                
                if water_to_use > 0:
                    flow = water_to_use / SECONDS_PER_HOUR
                    
                    # Calculate head loss (USE physics.py)
                    head_loss = calculate_total_head_loss(
                        flow, 
                        self.pipe_diam, 
                        self.head,
                        self.user.pipe_roughness_m
                    )
                    effective_head = max(0, self.head - head_loss)
                    
                    # Generate power (USE physics.py)
                    generated_power = calculate_turbine_power(flow, effective_head)
                    generated_power = min(generated_power, self.turbine_power)
                    
                    if generated_power >= deficit:
                        # Fully meet
                        generated = deficit
                        # Calculate actual water used (inverse of turbine_power)
                        actual_flow = deficit * 1000.0 / (
                            WATER_DENSITY * GRAVITY * effective_head * TURBINE_EFFICIENCY
                        )
                        actual_water = actual_flow * SECONDS_PER_HOUR
                        self.upper_volume -= actual_water
                        self.lower_volume += actual_water
                        state = 'generating_full'
                    else:
                        # Partial
                        generated = generated_power
                        unmet = deficit - generated_power
                        self.upper_volume -= water_to_use
                        self.lower_volume += water_to_use
                        state = 'generating_partial'
            else:
                unmet = deficit
                state = 'idle_empty' if water_upper <= 0 else 'idle_full'
        
        # --- APPLY LOSSES (monthly) ---
        self.hour += 1
        if self.hour % 720 == 0:  # Every 30 days
            # Estimate surface areas (USE physics.py)
            upper_area = estimate_reservoir_surface_area(self.upper_volume)
            lower_area = estimate_reservoir_surface_area(self.lower_volume)
            
            # Evaporation loss (USE physics.py)
            upper_evap = calculate_evaporation_loss(
                self.upper_volume, upper_area, 
                self.user.evaporation_rate_mm_month, 720
            )
            lower_evap = calculate_evaporation_loss(
                self.lower_volume, lower_area,
                self.user.evaporation_rate_mm_month, 720
            )
            
            # Seepage (from reservoir type)
            upper_seep = self.upper_volume * self.upper_seepage
            lower_seep = self.lower_volume * self.lower_seepage
            
            # Apply all losses
            self.upper_volume = max(self.min_upper, 
                                   self.upper_volume - upper_evap - upper_seep)
            self.lower_volume = max(self.min_lower,
                                   self.lower_volume - lower_evap - lower_seep)
        
        # Track totals
        self.total_pumped += pumped
        self.total_generated += generated
        self.total_unmet += unmet
        self.total_curtailed += curtailed
        
        # Store history
        self.history['upper_volume'].append(self.upper_volume)
        self.history['lower_volume'].append(self.lower_volume)
        self.history['pumped_energy'].append(pumped)
        self.history['generated_energy'].append(generated)
        self.history['unmet_load'].append(unmet)
        self.history['curtailed_energy'].append(curtailed)
        self.history['grid_used'].append(grid)
        self.history['state'].append(state)
        self.history['solar_power'].append(solar_kw)
        self.history['load_power'].append(load_kw)
        self.history['net_power'].append(net)
        
        return {
            'pumped': pumped, 'generated': generated,
            'unmet': unmet, 'curtailed': curtailed,
            'grid_used': grid, 'state': state
        }
    
    def simulate(self, solar_data, load_data):
        """Run full year simulation"""
        # Reset
        self.hour = 0
        self.upper_volume = self.max_upper * 0.70
        self.lower_volume = self.max_lower * 0.70
        self.total_pumped = 0.0
        self.total_generated = 0.0
        self.total_unmet = 0.0
        self.total_curtailed = 0.0
        self.grid_used = 0.0
        
        for key in self.history:
            self.history[key] = []
        
        # Simulate each hour
        for hour in range(len(solar_data)):
            self.simulate_hour(solar_data[hour], load_data[hour])
        
        # Calculate metrics
        total_load = sum(load_data)
        
        # Efficiency (USE physics.py)
        efficiency = calculate_round_trip_efficiency(
            self.total_pumped, self.total_generated
        )
        
        # Autonomy (days) (USE physics.py)
        stored_energy = calculate_stored_energy(self.upper_volume, self.head)
        avg_daily_load = total_load / 365.0 if total_load > 0 else 1.0
        autonomy = calculate_autonomy_days(stored_energy, avg_daily_load)
        
        # Check if autonomy requirement met
        autonomy_met = autonomy >= self.user.autonomy_days
        
        # Cost (USE physics.py)
        cost = self._calculate_cost()
        
        return {
            'metrics': {
                'efficiency_percent': efficiency,
                'autonomy_days': autonomy,
                'autonomy_met': autonomy_met,
                'capital_cost_lkr': cost,
                'total_pumped_kwh': self.total_pumped,
                'total_generated_kwh': self.total_generated,
                'total_unmet_kwh': self.total_unmet,
                'total_curtailed_kwh': self.total_curtailed,
                'grid_used_kwh': self.grid_used,
                'head_m': self.head,
                'volume_m3': self.volume,
                'pipe_diameter_m': self.pipe_diam,
                'pump_power_kw': self.pump_power,
                'turbine_power_kw': self.turbine_power,
                'upper_reservoir_type': self.user.upper_reservoir_type,
                'lower_reservoir_type': self.user.lower_reservoir_type
            },
            'history': self.history
        }
    
    def _calculate_cost(self):
        """Simple cost calculation with reservoir type factors"""
        # USE physics.py
        upper_factor = get_reservoir_cost_factor(self.user.upper_reservoir_type)
        lower_factor = get_reservoir_cost_factor(self.user.lower_reservoir_type)
        
        # Two reservoirs
        reservoir_cost = (self.volume * 1500.0 * upper_factor) + \
                        (self.volume * 1500.0 * lower_factor)
        
        # Equipment
        pump_cost = self.pump_power * 1200.0
        turbine_cost = self.turbine_power * 1200.0
        pipe_cost = self.head * 3.0 * 2.0 * 800.0
        pv_cost = self.user.pv_kwp * 120000.0
        
        # Balance of system + civil
        equipment = reservoir_cost + pump_cost + turbine_cost + pipe_cost + pv_cost
        total = equipment * 1.35
        
        return total