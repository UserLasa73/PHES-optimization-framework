import numpy as np
from constants import *
from physics import *
from cost_model import calculate_capital_cost


class PumpedHydroSimulator:
    def __init__(self, user_inputs, design_params):
        self.user = user_inputs
        self.design = design_params
        
        # Design parameters
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
        
        # Seepage losses
        self.upper_seepage = get_seepage_loss_factor(user_inputs.upper_reservoir_type)
        self.lower_seepage = get_seepage_loss_factor(user_inputs.lower_reservoir_type)
        
        # Reset state
        self._reset()
        
        # History
        self.history = {
            'upper_volume': [], 'lower_volume': [],
            'pumped_energy': [], 'generated_energy': [],
            'unmet_load': [], 'curtailed_energy': [],
            'grid_used': [], 'state': [],
            'solar_power': [], 'load_power': [], 'net_power': []
        }
    
    def _reset(self):
        """Reset simulator state"""
        self.hour = 0
        self.upper_volume = self.max_upper * 0.70
        self.lower_volume = self.max_lower * 0.70
        self.total_pumped = 0.0
        self.total_generated = 0.0
        self.total_unmet = 0.0
        self.total_curtailed = 0.0
        self.grid_used = 0.0
    
    def simulate_hour(self, solar_kw, load_kw):
        """Simulate ONE hour"""
        net = solar_kw - load_kw
        
        # Demand spike
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
        
        # ===== PUMPING (CORRECTED - WITH POWER LIMIT) =====
        if net > 0:
            excess = net
            space_upper = self.max_upper - self.upper_volume
            water_lower = self.lower_volume - self.min_lower
            
            if space_upper > 0 and water_lower > 0:
                # Max flow pump CAN achieve (with full head)
                max_flow = calculate_pump_flow_rate(self.pump_power, self.head)
                max_pump_vol = max_flow * SECONDS_PER_HOUR
                
                # Max flow we CAN AFFORD with available excess power
                # Q = (P * efficiency) / (rho * g * H)
                max_flow_with_power = (excess * 1000.0 * PUMP_EFFICIENCY) / (WATER_DENSITY * GRAVITY * self.head)
                max_vol_with_power = max_flow_with_power * SECONDS_PER_HOUR
                
                # Actual pump volume = minimum of all limits
                pump_vol = min(max_pump_vol, max_vol_with_power, space_upper, water_lower)
                
                if pump_vol > 0:
                    flow = pump_vol / SECONDS_PER_HOUR
                    
                    # Head loss during pumping
                    head_loss = calculate_total_head_loss(
                        flow, self.pipe_diam, self.head, self.user.pipe_roughness_m
                    )
                    effective_head = max(0, self.head - head_loss)
                    
                    # ACTUAL power needed
                    power_used = calculate_pump_power_from_flow(flow, effective_head)
                    
                    # Move water UP
                    self.upper_volume += pump_vol
                    self.lower_volume -= pump_vol
                    
                    # Record energy
                    pumped = power_used  # ← ALWAYS power_used
                    curtailed = max(0, excess - power_used)  # Remaining solar wasted
                    state = 'pumping'
                else:
                    curtailed = excess
                    state = 'idle'
            else:
                curtailed = excess
                state = 'idle'
        
        # ===== GENERATING =====
        elif net < 0:
            deficit = -net
            water_upper = self.upper_volume - self.min_upper
            space_lower = self.max_lower - self.lower_volume
            
            # Grid backup
            if self.user.has_grid_backup and water_upper <= 0:
                grid = deficit
                state = 'grid_backup'
                self.grid_used += grid
            
            elif water_upper > 0 and space_lower > 0:
                # Calculate flow from turbine
                max_flow = self.turbine_power * 1000.0 / (WATER_DENSITY * GRAVITY * self.head * TURBINE_EFFICIENCY)
                max_water_use = max_flow * SECONDS_PER_HOUR
                
                water_to_use = min(max_water_use, water_upper, space_lower)
                
                if water_to_use > 0:
                    flow = water_to_use / SECONDS_PER_HOUR
                    
                    # Head loss
                    head_loss = calculate_total_head_loss(
                        flow, self.pipe_diam, self.head, self.user.pipe_roughness_m
                    )
                    effective_head = max(0, self.head - head_loss)
                    
                    # ACTUAL power generated
                    generated_power = calculate_turbine_power(flow, effective_head)
                    generated_power = min(generated_power, self.turbine_power)
                    
                    # Move water DOWN (use ALL water_to_use)
                    self.upper_volume -= water_to_use
                    self.lower_volume += water_to_use
                    
                    # Record ACTUAL energy generated
                    generated = generated_power  # 1 hour of this power
                    
                    # Check if we met the deficit
                    if generated >= deficit:
                        # Fully met (surplus generation is curtailed)
                        state = 'generating_full'
                    else:
                        # Partially met
                        unmet = deficit - generated
                        state = 'generating_partial'
                else:
                    unmet = deficit
                    state = 'idle'
            else:
                unmet = deficit
                state = 'idle'
        
        # ===== LOSSES (monthly) =====
        self.hour += 1
        if self.hour % 720 == 0:
            upper_area = estimate_reservoir_surface_area(self.upper_volume)
            lower_area = estimate_reservoir_surface_area(self.lower_volume)
            
            upper_evap = calculate_evaporation_loss(
                self.upper_volume, upper_area, self.user.evaporation_rate_mm_month, 720
            )
            lower_evap = calculate_evaporation_loss(
                self.lower_volume, lower_area, self.user.evaporation_rate_mm_month, 720
            )
            
            upper_seep = self.upper_volume * self.upper_seepage
            lower_seep = self.lower_volume * self.lower_seepage
            
            self.upper_volume = max(self.min_upper, self.upper_volume - upper_evap - upper_seep)
            self.lower_volume = max(self.min_lower, self.lower_volume - lower_evap - lower_seep)
        
        # ===== TRACK TOTALS =====
        self.total_pumped += pumped
        self.total_generated += generated
        self.total_unmet += unmet
        self.total_curtailed += curtailed
        
        # ===== STORE HISTORY =====
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
        """Run full simulation"""
        self._reset()
        
        for key in self.history:
            self.history[key] = []
        
        for hour in range(len(solar_data)):
            self.simulate_hour(solar_data[hour], load_data[hour])
        
        total_load = sum(load_data)
        
        # Efficiency
        efficiency = calculate_round_trip_efficiency(
            self.total_pumped, self.total_generated
        )
        
        # Autonomy
        # Use the maximum water stored in upper reservoir
        if self.history['upper_volume']:
            max_upper_volume = max(self.history['upper_volume'])
        else:
            max_upper_volume = self.upper_volume
        
        total_stored_energy = calculate_stored_energy(max_upper_volume, self.head)
        avg_daily_load = total_load / 365.0 if total_load > 0 else 1.0
        autonomy = total_stored_energy / avg_daily_load
        autonomy_met = autonomy >= self.user.autonomy_days
        
        # Cost
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
        cost = calculate_capital_cost(
            self.volume,
            self.head,
            self.pipe_diam,
            self.pump_power,
            self.turbine_power,
            self.user.pv_kwp,
            self.user.upper_reservoir_type,
            self.user.lower_reservoir_type
        )
        # If it's a dict, extract total
        if isinstance(cost, dict):
            return cost['total_lkr']
        return cost