"""Hourly physics-based simulator for a closed-loop solar-PHES system."""

from __future__ import annotations

import math
from typing import Iterable

from src.constants import (
    MAX_WATER_FRACTION,
    MIN_WATER_FRACTION,
    SECONDS_PER_HOUR,
    TIME_STEP_HOURS,
)
from src.cost_model import calculate_capital_cost
from src.physics import (
    calculate_autonomy_days,
    calculate_evaporation_loss,
    calculate_pump_flow_rate,
    calculate_pump_power_from_flow,
    calculate_round_trip_efficiency,
    calculate_stored_energy,
    calculate_total_head_loss,
    calculate_turbine_flow_for_power,
    calculate_turbine_power,
    estimate_reservoir_surface_area,
    get_seepage_loss_factor,
)


class PumpedHydroSimulator:
    """Simulate a candidate design using one-hour timesteps.

    Volume convention
    -----------------
    ``design_params['volume_m3']`` is the combined gross installed capacity of
    the upper and lower reservoirs. Each reservoir is assigned half. This now
    matches the cost model and Streamlit display.
    """

    def __init__(self, user_inputs, design_params):
        self.user = user_inputs
        self.design = design_params

        self.head = float(design_params["head_m"])
        self.total_installed_volume = float(design_params["volume_m3"])
        self.pipe_diam = float(design_params["pipe_diameter_m"])
        self.pump_power = float(design_params["pump_power_kw"])
        self.turbine_power = float(design_params["turbine_power_kw"])

        if min(
            self.head,
            self.total_installed_volume,
            self.pipe_diam,
            self.pump_power,
            self.turbine_power,
        ) <= 0:
            raise ValueError("All PHES design parameters must be positive.")

        self.upper_capacity = self.total_installed_volume / 2.0
        self.lower_capacity = self.total_installed_volume / 2.0
        self.max_upper = self.upper_capacity * MAX_WATER_FRACTION
        self.max_lower = self.lower_capacity * MAX_WATER_FRACTION
        self.min_upper = self.upper_capacity * MIN_WATER_FRACTION
        self.min_lower = self.lower_capacity * MIN_WATER_FRACTION

        self.upper_surface_area = estimate_reservoir_surface_area(self.upper_capacity)
        self.lower_surface_area = estimate_reservoir_surface_area(self.lower_capacity)
        self.upper_seepage = get_seepage_loss_factor(user_inputs.upper_reservoir_type)
        self.lower_seepage = get_seepage_loss_factor(user_inputs.lower_reservoir_type)

        self.history = {}
        self._reset()

    def _new_history(self):
        return {
            "upper_volume": [],
            "lower_volume": [],
            "pumped_energy": [],
            "generated_energy": [],
            "unmet_load": [],
            "curtailed_energy": [],
            "grid_used": [],
            "state": [],
            "solar_power": [],
            "load_power": [],
            "net_power": [],
            "flow_m3s": [],
            "head_loss_m": [],
            "evaporation_loss_m3": [],
            "seepage_loss_m3": [],
        }

    def _reset(self):
        self.hour = 0

        # Start with no usable energy in the upper reservoir. This prevents the
        # annual simulation from generating unearned energy at the beginning.
        self.upper_volume = self.min_upper
        self.lower_volume = self.max_lower
        self.initial_upper_volume = self.upper_volume
        self.initial_lower_volume = self.lower_volume

        self.total_pumped = 0.0
        self.total_generated = 0.0
        self.total_unmet = 0.0
        self.total_curtailed = 0.0
        self.total_grid_used = 0.0
        self.total_evaporation_loss = 0.0
        self.total_seepage_loss = 0.0
        self.history = self._new_history()

    def _apply_water_losses(self, hours: float):
        """Apply prorated evaporation and seepage without dropping below dead storage."""
        upper_evap_requested = calculate_evaporation_loss(
            self.upper_surface_area,
            self.user.evaporation_rate_mm_month,
            hours,
        )
        lower_evap_requested = calculate_evaporation_loss(
            self.lower_surface_area,
            self.user.evaporation_rate_mm_month,
            hours,
        )
        month_fraction = hours / 720.0
        upper_seep_requested = self.upper_volume * self.upper_seepage * month_fraction
        lower_seep_requested = self.lower_volume * self.lower_seepage * month_fraction

        upper_available = max(0.0, self.upper_volume - self.min_upper)
        lower_available = max(0.0, self.lower_volume - self.min_lower)

        # Split actual losses proportionally when the requested total exceeds
        # water available above minimum operating storage.
        def actual_losses(evap_requested, seep_requested, available):
            requested = evap_requested + seep_requested
            if requested <= 0 or available <= 0:
                return 0.0, 0.0
            scale = min(1.0, available / requested)
            return evap_requested * scale, seep_requested * scale

        upper_evap, upper_seep = actual_losses(
            upper_evap_requested, upper_seep_requested, upper_available
        )
        lower_evap, lower_seep = actual_losses(
            lower_evap_requested, lower_seep_requested, lower_available
        )

        self.upper_volume -= upper_evap + upper_seep
        self.lower_volume -= lower_evap + lower_seep

        evaporation = upper_evap + lower_evap
        seepage = upper_seep + lower_seep
        self.total_evaporation_loss += evaporation
        self.total_seepage_loss += seepage
        return evaporation, seepage

    def simulate_hour(self, solar_kw: float, load_kw: float):
        """Simulate one one-hour timestep and return its energy flows."""
        solar_kw = max(0.0, float(solar_kw))
        load_kw = max(0.0, float(load_kw))
        net_kw = solar_kw - load_kw

        pumped_kwh = 0.0
        generated_kwh = 0.0
        unmet_kwh = 0.0
        curtailed_kwh = 0.0
        grid_kwh = 0.0
        flow_m3s = 0.0
        head_loss_m = 0.0
        evaporation_loss_m3 = 0.0
        seepage_loss_m3 = 0.0
        state = "idle"

        if net_kw > 0:
            available_power_kw = min(net_kw, self.pump_power)
            space_upper = max(0.0, self.max_upper - self.upper_volume)
            transferable_lower = max(0.0, self.lower_volume - self.min_lower)

            if available_power_kw > 0 and space_upper > 0 and transferable_lower > 0:
                possible_flow = calculate_pump_flow_rate(
                    available_power_kw,
                    self.head,
                    self.pipe_diam,
                    self.user.pipe_roughness_m,
                )
                pump_volume = min(
                    possible_flow * SECONDS_PER_HOUR,
                    space_upper,
                    transferable_lower,
                )

                if pump_volume > 0:
                    flow_m3s = pump_volume / SECONDS_PER_HOUR
                    head_loss_m = calculate_total_head_loss(
                        flow_m3s,
                        self.pipe_diam,
                        self.head,
                        self.user.pipe_roughness_m,
                    )
                    total_dynamic_head = self.head + head_loss_m
                    actual_power_kw = calculate_pump_power_from_flow(
                        flow_m3s, total_dynamic_head
                    )
                    # Numerical guard; the flow solver should already satisfy it.
                    actual_power_kw = min(actual_power_kw, available_power_kw)

                    self.upper_volume += pump_volume
                    self.lower_volume -= pump_volume
                    pumped_kwh = actual_power_kw * TIME_STEP_HOURS
                    curtailed_kwh = max(0.0, net_kw - actual_power_kw) * TIME_STEP_HOURS
                    state = "pumping"
                else:
                    curtailed_kwh = net_kw * TIME_STEP_HOURS
            else:
                curtailed_kwh = net_kw * TIME_STEP_HOURS

        elif net_kw < 0:
            deficit_kw = -net_kw
            target_output_kw = min(deficit_kw, self.turbine_power)
            transferable_upper = max(0.0, self.upper_volume - self.min_upper)
            space_lower = max(0.0, self.max_lower - self.lower_volume)

            if target_output_kw > 0 and transferable_upper > 0 and space_lower > 0:
                requested_flow = calculate_turbine_flow_for_power(
                    target_output_kw,
                    self.head,
                    self.pipe_diam,
                    self.user.pipe_roughness_m,
                )
                water_to_use = min(
                    requested_flow * SECONDS_PER_HOUR,
                    transferable_upper,
                    space_lower,
                )

                if water_to_use > 0:
                    flow_m3s = water_to_use / SECONDS_PER_HOUR
                    head_loss_m = calculate_total_head_loss(
                        flow_m3s,
                        self.pipe_diam,
                        self.head,
                        self.user.pipe_roughness_m,
                    )
                    net_head = max(0.0, self.head - head_loss_m)
                    actual_output_kw = min(
                        calculate_turbine_power(flow_m3s, net_head),
                        self.turbine_power,
                        deficit_kw,
                    )

                    self.upper_volume -= water_to_use
                    self.lower_volume += water_to_use
                    generated_kwh = actual_output_kw * TIME_STEP_HOURS
                    state = "generating"

            remaining_deficit_kwh = max(
                0.0, deficit_kw * TIME_STEP_HOURS - generated_kwh
            )
            if self.user.has_grid_backup and remaining_deficit_kwh > 0:
                grid_kwh = remaining_deficit_kwh
                state = "grid_backup" if generated_kwh == 0 else "generating_plus_grid"
            else:
                unmet_kwh = remaining_deficit_kwh
                if generated_kwh > 0 and unmet_kwh > 0:
                    state = "generating_partial"
                elif generated_kwh == 0:
                    state = "unmet"

        self.hour += 1
        if self.hour % 720 == 0:
            evaporation_loss_m3, seepage_loss_m3 = self._apply_water_losses(720.0)

        self.total_pumped += pumped_kwh
        self.total_generated += generated_kwh
        self.total_unmet += unmet_kwh
        self.total_curtailed += curtailed_kwh
        self.total_grid_used += grid_kwh

        values = {
            "upper_volume": self.upper_volume,
            "lower_volume": self.lower_volume,
            "pumped_energy": pumped_kwh,
            "generated_energy": generated_kwh,
            "unmet_load": unmet_kwh,
            "curtailed_energy": curtailed_kwh,
            "grid_used": grid_kwh,
            "state": state,
            "solar_power": solar_kw,
            "load_power": load_kw,
            "net_power": net_kw,
            "flow_m3s": flow_m3s,
            "head_loss_m": head_loss_m,
            "evaporation_loss_m3": evaporation_loss_m3,
            "seepage_loss_m3": seepage_loss_m3,
        }
        for key, value in values.items():
            self.history[key].append(value)

        return {
            "pumped": pumped_kwh,
            "generated": generated_kwh,
            "unmet": unmet_kwh,
            "curtailed": curtailed_kwh,
            "grid_used": grid_kwh,
            "state": state,
        }

    def simulate(self, solar_data: Iterable[float], load_data: Iterable[float]):
        solar = list(solar_data)
        load = list(load_data)
        if len(solar) != len(load):
            raise ValueError("Solar and load profiles must have equal length.")
        if not solar:
            raise ValueError("Solar and load profiles cannot be empty.")

        self._reset()
        for solar_kw, load_kw in zip(solar, load):
            self.simulate_hour(solar_kw, load_kw)

        # Apply losses for the final partial month (8760 is not divisible by 720).
        remaining_hours = self.hour % 720
        if remaining_hours:
            evaporation, seepage = self._apply_water_losses(float(remaining_hours))
            # Attach the final prorated loss to the last history row.
            self.history["evaporation_loss_m3"][-1] += evaporation
            self.history["seepage_loss_m3"][-1] += seepage
            self.history["upper_volume"][-1] = self.upper_volume
            self.history["lower_volume"][-1] = self.lower_volume

        total_load = sum(self.history["load_power"]) * TIME_STEP_HOURS
        total_grid_or_storage_served = total_load - self.total_unmet
        load_served_ratio = (
            total_grid_or_storage_served / total_load if total_load > 0 else 1.0
        )

        realized_efficiency = calculate_round_trip_efficiency(
            self.total_pumped, self.total_generated
        )

        net_stored_volume = self.upper_volume - self.initial_upper_volume
        net_stored_energy = calculate_stored_energy(
            max(0.0, net_stored_volume), self.head
        )
        balance_adjusted_output = self.total_generated + net_stored_energy
        adjusted_efficiency = calculate_round_trip_efficiency(
            self.total_pumped, balance_adjusted_output
        )

        usable_upper_volume = self.max_upper - self.min_upper
        usable_stored_energy = calculate_stored_energy(usable_upper_volume, self.head)
        average_daily_load = total_load / (len(load) / 24.0) if total_load > 0 else 0.0
        autonomy = calculate_autonomy_days(usable_stored_energy, average_daily_load)

        initial_water = self.initial_upper_volume + self.initial_lower_volume
        final_water = self.upper_volume + self.lower_volume
        measured_water_loss = self.total_evaporation_loss + self.total_seepage_loss
        water_balance_residual = initial_water - measured_water_loss - final_water

        finite_metrics = all(
            math.isfinite(value)
            for value in (
                adjusted_efficiency,
                realized_efficiency,
                autonomy,
                water_balance_residual,
            )
        )
        physically_valid = (
            finite_metrics
            and adjusted_efficiency <= 100.0 + 1e-6
            and adjusted_efficiency >= -1e-6
            and abs(water_balance_residual) <= 1e-6 * max(1.0, initial_water)
            and self.min_upper - 1e-8 <= self.upper_volume <= self.max_upper + 1e-8
            and self.min_lower - 1e-8 <= self.lower_volume <= self.max_lower + 1e-8
        )

        cost = self._calculate_cost()
        metrics = {
            # Main optimization target: endpoint-storage-adjusted annual efficiency.
            "efficiency_percent": adjusted_efficiency,
            "realized_efficiency_percent": realized_efficiency,
            "net_stored_energy_kwh": net_stored_energy,
            "autonomy_days": autonomy,
            "autonomy_met": autonomy >= self.user.autonomy_days,
            "load_served_ratio": load_served_ratio,
            "capital_cost_lkr": cost,
            "total_load_kwh": total_load,
            "total_pumped_kwh": self.total_pumped,
            "total_generated_kwh": self.total_generated,
            "total_unmet_kwh": self.total_unmet,
            "total_curtailed_kwh": self.total_curtailed,
            "grid_used_kwh": self.total_grid_used,
            "total_evaporation_loss_m3": self.total_evaporation_loss,
            "total_seepage_loss_m3": self.total_seepage_loss,
            "water_balance_residual_m3": water_balance_residual,
            "initial_upper_volume_m3": self.initial_upper_volume,
            "final_upper_volume_m3": self.upper_volume,
            "initial_lower_volume_m3": self.initial_lower_volume,
            "final_lower_volume_m3": self.lower_volume,
            "is_physically_valid": physically_valid,
            "head_m": self.head,
            "volume_m3": self.total_installed_volume,
            "upper_capacity_m3": self.upper_capacity,
            "lower_capacity_m3": self.lower_capacity,
            "pipe_diameter_m": self.pipe_diam,
            "pump_power_kw": self.pump_power,
            "turbine_power_kw": self.turbine_power,
            "upper_reservoir_type": self.user.upper_reservoir_type,
            "lower_reservoir_type": self.user.lower_reservoir_type,
        }
        return {"metrics": metrics, "history": self.history}

    def _calculate_cost(self):
        result = calculate_capital_cost(
            self.total_installed_volume,
            self.head,
            self.pipe_diam,
            self.pump_power,
            self.turbine_power,
            self.user.pv_kwp,
            self.user.upper_reservoir_type,
            self.user.lower_reservoir_type,
        )
        return result["total_lkr"]
