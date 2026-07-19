import math
import unittest

from src.cost_model import calculate_capital_cost
from src.physics import (
    calculate_pump_flow_rate,
    calculate_stored_energy,
    calculate_total_head_loss,
)
from src.simulator import PumpedHydroSimulator
from src.solar_data_loader import generate_load_profile
from src.user_inputs import UserInputs


class CostModelTests(unittest.TestCase):
    def test_reservoir_cost_is_monotonic_across_old_thresholds(self):
        common = dict(
            head_m=20.0,
            pipe_diameter_m=0.15,
            pump_power_kw=5.0,
            turbine_power_kw=5.0,
            pv_kwp=20.0,
            upper_type="new_tank",
            lower_type="new_tank",
        )
        costs = [
            calculate_capital_cost(volume_m3=v, **common)["total_lkr"]
            for v in [99.0, 100.0, 101.0, 299.0, 300.0, 301.0, 499.0, 500.0, 501.0]
        ]
        self.assertTrue(all(b > a for a, b in zip(costs, costs[1:])))

    def test_larger_pipe_costs_more(self):
        common = dict(
            volume_m3=200.0,
            head_m=20.0,
            pump_power_kw=5.0,
            turbine_power_kw=5.0,
            pv_kwp=20.0,
            upper_type="new_tank",
            lower_type="new_tank",
        )
        small = calculate_capital_cost(pipe_diameter_m=0.10, **common)["total_lkr"]
        large = calculate_capital_cost(pipe_diameter_m=0.30, **common)["total_lkr"]
        self.assertGreater(large, small)


class LoadProfileTests(unittest.TestCase):
    def test_each_day_matches_requested_energy_without_spikes(self):
        user = UserInputs()
        user.daily_energy_kwh = 24.0
        user.demand_spike_factor = 1.0
        load = generate_load_profile(user)
        self.assertEqual(len(load), 8760)
        for day in [0, 1, 100, 364]:
            daily_sum = sum(load[day * 24 : (day + 1) * 24])
            self.assertAlmostEqual(daily_sum, 24.0, places=10)


class PhysicsTests(unittest.TestCase):
    def test_friction_reduces_pump_flow(self):
        no_friction_geometry = calculate_pump_flow_rate(10.0, 20.0)
        with_friction = calculate_pump_flow_rate(10.0, 20.0, 0.10, 0.00015)
        self.assertGreater(no_friction_geometry, with_friction)

    def test_head_loss_decreases_with_pipe_diameter(self):
        small = calculate_total_head_loss(0.01, 0.10, 20.0)
        large = calculate_total_head_loss(0.01, 0.30, 20.0)
        self.assertGreater(small, large)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.user = UserInputs()
        self.user.daily_energy_kwh = 20.0
        self.user.evaporation_rate_mm_month = 0.0
        self.user.upper_reservoir_type = "new_tank"
        self.user.lower_reservoir_type = "new_tank"
        self.user.has_grid_backup = False
        self.design = {
            "volume_m3": 200.0,
            "head_m": 20.0,
            "pipe_diameter_m": 0.30,
            "pump_power_kw": 10.0,
            "turbine_power_kw": 10.0,
        }

    def test_no_generation_from_unearned_initial_storage(self):
        result = PumpedHydroSimulator(self.user, self.design).simulate([0.0], [10.0])
        metrics = result["metrics"]
        self.assertEqual(metrics["total_generated_kwh"], 0.0)
        self.assertEqual(metrics["total_unmet_kwh"], 10.0)

    def test_generation_never_exceeds_deficit(self):
        simulator = PumpedHydroSimulator(self.user, self.design)
        result = simulator.simulate([20.0, 0.0], [0.0, 3.0])
        self.assertLessEqual(result["metrics"]["total_generated_kwh"], 3.0 + 1e-9)

    def test_water_and_energy_diagnostics_are_physical(self):
        simulator = PumpedHydroSimulator(self.user, self.design)
        result = simulator.simulate([20.0, 0.0], [0.0, 20.0])
        metrics = result["metrics"]
        self.assertTrue(metrics["is_physically_valid"])
        self.assertLessEqual(metrics["efficiency_percent"], 100.0 + 1e-8)
        self.assertAlmostEqual(metrics["water_balance_residual_m3"], 0.0, places=8)

    def test_volume_semantics_match_total_installed_capacity(self):
        simulator = PumpedHydroSimulator(self.user, self.design)
        self.assertEqual(simulator.upper_capacity, 100.0)
        self.assertEqual(simulator.lower_capacity, 100.0)
        usable = simulator.max_upper - simulator.min_upper
        expected = calculate_stored_energy(usable, self.design["head_m"])
        result = simulator.simulate([0.0], [0.0])
        actual_autonomy = result["metrics"]["autonomy_days"]
        # Zero load yields zero autonomy by definition in this implementation.
        self.assertEqual(actual_autonomy, 0.0)
        self.assertGreater(expected, 0.0)


if __name__ == "__main__":
    unittest.main()
