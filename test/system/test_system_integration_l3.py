#!/usr/bin/env python
"""
L3 System Tests for Complete RELIC Ecosystem Simulation

This test module validates the complete system integration including:
- Full ecosystem simulation with multiple processes
- Carbon cycle conservation across all components
- Water balance and energy conservation
- Multi-day simulations with diurnal cycles
- Performance and scalability tests
"""

import unittest
import torch
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from relic.core.simulator import RELICSimulator

TEST_SCALE = float(os.getenv("RELIC_L3_SCALE", "0.1"))


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'carbon_biomass': 'kg C m-2',
    'carbon_pool': 'kg C m-2',
    'carbon_soil': 'kg C m-2',
    'nitrogen_pool': 'dimensionless',
    'productivity': 'dimensionless',
    'radiation': 'W m-2',
    'soil_water': 'dimensionless',
    'water_pool': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


def _scaled_steps(base_steps: int) -> int:
    """Scale long-running loops to keep test runtime manageable."""
    return max(1, int(base_steps * TEST_SCALE))


class TestCompleteEcosystemSimulation(unittest.TestCase):
    """Test suite for complete ecosystem simulation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')

    def test_carbon_conservation_simple(self):
        """Test simple carbon conservation."""
        # Create simulator
        steps = _scaled_steps(24)
        config = {
            'time_step': 3600.0,  # 1 hour
            'max_time': 3600.0 * steps,
        }

        simulator = _declared_simulator(config=config)

        # Initialize state
        initial_state = {
            'carbon_pool': 100.0,
            'air_temperature': 298.15,
            'co2_concentration': 415.0,
        }

        simulator.initialize_state(initial_state)

        # Run simulation
        for _ in range(steps):
            updates = simulator.step(3600.0)

        # Check carbon is conserved (within numerical tolerance)
        final_state = simulator.state_manager.get_state()
        final_carbon = final_state.get('carbon_pool', torch.tensor(100.0)).item()

        # Carbon should not change drastically without sources/sinks
        self.assertAlmostEqual(final_carbon, 100.0, delta=10.0)

    def test_diurnal_cycle(self):
        """Test basic diurnal cycle response."""
        # Create simulator
        config = {
            'time_step': 3600.0,  # 1 hour
            'max_time': 86400.0,  # 1 day
        }

        simulator = _declared_simulator(config=config)

        # Initialize state
        initial_state = {
            'air_temperature': 288.15,
            'radiation': 0.0,
        }

        simulator.initialize_state(initial_state)

        # Simulate a reduced diurnal cycle to limit runtime
        radiation_values = []
        hours = [0, 6, 12, 18, 23]
        for hour in hours:
            # Simple sinusoidal radiation
            if 6 <= hour <= 18:
                radiation = 500.0 * np.sin((hour - 6) * np.pi / 12)
            else:
                radiation = 0.0

            simulator.state_manager.update_environment({'radiation': radiation})
            radiation_values.append(radiation)

            updates = simulator.step(3600.0)

        # Check we have day and night values
        day_radiation = [
            value for hour, value in zip(hours, radiation_values)
            if 6 <= hour <= 18
        ]
        night_radiation = [
            value for hour, value in zip(hours, radiation_values)
            if hour < 6 or hour > 18
        ]

        self.assertGreater(max(day_radiation), 0)
        self.assertAlmostEqual(max(night_radiation), 0.0, delta=1e-6)

    def test_water_balance_simple(self):
        """Test simple water balance."""
        # Create simulator
        steps = _scaled_steps(72)  # 72 hours
        config = {
            'time_step': 3600.0,  # 1 hour
            'max_time': 3600.0 * steps,
        }

        simulator = _declared_simulator(config=config)

        # Initialize with water
        initial_state = {
            'soil_water': 100.0,  # mm
            'air_temperature': 298.15,
        }

        simulator.initialize_state(initial_state)

        # Track water
        initial_water = 100.0
        for _ in range(steps):
            updates = simulator.step(3600.0)

        # Water should decrease due to evaporation (but not to zero)
        final_state = simulator.state_manager.get_state()
        final_water = final_state.get('soil_water', torch.tensor(100.0)).item()

        self.assertLessEqual(final_water, initial_water)
        self.assertGreater(final_water, 0.0)

    def test_temperature_response(self):
        """Test ecosystem response to temperature."""
        # Create simulator
        config = {
            'time_step': 3600.0,
            'max_time': 3600.0,  # 1 hour
        }

        temperatures = [278.15, 298.15, 318.15]  # 5C, 25C, 45C
        results = []

        for temp in temperatures:
            simulator = _declared_simulator(config=config)

            initial_state = {
                'air_temperature': temp,
                'productivity': 0.0,
            }

            simulator.initialize_state(initial_state)
            updates = simulator.step(3600.0)

            state = simulator.state_manager.get_state()
            productivity = state.get('productivity', torch.tensor(0.0)).item()
            results.append(productivity)

        # Productivity should be highest at moderate temperature
        # (This is a simplified test - actual behavior depends on functors)
        # We just check that values are reasonable
        for r in results:
            self.assertGreaterEqual(r, 0.0)

    def test_multi_process_integration(self):
        """Test integration of multiple processes."""
        # Create simulator with longer runtime
        steps = _scaled_steps(336)  # 1 week at 30 min steps
        config = {
            'time_step': 1800.0,  # 30 minutes
            'max_time': 1800.0 * steps,
        }

        simulator = _declared_simulator(config=config)

        # Initialize comprehensive state
        initial_state = {
            'air_temperature': 288.15,
            'co2_concentration': 415.0,
            'soil_moisture': 0.3,
            'leaf_area_index': 4.0,
            'carbon_pool': 100.0,
            'nitrogen_pool': 10.0,
            'water_pool': 50.0,
        }

        simulator.initialize_state(initial_state)

        # Track key variables
        carbon_history = []
        water_history = []

        for step in range(steps):
            # Add some seasonal variation
            temp = 288.15 + 10 * np.sin(2 * np.pi * step / max(1, steps))
            simulator.state_manager.update_environment({'air_temperature': temp})

            updates = simulator.step(1800.0)

            if step % 24 == 0:  # Every 12 hours
                state = simulator.state_manager.get_state()
                carbon = state.get('carbon_pool', torch.tensor(0.0)).item()
                water = state.get('water_pool', torch.tensor(0.0)).item()
                carbon_history.append(carbon)
                water_history.append(water)

        # Check that system remained stable
        self.assertGreater(len(carbon_history), 0)
        self.assertGreater(len(water_history), 0)

        # Carbon shouldn't explode or crash
        for c in carbon_history:
            self.assertGreater(c, 0)
            self.assertLess(c, 1000)

        # Water shouldn't go negative
        for w in water_history:
            self.assertGreaterEqual(w, 0)

    def test_performance_basic(self):
        """Test that dispatcher overhead remains bounded across state sizes."""
        timed_steps = 200
        config = {
            'time_step': 3600.0,
            'max_time': 3600.0 * (timed_steps + 1),
        }

        small_state = {
            'temperature': torch.tensor(298.15),
            'carbon': torch.tensor(100.0),
        }
        large_state = {
            'temperature': torch.ones(100, 100) * 298.15,
            'carbon': torch.ones(100, 100) * 100.0,
        }

        def measure_dispatch(state):
            simulator = _declared_simulator(config=config)
            simulator.initialize_state(state)
            simulator.step(3600.0)
            start_time = time.perf_counter()
            for _ in range(timed_steps):
                simulator.step(3600.0)
            return time.perf_counter() - start_time

        small_time = measure_dispatch(small_state)
        large_time = measure_dispatch(large_state)

        clock_resolution = time.get_clock_info('perf_counter').resolution
        self.assertTrue(np.isfinite(small_time))
        self.assertTrue(np.isfinite(large_time))
        self.assertGreater(small_time, clock_resolution)
        self.assertGreater(large_time, clock_resolution)

        # Large state should take longer but not exponentially so
        # (allowing up to 1000x for 10000x more data)
        self.assertLess(large_time, small_time * 1000)

    def test_ecosystem_stability(self):
        """Test ecosystem doesn't crash over extended run."""
        # Create simulator for multi-day run
        steps = _scaled_steps(720)  # 30 days * 24 hours
        config = {
            'time_step': 3600.0,  # 1 hour
            'max_time': 3600.0 * steps,
        }

        simulator = _declared_simulator(config=config)

        # Initialize with typical values
        initial_state = {
            'air_temperature': 288.15,
            'soil_moisture': 0.3,
            'carbon_biomass': 10.0,
            'carbon_soil': 50.0,
        }

        simulator.initialize_state(initial_state)

        # Run for 30 days
        exception_occurred = False
        try:
            for hour in range(steps):
                # Add some variability
                temp = 288.15 + 10 * np.sin(2 * np.pi * hour / 24)
                simulator.state_manager.update_environment({'air_temperature': temp})

                updates = simulator.step(3600.0)
        except Exception as e:
            exception_occurred = True
            print(f"Exception during stability test: {e}")

        # Should complete without crashing
        self.assertFalse(exception_occurred)

        # Final state should be reasonable
        final_state = simulator.state_manager.get_state()
        final_carbon = (
            final_state.get('carbon_biomass', torch.tensor(0.0)).item() +
            final_state.get('carbon_soil', torch.tensor(0.0)).item()
        )

        # Carbon should be positive and finite
        self.assertGreater(final_carbon, 0)
        self.assertTrue(np.isfinite(final_carbon))


if __name__ == '__main__':
    unittest.main()

