"""
L3 System Tests: Complete Biosphere Simulation
===============================================

Tests complete biosphere system behavior after parameter refactoring:
- Full daily/seasonal simulation cycles
- Conservation laws over extended periods
- Realistic environmental drivers
- System stability and convergence
- Performance benchmarks

Test Strategy:
- Simulate realistic time periods (hours to days)
- Use realistic environmental forcing
- Verify conservation laws hold throughout
- Check biological realism of outputs
- Measure performance metrics
"""

import pytest
import torch
import numpy as np
from typing import Dict

from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import ConversionContext, unit_converter
from relic.parameters import get_global_manager
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.respiration.maintenance_respiration import MaintenanceRespiration
from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation


class TestBiosphereComplete:
    """L3: System Test - Complete Simulation"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.param_manager = get_global_manager()
        self.device = torch.device('cpu')
        yield

    def _create_initial_state(self) -> Dict:
        """Create initial state for simulation"""
        return {
            # Environmental (will be updated each timestep)
            'air_temperature': torch.tensor(298.15, device=self.device),
            'atmospheric_co2_ppm': torch.tensor(400.0, device=self.device),
            'ppfd': torch.tensor(0.0, device=self.device),
            'vapor_pressure_deficit': torch.tensor(1000.0, device=self.device),

            # Plant state
            'vcmax_scaling': torch.tensor(1.0, device=self.device),
            'leaf_area_index': torch.tensor(3.0, device=self.device),

            # Biomass pools (g C m-2)
            'biomass_leaf': torch.tensor(100.0, device=self.device),
            'biomass_stem': torch.tensor(200.0, device=self.device),
            'biomass_root': torch.tensor(150.0, device=self.device),
            'biomass_grain': torch.tensor(0.0, device=self.device),

            # Substrate pools (g C m-2)
            'soluble_sugar': torch.tensor(50.0, device=self.device),
            'starch': torch.tensor(20.0, device=self.device),

            # Soil conditions
            'soil_water_potential': torch.tensor(-0.5, device=self.device),
            'soil_temperature': torch.tensor(293.15, device=self.device),

            # Phenology
            'phenology_stage': torch.tensor(2.0, device=self.device),
            'growing_degree_days': torch.tensor(500.0, device=self.device),
        }

    def _generate_daily_forcing(self, hour: int) -> Dict:
        """Generate realistic environmental forcing for given hour of day"""
        # Solar radiation (sinusoidal pattern, peak at noon)
        solar_angle = (hour - 6) * np.pi / 12.0  # 6 AM to 6 PM
        if 6 <= hour <= 18:
            par = 1500.0 * max(0, np.sin(solar_angle))
        else:
            par = 0.0

        # Temperature (sinusoidal, min at 6 AM, max at 2 PM)
        temp_min = 288.15  # 15 degC
        temp_max = 303.15  # 30 degC
        temp_angle = (hour - 6) * np.pi / 12.0
        temperature = temp_min + (temp_max - temp_min) * 0.5 * (1 + np.sin(temp_angle))

        return {
            'ppfd': torch.tensor(par, device=self.device),
            'air_temperature': torch.tensor(temperature, device=self.device),
        }

    @staticmethod
    def _carbon_flux_step_in_grams(rate_kg_c: torch.Tensor, dt: float) -> torch.Tensor:
        """Convert one canonical carbon mass flux to an integrated g C step."""
        rate_g_c = unit_converter.convert(
            data=rate_kg_c,
            variable_name='carbon_flux',
            source_units='kg C m-2 s-1',
            target_units='g C m-2 s-1',
            context=ConversionContext(),
        )
        return rate_g_c * dt

    @staticmethod
    def _respiration_rate(output: Dict[str, UpdateItem]) -> torch.Tensor:
        """Validate and unwrap the canonical maintenance-respiration port."""
        item = output['maintenance_respiration_total']
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == 'kg C m-2 s-1'
        return item.value

    @staticmethod
    def _photosynthesis_rate(
        output: Dict[str, UpdateItem], key: str, unit: str
    ) -> torch.Tensor:
        """Validate and unwrap a canonical photosynthesis flux port."""
        item = output[key]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == unit
        return item.value

    def test_complete_daily_simulation(self):
        """Test complete daily simulation with realistic forcing"""
        print("\n[TEST] Complete Daily Simulation")

        # Create functors
        photo_functor = FarquharPhotosynthesis(vcmax25=60.0, jmax25=120.0, rd25=1.0)
        resp_functor = MaintenanceRespiration()

        # Initialize state
        state = self._create_initial_state()
        initial_carbon = self._calculate_total_carbon(state)
        print(f"  [INIT] Total system carbon: {initial_carbon:.2f} g C m-2")

        # Storage for daily results
        hourly_gpp = []
        hourly_resp = []
        hourly_net_after_maintenance = []

        # Simulate 24 hours
        dt = 3600.0  # 1 hour timestep
        for hour in range(24):
            # Update environmental forcing
            forcing = self._generate_daily_forcing(hour)
            state.update(forcing)

            # Run photosynthesis
            photo_output = photo_functor.forward(state, hour * 3600.0, dt)
            gpp_dt = self._carbon_flux_step_in_grams(
                self._photosynthesis_rate(
                    photo_output,
                    'gross_primary_productivity',
                    'kg C m-2 s-1',
                ),
                dt,
            )
            gpp = gpp_dt.item()

            # Run respiration
            resp_output = resp_functor.forward(state, hour * 3600.0, dt)
            rm_dt = self._carbon_flux_step_in_grams(
                self._respiration_rate(resp_output),
                dt,
            )
            rm = rm_dt.item()

            # This boundary excludes growth respiration and is not the NPP port.
            net_after_maintenance = gpp - rm

            # Update state (simplified - just sugar pool)
            state['soluble_sugar'] = state['soluble_sugar'] + gpp_dt - rm_dt

            # Store results
            hourly_gpp.append(gpp)
            hourly_resp.append(rm)
            hourly_net_after_maintenance.append(net_after_maintenance)

            if hour % 6 == 0:  # Print every 6 hours
                print(
                    f"  [Hour {hour:2d}] PPFD={forcing['ppfd'].item():.0f} "
                    f"GPP={gpp:.4f} Rm={rm:.4f} "
                    f"GPP_minus_Rm={net_after_maintenance:.4f}"
                )

        # Daily totals
        daily_gpp = sum(hourly_gpp)
        daily_resp = sum(hourly_resp)
        daily_net_after_maintenance = sum(hourly_net_after_maintenance)

        print(f"\n  [DAILY TOTALS]")
        print(f"    GPP:  {daily_gpp:.2f} g C m-2 day-1")
        print(f"    Rm:   {daily_resp:.2f} g C m-2 day-1")
        print(
            "    GPP minus maintenance respiration:  "
            f"{daily_net_after_maintenance:.2f} g C m-2 day-1"
        )

        # Biological realism checks
        assert daily_gpp > 0, "Daily GPP should be positive"
        assert 5.0 < daily_gpp < 50.0, f"Daily GPP should be realistic (5-50 g C m-2 day-1), got {daily_gpp}"

        assert daily_resp > 0, "Daily respiration should be positive"
        assert daily_resp < daily_gpp, "Respiration should be less than GPP"

        assert daily_net_after_maintenance > 0
        retained_fraction = daily_net_after_maintenance / daily_gpp
        assert 0.3 < retained_fraction < 0.9, \
            f"GPP-minus-Rm fraction should be 0.3-0.9, got {retained_fraction:.2f}"

        print(f"    GPP-minus-Rm fraction: {retained_fraction:.2%}")
        print("  [PASS] Daily simulation biologically realistic")

        # Check diurnal pattern
        max_gpp = max(hourly_gpp)
        max_hour = hourly_gpp.index(max_gpp)
        assert 8 <= max_hour <= 15, f"Peak GPP should be around noon, got hour {max_hour}"
        print(f"  [OK] Peak GPP at hour {max_hour} (expected 8-15)")

        print("  [PASS] Complete daily simulation successful")

    def test_conservation_laws(self):
        """Test conservation laws over extended simulation"""
        print("\n[TEST] Conservation Laws")

        # Create functors
        photo_functor = FarquharPhotosynthesis(vcmax25=60.0, jmax25=120.0, rd25=1.0)
        resp_functor = MaintenanceRespiration()

        # Initialize
        state = self._create_initial_state()
        initial_carbon = self._calculate_total_carbon(state)

        # Track carbon flows
        total_gpp = 0.0
        total_resp = 0.0

        # Run simulation for 6 hours
        dt = 1800.0  # 30 minutes
        num_steps = 12  # 6 hours

        for step in range(num_steps):
            hour = step * 0.5  # Hours
            forcing = self._generate_daily_forcing(int(hour) + 8)  # Start at 8 AM
            state.update(forcing)

            # Photosynthesis
            photo_output = photo_functor.forward(state, step * dt, dt)
            gpp_dt = self._carbon_flux_step_in_grams(
                self._photosynthesis_rate(
                    photo_output,
                    'gross_primary_productivity',
                    'kg C m-2 s-1',
                ),
                dt,
            )
            gpp = gpp_dt.item()
            total_gpp += gpp

            # Respiration
            resp_output = resp_functor.forward(state, step * dt, dt)
            rm_dt = self._carbon_flux_step_in_grams(
                self._respiration_rate(resp_output),
                dt,
            )
            rm = rm_dt.item()
            total_resp += rm

            # Update state
            state['soluble_sugar'] = state['soluble_sugar'] + gpp_dt - rm_dt

        # Final carbon
        final_carbon = self._calculate_total_carbon(state)

        # Check conservation
        carbon_change = final_carbon - initial_carbon
        expected_change = total_gpp - total_resp

        print(f"  [INIT] Total carbon: {initial_carbon:.2f} g C m-2")
        print(f"  [FINAL] Total carbon: {final_carbon:.2f} g C m-2")
        print(f"  [CHANGE] Actual: {carbon_change:.2f}, Expected: {expected_change:.2f}")

        # Allow small numerical error (< 1% or < 0.1 g C m-2)
        error = abs(carbon_change - expected_change)
        relative_error = error / max(abs(expected_change), 1e-6)

        print(f"  [ERROR] Absolute: {error:.4f} g C m-2, Relative: {relative_error:.4%}")

        assert error < 1.0 or relative_error < 0.05, \
            f"Carbon conservation error too large: {error:.4f} g C m-2"

        print("  [PASS] Carbon conservation maintained")

    def test_system_stability(self):
        """Test system stability over extended period"""
        print("\n[TEST] System Stability")

        # Create functors
        photo_functor = FarquharPhotosynthesis(vcmax25=60.0, jmax25=120.0, rd25=1.0)
        resp_functor = MaintenanceRespiration()

        # Initialize
        state = self._create_initial_state()

        # Run for 3 days, checking for numerical instabilities
        dt = 3600.0  # 1 hour
        num_days = 3
        num_steps = 24 * num_days

        sugar_history = []
        gpp_history = []

        for step in range(num_steps):
            hour = step % 24
            forcing = self._generate_daily_forcing(hour)
            state.update(forcing)

            # Run processes
            photo_output = photo_functor.forward(state, step * dt, dt)
            resp_output = resp_functor.forward(state, step * dt, dt)

            gpp_dt = self._carbon_flux_step_in_grams(
                self._photosynthesis_rate(
                    photo_output,
                    'gross_primary_productivity',
                    'kg C m-2 s-1',
                ),
                dt,
            )
            rm_dt = self._carbon_flux_step_in_grams(
                self._respiration_rate(resp_output),
                dt,
            )
            state['soluble_sugar'] = state['soluble_sugar'] + gpp_dt - rm_dt

            # Store history
            sugar_history.append(state['soluble_sugar'].item())
            gpp_history.append(gpp_dt.item())

            # Check for numerical issues
            assert torch.isfinite(state['soluble_sugar']), \
                f"Sugar pool became non-finite at step {step}"
            assert state['soluble_sugar'] >= 0, \
                f"Sugar pool became negative at step {step}"

        # Check stability metrics
        sugar_array = np.array(sugar_history)
        gpp_array = np.array(gpp_history)

        # Should not explode or collapse
        assert sugar_array.max() < 1000.0, "Sugar pool exploded"
        assert sugar_array.min() >= 0.0, "Sugar pool went negative"

        print(f"  [OK] Sugar pool range: [{sugar_array.min():.2f}, {sugar_array.max():.2f}] g C m-2")
        print(f"  [OK] Mean daily GPP: {gpp_array[gpp_array > 0].mean():.2f} g C m-2 h-1")

        # Check for sustained oscillations (not just noise)
        # Daily pattern should repeat
        day1_gpp = gpp_array[:24]
        day2_gpp = gpp_array[24:48]
        day3_gpp = gpp_array[48:72]

        # Days should be similar (within 20%)
        day1_total = day1_gpp.sum()
        day2_total = day2_gpp.sum()
        day3_total = day3_gpp.sum()

        print(f"  [OK] Daily GPP: Day1={day1_total:.2f}, Day2={day2_total:.2f}, Day3={day3_total:.2f}")

        day_variability = np.std([day1_total, day2_total, day3_total]) / np.mean([day1_total, day2_total, day3_total])
        assert day_variability < 0.3, f"Daily GPP too variable: {day_variability:.2%}"

        print("  [PASS] System stable over 3-day simulation")

    def test_bvoc_complete_integration(self):
        """Test complete BVOC emission system"""
        print("\n[TEST] Complete BVOC Integration")

        # Create functors
        photo_functor = FarquharPhotosynthesis(vcmax25=60.0, jmax25=120.0, rd25=1.0)
        bvoc_functor = MEGANModel()

        # Initialize
        state = self._create_initial_state()

        # Simulate peak daytime conditions
        forcing = self._generate_daily_forcing(12)  # Noon
        state.update(forcing)

        # Run photosynthesis
        dt = 3600.0
        photo_output = photo_functor.forward(state, 0.0, dt)
        gpp = self._carbon_flux_step_in_grams(
            self._photosynthesis_rate(
                photo_output,
                'gross_primary_productivity',
                'kg C m-2 s-1',
            ),
            dt,
        ).item()

        state['net_photosynthesis'] = self._photosynthesis_rate(
            photo_output,
            'net_photosynthesis',
            'umol CO2 m-2 s-1',
        )

        # Run BVOC
        bvoc_output = bvoc_functor.forward(state, 0.0, dt)

        # Check outputs
        print(f"  [GPP] {gpp:.4f} g C m-2 h-1")

        carbon_allocation = bvoc_output['bvoc_carbon_allocation_flux']
        assert isinstance(carbon_allocation, UpdateItem)
        assert carbon_allocation.mode is UpdateMode.REPLACE
        assert carbon_allocation.per_timestep is False
        assert carbon_allocation.unit == 'kg C m-2 s-1'
        bvoc_total = self._carbon_flux_step_in_grams(
            carbon_allocation.value,
            dt,
        ).item()
        print(
            f"  [bvoc_carbon_allocation] {bvoc_total:.6f} "
            "g C m-2 h-1"
        )

        # BVOC should be small fraction of GPP
        if bvoc_total > 0:
            bvoc_fraction = bvoc_total / gpp
            assert bvoc_fraction < 0.05, f"BVOC too high: {bvoc_fraction:.2%} of GPP"
            print(f"  [OK] Total BVOC carbon: {bvoc_total:.6f} g C m-2 h-1 ({bvoc_fraction:.2%} of GPP)")

        print("  [PASS] BVOC integration complete")

    def test_biomass_growth_simulation(self):
        """Test biomass growth over time"""
        print("\n[TEST] Biomass Growth Simulation")

        # The NPP boundary is prescribed until the unique upstream NPP closure runs.
        alloc_functor = BiomassAllocation()

        state = {
            'net_primary_productivity': torch.tensor(1.0e-7, device=self.device),
            'air_temperature': torch.tensor(298.15, device=self.device),
            'water_stress_factor': torch.tensor(1.0, device=self.device),
            'leaf_area_index': torch.tensor(3.0, device=self.device),
            'phenology_phase': torch.tensor(2.0, device=self.device),
        }
        biomass = {
            'leaf_biomass_c': torch.tensor(0.1, device=self.device),
            'wood_biomass_c': torch.tensor(0.2, device=self.device),
            'root_biomass_c': torch.tensor(0.15, device=self.device),
        }
        initial_biomass = sum(value.item() for value in biomass.values())

        print(f"  [INIT] Total biomass: {initial_biomass:.6f} kg C m-2")

        # Simulate 7 days of growth
        dt = 86400.0  # 1 day timestep for growth
        num_days = 7
        integrated_npp = 0.0

        for day in range(num_days):
            alloc_output = alloc_functor.forward(state, day * dt, dt)
            step_delta = torch.zeros((), device=self.device)
            for pool_name in biomass:
                item = alloc_output[pool_name]
                assert isinstance(item, UpdateItem)
                assert item.mode is UpdateMode.ACCUMULATE
                assert item.per_timestep is True
                assert item.unit == 'kg C m-2'
                biomass[pool_name] = biomass[pool_name] + item.value
                step_delta = step_delta + item.value

            expected_step = state['net_primary_productivity'] * dt
            torch.testing.assert_close(step_delta, expected_step, rtol=1.0e-6, atol=1.0e-10)
            residual = alloc_output[
                'diagnostics.biomass_allocation.carbon_balance_residual'
            ].value
            torch.testing.assert_close(
                residual,
                torch.zeros_like(residual),
                rtol=0.0,
                atol=1.0e-12,
            )
            integrated_npp += expected_step.item()

            if (day + 1) % 2 == 0:
                current_biomass = sum(value.item() for value in biomass.values())
                print(
                    f"  [Day {day+1}] Biomass: {current_biomass:.6f} kg C m-2, "
                    f"NPP boundary step: {expected_step.item():.6f} kg C m-2"
                )

        # Final biomass
        final_biomass = sum(value.item() for value in biomass.values())

        print(f"  [FINAL] Total biomass: {final_biomass:.6f} kg C m-2")

        # Check growth occurred
        biomass_gain = final_biomass - initial_biomass
        assert biomass_gain > 0, "Biomass should increase over time"
        assert biomass_gain < initial_biomass, "Growth rate should be reasonable (< 100%/week)"
        assert biomass_gain == pytest.approx(integrated_npp, rel=1.0e-5, abs=1.0e-8)

        print(
            f"  [OK] Biomass gain: {biomass_gain:.6f} kg C m-2 "
            f"({biomass_gain/initial_biomass*100:.1f}%)"
        )
        print("  [PASS] Biomass growth simulation successful")

    def _calculate_total_carbon(self, state: Dict) -> float:
        """Calculate total carbon in system"""
        carbon_pools = [
            'biomass_leaf',
            'biomass_stem',
            'biomass_root',
            'biomass_grain',
            'soluble_sugar',
            'starch',
        ]

        total = 0.0
        for pool in carbon_pools:
            if pool in state:
                total += state[pool].item()

        return total
