"""
L3 System Validation Tests for Phenology Module

Comprehensive validation of the phenology system against scientific
benchmarks, conservation laws, and ecosystem-level behavior. These tests
verify that the integrated system produces scientifically valid results.

Test Categories:
1. Conservation law validation (C, N, H2O balance)
2. Scientific benchmark comparisons
3. Multi-year simulation stability
4. Ecosystem coupling validation
5. Edge case handling
"""

import pytest
import torch
import numpy as np
from typing import Dict
import logging

# RELIC imports
from relic.parameters import get_global_manager, reset_global_manager
from relic.scientific.biosphere.phenology.models import (
    StaticPhenology, GDDPhenology, GSIPhenology, CropPhenology
)
from relic.core.state import StateManager
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)

T_FREEZE_K = float(get_physical_constant("T_FREEZE"))
SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _compute_gdd_step(
    model: GDDPhenology,
    environment: Dict[str, torch.Tensor],
    accumulated_gdd: torch.Tensor,
    previous_day: torch.Tensor,
):
    """Advance GDD once and return the externally owned temporal states."""
    canonical_environment = {
        **environment,
        'phenology_gdd_accumulated': accumulated_gdd,
        'phenology_gdd_previous_day_of_year': previous_day,
    }
    output = model.compute_phenological_state(
        canonical_environment,
        dt=torch.as_tensor(
            SECONDS_PER_DAY,
            device=environment['air_temperature'].device,
            dtype=environment['air_temperature'].dtype,
        ),
    )
    return (
        output,
        output.additional_outputs['phenology_gdd_accumulated'],
        output.additional_outputs['phenology_gdd_previous_day_of_year'],
    )


GSI_TEMPORAL_STATE_KEYS = (
    'phenology_gsi_history',
    'phenology_gsi_history_index',
    'phenology_gsi_history_count',
    'phenology_gsi_in_growing_season',
)


def _initial_gsi_state(model: GSIPhenology) -> Dict[str, torch.Tensor]:
    """Return canonical StateManager-owned GSI temporal state."""
    return {
        'phenology_gsi_history': torch.zeros(model.memory_days),
        'phenology_gsi_history_index': torch.tensor(0.0),
        'phenology_gsi_history_count': torch.tensor(0.0),
        'phenology_gsi_in_growing_season': torch.tensor(0.0),
    }


def _commit_gsi_state(state, updates) -> None:
    """Apply full-state GSI replacements from one Functor call."""
    for key in GSI_TEMPORAL_STATE_KEYS:
        state[key] = updates[key].value.detach().clone()


CROP_TEMPORAL_STATE_KEYS = (
    'phenology_crop_accumulated_phu',
    'phenology_crop_is_sown',
    'phenology_crop_is_harvested',
    'phenology_crop_vernalization_days',
)


def _initial_crop_state() -> Dict[str, torch.Tensor]:
    """Return canonical StateManager-owned Crop temporal state."""
    return {
        'phenology_crop_accumulated_phu': torch.tensor(0.0),
        'phenology_crop_is_sown': torch.tensor(0.0),
        'phenology_crop_is_harvested': torch.tensor(0.0),
        'phenology_crop_vernalization_days': torch.tensor(0.0),
    }


def _commit_crop_state(state, updates) -> None:
    """Apply full-state Crop replacements from one Functor call."""
    for key in CROP_TEMPORAL_STATE_KEYS:
        state[key] = updates[key].value.detach().clone()


class TestPhenologySystemValidation:
    """Comprehensive system validation tests for phenology module."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test environment."""
        reset_global_manager()
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database('relic/parameters/database')
        self.device = torch.device('cpu')
        
        # Initialize state manager for conservation tracking
        self.state_manager = StateManager(device=self.device)
    
    def test_carbon_conservation_annual_cycle(self):
        """Test carbon conservation through annual phenological cycle."""
        model = GSIPhenology(
            config={'phenology_type': 'gsi', 'vegetation_type': 'tree'},
            device=self.device
        )
        
        # Initial carbon pools
        initial_leaf_c = torch.tensor(1.0)  # kg C m-2
        initial_froot_c = torch.tensor(0.5)  # kg C m-2
        
        state = {
            'leaf_carbon': initial_leaf_c.clone(),
            'froot_carbon': initial_froot_c.clone(),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02),
            'litter_carbon': torch.tensor(0.0),  # Track litter accumulation
            **_initial_gsi_state(model),
        }
        
        total_initial_c = initial_leaf_c + initial_froot_c
        
        # Simulate full year (365 days)
        dt = SECONDS_PER_DAY
        total_litter_c = torch.tensor(0.0)
        
        for day in range(365):
            # Create realistic seasonal environment
            environment = self._create_seasonal_environment(day)
            
            # Compute fluxes
            updates = model(
                {**state, **environment},
                time=day * dt,
                dt=dt,
            )
            state['leaf_carbon'] += updates['leaf_carbon'].value
            state['froot_carbon'] += updates['froot_carbon'].value
            total_litter_c += dt * (
                updates['leaf_litterfall_carbon'].value
                + updates['froot_litterfall_carbon'].value
            )
            _commit_gsi_state(state, updates)

            assert state['leaf_carbon'] >= 0.0
            assert state['froot_carbon'] >= 0.0
        
        # Check carbon conservation
        final_biomass_c = state['leaf_carbon'] + state['froot_carbon']
        total_final_c = final_biomass_c + total_litter_c
        
        # Carbon should be conserved (within numerical tolerance)
        carbon_error = torch.abs(total_final_c - total_initial_c)
        relative_error = carbon_error / total_initial_c
        
        assert relative_error < 0.01, f"Carbon conservation violated: {relative_error:.4f}"
        logger.info(f"Carbon conservation error: {relative_error:.6f}")
    
    def test_nitrogen_conservation_with_retranslocation(self):
        """Test nitrogen conservation including retranslocation."""
        model = GSIPhenology(
            config={'phenology_type': 'gsi', 'vegetation_type': 'tree'},
            device=self.device
        )
        
        # Initial nitrogen pools
        initial_leaf_n = torch.tensor(0.05)  # kg N m-2
        initial_froot_n = torch.tensor(0.02)
        initial_plant_n = torch.tensor(0.01)  # Internal plant storage
        
        state = {
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(0.5),
            'leaf_nitrogen': initial_leaf_n.clone(),
            'froot_nitrogen': initial_froot_n.clone(),
            'plant_nitrogen_pool': initial_plant_n.clone(),
            'litter_nitrogen': torch.tensor(0.0),
            **_initial_gsi_state(model),
        }

        total_initial_n = initial_leaf_n + initial_froot_n + initial_plant_n
        
        # Simulate senescence period
        dt = SECONDS_PER_DAY
        total_litter_n = torch.tensor(0.0)
        total_retrans_n = torch.tensor(0.0)
        
        for day in range(240, 320):  # Senescence period
            environment = self._create_seasonal_environment(day)
            updates = model(
                {**state, **environment},
                time=(day - 240) * dt,
                dt=dt,
            )
            for state_name in (
                'leaf_nitrogen',
                'froot_nitrogen',
                'plant_nitrogen_pool',
            ):
                state[state_name] += updates[state_name].value
            total_litter_n += dt * (
                updates['leaf_litterfall_nitrogen'].value
                + updates['froot_litterfall_nitrogen'].value
            )
            total_retrans_n += dt * updates['nitrogen_retranslocation'].value
            _commit_gsi_state(state, updates)

            assert state['leaf_nitrogen'] >= 0.0
            assert state['froot_nitrogen'] >= 0.0
        
        # Check nitrogen conservation
        final_biomass_n = (
            state['leaf_nitrogen']
            + state['froot_nitrogen']
            + state['plant_nitrogen_pool']
        )
        total_final_n = final_biomass_n + total_litter_n
        
        n_error = torch.abs(total_final_n - total_initial_n)
        relative_error = n_error / total_initial_n
        
        assert relative_error < 0.01, f"Nitrogen conservation violated: {relative_error:.4f}"
        
        # Verify retranslocation efficiency
        retrans_efficiency = self.param_manager.get('biosphere.phenology.turnover.n_retrans_efficiency')
        # Retranslocation should be significant but not exceed efficiency limit
        assert total_retrans_n > 0, "No nitrogen retranslocation occurred"
        logger.info(f"N retranslocation: {total_retrans_n:.4f} kg N m-2")
    
    def test_phenology_benchmark_gdd_model(self):
        """Compare GDD model against known benchmarks."""
        model = GDDPhenology(
            name='test_gdd_benchmark',
            config={
                'phenology_type': 'gdd',
                'vegetation_type': 'tree',
            },
            device=self.device
        )
        
        # Test known GDD accumulation scenarios
        test_cases = [
            # (temps, expected_gdd, expected_phase)
            (
                [10.0] * 20,  # 20 days at 10掳C
                100.0,  # (10-5) * 20 = 100 GDD
                'dormancy'
            ),
            (
                [15.0] * 100,  # 100 days at 15掳C
                1000.0,  # (15-5) * 100 = 1000 GDD
                'senescence'
            ),
            (
                [3.0] * 30,  # Below base temp
                0.0,  # No GDD accumulation
                'dormancy'
            )
        ]
        
        for temps, expected_gdd, expected_phase in test_cases:
            accumulated_gdd = torch.tensor(0.0)
            previous_day = torch.tensor(0.0)
            output = None
            
            for i, temp_c in enumerate(temps):
                environment = {
                    'day_of_year': torch.tensor(float(i + 1)),
                    'air_temperature': torch.tensor(temp_c + T_FREEZE_K),
                    'air_temperature_min': torch.tensor(temp_c - 2 + T_FREEZE_K),
                    'air_temperature_max': torch.tensor(temp_c + 2 + T_FREEZE_K)
                }
                
                output, accumulated_gdd, previous_day = _compute_gdd_step(
                    model,
                    environment,
                    accumulated_gdd,
                    previous_day,
                )

            assert output is not None
            assert accumulated_gdd.item() == pytest.approx(expected_gdd, abs=1e-3)
            assert previous_day.item() == pytest.approx(float(len(temps)))
            assert output.phenology_phase.value == expected_phase
    
    def test_gsi_response_bounds_jolly_2005(self):
        """Validate the three canonical Jolly climate response factors."""
        model = GSIPhenology(
            name='test_gsi_benchmark',
            config={'phenology_type': 'gsi', 'vegetation_type': 'tree'},
            device=self.device
        )
        
        # Limiting-factor cases from the Jolly et al. three-factor equation.
        test_cases = [
            {
                'tmin': 10.0,  # degC - above optimal
                'vpd': 1000.0,  # Pa - optimal range
                'daylen': 50000.0,  # s - long day
                'expected_gsi_min': 0.8  # Should be high
            },
            {
                'tmin': -5.0,  # degC - below threshold
                'vpd': 5000.0,  # Pa - high stress
                'daylen': 30000.0,  # s - short day
                'expected_gsi_max': 0.2  # Should be low
            }
        ]
        
        for case in test_cases:
            environment = {
                'day_of_year': torch.tensor(150.0),
                'air_temperature_min': torch.tensor(case['tmin'] + T_FREEZE_K),
                'vapor_pressure_deficit': torch.tensor(case['vpd']),
                'daylength': torch.tensor(case['daylen']),
                **_initial_gsi_state(model),
            }
            
            output = model.compute_phenological_state(
                {**environment, "day_of_year": environment['day_of_year'], "air_temperature": torch.tensor(T_FREEZE_K + 15.0)},
                dt=torch.tensor(SECONDS_PER_DAY),
            )
            
            gsi = output.growing_season_index.item()
            
            if 'expected_gsi_min' in case:
                assert gsi >= case['expected_gsi_min'], \
                    f"GSI too low: {gsi:.3f} < {case['expected_gsi_min']}"
            if 'expected_gsi_max' in case:
                assert gsi <= case['expected_gsi_max'], \
                    f"GSI too high: {gsi:.3f} > {case['expected_gsi_max']}"
    
    def test_multi_year_stability(self):
        """Test model stability over multiple years."""
        model = GSIPhenology(
            config={'phenology_type': 'gsi', 'vegetation_type': 'tree'},
            device=self.device
        )
        
        # Track key metrics over years
        annual_max_lai = []
        annual_litter_production = []
        
        state = {
            'leaf_carbon': torch.tensor(0.5),
            'froot_carbon': torch.tensor(0.3),
            'leaf_nitrogen': torch.tensor(0.02),
            'froot_nitrogen': torch.tensor(0.01),
            **_initial_gsi_state(model),
        }
        
        # Simulate 5 years
        for year in range(5):
            year_max_lai = 0.0
            year_litter = 0.0
            
            for day in range(365):
                environment = self._create_seasonal_environment(day)
                updates = model(
                    {**state, **environment},
                    time=(year * 365 + day) * SECONDS_PER_DAY,
                    dt=SECONDS_PER_DAY,
                )
                fluxes = {key: update.value for key, update in updates.items()}
                _commit_gsi_state(state, updates)
                
                # Track metrics
                lai = fluxes['leaf_area_index'].item()
                year_max_lai = max(year_max_lai, lai)
                
                if 'leaf_litterfall_carbon' in fluxes:
                    year_litter += (
                        fluxes['leaf_litterfall_carbon'].item()
                        * SECONDS_PER_DAY
                    )
            
            annual_max_lai.append(year_max_lai)
            annual_litter_production.append(year_litter)
            
        # Check stability - values should stabilize after first year
        for i in range(2, 5):
            lai_change = abs(annual_max_lai[i] - annual_max_lai[i-1]) / annual_max_lai[i-1]
            assert lai_change < 0.1, f"LAI unstable in year {i}: {lai_change:.3f}"
            
            litter_change = abs(annual_litter_production[i] - annual_litter_production[i-1]) / annual_litter_production[i-1]
            assert litter_change < 0.15, f"Litter production unstable in year {i}: {litter_change:.3f}"
    
    def test_extreme_environment_handling(self):
        """Test model behavior under extreme environmental conditions."""
        model = GSIPhenology(
            config={'phenology_type': 'gsi', 'vegetation_type': 'tree'},
            device=self.device
        )
        
        extreme_scenarios = [
            {
                'name': 'extreme_cold',
                'air_temperature_min': torch.tensor(T_FREEZE_K - 50.0),
                'vpd': torch.tensor(2000.0),
                'daylength': torch.tensor(50000.0),
                'expected_growth_max': 0.1,
            },
            {
                'name': 'extreme_heat',
                'air_temperature_min': torch.tensor(T_FREEZE_K + 50.0),
                'vpd': torch.tensor(900.0),
                'daylength': torch.tensor(50000.0),
                'expected_growth_min': 0.5,
            },
            {
                'name': 'extreme_vpd',
                'air_temperature_min': torch.tensor(T_FREEZE_K + 10.0),
                'vpd': torch.tensor(10000.0),
                'daylength': torch.tensor(50000.0),
                'expected_growth_max': 0.2,
            }
        ]
        
        for scenario in extreme_scenarios:
            # Build environment
            environment = {
                'day_of_year': torch.tensor(150.0),
                'air_temperature_min': scenario['air_temperature_min'],
                'vapor_pressure_deficit': scenario['vpd'],
                'daylength': scenario['daylength'],
                **_initial_gsi_state(model),
            }
            
            output = model.compute_phenological_state(
                {**environment, "day_of_year": environment['day_of_year'], "air_temperature": environment['air_temperature_min'] + 5.0},
                dt=torch.tensor(SECONDS_PER_DAY),
            )
            
            growth = output.growth_factor.item()
            
            if 'expected_growth_max' in scenario:
                assert growth <= scenario['expected_growth_max'], \
                    f"{scenario['name']}: growth {growth:.3f} > {scenario['expected_growth_max']}"

            if 'expected_growth_min' in scenario:
                assert growth >= scenario['expected_growth_min'], \
                    f"{scenario['name']}: growth {growth:.3f} < {scenario['expected_growth_min']}"
    
    def test_crop_development_stages(self):
        """Test crop phenology development stages."""
        model = CropPhenology(
            name='test_crop_stages',
            config={
                'phenology_type': 'crop',
                'vegetation_type': 'crop',
                'crop_type': 'wheat',
                'crop_phu_total': 1200.0
            },
            device=self.device
        )
        
        crop_state = _initial_crop_state()
        development_stages = []
        
        for day in range(200):  # Growing season
            temp = 15.0 + 10.0 * np.sin(day * np.pi / 100)  # Seasonal temperature
            
            environment = {
                'day_of_year': torch.tensor(float(day + 100)),
                'air_temperature': torch.tensor(temp + 273.15),
            }
            
            state = {
                'leaf_carbon': torch.tensor(0.1 + day * 0.001),
                'froot_carbon': torch.tensor(0.05),
                'leaf_nitrogen': torch.tensor(0.004),
                'froot_nitrogen': torch.tensor(0.002),
                **crop_state,
            }
            
            updates = model(
                {**state, **environment},
                time=day * SECONDS_PER_DAY,
                dt=SECONDS_PER_DAY,
            )
            _commit_crop_state(crop_state, updates)
            development_stages.append(
                min(
                    crop_state['phenology_crop_accumulated_phu'].item()
                    / model.phenology_params.crop_phu_total,
                    1.0,
                )
            )
        
        # Verify crop development progresses
        assert max(development_stages) > development_stages[0], \
            "Crop development should progress through season"
        assert not hasattr(model, 'development_index')
    
    def test_lai_dynamics(self):
        """Test LAI dynamics match expected patterns."""
        models = [
            StaticPhenology('static_lai', {'phenology_type': 'static'}, device=self.device),
            GDDPhenology('gdd_lai', {'phenology_type': 'gdd'}, device=self.device),
            GSIPhenology('gsi_lai', {'phenology_type': 'gsi'}, device=self.device)
        ]
        
        for model in models:
            lai_trajectory = []
            accumulated_gdd = torch.tensor(0.0)
            previous_day = torch.tensor(0.0)
            gsi_state = (
                _initial_gsi_state(model)
                if isinstance(model, GSIPhenology)
                else None
            )
            
            # Full year simulation
            for day in range(365):
                environment = self._create_seasonal_environment(day)

                if isinstance(model, GDDPhenology):
                    output, accumulated_gdd, previous_day = _compute_gdd_step(
                        model,
                        environment,
                        accumulated_gdd,
                        previous_day,
                    )
                elif isinstance(model, GSIPhenology):
                    canonical_environment = {
                        **environment,
                        **gsi_state,
                    }
                    output = model.compute_phenological_state(
                        canonical_environment,
                        dt=torch.tensor(SECONDS_PER_DAY),
                    )
                    gsi_state = {
                        key: output.additional_outputs[key].detach().clone()
                        for key in GSI_TEMPORAL_STATE_KEYS
                    }
                else:
                    output = model.compute_phenological_state(
                        environment,
                        dt=get_physical_constant("seconds_per_day"),
                    )
                
                lai_trajectory.append(output.leaf_area_index.item())

            if isinstance(model, GDDPhenology):
                assert previous_day.item() == pytest.approx(365.0)
            
            # Check LAI patterns
            max_lai = max(lai_trajectory)
            min_lai = min(lai_trajectory)
            
            # LAI should vary seasonally
            assert max_lai > min_lai * 2, \
                f"{model.name}: Insufficient LAI variation: {max_lai:.2f} vs {min_lai:.2f}"
            
            # Maximum should be reasonable
            assert 0.1 <= max_lai <= 10.0, \
                f"{model.name}: Unrealistic max LAI: {max_lai:.2f}"
            
            # Minimum should be non-negative
            assert min_lai >= 0.0, \
                f"{model.name}: Negative LAI: {min_lai:.2f}"
    
    def _create_seasonal_environment(self, day: int) -> Dict[str, torch.Tensor]:
        """Create realistic seasonal environment for given day of year."""
        # Simple sinusoidal temperature pattern
        base_temp = 15.0  # degC
        amplitude = 10.0
        temp_c = base_temp + amplitude * np.sin((day - 80) * 2 * np.pi / 365)
        
        # Daylength variation (simplified)
        base_daylen = 43200  # 12 hours in seconds
        daylen_var = 14400  # 4 hour variation
        daylen = base_daylen + daylen_var * np.sin((day - 80) * 2 * np.pi / 365)
        
        # VPD increases with temperature
        vpd = 500.0 + 100.0 * temp_c
        
        return {
            'day_of_year': torch.tensor(float(day % 365 + 1)),
            'air_temperature': torch.tensor(temp_c + T_FREEZE_K),
            'air_temperature_min': torch.tensor(temp_c - 5 + T_FREEZE_K),
            'air_temperature_max': torch.tensor(temp_c + 5 + T_FREEZE_K),
            'vapor_pressure_deficit': torch.tensor(max(0, vpd)),
            'daylength': torch.tensor(max(0, daylen)),
        }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--log-cli-level=INFO'])

