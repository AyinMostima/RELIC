"""
Integration tests for phenology module parameter management fixes.

Tests verify:
1. All hardcoded constants are replaced with ParameterManager calls
2. Device binding is consistent
3. PFT integration works correctly
4. Conservation laws are enforced
5. Dependencies are properly declared
"""

import pytest
import torch
import numpy as np

from relic.scientific.biosphere.phenology.models import (
    GDDPhenology, GSIPhenology, CropPhenology
)
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant


def _gsi_temporal_state(model, device):
    """Return canonical StateManager-owned GSI temporal inputs."""
    return {
        'phenology_gsi_history': torch.zeros(model.memory_days, device=device),
        'phenology_gsi_history_index': torch.tensor(0.0, device=device),
        'phenology_gsi_history_count': torch.tensor(0.0, device=device),
        'phenology_gsi_in_growing_season': torch.tensor(0.0, device=device),
    }


def _crop_temporal_state(
    device,
    *,
    accumulated_phu=0.0,
    is_sown=False,
    is_harvested=False,
    vernalization_days=0.0,
):
    """Return canonical StateManager-owned Crop temporal inputs."""
    return {
        'phenology_crop_accumulated_phu': torch.tensor(
            accumulated_phu,
            device=device,
        ),
        'phenology_crop_is_sown': torch.tensor(
            float(is_sown),
            device=device,
        ),
        'phenology_crop_is_harvested': torch.tensor(
            float(is_harvested),
            device=device,
        ),
        'phenology_crop_vernalization_days': torch.tensor(
            vernalization_days,
            device=device,
        ),
    }


CROP_TEMPORAL_STATE_KEYS = (
    'phenology_crop_accumulated_phu',
    'phenology_crop_is_sown',
    'phenology_crop_is_harvested',
    'phenology_crop_vernalization_days',
)


class TestPhenologyParameterIntegration:
    """Test suite for phenology parameter management integration."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.param_manager = get_global_manager()
        
        # Basic config
        self.config = {
            'device': self.device,
            'conservation_tolerance': 1e-8
        }
        
        # Mock environment data
        self.environment = {
            'day_of_year': torch.tensor(150.0, device=self.device),
            'air_temperature': torch.tensor(293.15, device=self.device),  # 20 degC in K
            'relative_humidity': torch.tensor(0.6, device=self.device),
            'vapor_pressure_deficit': torch.tensor(1000.0, device=self.device),  # Pa
            'daylength': torch.tensor(50400.0, device=self.device),  # seconds
            'air_temperature_min': torch.tensor(288.15, device=self.device),  # 15 degC
            'air_temperature_max': torch.tensor(298.15, device=self.device),  # 25 degC
            'phenology_gdd_accumulated': torch.tensor(0.0, device=self.device),
            'phenology_gdd_previous_day_of_year': torch.tensor(149.0, device=self.device),
        }
        
        # Mock state data
        self.state = {
            'leaf_carbon': torch.tensor(0.5, device=self.device),  # kg C m^-2
            'froot_carbon': torch.tensor(0.3, device=self.device),
            'leaf_nitrogen': torch.tensor(0.02, device=self.device),  # kg N m^-2
            'froot_nitrogen': torch.tensor(0.01, device=self.device)
        }
        
    def test_gdd_model_parameter_loading(self):
        """Test GDD model loads all parameters from ParameterManager."""
        config = {
            **self.config,
            'phenology_type': 'gdd',
            'vegetation_type': 'tree'
        }
        
        model = GDDPhenology(name="test_gdd", config=config)
        
        # Verify critical parameters are loaded from ParameterManager
        assert model.phenology_params.gdd_base_temp is not None
        assert model.phenology_params.gdd_threshold_start is not None
        assert model.phenology_params.gdd_threshold_end is not None
        
        # Test forward pass
        output = model.compute_phenological_state(
            {**self.environment, "day_of_year": self.environment['day_of_year'], "air_temperature": self.environment['air_temperature']},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output.phenology_phase is not None
        assert output.growth_factor is not None
        assert output.senescence_rate is not None
        
    def test_gsi_model_parameter_loading(self):
        """Test GSI model loads all parameters from ParameterManager."""
        config = {
            **self.config,
            'phenology_type': 'gsi',
            'vegetation_type': 'tree'
        }
        
        model = GSIPhenology(name="test_gsi", config=config)
        
        # Verify critical parameters are loaded
        assert model.phenology_params.gsi_tmin_threshold is not None
        assert model.phenology_params.gsi_vpd_min is not None
        assert model.phenology_params.gsi_vpd_max is not None

        # Test forward pass
        environment = {
            **self.environment,
            **_gsi_temporal_state(model, self.device),
        }
        output = model.compute_phenological_state(
            environment,
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output.phenology_phase is not None
        assert output.growing_season_index is not None
        
    def test_crop_model_parameter_loading(self):
        """Test crop model loads all parameters including new ones."""
        config = {
            **self.config,
            'phenology_type': 'crop',
            'vegetation_type': 'crop',
            'crop_type': 'wheat'
        }
        
        model = CropPhenology(name="test_crop", config=config)
        
        crop_state = _crop_temporal_state(self.device, is_sown=True)
        for day in range(100, 200):
            env = {**self.environment, **crop_state}
            env['day_of_year'] = torch.tensor(float(day), device=self.device)
            output = model.compute_phenological_state(
                env,
                dt=get_physical_constant("seconds_per_day"),
            )
            crop_state = {
                key: output.additional_outputs[key]
                for key in CROP_TEMPORAL_STATE_KEYS
            }

        assert output.diagnostic_variables['development_index'] > 0.0
        assert crop_state['phenology_crop_accumulated_phu'] > 0.0
        assert not hasattr(model, 'development_index')
        
    def test_no_hardcoded_constants(self):
        """Verify no hardcoded constants remain in critical paths."""
        config = {
            **self.config,
            'phenology_type': 'gdd',
            'vegetation_type': 'tree'
        }
        
        model = GDDPhenology(name="test_no_hardcoded", config=config)
        
        # Test that phase offset parameters are loaded from ParameterManager
        full_growth_offset = self.param_manager.get('biosphere.phenology.gdd.full_growth_offset')
        senescence_start_offset = self.param_manager.get('biosphere.phenology.gdd.senescence_start_offset')
        
        assert full_growth_offset is not None
        assert senescence_start_offset is not None
        assert full_growth_offset == 300.0  # Check expected value from YAML
        assert senescence_start_offset == 200.0
        
    def test_device_binding_consistency(self):
        """Test that all tensors are created with correct device."""
        for device_type in ['cpu']:  # Add 'cuda' if available
            device = torch.device(device_type)
            config = {
                'device': device,
                'phenology_type': 'gsi',
                'vegetation_type': 'tree'
            }
            
            model = GSIPhenology(name=f"test_device_{device_type}", config=config)
            
            # Update environment to use correct device
            env = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in self.environment.items()}
            env.update(_gsi_temporal_state(model, device))
            
            output = model.compute_phenological_state(
                env,
                dt=get_physical_constant("seconds_per_day"),
            )
            
            # Check all output tensors are on correct device
            assert output.growth_factor.device.type == device_type
            assert output.senescence_rate.device.type == device_type
            assert output.leaf_area_index.device.type == device_type
            
    def test_conservation_law_enforcement(self):
        """Test that conservation laws are properly enforced."""
        config = {
            **self.config,
            'phenology_type': 'gdd',
            'vegetation_type': 'tree'
        }
        
        model = GDDPhenology(name="test_conservation", config=config)
        
        dt = 3600.0
        updates = model(
            {**self.state, **self.environment},
            time=0.0,
            dt=dt,
        )

        for key in ('leaf_carbon', 'froot_carbon'):
            assert isinstance(updates[key], UpdateItem)
            assert updates[key].mode is UpdateMode.ACCUMULATE
            assert updates[key].per_timestep is True

        carbon_residual = (
            updates['leaf_carbon'].value
            + updates['froot_carbon'].value
            + dt
            * (
                updates['leaf_litterfall_carbon'].value
                + updates['froot_litterfall_carbon'].value
            )
        )
        assert torch.allclose(carbon_residual, torch.zeros_like(carbon_residual))
        assert torch.allclose(
            carbon_residual,
            updates[
                'diagnostics.phenology.carbon_turnover_balance_residual'
            ].value,
        )
        
    def test_dependency_declaration(self):
        """State ports must not be represented as reverse Functor dependencies."""
        config = {
            **self.config,
            'phenology_type': 'gsi',
            'vegetation_type': 'tree'
        }
        
        model = GSIPhenology(name="test_dependencies", config=config)
        
        assert model.get_dependencies() == []
        assert set(model.metadata.input_variables) == {
            "day_of_year",
            "air_temperature",
            "air_temperature_min",
            "vapor_pressure_deficit",
            "daylength",
            "phenology_gsi_history",
            "phenology_gsi_history_index",
            "phenology_gsi_history_count",
            "phenology_gsi_in_growing_season",
            "leaf_carbon",
            "froot_carbon",
            "leaf_nitrogen",
            "froot_nitrogen",
        }
        
    def test_pft_integration(self):
        """Test integration with PFT system."""
        # Test different PFT types
        pft_configs = [
            {'pft_type': 'broadleaf_deciduous_temperate', 'vegetation_type': 'tree'},
            {'pft_type': 'c3_grass_temperate', 'vegetation_type': 'grass'},
            {'pft_type': 'crop_temperate', 'vegetation_type': 'crop'}
        ]
        
        for pft_config in pft_configs:
            config = {
                **self.config,
                'phenology_type': 'gsi',
                **pft_config
            }
            
            model = GSIPhenology(name=f"test_pft_{pft_config['pft_type']}", config=config)

            # Verify model can compute with PFT-specific parameters
            environment = {
                **self.environment,
                **_gsi_temporal_state(model, self.device),
            }
            output = model.compute_phenological_state(
                environment,
                dt=get_physical_constant("seconds_per_day"),
            )
            
            assert output is not None
            assert output.phenology_phase is not None
            
    def test_temperature_unit_consistency(self):
        """Test temperature unit handling is consistent."""
        config = {
            **self.config,
            'phenology_type': 'gdd',
            'vegetation_type': 'tree'
        }
        
        model = GDDPhenology(name="test_temp_units", config=config)
        
        # Test with Kelvin input
        output_k = model.compute_phenological_state(
            {**self.environment, "day_of_year": self.environment['day_of_year'], "air_temperature": torch.tensor(293.15, device=self.device)},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        # GDD model should handle temperature correctly
        assert output_k.thermal_time is not None
        assert output_k.thermal_time >= 0  # GDD should be non-negative
        
    def test_crop_does_not_duplicate_biomass_allocation_owner(self):
        """Keep organ allocation outside the crop phenology closure."""
        config = {
            **self.config,
            'phenology_type': 'crop',
            'vegetation_type': 'crop',
            'crop_type': 'wheat'
        }
        
        model = CropPhenology(name="test_allocation", config=config)
        for di in [0.1, 0.3, 0.5, 0.8]:
            environment = {
                **self.environment,
                **_crop_temporal_state(
                    self.device,
                    accumulated_phu=(
                        di * model.phenology_params.crop_phu_total
                    ),
                    is_sown=True,
                ),
            }
            environment['air_temperature'] = torch.tensor(
                get_physical_constant('T_FREEZE')
                + model.phenology_params.gdd_base_temp,
                device=self.device,
            )
            output = model.compute_phenological_state(
                environment,
                dt=get_physical_constant("seconds_per_day"),
            )
            assert output.allocation_fractions == {}

        assert not hasattr(model, '_compute_allocation_fractions')
            

class TestPhenologyStressIntegration:
    """Test phenology response to stress factors."""
    
    @pytest.fixture(autouse=True) 
    def setup(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.config = {'device': self.device}
        
    def test_vpd_stress_response(self):
        """Test the canonical Jolly GSI response to high VPD."""
        model = GSIPhenology(name="test_vpd_stress", config=self.config)
        
        # Normal conditions
        env_normal = {
            'day_of_year': torch.tensor(150.0, device=self.device),
            'air_temperature': torch.tensor(293.15, device=self.device),
            'air_temperature_min': torch.tensor(288.15, device=self.device),
            'vapor_pressure_deficit': torch.tensor(1000.0, device=self.device),
            'daylength': torch.tensor(50400.0, device=self.device),
            **_gsi_temporal_state(model, self.device),
        }
        
        output_normal = model.compute_phenological_state(
            env_normal,
            dt=torch.tensor(
                get_physical_constant("seconds_per_day"),
                device=self.device,
            ),
        )
        
        # High atmospheric demand limits the canonical VPD response factor.
        env_stress = {**env_normal}
        env_stress['vapor_pressure_deficit'] = torch.tensor(
            5000.0,
            device=self.device,
        )
        
        output_stress = model.compute_phenological_state(
            env_stress,
            dt=get_physical_constant("seconds_per_day"),
        )
        
        # GSI should be lower under high VPD.
        assert output_stress.growing_season_index < output_normal.growing_season_index


class TestPhenologyValidation:
    """Validation tests against expected behaviors."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.device = torch.device('cpu')
        self.config = {'device': self.device}
        
    def test_seasonal_progression(self):
        """Test realistic seasonal progression."""
        model = GDDPhenology(name="test_seasonal", config=self.config)
        
        # Track phases through a year
        phases_observed = set()
        accumulated_gdd = torch.tensor(0.0, device=self.device)
        previous_day = torch.tensor(0.0, device=self.device)
        seconds_per_day = torch.tensor(
            get_physical_constant("seconds_per_day"),
            device=self.device,
        )
        
        for doy in range(1, 366):
            temp = 273.15 + 10 + 15 * np.sin(2 * np.pi * (doy - 80) / 365)  # Seasonal temp
            
            env = {
                'day_of_year': torch.tensor(float(doy), device=self.device),
                'air_temperature': torch.tensor(temp, device=self.device),
                'air_temperature_min': torch.tensor(temp - 5, device=self.device),
                'air_temperature_max': torch.tensor(temp + 5, device=self.device),
                'phenology_gdd_accumulated': accumulated_gdd,
                'phenology_gdd_previous_day_of_year': previous_day,
            }
            
            output = model.compute_phenological_state(
                env,
                dt=seconds_per_day,
            )
            
            phases_observed.add(output.phenology_phase)
            accumulated_gdd = output.additional_outputs[
                'phenology_gdd_accumulated'
            ]
            previous_day = output.additional_outputs[
                'phenology_gdd_previous_day_of_year'
            ]
        
        # Should observe multiple phases through the year
        assert len(phases_observed) >= 3, f"Only observed phases: {phases_observed}"
        
    def test_parameter_bounds(self):
        """Test that all parameters respect their defined bounds."""
        param_manager = get_global_manager()
        
        # Check some critical parameters
        params_to_check = [
            ('biosphere.phenology.gdd.base_temp', 0.0, 15.0),
            ('biosphere.phenology.gdd.threshold_start', 50.0, 500.0),
            ('biosphere.phenology.gsi.onset_threshold', 0.0, 1.0),
            ('biosphere.phenology.crop.hi_max', 0.0, 1.0)
        ]
        
        for param_key, min_val, max_val in params_to_check:
            if param_key == 'biosphere.phenology.gdd.base_temp':
                value = param_manager.get(param_key, units="degC")
            else:
                value = param_manager.get(param_key)
            if hasattr(value, 'item'):
                value = value.item()
            if value is not None:
                assert min_val <= value <= max_val, \
                    f"Parameter {param_key}={value} outside bounds [{min_val}, {max_val}]"
                    

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

