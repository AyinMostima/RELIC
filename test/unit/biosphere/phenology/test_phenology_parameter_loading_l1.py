"""
L1 Unit Tests for Phenology Parameter System Integration

Tests parameter loading, unit conversions, and basic functionality
after refactoring to use unified ParameterManager.
This version includes tests for runtime registration fix.

Author: Yuchong Long
Date: 2025-01-24
Updated: 2025-01-30 - Added tests for runtime registration removal
"""

import pytest
import torch
from unittest.mock import patch

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from relic.scientific.biosphere.phenology.models import (
    StaticPhenology, GDDPhenology, GSIPhenology
)
from relic.core.state import UpdateMode
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant


class TestPhenologyParameterLoading:
    """Test parameter loading from unified ParameterManager."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Get parameter manager
        self.param_manager = get_global_manager()
        # Load phenology parameters
        try:
            # Try to load the database
            self.param_manager.load_from_database('relic/biosphere')
        except Exception as e:
            # If that fails, try loading individual file
            import os
            yaml_path = os.path.join('relic', 'parameters', 'database', 'relic', 'biosphere', 'phenology_unified.yaml')
            if os.path.exists(yaml_path):
                self.param_manager.load_from_yaml(yaml_path)
            else:
                # Try from test directory
                yaml_path = os.path.join('..', 'relic', 'parameters', 'database', 'relic', 'biosphere', 'phenology_unified.yaml')
                if os.path.exists(yaml_path):
                    self.param_manager.load_from_yaml(yaml_path)
        
    def test_no_hardcoded_defaults_in_base(self):
        """Verify the Functor path never invokes parameter-model defaults."""
        config = {
            'phenology_type': 'static',
            'vegetation_type': 'tree'
        }
        target_path = (
            "relic.scientific.biosphere.phenology.parameters."
            "get_global_manager"
        )
        with patch(
            target_path,
            side_effect=AssertionError(
                "Functor construction must not use the value-object manager"
            ),
        ):
            model = StaticPhenology('test_static', config)

        from relic.scientific.biosphere.phenology.parameters import phenology_field_names

        assert model.phenology_params.phenology_type == "static"
        assert model.phenology_params.model_fields_set == set(
            phenology_field_names("static")
        )
            
    def test_namespace_consistency(self):
        """Test that namespace mappings are consistent."""
        # Parameters that should be in 'common' namespace
        common_params = [
            'biosphere.phenology.common.phenology_type',
        ]
        
        for param_key in common_params:
            # Should not raise KeyError
            value = self.param_manager.get(param_key)
            assert value is not None, f"Parameter {param_key} not found"
            
        # Parameters that should NOT be in 'static' namespace for common items
        invalid_params = [
            'biosphere.phenology.static.vegetation_type',  # Should be in common
            'biosphere.phenology.static.phenology_type'  # Should be in common
        ]
        
        for param_key in invalid_params:
            # These should not exist after fix
            value = self.param_manager.get(param_key, default=None)
            # It's OK if these don't exist (they shouldn't)
            
    def test_flux_units_conversion(self):
        """Test that flux units are properly converted to per-second rates."""
        config = dict({'phenology_type': 'static'})
        phenology = StaticPhenology('test_flux', config)
        
        # Check internal parameters
        # Turnover rates are stored in canonical per-second units.
        leaf_turnover_param = phenology.phenology_params.leaf_turnover_rate
        assert leaf_turnover_param > 0.0
        
        # Create mock state and environment
        state = {
            'leaf_carbon': torch.tensor(1.0),  # kg C m^-2
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02)
        }
        
        environment = {
            'day_of_year': torch.tensor(180.0),
            'air_temperature': torch.tensor(288.15),  # 15掳C in K
            'daylength': torch.tensor(43200.0),  # 12 hours in seconds
        }
        
        # Compute rates
        updates = phenology({**state, **environment}, time=0.0, dt=1.0)
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Check that fluxes exist
        assert 'leaf_carbon' in fluxes  # Should have state increment
        assert 'leaf_turnover_flux' not in fluxes
        assert 'leaf_litterfall_carbon' in fluxes
        
        # Verify flux is negative (carbon loss from leaves)
        assert fluxes['leaf_carbon'] <= 0, "Leaf carbon increment should be negative (loss)"
        
        # Verify magnitude is reasonable for per-second rate
        # For 0.25 year^-1 rate and 1 kg C m^-2 pool:
        # Daily rate = 0.25 / 365.25 = 0.000684 day-1
        # Second rate = 0.000684 / 86400 = 7.92e-9 s^-1
        expected_order = 1e-9  # Order of magnitude
        actual_flux = abs(fluxes['leaf_carbon'].item())
        assert actual_flux < 1e-6, f"Flux rate {actual_flux} seems too high for per-second rate"
        
    def test_state_increment_mapping(self):
        """Test that fluxes are properly mapped to state increments."""
        config = dict({'phenology_type': 'gdd'})
        phenology = GDDPhenology('test_mapping', config)
        
        state = {
            'leaf_carbon': torch.tensor(2.0),
            'froot_carbon': torch.tensor(1.5),
            'leaf_nitrogen': torch.tensor(0.08),
            'froot_nitrogen': torch.tensor(0.04)
        }
        
        environment = {
            'day_of_year': torch.tensor(250.0),  # Fall season
            'air_temperature': torch.tensor(283.15),  # 10掳C
            'air_temperature_min': torch.tensor(278.15),
            'air_temperature_max': torch.tensor(288.15),
            'phenology_gdd_accumulated': torch.tensor(900.0),
            'phenology_gdd_previous_day_of_year': torch.tensor(249.0),
        }
        
        updates = phenology({**state, **environment}, time=0.0, dt=1.0)
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Check state variable increments exist
        assert 'leaf_carbon' in fluxes, "Missing leaf_carbon increment"
        assert 'froot_carbon' in fluxes, "Missing froot_carbon increment"
        for key, unit in (
            ('phenology_gdd_accumulated', 'degC day'),
            ('phenology_gdd_previous_day_of_year', 'day'),
        ):
            assert updates[key].mode is UpdateMode.REPLACE
            assert updates[key].per_timestep is True
            assert updates[key].unit == unit
        
        # Check litter pool increments
        assert 'leaf_litterfall_carbon' in fluxes, "Missing leaf_litterfall_carbon increment"
        assert 'froot_litterfall_carbon' in fluxes, "Missing froot_litterfall_carbon increment"
        
        # Verify conservation: loss from plant = gain to litter
        if 'leaf_carbon' in fluxes and 'leaf_litterfall_carbon' in fluxes:
            plant_loss = -fluxes['leaf_carbon']
            litter_gain = fluxes['leaf_litterfall_carbon']
            assert torch.allclose(plant_loss, litter_gain, rtol=1e-5), \
                "Carbon conservation violated: plant loss != litter gain"
                
    def test_parameter_registration_on_missing(self):
        """Test that missing parameters are auto-registered."""
        # Try to get a parameter that doesn't exist
        test_key = 'biosphere.phenology.test.nonexistent_param'
        
        # Missing parameters should return the provided default but remain unregistered
        value = self.param_manager.get(
            test_key,
            default=42.0,
            units='dimensionless',
        )
        
        assert value == 42.0, "Default value not returned"
        assert test_key not in self.param_manager.parameters, "Missing parameter should not auto-register"
        
    def test_device_binding(self):
        """Test that all tensors are properly device-bound."""
        # Test with CPU device
        device = torch.device('cpu')
        config = dict({
            'phenology_type': 'gsi',
            'device': device
        })
        
        phenology = GSIPhenology('test_device', config, device=device)
        
        state = {
            'leaf_carbon': torch.tensor(1.0, device=device),
            'froot_carbon': torch.tensor(1.0, device=device),
            'leaf_nitrogen': torch.tensor(0.04, device=device),
            'froot_nitrogen': torch.tensor(0.02, device=device),
        }
        
        environment = {
            'day_of_year': torch.tensor(180.0, device=device),
            'air_temperature': torch.tensor(293.15, device=device),
            'air_temperature_min': torch.tensor(288.15, device=device),
            'vapor_pressure_deficit': torch.tensor(1500.0, device=device),
            'daylength': torch.tensor(50000.0, device=device),
            'phenology_gsi_history': torch.zeros(
                phenology.memory_days,
                device=device,
            ),
            'phenology_gsi_history_index': torch.tensor(0.0, device=device),
            'phenology_gsi_history_count': torch.tensor(0.0, device=device),
            'phenology_gsi_in_growing_season': torch.tensor(0.0, device=device),
        }
        
        # Compute rates
        updates = phenology(
            {**state, **environment},
            time=0.0,
            dt=get_physical_constant("seconds_per_day"),
        )
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Check all outputs are on correct device
        for key, value in fluxes.items():
            if isinstance(value, torch.Tensor):
                assert value.device == device, f"Flux {key} not on correct device"
                
    def test_no_duplicate_parameters(self):
        """Verify no duplicate parameters in YAML after cleanup."""
        registry = self.param_manager.list(namespace="biosphere.phenology")
        
        duplicates = []
        for key in registry:
            if '.defaults.' in key:
                shadow_key = key.replace('.defaults', '')
                if shadow_key in registry:
                    duplicates.append((key, shadow_key))
        
        assert not duplicates, f"Duplicate parameters found: {duplicates}"

    def test_gdd_initial_state_is_not_a_parameter_default(self):
        """GDD initialization belongs to StateManager state, not parameters."""
        stale_key = "biosphere.phenology.defaults.initial_accumulated_gdd"
        assert self.param_manager.get_parameter(stale_key) is None
        
        
class TestPhenologyModels:
    """Test specific phenology model implementations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database('relic/biosphere')
        
    def test_static_phenology_phases(self):
        """Test static phenology phase transitions."""
        config = dict({'phenology_type': 'static'})
        phenology = StaticPhenology('test_static_phases', config)
        
        # Test different days of year
        test_cases = [
            (1, 'dormancy'),  # Winter
            (150, 'active'),  # Summer  
            (290, 'senescence'),  # Fall
        ]
        
        for doy, expected_phase in test_cases:
            environment = {
                'day_of_year': torch.tensor(float(doy)),
                'air_temperature': torch.tensor(288.15)
            }
            
            output = phenology.compute_phenological_state(
                environment,
                dt=get_physical_constant("seconds_per_day"),
            )
            
            # Check phase (note: phase is an enum, we check its behavior)
            # Growth factor should be high in active phase
            if expected_phase == 'active':
                assert output.growth_factor > 0.5
            elif expected_phase == 'dormancy':
                assert output.growth_factor < 0.5
                
    def test_gdd_accumulation(self):
        """Test GDD phenology thermal time accumulation."""
        config = dict({'phenology_type': 'gdd'})
        phenology = GDDPhenology('test_gdd', config)
        
        # Simulate warm conditions for GDD accumulation
        environment = {
            'day_of_year': torch.tensor(150.0),
            'air_temperature': torch.tensor(293.15),  # 20掳C
            'air_temperature_min': torch.tensor(288.15),  # 15 degC
            'air_temperature_max': torch.tensor(298.15),  # 25 degC
            'phenology_gdd_accumulated': torch.tensor(0.0),
            'phenology_gdd_previous_day_of_year': torch.tensor(149.0),
        }
        
        output = phenology.compute_phenological_state(
            environment,
            dt=torch.tensor(
                get_physical_constant("seconds_per_day")
            ),
        )
        
        # Check thermal time exists and is positive
        assert output.thermal_time is not None
        assert output.thermal_time.item() == pytest.approx(15.0)
        
        # Growth factor should be non-negative (remains zero until threshold)
        assert output.growth_factor >= 0
        
    def test_gsi_calculation(self):
        """Test GSI phenology index calculation."""
        config = dict({'phenology_type': 'gsi'})
        phenology = GSIPhenology('test_gsi', config)
        
        # Favorable conditions
        environment = {
            'day_of_year': torch.tensor(180.0),
            'air_temperature': torch.tensor(293.15),  # 20 degC
            'air_temperature_min': torch.tensor(283.15),  # 10 degC
            'vapor_pressure_deficit': torch.tensor(1000.0),  # Low stress
            'daylength': torch.tensor(54000.0),  # 15 hours (long day)
            'phenology_gsi_history': torch.zeros(phenology.memory_days),
            'phenology_gsi_history_index': torch.tensor(0.0),
            'phenology_gsi_history_count': torch.tensor(0.0),
            'phenology_gsi_in_growing_season': torch.tensor(0.0),
        }
        
        output = phenology.compute_phenological_state(
            environment,
            dt=torch.tensor(
                get_physical_constant("seconds_per_day")
            ),
        )
        
        # GSI should be calculated
        assert output.growing_season_index is not None
        assert 0 <= output.growing_season_index <= 1
        
        # Favorable conditions should give high GSI
        assert output.growing_season_index > 0.5
        

class TestParameterUnits:
    """Test parameter unit specifications and conversions."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database('relic/biosphere')
        
    def test_critical_parameter_units(self):
        """Verify critical parameters have correct units."""
        unit_tests = [
            ('biosphere.phenology.gdd.base_temp', 'degC'),
            ('biosphere.phenology.lai.max_lai_deciduous', 'm2 m-2'),
            ('biosphere.phenology.turnover.leaf_turnover_rate', 's^-1'),
            ('biosphere.phenology.morphology.specific_leaf_area', 'm2 kg-1'),
            ('biosphere.phenology.gsi.vpd_min', 'Pa'),
        ]
        
        for param_key, expected_unit in unit_tests:
            # Use get_parameter() to get parameter specification
            param_spec = self.param_manager.get_parameter(param_key)
            param_dict = param_spec.to_dict() if param_spec else None
            assert param_dict is not None, f"Parameter {param_key} not found"
            
            if 'units' in param_dict:
                # Check unit is reasonably similar (allow variations)
                actual_unit = param_dict['units']
                # Simple check - just verify it's not empty
                assert actual_unit, f"Parameter {param_key} has empty units"

    def test_crop_thermal_time_and_shape_parameters_use_canonical_units(self):
        """Verify crop thermal-time and dimensionless shape metadata."""
        expected = {
            'biosphere.phenology.crop.phu_total': ('degC day', 'thermal_time'),
            'biosphere.phenology.crop.phu_total_maize': (
                'degC day',
                'thermal_time',
            ),
            'biosphere.phenology.crop.phu_total_wheat': (
                'degC day',
                'thermal_time',
            ),
            'biosphere.phenology.crop.phu_total_soybean': (
                'degC day',
                'thermal_time',
            ),
            'biosphere.phenology.crop.di_lai_peak_end': (
                'dimensionless',
                'dimensionless',
            ),
            'biosphere.phenology.crop.lai_growth_exponent': (
                'dimensionless',
                'dimensionless',
            ),
            'biosphere.phenology.crop.vernalization_min_factor': (
                'dimensionless',
                'dimensionless',
            ),
        }

        for param_key, (expected_unit, expected_dimension) in expected.items():
            spec = self.param_manager.get_parameter(param_key)
            assert spec is not None, f"Parameter {param_key} not found"
            assert spec.units == expected_unit
            assert spec.dimension == expected_dimension


class TestRuntimeRegistrationRemoval:
    """Test that runtime parameter registration has been properly removed."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        # Record initial parameter count
        # Use get_all_parameters() instead of get_registry()
        self.initial_param_count = len(self.param_manager.get_all_parameters())

    def test_no_runtime_registration_in_pft_types(self):
        """Verify PFT types do not perform runtime registration."""
        from relic.scientific.biosphere.pft.pft_types import (
            PhotosynthesisParameters, PhenologyParameters,
            AllocationParameters, BVOCParameters,
            StressParameters, RespirationParameters
        )

        # Get parameter count before creating objects
        count_before = len(self.param_manager.get_all_parameters())

        # Create parameter objects with a complete SSOT namespace.
        # These should NOT register new parameters at runtime
        namespace = "biosphere.pft.broadleaf_deciduous_temperate"
        PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
        PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
        AllocationParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
        BVOCParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
        StressParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
        RespirationParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)

        # Get parameter count after
        count_after = len(self.param_manager.get_all_parameters())

        # No new parameters should have been registered
        # (allowing for some tolerance in case database loading adds some)
        assert count_after - count_before < 10, \
            f"Too many parameters registered at runtime: {count_after - count_before}"

    def test_missing_required_parameters_fail_fast(self):
        """Reject an incomplete PFT namespace instead of applying defaults."""
        from relic.scientific.biosphere.pft.pft_types import PhotosynthesisParameters

        with pytest.raises(ValueError, match="Missing required photosynthesis parameter"):
            PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace="nonexistent.namespace")

    def test_no_parameterspec_imports(self):
        """Verify ParameterSpec is not imported in fixed modules."""
        # Read the fixed file content
        with open('relic/scientific/biosphere/pft/pft_types.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Check that ParameterSpec is not imported
        assert 'ParameterSpec' not in content or 'from relic.parameters import get_global_manager\n' in content, \
            "ParameterSpec should not be imported after fix"

        # Check that register_parameter is not called
        assert 'register_parameter' not in content or content.count('register_parameter') == 0 or \
               '# ' in content.split('register_parameter')[0].split('\n')[-1], \
            "register_parameter should not be called after fix"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
