"""
L2 Integration Tests for Stress Module

Tests integration between stress functors, parameter system,
and other RELIC components.
"""

import pytest
import torch
from test.utils.forward_call import call_forward

# Import stress functors
from relic.scientific.biosphere.stress import (
    WaterStress,
    ThermalStress,
    NutrientStress,
    OzoneStress
)

# Import system components
from relic.core.state import StateManager
from relic.parameters import get_global_manager

# StateDict and FluxDict are typing aliases; use dict for runtime construction in tests.
StateDict = dict
FluxDict = dict


@pytest.fixture
def param_manager(fresh_global_parameter_manager):
    """A fresh process-wide parameter manager holding only the stress parameters
    (the previous manager is put back afterwards)."""
    manager = get_global_manager()
    manager.clear_parameters()  # Clear any existing parameters
    
    # Load stress parameters
    manager.load_from_database("relic/parameters/database/relic/biosphere/stress.yaml")
    
    return manager


@pytest.fixture
def state_manager():
    """Create state manager for testing."""
    sm = StateManager()
    
    # Initialize common states
    sm.set_state('temperature', torch.tensor(298.0))  # 25C
    sm.set_state('soil_water_potential', torch.tensor(-0.5))  # MPa
    sm.set_state('soil_nitrogen', torch.tensor(0.02))  # kg N/kg soil
    sm.set_state('soil_phosphorus', torch.tensor(0.001))  # kg P/kg soil
    sm.set_state('o3_mixing_ratio', torch.tensor(40.0e-9))  # mol mol-1 (40 ppb)
    sm.set_state('stomatal_conductance', torch.tensor(0.2))  # mol m-2 s-1
    
    return sm


class TestStressIntegration:
    """Test integration of stress functors with system components."""
    
    def test_stress_functor_with_parameter_manager(self, param_manager):
        """Test that stress functors correctly use parameter manager."""
        # Create water stress functor
        water_stress = WaterStress()
        
        # Verify it can get parameters
        wilting = water_stress.get_parameter_value(
            'wilting_point_potential',
            namespace='biosphere.stress.water',
            required=True
        )
        assert wilting is not None
        assert torch.is_tensor(wilting) or isinstance(wilting, (float, int))
        
        # Create thermal stress functor
        thermal_stress = ThermalStress()
        
        # Verify it can get parameters
        temp_opt = thermal_stress.get_parameter_value(
            'optimal_temperature',
            namespace='biosphere.stress.thermal',
            required=True
        )
        assert temp_opt is not None
        assert torch.is_tensor(temp_opt) or isinstance(temp_opt, (float, int))
    
    def test_stress_functor_with_state_manager(self, state_manager):
        """Test stress functors with state manager."""
        water_stress = WaterStress()
        
        # Create state dict from state manager
        state = StateDict({
            'soil_water_potential': state_manager.get_state('soil_water_potential')
        })
        environment = StateDict({})
        
        # Compute stress
        result = call_forward(water_stress, {**state, **environment})
        
        # Verify result structure
        assert isinstance(result, FluxDict)
        assert 'water_stress_factor' in result
        assert isinstance(result['water_stress_factor'], torch.Tensor)
        assert 0.0 <= result['water_stress_factor'].item() <= 1.0
    
    def test_multiple_stress_interaction(self, state_manager, param_manager):
        """Test interaction between multiple stress factors."""
        # Create all stress functors
        water_stress = WaterStress()
        thermal_stress = ThermalStress()
        nutrient_stress = NutrientStress()
        ozone_stress = OzoneStress()
        
        # Create state and environment
        state = StateDict({
            'nitrogen_concentration': torch.tensor(0.02),
            'phosphorus_concentration': torch.tensor(0.002)
        })

        environment = StateDict({
            'soil_water_potential': torch.tensor(-0.5),
            'air_temperature': torch.tensor(298.0),
            'o3_mixing_ratio': torch.tensor(40.0e-9),  # mol mol-1 (40 ppb)
            'stomatal_conductance': torch.tensor(0.2)
        })
        
        # Compute all stress factors
        water_result = call_forward(water_stress, {**state, **environment})
        thermal_result = call_forward(thermal_stress, {**state, **environment})
        nutrient_result = call_forward(nutrient_stress, {**state, **environment})
        ozone_result = call_forward(ozone_stress, {**state, **environment})
        
        # Verify all return valid stress factors
        assert 'water_stress_factor' in water_result
        assert 'temperature_stress_factor' in thermal_result
        assert 'nutrient_stress_factor' in nutrient_result
        assert 'ozone_stress_factor' in ozone_result
        
        # Compute combined stress (multiplicative)
        combined_stress = (
            water_result['water_stress_factor'] *
            thermal_result['temperature_stress_factor'] *
            nutrient_result['nutrient_stress_factor'] *
            ozone_result['ozone_stress_factor']
        )
        
        # Verify combined stress is valid
        assert isinstance(combined_stress, torch.Tensor)
        assert 0.0 <= combined_stress.item() <= 1.0
    
    def test_stress_with_unit_conversion(self, param_manager):
        """Test stress functors handle unit conversions correctly."""
        thermal_stress = ThermalStress()
        
        environment = StateDict({
            'air_temperature': torch.tensor(298.15)
        })

        result_kelvin = call_forward(thermal_stress, {**StateDict({}), **environment})
        assert 'temperature_stress_factor' in result_kelvin
        assert 0.0 <= result_kelvin['temperature_stress_factor'].item() <= 1.0
    
    def test_stress_dependency_chain(self, param_manager):
        """Test that stress functors correctly declare dependencies."""
        water_stress = WaterStress()
        
        dependencies = water_stress.get_dependencies()
        assert dependencies == []

        metadata = water_stress.metadata
        assert metadata.input_variables == ['soil_water_potential']

        provides = water_stress.get_provides()
        assert provides == ['water_stress_factor']
    
    def test_stress_conservation(self, state_manager, param_manager):
        """Test that stress functors maintain conservation laws."""
        # Stress factors don't change mass/energy, only modulate rates
        water_stress = WaterStress()
        
        state = StateDict({
            'soil_water_potential': torch.tensor(-0.5)
        })
        environment = StateDict({})
        
        # Compute stress multiple times
        result1 = call_forward(water_stress, {**state, **environment})
        result2 = call_forward(water_stress, {**state, **environment})
        
        # Results should be deterministic
        torch.testing.assert_close(
            result1['water_stress_factor'],
            result2['water_stress_factor']
        )
    
    def test_stress_parameter_validation(self, param_manager):
        """Test parameter validation in stress functors."""
        # Create nutrient stress with validation
        nutrient_stress = NutrientStress()
        
        # Get critical concentration with bounds checking
        n_crit = nutrient_stress.get_parameter_value(
            'n_critical_concentration',
            namespace='biosphere.stress.nutrient',
            required=True
        )
        n_crit_value = float(n_crit) if not torch.is_tensor(n_crit) else float(n_crit.item())

        # Verify it's within reasonable bounds
        assert n_crit_value > 0.0
        assert n_crit_value < 1.0  # kg N/kg DM should be less than 100%
    
    def test_stress_with_missing_state(self, param_manager):
        """Test stress functors handle missing state gracefully."""
        ozone_stress = OzoneStress()
        
        state = StateDict({})
        environment = StateDict({})

        with pytest.raises(KeyError):
            call_forward(ozone_stress, {**state, **environment})
    
    def test_stress_tensor_operations(self, param_manager):
        """Test tensor operations in stress computations."""
        thermal_stress = ThermalStress()
        
        batch_size = 10
        environment = StateDict({
            'air_temperature': torch.linspace(273, 313, batch_size)  # 0-40C
        })

        result = call_forward(thermal_stress, {**StateDict({}), **environment})
        
        # Should return batch of stress factors
        assert result['temperature_stress_factor'].shape == (batch_size,)
        assert torch.all(result['temperature_stress_factor'] >= 0.0)
        assert torch.all(result['temperature_stress_factor'] <= 1.0)
        
        # Optimal temperature should have highest stress factor
        temp_opt_idx = torch.argmax(result['temperature_stress_factor'])
        optimal_temp = environment['air_temperature'][temp_opt_idx]
        
        # Should be near the configured optimum
        temp_optimum = thermal_stress.get_parameter_value(
            'optimal_temperature',
            namespace='biosphere.stress.thermal',
            required=True
        )
        assert torch.abs(optimal_temp - temp_optimum) < 10.0  # Within 10K


class TestStressParameterIntegration:
    """Test parameter system integration specifics."""
    
    def test_namespace_hierarchy(self, param_manager):
        """Test hierarchical namespace resolution."""
        water_stress = WaterStress()
        
        # Should find parameters under biosphere.stress.water
        wilting = water_stress.get_parameter_value(
            'wilting_point_potential',
            namespace='biosphere.stress.water',
            required=True
        )
        assert wilting is not None
        
        # Should also work with partial namespace
        wilting2 = water_stress.get_parameter_value(
            'water.wilting_point_potential',
            namespace='biosphere.stress',
            required=True
        )
        assert wilting2 is not None
        assert torch.equal(wilting, wilting2)
    
    def test_parameter_registration(self, param_manager):
        """Test that parameters are pre-registered, not at runtime."""
        # Clear missing requests
        param_manager.clear_missing_requests()
        
        # Create all stress functors
        water_stress = WaterStress()
        thermal_stress = ThermalStress()
        nutrient_stress = NutrientStress()
        ozone_stress = OzoneStress()

        param_manager.clear_missing_requests()
        
        # Get some parameters
        water_stress.get_parameter_value('wilting_point_potential', namespace='biosphere.stress.water')
        thermal_stress.get_parameter_value('optimal_temperature', namespace='biosphere.stress.thermal')
        
        # Check for missing parameters
        missing = param_manager.get_missing_requests()
        
        # Should have no missing parameters if properly configured
        missing_relevant = {
            key: count
            for key, count in missing.items()
            if key.startswith((
                "biosphere.stress.global.",
                "biosphere.stress.water.",
                "biosphere.stress.thermal.",
                "biosphere.stress.nutrient.",
                "biosphere.stress.ozone.",
            ))
        }
        assert len(missing_relevant) == 0, f"Missing parameters: {missing_relevant}"
    
    def test_parameter_metadata(self, param_manager):
        """Test that parameters have proper metadata."""
        # Get parameter with metadata
        param_spec = param_manager.get_parameter('biosphere.stress.water.wilting_point_potential')

        if param_spec is not None:
            assert param_spec.units == 'MPa'
            assert param_spec.description
            assert param_spec.bounds[0] < param_spec.bounds[1]
    
    def test_cross_module_parameter_sharing(self, param_manager):
        """Test parameters shared across stress modules."""
        # Both water and thermal stress use min/max clamps
        water_stress = WaterStress()
        thermal_stress = ThermalStress()
        
        water_min = water_stress.get_parameter_value(
            'global.factor_min',
            namespace='biosphere.stress',
            required=True
        )

        thermal_min = thermal_stress.get_parameter_value(
            'global.factor_min',
            namespace='biosphere.stress',
            required=True
        )

        water_min_value = float(water_min) if not torch.is_tensor(water_min) else float(water_min.item())
        thermal_min_value = float(thermal_min) if not torch.is_tensor(thermal_min) else float(thermal_min.item())

        assert water_min_value >= 0.0
        assert thermal_min_value >= 0.0
        assert water_min_value < 0.1
        assert thermal_min_value < 0.1


class TestStressErrorHandling:
    """Test error handling and edge cases."""
    
    def test_extreme_values(self, param_manager):
        """Test stress functors with extreme input values."""
        water_stress = WaterStress()
        
        # Test with very dry conditions
        environment_dry = StateDict({
            'soil_water_potential': torch.tensor(-10.0)  # Very dry
        })

        environment_wet = StateDict({
            'soil_water_potential': torch.tensor(-0.01)  # Very wet
        })

        result_dry = call_forward(water_stress, {**StateDict({}), **environment_dry})
        result_wet = call_forward(water_stress, {**StateDict({}), **environment_wet})
        
        # Dry should have low stress factor
        assert result_dry['water_stress_factor'].item() < 0.2
        
        # Wet should have high stress factor
        assert result_wet['water_stress_factor'].item() > 0.8
    
    def test_nan_handling(self, param_manager):
        """Test handling of NaN values."""
        thermal_stress = ThermalStress()
        
        # Test with NaN temperature
        environment = StateDict({
            'air_temperature': torch.tensor(float('nan'))
        })

        result = call_forward(thermal_stress, {**StateDict({}), **environment})
        
        # Should handle NaN gracefully (return min stress or NaN)
        stress = result['temperature_stress_factor']
        stress_min = thermal_stress.get_parameter_value(
            'global.factor_min',
            namespace='biosphere.stress'
        )
        stress_min_value = float(stress_min) if not torch.is_tensor(stress_min) else float(stress_min.item())
        assert torch.isnan(stress).item() or float(stress.item()) == stress_min_value
    
    def test_device_consistency(self, param_manager):
        """Test stress functors work with different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        water_stress = WaterStress(device=torch.device('cuda'))
        water_stress = water_stress.to('cuda')
        
        state = StateDict({})
        environment = StateDict({
            'soil_water_potential': torch.tensor(-0.5).to('cuda')
        })

        result = call_forward(water_stress, {**state, **environment})
        
        # Result should be on same device
        assert result['water_stress_factor'].device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


