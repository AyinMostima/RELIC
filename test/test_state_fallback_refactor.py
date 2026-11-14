"""
Test Suite for state.get() default Refactoring

Verifies that the refactoring from state.get() default patterns to
strict validation or parameter-based defaults works correctly.

Test Coverage:
1. NutrientStress - Strict validation (nitrogen/phosphorus concentration required)
2. ThermalStress - Intelligent default (air_temperature OR leaf_temperature)
3. PhenologyGrowth - Selected-owner phenological-factor observation
4. StructuralGrowth - Parameter-based defaults (leaf growth rate, current LAI)
"""

import pytest
import torch
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.stress.nutrient_stress import NutrientStress
from relic.scientific.biosphere.stress.thermal_stress import ThermalStress
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from test.utils.forward_call import call_forward


class TestNutrientStressRefactoring:
    """Test strict validation for nutrient concentrations."""
    
    def test_missing_nitrogen_concentration_raises_error(self):
        """Should raise ValueError when nitrogen_concentration is missing."""
        functor = NutrientStress()
        state = {
            'phosphorus_concentration': torch.tensor(0.002)
        }
        
        with pytest.raises(ValueError) as exc_info:
            functor.forward(state, time=0.0, dt=1800.0)
        
        assert 'nitrogen_concentration' in str(exc_info.value)
        assert 'Available keys' in str(exc_info.value)
        
    def test_missing_phosphorus_concentration_raises_error(self):
        """Should raise ValueError when phosphorus_concentration is missing."""
        functor = NutrientStress()
        state = {
            'nitrogen_concentration': torch.tensor(0.02)
        }
        
        with pytest.raises(ValueError) as exc_info:
            functor.forward(state, time=0.0, dt=1800.0)
        
        assert 'phosphorus_concentration' in str(exc_info.value)
        assert 'Available keys' in str(exc_info.value)
        
    def test_both_concentrations_present_succeeds(self):
        """Should succeed when both required concentrations are present."""
        functor = NutrientStress()
        state = {
            'nitrogen_concentration': torch.tensor(0.02),
            'phosphorus_concentration': torch.tensor(0.002)
        }
        
        result = functor.forward(state, time=0.0, dt=1800.0)

        assert set(result) == {
            'nutrient_stress_factor',
            'diagnostics.nutrient_stress.nitrogen_factor',
            'diagnostics.nutrient_stress.phosphorus_factor',
        }
        assert all(isinstance(value, UpdateItem) for value in result.values())
        assert all(value.mode == UpdateMode.REPLACE for value in result.values())

    @pytest.mark.parametrize(
        ("state", "error_type", "message"),
        [
            (
                {
                    'nitrogen_concentration': 0.02,
                    'phosphorus_concentration': torch.tensor(0.002),
                },
                TypeError,
                'nitrogen_concentration must be a torch.Tensor',
            ),
            (
                {
                    'nitrogen_concentration': torch.tensor(0.02),
                    'phosphorus_concentration': torch.tensor(float('inf')),
                },
                ValueError,
                'phosphorus_concentration must contain only finite values',
            ),
            (
                {
                    'leaf_nitrogen': torch.tensor(0.02),
                    'leaf_phosphorus': torch.tensor(0.002),
                },
                ValueError,
                "Required state variable 'nitrogen_concentration'",
            ),
        ],
    )
    def test_invalid_or_deprecated_inputs_fail_fast(
        self, state, error_type, message
    ):
        """Reject non-tensors, non-finite values, and removed aliases."""
        functor = NutrientStress()

        with pytest.raises(error_type, match=message):
            functor.forward(state, time=0.0, dt=1800.0)
        

class TestThermalStressRefactoring:
    """Test intelligent default for temperature variables."""
    
    def test_missing_air_temperature_raises_error(self):
        """The declared input air_temperature is required (item 117)."""
        functor = ThermalStress()
        state = {}

        with pytest.raises(KeyError) as exc_info:
            functor.forward(state, time=0.0, dt=1800.0)

        assert 'air_temperature' in str(exc_info.value)
        assert 'available keys' in str(exc_info.value)
        
    def test_air_temperature_present_succeeds(self):
        """Should succeed with air_temperature."""
        functor = ThermalStress()
        state = {
            'air_temperature': torch.tensor(298.15)
        }
        
        result = functor.forward(state, time=0.0, dt=1800.0)
        
        assert 'temperature_stress_factor' in result
        item = result['temperature_stress_factor']
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.REPLACE
        assert isinstance(item.value, torch.Tensor)
        
    def test_leaf_temperature_is_not_an_alias(self):
        """leaf_temperature does not stand in for air_temperature (item 117)."""
        functor = ThermalStress()
        state = {
            'leaf_temperature': torch.tensor(300.15)
        }

        with pytest.raises(KeyError, match='air_temperature'):
            functor.forward(state, time=0.0, dt=1800.0)


class TestPhenologyGrowthRefactoring:
    """Test strict canonical state interconnections for phenology controls."""
    
    def test_missing_phenological_factor_fails_fast(self):
        """The selected main owner must supply phenological_factor."""
        functor = PhenologyGrowth(phenology_model_owner="gsi_phenology")

        with pytest.raises(KeyError, match='phenological_factor'):
            functor.forward({}, time=0.0, dt=1800.0)
        
    def test_old_environmental_ports_do_not_restore_fifth_formula(self):
        """Raw GDD and forcing inputs cannot reactivate the removed closure."""
        functor = PhenologyGrowth(phenology_model_owner="gsi_phenology")
        state = {
            'accumulated_gdd': torch.tensor(100.0),
            'air_temperature': torch.tensor(288.15),
            'daylength': torch.tensor(43200.0),
        }

        with pytest.raises(KeyError, match='phenological_factor'):
            functor.forward(state, time=0.0, dt=1800.0)
        
    def test_missing_selected_owner_fails_fast(self):
        """Composition must select one of the four main phenology owners."""
        with pytest.raises(ValueError, match='phenology_model_owner'):
            PhenologyGrowth()
        
    def test_selected_owner_variable_present_succeeds(self):
        """The canonical factor produces only one namespaced diagnostic."""
        functor = PhenologyGrowth(phenology_model_owner="gsi_phenology")
        state = {'phenological_factor': torch.tensor(0.2)}

        result = functor.forward(state, time=0.0, dt=1800.0)

        assert set(result) == set(functor.metadata.output_variables)
        assert set(result) == {'diagnostics.phenology.growth_active'}
        assert 'rate' not in result
        assert all(isinstance(item, UpdateItem) for item in result.values())
        assert all(item.mode is UpdateMode.REPLACE for item in result.values())
        diagnostic = result['diagnostics.phenology.growth_active']
        assert diagnostic.per_timestep is False
        assert diagnostic.value.item() is True


class TestStructuralGrowthRefactoring:
    """Test explicit StateManager ports for structural growth."""
    
    def test_missing_leaf_growth_fails_fast(self):
        """Missing upstream allocation output must fail fast."""
        functor = StructuralGrowth()
        state = {
            'leaf_area_index': torch.tensor(3.0)
        }
        
        with pytest.raises(KeyError, match="growth_rate_leaf"):
            call_forward(functor, state)
        
    def test_missing_leaf_area_index_fails_fast(self):
        """Missing StateManager-owned LAI must fail fast."""
        functor = StructuralGrowth()
        state = {
            'growth_rate_leaf': torch.tensor(0.5)
        }
        
        with pytest.raises(KeyError, match="leaf_area_index"):
            call_forward(functor, state)
        
    def test_both_variables_present_succeeds(self):
        """Should succeed with both state variables present."""
        functor = StructuralGrowth()
        state = {
            'growth_rate_leaf': torch.tensor(2.0e-7),
            'leaf_area_index': torch.tensor(3.0)
        }
        
        dt = 3600.0
        result = functor.forward(state, time=0.0, dt=dt)

        assert set(result) == set(functor.metadata.output_variables)
        assert 'rate' not in result
        assert all(isinstance(value, UpdateItem) for value in result.values())
        assert all(value.mode is UpdateMode.REPLACE for value in result.values())

        gross = result[
            'diagnostics.structural_growth.gross_leaf_area_growth_tendency'
        ]
        loss = result[
            'diagnostics.structural_growth.leaf_area_senescence_tendency'
        ]
        net = result['diagnostics.structural_growth.net_leaf_area_tendency']
        step = result['diagnostics.structural_growth.step_leaf_area_delta']
        assert gross.unit == 'm2 m-2 s-1'
        assert loss.unit == 'm2 m-2 s-1'
        assert net.unit == 'm2 m-2 s-1'
        assert step.unit == 'm2 m-2'
        assert gross.per_timestep is False
        assert loss.per_timestep is False
        assert net.per_timestep is False
        assert step.per_timestep is True
        torch.testing.assert_close(net.value, gross.value - loss.value)
        torch.testing.assert_close(step.value, net.value * dt)


class TestRefactoringImpact:
    """Test that refactoring didn't break existing functionality."""
    
    def test_nutrient_stress_numerical_consistency(self):
        """Verify nutrient stress calculations remain numerically consistent."""
        functor = NutrientStress()
        state = {
            'nitrogen_concentration': torch.tensor(0.025),
            'phosphorus_concentration': torch.tensor(0.003)
        }
        
        result = functor.forward(state, time=0.0, dt=1800.0)
        
        overall = result['nutrient_stress_factor'].value
        nitrogen = result[
            'diagnostics.nutrient_stress.nitrogen_factor'
        ].value
        phosphorus = result[
            'diagnostics.nutrient_stress.phosphorus_factor'
        ].value

        for factor in (overall, nitrogen, phosphorus):
            assert 0.0 <= factor.item() <= 1.0
        torch.testing.assert_close(overall, torch.minimum(nitrogen, phosphorus))
        
    def test_thermal_stress_numerical_consistency(self):
        """Verify thermal stress calculations remain numerically consistent."""
        functor = ThermalStress()
        state = {
            'air_temperature': torch.tensor(298.15)
        }
        
        result = functor.forward(state, time=0.0, dt=1800.0)
        
        # Stress factors should be in valid range [0, 1]
        assert 0.0 <= result['temperature_stress_factor'].value.item() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


