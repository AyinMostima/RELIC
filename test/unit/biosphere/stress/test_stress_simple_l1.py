"""
Simple L1 Unit Test for Stress Module
Tests basic functionality without loading full parameter database
"""

import pytest
import torch

from relic.scientific.biosphere.stress import (
    WaterStress,
    ThermalStress,
    NutrientStress,
    OzoneStress
)
from relic.core.state import UpdateItem, UpdateMode
from test.utils.forward_call import call_forward


class TestStressSimple:
    """Test stress functors with simple inputs."""

    def test_water_stress_basic(self):
        """Test water stress basic functionality."""
        # Create functor
        water_stress = WaterStress()

        # Create simple state (use plain dict, StateDict is just a type alias)
        state = {
            'soil_water_potential': torch.tensor(-1.0)  # MPa
        }
        environment = {}
        
        # Compute stress
        result = call_forward(water_stress, {**state, **environment})
        
        # Check result
        assert isinstance(result, dict)
        assert 'water_stress_factor' in result
        assert isinstance(result['water_stress_factor'], torch.Tensor)
        
        # Check range [0, 1]
        stress_value = result['water_stress_factor'].item()
        assert 0.0 <= stress_value <= 1.0
        print(f"Water stress factor: {stress_value:.3f}")
    
    def test_thermal_stress_basic(self):
        """Test thermal stress basic functionality."""
        # Create functor
        thermal_stress = ThermalStress()

        # Test at optimal temperature (should give high stress factor)
        state = {}
        environment = {
            'air_temperature': torch.tensor(298.15)  # 25C in Kelvin
        }

        # Compute stress
        result = call_forward(thermal_stress, {**state, **environment})
        
        # Check result
        assert isinstance(result, dict)
        assert 'temperature_stress_factor' in result
        assert isinstance(result['temperature_stress_factor'], torch.Tensor)

        # At optimal temperature, stress should be high
        stress_value = result['temperature_stress_factor'].item()
        assert 0.0 <= stress_value <= 1.0
        assert stress_value > 0.7  # Should be near maximum at optimal temp
        print(f"Thermal stress factor at 25C: {stress_value:.3f}")

        # Test at extreme temperature (should give low stress factor)
        state_cold = {}
        environment_cold = {
            'air_temperature': torch.tensor(275.15)  # 2C in Kelvin
        }
        result_cold = call_forward(thermal_stress, {**state_cold, **environment_cold})
        stress_cold = result_cold['temperature_stress_factor'].item()
        assert stress_cold < stress_value  # Cold stress should be lower
        print(f"Thermal stress factor at 2C: {stress_cold:.3f}")
    
    def test_nutrient_stress_basic(self):
        """Test the canonical nutrient limitation control contract."""
        nutrient_stress = NutrientStress()

        state = {
            'nitrogen_concentration': torch.tensor(0.02),  # kg N/kg DM
            'phosphorus_concentration': torch.tensor(0.002)  # kg P/kg DM
        }
        result = nutrient_stress.forward(state, time=0.0, dt=3600.0)

        expected_outputs = {
            'nutrient_stress_factor',
            'diagnostics.nutrient_stress.nitrogen_factor',
            'diagnostics.nutrient_stress.phosphorus_factor',
        }
        assert set(nutrient_stress.metadata.input_variables) == {
            'nitrogen_concentration',
            'phosphorus_concentration',
        }
        assert set(nutrient_stress.metadata.output_variables) == expected_outputs
        assert set(result) == expected_outputs
        assert nutrient_stress.get_dependencies() == []
        assert nutrient_stress.get_provides() == list(
            nutrient_stress.metadata.output_variables
        )
        assert 'leaf_nitrogen' not in nutrient_stress.metadata.input_variables
        assert 'soil_nitrogen' not in nutrient_stress.metadata.input_variables

        for item in result.values():
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == '1'
            assert item.source_functor == nutrient_stress.name

        stress_value = result['nutrient_stress_factor'].value.item()
        assert 0.0 <= stress_value <= 1.0
        print(f"Nutrient stress factor: {stress_value:.3f}")

        state_low = {
            'nitrogen_concentration': torch.tensor(0.001),  # Very low N
            'phosphorus_concentration': torch.tensor(0.0001)  # Very low P
        }
        result_low = nutrient_stress.forward(state_low, time=0.0, dt=3600.0)
        stress_low = result_low['nutrient_stress_factor'].value.item()
        assert stress_low < stress_value  # Low nutrients = more stress (lower factor)
        print(f"Nutrient stress with low nutrients: {stress_low:.3f}")

        with pytest.raises(ValueError, match='must contain only finite values'):
            nutrient_stress.forward(
                {
                    'nitrogen_concentration': torch.tensor(float('nan')),
                    'phosphorus_concentration': torch.tensor(0.001),
                },
                time=0.0,
                dt=3600.0,
            )

        with pytest.raises(ValueError, match='must be non-negative'):
            nutrient_stress.forward(
                {
                    'nitrogen_concentration': torch.tensor(-0.001),
                    'phosphorus_concentration': torch.tensor(0.001),
                },
                time=0.0,
                dt=3600.0,
            )
    
    def test_ozone_stress_basic(self):
        """Test ozone stress basic functionality."""
        # Create functor
        ozone_stress = OzoneStress()
        
        # Test with normal ozone levels
        state = {}
        environment = {
            'o3_mixing_ratio': torch.tensor(40e-9),  # mol/mol (40 ppb)
            'stomatal_conductance': torch.tensor(0.2)  # mol m-2 s-1
        }

        # Compute stress
        result = call_forward(ozone_stress, {**state, **environment})
        
        # Check result
        assert isinstance(result, dict)
        assert 'ozone_stress_factor' in result
        assert isinstance(result['ozone_stress_factor'], torch.Tensor)
        
        # Check range
        stress_value = result['ozone_stress_factor'].item()
        assert 0.0 <= stress_value <= 1.0
        print(f"Ozone stress factor at 40 ppb: {stress_value:.3f}")
        
        # Test with high ozone
        state_high = {}
        environment_high = {
            'o3_mixing_ratio': torch.tensor(120e-9),  # mol/mol (120 ppb)
            'stomatal_conductance': torch.tensor(0.2)
        }
        result_high = call_forward(ozone_stress, {**state_high, **environment_high})
        stress_high = result_high['ozone_stress_factor'].item()
        # High ozone should give equal or lower stress factor
        assert stress_high <= stress_value
        print(f"Ozone stress at 120 ppb: {stress_high:.3f}")
    
    def test_combined_stress(self):
        """Test combining multiple stress factors."""
        # Create all stress functors
        water_stress = WaterStress()
        thermal_stress = ThermalStress()
        nutrient_stress = NutrientStress()
        ozone_stress = OzoneStress()
        
        # Create comprehensive state and environment
        state = {
            'soil_water_potential': torch.tensor(-0.8),
            'nitrogen_concentration': torch.tensor(0.015),
            'phosphorus_concentration': torch.tensor(0.0008)
        }
        environment = {
            'air_temperature': torch.tensor(295.15),  # 22C
            'o3_mixing_ratio': torch.tensor(50e-9),  # mol/mol (50 ppb)
            'stomatal_conductance': torch.tensor(0.15)
        }

        # Compute all stress factors
        water_result = call_forward(water_stress, {**state, **environment})
        thermal_result = call_forward(thermal_stress, {**state, **environment})
        nutrient_result = call_forward(nutrient_stress, {**state, **environment})
        ozone_result = call_forward(ozone_stress, {**state, **environment})

        # Extract stress factors
        water_factor = water_result['water_stress_factor']
        thermal_factor = thermal_result['temperature_stress_factor']
        nutrient_factor = nutrient_result['nutrient_stress_factor']
        ozone_factor = ozone_result['ozone_stress_factor']
        
        # Combine multiplicatively (common approach)
        combined_stress = water_factor * thermal_factor * nutrient_factor * ozone_factor
        
        # Check combined stress
        assert isinstance(combined_stress, torch.Tensor)
        combined_value = combined_stress.item()
        assert 0.0 <= combined_value <= 1.0
        
        # Combined stress should be less than any individual factor
        assert combined_value <= water_factor.item()
        assert combined_value <= thermal_factor.item()
        assert combined_value <= nutrient_factor.item()
        assert combined_value <= ozone_factor.item()
        
        print(f"\nIndividual stress factors:")
        print(f"  Water: {water_factor.item():.3f}")
        print(f"  Thermal: {thermal_factor.item():.3f}")
        print(f"  Nutrient: {nutrient_factor.item():.3f}")
        print(f"  Ozone: {ozone_factor.item():.3f}")
        print(f"  Combined: {combined_value:.3f}")
    
    def test_metadata_exists(self):
        """Test that all stress functors have proper metadata."""
        functors = [
            WaterStress(),
            ThermalStress(),
            NutrientStress(),
            OzoneStress()
        ]

        for functor in functors:
            # Access metadata as attribute
            assert hasattr(functor, 'metadata'), f"{functor.name} missing metadata attribute"
            metadata = functor.metadata

            # Check required metadata fields
            assert hasattr(metadata, 'name')
            assert hasattr(metadata, 'description')
            assert hasattr(metadata, 'mathematical_expression')
            assert hasattr(metadata, 'references')
            assert hasattr(metadata, 'parameters')
            assert hasattr(metadata, 'category')
            assert hasattr(metadata, 'execution_phase')

            # Check that references is not empty
            assert len(metadata.references) > 0

            # Check that mathematical expression exists
            assert len(metadata.mathematical_expression) > 0

            # Check functor-level attributes (not in metadata)
            assert hasattr(functor, 'get_dependencies')
            assert hasattr(functor, 'get_provides')

            print(f"\n{metadata.name} metadata OK")
            print(f"  References: {len(metadata.references)} entries")
            print(f"  Parameters: {len(metadata.parameters)} defined")


def test_nutrient_stress_passes_no_unused_bridge_label():
    """12.15 and 12.16: no algorithm reads the bridge enzyme_kinetics, temperature_response
    or ph_dependence labels (they are only validated and copied into metadata), so
    NutrientStress passes none of them."""
    import inspect

    from relic.scientific.biosphere.stress import nutrient_stress

    source = inspect.getsource(nutrient_stress)
    for label in ("enzyme_kinetics", "temperature_response", "ph_dependence"):
        assert label not in source


if __name__ == "__main__":
    # Run tests manually
    test = TestStressSimple()
    
    print("Running stress module simple tests...")
    print("="*60)
    
    try:
        print("\n1. Testing Water Stress...")
        test.test_water_stress_basic()
        
        print("\n2. Testing Thermal Stress...")
        test.test_thermal_stress_basic()
        
        print("\n3. Testing Nutrient Stress...")
        test.test_nutrient_stress_basic()
        
        print("\n4. Testing Ozone Stress...")
        test.test_ozone_stress_basic()
        
        print("\n5. Testing Combined Stress...")
        test.test_combined_stress()
        
        print("\n6. Testing Metadata...")
        test.test_metadata_exists()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        raise

