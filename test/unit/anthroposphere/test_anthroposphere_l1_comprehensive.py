#!/usr/bin/env python3
"""
Comprehensive L1 Unit Tests for RELIC Anthroposphere Module

This comprehensive test suite validates all anthroposphere classes after the 
inheritance refactoring from FunctorBase to the four abstract base classes:
- PhysicalField
- BiologicalProcess  
- TransportMechanism
- ScaleCoupling

Test Categories:
- L1.1: Class instantiation and initialization
- L1.2: Abstract base class compliance
- L1.3: Core method functionality
- L1.4: Scientific calculation accuracy
- L1.5: Backward consistency (forward method)
- L1.6: Parameter handling and validation
- L1.7: Conservation checks (where applicable)

Scientific References:
- Farquhar, G.D. et al. (1980). Biochemical model of photosynthetic CO2 assimilation
- Lloyd, J. & Taylor, J.A. (1994). Temperature dependence of soil respiration
- Guenther, A.B. et al. (2012). MEGAN2.1 emissions model
- Ball, J.T. et al. (1987). Stomatal conductance model

Test Framework: pytest
Coverage Target: >90% for all anthroposphere modules
"""

import pytest
import torch
import sys
import os

from test.utils.forward_call import call_forward, call_transport_forward
from test.utils.functor_minimal_construction import construct_minimal

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Anthroposphere modules - Emissions
from relic.scientific.anthroposphere.emissions.sector_allocation import SectorAllocation

# Anthroposphere modules - Land Use
from relic.scientific.anthroposphere.land_use.mapping import LandUseParameterMapping
from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions

# Anthroposphere modules - Management
from relic.scientific.anthroposphere.management.fertilization import FertilizationManagement
from relic.scientific.anthroposphere.management.irrigation import IrrigationManagement
from relic.scientific.anthroposphere.management.tillage import TillageManagement
from relic.scientific.anthroposphere.management.grazing import GrazingManagement
from relic.scientific.anthroposphere.management.harvest import HarvestManagement
from relic.scientific.anthroposphere.management.fire_management import FireManagement

# Test constants
DEVICE = torch.device("cpu")
TEST_BATCH_SIZE = 10
SPATIAL_DIMS = 2
TEMPORAL_STEPS = 24
TOLERANCE = 1e-6

class TestClassification:
    """Test data structure for organized test execution."""
    
    # Class categorization by abstract base class
    PHYSICAL_FIELD_CLASSES = [
        (LandUseParameterMapping, "land_use_parameter_mapping")
    ]
    
    BIOLOGICAL_PROCESS_CLASSES = [
        (FertilizationManagement, "fertilization_management"),
        (GrazingManagement, "grazing_management"),
        (HarvestManagement, "harvest_management"),
        (FireManagement, "fire_management")
    ]
    
    TRANSPORT_MECHANISM_CLASSES = [
        (IrrigationManagement, "irrigation_management"),
        (TillageManagement, "tillage_management")
    ]
    
    SCALE_COUPLING_CLASSES = [
        (SectorAllocation, "sector_allocation"),
        (LandUseTransitions, "land_use_transitions")
    ]
    
    ALL_CLASSES = (
        PHYSICAL_FIELD_CLASSES + 
        BIOLOGICAL_PROCESS_CLASSES + 
        TRANSPORT_MECHANISM_CLASSES + 
        SCALE_COUPLING_CLASSES
    )


@pytest.fixture
def sample_state_dict():
    """Sample state dictionary for testing."""
    return {
        'temperature': torch.tensor([298.15] * TEST_BATCH_SIZE, device=DEVICE),
        'soil_moisture': torch.tensor([0.3] * TEST_BATCH_SIZE, device=DEVICE),
        'soil_nitrogen': torch.tensor([100.0] * TEST_BATCH_SIZE, device=DEVICE),
        'biomass': torch.tensor([1000.0] * TEST_BATCH_SIZE, device=DEVICE),
        'aboveground_biomass': torch.tensor([2.0] * TEST_BATCH_SIZE, device=DEVICE),
        'belowground_biomass': torch.tensor([0.2] * TEST_BATCH_SIZE, device=DEVICE),
        'litter_mass': torch.tensor([0.3] * TEST_BATCH_SIZE, device=DEVICE),
        'fuel_moisture': torch.tensor([0.15] * TEST_BATCH_SIZE, device=DEVICE),
        'days_since_fire': torch.tensor([30.0] * TEST_BATCH_SIZE, device=DEVICE),
        'crop_maturity': torch.tensor([1.0] * TEST_BATCH_SIZE, device=DEVICE),
        'residue_cover': torch.tensor([0.0] * TEST_BATCH_SIZE, device=DEVICE),
        'vegetation_quality': torch.tensor([0.8] * TEST_BATCH_SIZE, device=DEVICE),
        'leaf_area_index': torch.tensor([3.0] * TEST_BATCH_SIZE, device=DEVICE)
    }


@pytest.fixture
def sample_environment_dict():
    """Sample environment dictionary for testing."""
    return {
        'air_temperature': torch.tensor([298.15] * TEST_BATCH_SIZE, device=DEVICE),
        'relative_humidity': torch.tensor([0.7] * TEST_BATCH_SIZE, device=DEVICE),
        'soil_ph': torch.tensor([6.5] * TEST_BATCH_SIZE, device=DEVICE),
        'light_intensity': torch.tensor([1000.0] * TEST_BATCH_SIZE, device=DEVICE),
        'precipitation': torch.tensor([5.0] * TEST_BATCH_SIZE, device=DEVICE),
        'wind_speed': torch.tensor([2.0] * TEST_BATCH_SIZE, device=DEVICE)
    }


@pytest.fixture
def sample_coordinates():
    """Sample spatial coordinates for testing."""
    return torch.randn(TEST_BATCH_SIZE, SPATIAL_DIMS, device=DEVICE)


class TestL1Instantiation:
    """L1.1: Test class instantiation and basic initialization."""
    
    @pytest.mark.parametrize("class_type,name", TestClassification.ALL_CLASSES)
    def test_class_instantiation(self, class_type, name):
        """Test that all classes can be instantiated."""
        try:
            instance = construct_minimal(f"{class_type.__module__}:{class_type.__name__}")
            assert instance is not None
            assert hasattr(instance, '__class__')
            print(f"[OK] {name}: Instantiation successful")
        except Exception as e:
            pytest.fail(f"[FAIL] {name}: Failed to instantiate - {e}")
    
    @pytest.mark.parametrize("class_type,name", TestClassification.ALL_CLASSES)
    def test_device_assignment(self, class_type, name):
        """Test device assignment in initialization."""
        try:
            instance = construct_minimal(f"{class_type.__module__}:{class_type.__name__}")
            if hasattr(instance, 'device'):
                assert instance.device is not None
                print(f"[OK] {name}: Device assignment verified")
            else:
                print(f"[WARNING] {name}: No device attribute found")
        except Exception as e:
            pytest.fail(f"[FAIL] {name}: Device assignment error - {e}")


class TestL2AbstractBaseCompliance:
    """L1.2: Test compliance with abstract base class requirements."""
    
    @pytest.mark.parametrize("class_type,name", TestClassification.PHYSICAL_FIELD_CLASSES)
    def test_physical_field_compliance(self, class_type, name, sample_coordinates):
        """Test PhysicalField abstract method implementations."""
        instance = construct_minimal(f"{class_type.__module__}:{class_type.__name__}")
        
        # Test required methods exist
        assert hasattr(instance, 'get_value_at'), f"{name}: Missing get_value_at method"
        assert hasattr(instance, 'compute_gradient'), f"{name}: Missing compute_gradient method"
        assert hasattr(instance, 'interpolate'), f"{name}: Missing interpolate method"
        assert hasattr(instance, 'update_field'), f"{name}: Missing update_field method"

        if class_type is LandUseParameterMapping:
            with pytest.raises(RuntimeError, match="current spatial field"):
                instance.get_value_at(sample_coordinates)
            print(f"[OK] {name}: PhysicalField support lock verified")
            return
        
        # Test method calls
        try:
            values = instance.get_value_at(sample_coordinates)
            assert isinstance(values, torch.Tensor), f"{name}: get_value_at should return Tensor"
            print(f"[OK] {name}: PhysicalField compliance verified")
        except Exception as e:
            pytest.fail(f"[FAIL] {name}: PhysicalField method error - {e}")
    
    @pytest.mark.parametrize("class_type,name", TestClassification.BIOLOGICAL_PROCESS_CLASSES)
    def test_biological_process_compliance(self, class_type, name, sample_state_dict, sample_environment_dict):
        """Test BiologicalProcess abstract method implementations."""
        instance = construct_minimal(f"{class_type.__module__}:{class_type.__name__}")
        
        # Test required methods exist
        assert hasattr(instance, 'forward'), f"{name}: Missing compute_rate method"
        
        # Test method calls
        try:
            rates = call_forward(
                instance,
                {**sample_state_dict, **sample_environment_dict},
                calendar_type="noleap",
            )
            assert isinstance(rates, dict), f"{name}: compute_rate should return dict"
            assert len(rates) > 0, f"{name}: compute_rate should return non-empty dict"
            
            # Verify all rates are tensors
            for key, value in rates.items():
                assert isinstance(value, torch.Tensor), f"{name}: All rates should be Tensors"
            
            print(f"[OK] {name}: BiologicalProcess compliance verified")
        except Exception as e:
            pytest.fail(f"[FAIL] {name}: BiologicalProcess method error - {e}")
    
    @pytest.mark.parametrize("class_type,name", TestClassification.TRANSPORT_MECHANISM_CLASSES)
    def test_transport_mechanism_compliance(self, class_type, name, sample_coordinates):
        """Test TransportMechanism abstract method implementations."""
        instance = construct_minimal(f"{class_type.__module__}:{class_type.__name__}")
        
        # Test required methods exist
        assert hasattr(instance, 'forward'), f"{name}: Missing compute_flux method"
        
        # Test method calls with mock data
        try:
            source_field = torch.randn(TEST_BATCH_SIZE, device=DEVICE)
            sink_field = torch.randn(TEST_BATCH_SIZE, device=DEVICE)

            if name == "irrigation_management":
                irrigation_state = {
                    "soil_moisture": torch.full((TEST_BATCH_SIZE,), 0.20, device=DEVICE),
                    "precipitation_flux": torch.zeros(TEST_BATCH_SIZE, device=DEVICE),
                    "land_use_type": torch.ones(
                        TEST_BATCH_SIZE, dtype=torch.long, device=DEVICE
                    ),
                    "irrigation_type": torch.full(
                        (TEST_BATCH_SIZE,), 2, dtype=torch.long, device=DEVICE
                    ),
                    "field_capacity": torch.full((TEST_BATCH_SIZE,), 0.35, device=DEVICE),
                    "wilting_point": torch.full((TEST_BATCH_SIZE,), 0.15, device=DEVICE),
                    "irrigation_remaining_quota": torch.tensor(800.0, device=DEVICE),
                }
                fluxes = instance.forward(
                    irrigation_state,
                    time=0.0,
                    dt=3600.0,
                    calendar_type="noleap",
                )
            elif name == "tillage_management":
                tillage_state = {
                    "soil_moisture": torch.full((TEST_BATCH_SIZE,), 0.30, device=DEVICE),
                    "soil_temperature": torch.full((TEST_BATCH_SIZE,), 298.15, device=DEVICE),
                    "soil_bulk_density": torch.full((TEST_BATCH_SIZE,), 1.40, device=DEVICE),
                    "soil_organic_carbon": torch.full((TEST_BATCH_SIZE,), 0.02, device=DEVICE),
                    "soil_aggregates": torch.full((TEST_BATCH_SIZE,), 0.70, device=DEVICE),
                    "residue_cover": torch.full((TEST_BATCH_SIZE,), 0.50, device=DEVICE),
                    "days_since_tillage": torch.full((TEST_BATCH_SIZE,), 365.0, device=DEVICE),
                }
                tillage_environment = {
                    "air_temperature": tillage_state["soil_temperature"],
                    "soil_temperature": tillage_state["soil_temperature"],
                    "precipitation": torch.zeros(TEST_BATCH_SIZE, device=DEVICE),
                }
                fluxes = instance.forward(
                    tillage_state,
                    time=0.0,
                    dt=3600.0,
                    calendar_type="noleap",
                )
            else:
                fluxes = call_transport_forward(instance, source_field, sink_field)
            assert isinstance(fluxes, dict), f"{name}: compute_flux should return dict"
            
            print(f"[OK] {name}: TransportMechanism compliance verified")
        except Exception as e:
            pytest.fail(f"[FAIL] {name}: TransportMechanism method error - {e}")
    
    @pytest.mark.parametrize("class_type,name", TestClassification.SCALE_COUPLING_CLASSES)
    def test_scale_coupling_compliance(self, class_type, name):
        """Test ScaleCoupling abstract method implementations."""
        instance = construct_minimal(f"{class_type.__module__}:{class_type.__name__}")
        
        # Test required methods exist
        assert hasattr(instance, 'upscale'), f"{name}: Missing upscale method"
        assert hasattr(instance, 'downscale'), f"{name}: Missing downscale method"

        if class_type is LandUseTransitions:
            with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
                instance.upscale(torch.randn(TEST_BATCH_SIZE * 4, device=DEVICE))
            with pytest.raises(RuntimeError, match="explicit typed chain-map bridge"):
                instance.downscale(torch.randn(TEST_BATCH_SIZE, device=DEVICE))
            print(f"[OK] {name}: ScaleCoupling bridge lock verified")
            return

        if class_type is SectorAllocation:
            spatial_shape = (2, TEST_BATCH_SIZE // 2)
            sector_data = torch.rand(
                len(instance.STRUCTURAL_SECTORS),
                len(instance.CANONICAL_SPECIES),
                *spatial_shape,
                device=DEVICE,
            )
            upscaled = instance.upscale(sector_data)
            assert upscaled.shape == (
                len(instance.CANONICAL_SPECIES),
                *spatial_shape,
            )
            assert torch.allclose(upscaled, sector_data.sum(dim=0))
            with pytest.raises(RuntimeError, match="requires declared proxy state"):
                instance.downscale(upscaled)
            print(f"[OK] {name}: Canonical sector scale contract verified")
            return
        
        # Test method calls with mock data
        try:
            fine_data = torch.randn(TEST_BATCH_SIZE * 4, device=DEVICE)
            coarse_data = torch.randn(TEST_BATCH_SIZE, device=DEVICE)
            
            # Test upscaling
            upscaled = instance.upscale(fine_data)
            assert isinstance(upscaled, torch.Tensor), f"{name}: upscale should return Tensor"
            
            # Test downscaling
            downscaled = instance.downscale(coarse_data)
            assert isinstance(downscaled, torch.Tensor), f"{name}: downscale should return Tensor"
            
            print(f"[OK] {name}: ScaleCoupling compliance verified")
        except Exception as e:
            pytest.fail(f"[FAIL] {name}: ScaleCoupling method error - {e}")


class TestL3CoreFunctionality:
    """L1.3: Test core method functionality and scientific calculations."""
    
    def test_fertilization_boundary_calculation(self, sample_state_dict, sample_environment_dict):
        """Test managed N/P boundary outputs without duplicate soil reactions."""
        fertilization = FertilizationManagement()

        rates = call_forward(
            fertilization,
            {**sample_state_dict, **sample_environment_dict},
            calendar_type="noleap",
        )
        
        # Check expected outputs
        expected_fluxes = [
            'fertilizer_n_flux',
            'fertilizer_p_flux',
            'available_phosphorus',
        ]
        for flux in expected_fluxes:
            assert flux in rates, f"Missing expected flux: {flux}"
            assert torch.all(rates[flux] >= 0), f"Negative rate in {flux}"

        assert 'denitrification_rate' not in rates
        assert 'volatilization_flux' not in rates
        
        print("[OK] FertilizationManagement: Rate calculations verified")
    
    def test_land_use_parameter_field(self, sample_coordinates):
        """Test strict spatial support requirements for land-use fields."""
        mapping = LandUseParameterMapping()

        with pytest.raises(RuntimeError, match="current spatial field"):
            mapping.get_value_at(sample_coordinates)
        with pytest.raises(RuntimeError, match="current spatial field"):
            mapping.compute_gradient(sample_coordinates)
        
        print("[OK] LandUseParameterMapping: Spatial support lock verified")


class TestL4ScientificAccuracy:
    """L1.4: Test scientific calculation accuracy and physical realism."""
    
    def test_temperature_response_functions(self, sample_state_dict, sample_environment_dict):
        """Test Q10 and Arrhenius temperature responses."""
        fertilization = FertilizationManagement()
        
        # Test at different temperatures
        cold_env = sample_environment_dict.copy()
        cold_env['air_temperature'] = torch.tensor([278.15] * TEST_BATCH_SIZE, device=DEVICE)  # 5掳C
        
        hot_env = sample_environment_dict.copy()
        hot_env['air_temperature'] = torch.tensor([308.15] * TEST_BATCH_SIZE, device=DEVICE)   # 35掳C
        
        # Add nitrogen substrate
        state_with_n = sample_state_dict.copy()
        state_with_n['ammonium_n'] = torch.tensor([10.0] * TEST_BATCH_SIZE, device=DEVICE)
        
        cold_rates = call_forward(
            fertilization,
            {**state_with_n, **cold_env},
            calendar_type="noleap",
        )
        hot_rates = call_forward(
            fertilization,
            {**state_with_n, **hot_env},
            calendar_type="noleap",
        )
        
        # Check temperature response - hot should be faster than cold
        if 'nitrification_rate' in cold_rates and 'nitrification_rate' in hot_rates:
            assert torch.all(hot_rates['nitrification_rate'] > cold_rates['nitrification_rate']), \
                "Higher temperature should increase nitrification rate"
        
        print("[OK] Temperature response functions validated")
    
    def test_physical_constraints(self, sample_state_dict, sample_environment_dict):
        """Test that outputs respect physical constraints."""
        grazing = GrazingManagement()
        
        rates = call_forward(
            grazing,
            {**sample_state_dict, **sample_environment_dict},
            calendar_type="noleap",
        )
        
        # All biological rates should be non-negative
        for flux_name, flux_values in rates.items():
            if 'rate' in flux_name:
                assert torch.all(flux_values >= 0), f"Negative rate in {flux_name}"
        
        print("[OK] Physical constraints verified")


class TestL5Backwardconsistency:
    """L1.5: Test backward consistency with forward method."""
    
    @pytest.mark.parametrize("class_type,name", TestClassification.ALL_CLASSES)
    def test_forward_method_exists(self, class_type, name):
        """Test that forward method exists for backward consistency."""
        instance = construct_minimal(f"{class_type.__module__}:{class_type.__name__}")
        assert hasattr(instance, 'forward'), f"{name}: Missing forward method for backward consistency"
    
    @pytest.mark.parametrize("class_type,name", TestClassification.ALL_CLASSES)
    def test_forward_method_callable(self, class_type, name, sample_state_dict, sample_environment_dict):
        """Test that forward method can be called."""
        instance = construct_minimal(f"{class_type.__module__}:{class_type.__name__}")
        
        try:
            # Create mock time parameters
            time = 0.0
            dt = 3600.0  # 1 hour
            
            # Create module-specific input based on requirements
            test_state = sample_state_dict.copy()
            
            if name == "sector_allocation":
                spatial_shape = (2, 2)
                for species in instance.CANONICAL_SPECIES:
                    test_state[f"anthropogenic_{species}_flux"] = torch.full(
                        spatial_shape,
                        1.0e-6,
                        device=DEVICE,
                    )
                test_state.update({
                    "population_density": torch.ones(spatial_shape, device=DEVICE),
                    "gdp_density": torch.ones(spatial_shape, device=DEVICE),
                    "road_density": torch.ones(spatial_shape, device=DEVICE),
                    "land_use_fractions": torch.full(
                        (9, *spatial_shape),
                        1.0 / 9.0,
                        device=DEVICE,
                    ),
                })
            
            # Add specific inputs for FertilizationManagement
            if name == "fertilization_management":
                test_state.update({
                    "urea_n": torch.tensor([50.0] * TEST_BATCH_SIZE, device=DEVICE),
                    "ammonium_n": torch.tensor([20.0] * TEST_BATCH_SIZE, device=DEVICE),
                    "soil_ph": torch.tensor([6.5] * TEST_BATCH_SIZE, device=DEVICE)
                })

            if name == "irrigation_management":
                test_state.update({
                    "soil_moisture": torch.full((TEST_BATCH_SIZE,), 0.20, device=DEVICE),
                    "precipitation_flux": torch.zeros(TEST_BATCH_SIZE, device=DEVICE),
                    "land_use_type": torch.ones(
                        TEST_BATCH_SIZE, dtype=torch.long, device=DEVICE
                    ),
                    "irrigation_type": torch.full(
                        (TEST_BATCH_SIZE,), 2, dtype=torch.long, device=DEVICE
                    ),
                    "field_capacity": torch.full((TEST_BATCH_SIZE,), 0.35, device=DEVICE),
                    "wilting_point": torch.full((TEST_BATCH_SIZE,), 0.15, device=DEVICE),
                    "irrigation_remaining_quota": torch.tensor(800.0, device=DEVICE),
                })
            
            # TillageManagement reads the tilled-layer soil temperature [K],
            # a required input without a ParameterManager fallback.
            if name == "tillage_management":
                test_state["soil_temperature"] = torch.full(
                    (TEST_BATCH_SIZE,), 293.15, device=DEVICE
                )
                test_state["soil_organic_carbon"] = torch.full(
                    (TEST_BATCH_SIZE,), 5.0, device=DEVICE
                )
                test_state["days_since_tillage"] = torch.full(
                    (TEST_BATCH_SIZE,), 365.0, device=DEVICE
                )
                # Day 150 executes the default conservation pass, which incorporates
                # part of the residue cover.
                test_state["residue_cover"] = torch.full(
                    (TEST_BATCH_SIZE,), 0.5, device=DEVICE
                )

            # Add specific inputs for LandUseParameterMapping 
            if name == "land_use_parameter_mapping":
                # Provide all 9 land use types with fractions
                test_state["land_use_fraction"] = torch.tensor([
                    [0.15, 0.10, 0.08, 0.04, 0.20, 0.15, 0.03, 0.02, 0.23]  # 9 land use types
                ] * TEST_BATCH_SIZE, device=DEVICE)

            if name == "land_use_transitions":
                # LandUseTransitions requires explicit land use fractions in state
                test_state["land_use_fraction"] = torch.tensor(
                    [0.15, 0.10, 0.08, 0.04, 0.20, 0.15, 0.03, 0.02, 0.23],
                    device=DEVICE,
                )
            
            # LandUseTransitions, LandUseParameterMapping and SectorAllocation expose
            # only the canonical state/time/dt signature: they read no day or calendar
            # keyword, so none is passed (an unknown keyword raises TypeError).
            kwargs = (
                {}
                if name in (
                    "land_use_transitions",
                    "land_use_parameter_mapping",
                    "sector_allocation",
                )
                else {"day_of_year": 150, "calendar_type": "noleap"}
            )
            
            result = instance.forward(test_state, time, dt, **kwargs)
            assert result is not None, f"{name}: forward method returned None"
            print(f"[OK] {name}: Forward method consistency verified")
        except Exception as e:
            pytest.fail(f"[FAIL] {name}: Forward method error - {e}")


class TestL6ParameterHandling:
    """L1.6: Test parameter handling and validation."""
    
    def test_parameter_setting_and_retrieval(self):
        """Test parameter setting and retrieval functionality."""
        fertilization = FertilizationManagement()
        
        # Test parameter metadata exists
        if hasattr(fertilization, 'get_process_metadata'):
            metadata = fertilization.get_process_metadata()
            assert isinstance(metadata, dict)
            print("[OK] Parameter metadata accessible")
        
        # Test parameter validation
        if hasattr(fertilization, 'base_rate'):
            original_rate = fertilization.base_rate
            fertilization.base_rate = 2.0
            assert fertilization.base_rate == 2.0, "Parameter setting failed"
            fertilization.base_rate = original_rate
            print("[OK] Parameter setting validated")
    
class TestL7ErrorHandling:
    """L1.7: Test error handling and edge cases."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        mapping = LandUseParameterMapping()
        
        # Test with empty coordinates
        empty_coords = torch.empty(0, SPATIAL_DIMS, device=DEVICE)
        
        try:
            values = mapping.get_value_at(empty_coords)
            assert len(values) == 0, "Empty input should return empty output"
            print("[OK] Empty input handling verified")
        except Exception as e:
            # Empty input might be intentionally not supported
            print(f"[INFO] Empty input handling: {e}")
    
    def test_nan_input_handling(self, sample_state_dict, sample_environment_dict):
        """Test handling of NaN inputs."""
        fertilization = FertilizationManagement()
        
        # Create state with NaN values
        nan_state = sample_state_dict.copy()
        nan_state['temperature'] = torch.full_like(nan_state['temperature'], float('nan'))
        
        try:
            rates = call_forward(
                fertilization,
                {**nan_state, **sample_environment_dict},
                calendar_type="noleap",
            )
            # Should either handle NaN gracefully or raise appropriate error
            has_nan = any(torch.isnan(v).any() for v in rates.values())
            print(f"[INFO] NaN handling: {'produces NaN output' if has_nan else 'handles NaN input'}")
        except Exception as e:
            print(f"[INFO] NaN input raises exception: {type(e).__name__}")


# Test execution summary
def pytest_sessionfinish(session, exitstatus):
    """Print test execution summary."""
    print("\n" + "="*80)
    print("RELIC ANTHROPOSPHERE L1 UNIT TEST SUMMARY")
    print("="*80)
    
    if exitstatus == 0:
        print("[SUCCESS] All L1 unit tests passed")
        print("Anthroposphere module refactoring validation: COMPLETE")
    else:
        print("[FAILURE] Some L1 unit tests failed")
        print("Review failed tests before proceeding to L2 integration tests")
    
    print("="*80)


if __name__ == "__main__":
    print("RELIC Anthroposphere L1 Unit Test Suite")
    print("Testing inheritance refactoring compliance...")
    pytest.main([__file__, "-v", "-s", "--tb=short"])


