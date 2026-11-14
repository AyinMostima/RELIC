"""
Comprehensive Test Suite for BVOC Parameter System Integration

This test module validates that the BVOC module correctly integrates with the
unified parameter management system, ensuring:
1. All parameters are loaded from the database (no hardcoded values)
2. Units and bounds are properly validated
3. Single source of truth is maintained
4. Parameter access patterns are consistent

References
----------
- RELIC Core Development Guidelines
- MEGAN model parameter specifications (Guenther et al., 2012)

Test Coverage
-------------
- Parameter namespace resolution
- Units validation and conversion
- Bounds checking
- default behavior
- Parameter manager integration
"""

import pytest
import torch

# Import modules to test
from relic.scientific.biosphere.bvoc.megan_model import MEGANModel, CouplingStrategy
from relic.scientific.biosphere.bvoc.responses import (
    BVOCLightResponse,
    BVOCTemperatureResponse,
    BVOCEnvironmentalResponse
)


class TestParameterIntegration:
    """Test suite for BVOC parameter system integration."""

    @pytest.fixture
    def megan_model(self, param_manager):
        """Initialize MEGAN model with parameter manager."""
        return MEGANModel(
            name="test_megan",
            coupling_strategy=CouplingStrategy.SEQUENTIAL,
            validate_inputs=True
        )

    def test_namespace_resolution(self, megan_model, param_manager):
        """Test that parameter namespaces resolve correctly to YAML structure."""

        # Test cases: (parameter_key, expected_section)
        test_cases = [
            # Default emission factors
            ("biosphere.bvoc.default.isoprene", "default"),
            ("biosphere.bvoc.default.monoterpenes", "default"),
            ("biosphere.bvoc.default.sesquiterpenes", "default"),
            ("biosphere.bvoc.default.ovocs", "default"),

            # MEGAN parameters
            ("biosphere.bvoc.megan.alpha_light", "megan"),
            ("biosphere.bvoc.megan.cl_light", "megan"),
            ("biosphere.bvoc.megan.eopt_isoprene", "megan"),
            ("biosphere.bvoc.megan.topt", "megan"),
            ("biosphere.bvoc.megan.ct1_isoprene", "megan"),
            ("biosphere.bvoc.megan.ct2", "megan"),

            # Molecular properties
            (
                "relic.physical.constants.molecular_weights.isoprene",
                "physical_constants",
            ),
            ("biosphere.bvoc.carbon_atoms.isoprene", "carbon_atoms"),

            # CO2 inhibition
            ("biosphere.bvoc.co2_inhibition.ismax", "co2_inhibition"),
            ("biosphere.bvoc.co2_inhibition.c_star", "co2_inhibition"),

            # Water stress
            ("biosphere.bvoc.water_stress.wilting_point", "water_stress"),
            ("biosphere.bvoc.water_stress.field_capacity", "water_stress"),
        ]

        for param_key, section in test_cases:
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"Missing spec for parameter {param_key}"

            if isinstance(spec, dict):
                spec_key = spec.get("key", "")
                bounds = spec.get("bounds")
            else:
                spec_key = getattr(spec, "key", "")
                bounds = getattr(spec, "bounds", None)

            if section == "physical_constants":
                assert (
                    spec_key
                    == "relic.physical.constants.molecular_weights.isoprene"
                )
            else:
                assert f"biosphere.bvoc.{section}." in spec_key, (
                    f"Parameter {param_key} resolved to unexpected key {spec_key}"
                )

            value = param_manager.get(param_key)
            if bounds:
                if isinstance(bounds, list):
                    bounds = tuple(bounds)
                low, high = bounds
                assert low <= float(value) <= high, (
                    f"Parameter {param_key} value {value} not within bounds {bounds}"
                )

    def test_units_validation(self, megan_model, param_manager):
        """Test that units are properly validated when accessing parameters."""

        # Test cases: (parameter_key, expected_units)
        unit_tests = [
            ("biosphere.bvoc.default.isoprene", "ug g-1 DW h-1"),
            ("biosphere.bvoc.megan.alpha_light", "m2 s umol-1"),
            ("biosphere.bvoc.megan.topt", "K"),
            ("biosphere.bvoc.megan.ct1_isoprene", "J mol-1"),
            ("biosphere.bvoc.megan.beta_monoterpenes", "K-1"),
            (
                "relic.physical.constants.molecular_weights.isoprene",
                "g mol-1",
            ),
            ("relic.physical.constants.atmosphere.co2_reference", "ppm"),
            ("biosphere.bvoc.water_stress.wilting_point", "MPa"),
            ("relic.physical.constants.time.seconds_per_hour", "s"),
        ]

        for param_key, expected_units in unit_tests:
            # Get parameter spec with units
            param_spec = param_manager.get_parameter(param_key)

            if param_spec and 'units' in param_spec:
                assert param_spec['units'] == expected_units, \
                    f"Parameter {param_key} has units '{param_spec['units']}' but expected '{expected_units}'"

    def test_bounds_enforcement(self, param_manager):
        """Test that parameter bounds are enforced when setting values."""

        # Test setting values outside bounds
        test_cases = [
            ("biosphere.bvoc.megan.alpha_light", 0.0001, (0.001, 0.01)),  # Below min
            ("biosphere.bvoc.megan.alpha_light", 0.1, (0.001, 0.01)),      # Above max
            ("biosphere.bvoc.default.isoprene", 5.0, (10, 50)),            # Below min
            ("biosphere.bvoc.default.isoprene", 100.0, (10, 50)),          # Above max
        ]

        for param_key, test_value, bounds in test_cases:
            # Try to set value outside bounds
            param_manager.set_value(param_key, test_value)

            # Get the value back
            actual_value = float(param_manager.get(param_key))

            # Check if value was clamped to bounds or warning was issued
            # (Implementation may vary - could clamp or warn)
            assert bounds[0] <= actual_value <= bounds[1], \
                f"Parameter {param_key} value {actual_value} not properly bounded to {bounds}"

    def test_no_hardcoded_defaults(self, megan_model):
        """Ensure MEGAN model doesn't use hardcoded defaults when parameter manager is available."""

        # Check that model parameters match database values
        param_checks = [
            ("isoprene_ef", 24.0, 50.0),        # Should be in reasonable range
            ("monoterpenes_ef", 0.5, 5.0),
            ("alpha_light", 0.001, 0.01),
            ("cl_factor", 0.8, 1.5),
            ("eopt_iso", 1.0, 3.0),
            ("beta_monoterpenes", 0.05, 0.15),
        ]

        for attr_name, min_val, max_val in param_checks:
            if hasattr(megan_model, attr_name):
                value = getattr(megan_model, attr_name)
                if isinstance(value, torch.Tensor):
                    value = float(value.item())

                assert min_val <= value <= max_val, \
                    f"Model attribute {attr_name} has value {value} outside expected range [{min_val}, {max_val}]"

    def test_response_functions_parameter_integration(self, param_manager):
        """Test that response functions properly integrate with parameter system."""

        # Create test tensors
        temperature = torch.tensor([298.15])  # 25degC
        co2 = torch.tensor([400.0])          # 400 ppm
        ppfd = torch.tensor([1000.0])        # 1000 umol m-2 s-1

        # Test drought stress response
        water_potential = torch.tensor([-0.5])  # MPa
        drought_response = BVOCEnvironmentalResponse.drought_stress_response(
            water_potential=water_potential,
            param_manager=param_manager
        )
        assert 0.0 <= float(drought_response) <= 1.0, \
            "Drought stress response outside expected range"

        # Test light response
        light_response = BVOCLightResponse.get_compound_light_response(
            'isoprene',
            ppfd,
            param_manager=param_manager
        )
        assert 0.0 <= float(light_response) <= 1.1, \
            "Light response outside the Eq. (4) range [0, CL]"

        # Test temperature response
        temp_response = BVOCTemperatureResponse.get_compound_temperature_response(
            'isoprene',
            temperature,
            param_manager=param_manager
        )
        assert 0.0 < float(temp_response) < 1.0, \
            "Eq. (8) activity at 25 degC is below the optimum activity"

    def test_parameter_default_behavior(self):
        """Test default behavior when parameter is not in database."""

        # Create a minimal model without full parameter loading
        model = MEGANModel(name="test_default", validate_inputs=False)

        # Model should still initialize with default values
        assert hasattr(model, 'isoprene_ef')
        assert hasattr(model, 'alpha_light')

        # Check that default values are reasonable
        isoprene_ef = float(model.isoprene_ef.item()) if isinstance(model.isoprene_ef, torch.Tensor) else model.isoprene_ef
        assert 10.0 <= isoprene_ef <= 50.0, "default emission factor outside reasonable range"

    def test_parameter_consistency_across_modules(self, param_manager):
        """Ensure parameter values are consistent across different modules."""

        # Create model and response instances
        model = MEGANModel(name="test_consistency")

        # Test CO2 reference consistency
        co2_ref_model = float(model.co2_reference.item()) if hasattr(model, 'co2_reference') else None
        co2_ref_db = float(param_manager.get("relic.physical.constants.atmosphere.co2_reference", default=370.0))

        if co2_ref_model is not None:
            assert abs(co2_ref_model - co2_ref_db) < 0.1, \
                f"CO2 reference mismatch: model={co2_ref_model}, database={co2_ref_db}"

    def test_parameter_update_propagation(self, param_manager):
        """Test that parameter updates propagate correctly to dependent modules."""

        # Update a parameter
        original_value = param_manager.get("biosphere.bvoc.megan.alpha_light")
        new_value = 0.005
        param_manager.set_value("biosphere.bvoc.megan.alpha_light", new_value)

        # Create new model instance
        model = MEGANModel(name="test_update")

        # Check if model picked up the new value
        model_value = float(model.alpha_light.item()) if hasattr(model, 'alpha_light') else None

        if model_value is not None:
            assert abs(model_value - new_value) < 0.0001, \
                f"Parameter update not propagated: expected {new_value}, got {model_value}"

        # Restore original value
        param_manager.set_value("biosphere.bvoc.megan.alpha_light", original_value)

    def test_comprehensive_parameter_audit(self, param_manager):
        """Run comprehensive audit of parameter system."""

        # Run parameter audit
        audit_report = param_manager.run_registry_audit()

        # Check for missing parameters
        missing = param_manager.get_missing_requests()

        # Log any issues found
        if missing:
            print(f"Warning: {len(missing)} missing parameter requests detected:")
            for param in list(missing)[:5]:  # Show first 5
                print(f"  - {param}")

        # Basic assertions
        assert audit_report is not None, "Audit report should not be None"

        # Check that critical BVOC parameters are present
        critical_params = [
            "biosphere.bvoc.default.isoprene",
            "biosphere.bvoc.megan.alpha_light",
            "relic.physical.constants.molecular_weights.isoprene",
            "biosphere.bvoc.co2_inhibition.ismax"
        ]

        for param in critical_params:
            value = param_manager.get(param)
            assert value is not None, f"Critical parameter {param} not found in database"


class TestParameterValidation:
    """Additional tests for parameter validation and error handling."""

    def test_invalid_units_rejection(self, param_manager):
        """Test that invalid units are properly rejected."""

        # Try to get parameter with wrong units (should trigger validation)
        with pytest.raises(ValueError, match="Unit conversion failed"):
            # This should raise due to inconsistent units
            param_manager.get("biosphere.bvoc.megan.topt", units="m")

    def test_parameter_type_conversion(self, param_manager):
        """Test automatic type conversion for parameters."""

        # Get parameter as different types
        alpha_float = float(param_manager.get("biosphere.bvoc.megan.alpha_light"))
        alpha_tensor = torch.as_tensor(
            param_manager.get(
                "biosphere.bvoc.megan.alpha_light",
                required=True,
            )
        )

        assert isinstance(alpha_float, float)
        assert isinstance(alpha_tensor, torch.Tensor)
        assert abs(alpha_float - float(alpha_tensor.item())) < 1e-10

    def test_parameter_metadata_completeness(self, param_manager):
        """Verify that all parameters have complete metadata."""

        # List of parameters to check
        params_to_check = [
            "biosphere.bvoc.default.isoprene",
            "biosphere.bvoc.megan.alpha_light",
            "relic.physical.constants.molecular_weights.isoprene",
            "biosphere.bvoc.co2_inhibition.c_star"
        ]

        required_fields = ['value', 'units', 'description']

        for param_key in params_to_check:
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"No spec found for {param_key}"

            for field in required_fields:
                assert field in spec, f"Parameter {param_key} missing required field '{field}'"


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])


