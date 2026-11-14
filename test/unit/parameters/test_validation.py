"""
L1 Unit Tests for Parameter Validation Component

Tests for the ParameterValidator class and validation logic.
"""

import pytest
import torch

import relic.data.standards.dimensionality as dimensionality_module
from relic.parameters.validation import (
    ParameterValidator,
    validate_parameter_value,
    validate_parameter_relationships,
    check_physical_consistency
)
from relic.parameters.types import (
    ValidationLevel,
    ValidationResult
)
from relic.parameters import ParameterSpec, ParameterCategory


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating validation results."""
        # Valid result
        valid_result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor issue"],
        )
        assert valid_result.is_valid
        assert len(valid_result.errors) == 0
        assert len(valid_result.warnings) == 1
        # The result records no validation level: the field had no reader and
        # aggregate() wrote a fixed STANDARD for any mix of levels.
        assert "level" not in ValidationResult.__dataclass_fields__
        
        # Invalid result
        invalid_result = ValidationResult(
            is_valid=False,
            errors=["Critical error", "Another error"],
            warnings=[],
        )
        assert not invalid_result.is_valid
        assert len(invalid_result.errors) == 2
        assert len(invalid_result.warnings) == 0
    
    def test_validation_result_aggregation(self):
        """Test aggregating multiple validation results."""
        results = [
            ValidationResult(True, [], ["warning1"]),
            ValidationResult(True, [], ["warning2"]),
            ValidationResult(False, ["error1"], [])
        ]
        
        # Aggregate should be invalid if any result is invalid
        aggregated = ValidationResult.aggregate(results)
        assert not aggregated.is_valid
        assert "error1" in aggregated.errors
        assert "warning1" in aggregated.warnings
        assert "warning2" in aggregated.warnings


class TestParameterValidator:
    """Test ParameterValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a parameter validator instance."""
        return ParameterValidator()
    
    @pytest.fixture
    def sample_specs(self):
        """Create sample parameter specifications."""
        return {
            "temperature": ParameterSpec(
                key="env.temperature",
                value=298.15,  # 25掳C in Kelvin
                units="K",
                bounds=(200.0, 350.0),
                category=ParameterCategory.ENVIRONMENTAL,
                description="Temperature in Kelvin", dtype="float32"
            ),
            "pressure": ParameterSpec(
                key="env.pressure",
                value=101325.0,  # 1 atm in Pa
                units="Pa",
                bounds=(50000.0, 200000.0),
                category=ParameterCategory.ENVIRONMENTAL,
                description="Pressure in Pascals", dtype="float32"
            ),
            "humidity": ParameterSpec(
                key="env.humidity",
                value=0.6,  # 60% relative humidity
                units="dimensionless",
                bounds=(0.0, 1.0),
                category=ParameterCategory.ENVIRONMENTAL,
                description="Relative humidity", dtype="float32"
            )
        }
    
    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator.validation_level == ValidationLevel.STANDARD
        assert len(validator.custom_validators) == 0
        
        # Test with custom level
        strict_validator = ParameterValidator(level=ValidationLevel.STRICT)
        assert strict_validator.validation_level == ValidationLevel.STRICT
    
    def test_validate_single_parameter(self, validator, sample_specs):
        """Test validating a single parameter."""
        temp_spec = sample_specs["temperature"]
        
        # Valid parameter
        result = validator.validate_value(
            temp_spec.key,
            temp_spec.value,
            temp_spec
        )
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Value outside bounds
        result_invalid = validator.validate_value(
            temp_spec.key,
            400.0,  # Above upper bound
            temp_spec
        )
        assert not result_invalid.is_valid
        assert any("bounds" in err.lower() for err in result_invalid.errors)
    
    def test_validate_multiple_parameters(self, validator, sample_specs):
        """Test validating multiple parameters."""
        errors = validator.validate(sample_specs)
        
        # All sample specs should be valid
        assert len(errors) == 0
        
        # Add invalid parameter
        invalid_spec = ParameterSpec(
            key="invalid.param",
            value=-100.0,
            bounds=(0.0, 100.0),
            category=ParameterCategory.PHYSICAL,
            description="Invalid parameter", units="dimensionless", dtype="float32"
        )
        specs_with_invalid = {**sample_specs, "invalid": invalid_spec}
        
        errors = validator.validate(specs_with_invalid)
        assert len(errors) > 0
        assert any("invalid.param" in str(err) for err in errors)
    
    def test_validation_levels(self, validator):
        """Test different validation levels."""
        spec = ParameterSpec(
            key="test.param",
            value=50.0,
            bounds=(0.0, 100.0),
            units="meters",
            category=ParameterCategory.PHYSICAL,
            description="Test parameter", dtype="float32"
        )
        
        # NONE level - no validation
        validator.validation_level = ValidationLevel.NONE
        errors = validator.validate({"test": spec})
        assert len(errors) == 0
        
        # BASIC level - type and bounds only
        validator.validation_level = ValidationLevel.BASIC
        spec.value = 150.0  # Outside bounds
        errors = validator.validate({"test": spec})
        assert len(errors) > 0
        
        # STANDARD level - includes units
        validator.validation_level = ValidationLevel.STANDARD
        spec.value = 50.0  # Valid value
        spec.units = "invalid_unit"
        validator.clear_cache()  # Clear cache after changing spec
        errors = validator.validate({"test": spec})
        assert len(errors) > 0  # Should catch invalid unit
        
        # STRICT level - all checks including relationships
        validator.validation_level = ValidationLevel.STRICT
        errors = validator.validate({"test": spec})
        assert len(errors) > 0
    
    def test_type_validation(self, validator):
        """Test parameter type validation."""
        # Float parameter
        float_spec = ParameterSpec(
            key="float.param",
            value=3.14,
            dtype="float32",
            category=ParameterCategory.COMPUTATIONAL,
            description="Float parameter", units="dimensionless", bounds=(-float("inf"), float("inf"))
        )
        result = validator.validate_value("float.param", 3.14, float_spec)
        assert result.is_valid
        
        # Integer parameter
        int_spec = ParameterSpec(
            key="int.param",
            value=42,
            dtype="int32",
            category=ParameterCategory.COMPUTATIONAL,
            description="Integer parameter", units="dimensionless", bounds=(-float("inf"), float("inf"))
        )
        result = validator.validate_value("int.param", 42, int_spec)
        assert result.is_valid
        
        # Type mismatch
        result = validator.validate_value("int.param", 3.14, int_spec)
        assert len(result.warnings) > 0 or len(result.errors) > 0
    
    def test_bounds_validation(self, validator):
        """Test parameter bounds validation."""
        spec = ParameterSpec(
            key="bounded.param",
            value=50.0,
            bounds=(0.0, 100.0),
            category=ParameterCategory.PHYSICAL,
            description="Bounded parameter", units="dimensionless", dtype="float32"
        )
        
        # Within bounds
        result = validator.validate_value("bounded.param", 50.0, spec)
        assert result.is_valid
        
        # At lower bound
        result = validator.validate_value("bounded.param", 0.0, spec)
        assert result.is_valid
        
        # At upper bound
        result = validator.validate_value("bounded.param", 100.0, spec)
        assert result.is_valid
        
        # Below lower bound
        result = validator.validate_value("bounded.param", -1.0, spec)
        assert not result.is_valid
        
        # Above upper bound
        result = validator.validate_value("bounded.param", 101.0, spec)
        assert not result.is_valid
        
        # NaN value
        result = validator.validate_value("bounded.param", float('nan'), spec)
        assert not result.is_valid
    
    def test_unit_validation(self, validator):
        """Test parameter unit validation."""
        # Valid CF-1.8 units
        valid_units = [("K", "K"), ("Pa", "Pa"), ("m_per_s", "m/s"), ("kg_per_m3", "kg/m3"), ("dimensionless", "dimensionless"), ("unity", "1")]
        for unit_key, unit in valid_units:
            spec = ParameterSpec(
                key=f"unit.{unit_key}",
                value=1.0,
                units=unit,
                category=ParameterCategory.PHYSICAL,
                description=f"Parameter with unit {unit}", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
            result = validator.validate_value(spec.key, spec.value, spec)
            assert result.is_valid or len(result.warnings) > 0
        
        # Invalid units
        invalid_units = [("invalid", "invalid"), ("underscore", "m_per_s"), ("empty", "")]
        for unit_key, unit in invalid_units:
            spec = ParameterSpec(
                key=f"unit.{unit_key}",
                value=1.0,
                units=unit,
                category=ParameterCategory.PHYSICAL,
                description=f"Parameter with invalid unit {unit}", dtype="float32", bounds=(-float("inf"), float("inf"))
            )
            validator.validation_level = ValidationLevel.STRICT
            result = validator.validate_value(spec.key, spec.value, spec)
            assert len(result.errors) > 0 or len(result.warnings) > 0

    def test_unit_validation_delegates_to_canonical_dimensionality_owner(
        self,
        validator,
        monkeypatch,
    ):
        calls = []

        def fake_dimension_category(unit):
            calls.append(("category", unit))
            return "length"

        def fake_dimension_compatibility(unit, declared_dimension):
            calls.append(("compatibility", unit, declared_dimension))
            return True

        monkeypatch.setattr(
            dimensionality_module,
            "dimension_category_from_unit",
            fake_dimension_category,
        )
        monkeypatch.setattr(
            dimensionality_module,
            "is_declared_dimension_compatible",
            fake_dimension_compatibility,
        )
        validator.validation_level = ValidationLevel.STRICT
        spec = ParameterSpec(
            key="unit.delegated",
            value=1.0,
            units="custom_unit_expression",
            dimension="length",
            category=ParameterCategory.PHYSICAL,
            description="Delegation contract", dtype="float32", bounds=(-float("inf"), float("inf")),
        )

        result = validator.validate_value(spec.key, spec.value, spec)

        assert result.is_valid
        assert calls == [
            ("category", "custom_unit_expression"),
            ("compatibility", "custom_unit_expression", "length"),
        ]

    @pytest.mark.parametrize(
        ("units", "dimension"),
        [
            ("s day-1", "dimensionless"),
            ("g C m-2", "mass_areal_density"),
            ("umol photons m-2 s-1", "amount_flux"),
            ("degC_day", "thermal_time"),
        ],
    )
    def test_unit_validation_accepts_canonical_composite_dimensions(
        self,
        validator,
        units,
        dimension,
    ):
        validator.validation_level = ValidationLevel.STRICT
        spec = ParameterSpec(
            key=f"unit.{dimension}",
            value=1.0,
            units=units,
            dimension=dimension,
            category=ParameterCategory.PHYSICAL,
            description="Canonical dimensionality contract", dtype="float32", bounds=(-float("inf"), float("inf")),
        )

        result = validator.validate_value(spec.key, spec.value, spec)

        assert result.is_valid
        assert result.errors == []

    def test_unit_validation_rejects_inconsistent_declared_dimension(
        self,
        validator,
    ):
        validator.validation_level = ValidationLevel.STRICT
        spec = ParameterSpec(
            key="unit.time_ratio",
            value=1.0,
            units="s day-1",
            dimension="frequency",
            category=ParameterCategory.PHYSICAL,
            description="Incorrect dimension declaration", dtype="float32", bounds=(-float("inf"), float("inf")),
        )

        result = validator.validate_value(spec.key, spec.value, spec)

        assert not result.is_valid
        assert any("inconsistent" in error.lower() for error in result.errors)

    @pytest.mark.parametrize(
        ("units", "declared_dimension"),
        [
            ("g C m-2", "mass_per_area"),
            ("umol m-2 s-1", "molar_flux"),
            ("m3 m-3", "volume_fraction"),
            ("ug g-1 h-1", "emission_factor"),
            ("mol e- mol-1 photon", "quantum_yield"),
        ],
    )
    def test_unit_validation_compares_declared_semantics_by_si_vector(
        self,
        validator,
        units,
        declared_dimension,
    ):
        validator.validation_level = ValidationLevel.STRICT
        spec = ParameterSpec(
            key=f"unit.{declared_dimension}",
            value=1.0,
            units=units,
            dimension=declared_dimension,
            category=ParameterCategory.PHYSICAL,
            description="SI vector compatibility contract", dtype="float32", bounds=(-float("inf"), float("inf")),
        )

        result = validator.validate_value(spec.key, spec.value, spec)

        assert result.is_valid
        assert result.errors == []

    @pytest.mark.parametrize(
        ("units", "declared_dimension"),
        [
            ("K", "configuration"),
            ("J mol-1", "energy"),
            ("ppm", "concentration"),
            ("mm hr-1", "mass_flux"),
            ("K day-0.5", "thermal_time"),
            ("dimensionless", "temperature"),
            ("Pa kPa-1", "unit_conversion"),
        ],
    )
    def test_unit_validation_rejects_ambiguous_dimension_labels(
        self,
        validator,
        units,
        declared_dimension,
    ):
        validator.validation_level = ValidationLevel.STRICT
        spec = ParameterSpec(
            key=f"unit.invalid.{declared_dimension}",
            value=1.0,
            units=units,
            dimension=declared_dimension,
            category=ParameterCategory.PHYSICAL,
            description="Invalid dimension declaration", dtype="float32", bounds=(-float("inf"), float("inf")),
        )

        result = validator.validate_value(spec.key, spec.value, spec)

        assert not result.is_valid
        assert any("inconsistent" in error.lower() for error in result.errors)
    
    def test_custom_validators(self, validator):
        """Test adding custom validators."""
        # Define custom validator
        def positive_validator(key: str, value: float, spec: ParameterSpec) -> ValidationResult:
            if value <= 0:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"{key} must be positive"],
                    warnings=[],
                )
            return ValidationResult(True, [], [])
        
        # Add custom validator
        validator.add_custom_validator("positive", positive_validator)
        assert "positive" in validator.custom_validators
        
        # Test with positive value
        spec = ParameterSpec(
            key="positive.param",
            value=10.0,
            category=ParameterCategory.PHYSICAL,
            description="Must be positive", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        result = validator.validate_value("positive.param", 10.0, spec)
        assert result.is_valid
        
        # Test with negative value
        result = validator.validate_value("positive.param", -10.0, spec)
        assert not result.is_valid
        assert any("positive" in err.lower() for err in result.errors)
    
    def test_tensor_validation(self, validator):
        """Test validation of tensor parameters."""
        # Valid tensor
        tensor_value = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        spec = ParameterSpec(
            key="tensor.param",
            value=tensor_value,
            dtype="float32",
            category=ParameterCategory.COMPUTATIONAL,
            description="Tensor parameter", units="dimensionless", bounds=(-float("inf"), float("inf"))
        )
        result = validator.validate_value("tensor.param", tensor_value, spec)
        assert result.is_valid
        
        # Tensor with NaN
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        result = validator.validate_value("tensor.param", nan_tensor, spec)
        assert not result.is_valid
        assert any("nan" in err.lower() for err in result.errors)
        
        # Tensor with Inf
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        result = validator.validate_value("tensor.param", inf_tensor, spec)
        assert not result.is_valid
        assert any("inf" in err.lower() for err in result.errors)


class TestValidationFunctions:
    """Test standalone validation functions."""
    
    def test_validate_parameter_value(self):
        """Test validate_parameter_value function."""
        # Valid value
        result = validate_parameter_value(
            value=50.0,
            bounds=(0.0, 100.0),
            dtype="float32"
        )
        assert result.is_valid
        
        # Invalid bounds
        result = validate_parameter_value(
            value=150.0,
            bounds=(0.0, 100.0),
            dtype="float32"
        )
        assert not result.is_valid
        
        # Type check
        result = validate_parameter_value(
            value="not_a_number",
            bounds=None,
            dtype="float32"
        )
        assert not result.is_valid
    
    def test_validate_parameter_relationships(self):
        """Test validate_parameter_relationships function."""
        specs = {
            "water.field_capacity": ParameterSpec(
                key="water.field_capacity",
                value=0.3,
                units="m3/m3",
                bounds=(0.0, 1.0),
                category=ParameterCategory.PHYSICAL,
                description="Field capacity", dtype="float32"
            ),
            "water.wilting_point": ParameterSpec(
                key="water.wilting_point",
                value=0.1,
                units="m3/m3",
                bounds=(0.0, 1.0),
                category=ParameterCategory.PHYSICAL,
                description="Wilting point", dtype="float32"
            ),
            "water.saturation": ParameterSpec(
                key="water.saturation",
                value=0.5,
                units="m3/m3",
                bounds=(0.0, 1.0),
                category=ParameterCategory.PHYSICAL,
                description="Saturation", dtype="float32"
            )
        }
        
        # Valid relationships
        result = validate_parameter_relationships(specs)
        assert result.is_valid
        
        # Invalid: wilting point > field capacity
        specs["water.wilting_point"].value = 0.4
        result = validate_parameter_relationships(specs)
        assert not result.is_valid
        assert any("field capacity" in err.lower() for err in result.errors)
        
        # Invalid: field capacity > saturation
        specs["water.wilting_point"].value = 0.1
        specs["water.field_capacity"].value = 0.6
        result = validate_parameter_relationships(specs)
        assert not result.is_valid
        assert any("saturation" in err.lower() for err in result.errors)
    
    def test_check_physical_consistency(self):
        """Test check_physical_consistency function."""
        specs = {
            "temperature": ParameterSpec(
                key="env.temperature",
                value=298.15,  # 25掳C
                units="K",
                bounds=(200.0, 350.0),
                category=ParameterCategory.ENVIRONMENTAL,
                description="Temperature", dtype="float32"
            ),
            "pressure": ParameterSpec(
                key="env.pressure",
                value=101325.0,  # 1 atm
                units="Pa",
                bounds=(50000.0, 200000.0),
                category=ParameterCategory.ENVIRONMENTAL,
                description="Pressure", dtype="float32"
            ),
            "vapor_pressure": ParameterSpec(
                key="env.vapor_pressure",
                value=2000.0,  # Reasonable vapor pressure
                units="Pa",
                bounds=(0.0, 10000.0),
                category=ParameterCategory.ENVIRONMENTAL,
                description="Vapor pressure", dtype="float32"
            )
        }
        
        # Valid physical state
        result = check_physical_consistency(specs)
        assert result.is_valid
        
        # Invalid: vapor pressure > total pressure
        specs["vapor_pressure"].value = 150000.0
        result = check_physical_consistency(specs)
        assert not result.is_valid
        assert any("vapor" in err.lower() for err in result.errors)
        
        # Invalid: extreme temperature
        specs["vapor_pressure"].value = 2000.0
        specs["temperature"].value = 50.0  # Way too cold
        result = check_physical_consistency(specs)
        assert not result.is_valid


class TestValidationIntegration:
    """Test validation integration with parameter management."""
    
    def test_validation_with_parameter_manager(self):
        """Test validation integrated with parameter manager."""
        from relic.parameters import UnifiedParameterManager
        
        manager = UnifiedParameterManager()
        validator = ParameterValidator(level=ValidationLevel.STANDARD)
        
        # Register parameters
        specs = [
            ParameterSpec(
                key="valid.param",
                value=50.0,
                bounds=(0.0, 100.0),
                units="m/s",
                category=ParameterCategory.PHYSICAL,
                description="Valid parameter", dtype="float32"
            ),
            ParameterSpec(
                key="invalid.param",
                value=150.0,
                bounds=(0.0, 100.0),
                units="m/s",
                category=ParameterCategory.PHYSICAL,
                description="Invalid parameter", dtype="float32"
            )
        ]
        
        for spec in specs:
            manager.register_parameter(spec)
        
        # Validate through manager
        errors = manager.validate_parameters()
        assert len(errors) > 0
        assert any("invalid.param" in str(err) for err in errors)
    
    def test_validation_hooks(self):
        """Test validation hooks and callbacks."""
        validator = ParameterValidator()
        validation_called = []
        
        def validation_hook(key: str, value: float, spec: ParameterSpec):
            """Track validation calls."""
            validation_called.append(key)
        
        # Add hook
        validator.add_validation_hook(validation_hook)
        
        # Validate parameter
        spec = ParameterSpec(
            key="hooked.param",
            value=10.0,
            category=ParameterCategory.COMPUTATIONAL,
            description="Parameter with hook", units="dimensionless", dtype="float32", bounds=(-float("inf"), float("inf"))
        )
        
        validator.validate_value("hooked.param", 10.0, spec)
        assert "hooked.param" in validation_called



def test_validate_parameter_value_requires_the_expected_dtype():
    import inspect

    parameter = inspect.signature(validate_parameter_value).parameters["dtype"]
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="dtype"):
        validate_parameter_value(value=1.0, bounds=(0.0, 2.0))



class TestSignConstraintsComeFromDeclaredBounds:
    """A key containing "rate" says nothing about the sign; the declared bounds do."""

    @staticmethod
    def _spec(key, bounds):
        return ParameterSpec(
            key=key, value=1.0, units="s-1", dtype="float32", bounds=bounds,
            category=ParameterCategory.BIOLOGICAL_PARAMETER,
        )

    @pytest.mark.parametrize("key", ["test.calibrated_gain", "test.growth_rate", "test.carbon_flux"])
    def test_negative_value_passes_when_the_bounds_allow_it(self, key):
        spec = self._spec(key, (-10.0, 10.0))
        assert spec.validate_value(-1.0) == []
        assert ParameterValidator().validate_value(key, -1.0, spec).is_valid

    @pytest.mark.parametrize("key", ["test.calibrated_gain", "test.growth_rate"])
    def test_negative_value_is_rejected_by_a_zero_lower_bound(self, key):
        spec = self._spec(key, (0.0, 10.0))
        assert spec.validate_value(-1.0)
        assert not ParameterValidator().validate_value(key, -1.0, spec).is_valid



class TestNoConstraintIsInferredFromKeyNames:
    """Concentration, density, content and temperature keys are checked against their bounds only."""

    @staticmethod
    def _spec(key, units, bounds):
        return ParameterSpec(
            key=key, value=1.0, units=units, dtype="float32", bounds=bounds,
            category=ParameterCategory.MODEL_PARAMETER,
        )

    @pytest.mark.parametrize(
        ("key", "units", "value"),
        [
            ("test.concentration_change_scale", "mol m-3", -0.5),
            ("test.bulk_density_correction", "kg m-3", -20.0),
            ("test.water_content_bias", "m3 m-3", -0.05),
            ("test.temperature_shift", "K", -10.0),
            ("test.temperature_shift_celsius", "degC", -300.0),
        ],
    )
    def test_a_value_inside_the_declared_bounds_passes(self, key, units, value):
        spec = self._spec(key, units, (-1000.0, 1000.0))
        assert spec.validate_value(value) == []
        assert ParameterValidator().validate_value(key, value, spec).is_valid

    def test_strict_mode_has_no_earth_range_for_temperature_keys(self):
        spec = self._spec("test.combustion_temperature", "K", (0.0, 3000.0))
        strict = ParameterValidator(level=ValidationLevel.STRICT)
        assert strict.validate_value("test.combustion_temperature", 1500.0, spec).is_valid

    @pytest.mark.parametrize(
        ("key", "units"), [("test.nitrate_concentration", "mol m-3"), ("test.soil_temperature", "K")]
    )
    def test_the_declared_lower_bound_rejects_a_negative_value(self, key, units):
        spec = self._spec(key, units, (0.0, 1000.0))
        assert spec.validate_value(-1.0)
        assert not ParameterValidator().validate_value(key, -1.0, spec).is_valid



class TestComponentwiseBoundsOfListParameters:
    """A list-valued parameter that declares one (minimum, maximum) pair per component
    is judged component by component (relic.parameters.spec.componentwise_bounds_errors)."""

    # Declared as in hydrosphere.numerical.temperature_bounds [K]: the low end of the
    # admitted range lies in [223.15, 273.15], the high end in [323.15, 373.15].
    PAIRS = [[223.15, 273.15], [323.15, 373.15]]

    @staticmethod
    def _spec(bounds, *, value=None, dtype="list"):
        return ParameterSpec(
            key="test.numerical.temperature_domain",
            value=[273.15, 323.15] if value is None else value,
            units="K", dtype=dtype, bounds=bounds, category=ParameterCategory.MODEL_PARAMETER,
        )

    def test_only_numeric_pairs_are_a_componentwise_declaration(self):
        from relic.parameters.spec import componentwise_bounds

        assert componentwise_bounds(self.PAIRS) == ((223.15, 273.15), (323.15, 373.15))
        assert componentwise_bounds(([0, 1], (2.5, 3))) == ((0.0, 1.0), (2.5, 3.0))
        for other in (
            None,
            (0.0, 1.0),
            [0.0, 1.0],
            ("NO", "O2"),
            ["-2000.0 - 0.0", "1000.0 - 2500.0"],
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            [[0.0, "1.0"], [2.0, 3.0]],
            [[0.0, 1.0], 2.0],
        ):
            assert componentwise_bounds(other) is None, other

    def test_list_and_tuple_are_the_sequence_dtypes(self):
        from relic.parameters.spec import parameter_dtype_is_sequence

        assert all(parameter_dtype_is_sequence(dtype) for dtype in ("list", "tuple", "List[float]", list))
        assert not any(parameter_dtype_is_sequence(dtype) for dtype in ("float64", "dict", "str", None))

    @pytest.mark.parametrize("value", [[273.15, 323.15], (223.15, 373.15), [250, 350]])
    def test_components_inside_their_pairs_pass(self, value):
        spec = self._spec(self.PAIRS)
        assert spec.validate_value(value) == []
        assert ParameterValidator().validate_value(spec.key, value, spec).is_valid

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ([200.0, 323.15], "component 0 below minimum: 200.0 < 223.15"),
            ([280.0, 323.15], "component 0 above maximum: 280.0 > 273.15"),
            ((233.15, 400.0), "component 1 above maximum: 400.0 > 373.15"),
            ([273.15], "has length 1 but its bounds declare 2 components"),
            ([273.15, 300.0, 323.15], "has length 3 but its bounds declare 2 components"),
            (["273.15", 323.15], "component 0 must be a real number, got str '273.15'"),
            ([273.15, True], "component 1 must be a real number, got bool True"),
            ([float("nan"), 323.15], "component 0 is not finite: nan"),
            ([273.15, float("inf")], "component 1 is not finite: inf"),
        ],
    )
    def test_each_violation_of_a_list_value_is_named(self, value, message):
        spec = self._spec(self.PAIRS)
        expected = f"Parameter {spec.key} {message}"
        assert spec.validate_value(value) == [expected]
        result = ParameterValidator().validate_value(spec.key, value, spec)
        assert not result.is_valid
        assert result.errors == [expected]

    def test_every_violating_component_is_reported(self):
        spec = self._spec(self.PAIRS)
        assert spec.validate_value([200.0, 400.0]) == [
            f"Parameter {spec.key} component 0 below minimum: 200.0 < 223.15",
            f"Parameter {spec.key} component 1 above maximum: 400.0 > 373.15",
        ]

    def test_a_pair_that_cannot_order_its_component_rejects_it(self):
        # NaN compares false with every number; the rule must not pass such a pair.
        from relic.parameters.spec import componentwise_bounds_errors

        assert componentwise_bounds_errors([0.5, 0.5], [(float("nan"), 1.0), (0.0, 1.0)]) == [
            "component 0 = 0.5 cannot be ordered against its declared bounds [nan, 1.0]"
        ]

    @pytest.mark.parametrize(
        "value", [300.0, torch.tensor([273.15, 323.15]), "273.15, 323.15", {"low": 273.15, "high": 323.15}]
    )
    def test_spec_rejects_a_value_that_is_not_a_list_or_tuple(self, value):
        spec = self._spec(self.PAIRS)
        assert spec.validate_value(value) == [
            f"Parameter {spec.key} takes a list or tuple of 2 numbers "
            f"(one per declared bounds pair), got {type(value).__name__}"
        ]

    @pytest.mark.parametrize(
        "value", [300.0, float("nan"), torch.tensor(300.0), torch.tensor([273.15, 323.15])]
    )
    def test_validator_reports_a_non_sequence_value_instead_of_comparing_it_with_a_pair(self, value):
        # The scalar interval rule cannot order a number against a [min, max] pair
        # (TypeError); the componentwise rule alone judges such a parameter.
        spec = self._spec(self.PAIRS)
        result = ParameterValidator().validate_value(spec.key, value, spec)
        assert not result.is_valid
        assert result.errors == [
            f"Parameter {spec.key} takes a list or tuple of 2 numbers "
            f"(one per declared bounds pair), got {type(value).__name__}"
        ]
        assert "bounds" not in result.error_types

    @pytest.mark.parametrize(
        ("bounds", "dtype", "value"),
        [
            ((0.0, 1.0), "list", [5.0, -3.0]),
            (None, "list", [5.0, -3.0]),
            (("NO", "O2"), "list", ["NO", "NO2", "O3", "O2"]),
            (["-2000.0 - 0.0", "1000.0 - 2500.0"], "list", [-5000.0, 9000.0]),
            (([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), "list", [5.0, 5.0, 5.0]),
            (PAIRS, "float64", torch.tensor([0.0, 999.0], dtype=torch.float64)),
        ],
    )
    def test_the_componentwise_rule_does_not_apply_to_other_declarations(self, bounds, dtype, value):
        # A flat (lower, upper) interval, no bounds, labels, text ranges, lower/upper
        # vectors and a Tensor (numeric dtype) parameter are judged exactly as before.
        spec = self._spec(bounds, value=value, dtype=dtype)
        assert spec.validate_value(value) == []
        if isinstance(value, list):
            assert ParameterValidator().validate_value(spec.key, value, spec).is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
