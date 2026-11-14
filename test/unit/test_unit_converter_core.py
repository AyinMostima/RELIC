"""L1 Unit Test: UnitConverter Core Functionality.

Tests the core type preservation and basic conversion functionality
of the UnitConverter class.

Reference:
- relic/data/standards/unit_converter.py (Lines 115-210)
- test/unit_conversion/test_p0_simple.py (Phase 0 verification)
"""
import pytest
import torch
import numpy as np
import relic.data.standards.dimensionality as dimensionality_module
import relic.data.standards.unit_converter as unit_converter_module
from relic.data.standards.unit_converter import UnitConverter, ConversionContext
from relic.orchestration.adapters import AdapterRegistry, UnitCastAdapter
import relic.utils.physical_constants as physical_constants


class TestPartsPerConversions:
    """Verify mole-fraction scaling through the existing conversion owner."""

    @pytest.mark.parametrize("source,factor", [("ppm", 1e-6), ("ppb", 1e-9)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_roundtrip_retains_precision_and_gradient(self, unit_converter, source, factor, dtype):
        values = torch.tensor([0.0, 1e-4, 1.0, 1000.0], dtype=dtype, requires_grad=True)
        result = unit_converter.convert(values, "minimum_concentration", source, "mol mol-1")
        torch.testing.assert_close(result, values * factor, rtol=1e-6 if dtype == torch.float32 else 1e-12, atol=0)
        assert result.dtype == dtype and result.device == values.device
        restored = unit_converter.convert(result, "minimum_concentration", "mol mol-1", source)
        torch.testing.assert_close(restored, values)
        gradient = torch.autograd.grad(result.sum(), values)[0]
        torch.testing.assert_close(gradient, torch.full_like(values, factor), rtol=1e-6, atol=0)
        assert unit_converter.can_convert(source, "mol mol-1")

    def test_integer_ppb_input_does_not_truncate_fraction(self, unit_converter):
        result = unit_converter.convert(1, "generic", "ppb", "mol mol-1")
        assert isinstance(result, float)
        assert result == pytest.approx(1e-9, rel=1e-12, abs=0)


class TestTypePreservation:
    """Test that UnitConverter preserves input types correctly."""

    def test_tensor_type_preservation(self, unit_converter, conversion_context):
        """Test Tensor input returns Tensor output.

        Verifies that torch.Tensor inputs are returned as torch.Tensor,
        maintaining PyTorch consistency for differentiable computation.
        """
        input_value = torch.tensor([25.0, 30.0])
        result = unit_converter.convert(
            input_value,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )

        assert isinstance(result, torch.Tensor), \
            f"Expected torch.Tensor, got {type(result)}"
        assert result.shape == input_value.shape, \
            f"Shape mismatch: {result.shape} != {input_value.shape}"

    def test_float_type_preservation(self, unit_converter, conversion_context):
        """Test float input returns float output.

        Verifies that Python float inputs maintain float type,
        important for lightweight scalar computations.
        """
        input_value = 25.0
        result = unit_converter.convert(
            input_value,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )

        assert isinstance(result, float), \
            f"Expected float, got {type(result)}"
        assert abs(result - 298.15) < 1e-5, \
            f"Incorrect conversion: {result} != 298.15"

    def test_int_type_preservation(self, unit_converter, conversion_context):
        """Integer input must retain the fractional part of a scientific conversion."""
        input_value = 25
        result = unit_converter.convert(
            input_value,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )

        assert isinstance(result, float)
        assert result == pytest.approx(298.15, rel=0, abs=1e-12)

    def test_integer_mass_conversion_does_not_erase_the_mass(self, unit_converter, conversion_context):
        assert unit_converter.convert(1, "generic", "g", "kg", conversion_context) == 0.001

    @pytest.mark.parametrize("layout", ["scalar", "reversed", "transposed", "readonly", "non_native"])
    def test_numpy_float64_precision_and_supported_layouts(self, unit_converter, conversion_context, layout):
        if layout == "scalar":
            values = np.array(1e-50, dtype=np.float64)
        else:
            values = np.array([[1e-50, 0.1, 1.0], [2.0, 3.0, 4.0]], dtype=np.float64)
            if layout == "reversed":
                values = values[:, ::-1]
            elif layout == "transposed":
                values = values.T
            elif layout == "readonly":
                values.setflags(write=False)
            elif layout == "non_native":
                values = values.astype(">f8")
        result = unit_converter.convert(values, "generic", "g", "kg", conversion_context)
        assert result.shape == values.shape
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, values / 1000.0, rtol=1e-15, atol=0)

    @pytest.mark.parametrize("container", ["numpy", "tensor"])
    def test_integer_arrays_promote_to_float64_for_fractional_units(self, unit_converter, conversion_context, container):
        values = np.array([1, 2, 3], dtype=np.int64)
        if container == "tensor":
            values = torch.from_numpy(values)
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            result = unit_converter.convert(values, "generic", "g", "kg", conversion_context)
        finally:
            torch.set_default_dtype(original_dtype)
        if container == "tensor":
            assert result.dtype == torch.float64
            torch.testing.assert_close(result, torch.tensor([0.001, 0.002, 0.003], dtype=torch.float64))
        else:
            assert result.dtype == np.float64
            np.testing.assert_allclose(result, [0.001, 0.002, 0.003], rtol=1e-15)

    def test_float64_tensor_conversion_retains_small_values_and_gradient(self, unit_converter, conversion_context):
        values = torch.tensor([1e-50, 0.1], dtype=torch.float64, requires_grad=True)
        result = unit_converter.convert(values, "generic", "g", "kg", conversion_context)
        assert result.device == values.device
        torch.testing.assert_close(result, values / 1000.0, rtol=0, atol=0)
        result.sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, 0.001))

    def test_numpy_type_preservation(self, unit_converter, conversion_context):
        """Test numpy.ndarray input returns numpy.ndarray output.

        Verifies numpy array consistency for data I/O operations.
        """
        input_value = np.array([25.0, 30.0, 35.0])
        result = unit_converter.convert(
            input_value,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )

        assert isinstance(result, np.ndarray), \
            f"Expected np.ndarray, got {type(result)}"
        assert result.shape == input_value.shape, \
            f"Shape mismatch: {result.shape} != {input_value.shape}"

    def test_mixed_type_batch(self, mixed_type_input, unit_converter, conversion_context):
        """Test multiple input types with parametrized fixture.

        Uses pytest parametrization to test all supported types
        in a single test case.
        """
        result = unit_converter.convert(
            mixed_type_input,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )

        # Result should match input type
        if isinstance(mixed_type_input, torch.Tensor):
            assert isinstance(result, torch.Tensor)
        elif isinstance(mixed_type_input, np.ndarray):
            assert isinstance(result, np.ndarray)
        elif isinstance(mixed_type_input, float):
            assert isinstance(result, float)
        elif isinstance(mixed_type_input, int):
            assert isinstance(result, (int, float))


class TestSameUnitFastPath:
    """Test fast-path optimization for same-unit conversions."""

    @pytest.mark.parametrize("value", [0.1, 2**60 + 1])
    @pytest.mark.parametrize("target", ["1", "dimensionless"])
    def test_identity_preserves_scalar_bits(self, unit_converter, value, target):
        result = unit_converter.convert(value, "generic", "1", target, ConversionContext(allow_identity=True))
        assert type(result) is type(value)
        assert result == value

    def test_identity_preserves_numpy_float64_without_copy(self, unit_converter):
        values = np.array([0.1, 1e-50], dtype=np.float64)[::-1]
        result = unit_converter.convert(values, "generic", "1", "dimensionless", ConversionContext(allow_identity=True))
        assert result is values

    def test_same_unit_identity(self, unit_converter, conversion_context):
        """Test that converting from same unit to same unit returns input unchanged.

        This tests the fast-path optimization that avoids unnecessary computation.
        """
        input_value = torch.tensor([298.15, 300.0])
        result = unit_converter.convert(
            input_value,
            'air_temperature',
            'K',
            'K',
            conversion_context
        )

        assert torch.allclose(result, input_value), \
            "Same-unit conversion should return identical values"

    def test_same_unit_preserves_type(self, unit_converter, conversion_context):
        """Test that same-unit conversion preserves exact type."""
        # Test with float
        result_float = unit_converter.convert(
            298.15,
            'air_temperature',
            'K',
            'K',
            conversion_context
        )
        assert isinstance(result_float, float)

        # Test with numpy
        result_numpy = unit_converter.convert(
            np.array([298.15]),
            'air_temperature',
            'K',
            'K',
            conversion_context
        )
        assert isinstance(result_numpy, np.ndarray)

    def test_same_unknown_unit_is_not_convertible(self, unit_converter):
        """Reject identical labels that the canonical parser cannot recognize."""
        assert unit_converter.can_convert("banana", "banana") is False

    def test_same_unknown_unit_conversion_fails_closed(
        self,
        unit_converter,
        conversion_context,
    ):
        """Do not let the identity fast path certify an unknown unit label."""
        with pytest.raises(ValueError, match="Unknown or unsupported unit label"):
            unit_converter.convert(
                1.0,
                "generic",
                "banana",
                "banana",
                conversion_context,
            )


class TestIdentityConversions:
    """Require explicit authorization for unit-spelling equivalence."""

    def test_identity_alias_requires_explicit_opt_in(self, unit_converter):
        with pytest.raises(ValueError, match="allow_identity=True"):
            unit_converter.convert(
                1.0,
                "generic",
                "dimensionless",
                "1",
                ConversionContext(),
            )

        result = unit_converter.convert(
            1.0,
            "generic",
            "dimensionless",
            "1",
            ConversionContext(allow_identity=True),
        )
        assert result == 1.0


class TestVariableNameGovernance:
    """Enforce canonical scientific variable names at conversion boundaries."""

    def test_removed_productivity_abbreviations_are_not_normalized(self):
        assert "gpp" not in UnitConverter.VARIABLE_ALIASES
        assert "npp" not in UnitConverter.VARIABLE_ALIASES
        assert UnitConverter._normalize_variable_name("gpp") == "gpp"
        assert UnitConverter._normalize_variable_name("npp") == "npp"


class TestInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_invalid_type_raises_error(self, unit_converter, conversion_context):
        """Test that unsupported input types raise appropriate errors.

        Verifies that the converter rejects invalid input types
        with clear error messages.
        """
        with pytest.raises((TypeError, ValueError)):
            unit_converter.convert(
                "not_a_number",
                'air_temperature',
                'celsius',
                'K',
                conversion_context
            )

    def test_none_input_raises_error(self, unit_converter, conversion_context):
        """Test that None input raises error."""
        with pytest.raises((TypeError, ValueError)):
            unit_converter.convert(
                None,
                'air_temperature',
                'celsius',
                'K',
                conversion_context
            )

    def test_empty_tensor_handled(self, unit_converter, conversion_context):
        """Test that empty tensors are handled gracefully."""
        empty_tensor = torch.tensor([])
        result = unit_converter.convert(
            empty_tensor,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )

        assert isinstance(result, torch.Tensor)
        assert len(result) == 0


class TestConversionContext:
    """Test ConversionContext integration."""

    def test_context_required(self, unit_converter):
        """Test that ConversionContext is required for convert method."""
        # Should work with context
        context = ConversionContext()
        result = unit_converter.convert(
            25.0,
            'air_temperature',
            'celsius',
            'K',
            context
        )
        assert isinstance(result, float)

    def test_context_empty_works(self, unit_converter):
        """Test that empty ConversionContext works for basic conversions."""
        context = ConversionContext()
        result = unit_converter.convert(
            100.0,
            'length',
            'm',
            'km',
            context
        )
        assert abs(result - 0.1) < 1e-6


class TestPhysicalConstantSSOT:
    """Verify that UnitConverter uses the physical-constant access API."""

    def test_constant_resolvers_delegate_to_ssot_accessor(self, monkeypatch):
        calls = []
        values = {
            "relic.physical.constants.universal.gas_constant": 8.5,
            "relic.physical.constants.molecular_weights.carbon": 12.5,
            "relic.physical.constants.water.molecular_weight": 19.0,
        }

        def fake_get_physical_constant(name):
            calls.append(name)
            return values[name]

        monkeypatch.setattr(
            physical_constants,
            "get_physical_constant",
            fake_get_physical_constant,
        )
        converter = UnitConverter()

        assert converter._resolve_physical_constant("universal.gas_constant") == 8.5
        assert converter._resolve_molecular_weight("molecular_weights.carbon") == 12.5
        assert converter._resolve_molecular_weight("water.molecular_weight") == 19.0
        assert calls == list(values)

    def test_physical_constants_cannot_be_overridden_by_context(self):
        converter = UnitConverter()

        with pytest.raises(TypeError):
            ConversionContext(molecular_weight_water=99.0)
        with pytest.raises(TypeError):
            ConversionContext(molecular_weight_carbon=99.0)
        with pytest.raises(TypeError):
            converter._resolve_molecular_weight("water.molecular_weight", 99.0)
        with pytest.raises(TypeError):
            converter._resolve_physical_constant("universal.gas_constant", 99.0)

    def test_context_physical_defaults_delegate_to_ssot_accessor(self, monkeypatch):
        class ConversionContextManager:
            def get(self, key, **kwargs):
                if key == "soil_depth":
                    return 1.25
                raise AssertionError(f"Unexpected ParameterManager lookup: {key}")

        values = {
            "reference_temperature": 299.0,
            "reference_pressure": 100500.0,
            "water_density": 997.0,
        }
        monkeypatch.setattr(
            physical_constants,
            "get_physical_constant",
            lambda name: values[name],
        )
        converter = UnitConverter()
        converter._parameter_manager_cache = ConversionContextManager()

        result = converter._apply_context_defaults(
            ConversionContext(allow_defaults=True)
        )

        assert result.soil_depth == 1.25
        assert result.temperature == 299.0
        assert result.pressure == 100500.0
        assert result.water_density == 997.0

    def test_celsius_offset_delegates_to_ssot_accessor(self, monkeypatch):
        monkeypatch.setattr(
            physical_constants,
            "get_physical_constant",
            lambda name: 274.0 if name == "celsius_to_kelvin" else None,
        )
        converter = UnitConverter()

        result = converter.celsius_to_kelvin(
            torch.tensor(0.0),
            ConversionContext(allow_defaults=False),
        )

        assert result.item() == 274.0

    def test_time_and_radiation_conversions_delegate_to_ssot_accessor(
        self,
        monkeypatch,
    ):
        values = {
            "seconds_per_hour": 10.0,
            "seconds_per_day": 100.0,
            "relic.physical.constants.conversions.j_to_cal": 0.25,
        }
        monkeypatch.setattr(
            physical_constants,
            "get_physical_constant",
            lambda name: values[name],
        )
        converter = UnitConverter()
        context = ConversionContext(water_density=1000.0)

        assert converter.second_to_hour(torch.tensor(20.0), context).item() == 2.0
        assert converter.hour_to_second(torch.tensor(2.0), context).item() == 20.0
        assert converter.day_to_second(torch.tensor(2.0), context).item() == 200.0
        assert converter.second_to_day(torch.tensor(200.0), context).item() == 2.0
        assert converter.external_radiation_langley_to_wm2(
            torch.tensor(1.0), context
        ).item() == pytest.approx(400.0)
        assert converter.external_radiation_cal_to_wm2(
            torch.tensor(1.0), context
        ).item() == pytest.approx(400.0)
        assert converter.carbon_flux_g_day_to_kg_s(
            torch.tensor(1000.0), context
        ).item() == pytest.approx(0.01)
        assert converter.carbon_flux_kg_s_to_g_day(
            torch.tensor(0.01), context
        ).item() == pytest.approx(1000.0)

    @pytest.mark.parametrize(
        "method_name,input_value",
        [
            ("volumetric_to_mass_soil_moisture", 0.3),
            ("mass_to_volumetric_soil_moisture", 300.0),
        ],
    )
    def test_soil_water_conversion_has_no_hidden_density_fallback(
        self,
        method_name,
        input_value,
    ):
        converter = UnitConverter()
        method = getattr(converter, method_name)
        context = ConversionContext(soil_depth=1.0, allow_defaults=False)

        with pytest.raises(ValueError, match="Water density"):
            method(torch.tensor(input_value), context)


class TestUnitConverterRuntimeOwner:
    """Require one public runtime instance and no compatibility getters."""

    def test_module_level_instance_is_the_only_runtime_owner(self):
        assert isinstance(unit_converter_module.unit_converter, UnitConverter)
        assert not hasattr(dimensionality_module, "unit_" + "converter")
        assert not hasattr(
            unit_converter_module,
            "get_global_" + "unit_converter",
        )
        assert not hasattr(unit_converter_module, "get_unit_" + "converter")

    def test_orchestration_adapters_share_the_canonical_runtime_owner(self):
        owner = unit_converter_module.unit_converter

        assert UnitCastAdapter().unit_converter is owner
        registry = AdapterRegistry()
        assert registry.unit_converter is owner
        assert registry.get("unit_cast").unit_converter is owner

        injected = UnitConverter()
        assert UnitCastAdapter(injected).unit_converter is injected
        injected_registry = AdapterRegistry(injected)
        assert injected_registry.unit_converter is injected
        assert injected_registry.get("unit_cast").unit_converter is injected

    def test_duplicate_cf_unit_compliance_checker_is_removed(self):
        assert not hasattr(unit_converter_module, "CFUnitComplianceChecker")

    def test_soil_convenience_wrappers_do_not_construct_runtime_owners(
        self,
        monkeypatch,
    ):
        owner = unit_converter_module.unit_converter

        def reject_second_owner():
            raise AssertionError("UnitConverter runtime owner must be reused")

        monkeypatch.setattr(unit_converter_module, "UnitConverter", reject_second_owner)
        mass = owner.soil_moisture_m3m3_to_kgm2(
            torch.tensor(0.3),
            soil_depth=1.0,
            water_density=1000.0,
        )
        volumetric = owner.soil_moisture_kgm2_to_m3m3(
            mass,
            soil_depth=1.0,
            water_density=1000.0,
        )

        assert mass.item() == pytest.approx(300.0)
        assert volumetric.item() == pytest.approx(0.3)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_value_conversion(self, unit_converter, conversion_context):
        """Test conversion of zero values."""
        result = unit_converter.convert(
            0.0,
            'length',
            'm',
            'km',
            conversion_context
        )
        assert abs(result - 0.0) < 1e-10

    def test_negative_value_conversion(self, unit_converter, conversion_context):
        """Test conversion of negative values (e.g., Celsius temperatures)."""
        result = unit_converter.convert(
            -40.0,
            'air_temperature',
            'celsius',
            'K',
            conversion_context
        )
        # -40C = 233.15K
        assert abs(result - 233.15) < 1e-5

    def test_large_value_conversion(self, unit_converter, conversion_context):
        """Test conversion of large values."""
        large_value = 1e6  # 1000 km in meters
        result = unit_converter.convert(
            large_value,
            'length',
            'm',
            'km',
            conversion_context
        )
        assert abs(result - 1000.0) < 1e-5

    def test_very_small_value_conversion(self, unit_converter, conversion_context):
        """Test conversion of very small values."""
        small_value = 1e-6  # 1 micrometer in meters
        result = unit_converter.convert(
            small_value,
            'length',
            'm',
            'mm',
            conversion_context
        )
        assert abs(result - 1e-3) < 1e-9


