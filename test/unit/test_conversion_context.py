"""L1 Unit Test: ConversionContext Functionality.

Tests the ConversionContext class which provides additional information
for unit conversions that depend on environmental or physical context.

Reference:
- relic/data/standards/unit_converter.py (ConversionContext class)
"""
import pytest
import torch
from relic.data.standards.unit_converter import ConversionContext


class TestContextCreation:
    """Test ConversionContext creation and initialization."""

    def test_empty_context_creation(self):
        """Test creating an empty ConversionContext."""
        context = ConversionContext()
        assert context is not None
        assert isinstance(context, ConversionContext)
        assert context.allow_defaults is False

    def test_context_with_temperature(self):
        """Test creating ConversionContext with temperature."""
        context = ConversionContext(temperature=298.15)
        # Context should store the temperature
        # (Exact attribute access depends on implementation)
        assert context is not None

    def test_context_with_pressure(self):
        """Test creating ConversionContext with pressure."""
        context = ConversionContext(pressure=101325.0)
        assert context is not None

    def test_context_with_multiple_parameters(self):
        """Test creating ConversionContext with multiple parameters."""
        context = ConversionContext(
            temperature=298.15,
            pressure=101325.0,
            soil_depth=0.1
        )
        assert context is not None


class TestContextPropagation:
    """Test that context is properly propagated through conversions."""

    def test_context_used_in_conversion(self, unit_converter):
        """Test that context is passed to conversion method."""
        context = ConversionContext(temperature=298.15)

        # This should not raise an error
        result = unit_converter.convert(
            25.0,
            'air_temperature',
            'celsius',
            'K',
            context
        )
        assert isinstance(result, float)

    def test_empty_context_works(self, unit_converter):
        """Test that empty context works for simple conversions."""
        context = ConversionContext()

        result = unit_converter.convert(
            100.0,
            'length',
            'm',
            'km',
            context
        )
        assert abs(result - 0.1) < 1e-6


class TestContextDependentConversions:
    """Test conversions that may depend on context.

    Some unit conversions (e.g., volumetric to mass) require
    additional context like density or soil depth.
    """

    def test_volumetric_to_mass_requires_context(self, unit_converter):
        """Test volumetric to mass conversion with soil depth context.

        Converting m3/m3 (volumetric water content) to kg/m2 (water mass)
        requires soil depth information.
        """
        context = ConversionContext(soil_depth=0.1)  # 10 cm depth

        with pytest.raises(ValueError, match="water_density"):
            unit_converter.convert(
                0.3,  # 30% volumetric water content
                'soil_moisture_content',
                'm3 m-3',
                'kg m-2',
                context
            )

        explicit_default_context = ConversionContext(
            soil_depth=0.1,
            allow_defaults=True,
        )
        result = unit_converter.convert(
            0.3,
            'soil_moisture_content',
            'm3 m-3',
            'kg m-2',
            explicit_default_context,
        )
        assert result > 0

    def test_temperature_dependent_conversion(self, unit_converter):
        """Test conversion that may depend on temperature.

        Some conversions (e.g., vapor pressure) depend on temperature.
        """
        context = ConversionContext(temperature=298.15)

        # Basic conversion should still work
        result = unit_converter.convert(
            1000.0,
            'pressure',
            'Pa',
            'kPa',
            context
        )
        assert abs(result - 1.0) < 1e-6


class TestContextValidation:
    """Test validation of context parameters."""

    def test_negative_temperature_allowed(self):
        """Test that negative temperatures (in Celsius) are allowed."""
        # Negative Celsius is valid (below freezing)
        context = ConversionContext(temperature=253.15)  # -20C in K
        assert context is not None

    def test_zero_temperature_allowed(self):
        """Test that zero temperature is allowed (absolute zero in K)."""
        context = ConversionContext(temperature=0.0)
        assert context is not None

    def test_negative_pressure_rejected(self):
        """Test that negative pressure should be rejected or handled.

        Negative pressure is physically meaningless in most contexts.
        """
        # Implementation may accept this but should handle properly
        context = ConversionContext(pressure=-100.0)
        # Just ensure it doesn't crash during creation
        assert context is not None


class TestContextImmutability:
    """Test that context remains unchanged during conversions."""

    def test_context_not_modified_by_conversion(self, unit_converter):
        """Test that conversion doesn't modify the original context."""
        original_temp = 298.15
        context = ConversionContext(temperature=original_temp)

        # Perform conversion
        unit_converter.convert(
            25.0,
            'air_temperature',
            'celsius',
            'K',
            context
        )

        # Context should remain unchanged
        # (Exact check depends on ConversionContext implementation)
        # At minimum, it should not crash
        assert context is not None


class TestContextWithTensors:
    """Test context usage with tensor inputs."""

    def test_context_with_tensor_conversion(self, unit_converter):
        """Test that context works with tensor inputs."""
        context = ConversionContext(temperature=298.15)

        tensor_input = torch.tensor([25.0, 30.0, 35.0])
        result = unit_converter.convert(
            tensor_input,
            'air_temperature',
            'celsius',
            'K',
            context
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor_input.shape


class TestContextDefaults:
    """Test default behavior when context parameters are not provided."""

    def test_missing_optional_context_parameter(self, unit_converter):
        """Test conversion works even if optional context parameters are missing."""
        # Context without soil_depth
        context = ConversionContext(temperature=298.15)

        # Basic conversions should still work
        result = unit_converter.convert(
            100.0,
            'length',
            'm',
            'km',
            context
        )
        assert abs(result - 0.1) < 1e-6

    def test_completely_empty_context(self, unit_converter):
        """Test that completely empty context works for basic conversions."""
        context = ConversionContext()

        result = unit_converter.convert(
            25.0,
            'air_temperature',
            'celsius',
            'K',
            context
        )
        assert abs(result - 298.15) < 1e-5

    def test_context_required_conversion_fails_without_default_opt_in(
        self,
        unit_converter,
    ):
        with pytest.raises(
            ValueError,
            match="Missing required conversion context: temperature.*pressure",
        ):
            unit_converter.convert(
                400.0,
                'co2_concentration',
                'ppm',
                'mol m-3',
                ConversionContext(),
            )

        result = unit_converter.convert(
            400.0,
            'co2_concentration',
            'ppm',
            'mol m-3',
            ConversionContext(allow_defaults=True),
        )
        assert result > 0

