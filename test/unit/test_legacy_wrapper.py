"""L1 Unit Test: Unit Converter Interface.

Tests the authoritative UnitConverter interface and conversion behavior.
"""
import pytest
import numpy as np
from relic.data.standards.unit_converter import ConversionContext


def _convert(converter, value, from_unit, to_unit, variable_name="generic"):
    """Convert units using the authoritative converter."""
    context = ConversionContext()
    return converter.convert(value, variable_name, from_unit, to_unit, context)


class TestConverterSignature:
    """Test UnitConverter signature consistency."""

    def test_convert_with_float(self, unit_converter):
        """Test convert method with float input."""
        result = _convert(unit_converter, 25.0, 'celsius', 'K')
        assert isinstance(result, float), \
            f"Expected float, got {type(result)}"
        assert abs(result - 298.15) < 1e-5, \
            f"Expected 298.15, got {result}"

    def test_convert_with_numpy(self, unit_converter):
        """Test convert method with numpy array."""
        arr = np.array([0.0, 25.0, 100.0])
        result = _convert(unit_converter, arr, 'celsius', 'K')

        assert isinstance(result, np.ndarray), \
            f"Expected np.ndarray, got {type(result)}"
        expected = np.array([273.15, 298.15, 373.15])
        assert np.allclose(result, expected, atol=1e-5), \
            f"Expected {expected}, got {result}"

    def test_convert_with_element_parameter(self, unit_converter):
        """Test convert with an explicit variable name."""
        result = _convert(unit_converter, 1000.0, 'm', 'km', 'length')
        assert abs(result - 1.0) < 1e-6


class TestConverterCanConvert:
    """Test UnitConverter.can_convert method."""

    def test_can_convert_valid_conversion(self, unit_converter):
        """Test can_convert returns True for valid conversions."""
        # Temperature conversion should be supported
        result = unit_converter.can_convert('celsius', 'K')
        assert result is True, \
            "celsius to K conversion should be supported"

    def test_can_convert_with_variable_name(self, unit_converter):
        """Test can_convert with variable_name parameter."""
        result = unit_converter.can_convert(
            'celsius',
            'K',
            variable_name='air_temperature'
        )
        assert result is True

    def test_can_convert_generic_default(self, unit_converter):
        """Test can_convert falls back to 'generic' if specific not found."""
        # Should find conversion even without specific variable name
        result = unit_converter.can_convert('m', 'km')
        assert result is True

    def test_can_convert_invalid_conversion(self, unit_converter):
        """Test can_convert returns False for invalid conversions."""
        # Converting meters to seconds should not be supported
        result = unit_converter.can_convert('m', 's')
        assert result is False, \
            "m to s conversion should not be supported"


class TestConvenienceConversion:
    """Test helper conversion patterns."""

    def test_convert_units_basic(self, unit_converter):
        """Test basic conversion helper."""
        result = _convert(unit_converter, 100.0, 'm', 'km')
        assert abs(result - 0.1) < 1e-6, \
            f"Expected 0.1, got {result}"

    def test_convert_units_temperature(self, unit_converter):
        """Test temperature conversion helper."""
        result = _convert(unit_converter, 0.0, 'celsius', 'K')
        assert abs(result - 273.15) < 1e-5, \
            f"Expected 273.15, got {result}"

    def test_convert_units_with_array(self, unit_converter):
        """Test array conversion helper."""
        arr = np.array([1.0, 2.0, 3.0])
        result = _convert(unit_converter, arr, 'km', 'm')

        assert isinstance(result, np.ndarray)
        expected = np.array([1000.0, 2000.0, 3000.0])
        assert np.allclose(result, expected)


class TestUsagePatterns:
    """Test common usage patterns for the unit converter."""

    def test_simple_conversion(self, unit_converter):
        """Test simple unit conversion."""
        temp_c = 25.0
        temp_k = _convert(unit_converter, temp_c, 'celsius', 'K')
        assert abs(temp_k - 298.15) < 1e-5

    def test_check_before_convert(self, unit_converter):
        """Test checking if conversion is possible before converting."""
        if unit_converter.can_convert('m', 'km'):
            result = _convert(unit_converter, 1000.0, 'm', 'km')
            assert abs(result - 1.0) < 1e-6

    def test_batch_conversion(self, unit_converter):
        """Test converting arrays of values."""
        temperatures = np.array([0, 10, 20, 30])
        temps_k = _convert(unit_converter, temperatures, 'celsius', 'K')
        expected = np.array([273.15, 283.15, 293.15, 303.15])
        assert np.allclose(temps_k, expected, atol=1e-5)


class TestErrorHandling:
    """Test error handling in unit conversion."""

    def test_invalid_units_raises_error(self, unit_converter):
        """Test that invalid units raise appropriate errors."""
        with pytest.raises((ValueError, KeyError)):
            _convert(unit_converter, 100.0, 'invalid_unit', 'K')

    def test_inconsistent_units_raises_error(self, unit_converter):
        """Test that inconsistent unit conversions raise errors."""
        # Converting length to temperature should fail
        with pytest.raises((ValueError, KeyError)):
            _convert(unit_converter, 100.0, 'm', 'K')


class TestExtendedUnits:
    """Test that extended units are supported."""

    def test_pressure_conversion(self, unit_converter):
        """Test pressure unit conversion."""
        result = _convert(unit_converter, 1000.0, 'Pa', 'kPa')
        assert abs(result - 1.0) < 1e-6

    def test_mass_conversion(self, unit_converter):
        """Test mass unit conversion."""
        result = _convert(unit_converter, 1000.0, 'g', 'kg')
        assert abs(result - 1.0) < 1e-6


class TestTypeConsistency:
    """Test that the converter maintains type consistency."""

    def test_float_input_float_output(self, unit_converter):
        """Test float input returns float output."""
        result = _convert(unit_converter, 25.0, 'celsius', 'K')
        assert isinstance(result, float)

    def test_numpy_input_numpy_output(self, unit_converter):
        """Test numpy input returns numpy output."""
        arr = np.array([25.0])
        result = _convert(unit_converter, arr, 'celsius', 'K')
        assert isinstance(result, np.ndarray)

    def test_int_input_handled(self, unit_converter):
        """Test integer input is handled correctly.
        
        Note: Integer inputs may be rounded to maintain int type.
        25 celsius = 298.15 K, which rounds to 298 K as int.
        """
        result = _convert(unit_converter, 25, 'celsius', 'K')
        assert isinstance(result, (int, float))
        # Accept either exact float (298.15) or rounded int (298)
        assert abs(float(result) - 298.15) < 0.5, \
            f"Expected ~298.15, got {result}"


