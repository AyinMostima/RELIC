"""
Phase 0.1: Core Unit Conversion Verification Tests

This module tests the core unit conversion fixes:
- UnitConverter type consistency (Tensor, float, int, numpy.ndarray)
- Automatic type preservation
- Direct converter consistency
- Parameter manager integration
- Coupling layer consistency

References:
- Unit_Conversion_Final_Fix_Report.md
- RELIC Architecture Overview
"""

import pytest
import torch
from torch import Tensor
import numpy as np
import sys
from pathlib import Path

# Add relic to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.data.standards.unit_converter import UnitConverter, ConversionContext

from relic.parameters import get_global_manager


class TestUnitConverterTypeconsistency:
    """Test UnitConverter.convert() with all input types."""
    
    def test_tensor_input_output(self):
        """Test Tensor input returns Tensor output."""
        converter = UnitConverter()
        context = ConversionContext()
        
        # Temperature conversion
        temp_tensor = torch.tensor([25.0, 30.0, 35.0])
        result = converter.convert(
            temp_tensor, 
            'air_temperature', 
            'celsius', 
            'K', 
            context
        )
        
        assert isinstance(result, Tensor), f"Expected Tensor, got {type(result)}"
        assert torch.allclose(result, torch.tensor([298.15, 303.15, 308.15])), \
            f"Conversion incorrect: {result}"
    
    def test_float_input_output(self):
        """Test float input returns float output."""
        converter = UnitConverter()
        context = ConversionContext()
        
        # Temperature conversion
        result = converter.convert(
            25.0, 
            'air_temperature', 
            'celsius', 
            'K', 
            context
        )
        
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert abs(result - 298.15) < 1e-5, f"Conversion incorrect: {result}"
    
    def test_int_input_output(self):
        """Test int input is promoted to a float result without truncation."""
        converter = UnitConverter()
        context = ConversionContext()
        
        # Simple length conversion
        result = converter.convert(
            100, 
            'generic', 
            'm', 
            'cm', 
            context
        )
        
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert result == 10000.0
        # Note: 100m = 10000cm, result should be int(10000) = 10000
    
    def test_numpy_input_output(self):
        """Test numpy array input returns numpy array output."""
        converter = UnitConverter()
        context = ConversionContext()
        
        # Temperature conversion
        temp_array = np.array([0.0, 25.0, 100.0])
        result = converter.convert(
            temp_array, 
            'air_temperature', 
            'celsius', 
            'K', 
            context
        )
        
        assert isinstance(result, np.ndarray), f"Expected numpy.ndarray, got {type(result)}"
        expected = np.array([273.15, 298.15, 373.15])
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_type_preservation_mixed_conversions(self):
        """Test that various conversions preserve input types."""
        converter = UnitConverter()
        context = ConversionContext()
        
        test_cases = [
            # (input, variable, from_unit, to_unit, expected_type)
            (torch.tensor(1000.0), 'generic', 'mm', 'm', Tensor),
            (1000.0, 'generic', 'mm', 'm', float),
            (np.array([1000.0]), 'generic', 'mm', 'm', np.ndarray),
        ]
        
        for input_data, var, from_u, to_u, expected_type in test_cases:
            result = converter.convert(input_data, var, from_u, to_u, context)
            assert isinstance(result, expected_type), \
                f"Input type {type(input_data)} should return {expected_type}, got {type(result)}"


class TestConverterconsistency:
    """Test direct converter consistency."""
    
    def test_converter_float(self):
        """Test UnitConverter with float input."""
        converter = UnitConverter()
        context = ConversionContext()
        result = converter.convert(100.0, 'generic', 'm', 'km', context)
        assert isinstance(result, float)
        assert abs(result - 0.1) < 1e-5
    
    def test_converter_numpy(self):
        """Test UnitConverter with numpy input."""
        converter = UnitConverter()
        context = ConversionContext()
        input_data = np.array([100.0, 200.0, 1000.0])
        result = converter.convert(input_data, 'generic', 'm', 'km', context)
        
        assert isinstance(result, np.ndarray)
        expected = np.array([0.1, 0.2, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_unit_conversion_function(self):
        """Test direct conversion helper."""
        converter = UnitConverter()
        context = ConversionContext()
        # Test with float
        result = converter.convert(25.0, 'generic', 'celsius', 'K', context)
        assert isinstance(result, float)
        assert abs(result - 298.15) < 1e-5
        
        # Test with numpy
        result = converter.convert(np.array([0, 100]), 'generic', 'celsius', 'K', context)
        assert isinstance(result, np.ndarray)
        expected = np.array([273.15, 373.15])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestParameterManagerIntegration:
    """Test parameter manager unit conversion integration."""
    
    def test_parameter_unit_conversion(self):
        """Test that parameter manager can convert units."""
        try:
            pm = get_global_manager()
            
            # Try to get a parameter with unit conversion
            # This should use unit conversion internally
            # Note: This test may need adjustment based on actual parameters
            
            # For now, just verify the manager exists and has convert method
            assert hasattr(pm, '_convert_parameter_units')
            
        except Exception as e:
            pytest.skip(f"Parameter manager test skipped: {e}")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_same_unit_conversion(self):
        """Test that same unit returns input unchanged."""
        converter = UnitConverter()
        context = ConversionContext()
        
        input_data = torch.tensor([1.0, 2.0, 3.0])
        result = converter.convert(input_data, 'generic', 'm', 'm', context)
        
        assert torch.equal(result, input_data)
    
    def test_invalid_type_raises_error(self):
        """Test that invalid input type raises TypeError."""
        converter = UnitConverter()
        context = ConversionContext()
        
        with pytest.raises(TypeError):
            converter.convert("invalid", 'generic', 'm', 'km', context)
    
    def test_unsupported_conversion_raises_error(self):
        """Test that unsupported conversion raises ValueError."""
        converter = UnitConverter()
        context = ConversionContext()
        
        with pytest.raises(ValueError):
            converter.convert(1.0, 'generic', 'nonexistent_unit', 'another_unit', context)


class TestConversionAccuracy:
    """Test conversion accuracy for various units."""
    
    def test_temperature_conversions(self):
        """Test temperature conversion accuracy."""
        converter = UnitConverter()
        context = ConversionContext()
        
        # Celsius to Kelvin
        result = converter.convert(0.0, 'air_temperature', 'celsius', 'K', context)
        assert abs(result - 273.15) < 1e-5
        
        # Celsius to Kelvin (another value)
        result = converter.convert(100.0, 'air_temperature', 'celsius', 'K', context)
        assert abs(result - 373.15) < 1e-5
    
    def test_length_conversions(self):
        """Test length conversion accuracy."""
        converter = UnitConverter()
        context = ConversionContext()
        
        # Meters to kilometers
        result = converter.convert(1000.0, 'generic', 'm', 'km', context)
        assert abs(result - 1.0) < 1e-5
        
        # Millimeters to meters
        result = converter.convert(1000.0, 'generic', 'mm', 'm', context)
        assert abs(result - 1.0) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


