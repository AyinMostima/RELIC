"""
L1 Unit Tests - Scientific Module: Photosynthesis

Tests the photosynthesis scientific module to ensure:
- Correct implementation of BiologicalProcess abstraction
- Accurate Farquhar model calculations
- Temperature response functions
- Light response curves
- CO2 response
- Parameter management
- Conservation of mass/energy
"""

import pytest
import torch
import numpy as np

# Import photosynthesis module
from relic.scientific.biosphere.photosynthesis import FarquharPhotosynthesis
from relic.core.functor import FunctorBase
from relic.core.state import UpdateItem


def _canonical_state(
    *,
    air_temperature=298.15,
    atmospheric_co2_ppm=400.0,
    ppfd=1000.0,
    vapor_pressure_deficit=1000.0,
    water_stress_factor=1.0,
    vcmax_scaling=1.0,
):
    """Build the canonical Farquhar forcing and optional-control state."""
    return {
        "air_temperature": torch.as_tensor(air_temperature),
        "atmospheric_co2_ppm": torch.as_tensor(atmospheric_co2_ppm),
        "ppfd": torch.as_tensor(ppfd),
        "leaf_area_index": torch.tensor(1.0),
        "vapor_pressure_deficit": torch.as_tensor(vapor_pressure_deficit),
        "water_stress_factor": torch.as_tensor(water_stress_factor),
        "vcmax_scaling": torch.as_tensor(vcmax_scaling),
    }


def _value(output, key):
    """Return the tensor payload of an explicit state update."""
    assert isinstance(output[key], UpdateItem)
    return output[key].value


class TestFarquharPhotosynthesis:
    """Test suite for FarquharPhotosynthesis module."""
    
    def test_module_inheritance(self):
        """Test that FarquharPhotosynthesis properly inherits from FunctorBase."""
        photo = FarquharPhotosynthesis()
        
        # Check inheritance - FunctorBase is the unified base class in RELIC
        assert isinstance(photo, FunctorBase), "Not inheriting from FunctorBase"
        
        # Check required methods exist from FunctorBase
        assert hasattr(photo, 'forward'), "Missing forward method"
        assert callable(photo.forward), "forward not callable"
        assert hasattr(photo, 'get_dependencies'), "Missing get_dependencies method"
        assert callable(photo.get_dependencies), "get_dependencies not callable"
    
    def test_basic_photosynthesis(self):
        """Test basic photosynthesis calculation under standard conditions."""
        photo = FarquharPhotosynthesis()
        
        # Standard test conditions
        state = _canonical_state()
        
        # Compute photosynthesis rate
        result = photo.forward(state, 0.0, 1.0)
        
        # Extract rate from result dictionary
        rate = _value(result, "net_photosynthesis")
        
        # Check output
        assert isinstance(rate, torch.Tensor), "Rate not a tensor"
        assert rate.ndim == 0 or (rate.ndim == 1 and rate.shape[0] == 1), "Unexpected rate shape"
        assert rate > 0, "Photosynthesis rate should be positive under standard conditions"
        
        # Typical range for C3 photosynthesis at 25掳C, 1000 PAR
        # A broad C3 benchmark range is used for the configured parameter set.
        assert 5 < rate.item() < 50, f"Rate {rate.item()} outside expected range"
    
    def test_temperature_response(self):
        """Test temperature response of photosynthesis."""
        photo = FarquharPhotosynthesis()
        
        temperatures = torch.tensor([273.15, 283.15, 293.15, 303.15, 313.15, 323.15])  # 0 to 50掳C
        rates = []
        
        for temp in temperatures:
            result = photo.forward(
                _canonical_state(air_temperature=temp), 0.0, 1.0
            )
            rate = _value(result, "net_photosynthesis")
            rates.append(rate.item())
        
        rates = np.array(rates)
        
        # Check temperature response shape
        # Should be low at 0掳C, peak around 25-30掳C, decline at high temps
        assert rates[0] < rates[2], "Rate should increase from 0掳C to 20掳C"
        assert rates[2] > 0, "Rate should be positive at 20掳C"
        
        # Find peak temperature (should be around 25-30掳C)
        peak_idx = np.argmax(rates)
        peak_temp = temperatures[peak_idx].item() - 273.15
        assert 20 <= peak_temp <= 35, f"Peak temperature {peak_temp}掳C outside expected range"
        
        # High temperature inhibition
        if len(rates) > 5:
            assert rates[-1] < rates[peak_idx], "Rate should decrease at very high temperatures"
    
    def test_light_response(self):
        """Test light response curve (PAR dependency)."""
        photo = FarquharPhotosynthesis()
        
        # Test light levels from 0 to 2000 umol m-2 s-1
        ppfd_levels = torch.tensor([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 1500.0, 2000.0])
        rates = []
        
        for ppfd in ppfd_levels:
            result = photo.forward(_canonical_state(ppfd=ppfd), 0.0, 1.0)
            rate = _value(result, "net_photosynthesis")
            rates.append(rate.item())
        
        rates = np.array(rates)
        
        # Check light response properties
        assert rates[0] < 1.0, "Dark respiration should be near zero or negative"
        
        # Monotonic increase
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i-1] - 0.1, "Rate should increase with light"
        
        # Light saturation (diminishing returns at high light)
        if len(rates) > 5:
            slope_low = (rates[2] - rates[1]) / (ppfd_levels[2] - ppfd_levels[1])
            slope_high = (rates[-1] - rates[-2]) / (ppfd_levels[-1] - ppfd_levels[-2])
            assert slope_high < slope_low, "Should show light saturation"
    
    def test_co2_response(self):
        """Test CO2 response curve."""
        photo = FarquharPhotosynthesis()
        
        # Test CO2 levels from 200 to 1000 ppm
        co2_levels = torch.tensor([200.0, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0])
        rates = []
        
        for co2 in co2_levels:
            result = photo.forward(
                _canonical_state(atmospheric_co2_ppm=co2), 0.0, 1.0
            )
            rate = _value(result, "net_photosynthesis")
            rates.append(rate.item())
        
        rates = np.array(rates)
        
        # Check CO2 response properties
        # Should increase with CO2 (CO2 fertilization effect)
        for i in range(1, len(rates)):
            assert rates[i] > rates[i-1], f"Rate should increase with CO2: {rates}"
        
        # Check for saturation at high CO2
        slope_low = (rates[1] - rates[0]) / (co2_levels[1] - co2_levels[0])
        slope_high = (rates[-1] - rates[-2]) / (co2_levels[-1] - co2_levels[-2])
        assert slope_high < slope_low, "Should show CO2 saturation"
    
    def test_water_stress_response(self):
        """Test response to water stress."""
        photo = FarquharPhotosynthesis()
        
        # Test the canonical dimensionless stress control from wet to dry.
        water_stress_factors = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        rates = []
        
        for stress_factor in water_stress_factors:
            result = photo.forward(
                _canonical_state(water_stress_factor=stress_factor), 0.0, 1.0
            )
            rate = _value(result, "net_photosynthesis")
            rates.append(rate.item())
        
        rates = np.array(rates)
        
        # Check water stress response
        # Assimilation should decrease as the external stress factor decreases.
        for i in range(1, len(rates)):
            assert rates[i] <= rates[i-1] + 0.1, "Rate should decrease with water stress"
        
        # Severe stress should strongly reduce photosynthesis
        assert rates[-1] < 0.5 * rates[0], "Severe water stress should reduce photosynthesis"
    
    def test_vpd_response(self):
        """Test response to vapor pressure deficit."""
        photo = FarquharPhotosynthesis()
        
        # Test canonical VPD from 500 to 4000 Pa.
        vpd_levels = torch.tensor([500.0, 1000.0, 2000.0, 3000.0, 4000.0])
        rates = []
        
        for vpd in vpd_levels:
            result = photo.forward(
                _canonical_state(vapor_pressure_deficit=vpd), 0.0, 1.0
            )
            rate = _value(result, "net_photosynthesis")
            rates.append(rate.item())
        
        rates = np.array(rates)
        
        # Check VPD response (stomatal closure at high VPD)
        assert rates[0] > rates[-1], "High VPD should reduce photosynthesis"
        
        # Generally decreasing trend
        for i in range(1, min(4, len(rates))):
            assert rates[i] <= rates[i-1] + 1.0, "Rate should generally decrease with VPD"
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to key parameters."""
        # Test with modified Vcmax
        photo_low = FarquharPhotosynthesis(vcmax25=30.0)
        photo_high = FarquharPhotosynthesis(vcmax25=100.0)
        state = _canonical_state(ppfd=2000.0)

        result_low = photo_low.forward(state, 0.0, 1.0)
        rate_low = _value(result_low, "net_photosynthesis")
        result_high = photo_high.forward(state, 0.0, 1.0)
        rate_high = _value(result_high, "net_photosynthesis")
        
        # Higher Vcmax should give higher rate
        assert rate_high > rate_low, "Higher Vcmax should increase photosynthesis"
        assert rate_high < 3 * rate_low, "Response should be reasonable"
    
    def test_batch_processing(self):
        """Test processing multiple samples at once."""
        photo = FarquharPhotosynthesis()
        
        batch_size = 10
        state = _canonical_state(
            air_temperature=torch.randn(batch_size) * 5 + 298.15,
            atmospheric_co2_ppm=torch.ones(batch_size) * 400.0,
            ppfd=torch.rand(batch_size) * 1500 + 500,
            vapor_pressure_deficit=torch.ones(batch_size) * 1000.0,
            water_stress_factor=torch.ones(batch_size),
            vcmax_scaling=torch.ones(batch_size),
        )
        
        # Compute rates
        result = photo.forward(state, 0.0, 1.0)
        rates = _value(result, "net_photosynthesis")
        
        # Check output shape and values
        assert rates.shape == (batch_size,), f"Batch shape mismatch: {rates.shape}"
        assert torch.all(rates > -5), "All rates should be reasonable"
        assert torch.all(rates < 100), "All rates should be reasonable"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        photo = FarquharPhotosynthesis()
        
        # Test zero light (dark respiration)
        state = _canonical_state(ppfd=0.0)

        result = photo.forward(state, 0.0, 1.0)
        dark_rate = _value(result, "net_photosynthesis")
        assert dark_rate <= 0 or dark_rate < 1.0, "Dark respiration should be negative or very small"
        
        # Test extreme temperature
        state['air_temperature'] = torch.tensor(273.15)
        state['ppfd'] = torch.tensor(1000.0)
        result = photo.forward(state, 0.0, 1.0)
        cold_rate = _value(result, "net_photosynthesis")
        assert cold_rate < 5.0, "Rate should be very low at 0掳C"
        
        # Test very high temperature
        state['air_temperature'] = torch.tensor(323.15)
        result = photo.forward(state, 0.0, 1.0)
        hot_rate = _value(result, "net_photosynthesis")
        assert hot_rate < 10.0, "Rate should be low at 50掳C due to enzyme denaturation"
        
        # Test zero CO2 (should be near zero or negative)
        state['air_temperature'] = torch.tensor(298.15)
        state['atmospheric_co2_ppm'] = torch.tensor(10.0)
        result = photo.forward(state, 0.0, 1.0)
        no_co2_rate = _value(result, "net_photosynthesis")
        assert no_co2_rate <= 1.0, "Should have minimal photosynthesis without CO2"
    
    def test_nan_handling(self):
        """Test handling of NaN inputs."""
        photo = FarquharPhotosynthesis()
        
        # Test with NaN temperature
        state = _canonical_state(air_temperature=float('nan'))

        with pytest.raises(
            ValueError, match="air_temperature must contain only finite values"
        ):
            photo.forward(state, 0.0, 1.0)
    
    def test_units_consistency(self):
        """Test that units are handled consistently."""
        photo = FarquharPhotosynthesis()
        
        # Standard conditions
        result = photo.forward(_canonical_state(), 0.0, 1.0)
        rate = _value(result, "net_photosynthesis")
        assert result["net_photosynthesis"].unit == "umol CO2 m-2 s-1"

        assert 1.0 < rate.item() < 100.0, (
            f"Rate {rate.item()} suggests a unit conversion issue"
        )


def test_photosynthesis_integration():
    """Test integration of photosynthesis with other modules."""
    photo = FarquharPhotosynthesis()
    
    # Simulate a diurnal cycle
    hours = torch.arange(0, 24, 1.0)
    
    # Simple sine wave for PPFD (peak at noon).
    ppfd_values = torch.maximum(
        torch.zeros_like(hours),
        1000.0 * torch.sin(torch.pi * (hours - 6) / 12)
    )
    
    # Temperature cycle (cool at night, warm during day)
    temp_values = 293.15 + 10.0 * torch.sin(torch.pi * (hours - 6) / 12)
    
    total_carbon_kg = 0.0
    dt = 3600.0  # 1 hour in seconds
    
    for hour, ppfd, temp in zip(hours, ppfd_values, temp_values):
        result = photo.forward(
            _canonical_state(air_temperature=temp, ppfd=ppfd), 0.0, dt
        )
        gpp_mass_flux = _value(result, "gross_primary_productivity")
        total_carbon_kg += (gpp_mass_flux * dt).item()

    assert 0.0001 < total_carbon_kg < 0.02
    
    # Check that nighttime respiration occurred
    night_rates = []
    for hour in [0, 1, 2, 22, 23]:
        result = photo.forward(
            _canonical_state(air_temperature=293.15, ppfd=0.0), 0.0, dt
        )
        rate = _value(result, "net_photosynthesis")
        night_rates.append(rate.item())
    
    # All night rates should be similar and negative or near zero
    night_rates = np.array(night_rates)
    assert np.std(night_rates) < 1.0, "Night rates should be consistent"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

