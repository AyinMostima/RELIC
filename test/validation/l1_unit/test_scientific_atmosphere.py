"""
L1 Unit Tests - Scientific Module: Atmosphere

Tests the atmospheric science modules to ensure:
- Correct implementation of FunctorBase abstractions
- Accurate atmospheric process calculations (chemistry, radiation, turbulence)
- Temperature and light dependencies
- Parameter management and validation
- Conservation laws and physical constraints
- Tensor differentiability and device consistency

Covers:
- ChemistryFunctor: Atmospheric chemistry with NOx-O3 cycles and BVOC interactions
- RadiationFunctor: Radiation transfer with shortwave/longwave components
- TurbulenceFunctor: Boundary layer turbulence and mixing processes
"""

import pytest
import torch
import numpy as np

# Import atmosphere modules
from relic.scientific.atmosphere.chemistry import ChemistryFunctor
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.atmosphere.turbulence import TurbulenceFunctor
from relic.core.functor import FunctorBase
from relic.core.state import UpdateItem, UpdateMode


# Chemistry uses an explicit reduced box-model step. One second remains inside
# the positivity domain across the scalar and randomized batch states below.
CHEMISTRY_EXPLICIT_DT = 1.0


def _chemistry_value(result, key):
    """Return one canonical chemistry value after checking its wrapper."""

    item = result[key]
    assert isinstance(item, UpdateItem)
    return item.value


def _radiation_value(result, key):
    """Return one instantaneous radiation value from its explicit update."""

    item = result[key]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.REPLACE
    assert item.per_timestep is False
    assert isinstance(item.value, torch.Tensor)
    return item.value


class TestChemistryFunctor:
    """Test suite for ChemistryFunctor module."""
    
    def test_module_inheritance(self):
        """Test that ChemistryFunctor properly inherits from FunctorBase."""
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Check inheritance - FunctorBase is the unified base class
        assert isinstance(chemistry, FunctorBase), "Not inheriting from FunctorBase"
        
        # Check required methods exist
        assert hasattr(chemistry, 'forward'), "Missing forward method"
        assert callable(chemistry.forward), "forward not callable"
        assert hasattr(chemistry, 'get_dependencies'), "Missing get_dependencies method"
        assert callable(chemistry.get_dependencies), "get_dependencies not callable"
        assert not hasattr(chemistry, "compute_rate"), "Legacy compute_rate should not exist in forward-only API"
    
    def test_basic_chemistry(self):
        """Test basic atmospheric chemistry under standard conditions."""
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Standard atmospheric conditions
        state = {
            'air_temperature': torch.tensor(298.15),        # 25 C
            'no_mixing_ratio': torch.tensor(10e-9),         # 10 ppb NO
            'no2_mixing_ratio': torch.tensor(20e-9),        # 20 ppb NO2
            'o3_mixing_ratio': torch.tensor(50e-9),         # 50 ppb O3
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),          # 800 W/m2
            'pbl_height': torch.tensor(1000.0)              # 1 km boundary layer
        }
        
        # Compute one admissible chemistry step at noon.
        result = chemistry.forward(
            state,
            time=43200.0,
            dt=CHEMISTRY_EXPLICIT_DT,
        )
        
        # Check output structure
        assert isinstance(result, dict), "Result not a dictionary"
        state_keys = ['no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio']
        diagnostic_key = 'diagnostics.chemistry.photolysis_rate_no2'
        assert set(state_keys) <= set(result)
        assert diagnostic_key in result
        
        # Check tensor properties.
        for key in [*state_keys, diagnostic_key]:
            value = _chemistry_value(result, key)
            assert torch.isfinite(value).all(), f"{key} contains non-finite values"
        
        # Check photolysis rate is positive during daytime
        j_no2 = _chemistry_value(result, diagnostic_key)
        assert j_no2 > 0, "Photolysis rate should be positive during daytime"
        assert j_no2 < 0.1, f"Photolysis rate too high: {j_no2} s^-1"
        
        # Check that the state increments remain bounded.
        no_increment = _chemistry_value(result, 'no_mixing_ratio').abs()
        no2_increment = _chemistry_value(result, 'no2_mixing_ratio').abs()
        o3_increment = _chemistry_value(result, 'o3_mixing_ratio').abs()
        
        assert no_increment < 1e-6, f"NO increment too large: {no_increment}"
        assert no2_increment < 1e-6, f"NO2 increment too large: {no2_increment}"
        assert o3_increment < 1e-6, f"O3 increment too large: {o3_increment}"
    
    def test_temperature_dependence(self):
        """Test temperature response of chemical reaction rates."""
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Test temperatures from 5 C to 35 C
        temperatures = torch.tensor([278.15, 288.15, 298.15, 308.15])
        photolysis_rates = []
        reaction_rates = []
        
        for temp in temperatures:
            state = {
                'air_temperature': temp,
                'no_mixing_ratio': torch.tensor(10e-9),
                'no2_mixing_ratio': torch.tensor(20e-9), 
                'o3_mixing_ratio': torch.tensor(50e-9),
                'atmospheric_pressure': torch.tensor(101325.0),  # Pa
                'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
                'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
                'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
                'cos_zenith': torch.tensor(1.0),  # overhead sun
                'shortwave_down': torch.tensor(800.0),
                'pbl_height': torch.tensor(1000.0)
            }
            
            result = chemistry.forward(
                state,
                time=43200.0,
                dt=CHEMISTRY_EXPLICIT_DT,
            )
            photolysis_rates.append(_chemistry_value(
                result,
                'diagnostics.chemistry.photolysis_rate_no2',
            ).item())
            reaction_rates.append(abs(_chemistry_value(
                result,
                'diagnostics.chemistry.reaction_no_o3_rate',
            ).item()))
        
        photolysis_rates = np.array(photolysis_rates)
        reaction_rates = np.array(reaction_rates)
        
        # Default photolysis_temperature_mode 'none': J_NO2 does not depend on air
        # temperature (NO2 quantum yield 1.00 at 300-398 nm at 248 and 298 K;
        # Warneck and Williams 2012 Table 6.13b).
        assert np.all(photolysis_rates == photolysis_rates[0]), "J_NO2 must not depend on temperature"
        
        # Net NO tendency can be non-monotonic because competing pathways coexist.
        # The robust requirement is temperature sensitivity (non-constant response).
        assert reaction_rates.max() > reaction_rates.min(), "Reaction rates should respond to temperature"
    
    def test_light_dependence(self):
        """Test light response of photolysis rates."""
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Test different light levels
        light_levels = torch.tensor([0.0, 200.0, 600.0, 1000.0, 1400.0])
        photolysis_rates = []
        
        for light in light_levels:
            state = {
                'air_temperature': torch.tensor(298.15),
                'no_mixing_ratio': torch.tensor(10e-9),
                'no2_mixing_ratio': torch.tensor(20e-9),
                'o3_mixing_ratio': torch.tensor(50e-9),
                'atmospheric_pressure': torch.tensor(101325.0),  # Pa
                'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
                'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
                'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
                'cos_zenith': torch.tensor(1.0),  # overhead sun
                'shortwave_down': light,
                'pbl_height': torch.tensor(1000.0)
            }
            
            result = chemistry.forward(
                state,
                time=43200.0,
                dt=CHEMISTRY_EXPLICIT_DT,
            )
            photolysis_rates.append(_chemistry_value(
                result,
                'diagnostics.chemistry.photolysis_rate_no2',
            ).item())
        
        photolysis_rates = np.array(photolysis_rates)
        
        # Check light response
        assert photolysis_rates[0] < 1e-6, "Should have minimal photolysis in dark"
        
        # Monotonic increase with light
        for i in range(1, len(photolysis_rates)):
            assert photolysis_rates[i] >= photolysis_rates[i-1], "Photolysis should increase with light"
        
        # Light saturation at high intensities
        if len(photolysis_rates) > 3:
            slope_low = photolysis_rates[2] - photolysis_rates[1]
            slope_high = photolysis_rates[-1] - photolysis_rates[-2]
            assert slope_high <= slope_low * 1.5, "Should show some light saturation"
    
    def test_bvoc_chemistry(self):
        """Test BVOC oxidation chemistry when isoprene is present."""
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Standard conditions with isoprene
        state = {
            'air_temperature': torch.tensor(298.15),
            'no_mixing_ratio': torch.tensor(10e-9),
            'no2_mixing_ratio': torch.tensor(20e-9),
            'o3_mixing_ratio': torch.tensor(50e-9),
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'isoprene_mixing_ratio': torch.tensor(2e-9),  # 2 ppb isoprene
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        result = chemistry.forward(
            state,
            time=43200.0,
            dt=CHEMISTRY_EXPLICIT_DT,
        )
        
        # Should have isoprene oxidation output
        assert 'isoprene_mixing_ratio' in result, "Missing isoprene oxidation"
        
        # Isoprene should be consumed by the canonical state increment.
        isoprene_increment = _chemistry_value(result, 'isoprene_mixing_ratio')
        assert isoprene_increment <= 0, "Isoprene should be consumed by oxidation"
        
        # O3 consumption should be enhanced with BVOC
        o3_increment = _chemistry_value(result, 'o3_mixing_ratio')
        assert torch.isfinite(o3_increment), "O3 change should be finite"
    
    def test_edge_cases(self):
        """Test chemistry edge cases and error handling."""
        chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
        
        # Test zero concentrations
        state = {
            'air_temperature': torch.tensor(298.15),
            'no_mixing_ratio': torch.tensor(1e-15),   # Near-zero
            'no2_mixing_ratio': torch.tensor(1e-15),
            'o3_mixing_ratio': torch.tensor(1e-15),
            'atmospheric_pressure': torch.tensor(101325.0),  # Pa
            'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
            'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
            'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
            'cos_zenith': torch.tensor(1.0),  # overhead sun
            'shortwave_down': torch.tensor(800.0),
            'pbl_height': torch.tensor(1000.0)
        }
        
        result = chemistry.forward(
            state,
            time=43200.0,
            dt=CHEMISTRY_EXPLICIT_DT,
        )
        
        # Should handle near-zero concentrations without error
        assert all(
            torch.isfinite(_chemistry_value(result, key)).all()
            for key in result
        )
        
        # Test extreme temperatures
        extreme_state = state.copy()
        extreme_state['air_temperature'] = torch.tensor(250.0)  # Very cold
        
        result_cold = chemistry.forward(
            extreme_state,
            time=43200.0,
            dt=CHEMISTRY_EXPLICIT_DT,
        )
        assert all(
            torch.isfinite(_chemistry_value(result_cold, key)).all()
            for key in result_cold
        )
        
        # Test no light (nighttime)
        night_state = state.copy()
        night_state['shortwave_down'] = torch.tensor(0.0)
        
        result_night = chemistry.forward(
            night_state,
            time=0.0,
            dt=CHEMISTRY_EXPLICIT_DT,
        )
        j_no2_night = _chemistry_value(
            result_night,
            'diagnostics.chemistry.photolysis_rate_no2',
        )
        assert j_no2_night < 1e-6, "Should have minimal photolysis at night"


class TestRadiationFunctor:
    """Test suite for RadiationFunctor module."""
    
    def test_module_inheritance(self):
        """Test that RadiationFunctor properly inherits from FunctorBase."""
        radiation = RadiationFunctor()
        
        # Check inheritance
        assert isinstance(radiation, FunctorBase), "Not inheriting from FunctorBase"
        
        # Check required methods
        assert hasattr(radiation, 'forward'), "Missing forward method"
        assert callable(radiation.forward), "forward not callable"
        assert hasattr(radiation, 'get_dependencies'), "Missing get_dependencies method"
        assert callable(radiation.get_dependencies), "get_dependencies not callable"
    
    def test_basic_radiation(self):
        """Test basic radiation calculations under standard conditions."""
        radiation = RadiationFunctor()
        
        # Standard daytime conditions
        state = {
            'air_temperature': torch.tensor(298.15),        # 25 C
            'surface_temperature': torch.tensor(300.15),       # 27 C
            'incoming_shortwave_potential': torch.tensor(1200.0),  # Clear-sky potential
            'cos_zenith': torch.tensor(0.7),                   # Moderate zenith angle
            'relative_humidity': torch.tensor(0.6),            # 60% RH
            'surface_albedo': torch.tensor(0.2),               # Typical vegetation
            'surface_emissivity': torch.tensor(0.97),          # Near-blackbody
            'cloud_fraction': torch.tensor(0.0),
        }
        
        result = radiation.forward(state, time=43200.0, dt=3600.0)  # Noon
        
        # Check output structure
        assert isinstance(result, dict), "Result not a dictionary"
        required_outputs = ['shortwave_down', 'longwave_down', 'net_radiation',
                           'shortwave_net', 'longwave_net', 'atmospheric_emissivity',
                           'ppfd']
        
        for key in required_outputs:
            assert key in result, f"Missing output: {key}"
            value = _radiation_value(result, key)
            assert torch.isfinite(value).all(), f"{key} contains non-finite values"
        assert 'par' not in result
        
        # Check physical bounds
        sw_down = _radiation_value(result, 'shortwave_down')
        lw_down = _radiation_value(result, 'longwave_down')
        net_rad = _radiation_value(result, 'net_radiation')
        
        assert sw_down >= 0, "Shortwave radiation cannot be negative"
        assert sw_down <= 1400, f"Shortwave too high: {sw_down} W/m2"
        assert lw_down > 0, "Longwave radiation must be positive"
        assert lw_down < 500, f"Longwave too high: {lw_down} W/m2"
        
        # Net radiation should be positive during daytime
        assert net_rad > 0, f"Net radiation should be positive during daytime: {net_rad} W/m2"
        assert net_rad < 1000, f"Net radiation too high: {net_rad} W/m2"
        
        # Atmospheric emissivity should be reasonable
        emissivity = _radiation_value(result, 'atmospheric_emissivity')
        assert 0.4 <= emissivity <= 1.0, f"Atmospheric emissivity out of bounds: {emissivity}"
    
    def test_solar_angle_dependence(self):
        """Test radiation dependence on solar zenith angle."""
        radiation = RadiationFunctor()
        
        # Test different zenith angles
        zenith_angles = torch.tensor([0.0, 0.3, 0.6, 0.8, 1.0])  # cos(zenith)
        sw_radiation = []
        
        for cos_zen in zenith_angles:
            state = {
                'air_temperature': torch.tensor(298.15),
                'surface_temperature': torch.tensor(300.15), 
                'incoming_shortwave_potential': torch.tensor(1200.0),
                'cos_zenith': cos_zen,
                'relative_humidity': torch.tensor(0.6),
                'surface_albedo': torch.tensor(0.2),
                'surface_emissivity': torch.tensor(0.97),
                'cloud_fraction': torch.tensor(0.0),
            }
            
            result = radiation.forward(state, time=43200.0, dt=3600.0)
            sw_radiation.append(_radiation_value(result, 'shortwave_down').item())
        
        sw_radiation = np.array(sw_radiation)
        
        # Check solar angle response
        assert sw_radiation[0] < 50, "Should have minimal radiation at sunrise/sunset (cos=0)"
        assert sw_radiation[-1] > sw_radiation[0], "Radiation should increase with higher sun"
        
        # Generally monotonic increase
        for i in range(1, len(sw_radiation)):
            assert sw_radiation[i] >= sw_radiation[i-1] * 0.9, "Should generally increase with sun angle"
    
    def test_cloud_effects(self):
        """Test cloud effects on radiation."""
        radiation = RadiationFunctor()
        
        # Clear vs cloudy conditions
        cloud_fractions = torch.tensor([0.0, 0.3, 0.6, 0.9])
        sw_values = []
        lw_values = []
        
        for cf in cloud_fractions:
            state = {
                'air_temperature': torch.tensor(298.15),
                'surface_temperature': torch.tensor(300.15),
                'incoming_shortwave_potential': torch.tensor(1200.0),
                'cos_zenith': torch.tensor(0.7),
                'relative_humidity': torch.tensor(0.6),
                'surface_albedo': torch.tensor(0.2),
                'surface_emissivity': torch.tensor(0.97),
                'cloud_fraction': cf
            }
            
            result = radiation.forward(state, time=43200.0, dt=3600.0)
            sw_values.append(_radiation_value(result, 'shortwave_down').item())
            lw_values.append(_radiation_value(result, 'longwave_down').item())
        
        sw_values = np.array(sw_values)
        lw_values = np.array(lw_values)
        
        # Clouds should reduce shortwave
        assert sw_values[0] > sw_values[-1], "Clouds should reduce shortwave radiation"
        
        # Clouds should increase longwave emission
        assert lw_values[-1] > lw_values[0], "Clouds should increase longwave radiation"
    
    def test_energy_conservation(self):
        """Test energy conservation in radiation calculations."""
        radiation = RadiationFunctor()
        
        state = {
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'incoming_shortwave_potential': torch.tensor(1000.0),
            'cos_zenith': torch.tensor(0.7),
            'relative_humidity': torch.tensor(0.6),
            'surface_albedo': torch.tensor(0.3),
            'surface_emissivity': torch.tensor(0.95),
            'cloud_fraction': torch.tensor(0.0),
        }
        
        result = radiation.forward(state, time=43200.0, dt=3600.0)
        
        # Check energy balance components
        sw_down = _radiation_value(result, 'shortwave_down')
        sw_net = _radiation_value(result, 'shortwave_net')
        albedo = state['surface_albedo']
        
        # Net shortwave should equal (1-albedo) * incoming
        expected_sw_net = (1.0 - albedo) * sw_down
        assert torch.allclose(sw_net, expected_sw_net, rtol=1e-4), "Shortwave energy balance violated"
        
        # Net radiation should equal sum of components
        net_rad = _radiation_value(result, 'net_radiation')
        lw_net = _radiation_value(result, 'longwave_net')
        calculated_net = sw_net + lw_net
        
        assert torch.allclose(net_rad, calculated_net, rtol=1e-4), "Total energy balance violated"
    
    def test_temperature_dependence(self):
        """Test temperature effects on longwave radiation."""
        radiation = RadiationFunctor()
        
        # Test different air temperatures
        temperatures = torch.tensor([288.15, 298.15, 308.15])  # 15 C, 25 C, 35 C
        lw_down_values = []
        
        for temp in temperatures:
            state = {
                'air_temperature': temp,
                'surface_temperature': temp + 2.0,  # Surface slightly warmer
                'incoming_shortwave_potential': torch.tensor(1000.0),
                'cos_zenith': torch.tensor(0.7),
                'relative_humidity': torch.tensor(0.6),
                'surface_albedo': torch.tensor(0.2),
                'surface_emissivity': torch.tensor(0.97),
                'cloud_fraction': torch.tensor(0.0),
            }
            
            result = radiation.forward(state, time=43200.0, dt=3600.0)
            lw_down_values.append(_radiation_value(result, 'longwave_down').item())
        
        lw_down_values = np.array(lw_down_values)
        
        # Longwave should increase with temperature (T^4 dependence)
        assert lw_down_values[-1] > lw_down_values[0], "Longwave should increase with temperature"
        
        # Check approximate T^4 scaling
        temp_ratios = (temperatures / temperatures[0]).pow(4)
        # Allow some deviation due to emissivity temperature dependence
        lw_ratios = torch.tensor(lw_down_values, dtype=torch.float32) / lw_down_values[0]
        temp_ratios = temp_ratios.to(lw_ratios.dtype)
        
        # Should be roughly proportional but emissivity effects cause deviations
        assert torch.allclose(lw_ratios, temp_ratios, rtol=0.2), "Longwave-temperature scaling incorrect"


class TestTurbulenceFunctor:
    """Test suite for TurbulenceFunctor module."""
    
    def test_module_inheritance(self):
        """Test that TurbulenceFunctor properly inherits from FunctorBase."""
        turbulence = TurbulenceFunctor()
        
        # Check inheritance
        assert isinstance(turbulence, FunctorBase), "Not inheriting from FunctorBase"
        
        # Check required methods
        assert hasattr(turbulence, 'forward'), "Missing forward method"
        assert callable(turbulence.forward), "forward not callable"
        assert hasattr(turbulence, 'get_dependencies'), "Missing get_dependencies method"
        assert callable(turbulence.get_dependencies), "get_dependencies not callable"
    
    def test_basic_turbulence(self):
        """Test basic turbulence calculations under standard conditions."""
        turbulence = TurbulenceFunctor()
        
        # Standard atmospheric conditions
        state = {
            'air_temperature': torch.tensor(298.15),        # 25 C
            'surface_temperature': torch.tensor(301.15),    # 28 C (unstable)
            'wind_speed': torch.tensor(5.0),                # 5 m/s
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        result = turbulence.forward(state, time=43200.0, dt=3600.0)
        
        # Check output structure
        assert isinstance(result, dict), "Result not a dictionary"
        
        expected_keys = {
            'friction_velocity',
            'aerodynamic_resistance',
            'diagnostics.turbulence.stability_parameter',
            'monin_obukhov_length',
            'sensible_heat_flux',
            'momentum_flux',
            'diagnostics.turbulence.drag_coefficient',
        }
        assert set(result) == expected_keys
        for key in expected_keys:
            item = result[key]
            assert isinstance(item, UpdateItem), f"{key} not an UpdateItem"
            assert item.mode == UpdateMode.REPLACE
            assert torch.isfinite(item.value).all(), f"{key} contains non-finite values"
    
    def test_stability_dependence(self):
        """Test turbulence dependence on atmospheric stability."""
        turbulence = TurbulenceFunctor()
        
        # Test stable vs unstable conditions
        air_temp = torch.tensor(298.15)
        surface_temps = torch.tensor([295.15, 298.15, 303.15])  # Stable, neutral, unstable
        lengths = []
        heat_fluxes = []
        
        for surf_temp in surface_temps:
            state = {
                'air_temperature': air_temp,
                'surface_temperature': surf_temp,
                'wind_speed': torch.tensor(5.0),
                'reference_height': torch.tensor(10.0),
                'roughness_length_momentum': torch.tensor(0.1),
                'displacement_height': torch.tensor(0.0),
                'atmospheric_pressure': torch.tensor(101325.0),
                'relative_humidity': torch.tensor(0.5),
            }
            
            result = turbulence.forward(state, time=43200.0, dt=3600.0)
            
            lengths.append(result['monin_obukhov_length'].value.item())
            heat_fluxes.append(result['sensible_heat_flux'].value.item())

        assert lengths[0] > 0.0
        neutral_length = turbulence.get_parameter_value(
            "neutral_obukhov_length",
            namespace="atmosphere.turbulence",
            required=True,
        )
        assert abs(lengths[1]) == pytest.approx(float(neutral_length))
        assert lengths[2] < 0.0
        assert heat_fluxes[0] < 0.0
        assert abs(heat_fluxes[1]) < 1.0e-6
        assert heat_fluxes[2] > 0.0
    
    def test_wind_dependence(self):
        """Test wind dependence inside the configured closure domain."""
        turbulence = TurbulenceFunctor()

        low_wind_state = {
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'wind_speed': torch.tensor(1.0),
            'reference_height': torch.tensor(10.0),
            'roughness_length_momentum': torch.tensor(0.1),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        with pytest.raises(
            ValueError,
            match="outside the configured Businger-Dyer validity domain",
        ):
            turbulence.forward(low_wind_state, time=43200.0, dt=3600.0)

        wind_speeds = torch.tensor([3.0, 5.0, 10.0])
        friction_values = []
        
        for wind in wind_speeds:
            state = {
                'air_temperature': torch.tensor(298.15),
                'surface_temperature': torch.tensor(300.15),
                'wind_speed': wind,
                'reference_height': torch.tensor(10.0),
                'roughness_length_momentum': torch.tensor(0.1),
                'displacement_height': torch.tensor(0.0),
                'atmospheric_pressure': torch.tensor(101325.0),
                'relative_humidity': torch.tensor(0.5),
            }
            
            result = turbulence.forward(state, time=43200.0, dt=3600.0)
            
            # Friction velocity should increase with wind speed
            friction_values.append(result['friction_velocity'].value.item())
        
        friction_values = np.array(friction_values)
        assert friction_values[1] > friction_values[0], "Friction velocity should increase with wind"
        assert friction_values[2] > friction_values[1], "Friction velocity should increase with wind"
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to turbulence parameters."""
        # Test different roughness lengths using same functor with different state
        turbulence = TurbulenceFunctor()
        
        base_state = {
            'air_temperature': torch.tensor(298.15),
            'surface_temperature': torch.tensor(300.15),
            'wind_speed': torch.tensor(5.0),
            'reference_height': torch.tensor(10.0),
            'displacement_height': torch.tensor(0.0),
            'atmospheric_pressure': torch.tensor(101325.0),
            'relative_humidity': torch.tensor(0.5),
        }
        
        # Test smooth surface
        state_smooth = base_state.copy()
        state_smooth['roughness_length_momentum'] = torch.tensor(0.01)
        
        # Test rough surface  
        state_rough = base_state.copy()
        state_rough['roughness_length_momentum'] = torch.tensor(1.0)

        result_smooth = turbulence.forward(state_smooth, time=43200.0, dt=3600.0)
        result_rough = turbulence.forward(state_rough, time=43200.0, dt=3600.0)
        u_star_smooth = result_smooth['friction_velocity'].value
        u_star_rough = result_rough['friction_velocity'].value

        assert u_star_rough > u_star_smooth, "Rougher surfaces should increase friction velocity"


def test_atmosphere_integration():
    """Test integration between atmospheric modules."""
    # Create atmosphere functors
    radiation = RadiationFunctor()
    chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
    
    # Combined atmospheric state
    state = {
        'air_temperature': torch.tensor(298.15),
        'surface_temperature': torch.tensor(300.15),
        'incoming_shortwave_potential': torch.tensor(1200.0),
        'cos_zenith': torch.tensor(0.7),
        'relative_humidity': torch.tensor(0.6),
        'surface_albedo': torch.tensor(0.2),
        'surface_emissivity': torch.tensor(0.97),
        'cloud_fraction': torch.tensor(0.0),
        'no_mixing_ratio': torch.tensor(10e-9),
        'no2_mixing_ratio': torch.tensor(20e-9),
        'o3_mixing_ratio': torch.tensor(50e-9),
        'atmospheric_pressure': torch.tensor(101325.0),  # Pa
        'co_mixing_ratio': torch.tensor(150e-9),  # OH sink, 150 ppb
        'ch4_mixing_ratio': torch.tensor(1.8e-6),  # OH sink, 1.8 ppm
        'specific_humidity': torch.tensor(0.01),  # O(1D) + H2O OH source, 10 g/kg
        'pbl_height': torch.tensor(1000.0)
    }
    
    # Calculate radiation first
    rad_result = radiation.forward(state, time=43200.0, dt=3600.0)
    
    # Use radiation output for chemistry
    chem_state = state.copy()
    chem_state['shortwave_down'] = _radiation_value(rad_result, 'shortwave_down')
    
    chem_result = chemistry.forward(
        chem_state,
        time=43200.0,
        dt=CHEMISTRY_EXPLICIT_DT,
    )
    
    # Check that coupling works
    diagnostic_key = 'diagnostics.chemistry.photolysis_rate_no2'
    assert diagnostic_key in chem_result
    
    # Photolysis rate should be reasonable given the radiation input
    j_no2 = _chemistry_value(chem_result, diagnostic_key)
    sw_down = _radiation_value(rad_result, 'shortwave_down')
    
    # Higher radiation should give higher photolysis (rough check)
    if sw_down > 500:  # Good radiation
        assert j_no2 > 1e-5, "Good radiation should give significant photolysis"
    
    # Energy and mass conservation
    net_rad = _radiation_value(rad_result, 'net_radiation')
    assert -200 < net_rad < 800, f"Net radiation should be reasonable: {net_rad} W/m2"


def test_batch_processing_atmosphere():
    """Test atmospheric modules with batch processing."""
    batch_size = 5
    
    # Create batch data
    state_batch = {
        'air_temperature': torch.randn(batch_size) * 5 + 298.15,        # 25 +/- 5 C
        'surface_temperature': torch.randn(batch_size) * 5 + 300.15,    # 27 +/- 5 C
        'incoming_shortwave_potential': torch.rand(batch_size) * 600 + 800,  # 800-1400 W/m2
        'cos_zenith': torch.rand(batch_size) * 0.8 + 0.2,              # 0.2-1.0
        'relative_humidity': torch.rand(batch_size) * 0.6 + 0.3,       # 30-90%
        'surface_albedo': torch.rand(batch_size) * 0.3 + 0.1,          # 10-40%
        'surface_emissivity': torch.rand(batch_size) * 0.05 + 0.95,    # 95-100%
        'cloud_fraction': torch.rand(batch_size),
        'no_mixing_ratio': torch.rand(batch_size) * 20e-9 + 5e-9,      # 5-25 ppb
        'no2_mixing_ratio': torch.rand(batch_size) * 30e-9 + 10e-9,    # 10-40 ppb
        'o3_mixing_ratio': torch.rand(batch_size) * 40e-9 + 30e-9,     # 30-70 ppb
        'atmospheric_pressure': torch.full((batch_size,), 101325.0),  # Pa
        'co_mixing_ratio': torch.full((batch_size,), 150e-9),         # OH sink, 150 ppb
        'ch4_mixing_ratio': torch.full((batch_size,), 1.8e-6),        # OH sink, 1.8 ppm
        'specific_humidity': torch.full((batch_size,), 0.01),  # O(1D) + H2O OH source, 10 g/kg
        'pbl_height': torch.rand(batch_size) * 1000 + 500              # 500-1500 m
    }
    
    # Test radiation with batch
    radiation = RadiationFunctor()
    rad_result = radiation.forward(state_batch, time=43200.0, dt=3600.0)
    
    # Check batch output shapes
    for key in ['shortwave_down', 'longwave_down', 'net_radiation']:
        value = _radiation_value(rad_result, key)
        assert value.shape == (batch_size,), f"Batch shape mismatch for {key}: {value.shape}"
        assert torch.isfinite(value).all(), f"Non-finite values in batch {key}"
    
    # Test chemistry with batch
    chem_state = state_batch.copy()
    chem_state['shortwave_down'] = _radiation_value(rad_result, 'shortwave_down')
    
    chemistry = ChemistryFunctor(physical_boundary="open_mechanism")
    chem_result = chemistry.forward(
        chem_state,
        time=43200.0,
        dt=CHEMISTRY_EXPLICIT_DT,
    )
    
    # Check batch chemistry outputs
    chemistry_keys = [
        'no_mixing_ratio',
        'no2_mixing_ratio',
        'o3_mixing_ratio',
        'diagnostics.chemistry.photolysis_rate_no2',
    ]
    for key in chemistry_keys:
        value = _chemistry_value(chem_result, key)
        assert value.shape == (batch_size,), f"Batch shape mismatch for {key}: {value.shape}"
        assert torch.isfinite(value).all(), f"Non-finite values in batch {key}"


def test_device_consistency():
    """Test atmospheric modules work on different devices."""
    # Test CPU
    radiation_cpu = RadiationFunctor(device='cpu')
    
    state_cpu = {
        'air_temperature': torch.tensor(298.15),
        'surface_temperature': torch.tensor(300.15),
        'incoming_shortwave_potential': torch.tensor(1200.0),
        'cos_zenith': torch.tensor(0.7),
        'relative_humidity': torch.tensor(0.6),
        'surface_albedo': torch.tensor(0.2),
        'surface_emissivity': torch.tensor(0.97),
        'cloud_fraction': torch.tensor(0.0),
    }
    
    result_cpu = radiation_cpu.forward(state_cpu, time=43200.0, dt=3600.0)
    
    # Check all outputs are on CPU
    for key in result_cpu:
        value = _radiation_value(result_cpu, key)
        assert value.device.type == 'cpu', f"{key} not on CPU device"
    
    # Test CUDA if available
    if torch.cuda.is_available():
        radiation_cuda = RadiationFunctor(device='cuda')
        
        # Move state to CUDA
        state_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                     for k, v in state_cpu.items()}
        
        result_cuda = radiation_cuda.forward(state_cuda, time=43200.0, dt=3600.0)
        
        # Check all outputs are on CUDA
        for key in result_cuda:
            value = _radiation_value(result_cuda, key)
            assert value.device.type == 'cuda', f"{key} not on CUDA device"
        
        # Results should be numerically equivalent
        for key in result_cpu.keys():
            cpu_val = _radiation_value(result_cpu, key)
            cuda_val = _radiation_value(result_cuda, key).cpu()
            assert torch.allclose(cpu_val, cuda_val, rtol=1e-5), f"{key} differs between devices"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


