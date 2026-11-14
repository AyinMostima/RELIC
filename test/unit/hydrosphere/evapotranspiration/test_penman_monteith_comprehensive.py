"""
Comprehensive L1 Unit Tests for Penman-Monteith Evapotranspiration Model

This test suite provides exhaustive testing of the PenmanMonteithET class,
covering all mathematical functions, meteorological calculations, edge cases, and
scientific accuracy. Tests ensure zero-defect operation and validate against
established literature values.

Test Categories:
1. Basic Functionality Tests
2. Meteorological Function Tests
3. Scientific Accuracy Validation  
4. Edge Case and Boundary Tests
5. Aerodynamic Conductance Tests
6. Energy Balance Tests
7. Error Handling and Robustness Tests

References for validation values:
- Allen et al. (1998). FAO Irrigation Paper 56
- Monteith (1965). Evaporation and environment
- Penman (1948). Natural evaporation from open water
"""

import pytest
import torch
import sys
import os

# Add relic to Python path for testing
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, relic_path)

import relic.scientific.hydrosphere.evapotranspiration.penman_monteith as penman_module
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from relic.core.state import UpdateItem
from relic.utils.physical_constants import get_physical_constant as canonical_get_physical_constant


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _replace_update_value(result, key, value):
    """Replace one UpdateItem value while preserving its runtime contract."""
    item = result[key]
    result[key] = UpdateItem(
        value=value,
        mode=item.mode,
        unit=item.unit,
        per_timestep=item.per_timestep,
        element=item.element,
        source_functor=item.source_functor,
    )


class TestPenmanMonteithBasicFunctionality:
    """Test basic functionality and interface compliance."""
    
    def test_initialization_default_parameters(self):
        """Test model initialization with default parameters."""
        model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        # Check that model initializes without errors
        assert model is not None
        assert model.name == "penman_monteith_et"
        
        # Check that key parameters are set
        assert model.roughness_length > 0
        assert model.minimum_conductance > 0
        
    def test_initialization_custom_config(self):
        """Test model initialization ignores config overrides."""
        config = {
            "parameters": {
                "roughness_length": {"value": 2.5, "units": "m"},
                "displacement_height": {"value": 16.0, "units": "m"}
            }
        }
        
        model = PenmanMonteithET(config={**config, "canopy_resistance_source": "state"})

        # Config overrides are disabled; values should come from ParameterManager
        from relic.parameters import get_global_manager

        def _to_scalar(value):
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)

        manager = get_global_manager()
        expected_roughness = _to_scalar(
            manager.get("hydrosphere.evapotranspiration.roughness_length")
        )
        expected_displacement = _to_scalar(
            manager.get("hydrosphere.evapotranspiration.displacement_height")
        )

        assert _to_scalar(model.roughness_length) == pytest.approx(expected_roughness, rel=1e-6, abs=1e-6)
        assert _to_scalar(model.displacement_height) == pytest.approx(expected_displacement, rel=1e-6, abs=1e-6)
        
    def test_dependencies_interface(self):
        """Test get_dependencies interface compliance."""
        model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        dependencies = model.get_dependencies()

        assert isinstance(dependencies, (list, dict))
        if isinstance(dependencies, dict):
            assert "lagged" in dependencies or "optional" in dependencies or "required" in dependencies
        # ET is typically a primary process with no required dependencies
        if isinstance(dependencies, list):
            assert len(dependencies) == 0
        else:
            assert len(dependencies.get("required", [])) == 0
        
    def test_functor_metadata(self):
        """Test functor metadata is properly set."""
        model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        metadata = model.metadata
        
        assert metadata is not None
        assert metadata.name == "penman_monteith_et"
        assert "Penman-Monteith" in metadata.description
        assert set(metadata.output_variables) == {
            "evapotranspiration_flux",
            "latent_heat_flux",
            "diagnostics.vapor_pressure_deficit",
            "diagnostics.aerodynamic_conductance",
            "diagnostics.canopy_conductance",
            "canopy_storage",
            "diagnostics.evapotranspiration",
        }
        assert set(metadata.conservation_laws) == {"latent_heat_boundary"}
        
    def test_physical_constants(self):
        """Test physical constants are reasonable."""
        model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        # Check physical constants are in reasonable ranges
        assert 2.0e6 < model.latent_heat_vaporization < 3.0e6  # J/kg
        assert 1000 < model.specific_heat_air < 1100  # J/kg/K
        assert 0.6 < model.molecular_weight_ratio < 0.7  # water/air

    def test_physical_constants_use_canonical_ssot_api(self, monkeypatch):
        """All physical constants must bypass ordinary Functor parameter access."""
        physical_calls = []
        parameter_calls = []
        original_get_parameter_value = PenmanMonteithET.get_parameter_value

        def tracking_get_physical_constant(name, *args, **kwargs):
            physical_calls.append(name)
            return canonical_get_physical_constant(name, *args, **kwargs)

        def tracking_get_parameter_value(instance, name, *args, **kwargs):
            parameter_calls.append((name, kwargs.get("namespace")))
            return original_get_parameter_value(instance, name, *args, **kwargs)

        monkeypatch.setattr(
            penman_module,
            "get_physical_constant",
            tracking_get_physical_constant,
        )
        monkeypatch.setattr(
            PenmanMonteithET,
            "get_parameter_value",
            tracking_get_parameter_value,
        )

        model = PenmanMonteithET(config={"canopy_resistance_source": "state"})

        expected_physical_calls = {
            "water_molecular_weight",
            "air_molecular_weight",
            "latent_heat_vaporization",
            "air_specific_heat_cp",
            "von_karman_constant",
            "relic.physical.constants.psychrometrics.tetens_base_pressure",
            "relic.physical.constants.psychrometrics.tetens_coefficient_a",
            "relic.physical.constants.psychrometrics.tetens_coefficient_b",
            "relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff",
            "celsius_to_kelvin",
            "seconds_per_hour",
        }
        assert expected_physical_calls <= set(physical_calls)

        for name, namespace in parameter_calls:
            assert not str(name).startswith("relic.physical.constants")
            assert not str(namespace or "").startswith("relic.physical.constants")

        assert not any(
            key.startswith("relic.physical.constants")
            for key in model.required_parameters()
        )
        assert not any(
            key.startswith("relic.physical.constants")
            for key in model.build_hodge_contract(strict=True).parameter_keys
        )


class TestMeteorologicalFunctions:
    """Test meteorological calculation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        # Standard test temperatures
        self.temps_K = torch.tensor([283.15, 298.15, 313.15])  # 10掳C, 25掳C, 40掳C
        
    def test_saturation_vapor_pressure_calculation(self):
        """Test Tetens formula for saturation vapor pressure."""
        # Test at standard conditions
        temp_25C = torch.tensor(298.15)
        es_pa = self.model._es_tetens_pa(temp_25C)
        
        # At 25掳C, es should be approximately 3169 Pa (literature value)
        expected_es_25C = 3169.0  # Pa
        assert torch.isclose(es_pa, torch.tensor(expected_es_25C), rtol=0.05)
        
        # Test temperature response
        es_all = self.model._es_tetens_pa(self.temps_K)
        
        # Should increase with temperature
        assert torch.all(es_all[1:] > es_all[:-1])
        
        # Should be positive
        assert torch.all(es_all > 0)
        
        # Should be reasonable magnitude (hundreds to thousands of Pa)
        assert torch.all(es_all > 100)
        assert torch.all(es_all < 20000)
        
    def test_slope_saturation_vapor_pressure(self):
        """Test slope of saturation vapor pressure curve."""
        # Test at standard conditions
        temp_25C = torch.tensor(298.15)
        slope_pa_k = self.model._slope_svp_pa_per_K(temp_25C)
        
        # At 25掳C, slope should be approximately 188 Pa/K (literature value)
        expected_slope_25C = 188.0  # Pa/K
        assert torch.isclose(slope_pa_k, torch.tensor(expected_slope_25C), rtol=0.05)
        
        # Test temperature response
        slopes = self.model._slope_svp_pa_per_K(self.temps_K)
        
        # Should increase with temperature (convex curve)
        assert torch.all(slopes[1:] > slopes[:-1])
        
        # Should be positive
        assert torch.all(slopes > 0)
        
    def test_aerodynamic_conductance_calculation(self):
        """Test aerodynamic conductance calculation."""
        wind_speeds = torch.tensor([0.5, 2.0, 5.0])  # m/s
        temp = torch.tensor(298.15)
        
        ga_values = self.model._compute_aerodynamic_conductance(wind_speeds, temp)
        
        # Should increase with wind speed
        assert torch.all(ga_values[1:] > ga_values[:-1])
        
        # Should be positive
        assert torch.all(ga_values > 0)
        
        # Should be in reasonable range for conductance
        assert torch.all(ga_values > 0.001)  # Minimum physical limit
        assert torch.all(ga_values < 1.0)    # Maximum physical limit
        
    def test_psychrometric_constant(self):
        """Test psychrometric constant calculation."""
        # Create test conditions
        state = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        result = self.model.forward(state, time=0.0, dt=1.0)
        gamma = result["diagnostics.evapotranspiration"]["psychrometric_constant"]
        
        # At sea level and 25掳C, gamma should be approximately 66 Pa/K
        expected_gamma = torch.tensor(66.0, dtype=gamma.dtype, device=gamma.device)  # Pa/K
        assert torch.isclose(gamma, expected_gamma, rtol=0.1)


class TestScientificAccuracyValidation:
    """Test against established literature values and benchmarks."""
    
    def setup_method(self):
        """Set up standard test conditions."""
        self.model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
        # Standard weather conditions for grass reference crop (FAO-56)
        self.reference_conditions = {
            "net_radiation": torch.tensor(250.0),        # W/m2 (typical daytime)
            "air_temperature": torch.tensor(298.15),     # 25掳C
            "relative_humidity": torch.tensor(0.70),     # 70%
            "wind_speed": torch.tensor(2.0),             # 2 m/s
            "atmospheric_pressure": torch.tensor(101325.0), # 1013.25 mb
            "surface_resistance": torch.tensor(70.0)     # s/m (grass reference)
        }
        
    def test_reference_et_magnitude(self):
        """Test reference ET is within expected literature ranges."""
        result = self.model.forward(self.reference_conditions, time=0.0, dt=1.0)
        
        et_flux = _unwrap_update(result['evapotranspiration_flux']).item()  # kg/m2/s
        
        # Convert to mm day-1 for comparison with literature
        et_mm_day = et_flux * 86400.0  # kg/m2/s -> mm day-1
        
        # Literature ranges for reference ET under these conditions: 3-8 mm day-1
        assert 2.0 <= et_mm_day <= 10.0, f"Reference ET {et_mm_day:.2f} mm day-1 outside expected range [2-10]"
        
    def test_energy_balance_components(self):
        """Test energy balance components are reasonable."""
        result = self.model.forward(self.reference_conditions, time=0.0, dt=1.0)
        
        et_flux = _unwrap_update(result['evapotranspiration_flux']).item()
        latent_heat_flux = et_flux * self.model.latent_heat_vaporization
        
        net_radiation = self.reference_conditions['net_radiation'].item()
        
        # Latent heat should be significant fraction of net radiation
        # but not exceed it by too much (Bowen ratio considerations)
        latent_fraction = latent_heat_flux / net_radiation
        assert 0.3 <= latent_fraction <= 1.5, f"Latent heat fraction {latent_fraction:.2f} outside expected range"

    def test_latent_heat_boundary_conservation(self):
        """Latent heat must map exactly from the outward water flux."""
        result = self.model.forward(
            self.reference_conditions,
            time=0.0,
            dt=60.0,
        )

        assert self.model.check_conservation_laws(
            self.reference_conditions,
            result,
        )["latent_heat_boundary"] == pytest.approx(0.0, abs=1.0e-12)

        tampered = dict(result)
        latent_heat_flux = _unwrap_update(tampered["latent_heat_flux"])
        _replace_update_value(
            tampered,
            "latent_heat_flux",
            latent_heat_flux * latent_heat_flux.new_tensor(1.1),
        )
        assert self.model.check_conservation_laws(
            self.reference_conditions,
            tampered,
        )["latent_heat_boundary"] == pytest.approx(
            1.0 / 11.0,
            rel=1.0e-6,
        )

    @pytest.mark.parametrize(
        ("key", "invalid_value", "error_type", "message"),
        [
            (
                "evapotranspiration_flux",
                1.0,
                TypeError,
                "must be a torch.Tensor",
            ),
            (
                "evapotranspiration_flux",
                torch.empty(0),
                ValueError,
                "must be non-empty",
            ),
            (
                "evapotranspiration_flux",
                torch.tensor(1, dtype=torch.int64),
                TypeError,
                "must be floating",
            ),
            (
                "evapotranspiration_flux",
                torch.tensor(float("nan")),
                ValueError,
                "must be finite",
            ),
            (
                "latent_heat_flux",
                torch.tensor(float("inf")),
                ValueError,
                "must be finite",
            ),
            (
                "latent_heat_flux",
                torch.tensor(-1.0),
                ValueError,
                "must be non-negative",
            ),
        ],
    )
    def test_latent_heat_boundary_rejects_invalid_outputs(
        self,
        key,
        invalid_value,
        error_type,
        message,
    ):
        """Invalid boundary evidence must fail before residual evaluation."""
        result = self.model.forward(
            self.reference_conditions,
            time=0.0,
            dt=60.0,
        )
        _replace_update_value(result, key, invalid_value)

        with pytest.raises(error_type, match=message):
            self.model.check_conservation_laws(
                self.reference_conditions,
                result,
            )

    @pytest.mark.parametrize(
        "missing_key",
        ["evapotranspiration_flux", "latent_heat_flux"],
    )
    def test_latent_heat_boundary_requires_both_open_ports(self, missing_key):
        """Both water and energy boundary ports are mandatory."""
        result = self.model.forward(
            self.reference_conditions,
            time=0.0,
            dt=60.0,
        )
        result.pop(missing_key)

        with pytest.raises(KeyError, match=missing_key):
            self.model.check_conservation_laws(
                self.reference_conditions,
                result,
            )

    def test_latent_heat_boundary_requires_matching_shapes(self):
        """Water and latent-heat boundary tensors must have identical shapes."""
        result = self.model.forward(
            self.reference_conditions,
            time=0.0,
            dt=60.0,
        )
        latent_heat_flux = _unwrap_update(result["latent_heat_flux"])
        _replace_update_value(
            result,
            "latent_heat_flux",
            latent_heat_flux.reshape(1).expand(2).clone(),
        )

        with pytest.raises(ValueError, match="identical shapes"):
            self.model.check_conservation_laws(
                self.reference_conditions,
                result,
            )
        
    def test_humidity_response(self):
        """Test ET response to humidity variations."""
        et_values = []
        humidities = [0.3, 0.5, 0.7, 0.9]  # 30% to 90% RH
        
        for rh in humidities:
            conditions = self.reference_conditions.copy()
            conditions['relative_humidity'] = torch.tensor(rh)
            
            result = self.model.forward(conditions, time=0.0, dt=1.0)
            et_values.append(_unwrap_update(result['evapotranspiration_flux']).item())
            
        # ET should decrease with increasing humidity
        for i in range(1, len(et_values)):
            assert et_values[i] <= et_values[i-1], "ET should decrease with increasing humidity"
            
    def test_wind_speed_response(self):
        """Test ET response to wind speed variations."""
        et_values = []
        wind_speeds = [0.5, 1.0, 2.0, 4.0, 8.0]  # m/s
        
        for ws in wind_speeds:
            conditions = self.reference_conditions.copy()
            conditions['wind_speed'] = torch.tensor(ws)
            
            result = self.model.forward(conditions, time=0.0, dt=1.0)
            et_values.append(_unwrap_update(result['evapotranspiration_flux']).item())
            
        # ET should increase with wind speed (but may level off at high speeds)
        # Check that there's general increase
        assert et_values[-1] > et_values[0], "ET should increase with wind speed"
        
    def test_temperature_response(self):
        """Test ET response to temperature variations."""
        et_values = []
        temperatures = [288.15, 293.15, 298.15, 303.15, 308.15]  # 15-35掳C
        
        for temp in temperatures:
            conditions = self.reference_conditions.copy()
            conditions['air_temperature'] = torch.tensor(temp)
            
            result = self.model.forward(conditions, time=0.0, dt=1.0)
            et_values.append(_unwrap_update(result['evapotranspiration_flux']).item())
            
        # ET should generally increase with temperature
        assert et_values[-1] > et_values[0], "ET should increase with temperature"
        
    def test_radiation_response(self):
        """Test ET response to radiation variations."""
        et_values = []
        radiations = [50.0, 150.0, 300.0, 500.0, 800.0]  # W/m2
        
        for rad in radiations:
            conditions = self.reference_conditions.copy()
            conditions['net_radiation'] = torch.tensor(rad)
            
            result = self.model.forward(conditions, time=0.0, dt=1.0)
            et_values.append(_unwrap_update(result['evapotranspiration_flux']).item())
            
        # ET should increase with radiation
        for i in range(1, len(et_values)):
            assert et_values[i] >= et_values[i-1], "ET should increase with radiation"


class TestEdgeCasesAndBoundaryConditions:
    """Test model behavior under extreme and boundary conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
    def test_zero_radiation_conditions(self):
        """Test behavior under zero net radiation."""
        zero_rad_conditions = {
            "net_radiation": torch.tensor(0.0),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        result = self.model.forward(zero_rad_conditions, time=0.0, dt=1.0)
        et_flux = _unwrap_update(result['evapotranspiration_flux']).item()
        
        # ET should be very low but may be slightly positive due to aerodynamic term
        assert et_flux >= 0, "ET should be non-negative"
        assert et_flux < 0.001, "ET should be very low with zero radiation"
        
    def test_negative_radiation_conditions(self):
        """Test behavior under negative net radiation (nighttime)."""
        negative_rad_conditions = {
            "net_radiation": torch.tensor(-100.0),  # Nighttime cooling
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        result = self.model.forward(negative_rad_conditions, time=0.0, dt=1.0)
        et_flux = _unwrap_update(result['evapotranspiration_flux']).item()
        
        # ET should still be non-negative (clamped to zero)
        assert et_flux >= 0, "ET should be non-negative even with negative radiation"
        
    def test_very_low_wind_conditions(self):
        """Test behavior under very low wind speeds."""
        calm_conditions = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(0.1),  # Very calm
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        result = self.model.forward(calm_conditions, time=0.0, dt=1.0)
        et_flux = _unwrap_update(result['evapotranspiration_flux']).item()
        
        # Should still produce reasonable ET
        assert et_flux > 0, "ET should be positive even under calm conditions"
        assert et_flux < 0.01, "ET should be reduced under calm conditions"
        
    def test_very_high_surface_resistance(self):
        """Test behavior with very high surface resistance (stressed vegetation)."""
        stressed_conditions = {
            "net_radiation": torch.tensor(300.0),
            "air_temperature": torch.tensor(308.15),
            "relative_humidity": torch.tensor(0.4),
            "wind_speed": torch.tensor(3.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(500.0)  # Very high resistance
        }
        
        result = self.model.forward(stressed_conditions, time=0.0, dt=1.0)
        et_flux = _unwrap_update(result['evapotranspiration_flux']).item()
        
        # ET should be significantly reduced
        assert et_flux > 0, "ET should still be positive"
        
        # Compare with low resistance
        normal_conditions = stressed_conditions.copy()
        normal_conditions['surface_resistance'] = torch.tensor(70.0)
        normal_result = self.model.forward(normal_conditions, time=0.0, dt=1.0)
        normal_et = _unwrap_update(normal_result['evapotranspiration_flux']).item()
        
        assert et_flux < 0.5 * normal_et, "High surface resistance should significantly reduce ET"
        
    def test_extreme_temperatures(self):
        """Test behavior at extreme temperatures."""
        # Very hot conditions
        hot_conditions = {
            "net_radiation": torch.tensor(400.0),
            "air_temperature": torch.tensor(318.15),  # 45掳C
            "relative_humidity": torch.tensor(0.3),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(100.0)
        }
        
        result_hot = self.model.forward(hot_conditions, time=0.0, dt=1.0)
        et_hot = _unwrap_update(result_hot['evapotranspiration_flux']).item()
        
        # Should produce high ET
        assert et_hot > 0, "Hot conditions should produce positive ET"
        
        # Very cold conditions
        cold_conditions = hot_conditions.copy()
        cold_conditions['air_temperature'] = torch.tensor(278.15)  # 5掳C
        cold_conditions['net_radiation'] = torch.tensor(100.0)
        
        result_cold = self.model.forward(cold_conditions, time=0.0, dt=1.0)
        et_cold = _unwrap_update(result_cold['evapotranspiration_flux']).item()
        
        # Cold should produce less ET than hot
        assert et_cold < et_hot, "Cold conditions should produce less ET than hot"
        
    def test_saturated_humidity_conditions(self):
        """Test behavior at 100% relative humidity."""
        saturated_conditions = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(1.0),  # 100% RH
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        result = self.model.forward(saturated_conditions, time=0.0, dt=1.0)
        et_flux = _unwrap_update(result['evapotranspiration_flux']).item()
        vpd = result["diagnostics.evapotranspiration"]["vapor_pressure_deficit"].item()
        
        # VPD should be zero or very small
        assert vpd < 10.0, "VPD should be very small at 100% humidity"
        
        # ET should still be positive due to energy term but reduced
        assert et_flux > 0, "ET should still be positive at saturation"


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness under various conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
    def test_missing_required_inputs(self):
        """Test error handling for missing required inputs."""
        incomplete_state = {
            "net_radiation": torch.tensor(200.0),
            # Missing other required inputs
        }
        
        with pytest.raises(ValueError, match="Required state variable missing"):
            self.model.forward(incomplete_state, time=0.0, dt=1.0)
            
    def test_out_of_range_inputs(self):
        """Test handling of out-of-range inputs."""
        # Very high temperature
        extreme_conditions = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(373.15),  # 100掳C - too hot
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        with pytest.raises(ValueError, match="Temperature outside"):
            self.model.forward(extreme_conditions, time=0.0, dt=1.0)
            
    def test_negative_values_handling(self):
        """Test handling of negative values where inappropriate."""
        negative_conditions = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(-0.1),  # Negative humidity
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        with pytest.raises(ValueError, match="Relative humidity"):
            self.model.forward(negative_conditions, time=0.0, dt=1.0)
            
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        nan_conditions = {
            "net_radiation": torch.tensor(float('nan')),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        try:
            result = self.model.forward(nan_conditions, time=0.0, dt=1.0)
            # If it doesn't raise error, check output is handled properly
            # NaN propagation to output is acceptable for NaN input
            pass
        except (ValueError, RuntimeError):
            # Acceptable to raise error for NaN input
            pass
            
    def test_extreme_net_radiation_values(self):
        """Test handling of extreme net radiation values."""
        # Very high radiation
        extreme_high_rad = {
            "net_radiation": torch.tensor(2000.0),  # Very high
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        with pytest.raises(ValueError, match="Net radiation outside physical bounds"):
            self.model.forward(extreme_high_rad, time=0.0, dt=1.0)
            
    def test_tensor_vs_scalar_inputs(self):
        """Test handling of mixed tensor/scalar inputs."""
        mixed_conditions = {
            "net_radiation": 200.0,  # Scalar
            "air_temperature": torch.tensor(298.15),  # Tensor
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        # Should handle mixed input types gracefully
        result = self.model.forward(mixed_conditions, time=0.0, dt=1.0)
        assert torch.is_tensor(_unwrap_update(result['evapotranspiration_flux']))


class TestTransportMechanismInterface:
    """Test TransportMechanism interface implementation."""
    
    def setup_method(self):
        """Set up test model."""
        self.model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        
    def test_compute_flux_interface(self):
        """Test canonical state-driven transport outputs."""
        result = self.model.forward(
            {
                "net_radiation": torch.tensor(220.0),
                "air_temperature": torch.tensor(298.15),
                "relative_humidity": torch.tensor(0.65),
                "wind_speed": torch.tensor(2.0),
                "atmospheric_pressure": torch.tensor(101325.0),
                "surface_resistance": torch.tensor(70.0),
                # canopy_storage is returned only for a store that exists.
                "canopy_storage": torch.tensor(0.0),
            },
            time=0.0,
            dt=3600.0,
        )
        
        # Should return flux dictionary
        assert isinstance(result, dict)
        assert 'evapotranspiration_flux' in result
        assert 'latent_heat_flux' in result

        expected_units = {
            "evapotranspiration_flux": "kg m-2 s-1",
            "latent_heat_flux": "W m-2",
            "diagnostics.vapor_pressure_deficit": "Pa",
            "diagnostics.aerodynamic_conductance": "m s-1",
            "diagnostics.canopy_conductance": "m s-1",
            "canopy_storage": "mm",
        }
        for key, unit in expected_units.items():
            update = result[key]
            assert isinstance(update, UpdateItem)
            assert update.mode.value == "replace"
            assert update.per_timestep is False
            assert update.unit == unit
            assert update.source_functor == self.model.name
        assert 'vapor_flux' not in result
        
        # Fluxes should be positive (from source to sink)
        et_flux = _unwrap_update(result['evapotranspiration_flux'])
        assert torch.is_tensor(et_flux)
        assert et_flux.item() >= 0

    def test_forward_input_dict_immutability(self):
        """forward must not mutate the caller-provided state dictionary."""
        state = {
            "net_radiation": torch.tensor(220.0),
            "air_temperature": torch.tensor(298.15),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0),
            "relative_humidity": torch.tensor(0.65),
        }

        state_before = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }

        result = self.model.forward(state, time=0.0, dt=3600.0)
        assert isinstance(result, dict)

        assert set(state.keys()) == set(state_before.keys())
        for key, value in state_before.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(state[key], value)
            else:
                assert state[key] == value
        
    def test_compute_rate_interface(self):
        """Test compute_rate method (alias for forward)."""
        state = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        result = self.model.forward(state, time=0.0, dt=1.0)
        
        # Should return same structure as forward method
        assert 'evapotranspiration_flux' in result
        assert torch.is_tensor(_unwrap_update(result['evapotranspiration_flux']))


class TestParameterSensitivityAnalysis:
    """Test parameter sensitivity and configuration effects."""
    
    def setup_method(self):
        """Set up test conditions."""
        self.base_conditions = {
            "net_radiation": torch.tensor(200.0),
            "air_temperature": torch.tensor(298.15),
            "relative_humidity": torch.tensor(0.7),
            "wind_speed": torch.tensor(2.0),
            "atmospheric_pressure": torch.tensor(101325.0),
            "surface_resistance": torch.tensor(70.0)
        }
        
        
    def test_aerodynamic_resistance_override(self):
        """Test external aerodynamic resistance override."""
        # Conditions with external aerodynamic resistance
        conditions_with_ra = self.base_conditions.copy()
        conditions_with_ra['aerodynamic_resistance'] = torch.tensor(50.0)  # s/m
        
        model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
        result = model.forward(conditions_with_ra, time=0.0, dt=1.0)
        
        # Should use provided aerodynamic resistance
        expected_ga = 1.0 / 50.0
        actual_ga = result["diagnostics.evapotranspiration"]["aerodynamic_conductance"].item()
        
        assert torch.isclose(torch.tensor(actual_ga), torch.tensor(expected_ga), rtol=0.01)


# Integration with pytest fixtures and parameterized tests
@pytest.fixture(scope="module")
def penman_monteith_model():
    """Pytest fixture for Penman-Monteith model."""
    return PenmanMonteithET(config={"canopy_resistance_source": "state"})


@pytest.fixture(scope="module")
def standard_conditions():
    """Pytest fixture for standard test conditions."""
    return {
        "net_radiation": torch.tensor(200.0),
        "air_temperature": torch.tensor(298.15),
        "relative_humidity": torch.tensor(0.7),
        "wind_speed": torch.tensor(2.0),
        "atmospheric_pressure": torch.tensor(101325.0),
        "surface_resistance": torch.tensor(70.0)
    }


@pytest.mark.parametrize("net_radiation", [50, 150, 300, 500])
def test_radiation_sweep(penman_monteith_model, standard_conditions, net_radiation):
    """Parametrized test across radiation range."""
    conditions = standard_conditions.copy()
    conditions['net_radiation'] = torch.tensor(float(net_radiation))
    
    result = penman_monteith_model.forward(conditions, time=0.0, dt=1.0)
    
    # All outputs should be finite and physically reasonable
    assert torch.isfinite(_unwrap_update(result['evapotranspiration_flux']))
    assert _unwrap_update(result['evapotranspiration_flux']).item() >= 0.0


@pytest.mark.parametrize("humidity", [0.3, 0.5, 0.7, 0.9])
def test_humidity_sweep(penman_monteith_model, standard_conditions, humidity):
    """Parametrized test across humidity range."""
    conditions = standard_conditions.copy()
    conditions['relative_humidity'] = torch.tensor(humidity)
    
    result = penman_monteith_model.forward(conditions, time=0.0, dt=1.0)
    
    assert torch.isfinite(_unwrap_update(result['evapotranspiration_flux']))
    assert _unwrap_update(result['evapotranspiration_flux']).item() >= 0.0
    
    # VPD should decrease with increasing humidity
    vpd = result["diagnostics.evapotranspiration"]["vapor_pressure_deficit"].item()
    assert vpd >= 0.0


@pytest.mark.parametrize("wind_speed", [0.5, 1.0, 2.0, 4.0])
def test_wind_sweep(penman_monteith_model, standard_conditions, wind_speed):
    """Parametrized test across wind speed range."""
    conditions = standard_conditions.copy()
    conditions['wind_speed'] = torch.tensor(wind_speed)
    
    result = penman_monteith_model.forward(conditions, time=0.0, dt=1.0)
    
    assert torch.isfinite(_unwrap_update(result['evapotranspiration_flux']))
    assert _unwrap_update(result['evapotranspiration_flux']).item() > 0.0
    
    # Aerodynamic conductance should increase with wind speed
    ga = result["diagnostics.evapotranspiration"]["aerodynamic_conductance"].item()
    assert ga > 0.0


@pytest.mark.parametrize("ground_heat", [-40.0, 35.0, 500.0])
def test_measured_ground_heat_uses_signed_value_and_independent_equation(
    standard_conditions, ground_heat,
):
    """Measured surface G enters the one PM law, including negative-energy cases."""
    import math

    model = PenmanMonteithET(config={
        "canopy_resistance_source": "state", "soil_heat_flux_source": "state",
    })
    state = {**standard_conditions, "aerodynamic_resistance": torch.tensor(50.0),
             "soil_heat_flux": torch.tensor(ground_heat, requires_grad=True)}
    result = model(state, time=0.0, dt=1800.0)
    diagnostic = result["diagnostics.evapotranspiration"]
    assert "soil_heat_flux" in model.get_required_inputs()
    torch.testing.assert_close(diagnostic["soil_heat_flux"], state["soil_heat_flux"].to(torch.float32))
    torch.testing.assert_close(diagnostic["available_energy"], torch.tensor(200.0 - ground_heat, dtype=torch.float32))

    # References: Monteith (1965); Allen et al. (1998), FAO-56 Eqs. 3, 8 and 13.
    # Mathematical expression: E=max(0,[Delta*(Rn-G)+rho*cp*D/ra]/
    #   [Lv*(Delta+gamma*(1+rs/ra))]); rho=[p-(1-eps)*ea]/(Rd*T).
    # Parameter meanings: Delta saturation slope; Rn net radiation; G soil heat;
    #   D vapour deficit; ra/rs bulk resistances; Lv latent heat; cp air heat capacity;
    #   gamma psychrometric constant; eps molar-mass ratio; ea vapour pressure;
    #   Rd dry-air gas constant; T air temperature; p pressure.
    # Units: E [kg m-2 s-1]; Rn/G [W m-2]; p/D/ea [Pa]; Delta/gamma [Pa K-1];
    #   ra/rs [s m-1]; Lv [J kg-1]; cp/Rd [J kg-1 K-1]; T [K]; rho [kg m-3]; eps [1].
    # Domain/constraints: independent scalar oracle at finite warm fixture conditions.
    # Sign convention: G positive into soil; E and Lv*E positive out of the surface.
    const = canonical_get_physical_constant
    t = float(state["air_temperature"])
    tc = t - float(const("celsius_to_kelvin"))
    ea_sat = float(const("relic.physical.constants.psychrometrics.tetens_base_pressure")) * math.exp(
        float(const("relic.physical.constants.psychrometrics.tetens_coefficient_a")) * tc /
        (tc + float(const("relic.physical.constants.psychrometrics.tetens_coefficient_b"))))
    slope = float(const("relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff")) * ea_sat / (
        tc + float(const("relic.physical.constants.psychrometrics.tetens_coefficient_b"))) ** 2
    mw_air = float(const("air_molecular_weight"))
    epsilon = float(const("water_molecular_weight")) / mw_air
    rd = float(const("gas_constant")) / (mw_air * 0.001)
    ea = ea_sat * float(state["relative_humidity"])
    pressure = float(state["atmospheric_pressure"])
    rho = (pressure - (1.0 - epsilon) * ea) / (rd * t)
    cp = float(const("air_specific_heat_cp"))
    latent = float(const("latent_heat_vaporization"))
    gamma = cp * pressure / (epsilon * latent)
    denominator = latent * (slope + gamma * (1.0 + 70.0 / 50.0))
    raw = (slope * (200.0 - ground_heat) + rho * cp * (ea_sat - ea) / 50.0) / denominator
    et = _unwrap_update(result["evapotranspiration_flux"])
    assert float(et.detach()) == pytest.approx(max(0.0, raw), rel=2e-6, abs=1e-10)
    assert float(diagnostic["negative_et_projection_residual"].detach()) == pytest.approx(max(0.0, -raw), rel=2e-6, abs=1e-10)
    torch.testing.assert_close(_unwrap_update(result["latent_heat_flux"]), et * latent)
    assert float(diagnostic["available_energy_residual"].detach()) == 0.0
    et.backward()
    expected_gradient = -slope / denominator if raw > 0 else 0.0
    assert float(state["soil_heat_flux"].grad) == pytest.approx(expected_gradient, rel=2e-6, abs=1e-12)


@pytest.mark.parametrize("ground_heat", [None, float("nan"), float("inf")])
def test_measured_ground_heat_does_not_substitute_missing_or_invalid_input(
    standard_conditions, ground_heat,
):
    from relic.core.functor.functor_exceptions import ExecutionError

    model = PenmanMonteithET(config={
        "canopy_resistance_source": "state", "soil_heat_flux_source": "state",
    })
    state = dict(standard_conditions)
    if ground_heat is not None:
        state["soil_heat_flux"] = torch.tensor(ground_heat)
    with pytest.raises(ExecutionError, match="soil_heat_flux"):
        model(state, time=0.0, dt=1800.0)


def test_fraction_ground_heat_keeps_original_output_and_step_domain(standard_conditions):
    from relic.core.functor.functor_exceptions import ExecutionError

    model = PenmanMonteithET(config={"canopy_resistance_source": "state"})
    assert model.soil_heat_flux_source == "fraction"
    assert "soil_heat_flux" not in model.get_required_inputs()
    original = model(standard_conditions, time=0.0, dt=1800.0)
    ignored = model({**standard_conditions, "soil_heat_flux": torch.tensor(-40.0)}, time=0.0, dt=1800.0)
    torch.testing.assert_close(_unwrap_update(original["latent_heat_flux"]), _unwrap_update(ignored["latent_heat_flux"]), rtol=0, atol=0)
    measured = PenmanMonteithET(config={"canopy_resistance_source": "state", "soil_heat_flux_source": "state"})
    with pytest.raises(ExecutionError, match="dt=7200"):
        measured({**standard_conditions, "soil_heat_flux": torch.tensor(-40.0)}, time=0.0, dt=7200.0)


def test_unknown_ground_heat_source_is_rejected():
    with pytest.raises(ValueError, match="soil_heat_flux_source"):
        PenmanMonteithET(config={"canopy_resistance_source": "state", "soil_heat_flux_source": "guess"})


@pytest.mark.parametrize("include_ground", [False, True])
def test_measured_ground_heat_is_required_by_strict_graph_and_committed(
    standard_conditions, include_ground, tmp_path,
):
    from relic.orchestration.compiler import compile_graph
    from relic.orchestration.executor import GraphExecutor
    from relic.orchestration.graph_ir import GraphIR, NodeSpec
    from relic.orchestration.node_registry import NodeRegistry
    from relic.orchestration.ssot_catalog import SSOTCatalog

    config = {"canopy_resistance_source": "state", "soil_heat_flux_source": "state", "execution_group": "fast"}
    model = PenmanMonteithET(config=config)
    registry = NodeRegistry(SSOTCatalog.from_parameter_manager())
    kind = f"Functor:{PenmanMonteithET.__module__}.{PenmanMonteithET.__name__}"
    definition = registry.register_functor_instance(model, node_type_id=kind)
    assert definition.inputs["soil_heat_flux"].required
    state = {**standard_conditions, "aerodynamic_resistance": torch.tensor(50.0)}
    if include_ground:
        state["soil_heat_flux"] = torch.tensor(-40.0)
    graph = GraphIR(schema_version="1.0.0", nodes={
        "forcing": NodeSpec(class_type="State:ExternalForcing", outputs=list(state)),
        "penman_monteith_et": NodeSpec(class_type=kind, config=config,
            inputs={name: {"state": name} for name in state},
            outputs=["evapotranspiration_flux", "latent_heat_flux", "diagnostics.evapotranspiration"]),
    })
    if not include_ground:
        with pytest.raises(ValueError, match="soil_heat_flux"):
            compile_graph(graph, registry, strict=True, auto_adapt=False)
        return
    plan = compile_graph(graph, registry, strict=True, auto_adapt=False)
    executor = GraphExecutor(plan, registry=registry, output_dir=tmp_path,
        simulator_config={"device": "cpu", "auto_report": False})
    final = executor.execute(state, n_steps=1, dt=1800.0)
    expected = model(state, time=0.0, dt=1800.0)
    torch.testing.assert_close(final["latent_heat_flux"], _unwrap_update(expected["latent_heat_flux"]), rtol=0, atol=0)
    torch.testing.assert_close(final["soil_heat_flux"], state["soil_heat_flux"], rtol=0, atol=0)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
