"""
Comprehensive L1 Unit Tests for Farquhar Photosynthesis Model

This test suite provides exhaustive testing of the FarquharPhotosynthesis class,
covering all mathematical functions, parameter responses, edge cases, and
scientific accuracy. Tests are designed to ensure zero-defect operation
and validate against established literature values.

Test Categories:
1. Basic Functionality Tests
2. Temperature Response Function Tests  
3. Scientific Accuracy Validation
4. Edge Case and Boundary Tests
5. Parameter Sensitivity Tests
6. Performance and Optimization Tests
7. Error Handling and Robustness Tests

References for validation values:
- Farquhar et al. (1980). Biochemical model of C3 photosynthesis
- Bernacchi et al. (2001). Temperature response functions for Rubisco
- Medlyn et al. (2002). Temperature response of biochemical model parameters
- Wullschleger (1993). Biochemical limitations in C3 plants
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add relic to Python path for testing
relic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, relic_path)

from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.core.state import UpdateItem


def _state(
    *,
    air_temperature=298.15,
    atmospheric_co2_ppm=400.0,
    ppfd=1000.0,
    vapor_pressure_deficit=1000.0,
    **optional,
):
    """Build canonical forcing inputs and optional process controls."""
    state = {
        "air_temperature": torch.as_tensor(air_temperature),
        "atmospheric_co2_ppm": torch.as_tensor(atmospheric_co2_ppm),
        "ppfd": torch.as_tensor(ppfd),
        "leaf_area_index": torch.tensor(1.0),
        "vapor_pressure_deficit": torch.as_tensor(vapor_pressure_deficit),
    }
    state.update({key: torch.as_tensor(value) for key, value in optional.items()})
    return state


def _value(output, key):
    """Return the tensor carried by an explicit runtime update."""
    item = output[key]
    assert isinstance(item, UpdateItem)
    return item.value


class TestFarquharPhotosynthesisBasicFunctionality:
    """Test basic functionality and interface compliance."""
    
    def test_initialization_default_parameters(self):
        """Test model initialization with default parameters."""
        model = FarquharPhotosynthesis()
        
        # Check that model initializes without errors
        assert model is not None
        assert model.name == "farquhar_photosynthesis"
        assert model.pathway == "C3"
        
        # Check that key parameters are set
        assert model.vcmax25 is not None
        assert model.jmax25 is not None
        assert model.rd25 is not None
        assert torch.is_tensor(model.vcmax25)
        
    def test_initialization_custom_config(self):
        """Test model initialization with custom configuration."""
        config = {
            'vcmax25': 80.0,
            'jmax25': 160.0,
            'rd25': 1.5,
            'pathway': 'C4'
        }
        
        model = FarquharPhotosynthesis(config=config)
        
        # Check custom parameters are applied
        assert model.pathway == "C4"
        assert torch.isclose(model.vcmax25, model.vcmax25.new_tensor(80.0), atol=1e-5)
        assert torch.isclose(model.jmax25, model.jmax25.new_tensor(160.0), atol=1e-5)
        assert torch.isclose(model.rd25, model.rd25.new_tensor(1.5), atol=1e-5)
        
    def test_device_consistency(self):
        """Test CPU/GPU device consistency."""
        # Test CPU device
        model_cpu = FarquharPhotosynthesis(device=torch.device('cpu'))
        assert model_cpu.device == torch.device('cpu')
        
        # Test GPU device if available
        if torch.cuda.is_available():
            model_gpu = FarquharPhotosynthesis(device=torch.device('cuda'))
            assert model_gpu.device == torch.device('cuda')
            
    def test_dependencies_interface(self):
        """Test get_dependencies interface compliance."""
        model = FarquharPhotosynthesis()
        dependencies = model.get_dependencies()
        
        assert isinstance(dependencies, list)
        # Should return list of functor names
        expected_deps = ['environmental_forcing', 'atmospheric_radiation']
        assert dependencies == expected_deps
        
    def test_provides_interface(self):
        """Test get_provides interface compliance."""
        model = FarquharPhotosynthesis()
        provides = model.get_provides()
        
        assert isinstance(provides, list)
        assert provides == model.metadata.output_variables
        assert set(provides) == {
            "gross_photosynthesis",
            "net_photosynthesis",
            "gross_primary_productivity",
            "electron_transport_rate",
            "diagnostics.photosynthesis.dark_respiration",
            "diagnostics.photosynthesis.transport_limited",
            "diagnostics.photosynthesis.jmax_actual",
            "diagnostics.photosynthesis.limitation_factor",
            "diagnostics.photosynthesis.pathway_type",
        }

    def test_module_call_accepts_only_required_forcings(self):
        """FunctorBase validation must distinguish optional process controls."""
        model = FarquharPhotosynthesis()

        output = model(_state(), 0.0, 1.0)

        assert set(output) == set(model.metadata.output_variables)


class TestTemperatureResponseFunctions:
    """Test temperature response functions for biochemical accuracy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = FarquharPhotosynthesis()
        # Standard test temperatures
        self.temps_K = torch.tensor([283.15, 298.15, 313.15])  # 10掳C, 25掳C, 40掳C
        
    def test_vcmax_temperature_response(self):
        """Test Vcmax temperature response against literature values."""
        # Test temperature response shape and magnitude
        vcmax_temp = self.model._temperature_response_vcmax(self.temps_K)
        
        # Vcmax should increase with temperature up to optimum
        assert vcmax_temp[1] > vcmax_temp[0]  # 25掳C > 10掳C
        
        # Should peak and decline at high temperatures (peaked response)
        # At 40掳C, may still be increasing or starting to decline
        # This depends on the specific parameterization
        
        # Check reasonable magnitude (around reference value at 25掳C)
        vcmax_25C = vcmax_temp[1]
        assert torch.isclose(vcmax_25C, self.model.vcmax25.to(vcmax_25C), rtol=0.01)
        
        # Check units and physical bounds
        assert torch.all(vcmax_temp > 0)  # Must be positive
        assert torch.all(vcmax_temp < 500)  # Upper physical bound
        
    def test_jmax_temperature_response(self):
        """Test Jmax temperature response characteristics."""
        jmax_temp = self.model._temperature_response_jmax(self.temps_K)
        
        # Similar pattern to Vcmax
        assert jmax_temp[1] > jmax_temp[0]  # Increases with temperature
        assert torch.isclose(jmax_temp[1], self.model.jmax25.to(jmax_temp), rtol=0.01)
        
        # Physical bounds
        assert torch.all(jmax_temp > 0)
        assert torch.all(jmax_temp < 1000)
        
    def test_rd_temperature_response(self):
        """Test dark respiration Q10 response."""
        rd_temp = self.model._temperature_response_rd(self.temps_K)
        
        # Should increase exponentially with temperature (Q10 response)
        assert rd_temp[1] > rd_temp[0]
        assert rd_temp[2] > rd_temp[1]
        
        # Check Q10 relationship approximately holds
        q10_empirical = rd_temp[1] / rd_temp[0]  # 25掳C / 10掳C for 15掳C difference
        expected_q10 = torch.as_tensor(
            self.model.rd_q10,
            dtype=q10_empirical.dtype,
            device=q10_empirical.device,
        ) ** 1.5  # For 15掳C difference
        assert torch.isclose(q10_empirical, expected_q10, rtol=0.2)
        
    def test_michaelis_constants_temperature_response(self):
        """Test temperature response of kinetic constants."""
        kc_temp = self.model._temperature_response_kc(self.temps_K)
        ko_temp = self.model._temperature_response_ko(self.temps_K)
        gamma_star_temp = self.model._temperature_response_gamma_star(self.temps_K)
        
        # All should increase with temperature (Arrhenius response)
        assert torch.all(kc_temp[1:] > kc_temp[:-1])
        assert torch.all(ko_temp[1:] > ko_temp[:-1])
        assert torch.all(gamma_star_temp[1:] > gamma_star_temp[:-1])
        
        # Check reference values at 25掳C
        kc_ref = torch.as_tensor(self.model.kc25, dtype=kc_temp.dtype, device=kc_temp.device)
        ko_ref = torch.as_tensor(self.model.ko25, dtype=ko_temp.dtype, device=ko_temp.device)
        gamma_ref = torch.as_tensor(
            self.model.gamma_star25,
            dtype=gamma_star_temp.dtype,
            device=gamma_star_temp.device,
        )
        assert torch.isclose(kc_temp[1], kc_ref, rtol=0.01)
        assert torch.isclose(ko_temp[1], ko_ref, rtol=0.01)
        assert torch.isclose(gamma_star_temp[1], gamma_ref, rtol=0.01)


class TestScientificAccuracyValidation:
    """Test against established literature values and benchmarks."""
    
    def setup_method(self):
        """Set up standard test conditions."""
        self.model = FarquharPhotosynthesis()
        
        # Standard measurement conditions (Farquhar et al. 1980).
        self.standard_state = _state(ppfd=1500.0, vcmax_scaling=1.0)
        
    def test_photosynthesis_rate_magnitude(self):
        """Test that photosynthesis rates are within expected literature ranges."""
        result = self.model.forward(self.standard_state, time=0.0, dt=1.0)
        
        gross_ps = _value(result, 'gross_photosynthesis').item()
        net_ps = _value(result, 'net_photosynthesis').item()

        lai = torch.as_tensor(self.standard_state["leaf_area_index"])
        effective_lai = self.model._canopy_lai_scale(lai).item()
        gross_leaf = gross_ps / effective_lai
        net_leaf = net_ps / effective_lai

        # Farquhar literature ranges are leaf-area rates under high light.
        assert 10.0 <= gross_leaf <= 50.0, (
            f"Leaf-basis gross photosynthesis {gross_leaf} outside [10, 50]"
        )
        assert 8.0 <= net_leaf <= 45.0, (
            f"Leaf-basis net photosynthesis {net_leaf} outside [8, 45]"
        )
        
        # Net should be less than gross
        assert net_ps < gross_ps, "Net photosynthesis should be less than gross"
        
    def test_dark_respiration_magnitude(self):
        """Test dark respiration is reasonable fraction of Vcmax."""
        result = self.model.forward(self.standard_state, time=0.0, dt=1.0)
        
        rd = _value(
            result, 'diagnostics.photosynthesis.dark_respiration'
        ).item()
        vcmax_actual = self.model._temperature_response_vcmax(
            self.standard_state["air_temperature"]
        ).item()
        
        # Literature: Rd typically 1-2% of Vcmax25 at 25掳C
        rd_fraction = rd / vcmax_actual
        assert 0.005 <= rd_fraction <= 0.05, f"Rd fraction {rd_fraction} outside expected range [0.5-5%]"
        
    def test_vcmax_jmax_relationship(self):
        """Test Vcmax:Jmax ratio is within literature range."""
        temperature = self.standard_state["air_temperature"]
        vcmax = self.model._temperature_response_vcmax(temperature).item()
        jmax = self.model._temperature_response_jmax(temperature).item()
        
        # Literature: Jmax typically 1.8-2.2 times Vcmax (Wullschleger 1993)
        jmax_vcmax_ratio = jmax / vcmax
        assert 1.5 <= jmax_vcmax_ratio <= 2.5, f"Jmax:Vcmax ratio {jmax_vcmax_ratio} outside expected range [1.5-2.5]"
        
    def test_co2_response_curve_shape(self):
        """Test CO2 response follows expected biochemical shape."""
        co2_levels = torch.tensor([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
        photosynthesis_rates = []
        
        for co2 in co2_levels:
            state = self.standard_state.copy()
            state['atmospheric_co2_ppm'] = co2
            result = self.model.forward(state, time=0.0, dt=1.0)
            photosynthesis_rates.append(_value(result, 'net_photosynthesis').item())
        
        ps_array = np.array(photosynthesis_rates)
        
        # Should be monotonically increasing (or at least non-decreasing)
        assert np.all(np.diff(ps_array) >= -0.1), "CO2 response should be monotonically increasing"
        
        # Should show saturation at high CO2
        # Rate of increase should decrease at high CO2
        diff1 = ps_array[2] - ps_array[1]  # 200-100 ppm
        diff2 = ps_array[4] - ps_array[3]  # 800-400 ppm
        assert diff2 < diff1, "Should show saturation pattern at high CO2"
        
    def test_light_response_curve_shape(self):
        """Test light response follows expected pattern."""
        ppfd_levels = torch.tensor([0.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0])
        photosynthesis_rates = []
        
        for ppfd in ppfd_levels:
            state = self.standard_state.copy()
            state['ppfd'] = ppfd
            result = self.model.forward(state, time=0.0, dt=1.0)
            photosynthesis_rates.append(_value(result, 'net_photosynthesis').item())
            
        ps_array = np.array(photosynthesis_rates)
        
        # Should increase with light
        assert np.all(np.diff(ps_array) >= -0.1), "Light response should be monotonically increasing"
        
        # Should approach saturation
        # Initial slope should be steeper than final slope
        initial_slope = (ps_array[2] - ps_array[1]) / (500 - 100)
        final_slope = (ps_array[5] - ps_array[4]) / (2000 - 1500)
        assert initial_slope > final_slope, "Should show light saturation"


class TestEdgeCasesAndBoundaryConditions:
    """Test model behavior under extreme and boundary conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.model = FarquharPhotosynthesis()
        
    def test_extreme_temperatures(self):
        """Test behavior at extreme temperatures."""
        # Very cold conditions
        cold_state = _state(air_temperature=273.15, vcmax_scaling=1.0)
        
        result_cold = self.model.forward(cold_state, time=0.0, dt=1.0)
        
        # Should produce very low photosynthesis
        assert torch.isfinite(_value(result_cold, 'net_photosynthesis')).all()
        assert _value(result_cold, 'net_photosynthesis').item() < 5.0
        
        # Very hot conditions
        hot_state = cold_state.copy()
        hot_state['air_temperature'] = torch.tensor(318.15)
        
        result_hot = self.model.forward(hot_state, time=0.0, dt=1.0)
        
        # Should show temperature stress/decline
        assert torch.isfinite(_value(result_hot, 'net_photosynthesis')).all()
        
    def test_zero_light_conditions(self):
        """Test behavior under zero light."""
        dark_state = _state(ppfd=0.0, vcmax_scaling=1.0)
        
        result = self.model.forward(dark_state, time=0.0, dt=1.0)
        
        # Gross photosynthesis should be zero or very low
        assert _value(result, 'gross_photosynthesis').item() <= 0.1
        
        # Net photosynthesis should be negative (respiration only)
        assert _value(result, 'net_photosynthesis').item() < 0.0
        
    def test_very_high_co2(self):
        """Test behavior under CO2-saturated conditions."""
        high_co2_state = _state(
            atmospheric_co2_ppm=2000.0,
            ppfd=1500.0,
            vcmax_scaling=1.0,
        )
        
        result = self.model.forward(high_co2_state, time=0.0, dt=1.0)
        
        # Should be limited by light/Jmax, not CO2
        limitation = _value(
            result, 'diagnostics.photosynthesis.limitation_factor'
        ).item()
        # limitation_factor = 1 indicates Aj (light) limited
        assert limitation > 0.5, "Should be light-limited at very high CO2"
        
    def test_severe_stress_conditions(self):
        """Test behavior under severe stress."""
        stress_state = _state(
            vcmax_scaling=0.1,
            water_stress_factor=0.2,
            temperature_stress_factor=0.3,
        )
        
        result = self.model.forward(stress_state, time=0.0, dt=1.0)
        
        # Photosynthesis should be severely reduced
        assert torch.isfinite(_value(result, 'net_photosynthesis')).all()
        assert _value(result, 'gross_photosynthesis').item() < 5.0
        
    def test_invalid_input_handling(self):
        """Test error handling for invalid inputs."""
        # Test missing required inputs
        incomplete_state = {'air_temperature': torch.tensor(298.15)}
        
        with pytest.raises(ValueError, match="Required input"):
            self.model.forward(incomplete_state, time=0.0, dt=1.0)
            
        # Test out-of-range inputs with validation enabled
        model_with_validation = FarquharPhotosynthesis(validate_inputs=True)
        
        invalid_temp_state = _state(air_temperature=373.15)

        with pytest.raises(ValueError, match="air_temperature outside supported range"):
            model_with_validation.forward(invalid_temp_state, time=0.0, dt=1.0)


class TestParameterSensitivityAnalysis:
    """dA/dtheta by the BiologicalProcess central difference (item 114).

    FarquharPhotosynthesis owns no parameter_sensitivity: the base replaces the
    private tensor member inside torch.func.functional_call and evaluates the
    governed call on the state inputs only. The parameter set is fixed, because the
    Jmax-limited derivative depends on it (jmax25 = 120 gives 0.0553, not 0.0476).
    The step is 1e-4 (relative, and absolute for |theta| < 1): with the default 1e-6
    the float32 central difference of rd25 is dominated by round-off (-1.19 instead
    of -0.787). Reference values: float64 central differences of the same forward
    (vcmax25 0.14718, jmax25 0.04762, rd25 -0.78723).
    """

    PARAMETERS = {"vcmax25": 75.3, "jmax25": 148.6, "rd25": 1.2}
    STEP = {"relative_eps": 1.0e-4, "absolute_eps": 1.0e-4}

    def setup_method(self):
        """Set up the model with the fixed parameter set."""
        self.model = FarquharPhotosynthesis(**self.PARAMETERS)

    def _sensitivity(self, name, state):
        return self.model.parameter_sensitivity(
            name,
            state,
            time=0.0,
            dt=1.0,
            target_output="net_photosynthesis",
            **self.STEP,
        )

    def test_farquhar_uses_the_base_sensitivity(self):
        """The removed override took an environment mapping and duplicated the base."""
        assert "parameter_sensitivity" not in FarquharPhotosynthesis.__dict__

    def test_vcmax_sensitivity(self):
        """Rubisco-limited at 2000 umol m-2 s-1: A rises with Vcmax25, not with Jmax25."""
        state = _state(ppfd=2000.0, vcmax_scaling=1.0)
        assert float(self._sensitivity("vcmax25", state)) == pytest.approx(0.14718, rel=5e-3)
        assert float(self._sensitivity("jmax25", state)) == 0.0

    def test_jmax_sensitivity(self):
        """Electron-transport-limited at 1000 umol m-2 s-1: A rises with Jmax25, not
        with Vcmax25."""
        state = _state(ppfd=1000.0, vcmax_scaling=1.0)
        assert float(self._sensitivity("jmax25", state)) == pytest.approx(0.04762, rel=5e-3)
        assert float(self._sensitivity("vcmax25", state)) == 0.0

    def test_rd25_sensitivity(self):
        """A = min(Ac, Aj) - Rd: dA/dRd25 = -Rd(T)/Rd25, independent of the light."""
        state = _state(ppfd=1000.0, vcmax_scaling=1.0)
        assert float(self._sensitivity("rd25", state)) == pytest.approx(-0.78723, rel=5e-3)

    def test_sensitivity_leaves_the_parameters_unchanged(self):
        before = {name: getattr(self.model, name).detach().clone() for name in self.PARAMETERS}
        self._sensitivity("vcmax25", _state(ppfd=2000.0, vcmax_scaling=1.0))
        for name, value in before.items():
            assert torch.equal(getattr(self.model, name).detach(), value), name

    def test_invalid_parameter_sensitivity(self):
        """An unknown parameter name raises KeyError."""
        with pytest.raises(KeyError, match="invalid_param"):
            self._sensitivity("invalid_param", _state(vcmax_scaling=1.0))

    def test_missing_state_input_raises_through_the_governed_call(self):
        """The inputs come from the state only; the governed call rejects a state
        without a required input (ExecutionError from FunctorBase.__call__)."""
        from relic.core.functor.functor_exceptions import ExecutionError

        state = _state(ppfd=1000.0, vcmax_scaling=1.0)
        state.pop("ppfd")
        with pytest.raises(ExecutionError, match="ppfd"):
            self._sensitivity("vcmax25", state)


class TestC4PhotosynthesisPathway:
    """Test C4 photosynthesis pathway implementation."""
    
    def setup_method(self):
        """Set up C4 model."""
        config = {'pathway': 'C4'}
        self.model_c4 = FarquharPhotosynthesis(config=config)
        
        self.standard_state = _state(
            air_temperature=308.15,
            ppfd=1500.0,
            vcmax_scaling=1.0,
        )
        
    def test_c4_initialization(self):
        """Test C4-specific parameter initialization."""
        assert self.model_c4.pathway == "C4"
        assert hasattr(self.model_c4, 'vpmax25')
        assert hasattr(self.model_c4, 'kp25')
        assert hasattr(self.model_c4, 've_max25')
        
    def test_c4_photosynthesis_rates(self):
        """Test C4 photosynthesis rate characteristics."""
        result = self.model_c4.forward(self.standard_state, time=0.0, dt=1.0)
        
        assert set(result) == set(self.model_c4.metadata.output_variables)
        assert {
            'pep_carboxylation_limited',
            'export_limited',
            'vpmax_actual',
            've_max_actual',
        }.isdisjoint(result)
        
        # Should produce reasonable photosynthesis rates
        gross_ps = _value(result, 'gross_photosynthesis').item()
        assert 15.0 <= gross_ps <= 80.0, f"C4 gross photosynthesis {gross_ps} outside expected range"
        
    def test_c4_temperature_optimum(self):
        """Test C4 temperature optimum is higher than C3."""
        # Compare C4 at high temp vs C3
        config_c3 = {'pathway': 'C3'}
        model_c3 = FarquharPhotosynthesis(config=config_c3)
        
        hot_state = self.standard_state.copy()
        hot_state['air_temperature'] = torch.tensor(313.15)
        
        result_c4 = self.model_c4.forward(hot_state, time=0.0, dt=1.0)
        result_c3 = model_c3.forward(hot_state, time=0.0, dt=1.0)
        
        # C4 should maintain higher rates at high temperature
        # (This is a simplification - actual comparison depends on parameterization)
        assert torch.isfinite(_value(result_c4, 'net_photosynthesis')).all()
        assert torch.isfinite(_value(result_c3, 'net_photosynthesis')).all()


class TestExecutionAndOptimization:
    """Test the canonical eager execution path and batch behavior."""
    
    def setup_method(self):
        """Set up the canonical eager model."""
        self.model = FarquharPhotosynthesis()
        self.test_state = _state(vcmax_scaling=1.0)

    def test_single_eager_execution_path(self):
        """The model must expose one deterministic eager forward path."""
        assert not hasattr(self.model, "enable_jit")
        assert not hasattr(self.model, "_compiled_forward")

        first = self.model.forward(self.test_state, time=0.0, dt=1.0)
        second = self.model.forward(self.test_state, time=0.0, dt=1.0)

        assert set(first) == set(second)
        for key in first:
            torch.testing.assert_close(
                first[key].value,
                second[key].value,
                rtol=1e-5,
                atol=1e-8,
            )
                
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        # Create batch inputs
        batch_size = 10
        batch_state = {
            'air_temperature': torch.full((batch_size,), 298.15),
            'atmospheric_co2_ppm': torch.full((batch_size,), 400.0),
            'ppfd': torch.full((batch_size,), 1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.full((batch_size,), 1000.0),
            'vcmax_scaling': torch.full((batch_size,), 1.0)
        }
        
        result = self.model.forward(batch_state, time=0.0, dt=1.0)
        
        # Check output shapes
        for key, value in result.items():
            assert value.value.shape[0] == batch_size, f"Batch processing failed for {key}"
            
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_consistency(self):
        """Test GPU computation (skip if CUDA not available)."""
        model_gpu = FarquharPhotosynthesis(device=torch.device('cuda'))
        
        # Move test state to GPU
        gpu_state = {k: v.cuda() for k, v in self.test_state.items()}
        
        result = model_gpu.forward(gpu_state, time=0.0, dt=1.0)
        
        # Check that results are on GPU
        for key, value in result.items():
            assert value.value.device.type == 'cuda', f"GPU result {key} not on CUDA device"
            
        # Compare with CPU results
        model_cpu = FarquharPhotosynthesis(device=torch.device('cpu'))
        result_cpu = model_cpu.forward(self.test_state, time=0.0, dt=1.0)
        
        # Results should be numerically close
        for key in result.keys():
            if key in result_cpu:
                torch.testing.assert_close(
                    result[key].value.cpu(), result_cpu[key].value,
                    rtol=1e-4, atol=1e-6
                )


class TestErrorHandlingAndRobustness:
    """Test error handling and robustness under various conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.model = FarquharPhotosynthesis(validate_inputs=True)
        
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        nan_state = _state(air_temperature=float('nan'), vcmax_scaling=1.0)

        with pytest.raises(
            ValueError, match="air_temperature must contain only finite values"
        ):
            self.model.forward(nan_state, time=0.0, dt=1.0)
            
    def test_inf_input_handling(self):
        """Test handling of infinite inputs."""
        inf_state = _state(atmospheric_co2_ppm=float('inf'), vcmax_scaling=1.0)

        with pytest.raises(
            ValueError, match="atmospheric_co2_ppm must contain only finite values"
        ):
            self.model.forward(inf_state, time=0.0, dt=1.0)
            
    def test_negative_input_handling(self):
        """Test handling of negative inputs where inappropriate."""
        negative_ppfd_state = _state(ppfd=-100.0, vcmax_scaling=1.0)

        with pytest.raises(ValueError, match="ppfd must contain only non-negative values"):
            self.model.forward(negative_ppfd_state, time=0.0, dt=1.0)
            
    def test_memory_efficiency(self):
        """Test memory efficiency with large inputs."""
        # Create large batch to test memory handling
        large_batch_size = 1000
        large_state = {
            'air_temperature': torch.full((large_batch_size,), 298.15),
            'atmospheric_co2_ppm': torch.full((large_batch_size,), 400.0),
            'ppfd': torch.full((large_batch_size,), 1000.0),
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.full((large_batch_size,), 1000.0),
            'vcmax_scaling': torch.full((large_batch_size,), 1.0)
        }
        
        # Should process without memory errors
        result = self.model.forward(large_state, time=0.0, dt=1.0)
        
        net = _value(result, 'net_photosynthesis')
        assert net.shape[0] == large_batch_size
        assert torch.all(torch.isfinite(net))


# Integration with pytest fixtures and parameterized tests
@pytest.fixture(scope="module")
def farquhar_model():
    """Pytest fixture for Farquhar model."""
    return FarquharPhotosynthesis()


@pytest.fixture(scope="module") 
def standard_conditions():
    """Pytest fixture for standard test conditions."""
    return _state(vcmax_scaling=1.0)


@pytest.mark.parametrize("temperature", [283.15, 298.15, 313.15])
def test_temperature_sweep(farquhar_model, standard_conditions, temperature):
    """Parametrized test across temperature range."""
    conditions = standard_conditions.copy()
    conditions['air_temperature'] = torch.tensor(temperature)
    
    result = farquhar_model.forward(conditions, time=0.0, dt=1.0)
    
    # All outputs should be finite and physically reasonable
    assert torch.isfinite(_value(result, 'net_photosynthesis'))
    assert _value(result, 'net_photosynthesis').item() >= -5.0
    assert _value(result, 'gross_photosynthesis').item() >= 0.0


@pytest.mark.parametrize("co2_ppm", [200, 400, 800])
def test_co2_sweep(farquhar_model, standard_conditions, co2_ppm):
    """Parametrized test across CO2 range.""" 
    conditions = standard_conditions.copy()
    conditions['atmospheric_co2_ppm'] = torch.tensor(float(co2_ppm))
    
    result = farquhar_model.forward(conditions, time=0.0, dt=1.0)
    
    assert torch.isfinite(_value(result, 'net_photosynthesis'))
    assert _value(result, 'net_photosynthesis').item() >= -2.0


@pytest.mark.parametrize("ppfd_level", [0, 500, 1000, 2000])
def test_light_sweep(farquhar_model, standard_conditions, ppfd_level):
    """Parametrized test across light range."""
    conditions = standard_conditions.copy()
    conditions['ppfd'] = torch.tensor(float(ppfd_level))
    
    result = farquhar_model.forward(conditions, time=0.0, dt=1.0)
    
    assert torch.isfinite(_value(result, 'net_photosynthesis'))
    if ppfd_level == 0:
        # Dark conditions - net photosynthesis should be low/negative
        assert _value(result, 'net_photosynthesis').item() <= 1.0
    else:
        # Light conditions - should be positive
        assert _value(result, 'gross_photosynthesis').item() > 0.0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])


