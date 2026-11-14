"""
L1 Unit Tests for Farquhar Photosynthesis Model

Tests individual components and methods of the FarquharPhotosynthesis class
in isolation, verifying parameter loading, caching, and basic calculations.
"""

import pytest
import torch

# Import the module to test
from relic.scientific.biosphere.photosynthesis import FarquharPhotosynthesis
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager


def _value(item):
    """Return the tensor payload from an explicit state update."""
    assert isinstance(item, UpdateItem)
    return item.value


def _replace_update_value(output, key, value):
    """Replace one payload while preserving the declared runtime contract."""
    item = output[key]
    output[key] = UpdateItem(
        value=value,
        mode=item.mode,
        unit=item.unit,
        per_timestep=item.per_timestep,
        source_functor=item.source_functor,
    )


def _declared_parameter(model, name):
    """Return (description, units) of the metadata entry 'Description [units]' of the
    full key biosphere.photosynthesis.<name>."""
    entry = model.metadata.parameters[f"biosphere.photosynthesis.{name}"]
    description, separator, label = entry.rpartition(" [")
    assert separator and label.endswith("]"), entry
    return description, label[:-1]


class TestFarquharPhotosynthesisL1:
    """L1 unit tests for FarquharPhotosynthesis model."""
    
    @pytest.fixture
    def param_manager(self):
        """Get the global parameter manager."""
        return get_global_manager()
    
    @pytest.fixture
    def basic_model(self):
        """Create a basic photosynthesis model for testing."""
        model = FarquharPhotosynthesis(
            name="test_photosynthesis",
            enable_gpu=False,   # Use CPU for testing
            validate_inputs=True
        )
        return model
    
    @pytest.fixture
    def basic_state(self):
        """Create a basic state dictionary for testing."""
        return {
            'air_temperature': torch.tensor(298.15),  # K
            'atmospheric_co2_ppm': torch.tensor(400.0),  # umol mol-1
            'ppfd': torch.tensor(1000.0),  # umol photons m-2 s-1
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),  # Pa
            'vcmax_scaling': torch.tensor(1.0),
        }
    
    def test_initialization(self, basic_model):
        """Test that the model initializes correctly."""
        assert basic_model is not None
        assert basic_model.name == "test_photosynthesis"
        assert basic_model.pathway == "C3"
        assert hasattr(basic_model, '_param_cache')
        assert hasattr(basic_model, 'vcmax25')
        assert hasattr(basic_model, 'jmax25')
        assert hasattr(basic_model, 'rd25')

    def test_removed_jit_constructor_branch(self):
        """The failed compiled fallback must not remain a selectable runtime."""
        import inspect

        assert "enable_jit" not in inspect.signature(
            FarquharPhotosynthesis.__init__
        ).parameters
        with pytest.raises(TypeError, match="enable_jit"):
            FarquharPhotosynthesis(enable_jit=True)

    def test_metadata_declares_required_and_conditional_inputs(self, basic_model):
        """Metadata must cover every C3/C4 input consumed by forward."""
        assert basic_model.metadata.input_variables == [
            "air_temperature",
            "atmospheric_co2_ppm",
            "ppfd",
            "vapor_pressure_deficit",
            "vcmax_scaling",
            "vpmax_scaling",
            "jmax_scaling",
            "water_stress_factor",
            "temperature_stress_factor",
            "ozone_stress_factor",
            "leaf_area_index",
        ]
        assert set(basic_model.optional_input_variables) == set(
            basic_model.metadata.input_variables
        ).difference(basic_model.get_required_inputs())
        assert set(basic_model.metadata.conservation_laws) == {"carbon_boundary"}

    @pytest.mark.parametrize(
        "legacy_key, canonical_key",
        [
            ("temperature", "air_temperature"),
            ("co2_concentration", "atmospheric_co2_ppm"),
            ("par", "ppfd"),
            ("vpd", "vapor_pressure_deficit"),
        ],
    )
    def test_legacy_input_aliases_are_rejected(
        self, basic_model, basic_state, legacy_key, canonical_key
    ):
        """Legacy scientific input names must not select a second runtime path."""
        state = dict(basic_state)
        state[legacy_key] = state.pop(canonical_key)

        with pytest.raises(ValueError, match=canonical_key):
            basic_model.forward(state=state, time=0.0, dt=60.0)
    
    def test_parameter_caching(self, basic_model):
        """Test that numerical parameters are properly cached."""
        # Check that cache exists and is populated
        assert hasattr(basic_model, '_param_cache')
        assert isinstance(basic_model._param_cache, dict)
        
        # Check key cached parameters
        expected_cache_keys = [
            'epsilon_divide',
            'clamp_min_zero',
            'clamp_min_temperature',
            'clamp_max_temperature',
            'vpd_response_d0',
        ]
        
        for key in expected_cache_keys:
            assert key in basic_model._param_cache, f"Missing cached parameter: {key}"
            assert basic_model._param_cache[key] is not None

        assert {
            'nan_co2_default',
            'nan_par_default',
            'default_temperature',
        }.isdisjoint(basic_model._param_cache)
    
    def test_c3_initialization(self):
        """Test C3 pathway initialization."""
        model = FarquharPhotosynthesis(pathway="C3")
        assert model.pathway == "C3"
        assert not hasattr(model, 'vpmax25')  # C4-specific parameter should not exist
    
    def test_c4_initialization(self):
        """Test C4 pathway initialization."""
        model = FarquharPhotosynthesis(
            name="test_c4",
            config={'pathway': 'C4'},
            enable_gpu=False
        )
        assert model.pathway == "C4"
        assert hasattr(model, 'vpmax25')  # C4-specific parameter should exist
        assert hasattr(model, 'kp25')
        assert hasattr(model, 'phi')
    
    def test_invalid_pathway_raises_error(self):
        """Test that invalid pathway raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported pathway"):
            FarquharPhotosynthesis(config={'pathway': 'CAM'})
    
    def test_temperature_response_vcmax(self, basic_model):
        """Test temperature response function for Vcmax."""
        # Test at reference temperature (25 C)
        temp_ref = torch.tensor(298.15)
        vcmax_ref = basic_model._temperature_response_vcmax(temp_ref)
        
        # Should be close to vcmax25 at reference temperature
        expected = basic_model.vcmax25.to(temp_ref)
        assert torch.allclose(vcmax_ref, expected, rtol=0.1)
        
        # Test temperature dependency
        temp_cold = torch.tensor(283.15)  # 10 C
        temp_hot = torch.tensor(308.15)  # 35 C
        
        vcmax_cold = basic_model._temperature_response_vcmax(temp_cold)
        vcmax_hot = basic_model._temperature_response_vcmax(temp_hot)
        
        # Vcmax should be lower at cold temperature
        assert vcmax_cold < vcmax_ref
        # Vcmax should peak around 35 C then decline
        assert vcmax_hot > vcmax_ref
    
    def test_electron_transport_rate(self, basic_model):
        """Test electron transport rate calculation."""
        ppfd = torch.tensor([0.0, 500.0, 1000.0, 2000.0])
        jmax = torch.tensor(120.0)
        
        j_actual = basic_model._electron_transport_rate(ppfd, jmax)
        
        # Check that J increases with PAR
        assert torch.all(j_actual[1:] > j_actual[:-1])
        
        # Check that J is bounded by Jmax
        assert torch.all(j_actual <= jmax)
        
        # Check J = 0 when PAR = 0
        assert torch.allclose(j_actual[0], torch.zeros_like(j_actual[0]), atol=1e-6)

    def test_canopy_scaling_preserves_public_assimilation_energy_area_basis(
        self,
        basic_model,
        basic_state,
    ):
        """Canopy A and the public J coupling port share one ground-area scale."""
        baseline = basic_model.forward(basic_state, time=0.0, dt=1.0)
        canopy_state = dict(basic_state)
        canopy_state["leaf_area_index"] = torch.tensor(3.0)

        canopy = basic_model.forward(canopy_state, time=0.0, dt=1.0)
        extinction = torch.as_tensor(
            basic_model._param_cache["canopy_light_extinction"]
        )
        baseline_lai = torch.as_tensor(basic_state["leaf_area_index"])
        baseline_effective_lai = (
            1.0 - torch.exp(-extinction * baseline_lai)
        ) / extinction
        canopy_effective_lai = (
            1.0 - torch.exp(-extinction * canopy_state["leaf_area_index"])
        ) / extinction
        expected_ratio = canopy_effective_lai / baseline_effective_lai

        assert torch.allclose(
            basic_model._canopy_lai_scale(baseline_lai),
            baseline_effective_lai,
        )
        assert torch.allclose(
            _value(canopy["gross_photosynthesis"]),
            _value(baseline["gross_photosynthesis"]) * expected_ratio,
        )
        assert torch.allclose(
            _value(canopy["electron_transport_rate"]),
            _value(baseline["electron_transport_rate"]) * expected_ratio,
        )
    
    def test_carboxylation_limited_rate(self, basic_model):
        """Test RuBisCO-limited carboxylation rate calculation."""
        vcmax = torch.tensor(60.0)
        ci = torch.tensor([100.0, 200.0, 400.0])  # Different CO2 concentrations
        kc = torch.tensor(404.0)
        ko = torch.tensor(278.0)
        gamma_star = torch.tensor(42.75)
        
        ac = basic_model._carboxylation_limited_rate(vcmax, ci, kc, ko, gamma_star)
        
        # Check that Ac increases with Ci
        assert torch.all(ac[1:] > ac[:-1])
        
        # Check that all rates are positive
        assert torch.all(ac >= 0.0)
        
        # Check that Ac approaches vcmax as Ci increases
        assert ac[-1] < vcmax  # Should approach but not exceed
    
    def test_light_limited_rate(self, basic_model):
        """Test light-limited carboxylation rate calculation."""
        j = torch.tensor([50.0, 100.0, 150.0])
        ci = torch.tensor(400.0)
        gamma_star = torch.tensor(42.75)
        
        aj = basic_model._light_limited_rate(j, ci, gamma_star)
        
        # Check that Aj increases with J
        assert torch.all(aj[1:] > aj[:-1])
        
        # Check that all rates are positive
        assert torch.all(aj >= 0.0)
    
    def test_forward_c3_basic(self, basic_model, basic_state):
        """Test basic C3 forward pass."""
        output = basic_model.forward(basic_state, time=0.0, dt=1.0)

        assert set(output) == {
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
        for key, item in output.items():
            assert isinstance(item, UpdateItem), key
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert torch.isfinite(item.value).all(), key

    def test_forward_exposes_unique_unit_aware_carbon_ports(
        self, basic_model, basic_state
    ):
        """Molar assimilation and mass GPP must be distinct declared ports."""
        output = basic_model.forward(basic_state, time=0.0, dt=1.0)

        assert set(output) == set(basic_model.metadata.output_variables)
        assert {
            "gross_photosynthesis",
            "net_photosynthesis",
            "gross_primary_productivity",
        }.issubset(output)
        assert {
            "gpp",
            "an",
            "net_assimilation",
            "photosynthesis_rate",
            "ozone_stress_factor",
        }.isdisjoint(output)

        expected_gpp = (
            _value(output["gross_photosynthesis"])
            * basic_model.carbon_molar_mass_g_mol
            * 1.0e-9
        )
        assert torch.allclose(
            _value(output["gross_primary_productivity"]), expected_gpp, rtol=1.0e-6
        )

    def test_open_carbon_boundary_checker_matches_assimilation_identity(
        self, basic_model, basic_state
    ):
        """Molar assimilation and carbon-mass conversion must both close."""
        output = basic_model.forward(basic_state, time=0.0, dt=1.0)

        residuals = basic_model.check_conservation_laws(basic_state, output)
        assert residuals == {"carbon_boundary": pytest.approx(0.0, abs=1.0e-12)}

        tampered = dict(output)
        gpp = _value(tampered["gross_primary_productivity"])
        _replace_update_value(
            tampered,
            "gross_primary_productivity",
            gpp * gpp.new_tensor(1.1),
        )
        assert basic_model.check_conservation_laws(
            basic_state, tampered
        ) == {"carbon_boundary": pytest.approx(1.0 / 11.0, rel=1.0e-6)}

    def test_open_carbon_boundary_allows_nighttime_net_respiration(
        self, basic_model, basic_state
    ):
        """Zero light may yield negative net exchange while gross uptake is zero."""
        state = dict(basic_state)
        state["ppfd"] = torch.tensor(0.0)

        output = basic_model.forward(state, time=0.0, dt=60.0)

        assert _value(output["gross_photosynthesis"]).item() == pytest.approx(0.0)
        assert _value(output["gross_primary_productivity"]).item() == pytest.approx(
            0.0
        )
        assert _value(output["net_photosynthesis"]).item() < 0.0
        assert basic_model.check_conservation_laws(state, output) == {
            "carbon_boundary": pytest.approx(0.0, abs=1.0e-12)
        }

    def test_open_carbon_boundary_checker_fails_closed_on_malformed_ports(
        self, basic_model, basic_state
    ):
        """Malformed or sign-invalid carbon boundary ports must fail fast."""
        output = basic_model.forward(basic_state, time=0.0, dt=1.0)

        for key in (
            "gross_photosynthesis",
            "net_photosynthesis",
            "gross_primary_productivity",
            "diagnostics.photosynthesis.dark_respiration",
        ):
            missing = dict(output)
            missing.pop(key)
            with pytest.raises(KeyError, match=key):
                basic_model.check_conservation_laws(basic_state, missing)

        malformed_cases = [
            ("gross_photosynthesis", 1.0, TypeError, "Tensor"),
            (
                "gross_photosynthesis",
                torch.empty(0),
                ValueError,
                "empty",
            ),
            (
                "gross_photosynthesis",
                torch.tensor(1),
                TypeError,
                "floating",
            ),
            (
                "gross_photosynthesis",
                torch.tensor(float("nan")),
                ValueError,
                "finite",
            ),
            (
                "gross_photosynthesis",
                torch.tensor(float("inf")),
                ValueError,
                "finite",
            ),
            (
                "gross_photosynthesis",
                torch.tensor(-1.0),
                ValueError,
                "non-negative",
            ),
            (
                "gross_primary_productivity",
                torch.tensor(-1.0),
                ValueError,
                "non-negative",
            ),
            (
                "diagnostics.photosynthesis.dark_respiration",
                torch.tensor(-1.0),
                ValueError,
                "non-negative",
            ),
            (
                "net_photosynthesis",
                torch.ones(2),
                ValueError,
                "shape",
            ),
            (
                "net_photosynthesis",
                torch.tensor(
                    1.0,
                    dtype=(
                        torch.float32
                        if _value(output["net_photosynthesis"]).dtype
                        is torch.float64
                        else torch.float64
                    ),
                ),
                ValueError,
                "dtype",
            ),
        ]
        for key, value, error_type, match in malformed_cases:
            malformed = dict(output)
            _replace_update_value(malformed, key, value)
            try:
                basic_model.check_conservation_laws(basic_state, malformed)
            except Exception as exc:
                assert isinstance(exc, error_type), (key, value, exc)
                assert match in str(exc), (key, value, exc)
            else:
                pytest.fail(
                    f"Accepted malformed conservation port {key}: {value!r}"
                )

    def test_forward_is_the_single_rate_entry(self, basic_model, basic_state):
        """forward is the only calculation entry (item 114): the uncalled
        _compute_rate_core wrapper, which merged an environment mapping over the
        state, is removed. The governed call returns the forward net assimilation."""
        assert not hasattr(FarquharPhotosynthesis, "_compute_rate_core")
        governed = _value(basic_model(basic_state, time=0.0, dt=1.0)["net_photosynthesis"])
        kernel = _value(
            basic_model.forward(basic_state, time=0.0, dt=1.0)["net_photosynthesis"]
        )
        assert isinstance(governed, torch.Tensor)
        assert torch.equal(governed, kernel)
        assert bool((governed > 0.0).all())
    
    @pytest.mark.parametrize(
        "key,bad_value",
        [
            ("air_temperature", float("nan")),
            ("atmospheric_co2_ppm", float("inf")),
            ("ppfd", float("-inf")),
            ("vapor_pressure_deficit", float("nan")),
        ],
    )
    def test_non_finite_required_inputs_fail_fast(
        self, basic_model, basic_state, key, bad_value
    ):
        """Required forcing inputs must never be replaced by hidden defaults."""
        state = dict(basic_state)
        state[key] = torch.tensor(bad_value)

        with pytest.raises(ValueError, match=f"{key} must contain only finite values"):
            basic_model.forward(state, time=0.0, dt=1.0)
    
    @pytest.mark.parametrize("temperature", [200.0, 373.15])
    def test_out_of_range_temperature_fails_fast(
        self, basic_model, basic_state, temperature
    ):
        """Out-of-domain temperatures must not be silently clamped."""
        state = dict(basic_state)
        state["air_temperature"] = torch.tensor(temperature)

        with pytest.raises(ValueError, match="air_temperature outside supported range"):
            basic_model.forward(state, time=0.0, dt=1.0)
    
    def test_water_stress_effect(self, basic_model, basic_state):
        """Test that water stress reduces photosynthesis."""
        # Full water availability
        state_no_stress = basic_state.copy()
        state_no_stress['water_stress_factor'] = torch.tensor(1.0)
        output_no_stress = basic_model.forward(state_no_stress, time=0.0, dt=1.0)
        
        # Water stress
        state_stress = basic_state.copy()
        state_stress['water_stress_factor'] = torch.tensor(0.5)
        output_stress = basic_model.forward(state_stress, time=0.0, dt=1.0)
        
        # Photosynthesis should be reduced under water stress
        assert _value(output_stress['gross_photosynthesis']) < _value(output_no_stress['gross_photosynthesis'])
        assert _value(output_stress['net_photosynthesis']) < _value(output_no_stress['net_photosynthesis'])
    
    def test_leaf_water_potential_does_not_define_second_stress_closure(
        self, basic_model, basic_state
    ):
        """Leaf water potential must not bypass the canonical stress factor."""
        wet = dict(basic_state)
        wet["water_stress_factor"] = torch.tensor(0.5)
        wet["leaf_water_potential"] = torch.tensor(-0.1)
        dry = dict(wet)
        dry["leaf_water_potential"] = torch.tensor(-2.0)

        output_wet = basic_model.forward(wet, time=0.0, dt=1.0)
        output_dry = basic_model.forward(dry, time=0.0, dt=1.0)

        assert torch.equal(
            _value(output_wet["gross_photosynthesis"]),
            _value(output_dry["gross_photosynthesis"]),
        )

    @pytest.mark.parametrize(
        "factor_name",
        ["temperature_stress_factor", "ozone_stress_factor"],
    )
    def test_canonical_stress_factors_reduce_c3_assimilation(
        self, basic_model, basic_state, factor_name
    ):
        """Canonical external stress controls must reduce C3 assimilation."""
        unstressed = dict(basic_state)
        unstressed[factor_name] = torch.tensor(1.0)
        stressed = dict(basic_state)
        stressed[factor_name] = torch.tensor(0.5)

        gross_unstressed = _value(
            basic_model.forward(unstressed, time=0.0, dt=1.0)["gross_photosynthesis"]
        )
        gross_stressed = _value(
            basic_model.forward(stressed, time=0.0, dt=1.0)["gross_photosynthesis"]
        )

        assert gross_stressed < gross_unstressed

    @pytest.mark.parametrize(
        ("factor_name", "factor_value"),
        [
            ("water_stress_factor", -0.01),
            ("water_stress_factor", 1.01),
            ("temperature_stress_factor", -0.01),
            ("temperature_stress_factor", 1.01),
            ("ozone_stress_factor", -0.01),
            ("ozone_stress_factor", 1.01),
        ],
    )
    def test_stress_factors_reject_values_outside_unit_interval(
        self, basic_model, basic_state, factor_name, factor_value
    ):
        """Stress factors outside [0, 1] must fail instead of being clamped."""
        state = dict(basic_state)
        state[factor_name] = torch.tensor(factor_value)

        with pytest.raises(ValueError, match=rf"{factor_name} outside supported range"):
            basic_model.forward(state, time=0.0, dt=1.0)
    
    def test_vcmax_scaling(self, basic_model, basic_state):
        """Test that Vcmax scaling affects photosynthesis."""
        # Normal scaling
        state_normal = basic_state.copy()
        state_normal['vcmax_scaling'] = torch.tensor(1.0)
        output_normal = basic_model.forward(state_normal, time=0.0, dt=1.0)
        
        # Reduced scaling (e.g., nitrogen limitation)
        state_reduced = basic_state.copy()
        state_reduced['vcmax_scaling'] = torch.tensor(0.5)
        output_reduced = basic_model.forward(state_reduced, time=0.0, dt=1.0)
        
        # Photosynthesis should be reduced with lower Vcmax
        assert _value(output_reduced['gross_photosynthesis']) < _value(output_normal['gross_photosynthesis'])
    
    def test_light_response(self, basic_model):
        """Test photosynthetic light response curve."""
        ppfd_values = torch.tensor([0.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0])
        outputs = []
        
        for ppfd in ppfd_values:
            state = {
                'air_temperature': torch.tensor(298.15),
                'atmospheric_co2_ppm': torch.tensor(400.0),
                'ppfd': ppfd,
                'leaf_area_index': torch.tensor(1.0),
                'vapor_pressure_deficit': torch.tensor(1000.0),
                'vcmax_scaling': torch.tensor(1.0)
            }
            output = basic_model.forward(state, time=0.0, dt=1.0)
            outputs.append(_value(output['gross_photosynthesis']))
        
        gpp_values = torch.stack(outputs)
        
        # Check that GPP increases with PAR
        assert torch.all(gpp_values[1:] >= gpp_values[:-1])
        
        # Check that GPP is zero when PAR is zero
        assert torch.allclose(gpp_values[0], torch.zeros_like(gpp_values[0]), atol=0.1)
        
        # Check for light saturation (diminishing returns)
        increments = gpp_values[1:] - gpp_values[:-1]
        assert increments[-1] < increments[0]  # Smaller increase at high light
    
    def test_co2_response(self, basic_model):
        """Test photosynthetic CO2 response curve."""
        co2_values = torch.tensor([100.0, 200.0, 400.0, 600.0, 800.0, 1000.0])
        outputs = []
        
        for co2 in co2_values:
            state = {
                'air_temperature': torch.tensor(298.15),
                'atmospheric_co2_ppm': co2,
                'ppfd': torch.tensor(1000.0),
                'leaf_area_index': torch.tensor(1.0),
                'vapor_pressure_deficit': torch.tensor(1000.0),
                'vcmax_scaling': torch.tensor(1.0)
            }
            output = basic_model.forward(state, time=0.0, dt=1.0)
            outputs.append(_value(output['gross_photosynthesis']))
        
        gpp_values = torch.stack(outputs)
        
        # Check that GPP increases with CO2
        assert torch.all(gpp_values[1:] > gpp_values[:-1])
        
        # Check for CO2 saturation (diminishing returns)
        increments = gpp_values[1:] - gpp_values[:-1]
        assert increments[-1] < increments[0]  # Smaller increase at high CO2
    
    def test_limitation_factor(self, basic_model):
        """Test that limitation factor correctly identifies limiting process."""
        # Low light condition (should be light-limited)
        state_low_light = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(1000.0),  # High CO2
            'ppfd': torch.tensor(100.0),  # Low light
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'vcmax_scaling': torch.tensor(1.0)
        }
        output_low_light = basic_model.forward(state_low_light, time=0.0, dt=1.0)
        assert _value(output_low_light['diagnostics.photosynthesis.limitation_factor']) == 1.0
        
        # Low CO2 condition (should be RuBisCO-limited)
        state_low_co2 = {
            'air_temperature': torch.tensor(298.15),
            'atmospheric_co2_ppm': torch.tensor(100.0),  # Low CO2
            'ppfd': torch.tensor(2000.0),  # High light
            'leaf_area_index': torch.tensor(1.0),
            'vapor_pressure_deficit': torch.tensor(1000.0),
            'vcmax_scaling': torch.tensor(1.0)
        }
        output_low_co2 = basic_model.forward(state_low_co2, time=0.0, dt=1.0)
        assert _value(output_low_co2['diagnostics.photosynthesis.limitation_factor']) == 0.0

    def test_vpd_response_uses_pa(self, basic_model, basic_state):
        """The Leuning response must compare VPD and D0 in Pa."""
        state_dry = dict(basic_state)
        state_dry["vapor_pressure_deficit"] = torch.tensor(1500.0)
        state_humid = dict(basic_state)
        state_humid["vapor_pressure_deficit"] = torch.tensor(0.0)

        dry = _value(basic_model.forward(state_dry, 0.0, 1.0)["gross_photosynthesis"])
        humid = _value(basic_model.forward(state_humid, 0.0, 1.0)["gross_photosynthesis"])

        assert basic_model._param_cache["vpd_response_d0"] == pytest.approx(1500.0)
        assert dry < humid
    
    def test_dependencies(self, basic_model):
        """Test that dependencies are correctly declared."""
        deps = basic_model.get_dependencies()
        assert isinstance(deps, list)
        assert 'environmental_forcing' in deps
        assert 'atmospheric_radiation' in deps
        assert set(deps) == {'environmental_forcing', 'atmospheric_radiation'}
    
    def test_required_inputs(self, basic_model):
        """Test that required inputs are correctly declared."""
        inputs = basic_model.get_required_inputs()
        assert isinstance(inputs, list)
        assert inputs == [
            'air_temperature',
            'atmospheric_co2_ppm',
            'ppfd',
            'vapor_pressure_deficit',
            'leaf_area_index',
        ]
    
    def test_provides(self, basic_model):
        """Test that provided outputs are correctly declared."""
        outputs = basic_model.get_provides()
        assert isinstance(outputs, list)
        assert outputs == basic_model.metadata.output_variables

    def test_reference_rate_labels_carry_the_emitted_species(self, basic_model, param_manager):
        """Jmax is an electron-transport capacity (Farquhar et al. 1980: ueq m-2 s-1),
        Vcmax a carboxylation rate and Rd a CO2 release rate. The parameter labels in
        the YAML SSOT (global and per-PFT) and in the Functor metadata carry the same
        counted species as the emitted outputs, because the unit SSOT never converts
        an annotated label to a plain or differently annotated one."""
        from pathlib import Path

        from relic.data.standards.unit_converter import unit_converter

        expected = {
            "vcmax25": "umol CO2 m-2 s-1",
            "jmax25": "umol e- m-2 s-1",
            "rd25": "umol CO2 m-2 s-1",
        }
        pft_root = (
            Path(__file__).resolve().parents[4]
            / "relic" / "parameters" / "database" / "relic" / "biosphere" / "pft"
        )
        pft_names = sorted(
            path.stem for path in pft_root.glob("*.yaml")
            if path.stem not in {"__init__", "metadata"}
        )
        assert pft_names
        for name, units in expected.items():
            assert param_manager.get_parameter(f"biosphere.photosynthesis.{name}").units == units
            assert _declared_parameter(basic_model, name)[1] == units
            for pft in pft_names:
                key = f"biosphere.pft.{pft}.photosynthesis.{name}"
                assert param_manager.get_parameter(key).units == units, key

        emitted = {
            "diagnostics.photosynthesis.jmax_actual": "jmax25",
            "electron_transport_rate": "jmax25",
            "diagnostics.photosynthesis.dark_respiration": "rd25",
        }
        output = basic_model(
            {
                "air_temperature": torch.tensor(298.15),
                "atmospheric_co2_ppm": torch.tensor(400.0),
                "ppfd": torch.tensor(1000.0),
                "leaf_area_index": torch.tensor(1.0),
                "vapor_pressure_deficit": torch.tensor(1000.0),
            },
            0.0,
            1.0,
        )
        for port, parameter in emitted.items():
            assert unit_converter.can_convert(expected[parameter], output[port].unit, port), port

    def test_c4_capacity_labels_count_the_fixed_co2(self, param_manager):
        """The C4 capacities are CO2 fixation rates (von Caemmerer 2000, C4 model).

        Vpmax is the PEP carboxylation capacity (one CO2, as HCO3-, fixed per PEP
        carboxylated) and Ve_max caps the assimilation in A = min(Ap, Aj, Ae), so
        both are compared with the emitted gross assimilation [umol CO2 m-2 s-1].
        The YAML SSOT labels equal the Functor metadata labels, and the unit SSOT
        converts them to the emitted assimilation label.
        """
        from relic.data.standards.unit_converter import unit_converter

        model = FarquharPhotosynthesis(
            name="test_c4_capacity_units", config={'pathway': 'C4'}, enable_gpu=False
        )
        output = model(
            {
                "air_temperature": torch.tensor(298.15),
                "atmospheric_co2_ppm": torch.tensor(400.0),
                "ppfd": torch.tensor(1000.0),
                "leaf_area_index": torch.tensor(1.0),
                "vapor_pressure_deficit": torch.tensor(1000.0),
            },
            0.0,
            1.0,
        )
        for name in ("vpmax25", "ve_max25"):
            label = param_manager.get_parameter(f"biosphere.photosynthesis.{name}").units
            assert label == "umol CO2 m-2 s-1", name
            assert _declared_parameter(model, name)[1] == label, name
            assert unit_converter.can_convert(
                label, output["gross_photosynthesis"].unit, "gross_photosynthesis"
            ), name
            # Requesting the metadata label is an identity read of the YAML value.
            stored = model.get_parameter_value(
                name, namespace="biosphere.photosynthesis", required=True
            )
            annotated = model.get_parameter_value(
                name, namespace="biosphere.photosynthesis",
                units=_declared_parameter(model, name)[1], required=True,
            )
            assert torch.equal(
                torch.as_tensor(annotated).detach(), torch.as_tensor(stored).detach()
            ), name

    def test_c4_parameter_descriptions_match_their_role_in_the_model(self, param_manager):
        """The C4 path computes Ap = Vp (1 - phi) and A = min(Ap, Aj, Ae) with Ae = Ve_max.

        phi multiplies the PEP carboxylation rate as (1 - phi): it is the bundle-sheath
        leakiness phi = L / Vp of von Caemmerer (2000; 2021, Eq. 5), not the fraction
        alpha of O2 evolution in the bundle sheath. Ve_max is a constant cap on A, the
        Rubisco-limited rate V_max of Collatz et al. (1992) and the Rubisco activity
        Vcmax of the von Caemmerer model; C4 Rubisco sits in the bundle sheath, and
        no export (triose phosphate utilization) limitation is part of either model.
        The C3 path has no TPU limitation, so tpu25 had no reader and is retired.
        """
        phi = param_manager.get_parameter("biosphere.photosynthesis.phi").description
        assert "leakiness" in phi and "O2 evolution" not in phi
        ve_max = param_manager.get_parameter("biosphere.photosynthesis.ve_max25").description
        assert "Rubisco" in ve_max and "bundle sheath" in ve_max and "mesophyll" not in ve_max
        model = FarquharPhotosynthesis(
            name="test_c4_descriptions", config={'pathway': 'C4'}, enable_gpu=False
        )
        declared = _declared_parameter(model, "ve_max25")[0]
        assert "Rubisco" in declared and "export" not in declared
        assert param_manager.get_parameter("biosphere.photosynthesis.tpu25") is None

    def test_c4_limitation_code_2_names_the_bundle_sheath_rubisco_limit(self):
        """C4 code 2 marks A = Ae = Ve_max, the bundle-sheath Rubisco capacity (Collatz et
        al. 1992 V_max; von Caemmerer 2000 Vcmax), not an export (triose phosphate
        utilization) limit: lowering only that capacity gives code 2 with gross
        assimilation proportional to it, and the limitation_factor contract names code 2
        as the Rubisco (bundle-sheath carboxylation) limit."""
        from relic.contracts import load_contract_registry

        model = FarquharPhotosynthesis(
            name="test_c4_limitation_code", config={'pathway': 'C4'}, enable_gpu=False
        )

        def run(rubisco_scaling):
            # vcmax_scaling scales Ve_max only; PEP carboxylase and electron transport
            # capacities are held at their unscaled values.
            state = {
                'air_temperature': torch.tensor(298.15, dtype=torch.float64),
                'atmospheric_co2_ppm': torch.tensor(400.0, dtype=torch.float64),
                'ppfd': torch.tensor(1500.0, dtype=torch.float64),
                'leaf_area_index': torch.tensor(1.0),
                'vapor_pressure_deficit': torch.tensor(1000.0, dtype=torch.float64),
                'vcmax_scaling': torch.tensor(rubisco_scaling, dtype=torch.float64),
                'vpmax_scaling': torch.tensor(1.0, dtype=torch.float64),
                'jmax_scaling': torch.tensor(1.0, dtype=torch.float64),
            }
            output = model(state, 0.0, 1.0)
            return (
                _value(output["diagnostics.photosynthesis.limitation_factor"]),
                _value(output["gross_photosynthesis"]),
            )

        code_low, gross_low = run(0.125)
        code_high, gross_high = run(0.25)
        assert code_low.item() == 2.0 and code_high.item() == 2.0
        torch.testing.assert_close(gross_high, 2.0 * gross_low, rtol=1.0e-12, atol=0.0)

        description = load_contract_registry().get_contract(
            "diagnostics.photosynthesis.limitation_factor"
        ).raw["description"]
        assert "2 = Rubisco (bundle-sheath carboxylation) limited" in description
        assert "export" not in description.lower()

    def _reference_temperature_float64_state(self, ppfd):
        light = torch.as_tensor(ppfd, dtype=torch.float64)
        state = {
            'air_temperature': torch.full_like(light, 298.15),
            'atmospheric_co2_ppm': torch.full_like(light, 400.0),
            'ppfd': light,
            'vapor_pressure_deficit': torch.zeros_like(light),
            'leaf_area_index': torch.full_like(light, 3.0),
        }
        for key in (
            'vcmax_scaling', 'jmax_scaling', 'water_stress_factor',
            'temperature_stress_factor', 'ozone_stress_factor',
        ):
            state[key] = torch.ones_like(light)
        return state

    def test_float64_inputs_keep_precision_and_match_the_analytical_c3_oracle(self, basic_model):
        """Float64 forcings must not be narrowed to float32 anywhere in the C3 kernel."""
        state = self._reference_temperature_float64_state([0.0, 1000.0])
        output = basic_model(state, 0.0, 1.0)

        def p(key):
            return float(torch.as_tensor(basic_model.get_parameter_value(key, namespace="biosphere.photosynthesis", required=True)).detach())

        # References: Farquhar et al. (1980); de Pury and Farquhar (1997).
        # Mathematical expression: at T=Tref every temperature response equals its
        #   25 C value; J is the lower non-rectangular root; Ac=Vcmax(Ci-G*)/(Ci+Kc(1+O/Ko));
        #   Aj=J(Ci-G*)/(f1(Ci+f2 G*)); A=min(Ac,Aj); L_eff=(1-exp(-k LAI))/k.
        # Parameter meanings: alpha, theta, Jmax25, Vcmax25, Kc25, Ko25, O, G*25, Rd25,
        #   f1,f2 rubisco kinetic factors, k canopy extinction, LAI leaf area index.
        # Units: rates [umol CO2 m-2 s-1]; J [umol e- m-2 s-1]; Ci,G*,Kc [umol mol-1];
        #   Ko,O [mmol mol-1]; PPFD [umol photons m-2 s-1]; LAI [m2 m-2].
        # Domain/constraints: C3 pathway, zero VPD, unit stress and scaling factors;
        #   the only production regularization is the YAML epsilon in denominators.
        # Sign convention: gross uptake positive; respiration subtracted once.
        light = state['ppfd']
        co2 = state['atmospheric_co2_ppm']
        lai = state['leaf_area_index']
        alpha, theta, jmax = p("alpha"), p("theta"), p("jmax25")
        electrons = (
            alpha * light + jmax
            - torch.sqrt((alpha * light + jmax) ** 2 - 4.0 * theta * alpha * light * jmax)
        ) / (2.0 * theta)
        rubisco = p("vcmax25") * (co2 - p("gamma_star25")) / (
            co2 + p("kc25") * (1.0 + p("o2_concentration") / p("ko25"))
        )
        light_rate = electrons * (co2 - p("gamma_star25")) / (
            p("rubisco_kinetic_factor1") * (co2 + p("rubisco_kinetic_factor2") * p("gamma_star25"))
        )
        extinction = p("canopy_light_extinction")
        scale = (1.0 - torch.exp(-extinction * lai)) / extinction

        for key, item in output.items():
            assert _value(item).dtype == torch.float64, key
        gross = _value(output["gross_photosynthesis"])
        net = _value(output["net_photosynthesis"])
        transport = _value(output["electron_transport_rate"])
        torch.testing.assert_close(gross, torch.minimum(rubisco, light_rate) * scale, rtol=1.0e-9, atol=0.0)
        torch.testing.assert_close(net, (torch.minimum(rubisco, light_rate) - p("rd25")) * scale, rtol=1.0e-9, atol=0.0)
        torch.testing.assert_close(transport, electrons * scale, rtol=1.0e-9, atol=0.0)

    def test_float64_temperature_and_parameter_gradients_survive_the_forward_pass(self, basic_model):
        """Input precision and parameter autograd history must both reach the outputs."""
        state = self._reference_temperature_float64_state([2000.0, 2000.0])
        temperature = torch.tensor([298.15, 303.15], dtype=torch.float64, requires_grad=True)
        state['air_temperature'] = temperature
        basic_model.zero_grad(set_to_none=True)
        output = basic_model(state, 0.0, 1.0)
        gross = _value(output["gross_photosynthesis"])
        assert gross.dtype == torch.float64
        gross.sum().backward()
        assert temperature.grad is not None
        assert temperature.grad.dtype == torch.float64
        assert bool(torch.isfinite(temperature.grad).all())
        assert basic_model.vcmax25.grad is not None
        assert basic_model.vcmax25.grad.dtype == basic_model.vcmax25.dtype
        assert bool(torch.isfinite(basic_model.vcmax25.grad).all())
        limitation = _value(output["diagnostics.photosynthesis.limitation_factor"])
        assert limitation.dtype == torch.float64
        if bool(torch.any(limitation == 0.0)):
            assert bool(torch.any(basic_model.vcmax25.grad != 0.0))

    @staticmethod
    def _float32_state():
        return {
            'air_temperature': torch.tensor(298.15, dtype=torch.float32),
            'atmospheric_co2_ppm': torch.tensor(400.0, dtype=torch.float32),
            'ppfd': torch.tensor(1000.0, dtype=torch.float32),
            'leaf_area_index': torch.tensor(1.0, dtype=torch.float32),
            'vapor_pressure_deficit': torch.tensor(1000.0, dtype=torch.float32),
            'vcmax_scaling': torch.tensor(1.0, dtype=torch.float32),
        }

    def test_float32_inputs_stay_float32_even_with_wider_parameters(self, basic_model):
        """Runtime state owns precision: float32 forcings give float32 outputs."""
        output = basic_model(self._float32_state(), 0.0, 1.0)
        for key, item in output.items():
            assert _value(item).dtype == torch.float32, key

    def test_mixed_precision_inputs_promote_to_the_widest_floating_dtype(self, basic_model):
        """One float64 forcing lifts the whole kernel to float64 instead of narrowing it."""
        state = self._float32_state()
        state['air_temperature'] = torch.tensor(298.15, dtype=torch.float64)
        output = basic_model(state, 0.0, 1.0)
        for key, item in output.items():
            assert _value(item).dtype == torch.float64, key


def test_package_docstring_example_uses_the_declared_input_names():
    """Item 123: the usage example of the photosynthesis package names only declared
    FarquharPhotosynthesis inputs (the stress controls are water_stress_factor,
    temperature_stress_factor and ozone_stress_factor)."""
    import re as _re

    import relic.scientific.biosphere.photosynthesis as package

    example_keys = set(_re.findall(r'"([a-z_0-9]+)": \.\.\.', package.__doc__))
    assert example_keys
    assert example_keys <= set(FarquharPhotosynthesis().metadata.input_variables)
    for name in ("water_stress_factor", "temperature_stress_factor", "ozone_stress_factor"):
        assert name in example_keys, name


class TestFarquharParameterBindingsL1:
    """Farquhar must preserve the existing parameter and checkpoint owners."""

    CAPACITIES = {"vcmax25": 60.0, "jmax25": 120.0, "rd25": 1.2}

    @staticmethod
    def _manager(*, dtype="float32", requires_grad=True):
        from copy import deepcopy

        from relic.parameters import UnifiedParameterManager

        manager = UnifiedParameterManager()
        for name, value in TestFarquharParameterBindingsL1.CAPACITIES.items():
            spec = deepcopy(manager.get_parameter(f"biosphere.photosynthesis.{name}"))
            spec.value = spec.default = value
            spec.dtype = dtype
            spec.requires_grad = requires_grad
            manager.register_parameter(spec)
        return manager

    @staticmethod
    def _state(*, dtype=torch.float32):
        return {
            "air_temperature": torch.full((2,), 298.15, dtype=dtype),
            "atmospheric_co2_ppm": torch.full((2,), 400.0, dtype=dtype),
            "ppfd": torch.tensor([100.0, 2000.0], dtype=dtype),
            "vapor_pressure_deficit": torch.zeros(2, dtype=dtype),
            "leaf_area_index": torch.ones(2, dtype=dtype),
        }

    def test_default_capacities_are_the_original_manager_objects(self):
        model = FarquharPhotosynthesis(enable_gpu=False)
        manager = model.parameter_manager
        for name in self.CAPACITIES:
            key = f"biosphere.photosynthesis.{name}"
            assert getattr(model, name) is manager.nn_parameters[key]

    def test_capacities_keep_injected_manager_identity_and_optimizer_gradients(self):
        manager = self._manager()
        model = FarquharPhotosynthesis(parameter_manager=manager, enable_gpu=False)
        keys = [f"biosphere.photosynthesis.{name}" for name in self.CAPACITIES]
        bindings = [manager.nn_parameters[key] for key in keys]
        for name, key, binding in zip(self.CAPACITIES, keys, bindings, strict=True):
            assert getattr(model, name) is binding
            assert model.manager_binding_key(binding) == key
            assert name not in model.state_dict()
            assert name not in dict(model.named_parameters())
        output = model(self._state(), 0.0, 1.0)
        _value(output["net_photosynthesis"]).sum().backward()
        before = [binding.detach().clone() for binding in bindings]
        gradients = [binding.grad.detach().clone() for binding in bindings]
        for gradient in gradients:
            assert bool(torch.isfinite(gradient).all())
            assert bool(torch.any(gradient != 0.0))
        optimizer = torch.optim.SGD(bindings, lr=0.05)
        optimizer.step()
        for name, binding, value, gradient in zip(self.CAPACITIES, bindings, before, gradients, strict=True):
            assert getattr(model, name) is binding
            torch.testing.assert_close(binding, value - 0.05 * gradient)

    @pytest.mark.parametrize("freeze_before_construction", [False, True])
    def test_manager_freeze_and_unfreeze_keep_one_capacity(self, freeze_before_construction):
        manager = self._manager()
        keys = [f"biosphere.photosynthesis.{name}" for name in self.CAPACITIES]
        if freeze_before_construction:
            manager.freeze(keys)
        model = FarquharPhotosynthesis(parameter_manager=manager)
        if not freeze_before_construction:
            manager.freeze(keys)
        state = self._state()
        state["air_temperature"].requires_grad_(True)
        _value(model(state, 0.0, 1.0)["net_photosynthesis"]).sum().backward()
        for name, key in zip(self.CAPACITIES, keys, strict=True):
            assert getattr(model, name) is manager.nn_parameters[key]
            assert not getattr(model, name).requires_grad
            assert manager.nn_parameters[key].grad is None
        manager.unfreeze(keys)
        _value(model(self._state(), 0.0, 1.0)["net_photosynthesis"]).sum().backward()
        for name, key in zip(self.CAPACITIES, keys, strict=True):
            assert getattr(model, name) is manager.nn_parameters[key]
            assert getattr(model, name).requires_grad
            assert manager.nn_parameters[key].grad is not None

    def test_nonlearnable_specs_remain_typed_nonlearnable_module_state(self):
        manager = self._manager(dtype="float64", requires_grad=False)
        model = FarquharPhotosynthesis(parameter_manager=manager)
        for name in self.CAPACITIES:
            key = f"biosphere.photosynthesis.{name}"
            value = getattr(model, name)
            assert key not in manager.nn_parameters
            assert model.manager_binding_key(value) is None
            assert value.dtype == torch.float64
            assert not value.requires_grad
            assert name in dict(model.named_buffers())
            assert name not in dict(model.named_parameters())
            assert name in model.state_dict()

    def test_runtime_manager_dtype_is_preserved_even_when_spec_is_float32(self):
        manager = self._manager()
        manager.to("cpu", dtype=torch.float64)
        model = FarquharPhotosynthesis(parameter_manager=manager)
        for name in self.CAPACITIES:
            key = f"biosphere.photosynthesis.{name}"
            assert manager.get_parameter(key).dtype == "float32"
            assert getattr(model, name) is manager.nn_parameters[key]
            assert getattr(model, name).dtype == torch.float64
        _value(model(self._state(dtype=torch.float64), 0.0, 1.0)["net_photosynthesis"]).sum().backward()
        for name in self.CAPACITIES:
            assert getattr(model, name).grad.dtype == torch.float64

    def test_numeric_keywords_override_config_without_mutating_the_manager(self):
        manager = self._manager()
        before = {name: manager.nn_parameters[f"biosphere.photosynthesis.{name}"].detach().clone()
                  for name in self.CAPACITIES}
        config = {"parameters": {"vcmax25": 90.0, "jmax25": 180.0, "rd25": 1.5}}
        model = FarquharPhotosynthesis(
            parameter_manager=manager, config=config, vcmax25=50.0, jmax25=100.0, rd25=1.0,
        )
        assert config["parameters"] == {"vcmax25": 90.0, "jmax25": 180.0, "rd25": 1.5}
        for name, value in {"vcmax25": 50.0, "jmax25": 100.0, "rd25": 1.0}.items():
            assert getattr(model, name).item() == pytest.approx(value)
            assert model.get_parameter_value(
                name, namespace="biosphere.photosynthesis", source_order=("config", "manager"), required=True,
            ).item() == pytest.approx(value)
            assert model.manager_binding_key(getattr(model, name)) is None
            assert name in dict(model.named_buffers())
            torch.testing.assert_close(manager.nn_parameters[f"biosphere.photosynthesis.{name}"], before[name])

    def test_config_capacity_unit_identity_uses_the_existing_getter(self):
        manager = self._manager()
        model = FarquharPhotosynthesis(parameter_manager=manager, config={"parameters": {
            "vcmax25": {"value": 50.0, "units": "umol CO2 m-2 s-1"},
            "jmax25": {"value": 100.0, "units": "umol e- m-2 s-1"},
            "rd25": {"value": 1.0, "units": "umol CO2 m-2 s-1"},
        }})
        for name, value in {"vcmax25": 50.0, "jmax25": 100.0, "rd25": 1.0}.items():
            assert getattr(model, name).item() == pytest.approx(value)
            assert model.manager_binding_key(getattr(model, name)) is None

    def test_unsupported_config_capacity_conversion_is_not_silently_ignored(self):
        manager = self._manager()
        with pytest.warns(UserWarning, match="Performing conversion"), pytest.raises(ValueError, match="Unit conversion failed"):
            FarquharPhotosynthesis(parameter_manager=manager, config={"parameters": {
                "vcmax25": {"value": 0.05, "units": "mmol CO2 m-2 s-1"},
            }})

    def test_private_config_tensor_keeps_its_graph_dtype_and_buffer_checkpoint(self):
        manager = self._manager(dtype="float64")
        source = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
        candidate = 2.0 * source
        model = FarquharPhotosynthesis(parameter_manager=manager, config={"parameters": {"vcmax25": candidate}})
        assert model.vcmax25 is candidate
        assert dict(model.named_buffers())["vcmax25"] is candidate
        assert model.manager_binding_key(candidate) is None
        assert candidate.grad_fn is not None
        assert "vcmax25" in model.state_dict()
        _value(model(self._state(dtype=torch.float64), 0.0, 1.0)["net_photosynthesis"]).sum().backward()
        assert source.grad is not None and source.grad.item() > 0.0
        assert manager.nn_parameters["biosphere.photosynthesis.vcmax25"].grad is None

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_explicit_private_parameter_keeps_its_identity_and_gradient_flag(self, requires_grad):
        manager = self._manager()
        candidate = torch.nn.Parameter(torch.tensor(40.0, dtype=torch.float32), requires_grad=requires_grad)
        model = FarquharPhotosynthesis(parameter_manager=manager, config={"parameters": {"vcmax25": candidate}})
        assert model.vcmax25 is candidate
        assert dict(model.named_parameters())["vcmax25"] is candidate
        assert candidate.requires_grad is requires_grad
        assert model.manager_binding_key(candidate) is None
        assert candidate.data_ptr() != manager.nn_parameters["biosphere.photosynthesis.vcmax25"].data_ptr()

    def test_private_parameter_dtype_conversion_keeps_the_input_gradient_graph(self):
        manager = self._manager()
        candidate = torch.nn.Parameter(torch.tensor(40.0, dtype=torch.float64))
        model = FarquharPhotosynthesis(parameter_manager=manager, config={"parameters": {"vcmax25": candidate}})
        assert model.vcmax25.dtype == torch.float32
        assert model.vcmax25.grad_fn is not None
        assert dict(model.named_buffers())["vcmax25"] is model.vcmax25
        assert model.manager_binding_key(model.vcmax25) is None
        _value(model(self._state(), 0.0, 1.0)["net_photosynthesis"]).sum().backward()
        assert candidate.grad is not None and candidate.grad.item() > 0.0

    @pytest.mark.parametrize("base_source", ["manager", "config", "private_parameter"])
    @pytest.mark.parametrize("pft_requires_grad", [False, True])
    def test_pft_rebinds_its_sources_without_mutating_base_capacities(self, pft_requires_grad, base_source):
        from copy import deepcopy

        manager = self._manager()
        before = {name: manager.nn_parameters[f"biosphere.photosynthesis.{name}"].detach().clone()
                  for name in self.CAPACITIES}
        pft_values = {"vcmax25": 45.0, "jmax25": 90.0, "rd25": 0.9}
        for name, value in pft_values.items():
            spec = deepcopy(manager.get_parameter(f"biosphere.photosynthesis.{name}"))
            spec.key = f"biosphere.pft.binding_test.photosynthesis.{name}"
            spec.value = spec.default = value
            spec.dtype = "float64"
            spec.requires_grad = pft_requires_grad
            manager.register_parameter(spec)
        config = {}
        if base_source == "config":
            config = {"parameters": dict(self.CAPACITIES)}
        elif base_source == "private_parameter":
            config = {"parameters": {name: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
                                     for name, value in self.CAPACITIES.items()}}
        model = FarquharPhotosynthesis(parameter_manager=manager, pft_name="binding_test", config=config)
        for name, value in pft_values.items():
            base_key = f"biosphere.photosynthesis.{name}"
            pft_key = f"biosphere.pft.binding_test.photosynthesis.{name}"
            torch.testing.assert_close(manager.nn_parameters[base_key], before[name])
            assert getattr(model, name).item() == pytest.approx(value)
            assert getattr(model, name).dtype == torch.float64
            assert getattr(model, name).requires_grad is pft_requires_grad
            if pft_requires_grad:
                assert getattr(model, name) is manager.nn_parameters[pft_key]
                assert name not in model.state_dict()
            else:
                assert name in dict(model.named_buffers())
                assert model.manager_binding_key(getattr(model, name)) is None

    def test_manager_held_capacity_uses_the_existing_manager_sensitivity_path(self):
        manager = self._manager()
        model = FarquharPhotosynthesis(parameter_manager=manager)
        with pytest.raises(TypeError, match="manager calibration path"):
            model.parameter_sensitivity("vcmax25", self._state(), time=0.0, dt=1.0,
                                        target_output="net_photosynthesis")
        for name in self.CAPACITIES:
            key = f"biosphere.photosynthesis.{name}"
            assert manager.get_trainable_parameters()[key] is getattr(model, name)

    @pytest.mark.parametrize("source_kind", ["manager", "config"])
    def test_format3_checkpoint_restores_actual_capacity_owner_and_output(self, tmp_path, source_kind):
        from relic.core.simulator import RELICSimulator
        from relic.core.simulator.checkpoint import _decode_checkpoint_value

        manager = self._manager()
        config = {"parameters": {"vcmax25": 40.0}} if source_kind == "config" else {}
        model = FarquharPhotosynthesis(name="binding_checkpoint", parameter_manager=manager, config=config)
        model.set_dependencies([])
        state = self._state()
        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False,
                                    "strict_dependencies": False, "adaptive_dt_enabled": False})
        simulator.initialize_state(state)
        simulator.register_functor(model, execution_group="fast")
        expected = _value(model(state, 0.0, 1.0)["net_photosynthesis"]).detach().clone()
        identities = {name: getattr(model, name) for name in self.CAPACITIES}
        saved_values = {name: value.detach().clone() for name, value in identities.items()}
        path = tmp_path / f"farquhar_{source_kind}.pt"
        receipt = simulator.create_checkpoint(path)
        assert receipt["format_version"] == 3
        entry = torch.load(path, map_location="cpu", weights_only=True)["functors"][model.name]
        bindings = _decode_checkpoint_value(entry["parameter_bindings"], path="capacity_bindings")
        local = _decode_checkpoint_value(entry["state_dict"], path="capacity_module_state")
        for name in self.CAPACITIES:
            key = f"biosphere.photosynthesis.{name}"
            if source_kind == "config" and name == "vcmax25":
                assert name in local and key not in bindings
            else:
                assert key in bindings and name not in local
        with torch.no_grad():
            for name in self.CAPACITIES:
                getattr(model, name).add_(0.5)
        assert not torch.equal(_value(model(state, 0.0, 1.0)["net_photosynthesis"]), expected)
        simulator.restore_checkpoint(path)
        for name in self.CAPACITIES:
            assert getattr(model, name) is identities[name]
            torch.testing.assert_close(getattr(model, name), saved_values[name])
            if source_kind == "manager" or name != "vcmax25":
                assert getattr(model, name) is manager.nn_parameters[f"biosphere.photosynthesis.{name}"]
        torch.testing.assert_close(_value(model(state, 0.0, 1.0)["net_photosynthesis"]), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

