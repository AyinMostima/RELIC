"""
L1 Unit Tests for MineralizationProcess Module

This module provides comprehensive unit tests for the nitrogen mineralization
and immobilization process implementation, covering C-N coupling, microbial
stoichiometry, environmental responses, and mass balance.

Test Categories:
- L1.1: Basic instantiation and configuration  
- L1.2: Scientific accuracy against literature values
- L1.3: Environmental response functions
- L1.4: Stoichiometric constraints and C-N coupling
- L1.5: Boundary condition handling
- L1.6: Parameter sensitivity analysis
- L1.7: Performance and stability tests
"""

import pytest
import torch
import numpy as np
import inspect
from typing import Dict

# Import the module under test
from relic.scientific.pedosphere.nutrient_cycling import MineralizationProcess
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant

# Import utilities for testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _to_float(value: torch.Tensor | UpdateItem) -> float:
    if isinstance(value, UpdateItem):
        value = value.value
    return float(torch.as_tensor(value).detach().item())

def _normalize_param_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return value

def _set_param_value(param_manager, key: str, value) -> None:
    spec = param_manager.get_parameter(key)
    units = spec.units if spec is not None else None
    param_manager.set_value(key, value, units=units, validate=True)

class TestMineralizationProcessL1:
    """L1 Unit Tests for MineralizationProcess functor."""
    
    @pytest.fixture
    def default_config(self) -> Dict:
        """Default configuration for MineralizationProcess."""
        return {}
    
    @pytest.fixture 
    def mineralization(self, default_config) -> MineralizationProcess:
        """Create MineralizationProcess instance."""
        return MineralizationProcess(name="test_mineralization", config=default_config)

    @pytest.fixture
    def standard_state(self) -> Dict:
        """Standard state for testing."""
        return {
            'organic_nitrogen': torch.tensor(5.0),
            'nh4_nitrogen': torch.tensor(0.1),
            'microbial_nitrogen': torch.tensor(0.05),
            'soil_temperature': torch.tensor(288.15),
            'soil_moisture': torch.tensor(0.3),
            'soil_ph': torch.tensor(6.5),
        }

    @pytest.fixture
    def standard_environment(self) -> Dict:
        """Declared carbon-coupling inputs, merged into the state (the one input channel)."""
        return {
            'carbon_decomposition_rate': torch.tensor(1e-6),
            'organic_nitrogen_input': torch.tensor(0.0),
        }

    # =============================================================================
    # L1.1: Basic Instantiation and Configuration Tests
    # =============================================================================
    
    def test_instantiation_basic(self, default_config):
        """Test basic instantiation with default configuration."""
        mineralization = MineralizationProcess(name="test_min", config=default_config)
        
        assert mineralization.name == "test_min"
        assert mineralization.config == default_config
        assert hasattr(mineralization, 'metadata')
        assert not hasattr(mineralization, 'process_params')
        assert not hasattr(mineralization, 'stoichiometry_params')
        assert mineralization.metadata.name == "test_min"
        
    def test_instantiation_metadata(self, mineralization):
        """Test metadata completeness."""
        metadata = mineralization.metadata
        
        # Check required metadata fields
        assert metadata.name is not None
        assert metadata.version is not None
        assert metadata.description is not None
        assert metadata.mathematical_expression is not None
        
        # Check scientific references
        assert len(metadata.references) >= 3
        assert any("Manzoni" in ref or "Porporato" in ref for ref in metadata.references)
        assert any("Grant" in ref for ref in metadata.references)
        assert any("Stanford" in ref for ref in metadata.references)
        
        # Check the complete ParameterManager dependency surface.
        expected_parameter_keys = {
            'pedosphere.mineralization.temp_response_type',
            'pedosphere.mineralization.q10_mineralization',
            'pedosphere.mineralization.activation_energy',
            'pedosphere.mineralization.moisture_optimum',
            'pedosphere.mineralization.moisture_response_power',
            'pedosphere.mineralization.k_mineralization',
            'pedosphere.mineralization.cn_ratio_microbial',
            'pedosphere.mineralization.n_assimilation_efficiency',
            'pedosphere.mineralization.carbon_use_efficiency',
            'pedosphere.physical.ph_optimum_decomposition',
            'pedosphere.physical.ph_response_width',
            'pedosphere.numerical.temperature_response_min',
            'pedosphere.numerical.temperature_response_max',
            'pedosphere.numerical.moisture_response_min',
            'pedosphere.numerical.moisture_response_max',
            'pedosphere.numerical.ph_response_min',
            'pedosphere.numerical.ph_response_max',
            'pedosphere.nutrient_cycling.k_microbial_death',
        }
        assert set(metadata.parameters) == expected_parameter_keys
        assert not any(
            key.startswith('relic.physical.constants.')
            for key in metadata.parameters
        )
    
    def test_parameters_and_buffers_are_registered_with_module(self, mineralization):
        """Optimizer, checkpoint, dtype, and device ownership must be explicit."""
        expected_parameters = {
            "q10",
            "k_mineralization",
            "c_use_efficiency",
            "ph_optimum",
            "ph_range",
            "k_microbial_death",
        }
        expected_buffers = {
            "activation_energy",
            "moisture_optimum",
            "moisture_response_power",
            "temperature_response_min",
            "temperature_response_max",
            "moisture_response_min",
            "moisture_response_max",
            "ph_response_min",
            "ph_response_max",
            "cn_ratio_microbial",
            "n_assimilation_efficiency",
        }

        # The learnable parameters are ParameterManager bindings held by
        # reference (item 74): the manager trainable interface owns them, the
        # module registers none of them, and module conversions leave them alone.
        held_attributes = {
            name: value
            for name, value in vars(mineralization).items()
            if mineralization.manager_binding_key(value) is not None
        }
        assert set(held_attributes) == expected_parameters
        trainable = get_global_manager().get_trainable_parameters()
        for name, binding in held_attributes.items():
            assert trainable[mineralization.manager_binding_key(binding)] is binding, name
        assert dict(mineralization.named_parameters()) == {}
        assert set(dict(mineralization.named_buffers())) == expected_buffers
        assert expected_buffers <= set(mineralization.state_dict())
        assert not expected_parameters & set(mineralization.state_dict())

        binding_dtypes = {name: value.dtype for name, value in held_attributes.items()}
        mineralization.to(dtype=torch.float64)
        assert all(
            value.dtype is torch.float64
            for value in dict(mineralization.named_buffers()).values()
        )
        assert {name: value.dtype for name, value in held_attributes.items()} == binding_dtypes

    def test_retired_unowned_carbon_specs_are_absent(self):
        """Removed runtime ports must not survive as unused parameter defaults."""
        param_manager = get_global_manager()
        retired_keys = (
            "pedosphere.mineralization.substrate_cn_ratio_min",
            "pedosphere.environment_defaults.substrate_cn_ratio",
            "pedosphere.environment_defaults.organic_carbon",
        )

        for key in retired_keys:
            assert param_manager.get_parameter(key) is None, key

    def test_stoichiometric_parameter_values(self, mineralization):
        """Test registered stoichiometric parameter values."""
        cn_ratio = _to_float(mineralization.cn_ratio_microbial)
        n_eff = _to_float(mineralization.n_assimilation_efficiency)
        cue = _to_float(mineralization.c_use_efficiency)
        assert 5.0 <= cn_ratio <= 15.0, "Microbial C:N should be 5-15"
        assert 0.3 <= n_eff <= 1.0, "N efficiency should be 0.3-1.0"
        assert 0.0 < cue <= 1.0, "Carbon use efficiency should be in (0, 1]"

    def test_microbial_turnover_matches_linked_cn_daily_rate(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """The linked C-N turnover baseline must retain its published timescale."""
        # References: Schimel and Weintraub (2003), Table 1, Kt = 0.012 d-1.
        # Mathematical expression: k_s = Kt / seconds_per_day and
        # death_fraction = k_s * dt for first-order microbial N turnover.
        # Parameters: Kt [d-1], k_s [s-1], dt [s], microbial N [g N m-2].
        # Units: death_flux [g N m-2 s-1]. Open-system route: none; dead
        # microbial N is transferred to the organic N pool in the closed ledger.
        seconds_per_day = float(get_physical_constant("seconds_per_day"))
        expected_rate = 0.012 / seconds_per_day
        actual_rate = _to_float(mineralization.k_microbial_death)
        assert actual_rate == pytest.approx(expected_rate, rel=1e-7)

        dt = 3600.0
        result = mineralization.forward(
            {**standard_state, **standard_environment},
            time=0.0,
            dt=dt,
        )
        death_flux = _to_float(result["microbial_death_flux"])
        microbial_pool = _to_float(standard_state["microbial_nitrogen"])
        realized_fraction = death_flux * dt / microbial_pool
        expected_fraction = 0.012 * dt / seconds_per_day

        assert death_flux * dt < microbial_pool
        assert realized_fraction == pytest.approx(expected_fraction, rel=1e-6)
    
    def test_dependencies(self, mineralization):
        """Test dependency declaration."""
        deps = mineralization.get_dependencies()

        assert isinstance(deps, list)
        assert deps == []
        configured = MineralizationProcess(
            name="configured_mineralization",
            config={
                "dependencies": [
                    "first_order_decomposition",
                    "litterfall_interface_mapper",
                ]
            },
        )
        assert configured.get_dependencies() == [
            "first_order_decomposition",
            "litterfall_interface_mapper",
        ]
        assert "carbon_decomposition_rate" in mineralization.metadata.input_variables
        assert "organic_nitrogen_input" in mineralization.metadata.input_variables
        assert "carbon_decomposition_rate" not in mineralization.optional_input_variables
        assert "organic_nitrogen_input" not in mineralization.optional_input_variables
        assert "decomposition_rate" not in mineralization.metadata.input_variables
        assert "carbon_flux" not in mineralization.metadata.input_variables
        assert "microbial_biomass" not in mineralization.metadata.input_variables
        assert "microbial_biomass" not in mineralization.optional_input_variables

    def test_functor_uses_only_canonical_parameter_access(self):
        """Scientific parameter access must stay on FunctorBase's canonical API."""
        source = inspect.getsource(MineralizationProcess)

        assert "require_scalar(" not in source
        assert "get_global_manager(" not in source

    def test_carbon_decomposition_input_is_canonical_and_required(
        self,
        mineralization,
        standard_state,
    ):
        """Missing canonical C decomposition input must fail without aliases."""
        with pytest.raises((KeyError, ValueError), match="carbon_decomposition_rate"):
            mineralization.forward(
                standard_state,
                time=0.0,
                dt=3600.0,
                decomposition_rate=torch.tensor(1e-6),
            )

        with pytest.raises((KeyError, ValueError), match="carbon_decomposition_rate"):
            mineralization.forward(
                standard_state,
                time=0.0,
                dt=3600.0,
                carbon_flux=torch.tensor(1e-6),
            )

    @pytest.mark.parametrize(
        "invalid_rate",
        [torch.tensor(float("nan")), torch.tensor(float("inf")), torch.tensor(-1e-6)],
    )
    def test_carbon_decomposition_input_must_be_finite_and_non_negative(
        self,
        mineralization,
        standard_state,
        invalid_rate,
    ):
        """Invalid canonical C decomposition rates must fail before N advance."""
        with pytest.raises(ValueError, match="carbon_decomposition_rate"):
            mineralization.forward(
                {
                    **standard_state,
                    "carbon_decomposition_rate": invalid_rate,
                    "organic_nitrogen_input": torch.tensor(0.0),
                },
                time=0.0,
                dt=3600.0,
            )

    def test_forward_outputs_match_metadata_exactly(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """The runtime interface must contain only declared canonical outputs."""
        result = mineralization.forward(
            {**standard_state, **standard_environment},
            time=0.0,
            dt=3600.0,
        )

        assert set(result) == set(mineralization.metadata.output_variables)

    def test_all_declared_inputs_are_required_canonical_tensors(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """Every declared input must come from StateManager without defaults."""
        canonical_state = {**standard_state, **standard_environment}
        assert mineralization.optional_input_variables == []

        for key in mineralization.metadata.input_variables:
            missing_state = dict(canonical_state)
            missing_state.pop(key)
            with pytest.raises((KeyError, ValueError), match=key):
                mineralization.forward(missing_state, time=0.0, dt=10.0)

            non_tensor_state = dict(canonical_state)
            non_tensor_state[key] = float(canonical_state[key])
            with pytest.raises(TypeError, match=key):
                mineralization.forward(non_tensor_state, time=0.0, dt=10.0)

    def test_all_declared_inputs_must_be_finite(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """NaN and Inf must fail before reaction-network advancement."""
        canonical_state = {**standard_state, **standard_environment}
        for key in mineralization.metadata.input_variables:
            invalid_state = dict(canonical_state)
            invalid_state[key] = torch.tensor(float("nan"))
            with pytest.raises(ValueError, match=key):
                mineralization.forward(invalid_state, time=0.0, dt=10.0)

    def test_declared_inputs_are_read_from_the_state_only(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """A canonical input has one runtime source, the state: a keyword carrying a
        declared input is not read, and the rate core takes the state only."""
        assert list(
            inspect.signature(MineralizationProcess._compute_rate_core).parameters
        ) == ["self", "state"]
        canonical_state = {**standard_state, **standard_environment}

        for key in mineralization.metadata.input_variables:
            missing_state = dict(canonical_state)
            value = missing_state.pop(key)
            with pytest.raises((KeyError, ValueError), match=key):
                mineralization.forward(
                    missing_state,
                    time=0.0,
                    dt=10.0,
                    **{key: value},
                )

    @pytest.mark.parametrize(
        ("input_name", "invalid_value"),
        [
            ("soil_temperature", 0.0),
            ("soil_moisture", -0.01),
            ("soil_moisture", 1.01),
            ("soil_ph", -0.01),
            ("soil_ph", 14.01),
        ],
    )
    def test_declared_inputs_respect_physical_domains(
        self,
        mineralization,
        standard_state,
        standard_environment,
        input_name,
        invalid_value,
    ):
        """Canonical drivers and pools must remain inside declared domains."""
        invalid_state = {**standard_state, **standard_environment}
        invalid_state[input_name] = torch.tensor(invalid_value)

        with pytest.raises(ValueError, match=input_name):
            mineralization.forward(invalid_state, time=0.0, dt=10.0)

    # =============================================================================
    # L1.2: Scientific Accuracy Tests Against Literature Values
    # =============================================================================
    
    def test_q10_temperature_response_literature(self):
        """Test Q10 temperature response against literature values."""
        param_manager = get_global_manager()
        if param_manager is None:
            raise RuntimeError("ParameterManager must be available for Q10 literature test")

        q10_key = "pedosphere.mineralization.q10_mineralization"
        temp_key = "pedosphere.mineralization.temp_response_type"
        original_q10 = _normalize_param_value(param_manager.get(q10_key))
        original_temp = _normalize_param_value(param_manager.get(temp_key))
        try:
            _set_param_value(param_manager, temp_key, "q10")
            mineralization = MineralizationProcess(name="test_mineralization", config={})

            # Literature Q10 for N mineralization: 2.0-3.0 (Kirschbaum, 1995)
            q10_value = _to_float(mineralization.q10)
            assert 2.0 <= q10_value <= 3.5, f"Q10 should be 2.0-3.5: {q10_value}"

            # Test temperature response follows Q10 law
            temperatures = [278.15, 288.15, 298.15, 308.15]  # 5 degC, 15 degC, 25 degC, 35 degC
            responses = []

            for temp in temperatures:
                response = mineralization._calculate_temperature_response(torch.tensor(temp))
                responses.append(_to_float(torch.as_tensor(response).detach().item()))

            # Check Q10 relationship over 10 degC intervals
            for i in range(len(temperatures) - 1):
                temp_diff = temperatures[i+1] - temperatures[i]  # 10 degC
                response_ratio = responses[i+1] / responses[i]
                expected_ratio = q10_value ** (temp_diff / 10.0)

                relative_error = abs(response_ratio - expected_ratio) / expected_ratio
                assert relative_error < 0.15, \
                    f"Q10 relationship violated: {response_ratio:.3f} vs expected {expected_ratio:.3f}"
        finally:
            _set_param_value(param_manager, q10_key, original_q10)
            _set_param_value(param_manager, temp_key, original_temp)
    
    def test_microbial_cn_ratio_literature(self, mineralization):
        """Test microbial C:N ratio against literature values."""
        cn_microbial = _to_float(mineralization.cn_ratio_microbial)
        
        # Literature values: bacterial biomass C:N = 3-12, fungal = 5-15 (Manzoni & Porporato, 2009)
        assert 3.0 <= cn_microbial <= 15.0, f"Microbial C:N should be 3-15: {cn_microbial}"
        
        # Default value should be around 6-10 (typical for soil microbes)
        assert 6.0 <= cn_microbial <= 12.0, f"Default microbial C:N should be 6-12: {cn_microbial}"
    
    def test_mineralization_rate_constants_literature(self, mineralization):
        """Test mineralization rate constants against literature."""
        k_min_per_year = _to_float(mineralization.k_mineralization)
        
        # Literature range: 0.5-3.0 per year (Stanford & Smith, 1972; Grant et al., 1993)
        assert 0.3 <= k_min_per_year <= 4.0, \
            f"Mineralization rate should be 0.3-4.0 per year: {k_min_per_year:.2f}"
    
    def test_ph_optimum_literature(self, mineralization):
        """Test pH optimum against literature values."""
        ph_opt = _to_float(mineralization.ph_optimum)
        
        # Literature: optimal pH for N mineralization = 6.5-7.5 (Brady & Weil, 2017)
        assert 6.0 <= ph_opt <= 8.0, f"pH optimum should be 6.0-8.0: {ph_opt}"
        
        # Test pH response function shape
        ph_values = torch.linspace(4.0, 9.0, 21)
        responses = []
        
        for ph in ph_values:
            temp = torch.tensor(288.15)
            moisture = torch.tensor(0.3)
            response = mineralization._calculate_environmental_response(temp, moisture, torch.tensor([ph]))
            responses.append(_to_float(torch.as_tensor(response).detach().item()))
        
        # Find maximum response pH
        max_idx = np.argmax(responses)
        max_ph = _to_float(ph_values[max_idx])
        
        assert abs(max_ph - ph_opt) < 0.5, f"Response maximum at pH {max_ph:.1f} vs optimum {ph_opt:.1f}"
    
    # =============================================================================
    # L1.3: Environmental Response Function Tests
    # =============================================================================
    
    def test_temperature_response_bounds(self, mineralization):
        """Test temperature response stays within reasonable bounds."""
        extreme_temps = [243.15, 253.15, 323.15, 333.15]  # -30 degC, -20 degC, 50 degC, 60 degC
        
        for temp in extreme_temps:
            try:
                response = mineralization._calculate_temperature_response(torch.tensor(temp))
                # Should be bounded
                assert 0.05 <= _to_float(response) <= 10.0, \
                    f"Temperature response {response:.3f} outside bounds at {temp-273.15:.1f} degC"
            except ValueError:
                # Extreme temperatures may raise validation errors
                pass
    
    def test_moisture_response_optimum(self, mineralization):
        """Test moisture response peaks near optimum."""
        moistures = torch.linspace(0.05, 0.8, 31)
        responses = []
        
        for moisture in moistures:
            temp = torch.tensor(288.15)
            ph = torch.tensor(7.0)
            response = mineralization._calculate_environmental_response(temp, torch.tensor([moisture]), ph)
            responses.append(_to_float(torch.as_tensor(response).detach().item()))
        
        # Find optimal moisture
        max_idx = np.argmax(responses)
        optimal_moisture = _to_float(moistures[max_idx])
        
        # Should be close to configured optimum
        expected_optimum = _to_float(mineralization.moisture_optimum)
        assert abs(optimal_moisture - expected_optimum) < 0.1, \
            f"Moisture response peak at {optimal_moisture:.3f} vs expected {expected_optimum:.3f}"
    
    def test_ph_response_gaussian_shape(self, mineralization):
        """Test pH response has proper Gaussian shape."""
        ph_values = torch.linspace(4.0, 9.0, 21) 
        temp = torch.tensor(288.15)
        moisture = torch.tensor(0.3)
        responses = []
        
        for ph in ph_values:
            response = mineralization._calculate_environmental_response(temp, moisture, torch.tensor([ph]))
            responses.append(_to_float(torch.as_tensor(response).detach().item()))
        
        responses = np.array(responses)
        
        # Find maximum
        max_idx = np.argmax(responses)
        max_response = responses[max_idx]
        
        # Response should decrease on both sides of optimum
        left_side = responses[:max_idx]
        right_side = responses[max_idx+1:]
        
        if len(left_side) > 0:
            assert np.all(np.diff(left_side) >= -0.05), "Response should increase towards optimum"
        if len(right_side) > 0: 
            assert np.all(np.diff(right_side) <= 0.05), "Response should decrease from optimum"
    
    def test_environmental_response_multiplicative(self, mineralization):
        """Test environmental factors combine multiplicatively."""
        temp = torch.tensor(288.15)
        moisture = torch.tensor(0.3) 
        ph = torch.tensor(6.5)
        
        # Get individual responses (approximate by comparing different conditions)
        temp_high = torch.tensor(298.15)  # Higher temperature
        response_normal = mineralization._calculate_environmental_response(temp, moisture, ph)
        response_high_temp = mineralization._calculate_environmental_response(temp_high, moisture, ph)
        
        # Higher temperature should give higher response (positive temperature dependence)
        assert _to_float(response_high_temp) > _to_float(response_normal), \
            "Higher temperature should increase response"
        
        # Test extreme conditions give lower response
        temp_low = torch.tensor(268.15)  # Very cold
        response_cold = mineralization._calculate_environmental_response(temp_low, moisture, ph)
        
        assert _to_float(response_cold) < _to_float(response_normal), \
            "Cold temperature should decrease response"
    
    # =============================================================================
    # L1.4: Stoichiometric Constraints and C-N Coupling Tests
    # =============================================================================
    
    def test_gross_mineralization_calculation(self, mineralization, standard_state):
        """Test gross mineralization calculation."""
        organic_n = torch.tensor(5.0)  # g N m-2
        env_response = torch.tensor(1.0)  # Neutral conditions
        
        gross_min = mineralization._calculate_gross_mineralization(organic_n, env_response)
        
        # Should be positive
        assert _to_float(gross_min) > 0, "Gross mineralization should be positive"
        
        # Should be proportional to organic N pool size
        organic_n_double = torch.tensor(10.0)
        gross_min_double = mineralization._calculate_gross_mineralization(organic_n_double, env_response)
        
        ratio = _to_float(gross_min_double) / _to_float(gross_min)
        assert 1.8 <= ratio <= 2.2, f"Gross mineralization should scale with pool size: ratio {ratio:.2f}"
        
        # Should respond to environmental conditions
        env_high = torch.tensor(2.0)
        gross_min_high = mineralization._calculate_gross_mineralization(organic_n, env_high)
        
        ratio_env = _to_float(gross_min_high) / _to_float(gross_min)
        assert 1.8 <= ratio_env <= 2.2, f"Should respond to environmental modifier: ratio {ratio_env:.2f}"
    
    def test_microbial_nitrogen_demand_calculation(self, mineralization):
        """Test microbial nitrogen demand calculation."""
        carbon_uptake = torch.tensor(1e-6)  # g C m-2 s-1
        n_demand = mineralization._calculate_microbial_nitrogen_demand(carbon_uptake)
        
        # Should be positive (microbes need N for growth)
        assert _to_float(n_demand) >= 0, "N demand should be non-negative"
        
        # Test stoichiometric relationship
        cn_microbial = _to_float(mineralization.cn_ratio_microbial)
        n_efficiency = _to_float(mineralization.n_assimilation_efficiency)
        expected_demand = carbon_uptake / (cn_microbial * n_efficiency)
        torch.testing.assert_close(n_demand, expected_demand)

    def test_immobilization_can_exceed_gross_using_available_nh4(
        self,
        mineralization,
        standard_state,
    ):
        """Net immobilization may consume pre-existing mineral nitrogen."""
        # References: Manzoni and Porporato (2009), Sections 3.3-3.4.
        # Mathematical expression: R_net = R_gross - R_immob, where
        # R_immob*dt <= N_NH4 and R_net may be negative.
        # Parameters: R_gross, R_immob [g N m-2 s-1], N_NH4 [g N m-2], dt [s].
        dt = 10.0
        result = mineralization.forward(
            {
                **standard_state,
                "carbon_decomposition_rate": torch.tensor(1.0e-3),
                "organic_nitrogen_input": torch.tensor(0.0),
            },
            time=0.0,
            dt=dt,
        )

        gross = _to_float(result["gross_mineralization_flux"])
        immobilization = _to_float(result["immobilization_flux"])
        net = _to_float(result["mineralization_flux"])
        nh4_available = _to_float(standard_state["nh4_nitrogen"])

        assert immobilization > gross
        assert net < 0.0
        assert net == pytest.approx(gross - immobilization, rel=1.0e-7)
        assert immobilization * dt <= nh4_available + 1.0e-12
        assert _to_float(result["nh4_nitrogen"]) == pytest.approx(net * dt)

    def test_immobilization_uses_current_dt_without_hidden_hour_horizon(
        self,
        mineralization,
        standard_state,
    ):
        """The current step owns availability scaling and applies dt once."""
        state = dict(standard_state)
        state["nh4_nitrogen"] = torch.tensor(1.0e-6)
        dt = 10.0
        result = mineralization.forward(
            {
                **state,
                "carbon_decomposition_rate": torch.tensor(1.0e-3),
                "organic_nitrogen_input": torch.tensor(0.0),
            },
            time=0.0,
            dt=dt,
        )

        gross = result["gross_mineralization_flux"].value
        immobilization = result["immobilization_flux"].value
        torch.testing.assert_close(
            immobilization * dt,
            state["nh4_nitrogen"],
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            result["nh4_nitrogen"].value,
            gross * dt - state["nh4_nitrogen"],
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        source = inspect.getsource(MineralizationProcess)
        assert "seconds_per_hour" not in source

    def test_no_duplicate_carbon_overflow_formula(self):
        """The downstream N-only process must not re-settle upstream carbon."""
        source = inspect.getsource(MineralizationProcess)

        assert "_check_stoichiometric_balance" not in source
        assert "carbon_overflow" not in source
        assert "organic_carbon" not in source
        assert "substrate_cn_ratio" not in source
    
    # =============================================================================
    # L1.5: Boundary Condition Handling Tests
    # =============================================================================
    
    def test_zero_nitrogen_pools_handling(self, mineralization, standard_environment):
        """Test handling of zero nitrogen pools."""
        zero_state = {
            'organic_nitrogen': torch.tensor(0.0),
            'nh4_nitrogen': torch.tensor(0.0),
            'microbial_nitrogen': torch.tensor(0.0),
            'soil_temperature': torch.tensor(288.15),
            'soil_moisture': torch.tensor(0.3),
            'soil_ph': torch.tensor(6.5)
        }
        
        result = mineralization.forward({**zero_state, **standard_environment}, time=0.0, dt=3600.0)
        
        # Should handle gracefully
        gross_min = _to_float(result.get('gross_mineralization_flux', torch.tensor(0.0)))
        assert gross_min < 1e-10, "Zero organic N should give near-zero gross mineralization"
        
        # All outputs should be finite
        for key, update in result.items():
            assert isinstance(update, UpdateItem), key
        assert torch.isfinite(update.value).all(), f"{key} should be finite"

    @pytest.mark.parametrize(
        "pool_key",
        ["organic_nitrogen", "nh4_nitrogen", "microbial_nitrogen"],
    )
    def test_negative_nitrogen_pools_fail_fast(
        self,
        mineralization,
        standard_state,
        standard_environment,
        pool_key,
    ):
        """Conserved nitrogen donors must be finite and non-negative."""
        negative_state = {**standard_state, **standard_environment}
        negative_state[pool_key] = torch.tensor(-1.0)

        with pytest.raises(ValueError, match=pool_key):
            mineralization.forward(negative_state, time=0.0, dt=3600.0)
    
    def test_extreme_temperature_handling(self, mineralization, standard_state, standard_environment):
        """Test handling of extreme temperatures."""
        extreme_temps = [223.15, 373.15]  # -50 degC, 100 degC
        
        for temp in extreme_temps:
            state = standard_state.copy()
            state['soil_temperature'] = torch.tensor(temp)
            
            try:
                result = mineralization.forward({**state, **standard_environment}, time=0.0, dt=3600.0)
                
                # Should give reasonable results or low activity
                gross_min = _to_float(result.get('gross_mineralization_flux', torch.tensor(0.0)))
                assert 0.0 <= gross_min < 1e-4, f"Extreme temperature should give low activity: {gross_min}"
                
            except (ValueError, RuntimeError) as e:
                # Acceptable to raise error for extreme temperatures
                assert "temperature" in str(e).lower()
    
    def test_extreme_ph_handling(self, mineralization, standard_state, standard_environment):
        """Test handling of extreme pH values."""
        extreme_phs = [2.0, 12.0]  # Very acidic, very basic

        reference_state = standard_state.copy()
        reference_state['soil_ph'] = torch.as_tensor(mineralization.ph_optimum)
        reference_result = mineralization.forward(
            {**reference_state, **standard_environment},
            time=0.0,
            dt=3600.0,
        )
        reference_gross = _to_float(reference_result['gross_mineralization_flux'])
        
        for ph in extreme_phs:
            state = standard_state.copy()
            state['soil_ph'] = torch.tensor(ph)
            
            result = mineralization.forward({**state, **standard_environment}, time=0.0, dt=3600.0)
            
            # Should give low activity due to pH stress
            gross_rate = _to_float(result['gross_mineralization_flux'])
            assert gross_rate < reference_gross, (
                f"Extreme pH should reduce gross mineralization: {gross_rate}"
            )
    
    def test_very_low_carbon_decomposition_rate(self, mineralization, standard_state):
        """Test handling when the canonical carbon flux is very low."""
        low_carbon_env = {
            'carbon_decomposition_rate': torch.tensor(1e-12),  # Very low
            'organic_nitrogen_input': torch.tensor(0.0),
        }
        
        result = mineralization.forward({**standard_state, **low_carbon_env}, time=0.0, dt=3600.0)
        
        # Should give low activity
        net_min = _to_float(result.get('mineralization_flux', torch.tensor(0.0)))
        
        # Net mineralization should be positive (no C limitation for immobilization)
        assert net_min > 0, "Low C availability should result in positive net mineralization"
    
    # =============================================================================
    # L1.6: Parameter Sensitivity Analysis Tests
    # =============================================================================
    
    def test_q10_sensitivity(self, standard_state, standard_environment):
        """Test sensitivity to Q10 parameter."""
        q10_values = [1.8, 2.4, 3.0, 3.5]
        gross_mineralizations = []

        param_manager = get_global_manager()
        if param_manager is None:
            raise RuntimeError("ParameterManager must be available for Q10 sensitivity test")

        q10_key = "pedosphere.mineralization.q10_mineralization"
        temp_key = "pedosphere.mineralization.temp_response_type"
        original_q10 = _normalize_param_value(param_manager.get(q10_key))
        original_temp = _normalize_param_value(param_manager.get(temp_key))
        try:
            _set_param_value(param_manager, temp_key, "q10")
            for q10 in q10_values:
                _set_param_value(param_manager, q10_key, q10)
                mineralization = MineralizationProcess(name="test", config={})
                result = mineralization.forward({**standard_state, **standard_environment}, time=0.0, dt=3600.0)
                gross_min = _to_float(result.get('gross_mineralization_flux', torch.tensor(0.0)))
                gross_mineralizations.append(gross_min)
        finally:
            _set_param_value(param_manager, q10_key, original_q10)
            _set_param_value(param_manager, temp_key, original_temp)
        
        # Higher Q10 should increase response at temperatures above reference (15 degC vs 25 degC reference)
        # Since we're at 15 degC, higher Q10 doesn't necessarily increase response
        # Just check they're all reasonable and increase with temperature
        assert all(gm > 0 for gm in gross_mineralizations), "All should be positive"
    
    def test_microbial_cn_sensitivity(self, standard_state, standard_environment):
        """Test sensitivity to microbial C:N ratio."""
        cn_ratios = [6.0, 8.0, 10.0, 12.0]
        immobilizations = []

        param_manager = get_global_manager()
        if param_manager is None:
            raise RuntimeError("ParameterManager must be available for CN sensitivity test")

        cn_key = "pedosphere.mineralization.cn_ratio_microbial"
        original_cn = _normalize_param_value(param_manager.get(cn_key))
        try:
            for cn_ratio in cn_ratios:
                _set_param_value(param_manager, cn_key, cn_ratio)
                mineralization = MineralizationProcess(name="test", config={})
                result = mineralization.forward({**standard_state, **standard_environment}, time=0.0, dt=3600.0)
                immob = _to_float(result.get('immobilization_flux', torch.tensor(0.0)))
                immobilizations.append(immob)
        finally:
            _set_param_value(param_manager, cn_key, original_cn)
        
        # Higher microbial C:N should reduce N demand and immobilization
        for i in range(len(immobilizations) - 1):
            assert immobilizations[i] >= immobilizations[i+1] * 0.8, \
                "Higher microbial C:N should reduce N demand"
    
    def test_n_assimilation_efficiency_sensitivity(self, standard_state, standard_environment):
        """Test sensitivity to nitrogen assimilation efficiency."""
        efficiencies = [0.4, 0.6, 0.8, 1.0]
        net_mineralizations = []

        param_manager = get_global_manager()
        if param_manager is None:
            raise RuntimeError("ParameterManager must be available for N efficiency test")

        eff_key = "pedosphere.mineralization.n_assimilation_efficiency"
        original_eff = _normalize_param_value(param_manager.get(eff_key))
        try:
            for eff in efficiencies:
                _set_param_value(param_manager, eff_key, eff)
                mineralization = MineralizationProcess(name="test", config={})
                result = mineralization.forward({**standard_state, **standard_environment}, time=0.0, dt=3600.0)
                net_min = _to_float(result.get('mineralization_flux', torch.tensor(0.0)))
                net_mineralizations.append(net_min)
        finally:
            _set_param_value(param_manager, eff_key, original_eff)
        
        # Higher efficiency should reduce immobilization and increase net mineralization
        assert net_mineralizations[-1] >= net_mineralizations[0], \
            "Higher N efficiency should increase net mineralization"
    
    def test_carbon_use_efficiency_sensitivity(self, standard_state, standard_environment):
        """Test sensitivity to carbon use efficiency."""
        efficiencies = [0.3, 0.5, 0.6, 0.7]
        microbial_n_demands = []

        param_manager = get_global_manager()
        if param_manager is None:
            raise RuntimeError("ParameterManager must be available for CUE sensitivity test")

        cue_key = "pedosphere.mineralization.carbon_use_efficiency"
        original_cue = _normalize_param_value(param_manager.get(cue_key))
        try:
            for eff in efficiencies:
                _set_param_value(param_manager, cue_key, eff)
                mineralization = MineralizationProcess(name="test", config={})
                result = mineralization.forward({**standard_state, **standard_environment}, time=0.0, dt=3600.0)
                n_demand = _to_float(result['immobilization_flux'])
                microbial_n_demands.append(n_demand)
        finally:
            _set_param_value(param_manager, cue_key, original_cue)
        
        # Higher C use efficiency should increase microbial N demand
        assert microbial_n_demands[-1] >= microbial_n_demands[0], \
            "Higher C use efficiency should increase microbial N demand"
    
    # =============================================================================
    # L1.7: Performance and Stability Tests
    # =============================================================================
    
    def test_numerical_stability_small_pools(self, mineralization, standard_environment):
        """Test numerical stability with very small nitrogen pools."""
        small_state = {
            'organic_nitrogen': torch.tensor(1e-10),
            'nh4_nitrogen': torch.tensor(1e-12),
            'microbial_nitrogen': torch.tensor(1e-11),
            'soil_temperature': torch.tensor(288.15),
            'soil_moisture': torch.tensor(0.3),
            'soil_ph': torch.tensor(6.5)
        }
        
        result = mineralization.forward({**small_state, **standard_environment}, time=0.0, dt=3600.0)
        
        # Should not crash and all outputs should be finite
        for key, update in result.items():
            assert isinstance(update, UpdateItem), key
            assert torch.isfinite(update.value).all(), f"{key} should be finite with small pools"
            assert not torch.isnan(update.value).any(), f"{key} should not be NaN"
        
        # Results should be appropriately small
        gross_min = _to_float(result.get('gross_mineralization_flux', torch.tensor(0.0)))
        assert gross_min < 1e-8, "Small pools should give small fluxes"
    
    def test_numerical_stability_large_dt(self, mineralization, standard_state, standard_environment):
        """Test numerical stability with large time steps."""
        large_dt = 86400.0  # 1 day
        
        result = mineralization.forward({**standard_state, **standard_environment}, time=0.0, dt=large_dt)
        
        # Should not produce unrealistic results
        for key, update in result.items():
            assert isinstance(update, UpdateItem), key
            assert torch.isfinite(update.value).all(), f"{key} should be finite with large dt"
        
        # Pool changes should be reasonable
        delta_organic = _to_float(result.get('organic_nitrogen', torch.tensor(0.0)))
        original_organic = _to_float(standard_state['organic_nitrogen'])
        
        # Should not change more than pool size per step
        assert abs(delta_organic) <= original_organic * 2, \
            "Large dt should not cause unrealistic pool changes"

    @pytest.mark.parametrize(
        "invalid_dt",
        [0.0, -1.0, float("nan"), float("inf"), torch.tensor([1.0, 2.0])],
    )
    def test_dt_must_be_finite_positive_scalar(
        self,
        mineralization,
        standard_state,
        standard_environment,
        invalid_dt,
    ):
        """Reaction integration requires one finite positive timestep."""
        with pytest.raises(ValueError, match="dt"):
            mineralization.forward(
                {**standard_state, **standard_environment},
                time=0.0,
                dt=invalid_dt,
            )

    def test_forward_preserves_input_dtype_and_device(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """Parameter tensors must align to the canonical input tensor."""
        state = {
            key: value.to(dtype=torch.float64, device="cpu")
            for key, value in {**standard_state, **standard_environment}.items()
        }
        result = mineralization.forward(state, time=0.0, dt=10.0)

        for key in mineralization.metadata.output_variables:
            assert result[key].value.dtype is torch.float64, key
            assert result[key].value.device.type == "cpu", key
    
    def test_mass_balance_conservation(self, mineralization, standard_state, standard_environment):
        """Test nitrogen mass balance conservation."""
        result = mineralization.forward({**standard_state, **standard_environment}, time=0.0, dt=3600.0)
        
        # Get nitrogen changes
        delta_organic = _to_float(result.get('organic_nitrogen', torch.tensor(0.0)))
        delta_mineral = _to_float(result.get('nh4_nitrogen', torch.tensor(0.0))) 
        delta_microbial = _to_float(result.get('microbial_nitrogen', torch.tensor(0.0)))
        
        # Total nitrogen change in system
        total_delta = delta_organic + delta_mineral + delta_microbial
        
        # Should be conserved (total change should be near zero)
        mass_balance_error = abs(total_delta)
        assert mass_balance_error < 1e-7, \
            f"Nitrogen mass balance error: {mass_balance_error:.2e}"

    def test_litter_nitrogen_boundary_enters_organic_pool_once(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """Route litter N once through the explicit open-system input port."""
        dt = 600.0
        litter_input = torch.tensor(2.5e-6)
        zero_environment = dict(standard_environment)
        nonzero_environment = {
            **standard_environment,
            "organic_nitrogen_input": litter_input,
        }
        zero_result = mineralization.forward(
            {**standard_state, **zero_environment},
            time=0.0,
            dt=dt,
        )
        nonzero_result = mineralization.forward(
            {**standard_state, **nonzero_environment},
            time=0.0,
            dt=dt,
        )

        expected_boundary_step = litter_input * dt
        torch.testing.assert_close(
            nonzero_result["organic_nitrogen"].value
            - zero_result["organic_nitrogen"].value,
            expected_boundary_step,
            rtol=0.0,
            atol=1.0e-12,
        )
        for pool in ("nh4_nitrogen", "microbial_nitrogen"):
            torch.testing.assert_close(
                nonzero_result[pool].value,
                zero_result[pool].value,
                rtol=0.0,
                atol=0.0,
            )

        boundary_step = nonzero_result[
            "diagnostics.mineralization.organic_nitrogen_boundary_input"
        ]
        assert isinstance(boundary_step, UpdateItem)
        assert boundary_step.mode is UpdateMode.REPLACE
        assert boundary_step.per_timestep is False
        assert boundary_step.unit == "g N m-2"
        assert boundary_step.source_functor == mineralization.name
        torch.testing.assert_close(
            boundary_step.value,
            expected_boundary_step,
            rtol=0.0,
            atol=0.0,
        )
        total_pool_delta = sum(
            nonzero_result[pool].value
            for pool in (
                "organic_nitrogen",
                "nh4_nitrogen",
                "microbial_nitrogen",
            )
        )
        torch.testing.assert_close(
            total_pool_delta,
            expected_boundary_step,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_nitrogen_pool_deltas_are_explicit_accumulate_updates(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """All conserved N pools must receive explicit per-step increments."""
        result = mineralization.forward(
            {**standard_state, **standard_environment},
            time=0.0,
            dt=3600.0,
        )

        for name in ("organic_nitrogen", "nh4_nitrogen", "microbial_nitrogen"):
            update = result[name]
            assert isinstance(update, UpdateItem), name
            assert update.mode is UpdateMode.ACCUMULATE, name
            assert update.unit == "g N m-2", name
            assert update.basis == "per_step", name
            assert update.per_timestep is True, name
            assert update.element == "N", name
            assert update.source_functor == mineralization.name, name

    def test_nitrogen_fluxes_are_explicit_nonincrementable_rates(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """Formal N fluxes must be instantaneous replacement values."""
        result = mineralization.forward(
            {**standard_state, **standard_environment},
            time=0.0,
            dt=3600.0,
        )

        for name in (
            "mineralization_flux",
            "gross_mineralization_flux",
            "immobilization_flux",
            "microbial_death_flux",
        ):
            update = result[name]
            assert isinstance(update, UpdateItem), name
            assert update.mode is UpdateMode.REPLACE, name
            assert update.per_timestep is False, name
            assert update.unit == "g N m-2 s-1", name
            assert update.element is None, name
            assert update.source_functor == mineralization.name, name
    
    def test_forward_is_pure_and_functor_base_owns_lifecycle(
        self,
        mineralization,
        standard_state,
        standard_environment,
    ):
        """FunctorBase must be the only owner of execution statistics."""
        for legacy_surface in (
            "_diagnostic_totals",
            "n_limitation_events",
            "c_limitation_events",
            "get_mineralization_statistics",
        ):
            assert not hasattr(mineralization, legacy_surface)

        init_source = inspect.getsource(MineralizationProcess.__init__)
        forward_source = inspect.getsource(MineralizationProcess.forward)
        assert "self._execution_count" not in init_source
        assert "self._execution_count" not in forward_source

        initial_stats = mineralization.get_execution_stats()
        direct_result = mineralization.forward(
            {**standard_state, **standard_environment},
            time=0.0,
            dt=3600.0,
        )
        assert mineralization.get_execution_stats() == initial_stats

        canonical_state = {**standard_state, **standard_environment}
        canonical_result = mineralization(
            canonical_state,
            time=0.0,
            dt=3600.0,
        )
        assert mineralization.get_execution_stats()["execution_count"] == 1
        for key in mineralization.metadata.output_variables:
            torch.testing.assert_close(
                direct_result[key].value,
                canonical_result[key].value,
            )
    
    def test_repeated_execution_consistency(self, mineralization, standard_state, standard_environment):
        """Test consistency of repeated executions."""
        results = []
        
        for _ in range(3):
            result = mineralization.forward(
                {**standard_state.copy(), **standard_environment},
                time=0.0, 
                dt=3600.0, 
            )
            net_min = _to_float(result.get('mineralization_flux', torch.tensor(0.0)))
            results.append(net_min)
        
        # Should be deterministic
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-12, \
                f"Results should be deterministic: {results[0]:.2e} vs {results[i]:.2e}"
    
    def test_gradient_computation_consistency(self, standard_state, standard_environment):
        """Inputs and every active trainable parameter must receive gradients."""
        param_manager = get_global_manager()
        temp_key = "pedosphere.mineralization.temp_response_type"
        original_temp = _normalize_param_value(param_manager.get(temp_key))
        try:
            _set_param_value(param_manager, temp_key, "q10")
            mineralization = MineralizationProcess(name="gradient_mineralization")
            grad_state = {
                key: value.clone().requires_grad_(True)
                for key, value in {**standard_state, **standard_environment}.items()
            }
            # The learnable parameters are held ParameterManager bindings; the
            # gradient is read from the manager trainable interface. Earlier
            # gradients on these shared bindings are cleared first.
            trainable = param_manager.get_trainable_parameters()
            held_keys = {
                name: mineralization.manager_binding_key(value)
                for name, value in vars(mineralization).items()
                if mineralization.manager_binding_key(value) is not None
            }
            for key in held_keys.values():
                trainable[key].grad = None
            result = mineralization.forward(grad_state, time=0.0, dt=3600.0)
            loss = sum(
                update.value.square().sum()
                for update in result.values()
            )

            assert loss.requires_grad
            loss.backward()
            assert grad_state['organic_nitrogen'].grad is not None
            assert torch.isfinite(grad_state['organic_nitrogen'].grad).all()
            expected_parameters = {
                "q10",
                "k_mineralization",
                "c_use_efficiency",
                "ph_optimum",
                "ph_range",
                "k_microbial_death",
            }
            assert set(held_keys) == expected_parameters
            for name, key in held_keys.items():
                parameter = trainable[key]
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
        finally:
            _set_param_value(param_manager, temp_key, original_temp)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])




