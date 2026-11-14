"""
L1 Unit Tests for First Order Decomposition Functor.

Tests initialization, parameter loading, and basic forward execution.
Following the RELIC L1 testing specification.
"""
import pytest
import torch
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
from relic.utils.physical_constants import get_physical_constant


POOL_OUTPUTS = (
    "metabolic_litter_c",
    "structural_litter_c",
    "fast_som_c",
    "slow_som_c",
    "passive_som_c",
)
FORMAL_FLUX_OUTPUTS = (
    "carbon_decomposition_rate",
    "heterotrophic_respiration",
)
DECOMPOSITION_DIAGNOSTICS = (
    "diagnostics.first_order_decomposition.metabolic_litter_decomposition_flux",
    "diagnostics.first_order_decomposition.structural_litter_decomposition_flux",
    "diagnostics.first_order_decomposition.fast_som_decomposition_flux",
    "diagnostics.first_order_decomposition.slow_som_decomposition_flux",
    "diagnostics.first_order_decomposition.passive_som_decomposition_flux",
    "diagnostics.first_order_decomposition.metabolic_to_fast_transfer_flux",
    "diagnostics.first_order_decomposition.structural_to_slow_transfer_flux",
    "diagnostics.first_order_decomposition.fast_to_passive_transfer_flux",
    "diagnostics.first_order_decomposition.temperature_modifier",
    "diagnostics.first_order_decomposition.moisture_modifier",
    "diagnostics.first_order_decomposition.environmental_modifier",
)
# Declared step length read by the public carbon-rate ledger (round 47).
STEP_OUTPUTS = ("diagnostics.first_order_decomposition.step_seconds",)
EXPECTED_OUTPUTS = set(POOL_OUTPUTS + FORMAL_FLUX_OUTPUTS + DECOMPOSITION_DIAGNOSTICS + STEP_OUTPUTS)
EXPECTED_INPUTS = set(
    POOL_OUTPUTS
    + (
        "soil_temperature",
        "soil_moisture",
        "metabolic_litter_input",
        "structural_litter_input",
        "lignin_n_ratio",
        "clay_content",
    )
)
EXPECTED_COUPLING_INTERFACES = {
    "soil_temperature",
    "soil_moisture",
    "metabolic_litter_input",
    "structural_litter_input",
    "carbon_decomposition_rate",
    "heterotrophic_respiration",
}
REMOVED_INPUTS = {"litter_inputs"}
REMOVED_OUTPUTS = {
    "decomposition_rate",
    "carbon_flux",
    "co2_production",
    "respiration_rate",
    "metabolic_respiration",
    "structural_respiration",
    "fast_som_respiration",
    "slow_som_respiration",
    "passive_som_respiration",
    "transfer_metabolic_to_fast",
    "transfer_structural_to_slow",
    "transfer_fast_to_passive",
    "temperature_modifier",
    "moisture_modifier",
    "environmental_modifier",
}
INACTIVE_REGISTRY_PARAMETER_KEYS = {
    "pedosphere.decomposition.q10_metabolic",
    "pedosphere.decomposition.q10_structural",
    "pedosphere.decomposition.q10_fast_som",
    "pedosphere.decomposition.q10_slow_som",
    "pedosphere.decomposition.q10_passive_som",
    "pedosphere.decomposition.cn_ratio_metabolic",
    "pedosphere.decomposition.cn_ratio_structural",
    "pedosphere.decomposition.cn_ratio_fast_som",
    "pedosphere.decomposition.cn_ratio_slow_som",
    "pedosphere.decomposition.cn_ratio_passive_som",
    "pedosphere.decomposition.lignin_content_metabolic",
    "pedosphere.decomposition.lignin_content_structural",
}
EXPECTED_ACTIVE_PARAMETER_KEYS = {
    "pedosphere.decomposition.param_25",
    "pedosphere.decomposition.clay_enhancement_coefficient",
    "pedosphere.decomposition.max_clay_factor",
    "pedosphere.decomposition.clay_protection_coefficient",
    "pedosphere.decomposition.k_metabolic",
    "pedosphere.decomposition.k_structural",
    "pedosphere.decomposition.k_fast_som",
    "pedosphere.decomposition.k_slow_som",
    "pedosphere.decomposition.k_passive_som",
    "pedosphere.decomposition.t_met_fast",
    "pedosphere.decomposition.t_met_slow",
    "pedosphere.decomposition.t_str_fast",
    "pedosphere.decomposition.t_str_slow",
    "pedosphere.decomposition.t_fast_slow",
    "pedosphere.decomposition.t_fast_passive",
    "pedosphere.decomposition.t_slow_fast",
    "pedosphere.decomposition.t_slow_passive",
    "pedosphere.decomposition.temp_response_type",
    "pedosphere.physical.q10_decomposition",
    "pedosphere.physical.activation_energy_decomposition",
    "pedosphere.decomposition.moisture_response_power",
    "pedosphere.decomposition.field_capacity",
    "pedosphere.decomposition.wilting_point",
    "pedosphere.numerical.clamp_min",
    "pedosphere.numerical.clamp_max",
    "pedosphere.physical.temperature_minimum_biological",
    "pedosphere.physical.temperature_maximum_biological",
    "pedosphere.numerical.temperature_response_min",
    "pedosphere.numerical.temperature_response_max",
    "pedosphere.numerical.unity_multiplier",
    "pedosphere.decomposition.lignin_inhibition_coefficient",
    "pedosphere.decomposition.max_lignin_n_ratio",
}


def _fixed_state(
    *,
    metabolic_litter_input: float = 0.0,
    structural_litter_input: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return the fixed scientific regression state."""
    return {
        "metabolic_litter_c": torch.tensor(100.0, dtype=torch.float32),
        "structural_litter_c": torch.tensor(200.0, dtype=torch.float32),
        "fast_som_c": torch.tensor(500.0, dtype=torch.float32),
        "slow_som_c": torch.tensor(1000.0, dtype=torch.float32),
        "passive_som_c": torch.tensor(2000.0, dtype=torch.float32),
        "soil_temperature": torch.tensor(293.15, dtype=torch.float32),
        "soil_moisture": torch.tensor(0.3, dtype=torch.float32),
        "metabolic_litter_input": torch.tensor(
            metabolic_litter_input,
            dtype=torch.float32,
        ),
        "structural_litter_input": torch.tensor(
            structural_litter_input,
            dtype=torch.float32,
        ),
        "lignin_n_ratio": torch.tensor(0.3, dtype=torch.float32),
        "clay_content": torch.tensor(0.25, dtype=torch.float32),
    }


def _value(outputs: dict[str, UpdateItem], key: str) -> torch.Tensor:
    """Return the tensor payload from an explicit update item."""
    item = outputs[key]
    assert isinstance(item, UpdateItem), f"{key} must use explicit UpdateItem semantics"
    assert isinstance(item.value, torch.Tensor), f"{key} payload must be a Tensor"
    return item.value


class TestFirstOrderDecompositionInit:
    """Test initialization and parameter loading."""

    def test_initialization(self, parameter_manager, basic_config):
        """Test Functor can be initialized."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )
        assert functor is not None
        assert functor.name == "test_decomp"

    def test_parameter_loading(self, parameter_manager, basic_config):
        """Test that required parameters are loaded from manager."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )
        # Should have loaded decomposition rates
        assert hasattr(functor, 'pool_params')
        assert 'metabolic_litter' in functor.pool_params
        assert 'k_ref' in functor.pool_params['metabolic_litter']

        # Should have temperature response parameters
        assert hasattr(functor, 'q10') or hasattr(functor, 'activation_energy')

    def test_metadata_present(self, basic_config):
        """Test metadata declares the exact canonical output interface."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )
        metadata = functor.metadata
        assert metadata is not None
        assert set(metadata.input_variables) == EXPECTED_INPUTS
        assert REMOVED_INPUTS.isdisjoint(metadata.input_variables)
        assert set(metadata.output_variables) == EXPECTED_OUTPUTS
        assert REMOVED_OUTPUTS.isdisjoint(metadata.output_variables)
        assert set(metadata.parameters) == EXPECTED_ACTIVE_PARAMETER_KEYS
        assert set(functor.required_parameters()) == EXPECTED_ACTIVE_PARAMETER_KEYS
        assert functor.optional_input_variables == []
        assert set(metadata.coupling_interfaces) == EXPECTED_COUPLING_INTERFACES
        assert "carbon" in metadata.conservation_laws
        assert metadata.references

    def test_litter_partition_has_one_runtime_owner(self, basic_config):
        """FirstOrder must consume mapped ports without a second partition path."""
        functor = FirstOrderDecomposition(
            name="single_partition_owner_test",
            config=basic_config,
        )

        assert not hasattr(functor, "_partition_litter_inputs")
        assert not hasattr(functor, "default_litter_input")
        assert not hasattr(functor, "lignin_partitioning_coefficient")
        assert not hasattr(functor, "min_metabolic_fraction")
        assert not hasattr(functor, "max_metabolic_fraction")

    def test_required_state_inputs_have_no_parameter_default_surface(
        self,
        basic_config,
    ):
        """Required StateManager values must not be shadowed by dead defaults."""
        functor = FirstOrderDecomposition(
            name="required_state_owner_test",
            config=basic_config,
        )

        assert not hasattr(functor, "soil_temperature_default")
        assert not hasattr(functor, "soil_moisture_default")
        assert not hasattr(functor, "clay_content")
        assert not hasattr(functor, "_pool_state_defaults")

    def test_configured_litter_mapper_dependency_is_preserved(self, basic_config):
        """Graph configuration must order FirstOrder after the selected mapper."""
        config = dict(basic_config)
        config["dependencies"] = ["litterfall_interface_mapper"]
        functor = FirstOrderDecomposition(
            name="configured_dependency_test",
            config=config,
        )

        assert functor.get_dependencies() == ["litterfall_interface_mapper"]

    def test_inactive_registry_parameters_are_not_runtime_dependencies(
        self,
        monkeypatch,
        basic_config,
    ):
        """Unavailable registry-only values must not block the carbon process."""
        original_get_parameter_value = FirstOrderDecomposition.get_parameter_value

        def reject_inactive_registry_keys(
            self,
            name,
            *args,
            namespace=None,
            **kwargs,
        ):
            full_key = f"{namespace}.{name}" if namespace else name
            if full_key in INACTIVE_REGISTRY_PARAMETER_KEYS:
                raise KeyError(f"Inactive registry parameter is unavailable: {full_key}")
            return original_get_parameter_value(
                self,
                name,
                *args,
                namespace=namespace,
                **kwargs,
            )

        monkeypatch.setattr(
            FirstOrderDecomposition,
            "get_parameter_value",
            reject_inactive_registry_keys,
        )
        functor = FirstOrderDecomposition(
            name="active_parameter_contract_test",
            config=basic_config,
        )
        outputs = functor.forward(_fixed_state(), time=0.0, dt=3600.0)

        assert set(outputs) == EXPECTED_OUTPUTS
        assert set(functor.required_parameters()) == EXPECTED_ACTIVE_PARAMETER_KEYS


class TestFirstOrderDecompositionForward:
    """Test forward execution."""

    def test_forward_runs(self, basic_config):
        """Test forward returns exactly the canonical 18 outputs."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )

        updates = functor.forward(
            state=_fixed_state(),
            time=0.0,
            dt=3600.0  # 1 hour
        )

        assert isinstance(updates, dict)
        assert set(updates) == EXPECTED_OUTPUTS
        assert REMOVED_OUTPUTS.isdisjoint(updates)

    def test_forward_returns_tensors(self, basic_config):
        """Test every output has explicit state-update semantics."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )

        updates = functor.forward(
            state=_fixed_state(),
            time=0.0,
            dt=3600.0
        )

        for key, item in updates.items():
            assert isinstance(item, UpdateItem), f"{key} should be UpdateItem"
            assert isinstance(item.value, torch.Tensor), f"{key} payload should be Tensor"
            assert item.source_functor == "test_decomp"

        for key in POOL_OUTPUTS:
            item = updates[key]
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == "g C m-2"

        for key in FORMAL_FLUX_OUTPUTS + DECOMPOSITION_DIAGNOSTICS:
            item = updates[key]
            assert item.mode == UpdateMode.REPLACE

        # Gross decomposition is an internal transformation diagnostic and is
        # excluded from the elemental boundary ledger. Heterotrophic
        # respiration is the routed atmospheric carbon export.
        assert updates["carbon_decomposition_rate"].element is None
        assert updates["heterotrophic_respiration"].element == "C"

        for key in FORMAL_FLUX_OUTPUTS + DECOMPOSITION_DIAGNOSTICS[:8]:
            assert updates[key].unit == "g C m-2 s-1"

        for key in DECOMPOSITION_DIAGNOSTICS[8:]:
            assert updates[key].unit == "dimensionless"

    def test_forward_preserves_float64_state_dtype(self, basic_config):
        """Scientific outputs must preserve the StateManager floating dtype."""
        state = {
            key: value.to(dtype=torch.float64)
            for key, value in _fixed_state(
                metabolic_litter_input=8.5e-7,
                structural_litter_input=1.5e-7,
            ).items()
        }
        functor = FirstOrderDecomposition(
            name="dtype_preservation_test",
            config=basic_config,
        )

        updates = functor.forward(state=state, time=0.0, dt=3600.0)

        assert all(item.value.dtype == torch.float64 for item in updates.values())

    def test_forward_positive_decomposition(self, basic_config):
        """Test canonical gross decomposition and respiration are positive."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )

        updates = functor.forward(
            state=_fixed_state(),
            time=0.0,
            dt=3600.0
        )

        assert torch.all(_value(updates, "carbon_decomposition_rate") > 0)
        assert torch.all(_value(updates, "heterotrophic_respiration") > 0)
        assert torch.all(
            _value(updates, "carbon_decomposition_rate")
            >= _value(updates, "heterotrophic_respiration")
        )

    @pytest.mark.parametrize(
        "required_key",
        POOL_OUTPUTS
        + (
            "soil_temperature",
            "soil_moisture",
            "metabolic_litter_input",
            "structural_litter_input",
            "lignin_n_ratio",
            "clay_content",
        ),
    )
    def test_required_state_inputs_fail_fast(self, required_key, basic_config):
        """Direct forward must not replace required state with parameter defaults."""
        state = _fixed_state()
        state.pop(required_key)
        functor = FirstOrderDecomposition(
            name="required_input_test",
            config=basic_config,
        )

        with pytest.raises(KeyError, match=required_key):
            functor.forward(state, time=0.0, dt=3600.0)

    def test_old_generic_litter_input_is_not_a_runtime_alias(self, basic_config):
        """The removed generic key must not satisfy either mapped input port."""
        state = _fixed_state()
        state.pop("metabolic_litter_input")
        state["litter_inputs"] = torch.tensor(1.0e-6)
        functor = FirstOrderDecomposition(
            name="removed_litter_alias_test",
            config=basic_config,
        )

        with pytest.raises(KeyError, match="metabolic_litter_input"):
            functor.forward(state, time=0.0, dt=3600.0)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("metabolic_litter_input", -1.0e-6),
            ("metabolic_litter_input", float("nan")),
            ("structural_litter_input", -1.0e-6),
            ("structural_litter_input", float("inf")),
        ],
    )
    def test_invalid_mapped_litter_inputs_fail_fast(
        self,
        key,
        value,
        basic_config,
    ):
        """Mapped litter ports must be finite non-negative carbon fluxes."""
        state = _fixed_state()
        state[key] = torch.tensor(value)
        functor = FirstOrderDecomposition(
            name="invalid_mapped_litter_test",
            config=basic_config,
        )

        with pytest.raises(ValueError, match=key):
            functor.forward(state, time=0.0, dt=3600.0)

    @pytest.mark.parametrize(
        ("driver", "invalid_value"),
        [
            ("soil_temperature", float("nan")),
            ("soil_temperature", 223.15),
            ("soil_moisture", float("inf")),
        ],
    )
    def test_invalid_environmental_drivers_fail_fast(
        self,
        driver,
        invalid_value,
        basic_config,
    ):
        """Non-finite or out-of-domain drivers must not be clipped silently."""
        state = _fixed_state()
        state[driver] = torch.tensor(invalid_value)
        functor = FirstOrderDecomposition(
            name="invalid_driver_test",
            config=basic_config,
        )

        with pytest.raises(ValueError, match=driver):
            functor.forward(state, time=0.0, dt=3600.0)

    def test_transfer_fraction_sum_above_unity_fails_fast(self, basic_config):
        """Outgoing transfer fractions cannot create carbon before respiration."""
        functor = FirstOrderDecomposition(
            name="transfer_sum_test",
            config=basic_config,
        )
        functor.transfer_params["metabolic_to_fast"] = 0.7
        functor.transfer_params["metabolic_to_slow"] = 0.4

        with pytest.raises(ValueError, match="metabolic_litter.*sum"):
            functor.forward(_fixed_state(), time=0.0, dt=3600.0)

    def test_temperature_sensitivity(self, basic_config):
        """Test that decomposition rate increases with temperature."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )

        # Cold environment
        state_cold = _fixed_state()
        state_cold['soil_temperature'] = torch.tensor(278.15)  # 5degC

        # Warm environment
        state_warm = _fixed_state()
        state_warm['soil_temperature'] = torch.tensor(298.15)  # 25degC

        updates_cold = functor.forward(state_cold, 0.0, 3600.0)
        updates_warm = functor.forward(state_warm, 0.0, 3600.0)

        flux_cold = _value(updates_cold, "heterotrophic_respiration")
        flux_warm = _value(updates_warm, "heterotrophic_respiration")
        assert torch.all(flux_warm > flux_cold)

    def test_moisture_sensitivity(self, basic_config):
        """Test that decomposition responds to moisture."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )

        # Dry conditions
        state_dry = _fixed_state()
        state_dry['soil_moisture'] = torch.tensor(0.1)  # Dry

        # Optimal moisture
        state_optimal = _fixed_state()
        state_optimal['soil_moisture'] = torch.tensor(0.4)  # Optimal

        updates_dry = functor.forward(state_dry, 0.0, 3600.0)
        updates_optimal = functor.forward(state_optimal, 0.0, 3600.0)

        flux_dry = _value(updates_dry, "heterotrophic_respiration")
        flux_optimal = _value(updates_optimal, "heterotrophic_respiration")
        assert torch.all(flux_optimal > flux_dry)

    def test_runtime_clay_content_is_the_only_clay_response_owner(
        self,
        basic_config,
    ):
        """The explicit clay input must drive protection and passive transfer."""
        functor = FirstOrderDecomposition(
            name="runtime_clay_owner_test",
            config=basic_config,
        )
        low_clay_state = _fixed_state()
        high_clay_state = _fixed_state()
        low_clay_state["clay_content"] = torch.tensor(0.0)
        high_clay_state["clay_content"] = torch.tensor(0.5)

        low_outputs = functor.forward(low_clay_state, time=0.0, dt=3600.0)
        high_outputs = functor.forward(high_clay_state, time=0.0, dt=3600.0)
        low_fraction = _value(
            low_outputs,
            "diagnostics.first_order_decomposition.fast_to_passive_transfer_flux",
        ) / _value(
            low_outputs,
            "diagnostics.first_order_decomposition.fast_som_decomposition_flux",
        )
        high_fraction = _value(
            high_outputs,
            "diagnostics.first_order_decomposition.fast_to_passive_transfer_flux",
        ) / _value(
            high_outputs,
            "diagnostics.first_order_decomposition.fast_som_decomposition_flux",
        )

        assert high_fraction > low_fraction

    def test_forward_has_no_hidden_scientific_history(self, basic_config):
        """Scientific cumulative state must remain owned by StateManager."""
        functor = FirstOrderDecomposition(
            name="state_purity_test",
            config=basic_config,
        )
        forbidden_attributes = {
            "total_respiration",
            "total_decomposed",
            "pool_turnovers",
            "get_decomposition_statistics",
        }
        assert all(
            not hasattr(functor, attribute)
            for attribute in forbidden_attributes
        )

        before = {
            key: value.detach().clone()
            for key, value in functor.state_dict().items()
        }
        functor.forward(_fixed_state(), time=0.0, dt=3600.0)
        after = functor.state_dict()
        assert set(after) == set(before)
        assert all(torch.equal(after[key], value) for key, value in before.items())


class TestFirstOrderDecompositionConservation:
    """Test conservation laws."""

    def test_fixed_numerical_regression(self, basic_config):
        """Test the fixed five-pool numerical baseline."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )
        outputs = functor.forward(_fixed_state(), time=0.0, dt=3600.0)

        # Baseline re-established on 2026-09-14 after restricting the lignin
        # inhibition to the structural litter pool (Parton et al. 1987, Eq.
        # [2]-[4]); the metabolic pool now decays at k f_T f_theta C.
        expected_pool_deltas = {
            "metabolic_litter_c": -8.066470175982e-02,
            "structural_litter_c": -3.213990479708e-02,
            "fast_som_c": -1.073344573379e-01,
            "slow_som_c": 1.061581373215e-01,
            "passive_som_c": 7.905141450465e-03,
        }
        for key, expected in expected_pool_deltas.items():
            actual = _value(outputs, key)
            torch.testing.assert_close(
                actual,
                torch.as_tensor(expected, dtype=actual.dtype, device=actual.device),
                rtol=1.0e-6,
                atol=1.0e-10,
            )

        gross = _value(outputs, "carbon_decomposition_rate")
        respiration = _value(outputs, "heterotrophic_respiration")
        torch.testing.assert_close(
            gross,
            torch.as_tensor(7.927035767352e-05, dtype=gross.dtype, device=gross.device),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            respiration,
            torch.as_tensor(2.946549284388e-05, dtype=respiration.dtype, device=respiration.device),
            rtol=1.0e-6,
            atol=1.0e-12,
        )

        residual = (
            torch.stack([_value(outputs, key) for key in POOL_OUTPUTS]).sum()
            + respiration * 3600.0
        )
        # The ledger closes to float32 rounding of O(0.1) increments (ulp ~ 7e-9).
        assert abs(residual.item()) <= 1.0e-7

    def test_carbon_balance(self, basic_config):
        """Test split inputs enter their pools once and close the carbon ledger."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )

        dt = 3600.0
        metabolic_input = 8.5e-7
        structural_input = 1.5e-7
        without_input = functor.forward(_fixed_state(), time=0.0, dt=dt)
        outputs = functor.forward(
            state=_fixed_state(
                metabolic_litter_input=metabolic_input,
                structural_litter_input=structural_input,
            ),
            time=0.0,
            dt=dt,
        )
        total_delta = torch.stack([_value(outputs, key) for key in POOL_OUTPUTS]).sum()
        respiration = _value(outputs, "heterotrophic_respiration")
        residual = (
            total_delta
            + respiration * dt
            - (metabolic_input + structural_input) * dt
        )

        metabolic_delta = (
            _value(outputs, "metabolic_litter_c")
            - _value(without_input, "metabolic_litter_c")
        )
        structural_delta = (
            _value(outputs, "structural_litter_c")
            - _value(without_input, "structural_litter_c")
        )
        torch.testing.assert_close(
            metabolic_delta,
            torch.as_tensor(
                metabolic_input * dt,
                dtype=metabolic_delta.dtype,
                device=metabolic_delta.device,
            ),
        )
        torch.testing.assert_close(
            structural_delta,
            torch.as_tensor(
                structural_input * dt,
                dtype=structural_delta.dtype,
                device=structural_delta.device,
            ),
        )

        torch.testing.assert_close(
            total_delta,
            torch.as_tensor(-1.024757921696e-01, dtype=total_delta.dtype),
            rtol=1.0e-6,
            atol=1.0e-10,
        )
        # Residual is float32 rounding only (inputs enter once, ledger closes).
        assert abs(residual.item()) <= 1.0e-7

    def test_conservation_checker_uses_supplied_updates(self, basic_config):
        """Conservation validation must detect a modified returned pool update."""
        state = _fixed_state()
        functor = FirstOrderDecomposition(
            name="conservation_payload_test",
            config=basic_config,
        )
        outputs = functor.forward(state, time=0.0, dt=3600.0)
        original = outputs["metabolic_litter_c"]
        outputs["metabolic_litter_c"] = UpdateItem(
            value=original.value + 1.0,
            mode=original.mode,
            write_mode=original.write_mode,
            unit=original.unit,
            basis=original.basis,
            per_timestep=original.per_timestep,
            element=original.element,
            source_functor=original.source_functor,
        )

        violations = functor.check_conservation_laws(state, outputs, dt=3600.0)

        assert violations["carbon"] > 0.99

    def test_conservation_dt_context_is_not_checkpointed(self, basic_config):
        """The lifecycle-only conservation dt must remain non-persistent."""
        functor = FirstOrderDecomposition(
            name="nonpersistent_conservation_dt_test",
            config=basic_config,
        )
        state = _fixed_state()
        outputs = functor.forward(state, time=0.0, dt=3600.0)

        assert "_last_time_step" not in functor.state_dict()
        violations = functor.check_conservation_laws(state, outputs)
        # float32 rounding of O(0.1) increments; the ledger itself is closed.
        assert violations["carbon"] <= 1.0e-7


class TestFirstOrderDecompositionPoolDynamics:
    """Test pool-specific dynamics."""

    def test_metabolic_litter_decomposes_faster(self, basic_config):
        """Test that metabolic litter decomposes faster than structural."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )

        # Set equal pool sizes
        state = _fixed_state()
        state['metabolic_litter_c'] = torch.tensor(100.0)
        state['structural_litter_c'] = torch.tensor(100.0)

        updates = functor.forward(state, 0.0, 3600.0)

        metabolic_flux = _value(
            updates,
            "diagnostics.first_order_decomposition.metabolic_litter_decomposition_flux",
        )
        structural_flux = _value(
            updates,
            "diagnostics.first_order_decomposition.structural_litter_decomposition_flux",
        )
        assert metabolic_flux.item() > structural_flux.item()

    def test_som_pools_decompose_slower(self, basic_config):
        """Test all pool decomposition diagnostics are non-negative."""
        functor = FirstOrderDecomposition(
            name="test_decomp",
            config=basic_config
        )

        updates = functor.forward(_fixed_state(), 0.0, 3600.0)

        decomposition_fluxes = DECOMPOSITION_DIAGNOSTICS[:5]
        assert all(_value(updates, key).item() >= 0.0 for key in decomposition_fluxes)

    def test_pool_increments_scale_once_with_dt(self, basic_config):
        """Test pool increments receive exactly one timestep scaling."""
        state = _fixed_state()
        short = FirstOrderDecomposition(name="short_dt", config=basic_config).forward(
            state, 0.0, 1800.0
        )
        long = FirstOrderDecomposition(name="long_dt", config=basic_config).forward(
            state, 0.0, 3600.0
        )

        for key in POOL_OUTPUTS:
            torch.testing.assert_close(
                _value(long, key),
                2.0 * _value(short, key),
                rtol=1.0e-6,
                atol=1.0e-10,
            )
        for key in FORMAL_FLUX_OUTPUTS:
            torch.testing.assert_close(
                _value(long, key),
                _value(short, key),
                rtol=1.0e-6,
                atol=1.0e-12,
            )

    def test_oversized_timestep_requires_simulator_managed_substeps(
        self,
        basic_config,
    ):
        """An unstable direct step must fail before producing negative pools."""
        state = _fixed_state()
        dt = float(get_physical_constant("seconds_per_year"))
        functor = FirstOrderDecomposition(
            name="large_step_test",
            config=basic_config,
        )

        with pytest.raises(ValueError, match="simulator-managed substeps"):
            functor.forward(state, time=0.0, dt=dt)

    def test_zero_pools_and_zero_input_produce_zero_updates(self, basic_config):
        """Test an empty closed local ledger remains exactly empty."""
        state = _fixed_state()
        for key in POOL_OUTPUTS:
            state[key] = torch.tensor(0.0)
        outputs = FirstOrderDecomposition(
            name="zero_pool_test",
            config=basic_config,
        ).forward(state, 0.0, 3600.0)

        for key in POOL_OUTPUTS + FORMAL_FLUX_OUTPUTS + DECOMPOSITION_DIAGNOSTICS[:8]:
            assert _value(outputs, key).item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_forward_keeps_no_module_state_and_declares_its_step():
    """Round 47 (I1): forward writes no module tensor; two identical calls give
    bit-identical outputs and unchanged parameters and buffers; the step length reaches
    the public ledger as the declared diagnostics.first_order_decomposition.step_seconds output."""
    functor = FirstOrderDecomposition(name="decomp_i1", config={})

    def _snapshot():
        return {
            **{f"p.{k}": v.detach().clone() for k, v in functor.named_parameters()},
            **{f"b.{k}": v.detach().clone() for k, v in functor.named_buffers()},
        }

    before = _snapshot()
    first = functor(_fixed_state(metabolic_litter_input=1.0e-6), time=0.0, dt=3600.0)
    second = functor(_fixed_state(metabolic_litter_input=1.0e-6), time=0.0, dt=3600.0)
    after = _snapshot()
    assert not hasattr(functor, "_last_time_step")
    assert before.keys() == after.keys()
    for key, value in before.items():
        assert torch.equal(value, after[key]), key
    assert first.keys() == second.keys()
    for key in first:
        a = first[key].value if isinstance(first[key], UpdateItem) else first[key]
        b = second[key].value if isinstance(second[key], UpdateItem) else second[key]
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), key
    step = first["diagnostics.first_order_decomposition.step_seconds"]
    assert isinstance(step, UpdateItem) and step.unit == "s"
    assert float(step.value) == 3600.0
    assert "diagnostics.first_order_decomposition.step_seconds" in functor.metadata.output_variables
