"""L1 tests for the canonical autotrophic-respiration carbon ledger."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.scientific.biosphere.respiration.autotrophic_respiration import (
    AutotrophicRespiration,
)


REQUIRED_INPUTS = [
    "gross_primary_productivity",
    "maintenance_respiration_total",
    "growth_respiration_total",
]
OUTPUTS = [
    "total_autotrophic_respiration",
    "net_primary_productivity",
    "diagnostics.autotrophic_respiration.maintenance_component",
    "diagnostics.autotrophic_respiration.growth_component",
    "diagnostics.autotrophic_respiration.maintenance_fraction",
    "diagnostics.autotrophic_respiration.growth_fraction",
    "diagnostics.autotrophic_respiration.carbon_balance_residual",
]


def _state(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    state = {
        "gross_primary_productivity": torch.tensor(1.0e-7),
        "maintenance_respiration_total": torch.tensor(3.0e-8),
        "growth_respiration_total": torch.tensor(2.0e-8),
    }
    state.update(overrides)
    return state


def _value(output: dict[str, UpdateItem], key: str) -> torch.Tensor:
    item = output[key]
    assert isinstance(item, UpdateItem)
    assert isinstance(item.value, torch.Tensor)
    return item.value


@pytest.fixture
def functor() -> AutotrophicRespiration:
    return AutotrophicRespiration(pft_name="broadleaf_deciduous_temperate")


class TestAutotrophicRespirationL1:
    """Verify one aggregate carbon ledger and one NPP writer."""

    def test_initialization_and_exact_contract(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        assert isinstance(functor, torch.nn.Module)
        assert functor.pft_name == "broadleaf_deciduous_temperate"
        assert functor.get_dependencies() == [
            "maintenance_respiration",
            "growth_respiration",
        ]
        assert functor.get_required_inputs() == REQUIRED_INPUTS
        assert functor.metadata.input_variables == REQUIRED_INPUTS
        assert functor.get_provides() == OUTPUTS
        assert functor.metadata.output_variables == OUTPUTS
        assert set(functor.metadata.parameters) == {
            "biosphere.validation.epsilon.flux"
        }
        assert not hasattr(functor, "maintenance_respiration")
        assert not hasattr(functor, "growth_respiration")

    def test_runtime_dependency_override_uses_functor_base_state(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        functor.set_dependencies(["farquhar_photosynthesis"])

        assert functor.get_dependencies() == ["farquhar_photosynthesis"]

    def test_forward_exact_outputs_units_and_fixed_numerics(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        output = functor.forward(_state(), time=0.0, dt=3600.0)

        assert list(output) == OUTPUTS
        for key, item in output.items():
            assert isinstance(item, UpdateItem), key
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.source_functor == "autotrophic_respiration"

        expected = {
            "total_autotrophic_respiration": 5.0e-8,
            "net_primary_productivity": 5.0e-8,
            "diagnostics.autotrophic_respiration.maintenance_component": 3.0e-8,
            "diagnostics.autotrophic_respiration.growth_component": 2.0e-8,
            "diagnostics.autotrophic_respiration.maintenance_fraction": 0.6,
            "diagnostics.autotrophic_respiration.growth_fraction": 0.4,
            "diagnostics.autotrophic_respiration.carbon_balance_residual": 0.0,
        }
        for key, expected_value in expected.items():
            torch.testing.assert_close(
                _value(output, key),
                torch.tensor(expected_value),
                rtol=2.0e-6,
                atol=1.0e-14,
            )

        expected_units = {
            "total_autotrophic_respiration": "kg C m-2 s-1",
            "net_primary_productivity": "kg C m-2 s-1",
            "diagnostics.autotrophic_respiration.maintenance_component": "kg C m-2 s-1",
            "diagnostics.autotrophic_respiration.growth_component": "kg C m-2 s-1",
            "diagnostics.autotrophic_respiration.maintenance_fraction": "1",
            "diagnostics.autotrophic_respiration.growth_fraction": "1",
            "diagnostics.autotrophic_respiration.carbon_balance_residual": "kg C m-2 s-1",
        }
        assert {key: item.unit for key, item in output.items()} == expected_units

    def test_nonincrementable_fluxes_commit_through_state_manager(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        """Route both instantaneous carbon fluxes without storage-element hints."""
        output = functor.forward(_state(), time=0.0, dt=3600.0)
        flux_keys = (
            "total_autotrophic_respiration",
            "net_primary_productivity",
        )
        expected_directions = {
            "total_autotrophic_respiration": {
                "allowed_in": [],
                "allowed_out": ["autotrophic_respiration"],
            },
            "net_primary_productivity": {
                "allowed_in": ["autotrophic_respiration"],
                "allowed_out": ["autotrophic_respiration"],
            },
        }

        manager = StateManager(device="cpu")
        registry = manager._contract_registry
        assert isinstance(registry, ContractRegistry)
        for key in flux_keys:
            contract = registry.get_contract(key)
            assert contract is not None
            assert contract.kind == "flux"
            assert contract.incrementable is False
            assert contract.raw["accumulates"] is None
            assert contract.raw["flux_contract"] == {
                **expected_directions[key],
                "budget_check": {
                    "guardian": None,
                    "tolerance": {
                        "atol": 1.0e-10,
                        "rtol": 1.0e-8,
                    },
                },
            }

        manager.set_state(
            {key: torch.tensor(0.0) for key in flux_keys},
            validate=False,
        )
        committed = manager.apply_updates(
            {key: output[key] for key in flux_keys},
            dt=3600.0,
            check_conservation=False,
            update_sources={key: functor.name for key in flux_keys},
        )

        for key in flux_keys:
            assert output[key].element is None
            torch.testing.assert_close(
                committed[key],
                output[key].value,
                rtol=0.0,
                atol=0.0,
            )

    def test_exact_carbon_ledger(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        state = _state(
            gross_primary_productivity=torch.tensor(1.538169e-7),
            maintenance_respiration_total=torch.tensor(3.616369e-8),
            growth_respiration_total=torch.tensor(2.587189e-8),
        )
        output = functor.forward(state, 0.0, 60.0)

        total = _value(output, "total_autotrophic_respiration")
        npp = _value(output, "net_primary_productivity")
        residual = state["gross_primary_productivity"] - total - npp
        torch.testing.assert_close(
            total,
            state["maintenance_respiration_total"]
            + state["growth_respiration_total"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(residual, torch.tensor(0.0), rtol=0.0, atol=1.0e-14)
        torch.testing.assert_close(
            _value(
                output,
                "diagnostics.autotrophic_respiration.carbon_balance_residual",
            ),
            residual,
            rtol=0.0,
            atol=0.0,
        )

    def test_negative_npp_is_not_hidden_by_component_scaling(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        state = _state(
            gross_primary_productivity=torch.tensor(4.0e-8),
            maintenance_respiration_total=torch.tensor(3.0e-8),
            growth_respiration_total=torch.tensor(2.0e-8),
        )
        output = functor.forward(state, 0.0, 1.0)

        torch.testing.assert_close(
            _value(output, "total_autotrophic_respiration"),
            torch.tensor(5.0e-8),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            _value(output, "net_primary_productivity"),
            torch.tensor(-1.0e-8),
            rtol=2.0e-6,
            atol=1.0e-14,
        )
        torch.testing.assert_close(
            _value(
                output,
                "diagnostics.autotrophic_respiration.maintenance_component",
            ),
            state["maintenance_respiration_total"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            _value(
                output,
                "diagnostics.autotrophic_respiration.growth_component",
            ),
            state["growth_respiration_total"],
            rtol=0.0,
            atol=0.0,
        )

    def test_zero_total_has_zero_fractions(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        output = functor.forward(
            _state(
                maintenance_respiration_total=torch.tensor(0.0),
                growth_respiration_total=torch.tensor(0.0),
            ),
            0.0,
            1.0,
        )
        for key in (
            "total_autotrophic_respiration",
            "diagnostics.autotrophic_respiration.maintenance_fraction",
            "diagnostics.autotrophic_respiration.growth_fraction",
        ):
            torch.testing.assert_close(
                _value(output, key),
                torch.tensor(0.0),
                rtol=0.0,
                atol=0.0,
            )

    def test_fraction_epsilon_is_diagnostic_only(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        manager = functor.get_parameter_manager()
        spec = manager.get_parameter("biosphere.validation.epsilon.flux")
        assert spec is not None
        assert spec.units == "kg C m-2 s-1"
        epsilon = functor.get_parameter_value(
            "flux",
            namespace="biosphere.validation.epsilon",
            required=True,
        )
        epsilon = torch.as_tensor(epsilon, dtype=torch.float32)
        maintenance = 0.3 * epsilon
        growth = 0.2 * epsilon
        gpp = 2.0 * epsilon
        output = functor.forward(
            _state(
                gross_primary_productivity=gpp,
                maintenance_respiration_total=maintenance,
                growth_respiration_total=growth,
            ),
            0.0,
            1.0,
        )

        torch.testing.assert_close(
            _value(output, "total_autotrophic_respiration"),
            maintenance + growth,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            _value(output, "net_primary_productivity"),
            gpp - maintenance - growth,
            rtol=0.0,
            atol=0.0,
        )
        for key in (
            "diagnostics.autotrophic_respiration.maintenance_fraction",
            "diagnostics.autotrophic_respiration.growth_fraction",
        ):
            value = _value(output, key)
            torch.testing.assert_close(
                value,
                torch.zeros_like(value),
                rtol=0.0,
                atol=0.0,
            )

    def test_outputs_are_dt_invariant_replace_values(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        one_second = functor.forward(_state(), 0.0, 1.0)
        one_hour = functor.forward(_state(), 0.0, 3600.0)
        for key in OUTPUTS:
            torch.testing.assert_close(
                _value(one_second, key),
                _value(one_hour, key),
                rtol=0.0,
                atol=0.0,
            )

    @pytest.mark.parametrize("missing_key", REQUIRED_INPUTS)
    def test_missing_declared_input_fails_fast(
        self,
        functor: AutotrophicRespiration,
        missing_key: str,
    ) -> None:
        state = _state()
        state.pop(missing_key)
        with pytest.raises(KeyError, match=missing_key):
            functor.forward(state, 0.0, 1.0)

    @pytest.mark.parametrize("key", REQUIRED_INPUTS)
    def test_inputs_must_be_tensors(
        self,
        functor: AutotrophicRespiration,
        key: str,
    ) -> None:
        state = _state()
        state[key] = 1.0  # type: ignore[assignment]
        with pytest.raises(TypeError, match=key):
            functor.forward(state, 0.0, 1.0)

    @pytest.mark.parametrize("key", REQUIRED_INPUTS)
    def test_inputs_must_be_floating_point_tensors(
        self,
        functor: AutotrophicRespiration,
        key: str,
    ) -> None:
        state = _state()
        state[key] = torch.tensor(1, dtype=torch.int64)
        with pytest.raises(TypeError, match=key):
            functor.forward(state, 0.0, 1.0)

    @pytest.mark.parametrize("key", REQUIRED_INPUTS)
    def test_non_finite_inputs_fail_fast(
        self,
        functor: AutotrophicRespiration,
        key: str,
    ) -> None:
        with pytest.raises(ValueError, match=key):
            functor.forward(
                _state(**{key: torch.tensor(float("nan"))}),
                0.0,
                1.0,
            )

    @pytest.mark.parametrize("key", REQUIRED_INPUTS)
    def test_negative_inputs_fail_fast(
        self,
        functor: AutotrophicRespiration,
        key: str,
    ) -> None:
        with pytest.raises(ValueError, match=key):
            functor.forward(
                _state(**{key: torch.tensor(-1.0)}),
                0.0,
                1.0,
            )

    def test_forward_is_differentiable_and_does_not_mutate_state(
        self,
        functor: AutotrophicRespiration,
    ) -> None:
        maintenance = torch.tensor(3.0e-8, dtype=torch.float64, requires_grad=True)
        growth = torch.tensor(2.0e-8, dtype=torch.float64, requires_grad=True)
        state = {
            "gross_primary_productivity": torch.tensor(1.0e-7, dtype=torch.float64),
            "maintenance_respiration_total": maintenance,
            "growth_respiration_total": growth,
        }
        state_keys = set(state)

        output = functor.forward(state, time=0.0, dt=3600.0)
        _value(output, "net_primary_productivity").backward()

        assert set(state) == state_keys
        assert _value(output, "total_autotrophic_respiration").dtype == torch.float64
        assert maintenance.grad is not None
        assert growth.grad is not None

    def test_source_has_no_aggregate_alias_or_carbon_clip(self) -> None:
        source = Path(
            "relic/scientific/biosphere/respiration/autotrophic_respiration.py"
        ).read_text(encoding="utf-8")

        assert '"rate"' not in source
        assert '"autotrophic_respiration_rate"' not in source
        assert "carbon_limit" not in source
        assert "limit_factor" not in source
        assert "_compute_rate_core" not in source
