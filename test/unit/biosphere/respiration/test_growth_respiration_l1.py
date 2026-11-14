"""L1 tests for the canonical growth-respiration carbon flux."""

from __future__ import annotations

import pytest
import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.scientific.biosphere.respiration.growth_respiration import (
    GrowthRespiration,
)


REQUIRED_INPUTS = [
    "gross_primary_productivity",
    "maintenance_respiration_total",
    "air_temperature",
    "plant_nitrogen",
    "plant_phosphorus",
    "soluble_sugar",
]
OUTPUTS = [
    "growth_respiration_total",
    "diagnostics.growth_respiration.growth_efficiency",
    "diagnostics.growth_respiration.available_carbon",
    "diagnostics.growth_respiration.np_ratio_effect",
    "diagnostics.growth_respiration.temperature_factor",
    "diagnostics.growth_respiration.substrate_factor",
    "diagnostics.growth_respiration.carbon_limit_residual",
]


def _state(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    state = {
        "gross_primary_productivity": torch.tensor(1.5e-7),
        "maintenance_respiration_total": torch.tensor(3.6163690476190476e-8),
        "air_temperature": torch.tensor(298.15),
        "plant_nitrogen": torch.tensor(2.0),
        "plant_phosphorus": torch.tensor(0.2),
        "soluble_sugar": torch.tensor(50.0),
    }
    state.update(overrides)
    return state


def _value(output: dict[str, UpdateItem], key: str) -> torch.Tensor:
    item = output[key]
    assert isinstance(item, UpdateItem)
    assert isinstance(item.value, torch.Tensor)
    return item.value


@pytest.fixture
def growth_respiration() -> GrowthRespiration:
    return GrowthRespiration()


class TestGrowthRespirationL1:
    """Verify the single kg-C growth-respiration contract."""

    def test_initialization_and_metadata(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        assert isinstance(growth_respiration, torch.nn.Module)
        assert growth_respiration.metadata.input_variables == REQUIRED_INPUTS
        assert growth_respiration.get_required_inputs() == REQUIRED_INPUTS
        assert growth_respiration.metadata.output_variables == OUTPUTS
        assert growth_respiration.get_provides() == OUTPUTS
        assert growth_respiration.get_dependencies() == [
            "farquhar_photosynthesis",
            "maintenance_respiration",
        ]
        assert not hasattr(growth_respiration, "_compute_rate_core")

    def test_forward_exact_contract_and_fixed_numerics(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        output = growth_respiration.forward(_state(), time=0.0, dt=3600.0)

        assert list(output) == OUTPUTS
        assert set(output) == set(growth_respiration.metadata.output_variables)
        for key, item in output.items():
            assert isinstance(item, UpdateItem), key
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.source_functor == "growth_respiration"

        expected = {
            "growth_respiration_total": 2.58718912959921e-8,
            "diagnostics.growth_respiration.growth_efficiency": 0.75,
            "diagnostics.growth_respiration.available_carbon": 1.1383631459693788e-7,
            "diagnostics.growth_respiration.np_ratio_effect": 0.19867433607578278,
            "diagnostics.growth_respiration.temperature_factor": 1.0,
            "diagnostics.growth_respiration.substrate_factor": 50.0 / 55.0,
            "diagnostics.growth_respiration.carbon_limit_residual": 0.0,
        }
        for key, expected_value in expected.items():
            torch.testing.assert_close(
                _value(output, key),
                torch.tensor(expected_value),
                rtol=2.0e-6,
                atol=1.0e-14,
            )

        expected_units = {
            "growth_respiration_total": "kg C m-2 s-1",
            "diagnostics.growth_respiration.growth_efficiency": "1",
            "diagnostics.growth_respiration.available_carbon": "kg C m-2 s-1",
            "diagnostics.growth_respiration.np_ratio_effect": "1",
            "diagnostics.growth_respiration.temperature_factor": "1",
            "diagnostics.growth_respiration.substrate_factor": "1",
            "diagnostics.growth_respiration.carbon_limit_residual": "kg C m-2 s-1",
        }
        assert {key: item.unit for key, item in output.items()} == expected_units

    def test_nonincrementable_flux_commits_through_state_manager(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        """Route the instantaneous carbon flux without a storage-element hint."""
        output = growth_respiration.forward(_state(), time=0.0, dt=3600.0)
        key = "growth_respiration_total"
        item = output[key]

        manager = StateManager(device="cpu")
        registry = manager._contract_registry
        assert isinstance(registry, ContractRegistry)
        contract = registry.get_contract(key)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.incrementable is False
        assert contract.raw["accumulates"] is None

        manager.set_state({key: torch.tensor(0.0)}, validate=False)
        committed = manager.apply_updates(
            {key: item},
            dt=3600.0,
            check_conservation=False,
            update_sources={key: growth_respiration.name},
        )

        assert item.element is None
        torch.testing.assert_close(committed[key], item.value, rtol=0.0, atol=0.0)

    def test_temperature_response_before_carbon_limit(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        at_25 = growth_respiration.forward(_state(), 0.0, 1.0)
        at_35 = growth_respiration.forward(
            _state(air_temperature=torch.tensor(308.15)),
            0.0,
            1.0,
        )
        at_15 = growth_respiration.forward(
            _state(air_temperature=torch.tensor(288.15)),
            0.0,
            1.0,
        )

        total = "growth_respiration_total"
        torch.testing.assert_close(
            _value(at_35, total) / _value(at_25, total),
            torch.tensor(2.0),
            rtol=2.0e-6,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            _value(at_15, total) / _value(at_25, total),
            torch.tensor(0.5),
            rtol=2.0e-6,
            atol=1.0e-7,
        )

    def test_nutrient_ratio_changes_growth_efficiency(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        low_ratio = growth_respiration.forward(
            _state(plant_nitrogen=torch.tensor(0.0)),
            0.0,
            1.0,
        )
        high_ratio = growth_respiration.forward(
            _state(plant_nitrogen=torch.tensor(2.0)),
            0.0,
            1.0,
        )

        efficiency = "diagnostics.growth_respiration.growth_efficiency"
        assert _value(high_ratio, efficiency) > _value(low_ratio, efficiency)
        assert _value(high_ratio, "growth_respiration_total") < _value(
            low_ratio,
            "growth_respiration_total",
        )

    def test_substrate_limitation_and_zero_substrate_closure(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        high = growth_respiration.forward(
            _state(soluble_sugar=torch.tensor(100.0)),
            0.0,
            1.0,
        )
        low = growth_respiration.forward(
            _state(soluble_sugar=torch.tensor(1.0)),
            0.0,
            1.0,
        )
        zero = growth_respiration.forward(
            _state(soluble_sugar=torch.tensor(0.0)),
            0.0,
            1.0,
        )

        total = "growth_respiration_total"
        assert _value(high, total) > _value(low, total)
        assert _value(low, total) > _value(zero, total)
        torch.testing.assert_close(
            _value(zero, total),
            torch.tensor(0.0),
            rtol=0.0,
            atol=0.0,
        )

    def test_growth_flux_is_bounded_by_available_carbon(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        output = growth_respiration.forward(
            _state(
                air_temperature=torch.tensor(328.15),
                plant_nitrogen=torch.tensor(0.0),
                soluble_sugar=torch.tensor(1000.0),
            ),
            0.0,
            1.0,
        )

        available = _value(
            output,
            "diagnostics.growth_respiration.available_carbon",
        )
        respiration = _value(output, "growth_respiration_total")
        residual = _value(
            output,
            "diagnostics.growth_respiration.carbon_limit_residual",
        )
        torch.testing.assert_close(respiration, available, rtol=0.0, atol=0.0)
        assert residual > 0.0

    def test_no_available_carbon_has_zero_growth_respiration(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        output = growth_respiration.forward(
            _state(
                gross_primary_productivity=torch.tensor(1.0e-8),
                maintenance_respiration_total=torch.tensor(2.0e-8),
            ),
            0.0,
            1.0,
        )
        for key in (
            "growth_respiration_total",
            "diagnostics.growth_respiration.available_carbon",
            "diagnostics.growth_respiration.carbon_limit_residual",
        ):
            torch.testing.assert_close(
                _value(output, key),
                torch.tensor(0.0),
                rtol=0.0,
                atol=0.0,
            )

    def test_rate_outputs_are_dt_invariant_and_replace_only(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        one_second = growth_respiration.forward(_state(), 0.0, 1.0)
        one_hour = growth_respiration.forward(_state(), 0.0, 3600.0)

        for key in OUTPUTS:
            torch.testing.assert_close(
                _value(one_second, key),
                _value(one_hour, key),
                rtol=0.0,
                atol=0.0,
            )
            assert one_hour[key].mode is UpdateMode.REPLACE
            assert one_hour[key].per_timestep is False

    @pytest.mark.parametrize("missing_key", REQUIRED_INPUTS)
    def test_required_inputs_fail_fast(
        self,
        growth_respiration: GrowthRespiration,
        missing_key: str,
    ) -> None:
        state = _state()
        state.pop(missing_key)
        with pytest.raises(KeyError, match=missing_key):
            growth_respiration.forward(state, 0.0, 1.0)

    @pytest.mark.parametrize(
        ("canonical_key", "legacy_key"),
        [
            ("gross_primary_productivity", "gpp"),
            ("air_temperature", "temperature"),
            ("plant_nitrogen", "n_content_leaf"),
            ("plant_phosphorus", "p_content_leaf"),
        ],
    )
    def test_legacy_input_aliases_are_rejected(
        self,
        growth_respiration: GrowthRespiration,
        canonical_key: str,
        legacy_key: str,
    ) -> None:
        state = _state()
        state[legacy_key] = state.pop(canonical_key)
        with pytest.raises(KeyError, match=canonical_key):
            growth_respiration.forward(state, 0.0, 1.0)

    @pytest.mark.parametrize("key", REQUIRED_INPUTS)
    def test_non_finite_inputs_fail_fast(
        self,
        growth_respiration: GrowthRespiration,
        key: str,
    ) -> None:
        with pytest.raises(ValueError, match=key):
            growth_respiration.forward(
                _state(**{key: torch.tensor(float("nan"))}),
                0.0,
                1.0,
            )

    @pytest.mark.parametrize(
        "key",
        [
            "gross_primary_productivity",
            "maintenance_respiration_total",
            "plant_nitrogen",
            "plant_phosphorus",
            "soluble_sugar",
        ],
    )
    def test_negative_mass_inputs_fail_fast(
        self,
        growth_respiration: GrowthRespiration,
        key: str,
    ) -> None:
        with pytest.raises(ValueError, match=key):
            growth_respiration.forward(
                _state(**{key: torch.tensor(-1.0)}),
                0.0,
                1.0,
            )

    def test_nonpositive_kelvin_temperature_fails_fast(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        with pytest.raises(ValueError, match="air_temperature"):
            growth_respiration.forward(
                _state(air_temperature=torch.tensor(0.0)),
                0.0,
                1.0,
            )

    def test_parameter_units_are_loaded_from_yaml(
        self,
        growth_respiration: GrowthRespiration,
    ) -> None:
        manager = growth_respiration.get_parameter_manager()
        expected_units = {
            "biosphere.respiration.growth.ygrow_base": "dimensionless",
            "biosphere.respiration.growth.np_response_coeff": "dimensionless",
            "biosphere.respiration.growth.np_half_sat": "dimensionless",
            "biosphere.respiration.growth.q10_growth": "dimensionless",
            "biosphere.respiration.growth.km_substrate_growth": "g C m-2",
            "biosphere.respiration.growth.min_growth_efficiency": "dimensionless",
            "biosphere.respiration.growth.max_growth_efficiency": "dimensionless",
            "biosphere.validation.epsilon.nutrient": "g P m-2",
        }
        for key, units in expected_units.items():
            spec = manager.get_parameter(key)
            assert spec is not None
            assert spec.units == units


def test_growth_respiration_passes_and_assigns_no_bridge_label():
    """12.16: no algorithm reads the BiologicalProcess temperature_response or
    ph_dependence labels (the Q10 factor calls q10_temperature_response directly), so the module neither passes nor assigns them."""
    import inspect

    from relic.scientific.biosphere.respiration import growth_respiration

    source = inspect.getsource(growth_respiration)
    for label in ("temperature_response", "ph_dependence"):
        assert f"self.{label}" not in source
        assert f"{label}=" not in source
