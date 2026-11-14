"""L2 integration tests for the canonical plant respiration chain."""

from __future__ import annotations

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.respiration.autotrophic_respiration import (
    AutotrophicRespiration,
)
from relic.scientific.biosphere.respiration.growth_respiration import (
    GrowthRespiration,
)
from relic.scientific.biosphere.respiration.maintenance_respiration import (
    MaintenanceRespiration,
)


def _state(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    state = {
        "gross_primary_productivity": torch.tensor(1.538169e-7),
        "air_temperature": torch.tensor(298.15),
        "biomass_leaf": torch.tensor(200.0),
        "biomass_stem": torch.tensor(500.0),
        "biomass_root": torch.tensor(150.0),
        "soluble_sugar": torch.tensor(50.0),
        "soil_water_potential": torch.tensor(-0.5),
        "plant_nitrogen": torch.tensor(2.0),
        "plant_phosphorus": torch.tensor(0.2),
    }
    state.update(overrides)
    return state


def _value(output: dict[str, UpdateItem], key: str) -> torch.Tensor:
    item = output[key]
    assert isinstance(item, UpdateItem)
    return item.value


def _run_chain(
    state: dict[str, torch.Tensor],
    *,
    dt: float = 60.0,
) -> tuple[
    dict[str, UpdateItem],
    dict[str, UpdateItem],
    dict[str, UpdateItem],
]:
    scheduled_state = dict(state)
    maintenance = MaintenanceRespiration(enable_acclimation=False)
    maintenance_output = maintenance.forward(scheduled_state, 0.0, dt)
    scheduled_state["maintenance_respiration_total"] = _value(
        maintenance_output,
        "maintenance_respiration_total",
    )

    growth = GrowthRespiration()
    growth_output = growth.forward(scheduled_state, 0.0, dt)
    scheduled_state["growth_respiration_total"] = _value(
        growth_output,
        "growth_respiration_total",
    )

    aggregate = AutotrophicRespiration()
    aggregate_output = aggregate.forward(scheduled_state, 0.0, dt)
    return maintenance_output, growth_output, aggregate_output


class TestAutotrophicRespirationIntegration:
    """Verify scheduler-equivalent component composition and carbon closure."""

    def test_component_dependencies_are_state_driven(self) -> None:
        aggregate = AutotrophicRespiration()
        assert aggregate.get_dependencies() == [
            "maintenance_respiration",
            "growth_respiration",
        ]
        assert not hasattr(aggregate, "maintenance_respiration")
        assert not hasattr(aggregate, "growth_respiration")

    def test_scheduled_chain_closes_exact_carbon_ledger(self) -> None:
        state = _state()
        maintenance, growth, aggregate = _run_chain(state)

        rm = _value(maintenance, "maintenance_respiration_total")
        rg = _value(growth, "growth_respiration_total")
        ra = _value(aggregate, "total_autotrophic_respiration")
        npp = _value(aggregate, "net_primary_productivity")
        residual = state["gross_primary_productivity"] - ra - npp

        torch.testing.assert_close(ra, rm + rg, rtol=0.0, atol=0.0)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=1.0e-14)
        torch.testing.assert_close(
            _value(
                aggregate,
                "diagnostics.autotrophic_respiration.carbon_balance_residual",
            ),
            residual,
            rtol=0.0,
            atol=0.0,
        )

    def test_component_diagnostics_preserve_committed_fluxes(self) -> None:
        maintenance, growth, aggregate = _run_chain(_state())
        torch.testing.assert_close(
            _value(
                aggregate,
                "diagnostics.autotrophic_respiration.maintenance_component",
            ),
            _value(maintenance, "maintenance_respiration_total"),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            _value(
                aggregate,
                "diagnostics.autotrophic_respiration.growth_component",
            ),
            _value(growth, "growth_respiration_total"),
            rtol=0.0,
            atol=0.0,
        )

    def test_chain_uses_canonical_units_modes_and_writers(self) -> None:
        maintenance, growth, aggregate = _run_chain(_state())
        checks = [
            (
                maintenance["maintenance_respiration_total"],
                "maintenance_respiration",
            ),
            (growth["growth_respiration_total"], "growth_respiration"),
            (
                aggregate["total_autotrophic_respiration"],
                "autotrophic_respiration",
            ),
            (aggregate["net_primary_productivity"], "autotrophic_respiration"),
        ]
        for item, source in checks:
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == "kg C m-2 s-1"
            assert item.source_functor == source

    @pytest.mark.parametrize("temperature", [288.15, 298.15, 308.15])
    def test_temperature_response_propagates_through_components(
        self,
        temperature: float,
    ) -> None:
        _, _, output = _run_chain(
            _state(air_temperature=torch.tensor(temperature))
        )
        value = _value(output, "total_autotrophic_respiration")
        assert torch.isfinite(value)
        assert value >= 0.0

        if temperature > 298.15:
            _, _, reference = _run_chain(_state())
            assert value > _value(reference, "total_autotrophic_respiration")

    @pytest.mark.parametrize("substrate", [0.0, 1.0, 10.0, 100.0])
    def test_substrate_response_propagates_through_components(
        self,
        substrate: float,
    ) -> None:
        _, _, output = _run_chain(
            _state(soluble_sugar=torch.tensor(substrate))
        )
        respiration = _value(output, "total_autotrophic_respiration")
        assert torch.isfinite(respiration)
        assert respiration >= 0.0
        if substrate == 0.0:
            torch.testing.assert_close(
                respiration,
                torch.zeros_like(respiration),
                rtol=0.0,
                atol=0.0,
            )

    def test_low_gpp_retains_negative_npp_without_rescaling_components(self) -> None:
        state = _state(gross_primary_productivity=torch.tensor(1.0e-10))
        maintenance, growth, aggregate = _run_chain(state)
        rm = _value(maintenance, "maintenance_respiration_total")
        rg = _value(growth, "growth_respiration_total")
        ra = _value(aggregate, "total_autotrophic_respiration")
        npp = _value(aggregate, "net_primary_productivity")

        torch.testing.assert_close(ra, rm + rg, rtol=0.0, atol=0.0)
        assert npp < 0.0

    def test_rate_chain_is_dt_invariant(self) -> None:
        one_second = _run_chain(_state(), dt=1.0)
        one_hour = _run_chain(_state(), dt=3600.0)
        for first, second in zip(one_second, one_hour):
            assert set(first) == set(second)
            for key in first:
                torch.testing.assert_close(
                    _value(first, key),
                    _value(second, key),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_vector_state_preserves_shape_and_ledger(self) -> None:
        state = _state(
            gross_primary_productivity=torch.tensor([1.5e-7, 2.0e-7]),
            air_temperature=torch.tensor([298.15, 303.15]),
            biomass_leaf=torch.tensor([200.0, 220.0]),
            biomass_stem=torch.tensor([500.0, 550.0]),
            biomass_root=torch.tensor([150.0, 165.0]),
            soluble_sugar=torch.tensor([50.0, 60.0]),
            soil_water_potential=torch.tensor([-0.5, -0.4]),
            plant_nitrogen=torch.tensor([2.0, 2.2]),
            plant_phosphorus=torch.tensor([0.2, 0.22]),
        )
        _, _, output = _run_chain(state)
        ra = _value(output, "total_autotrophic_respiration")
        npp = _value(output, "net_primary_productivity")
        assert ra.shape == torch.Size([2])
        torch.testing.assert_close(
            state["gross_primary_productivity"] - ra - npp,
            torch.zeros_like(ra),
            rtol=0.0,
            atol=1.0e-14,
        )
