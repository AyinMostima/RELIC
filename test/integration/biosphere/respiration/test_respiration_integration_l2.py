"""L2 parameter and response integration tests for plant respiration."""

from __future__ import annotations

import pytest
import torch

from relic.core.state import UpdateItem
from relic.scientific.biosphere.respiration.autotrophic_respiration import (
    AutotrophicRespiration,
)
from relic.scientific.biosphere.respiration.growth_respiration import (
    GrowthRespiration,
)
from relic.scientific.biosphere.respiration.maintenance_respiration import (
    MaintenanceRespiration,
)


PFT_NAMES = [
    "broadleaf_deciduous_temperate",
    "needleleaf_evergreen_temperate",
    "c3_grass_temperate",
    "c4_grass_temperate",
    "shrub_deciduous_temperate",
]


def _state(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    state = {
        "gross_primary_productivity": torch.tensor(1.5e-7),
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
    pft_name: str | None = None,
    dt: float = 3600.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scheduled = dict(state)
    maintenance = MaintenanceRespiration(
        pft_name=pft_name,
        enable_acclimation=False,
    )
    maintenance_output = maintenance.forward(scheduled, 0.0, dt)
    rm = _value(maintenance_output, "maintenance_respiration_total")
    scheduled["maintenance_respiration_total"] = rm

    growth = GrowthRespiration(pft_name=pft_name)
    growth_output = growth.forward(scheduled, 0.0, dt)
    rg = _value(growth_output, "growth_respiration_total")
    scheduled["growth_respiration_total"] = rg

    aggregate = AutotrophicRespiration(pft_name=pft_name)
    aggregate_output = aggregate.forward(scheduled, 0.0, dt)
    ra = _value(aggregate_output, "total_autotrophic_respiration")
    npp = _value(aggregate_output, "net_primary_productivity")
    return rm, rg, ra, npp


class TestRespirationIntegrationL2:
    """Verify real parameters and cross-process response propagation."""

    @pytest.mark.parametrize("pft_name", PFT_NAMES)
    def test_maintenance_parameters_load_for_pft_fallback(
        self,
        pft_name: str,
    ) -> None:
        model = MaintenanceRespiration(pft_name=pft_name)
        assert model.rm_leaf_25 > 0.0
        assert model.rm_stem_25 > 0.0
        assert model.rm_root_25 > 0.0
        assert model.rm_leaf_25 >= model.rm_root_25 >= model.rm_stem_25
        assert 1.5 < model.q10_respiration < 3.0

    @pytest.mark.parametrize("pft_name", PFT_NAMES)
    def test_growth_parameters_load_for_pft_fallback(
        self,
        pft_name: str,
    ) -> None:
        model = GrowthRespiration(pft_name=pft_name)
        assert 0.4 < model.ygrow_base < 0.8
        assert 1.0 < model.q10_growth < 3.0
        assert 0.0 < model.np_response_coeff < 0.5
        assert 0.01 < model.np_half_sat < 0.2

    @pytest.mark.parametrize("pft_name", PFT_NAMES)
    def test_full_chain_closes_for_each_pft(self, pft_name: str) -> None:
        rm, rg, ra, npp = _run_chain(_state(), pft_name=pft_name)
        torch.testing.assert_close(ra, rm + rg, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            _state()["gross_primary_productivity"] - ra - npp,
            torch.zeros_like(ra),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_temperature_response_is_consistent(self) -> None:
        rates = []
        for temperature in (288.15, 298.15, 308.15):
            _, _, ra, _ = _run_chain(
                _state(air_temperature=torch.tensor(temperature))
            )
            rates.append(ra)
        assert rates[0] < rates[1] < rates[2]

    def test_water_stress_reduces_maintenance_and_total_respiration(self) -> None:
        rates = []
        for potential in (-0.1, -0.5, -1.0, -1.5):
            rm, _, ra, _ = _run_chain(
                _state(soil_water_potential=torch.tensor(potential))
            )
            rates.append((rm, ra))
        for previous, current in zip(rates, rates[1:]):
            assert previous[0] >= current[0]
            assert previous[1] >= current[1]

    def test_nutrient_ratio_changes_growth_component(self) -> None:
        _, low_n_growth, _, _ = _run_chain(
            _state(plant_nitrogen=torch.tensor(0.0))
        )
        _, high_n_growth, _, _ = _run_chain(
            _state(plant_nitrogen=torch.tensor(3.0))
        )
        assert low_n_growth > high_n_growth

    @pytest.mark.parametrize("gpp", [0.0, 1.0e-10, 1.0e-7, 2.0e-7])
    def test_gpp_range_preserves_exact_ledger(self, gpp: float) -> None:
        state = _state(gross_primary_productivity=torch.tensor(gpp))
        rm, rg, ra, npp = _run_chain(state)
        torch.testing.assert_close(ra, rm + rg, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            state["gross_primary_productivity"] - ra - npp,
            torch.zeros_like(ra),
            rtol=0.0,
            atol=1.0e-14,
        )
        if gpp == 0.0:
            assert npp < 0.0

    def test_combined_cold_drought_low_substrate_stress_is_finite(self) -> None:
        rm, rg, ra, npp = _run_chain(
            _state(
                gross_primary_productivity=torch.tensor(1.0e-9),
                air_temperature=torch.tensor(278.15),
                biomass_leaf=torch.tensor(50.0),
                biomass_stem=torch.tensor(100.0),
                biomass_root=torch.tensor(80.0),
                soluble_sugar=torch.tensor(2.0),
                soil_water_potential=torch.tensor(-1.3),
                plant_nitrogen=torch.tensor(1.0),
                plant_phosphorus=torch.tensor(0.05),
            ),
            pft_name="shrub_deciduous_temperate",
        )
        for value in (rm, rg, ra, npp):
            assert torch.isfinite(value)
        assert rm >= 0.0
        assert rg >= 0.0
        torch.testing.assert_close(ra, rm + rg, rtol=0.0, atol=0.0)
