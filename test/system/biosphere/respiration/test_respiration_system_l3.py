"""L3 system tests for the canonical plant respiration carbon ledger."""

from __future__ import annotations

import math

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


def _value(output: dict[str, UpdateItem], key: str) -> torch.Tensor:
    item = output[key]
    assert isinstance(item, UpdateItem)
    return item.value


def _state(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    state = {
        "gross_primary_productivity": torch.tensor(1.5e-7),
        "air_temperature": torch.tensor(298.15),
        "growing_temperature": torch.tensor(298.15),
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


class RespirationSystem:
    """Small scheduler-equivalent harness using committed tensor state."""

    def __init__(
        self,
        *,
        pft_name: str | None = None,
        enable_acclimation: bool = False,
    ) -> None:
        self.maintenance = MaintenanceRespiration(
            pft_name=pft_name,
            enable_acclimation=enable_acclimation,
        )
        self.growth = GrowthRespiration(pft_name=pft_name)
        self.aggregate = AutotrophicRespiration(pft_name=pft_name)

    def step(
        self,
        state: dict[str, torch.Tensor],
        *,
        dt: float,
    ) -> dict[str, dict[str, UpdateItem]]:
        scheduled = dict(state)
        maintenance = self.maintenance.forward(scheduled, 0.0, dt)
        scheduled["maintenance_respiration_total"] = _value(
            maintenance,
            "maintenance_respiration_total",
        )
        growth = self.growth.forward(scheduled, 0.0, dt)
        scheduled["growth_respiration_total"] = _value(
            growth,
            "growth_respiration_total",
        )
        aggregate = self.aggregate.forward(scheduled, 0.0, dt)
        return {
            "maintenance": maintenance,
            "growth": growth,
            "aggregate": aggregate,
        }


def _ledger(
    state: dict[str, torch.Tensor],
    output: dict[str, dict[str, UpdateItem]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aggregate = output["aggregate"]
    respiration = _value(aggregate, "total_autotrophic_respiration")
    npp = _value(aggregate, "net_primary_productivity")
    residual = state["gross_primary_productivity"] - respiration - npp
    return respiration, npp, residual


class TestRespirationSystemL3:
    """Exercise multi-step and multi-PFT system behavior."""

    def test_gpp_respiration_npp_coupling(self) -> None:
        state = _state(gross_primary_productivity=torch.tensor(1.538169e-7))
        output = RespirationSystem().step(state, dt=60.0)
        respiration, npp, residual = _ledger(state, output)
        rm = _value(output["maintenance"], "maintenance_respiration_total")
        rg = _value(output["growth"], "growth_respiration_total")

        torch.testing.assert_close(respiration, rm + rg, rtol=0.0, atol=0.0)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=1.0e-14)
        assert npp > 0.0

    def test_diurnal_cycle_preserves_every_step_ledger(self) -> None:
        system = RespirationSystem()
        residuals = []
        respiration_rates = []
        for hour in range(24):
            phase = 2.0 * math.pi * (hour - 6) / 24.0
            temperature = 298.15 + 8.0 * math.sin(phase)
            daylight = max(math.sin(math.pi * (hour - 6) / 12.0), 0.0)
            state = _state(
                air_temperature=torch.tensor(temperature),
                gross_primary_productivity=torch.tensor(1.8e-7 * daylight),
            )
            output = system.step(state, dt=3600.0)
            respiration, _, residual = _ledger(state, output)
            residuals.append(residual)
            respiration_rates.append(respiration)

        stacked = torch.stack(residuals)
        torch.testing.assert_close(
            stacked,
            torch.zeros_like(stacked),
            rtol=0.0,
            atol=1.0e-14,
        )
        assert torch.stack(respiration_rates).max() > torch.stack(respiration_rates).min()

    def test_explicit_seasonal_acclimation_changes_maintenance_flux(self) -> None:
        system = RespirationSystem(enable_acclimation=True)
        cool = system.step(
            _state(growing_temperature=torch.tensor(288.15)),
            dt=86400.0,
        )
        warm = system.step(
            _state(growing_temperature=torch.tensor(308.15)),
            dt=86400.0,
        )
        cool_rm = _value(cool["maintenance"], "maintenance_respiration_total")
        warm_rm = _value(warm["maintenance"], "maintenance_respiration_total")
        assert cool_rm > warm_rm

    def test_drought_response_reduces_system_respiration(self) -> None:
        system = RespirationSystem()
        wet_state = _state(soil_water_potential=torch.tensor(-0.1))
        dry_state = _state(soil_water_potential=torch.tensor(-1.4))
        wet, _, _ = _ledger(wet_state, system.step(wet_state, dt=3600.0))
        dry, _, _ = _ledger(dry_state, system.step(dry_state, dt=3600.0))
        assert wet > dry

    @pytest.mark.parametrize(
        "pft_name",
        [
            "broadleaf_deciduous_temperate",
            "needleleaf_evergreen_temperate",
            "c3_grass_temperate",
            "c4_grass_temperate",
            "shrub_deciduous_temperate",
        ],
    )
    def test_multi_pft_ecosystem_has_finite_exact_ledgers(
        self,
        pft_name: str,
    ) -> None:
        system = RespirationSystem(pft_name=pft_name)
        state = _state()
        output = system.step(state, dt=3600.0)
        respiration, npp, residual = _ledger(state, output)
        assert torch.isfinite(respiration)
        assert torch.isfinite(npp)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=1.0e-14)

    def test_daily_integrated_mass_ledger(self) -> None:
        system = RespirationSystem()
        dt = 3600.0
        cumulative_gpp = torch.tensor(0.0)
        cumulative_respiration = torch.tensor(0.0)
        cumulative_npp = torch.tensor(0.0)
        for hour in range(24):
            daylight = max(math.sin(math.pi * (hour - 6) / 12.0), 0.0)
            state = _state(
                gross_primary_productivity=torch.tensor(1.8e-7 * daylight)
            )
            respiration, npp, _ = _ledger(
                state,
                system.step(state, dt=dt),
            )
            cumulative_gpp = cumulative_gpp + state["gross_primary_productivity"] * dt
            cumulative_respiration = cumulative_respiration + respiration * dt
            cumulative_npp = cumulative_npp + npp * dt

        residual = cumulative_gpp - cumulative_respiration - cumulative_npp
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=1.0e-10)

    def test_biomass_sensitivity_is_monotonic(self) -> None:
        system = RespirationSystem()
        low_state = _state(
            biomass_leaf=torch.tensor(100.0),
            biomass_stem=torch.tensor(250.0),
            biomass_root=torch.tensor(75.0),
        )
        high_state = _state(
            biomass_leaf=torch.tensor(300.0),
            biomass_stem=torch.tensor(750.0),
            biomass_root=torch.tensor(225.0),
        )
        low, _, _ = _ledger(low_state, system.step(low_state, dt=3600.0))
        high, _, _ = _ledger(high_state, system.step(high_state, dt=3600.0))
        assert high > low

    @pytest.mark.parametrize("temperature", [233.15, 333.15])
    def test_supported_temperature_extremes_are_finite(
        self,
        temperature: float,
    ) -> None:
        system = RespirationSystem()
        state = _state(air_temperature=torch.tensor(temperature))
        respiration, npp, residual = _ledger(
            state,
            system.step(state, dt=3600.0),
        )
        assert torch.isfinite(respiration)
        assert torch.isfinite(npp)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=1.0e-14)
