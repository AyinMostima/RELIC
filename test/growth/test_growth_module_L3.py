"""L3 system tests for the canonical biosphere growth chain.

The scenarios exercise annual carbon routing, drought allocation, structural
observations, seasonal phenology, and PFT-specific allocation without creating
test-only Functor metadata or a second state-update path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pytest
import torch
from torch import Tensor

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from relic.utils.time_constants import SECONDS_PER_DAY


BIOMASS_STATE_KEYS = (
    "leaf_biomass_c",
    "wood_biomass_c",
    "root_biomass_c",
)
GROWTH_RATE_KEYS = (
    "growth_rate_leaf",
    "growth_rate_stem",
    "growth_rate_root",
)
STRUCTURAL_OUTPUTS = {
    "diagnostics.structural_growth.gross_leaf_area_growth_tendency",
    "diagnostics.structural_growth.leaf_area_senescence_tendency",
    "diagnostics.structural_growth.net_leaf_area_tendency",
    "diagnostics.structural_growth.step_leaf_area_delta",
}


@dataclass(frozen=True)
class GrowthSystem:
    """Canonical growth Functors shared by the L3 scenarios."""

    device: torch.device
    phenology: PhenologyGrowth
    allocation: BiomassAllocation
    structural: StructuralGrowth


@pytest.fixture(scope="module")
def growth_system() -> GrowthSystem:
    """Construct production Functors on a deterministic CPU device."""
    device = torch.device("cpu")
    pft_name = "broadleaf_deciduous_temperate"
    return GrowthSystem(
        device=device,
        phenology=PhenologyGrowth(
            device=device,
            phenology_model_owner="gsi_phenology",
        ),
        allocation=BiomassAllocation(
            name="l3_biomass_allocation",
            pft_name=pft_name,
            device=device,
        ),
        structural=StructuralGrowth(
            name="l3_structural_observer",
            device=device,
        ),
    )


def _scalar(value: Tensor) -> float:
    """Return one detached scalar for scenario-level assertions."""
    return float(value.detach().cpu().item())


def _allocation_state(
    system: GrowthSystem,
    *,
    npp: float,
    water_stress_factor: float,
    phenology_phase: float | None = None,
    leaf_area_index: float = 3.0,
    air_temperature: float = 298.15,
) -> dict[str, Tensor]:
    """Build one canonical kg-C allocation state without runtime aliases."""
    phase = (
        float(system.allocation.MATURE_PHASE)
        if phenology_phase is None
        else phenology_phase
    )
    return {
        "net_primary_productivity": torch.tensor(
            npp, device=system.device, dtype=torch.float32
        ),
        "air_temperature": torch.tensor(
            air_temperature, device=system.device, dtype=torch.float32
        ),
        "water_stress_factor": torch.tensor(
            water_stress_factor, device=system.device, dtype=torch.float32
        ),
        "leaf_area_index": torch.tensor(
            leaf_area_index, device=system.device, dtype=torch.float32
        ),
        "phenology_phase": torch.tensor(
            phase, device=system.device, dtype=torch.float32
        ),
        "leaf_biomass_c": torch.tensor(
            0.30, device=system.device, dtype=torch.float32
        ),
        "wood_biomass_c": torch.tensor(
            5.00, device=system.device, dtype=torch.float32
        ),
        "root_biomass_c": torch.tensor(
            1.50, device=system.device, dtype=torch.float32
        ),
    }


class TestGrowthSystemL3:
    """L3 growth scenarios using canonical ports and update semantics."""

    def test_annual_growth_cycle(self, growth_system: GrowthSystem) -> None:
        """Run 52 weekly steps with seasonal phenology and exact carbon routing."""
        system = growth_system
        week_seconds = 7.0 * SECONDS_PER_DAY
        leaf_area_index = torch.tensor(3.0, device=system.device)
        initial_lai = leaf_area_index.clone()
        stocks = {
            "leaf_biomass_c": torch.tensor(0.30, device=system.device),
            "wood_biomass_c": torch.tensor(5.00, device=system.device),
            "root_biomass_c": torch.tensor(1.50, device=system.device),
        }
        initial_total = torch.stack(tuple(stocks.values())).sum()
        integrated_npp = torch.zeros((), device=system.device)
        phenology_factors: list[float] = []

        for week in range(52):
            seasonal_angle = 2.0 * math.pi * (week - 13.0) / 52.0
            air_temperature = 288.15 + 10.0 * math.sin(seasonal_angle)
            phenological_factor = max(
                0.0, math.sin(math.pi * (week + 0.5) / 52.0)
            )
            npp = 2.0e-7 * phenological_factor
            if week < 8 or week >= 46:
                phenology_phase = float(system.allocation.DORMANT_PHASE)
            elif week < 18:
                phenology_phase = float(system.allocation.GREENUP_PHASE)
            elif week < 38:
                phenology_phase = float(system.allocation.MATURE_PHASE)
            else:
                phenology_phase = float(system.allocation.SENESCENCE_PHASE)

            phenology_output = system.phenology.forward(
                {
                    "phenological_factor": torch.tensor(
                        phenological_factor, device=system.device
                    ),
                },
                time=week * week_seconds,
                dt=week_seconds,
            )
            assert set(phenology_output) == set(
                system.phenology.metadata.output_variables
            )
            assert "rate" not in phenology_output
            assert all(
                isinstance(item, UpdateItem)
                and item.mode is UpdateMode.REPLACE
                and item.per_timestep is False
                and item.unit == "dimensionless"
                and item.source_functor == system.phenology.name
                for item in phenology_output.values()
            )
            active = phenology_output[
                "diagnostics.phenology.growth_active"
            ].value
            assert bool(active.item()) is (
                phenological_factor > system.phenology.growth_active_threshold
            )
            phenology_factors.append(phenological_factor)
            assert "leaf_area_index" not in phenology_output

            allocation_state = _allocation_state(
                system,
                npp=npp,
                water_stress_factor=0.9,
                phenology_phase=phenology_phase,
                leaf_area_index=_scalar(leaf_area_index),
                air_temperature=air_temperature,
            )
            allocation_output = system.allocation.forward(
                allocation_state,
                time=week * week_seconds,
                dt=week_seconds,
            )
            for key in BIOMASS_STATE_KEYS:
                update = allocation_output[key]
                assert isinstance(update, UpdateItem)
                assert update.mode is UpdateMode.ACCUMULATE
                assert update.per_timestep is True
                assert update.source_functor == system.allocation.name
                stocks[key] = stocks[key] + update.value
            integrated_npp = integrated_npp + (
                allocation_state["net_primary_productivity"] * week_seconds
            )

            structural_output = system.structural.forward(
                {
                    "growth_rate_leaf": allocation_output[
                        "growth_rate_leaf"
                    ].value,
                    "leaf_area_index": leaf_area_index,
                },
                time=week * week_seconds,
                dt=week_seconds,
            )
            net_tendency = structural_output[
                "diagnostics.structural_growth.net_leaf_area_tendency"
            ].value
            step_delta = structural_output[
                "diagnostics.structural_growth.step_leaf_area_delta"
            ].value
            torch.testing.assert_close(
                step_delta,
                net_tendency * week_seconds,
                rtol=1.0e-6,
                atol=1.0e-7,
            )

        final_total = torch.stack(tuple(stocks.values())).sum()
        torch.testing.assert_close(
            final_total - initial_total,
            integrated_npp,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        torch.testing.assert_close(leaf_area_index, initial_lai)
        assert max(phenology_factors) > 0.8
        assert min(phenology_factors) < 0.2
        assert all(stocks[key] > value for key, value in zip(
            BIOMASS_STATE_KEYS,
            (0.30, 5.00, 1.50),
        ))

    def test_drought_stress_response(self, growth_system: GrowthSystem) -> None:
        """Drought redistributes fixed NPP toward roots without losing carbon."""
        system = growth_system
        dt = SECONDS_PER_DAY
        npp = 2.0e-7
        control = system.allocation.forward(
            _allocation_state(system, npp=npp, water_stress_factor=1.0),
            time=0.0,
            dt=dt,
        )
        drought = system.allocation.forward(
            _allocation_state(system, npp=npp, water_stress_factor=0.3),
            time=0.0,
            dt=dt,
        )

        root_key = "diagnostics.biomass_allocation.allocation_fraction_root"
        leaf_key = "diagnostics.biomass_allocation.allocation_fraction_leaf"
        assert drought[root_key].value > control[root_key].value
        assert drought[leaf_key].value < control[leaf_key].value
        for output in (control, drought):
            total_growth = sum(output[key].value for key in GROWTH_RATE_KEYS)
            torch.testing.assert_close(
                total_growth,
                torch.tensor(
                    npp,
                    device=system.device,
                    dtype=total_growth.dtype,
                ),
                rtol=1.0e-6,
                atol=1.0e-12,
            )

    def test_carbon_balance_annual(self, growth_system: GrowthSystem) -> None:
        """An annual step routes integrated NPP once to three biomass stocks."""
        system = growth_system
        annual_seconds = 52.0 * 7.0 * SECONDS_PER_DAY
        npp = 1.25e-7
        output = system.allocation.forward(
            _allocation_state(system, npp=npp, water_stress_factor=0.9),
            time=0.0,
            dt=annual_seconds,
        )

        biomass_gain = sum(output[key].value for key in BIOMASS_STATE_KEYS)
        expected_gain = torch.tensor(
            npp * annual_seconds,
            device=system.device,
            dtype=biomass_gain.dtype,
        )
        torch.testing.assert_close(
            biomass_gain,
            expected_gain,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        rate_sum = sum(output[key].value for key in GROWTH_RATE_KEYS)
        torch.testing.assert_close(
            rate_sum,
            torch.tensor(
                npp,
                device=system.device,
                dtype=rate_sum.dtype,
            ),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
        residual = output[
            "diagnostics.biomass_allocation.carbon_balance_residual"
        ].value
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1.0e-12, rtol=0.0)

        for key in BIOMASS_STATE_KEYS:
            assert output[key].mode is UpdateMode.ACCUMULATE
            assert output[key].per_timestep is True
            assert output[key].unit == "kg C m-2"
        for key in GROWTH_RATE_KEYS:
            assert output[key].mode is UpdateMode.REPLACE
            assert output[key].per_timestep is False
            assert output[key].unit == "kg C m-2 s-1"

        assert set(output) == set(system.allocation.metadata.output_variables)
        assert set(system.allocation.metadata.input_variables) == {
            "net_primary_productivity",
            "air_temperature",
            "water_stress_factor",
            "leaf_area_index",
            "phenology_phase",
            "leaf_biomass_c",
            "wood_biomass_c",
            "root_biomass_c",
        }

    def test_lai_dynamics(self, growth_system: GrowthSystem) -> None:
        """StructuralGrowth reports LAI tendencies without owning the LAI state."""
        system = growth_system
        dt = 7.0 * SECONDS_PER_DAY
        allocation_output = system.allocation.forward(
            _allocation_state(system, npp=2.0e-7, water_stress_factor=0.9),
            time=0.0,
            dt=dt,
        )
        leaf_area_index = torch.tensor(3.0, device=system.device)
        initial_lai = leaf_area_index.clone()
        output = system.structural.forward(
            {
                "growth_rate_leaf": allocation_output["growth_rate_leaf"].value,
                "leaf_area_index": leaf_area_index,
            },
            time=0.0,
            dt=dt,
        )

        assert set(output) == STRUCTURAL_OUTPUTS
        assert "leaf_area_index" not in output
        assert all(item.mode is UpdateMode.REPLACE for item in output.values())
        assert all(
            item.source_functor == system.structural.name
            for item in output.values()
        )
        step = output[
            "diagnostics.structural_growth.step_leaf_area_delta"
        ]
        net = output[
            "diagnostics.structural_growth.net_leaf_area_tendency"
        ]
        gross = output[
            "diagnostics.structural_growth.gross_leaf_area_growth_tendency"
        ]
        loss = output[
            "diagnostics.structural_growth.leaf_area_senescence_tendency"
        ]
        assert step.per_timestep is True
        assert net.per_timestep is False
        assert gross.per_timestep is False
        assert loss.per_timestep is False
        torch.testing.assert_close(net.value, gross.value - loss.value)
        torch.testing.assert_close(
            step.value,
            net.value * dt,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        torch.testing.assert_close(leaf_area_index, initial_lai)


def test_pft_comparison(growth_system: GrowthSystem) -> None:
    """PFT ParameterSpecs produce distinct conservative allocation patterns."""
    device = growth_system.device
    pft_names = (
        "broadleaf_deciduous_temperate",
        "needleleaf_evergreen_boreal",
        "c3_grass_temperate",
    )
    fractions = []
    npp = torch.tensor(2.0e-7, device=device, dtype=torch.float32)

    for pft_name in pft_names:
        allocation = BiomassAllocation(
            name=f"l3_pft_{pft_name}",
            pft_name=pft_name,
            enable_environmental_control=False,
            enable_phenology_control=False,
            device=device,
        )
        output = allocation.forward(
            {"net_primary_productivity": npp},
            time=0.0,
            dt=SECONDS_PER_DAY,
        )
        fraction_vector = torch.stack(
            (
                output[
                    "diagnostics.biomass_allocation.allocation_fraction_leaf"
                ].value,
                output[
                    "diagnostics.biomass_allocation.allocation_fraction_stem"
                ].value,
                output[
                    "diagnostics.biomass_allocation.allocation_fraction_root"
                ].value,
            )
        )
        fractions.append(fraction_vector)
        torch.testing.assert_close(
            fraction_vector.sum(),
            torch.ones((), device=device, dtype=fraction_vector.dtype),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            sum(output[key].value for key in GROWTH_RATE_KEYS),
            npp,
            rtol=1.0e-6,
            atol=1.0e-12,
        )

    assert not torch.allclose(fractions[0], fractions[1])
    assert not torch.allclose(fractions[1], fractions[2])
    assert not torch.allclose(fractions[0], fractions[2])
