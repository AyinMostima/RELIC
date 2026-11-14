"""G04: real multi-step carbon and nitrogen budgets through the governed chain.

The pedosphere C/N chain (static phenology -> litterfall interface mapper ->
first-order decomposition -> mineralization -> nitrification ->
denitrification -> nitrogen leaching -> root nutrient uptake -> nitrogen
cycling) runs for many governed steps inside a RELICSimulator. Every step and
the whole run must close the physical ledgers against the explicit external
routes only: heterotrophic respiration for carbon; fertilizer and urine
inputs, leaching and gaseous (N2O, N2) exports for nitrogen. A pool that
would be driven negative by an oversized step is refused before any write,
and a restart from a durable checkpoint continues bitwise.

References: Parton et al. (1987) SSSAJ 51; Thornton et al. (2002) AFM 113;
Manzoni and Porporato (2009) SBB 41.
Mathematical expression (carbon): 1000 Delta C_veg + sum_i Delta C_i +
R_h dt = 0 with C_veg in kg C m-2 and soil pools in g C m-2.
Mathematical expression (nitrogen): 1000 Delta N_veg + Delta N_org +
Delta N_mic + Delta NH4 + Delta NO3 + M_N Delta N_plant = F_fert + F_urine -
(L_N + G_N) dt with the external deltas in g N m-2, M_N the nitrogen
atomic weight converting the plant storage (mol N m-2), and G_N the gaseous
export (aggregated N2O port of nitrification and denitrification plus N2).
Parameter meanings: R_h heterotrophic respiration [g C m-2 s-1], L_N
nitrate leaching [g N m-2 s-1], G_N gaseous nitrogen export [g N m-2 s-1],
F_fert and F_urine the external nitrogen deltas of the step [g N m-2].
Units: soil pools g element m-2, vegetation pools kg element m-2, dt s.
Domain/constraints: pools >= 0; the step is rejected when a pool would go
negative; residuals are compared at float64 rounding.
Sign convention: exports leave the ledger with positive rates; inputs
enter with positive deltas.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.utils.physical_constants import get_physical_constant
from relic.scientific.biosphere.phenology.models import StaticPhenology
from relic.scientific.pedosphere.decomposition import FirstOrderDecomposition
from relic.scientific.pedosphere.interface_mapping import LitterfallInterfaceMapper
from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrogen import NitrogenCycling
from relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching import NitrogenLeachingProcess
from relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake import RootNutrientUptake

DT = 600.0
STEPS = 48
SOIL_C = ("metabolic_litter_c", "structural_litter_c", "fast_som_c", "slow_som_c", "passive_som_c")
VEG_C = ("leaf_carbon", "froot_carbon")
SOIL_N = ("organic_nitrogen", "microbial_nitrogen", "nh4_nitrogen", "no3_nitrogen")
# plant_n_storage is carried in mol N m-2 (root uptake contract); the ledger
# converts it with the nitrogen atomic weight from the physical constants.
PLANT_N = "plant_n_storage"
VEG_N = ("leaf_nitrogen", "froot_nitrogen", "plant_nitrogen_pool")


def _initial_state(metabolic: float = 100.0) -> dict[str, torch.Tensor]:
    f64 = torch.float64
    return {
        "metabolic_litter_c": torch.tensor(metabolic, dtype=f64),
        "structural_litter_c": torch.tensor(200.0, dtype=f64),
        "fast_som_c": torch.tensor(500.0, dtype=f64),
        "slow_som_c": torch.tensor(1000.0, dtype=f64),
        "passive_som_c": torch.tensor(2000.0, dtype=f64),
        "soil_temperature": torch.tensor(288.15, dtype=f64),
        "soil_moisture": torch.tensor(0.3, dtype=f64),
        "water_filled_pore_space": torch.tensor(0.5, dtype=f64),
        "day_of_year": torch.tensor(280.0, dtype=f64),
        "air_temperature": torch.tensor(283.15, dtype=f64),
        "leaf_carbon": torch.tensor(2.0, dtype=f64),
        "froot_carbon": torch.tensor(1.5, dtype=f64),
        "leaf_nitrogen": torch.tensor(0.08, dtype=f64),
        "froot_nitrogen": torch.tensor(0.04, dtype=f64),
        "plant_nitrogen_pool": torch.tensor(0.2, dtype=f64),
        "lignin_n_ratio": torch.tensor(0.3, dtype=f64),
        "clay_content": torch.tensor(0.25, dtype=f64),
        "organic_nitrogen": torch.tensor(5.0, dtype=f64),
        "nh4_nitrogen": torch.tensor(0.1, dtype=f64),
        "no3_nitrogen": torch.tensor(0.2, dtype=f64),
        "microbial_nitrogen": torch.tensor(0.05, dtype=f64),
        "available_phosphorus": torch.tensor(1.0, dtype=f64),
        "plant_n_storage": torch.tensor(0.1, dtype=f64),
        "plant_p_storage": torch.tensor(0.05, dtype=f64),
        "soil_ph": torch.tensor(6.5, dtype=f64),
        "labile_carbon": torch.tensor(0.1, dtype=f64),
        "drainage_flux": torch.tensor(1.0e-6, dtype=f64),
        "fertilizer_n_flux": torch.tensor(1.0e-6, dtype=f64),
        # One-layer column: the depth and bulk density share the scalar shape of the
        # column pools (the leaching bucket must not enlarge the no3_nitrogen shape).
        "soil_depth": torch.tensor(0.1, dtype=f64),
        "bulk_density": torch.tensor(1300.0, dtype=f64),  # kg m-3 leaching forcing
        "root_density": torch.tensor([1.0], dtype=f64),
        "soil_water_content": torch.tensor([0.3], dtype=f64),
        "plant_n_demand": torch.tensor(1.0e-6, dtype=f64),
        "plant_p_demand": torch.tensor(1.0e-7, dtype=f64),
        "field_capacity": torch.tensor([0.45], dtype=f64),
        "wilting_point": torch.tensor([0.1], dtype=f64),
    }


def _build(initial_state: dict[str, torch.Tensor]) -> RELICSimulator:
    sim = RELICSimulator({
        "device": "cpu",
        "conservation_checks": False,
        "strict_dependencies": True,
        "strict_scale_detection": False,
        "adaptive_dt_enabled": False,
    })
    sim.initialize_state({key: value.clone() for key, value in initial_state.items()})
    for functor in (
        StaticPhenology(name="static_phenology"),
        LitterfallInterfaceMapper(name="litterfall_interface_mapper", config={"dependencies": ["static_phenology"]}),
        FirstOrderDecomposition(name="first_order_decomposition", config={"dependencies": ["litterfall_interface_mapper"]}),
        MineralizationProcess(name="mineralization", config={"dependencies": ["first_order_decomposition", "litterfall_interface_mapper"]}),
        NitrificationProcess(),
        DenitrificationProcess(),
        NitrogenLeachingProcess(),
        RootNutrientUptake(),
        NitrogenCycling(),
    ):
        sim.register_functor(functor, execution_group="fast")
    return sim


def _scalar(sim, key: str) -> float:
    return float(sim.state_manager.get_state(key).sum().item())


def _snapshot(sim) -> dict[str, float]:
    return {key: _scalar(sim, key) for key in SOIL_C + VEG_C + SOIL_N + VEG_N + (PLANT_N,)}


def _carbon_residual(before, after, sim) -> float:
    respiration = _scalar(sim, "heterotrophic_respiration")
    delta_veg = sum(after[k] - before[k] for k in VEG_C) * 1000.0
    delta_soil = sum(after[k] - before[k] for k in SOIL_C)
    return delta_veg + delta_soil + respiration * DT


def _nitrogen_residual(before, after, sim) -> float:
    current = sim.state_manager.get_diagnostics()["current"]
    fert = float(torch.as_tensor(current["NitrogenCycling.nitrogen_cycling.external_fertilizer_n_delta"]).sum().item())
    urine = float(torch.as_tensor(current["NitrogenCycling.nitrogen_cycling.external_urine_n_delta"]).sum().item())
    leaching = _scalar(sim, "nitrogen_leaching_flux")
    # Gaseous exports: the aggregated N2O port (nitrification + denitrification
    # N2O) plus the N2 production of denitrification.
    denitrification = _scalar(sim, "n2o_flux") + _scalar(sim, "n2_production")
    molar_mass_n = float(get_physical_constant("relic.physical.constants.molecular_weights.nitrogen"))
    delta_veg = sum(after[k] - before[k] for k in VEG_N) * 1000.0
    delta_soil = sum(after[k] - before[k] for k in SOIL_N)
    delta_plant = (after[PLANT_N] - before[PLANT_N]) * molar_mass_n
    return delta_veg + delta_soil + delta_plant - (fert + urine) + (leaching + denitrification) * DT


class TestGovernedMultiStepBudgets:
    def test_carbon_and_nitrogen_ledgers_close_every_step_and_over_the_run(self):
        sim = _build(_initial_state())
        start = _snapshot(sim)
        cumulative_respiration = 0.0
        cumulative_n_external = 0.0
        for step in range(STEPS):
            before = _snapshot(sim)
            sim.step(dt=DT)
            after = _snapshot(sim)
            assert _carbon_residual(before, after, sim) == pytest.approx(0.0, abs=1e-9), step
            assert _nitrogen_residual(before, after, sim) == pytest.approx(0.0, abs=1e-9), step
            cumulative_respiration += _scalar(sim, "heterotrophic_respiration") * DT
            current = sim.state_manager.get_diagnostics()["current"]
            cumulative_n_external += (
                float(torch.as_tensor(current["NitrogenCycling.nitrogen_cycling.external_fertilizer_n_delta"]).sum().item())
                + float(torch.as_tensor(current["NitrogenCycling.nitrogen_cycling.external_urine_n_delta"]).sum().item())
                - (_scalar(sim, "nitrogen_leaching_flux") + _scalar(sim, "n2o_flux") + _scalar(sim, "n2_production")) * DT
            )
            for key in SOIL_C + SOIL_N + (PLANT_N,):
                assert after[key] >= 0.0, key
        end = _snapshot(sim)
        assert sim.step_count == STEPS and sim.current_time == pytest.approx(STEPS * DT)
        # Whole-run balance from the same external routes only.
        total_carbon = sum(end[k] - start[k] for k in VEG_C) * 1000.0 + sum(end[k] - start[k] for k in SOIL_C)
        assert total_carbon + cumulative_respiration == pytest.approx(0.0, abs=1e-8)
        molar_mass_n = float(get_physical_constant("relic.physical.constants.molecular_weights.nitrogen"))
        total_nitrogen = (
            sum(end[k] - start[k] for k in VEG_N) * 1000.0
            + sum(end[k] - start[k] for k in SOIL_N)
            + (end[PLANT_N] - start[PLANT_N]) * molar_mass_n
        )
        assert total_nitrogen - cumulative_n_external == pytest.approx(0.0, abs=1e-8)
        # Real dynamics happened: litter moved to soil, respiration is positive.
        assert cumulative_respiration > 0.0
        assert end["metabolic_litter_c"] != start["metabolic_litter_c"]

    def test_pool_exhaustion_is_refused_before_any_write(self):
        # A metabolic pool of 1e-6 g C m-2 with dt so large that the explicit
        # first-order step would overshoot below zero.
        sim = _build(_initial_state(metabolic=1.0e-6))
        before = _snapshot(sim)
        with pytest.raises(Exception, match="first_order_decomposition"):
            sim.step(dt=5.0e8)
        assert _snapshot(sim) == before
        assert sim.current_time == 0.0 and sim.step_count == 0

    def test_restart_from_a_durable_checkpoint_continues_bitwise(self, tmp_path: Path):
        sim = _build(_initial_state())
        for _ in range(STEPS // 2):
            sim.step(dt=DT)
        checkpoint = tmp_path / "budget_midrun.ckpt"
        sim.create_checkpoint(checkpoint)
        for _ in range(STEPS // 2):
            sim.step(dt=DT)
        uninterrupted = {key: sim.state_manager.get_state(key).clone() for key in SOIL_C + SOIL_N + VEG_C + VEG_N + (PLANT_N,)}

        resumed = _build(_initial_state())
        resumed.restore_checkpoint(checkpoint)
        for _ in range(STEPS // 2):
            resumed.step(dt=DT)
        for key, value in uninterrupted.items():
            assert torch.equal(value, resumed.state_manager.get_state(key)), key
