"""G02: one governed production chain for a mature process (first-order decomposition).

The representative process is ``FirstOrderDecomposition`` (Parton et al. 1987
five-pool first-order kinetics with temperature and moisture modifiers). The
chain under test is the real simulator path: Functor object call through
``FunctorBase.__call__`` -> UpdateItem / diagnostic partition -> StateManager
commit -> state and diagnostics readable from the simulator, with provenance,
parameter dependence, dt applied exactly once, restart, and the negatives the
task card names (NaN input, hidden state write, diagnostics as state).

References: Parton et al. (1987) SSSAJ 51, 1173-1179; Manzoni and Porporato
(2009) SBB 41, 1355-1367.
Mathematical expression: dC_i/dt = I_i - k_i f_env C_i with f_env = f_T f_W;
per step Delta C_i = (I_i - k_i f_env C_i) dt, and
sum_i Delta C_i + R_h dt = 0 for the closed five-pool ledger.
Parameter meanings: k_i first-order rate constants [s-1], f_T, f_W
dimensionless modifiers, I_i litter inputs [g C m-2 s-1], R_h heterotrophic
respiration [g C m-2 s-1].
Units: pools g C m-2, rates g C m-2 s-1, dt s.
Domain/constraints: pools >= 0, 0 <= f_env <= 1, dt within the explicit-Euler
stability range enforced by the functor.
Sign convention: decomposition removes carbon from the source pool (negative
Delta), transfers add to the receiving pool, respiration leaves the ledger.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.core.state.manager import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
from relic.utils.physical_constants import get_physical_constant

POOLS = ("metabolic_litter_c", "structural_litter_c", "fast_som_c", "slow_som_c", "passive_som_c")
DIAG = "diagnostics.first_order_decomposition."
# Ingested diagnostics are keyed by writer name plus the diagnostic name.
INGESTED = "first_order_decomposition.first_order_decomposition."
# Fixed five-pool regression baseline (per-step increments at dt = 3600 s),
# established in float32 by test/pedosphere/decomposition/test_first_order.py;
# the governed chain here runs in float64, so the baseline is matched to the
# float32 rounding level (rel 1e-4) and the exact kinetics are checked
# separately at float64 precision.
EXPECTED_DELTAS = {
    "metabolic_litter_c": -8.066470175982e-02,
    "structural_litter_c": -3.213990479708e-02,
    "fast_som_c": -1.073344573379e-01,
    "slow_som_c": 1.061581373215e-01,
    "passive_som_c": 7.905141450465e-03,
}
EXPECTED_RESPIRATION = 2.946549284388e-05


def _initial_state(soil_temperature: float = 293.15) -> dict[str, torch.Tensor]:
    return {
        "metabolic_litter_c": torch.tensor(100.0, dtype=torch.float64),
        "structural_litter_c": torch.tensor(200.0, dtype=torch.float64),
        "fast_som_c": torch.tensor(500.0, dtype=torch.float64),
        "slow_som_c": torch.tensor(1000.0, dtype=torch.float64),
        "passive_som_c": torch.tensor(2000.0, dtype=torch.float64),
        "soil_temperature": torch.tensor(soil_temperature, dtype=torch.float64),
        "soil_moisture": torch.tensor(0.3, dtype=torch.float64),
        "metabolic_litter_input": torch.tensor(0.0, dtype=torch.float64),
        "structural_litter_input": torch.tensor(0.0, dtype=torch.float64),
        "lignin_n_ratio": torch.tensor(0.3, dtype=torch.float64),
        "clay_content": torch.tensor(0.25, dtype=torch.float64),
        "carbon_decomposition_rate": torch.tensor(0.0, dtype=torch.float64),
        "heterotrophic_respiration": torch.tensor(0.0, dtype=torch.float64),
    }


def _simulator(functor=None, soil_temperature: float = 293.15) -> tuple[RELICSimulator, FirstOrderDecomposition]:
    sim = RELICSimulator({
        "device": "cpu",
        "conservation_checks": False,
        "strict_dependencies": False,
        "adaptive_dt_enabled": False,
    })
    sim.initialize_state(_initial_state(soil_temperature))
    functor = functor or FirstOrderDecomposition(name="first_order_decomposition", config={})
    sim.register_functor(functor, execution_group="fast")
    return sim, functor


def _pools(sim) -> dict[str, float]:
    state = sim.get_state()
    return {name: float(state[name].item()) for name in POOLS}


class TestGovernedChain:
    def test_governed_step_commits_the_reference_increments_and_closes_the_ledger(self):
        sim, functor = _simulator()
        before = _pools(sim)
        sim.step(dt=3600.0)
        after = _pools(sim)
        assert functor._execution_count == 1
        for name, expected in EXPECTED_DELTAS.items():
            assert after[name] - before[name] == pytest.approx(expected, rel=1e-4)
        state = sim.get_state()
        respiration = float(state["heterotrophic_respiration"].item())
        assert respiration == pytest.approx(EXPECTED_RESPIRATION, rel=1e-4)
        # Closed ledger through the StateManager: sum of pool increments plus
        # respiration over the step vanishes to float32 rounding.
        total_delta = sum(after[name] - before[name] for name in POOLS)
        assert total_delta + respiration * 3600.0 == pytest.approx(0.0, abs=1e-9)

    def test_metabolic_pool_follows_the_independent_first_order_kinetics(self):
        sim, _ = _simulator()
        before = _pools(sim)["metabolic_litter_c"]
        sim.step(dt=3600.0)
        diagnostics = sim.state_manager.get_diagnostics()["current"]
        env = float(torch.as_tensor(diagnostics[INGESTED + "environmental_modifier"]).item())
        # k_metabolic is declared in yr-1 (ParameterManager); the kinetics use
        # the physical-constant year length to express it per second.
        k_met = float(get_global_manager().get("pedosphere.decomposition.k_metabolic", required=True))
        k_met = k_met / float(get_physical_constant("seconds_per_year"))
        # Independent hand calculation: Delta C_met = -k_met f_env C_met dt.
        expected = -k_met * env * before * 3600.0
        actual = _pools(sim)["metabolic_litter_c"] - before
        assert actual == pytest.approx(expected, rel=1e-9)
        flux = float(torch.as_tensor(diagnostics[INGESTED + "metabolic_litter_decomposition_flux"]).item())
        assert flux == pytest.approx(k_met * env * before, rel=1e-9)
        assert 0.0 < env <= 1.0

    def test_diagnostics_are_ingested_not_committed_as_state(self):
        sim, _ = _simulator()
        sim.step(dt=3600.0)
        state = sim.get_state()
        current = sim.state_manager.get_diagnostics()["current"]
        for key in ("temperature_modifier", "moisture_modifier", "environmental_modifier"):
            assert DIAG + key not in state
            assert INGESTED + key in current
        assert "carbon_decomposition_rate" in state and "heterotrophic_respiration" in state

    def test_provenance_names_the_functor_as_the_writer(self):
        sim, functor = _simulator()
        sim.step(dt=3600.0)
        provenance = sim.get_variable_provenance("metabolic_litter_c")
        assert functor.name in str(provenance)
        assert "accumulate" in str(provenance).lower()

    def test_dt_is_applied_exactly_once(self):
        sim_full, _ = _simulator()
        sim_half, _ = _simulator()
        before = _pools(sim_full)
        sim_full.step(dt=3600.0)
        sim_half.step(dt=1800.0)
        full = _pools(sim_full)
        half = _pools(sim_half)
        for name in POOLS:
            assert (half[name] - before[name]) * 2.0 == pytest.approx(full[name] - before[name], rel=1e-9)

    def test_parameter_dependence_flows_through_the_parameter_system(self):
        manager = get_global_manager()
        key = "pedosphere.decomposition.k_metabolic"
        original = float(manager.get(key, required=True))
        sim_ref, _ = _simulator()
        before = _pools(sim_ref)
        sim_ref.step(dt=3600.0)
        reference_delta = _pools(sim_ref)["metabolic_litter_c"] - before["metabolic_litter_c"]
        try:
            manager.set_value(key, original * 2.0)
            sim_double, _ = _simulator()
            sim_double.step(dt=3600.0)
            doubled_delta = _pools(sim_double)["metabolic_litter_c"] - before["metabolic_litter_c"]
        finally:
            manager.set_value(key, original)
        assert doubled_delta == pytest.approx(2.0 * reference_delta, rel=1e-9)
        sim_back, _ = _simulator()
        sim_back.step(dt=3600.0)
        assert _pools(sim_back)["metabolic_litter_c"] - before["metabolic_litter_c"] == pytest.approx(reference_delta, rel=1e-12)

    def test_restart_from_checkpoint_continues_bitwise(self, tmp_path: Path):
        sim, _ = _simulator()
        sim.step(dt=3600.0)
        checkpoint = tmp_path / "first_order_step1.ckpt"
        sim.create_checkpoint(checkpoint)
        sim.step(dt=3600.0)
        uninterrupted = sim.get_state()

        restarted, _ = _simulator()
        restarted.restore_checkpoint(checkpoint)
        restarted.step(dt=3600.0)
        resumed = restarted.get_state()
        for name in POOLS + ("carbon_decomposition_rate", "heterotrophic_respiration"):
            assert torch.equal(uninterrupted[name], resumed[name]), name


class TestDifferentiability:
    def test_governed_call_keeps_the_analytic_gradient_of_the_smooth_path(self):
        """d(Delta C_met)/dC_met = -k_met f_env dt through the governed object call."""
        functor = FirstOrderDecomposition(name="first_order_decomposition", config={})
        state = _initial_state()
        state["metabolic_litter_c"] = state["metabolic_litter_c"].clone().requires_grad_(True)
        outputs = functor(state, time=0.0, dt=3600.0)
        delta = outputs["metabolic_litter_c"].value
        assert delta.requires_grad and delta.grad_fn is not None
        (gradient,) = torch.autograd.grad(delta, state["metabolic_litter_c"])
        env = float(outputs[DIAG + "environmental_modifier"].value)
        k_met = float(get_global_manager().get("pedosphere.decomposition.k_metabolic", required=True))
        k_met = k_met / float(get_physical_constant("seconds_per_year"))
        assert float(gradient) == pytest.approx(-k_met * env * 3600.0, rel=1e-9)


class TestNegatives:
    def test_non_finite_driver_is_refused_and_the_step_leaves_state_untouched(self):
        sim, _ = _simulator(soil_temperature=float("nan"))
        before = _pools(sim)
        with pytest.raises(Exception, match="first_order_decomposition"):
            sim.step(dt=3600.0)
        assert _pools(sim) == before
        assert sim.current_time == 0.0

    def test_hidden_state_write_by_a_non_owner_is_refused(self):
        class LeakingDecomposition(FirstOrderDecomposition):
            def forward(self, state, time, dt, **kwargs):
                outputs = super().forward(state, time, dt, **kwargs)
                # organic_nitrogen belongs to the mineralization owner.
                outputs["organic_nitrogen"] = UpdateItem(
                    value=torch.tensor(1.0, dtype=torch.float64),
                    mode=UpdateMode.ACCUMULATE,
                    unit="g N m-2",
                    per_timestep=True,
                    source_functor=self.name,
                )
                return outputs

        sim = RELICSimulator({
            "device": "cpu",
            "conservation_checks": False,
            "strict_dependencies": False,
            "adaptive_dt_enabled": False,
        })
        initial = _initial_state()
        initial["organic_nitrogen"] = torch.tensor(5.0, dtype=torch.float64)
        sim.initialize_state(initial)
        sim.register_functor(LeakingDecomposition(name="first_order_decomposition", config={}), execution_group="fast")
        before = _pools(sim)
        with pytest.raises(Exception):
            sim.step(dt=3600.0)
        assert _pools(sim) == before
        assert float(sim.get_state()["organic_nitrogen"].item()) == 5.0
