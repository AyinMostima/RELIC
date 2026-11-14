"""CR-A-04 S2 and T4: Strang step means of REPLACE flux rates close the ledgers.

Design 20261019_cr_a04_design_01, D2, S2 and T4 (step 5, execution_group.py). In the
Strang palindrome every fast member except the last runs two half-step calls. A
member that writes an SCM flux contract as a REPLACE per-second rate commits q1 over
[t, t + h/2] and q2 over [t + h/2, t + h]; before S2 the stored value after the step
was q2, so a ledger built from the reported rate, q2 h, missed (q1 - q2) h / 2 of
the amount actually released (q1 + q2) h / 2 (D2). S2: the group composition (the
step-transaction object of the fast group, empty at every accepted-step boundary)
accumulates such rates with the coverage h_k / H and publishes the step mean once,
in the same owner commit as the boundary-rate finalization, after the palindrome.
The half-step commits keep their own rates (each commit's ledger uses its own
interval), consumers inside the palindrome see the writer's latest stage rate, and
step() returns the published mean.

Decisions pinned here:
- Lie and members that run once (the last member of the palindrome, medium and slow
  group calls) are unchanged: one call over H, no mean, no extra commit.
- Historical step-5 decision kept every REPLACE child's last value. The
  2026-10-20 P-N evidence falsified its physical N2 ledger (retained old failure).
  A real SCM allowed_in/out boundary port now contributes its actual child
  integral to the call mean before S2; explicit conflicting temporal support
  refuses the step. Ordinary state/observation REPLACE and ACCUMULATE keep
  their existing rules. S2's wider historical publication selector is separate.
- A writer that emits the rate in only some of its calls of one composition has no
  defined step mean: the step is refused and rolled back.
- C7-like compositions (Priestley-Taylor ET without persistent state) stay bitwise:
  q1 = q2 = q and 0.5 q + 0.5 q = q in binary64.

T4 cases: S1 refuses the governed G04 chain under Strang (phenology -> litterfall
mapper -> decomposition and decomposition -> mineralization via
carbon_decomposition_rate), so T4 uses compositions of the same owners that S1
admits (S1 is not loosened): first-order decomposition before static phenology
(carbon: decomposition runs twice; pre-change residual -8.465e-8 g C m-2 per step at
dt = 600 s), and mineralization, nitrification, denitrification and nitrate leaching
(nitrogen: denitrification runs twice; pre-change residual about -3.2e-7 g N m-2 per
step at dt = 3600 s with the warm, wet driver state below). Pre-change values:
20261019_cr_a04_impl_03/logs/03_diag_prechange_d2_pairs.log.

References: Strang (1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer, Lubich and
Wanner (2006) Geometric Numerical Integration, 2nd ed., Sect. II.5; LeVeque (2002)
Finite Volume Methods, finite-volume time integration; Parton et al. (1987) SSSAJ 51;
Li et al. (1992) JGR 97 (denitrification products); Priestley and Taylor (1972)
Mon. Weather Rev. 100:81-92.
Mathematical expression: M = sum_k (h_k / H) q_k over the calls k of one writer
(h_k = H / 2 for both half-steps, so M = 0.5 q1 + 0.5 q2); carbon ledger
r_C = sum_i Delta C_i - (I_met + I_str) H + R_h H; nitrogen ledger
r_N = sum_j Delta N_j - I_N H + (L_N + F_N2O + F_N2) H.
Parameter meanings: q_k the rate committed by call k; h_k its interval; H the fast
step; C_i soil carbon pools; N_j soil nitrogen pools; I_met, I_str, I_N held inputs;
R_h, L_N, F_N2O, F_N2 the stored (reported) rates after the step.
Units: q, R_h, L_N, F_N2O, F_N2, I in g m-2 s-1; pools g m-2; h_k, H [s]; the probe
rate in g N m-2 s-1.
Domain/constraints: CPU, float64 (test/conftest.py); ledger tolerance 1e-9 g m-2 per
step as in budget_ledger_governed_multistep.test.py.
Sign convention: exports positive, inputs positive.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
import torch

from relic.core.functor import BiologicalProcessFunctor, FunctorMetadata, TemporalScale, VariableKind
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode

_TEST_ROOT = Path(__file__).resolve().parents[1]


def _load_test_module(relative: str, alias: str):
    """Load an existing test module by path to reuse its fixtures (no copies)."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, _TEST_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_T0 = _load_test_module("core/splitting_regression_digests.test.py", "_cr_a04_t0_digests")
_T1 = _load_test_module("core/splitting_admissibility.test.py", "_cr_a04_t1_admissibility")
_T6 = _load_test_module("core/splitting_checkpoint_restore.test.py", "_cr_a04_t6_checkpoint_restore")
_BUDGET = _T0._BUDGET

_LEDGER_TOL = 1.0e-9  # g m-2 per step, the G04 ledger tolerance
_PROBE_Q0 = 1.0e-6  # g N m-2 s-1, probe rate at t = 0
_PROBE_TIME_SCALE = 3600.0  # s, probe rate doubles over this time
_SOIL_N = ("organic_nitrogen", "microbial_nitrogen", "nh4_nitrogen", "no3_nitrogen")
# Held litter inputs of first-order decomposition [g C m-2 s-1] (the litterfall
# mapper that provides them in G04 is refused under Strang by S1).
_LITTER_INPUTS = {"metabolic_litter_input": 2.0e-6, "structural_litter_input": 1.0e-6}
# Warm, wet driver state for the nitrogen chain: denitrification changes NO3 within
# one step, so the two half-step rates differ (pre-change residual about -3e-7).
_WET_DRIVERS = {"water_filled_pore_space": 0.9, "labile_carbon": 1.0, "soil_temperature": 303.15}


class _RateProbe(BiologicalProcessFunctor):
    """Test writer of the SCM flux n2_production as a REPLACE per-second rate.

    The probe carries the name of the contract's writer (denitrification) so the
    ContractRegistry authorizes the write; it holds no state. Its rate depends on the
    start time of each call (or sub-step), q(t) = q0 (1 + t / T), so successive calls
    report different rates. ``emit(call_number)`` decides whether a call returns it.
    """

    def __init__(
        self, emit: Optional[Callable[[int], bool]] = None, *, basis: Optional[str] = "per_second"
    ) -> None:
        metadata = FunctorMetadata(
            name="denitrification",
            description="REPLACE flux-rate probe (test fixture)",
            references=["Strang (1968) SIAM J. Numer. Anal. 5:506-517"],
            temporal_scale=TemporalScale.SECOND,
            input_variables=[],
            output_variables=["n2_production"],
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(name="denitrification", config={}, metadata=metadata)
        self.execution_group = "fast"
        self.calls: List[Tuple[float, float, float]] = []
        self.emit = emit
        self.basis = basis

    def get_dependencies(self) -> Dict[str, List[str]]:
        return {"required": [], "optional": [], "lagged": []}

    def forward(self, state: Dict[str, Any], time: float, dt: torch.Tensor) -> Dict[str, UpdateItem]:
        # References: test fixture (no scientific law).
        # Mathematical expression: q(t) = q0 (1 + t / T).
        # Parameter meanings: t the start of this call; q0 = _PROBE_Q0; T = _PROBE_TIME_SCALE.
        # Units: q, q0 g N m-2 s-1; t, T [s].
        # Domain/constraints: t >= 0.
        # Sign convention: an export (positive).
        rate = _probe_rate(float(time))
        self.calls.append((float(time), float(dt.item()), rate))
        if self.emit is not None and not self.emit(len(self.calls)):
            return {}
        item = UpdateItem(
            value=torch.tensor(rate, dtype=torch.float64),
            mode=UpdateMode.REPLACE,
            unit="g N m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
            # None omits the deprecated field and exercises UpdateItem's real default.
            **({} if self.basis is None else {"basis": self.basis}),
        )
        assert item.basis == ("per_step" if self.basis is None else self.basis)
        return {"n2_production": item}


def _probe_rate(time: float) -> float:
    return _PROBE_Q0 * (1.0 + time / _PROBE_TIME_SCALE)


class _Recorder:
    """Provenance recorder capturing the StateManager commit events (step = commit index)."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def record_state_update(self, step, variable, pre_value, post_value, functor_increments, unit, **options):
        self.events.append({
            "commit": int(step),
            "variable": str(variable),
            "pre": torch.as_tensor(pre_value).detach().clone(),
            "post": torch.as_tensor(post_value).detach().clone(),
            "options": dict(options),
        })

    def posts(self, variable: str) -> List[Tuple[int, torch.Tensor]]:
        return [(event["commit"], event["post"]) for event in self.events if event["variable"] == variable]


def _commit_count(simulator: RELICSimulator) -> int:
    return int(simulator.state_manager.get_diagnostics()["update_counter"])


def _count_calls(simulator: RELICSimulator) -> Dict[str, int]:
    """Count forward calls of every fast member (wraps the bound forward of each instance)."""
    calls: Dict[str, int] = {}
    for name in list(simulator._order_fast):
        functor = simulator.functors[name]
        original = functor.forward

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args, **kwargs)

        functor.forward = counted
    return calls


def _probe_simulator(
    splitting_mode: str, *, emit=None, basis="per_second", **config
) -> Tuple[RELICSimulator, _RateProbe]:
    settings = {
        "time_step": 600.0,
        "device": "cpu",
        "splitting_mode": splitting_mode,
        "state_units": {"relax_a": "dimensionless"},
    }
    settings.update(config)
    simulator = RELICSimulator(settings)
    simulator.initialize_state({"relax_a": 1.0, "n2_production": 0.0})
    probe = _RateProbe(emit=emit, basis=basis)
    simulator.register_functor(probe)
    simulator.register_functor(_T1._Relaxation("relax_a", "relax_a"))
    assert list(simulator._order_fast) == ["denitrification", "relax_a"]
    return simulator, probe


def _scalar(simulator: RELICSimulator, key: str) -> float:
    return float(torch.as_tensor(simulator.state_manager.get_state(key)).sum().item())


# --------------------------------------------------------------------------
# T4: ledgers from the reported rates under Strang (RED before S2).
# --------------------------------------------------------------------------


def _carbon_simulator(splitting_mode: str) -> RELICSimulator:
    """First-order decomposition before static phenology (no flux path; S1 admits it)."""
    from relic.scientific.biosphere.phenology.models import StaticPhenology
    from relic.scientific.pedosphere.decomposition import FirstOrderDecomposition

    simulator = RELICSimulator({
        "device": "cpu",
        "splitting_mode": splitting_mode,
        "conservation_checks": False,
        "strict_dependencies": True,
        "strict_scale_detection": False,
        "adaptive_dt_enabled": False,
    })
    initial = {key: value.clone() for key, value in _BUDGET._initial_state().items()}
    initial.update({key: torch.tensor(value, dtype=torch.float64) for key, value in _LITTER_INPUTS.items()})
    simulator.initialize_state(initial)
    simulator.register_functor(FirstOrderDecomposition(name="first_order_decomposition"), execution_group="fast")
    simulator.register_functor(StaticPhenology(name="static_phenology"), execution_group="fast")
    assert list(simulator._order_fast) == ["first_order_decomposition", "static_phenology"]
    return simulator


def _carbon_residual(simulator: RELICSimulator, before: Dict[str, float], after: Dict[str, float], dt: float) -> float:
    inputs = sum(_scalar(simulator, key) for key in _LITTER_INPUTS) * dt
    respiration = _scalar(simulator, "heterotrophic_respiration")
    return sum(after[key] - before[key] for key in _BUDGET.SOIL_C) - inputs + respiration * dt


def _nitrogen_simulator(splitting_mode: str, *, substep_overrides=None, initial_updates=None) -> RELICSimulator:
    """Mineralization, nitrification, denitrification, leaching (T6 fixture members), wet drivers."""
    from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess
    from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess
    from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
    from relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching import NitrogenLeachingProcess

    simulator = RELICSimulator({
        "device": "cpu",
        "splitting_mode": splitting_mode,
        "conservation_checks": False,
        "strict_dependencies": True,
        "strict_scale_detection": False,
        "adaptive_dt_enabled": False,
        "substep_overrides": substep_overrides or {},
    })
    initial = {key: value.clone() for key, value in _BUDGET._initial_state().items()}
    initial.update({key: torch.tensor(value, dtype=torch.float64) for key, value in _T6._N_PRESCRIBED.items()})
    initial.update({key: torch.tensor(value, dtype=torch.float64) for key, value in _WET_DRIVERS.items()})
    initial.update(initial_updates or {})
    simulator.initialize_state(initial)
    for functor in (MineralizationProcess(), NitrificationProcess(), DenitrificationProcess(), NitrogenLeachingProcess()):
        simulator.register_functor(functor, execution_group="fast")
    assert list(simulator._order_fast) == ["mineralization", "nitrification", "denitrification", "nitrogen_leaching"]
    return simulator


def _nitrogen_residual(simulator: RELICSimulator, before: Dict[str, float], after: Dict[str, float], dt: float) -> float:
    inputs = _scalar(simulator, "organic_nitrogen_input") * dt
    exports = (
        _scalar(simulator, "nitrogen_leaching_flux")
        + _scalar(simulator, "n2o_flux")
        + _scalar(simulator, "n2_production")
    ) * dt
    return sum(after[key] - before[key] for key in _SOIL_N) - inputs + exports


def test_t4_decomposition_carbon_ledger_closes_from_the_reported_respiration_under_strang():
    dt, steps = 600.0, 6
    simulator = _carbon_simulator("strang")
    calls = _count_calls(simulator)
    start = {key: _scalar(simulator, key) for key in _BUDGET.SOIL_C}
    cumulative = 0.0
    for step in range(steps):
        before = {key: _scalar(simulator, key) for key in _BUDGET.SOIL_C}
        calls.clear()
        simulator.step(dt=dt)
        after = {key: _scalar(simulator, key) for key in _BUDGET.SOIL_C}
        # D2 is exhibited: the respiration writer runs two half-steps.
        assert calls == {"first_order_decomposition": 2, "static_phenology": 1}, step
        assert _carbon_residual(simulator, before, after, dt) == pytest.approx(0.0, abs=_LEDGER_TOL), step
        cumulative += (_scalar(simulator, "heterotrophic_respiration")
                       - sum(_scalar(simulator, key) for key in _LITTER_INPUTS)) * dt
    end = {key: _scalar(simulator, key) for key in _BUDGET.SOIL_C}
    assert sum(end[key] - start[key] for key in _BUDGET.SOIL_C) + cumulative == pytest.approx(
        0.0, abs=_LEDGER_TOL * steps
    )


def test_t4_nitrogen_ledger_closes_from_the_reported_rates_under_strang():
    dt, steps = 3600.0, 4
    simulator = _nitrogen_simulator("strang")
    calls = _count_calls(simulator)
    for step in range(steps):
        before = {key: _scalar(simulator, key) for key in _SOIL_N}
        calls.clear()
        simulator.step(dt=dt)
        after = {key: _scalar(simulator, key) for key in _SOIL_N}
        assert calls == {"mineralization": 2, "nitrification": 2, "denitrification": 2, "nitrogen_leaching": 1}
        assert _nitrogen_residual(simulator, before, after, dt) == pytest.approx(0.0, abs=_LEDGER_TOL), step


@pytest.mark.parametrize("builder", [_carbon_simulator, _nitrogen_simulator], ids=["carbon", "nitrogen"])
def test_t4_lie_ledgers_of_the_same_compositions_close(builder):
    dt = 600.0 if builder is _carbon_simulator else 3600.0
    residual = _carbon_residual if builder is _carbon_simulator else _nitrogen_residual
    pools = _BUDGET.SOIL_C if builder is _carbon_simulator else _SOIL_N
    simulator = builder("lie")
    for step in range(3):
        before = {key: _scalar(simulator, key) for key in pools}
        simulator.step(dt=dt)
        after = {key: _scalar(simulator, key) for key in pools}
        assert residual(simulator, before, after, dt) == pytest.approx(0.0, abs=_LEDGER_TOL), step


# --------------------------------------------------------------------------
# S2 semantics.
# --------------------------------------------------------------------------


def test_s2_stored_rate_is_the_coverage_weighted_mean_of_the_half_step_rates():
    simulator, probe = _probe_simulator("strang")
    recorder = _Recorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    first_commit = _commit_count(simulator)
    returned = simulator.step(dt=600.0)
    assert [(start, length) for start, length, _ in probe.calls] == [(0.0, 300.0), (300.0, 300.0)]
    q1 = torch.tensor(probe.calls[0][2], dtype=torch.float64)
    q2 = torch.tensor(probe.calls[1][2], dtype=torch.float64)
    expected = q1 * torch.tensor(0.5, dtype=torch.float64) + q2 * torch.tensor(0.5, dtype=torch.float64)
    stored = simulator.state_manager.get_state("n2_production")
    assert torch.equal(stored.reshape(()), expected)
    assert not torch.equal(expected, q2)
    # step() returns the published mean, not the last half-step rate.
    assert torch.equal(torch.as_tensor(returned["n2_production"]).reshape(()), expected)
    # The half-step commits keep their own rates; one owner commit publishes the mean.
    posts = recorder.posts("n2_production")
    assert [float(value) for _, value in posts] == [float(q1), float(q2), float(expected)]
    assert [commit for commit, _ in posts] == [first_commit, first_commit + 2, first_commit + 3]
    assert _commit_count(simulator) == first_commit + 4
    # The accumulator lives in the composition of the step and is gone afterwards.
    assert simulator._active_composition is None


def test_s2_lie_keeps_the_single_call_rate_and_commit_count():
    simulator, probe = _probe_simulator("lie")
    first_commit = _commit_count(simulator)
    returned = simulator.step(dt=600.0)
    assert [(start, length) for start, length, _ in probe.calls] == [(0.0, 600.0)]
    stored = simulator.state_manager.get_state("n2_production").reshape(())
    assert float(stored) == probe.calls[0][2] == _probe_rate(0.0)
    assert float(torch.as_tensor(returned["n2_production"])) == probe.calls[0][2]
    assert _commit_count(simulator) == first_commit + 1


def test_s2_nitrogen_chain_publishes_the_mean_of_the_twice_run_writer_only():
    simulator = _nitrogen_simulator("strang")
    recorder = _Recorder()
    simulator.state_manager.set_provenance_recorder(recorder)
    first_commit = _commit_count(simulator)
    simulator.step(dt=3600.0)
    n2 = recorder.posts("n2_production")
    # Two denitrification half-step commits, then the published mean.
    assert len(n2) == 3
    half = torch.tensor(0.5, dtype=torch.float64)
    expected = n2[0][1] * half + n2[1][1] * half
    assert not torch.equal(n2[0][1], n2[1][1])
    assert torch.equal(n2[2][1], expected)
    assert torch.equal(simulator.state_manager.get_state("n2_production"), expected)
    # Nitrate leaching (last member, one call over H) keeps its single committed rate.
    leaching = recorder.posts("nitrogen_leaching_flux")
    assert len(leaching) == 1
    assert torch.equal(simulator.state_manager.get_state("nitrogen_leaching_flux"), leaching[0][1])
    assert all(commit >= first_commit for commit, _ in n2 + leaching)


def test_s2_c7_like_priestley_taylor_mean_is_bitwise_the_lie_value():
    from relic.scientific.hydrosphere.evapotranspiration.priestley_taylor import PriestleyTaylorET
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture

    def build(mode: str) -> RELICSimulator:
        simulator = RELICSimulator({"time_step": 1800.0, "device": "cpu", "splitting_mode": mode})
        simulator.initialize_state({
            "net_radiation": torch.tensor([400.0]),
            "air_temperature": torch.tensor([298.15]),
            "atmospheric_pressure": torch.tensor([101325.0]),
            "soil_heat_flux": torch.tensor([40.0]),
            "soil_moisture": torch.tensor([0.25]),
            "precipitation_flux": torch.tensor([1.0e-5]),
            "evapotranspiration_flux": torch.tensor([0.0]),
        })
        simulator.register_functor(PriestleyTaylorET(), execution_group="fast")
        simulator.register_functor(BucketSoilMoisture(), execution_group="fast")
        return simulator

    lie, strang = build("lie"), build("strang")
    recorder = _Recorder()
    strang.state_manager.set_provenance_recorder(recorder)
    for _ in range(3):
        lie.step(dt=1800.0)
        strang.step(dt=1800.0)
        for key in ("evapotranspiration_flux", "latent_heat_flux", "soil_moisture"):
            assert torch.equal(lie.state_manager.get_state(key), strang.state_manager.get_state(key)), key
    # S2 acted (non-vacuous): per step two half-step commits and one published mean.
    posts = recorder.posts("evapotranspiration_flux")
    assert len(posts) == 9
    for step in range(3):
        first, second, mean = (value for _, value in posts[3 * step: 3 * step + 3])
        assert torch.equal(first, second) and torch.equal(mean, first)


@pytest.mark.parametrize("basis", [None, "per_second"], ids=["default_per_step", "per_second"])
def test_physical_replace_rate_is_the_substep_mean_under_lie(basis):
    # Revision 2026-10-20: preserve the physical integral, not step-5 last-child loss.
    simulator, probe = _probe_simulator("lie", basis=basis, substep_overrides={"denitrification": {"count": 2}})
    first_commit = _commit_count(simulator)
    simulator.step(dt=600.0)
    assert [(start, length) for start, length, _ in probe.calls] == [(0.0, 300.0), (300.0, 300.0)]
    stored = simulator.state_manager.get_state("n2_production").reshape(())
    assert float(stored) == (_probe_rate(0.0) + _probe_rate(300.0)) * 0.5
    assert _commit_count(simulator) == first_commit + 1


@pytest.mark.parametrize("basis", [None, "per_second"], ids=["default_per_step", "per_second"])
def test_strang_mean_uses_the_call_value_of_each_half_step_with_substeps(basis):
    simulator, probe = _probe_simulator("strang", basis=basis, substep_overrides={"denitrification": {"count": 2}})
    simulator.step(dt=600.0)
    assert [start for start, _, _ in probe.calls] == [0.0, 150.0, 300.0, 450.0]
    half = torch.tensor(0.5, dtype=torch.float64)
    expected = ((torch.tensor(_probe_rate(0.0), dtype=torch.float64) * half
                 + torch.tensor(_probe_rate(150.0), dtype=torch.float64) * half) * half
                + (torch.tensor(_probe_rate(300.0), dtype=torch.float64) * half
                   + torch.tensor(_probe_rate(450.0), dtype=torch.float64) * half) * half)
    assert torch.equal(simulator.state_manager.get_state("n2_production").reshape(()), expected)


@pytest.mark.parametrize("emitting_call", [1, 2], ids=["first_half_only", "second_half_only"])
def test_s2_refuses_a_rate_missing_from_one_half_step_and_rolls_back(emitting_call):
    simulator, probe = _probe_simulator("strang", emit=lambda number: number == emitting_call)
    before = {key: simulator.state_manager.get_state(key).clone() for key in ("relax_a", "n2_production")}
    with pytest.raises(RuntimeError, match="S2"):
        simulator.step(dt=600.0)
    assert simulator.current_time == 0.0 and simulator.step_count == 0
    for key, value in before.items():
        assert torch.equal(simulator.state_manager.get_state(key), value), key


def test_s2_rejected_adaptive_attempt_leaves_no_partial_mean():
    # dt_min = 1 s and a tiny atol force one rejection: retry = max(0.1 dt, dt_min) = 60 s.
    simulator, probe = _probe_simulator(
        "strang", adaptive_dt_enabled=True, adaptive_atol=1.0e-12, adaptive_rtol=0.0, adaptive_dt_min=1.0
    )
    simulator.step(dt=600.0)
    starts = [(start, length) for start, length, _ in probe.calls]
    assert starts == [(0.0, 300.0), (300.0, 300.0), (0.0, 30.0), (30.0, 30.0)]
    half = torch.tensor(0.5, dtype=torch.float64)
    expected = (torch.tensor(_probe_rate(0.0), dtype=torch.float64) * half
                + torch.tensor(_probe_rate(30.0), dtype=torch.float64) * half)
    assert simulator.current_time == pytest.approx(60.0)
    assert torch.equal(simulator.state_manager.get_state("n2_production").reshape(()), expected)


def test_s2_strang_restart_is_bitwise_including_the_published_means(tmp_path):
    simulator = _nitrogen_simulator("strang")
    for _ in range(2):
        simulator.step(dt=3600.0)
    path = tmp_path / "s2_nitrogen.ckpt"
    simulator.create_checkpoint(path)
    keys = _SOIL_N + ("n2_production", "nitrification_flux", "denitrification_flux", "n2o_flux")
    uninterrupted = []
    for _ in range(2):
        simulator.step(dt=3600.0)
        uninterrupted.append({key: simulator.state_manager.get_state(key).clone() for key in keys})
    resumed = _nitrogen_simulator("strang")
    resumed.restore_checkpoint(path)
    for row in uninterrupted:
        resumed.step(dt=3600.0)
        for key in keys:
            assert torch.equal(row[key], resumed.state_manager.get_state(key)), key


# --------------------------------------------------------------------------
# 2026-10-20 physical child-interval rate qualification (step-5 revision).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["lie", "strang"])
@pytest.mark.parametrize("basis", [None, "per_second"], ids=["default_per_step", "per_second"])
@pytest.mark.parametrize("substep_override", [{"max_seconds": 20.0}, {"count": 3}], ids=["max_seconds", "three_children"])
def test_physical_rate_integrates_actual_equal_substeps_for_unequal_call_lengths(mode, basis, substep_override):
    simulator, probe = _probe_simulator(
        mode, basis=basis, substep_overrides={"denitrification": substep_override}
    )
    for interval in (65.0, 17.5, 82.25):
        probe.calls.clear()
        simulator.step(dt=interval)
        # References: LeVeque (2002); existing physical boundary-rate contract.
        # Mathematical expression: A = sum_k q_k h_k = H * q_mean.
        # Parameter meanings: q_k is the observed child rate; h_k its actual duration;
        #   H is the complete macrostep interval, including all Strang stages.
        # Units: q_k [g N m-2 s-1]; h_k and H [s]; A [g N m-2].
        # Domain/constraints: positive actual supports; one physical integration;
        #   omitting deprecated basis must not change the SCM rate interpretation.
        # Sign convention: positive amounts leave the declared nitrogen boundary.
        integral = sum(rate * length for _, length, rate in probe.calls)
        actual = _scalar(simulator, "n2_production") * interval
        assert actual == pytest.approx(integral, abs=1.0e-15), {
            "expected_integral": integral, "actual_integral": actual,
            "basis": basis, "substep_override": substep_override,
        }
        for start, length, _ in probe.calls:
            if "max_seconds" in substep_override:
                assert 0.0 < length <= 20.0
            else:
                assert length == interval / (3 if mode == "lie" else 6)
        if "count" in substep_override:
            assert len(probe.calls) == (3 if mode == "lie" else 6)
    # The owner uses ceil then equal subdivision, never an invented final tail.
    if mode == "lie" and "max_seconds" in substep_override:
        assert len(probe.calls) == 5
        assert {length for _, length, _ in probe.calls} == {82.25 / 5.0}


@pytest.mark.parametrize("mode", ["lie", "strang"])
@pytest.mark.parametrize("basis", [None, "per_second"], ids=["default_per_step", "per_second"])
def test_constant_physical_substep_rates_keep_the_original_value(mode, basis, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_probe_rate", lambda time: _PROBE_Q0)
    simulator, _ = _probe_simulator(mode, basis=basis, substep_overrides={"denitrification": {"count": 4}})
    simulator.step(dt=65.0)
    assert _scalar(simulator, "n2_production") == _PROBE_Q0


@pytest.mark.parametrize("mode", ["lie", "strang"])
@pytest.mark.parametrize("basis", [None, "per_second"], ids=["default_per_step", "per_second"])
def test_explicit_instantaneous_physical_support_is_not_reinterpreted_by_substeps(mode, basis, monkeypatch):
    simulator, _ = _probe_simulator(mode, basis=basis, substep_overrides={"denitrification": {"count": 2}})
    contract = simulator.state_manager._contract_registry.get_contract("n2_production")
    monkeypatch.setitem(contract.raw, "support", {
        "spatial_kind": "grid_cell", "area_basis": "ground_area",
        "temporal_basis": "instantaneous", "vertical_basis": "none",
    })
    before = simulator.state_manager.get_state("n2_production").clone()
    with pytest.raises(RuntimeError, match="Group execution failed") as caught:
        simulator.step(dt=600.0)
    assert isinstance(caught.value.__cause__, ValueError)
    assert "temporal support" in str(caught.value.__cause__)
    assert simulator.current_time == 0.0 and simulator.step_count == 0
    assert torch.equal(before, simulator.state_manager.get_state("n2_production"))
    # The in-memory support declaration is a test fixture, not a YAML edit.


@pytest.mark.parametrize("mode", ["lie", "strang"])
@pytest.mark.parametrize("basis", [None, "per_second"], ids=["default_per_step", "per_second"])
def test_single_child_keeps_the_existing_instantaneous_and_s2_publication_boundaries(mode, basis):
    simulator, probe = _probe_simulator(mode, basis=basis)
    contract = simulator.state_manager._contract_registry.get_contract("n2_production")
    contract.raw["support"] = {
        "spatial_kind": "grid_cell", "area_basis": "ground_area",
        "temporal_basis": "instantaneous", "vertical_basis": "none",
    }
    try:
        simulator.step(dt=600.0)
        expected = sum(rate * length for _, length, rate in probe.calls) / 600.0
        assert _scalar(simulator, "n2_production") == pytest.approx(expected, abs=1.0e-18)
        # Historical S2 still averages the two call values; this does not certify
        # all instantaneous SCM fluxes as interval means or endpoint observations.
        assert len(probe.calls) == (1 if mode == "lie" else 2)
    finally:
        del contract.raw["support"]


@pytest.mark.parametrize("missing", [1, 2, 4])
@pytest.mark.parametrize("basis", [None, "per_second"], ids=["default_per_step", "per_second"])
def test_a_physical_rate_missing_from_one_actual_child_refuses_and_rolls_back(missing, basis):
    simulator, _ = _probe_simulator(
        "lie", basis=basis, emit=lambda call: call != missing,
        substep_overrides={"denitrification": {"count": 4}},
    )
    before = simulator.state_manager.get_state("n2_production").clone()
    with pytest.raises(RuntimeError, match="Group execution failed") as caught:
        simulator.step(dt=600.0)
    assert isinstance(caught.value.__cause__, ValueError)
    assert "substep coverage" in str(caught.value.__cause__)
    assert simulator.current_time == 0.0 and simulator.step_count == 0
    assert torch.equal(before, simulator.state_manager.get_state("n2_production"))


def _observe_nitrogen_gases(simulator):
    """Read actual production outputs through the existing governor, without writes."""
    records = []
    for name in ("nitrification", "denitrification"):
        def observe(call, state, *, time, dt, _name=name, **kwargs):
            output = call(state, time=time, dt=dt, **kwargs)
            records.append({
                "name": _name, "dt": dt,
                "n2": output["n2_production"].value if _name == "denitrification" else dt * 0.0,
                "n2o": output["n2o_flux"].value,
                "no3_delta": output["no3_nitrogen"].value,
                "no3_pre": state["no3_nitrogen"],
            })
            return output
        simulator.functors[name].set_call_governor(observe)
    return records


@pytest.mark.parametrize("mode", ["lie", "strang"])
def test_real_denitrification_child_integrals_and_published_gas_ledger_agree(mode):
    simulator = _nitrogen_simulator(mode, substep_overrides={"denitrification": {"count": 8}})
    observed = _observe_nitrogen_gases(simulator)
    before = {key: _scalar(simulator, key) for key in _SOIL_N}
    simulator.step(dt=7200.0)
    after = {key: _scalar(simulator, key) for key in _SOIL_N}
    n2 = sum(float((row["n2"] * row["dt"]).detach()) for row in observed)
    n2o = sum(float((row["n2o"] * row["dt"]).detach()) for row in observed)
    assert _scalar(simulator, "n2_production") * 7200.0 == pytest.approx(n2, abs=1.0e-12)
    # ACCUMULATE is already averaged: the new physical REPLACE path must not weight it twice.
    assert _scalar(simulator, "n2o_flux") * 7200.0 == pytest.approx(n2o, abs=1.0e-12)
    assert _nitrogen_residual(simulator, before, after, 7200.0) == pytest.approx(0.0, abs=_LEDGER_TOL)


def test_donor_limited_real_gases_keep_actual_amounts_instead_of_the_final_zero_rate():
    simulator = _nitrogen_simulator(
        "lie", substep_overrides={"denitrification": {"count": 8}},
        initial_updates={
            "nh4_nitrogen": torch.tensor(0.0, dtype=torch.float64),
            "no3_nitrogen": torch.tensor(1.0e-4, dtype=torch.float64),
            "organic_nitrogen": torch.tensor(0.0, dtype=torch.float64),
            "microbial_nitrogen": torch.tensor(0.0, dtype=torch.float64),
        },
    )
    observed = _observe_nitrogen_gases(simulator)
    simulator.step(dt=7200.0)
    den = [row for row in observed if row["name"] == "denitrification"]
    assert any(float((row["no3_pre"] + row["no3_delta"]).detach()) <= 1.0e-18 for row in den)
    expected = sum(float((row["n2"] * row["dt"]).detach()) for row in den)
    assert expected > 0.0
    assert _scalar(simulator, "n2_production") * 7200.0 == pytest.approx(expected, abs=1.0e-12)


def test_physical_mean_preserves_cpu_float64_and_the_executed_gradient():
    initial = torch.tensor(0.002, dtype=torch.float64, requires_grad=True)
    simulator = _nitrogen_simulator(
        "lie", substep_overrides={"denitrification": {"count": 4}},
        initial_updates={"no3_nitrogen": initial},
    )
    observed = _observe_nitrogen_gases(simulator)
    simulator.step(dt=65.0)
    stored = simulator.state_manager.get_state("n2_production")
    assert stored.dtype == torch.float64 and stored.device.type == "cpu"
    expected = sum(row["n2"] * row["dt"] for row in observed)
    observed_gradient = torch.autograd.grad(stored * 65.0, initial, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, initial)[0]
    assert float(expected_gradient.detach().abs()) > 0.0
    assert torch.allclose(observed_gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)


def test_physical_substep_means_restart_bitwise_through_the_existing_owner(tmp_path):
    settings = {"substep_overrides": {"denitrification": {"count": 3}}}
    writer = _nitrogen_simulator("strang", **settings)
    writer.step(dt=65.0)
    checkpoint = tmp_path / "physical_substep_mean.ckpt"
    writer.create_checkpoint(checkpoint)
    writer.step(dt=82.25)
    resumed = _nitrogen_simulator("strang", **settings)
    resumed.restore_checkpoint(checkpoint)
    resumed.step(dt=82.25)
    for key in _SOIL_N + ("n2_production", "n2o_flux", "denitrification_flux"):
        assert torch.equal(writer.state_manager.get_state(key), resumed.state_manager.get_state(key)), key


def test_plain_state_and_derived_replace_outputs_retain_the_last_actual_child(monkeypatch):
    from dataclasses import replace
    from relic.core.state import StateWriteMode

    simulator, probe = _probe_simulator("lie", substep_overrides={"denitrification": {"count": 4}})
    original = probe.forward
    samples = []

    def observed(state, time, dt):
        result = original(state, time, dt)
        value = result["n2_production"].value
        samples.append(value)
        # These are manufactured observations, not a second scientific law.
        result["relax_a"] = UpdateItem(value=value, mode=UpdateMode.REPLACE, unit="dimensionless")
        result["diagnostics.last_rate"] = replace(result["n2_production"], write_mode=StateWriteMode.DERIVED)
        return result

    probe.metadata.output_variables.append("relax_a")
    monkeypatch.setattr(probe, "forward", observed)
    # Inspect the existing call owner directly so no second writer of relax_a is committed.
    from relic.core.simulator.execution_group import GroupComposition, _run_functor_substeps
    simulator._active_composition = GroupComposition(interval=65.0)
    output = _run_functor_substeps(simulator, probe.name, probe, simulator.state_manager.get_state(), 65.0, 0.0, [])
    assert torch.equal(output["relax_a"].value, samples[-1])
    assert torch.equal(output["diagnostics.last_rate"].value, samples[-1])
    assert not torch.equal(output["n2_production"].value, samples[-1])


@pytest.mark.parametrize("mode", ["lie", "strang"])
def test_rank_mean_hooks_reduce_accumulate_children_and_replace_calls_once(mode, monkeypatch):
    from dataclasses import replace
    from relic.core.simulator.execution_loop import _apply_distributed_allreduce

    local_values = []
    original_probe = _RateProbe

    class _MixedProbe(original_probe):
        def __init__(self, emit=None, *, basis="per_second"):
            super().__init__(emit=emit, basis=basis)
            self.metadata.output_variables.append("n2o_flux")

        def forward(self, state, time, dt):
            output = super().forward(state, time, dt)
            original = output["n2_production"]
            local_values.append((original.value, float(original.value)))
            output["n2o_flux"] = replace(original, mode=UpdateMode.ACCUMULATE)
            return output

    monkeypatch.setattr(sys.modules[__name__], "_RateProbe", _MixedProbe)
    simulator, probe = _probe_simulator(mode, substep_overrides={"denitrification": {"count": 4}})
    reduced = {"n2_production": [], "n2o_flux": []}

    class _TwoRankSum:
        def barrier(self):
            pass

        def all_reduce(self, payload):
            # References: existing _apply_distributed_allreduce; PyTorch all_reduce SUM.
            # Mathematical expression: mean(x,3*x) = (x+3*x)/2 = 2*x.
            # Parameter meanings: this hook simulates two different rank values;
            #   the real reducer divides the mutated SUM payload by world_size=2.
            # Units: unchanged UpdateItem units.
            # Domain/constraints: reducer-protocol probe only, not a multi-process run;
            #   use in-place mutation to exercise the actual cloned child payload.
            # Sign convention: retain the sign of the original update.
            payload.mul_(4.0)

    monkeypatch.setattr(simulator, "process_group_manager", _TwoRankSum())

    def rank_mean(updates):
        for name in reduced:
            if name in updates:
                reduced[name].append(float(updates[name].value.detach()))
        previous = simulator.enable_distributed
        simulator.enable_distributed = True
        try:
            _apply_distributed_allreduce(
                simulator, updates, has_distributed_fn=lambda: True,
                is_distributed_fn=lambda: True, get_world_size_fn=lambda: 2,
            )
        finally:
            simulator.enable_distributed = previous

    monkeypatch.setattr(simulator, "_reduce_updates_across_ranks", rank_mean)
    simulator.step(dt=65.0)
    local_integral = sum(rate * length for _, length, rate in probe.calls)
    expected = 2.0 * local_integral / 65.0
    assert _scalar(simulator, "n2_production") == pytest.approx(expected, abs=1.0e-18)
    assert _scalar(simulator, "n2o_flux") == pytest.approx(expected, abs=1.0e-18)
    assert len(reduced["n2o_flux"]) == len(probe.calls) == (4 if mode == "lie" else 8)
    assert len(reduced["n2_production"]) == (1 if mode == "lie" else 2)
    assert all(float(value.detach()) == before for value, before in local_values)
