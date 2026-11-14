"""CR-A-04 T6 and restore admissibility: fast-group splitting controls across checkpoints.

Design 20261019_cr_a04_design_01, sections 2-4 and 6 (step 3, checkpoint.py):

- T6: a Strang composition restarted from a durable checkpoint in the middle of
  the run continues bitwise (stateful nitrogen members coupled through the NO3
  state; a Strang palindrome over a reversed fast order). The pinned digests were
  computed before step 3.
- The resolved splitting controls (_splitting_mode, _splitting_policy,
  _fast_order_mode, _substep_overrides, _multirate_exchanges) are runtime fields
  of the checkpoint, including values that come from the parameter database
  rather than from the configuration, so an exact restart continues with the
  controls the saved run executed.
- D1 at restore: a restored control outside its allowed values, malformed, or not
  in the canonical form the simulator writes is refused before any mutation. Since
  step 6 every well-formed control is executed; a restored interval_mean exchange
  whose running mean is missing from the restored state in the middle of its
  consumer interval is refused when the restore republishes, and rolled back.
- A restore republishes the provider view through the splitting admissibility
  resolver: a restored composition that is inadmissible (S1, S3, S5) is refused
  and rolled back, leaving the committed view, the controls and the state
  unchanged.
- A checkpoint written before the controls existed has no such runtime field; it
  restores with the documented database defaults (computation.scheduling.*
  default), which equal the behaviour of the scheduler that wrote it: policy
  warn, forward fast order, no sub-step overrides, no multirate exchanges.

References: Strang (1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer, Lubich and
Wanner (2006) Geometric Numerical Integration, 2nd ed., Sect. II.4-II.5; Parton
et al. (1987) SSSAJ 51 and Del Grosso et al. (2000) Global Biogeochem. Cycles 14
(nitrogen owners of the reused fixtures).
Mathematical expression: digest = SHA-256 over (name, shape, float64 bytes) of each
recorded trajectory (T0 helper); relaxation fixtures x(t + dt) = x(t) exp(-k dt).
Parameter meanings: recorded tensors are per-step trajectories of the named state
variables; k the fixture relaxation rate [s-1].
Units: nitrogen pools g N m-2, nitrogen rates g N m-2 s-1; fixtures dimensionless;
t, dt [s].
Domain/constraints: CPU, float64 test default dtype (test/conftest.py). The pinned
nitrogen digest covers the pools only: under Strang the stored REPLACE rates of the
members run twice are the subject of design step 5 (S2), the pools are not; every
recorded variable is still compared bitwise between the uninterrupted and the
restarted run.
Sign convention: as declared by the owning Functors.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
import torch

from relic.core.functor import SubFlowAccuracy
from relic.core.simulator import RELICSimulator
from relic.core.simulator import checkpoint as checkpoint_module
from relic.parameters import get_global_manager

_TEST_ROOT = Path(__file__).resolve().parents[1]


def _load_test_module(relative: str, alias: str):
    """Load an existing test module by path to reuse its fixtures (no copies).

    The module is entered in sys.modules so that inspect finds the source of its
    classes (the provider kernel identity of a durable checkpoint needs it).
    """
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
_BUDGET = _load_test_module(
    "integration/pedosphere/budget_ledger_governed_multistep.test.py", "_cr_a04_budget_fixtures"
)

# Runtime fields of the splitting controls and the scheduling entries they resolve.
_CONTROLS = {
    "_splitting_mode": "computation.scheduling.splitting_mode",
    "_splitting_policy": "computation.scheduling.splitting_policy",
    "_fast_order_mode": "computation.scheduling.fast_order",
    "_substep_overrides": "computation.scheduling.substep_overrides",
    "_multirate_exchanges": "computation.scheduling.multirate_exchanges",
}
# Fields a checkpoint written before CR-A-04 step 3 does not carry.
_NEW_FIELDS = ("_splitting_policy", "_fast_order_mode", "_substep_overrides", "_multirate_exchanges")
_DOCUMENTED_DEFAULTS = {
    "_splitting_mode": "lie",
    "_splitting_policy": "warn",
    "_fast_order_mode": "forward",
    "_substep_overrides": {},
    "_multirate_exchanges": (),
}


def _rewrite_runtime(source: Path, target: Path, mutate: Callable[[Dict[str, Any]], None]) -> Path:
    """Write a resealed copy of a checkpoint whose decoded runtime ``mutate`` changed.

    Stands for a checkpoint written by another scheduler version (fields absent)
    or for a damaged or foreign payload (values outside the contract); the
    integrity record is recomputed with the owner's sealing function.
    """
    payload = torch.load(source, map_location="cpu", weights_only=True)
    runtime = {
        name: checkpoint_module._decode_checkpoint_value(value, path=f"runtime.{name}")
        for name, value in payload["runtime"].items()
    }
    mutate(runtime)
    body = {key: payload[key] for key in checkpoint_module._CHECKPOINT_BODY_FIELDS}
    body["runtime"] = {
        name: checkpoint_module._encode_checkpoint_value(value, path=f"runtime.{name}")
        for name, value in runtime.items()
    }
    torch.save(checkpoint_module._seal_checkpoint(body, payload["delivery_state"]), target)
    return target


def _saved_runtime(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        name: checkpoint_module._decode_checkpoint_value(value, path=f"runtime.{name}")
        for name, value in payload["runtime"].items()
    }


def _record(simulator: RELICSimulator) -> Dict[str, Any]:
    return {
        "view": simulator.get_committed_provider_view(),
        "epoch": simulator.provider_epoch,
        "order_fast": tuple(simulator._order_fast),
        "controls": {name: getattr(simulator, name) for name in _CONTROLS},
        "state": {key: value.clone() for key, value in simulator.state_manager.get_state().items()},
        "current_time": simulator.current_time,
        "step_count": simulator.step_count,
    }


def _assert_unchanged(simulator: RELICSimulator, before: Dict[str, Any]) -> None:
    """A refused restore leaves the committed view, the controls and the state unchanged."""
    assert simulator.get_committed_provider_view() is before["view"]
    assert simulator.provider_epoch == before["epoch"]
    assert tuple(simulator._order_fast) == before["order_fast"]
    assert {name: getattr(simulator, name) for name in _CONTROLS} == before["controls"]
    state = simulator.state_manager.get_state()
    assert set(state) == set(before["state"])
    for key, value in before["state"].items():
        assert torch.equal(state[key], value), key
    assert simulator.current_time == before["current_time"]
    assert simulator.step_count == before["step_count"]


def _relaxation_simulator(calls: List[tuple], *, dependencies=None, accuracy=None, **config) -> RELICSimulator:
    """Two (or three) fixture relaxations; relax_b may declare a dependency on relax_a."""
    simulator = _T1._fixture_simulator(**config)
    accuracy = SubFlowAccuracy.UNDECLARED if accuracy is None else accuracy
    simulator.register_functor(_T1._Relaxation("relax_a", "relax_a", accuracy=accuracy, call_log=calls))
    simulator.register_functor(
        _T1._Relaxation("relax_b", "relax_b", accuracy=accuracy, dependencies=dependencies, call_log=calls)
    )
    return simulator


def _relaxation_row(simulator: RELICSimulator) -> Dict[str, torch.Tensor]:
    state = simulator.state_manager.get_state()
    return {name: state[name].clone() for name in ("relax_a", "relax_b", "relax_c")}


# --------------------------------------------------------------------------
# T6: Strang checkpoint/restore continues bitwise (pinned before step 3).
# --------------------------------------------------------------------------

_N_DT = 600.0
_N_HALF = 6
_N_POOLS = ("organic_nitrogen", "microbial_nitrogen", "nh4_nitrogen", "no3_nitrogen")
_N_RATES = ("nitrification_flux", "n2o_flux", "denitrification_flux", "n2_production", "nitrogen_leaching_flux")


# Prescribed (held) inputs of mineralization, which the full G04 chain receives
# from first-order decomposition and the litterfall mapper; that relay is refused
# under Strang by S1 (T1), so it is replaced by constant forcing here.
_N_PRESCRIBED = {
    "carbon_decomposition_rate": 1.0e-5,  # g C m-2 s-1 (SCM flux contract)
    "organic_nitrogen_input": 1.0e-7,  # g N m-2 s-1 (SCM flux contract)
}


def _nitrogen_simulator() -> RELICSimulator:
    """Mineralization, nitrification, denitrification and nitrate leaching, all fast, Strang.

    The members share the organic, microbial, NH4 and NO3 states (state-coupled
    transfers, admissible under S1); the G04 forcing state of
    budget_ledger_governed_multistep.test.py is reused.
    """
    from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess
    from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess
    from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
    from relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching import NitrogenLeachingProcess

    simulator = RELICSimulator(
        {
            "device": "cpu",
            "splitting_mode": "strang",
            "conservation_checks": False,
            "strict_dependencies": True,
            "strict_scale_detection": False,
        }
    )
    initial = {key: value.clone() for key, value in _BUDGET._initial_state().items()}
    initial.update({key: torch.tensor(value, dtype=torch.float64) for key, value in _N_PRESCRIBED.items()})
    simulator.initialize_state(initial)
    for functor in (
        MineralizationProcess(),
        NitrificationProcess(),
        DenitrificationProcess(),
        NitrogenLeachingProcess(),
    ):
        simulator.register_functor(functor, execution_group="fast")
    assert simulator.get_committed_provider_view().order_fast == (
        "mineralization",
        "nitrification",
        "denitrification",
        "nitrogen_leaching",
    )
    assert set(simulator.get_splitting_order_report()["member_class"].values()) == {"stateful"}
    return simulator


def _nitrogen_row(simulator: RELICSimulator) -> Dict[str, torch.Tensor]:
    state = simulator.state_manager.get_state()
    return {key: state[key].clone() for key in _N_POOLS + _N_RATES}


# Pinned digests (computed on 2026-10-19 before CR-A-04 step 3; two identical runs,
# logs 04/06 of 20261019_cr_a04_impl_02).
_PINNED = {
    "strang_nitrogen_restart_pools": "1aa07f46aa72f89eefe93cb072032721de159bde1729e1daf94afe0c7c97fd14",
    "strang_reverse_restart": "fa3ffe50b6d40d2fdf321d3c04c42db087dfab5c1b1ae7374a86a4a84a1e1829",
}


def _assert_pinned(name: str, records: Dict[str, torch.Tensor]) -> None:
    observed = _T0._digest(records)
    assert observed == _PINNED[name], (
        f"T6 digest of '{name}' changed: {observed} != {_PINNED[name]}; "
        f"records: { {key: value.tolist() for key, value in records.items()} }"
    )


def test_strang_nitrogen_restart_continues_bitwise_and_is_pinned(tmp_path):
    writer = _nitrogen_simulator()
    for _ in range(_N_HALF):
        writer.step(dt=_N_DT)
    path = tmp_path / "strang_nitrogen.ckpt"
    writer.create_checkpoint(path)
    uninterrupted = []
    for _ in range(_N_HALF):
        writer.step(dt=_N_DT)
        uninterrupted.append(_nitrogen_row(writer))

    resumed = _nitrogen_simulator()
    resumed.restore_checkpoint(path)
    assert resumed._splitting_mode == "strang"
    restarted = []
    for _ in range(_N_HALF):
        resumed.step(dt=_N_DT)
        restarted.append(_nitrogen_row(resumed))
    uninterrupted_records = _T0._stack(uninterrupted)
    restarted_records = _T0._stack(restarted)
    for key in uninterrupted_records:
        assert torch.equal(uninterrupted_records[key], restarted_records[key]), key
    _assert_pinned(
        "strang_nitrogen_restart_pools", {key: restarted_records[key] for key in _N_POOLS}
    )


def test_strang_reverse_order_restart_continues_bitwise_and_is_pinned(tmp_path):
    def build(calls: List[tuple]) -> RELICSimulator:
        simulator = _T1._fixture_simulator(splitting_mode="strang", fast_order="reverse")
        for name in ("relax_a", "relax_b", "relax_c"):
            simulator.register_functor(_T1._Relaxation(name, name, call_log=calls))
        assert simulator.get_committed_provider_view().order_fast == ("relax_c", "relax_b", "relax_a")
        return simulator

    writer_calls: List[tuple] = []
    writer = build(writer_calls)
    for _ in range(3):
        writer.step()
    path = tmp_path / "strang_reverse.ckpt"
    writer.create_checkpoint(path)
    del writer_calls[:]
    uninterrupted = []
    for _ in range(3):
        writer.step()
        uninterrupted.append(_relaxation_row(writer))

    resumed_calls: List[tuple] = []
    resumed = build(resumed_calls)
    resumed.restore_checkpoint(path)
    assert resumed.get_committed_provider_view().order_fast == ("relax_c", "relax_b", "relax_a")
    del resumed_calls[:]
    restarted = []
    for _ in range(3):
        resumed.step()
        restarted.append(_relaxation_row(resumed))
    # The palindrome over the reversed order: c and b run two half-steps, a one step.
    assert resumed_calls == writer_calls
    assert resumed_calls[:5] == [
        ("relax_c", 180.0, 30.0),
        ("relax_b", 180.0, 30.0),
        ("relax_a", 180.0, 60.0),
        ("relax_b", 210.0, 30.0),
        ("relax_c", 210.0, 30.0),
    ]
    uninterrupted_records = _T0._stack(uninterrupted)
    restarted_records = _T0._stack(restarted)
    for key in uninterrupted_records:
        assert torch.equal(uninterrupted_records[key], restarted_records[key]), key
    _assert_pinned("strang_reverse_restart", restarted_records)


# --------------------------------------------------------------------------
# The resolved controls are runtime fields (database values included).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, dict(_DOCUMENTED_DEFAULTS)),
        (
            {"splitting_mode": "strang", "splitting_policy": "require_declared", "fast_order": "reverse"},
            dict(
                _DOCUMENTED_DEFAULTS,
                _splitting_mode="strang",
                _splitting_policy="require_declared",
                _fast_order_mode="reverse",
            ),
        ),
    ],
)
def test_checkpoint_runtime_carries_the_resolved_splitting_controls(tmp_path, config, expected):
    simulator = _relaxation_simulator([], accuracy=SubFlowAccuracy.EXACT, **config)
    simulator.step()
    path = tmp_path / "controls.ckpt"
    simulator.create_checkpoint(path)
    runtime = _saved_runtime(path)
    for name, value in expected.items():
        assert name in runtime, name
        assert runtime[name] == value and type(runtime[name]) is type(value), name
    if not config:
        # Nothing configured: the saved values are the parameter database values.
        manager = get_global_manager()
        for name in ("_splitting_mode", "_splitting_policy", "_fast_order_mode"):
            assert runtime[name] == manager.get(_CONTROLS[name])


# --------------------------------------------------------------------------
# D1 at restore: restored controls are checked against their allowed values.
# --------------------------------------------------------------------------

_EXCHANGE = {"id": "relax_mean", "variable": "relax_a", "consumer_group": "medium", "mode": "interval_mean"}


def _checkpointed_relaxation(tmp_path: Path, **config):
    """A checkpoint after two steps and a target that ran one step of its own."""
    writer = _relaxation_simulator([], **config)
    writer.step()
    writer.step()
    path = tmp_path / "relaxation.ckpt"
    writer.create_checkpoint(path)
    target = _relaxation_simulator([], **config)
    target.step()
    return path, target


@pytest.mark.parametrize(
    "field, value",
    [
        ("_splitting_mode", "palindromic"),
        ("_splitting_mode", "Strang"),
        ("_splitting_mode", 2),
        ("_splitting_policy", "strict"),
        ("_splitting_policy", None),
        ("_fast_order_mode", "backward"),
        ("_substep_overrides", {"relax_a": {"count": 0}}),
        ("_substep_overrides", [("relax_a", 2)]),
        ("_multirate_exchanges", (dict(_EXCHANGE, mode="average"),)),
        ("_multirate_exchanges", (dict(_EXCHANGE, mode="interpolate"),)),
        # Not the canonical form the simulator writes (a tuple of exchanges).
        ("_multirate_exchanges", []),
    ],
)
def test_restore_refuses_a_splitting_control_outside_its_contract(tmp_path, field, value):
    path, target = _checkpointed_relaxation(tmp_path)
    tampered = _rewrite_runtime(path, tmp_path / "tampered.ckpt", lambda runtime: runtime.__setitem__(field, value))
    before = _record(target)
    with pytest.raises(ValueError, match=rf"Checkpoint runtime field '{field}'"):
        target.restore_checkpoint(tampered)
    _assert_unchanged(target, before)


# A restored well-formed sub-step override is executed since step 4
# (splitting_substep_overrides.test.py), a multirate exchange since step 6
# (splitting_multirate_exchanges.test.py). The checkpoint below was written
# without the exchange after two fast steps, with the medium group never run, so
# the restored exchange would start its mean in the middle of its interval.
def test_restore_refuses_an_interval_mean_exchange_whose_mean_is_missing_mid_interval(tmp_path):
    path, target = _checkpointed_relaxation(tmp_path)
    tampered = _rewrite_runtime(
        path, tmp_path / "tampered.ckpt",
        lambda runtime: runtime.__setitem__("_multirate_exchanges", (dict(_EXCHANGE),)),
    )
    before = _record(target)
    with pytest.raises(RuntimeError, match=r"relax_mean.*mean"):
        target.restore_checkpoint(tampered)
    _assert_unchanged(target, before)
    assert target._multirate_exchanges == ()
    assert "exchange_mean__relax_mean" not in target.state_manager.schema


# --------------------------------------------------------------------------
# A restore republishes through the admissibility resolver (S1, S3, S5).
# --------------------------------------------------------------------------


def _snow_bucket(splitting_mode: str) -> RELICSimulator:
    from relic.scientific.hydrosphere.snow.snowpack import SnowPack
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture

    simulator = _T1._snow_bucket_simulator(splitting_mode)
    simulator.register_functor(SnowPack(), execution_group="fast")
    simulator.register_functor(BucketSoilMoisture(config=dict(_T1._RAIN_AND_MELT)), execution_group="fast")
    return simulator


def _assert_runs_like_an_untouched_twin(target: RELICSimulator, twin: RELICSimulator, dt: float) -> None:
    target.step(dt=dt)
    twin.step(dt=dt)
    state, expected = target.state_manager.get_state(), twin.state_manager.get_state()
    for key, value in expected.items():
        assert torch.equal(state[key], value), key


def test_restore_into_strang_with_a_stateful_flux_transfer_is_refused_and_rolled_back(tmp_path):
    """S1 at restore: snow_pack -> bucket via snow_melt_flux and rain_flux."""
    writer = _snow_bucket("lie")
    writer.step(dt=3600.0)
    writer.step(dt=3600.0)
    path = tmp_path / "snow_bucket_lie.ckpt"
    writer.create_checkpoint(path)
    tampered = _rewrite_runtime(
        path, tmp_path / "snow_bucket_strang.ckpt", lambda runtime: runtime.__setitem__("_splitting_mode", "strang")
    )
    target, twin = _snow_bucket("lie"), _snow_bucket("lie")
    target.step(dt=3600.0)
    twin.step(dt=3600.0)
    before = _record(target)
    with pytest.raises(RuntimeError, match=r"S1") as refusal:
        target.restore_checkpoint(tampered)
    assert "snow_pack" in str(refusal.value) and "bucket_soil_moisture" in str(refusal.value)
    _assert_unchanged(target, before)
    assert target._splitting_mode == "lie"
    _assert_runs_like_an_untouched_twin(target, twin, 3600.0)


def test_restore_of_a_reverse_order_over_a_same_step_edge_is_refused_and_rolled_back(tmp_path):
    """S5 at restore: relax_b requires relax_a in the same step."""
    edge = {"required": ["relax_a"], "optional": [], "lagged": []}
    path, target = _checkpointed_relaxation(tmp_path, dependencies=edge)
    twin = _relaxation_simulator([], dependencies=edge)
    twin.step()
    tampered = _rewrite_runtime(
        path, tmp_path / "reverse.ckpt", lambda runtime: runtime.__setitem__("_fast_order_mode", "reverse")
    )
    before = _record(target)
    with pytest.raises(RuntimeError, match=r"S5.*relax_b.*relax_a"):
        target.restore_checkpoint(tampered)
    _assert_unchanged(target, before)
    assert target._fast_order_mode == "forward"
    _assert_runs_like_an_untouched_twin(target, twin, 60.0)


def test_restore_of_require_declared_over_undeclared_stateful_members_is_refused_and_rolled_back(tmp_path):
    """S3 at restore: two undeclared stateful members under Strang."""
    path, target = _checkpointed_relaxation(tmp_path, splitting_mode="strang")
    twin = _relaxation_simulator([], splitting_mode="strang")
    twin.step()
    tampered = _rewrite_runtime(
        path, tmp_path / "require_declared.ckpt",
        lambda runtime: runtime.__setitem__("_splitting_policy", "require_declared"),
    )
    before = _record(target)
    with pytest.raises(RuntimeError, match=r"S3"):
        target.restore_checkpoint(tampered)
    _assert_unchanged(target, before)
    assert target._splitting_policy == "warn"
    _assert_runs_like_an_untouched_twin(target, twin, 60.0)


# --------------------------------------------------------------------------
# Checkpoints written before the controls existed.
# --------------------------------------------------------------------------


def test_documented_defaults_equal_the_scheduler_before_the_controls():
    """The defaults an old checkpoint restores with are the pre-CR-A-04 behaviour."""
    manager = get_global_manager()
    defaults = {name: manager.get_parameter(key).default for name, key in _CONTROLS.items()}
    assert defaults == {
        "_splitting_mode": "lie",
        "_splitting_policy": "warn",
        "_fast_order_mode": "forward",
        "_substep_overrides": {},
        "_multirate_exchanges": [],
    }


def test_checkpoint_without_the_new_controls_restores_the_documented_defaults(tmp_path):
    """The saved run executed the forward order without policy; the restore continues it."""
    config = {"splitting_mode": "strang", "splitting_policy": "require_declared", "fast_order": "reverse"}
    writer = _relaxation_simulator([], accuracy=SubFlowAccuracy.EXACT, **config)
    writer.step()
    path = tmp_path / "current.ckpt"
    writer.create_checkpoint(path)

    def strip(runtime: Dict[str, Any]) -> None:
        for name in _NEW_FIELDS:
            runtime.pop(name, None)

    old = _rewrite_runtime(path, tmp_path / "old.ckpt", strip)
    assert not set(_NEW_FIELDS) & set(_saved_runtime(old))
    calls: List[tuple] = []
    target = _relaxation_simulator(calls, accuracy=SubFlowAccuracy.EXACT, **config)
    assert target.get_committed_provider_view().order_fast == ("relax_b", "relax_a")
    epoch = target.provider_epoch
    target.restore_checkpoint(old)
    assert target.provider_epoch == epoch + 1
    assert {name: getattr(target, name) for name in _CONTROLS} == dict(_DOCUMENTED_DEFAULTS, _splitting_mode="strang")
    assert target.get_committed_provider_view().order_fast == ("relax_a", "relax_b")
    report = target.get_splitting_order_report()
    assert report["fast_order_mode"] == "forward" and report["splitting_policy"] == "warn"
    del calls[:]
    target.step()
    assert calls == [("relax_a", 60.0, 30.0), ("relax_b", 60.0, 60.0), ("relax_a", 90.0, 30.0)]


def test_checkpoint_without_any_splitting_control_restarts_bitwise(tmp_path):
    writer = _relaxation_simulator([])
    for _ in range(3):
        writer.step()
    path = tmp_path / "current.ckpt"
    writer.create_checkpoint(path)

    def strip(runtime: Dict[str, Any]) -> None:
        for name in _CONTROLS:
            runtime.pop(name, None)

    old = _rewrite_runtime(path, tmp_path / "old.ckpt", strip)
    uninterrupted = []
    for _ in range(3):
        writer.step()
        uninterrupted.append(_relaxation_row(writer))
    resumed = _relaxation_simulator([])
    resumed.restore_checkpoint(old)
    assert {name: getattr(resumed, name) for name in _CONTROLS} == _DOCUMENTED_DEFAULTS
    restarted = []
    for _ in range(3):
        resumed.step()
        restarted.append(_relaxation_row(resumed))
    for expected, observed in zip(uninterrupted, restarted):
        for key in expected:
            assert torch.equal(expected[key], observed[key]), key


def test_local_transaction_restore_refuses_controls_that_differ_from_the_published_plan(tmp_path):
    """A process-local restore publishes no provider view, so it cannot change the controls.

    The published fast order belongs to the committed view; restoring different
    splitting controls without republishing would leave the view and the
    controls inconsistent, so only the durable restore (which republishes) may.
    """
    from relic.core.simulator.checkpoint import capture_runtime_transaction, restore_runtime_transaction

    config = {"splitting_mode": "strang", "splitting_policy": "require_declared", "fast_order": "reverse"}
    writer = _relaxation_simulator([], accuracy=SubFlowAccuracy.EXACT, **config)
    writer.step()
    path = tmp_path / "current.ckpt"
    writer.create_checkpoint(path)

    def strip(runtime: Dict[str, Any]) -> None:
        for name in _NEW_FIELDS:
            runtime.pop(name, None)

    old = _rewrite_runtime(path, tmp_path / "old.ckpt", strip)
    target = _relaxation_simulator([], accuracy=SubFlowAccuracy.EXACT, **config)
    transaction = capture_runtime_transaction(target)
    target.restore_checkpoint(old)
    before = _record(target)
    with pytest.raises(RuntimeError, match=r"splitting control"):
        restore_runtime_transaction(target, transaction)
    _assert_unchanged(target, before)
    # Equal controls restore as before (the ensemble rollback path).
    same = capture_runtime_transaction(target)
    target.step()
    restore_runtime_transaction(target, same)
    _assert_unchanged(target, before)
