"""CR-A-04 T9: per-run sub-step overrides enter the provider binding and are executed.

Design 20261019_cr_a04_design_01, sections 3 and 4 (step 4): the simulator setting
substep_overrides ({functor: {count: n}} or {functor: {max_seconds: s}},
computation.scheduling.substep_overrides) replaces FunctorMetadata.substep_count of
the named Functor. The override is recorded in the provider binding payload of the
registration owner, so it enters the binding digest of the committed provider view
(and through it the exact-restart topology), and the group executor reads it when it
sub-steps a call: n = count, or n = ceil(dt_call / s) for a call over dt_call.
A Functor without an override keeps its binding digest (pinned before step 4), so
checkpoints written before the change keep their topology identity.

References: Gear and Wells (1984) BIT 24(4):484-502 (multirate methods, different step
sizes for different components); Hairer, Lubich and Wanner (2006) Geometric Numerical
Integration, 2nd ed., Sect. II.5 (sub-flows of a splitting); RELIC architecture
overview 4.4.1 (sub-stepping through FunctorMetadata.substep_count).
Mathematical expression: n = count; n = ceil(dt_call / s); h = dt_call / n; under
Strang the members before the last run two calls over dt_call = h_fast / 2 each.
Parameter meanings: n the number of sub-steps of one call; s the largest admissible
sub-step; dt_call the interval of one group call; h the sub-step; h_fast the fast step.
Units: dt_call, s, h, h_fast [s]; n dimensionless; fixtures dimensionless.
Domain/constraints: n integer >= 1; s finite > 0; dt_call > 0.
Sign convention: none (positive magnitudes); relaxation decreases |x|.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

from relic.core.simulator import RELICSimulator

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


_T1 = _load_test_module("core/splitting_admissibility.test.py", "_cr_a04_t1_admissibility")
_T6 = _load_test_module("core/splitting_checkpoint_restore.test.py", "_cr_a04_t6_checkpoint_restore")

# Binding digests of the two fixture relaxations without overrides, computed on
# 2026-10-19 before CR-A-04 step 4 (registration.py); unchanged by the step.
_PINNED_BINDING = {
    "relax_a": "770cc4d551d4849347ea20c7e4566c181d4024731fb5a9fb7eb0aa8b13820b68",
    "relax_b": "ac2d0c98d909218c16114f4ae44977d33db1c7ca647e41ce0b1b9f399672447d",
}


def _simulator(calls: Optional[List[tuple]] = None, *, substep_count: Optional[int] = None, **config):
    calls = [] if calls is None else calls
    simulator = _T1._fixture_simulator(**config)
    relax_a = _T1._Relaxation("relax_a", "relax_a", call_log=calls)
    relax_b = _T1._Relaxation("relax_b", "relax_b", call_log=calls)
    if substep_count is not None:
        relax_b.metadata.substep_count = substep_count
    simulator.register_functor(relax_a)
    simulator.register_functor(relax_b)
    return simulator


def _identity(simulator: RELICSimulator, name: str):
    return simulator.get_committed_provider_view().providers[name]


def test_binding_digest_without_overrides_is_unchanged():
    simulator = _simulator()
    for name, digest in _PINNED_BINDING.items():
        assert _identity(simulator, name).binding_digest == digest, (
            name, _identity(simulator, name).binding_digest
        )


def test_substep_override_enters_the_binding_digest_of_its_functor_only():
    plain = _simulator()
    overridden = _simulator(substep_overrides={"relax_b": {"count": 3}})
    assert _identity(overridden, "relax_a").binding_digest == _identity(plain, "relax_a").binding_digest
    assert _identity(overridden, "relax_b").binding_digest != _identity(plain, "relax_b").binding_digest
    for name in ("relax_a", "relax_b"):
        assert _identity(overridden, name).contract_digest == _identity(plain, name).contract_digest
        assert _identity(overridden, name).implementation_digest == _identity(plain, name).implementation_digest


def test_distinct_overrides_give_distinct_binding_digests():
    digests = {
        repr(override): _identity(_simulator(substep_overrides={"relax_b": override}), "relax_b").binding_digest
        for override in ({"count": 3}, {"count": 4}, {"max_seconds": 20.0}, {"max_seconds": 30.0})
    }
    digests["none"] = _identity(_simulator(), "relax_b").binding_digest
    assert len(set(digests.values())) == len(digests), digests


def test_count_override_runs_n_substeps_and_replaces_the_declared_count():
    calls: List[tuple] = []
    simulator = _simulator(calls, substep_count=4, substep_overrides={"relax_b": {"count": 3}})
    simulator.step()
    assert calls == [
        ("relax_a", 0.0, 60.0),
        ("relax_b", 0.0, 20.0),
        ("relax_b", 20.0, 20.0),
        ("relax_b", 40.0, 20.0),
    ]
    declared: List[tuple] = []
    reference = _simulator(declared, substep_count=4)
    reference.step()
    assert [entry for entry in declared if entry[0] == "relax_b"] == [
        ("relax_b", 0.0, 15.0), ("relax_b", 15.0, 15.0), ("relax_b", 30.0, 15.0), ("relax_b", 45.0, 15.0)
    ]


def test_max_seconds_override_uses_the_ceiling_of_the_call_over_max_seconds():
    calls: List[tuple] = []
    simulator = _simulator(calls, substep_overrides={"relax_b": {"max_seconds": 25.0}})
    simulator.step()
    assert [entry for entry in calls if entry[0] == "relax_b"] == [
        ("relax_b", 0.0, 20.0), ("relax_b", 20.0, 20.0), ("relax_b", 40.0, 20.0)
    ]
    del calls[:]
    simulator.step(dt=50.0)
    assert [entry for entry in calls if entry[0] == "relax_b"] == [("relax_b", 60.0, 25.0), ("relax_b", 85.0, 25.0)]


def test_overrides_follow_the_strang_half_steps_and_the_report():
    calls: List[tuple] = []
    simulator = _simulator(
        calls,
        splitting_mode="strang",
        substep_overrides={"relax_a": {"max_seconds": 25.0}, "relax_b": {"count": 3}, "relax_z": {"count": 2}},
    )
    simulator.step()
    # relax_a runs two half-steps of 30 s, each ceil(30 / 25) = 2 sub-steps of 15 s;
    # relax_b (last) runs one call of 60 s in 3 sub-steps.
    assert calls == [
        ("relax_a", 0.0, 15.0),
        ("relax_a", 15.0, 15.0),
        ("relax_b", 0.0, 20.0),
        ("relax_b", 20.0, 20.0),
        ("relax_b", 40.0, 20.0),
        ("relax_a", 30.0, 15.0),
        ("relax_a", 45.0, 15.0),
    ]
    report = simulator.get_splitting_order_report()
    assert report["effective_substep_seconds"] == {"relax_a": 15.0, "relax_b": 20.0}
    assert report["substep_overrides"] == {
        "relax_a": {"max_seconds": 25.0}, "relax_b": {"count": 3}, "relax_z": {"count": 2}
    }
    # An override that names no registered Functor is reported, not silently dropped.
    assert report["unmatched_substep_overrides"] == ("relax_z",)


def test_checkpoint_restart_with_overrides_continues_bitwise(tmp_path):
    overrides = {"relax_a": {"max_seconds": 25.0}, "relax_b": {"count": 3}}
    writer = _simulator(splitting_mode="strang", substep_overrides=overrides)
    writer.step()
    writer.step()
    path = tmp_path / "overrides.ckpt"
    writer.create_checkpoint(path)
    assert _T6._saved_runtime(path)["_substep_overrides"] == overrides
    uninterrupted = []
    for _ in range(3):
        writer.step()
        uninterrupted.append(_T6._relaxation_row(writer))
    resumed = _simulator(splitting_mode="strang", substep_overrides=overrides)
    resumed.restore_checkpoint(path)
    restarted = []
    for _ in range(3):
        resumed.step()
        restarted.append(_T6._relaxation_row(resumed))
    for expected, observed in zip(uninterrupted, restarted):
        for key in expected:
            assert torch.equal(expected[key], observed[key]), key


def test_restored_override_is_republished_in_the_binding_and_executed(tmp_path):
    writer = _simulator()
    writer.step()
    path = tmp_path / "plain.ckpt"
    writer.create_checkpoint(path)
    restored = _T6._rewrite_runtime(
        path, tmp_path / "override.ckpt",
        lambda runtime: runtime.__setitem__("_substep_overrides", {"relax_b": {"count": 3}}),
    )
    calls: List[tuple] = []
    target = _simulator(calls)
    plain_digest = _identity(target, "relax_b").binding_digest
    target.restore_checkpoint(restored)
    assert target._substep_overrides == {"relax_b": {"count": 3}}
    assert _identity(target, "relax_b").binding_digest != plain_digest
    assert _identity(target, "relax_b").binding_digest == _identity(
        _simulator(substep_overrides={"relax_b": {"count": 3}}), "relax_b"
    ).binding_digest
    del calls[:]
    target.step()
    assert [entry for entry in calls if entry[0] == "relax_b"] == [
        ("relax_b", 60.0, 20.0), ("relax_b", 80.0, 20.0), ("relax_b", 100.0, 20.0)
    ]
