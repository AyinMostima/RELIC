"""
Step/run execution helpers for RELICSimulator.

This module extracts time-loop logic from ``main.py``.
"""

from __future__ import annotations

import itertools
import hashlib
import json
import logging
import math
import os
import time
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

import torch
from torch import Tensor

from relic.core import device_policy

from ...parameters import get_global_manager
from ..abstractions import FluxDict
from ..dependency_graph import normalize_dependency_declaration
from ..functor.functor_types import SubFlowAccuracy, VariableKind
from ..state.manager import StateManager, UpdateItem, UpdateMode
from ..state import ExternalIntent
from ..state.types import StateVariableSpec, StateWriteMode, partition_outputs
from .checkpoint import (
    _capture_rng_state, _restore_rng_state, capture_checkpoint_payload,
    checkpoint_delivery_snapshot,
)
from .diagnostics import capture_diagnostic_export
from .execution_group import (
    EXCHANGE_ROLE,
    GroupComposition,
    build_commit_context,
    merge_sequential_updates,
    replaces_earlier_output,
    resolve_call_substep_count,
)
from .dependency_ops import get_temporal_zone_label
from .time_ops import commit_rate_group_interval

logger = logging.getLogger(__name__)


_PERFORMANCE_STAGE_KEYS = (
    "manager_snapshot",
    "runtime_snapshot",
    "functor_state_snapshot",
    "rng_snapshot",
    "fast_group",
    "medium_group",
    "slow_group",
    "distributed_collective",
    "cross_domain_validation",
    "conservation",
    "state_apply_updates",
    "output",
    "diagnostic_export",
    "checkpoint",
    "cpu_gpu_transfer",
    "cuda_compute",
    "functor_forward",
    "functor_validation",
    "functor_conservation",
    "step_compute",
    "step_total",
)


def _new_step_stage_timings(enabled: bool) -> Optional[Dict[str, float]]:
    """Create one fixed-schema stage record only when profiling is enabled."""
    if not enabled:
        return None
    return {stage_name: 0.0 for stage_name in _PERFORMANCE_STAGE_KEYS}


def _stage_start(stage_timings: Optional[Dict[str, float]]) -> Optional[float]:
    """Return a wall-clock marker without adding work to disabled profiling."""
    return time.perf_counter() if stage_timings is not None else None


def _record_stage(
    stage_timings: Optional[Dict[str, float]],
    stage_name: str,
    started_at: Optional[float],
) -> None:
    """Accumulate one nonnegative wall-clock stage duration in seconds."""
    if stage_timings is None or started_at is None:
        return
    stage_timings[stage_name] += max(0.0, time.perf_counter() - started_at)


def _commit_performance_diagnostics(
    simulator: Any,
    stage_timings: Optional[Dict[str, float]],
    functor_timing_snapshot: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Commit accepted-step timings to the simulator diagnostics owner."""
    if stage_timings is None:
        return

    functor_step_deltas: Dict[str, Dict[str, Any]] = {}
    for functor_name, before in (functor_timing_snapshot or {}).items():
        if functor_name not in simulator.functors:
            continue
        functor = simulator.functors[functor_name]
        before_count = int(before.get("count", 0))
        after_count = int(getattr(functor, "_pipeline_timing_count", 0))
        call_count = after_count - before_count
        if call_count <= 0:
            continue

        before_totals = before.get("totals_s", {})
        after_totals = getattr(functor, "_pipeline_timing_totals_s", {})
        stage_delta = {
            stage_name: max(
                0.0,
                float(after_totals.get(stage_name, 0.0))
                - float(before_totals.get(stage_name, 0.0)),
            )
            for stage_name in after_totals
        }
        functor_step_deltas[functor_name] = {
            "calls": call_count,
            "stage_s": stage_delta,
        }
        stage_timings["functor_forward"] += stage_delta.get("forward", 0.0)
        stage_timings["functor_validation"] += stage_delta.get(
            "state_validation",
            0.0,
        ) + stage_delta.get("framework_validation", 0.0)
        stage_timings["functor_conservation"] += stage_delta.get(
            "local_conservation",
            0.0,
        )

    last_step = {
        stage_name: max(0.0, float(stage_timings.get(stage_name, 0.0)))
        for stage_name in _PERFORMANCE_STAGE_KEYS
    }
    existing = getattr(simulator, "_performance_diagnostics", None)
    if not isinstance(existing, dict) or existing.get("schema_version") != 1:
        existing = {
            "schema_version": 1,
            "accepted_steps": 0,
            "last_step_s": {stage_name: 0.0 for stage_name in _PERFORMANCE_STAGE_KEYS},
            "cumulative_s": {stage_name: 0.0 for stage_name in _PERFORMANCE_STAGE_KEYS},
            "functors": {},
        }

    cumulative = existing.get("cumulative_s")
    if not isinstance(cumulative, dict):
        cumulative = {stage_name: 0.0 for stage_name in _PERFORMANCE_STAGE_KEYS}
    for stage_name, duration in last_step.items():
        cumulative[stage_name] = max(0.0, float(cumulative.get(stage_name, 0.0))) + duration

    existing["accepted_steps"] = int(existing.get("accepted_steps", 0)) + 1
    existing["last_step_s"] = last_step
    existing["cumulative_s"] = cumulative

    functor_diagnostics = existing.get("functors")
    if not isinstance(functor_diagnostics, dict):
        functor_diagnostics = {}
    for functor_name, delta in functor_step_deltas.items():
        stage_delta = delta["stage_s"]
        current = functor_diagnostics.get(functor_name)
        if not isinstance(current, dict):
            current = {
                "calls": 0,
                "last_step_calls": 0,
                "last_step_s": {key: 0.0 for key in stage_delta},
                "cumulative_s": {key: 0.0 for key in stage_delta},
            }
        current_cumulative = current.get("cumulative_s")
        if not isinstance(current_cumulative, dict):
            current_cumulative = {key: 0.0 for key in stage_delta}
        for stage_name, duration in stage_delta.items():
            current_cumulative[stage_name] = max(
                0.0,
                float(current_cumulative.get(stage_name, 0.0)),
            ) + duration
        current["calls"] = int(current.get("calls", 0)) + int(delta["calls"])
        current["last_step_calls"] = int(delta["calls"])
        current["last_step_s"] = dict(stage_delta)
        current["cumulative_s"] = current_cumulative
        functor_diagnostics[functor_name] = current
    existing["functors"] = functor_diagnostics
    simulator._performance_diagnostics = existing


class _AppendOnlyHistorySavepoint:
    """Short transaction savepoint for one append-only in-memory history map."""

    # References: Gray and Reuter (1993), Transaction Processing: Concepts and
    #   Techniques (savepoint and rollback to a savepoint); RELIC accepted-step
    #   transaction T_step (docs/source/architecture_overview.md);
    #   RELICSimulator.get_variable_provenance (writer records accumulate across
    #   accepted steps).
    # Mathematical expression: at the savepoint n_v = len(H_v) for every v in V;
    #   rollback sets H_v := H_v[0:n_v] for v in V and removes every v not in V.
    #   Cost: O(|V|) to open the savepoint, O(|V| + a) to roll back.
    # Parameter meanings: H_v is the list of records of variable v in the map held
    #   by the simulator attribute ``attribute`` (time_series: stored outputs;
    #   _step_provenance_detail: writer records); V the variables present at the
    #   savepoint; n_v the saved length; a the number of records appended since.
    # Units: n_v, |V| and a are counts (dimensionless); the records keep their own
    #   units and are neither visited nor cloned.
    # Domain/constraints: inside the transaction a history only grows at its end,
    #   either in place or as a copy stored in the map (the saved list then keeps
    #   its n_v records); a saved list shorter than n_v at rollback raises. Every
    #   map entry must be a list.
    # Sign convention: not applicable (no flux or signed quantity).

    __slots__ = ("attribute", "mapping", "entries")

    def __init__(self, attribute: str, mapping: Dict[str, Any]) -> None:
        self.attribute = attribute
        self.mapping = mapping
        self.entries: Dict[str, Tuple[List[Any], int]] = {}
        for variable_name, history in mapping.items():
            if not isinstance(history, list):
                raise TypeError(
                    f"Simulator {attribute} entries must be lists inside a step "
                    "transaction."
                )
            self.entries[variable_name] = (history, len(history))


def _restore_history_savepoint(
    simulator: Any,
    savepoint: _AppendOnlyHistorySavepoint,
) -> None:
    """Rollback appended history records without cloning the earlier ones."""
    target = savepoint.mapping
    original_names = set(savepoint.entries)
    for variable_name in list(target):
        if variable_name not in original_names:
            del target[variable_name]

    for variable_name, (history, original_length) in savepoint.entries.items():
        if len(history) < original_length:
            raise RuntimeError(
                f"Simulator {savepoint.attribute} history was truncated inside an "
                "active append-only step transaction."
            )
        del history[original_length:]
        target[variable_name] = history

    setattr(simulator, savepoint.attribute, target)


def _clone_runtime_value(value: Any) -> Any:
    """Clone mutable simulator bookkeeping used by a step transaction."""
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, set):
        return {_clone_runtime_value(item) for item in value}
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    return value


def _snapshot_runtime_value(name: str, value: Any) -> Any:
    """Savepoint an append-only history map; clone any other bookkeeping value."""
    if name == "time_series" or (
        name == "_step_provenance_detail" and isinstance(value, dict)
    ):
        return _AppendOnlyHistorySavepoint(name, value)
    return _clone_runtime_value(value)


def _restore_runtime_value(simulator: Any, name: str, snapshot: Any) -> None:
    """Restore one value captured by ``_snapshot_runtime_value``."""
    if isinstance(snapshot, _AppendOnlyHistorySavepoint):
        _restore_history_savepoint(simulator, snapshot)
    else:
        setattr(simulator, name, _clone_runtime_value(snapshot))


def _clone_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """Clone state mapping to support adaptive step rejection rollback."""
    snapshot: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, Tensor):
            snapshot[key] = value.clone()
        else:
            snapshot[key] = value
    return snapshot


def _restore_state_snapshot(simulator: Any, snapshot: Dict[str, Any]) -> None:
    """Restore simulator state from a previously captured snapshot."""
    state_manager = getattr(simulator, "state_manager", None)
    if state_manager is None:
        return
    if hasattr(state_manager, "_state_variables"):
        state_manager._state_variables = _clone_state_snapshot(snapshot)
        return
    state_manager.set_state(_clone_state_snapshot(snapshot), validate=False)


def _snapshot_functor_state_dicts(simulator: Any) -> Dict[str, Dict[str, Any]]:
    """Capture parameter and buffer state for an accepted-step transaction."""
    functors = getattr(simulator, "functors", {})
    return {
        str(name): _clone_runtime_value(functor.state_dict())
        for name, functor in functors.items()
    }


def _restore_functor_state_dicts(
    simulator: Any,
    snapshot: Dict[str, Dict[str, Any]],
) -> None:
    """Restore parameter and buffer state after rejection or failure."""
    functors = getattr(simulator, "functors", {})
    if set(functors) != set(snapshot):
        raise RuntimeError(
            "Functor registry changed inside an active step transaction."
        )
    for name, state_dict in snapshot.items():
        functors[name].load_state_dict(
            _clone_runtime_value(state_dict),
            strict=True,
        )


def _compute_error_norm(
    simulator: Any,
    state_before: Dict[str, Any],
    updates: Dict[str, Any],
) -> float:
    """Compute the adaptive-step control metric from the committed state change.

    ``state_before`` is the snapshot taken before the fast group; the state
    after it is the one StateManager committed (``simulator._get_current_state()``),
    so every update form is measured by what was actually committed: a REPLACE
    value, an integrated amount, a per-second rate or FLUX that StateManager or
    the group owner integrated once over the real interval (with substeps and
    Strang half-steps), a declared boundary-rate aggregate that StateManager
    stores as a rate, and clipping or distributed reduction applied at commit.
    ``updates`` (the merged fast-group return) only names the written
    variables; its payloads are not re-interpreted here.

    References:
        Hairer and Wanner (1996), ODE II, Sect. IV.2 (weighted RMS norm);
        Uruci (2018), Appendix A Eq. A.5; LeVeque (2002), realized storage
        change of a finite-volume commit (StateManager.apply_updates).
    Mathematical expression:
        tol_k = atol + rtol * |x_k|
        u_k   = x'_k - x_k
        e_k   = mean_elements((u_k / tol_k)^2)
        Err   = sqrt((1 / m) * sum_k e_k)
    Parameter meanings:
        x_k is variable k before the fast group (``state_before``); x'_k is its
        committed value after the fast group; u_k is the committed change;
        atol and rtol are ``adaptive_atol`` and ``adaptive_rtol``; the k run
        over distinct StateManager targets the fast group wrote (the state channel of
        ``partition_outputs(updates)``; diagnostics and DERIVED outputs never
        change the state and are excluded), restricted to
        ``adaptive_monitor_vars`` when it is set, that have a numeric pre-step
        value; m is their number (equal weight).
    Units: x_k, x'_k, u_k and atol in the storage units of variable k (for a
        declared boundary-rate aggregate the stored value is a rate, so u_k is
        the change of that rate); rtol, e_k and Err are dimensionless.
    Domain/constraints: a written variable without a numeric pre-step value is
        skipped; m = 0 gives Err = 0. A skipped variable is not an error: it
        has no pre-step reference to scale against. A variable with a pre-step
        value that is missing from the committed state, or whose shape changed
        in the fast group, cannot be measured and raises. The metric is an
        update-magnitude heuristic, not a Richardson, embedded-pair, or
        local-truncation-error estimate.
    Sign convention: u_k is positive when variable k increases; only |u_k|
        enters Err.
    """
    # atol > 0 and rtol >= 0 are validated where the settings are read
    # (RELICSimulator.__init__), so tol_k > 0 without a guard term.
    atol = float(simulator._adaptive_atol)
    rtol = float(simulator._adaptive_rtol)
    monitored = simulator._adaptive_monitor_vars
    monitored_set = set(monitored) if monitored else None
    state_after = simulator._get_current_state()
    written_state, _, _ = partition_outputs(updates)

    def _numeric(value: Any) -> Optional[Tensor]:
        if isinstance(value, Tensor):
            if value.dtype == torch.bool or value.is_complex():
                return None
            return value.detach().to(dtype=torch.float64)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return torch.tensor(float(value), dtype=torch.float64)
        return None

    sum_sq = 0.0
    count = 0
    # Measure each actual commit target once; raw delta_ names are not stores.
    written_targets = dict.fromkeys(
        StateManager._update_target_name(name) for name in written_state
    )
    for var_name in written_targets:
        if monitored_set is not None and var_name not in monitored_set:
            continue
        before = _numeric(state_before.get(var_name))
        if before is None:
            continue
        if var_name not in state_after:
            raise RuntimeError(
                f"Adaptive metric: '{var_name}' had a pre-step value but is missing "
                "from the committed state after the fast group"
            )
        after = _numeric(state_after[var_name])
        if after is None:
            raise TypeError(
                f"Adaptive metric: committed value of '{var_name}' is not numeric "
                f"({type(state_after[var_name]).__name__})"
            )
        if after.shape != before.shape:
            raise ValueError(
                f"Adaptive metric: shape of '{var_name}' changed in the fast group "
                f"from {tuple(before.shape)} to {tuple(after.shape)}; its change "
                "cannot be measured"
            )
        before = before.to(device=after.device)
        change = after - before
        tol = atol + rtol * before.abs()
        sum_sq += float(((change / tol) ** 2).mean().item())
        count += 1

    if count == 0:
        return 0.0
    return float((sum_sq / float(count)) ** 0.5)


def _prepare_step_state(simulator: Any, simulation_state_enum: Any) -> None:
    """Validate and transition simulator state for a single step."""
    if simulator.state == simulation_state_enum.UNINITIALIZED:
        if simulator.allow_uninitialized_step:
            simulator.initialize_state({})
        else:
            raise RuntimeError(f"Cannot step simulation in state: {simulator.state}")

    # Allow stepping from INITIALIZED by promoting to RUNNING automatically.
    if simulator.state != simulation_state_enum.RUNNING:
        if simulator.state == simulation_state_enum.INITIALIZED:
            logger.debug(
                "Auto-promoting simulator state from INITIALIZED to RUNNING "
                "for step execution"
            )
            simulator.state = simulation_state_enum.RUNNING
        else:
            raise RuntimeError(f"Cannot step simulation in state: {simulator.state}")


def _sync_state_reset_if_needed(simulator: Any) -> None:
    """Reset simulator when state manager indicates an external state reset."""
    if simulator.state_manager is None:
        return
    reset_counter = getattr(simulator.state_manager, "_state_reset_counter", None)
    if reset_counter is not None and reset_counter != simulator._last_state_reset_counter:
        simulator.reset()
        simulator._last_state_reset_counter = reset_counter


def _initialize_step_timing_events(simulator: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """Create CUDA timing events when GPU timing is available."""
    use_cuda_timing = (
        bool(getattr(simulator, "enable_profiling", False))
        and getattr(simulator, "device", torch.device("cpu")).type == "cuda"
        and device_policy.is_cuda_available()
    )
    if not use_cuda_timing:
        return None, None

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    return start_event, end_event


def _accumulate_updates(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Fold one group result into the step's returned mapping of payloads.

    The single merge rule (execution_group.merge_sequential_updates) applied to
    payloads through its shared decision ``replaces_earlier_output``: a plain
    diagnostic (the last observation) and a REPLACE item replace the earlier
    value; an ACCUMULATE item or a plain state payload adds to it. The group
    results of one step account for their own intervals ending at the step end
    and are concurrent contributions: a per-second rate adds with the coverage
    1. Non-numeric payloads are not returned.
    """
    _, diagnostics, _ = partition_outputs(updates)
    for var_name, update in updates.items():
        payload = update.value if isinstance(update, UpdateItem) else update
        if not isinstance(payload, (Tensor, float, int)):
            continue
        if var_name in target and not replaces_earlier_output(var_name, update, diagnostics):
            target[var_name] = target[var_name] + payload
        else:
            target[var_name] = payload


def _step_view(simulator: Any) -> Any:
    """Return the provider view frozen for the active step transaction."""
    view = getattr(simulator, "_active_step_view", None)
    if view is None:
        raise RuntimeError(
            "No committed provider view is frozen for the active step; "
            "group execution requires step_simulation to admit a reader first"
        )
    return view


def splitting_order_report(
    splitting_mode: str,
    fast_order: Tuple[str, ...],
    functors: Mapping[str, Any],
    *,
    epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Effective order of the fast-group composition bound to the sub-integrators.

    References:
        Hairer, Lubich and Wanner (2006), Geometric Numerical Integration,
        Sect. II.4-II.5 and III.3 (order of compositions of methods);
        Strang (1968) SIAM J. Numer. Anal. 5:506-517.
    Mathematical expression:
        q = min(s, min_i p_i) over the fast Functors f_1..f_n (n >= 2), with the
        splitting order s = 1 for "lie" and s = 2 for "strang"; for n = 1 there is
        no splitting error and q = p_1; any UNDECLARED p_i makes q "unverified".
    Parameter meanings:
        p_i is ``FunctorMetadata.sub_flow_accuracy.order`` of f_i (EXACT = inf,
        ORDER_2 = 2, ORDER_1 = 1); s is the order of the composition with exact
        sub-flows at correct stage times; q is the global order of one step.
    Units: dimensionless orders.
    Domain/constraints: q assumes smooth sub-flows and suitable boundary
        treatment, which are not checked; an adjoint pair of first-order
        sub-integrators that would restore symmetry cannot be declared and is
        reported conservatively. The mode is "lie" or "strang", validated when
        the simulator is built (computation.scheduling.splitting_mode).
    Sign convention: not applicable.
    """
    mode = str(splitting_mode).strip().lower()
    names = tuple(fast_order)
    accuracies: Dict[str, str] = {}
    orders: List[float] = []
    undeclared: List[str] = []
    for name in names:
        metadata = getattr(functors[name], "metadata", None)
        accuracy = (
            SubFlowAccuracy(getattr(metadata, "sub_flow_accuracy"))
            if metadata is not None
            else SubFlowAccuracy.UNDECLARED
        )
        accuracies[name] = accuracy.value
        if accuracy.order is None:
            undeclared.append(name)
        else:
            orders.append(float(accuracy.order))

    splitting_limit: Optional[int]
    if len(names) < 2:
        splitting_limit = None
    elif mode == "strang":
        splitting_limit = 2
    else:
        splitting_limit = 1

    effective_order: Any
    if not names:
        effective_order = None
    elif undeclared:
        effective_order = "unverified"
    else:
        bound = min(orders)
        if splitting_limit is not None:
            bound = min(bound, float(splitting_limit))
        effective_order = "exact" if math.isinf(bound) else int(bound)
    return {
        "epoch": epoch,
        "splitting_mode": mode,
        "fast_order": names,
        "sub_flow_accuracy": accuracies,
        "splitting_order_limit": splitting_limit,
        "effective_order": effective_order,
        "undeclared": tuple(undeclared),
    }


# Fast-member classes of the splitting admissibility resolver (see
# classify_fast_member) and the composition of the execution groups, which is
# sequential (Lie type) across groups: fast composition, then medium, then slow.
FAST_MEMBER_CLASSES = ("stateful", "stateless", "unclassified")
_STATELESS_VARIABLE_KINDS = frozenset(
    {VariableKind.FLUX, VariableKind.DIAGNOSTIC, VariableKind.FORCING, VariableKind.PARAMETER}
)
CROSS_GROUP_COMPOSITION = "lie_fast_medium_slow"


def _canonical_variable(name: str, contract_registry: Any) -> str:
    return contract_registry.resolve_alias(name) if contract_registry is not None else name


def _contract_kind(name: str, contract_registry: Any) -> Optional[str]:
    """SCM contract kind of a canonical variable name, or None without a contract."""
    if contract_registry is None:
        return None
    contract = contract_registry.get_contract(name)
    return None if contract is None else str(contract.kind).strip().lower()


def _declared_state_names(functor: Any, field_name: str, contract_registry: Any) -> Tuple[str, ...]:
    """Canonical declared names of one metadata field, reserved diagnostic names removed.

    The reserved names (``diagnostics``, ``diagnostics.*``, ``__*``) are
    separated by the one channel classifier ``partition_outputs``.
    """
    names = tuple(getattr(functor.metadata, field_name, ()) or ())
    state_names, _, _ = partition_outputs(dict.fromkeys(names))
    return tuple(dict.fromkeys(_canonical_variable(name, contract_registry) for name in state_names))


def classify_fast_member(functor: Any, contract_registry: Any) -> Tuple[str, Tuple[str, ...]]:
    """Classify one fast member as holding persistent state or not, from declarations only.

    References:
        Hairer, Lubich and Wanner (2006), Geometric Numerical Integration,
        Sect. II.5 (a splitting step composes the flows of state-advancing
        sub-problems); RELIC SCM variable contracts (``kind``) and
        ``FunctorMetadata.variable_kind``.
    Mathematical expression:
        stateful  <=> variable_kind = state, or some written variable v has
                      SCM kind(v) = state, or v in W(f) and v in R(f)
                      (a recurrence x_{n+1} = Phi(x_n));
        stateless <=> not stateful and (variable_kind in {flux, diagnostic,
                      forcing, parameter} or every v in W(f) has an SCM contract);
        unclassified otherwise.
    Parameter meanings:
        W(f) the declared output variables of f and R(f) its declared inputs,
        both canonical (SCM aliases resolved) and without reserved diagnostic
        names; kind(v) the SCM contract kind of v.
    Units: not applicable (a classification).
    Domain/constraints: declarations are trusted, nothing is inferred from
        names or values; hidden per-call state that is not declared (for
        example a window over calls kept in a Python attribute) is not seen;
        without a ContractRegistry only variable_kind and recurrences classify.
    Sign convention: not applicable.
    """
    metadata = functor.metadata
    variable_kind = VariableKind(metadata.variable_kind)
    written = _declared_state_names(functor, "output_variables", contract_registry)
    read = set(_declared_state_names(functor, "input_variables", contract_registry))
    reasons: List[str] = []
    if variable_kind is VariableKind.STATE:
        reasons.append("variable_kind=state")
    state_contracts = [name for name in written if _contract_kind(name, contract_registry) == "state"]
    if state_contracts:
        reasons.append("writes SCM state variables " + ", ".join(state_contracts))
    recurrent = [name for name in written if name in read]
    if recurrent:
        reasons.append("reads and writes " + ", ".join(recurrent))
    if reasons:
        return "stateful", tuple(reasons)
    if variable_kind in _STATELESS_VARIABLE_KINDS:
        return "stateless", (f"variable_kind={variable_kind.value}",)
    uncontracted = [name for name in written if _contract_kind(name, contract_registry) is None]
    if not uncontracted:
        return "stateless", ("every written variable has a non-state SCM contract",)
    return "unclassified", (
        "variable_kind undeclared and no SCM contract for " + ", ".join(uncontracted),
    )


def resolve_splitting_admissibility(
    splitting_mode: str,
    splitting_policy: str,
    fast_order_mode: str,
    planned_fast_order: Tuple[str, ...],
    functors: Mapping[str, Any],
    contract_registry: Any,
    *,
    time_step: float,
    multirate_exchanges: Tuple[Mapping[str, str], ...] = (),
    substep_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Resolve whether a fast-group composition may be published (pure; nothing is stored).

    References:
        Strang (1968) SIAM J. Numer. Anal. 5(3):506-517; Hairer, Lubich and
        Wanner (2006), Geometric Numerical Integration, Sect. II.4-II.5;
        McLachlan and Quispel (2002) Acta Numerica 11:341-434; Gear and Wells
        (1984) BIT 24(4):484-502 (multirate exchange between groups).
    Mathematical expression:
        S1 (strang, n >= 2): a flux path P -> R_1 -> ... -> C over SCM variables
           of kind flux, written by one member and declared as input by the
           next, with P and C stateful and every R_i stateless, is refused; a
           path with an unclassified endpoint is flagged (S3 with
           require_declared refuses undeclared unclassified members). The
           palindrome runs every member but the last twice, so
           the provider releases (q1 + q2) h / 2 while the consumer integrates
           the value it read; the residual (q1 - q2) h / 2 per step does not
           vanish when the provider rate changes inside the step (design D3).
           Couplings through shared state variables are not flux paths.
        S3 (strang, n >= 2, policy require_declared): refused while a member
           that is not stateless declares sub_flow_accuracy undeclared.
        S5 (fast_order reverse): refused while a fast member declares a
           required or optional (same-step) dependency on another fast
           member; otherwise the published order is the planned order
           reversed (the adjoint Lie composition).
        Effective sub-step of member i: h_i = (h / 2) / n_i for the members
        that run two half-steps under strang (all but the last, n >= 2) and
        h_i = h / n_i otherwise, with n_i the sub-step count of one call over
        that interval (execution_group.resolve_call_substep_count).
    Parameter meanings:
        n the number of fast members; h = time_step, the configured fast step;
        n_i the sub-step count of member i: its substep_overrides entry
        ({count: n_i}, or {max_seconds: s} giving n_i = ceil(dt_call / s)),
        else its declared FunctorMetadata.substep_count; q1, q2 the
        provider rates of its two half-steps; planned_fast_order the dependency
        order computed for the publication.
    Units: h, h_i [s]; q in the flux units of the contract.
    Domain/constraints: the modes are already validated by the simulator
        (lie|strang, warn|require_declared, forward|reverse); the effective
        sub-steps refer to the configured time_step (an adaptive retry or an
        explicit step(dt) scales them); the groups compose sequentially
        (lie_fast_medium_slow), and a provider in a slower group is held, not
        split. Without a ContractRegistry S1 cannot be evaluated, so a strang
        composition with two stateful members is refused.
    Sign convention: not applicable.
    """
    mode = str(splitting_mode)
    planned = tuple(planned_fast_order)
    classes: Dict[str, str] = {}
    class_basis: Dict[str, Tuple[str, ...]] = {}
    for name in planned:
        classes[name], class_basis[name] = classify_fast_member(functors[name], contract_registry)
    split = mode == "strang" and len(planned) >= 2
    refusals: List[str] = []
    edges: List[Dict[str, Any]] = []

    # S5: the reverse order needs a fast group without same-step edges.
    published = planned
    if fast_order_mode == "reverse":
        members = set(planned)
        same_step: List[Tuple[str, str]] = []
        for consumer in planned:
            declared = normalize_dependency_declaration(
                functors[consumer].get_dependencies(), owner=consumer
            )
            same_step.extend(
                (consumer, provider)
                for provider in declared.same_step
                if provider in members and provider != consumer
            )
        for consumer, provider in same_step:
            edges.append({
                "rule": "S5", "kind": "dependency", "status": "refused",
                "provider": provider, "consumer": consumer, "relays": (), "variables": (),
                "reason": "fast_order reverse would run the consumer before its same-step provider",
            })
        if same_step:
            refusals.append(
                "S5: fast_order reverse requires a fast group without same-step dependency "
                "edges; found " + ", ".join(f"{consumer} -> {provider}" for consumer, provider in same_step)
            )
        else:
            published = tuple(reversed(planned))

    # S1: flux paths between fast members that are not stateless.
    if contract_registry is None:
        stateful = [name for name in planned if classes[name] == "stateful"]
        if split and len(stateful) >= 2:
            refusals.append(
                "S1: the strang composition cannot be checked for flux transfers between "
                f"stateful fast members {stateful} without the SCM ContractRegistry"
            )
    else:
        writes = {
            name: {
                variable
                for variable in _declared_state_names(functors[name], "output_variables", contract_registry)
                if _contract_kind(variable, contract_registry) == "flux"
            }
            for name in planned
        }
        reads = {
            name: set(_declared_state_names(functors[name], "input_variables", contract_registry))
            for name in planned
        }
        for provider in planned:
            if classes[provider] == "stateless":
                continue
            found: Dict[str, Dict[str, Any]] = {}
            frontier: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = [(provider, (), ())]
            visited = {provider}
            while frontier:
                node, relays, variables = frontier.pop(0)
                for consumer in planned:
                    if consumer == node or consumer == provider or consumer in relays:
                        continue
                    shared = tuple(sorted(writes[node] & reads[consumer]))
                    if not shared:
                        continue
                    path_variables = tuple(dict.fromkeys(variables + shared))
                    if classes[consumer] == "stateless":
                        if consumer not in visited:
                            visited.add(consumer)
                            frontier.append((consumer, relays + (consumer,), path_variables))
                    elif consumer not in found:
                        refused = (
                            split and classes[provider] == "stateful" and classes[consumer] == "stateful"
                        )
                        if refused:
                            reason = (
                                "the palindrome releases (q1 + q2) h / 2 from the provider while the "
                                "consumer integrates the rate it read"
                            )
                        elif not split:
                            reason = "admissible under lie: each member runs once over [t, t + h]"
                        else:
                            reason = (
                                "admissible: an endpoint is not classified as holding persistent "
                                "state (declare variable_kind or SCM contracts)"
                            )
                        found[consumer] = {
                            "rule": "S1", "kind": "flux_transfer",
                            "status": "refused" if refused else "flagged",
                            "provider": provider, "consumer": consumer,
                            "relays": relays, "variables": path_variables,
                            "reason": reason,
                        }
            for consumer in planned:
                if consumer in found:
                    edges.append(found[consumer])
        refused_paths = [edge for edge in edges if edge["rule"] == "S1" and edge["status"] == "refused"]
        if refused_paths:
            refusals.append(
                "S1: strang splitting refuses SCM flux transfers between fast members that hold "
                "persistent state: " + "; ".join(
                    f"{edge['provider']} -> "
                    + "".join(f"{relay} -> " for relay in edge["relays"])
                    + f"{edge['consumer']} via {', '.join(edge['variables'])}"
                    for edge in refused_paths
                )
                + " (run them under lie, or place the provider in a slower group)"
            )

    # S3: require_declared needs declared sub-flows of every member not shown stateless.
    if split and splitting_policy == "require_declared":
        undeclared = [
            name for name in planned
            if classes[name] != "stateless"
            and SubFlowAccuracy(functors[name].metadata.sub_flow_accuracy)
            is SubFlowAccuracy.UNDECLARED
        ]
        if undeclared:
            refusals.append(
                "S3: splitting_policy require_declared refuses the strang composition while fast "
                f"members {undeclared} hold persistent state (or are not shown to be without it) "
                "and declare no sub_flow_accuracy"
            )

    half = float(time_step) * 0.5
    substep_seconds: Dict[str, float] = {}
    for index, name in enumerate(published):
        call_seconds = half if split and index < len(published) - 1 else float(time_step)
        count = resolve_call_substep_count(
            name, functors[name], (substep_overrides or {}).get(name), call_seconds
        )
        substep_seconds[name] = call_seconds / count
    return {
        "splitting_policy": str(splitting_policy),
        "fast_order_mode": str(fast_order_mode),
        "planned_fast_order": planned,
        "published_fast_order": published,
        "member_class": dict(classes),
        "member_class_basis": dict(class_basis),
        "refused_or_flagged_edges": tuple(edges),
        "refusals": tuple(refusals),
        "admissible": not refusals,
        "contract_registry": "unavailable" if contract_registry is None else "available",
        "time_step_seconds": float(time_step),
        "effective_substep_seconds": substep_seconds,
        "exchanges": tuple(dict(item) for item in multirate_exchanges),
        "cross_group_composition": CROSS_GROUP_COMPOSITION,
    }


def _splitting_contract_registry(
    sim: Any,
    fast_order: Tuple[str, ...],
    exchanges: Tuple[Mapping[str, str], ...] = (),
) -> Any:
    """ContractRegistry governing the simulator's writes (loaded once per process).

    The StateManager's registry when a StateManager exists (None after an
    explicit opt-out), else the packaged registry the StateManager would
    load; nothing is loaded for an empty fast group without exchanges.
    """
    if not fast_order and not exchanges:
        return None
    if sim.state_manager is not None:
        return getattr(sim.state_manager, "_contract_registry", None)
    from ...contracts.contract_loader import load_contract_registry

    return load_contract_registry()


def splitting_admissibility_report(sim: Any, view: Any) -> Dict[str, Any]:
    """Admissibility fields of a committed provider view (recomputed; nothing is stored)."""
    published = tuple(view.order_fast)
    # A published reverse order has no same-step edges, so reversing it again
    # gives back the planned dependency order.
    planned = tuple(reversed(published)) if sim._fast_order_mode == "reverse" else published
    resolution = resolve_splitting_admissibility(
        sim._splitting_mode,
        sim._splitting_policy,
        sim._fast_order_mode,
        planned,
        {name: view.providers[name].functor for name in published},
        _splitting_contract_registry(sim, published),
        time_step=float(sim.time_step),
        multirate_exchanges=sim._multirate_exchanges,
        substep_overrides=sim._substep_overrides,
    )
    report = {
        key: resolution[key]
        for key in (
            "splitting_policy", "fast_order_mode", "member_class", "member_class_basis",
            "refused_or_flagged_edges", "contract_registry", "time_step_seconds",
            "effective_substep_seconds", "exchanges", "cross_group_composition",
        )
    }
    # The configured sub-step overrides of every group, and those that name no
    # registered provider (reported, never silently dropped).
    report["substep_overrides"] = {name: dict(entry) for name, entry in sim._substep_overrides.items()}
    report["unmatched_substep_overrides"] = tuple(
        sorted(name for name in sim._substep_overrides if name not in view.providers)
    )
    # The declared multirate exchanges resolved against the committed groups.
    report["exchanges"] = resolve_multirate_exchanges(
        sim._multirate_exchanges,
        {
            "fast": published,
            "medium": tuple(view.order_medium),
            "slow": tuple(view.order_slow),
        },
        {name: provider.functor for name, provider in view.providers.items()},
        _splitting_contract_registry(sim, published, sim._multirate_exchanges),
    )["exchanges"]
    return report


def _govern_splitting_admissibility(sim: Any) -> None:
    """Resolve the splitting admissibility of the plan about to be published.

    Called once at the single publication point (registration owner). A
    refusal raises RuntimeError before the view is built, so the registration,
    retirement, or restore rolls back and the committed view is unchanged. An
    admissible plan installs its published fast order (reversed for
    fast_order reverse). Under strang the effective-order warning is logged
    only when the effective order differs from the previously committed view's
    (no stored state), so registering many Functors with an unchanged verdict
    logs it once.
    """
    if sim._fast_order_mode == "reverse":
        # The cached fast order is a previously published (reversed) order when
        # the groups were not dirty; recompute the dependency order to reverse.
        sim._groups_dirty = True
        sim._compute_group_execution_orders()
    planned = tuple(sim._order_fast)
    resolution = resolve_splitting_admissibility(
        sim._splitting_mode,
        sim._splitting_policy,
        sim._fast_order_mode,
        planned,
        sim.functors,
        _splitting_contract_registry(sim, planned),
        time_step=float(sim.time_step),
        multirate_exchanges=sim._multirate_exchanges,
        substep_overrides=sim._substep_overrides,
    )
    if resolution["refusals"]:
        raise RuntimeError(
            "Splitting admissibility refused the provider-view publication: "
            + " | ".join(resolution["refusals"])
        )
    # Multirate exchanges of the plan about to be published (design section 3):
    # a refusal raises before any mutation; an admitted plan makes sure the bound
    # StateManager declares the interval means (idempotent; the binding of a
    # StateManager declares them first, see plan_exchange_means).
    exchange_resolution = resolve_multirate_exchanges(
        sim._multirate_exchanges,
        {
            "fast": tuple(planned),
            "medium": tuple(sim._order_medium),
            "slow": tuple(sim._order_slow),
        },
        sim.functors,
        _splitting_contract_registry(sim, planned, sim._multirate_exchanges),
    )
    if exchange_resolution["refusals"]:
        raise RuntimeError(
            "Multirate exchange admissibility refused the provider-view publication: "
            + " | ".join(exchange_resolution["refusals"])
        )
    register_exchange_means(sim.state_manager, plan_exchange_means(sim, sim.state_manager))
    sim._order_fast = list(resolution["published_fast_order"])
    for edge in resolution["refused_or_flagged_edges"]:
        logger.debug("Splitting admissibility flagged %s", edge)
    splitting_mode = sim._splitting_mode
    report = splitting_order_report(splitting_mode, tuple(sim._order_fast), sim.functors)
    if report["splitting_mode"] != "strang" or not report["fast_order"]:
        return
    previous_view = getattr(getattr(sim, "_provider_lifecycle", None), "committed_view", None)
    previous_order: Any = None
    if previous_view is not None:
        previous_order = splitting_order_report(
            splitting_mode,
            previous_view.order_fast,
            {name: previous_view.providers[name].functor for name in previous_view.order_fast},
        )["effective_order"]
    effective_order = report["effective_order"]
    if effective_order == previous_order:
        return
    if effective_order == "unverified":
        logger.warning(
            "Strang splitting is configured but its effective order is unverified: "
            "fast Functors %s declare no sub_flow_accuracy, so the second-order claim "
            "of the palindromic composition is not established.",
            list(report["undeclared"]),
        )
    elif isinstance(effective_order, int) and effective_order < 2:
        limiting = [
            name
            for name, accuracy in report["sub_flow_accuracy"].items()
            if accuracy == SubFlowAccuracy.ORDER_1.value
        ]
        logger.warning(
            "Strang splitting is configured but the fast group has effective order %s: "
            "first-order sub-flows %s bound the composed step.",
            effective_order,
            limiting,
        )


# Multirate exchanges between the fast group and a slower consumer group
# (computation.scheduling.multirate_exchanges; design 20261019_cr_a04_design_01,
# section 3). An interval_mean exchange keeps its running interval mean in the
# StateManager variable EXCHANGE_MEAN_PREFIX + id, written only by the owner
# identity EXCHANGE_OWNER_PREFIX + id (precedent: topology:<transition>), so the
# step transaction, checkpoints and restarts carry it like any other state. A
# hold exchange keeps nothing: the consumer reads the committed value.
EXCHANGE_MEAN_PREFIX = "exchange_mean__"
EXCHANGE_OWNER_PREFIX = "exchange:"
_CONSUMER_LAST_EXECUTION = {
    "medium": "_last_medium_execution_time",
    "slow": "_last_slow_execution_time",
}


def exchange_mean_variable(exchange: Mapping[str, str]) -> str:
    """StateManager name of the running interval mean of one exchange."""
    return f"{EXCHANGE_MEAN_PREFIX}{exchange['id']}"


def exchange_owner(exchange: Mapping[str, str]) -> str:
    """Owner identity (update source) of the interval mean of one exchange."""
    return f"{EXCHANGE_OWNER_PREFIX}{exchange['id']}"


def _interval_mean_exchanges(sim: Any) -> Tuple[Mapping[str, str], ...]:
    return tuple(
        exchange for exchange in getattr(sim, "_multirate_exchanges", ())
        if exchange["mode"] == "interval_mean"
    )


def _declared_targets(functor: Any, field_name: str, contract_registry: Any) -> Tuple[str, ...]:
    """Canonical declared names of one metadata field with the delta_ prefix removed."""
    return tuple(dict.fromkeys(
        _canonical_variable(StateManager._update_target_name(name), contract_registry)
        for name in _declared_state_names(functor, field_name, contract_registry)
    ))


def resolve_multirate_exchanges(
    exchanges: Tuple[Mapping[str, str], ...],
    group_orders: Mapping[str, Tuple[str, ...]],
    functors: Mapping[str, Any],
    contract_registry: Any,
) -> Dict[str, Any]:
    """Resolve whether the declared multirate exchanges may be published (pure; nothing is stored).

    References:
        Gear and Wells (1984) BIT 24(4):484-502 (multirate methods); design
        20261019_cr_a04_design_01, section 3 (exchange interval_mean).
    Mathematical expression:
        For an exchange e = (id, v, g, mode) with consumer group g:
        R_e = {f in g : v in R(f)} (readers), W_e = {f in g : v in W(f)}
        (consumer-group writers), P_e = {f in fast : v in W(f)} (fast writers),
        M_e = {f : m_e in W(f)} (writers of the mean variable m_e).
        refused  <=> v is an SCM alias, or M_e is not empty, or
                     (mode = interval_mean and W_e is not empty);
        flagged  <=> not refused and (R_e, P_e empty, or W_e not empty);
        admitted otherwise.
    Parameter meanings:
        W(f), R(f) the declared outputs (delta_ prefix removed) and inputs of
        Functor f, canonical; m_e = exchange_mean_variable(e); the group orders
        are those of the plan about to be published (or of the committed view).
    Units: not applicable (a classification).
    Domain/constraints: the exchanges were validated by the simulator (keys,
        consumer groups slower than fast, modes, unique ids and routes). The
        overlay of an interval_mean exchange replaces the value its consumer
        group reads, so a consumer group that writes v (its own value) refuses
        it; the mean variable belongs to the exchange owner alone; the overlay
        key is the committed state name of v, so v must be canonical.
    Sign convention: not applicable.
    """
    records: List[Dict[str, Any]] = []
    refusals: List[str] = []
    written = {
        name: set(_declared_targets(functor, "output_variables", contract_registry))
        for name, functor in functors.items()
    }
    read = {
        name: set(_declared_targets(functor, "input_variables", contract_registry))
        for name, functor in functors.items()
    }
    for exchange in exchanges:
        exchange_id = exchange["id"]
        variable = exchange["variable"]
        group = exchange["consumer_group"]
        mode = exchange["mode"]
        mean_variable = exchange_mean_variable(exchange) if mode == "interval_mean" else None
        canonical = _canonical_variable(variable, contract_registry)
        consumers = tuple(name for name in group_orders.get(group, ()) if name in functors)
        readers = tuple(name for name in consumers if canonical in read[name])
        consumer_writers = tuple(name for name in consumers if canonical in written[name])
        fast_writers = tuple(
            name for name in group_orders.get("fast", ()) if name in functors and canonical in written[name]
        )
        mean_writers = tuple(
            name for name in functors if mean_variable is not None and mean_variable in written[name]
        )
        refused: List[str] = []
        flagged: List[str] = []
        if canonical != variable:
            refused.append(f"'{variable}' is an SCM alias of '{canonical}'; declare the canonical name")
        if mean_writers:
            refused.append(
                f"its mean variable '{mean_variable}' is written by Functor(s) {list(mean_writers)}; "
                "only the exchange owner writes it"
            )
        if mode == "interval_mean" and consumer_writers:
            refused.append(
                f"'{variable}' is written by its consumer group {group} ({list(consumer_writers)}); "
                "the interval-mean overlay would replace that group's own value"
            )
        if mode == "hold" and consumer_writers:
            flagged.append(f"'{variable}' is written by its consumer group {group} ({list(consumer_writers)})")
        if not readers:
            flagged.append(f"no member of the {group} group declares '{variable}' as an input")
        if not fast_writers:
            flagged.append(f"no fast member writes '{variable}' (the mean is that of its committed value)")
        status = "refused" if refused else ("flagged" if flagged else "admitted")
        records.append({
            "id": exchange_id,
            "variable": variable,
            "consumer_group": group,
            "mode": mode,
            "mean_variable": mean_variable,
            "owner": exchange_owner(exchange) if mean_variable is not None else None,
            "readers": readers,
            "consumer_writers": consumer_writers,
            "fast_writers": fast_writers,
            "status": status,
            "reasons": tuple(refused + flagged),
        })
        if refused:
            refusals.append(f"exchange '{exchange_id}': " + "; ".join(refused))
    return {"exchanges": tuple(records), "refusals": tuple(refusals)}


def _exchange_variable_units(sim: Any, state_manager: Any, exchange: Mapping[str, str]) -> str:
    """Declared units of an exchange variable: StateManager schema, SCM contract, state_units."""
    variable = exchange["variable"]
    spec = state_manager.schema.get(variable)
    if spec is not None and getattr(spec, "units", None):
        return str(spec.units)
    registry = getattr(state_manager, "_contract_registry", None)
    contract = registry.get_contract(variable) if registry is not None else None
    if contract is not None and getattr(contract, "udunits", None):
        return str(contract.udunits)
    state_units = sim._get_setting("state_units")
    if isinstance(state_units, Mapping) and state_units.get(variable):
        return str(state_units[variable])
    raise ValueError(
        f"Multirate exchange '{exchange['id']}': variable '{variable}' has no declared units "
        "(StateManager schema, SCM contract, or the simulator setting 'state_units'), so its "
        "interval mean cannot be declared"
    )


def plan_exchange_means(sim: Any, state_manager: Any) -> List[StateVariableSpec]:
    """Schema entries a StateManager still needs for the declared interval means (pure).

    References: design 20261019_cr_a04_design_01, sections 2 and 3 (exchange
        means are StateManager variables of their owner); StateManager
        register_variable (the schema owner's registration API).
    Mathematical expression: one entry m_e per interval_mean exchange e that is
        missing from the schema or the state; its initial value is the
        placeholder 0 (float64), which the first exchange commit of a block
        replaces (tau = 0 gives m_e = vbar without reading the old value).
    Parameter meanings: m_e = exchange_mean_variable(e); tau = t - t_last(g)
        the elapsed time of the consumer interval of group g at time t.
    Units: m_e in the declared units of its variable (_exchange_variable_units).
    Domain/constraints: without a StateManager there is nothing to declare; a
        mean missing from the state while its consumer interval is already
        running (tau > 0) cannot be started there and raises RuntimeError (a
        restored run whose state lacks it is refused); undeclared variable units
        raise ValueError. The entries are not conserved, unbounded, and not
        required by StateManager.initialize.
    Sign convention: the mean keeps the sign convention of its variable.
    """
    specs: List[StateVariableSpec] = []
    if state_manager is None:
        return specs
    schema = state_manager.schema
    state_names = state_manager.state_vars
    for exchange in _interval_mean_exchanges(sim):
        name = exchange_mean_variable(exchange)
        in_schema = name in schema
        in_state = name in state_names
        if in_schema and in_state:
            continue
        group = exchange["consumer_group"]
        elapsed = float(sim.current_time) - float(getattr(sim, _CONSUMER_LAST_EXECUTION[group]))
        if not in_state and elapsed > 0.0:
            raise RuntimeError(
                f"Multirate exchange '{exchange['id']}': the {group} consumer interval has run for "
                f"{elapsed} s, but its interval mean '{name}' is not part of the state; a running mean "
                "cannot start in the middle of its interval"
            )
        if in_schema:
            specs.append(schema[name])
            continue
        specs.append(StateVariableSpec(
            name=name,
            units=_exchange_variable_units(sim, state_manager, exchange),
            bounds=None,
            conserved=False,
            spatial_dims=0,
            required=False,
            default=torch.zeros((), dtype=torch.float64),
            description=(
                f"Interval mean of '{exchange['variable']}' over the running {group} consumer "
                f"interval (multirate exchange '{exchange['id']}', owner {exchange_owner(exchange)})"
            ),
        ))
    return specs


def register_exchange_means(state_manager: Any, specs: List[StateVariableSpec]) -> None:
    """Register planned interval-mean entries through the StateManager schema owner."""
    for spec in specs:
        state_manager.register_variable(spec)


def _commit_multirate_exchanges(simulator: Any, dt: float) -> None:
    """Advance the running mean of every interval_mean exchange over the accepted fast step.

    References:
        Gear and Wells (1984) BIT 24(4):484-502 (multirate methods); design
        20261019_cr_a04_design_01, section 3 (exchange interval_mean); Welford
        (1962) Technometrics 4(3):419-420 (updating form of a running mean).
    Mathematical expression:
        m_e <- (m_e tau + vbar h) / (tau + h)
             = m_e + ((vbar - m_e) h) / (tau + h)     for tau > 0,
        m_e <- vbar                                    for tau = 0,
        so after the steps k of one consumer interval m_e = sum_k vbar_k h_k / T,
        the exact interval mean of the piecewise-constant reconstruction.
    Parameter meanings:
        vbar the committed value of the exchange variable v after the accepted
        fast composition (after its boundary-rate and S2 finalization); h = dt
        the accepted fast step; t_k the step start; tau = t_k - t_last(g) with
        t_last(g) the last execution time of the consumer group g; T = sum_k h_k.
    Units: m_e and vbar in the units of v; h, tau, T [s].
    Domain/constraints: one owner commit per exchange (REPLACE, source
        exchange:<id>, no conservation check: a mean is not a budget term),
        after the adaptive acceptance and before the consumer groups run, inside
        the step transaction (a failed step restores m_e); computed in
        promote(dtype(v), float64) and stored in the float64 entry of the
        schema; v must be a floating-point committed state variable, and a mean
        whose shape differs from v while tau > 0 raises (it cannot cover the
        elapsed part of the interval).
        Commit context (D7): step, composition_start t_k and
        composition_interval h of the accepted fast step, stage the index of
        the exchange among the interval_mean exchanges, role exchange,
        interval_start t_last(g) (the start of the consumer interval the mean
        covers) and dt h (the part this commit adds), so the mean covers
        [interval_start, composition_start + dt].
    Sign convention: m_e keeps the sign convention of v.
    """
    exchanges = _interval_mean_exchanges(simulator)
    if not exchanges:
        return
    state_manager = getattr(simulator, "state_manager", None)
    if state_manager is None:
        raise RuntimeError("Multirate interval_mean exchanges require a StateManager")
    step_start = float(simulator.current_time)
    h = float(dt)
    step_counter = getattr(simulator, "step_count", None)
    step_number = step_counter if type(step_counter) is int else None
    for exchange_index, exchange in enumerate(exchanges):
        name = exchange_mean_variable(exchange)
        owner = exchange_owner(exchange)
        variable = exchange["variable"]
        spec = state_manager.schema.get(name)
        if spec is None:
            raise RuntimeError(
                f"Multirate exchange '{exchange['id']}': its interval mean '{name}' is not declared "
                "in the StateManager schema"
            )
        value = state_manager.get_state(variable, clone=False)
        if not isinstance(value, Tensor):
            raise RuntimeError(
                f"Multirate exchange '{exchange['id']}': variable '{variable}' is not part of the "
                "committed state"
            )
        if not value.dtype.is_floating_point:
            raise TypeError(
                f"Multirate exchange '{exchange['id']}': variable '{variable}' has dtype {value.dtype}; "
                "an interval mean needs a floating-point value"
            )
        work_dtype = torch.promote_types(value.dtype, torch.float64)
        current = value.to(dtype=work_dtype)
        group = exchange["consumer_group"]
        interval_start = float(getattr(simulator, _CONSUMER_LAST_EXECUTION[group]))
        tau = step_start - interval_start
        if tau < 0.0:
            raise RuntimeError(
                f"Multirate exchange '{exchange['id']}': the {group} group last executed after the "
                f"step start (tau = {tau} s); the last-execution bookkeeping is inconsistent"
            )
        if tau == 0.0:
            mean = current.clone()
        else:
            previous = state_manager.get_state(name, clone=False)
            if not isinstance(previous, Tensor) or previous.shape != current.shape:
                raise RuntimeError(
                    f"Multirate exchange '{exchange['id']}': the running mean '{name}' does not cover "
                    f"the elapsed {tau} s of its {group} interval for '{variable}' (shape "
                    f"{None if previous is None else tuple(previous.shape)} vs {tuple(current.shape)})"
                )
            previous = previous.to(device=current.device, dtype=work_dtype)
            mean = previous + ((current - previous) * h) / (tau + h)
        state_manager.apply_updates(
            {
                name: UpdateItem(
                    value=mean,
                    mode=UpdateMode.REPLACE,
                    write_mode=StateWriteMode.MUTABLE,
                    unit=spec.units,
                    source_functor=owner,
                )
            },
            None,
            check_conservation=False,
            rollback_on_failure=True,
            update_sources={name: owner},
            commit_context=build_commit_context(
                step=step_number,
                composition_start=step_start,
                composition_interval=h,
                stage=exchange_index,
                role=EXCHANGE_ROLE,
                interval_start=interval_start,
                dt=h,
            ),
        )


def _exchange_overlay(simulator: Any, group_name: str) -> Dict[str, Tensor]:
    """Input overlay of one consumer group call: v -> m_e for its interval_mean exchanges.

    References: design 20261019_cr_a04_design_01, section 3 (overlay v := m_e for
        the consumer group call only, before declared input projection).
    Mathematical expression: O[v] = m_e cast to the dtype and device of the
        committed v, for every interval_mean exchange with consumer group g.
    Parameter meanings: g = group_name; m_e the mean committed this step by
        _commit_multirate_exchanges.
    Units: those of v.
    Domain/constraints: read by execution_group._prepare_group_state while
        simulator._active_exchange_overlay holds it (the consumer group call
        only); nothing is written.
    Sign convention: as v.
    """
    overlay: Dict[str, Tensor] = {}
    for exchange in _interval_mean_exchanges(simulator):
        if exchange["consumer_group"] != group_name:
            continue
        state_manager = simulator.state_manager
        mean = state_manager.get_state(exchange_mean_variable(exchange), clone=False)
        value = state_manager.get_state(exchange["variable"], clone=False)
        if not isinstance(mean, Tensor) or not isinstance(value, Tensor):
            raise RuntimeError(
                f"Multirate exchange '{exchange['id']}': the mean or the variable is missing from "
                f"the committed state before the {group_name} group call"
            )
        overlay[exchange["variable"]] = mean.to(device=value.device, dtype=value.dtype)
    return overlay


def _execute_fast_group(simulator: Any, step_dt: float) -> Dict[str, Any]:
    """Execute fast group using configured operator-splitting mode."""
    # References: Strang (1968) SIAM J. Numer. Anal. 5:506-517; Hairer, Lubich and
    #   Wanner (2006) Geometric Numerical Integration, Chapter II.5 (symmetric
    #   composition of n flows); McLachlan and Quispel (2002) Acta Numerica.
    # Mathematical expression: for the fast order (f_1, ..., f_n) with n >= 2 the
    #   "strang" mode evaluates the palindromic composition
    #   Phi_h = Phi_{f_1}^{h/2} o ... o Phi_{f_{n-1}}^{h/2} o Phi_{f_n}^{h}
    #           o Phi_{f_{n-1}}^{h/2} o ... o Phi_{f_1}^{h/2}
    #   (2n-1 operator evaluations); "lie" mode evaluates Phi_{f_1}^{h} o ... o Phi_{f_n}^{h}.
    #   Stage times: each call receives the real interval it accounts for, so
    #   Phi_{f_i}^{h/2} (first half) covers [t, t + h/2], Phi_{f_n}^{h} covers
    #   [t, t + h], Phi_{f_i}^{h/2} (second half) covers [t + h/2, t + h]; in "lie"
    #   mode every Phi_{f_i}^{h} covers [t, t + h].
    # Parameter meanings: f_i are the fast-group Functors in committed order; h = step_dt;
    #   t = simulator.current_time, the start of the step.
    # Units: t, h [s]; the accumulated updates keep each Functor's declared units.
    # Domain/constraints: when every Phi_{f_i} is the exact sub-flow the palindromic
    #   composition is symmetric and second order for smooth sub-flows for any n
    #   (Hairer et al. 2006, Theorem II.3.2); for non-autonomous sub-flows this needs
    #   the stage times above (a second half-step started at t instead of t + h/2
    #   leaves an O(h^2) local error, i.e. first order). The call order alone does
    #   not make a composition of numerical sub-integrators symmetric: the effective
    #   order is min(2, p_i) over the declared orders p_i of the sub-integrators
    #   (FunctorMetadata.sub_flow_accuracy; see splitting_order_report); with
    #   first-order sub-integrators such as explicit Euler the step stays first order.
    #   Each sub-execution is its own group commit. n < 2 or "lie" runs the plain
    #   sequential order. The results of the sub-executions merge by the single
    #   rule of execution_group.merge_sequential_updates (plain diagnostics: last
    #   observation; UpdateItems by their mode; amounts and plain state payloads:
    #   sum; a per-second rate of a sub-execution over h_k adds with the coverage
    #   h_k / h, i.e. 1/2 for a half-step and 1 for the middle, so the returned rate
    #   is the rate over the step); _compute_error_norm measures the committed state
    #   change of the written variables, not these payloads. Boundary-rate
    #   state during a composition is the sum of writers' latest reduced stage
    #   rates, not its partial full-step mean. Only successful completion
    #   publishes the full-step mean, before the adaptive metric is evaluated.
    # Sign convention: sub-flows are applied left to right in time order.
    #
    # Dependency outputs of the composition (the subprocess_outputs channel):
    # References: Strang (1968); Hairer, Lubich and Wanner (2006), Sect. II.5 (each
    #   sub-flow of a composition acts on what the preceding sub-flows produced);
    #   RELIC dependency contract (FunctorBase.get_dependencies,
    #   normalize_dependency_declaration).
    # Mathematical expression: the sub-executions e_1, ..., e_m of one composition
    #   (m = 2n - 1 in "strang" mode with n >= 2; one group call over the n Functors
    #   otherwise) run in time order. D_0 = {}; after e_k = (f, [a_k, b_k]),
    #   D_k = D_{k-1} with D_k[f] = U_k, the merged substep updates e_k returned. A
    #   consumer executed at e_k receives subprocess_outputs[p] = D_{k-1}[p] for each
    #   declared provider p; a required p missing from D_{k-1} raises. With the
    #   topological fast order (providers before consumers) every Strang
    #   sub-execution of a consumer, including its second half-step over
    #   [t + h/2, t + h], therefore sees each provider's first half-step outputs over
    #   [t, t + h/2]; in "lie" mode it sees the provider's outputs over [t, t + h].
    # Parameter meanings: D is the composition's dependency-output map; U_k the
    #   updates of one sub-execution; f the Functor of e_k; p a provider name;
    #   h = step_dt; t the step start.
    # Units: an entry keeps the provider's declared units and is accumulated over the
    #   interval [a_k, b_k] of its own sub-execution (h/2 or h), not over the
    #   consumer's interval.
    # Domain/constraints: D is created empty for each composition, i.e. for each
    #   fast-group attempt of a step (a rejected adaptive attempt's map is discarded
    #   with the attempt and the retry starts empty), and is dropped when the
    #   composition returns; a failed step restores the step runtime, so no map
    #   outlives its step. Medium and slow group calls are compositions of their own;
    #   a provider in another execution group is never in D (its values are read
    #   from the committed state with the declared cross-group lag).
    # Sign convention: entries keep the sign conventions of the provider's updates.
    #
    # Stored diagnostics of the composition: the same GroupComposition keeps, per
    #   Functor f, M_k[f] = merge_sequential_updates(M_{k-1}[f], G_k), where G_k are
    #   the diagnostics f returned at its sub-execution e_k (modes kept); after e_k
    #   StateManager stores M_k[f] for f (ingest_module_diagnostics replaces f's
    #   entries). The stored value after the composition therefore equals the value
    #   step() returns: plain diagnostics and REPLACE items the last value,
    #   amounts the sum over f's sub-executions, per-second rates the sum of the
    #   sub-execution rates weighted by h_k / h (the rate over the step). There is
    #   no second merge rule; a rejected attempt's stored diagnostics are restored
    #   with the StateManager checkpoint of the attempt, and its M is discarded
    #   with it. Exception for committed state (S2, CR-A-04 step 5): a REPLACE
    #   per-second rate on an SCM flux contract whose writer ran in two Strang
    #   half-steps is not kept at its last value; the finalization commit stores,
    #   and the completing call returns, its step mean sum_k (h_k / h) q_k
    #   (GroupComposition.finalize_boundary_rates); DERIVED rates, diagnostics,
    #   Lie calls and sub-steps keep the last value.
    splitting_mode = str(getattr(simulator, "_splitting_mode", "lie")).strip().lower()
    order_fast = list(_step_view(simulator).order_fast)
    step_start = float(simulator.current_time)
    composition = GroupComposition(interval=step_dt)
    if splitting_mode == "strang" and len(order_fast) >= 2:
        half_dt = step_dt * 0.5
        # Coverage of a half-step within the step (exactly 1/2 in binary64).
        half_coverage = half_dt / step_dt
        split_updates: Dict[str, Any] = {}
        forward_half = order_fast[:-1]
        for name in forward_half:
            merge_sequential_updates(
                split_updates,
                simulator._execute_group(
                    [name], half_dt, start_time=step_start,
                    composition=composition,
                ),
                coverage=half_coverage,
            )
        merge_sequential_updates(
            split_updates,
            simulator._execute_group(
                [order_fast[-1]], step_dt, start_time=step_start,
                composition=composition,
            ),
            coverage=1.0,
        )
        for name in reversed(forward_half):
            merge_sequential_updates(
                split_updates,
                simulator._execute_group(
                    [name], half_dt, start_time=step_start + half_dt,
                    composition=composition,
                ),
                coverage=half_coverage,
            )
        composition.finalize_boundary_rates(simulator.state_manager)
        return split_updates
    updates = cast(
        Dict[str, Any],
        simulator._execute_group(
            order_fast, step_dt, start_time=step_start,
            composition=composition,
        ),
    )
    composition.finalize_boundary_rates(simulator.state_manager)
    return updates


def _execute_fast_with_adaptive_retry(
    simulator: Any,
    dt: float,
    adaptive_enabled: bool,
    all_updates: Dict[str, Any],
) -> float:
    """Execute the fast group; with adaptive stepping, retry a rejected attempt once.

    Returns the interval actually integrated, which is never longer than ``dt``,
    so ``run()`` (which truncates its last step to ``end_time - current_time``)
    never passes ``end_time``. ``step(dt)`` integrates ``dt`` or, after one
    rejection, the shorter retry; ``_adaptive_next_dt`` only feeds ``run()``.

    References:
        Hairer, Norsett and Wanner (1993), Solving ODEs I, Sect. II.4
        (step-size selection with safety factor and growth limits);
        Gaines and Lyons (1997), SIAM J. Appl. Math. 57 (Brownian tree: a
        retry refines the rejected increment on a sub-interval).
    Mathematical expression:
        Err > 1 and dt > dt_min:  retry = max(dt * max(0.1, 0.9 / sqrt(Err)), dt_min)
                                  (dt_min < dt and the factor < 1, so retry < dt);
                                  roll back, run once over [t, t + retry], accept,
                                  dt_next = retry.
        Err > 1 and dt <= dt_min: no shorter retry exists; accept the attempt
                                  (WARNING naming Err and dt_min) and propose as
                                  for an accepted step.
        accepted step:            dt_next = dt * min(5, max(0.1, 0.9 / sqrt(Err)))
                                  for Err > 0, 2 * dt for Err = 0; then clipped
                                  to [dt_min, dt_max].
    Parameter meanings:
        dt the attempted interval; Err the update-magnitude metric of
        ``_compute_error_norm``; dt_min, dt_max ``adaptive_dt_min`` and
        ``adaptive_dt_max``; 0.9, 0.1, 5, 2 fixed safety and limit factors.
    Units: dt, retry, dt_next, dt_min, dt_max [s]; Err dimensionless.
    Domain/constraints: dt > 0; at most one retry, accepted without a second
        check (an accepted step can have Err > 1); a retry is never longer
        than the rejected attempt.
    Sign convention: time increases; all intervals are positive.
    """
    if not _step_view(simulator).order_fast:
        return dt

    if not adaptive_enabled:
        fast_updates = _execute_fast_group(simulator, dt)
        _accumulate_updates(all_updates, fast_updates)
        simulator._adaptive_next_dt = dt
        return dt

    state_before_fast = _clone_state_snapshot(simulator._get_current_state())
    state_manager = getattr(simulator, "state_manager", None)
    manager_snapshot = (
        state_manager.checkpoint()
        if state_manager is not None and hasattr(state_manager, "checkpoint")
        else None
    )
    sources_snapshot = _clone_runtime_value(
        getattr(simulator, "_step_update_sources", None)
    )
    increments_snapshot = _clone_runtime_value(
        getattr(simulator, "_step_functor_increments", None)
    )
    provenance_snapshot = _snapshot_runtime_value(
        "_step_provenance_detail",
        getattr(simulator, "_step_provenance_detail", None),
    )
    intent_count = len(getattr(simulator, "_active_step_external_intents", ()))
    functor_state_snapshot = _snapshot_functor_state_dicts(simulator)
    rng_snapshot = _capture_rng_state(simulator)
    fast_updates = _execute_fast_group(simulator, dt)
    _accumulate_updates(all_updates, fast_updates)

    error_norm = _compute_error_norm(simulator, state_before_fast, fast_updates)
    dt_min = float(simulator._adaptive_dt_min)
    if error_norm > 1.0 and dt > dt_min:
        # dt_min < dt and the factor is below 1, so the retry is strictly shorter
        # than the rejected attempt and lies inside [t, t + dt].
        retry_dt = max(dt * max(0.1, 0.9 / (error_norm ** 0.5)), dt_min)
        logger.debug(
            "Adaptive step rejected (err=%.3f), retrying with dt=%.6fs",
            error_norm,
            retry_dt,
        )
        if manager_snapshot is not None:
            assert state_manager is not None
            state_manager.restore(manager_snapshot)
        else:
            _restore_state_snapshot(simulator, state_before_fast)
        _restore_functor_state_dicts(simulator, functor_state_snapshot)
        _restore_rng_state(rng_snapshot, simulator)
        # Variable steps stay on one Wiener path: every fast-group Functor
        # that owns a Brownian tree pins the rejected trial's increment after
        # its substream was rewound, so the retry refines it by the bridge
        # instead of redrawing a different path (Gaines and Lyons 1997).
        for functor_name in _step_view(simulator).order_fast:
            provider = _step_view(simulator).providers.get(functor_name)
            if provider is None:
                continue
            record_path = getattr(provider.functor, "record_rejected_trial_path", None)
            if callable(record_path):
                record_path()
        all_updates.clear()
        simulator._step_update_sources = _clone_runtime_value(sources_snapshot)
        simulator._step_functor_increments = _clone_runtime_value(increments_snapshot)
        _restore_runtime_value(simulator, "_step_provenance_detail", provenance_snapshot)
        if hasattr(simulator, "_active_step_external_intents"):
            del simulator._active_step_external_intents[intent_count:]
        fast_updates = _execute_fast_group(simulator, retry_dt)
        _accumulate_updates(all_updates, fast_updates)
        simulator._adaptive_next_dt = retry_dt
        return retry_dt

    if error_norm > 1.0:
        logger.warning(
            "Adaptive step accepted with Err=%.6g > 1: the attempt dt=%.6g s is not "
            "longer than adaptive_dt_min=%.6g s, so no shorter retry exists; the "
            "interval is integrated once without rollback.",
            error_norm,
            dt,
            dt_min,
        )
    if error_norm > 0.0:
        dt_next = dt * min(5.0, max(0.1, 0.9 / (error_norm ** 0.5)))
    else:
        dt_next = dt * 2.0
    dt_next = min(dt_next, float(simulator._adaptive_dt_max))
    dt_next = max(dt_next, dt_min)
    simulator._adaptive_next_dt = dt_next
    return dt


def _should_execute_rate_group(
    simulator: Any,
    group_name: str,
    group_order: List[str],
    future_time: float,
) -> bool:
    """Check whether a medium/slow group should execute at ``future_time``.

    Order: interval boundary, then the bootstrap mark, then event triggers
    (triggers are consulted only when neither of the first two starts the
    group). A pending bootstrap mark (``force_initial_execution``) is consumed
    whenever the group executes, whatever started it, so a group that ran for
    its first boundary does not run once more at the next step without one.
    ``_bootstrap_groups`` is a step-transaction attribute: a rejected or failed
    step restores the mark.
    """
    if not group_order:
        return False

    if group_name == "medium":
        should_execute = bool(simulator._should_execute_medium(future_time))
    else:
        should_execute = bool(simulator._should_execute_slow(future_time))
    if not should_execute:
        should_execute = group_name in simulator._bootstrap_groups
    if not should_execute:
        should_execute = bool(
            simulator._check_event_triggers(group_name, simulator._get_current_state())
        )
    if should_execute:
        simulator._bootstrap_groups.discard(group_name)
    return should_execute


def _execute_rate_group_if_needed(
    simulator: Any,
    group_name: str,
    group_order: List[str],
    future_time: float,
    dt: float,
    all_updates: Dict[str, Any],
) -> bool:
    """Execute medium/slow group when schedule or trigger conditions are met.

    The group call reads the interval means of its interval_mean exchanges in
    place of the committed variables (_exchange_overlay); nothing else changes.
    """
    if not _should_execute_rate_group(simulator, group_name, group_order, future_time):
        return False

    if group_name == "medium":
        last_execution_attr = "_last_medium_execution_time"
    else:
        last_execution_attr = "_last_slow_execution_time"
    last_execution_time = getattr(simulator, last_execution_attr)
    # References: architecture_overview.md 4.4.1 (effective dt contract); Lawrence et
    #   al. (2019) CLM5 Technical Note, Section 2.7 (multi-rate stepping).
    # Mathematical expression: dt_eff = t_future - t_last, the elapsed time since the
    #   group last executed (or the run start / restored checkpoint time); the group
    #   call accounts for [t_last, t_future] and therefore receives (time, dt) =
    #   (t_last, dt_eff), with sub-step k at t_last + k * dt_eff / n_substeps.
    # Parameter meanings: t_future is the end of the current step; t_last is the
    #   committed last-execution time of this group.
    # Units: t_future, t_last, dt_eff [s].
    # Domain/constraints: dt_eff > 0 whenever the group is due; a non-positive value
    #   means the last-execution bookkeeping is ahead of the clock (corrupted restore
    #   or manual mutation) and is an error, never silently replaced by the step dt.
    # Sign convention: time increases; dt_eff is the positive elapsed interval.
    effective_dt = future_time - last_execution_time
    if effective_dt <= 0:
        raise RuntimeError(
            f"{group_name} group effective dt is {effective_dt} s (future_time={future_time}, "
            f"last_execution={last_execution_time}); the last-execution bookkeeping is inconsistent."
        )

    logger.debug(
        "Executing %s group over [%ss, %ss] (effective_dt=%ss)",
        group_name,
        last_execution_time,
        future_time,
        effective_dt,
    )
    overlay = _exchange_overlay(simulator, group_name)
    simulator._active_exchange_overlay = overlay or None
    try:
        group_updates = simulator._execute_group(
            group_order,
            effective_dt,
            start_time=last_execution_time,
        )
    finally:
        simulator._active_exchange_overlay = None
    _accumulate_updates(all_updates, group_updates)
    setattr(simulator, last_execution_attr, future_time)
    # The boundary predicate is pure; the interval bookkeeping advances here, inside the
    # step transaction, so a rejected trial cannot consume a trigger.
    commit_rate_group_interval(simulator, group_name, future_time)
    return True


def _apply_distributed_allreduce(
    simulator: Any,
    all_updates: Dict[str, Any],
    has_distributed_fn: Any,
    is_distributed_fn: Any,
    get_world_size_fn: Optional[Callable[[], int]],
) -> None:
    """Run distributed AllReduce over tensor updates when distributed mode is active."""
    distributed_enabled = (
        simulator.enable_distributed
        and callable(has_distributed_fn)
        and has_distributed_fn()
        and callable(is_distributed_fn)
        and is_distributed_fn()
    )
    if not distributed_enabled:
        return

    # Synchronize all processes before aggregation.
    simulator.process_group_manager.barrier()

    # References: PyTorch distributed all_reduce (sum) semantics; the
    #   coordinated registry (E02) guarantees identical provider sets, so
    #   every rank issues the same collective sequence (sorted names).
    # Mathematical expression: x_bar = (1 / N) * sum_{r=0}^{N-1} x_r for
    #   every tensor payload x_r of an update on rank r.
    # Parameter meanings: N world size, x_r local update tensor (raw tensor
    #   or the value of an UpdateItem), x_bar the ensemble-mean update.
    # Units: those of the update itself.
    # Domain/constraints: N >= 1; only tensor payloads are reduced; the sum
    #   over ranks equals N * x_bar (reduction balance).
    # Sign convention: not applicable.
    world_size = get_world_size_fn() if callable(get_world_size_fn) else 1
    reduced = 0
    for var_name in sorted(all_updates):
        update = all_updates[var_name]
        if isinstance(update, UpdateItem) and isinstance(update.value, Tensor):
            payload = update.value
            simulator.process_group_manager.all_reduce(payload)
            update.value = payload / world_size
            reduced += 1
        elif isinstance(update, Tensor):
            simulator.process_group_manager.all_reduce(update)
            all_updates[var_name] = update / world_size
            reduced += 1

    logger.debug(
        "Distributed AllReduce completed for %s of %s variables", reduced, len(all_updates)
    )


def _collect_executed_groups(
    has_fast_group: bool,
    execute_medium: bool,
    execute_slow: bool,
) -> List[str]:
    """Build executed group list for cross-domain validation."""
    executed_groups: List[str] = []
    if has_fast_group:
        executed_groups.append("fast")
    if execute_medium:
        executed_groups.append("medium")
    if execute_slow:
        executed_groups.append("slow")
    return executed_groups


def _get_temporal_zone_parameter(key: str) -> Any:
    """Read one required ``computation.temporal_zones.*`` parameter (YAML SSOT).

    Values, units, dtype, and bounds live in
    ``relic\\parameters\\database\\relic\\computation\\temporal_zones.yaml``;
    there is no Python fallback value.
    """
    return get_global_manager().get(key, required=True)


def _get_functor_execution_group(simulator: Any, functor_name: str) -> str:
    """Resolve execution group for a functor from simulator group map."""
    groups = getattr(simulator, "_groups", {})
    if isinstance(groups, dict):
        for group_name, members in groups.items():
            if isinstance(members, list) and functor_name in members:
                return str(group_name)
    return "fast"


def _check_cross_zone_error_transfer(
    simulator: Any,
    all_updates: Dict[str, Any],
    dt: float,
) -> Dict[str, Any]:
    """Monitor cross-zone contribution imbalance for updated variables.

    References:
        Knoll and Keyes (2004), J. Comput. Phys. 193:357-397;
        Sportisse (2000), J. Comput. Phys. 161:140-168.
    Mathematical expression:
        zone_contrib[z] = sum over writers w in zone z of sum_elements |delta_{v,w}|
        cross_transfer_ratio = max_z zone_contrib[z] / sum_z zone_contrib[z]
        warn when cross_transfer_ratio > theta.
    Parameter meanings:
        v is an updated variable; z a temporal zone (core, physics, coupling of
        the fast, medium, slow groups); delta_{v,w} the increment of writer w;
        theta = computation.temporal_zones.cross_transfer_threshold; monitoring
        runs only when computation.temporal_zones.monitoring_enabled is true.
    Units: delta in the units of v; ratio and theta dimensionless; dt [s].
    Domain/constraints: both keys are required YAML parameters with no Python
        fallback; monitoring_enabled must be a bool and theta must lie in
        [0, 1] (a value outside raises; it is not clipped).
    Sign convention: magnitudes only; the ratio is non-negative.
    """
    monitoring_enabled = _get_temporal_zone_parameter(
        "computation.temporal_zones.monitoring_enabled"
    )
    if not isinstance(monitoring_enabled, bool):
        raise TypeError(
            "computation.temporal_zones.monitoring_enabled must be a bool, got "
            f"{type(monitoring_enabled).__name__}"
        )
    if not monitoring_enabled:
        return {"enabled": False, "checked_variables": 0, "violations": []}

    threshold = float(
        _get_temporal_zone_parameter("computation.temporal_zones.cross_transfer_threshold")
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "computation.temporal_zones.cross_transfer_threshold must lie in [0, 1], "
            f"got {threshold}"
        )

    increment_map_all = getattr(simulator, "_step_functor_increments", {}) or {}
    violations: List[Dict[str, Any]] = []
    checked_variables = 0

    for var_name, update in all_updates.items():
        payload = update.value if isinstance(update, UpdateItem) else update
        if not isinstance(payload, Tensor):
            continue

        checked_variables += 1
        zone_contrib: Dict[str, float] = {"core": 0.0, "physics": 0.0, "coupling": 0.0}

        increment_map = increment_map_all.get(var_name) if isinstance(increment_map_all, dict) else None
        if isinstance(increment_map, dict) and increment_map:
            for writer_name, increment_tensor in increment_map.items():
                if not isinstance(writer_name, str):
                    continue
                if not isinstance(increment_tensor, Tensor):
                    continue
                group_name = _get_functor_execution_group(simulator, writer_name)
                zone_name = get_temporal_zone_label(group_name)
                contribution = float(increment_tensor.detach().abs().sum().cpu().item())
                zone_contrib[zone_name] = zone_contrib.get(zone_name, 0.0) + contribution
        else:
            source_map = getattr(simulator, "_step_update_sources", {}) or {}
            if isinstance(source_map, dict) and var_name in source_map:
                writer_name = source_map[var_name][0]
                group_name = _get_functor_execution_group(simulator, writer_name)
                zone_name = get_temporal_zone_label(group_name)
            else:
                zone_name = "core"
            zone_contrib[zone_name] = float(payload.detach().abs().sum().cpu().item())

        total_contrib = sum(zone_contrib.values())
        if total_contrib <= 0.0:
            continue
        ratio = max(zone_contrib.values()) / total_contrib
        if ratio > threshold:
            violation = {
                "variable": var_name,
                "cross_transfer_ratio": ratio,
                "threshold": threshold,
                "zone_contributions": zone_contrib,
                "dt": float(dt),
            }
            violations.append(violation)
            logger.warning(
                "Cross-zone error transfer alert for '%s': ratio=%.3f > %.3f, "
                "contrib=%s",
                var_name,
                ratio,
                threshold,
                zone_contrib,
            )

    return {
        "enabled": True,
        "checked_variables": checked_variables,
        "violations": violations,
        "threshold": threshold,
    }


def _advance_forcing_cursor(simulator: Any) -> None:
    """Consume one forcing snapshot inside the accepted step transaction.

    References:
        RELIC restart contract (F02): the consumed forcing position is part
        of the accepted step state, so a checkpoint delivered after this step
        carries the position of the next snapshot to apply.
    Mathematical expression:
        consumed_{n+1} = consumed_n + 1, with 0 <= consumed <= length.
    Parameter meanings:
        consumed counts accepted steps that applied a sequence snapshot;
        length is the bound sequence length.
    Units: dimensionless counts.
    Domain/constraints: a step beyond the sequence end is refused before the
        cursor moves.
    Sign convention: the cursor only increases inside accepted steps and is
        restored by the step transaction on rejection.
    """
    cursor = getattr(simulator, "_forcing_cursor", None)
    if cursor is None:
        return
    if cursor["consumed"] >= cursor["length"]:
        raise RuntimeError(
            "Forcing sequence exhausted: "
            f"{cursor['consumed']}/{cursor['length']} snapshots already consumed"
        )
    simulator._forcing_cursor = dict(cursor, consumed=cursor["consumed"] + 1)


def _finalize_step(
    simulator: Any,
    dt: float,
    all_updates: Dict[str, Any],
    start_time: Optional[Any],
    end_time: Optional[Any],
    compute_wall_start: float,
    stage_timings: Optional[Dict[str, float]],
) -> None:
    """Finalize step bookkeeping, timing, and in-memory outputs."""
    if all_updates:
        _check_cross_zone_error_transfer(simulator, all_updates, dt)

    simulator.current_time += dt
    simulator.step_count += 1
    _advance_forcing_cursor(simulator)

    if start_time is not None and end_time is not None:
        end_time.record()
        torch.cuda.synchronize()
        step_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        simulator.total_execution_time += step_time
        if stage_timings is not None:
            stage_timings["cuda_compute"] = max(0.0, float(step_time))
    else:
        simulator.total_execution_time += max(
            0.0,
            time.perf_counter() - compute_wall_start,
        )

    if stage_timings is not None:
        stage_timings["step_compute"] = max(
            0.0,
            time.perf_counter() - compute_wall_start,
        )

    if simulator.step_count % simulator.output_frequency == 0:
        output_started = _stage_start(stage_timings)
        simulator._store_outputs(all_updates)
        _record_stage(stage_timings, "output", output_started)


def _automatic_delivery_records(simulator: Any) -> List[Dict[str, Any]]:
    """Automatic diagnostic and checkpoint requests due at the current step."""
    records: List[Dict[str, Any]] = []
    if simulator.enable_diagnostic_export and simulator.step_count % simulator.diagnostic_output_frequency == 0:
        if simulator.diagnostic_output_path is None:
            raise RuntimeError("Diagnostic export requires an output path")
        records.append({"functor": None, "substep": None, "time": simulator.current_time,
                        "output": "automatic_diagnostic", "automatic": True,
                        "intent": ExternalIntent("diagnostic", str(simulator.diagnostic_output_path), "automatic")})
    if simulator.enable_checkpointing and simulator.current_time - simulator.last_checkpoint_time >= simulator.checkpoint_frequency:
        records.append({"functor": None, "substep": None, "time": simulator.current_time,
                        "output": "automatic_checkpoint", "automatic": True,
                        "intent": ExternalIntent("checkpoint", str(simulator._resolve_checkpoint_path()), "automatic")})
    return records


def _delivery_target(simulator: Any, intent: ExternalIntent) -> Tuple[Dict[str, Any], str, str]:
    """Validate one intent and return its (parameters, resource, dedup target)."""
    if intent.operation not in ("checkpoint", "diagnostic", "topology"):
        raise ValueError(f"Unsupported external intent operation: {intent.operation}")
    if intent.operation == "diagnostic":
        allowed = {"reference_date"}
    elif intent.operation == "topology":
        allowed = {"transition"}
    else:
        allowed = set()
    if set(intent.payload).difference(allowed):
        raise ValueError(f"Unsupported {intent.operation} delivery options: {sorted(intent.payload)}")
    parameters: Dict[str, Any] = {}
    if intent.operation == "diagnostic":
        parameters["reference_date"] = intent.payload.get(
            "reference_date", simulator.diagnostic_reference_date,
        )
    if intent.operation == "topology":
        # A topology request names one declared transition; it is checked
        # here, while the scientific step can still be rolled back, and
        # committed by the lifecycle owner after acceptance.
        transition_name = str(intent.payload.get("transition", intent.resource))
        if transition_name != intent.resource:
            raise ValueError("Topology intent payload transition must match its resource")
        if transition_name not in simulator._provider_lifecycle.topology_transitions:
            raise ValueError(f"Undeclared topology transition requested: {transition_name}")
        parameters["transition"] = transition_name
        resource = f"topology:{transition_name}"
        return parameters, resource, resource
    resource = str(Path(intent.resource).resolve())
    return parameters, resource, os.path.normcase(resource)


def _prepare_external_deliveries(
    simulator: Any, collected: List[Dict[str, Any]], *, accepted_auto_report_enabled: bool,
) -> List[Dict[str, Any]]:
    """Validate and combine requests while scientific rollback is available."""
    records = list(collected) + _automatic_delivery_records(simulator)
    by_target: Dict[str, Dict[str, Any]] = {}
    for record in records:
        intent = record["intent"]
        parameters, resource, target = _delivery_target(simulator, intent)
        source = {name: value for name, value in record.items() if name != "intent"}
        source["request_id"] = intent.request_id
        if target in by_target:
            job = by_target[target]
            if job["operation"] != intent.operation or job["parameters"] != parameters:
                raise ValueError(f"External requests have conflicting parameters for {resource}")
            job["sources"].append(source)
            job["automatic"] = job["automatic"] or bool(record.get("automatic", False))
            continue
        identity = [simulator.run_id, simulator.step_count, intent.operation, target, intent.request_id]
        key = hashlib.sha256(json.dumps(identity, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        by_target[target] = {
            "key": key, "run_id": simulator.run_id, "accepted_step": simulator.step_count,
            "accepted_time": simulator.current_time, "operation": intent.operation,
            "resource": resource, "parameters": parameters, "sources": [source],
            "automatic": bool(record.get("automatic", False)), "snapshot": None,
            "attempts": 0, "error": None,
        }
    jobs = []
    for job in by_target.values():
        if job["operation"] == "checkpoint":
            if Path(job["resource"]).exists():
                raise FileExistsError(f"Checkpoint already exists: {job['resource']}")
        elif job["operation"] == "topology":
            job["snapshot"] = {
                "transition": job["parameters"]["transition"],
                "requested_step": int(simulator.step_count),
                "requested_time": float(simulator.current_time),
            }
        else:
            job["snapshot"] = capture_diagnostic_export(
                simulator, output_path=job["resource"], **job["parameters"],
            )
            if job["snapshot"] is None:
                if all(source.get("automatic", False) for source in job["sources"]):
                    continue
                raise ValueError("Explicit diagnostic delivery has no summaries to export")
        jobs.append(job)
    if any(job["operation"] == "checkpoint" for job in jobs):
        # Freeze one scientific state after every request is prepared. Each
        # checkpoint root retains the complete accepted batch through refs.
        batch = capture_checkpoint_payload(
            simulator, pending_commit=True,
            accepted_auto_report_enabled=accepted_auto_report_enabled,
            pending_deliveries=jobs,
        )
        for job in jobs:
            if job["operation"] == "checkpoint":
                job["snapshot"] = checkpoint_delivery_snapshot(batch, job["key"])
    return jobs


def retry_external_deliveries(
    simulator: Any, *, stage_timings: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Drain the one simulator delivery queue without advancing science."""
    state_manager = getattr(simulator, "state_manager", None)
    if state_manager is not None:
        if state_manager._provenance_transaction_active:
            raise RuntimeError("External delivery requires an accepted, inactive provenance transaction")
        # Confirm old notifications before any new savepoint or file effect.
        # A successful prefix must never be restored by a later rejected step.
        if state_manager._pending_provenance_events:
            state_manager.commit_provenance_transaction()
    completed: List[Dict[str, Any]] = []
    pending = simulator._pending_external_deliveries
    while pending:
        job = pending[0]
        if job["snapshot"] is None:
            raise RuntimeError(f"Delivery {job['key']} has no captured accepted payload: {job['error']}")
        job["attempts"] += 1
        delivery_started = _stage_start(stage_timings)
        try:
            if job["operation"] == "checkpoint":
                receipt = simulator._create_checkpoint(job["resource"], snapshot=job["snapshot"])
                if job["automatic"] and job["run_id"] == simulator.run_id:
                    simulator.last_checkpoint_time = job["accepted_time"]
            elif job["operation"] == "topology":
                from .registration import apply_topology_transition

                lifecycle_receipt = apply_topology_transition(
                    simulator,
                    job["parameters"]["transition"],
                    accepted_step=job["accepted_step"],
                    accepted_time=job["accepted_time"],
                )
                receipt = {
                    "transition": job["parameters"]["transition"],
                    "receipt_id": lifecycle_receipt.receipt_id,
                    "epoch_before": lifecycle_receipt.epoch_before,
                    "epoch_after": lifecycle_receipt.epoch_after,
                    "status": lifecycle_receipt.status,
                }
            else:
                written = simulator._export_diagnostics(snapshot=job["snapshot"], delivery_key=job["key"])
                if written is None or not written.validation_passed:
                    raise OSError("Diagnostic delivery returned no successful receipt")
                receipt = {"path": str(written.output_path), "sha256": written.sha256,
                           "size_bytes": written.file_size, "replayed": written.replayed}
            record = {name: job[name] for name in (
                "key", "run_id", "accepted_step", "accepted_time", "operation", "sources", "attempts",
            )}
            record["receipt"] = receipt
            previous = simulator._last_external_delivery_receipts
            matching = next((index for index, item in enumerate(previous) if item["key"] == job["key"]), None)
            if matching is None:
                previous.append(record)
            else:
                previous[matching] = record
            completed.append(record)
            pending.pop(0)
        except BaseException as exc:
            job["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if job["operation"] == "checkpoint":
                stage = "checkpoint"
            elif job["operation"] == "topology":
                stage = "topology_transition"
            else:
                stage = "diagnostic_export"
            _record_stage(stage_timings, stage, delivery_started)
    return completed


_STEP_TRANSACTION_ATTRIBUTES = (
    "state",
    "run_id",
    "current_time",
    "step_count",
    "total_execution_time",
    "last_checkpoint_time",
    "time_series",
    "_previous_day_index",
    "_previous_month_index",
    "_previous_medium_index",
    "_previous_slow_index",
    "_last_medium_execution_time",
    "_last_slow_execution_time",
    "_bootstrap_groups",
    "_adaptive_next_dt",
    "_last_step_had_violation",
    "_step_update_sources",
    "_step_functor_increments",
    "_step_provenance_detail",
    "_provenance_evicted_counts",
    "_active_composition",
    "_last_state_reset_counter",
    "_initial_state",
    "_scale_check_history",
    "_forcing_cursor",
)


def _snapshot_step_runtime(simulator: Any) -> Dict[str, Any]:
    """Capture simulator bookkeeping that must change only on accepted steps."""
    snapshot: Dict[str, Any] = {}
    for name in _STEP_TRANSACTION_ATTRIBUTES:
        if not hasattr(simulator, name):
            continue
        snapshot[name] = _snapshot_runtime_value(name, getattr(simulator, name))
    return snapshot


def _restore_step_runtime(simulator: Any, snapshot: Dict[str, Any]) -> None:
    """Restore simulator bookkeeping after a rejected or failed step."""
    for name in _STEP_TRANSACTION_ATTRIBUTES:
        if name not in snapshot and hasattr(simulator, name):
            delattr(simulator, name)
    for name, value in snapshot.items():
        _restore_runtime_value(simulator, name, value)


def step_simulation(
    self: Any,
    dt: Optional[float],
    simulation_state_enum: Any,
    has_distributed_fn: Any,
    is_distributed_fn: Any,
    get_world_size_fn: Any,
) -> FluxDict:
    """
    Execute one simulation step.

    Parameters
    ----------
    self : Any
        RELICSimulator instance.
    dt : Optional[float]
        Step size.
    simulation_state_enum : Any
        SimulationState enum class from ``main.py``.

    References:
        Hairer, Norsett, and Wanner (1993), Solving Ordinary Differential
        Equations I, time-marching interval requirements.
    Mathematical expression:
        dt is in Real excluding bool, is finite, and dt > 0.
    Parameter meanings:
        dt is the simulator integration interval [s].
    """
    if dt is None:
        dt = self.time_step
    if isinstance(dt, bool) or not isinstance(dt, Real):
        raise TypeError("Simulation time step must be a real scalar")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Simulation time step must be finite and positive")
    provenance_limit = self._get_setting("max_provenance_records_per_variable", 256)
    if type(provenance_limit) is not int or provenance_limit < 1:
        raise ValueError("max_provenance_records_per_variable must be a positive integer")
    retry_external_deliveries(self)

    step_wall_start = time.perf_counter()
    profiling_enabled = bool(
        getattr(
            self,
            "enable_profiling",
            self._get_setting("enable_profiling", False),
        )
    )
    stage_timings = _new_step_stage_timings(profiling_enabled)

    # Explicit auto-initialization is a lifecycle bootstrap that precedes the
    # scientific step transaction. An ordinary INITIALIZED -> RUNNING
    # transition is part of the transaction and must therefore be reversible.
    prepared_before_transaction = (
        self.state == simulation_state_enum.UNINITIALIZED
        and bool(getattr(self, "allow_uninitialized_step", False))
    )
    if prepared_before_transaction:
        _prepare_step_state(self, simulation_state_enum)

    state_manager = getattr(self, "state_manager", None)
    manager_snapshot_started = _stage_start(stage_timings)
    manager_snapshot = (
        state_manager.create_transaction_savepoint()
        if state_manager is not None
        else None
    )
    _record_stage(stage_timings, "manager_snapshot", manager_snapshot_started)

    runtime_snapshot_started = _stage_start(stage_timings)
    runtime_snapshot = _snapshot_step_runtime(self)
    _record_stage(stage_timings, "runtime_snapshot", runtime_snapshot_started)

    functor_snapshot_started = _stage_start(stage_timings)
    functor_state_snapshot = _snapshot_functor_state_dicts(self)
    _record_stage(
        stage_timings,
        "functor_state_snapshot",
        functor_snapshot_started,
    )

    rng_snapshot_started = _stage_start(stage_timings)
    rng_snapshot = _capture_rng_state(self)
    _record_stage(stage_timings, "rng_snapshot", rng_snapshot_started)
    auto_report_enabled = (
        bool(getattr(state_manager, "_enable_auto_report", True))
        if state_manager is not None
        else True
    )
    provenance_transaction_started = False
    step_accepted = False
    start_time: Optional[Any] = None
    end_time: Optional[Any] = None
    compute_wall_start = time.perf_counter()
    all_updates: Dict[str, Any] = {}
    functor_timing_snapshot: Dict[str, Dict[str, Any]] = {}
    functor_profiling_flags: Dict[str, bool] = {}
    self._active_step_external_intents = []
    prepared_deliveries: List[Dict[str, Any]] = []
    step_view_token: Optional[Any] = None
    step_view_token: Optional[Any] = None

    if stage_timings is not None:
        self._active_step_stage_timings = stage_timings
        for functor_name, functor in self.functors.items():
            functor_profiling_flags[functor_name] = bool(
                getattr(functor, "_performance_profiling_enabled", False)
            )
            functor_timing_snapshot[functor_name] = {
                "count": int(getattr(functor, "_pipeline_timing_count", 0)),
                "totals_s": dict(
                    getattr(functor, "_pipeline_timing_totals_s", {})
                ),
            }
            functor._performance_profiling_enabled = True

    try:
        try:
            if not prepared_before_transaction:
                _prepare_step_state(self, simulation_state_enum)
            _sync_state_reset_if_needed(self)
            # The committed provider view (P_e, Omega_e, e) is frozen for the
            # whole step transaction: every group, substep, and compensation
            # resolves providers and orders from this one reader token.
            admitted_token = self.acquire_provider_view()
            step_view_token = admitted_token
            step_view = admitted_token.view
            self._active_step_view = step_view
            compute_wall_start = time.perf_counter()
            start_time, end_time = _initialize_step_timing_events(self)

            self._step_update_sources = {}
            self._step_functor_increments = {}
            self._last_step_had_violation = False
            adaptive_enabled = bool(getattr(self, "_adaptive_dt_enabled", False))

            # Multi-rate execution applies fast, medium, and slow groups inside one
            # accepted-step transaction. Rejected trials never reach observers.

            if state_manager is not None:
                state_manager.set_auto_report_enabled(False)
                if hasattr(state_manager, "begin_provenance_transaction"):
                    state_manager.begin_provenance_transaction()
                    provenance_transaction_started = True

            fast_started = _stage_start(stage_timings)
            dt = _execute_fast_with_adaptive_retry(
                self,
                dt,
                adaptive_enabled,
                all_updates,
            )
            _record_stage(stage_timings, "fast_group", fast_started)
            future_time = self.current_time + dt

            # Multirate exchanges: one owner commit per interval_mean exchange over
            # the accepted fast step (after its boundary-rate and S2 finalization
            # and after the adaptive acceptance), before the consumer groups run
            # with their overlays (design 20261019_cr_a04_design_01, section 2).
            exchange_started = _stage_start(stage_timings)
            _commit_multirate_exchanges(self, dt)
            _record_stage(stage_timings, "state_apply_updates", exchange_started)

            medium_started = _stage_start(stage_timings)
            execute_medium = _execute_rate_group_if_needed(
                self,
                group_name="medium",
                group_order=list(step_view.order_medium),
                future_time=future_time,
                dt=dt,
                all_updates=all_updates,
            )
            _record_stage(stage_timings, "medium_group", medium_started)

            slow_started = _stage_start(stage_timings)
            execute_slow = _execute_rate_group_if_needed(
                self,
                group_name="slow",
                group_order=list(step_view.order_slow),
                future_time=future_time,
                dt=dt,
                all_updates=all_updates,
            )
            _record_stage(stage_timings, "slow_group", slow_started)

            # The distributed reduction runs inside each group before its
            # updates are written (execution_group); its time is accumulated
            # under the "distributed_collective" stage there.
            executed_groups = _collect_executed_groups(
                bool(step_view.order_fast),
                execute_medium,
                execute_slow,
            )
            if executed_groups:
                validation_started = _stage_start(stage_timings)
                self._validate_cross_domain_coupling(executed_groups)
                _record_stage(
                    stage_timings,
                    "cross_domain_validation",
                    validation_started,
                )

            # No step-level conservation report is built here. Every group commit
            # runs the commit-time conservation checks (StateManager.apply_updates
            # with check_conservation), and StateManager's per-commit report is
            # suspended for the step transaction (set_auto_report_enabled above).
            # A report needs the committed, normalized UpdateItems and the commit
            # interval of each commit (a boundary flux is a REPLACE rate with its
            # dt); all_updates is the returned mapping of payloads, so a report
            # built from it rejects every declared boundary flux and counts
            # REPLACE values and per-second rates as increments.

            _finalize_step(
                self,
                dt,
                all_updates,
                start_time,
                end_time,
                compute_wall_start,
                stage_timings,
            )
            # References: Gray and Reuter (1993), Transaction Processing;
            #   RELIC architecture Section 4.4 accepted-step and restart contract.
            # Mathematical expression: q_v = max(0, len(H_v) - M);
            #   H_v_new = H_v[q_v:]; E_v_new = E_v + q_v.
            # Parameter meanings: H_v is the writer audit list for variable v;
            #   M is the configured record limit; E_v counts evicted records.
            # Units: M, q_v and E_v are dimensionless record counts.
            # Domain/constraints: M is a positive integer. Copy on eviction
            #   preserves the old list for append-only rollback. Trim before
            #   checkpoint preparation; never remove scientific/numerical state.
            # Sign convention: eviction counts are non-negative; no flux changes.
            evicted_counts = dict(getattr(self, "_provenance_evicted_counts", {}))
            for variable, history in self._step_provenance_detail.items():
                excess = len(history) - provenance_limit
                if excess > 0:
                    self._step_provenance_detail[variable] = history[excess:]
                    evicted_counts[variable] = evicted_counts.get(variable, 0) + excess
            self._provenance_evicted_counts = evicted_counts
            prepared_deliveries = _prepare_external_deliveries(
                self, self._active_step_external_intents,
                accepted_auto_report_enabled=auto_report_enabled,
            )
            if provenance_transaction_started:
                assert state_manager is not None
                try:
                    state_manager.commit_provenance_transaction()
                finally:
                    # StateManager closes this flag before notifying observers.
                    # A later cancellation cannot undo that accepted science.
                    if not state_manager._provenance_transaction_active:
                        step_accepted = True
                        provenance_transaction_started = False
            else:
                step_accepted = True
        except BaseException:
            if not step_accepted:
                if provenance_transaction_started:
                    assert state_manager is not None
                    state_manager.rollback_provenance_transaction()
                    provenance_transaction_started = False
                if manager_snapshot is not None:
                    assert state_manager is not None
                    state_manager.restore_transaction_savepoint(manager_snapshot)
                _restore_functor_state_dicts(self, functor_state_snapshot)
                _restore_step_runtime(self, runtime_snapshot)
                _restore_rng_state(rng_snapshot, self)
            raise
        finally:
            if step_accepted:
                # Publish the complete in-memory batch before cleanup can fail.
                # Payloads are excluded from the scientific savepoint, avoiding
                # recursive capture of the checkpoint delivery itself.
                self._pending_external_deliveries.extend(prepared_deliveries)
                self._last_external_delivery_receipts = []
                self._step_update_sources = None
            del self._active_step_external_intents
            self._active_step_view = None
            if step_view_token is not None:
                step_view_token.release()
            if provenance_transaction_started:
                assert state_manager is not None
                state_manager.rollback_provenance_transaction()
            try:
                if state_manager is not None:
                    state_manager.set_auto_report_enabled(auto_report_enabled)
            finally:
                if (
                    stage_timings is not None
                    and getattr(self, "_active_step_stage_timings", None) is stage_timings
                ):
                    delattr(self, "_active_step_stage_timings")
                for functor_name, previous_flag in functor_profiling_flags.items():
                    if functor_name in self.functors:
                        self.functors[functor_name]._performance_profiling_enabled = previous_flag

        retry_external_deliveries(self, stage_timings=stage_timings)
    finally:
        if step_accepted:
            if stage_timings is not None:
                stage_timings["step_total"] = max(0.0, time.perf_counter() - step_wall_start)
            _commit_performance_diagnostics(self, stage_timings, functor_timing_snapshot)
    return all_updates


def run_simulation(
    self: Any,
    start_time: float,
    end_time: Optional[float],
    dt: Optional[float],
    simulation_state_enum: Any,
) -> Dict[str, Any]:
    """
    Run simulation loop from start to end.

    Parameters
    ----------
    self : Any
        RELICSimulator instance.
    start_time : float
        Start time.
    end_time : Optional[float]
        End time.
    dt : Optional[float]
        Step size.
    simulation_state_enum : Any
        SimulationState enum class from ``main.py``.
    """
    if self.state not in [
        simulation_state_enum.INITIALIZED,
        simulation_state_enum.PAUSED,
        simulation_state_enum.COMPLETED,
    ]:
        raise RuntimeError(f"Cannot start simulation from state: {self.state}")

    if end_time is None:
        end_time = float(self.max_time)
    if dt is None:
        dt = float(self.time_step)

    self.current_time = start_time
    self.state = simulation_state_enum.RUNNING

    # Initialize time boundary tracking for multi-rate execution
    self._previous_day_index = self._compute_day_index(start_time)
    self._previous_month_index = self._compute_month_index(start_time)
    self._previous_medium_index = self._compute_interval_index(
        start_time,
        self._medium_interval_seconds,
    )
    slow_interval_seconds = getattr(self, "_slow_interval_seconds", None)
    if slow_interval_seconds is not None and float(slow_interval_seconds) > 0.0:
        self._previous_slow_index = self._compute_interval_index(
            start_time,
            float(slow_interval_seconds),
        )
    else:
        self._previous_slow_index = -1
    self._last_medium_execution_time = start_time
    self._last_slow_execution_time = start_time
    self._adaptive_next_dt = dt

    # The committed provider view owns the group execution orders.
    committed_view = self.get_committed_provider_view()
    if committed_view is None:
        raise RuntimeError("Cannot start the simulation without a committed provider view.")

    logger.info("Starting simulation: t=%ss to %ss, dt=%ss", start_time, end_time, dt)
    logger.info(
        "Multi-rate groups: fast=%s, medium=%s, slow=%s (provider epoch %s)",
        len(committed_view.order_fast),
        len(committed_view.order_medium),
        len(committed_view.order_slow),
        committed_view.epoch,
    )
    if self.enable_auto_scale:
        logger.info("Automatic multi-scale coupling enabled")

    try:
        while self.current_time < end_time:
            next_dt = dt
            if getattr(self, "_adaptive_dt_enabled", False):
                next_dt = float(getattr(self, "_adaptive_next_dt", dt))
            # Adjust final time step
            actual_dt = min(next_dt, end_time - self.current_time)

            # Execute time step
            self.step(actual_dt)
            if self._last_step_had_violation:
                raise RuntimeError("Conservation violation occurred during run")

            # Progress logging
            if self.step_count % 100 == 0:
                progress = (self.current_time - start_time) / (end_time - start_time) * 100
                logger.info("Simulation progress: %.1f%% (step %s)", progress, self.step_count)

        self.state = simulation_state_enum.COMPLETED
        logger.info("Simulation completed successfully after %s steps", self.step_count)

        # Report scale coupling statistics if enabled
        if self.enable_auto_scale and self._scale_check_history:
            regrid_events = sum(
                1 for check in self._scale_check_history if check.get("needs_regrid", False)
            )
            logger.info(
                "Multi-scale coupling statistics: %s regridding events performed",
                regrid_events,
            )

    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        self.state = simulation_state_enum.FAILED
        logger.error("Simulation failed: %s", e)
        raise

    return cast(Dict[str, Any], self._get_current_state())


_GROUP_RANK = {"fast": 0, "medium": 1, "slow": 2}


def _iter_declared_variables(value: Any) -> List[str]:
    """Return the declared variable names of one metadata field."""
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return []


def _strict_cross_group_setting(sim: Any) -> bool:
    """Read ``strict_cross_group_dependencies`` (scheduler key, default False)."""
    value = sim._get_setting("strict_cross_group_dependencies", False)
    if not isinstance(value, bool):
        raise TypeError(
            "strict_cross_group_dependencies must be a bool, got "
            f"{type(value).__name__} ({value!r})"
        )
    return value


def _cross_group_reads(sim: Any) -> List[Dict[str, Any]]:
    """List the declared cross-group reads of the current group membership.

    A read is a pair (consumer, producer) in different execution groups where a
    declared input of the consumer is a declared output of the producer. Its
    kind is ``sampling`` when the consumer group is slower than the producer
    group and ``hold`` when it is faster. It is acknowledged when the consumer
    names the producer in the ``lagged`` part of its dependency declaration.
    """
    group_of: Dict[str, str] = {}
    for group_name, members in sim._groups.items():
        for member in members:
            group_of[member] = group_name
    producers: Dict[str, List[str]] = {}
    for name, functor in sim.functors.items():
        metadata = getattr(functor, "metadata", None)
        for variable in _iter_declared_variables(getattr(metadata, "output_variables", None)):
            producers.setdefault(variable, []).append(name)

    reads: List[Dict[str, Any]] = []
    for consumer, functor in sim.functors.items():
        metadata = getattr(functor, "metadata", None)
        inputs = _iter_declared_variables(getattr(metadata, "input_variables", None))
        if not inputs:
            continue
        acknowledged = set(
            normalize_dependency_declaration(functor.get_dependencies(), owner=consumer).lagged
        )
        for variable in inputs:
            for producer in producers.get(variable, []):
                if producer == consumer:
                    continue
                if consumer not in group_of or producer not in group_of:
                    raise RuntimeError(
                        f"Registered providers '{consumer}' and '{producer}' must both "
                        "have an execution group membership"
                    )
                consumer_group = group_of[consumer]
                producer_group = group_of[producer]
                if consumer_group == producer_group:
                    continue
                reads.append(
                    {
                        "consumer": consumer,
                        "consumer_group": consumer_group,
                        "producer": producer,
                        "producer_group": producer_group,
                        "variable": variable,
                        "kind": (
                            "sampling"
                            if _GROUP_RANK[consumer_group] > _GROUP_RANK[producer_group]
                            else "hold"
                        ),
                        "acknowledged": producer in acknowledged,
                    }
                )
    return reads


def _require_contiguous_promotion(sim: Any, scheduled_members: List[str]) -> None:
    """Refuse a slow -> medium move that would leave an interval unaccounted.

    References: architecture_overview.md 4.4.1 (effective dt contract,
        dt_eff = t_future - t_last per group).
    Mathematical expression: a scheduled slow member has integrated up to
        t_s = t_last,slow; after the move its first medium call covers
        [t_m, t_future] with t_m = t_last,medium. The move is admitted only when
        t_s == t_m; t_s < t_m leaves [t_s, t_m] uncovered and t_s > t_m would
        integrate [t_m, t_s] twice.
    Parameter meanings: t_s and t_m are ``_last_slow_execution_time`` and
        ``_last_medium_execution_time`` (step-transaction attributes, part of the
        durable checkpoint runtime, so the check is identical after restore).
    Units: t_s, t_m [s].
    Domain/constraints: only members of the slow group in the previously
        committed view have an accounting history; a provider first published in
        this view is placed directly and starts with its first call. One group
        commit integrates with one interval (StateManager.apply_updates receives
        a single dt), so a per-Functor catch-up inside the medium call is not
        available; the move fails closed instead.
    Sign convention: time increases; intervals are [start, end] with start <= end.
    """
    if not scheduled_members:
        return
    slow_last = float(sim._last_slow_execution_time)
    medium_last = float(sim._last_medium_execution_time)
    if slow_last == medium_last:
        return
    if slow_last < medium_last:
        effect = f"leave the uncovered interval [{slow_last} s, {medium_last} s]"
    else:
        effect = f"integrate the interval [{medium_last} s, {slow_last} s] twice"
    raise RuntimeError(
        f"Cannot promote scheduled slow members {scheduled_members} to the medium group "
        f"(resolve_cross_group_deps=True): the slow group last ran at {slow_last} s and the "
        f"medium group at {medium_last} s, so the move would {effect} for them. "
        "Register the cross-group producer when both groups "
        "share their last execution time (for example before the run), declare the "
        "producer in the consumer's lagged dependencies, or place the consumer in the "
        "medium group from the start."
    )


def _promote_slow_consumers(sim: Any) -> List[str]:
    """Move slow consumers with an unacknowledged sampling lag to the medium group.

    The move changes group membership (``sim._groups``) and therefore the
    published orders and provider identities; a Functor is never a member of
    two groups. The loop runs to a fixed point because a moved consumer can
    become a faster producer for another slow consumer. A pending slow
    bootstrap mark of a moved Functor with ``force_initial_execution`` moves
    with it.
    """
    lifecycle = getattr(sim, "_provider_lifecycle", None)
    previous_view = getattr(lifecycle, "committed_view", None)
    scheduled_slow = set(previous_view.groups["slow"]) if previous_view is not None else set()
    promoted: List[str] = []
    while True:
        candidates: Dict[str, List[str]] = {}
        for read in _cross_group_reads(sim):
            if read["acknowledged"] or read["kind"] != "sampling":
                continue
            if read["consumer_group"] != "slow":
                continue
            producers = candidates.setdefault(read["consumer"], [])
            if read["producer"] not in producers:
                producers.append(read["producer"])
        if not candidates:
            break
        _require_contiguous_promotion(sim, sorted(set(candidates) & scheduled_slow))
        for name, producers in candidates.items():
            sim._groups["slow"] = [member for member in sim._groups["slow"] if member != name]
            sim._groups["medium"] = [*sim._groups["medium"], name]
            promoted.append(name)
            logger.info(
                "Promoted '%s' from the slow to the medium group "
                "(resolve_cross_group_deps=True): it reads outputs of faster producers %s. "
                "The move is part of the published provider view; the Functor runs "
                "only in the medium group.",
                name,
                producers,
            )
    if not promoted:
        return promoted

    bootstrap_groups = sim._bootstrap_groups
    if "slow" in bootstrap_groups:
        if any(
            getattr(sim.functors[name], "force_initial_execution", False) for name in promoted
        ):
            bootstrap_groups.add("medium")
        if not any(
            getattr(sim.functors[name], "force_initial_execution", False)
            for name in sim._groups["slow"]
        ):
            bootstrap_groups.discard("slow")
    sim._groups_dirty = True
    sim._compute_group_execution_orders()
    return promoted


def _warn_cross_group_dependencies(sim: Any) -> List[str]:
    """Govern declared cross-group reads when a provider view is published.

    References:
        architecture_overview.md 4.4.1 (multi-rate groups, cross-group lag);
        Gear and Wells (1984) BIT 24:484-502 (multirate methods: slow
        components sample fast ones at their own macro steps); Lawrence et al.
        (2019) CLM5 Technical Note, Sect. 2.7 (multi-rate stepping).
    Mathematical expression:
        rank(fast) = 0 < rank(medium) = 1 < rank(slow) = 2. For a declared read
        of variable x by consumer c from producer p with g(c) != g(p):
        sampling lag if rank(g(c)) > rank(g(p)) (c sees x only at its own
        group boundaries); hold lag if rank(g(c)) < rank(g(p)) (c holds the
        value x(t_last,p) committed when p last ran, a zero-order hold).
    Parameter meanings:
        g(f) is the execution group of Functor f; x is a declared input of c
        and a declared output of p; the read is acknowledged when p is listed
        in the ``lagged`` dependencies of c.
    Units: not applicable (ranks and names).
    Domain/constraints:
        1. With ``resolve_cross_group_deps=True`` every slow consumer with an
           unacknowledged sampling lag is moved to the medium group (fixed
           point), recorded in the published plan, and logged at INFO.
        2. Every unacknowledged read of the resulting plan is warned once,
           "Cross-group sampling lag" or "Cross-group hold lag".
        3. With ``strict_cross_group_dependencies=True`` any unacknowledged
           read raises RuntimeError, so the registration, retirement, or
           restore that would publish the plan is rolled back.
        No staleness bound is computed; nothing moves into the fast group.
    Sign convention: not applicable.

    Returns the names moved to the medium group by this publication.
    """
    strict = _strict_cross_group_setting(sim)
    promoted: List[str] = []
    if getattr(sim, "_resolve_cross_group_deps", False):
        promoted = _promote_slow_consumers(sim)

    unacknowledged: List[Dict[str, Any]] = []
    for read in _cross_group_reads(sim):
        if read["acknowledged"]:
            logger.debug(
                "Cross-group read acknowledged by a lagged declaration: '%s' (group=%s) "
                "reads '%s' from '%s' (group=%s)",
                read["consumer"], read["consumer_group"], read["variable"],
                read["producer"], read["producer_group"],
            )
            continue
        unacknowledged.append(read)
        if read["kind"] == "sampling":
            logger.warning(
                "Cross-group sampling lag: '%s' (group=%s) reads '%s' produced by '%s' "
                "(group=%s). The consumer samples the producer's state only when its "
                "own group executes. Declare '%s' in the lagged dependencies of '%s' "
                "to acknowledge the lag.",
                read["consumer"], read["consumer_group"], read["variable"],
                read["producer"], read["producer_group"],
                read["producer"], read["consumer"],
            )
        else:
            logger.warning(
                "Cross-group hold lag: '%s' (group=%s) reads '%s' produced by '%s' "
                "(group=%s). The consumer holds the value committed when the slower "
                "producer last ran. Declare '%s' in the lagged dependencies of '%s' "
                "to acknowledge the lag.",
                read["consumer"], read["consumer_group"], read["variable"],
                read["producer"], read["producer_group"],
                read["producer"], read["consumer"],
            )

    if strict and unacknowledged:
        summary = "; ".join(
            f"'{read['consumer']}' (group={read['consumer_group']}) reads "
            f"'{read['variable']}' from '{read['producer']}' "
            f"(group={read['producer_group']}, {read['kind']} lag)"
            for read in unacknowledged
        )
        raise RuntimeError(
            "Unacknowledged cross-group reads with strict_cross_group_dependencies=True: "
            f"{summary}. Declare each producer in the consumer's lagged dependencies "
            "or place both Functors in one execution group."
        )
    return promoted


def compute_splitting_sensitivity(
    simulator: Any,
    state: Dict[str, Any],
    dt: float,
    group_order: List[str],
    n_permutations: Optional[int] = None,
) -> Dict[str, Any]:
    """Quantify order sensitivity of operator splitting for a given functor order.

    Mathematical Expression:
        For each permutation sigma:
            result_sigma = execute(state, dt, sigma)
        sensitivity[var] = std(result_sigma[var]) / mean(|result_sigma[var]|)
        commutativity_violation = max sensitivity[var]

    References:
        - Strang (1968), SIAM J. Numer. Anal.
        - Sportisse (2000), J. Comput. Phys.
        - McLachlan and Quispel (2002), Acta Numerica.

    Parameters:
        simulator: RELICSimulator instance
        state: base state snapshot
        dt: step size [s]
        group_order: base functor order for one execution group
        n_permutations: number of permutations to sample; None reads
            computation.diagnostics.splitting_sensitivity.max_permutations
    """
    if not group_order:
        return {
            "per_variable_sensitivity": {},
            "commutativity_violation": 0.0,
            "permutations_tested": 0,
            "heatmap_data": [],
        }

    if n_permutations is None:
        n_permutations = int(
            get_global_manager().get(
                "computation.diagnostics.splitting_sensitivity.max_permutations", required=True
            )
        )
    unique_permutations = list(dict.fromkeys(itertools.permutations(group_order)))
    sampled_permutations = unique_permutations[: max(1, int(n_permutations))]

    original_state = _clone_state_snapshot(simulator._get_current_state())
    state_manager = getattr(simulator, "state_manager", None)
    manager_snapshot = (
        state_manager.checkpoint()
        if state_manager is not None and hasattr(state_manager, "checkpoint")
        else None
    )
    original_sources = _clone_runtime_value(
        getattr(simulator, "_step_update_sources", None)
    )
    original_increments = _clone_runtime_value(
        getattr(simulator, "_step_functor_increments", None)
    )
    original_provenance = _snapshot_runtime_value(
        "_step_provenance_detail",
        getattr(simulator, "_step_provenance_detail", None),
    )
    original_functor_state = _snapshot_functor_state_dicts(simulator)

    variable_values: Dict[str, List[float]] = {}
    heatmap_data: List[Dict[str, Any]] = []

    try:
        for permutation in sampled_permutations:
            if manager_snapshot is not None:
                assert state_manager is not None
                state_manager.restore(manager_snapshot)
            _restore_state_snapshot(simulator, _clone_state_snapshot(state))
            _restore_functor_state_dicts(simulator, original_functor_state)
            if (
                state_manager is not None
                and hasattr(state_manager, "begin_provenance_transaction")
                and not getattr(
                    state_manager,
                    "_provenance_transaction_active",
                    False,
                )
            ):
                state_manager.begin_provenance_transaction()
            if state_manager is not None and hasattr(
                state_manager,
                "set_auto_report_enabled",
            ):
                state_manager.set_auto_report_enabled(False)
            simulator._step_update_sources = {}
            simulator._step_functor_increments = {}
            simulator._step_provenance_detail = {}

            _ = simulator._execute_group(
                list(permutation),
                dt,
                start_time=float(simulator.current_time),
            )
            current_state = simulator._get_current_state()

            row: Dict[str, Any] = {"order": list(permutation)}
            for var_name, value in current_state.items():
                if not isinstance(value, Tensor):
                    continue
                if not (torch.is_floating_point(value) or torch.is_complex(value)):
                    continue
                magnitude = float(
                    value.detach().abs().to(dtype=torch.float64).mean().cpu().item()
                )
                variable_values.setdefault(var_name, []).append(magnitude)
                row[var_name] = magnitude
            heatmap_data.append(row)

    finally:
        if manager_snapshot is not None:
            assert state_manager is not None
            state_manager.restore(manager_snapshot)
        else:
            _restore_state_snapshot(simulator, original_state)
        _restore_functor_state_dicts(simulator, original_functor_state)
        simulator._step_update_sources = _clone_runtime_value(original_sources)
        simulator._step_functor_increments = _clone_runtime_value(original_increments)
        _restore_runtime_value(simulator, "_step_provenance_detail", original_provenance)

    per_variable_sensitivity: Dict[str, float] = {}
    for var_name, values in variable_values.items():
        if not values:
            continue
        values_tensor = torch.as_tensor(values, dtype=torch.float64)
        mean_abs = torch.mean(torch.abs(values_tensor))
        if float(mean_abs.item()) <= 0.0:
            sensitivity = 0.0
        else:
            sensitivity = float(torch.std(values_tensor, unbiased=False).item() / mean_abs.item())
        per_variable_sensitivity[var_name] = sensitivity

    commutativity_violation = (
        max(per_variable_sensitivity.values()) if per_variable_sensitivity else 0.0
    )
    return {
        "per_variable_sensitivity": per_variable_sensitivity,
        "commutativity_violation": float(commutativity_violation),
        "permutations_tested": len(sampled_permutations),
        "heatmap_data": heatmap_data,
    }
