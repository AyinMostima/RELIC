"""
Execution helpers for simulator functor groups.

This module extracts the heavy group-execution logic from ``main.py`` to keep
the simulator entry class focused on orchestration APIs.
"""

from __future__ import annotations

import inspect
import logging
import math
import time
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, Collection, Dict, List, Mapping, Optional, cast

import torch
from torch import Tensor

from ..abstractions import FluxDict
from ..dependency_graph import normalize_dependency_declaration
from ..state.manager import StateManager, UpdateItem, UpdateMode
from ..state.conservation import _validate_boundary_flux_update
from ..state.types import StateWriteMode, partition_outputs
from .registration import _current_provider_view
from .output_overlap_governance import (
    OVERLAP_POLICY_MUTUALLY_EXCLUSIVE,
    get_output_overlap_policy,
)

logger = logging.getLogger(__name__)


@dataclass
class ReplaceRateMean:
    """S2 accumulator of one REPLACE per-second rate on an SCM flux contract.

    writer: the Functor that committed the rate (one REPLACE writer per
        variable and step; a second writer is refused at the overlap check).
    item: the committed UpdateItem of the latest contribution with the value
        sum_k (h_k / H) q_k of the contributions so far.
    coverage: sum_k h_k / H of the contributions [1].
    contributions: the number of calls of the writer that committed the rate.
    """

    writer: str
    item: UpdateItem
    coverage: float
    contributions: int


@dataclass
class GroupComposition:
    """State owned by one group composition and shared by its sub-executions.

    A composition is the fast-group attempt of a step (the ``2n - 1`` Strang
    sub-executions, or the one Lie group call) or one medium or slow group
    call. It is created empty for each attempt and dropped when the
    composition returns, so a rejected adaptive attempt leaves nothing behind.

    interval: length H [s] of the interval the composition accounts for: the
        step interval of the fast-group composition, the dt of a group call
        that is a composition of its own. A sub-execution over h merges its
        per-second rates with the coverage h / H (``merge_sequential_updates``).
    dependency_outputs: provider name -> the merged updates of the provider's
        most recent earlier sub-execution (the ``subprocess_outputs`` source).
    diagnostics: Functor name -> the Functor's diagnostics merged by
        ``merge_sequential_updates`` over its sub-executions so far, with the
        UpdateItem modes kept. After each sub-execution StateManager stores
        this merged payload for the Functor, so the stored value follows the
        same rule as the value ``step()`` returns.
    boundary_rates: complete-step contributions merged with coverage h / H.
        This partial sum is private until the composition completes; it is not
        the current stage rate. Final publication restates the complete mean
        through StateManager without a second boundary-budget entry.
    stage_boundary_rates: writer -> its latest emitted boundary-rate payloads
        in this composition, after the existing distributed reduction. The
        visible stage rate sums these writer contributions; a later emission
        replaces only that writer's contribution. Old step averages are not
        carried into this map. Within a provider's substeps its latest rate is
        visible; after the provider call its own interval mean is visible and
        is also retained by dependency_outputs. An absent emission leaves that
        writer's latest observation held; boundary_rates counts only emitted
        contributions over their declared intervals.
    replace_rate_means: variable -> S2 accumulator (ReplaceRateMean) of a
        REPLACE per-second rate on an SCM flux contract: the rates its writer
        committed in this composition, weighted with their coverage h_k / H
        (``_store_composition_boundary_rates``). Each commit keeps its own
        rate (its ledger uses its own interval) and later sub-executions see
        that latest stage rate; ``finalize_boundary_rates`` publishes the step
        mean once when the writer ran more than once (the Strang members
        before the last). A writer that ran once keeps its committed value.
    member_calls: Functor name -> the coverages h_k / H of its calls in this
        composition, in time order (two for a Strang member before the last,
        one otherwise).
    stage_count: the number of group calls (sub-executions) of this
        composition so far, i.e. the stage index k of the next call.
    start_time: start [s] of the first group call of this composition.
    step: the 0-based index of the step of the first group call (the
        ``step_count`` of the simulator then; None without a step counter),
        for the commit context of the finalization commit.
    """

    interval: float
    dependency_outputs: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    boundary_rates: Dict[str, Any] = field(default_factory=dict)
    stage_boundary_rates: Dict[str, Dict[str, UpdateItem]] = field(default_factory=dict)
    boundary_rates_need_finalization: bool = False
    replace_rate_means: Dict[str, ReplaceRateMean] = field(default_factory=dict)
    member_calls: Dict[str, List[float]] = field(default_factory=dict)
    stage_count: int = 0
    start_time: Optional[float] = None
    step: Optional[int] = None

    def boundary_stage_view(self) -> Dict[str, Any]:
        """Derive the aggregate of current writer contributions using the one merge rule."""
        # References: RELIC aggregate boundary contract; Strang (1968).
        # Mathematical expression: q_stage = sum_f q_latest,f (coverage 1).
        # Parameter meanings: f is a writer that emitted a rate in this composition.
        # Units: the declared boundary-rate unit, with identical metadata enforced by merge.
        # Domain/constraints: contributions were reduced once; no old-step average is added.
        # Sign convention: each contribution keeps its contract's boundary direction.
        rates: Dict[str, Any] = {}
        for contributions in self.stage_boundary_rates.values():
            merge_sequential_updates(rates, contributions, coverage=1.0)
        return rates

    def replace_rate_step_means(self) -> Dict[str, UpdateItem]:
        """The S2 step means to publish; refuse a rate missing from a call of its writer."""
        # References: Strang (1968) SIAM J. Numer. Anal. 5:506-517; Hairer, Lubich
        #   and Wanner (2006), Sect. II.5; design 20261019_cr_a04_design_01 (D2, S2).
        # Mathematical expression: M = sum_k (h_k / H) q_k over the m calls of the
        #   writer, published when m >= 2 and sum_k h_k / H = 1 (M = 0.5 q1 + 0.5 q2
        #   for the two half-steps of a Strang member, so M H = q1 h/2 + q2 h/2, the
        #   amount released; 0.5 q + 0.5 q = q exactly in binary64).
        # Parameter meanings: q_k the rate committed by call k over h_k; H the
        #   composition interval; m the number of calls of the writer.
        # Units: M and q_k in the contract's rate unit; h_k, H [s].
        # Domain/constraints: every call of the writer must have committed the rate
        #   (a REPLACE value missing from a call has no defined interval mean), else
        #   RuntimeError before any write; a writer with one call keeps its value.
        # Sign convention: rates keep the contract's boundary direction.
        means: Dict[str, UpdateItem] = {}
        for var_name, mean in self.replace_rate_means.items():
            calls = len(self.member_calls.get(mean.writer, ()))
            if mean.contributions != calls:
                raise RuntimeError(
                    f"S2: the REPLACE flux rate '{var_name}' of '{mean.writer}' was committed in "
                    f"{mean.contributions} of its {calls} calls of this composition, so its step "
                    "mean over the composition interval is undefined; emit it in every call, or "
                    "write it as an ACCUMULATE per-second rate"
                )
            if calls < 2:
                continue
            if not math.isclose(mean.coverage, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError(
                    f"S2: the calls of '{mean.writer}' cover {mean.coverage!r} of the composition "
                    f"interval, not 1, so '{var_name}' has no step mean"
                )
            means[var_name] = mean.item
        return means

    def finalize_boundary_rates(self, state_manager: StateManager) -> None:
        """Publish the complete interval means once, after successful composition execution.

        One owner commit carries the boundary-rate means (ACCUMULATE per-second
        aggregates) and the S2 step means of REPLACE flux rates.
        """
        # References: Strang (1968); StateManager boundary-rate commit contract.
        # Mathematical expression: B = sum_k (h_k/H) q_k; no additional amount B*H is advanced.
        # Parameter meanings: H is this composition interval; h_k is one committed interval.
        # Units: B retains rate units; H and h_k are seconds.
        # Domain/constraints: called only at successful completion, before the adaptive metric;
        #   every call already committed its own rate and ledger, so the publication
        #   restates means without a conservation check (no second budget entry).
        # Sign convention: rates retain the original outward/inward convention.
        publication = self.replace_rate_step_means()
        if self.boundary_rates_need_finalization:
            shared = sorted(set(publication) & set(self.boundary_rates))
            if shared:
                raise RuntimeError(
                    f"S2: variables {shared} are both boundary-rate aggregates and REPLACE rates"
                )
            publication = {**self.boundary_rates, **publication}
        if publication:
            state_manager.apply_updates(
                publication, self.interval,
                check_conservation=False, rollback_on_failure=False,
                commit_context=self.finalization_commit_context(),
            )
        self.boundary_rates_need_finalization = False
        self.replace_rate_means = {}

    def finalization_commit_context(self) -> Optional[Dict[str, Any]]:
        """Commit context of the owner commit that publishes the composition means.

        References: W3C PROV-DM (2013); design 20261019_cr_a04_design_01,
            section 2 (commit_context; the finalization commit has its own role).
        Mathematical expression: the commit restates the means over the whole
            composition [t0, t0 + H]: role composition_mean, stage m (the commit
            after the m group calls k = 0..m-1), interval_start t0, dt H.
        Parameter meanings: t0 = start_time, H = interval, m = stage_count,
            step the step index of the first call.
        Units: t0, H [s]; m dimensionless.
        Domain/constraints: None while no group call has started the
            composition (no call, nothing to publish).
        Sign convention: not applicable.
        """
        if self.start_time is None:
            return None
        return build_commit_context(
            step=self.step,
            composition_start=self.start_time,
            composition_interval=self.interval,
            stage=self.stage_count,
            role=COMPOSITION_MEAN_ROLE,
            interval_start=self.start_time,
            dt=self.interval,
        )

    def __post_init__(self) -> None:
        if isinstance(self.interval, bool) or not isinstance(self.interval, Real):
            raise TypeError(
                f"GroupComposition interval must be a real scalar in seconds, "
                f"got {type(self.interval).__name__}"
            )
        self.interval = float(self.interval)
        if not math.isfinite(self.interval) or self.interval <= 0.0:
            raise ValueError(
                f"GroupComposition interval must be finite and positive, got {self.interval}"
            )


def execute_group(
    self: Any,
    group_order: List[str],
    dt: float,
    *,
    start_time: float,
    composition: Optional[GroupComposition] = None,
) -> FluxDict:
    """
    Execute functors in a specific group following topological order.

    Parameters
    ----------
    self : Any
        RELICSimulator instance.
    group_order : List[str]
        Execution order for the group.
    dt : float
        Length of the interval this group call accounts for [s].
    start_time : float
        Start of that interval [s]. The pair ``(start_time, dt)`` is the real
        interval ``[start_time, start_time + dt]``: the step start for a fast
        call, ``t + h/2`` for a second Strang half-step, and the last execution
        time of the group for a medium or slow call. There is no default.
    composition : GroupComposition, optional
        The group composition this call belongs to: its dependency-output map
        (``subprocess_outputs`` source) and its merged per-Functor
        diagnostics. The fast-group composition owner passes one composition
        to all of its sub-executions; without it the call is a composition of
        its own and starts empty.

    Returns
    -------
    FluxDict
        Combined updates for the execution group.
    """
    return _execute_group_impl(self, group_order, dt, start_time, composition)


def _get_or_init_step_update_sources(self: Any) -> Dict[str, Any]:
    step_update_sources = getattr(self, "_step_update_sources", None)
    if step_update_sources is None:
        step_update_sources = {}
        self._step_update_sources = step_update_sources
    return step_update_sources


def _align_state_tensors_to_device(
    state: Dict[str, Any],
    target_device: Optional[torch.device],
) -> Dict[str, Any]:
    """Return a shallow-copied state with tensor entries aligned to target device."""
    if target_device is None:
        return state
    aligned = dict(state)
    for key, value in aligned.items():
        if isinstance(value, Tensor) and value.device != target_device:
            aligned[key] = value.to(device=target_device)
    return aligned


def _resolve_functor_device(
    simulator: Any,
    functor: Any,
    state: Dict[str, Any],
) -> Optional[torch.device]:
    """Resolve execution device for a functor call."""
    functor_device = getattr(functor, "device", None)
    if functor_device is not None:
        return torch.device(functor_device)

    simulator_device = getattr(simulator, "device", None)
    if simulator_device is not None:
        return torch.device(simulator_device)

    for value in state.values():
        if isinstance(value, Tensor):
            return value.device
    return None


def _build_active_group_scale_state(
    simulator: Any,
    current_state: Dict[str, Any],
    group_order: List[str],
) -> Dict[str, Any]:
    """Select retained outputs owned by the currently executing group."""
    active_functors = [
        simulator.functors[name]
        for name in group_order
        if name in simulator.functors
    ]
    if not active_functors:
        return current_state

    active_outputs = {
        output_name
        for functor in active_functors
        for output_name in (
            getattr(getattr(functor, "metadata", None), "output_variables", [])
            or []
        )
    }
    return {
        name: value
        for name, value in current_state.items()
        if name in active_outputs
    }


def _is_numeric_like(value: Any) -> bool:
    return isinstance(value, (Tensor, float, int)) or hasattr(value, "dtype")


def _combine_update_values(lhs: Any, rhs: Any) -> Any:
    """Combine update payloads with dtype/device alignment."""
    if isinstance(lhs, Tensor) or isinstance(rhs, Tensor):
        lhs_tensor = lhs if isinstance(lhs, Tensor) else torch.as_tensor(lhs)
        rhs_tensor = rhs if isinstance(rhs, Tensor) else torch.as_tensor(
            rhs, device=lhs_tensor.device, dtype=lhs_tensor.dtype
        )
        return lhs_tensor + rhs_tensor.to(
            device=lhs_tensor.device,
            dtype=lhs_tensor.dtype,
        )
    return float(lhs) + float(rhs)


def _diff_update_values(new_value: Any, base_value: Any) -> Any:
    """Compute delta for replace-mode updates."""
    if base_value is None:
        return new_value
    if isinstance(new_value, Tensor) or isinstance(base_value, Tensor):
        new_tensor = new_value if isinstance(new_value, Tensor) else torch.as_tensor(new_value)
        base_tensor = (
            base_value
            if isinstance(base_value, Tensor)
            else torch.as_tensor(
                base_value,
                device=new_tensor.device,
                dtype=new_tensor.dtype,
            )
        )
        return new_tensor - base_tensor
    return float(new_value) - float(base_value)


_UPDATE_COMPATIBILITY_FIELDS = (
    "mode",
    "write_mode",
    "unit",
    "basis",
    "per_timestep",
    "element",
)


def _merge_update_items(var_name: str, lhs: UpdateItem, rhs: UpdateItem) -> UpdateItem:
    """Merge compatible update items without discarding semantic metadata."""
    mismatches = [
        field_name
        for field_name in _UPDATE_COMPATIBILITY_FIELDS
        if getattr(lhs, field_name) != getattr(rhs, field_name)
    ]
    if mismatches:
        details = ", ".join(
            f"{field_name}={getattr(lhs, field_name)!r}/{getattr(rhs, field_name)!r}"
            for field_name in mismatches
        )
        raise ValueError(
            f"Incompatible UpdateItem metadata for '{var_name}': {details}"
        )

    source_functor = (
        lhs.source_functor
        if lhs.source_functor == rhs.source_functor
        else None
    )
    if lhs.mode == UpdateMode.REPLACE:
        return replace(rhs, source_functor=source_functor)
    return replace(
        lhs,
        value=_combine_update_values(lhs.value, rhs.value),
        source_functor=source_functor,
    )


def _scale_update_value(value: Any, dt: float) -> Any:
    """Scale a payload by one scalar factor (a timestep or a coverage fraction)."""
    if isinstance(value, Tensor):
        dt_tensor = torch.as_tensor(dt, device=value.device, dtype=value.dtype)
        return value * dt_tensor
    return float(value) * float(dt)


def _lower_flux_update(item: UpdateItem, dt: float) -> UpdateItem:
    """Convert a FLUX item to one integrated ACCUMULATE item exactly once."""
    if item.mode != UpdateMode.FLUX:
        return item
    integrated_value = (
        item.value
        if item.per_timestep
        else _scale_update_value(item.value, dt)
    )
    return replace(
        item,
        value=integrated_value,
        mode=UpdateMode.ACCUMULATE,
        basis="per_step",
        per_timestep=True,
    )


# Declared time bases of an unintegrated ACCUMULATE rate: "per_second" (integrated
# by StateManager over the commit interval) and "instantaneous" (a declared
# boundary-rate aggregate that StateManager keeps as a rate).
_PER_SECOND_BASES = frozenset({"per_second", "instantaneous"})


def _is_per_second_rate(update: Any) -> bool:
    """True for an ACCUMULATE UpdateItem whose payload is a per-second rate.

    The item's own time basis decides: ``per_timestep=False`` with a
    per-second basis (``_PER_SECOND_BASES``). A per_timestep item, a per-step
    basis, and a plain payload are amounts; FLUX items are lowered to amounts
    before they are merged (``_lower_flux_update``).
    """
    return (
        isinstance(update, UpdateItem)
        and update.mode == UpdateMode.ACCUMULATE
        and not update.per_timestep
        and str(update.basis or "per_step").strip().lower() in _PER_SECOND_BASES
    )


def _is_boundary_rate_target(self: Any, var_name: Any) -> bool:
    """True when StateManager keeps an unintegrated rate for ``var_name`` as is.

    StateManager commits an unintegrated ACCUMULATE rate for a ContractRegistry
    variable of kind ``flux`` as a boundary-rate aggregate (the stored value is
    the rate); a ``delta_`` name is a tendency of its target and is integrated.
    """
    if not isinstance(var_name, str) or var_name.startswith("delta_"):
        return False
    contract_registry = (
        getattr(self.state_manager, "_contract_registry", None)
        if self.state_manager is not None
        else None
    )
    if contract_registry is None:
        return False
    contract = contract_registry.get_contract(var_name)
    return contract is not None and str(contract.kind).lower() == "flux"


def _is_replace_boundary_rate(
    self: Any, var_name: Any, update: Any, *, interval: Optional[float] = None,
) -> bool:
    """Classify S2 rates; an actual child interval additionally requires a physical port.

    No interval preserves the historical S2 publication selector. A child
    interval invokes the existing StateManager boundary contract: the writer
    has an allowed input/output route, exact rate units, and a positive dt.
    """
    candidate = (
        isinstance(update, UpdateItem)
        and update.mode == UpdateMode.REPLACE
        and not update.per_timestep
        and update.write_mode != StateWriteMode.DERIVED
        and _is_boundary_rate_target(self, var_name)
    )
    if not candidate or interval is None:
        return candidate
    registry = self.state_manager._contract_registry
    contract = registry.get_contract(var_name)
    whitelist = registry.get_flux_whitelist(var_name)
    source = str(update.source_functor or "").strip()
    if source not in set(whitelist["flux_in"]) | set(whitelist["flux_out"]):
        return False
    # References: StateManager conservation._validate_boundary_flux_update;
    #   LeVeque (2002), finite-volume boundary balances.
    # Mathematical expression: emitted amount = physical rate * actual child dt.
    # Parameter meanings: the existing SCM writer route decides physical direction;
    #   interval is the actual child duration, not a default temporal support.
    # Units: contract rate units multiplied by seconds yield the physical amount.
    # Domain/constraints: reuse the canonical unit/direction/interval validator;
    #   an explicit instantaneous or integral support cannot become a mean rate.
    # Sign convention: retain the existing declared input/output direction.
    _validate_boundary_flux_update(self.state_manager, var_name, update, contract, interval)
    support = contract.raw.get("support")
    if support is not None and support.get("temporal_basis") in {"instantaneous", "interval_integral"}:
        raise ValueError(
            f"Physical boundary rate '{var_name}' has conflicting temporal support "
            f"{support['temporal_basis']!r} for substep interval aggregation"
        )
    return True


def _validate_merge_coverage(coverage: Any) -> float:
    """Return the coverage h / H as a float; raise unless it is a fraction in (0, 1]."""
    if isinstance(coverage, bool) or not isinstance(coverage, Real):
        raise TypeError(f"Merge coverage must be a real fraction, got {type(coverage).__name__}")
    fraction = float(coverage)
    if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"Merge coverage must be a finite fraction in (0, 1], got {coverage!r}")
    return fraction


def _weight_per_second_rate(update: Any, coverage: float) -> Any:
    """Scale a per-second rate by its coverage h / H; amounts and others pass unchanged."""
    if coverage == 1.0 or not _is_per_second_rate(update):
        return update
    return replace(update, value=_scale_update_value(update.value, coverage))


def replaces_earlier_output(var_name: str, later: Any, diagnostic_names: Collection[str]) -> bool:
    """The merge decision shared by every merge site (see merge_sequential_updates).

    True when a later output replaces the earlier one: a plain diagnostic (an
    observation) or a REPLACE UpdateItem. False when it adds to it: an
    ACCUMULATE UpdateItem or a plain state payload (an increment).
    ``diagnostic_names`` are the diagnostics partition_outputs classified.
    """
    if isinstance(later, UpdateItem):
        return later.mode == UpdateMode.REPLACE
    return var_name in diagnostic_names


def merge_sequential_updates(
    accumulated: Dict[str, Any],
    later_updates: Dict[str, Any],
    *,
    coverage: float,
) -> Dict[str, Any]:
    """Merge the outputs of a later execution onto the earlier ones (the single merge rule).

    Used for the substeps of one Functor call, for the diagnostics of the
    Functors of one group call, and for the sub-executions of one fast-group
    composition (Strang); the step return folds group results with the same
    decision (``replaces_earlier_output``).

    ``coverage`` is h / H, the length of the interval the later execution
    accounts for over the length of the interval the merged outputs account
    for: 1 / n for each of n substeps of one call, h_k / h for a Strang
    sub-execution of a step of length h, and 1 for the Functors of one group
    call (they cover the same interval). It weights per-second rates only.
    """
    # References: RELIC UpdateItem contract (relic/core/state/types.py:
    #   partition_outputs is the sole channel classifier; UpdateMode semantics;
    #   per_timestep=False is an unintegrated rate); StateManager.apply_updates
    #   (an ACCUMULATE rate with a per-second basis is integrated once over the
    #   commit interval, a boundary-rate aggregate is kept as a rate); LeVeque
    #   (2002), finite-volume time integration; architecture_overview.md
    #   Section 4.4.1 (fast-group composition).
    # Mathematical expression: for successive outputs o_1, ..., o_m of one name,
    #   o_j accounting for an interval of length h_j within the merged interval
    #   of length H (coverage w_j = h_j / H),
    #   merged = o_m                    for a plain diagnostic (observation) or a
    #                                   REPLACE UpdateItem;
    #   merged = sum_j o_j              for an amount: an ACCUMULATE UpdateItem
    #                                   with per_timestep=True or a per-step basis
    #                                   (FLUX items arrive lowered to such amounts)
    #                                   or a plain state payload (an increment);
    #   merged = sum_j w_j q_j          for a per-second rate q_j (an ACCUMULATE
    #                                   UpdateItem with per_timestep=False and a
    #                                   per-second basis), so merged * H =
    #                                   sum_j q_j h_j: the time-weighted mean over
    #                                   substeps that partition H (sum_j w_j = 1),
    #                                   the sum over Functors of one group call
    #                                   (w_j = 1 each), and the sum of the Functors'
    #                                   time-weighted means over a Strang step;
    #   dict payloads merge key by key (later keys win).
    # Parameter meanings: o_j is the output of the j-th execution in time order
    #   (substep, Functor, or sub-execution); q_j a per-second rate; h_j the length
    #   of the interval of execution j; H the length of the merged interval; w_j the
    #   coverage; a diagnostic is a reserved diagnostics.* / __* name or a DERIVED
    #   UpdateItem.
    # Units: h_j, H [s]; w_j [1]; merged values keep the units of the outputs (an
    #   amount its amount unit, a rate its per-second unit); UpdateItems of one name
    #   must agree on mode, write mode, unit, basis, per_timestep and element.
    # Domain/constraints: 0 < w_j <= 1 (TypeError or ValueError otherwise); an
    #   output absent from an execution contributes nothing over its interval;
    #   incompatible UpdateItem metadata raises ValueError; external intents must be
    #   collected before merging. A diagnostic whose whole-interval total is wanted
    #   must be returned as an ACCUMULATE UpdateItem; a plain diagnostic is the last
    #   observation.
    # Sign convention: increments and rates keep their sign; positive increases the
    #   target.
    coverage = _validate_merge_coverage(coverage)
    _, diagnostics, intents = partition_outputs(later_updates)
    if intents:
        raise ValueError("External intents must be collected before numeric substep merging")
    for var_name, sub_value in later_updates.items():
        if var_name not in accumulated:
            accumulated[var_name] = _weight_per_second_rate(sub_value, coverage)
            continue

        current_value = accumulated[var_name]
        if not isinstance(sub_value, UpdateItem) and replaces_earlier_output(
            var_name, sub_value, diagnostics
        ):
            # Plain diagnostics are observations, not implicit accumulators.
            accumulated[var_name] = sub_value
            continue
        if isinstance(current_value, UpdateItem) or isinstance(sub_value, UpdateItem):
            lhs = (
                current_value
                if isinstance(current_value, UpdateItem)
                else replace(sub_value, value=current_value)
            )
            rhs = (
                sub_value
                if isinstance(sub_value, UpdateItem)
                else replace(lhs, value=sub_value)
            )
            accumulated[var_name] = _merge_update_items(
                var_name, lhs, _weight_per_second_rate(rhs, coverage)
            )
            continue

        if isinstance(current_value, dict) and isinstance(sub_value, dict):
            merged_payload = dict(current_value)
            merged_payload.update(sub_value)
            accumulated[var_name] = merged_payload
            continue

        if _is_numeric_like(current_value) and _is_numeric_like(sub_value):
            accumulated[var_name] = _combine_update_values(current_value, sub_value)
        else:
            accumulated[var_name] = sub_value

    return accumulated


def _prepare_group_state(self: Any, group_order: List[str]) -> Dict[str, Any]:
    current_state = self._get_current_state()
    # Multirate exchange overlay (CR-A-04 step 6; owned by the execution loop).
    # References: Gear and Wells (1984) BIT 24(4):484-502 (multirate methods);
    #   design 20261019_cr_a04_design_01, section 3 (exchange interval_mean).
    # Mathematical expression: x_in[v] = m_e for every variable v of the overlay O
    #   of this group call; x_in[u] = x[u] for every other u, with x the committed
    #   state (StateManager).
    # Parameter meanings: O = simulator._active_exchange_overlay, set by
    #   execution_loop only around the call of an exchange's consumer group
    #   (variable -> its interval mean m_e, already in the dtype and device of v);
    #   x_in the input state of every Functor of this group call.
    # Units: m_e in the units of v.
    # Domain/constraints: applied before device alignment and regridding (before
    #   any input projection); never committed, since a variable written by its
    #   consumer group is refused at publication; an overlaid variable that is not
    #   part of the committed state raises RuntimeError.
    # Sign convention: v keeps its own sign convention.
    overlay = getattr(self, "_active_exchange_overlay", None)
    if overlay:
        missing = sorted(name for name in overlay if name not in current_state)
        if missing:
            raise RuntimeError(
                f"Multirate exchange overlay names variables that are not part of the committed "
                f"state: {missing}"
            )
        current_state = dict(current_state)
        current_state.update(overlay)
    target_device = getattr(self, "device", None)
    if target_device is not None:
        current_state = _align_state_tensors_to_device(
            current_state,
            torch.device(target_device),
        )
    if not (self.enable_auto_scale or self.strict_scale_detection):
        return current_state

    scale_state = _build_active_group_scale_state(
        self,
        current_state,
        group_order,
    )
    scale_check = self._check_scale_consistency(
        scale_state,
        force=self.strict_scale_detection and not self.enable_auto_scale,
    )
    if not scale_check.get("needs_regrid", False):
        return current_state

    if not self.enable_auto_scale:
        detected = scale_check.get("detected_scales", {})
        reason = scale_check.get("reason", "scale mismatch detected")
        raise ValueError(
            f"{reason} detected while enable_auto_scale=False. "
            "Detected scales: "
            f"{detected}. Provide consistent scales or enable_auto_scale."
        )

    logger.debug(
        "Performing regridding before %s group execution",
        group_order[0] if group_order else "unknown",
    )
    return self._perform_regridding(current_state, scale_check)


def _build_integration_kwargs(self: Any, functor: Any) -> Dict[str, Any]:
    integration_kwargs: Dict[str, Any] = {}
    forward_callable = getattr(functor, "forward", None)
    if not callable(forward_callable):
        return integration_kwargs
    try:
        parameters = inspect.signature(forward_callable).parameters
    except (TypeError, ValueError):
        # A callable without an introspectable signature receives no optional
        # keyword controls; declaration errors below are not masked.
        return integration_kwargs
    if "calendar_type" in parameters:
        integration_kwargs["calendar_type"] = self._calendar_type
    if "time_origin" in parameters:
        integration_kwargs["time_origin"] = self._time_origin
    # The simulator resolved these controls from its settings or the
    # transport.numerical.* parameter database entries at construction.
    if "time_integration_method" in parameters:
        integration_kwargs["time_integration_method"] = self._time_integration_method
    if "implicit_midpoint_max_iter" in parameters:
        integration_kwargs["implicit_midpoint_max_iter"] = self._implicit_midpoint_max_iter
    if "implicit_midpoint_tolerance" in parameters:
        integration_kwargs["implicit_midpoint_tolerance"] = self._implicit_midpoint_tolerance
    if "subprocess_outputs" in parameters:
        declared = normalize_dependency_declaration(
            functor.get_dependencies(), owner=functor.name
        )
        required = list(declared.required)
        optional = list(declared.optional)
        composition = getattr(self, "_active_composition", None)
        output_cache = {} if composition is None else composition.dependency_outputs
        missing = [name for name in required if name not in output_cache]
        if missing:
            raise RuntimeError(
                "Declared dependency outputs are unavailable for "
                f"{functor.name}: {', '.join(missing)}. subprocess_outputs carry the "
                "outputs of providers that ran earlier in the same group composition of "
                "this step; a provider in another execution group is read from the "
                "committed state instead."
            )
        integration_kwargs["subprocess_outputs"] = {
            name: output_cache[name]
            for name in required + optional
            if name in output_cache
        }
    return integration_kwargs


def _coerce_runtime_update_item(
    self: Any,
    var_name: str,
    update: Any,
    output_specs: Dict[str, Any],
) -> UpdateItem:
    """Resolve one update through the simulator's canonical mode policy."""
    if isinstance(update, UpdateItem):
        item = update
    elif isinstance(update, (tuple, dict)):
        item = StateManager._coerce_update_item(update)
        if item is None:
            raise TypeError(f"Cannot normalize state output '{var_name}'")
    else:
        if self.run_specs is not None:
            default_mode = UpdateMode.REPLACE
        else:
            default_mode = (
                UpdateMode.REPLACE
                if _is_rate_like(var_name, output_specs)
                else UpdateMode.ACCUMULATE
            )
        item = UpdateItem(update, default_mode)

    if self.state_manager is not None:
        write_mode = self.state_manager._resolve_write_mode(var_name, item)
        if item.write_mode != write_mode:
            item = replace(item, write_mode=write_mode)
    if self.run_specs is not None and item.write_mode != StateWriteMode.DERIVED:
        resolved_mode = self._resolve_update_mode(var_name, item.mode)
        if resolved_mode != item.mode:
            item = replace(item, mode=resolved_mode)
    return item


def _normalize_substep_result(
    self: Any,
    raw_result: Dict[str, Any],
    output_specs: Dict[str, Any],
    sub_dt: float,
    declared_outputs: Collection[str] = (),
) -> Dict[str, Any]:
    """Normalize numeric substep outputs while retaining diagnostic payloads."""
    normalized: Dict[str, Any] = {}
    state, diagnostics, intents = partition_outputs(raw_result, declared_outputs=declared_outputs)
    if intents:
        raise ValueError("External intents require substep collection")
    for var_name, value in state.items():
        item = _coerce_runtime_update_item(
            self=self, var_name=var_name, update=value, output_specs=output_specs,
        )
        normalized[var_name] = _lower_flux_update(item, sub_dt)
    for var_name, value in diagnostics.items():
        normalized[var_name] = _lower_flux_update(value, sub_dt) if isinstance(value, UpdateItem) else value
    resolved_state, _, _ = partition_outputs(normalized)
    declared = set(declared_outputs)
    undeclared = [
        name for name in resolved_state
        if declared and name not in declared
        and not (name.startswith("delta_") and name[len("delta_"):] in declared)
    ]
    if undeclared:
        raise ValueError(f"Undeclared scientific output(s): {', '.join(undeclared)}")
    return normalized


def _advance_substep_state(
    self: Any,
    execution_state: Dict[str, Any],
    normalized_result: Dict[str, Any],
    output_specs: Dict[str, Any],
    sub_dt: float,
) -> Dict[str, Any]:
    """Advance the local state seen by the next substep without global writes.

    ``sub_dt`` is the substep interval [s]; a state tendency advances its
    target by value times ``sub_dt``. A boundary rate instead uses the
    composition's current per-writer stage view without time integration.
    """
    state_keys = set(execution_state)
    schema_keys = (
        set(getattr(self.state_manager, "schema", {}).keys())
        if self.state_manager is not None
        else set()
    )
    state_candidates, _, _ = partition_outputs(normalized_result)
    state_updates = {
        var_name: value
        for var_name, value in state_candidates.items()
        if _is_state_target(var_name, state_keys, schema_keys)
        and (isinstance(value, UpdateItem) or _is_numeric_like(value))
    }
    if not state_updates:
        return execution_state
    provisional_state, _ = _build_provisional_state_and_converted_updates(
        self=self,
        current_state=execution_state,
        raw_updates=state_updates,
        output_specs=output_specs,
        dt=sub_dt,
    )
    return provisional_state


def resolve_call_substep_count(
    functor_name: str,
    functor: Any,
    override: Optional[Mapping[str, Any]],
    dt: float,
) -> int:
    """Sub-step count of one call of a Functor over dt (the one rule; pure).

    References: Gear and Wells (1984) BIT 24(4):484-502 (different step sizes
        for different components); architecture_overview.md 4.4.1
        (sub-stepping through FunctorMetadata.substep_count).
    Mathematical expression: n = count for an override {count: n};
        n = max(1, ceil(dt / s)) for an override {max_seconds: s}; otherwise
        n = FunctorMetadata.substep_count; each sub-step covers h = dt / n.
    Parameter meanings: dt the interval of the call (half the fast step for
        the first Strang half-steps); s the largest admissible sub-step; n the
        sub-step count; override the simulator setting substep_overrides of
        this Functor (computation.scheduling.substep_overrides), which the
        registration owner records in its provider binding.
    Units: dt, s, h [s]; n dimensionless.
    Domain/constraints: dt > 0 (checked by the group owner); an override is
        validated when the simulator is built or restored (count integer >= 1,
        s finite > 0); h <= s up to binary64 rounding of dt / s; a declared
        substep_count changed after construction is checked again here.
    Sign convention: none (positive magnitudes).
    """
    if override is None:
        n_substeps = functor.metadata.substep_count
        if isinstance(n_substeps, bool) or not isinstance(n_substeps, int) or n_substeps < 1:
            raise ValueError(
                f"Functor '{functor_name}' metadata.substep_count must be an integer >= 1, "
                f"got {n_substeps!r}"
            )
        return n_substeps
    if "count" in override:
        return int(override["count"])
    return max(1, math.ceil(float(dt) / float(override["max_seconds"])))


def _run_functor_substeps(
    self: Any,
    functor_name: str,
    functor: Any,
    current_state: Dict[str, Any],
    dt: float,
    start_time: float,
    external_intents: List[Dict[str, Any]],
) -> Any:
    # The sub-step count of this call: the per-run override of the simulator
    # (substep_overrides), else the validated FunctorMetadata.substep_count.
    n_substeps = resolve_call_substep_count(
        functor_name, functor, getattr(self, "_substep_overrides", {}).get(functor_name), dt
    )
    sub_dt = dt / float(n_substeps)
    functor_device = _resolve_functor_device(self, functor, current_state)
    execution_state = _align_state_tensors_to_device(current_state, functor_device)
    output_specs = _safe_get_output_specs(functor)

    raw_updates: Any = {}
    physical_replace_means: Dict[str, Any] = {}
    physical_replace_counts: Dict[str, int] = {}
    for substep_index in range(n_substeps):
        # References: Higham (2002), floating-point representation and rounding;
        #   Hairer, Lubich and Wanner (2006), Sect. II.5 (stage times of a
        #   composition); architecture_overview.md 4.4.1 (time argument contract).
        # Mathematical expression: h = dt / n_substeps; sub-step k covers
        #   [t_s + k h, t_s + (k + 1) h], so it receives time = t_s + k h and dt = h;
        #   sum over k of h advances the whole interval [t_s, t_s + dt].
        # Parameter meanings: t_s = start_time, the start of the interval this group
        #   call accounts for; dt its length; h one subinterval; k the sub-step index.
        # Units: t_s, dt, h, time [s]; n_substeps and k [1].
        # Domain/constraints: dt > 0 and t_s finite (checked by the group owner);
        #   retain the binary64 scheduler value when forming the scalar Tensor;
        #   scientific kernels own their state precision.
        # Sign convention: positive h advances the same clock as the scheduler.
        if functor_device is not None:
            dt_tensor = torch.tensor(sub_dt, device=functor_device, dtype=torch.float64)
        else:
            dt_tensor = torch.tensor(sub_dt, dtype=torch.float64)
        integration_kwargs = _build_integration_kwargs(self, functor)
        substep_time = start_time + substep_index * sub_dt
        if integration_kwargs:
            sub_result = functor(
                execution_state,
                substep_time,
                dt_tensor,
                validation_report=self._validation_findings,
                **integration_kwargs,
            )
        else:
            sub_result = functor(
                execution_state,
                substep_time,
                dt_tensor,
                validation_report=self._validation_findings,
            )
        if not isinstance(sub_result, dict):
            raw_updates = sub_result
            break
        declared_outputs = functor.metadata.output_variables or ()
        _, _, intents = partition_outputs(sub_result, declared_outputs=declared_outputs)
        scientific_result = (
            {key: value for key, value in sub_result.items() if key not in intents}
            if intents else sub_result
        )
        normalized_result = _normalize_substep_result(
            self=self,
            raw_result=scientific_result,
            output_specs=output_specs,
            sub_dt=sub_dt,
            declared_outputs=declared_outputs,
        )
        state_candidates, _, _ = partition_outputs(normalized_result)
        boundary_updates: Dict[str, UpdateItem] = {}
        for var_name, item in state_candidates.items():
            if not (_is_per_second_rate(item) and _is_boundary_rate_target(self, var_name)):
                continue
            reference = execution_state.get(var_name)
            value = item.value
            if isinstance(reference, Tensor):
                value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
            elif not isinstance(value, Tensor):
                value = torch.as_tensor(value, device=functor_device, dtype=torch.float32)
            boundary_updates[var_name] = replace(item, value=value.clone())
        if boundary_updates:
            # References: PyTorch AllReduce mean; RELIC stage and commit rate contracts.
            # Mathematical expression: q_bar,k = sum_r q_r,k/N before q_stage or sum h_k*q_bar,k/H.
            # Parameter meanings: r rank, N world size, k substep; each writer is reduced once.
            # Units: original boundary-rate units; no time integration occurs here.
            # Domain/constraints: use the existing reducer; group commit excludes these rates
            #   from a second reduction. Clone payloads so the reducer cannot mutate input state.
            # Sign convention: reduction preserves the declared boundary direction.
            reduce_started = time.perf_counter()
            self._reduce_updates_across_ranks(boundary_updates)
            timings = getattr(self, "_active_step_stage_timings", None)
            if timings is not None:
                timings["distributed_collective"] += max(0.0, time.perf_counter() - reduce_started)
            normalized_result.update(boundary_updates)
            composition = self._active_composition
            composition.stage_boundary_rates.setdefault(functor_name, {}).update(boundary_updates)
        external_intents.extend(
            {"functor": functor_name, "substep": substep_index, "time": float(substep_time),
             "output": name, "intent": intent}
            for name, intent in intents.items()
        )
        execution_state = _advance_substep_state(
            self=self,
            execution_state=execution_state,
            normalized_result=normalized_result,
            output_specs=output_specs,
            sub_dt=sub_dt,
        )
        # Each substep accounts for h = dt / n_substeps of the call interval dt,
        # so its per-second rates merge with the coverage h / dt = 1 / n_substeps
        # (the time-weighted mean); amounts add (merge_sequential_updates).
        raw_updates = merge_sequential_updates(
            raw_updates, normalized_result, coverage=1.0 / float(n_substeps)
        )
        if n_substeps > 1:
            # References: LeVeque (2002); StateManager boundary-ledger contract;
            #   20261020 P-N qualification, actual N2 child integrals (step-5 revision).
            # Mathematical expression: q_call = sum_k q_k * dt_k / dt_call.
            # Parameter meanings: q_k is an emitted physical port rate; dt_k is
            #   this actual child duration; dt_call is this one governed call.
            # Units: q_call retains the contract rate unit; dt_k and dt_call [s].
            # Domain/constraints: no state advance or second physical transfer;
            #   ordinary REPLACE states/observations and ACCUMULATE remain unchanged.
            #   The existing commit reducer still reduces this mean exactly once.
            # Sign convention: preserve the actual physical port direction.
            for name, item in state_candidates.items():
                if not _is_replace_boundary_rate(self, name, item, interval=sub_dt):
                    continue
                # The physical port contract supplies the rate semantics even
                # when the original REPLACE item keeps its default per-step basis.
                merge_sequential_updates(
                    physical_replace_means,
                    {name: replace(
                        item, mode=UpdateMode.ACCUMULATE,
                        value=_scale_update_value(item.value, sub_dt / dt),
                    )},
                    coverage=1.0,
                )
                physical_replace_counts[name] = physical_replace_counts.get(name, 0) + 1
    for name, mean in physical_replace_means.items():
        if physical_replace_counts[name] != n_substeps:
            raise ValueError(
                f"Physical boundary substep coverage for '{name}' is "
                f"{physical_replace_counts[name]} of {n_substeps} actual children"
            )
        raw_updates[name] = replace(mean, mode=UpdateMode.REPLACE)
    composition = getattr(self, "_active_composition", None)
    if composition is not None and isinstance(raw_updates, dict):
        # A completed provider stage exposes its own time-weighted substep mean.
        stage_rates = {
            name: item for name, item in raw_updates.items()
            if _is_per_second_rate(item) and _is_boundary_rate_target(self, name)
            and item.write_mode != StateWriteMode.DERIVED
        }
        composition.stage_boundary_rates.setdefault(functor_name, {}).update(stage_rates)
        composition.dependency_outputs[functor_name] = dict(raw_updates)
    return raw_updates


def _flatten_diagnostics(prefix: str, value: Any, output: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            key_name = f"{prefix}.{child_key}" if prefix else str(child_key)
            _flatten_diagnostics(key_name, child_value, output)
        return
    output[prefix] = value


def _is_state_target(name: Any, state_keys: set, schema_keys: set) -> bool:
    if name in state_keys or name in schema_keys:
        return True
    if isinstance(name, str) and name.startswith("delta_"):
        target_name = name[len("delta_") :]
        return target_name in state_keys or target_name in schema_keys
    return False


def _clean_updates_and_collect_diagnostics(
    self: Any,
    functor_name: str,
    functor: Any,
    raw_updates: Dict[str, Any],
    group_diagnostics: Dict[str, Any],
    dt: Optional[float] = None,
) -> Dict[str, Any]:
    """Partition one Functor call's outputs, store its diagnostics, and return the rest.

    ``dt`` is the interval [s] of this group call; within the active group
    composition of interval H it covers the fraction dt / H
    (``merge_sequential_updates``). It is required when a composition is
    active; without one the call stores its own payload and ``dt`` is unused.
    """
    diagnostic_payload: Dict[str, Any] = {}
    output_vars = getattr(getattr(functor, "metadata", None), "output_variables", None)
    cleaned_updates, diagnostics, intents = partition_outputs(
        raw_updates, declared_outputs=output_vars or (),
    )
    if intents:
        raise ValueError("External intents must not enter diagnostic ingestion")
    # A declared output written with the DERIVED mode is a derived quantity of
    # this step: it is never committed as prognostic state (the commit owner
    # partitions it out again), but it stays visible to the Functors that
    # consume it later in the same group and keeps its write provenance.
    derived_outputs: Dict[str, Any] = {}
    for var_name in list(diagnostics):
        if not var_name.startswith(("diagnostics.", "__")) and var_name != "diagnostics":
            derived_outputs[var_name] = diagnostics.pop(var_name)
    cleaned_updates.update(derived_outputs)
    # Stored diagnostics follow the single merge rule over the executions of
    # this Functor within its composition (its Strang sub-executions): the
    # composition keeps the merged items with their modes, and StateManager
    # stores the merged payload, replacing the Functor's earlier entries. A
    # call that is a composition of its own stores its own payload. This call
    # covers dt of the composition interval H, so its per-second rates merge
    # with the coverage dt / H (1/2 for a Strang half-step, 1 otherwise).
    composition = getattr(self, "_active_composition", None)
    stored_diagnostics = {**diagnostics, **derived_outputs}
    if composition is not None and stored_diagnostics:
        if dt is None:
            raise ValueError(
                f"The call interval of '{functor_name}' is required to merge its "
                "diagnostics within the active group composition"
            )
        stored_diagnostics = merge_sequential_updates(
            composition.diagnostics.setdefault(functor_name, {}),
            stored_diagnostics,
            coverage=dt / composition.interval,
        )
    for var_name, update in stored_diagnostics.items():
        candidate = update.value if isinstance(update, UpdateItem) else update
        diag_key = var_name[len("diagnostics."):] if var_name.startswith("diagnostics.") else var_name
        _flatten_diagnostics(diag_key, candidate, diagnostic_payload)
    # Diagnostics of successive Functors of one group call merge by the single
    # rule (merge_sequential_updates), not by overwriting. Those Functors cover
    # the same interval, so a per-second rate merges with the coverage 1 (the
    # rates of concurrent processes add).
    functor_group_diagnostics: Dict[str, Any] = {}
    for var_name, update in diagnostics.items():
        _flatten_diagnostics(var_name, update, functor_group_diagnostics)
    merge_sequential_updates(group_diagnostics, functor_group_diagnostics, coverage=1.0)

    if diagnostic_payload:
        if self.state_manager is not None:
            self.state_manager.ingest_module_diagnostics(functor_name, diagnostic_payload)

    return cleaned_updates


def _safe_get_output_specs(functor: Any) -> Dict[str, Any]:
    if not hasattr(functor, "get_output_specifications"):
        return {}
    return functor.get_output_specifications() or {}


def _is_rate_like(name: Any, output_specs: Dict[str, Any]) -> bool:
    if not isinstance(name, str):
        return False
    if name.endswith("_flux") or name.endswith("_rate"):
        return True
    spec = output_specs.get(name)
    return spec is not None and getattr(spec, "is_rate", False)


def _build_provisional_state_and_converted_updates(
    self: Any,
    current_state: Dict[str, Any],
    raw_updates: Dict[str, Any],
    output_specs: Dict[str, Any],
    *,
    dt: float,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Return the state seen after these updates and the converted updates.

    ``dt`` is the interval [s] the updates account for: the substep interval
    for the state seen by the next substep, the group call interval for the
    state seen by the next Functor of the group.
    """
    provisional_state = current_state.copy()
    converted_updates: Dict[str, Dict[str, Any]] = {}
    composition = getattr(self, "_active_composition", None)
    stage_rates = {} if composition is None else composition.boundary_stage_view()

    for var_name, update in raw_updates.items():
        item = _coerce_runtime_update_item(
            self=self,
            var_name=var_name,
            update=update,
            output_specs=output_specs,
        )
        if item.mode == UpdateMode.FLUX:
            raise RuntimeError(
                f"FLUX update for '{var_name}' reached provisional state before integration"
            )

        base_value = current_state.get(var_name)
        value = item.value

        if isinstance(base_value, Tensor) and isinstance(value, Tensor):
            value = value.to(device=base_value.device, dtype=base_value.dtype)
        elif isinstance(base_value, Tensor):
            value = torch.as_tensor(
                value,
                device=base_value.device,
                dtype=base_value.dtype,
            )
        elif not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)

        if item.mode == UpdateMode.REPLACE:
            provisional_value = value
        elif _is_per_second_rate(item) and _is_boundary_rate_target(self, var_name):
            # References: StateManager.apply_updates; RELIC aggregate boundary contract.
            # Mathematical expression: q_visible = sum_f q_latest,f, never q_old + q_new.
            # Parameter meanings: writer contributions belong only to the current composition.
            # Units: the same per-second rate unit as the boundary variable.
            # Domain/constraints: isolated provisional calls contain one writer; group/substep
            #   execution uses the existing composition's post-reduction writer observations.
            # Sign convention: rates retain their boundary direction.
            provisional_value = stage_rates[var_name].value if var_name in stage_rates else value
        else:
            # References: StateManager.apply_updates (an ACCUMULATE rate with a
            #   per-second basis is integrated once over the commit interval; an
            #   unintegrated rate of a ContractRegistry flux variable is a
            #   boundary-rate aggregate kept as a rate); LeVeque (2002),
            #   finite-volume time integration.
            # Mathematical expression: x_prov = x + a for an amount a;
            #   x_prov = x + q dt for a per-second rate q that StateManager
            #   integrates; boundary-rate aggregates use the stage-view branch above.
            # Parameter meanings: x the state before these updates; a an amount;
            #   q a per-second rate; dt the interval the updates account for (the
            #   substep interval h or the group call interval).
            # Units: x, a in the target unit; q in the target unit per second;
            #   dt [s].
            # Domain/constraints: dt > 0 (checked by the group owner); FLUX items
            #   are lowered to amounts before this point; the committed update of
            #   a rate stays the rate (_build_adjusted_updates), so StateManager
            #   integrates it exactly once.
            # Sign convention: positive a and q increase the target.
            increment = _scale_update_value(value, dt) if _is_per_second_rate(item) else value
            if isinstance(base_value, Tensor):
                provisional_value = base_value + increment
            elif base_value is not None:
                base_tensor = torch.as_tensor(
                    float(base_value),
                    device=increment.device if isinstance(increment, Tensor) else None,
                    dtype=(increment.dtype if isinstance(increment, Tensor) else torch.float32),
                )
                delta_tensor = (
                    increment
                    if isinstance(increment, Tensor)
                    else torch.as_tensor(
                        increment,
                        device=base_tensor.device,
                        dtype=base_tensor.dtype,
                    )
                )
                provisional_value = base_tensor + delta_tensor
            else:
                provisional_value = increment

        provisional_state[var_name] = provisional_value
        converted_updates[var_name] = {
            "item": replace(item, value=value),
            "base_value": base_value,
            "value": value,
        }

    return provisional_state, converted_updates


def _build_adjusted_updates(
    provisional_state: Dict[str, Any],
    converted_updates: Dict[str, Dict[str, Any]],
) -> Dict[str, UpdateItem]:
    adjusted_updates: Dict[str, UpdateItem] = {}
    for var_name, info in converted_updates.items():
        item = info["item"]
        mode = item.mode
        base_value = info["base_value"]
        clamped_value = provisional_state.get(var_name, info["value"])

        if mode == UpdateMode.REPLACE:
            adjusted_updates[var_name] = replace(
                item,
                value=clamped_value,
                mode=UpdateMode.REPLACE,
            )
            continue

        if _is_per_second_rate(item):
            # A per-second rate is committed as the rate: StateManager integrates
            # it once over the commit interval (or keeps a boundary-rate
            # aggregate), so it is not re-derived from the provisional state.
            adjusted_updates[var_name] = item
            continue

        if isinstance(clamped_value, Tensor):
            if isinstance(base_value, Tensor):
                base_tensor = base_value.to(
                    device=clamped_value.device,
                    dtype=clamped_value.dtype,
                )
            elif base_value is not None:
                base_tensor = torch.as_tensor(
                    float(base_value),
                    device=clamped_value.device,
                    dtype=clamped_value.dtype,
                )
            else:
                base_tensor = torch.as_tensor(
                    0.0,
                    device=clamped_value.device,
                    dtype=clamped_value.dtype,
                )
            delta_value = clamped_value - base_tensor
        else:
            base_scalar = float(base_value) if base_value is not None else 0.0
            delta_value = float(clamped_value) - base_scalar

        adjusted_updates[var_name] = replace(
            item,
            value=delta_value,
            mode=UpdateMode.ACCUMULATE,
        )

    return adjusted_updates


def _check_flux_whitelist(self: Any, functor_name: str, adjusted_updates: Dict[str, Any]) -> None:
    contract_registry = (
        getattr(self.state_manager, "_contract_registry", None)
        if self.state_manager is not None
        else None
    )
    if contract_registry is None:
        return

    for var_name, update_item in adjusted_updates.items():
        if not isinstance(update_item, UpdateItem):
            continue
        flux_wl = contract_registry.get_flux_whitelist(var_name)
        if not flux_wl:
            continue
        allowed_in = flux_wl.get("flux_in", [])
        allowed_out = flux_wl.get("flux_out", [])
        if not allowed_in and not allowed_out:
            continue
        delta_val = update_item.value
        contract = contract_registry.get_contract(var_name)
        is_instantaneous_flux = (
            contract is not None
            and str(contract.kind).lower() == "flux"
            and not bool(update_item.per_timestep)
        )
        if is_instantaneous_flux:
            has_nonzero = (
                bool(torch.any(delta_val != 0).item())
                if isinstance(delta_val, Tensor) else float(delta_val) != 0.0
            )
            # Boundary fluxes commonly use non-negative magnitudes. Their
            # direction is owned by the SCM writer declaration, not sign.
            has_input = has_nonzero and functor_name in allowed_in
            has_output = has_nonzero and functor_name in allowed_out
            if has_nonzero and not has_input and not has_output:
                has_input = bool(allowed_in)
                has_output = bool(allowed_out)
        elif isinstance(delta_val, Tensor):
            has_input = bool(torch.any(delta_val > 0).item())
            has_output = bool(torch.any(delta_val < 0).item())
        else:
            delta_scalar = float(delta_val)
            has_input = delta_scalar > 0.0
            has_output = delta_scalar < 0.0

        # Zero increments carry no material-flow direction.For fields, inspect
        # every element so a net sum cannot hide a locally unauthorized flow.
        unauthorized_directions = []
        if has_input and functor_name not in allowed_in:
            unauthorized_directions.append("input")
        if has_output and functor_name not in allowed_out:
            unauthorized_directions.append("output")
        if unauthorized_directions:
            logger.warning(
                "Functor %s is not authorized for %s flux direction(s) of %s",
                functor_name,
                ",".join(unauthorized_directions),
                var_name,
            )


def _update_step_functor_increment(
    step_functor_increments: Dict[str, Any],
    var_name: str,
    functor_name: str,
    increment_value: Any,
) -> None:
    functor_map = step_functor_increments.setdefault(var_name, {})
    existing_inc = functor_map.get(functor_name)
    if existing_inc is None:
        functor_map[functor_name] = increment_value
    else:
        functor_map[functor_name] = _combine_update_values(existing_inc, increment_value)


def _provider_provenance(simulator: Any, functor_name: str) -> Optional[Dict[str, Any]]:
    """Provider UID and epoch of the view that executed ``functor_name``."""
    view = _current_provider_view(simulator)
    provider = view.providers.get(functor_name)
    if provider is None:
        return None
    return {"provider_uid": provider.provider_uid, "provider_epoch": view.epoch}


COMMIT_ROLES = ("lie", "first_half", "middle", "second_half", "group")
# Roles of the commits that are not a group commit (commit context, D7): the
# restatement of the aggregate boundary-rate stage view after a group call
# (_store_composition_boundary_rates; same stage and interval as that call), the
# finalization commit of a composition (GroupComposition.finalize_boundary_rates)
# and the commit of one multirate exchange mean (execution_loop). None of them is
# a budget entry (each commits without a conservation check).
STAGE_VIEW_ROLE = "stage_view"
COMPOSITION_MEAN_ROLE = "composition_mean"
EXCHANGE_ROLE = "exchange"


def resolve_commit_role(
    own_composition: bool,
    stage: int,
    dt: float,
    interval: float,
    earlier_calls: int,
) -> str:
    """Role of one Functor call in its group composition (pure; one of COMMIT_ROLES).

    References: Strang (1968) SIAM J. Numer. Anal. 5:506-517; Hairer, Lubich and
        Wanner (2006), Sect. II.5 (palindromic composition); design
        20261019_cr_a04_design_01, section 2 (provenance).
    Mathematical expression: group if the call is a composition of its own
        (medium and slow group calls); for a call over dt within a composition
        of interval H: lie if dt = H at stage 0, middle if dt = H at a later
        stage, first_half if dt < H and the Functor has no earlier call in the
        composition, second_half if dt < H after an earlier call.
    Parameter meanings: stage the index k of the group call in its composition;
        earlier_calls the calls of this Functor in the composition before this
        one; dt the call interval; H = interval.
    Units: dt, H [s]; stage and earlier_calls dimensionless.
    Domain/constraints: 0 < dt <= H; the fast-group owner passes H itself to a
        full-step call, so dt = H is compared exactly.
    Sign convention: not applicable.
    """
    if own_composition:
        return "group"
    if dt == interval:
        return "lie" if stage == 0 else "middle"
    return "second_half" if earlier_calls else "first_half"


def build_commit_context(
    *,
    step: Optional[int],
    composition_start: float,
    composition_interval: float,
    stage: int,
    role: str,
    interval_start: float,
    dt: float,
) -> Dict[str, Any]:
    """Provenance context of one Functor call and its group commit (pure, plain values).

    References: W3C PROV-DM (2013), generation records; design
        20261019_cr_a04_design_01, section 2 (commit_context; the group commit
        passes it to StateManager.apply_updates, whose events store a copy).
    Mathematical expression: the call covers [interval_start, interval_start + dt]
        inside the composition [composition_start, composition_start +
        composition_interval] of step ``step``.
    Parameter meanings: step the 0-based index of the step being executed (the
        accepted steps before it), None without a step counter; stage the group
        call index k; role from resolve_commit_role.
    Units: composition_start, composition_interval, interval_start, dt [s].
    Domain/constraints: plain int, float and str values only (checkpoint grammar).
    Sign convention: not applicable.
    """
    return {
        "step": None if step is None else int(step),
        "composition_start": float(composition_start),
        "composition_interval": float(composition_interval),
        "stage": int(stage),
        "role": str(role),
        "interval_start": float(interval_start),
        "dt": float(dt),
    }


def resolve_group_commit_context(call_contexts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The commit context of one group commit from the contexts of its Functor calls (pure).

    References: W3C PROV-DM (2013); design 20261019_cr_a04_design_01, D7 and
        section 2 (one commit, one context).
    Mathematical expression: C = c_1 if c_1 = c_2 = ... = c_n, else undefined.
    Parameter meanings: c_i the context of the i-th Functor call of the group
        call (build_commit_context); they share step, composition, stage,
        interval start and interval, and the role of a full-step call (lie,
        middle, group) does not depend on the Functor.
    Units: as build_commit_context.
    Domain/constraints: None without calls; a group call whose calls have
        different contexts (half-step calls of several Functors with different
        earlier calls; no current builder makes one) raises RuntimeError before
        the commit, because one commit cannot carry two roles.
    Sign convention: not applicable.
    """
    if not call_contexts:
        return None
    first = call_contexts[0]
    different = [context for context in call_contexts[1:] if context != first]
    if different:
        raise RuntimeError(
            "D7: the Functor calls of one group commit have different commit contexts "
            f"({first!r} and {different[0]!r}); a group commit carries one context"
        )
    return dict(first)


def _state_commit_index(state_manager: Any) -> Optional[int]:
    """Index of the next StateManager commit (the ``step`` of its provenance events)."""
    getter = getattr(state_manager, "get_diagnostics", None)
    if not callable(getter):
        return None
    counter = getter().get("update_counter")
    return counter if type(counter) is int else None


def _stamp_commit_index(step_provenance_detail: Dict[str, Any], commit_index: Optional[int]) -> None:
    """Record the group commit index in the provider dict of this call's writer records."""
    if commit_index is None:
        return
    for records in step_provenance_detail.values():
        for entry in records:
            provider = entry[4] if len(entry) > 4 else None
            if isinstance(provider, dict):
                provider["commit_index"] = commit_index


def _track_step_provenance(
    step_provenance_detail: Dict[str, Any],
    var_name: str,
    functor_name: str,
    mode: UpdateMode,
    step_index: Any,
    provider: Optional[Dict[str, Any]] = None,
) -> None:
    if var_name not in step_provenance_detail:
        step_provenance_detail[var_name] = []
    step_provenance_detail[var_name].append(
        (functor_name, mode, step_index, "writer", provider)
    )


def _resolve_overlap_and_maybe_skip(
    var_name: str,
    functor_name: str,
    update_item: UpdateItem,
    step_update_sources: Dict[str, Any],
) -> bool:
    existing_source = step_update_sources.get(var_name)
    if existing_source is None or existing_source[0] == functor_name:
        return False

    existing_mode = existing_source[1]
    overlap_policy = get_output_overlap_policy(var_name)
    if overlap_policy == OVERLAP_POLICY_MUTUALLY_EXCLUSIVE:
        raise RuntimeError(
            "Mutually exclusive overlap for '%s' encountered during step: "
            "%s (%s) vs %s (%s). Register only one model variant."
            % (
                var_name,
                existing_source[0],
                existing_mode,
                functor_name,
                update_item.mode,
            )
        )

    if update_item.mode == UpdateMode.REPLACE or existing_mode == UpdateMode.REPLACE:
        message = (
            "Conflicting REPLACE updates for '%s': %s (%s) vs %s (%s). "
            "Register only one functor per output variable or "
            "refactor outputs."
        )
        raise RuntimeError(
            message
            % (
                var_name,
                existing_source[0],
                existing_mode,
                functor_name,
                update_item.mode,
            )
        )

    return False


def _merge_into_all_updates(all_updates: Dict[str, Any], var_name: str, update_item: UpdateItem) -> None:
    existing = all_updates.get(var_name)
    if isinstance(existing, UpdateItem):
        all_updates[var_name] = _merge_update_items(var_name, existing, update_item)
        return

    if existing is not None:
        existing_item = replace(update_item, value=existing)
        all_updates[var_name] = _merge_update_items(
            var_name,
            existing_item,
            update_item,
        )
        return

    all_updates[var_name] = update_item


def _accumulate_adjusted_updates(
    functor_name: str,
    adjusted_updates: Dict[str, UpdateItem],
    all_updates: Dict[str, Any],
    step_update_sources: Dict[str, Any],
    step_functor_increments: Dict[str, Any],
    step_provenance_detail: Dict[str, Any],
    step_index: Any,
    provider: Optional[Dict[str, Any]] = None,
) -> None:
    for var_name, update_item in adjusted_updates.items():
        update_value = update_item.value
        if _is_numeric_like(update_value):
            should_skip = _resolve_overlap_and_maybe_skip(
                var_name=var_name,
                functor_name=functor_name,
                update_item=update_item,
                step_update_sources=step_update_sources,
            )
            if should_skip:
                continue

            if update_item.mode == UpdateMode.ACCUMULATE:
                _update_step_functor_increment(
                    step_functor_increments=step_functor_increments,
                    var_name=var_name,
                    functor_name=functor_name,
                    increment_value=update_value,
                )
            step_update_sources[var_name] = (functor_name, update_item.mode)
            _track_step_provenance(
                step_provenance_detail=step_provenance_detail,
                var_name=var_name,
                functor_name=functor_name,
                mode=update_item.mode,
                step_index=step_index,
                provider=provider,
            )

        _merge_into_all_updates(all_updates, var_name, update_item)


def _process_functor_execution(
    self: Any,
    functor_name: str,
    functor: Any,
    current_state: Dict[str, Any],
    dt: float,
    start_time: float,
    all_updates: Dict[str, Any],
    group_diagnostics: Dict[str, Any],
    step_update_sources: Dict[str, Any],
    step_functor_increments: Dict[str, Any],
    step_provenance_detail: Dict[str, Any],
    external_intents: List[Dict[str, Any]],
    commit_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_updates = _run_functor_substeps(
        self=self,
        functor_name=functor_name,
        functor=functor,
        current_state=current_state,
        dt=dt,
        start_time=start_time,
        external_intents=external_intents,
    )

    if isinstance(raw_updates, dict):
        raw_updates = _clean_updates_and_collect_diagnostics(
            self=self,
            functor_name=functor_name,
            functor=functor,
            raw_updates=raw_updates,
            group_diagnostics=group_diagnostics,
            dt=dt,
        )

    output_specs = _safe_get_output_specs(functor)
    provisional_state, converted_updates = _build_provisional_state_and_converted_updates(
        self=self,
        current_state=current_state,
        raw_updates=raw_updates,
        output_specs=output_specs,
        dt=dt,
    )
    adjusted_updates = _build_adjusted_updates(
        provisional_state=provisional_state,
        converted_updates=converted_updates,
    )

    self._collect_functor_diagnostics(functor, current_state, adjusted_updates)
    _check_flux_whitelist(
        self=self,
        functor_name=functor_name,
        adjusted_updates=adjusted_updates,
    )
    # The writer records of this call share one provider dict: provider UID and
    # epoch, the commit context of the call, and (after the group commit) the
    # StateManager commit index.
    provider = _provider_provenance(self, functor_name)
    if provider is not None and commit_context is not None:
        provider["commit_context"] = commit_context
    _accumulate_adjusted_updates(
        functor_name=functor_name,
        adjusted_updates=adjusted_updates,
        all_updates=all_updates,
        step_update_sources=step_update_sources,
        step_functor_increments=step_functor_increments,
        step_provenance_detail=step_provenance_detail,
        step_index=getattr(self, "_current_step", None),
        provider=provider,
    )
    return provisional_state


def _apply_accumulated_updates(
    self: Any,
    all_updates: Dict[str, Any],
    step_update_sources: Dict[str, Any],
    step_functor_increments: Dict[str, Any],
    dt: float,
    commit_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Commit the numeric state updates of one group call; return the committed payloads
    (after the distributed reduction), keyed by variable name. ``commit_context``
    is the provenance context of the commit (resolve_group_commit_context)."""
    baseline_state_keys = set(getattr(self.state_manager, "state_vars", {}).keys())
    baseline_schema_keys = set(getattr(self.state_manager, "schema", {}).keys())

    def _is_numeric_payload(value: Any) -> bool:
        if isinstance(value, UpdateItem):
            return _is_numeric_like(value.value)
        return _is_numeric_like(value)

    def _should_apply_to_state(var_name: str) -> bool:
        if var_name in baseline_state_keys or var_name in baseline_schema_keys:
            return True
        if isinstance(var_name, str) and var_name.startswith("delta_"):
            target_name = var_name[len("delta_") :]
            return target_name in baseline_state_keys or target_name in baseline_schema_keys
        source_info = step_update_sources.get(var_name)
        if source_info is None:
            return False
        source_functor_name = source_info[0]
        source_provider = _current_provider_view(self).providers.get(source_functor_name)
        functor_obj = None if source_provider is None else source_provider.functor
        output_vars = getattr(getattr(functor_obj, "metadata", None), "output_variables", None)
        if not output_vars:
            return True
        return var_name in output_vars

    state_updates, _, intents = partition_outputs(all_updates)
    if intents:
        raise ValueError("External intents cannot be applied to scientific state")
    numeric_updates = {
        k: v
        for k, v in state_updates.items()
        if _is_numeric_payload(v) and _should_apply_to_state(k)
    }
    check_conservation = self.enable_conservation_checks
    rollback_on_failure = self.strict_conservation_enabled
    source_map = {k: v[0] for k, v in step_update_sources.items() if k in numeric_updates}
    increment_map = {
        k: v for k, v in step_functor_increments.items() if k in numeric_updates
    }
    # Distributed ranks agree on the ensemble-mean update before it is
    # written; reducing after the write would average a stale copy (E02).
    reduce_started = time.perf_counter()
    # Boundary rates were reduced at their writer substeps, before consumers read them.
    other_updates = {
        name: item for name, item in numeric_updates.items()
        if not (_is_per_second_rate(item) and _is_boundary_rate_target(self, name))
    }
    if other_updates:
        self._reduce_updates_across_ranks(other_updates)
        numeric_updates.update(other_updates)
    reduce_timings = getattr(self, "_active_step_stage_timings", None)
    if reduce_timings is not None:
        cast(Dict[str, float], reduce_timings)["distributed_collective"] += max(
            0.0, time.perf_counter() - reduce_started
        )
    self.state_manager.apply_updates(
        numeric_updates,
        dt,
        check_conservation=check_conservation,
        rollback_on_failure=rollback_on_failure,
        update_sources=source_map,
        update_increments=increment_map,
        commit_context=commit_context,
    )
    return numeric_updates


def _accumulate_replace_rate_mean(
    composition: GroupComposition,
    var_name: str,
    writer: str,
    committed: UpdateItem,
    coverage: float,
) -> None:
    """Add one committed REPLACE flux rate to its S2 accumulator with the coverage h_k / H."""
    # References: Strang (1968); design 20261019_cr_a04_design_01 (S2); RELIC
    #   UpdateItem metadata contract (_merge_update_items).
    # Mathematical expression: M_k = M_(k-1) + (h_k / H) q_k, M_0 = 0.
    # Parameter meanings: q_k the reduced committed rate of this call; h_k = dt;
    #   H = composition.interval.
    # Units: M_k and q_k in the contract's rate unit; coverage [1].
    # Domain/constraints: one writer per variable; the contributions of one
    #   variable agree on unit, write mode, time flag and element (ValueError).
    # Sign convention: rates keep the contract's boundary direction.
    weighted = committed.value if coverage == 1.0 else _scale_update_value(committed.value, coverage)
    mean = composition.replace_rate_means.get(var_name)
    if mean is None:
        composition.replace_rate_means[var_name] = ReplaceRateMean(
            writer=writer, item=replace(committed, value=weighted), coverage=coverage, contributions=1,
        )
        return
    if mean.writer != writer:
        raise RuntimeError(
            f"S2: REPLACE flux rate '{var_name}' has two writers in one composition: "
            f"{mean.writer} and {writer}"
        )
    merged = _merge_update_items(
        var_name,
        replace(mean.item, mode=UpdateMode.ACCUMULATE),
        replace(committed, value=weighted, mode=UpdateMode.ACCUMULATE),
    )
    mean.item = replace(merged, mode=UpdateMode.REPLACE)
    mean.coverage += coverage
    mean.contributions += 1


def _return_completed_step_means(
    composition: GroupComposition,
    all_updates: Dict[str, Any],
    group_order: Collection[str],
) -> None:
    """Let the call that completes a writer's S2 mean return that mean (the step() value)."""
    # References: design 20261019_cr_a04_design_01 (S2); single merge rule
    #   (a later REPLACE item replaces the earlier one in the step return).
    # Mathematical expression: returned q = M once the writer's calls cover H
    #   (sum_k h_k / H = 1, m >= 2); the commit of this call kept q_m.
    # Parameter meanings: M the accumulated mean; m the writer's calls so far.
    # Units: the contract's rate unit.
    # Domain/constraints: only the composition owner consumes this return; the
    #   published state equals it after finalize_boundary_rates.
    # Sign convention: rates keep the contract's boundary direction.
    for var_name, mean in composition.replace_rate_means.items():
        if (
            mean.writer in group_order
            and mean.contributions >= 2
            and mean.contributions == len(composition.member_calls.get(mean.writer, ()))
            and math.isclose(mean.coverage, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            and isinstance(all_updates.get(var_name), UpdateItem)
        ):
            all_updates[var_name] = replace(all_updates[var_name], value=mean.item.value)


def _store_composition_boundary_rates(
    self: Any,
    composition: GroupComposition,
    committed_updates: Dict[str, Any],
    dt: float,
    group_order: Collection[str],
    writers: Optional[Mapping[str, Any]] = None,
    commit_context: Optional[Dict[str, Any]] = None,
) -> None:
    """Accumulate final means privately and publish only complete stage contributions.

    REPLACE per-second rates on SCM flux contracts enter the S2 accumulators of
    the composition (``writers`` maps a variable to its (writer, mode) source).
    ``commit_context`` is the context of the group commit of this call; the
    restatement commit of the stage view carries it with role stage_view.
    """
    # References: Strang (1968); StateManager per-commit boundary ledger; RELIC merge rule.
    # Mathematical expression: B_k=B_(k-1)+(h_k/H)*q_k; q_stage=sum_f q_latest,f.
    # Parameter meanings: q_k is this call's reduced committed rate; h_k=dt; H=composition.interval.
    # Units: B_k and q_stage keep the boundary rate units; h_k and H are seconds.
    # Domain/constraints: each call's physical q_k*h_k was accounted for by its one commit.
    #   Prior writers are retained in stage observations, not in the next physical ledger.
    #   B_k is not visible to stage consumers; finalization publishes B only on completion.
    # Sign convention: each rate retains its contract's boundary direction.
    coverage = _validate_merge_coverage(dt / composition.interval)
    stage_view = composition.boundary_stage_view()
    stage_restated: Dict[str, Any] = {}
    for var_name, committed in committed_updates.items():
        if _is_replace_boundary_rate(self, var_name, committed):
            source = (writers or {}).get(var_name)
            writer = source[0] if source else committed.source_functor
            _accumulate_replace_rate_mean(composition, var_name, str(writer), committed, coverage)
            continue
        if not (_is_per_second_rate(committed) and _is_boundary_rate_target(self, var_name)):
            continue
        earlier = var_name in composition.boundary_rates
        merge_sequential_updates(composition.boundary_rates, {var_name: committed}, coverage=coverage)
        composition.boundary_rates_need_finalization |= earlier or coverage != 1.0
        if any(var_name in rates and writer not in group_order
               for writer, rates in composition.stage_boundary_rates.items()):
            stage_restated[var_name] = stage_view[var_name]
    if stage_restated:
        self.state_manager.apply_updates(
            stage_restated, dt, check_conservation=False, rollback_on_failure=False,
            commit_context=(
                None if commit_context is None else {**commit_context, "role": STAGE_VIEW_ROLE}
            ),
        )


def _merge_group_diagnostics_into_updates(
    all_updates: Dict[str, Any],
    group_diagnostics: Dict[str, Any],
) -> None:
    for key, value in group_diagnostics.items():
        if key in all_updates:
            raise ValueError(f"Diagnostic output '{key}' conflicts with a scientific state update")
        if isinstance(value, Tensor):
            # Returned diagnostics may participate in differentiable losses.
            all_updates[key] = value
        elif isinstance(value, (float, int)):
            all_updates[key] = torch.as_tensor(float(value), dtype=torch.float32)
        else:
            all_updates[key] = value


def _execute_group_impl(
    self: Any,
    group_order: List[str],
    dt: float,
    start_time: float,
    composition: Optional[GroupComposition] = None,
) -> FluxDict:
    """
    Internal implementation for group execution over ``[start_time, start_time + dt]``.
    """
    if isinstance(start_time, bool) or not isinstance(start_time, Real):
        raise TypeError(
            f"Group start_time must be a real scalar in seconds, got {type(start_time).__name__}"
        )
    start_time = float(start_time)
    if not math.isfinite(start_time):
        raise ValueError(f"Group start_time must be finite, got {start_time}")
    if not group_order:
        return {}

    current_state = _prepare_group_state(self, group_order)
    all_updates: Dict[str, Any] = {}
    group_diagnostics: Dict[str, Any] = {}
    external_intents: List[Dict[str, Any]] = []
    committed_step_update_sources = _get_or_init_step_update_sources(self)
    step_update_sources = dict(committed_step_update_sources)
    committed_step_functor_increments = getattr(self, "_step_functor_increments", None)
    step_functor_increments = {
        var_name: dict(increments) if isinstance(increments, dict) else increments
        for var_name, increments in (committed_step_functor_increments or {}).items()
    }
    committed_step_provenance = getattr(self, "_step_provenance_detail", None)
    # Writer records of this group call only; the recorded history is not copied.
    # References: RELICSimulator.get_variable_provenance (writer records accumulate
    #   across accepted steps); execution_loop._AppendOnlyHistorySavepoint (step
    #   rollback by saved lengths).
    # Mathematical expression: N_v collects the records this call writes for v; a
    #   successful call publishes H_v := H_v + N_v (list extension in place, H_v
    #   created empty for a new v); a failed call publishes nothing.
    #   Cost: O(sum_v len(N_v)) per call, independent of len(H_v).
    # Parameter meanings: H_v is the committed record list of variable v in
    #   simulator._step_provenance_detail; N_v the new records of this call.
    # Units: counts of records (dimensionless).
    # Domain/constraints: H_v only grows at its end, so the step savepoint restores
    #   it by truncation; publication follows the state commit of this call.
    # Sign convention: not applicable (no flux or signed quantity).
    step_provenance_detail: Dict[str, Any] = {}
    state_manager = getattr(self, "state_manager", None)
    diagnostics_snapshot = None
    if state_manager is not None and hasattr(state_manager, "_current_diagnostics"):
        diagnostics_snapshot = {
            key: value.clone() if isinstance(value, Tensor) else value
            for key, value in state_manager._current_diagnostics.items()
        }
    group_succeeded = False
    # The group composition (see _execute_fast_group): the caller's composition
    # for a sub-execution of a composition, else a new empty one over this call.
    self._active_composition = (
        GroupComposition(interval=dt) if composition is None else composition
    )
    active_composition = self._active_composition
    # Stage k of this call and the start of the composition (commit context, D7).
    stage = active_composition.stage_count
    step_counter = getattr(self, "step_count", None)
    step_number = step_counter if type(step_counter) is int else None
    if active_composition.start_time is None:
        active_composition.start_time = start_time
        active_composition.step = step_number
    # The contexts of this call's Functor calls; the group commit carries their
    # common value (resolve_group_commit_context).
    call_contexts: List[Dict[str, Any]] = []
    # Providers are resolved from the frozen step view (or the committed view
    # for direct group calls), never from the live registry, so a concurrent
    # registration cannot mix old and new providers inside one group.
    providers = _current_provider_view(self).providers
    try:
        for functor_name in group_order:
            provider = providers.get(functor_name)
            if provider is None:
                continue

            functor = provider.functor
            commit_context = build_commit_context(
                step=step_number,
                composition_start=cast(float, active_composition.start_time),
                composition_interval=active_composition.interval,
                stage=stage,
                role=resolve_commit_role(
                    composition is None,
                    stage,
                    float(dt),
                    active_composition.interval,
                    len(active_composition.member_calls.get(functor_name, ())),
                ),
                interval_start=start_time,
                dt=dt,
            )
            call_contexts.append(commit_context)
            try:
                current_state = _process_functor_execution(
                    self=self,
                    functor_name=functor_name,
                    functor=functor,
                    current_state=current_state,
                    dt=dt,
                    start_time=start_time,
                    all_updates=all_updates,
                    group_diagnostics=group_diagnostics,
                    step_update_sources=step_update_sources,
                    step_functor_increments=step_functor_increments,
                    step_provenance_detail=step_provenance_detail,
                    external_intents=external_intents,
                    commit_context=commit_context,
                )
                # Coverage h_k / H of this call within the composition (S2, roles).
                active_composition.member_calls.setdefault(functor_name, []).append(
                    float(dt) / active_composition.interval
                )
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                logger.error("Functor %s failed: %s", functor_name, e)
                self.state = self.state.__class__.FAILED
                raise RuntimeError(f"Group execution failed in functor {functor_name}") from e

        accepted_step_intents = getattr(self, "_active_step_external_intents", None)
        if external_intents and not isinstance(accepted_step_intents, list):
            raise RuntimeError("External intents require an active accepted-step transaction")
        has_state_updates = bool(all_updates)
        _merge_group_diagnostics_into_updates(all_updates, group_diagnostics)
        if self.state_manager is not None and has_state_updates:
            stage_timings = getattr(self, "_active_step_stage_timings", None)
            apply_started = (
                time.perf_counter() if isinstance(stage_timings, dict) else None
            )
            try:
                commit_index = _state_commit_index(self.state_manager)
                group_commit_context = resolve_group_commit_context(call_contexts)
                committed_updates = _apply_accumulated_updates(
                    self=self,
                    all_updates=all_updates,
                    step_update_sources=step_update_sources,
                    step_functor_increments=step_functor_increments,
                    dt=dt,
                    commit_context=group_commit_context,
                )
                _stamp_commit_index(step_provenance_detail, commit_index)
                _store_composition_boundary_rates(
                    self, self._active_composition, committed_updates, dt, group_order,
                    writers=step_update_sources, commit_context=group_commit_context,
                )
                _return_completed_step_means(self._active_composition, all_updates, group_order)
            except ValueError as e:
                logger.error("Group updates violated constraints: %s", e)
                self._last_step_had_violation = True
                raise RuntimeError(f"Group updates violated constraints: {e}") from e
            finally:
                if apply_started is not None:
                    cast(Dict[str, float], stage_timings)["state_apply_updates"] += max(
                        0.0,
                        time.perf_counter() - apply_started,
                    )

        committed_step_update_sources.clear()
        committed_step_update_sources.update(step_update_sources)
        if committed_step_functor_increments is None:
            self._step_functor_increments = step_functor_increments
        else:
            committed_step_functor_increments.clear()
            committed_step_functor_increments.update(step_functor_increments)
        if committed_step_provenance is None:
            self._step_provenance_detail = step_provenance_detail
        else:
            for var_name, records in step_provenance_detail.items():
                committed_step_provenance.setdefault(var_name, []).extend(records)

        if external_intents:
            cast(List[Any], accepted_step_intents).extend(external_intents)

        active_composition.stage_count += 1
        if composition is None:
            self._active_composition.finalize_boundary_rates(self.state_manager)
        group_succeeded = True
        return all_updates
    finally:
        if (
            not group_succeeded
            and diagnostics_snapshot is not None
            and state_manager is not None
        ):
            state_manager._current_diagnostics = diagnostics_snapshot
        self._active_composition = None
