"""Dependency and execution-group helpers for RELICSimulator."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any, List, Tuple

from ..conservation_registry import build_conservation_ledger
from ..dependency_graph import normalize_dependency_declaration
from ..functor import FunctorBase, TemporalScale

_VALID_EXECUTION_GROUPS = {"fast", "medium", "slow"}

# Temporal zone semantic mapping over existing execution groups.
TEMPORAL_ZONE_LABELS = {
    "fast": "core",
    "medium": "physics",
    "slow": "coupling",
}

# Zone-level physical constraints (values are configured via ParameterManager).
TEMPORAL_ZONE_CONSTRAINTS = {
    "core": {"max_energy_injection_rate": None},
    "physics": {"max_energy_injection_rate": None},
    "coupling": {"max_energy_injection_rate": None},
}


def _is_valid_execution_group(value: Any) -> bool:
    """Return True when value is a supported execution group string."""
    return isinstance(value, str) and value in _VALID_EXECUTION_GROUPS


def _invalid_execution_group_message(value: Any, *, source: str) -> str:
    """The one message for an execution group that ``_is_valid_execution_group`` rejects.

    Shared by registration (``_require_execution_group``) and GraphIR
    validation, so compile-time and runtime report the same text.
    """
    return (
        f"Invalid execution_group {value!r} from {source}; expected one of "
        f"{sorted(_VALID_EXECUTION_GROUPS)}."
    )


def _require_execution_group(value: Any, *, source: str) -> str:
    """Return a declared execution group; raise ValueError naming its source."""
    if not _is_valid_execution_group(value):
        raise ValueError(_invalid_execution_group_message(value, source=source))
    return str(value)


def _require_characteristic_days(value: Any, *, owner: str) -> float:
    """Return a declared characteristic time in days; raise ValueError when invalid."""
    days = None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        days = float(value)
    if days is None or not math.isfinite(days) or days < 0.0:
        raise ValueError(
            f"characteristic_time_days of Functor '{owner}' must be a finite "
            f"non-negative real number of days; got {value!r}."
        )
    return days


_SCHEDULING_NS = "computation.scheduling"


def resolve_scheduling_thresholds() -> Tuple[float, float]:
    """Return the (fast_max, medium_max) characteristic-time thresholds in days.

    References: RELIC architecture_overview.md section 4.4.1; Lawrence et al. (2019)
        CLM5 Technical Note, Section 2.7 (multi-rate time stepping).
    Mathematical expression: gamma(tau_c) = fast if tau_c <= T_f, medium if
        T_f < tau_c <= T_m, slow otherwise, with T_f and T_m read from the
        parameter database (computation.scheduling.*_max_characteristic_days).
    Parameter meanings: tau_c is the Functor characteristic time; T_f and T_m are
        the fast and medium upper bounds.
    Units: tau_c, T_f, T_m [day].
    Domain/constraints: 0 < T_f < T_m; both keys are required YAML parameters and
        no Python literal may stand in for them.
    Sign convention: thresholds are positive durations.
    """
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    fast_max = float(
        manager.get(
            f"{_SCHEDULING_NS}.fast_group_max_characteristic_days",
            units="day",
            required=True,
        )
    )
    medium_max = float(
        manager.get(
            f"{_SCHEDULING_NS}.medium_group_max_characteristic_days",
            units="day",
            required=True,
        )
    )
    if not (0.0 < fast_max < medium_max):
        raise ValueError(
            "computation.scheduling thresholds must satisfy 0 < fast_group_max_characteristic_days "
            f"< medium_group_max_characteristic_days; got {fast_max} and {medium_max}."
        )
    return fast_max, medium_max


def _group_from_characteristic_days(char_days: float) -> str:
    """Map validated characteristic days to an execution group using the YAML thresholds."""
    fast_max, medium_max = resolve_scheduling_thresholds()
    if char_days <= fast_max:
        return "fast"
    if char_days <= medium_max:
        return "medium"
    return "slow"


def _attribute_group(functor: FunctorBase) -> str | None:
    """Group of the Functor attribute ``execution_group``; None when undeclared."""
    declared = getattr(functor, "execution_group", None)
    if declared is None:
        return None
    return _require_execution_group(
        declared,
        source=f"the execution_group attribute of Functor {getattr(functor, 'name', None)!r}",
    )


def _characteristic_time_group(functor: FunctorBase) -> str | None:
    """Group of the Functor attribute ``characteristic_time_days``; None when undeclared."""
    declared = getattr(functor, "characteristic_time_days", None)
    if declared is None:
        return None
    days = _require_characteristic_days(declared, owner=str(getattr(functor, "name", None)))
    return _group_from_characteristic_days(days)


def _resolve_temporal_scale_group(functor: FunctorBase) -> str | None:
    """Derive an execution group from ordered temporal-scale semantics.

    References: CF Conventions calendar semantics and RELIC execution groups.
    Mathematical expression: gamma(tau) is fast, medium, or slow over ordered
    scale classes whose supported calendar realizations cannot cross a group
    boundary.
    Parameter meanings: tau is FunctorMetadata.temporal_scale; gamma is the
    scheduler execution group.
    Units: fixed SI durations are converted to days by TemporalScale.
    Domain/constraints: the value is parsed by TemporalScale.from_value, the
    one parser, which raises for a value that names no scale; UNKNOWN (also
    from None or "unknown") means no declaration and returns None.
    Every declared scale must have one explicit scheduling class.
    Sign convention: elapsed durations are non-negative and increase from fast
    to slow classes.
    """
    meta = getattr(functor, "metadata", None)
    declared = getattr(meta, "temporal_scale", None) if meta is not None else None
    try:
        scale = TemporalScale.from_value(declared)
    except (TypeError, ValueError) as exc:
        raise type(exc)(
            f"metadata.temporal_scale of Functor {getattr(functor, 'name', None)!r}: {exc}"
        ) from exc
    if scale is TemporalScale.UNKNOWN:
        return None
    if scale in (
        TemporalScale.SECOND,
        TemporalScale.MINUTE,
        TemporalScale.HOUR,
        TemporalScale.DAY,
        TemporalScale.STATIC,
        TemporalScale.STEADY_STATE,
    ):
        return _group_from_characteristic_days(scale.to_characteristic_time_days())
    if scale is TemporalScale.SUB_DAILY:
        return "fast"
    if scale is TemporalScale.WEEK:
        return "medium"
    if scale in (TemporalScale.MONTH, TemporalScale.SEASON, TemporalScale.YEAR):
        return "slow"
    raise ValueError(f"Temporal scale '{scale.value}' has no execution-group contract")


def _process_groups_entry(simulator: Any, functor_name: str) -> str | None:
    """Return the validated process_groups entry of a Functor; None when it has none."""
    process_groups = simulator._get_setting("process_groups", None)
    if process_groups is None:
        return None
    if not isinstance(process_groups, Mapping):
        raise ValueError(
            "process_groups must be a mapping of Functor name to execution group; "
            f"got {type(process_groups).__name__}."
        )
    if functor_name not in process_groups:
        return None
    return _require_execution_group(
        process_groups[functor_name],
        source=f"the process_groups entry of Functor '{functor_name}'",
    )


def find_missing_required_dependencies(simulator: Any) -> List[Tuple[str, str]]:
    """Return missing required dependencies across registered functors."""
    missing: List[Tuple[str, str]] = []
    for functor_name, functor in simulator.functors.items():
        if not hasattr(functor, "get_dependencies"):
            continue
        declared = normalize_dependency_declaration(
            functor.get_dependencies(), owner=functor_name
        )
        for dep in declared.required:
            if dep not in simulator.functors:
                missing.append((functor_name, dep))
    return missing


def refresh_conservation_ledger(simulator: Any) -> None:
    """Rebuild the StateManager ledger from contract and parameter SSOTs."""
    if simulator.state_manager is None:
        return

    contract_registry = getattr(simulator.state_manager, "_contract_registry", None)
    canonical_ledger = build_conservation_ledger(
        contract_registry,
        force_reload=True,
    )
    simulator.state_manager.set_conservation_ledger(canonical_ledger)


def determine_execution_group(
    simulator: Any,
    functor: FunctorBase,
    execution_group: Any = None,
) -> str:
    """Resolve the execution group of a Functor by the one priority chain.

    Priority: the register_functor argument > the execution_group attribute >
    characteristic_time_days > metadata.temporal_scale > the process_groups
    entry > "fast" when nothing is declared. Every declared value is validated,
    also when a higher rule decides, and an invalid value raises ValueError.
    A process_groups entry fills in only for a Functor without an attribute,
    characteristic time or temporal scale; an entry that disagrees with the
    group of such a rule raises ValueError, and the register argument overrides
    the entry.
    """
    # References: RELIC architecture_overview.md Section 4.4.1 (multi-rate
    #   execution groups); Lawrence et al. (2019) CLM5 Technical Note,
    #   Section 2.7 (multi-rate time stepping).
    # Mathematical expression: g = a if a is given; else g = e if e is
    #   declared; else g = gamma(tau_c) if tau_c is declared; else
    #   g = gamma_s(s) if s is declared; else g = p if p is declared; else
    #   g = fast. When e, tau_c or s decides and p is declared, p must equal g.
    # Parameter meanings: a is the register_functor argument; e the Functor's
    #   execution_group attribute; tau_c its characteristic_time_days; s its
    #   metadata.temporal_scale; p its process_groups configuration entry;
    #   gamma maps tau_c to a group with the computation.scheduling thresholds
    #   (resolve_scheduling_thresholds); gamma_s maps s by its scale class
    #   (_resolve_temporal_scale_group).
    # Units: tau_c [day]; a, e, p and g are dimensionless group labels.
    # Domain/constraints: a, e and p are one of fast, medium, slow; tau_c is a
    #   finite non-negative real number (not a bool); s is parsed by
    #   TemporalScale.from_value, and UNKNOWN (also None) means no declaration.
    #   None of a, e, tau_c, p means no declaration. Every declared value is validated
    #   even when a higher rule decides; an invalid value or a conflicting p
    #   raises ValueError. No silent fallback exists: only the absence of every
    #   declaration resolves to fast.
    # Sign convention: none; tau_c is a non-negative duration.
    name = functor.name
    argument = None
    if execution_group is not None:
        argument = _require_execution_group(
            execution_group, source=f"the register_functor argument of Functor '{name}'"
        )
    # Every rule is evaluated, so every declaration is validated even when a
    # higher rule decides.
    declared_rules = (
        ("execution_group attribute", _attribute_group(functor)),
        ("characteristic_time_days", _characteristic_time_group(functor)),
        ("metadata.temporal_scale", _resolve_temporal_scale_group(functor)),
    )
    entry = _process_groups_entry(simulator, name)

    if argument is not None:
        return argument
    for rule, group in declared_rules:
        if group is None:
            continue
        if entry is not None and entry != group:
            raise ValueError(
                f"process_groups assigns Functor '{name}' to {entry!r}, but its {rule} "
                f"(a higher-priority rule) assigns {group!r}; remove the process_groups "
                "entry, or override the group with the register_functor(..., "
                "execution_group=...) argument."
            )
        return group
    return "fast" if entry is None else entry


def get_temporal_zone_label(execution_group: str) -> str:
    """Map execution group to temporal zone label."""
    return TEMPORAL_ZONE_LABELS.get(execution_group, "core")


__all__ = [
    "find_missing_required_dependencies",
    "refresh_conservation_ledger",
    "determine_execution_group",
    "resolve_scheduling_thresholds",
    "TEMPORAL_ZONE_LABELS",
    "TEMPORAL_ZONE_CONSTRAINTS",
    "get_temporal_zone_label",
]
