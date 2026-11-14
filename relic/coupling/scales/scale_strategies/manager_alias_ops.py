"""Delegated alias transition helpers for ScaleManager."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from torch import Tensor

from ...core.policy import ScalingMode
from ..types import ScalingStrategy

logger = logging.getLogger(__name__)


def _normalize_transition_list(raw: Any, transition_type: str) -> Set[Tuple[str, str, str]]:
    normalized: Set[Tuple[str, str, str]] = set()
    if not raw:
        return normalized
    for item in raw:
        if not (isinstance(item, (list, tuple)) and len(item) == 3):
            raise ValueError(
                f"Alias entry for {transition_type} must be [source, target, direction]; got {item!r}"
            )
        source, target, direction = item
        dir_lower = str(direction).lower()
        if dir_lower not in {"up", "down"}:
            raise ValueError(f"Alias entry direction must be 'up' or 'down'; got {direction!r}")
        normalized.add((str(source).lower(), str(target).lower(), dir_lower))
    return normalized


def _load_transition_set(
    parameter_manager: Any,
    *,
    alias_namespace: str,
    key: str,
    transition_type: str,
) -> Set[Tuple[str, str, str]]:
    raw = parameter_manager.get(
        key,
        namespace=alias_namespace,
        expected_type=list,
        required=False,
    )
    return _normalize_transition_list(raw or [], transition_type)


def _merge_transition_types(
    *,
    priority_order: List[str],
    transitions_by_type: Dict[str, Set[Tuple[str, str, str]]],
    warning_callback,
) -> Dict[Tuple[str, str, str], str]:
    final_types: Dict[Tuple[str, str, str], str] = {}
    for t_type in priority_order:
        for triple in transitions_by_type.get(t_type, ()):
            existing = final_types.get(triple)
            if existing is not None and existing != t_type:
                message = (
                    f"Alias conflict for {triple}: keeping '{existing}' per priority order, "
                    f"ignoring '{t_type}'."
                )
                warning_callback(message)
                continue
            final_types.setdefault(triple, t_type)
    return final_types


def _pair_set(
    final_types: Dict[Tuple[str, str, str], str],
    expected_type: str,
    expected_direction: str,
) -> Set[Tuple[str, str]]:
    return {
        (source, target)
        for (source, target, direction), triple_type in final_types.items()
        if triple_type == expected_type and direction == expected_direction
    }


def load_alias_configuration_impl(
    self,
    *,
    alias_namespace: str,
    default_priority_order: List[str],
) -> None:
    pm = self.parameter_manager
    if pm is None:
        raise RuntimeError("ParameterManager is required to load alias configuration.")

    priority_order = pm.get(
        "priority_order",
        namespace=alias_namespace,
        expected_type=list,
        required=False,
    )
    if not isinstance(priority_order, list) or not priority_order:
        priority_order = list(default_priority_order)
    priority_order = [str(item).lower() for item in priority_order]

    transitions_by_type: Dict[str, Set[Tuple[str, str, str]]] = {}
    loaded_any = False

    for key, t_type in (
        ("temporal_transitions", "temporal"),
        ("spatial_transitions", "spatial"),
        ("conservative_transitions", "conservative"),
    ):
        normalized = _load_transition_set(
            pm,
            alias_namespace=alias_namespace,
            key=key,
            transition_type=t_type,
        )
        transitions_by_type[t_type] = normalized
        loaded_any = loaded_any or bool(normalized)

    if not loaded_any:
        logger.info("No alias transitions configured under %s; alias sets remain empty.", alias_namespace)
        return

    def _warn_alias(message: str) -> None:
        self._warnings.append(message)
        logger.warning("ScaleManager alias warning: %s", message)

    final_types = _merge_transition_types(
        priority_order=priority_order,
        transitions_by_type=transitions_by_type,
        warning_callback=_warn_alias,
    )

    # Every alias pair is a valid scale transition. Spatial and conservative pairs run through the
    # hierarchical path (declared topology, basis-aware kernels, one ledger audit); only temporal
    # pairs are routed separately (they fail closed until a calendar period contract is bound).
    self._alias_transitions = set(final_types.keys())
    self._temporal_alias_up = _pair_set(final_types, "temporal", "up")
    self._temporal_alias_down = _pair_set(final_types, "temporal", "down")


def _require_temporal_alias_guard(self) -> None:
    """A temporal alias pair must never fall through to the spatial hierarchy path."""
    if self.temporal_strategy is None:
        raise RuntimeError(
            "The temporal alias guard is not bound: ScaleManager.initialize() must create the "
            "temporal strategy before any alias scale pair is aggregated or disaggregated."
        )


def handle_alias_aggregation_impl(
    self,
    data_dict: Dict[str, Tensor],
    source_scale: str,
    target_scale: str,
    variables: List[str],
    strategy: ScalingStrategy,
    mode: ScalingMode,
    metadata: Optional[Dict[str, Any]],
    var_metadata: Dict[str, Dict[str, Any]],
) -> None:
    """Reject a temporal alias pair (no calendar period contract); every other pair returns.

    Spatial and conservative alias pairs are aggregated by the hierarchical path like any other
    scale pair: the declared topology and areas, the basis-aware kernels and the ledger audit of
    the returned result. There is no alias-specific remapper.
    """
    _require_temporal_alias_guard(self)
    self.temporal_strategy.handle_alias_aggregation(
        data_dict=data_dict,
        source_scale=source_scale,
        target_scale=target_scale,
        variables=variables,
        strategy=strategy,
        mode=mode,
        metadata=metadata,
        var_metadata=var_metadata,
    )


def handle_alias_disaggregation_impl(
    self,
    data_dict: Dict[str, Tensor],
    source_scale: str,
    target_scale: str,
    variables: List[str],
    strategy: ScalingStrategy,
    mode: ScalingMode,
    metadata: Optional[Dict[str, Any]],
    var_metadata: Dict[str, Dict[str, Any]],
) -> None:
    """Reject a temporal alias pair; spatial and conservative pairs disaggregate hierarchically."""
    _require_temporal_alias_guard(self)
    self.temporal_strategy.handle_alias_disaggregation(
        data_dict=data_dict,
        source_scale=source_scale,
        target_scale=target_scale,
        variables=variables,
        strategy=strategy,
        mode=mode,
        metadata=metadata,
        var_metadata=var_metadata,
    )
