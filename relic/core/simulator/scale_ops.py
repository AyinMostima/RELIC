"""
Scale consistency and group-order helpers for RELICSimulator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from torch import Tensor

from ..dependency_graph import (
    CyclicDependencyError,
    DependencyGraphResolver as DependencyGraph,
    MissingDependencyError,
    normalize_dependency_declaration,
)
from ..functor.functor_types import SpatialScale

logger = logging.getLogger(__name__)

_SCALE_PREFERENCE = {
    SpatialScale.POINT.value: 6,
    SpatialScale.PATCH.value: 5,
    SpatialScale.GRID_CELL.value: 4,
    SpatialScale.HILLSLOPE.value: 3,
    SpatialScale.BASIN.value: 2,
    SpatialScale.LANDSCAPE.value: 2,
    SpatialScale.GRID_2D.value: 1,
    SpatialScale.GRID_3D.value: 0,
    SpatialScale.REGIONAL.value: -1,
    SpatialScale.GLOBAL.value: -2,
    SpatialScale.UNKNOWN.value: -3,
}

_SIMULATOR_CONTROL_STATE_KEYS = frozenset({"time", "step", "state_var"})

_GROUP_ORDER_ATTRIBUTES = {
    "fast": "_order_fast",
    "medium": "_order_medium",
    "slow": "_order_slow",
}


def _build_metadata_scale_map(simulator: Any) -> Dict[str, str]:
    """Tier 1 of scale detection: the declared output scales of the Functors.

    Each declaration is parsed by SpatialScale.from_value, the one parser, so a
    misspelled label raises instead of reaching the Tier 2 heuristics; only an
    undeclared scale (UNKNOWN) is left to them.
    """
    metadata_scale_map: Dict[str, str] = {}
    for functor_name, functor_obj in getattr(simulator, "functors", {}).items():
        meta = getattr(functor_obj, "metadata", None)
        if meta is None:
            continue
        try:
            declared_scale = SpatialScale.from_value(getattr(meta, "spatial_scale", None))
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"metadata.spatial_scale of Functor {functor_name!r}: {exc}"
            ) from exc
        declared_scale_value = declared_scale.value
        if declared_scale_value == SpatialScale.UNKNOWN.value:
            continue
        for out_var in getattr(meta, "output_variables", []) or []:
            existing_scale = metadata_scale_map.get(out_var)
            if existing_scale is not None and existing_scale != declared_scale_value:
                raise ValueError(
                    "Conflicting spatial scales declared for output "
                    f"'{out_var}': {existing_scale}, {declared_scale_value}"
                )
            metadata_scale_map[out_var] = declared_scale_value
    return metadata_scale_map


def _infer_scale_from_shape(var_value: Tensor) -> str:
    var_shape = var_value.shape
    numel = int(var_value.numel())
    if len(var_shape) == 0 or numel <= 10:
        return SpatialScale.PATCH.value
    if numel <= 50:
        return SpatialScale.HILLSLOPE.value
    if numel <= 500:
        return SpatialScale.BASIN.value
    if len(var_shape) == 1:
        return SpatialScale.BASIN.value
    if len(var_shape) == 2:
        return SpatialScale.GRID_2D.value
    if len(var_shape) == 3:
        return SpatialScale.GRID_3D.value
    return SpatialScale.UNKNOWN.value


def _infer_detected_scale(
    var_name: str,
    var_value: Tensor,
    metadata_scale_map: Dict[str, str],
) -> str:
    if var_name in metadata_scale_map:
        return metadata_scale_map[var_name]

    name_lower = var_name.lower()
    if "basin" in name_lower:
        return SpatialScale.BASIN.value
    if "hillslope" in name_lower:
        return SpatialScale.HILLSLOPE.value
    if "patch" in name_lower:
        return SpatialScale.PATCH.value
    return _infer_scale_from_shape(var_value)


def _needs_regrid(unique_scales: set[str]) -> bool:
    known_scales = unique_scales - {SpatialScale.UNKNOWN.value}
    return len(known_scales) > 1


def _select_target_scale(unique_scales: set[str]) -> str:
    return max(unique_scales, key=lambda s: _SCALE_PREFERENCE.get(s, 0))


def _record_scale_check(simulator: Any, scale_check: Dict[str, Any]) -> None:
    scale_check["timestamp"] = simulator.current_time
    simulator._scale_check_history.append(scale_check)
    if len(simulator._scale_check_history) > 100:
        simulator._scale_check_history = simulator._scale_check_history[-50:]


def check_scale_consistency(
    self: Any,
    state: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Detect potential scale mismatches in state tensors.
    """
    auto_scale_disabled = not self.enable_auto_scale or self.scale_manager is None
    if auto_scale_disabled and not force:
        return {"needs_regrid": False, "reason": "auto_scale disabled"}

    scale_check: Dict[str, Any] = {
        "needs_regrid": False,
        "reason": None,
        "detected_scales": {},
        "target_scale": None,
        "affected_variables": [],
    }

    # Metadata-declared scale is authoritative; heuristic detection is fallback.
    metadata_scale_map = _build_metadata_scale_map(self)

    spatial_tensors = {
        var_name: var_value
        for var_name, var_value in state.items()
        if isinstance(var_value, Tensor)
        and var_name not in _SIMULATOR_CONTROL_STATE_KEYS
    }
    authoritative_outputs = {
        var_name: metadata_scale_map[var_name]
        for var_name in spatial_tensors
        if var_name in metadata_scale_map
    }
    if authoritative_outputs:
        scale_check["detected_scales"].update(authoritative_outputs)
    else:
        for var_name, var_value in spatial_tensors.items():
            scale_check["detected_scales"][var_name] = _infer_detected_scale(
                var_name,
                var_value,
                metadata_scale_map,
            )

    unique_scales = set(scale_check["detected_scales"].values())
    if _needs_regrid(unique_scales):
        scale_check["needs_regrid"] = True
        scale_check["reason"] = "multiple scales detected"
        scale_check["affected_variables"] = list(scale_check["detected_scales"].keys())
        best_scale = _select_target_scale(unique_scales)
        scale_check["target_scale"] = best_scale
        logger.debug(
            "Scale mismatch detected: %s, target: %s",
            unique_scales,
            best_scale,
        )

    _record_scale_check(self, scale_check)

    if auto_scale_disabled and scale_check.get("needs_regrid", False):
        scale_check["reason"] = scale_check.get("reason") or "scale mismatch detected"

    return scale_check


def compute_group_execution_orders(self: Any) -> None:
    """
    Compute dependency-aware execution order for each multi-rate group.

    Only same-group required and optional edges constrain a group order; an
    empty group gets an empty order. A missing provider, a cycle, or any other
    sort failure propagates with a note naming the group (fail closed); no
    unsorted fallback order is ever installed.
    """
    if not self._groups_dirty:
        return

    for group_name, functor_names in self._groups.items():
        if not functor_names:
            # A group emptied by retirement or by cross-group promotion must not
            # keep a stale order.
            setattr(self, _GROUP_ORDER_ATTRIBUTES[group_name], [])
            continue

        group_graph = DependencyGraph()
        group_members = set(functor_names)

        for functor_name in functor_names:
            if functor_name in self.functors:
                functor = self.functors[functor_name]
                group_graph.add_functor(functor, functor_name)

                declared = normalize_dependency_declaration(
                    functor.get_dependencies(), owner=functor_name
                )
                deps_in_group = [dep for dep in declared.same_step if dep in group_members]

                node = group_graph.nodes[functor_name]
                node.dependencies = set(deps_in_group)

                optional_in_group = {dep for dep in declared.optional if dep in group_members}
                group_graph._optional_deps[functor_name] = optional_in_group

                lagged_in_group = {dep for dep in declared.lagged if dep in group_members}
                group_graph._lagged_deps[functor_name] = lagged_in_group
                node.lagged_dependencies = set(lagged_in_group)

        try:
            group_order = group_graph.topological_sort()
        except (MissingDependencyError, CyclicDependencyError, ValueError) as exc:
            # Fail closed: a group whose dependency order cannot be computed is
            # never executed in unsorted registration order. The previous orders
            # stay untouched and the groups stay dirty; the registration owner
            # rolls the publishing transaction back.
            registered = sorted(group_graph.nodes.keys())
            logger.error(
                "Cannot compute the execution order of the '%s' group (members %s): %s",
                group_name,
                registered,
                exc,
            )
            exc.add_note(f"Execution group: '{group_name}' (members: {registered})")
            raise
        setattr(self, _GROUP_ORDER_ATTRIBUTES[group_name], group_order)
        logger.debug("Computed execution order for %s group: %s", group_name, group_order)

    self._groups_dirty = False
    logger.info("Group execution orders computed successfully")
