"""
State regridding helpers for RELICSimulator.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ...data.standards.cf_conventions import CF_REGRID_TARGET_GRID_FIELDS
from ..abstractions import StateDict

logger = logging.getLogger(__name__)

# CF_REGRID_TARGET_GRID_FIELDS is owned by the CF synchronizer
# (CFAttributeSynchronizer.sync_after_regridding); the grid shape is taken from
# the descriptor geometry. The fields are declared on the explicit auto_scale
# descriptors and never inferred.


def _split_grid_descriptor(
    origin: str,
    descriptor: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split one explicit grid descriptor into (geometry, CF grid metadata)."""
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"{origin} must be a mapping, got {type(descriptor).__name__}")
    cf_fields = {
        name: descriptor[name] for name in CF_REGRID_TARGET_GRID_FIELDS if name in descriptor
    }
    geometry = {
        name: value
        for name, value in descriptor.items()
        if name not in CF_REGRID_TARGET_GRID_FIELDS
    }
    return geometry, cf_fields


def _require_cf_grid_fields(simulator: Any, origin: str, descriptor: Mapping) -> None:
    """Reject a target descriptor lacking the CF fields of the attached CF synchronizer."""
    scale_manager = getattr(simulator, "scale_manager", None)
    if scale_manager is None or getattr(scale_manager, "cf_synchronizer", None) is None:
        return
    missing = [
        name
        for name in CF_REGRID_TARGET_GRID_FIELDS
        if not isinstance(descriptor.get(name), str) or not descriptor[name].strip()
    ]
    if missing:
        raise ValueError(
            f"{origin} lacks the CF grid metadata required by the attached CF "
            f"synchronizer: {', '.join(missing)}. Declare them explicitly on the "
            "descriptor; they are never inferred from tensor shape or scale labels."
        )


def _is_identity_regrid(source_descriptor: Any, target_descriptor: Any) -> bool:
    """True only when both descriptors carry literally the same geometry.

    ScaleManager.regrid_variable returns the input unchanged for identical grids
    and does not call the CF synchronizer then (``identity_operation`` branch of
    ``manager_spatial_ops.regrid_variable_impl``). Only exact equality of the
    geometry entries qualifies; geometrically equal but differently written
    descriptors are treated as a real regrid and need the CF fields.
    """
    if not isinstance(source_descriptor, Mapping) or not isinstance(target_descriptor, Mapping):
        return False
    source_geometry, _ = _split_grid_descriptor("source_grid", source_descriptor)
    target_geometry, _ = _split_grid_descriptor("target_grid", target_descriptor)
    return source_geometry == target_geometry


def validate_auto_scale_grid_descriptors(simulator: Any) -> None:
    """Validate the configured ``auto_scale`` grid descriptors (configuration time).

    Called once the ScaleManager lifecycle has attached its CF synchronizer.
    Descriptors are optional here (regridding requires them when it runs); a
    configured descriptor must be a mapping, and a configured target descriptor
    must carry the CF grid fields unless the pair is an identity regrid, for
    which the synchronizer never runs.
    """
    auto_scale_config = simulator._get_setting("auto_scale", {}) or {}
    if not isinstance(auto_scale_config, Mapping):
        raise TypeError("auto_scale configuration must be a mapping")
    for name in ("source_grid", "target_grid"):
        descriptor = auto_scale_config.get(name)
        if descriptor is None:
            continue
        _split_grid_descriptor(f"auto_scale.{name}", descriptor)
    target = auto_scale_config.get("target_grid")
    if target is not None and not _is_identity_regrid(
        auto_scale_config.get("source_grid"), target
    ):
        _require_cf_grid_fields(simulator, "auto_scale.target_grid", target)


def _coerce_regular_grid_descriptor(grid_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert StateManager grid-resolution payloads to GridDescriptor-compatible mappings.
    """
    if not isinstance(grid_info, dict):
        return grid_info

    shape_raw = grid_info.get("shape")
    resolution_raw = grid_info.get("resolution")
    if not isinstance(shape_raw, Sequence) or not shape_raw:
        return grid_info
    if not isinstance(resolution_raw, Sequence) or not resolution_raw:
        return grid_info

    try:
        shape_vals = [max(int(v), 1) for v in shape_raw]
    except (TypeError, ValueError):
        return grid_info

    # GridDescriptor compact initializer expects shape=(nx, ny).
    if len(shape_vals) == 1:
        shape: Tuple[int, int] = (shape_vals[0], 1)
    else:
        shape = (shape_vals[-1], shape_vals[-2])

    try:
        resolution_vals = [float(v) for v in resolution_raw]
    except (TypeError, ValueError):
        return grid_info

    if len(resolution_vals) == 1:
        spacing = (resolution_vals[0], 1.0)
    else:
        spacing = (resolution_vals[-1], resolution_vals[-2])

    return {
        "grid_type": "regular",
        "coord_system": "cartesian",
        "shape": shape,
        "spacing": spacing,
        "origin": (0.0, 0.0),
    }


def perform_regridding(self: Any, state: StateDict, scale_check: Dict[str, Any]) -> StateDict:
    """
    Regrid state variables when scale mismatch is detected.

    References
    ----------
    Jones (1999), First- and second-order conservative remapping schemes.

    Mathematical expression
    -----------------------
    For an area density, q_t[j] = sum_i(A_ij * q_s[i]) / A_t[j], so
    sum_j(A_t[j] * q_t[j]) = sum_i(A_s[i] * q_s[i]).

    Parameter meanings
    ------------------
    A_ij is source-target overlap area [m2], A_t is target-cell area [m2],
    q is a field expressed per unit area. Cell-integrated quantities require
    an explicit variable-role contract in the canonical ScaleManager audit.
    """
    self._ensure_scale_manager_ready()
    scale_manager = self.scale_manager
    if not self.enable_auto_scale:
        logger.warning(
            "Regridding requested but auto_scale is disabled"
        )
        return state
    if scale_manager is None:
        raise RuntimeError(
            "ScaleManager is required for regridding but is unavailable."
        )

    if not scale_check.get("needs_regrid", False):
        return state

    target_scale = scale_check.get("target_scale")
    if not target_scale:
        logger.warning("No target scale specified for regridding")
        return state

    regridded_state = state.copy()
    regrid_count = 0

    auto_scale_config = self._get_setting("auto_scale", {}) or {}
    if not isinstance(auto_scale_config, Mapping):
        raise TypeError("auto_scale configuration must be a mapping")
    descriptors: Dict[str, Tuple[str, Any]] = {}
    for name in ("source_grid", "target_grid"):
        if name in scale_check:
            descriptors[name] = (f"scale_check.{name}", scale_check[name])
        else:
            descriptors[name] = (f"auto_scale.{name}", auto_scale_config.get(name))
    if descriptors["source_grid"][1] is None or descriptors["target_grid"][1] is None:
        raise ValueError(
            "Regridding requires explicit source_grid and target_grid descriptors; "
            "tensor shape and scale labels are insufficient to reconstruct geometry."
        )
    from relic.coupling.scales import GridDescriptor

    identity_regrid = _is_identity_regrid(
        descriptors["source_grid"][1], descriptors["target_grid"][1]
    )
    grids: Dict[str, Any] = {}
    grid_declarations: Dict[str, Dict[str, Any]] = {}
    for name, (origin, raw_descriptor) in descriptors.items():
        geometry, cf_fields = _split_grid_descriptor(origin, raw_descriptor)
        if name == "target_grid" and not identity_regrid:
            _require_cf_grid_fields(self, origin, raw_descriptor)
        geometry = dict(_coerce_regular_grid_descriptor(geometry))
        geometry.setdefault("device", torch.device(self.device))
        grids[name] = GridDescriptor(**geometry)
        # The CF synchronizer reads the grid shape and the declared CF fields; the
        # ScaleManager checks that the declared shape equals the runtime grid.
        grid_declarations[name] = {**cf_fields, "shape": tuple(grids[name].dims)}
    source_grid = grids["source_grid"]
    target_grid = grids["target_grid"]

    if identity_regrid:
        # Identical source and target geometry map every cell onto itself: no
        # value changes, so neither the CF grid fields nor a variable's spatial
        # basis (its catalog metadata) is needed. Any other descriptor pair is a
        # real regrid and keeps the per-variable metadata requirement below.
        logger.debug("Identity regridding to %s scale: state unchanged", target_scale)
        return state

    for var_name, var_value in state.items():
        if (
            isinstance(var_value, Tensor)
            and var_name in scale_check["detected_scales"]
        ):
            current_scale = scale_check["detected_scales"][var_name]
            if current_scale != target_scale and current_scale != "unknown":
                # One explicit metadata entry per variable: the ScaleManager variable
                # catalog entry (the same source its conservative kernel reads when no
                # metadata is given; an undeclared variable raises). Its spatial basis
                # selects the method; the CF grid declarations of both descriptors are
                # added for the CF synchronizer.
                variable_metadata = scale_manager.get_variable_metadata(
                    var_name, scale=current_scale
                )
                regrid_strategy = self._determine_regrid_strategy(var_name, variable_metadata)
                variable_metadata["source_grid"] = dict(grid_declarations["source_grid"])
                variable_metadata["target_grid"] = dict(grid_declarations["target_grid"])
                regrid_result = scale_manager.regrid_variable(
                    var_name=var_name,
                    var_value=var_value,
                    source_scale=current_scale,
                    target_scale=target_scale,
                    strategy=regrid_strategy,
                    source_grid=source_grid,
                    target_grid=target_grid,
                    metadata=variable_metadata,
                )
                regridded_value = (
                    regrid_result[0]
                    if isinstance(regrid_result, tuple)
                    else regrid_result
                )
                if regridded_value is None:
                    raise RuntimeError(f"Regridding returned None for variable '{var_name}'.")

                regridded_state[var_name] = regridded_value
                regrid_count += 1

                logger.debug(
                    "Regridded %s: %s -> %s",
                    var_name,
                    current_scale,
                    target_scale,
                )

    if regrid_count > 0:
        logger.info(
            "Regridding completed: %s variables to %s scale",
            regrid_count,
            target_scale,
        )

    return regridded_state
