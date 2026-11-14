"""Snapshot and diff helpers for parameter manager state."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, Tuple

import torch

from ..spec import ParameterSpec
from ..storage import ParameterSnapshot
from ..serialization import deserialize_tensor
from ..types import ValidationLevel, UnitValidationMode
from .compare import values_differ

logger = logging.getLogger(__name__)


def _live_learnable_values(manager) -> Dict[str, torch.Tensor]:
    """The live tensors of the learnable bindings, copied.

    An optimizer may have moved a binding away from its declared value; the
    snapshot keeps that live value apart from the declared (spec) and stored
    values, so each owner is captured and restored with its own type.
    """
    return {key: parameter.detach().clone() for key, parameter in manager.nn_parameters.items()}


def snapshot(manager) -> Dict[str, Any]:
    """Create an exact snapshot: specs, stored values and live learnable tensors, each as it is.

    The storage is captured by its own snapshot, so a spec whose value is not
    stored yet stays without a stored value after the restore.
    """
    storage_snapshot = manager.storage.snapshot(
        "Parameter state snapshot", metadata={"nn_values": _live_learnable_values(manager)}
    )

    return {
        "timestamp": storage_snapshot.timestamp.isoformat() if storage_snapshot.timestamp else None,
        "parameters": {key: deepcopy(spec.to_dict()) for key, spec in manager.parameters.items()},
        "device": str(manager.device),
        "validation_level": manager.validation_level.value,
        "unit_validation_mode": manager.unit_validation_mode.value,
        "unit_validation_issues": list(manager._unit_validation_issues),
        "storage_snapshot": storage_snapshot,
        "nn_requires_grad": {
            key: parameter.requires_grad for key, parameter in manager.nn_parameters.items()
        },
        "nn_values": deepcopy(storage_snapshot.metadata["nn_values"]),
    }


def manager_to_dict(manager) -> Dict[str, Any]:
    """Convert parameter manager state to a serializable dictionary."""
    parameters_dict = {key: spec.to_dict() for key, spec in manager.parameters.items()}

    nn_parameters_dict = {}
    for key, param in manager.nn_parameters.items():
        nn_parameters_dict[key] = {
            "value": param.data.tolist() if param.numel() > 1 else param.data.item(),
            "requires_grad": param.requires_grad,
            "device": str(param.device),
            "dtype": str(param.dtype),
        }

    return {
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "parameters": parameters_dict,
        "nn_parameters": nn_parameters_dict,
        "metadata": {
            "total_parameters": len(manager.parameters),
            "learnable_parameters": sum(1 for spec in manager.parameters.values() if spec.requires_grad),
            "device": str(manager.device),
            "validation_level": str(manager.validation_level),
            "database_path": str(manager.database_path),
            "change_history_length": len(manager.change_history),
            "missing_requests_count": len(manager._missing_requests),
        },
        "runtime_state": {
            "accessed_parameters": list(manager._access_history),
            "missing_requests": dict(manager._missing_requests),
            "change_callbacks_count": len(manager.change_callbacks),
        },
    }


def _prepare_snapshot_specs(manager, snapshot_data: Dict[str, Any]) -> Dict[str, ParameterSpec]:
    """Rebuild and validate the ParameterSpec objects carried by a snapshot."""
    prepared_specs: Dict[str, ParameterSpec] = {}
    for key, param_data in snapshot_data["parameters"].items():
        spec = ParameterSpec.from_dict(deepcopy(param_data))
        if spec.key != key:
            raise ValueError(f"Parameter snapshot key does not match spec: {key}")
        if hasattr(manager, "database") and hasattr(manager.database, "_validate_spec"):
            manager.database._validate_spec(spec)
        prepared_specs[key] = spec
    return prepared_specs


def _check_storage_snapshot(prepared_specs: Dict[str, ParameterSpec], storage_snapshot: Any) -> None:
    """Require every captured stored value to agree with its snapshot spec.

    A spec without a stored value (reloaded by set_parameter, not accessed yet)
    and a stored value without a spec are states of the live owners; the
    snapshot keeps them as they are.
    """
    if not isinstance(storage_snapshot, ParameterSnapshot):
        raise TypeError("Parameter storage snapshot has an invalid type")
    stored_values = storage_snapshot.parameters
    for key, spec in prepared_specs.items():
        if key not in stored_values:
            continue
        stored = stored_values[key]
        dtype_changed = (
            isinstance(spec.value, torch.Tensor) and isinstance(stored, torch.Tensor)
            and spec.value.dtype != stored.dtype
        )
        if dtype_changed or values_differ(spec.value, stored):
            raise ValueError(f"Parameter snapshot contains conflicting values: {key}")


def _check_gradient_flags(prepared_specs: Dict[str, ParameterSpec], flags: Any) -> Dict[str, bool]:
    """Require explicit gradient flags, one per live binding, each of a learnable spec.

    A learnable spec gets its binding on first access, so a snapshot may hold
    learnable specs without a binding; the restore leaves them without one.
    """
    if not isinstance(flags, dict) or any(type(value) is not bool for value in flags.values()):
        raise ValueError("Parameter snapshot requires explicit nn_requires_grad flags")
    if not set(flags) <= {key for key, spec in prepared_specs.items() if spec.requires_grad}:
        raise ValueError("Parameter snapshot gradient flags name parameters that are not learnable specs")
    return flags


def _check_live_values(
    prepared_specs: Dict[str, ParameterSpec],
    flags: Dict[str, bool],
    live_values: Any,
) -> Dict[str, torch.Tensor]:
    """Require one live tensor per live binding, with its spec's shape and dtype."""
    if not isinstance(live_values, dict) or set(live_values) != set(flags):
        raise ValueError("Parameter snapshot requires nn_values for exactly its live bindings")
    for key, tensor in live_values.items():
        declared = prepared_specs[key].to_tensor(torch.device("cpu"))
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != declared.shape
            or tensor.dtype != declared.dtype
        ):
            raise ValueError(f"Parameter snapshot live value does not match its spec binding: {key}")
    return live_values


def restore(manager, snapshot_data: Dict[str, Any]) -> None:
    """Restore parameter state from snapshot data."""
    if "parameters" not in snapshot_data:
        raise ValueError("Invalid snapshot format")

    prepared_specs = _prepare_snapshot_specs(manager, snapshot_data)

    storage_snapshot = snapshot_data.get("storage_snapshot")
    if storage_snapshot is not None:
        _check_storage_snapshot(prepared_specs, storage_snapshot)

    flags = _check_gradient_flags(prepared_specs, snapshot_data.get("nn_requires_grad"))
    live_values = _check_live_values(prepared_specs, flags, snapshot_data.get("nn_values"))
    target_device = torch.device(snapshot_data.get("device", manager.device))
    target_validation = ValidationLevel(snapshot_data.get("validation_level", manager.validation_level.value))
    target_units = UnitValidationMode(snapshot_data.get("unit_validation_mode", manager.unit_validation_mode.value))
    prepared_tensors = {key: live_values[key].to(target_device) for key in flags}
    for key, tensor in prepared_tensors.items():
        existing = manager.nn_parameters.get(key)
        live_spec = manager.parameters.get(key)
        # A live learnable binding may sit on the device of the Functor that
        # aliases it (registration migrates aliased Parameters in place);
        # restore copies values across devices and never relocates the binding.
        if existing is not None and (
            existing.shape != tensor.shape or existing.dtype != tensor.dtype
            or live_spec is None or live_spec.dtype != prepared_specs[key].dtype
        ):
            raise ValueError(f"Cannot restore incompatible live Parameter binding: {key}")

    manager.parameters.clear()
    manager.parameters.update(prepared_specs)
    manager.device = target_device
    for key in list(manager.nn_parameters):
        if key not in prepared_tensors:
            del manager.nn_parameters[key]
    for key, tensor in prepared_tensors.items():
        existing = manager.nn_parameters.get(key)
        if existing is None:
            manager._register_nn_parameter(prepared_specs[key])
            existing = manager.nn_parameters[key]
        with torch.no_grad():
            existing.copy_(tensor)
        existing.requires_grad_(flags[key])

    manager.validation_level = target_validation
    manager.unit_validation_mode = target_units
    if "unit_validation_issues" in snapshot_data:
        manager._unit_validation_issues = list(snapshot_data["unit_validation_issues"])
    manager._unit_validation_checked.clear()

    # The stored values are written directly: this restore sets the exact
    # specifications itself (including those whose value is not stored), so the
    # storage restore callback, which re-derives specifications, is not involved.
    manager.storage.clear()
    if storage_snapshot is not None:
        manager.storage.update(dict(storage_snapshot.parameters))
    else:
        manager.storage.update(
            {key: spec.value for key, spec in manager.parameters.items()}
        )
    manager.clear_cache()

    logger.info("Restored %s parameters from snapshot", len(manager.parameters))


def _restored_binding_value(key: str, binding: torch.Tensor, value: Any) -> torch.Tensor:
    """The tensor a live learnable binding takes for a restored stored value.

    The value keeps the binding's dtype and device; a single value fills the
    binding (as ``set_value`` does), any other shape must match the binding.
    """
    tensor = torch.as_tensor(
        value.detach() if isinstance(value, torch.Tensor) else value,
        dtype=binding.dtype,
        device=binding.device,
    )
    if tensor.shape == binding.shape:
        return tensor
    if tensor.numel() == 1:
        return tensor.reshape(()).expand(binding.shape)
    raise ValueError(
        f"Restored value of '{key}' has shape {tuple(tensor.shape)}, its learnable "
        f"binding has shape {tuple(binding.shape)}"
    )


def on_storage_restore(manager, snapshot_data: Any) -> None:
    """Bring the specs and live bindings in line with values restored into storage.

    Every restored value needs a specification: the loaded one, or the
    database one when it is not loaded yet. A value without any specification
    has no units, bounds or dtype, and none is invented: the snapshot is
    rejected (KeyError) before anything changes, and the storage takes back its
    previous values. Live learnable bindings take the restored values (same
    Parameter objects); specifications and bindings of parameters absent from
    the restored values are dropped.
    """
    restored_params = dict(getattr(snapshot_data, "parameters", {}) or {})
    specs: Dict[str, ParameterSpec] = {}
    unknown = []
    for key in restored_params:
        spec = manager.parameters.get(key)
        if spec is None:
            spec = manager.database.get_parameter(key)
        if spec is None:
            unknown.append(key)
        else:
            specs[key] = spec
    if unknown:
        raise KeyError(
            "Storage snapshot holds values for parameters without a specification: "
            f"{sorted(unknown)}; register them with register_parameter() before restoring"
        )
    binding_values = {
        key: _restored_binding_value(key, manager.nn_parameters[key], value)
        for key, value in restored_params.items()
        if key in manager.nn_parameters
    }

    for key in list(manager.parameters.keys()):
        if key not in restored_params:
            manager.parameters.pop(key)
            manager.nn_parameters.pop(key, None)
    for key, value in restored_params.items():
        manager.parameters[key] = replace(specs[key], value=value)
    for key, tensor in binding_values.items():
        with torch.no_grad():
            manager.nn_parameters[key].copy_(tensor)

    manager.clear_cache()


def diff_snapshots(manager, other_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the resolved values of the current state with another snapshot.

    Both sides use the projection of ``_extract_parameter_values``: declared
    values with the live learnable values over them.
    """
    current_snapshot = snapshot(manager)

    diff_result = {
        "added": [],
        "removed": [],
        "modified": {},
        "summary": {"total_changes": 0},
    }

    current_params = _extract_parameter_values(current_snapshot)
    other_params = _extract_parameter_values(other_snapshot)

    for key in current_params:
        if key not in other_params:
            diff_result["added"].append(key)

    for key in other_params:
        if key not in current_params:
            diff_result["removed"].append(key)

    for key in current_params:
        if key in other_params:
            current_value = current_params[key]
            other_value = other_params[key]
            if values_differ(other_value, current_value):
                diff_result["modified"][key] = {
                    "old": other_value,
                    "new": current_value,
                }

    diff_result["summary"]["total_changes"] = (
        len(diff_result["added"]) + len(diff_result["removed"]) + len(diff_result["modified"])
    )

    return diff_result


def _stored_parameter_values(manager) -> Dict[str, Any]:
    """Every parameter's stored value, or its spec value while none is stored, read-only."""
    stored = {key: spec.value for key, spec in manager.parameters.items()}
    stored.update(manager.storage.get_all())
    return stored


def _resolved_parameter_values(manager) -> Dict[str, Any]:
    """Project resolved values (live learnable tensors over stored values), read-only."""
    resolved = _stored_parameter_values(manager)
    resolved.update(
        {key: parameter.detach() for key, parameter in manager.nn_parameters.items()}
    )
    return resolved


def _resolved_snapshot_values(snapshot_data: ParameterSnapshot) -> Dict[str, Any]:
    """The same projection for a snapshot: its live learnable tensors over its stored values.

    A snapshot taken by the storage alone records no learnable bindings.
    """
    resolved = dict(snapshot_data.parameters)
    resolved.update(deepcopy(snapshot_data.metadata.get("nn_values", {})))
    return resolved


def get_diff(manager, snapshot_data: ParameterSnapshot) -> Dict[str, Tuple[Any, Any]]:
    """Get differences between current state and a snapshot (resolved values on both sides)."""
    diff = {}

    current_params = _resolved_parameter_values(manager)
    snapshot_params = _resolved_snapshot_values(snapshot_data)

    for key, old_value in snapshot_params.items():
        if key in current_params:
            new_value = current_params[key]
            if values_differ(old_value, new_value):
                diff[key] = (old_value, new_value)
        else:
            diff[key] = (old_value, None)

    for key, new_value in current_params.items():
        if key not in snapshot_params:
            diff[key] = (None, new_value)

    return diff


def create_snapshot(manager, description: str = "") -> ParameterSnapshot:
    """Snapshot every parameter's value exactly (types kept): its stored value, or its
    spec value while none is stored; the live learnable tensors are recorded apart
    in ``metadata["nn_values"]``."""
    return manager.storage.snapshot(
        description,
        parameters=_stored_parameter_values(manager),
        metadata={"nn_values": _live_learnable_values(manager)},
    )


def restore_snapshot(manager, snapshot_data: ParameterSnapshot) -> None:
    """Restore the snapshotted values of the parameters that still exist, exactly.

    Stored values are written back as they were (no validation, clamping or
    conversion: they were valid when captured), then the live learnable tensors
    are copied into their bindings. A snapshot taken by the storage alone
    records no learnable bindings.
    """
    for key, value in snapshot_data.parameters.items():
        if key in manager.parameters:
            manager.set_value(key, value, validate=False)
    for key, tensor in snapshot_data.metadata.get("nn_values", {}).items():
        binding = manager.nn_parameters.get(key)
        if binding is not None:
            with torch.no_grad():
                binding.copy_(tensor.to(device=binding.device, dtype=binding.dtype))


def _decode_snapshot_value(value: Any) -> Any:
    """Decode a snapshot value that uses the Tensor wire format; other values pass through."""
    if isinstance(value, dict) and value.get("type") == "tensor":
        return deserialize_tensor(value)
    return value


def _extract_parameter_values(snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolved parameter values of a snapshot payload.

    The declared values of its specs, with the live learnable values
    (``nn_values``, raw Tensors or the Tensor wire of a JSON baseline) over
    them: the value ``manager.get`` returns for a learnable parameter.
    """
    raw_parameters = snapshot_data.get("parameters", {})
    extracted: Dict[str, Any] = {}
    for key, value in raw_parameters.items():
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        extracted[key] = _decode_snapshot_value(value)
    for key, value in snapshot_data.get("nn_values", {}).items():
        extracted[key] = _decode_snapshot_value(value)
    return extracted


def _is_numeric(value: Any) -> bool:
    """Return True for numeric scalar values excluding booleans."""
    if isinstance(value, torch.Tensor):
        return value.numel() == 1 and value.dtype != torch.bool and not value.is_complex()
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_mapping(mapping: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionaries into dot-path key mapping."""
    flattened: Dict[str, Any] = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def compute_parameter_diff(
    current_snapshot: Dict[str, Any],
    baseline_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute parameter diff between current and baseline snapshots.

    Mathematical Expression:
        Numeric:
            delta = current - baseline
            relative_change = delta / |baseline|
        Categorical:
            changed = current != baseline

    References:
        - Easterbrook (2014), Geosci. Model Dev. model evolution tracking.

    Parameters:
        current_snapshot: snapshot dict for current run
        baseline_snapshot: snapshot dict for baseline run

    Units: delta retains the parameter unit; relative_change is dimensionless.
    Domain/constraints: scalar numeric differences; zero baseline has no ratio.
    Sign convention: positive delta denotes an increase from the baseline.
    """
    current_values = _extract_parameter_values(current_snapshot)
    baseline_values = _extract_parameter_values(baseline_snapshot)

    changed_parameters = []
    added_parameters = []
    removed_parameters = []

    current_keys = set(current_values.keys())
    baseline_keys = set(baseline_values.keys())

    for key in sorted(current_keys - baseline_keys):
        added_parameters.append({"key": key, "value": current_values[key]})

    for key in sorted(baseline_keys - current_keys):
        removed_parameters.append({"key": key, "value": baseline_values[key]})

    for key in sorted(current_keys & baseline_keys):
        current_value = current_values[key]
        baseline_value = baseline_values[key]
        tensor_values = isinstance(current_value, torch.Tensor) or isinstance(baseline_value, torch.Tensor)
        contract_changes = {}
        if isinstance(current_value, torch.Tensor) and isinstance(baseline_value, torch.Tensor):
            for field in ("shape", "dtype", "requires_grad"):
                old = getattr(baseline_value, field)
                new = getattr(current_value, field)
                if old != new:
                    contract_changes[field] = {"baseline": old, "current": new}
        if not contract_changes and not values_differ(baseline_value, current_value):
            continue
        if isinstance(current_value, torch.Tensor) and _is_numeric(current_value):
            current_value = current_value.detach().item()
        if isinstance(baseline_value, torch.Tensor) and _is_numeric(baseline_value):
            baseline_value = baseline_value.detach().item()
        entry: Dict[str, Any] = {
            "key": key,
            "current": current_value,
            "baseline": baseline_value,
            "kind": "tensor" if tensor_values else "categorical",
        }
        if contract_changes:
            entry["contract_changes"] = contract_changes
        if _is_numeric(current_value) and _is_numeric(baseline_value):
            delta = current_value - baseline_value
            if abs(baseline_value) > 0.0:
                relative_change = delta / abs(baseline_value)
            else:
                relative_change = None
            entry.update(
                {
                    "kind": "numeric",
                    "delta": delta,
                    "relative_change": relative_change,
                }
            )
        changed_parameters.append(entry)

    total_changes = (
        len(changed_parameters) + len(added_parameters) + len(removed_parameters)
    )
    summary = (
        f"Parameter diff summary: {len(changed_parameters)} changed, "
        f"{len(added_parameters)} added, {len(removed_parameters)} removed."
    )
    return {
        "changed_parameters": changed_parameters,
        "added_parameters": added_parameters,
        "removed_parameters": removed_parameters,
        "summary": summary,
        "total_changes": total_changes,
    }


def generate_experiment_diff_report(
    current_snapshot: Dict[str, Any],
    baseline_snapshot: Dict[str, Any],
    current_config: Dict[str, Any],
    baseline_config: Dict[str, Any],
) -> str:
    """Generate markdown report summarizing parameter and config differences."""
    parameter_diff = compute_parameter_diff(current_snapshot, baseline_snapshot)
    current_config_flat = _flatten_mapping(current_config or {})
    baseline_config_flat = _flatten_mapping(baseline_config or {})

    config_added = sorted(set(current_config_flat.keys()) - set(baseline_config_flat.keys()))
    config_removed = sorted(set(baseline_config_flat.keys()) - set(current_config_flat.keys()))
    config_changed = []
    for key in sorted(set(current_config_flat.keys()) & set(baseline_config_flat.keys())):
        if current_config_flat[key] != baseline_config_flat[key]:
            config_changed.append(
                {
                    "key": key,
                    "current": current_config_flat[key],
                    "baseline": baseline_config_flat[key],
                }
            )

    lines = [
        "# Experiment Differential Report",
        "",
        "## Parameter Differences",
        parameter_diff["summary"],
        "",
        f"- Changed: {len(parameter_diff['changed_parameters'])}",
        f"- Added: {len(parameter_diff['added_parameters'])}",
        f"- Removed: {len(parameter_diff['removed_parameters'])}",
        "",
    ]

    if parameter_diff["changed_parameters"]:
        lines.append("### Changed Parameters")
        for item in parameter_diff["changed_parameters"]:
            if item.get("kind") == "numeric":
                lines.append(
                    f"- `{item['key']}`: {item['baseline']} -> {item['current']} "
                    f"(delta={item.get('delta')}, rel={item.get('relative_change')})"
                )
            else:
                lines.append(f"- `{item['key']}`: {item['baseline']} -> {item['current']}")
            if item.get("contract_changes"):
                changes = ", ".join(
                    f"{field}: {values['baseline']} -> {values['current']}"
                    for field, values in item["contract_changes"].items()
                )
                lines[-1] += f"; contract changes: {changes}"
        lines.append("")

    if parameter_diff["added_parameters"]:
        lines.append("### Added Parameters")
        for item in parameter_diff["added_parameters"]:
            lines.append(f"- `{item['key']}`: {item['value']}")
        lines.append("")

    if parameter_diff["removed_parameters"]:
        lines.append("### Removed Parameters")
        for item in parameter_diff["removed_parameters"]:
            lines.append(f"- `{item['key']}`: {item['value']}")
        lines.append("")

    lines.extend(
        [
            "## Configuration Differences",
            f"- Changed: {len(config_changed)}",
            f"- Added: {len(config_added)}",
            f"- Removed: {len(config_removed)}",
            "",
        ]
    )

    if config_changed:
        lines.append("### Changed Config Entries")
        for item in config_changed:
            lines.append(f"- `{item['key']}`: {item['baseline']} -> {item['current']}")
        lines.append("")

    if config_added:
        lines.append("### Added Config Entries")
        for key in config_added:
            lines.append(f"- `{key}`: {current_config_flat[key]}")
        lines.append("")

    if config_removed:
        lines.append("### Removed Config Entries")
        for key in config_removed:
            lines.append(f"- `{key}`: {baseline_config_flat[key]}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
