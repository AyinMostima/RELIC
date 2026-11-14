"""Checkpoint helpers for RELICSimulator."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
from copy import deepcopy
from collections.abc import Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Optional, cast
from uuid import uuid4

import torch

from relic.core.device_policy import get_device_overrides, is_cuda_available
from relic.core.functor.functor_base import FunctorBase
from relic.parameters import get_global_manager
from relic.parameters.ops.nn_params import capture_binding_values, restore_binding_values
from relic.core.state import UpdateMode
from relic.core.state.manager import COMMIT_CONTEXT_RESERVED_KEYS, validate_commit_context
from relic.core.state.types import (
    ConservationSummary,
    ElementConservationReport,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "relic.simulator.checkpoint"
CHECKPOINT_FORMAT_VERSION = 3
# Version 2 kept a ParameterManager binding held by a Functor as an entry of
# the Functor state_dict; version 3 saves it through the manager in the
# Functor entry's parameter_bindings. Restore reads version 2 through
# _upgrade_version_2_payload (data continuity); only version 3 is written.
_READABLE_FORMAT_VERSIONS = (2, CHECKPOINT_FORMAT_VERSION)
_TYPE_KEY = "__relic_checkpoint_type__"
_CHECKPOINT_BODY_FIELDS = {
    "format", "format_version", "created_utc", "runtime", "state_manager",
    "functors", "topology", "rng",
}
_DELIVERY_FIELDS = {
    "run_id", "accepted_step", "accepted_time", "operation", "resource",
    "parameters", "sources", "automatic",
}
# Delivery settings name where and how often accepted state is exported.
# They are incidental paths and cadences, not scientific content, so they
# are excluded from the checkpoint scientific contract digest.
_DELIVERY_CONFIG_KEYS = frozenset({
    "enable_checkpointing",
    "checkpoint_frequency",
    "checkpoint_directory",
    "enable_diagnostic_export",
    "diagnostic_output_frequency",
    "diagnostic_output_path",
})
_CORE_RUNTIME_FIELDS = (
    "state",
    "run_id",
    "current_time",
    "time_step",
    "step_count",
    "total_execution_time",
    "last_checkpoint_time",
    "time_series",
)
_ADDITIONAL_RUNTIME_FIELDS = (
    "max_time",
    "output_frequency",
    "checkpoint_frequency",
    "_adaptive_dt_enabled",
    "_adaptive_atol",
    "_adaptive_rtol",
    "_adaptive_dt_min",
    "_adaptive_dt_max",
    "_adaptive_monitor_vars",
    "_adaptive_next_dt",
    "_splitting_mode",
    "_splitting_policy",
    "_fast_order_mode",
    "_substep_overrides",
    "_multirate_exchanges",
    "_time_integration_method",
    "_implicit_midpoint_max_iter",
    "_implicit_midpoint_tolerance",
    "_resolve_cross_group_deps",
    "enable_auto_scale",
    "strict_scale_detection",
    "enable_conservation_checks",
    "strict_conservation_enabled",
    "_scale_check_history",
    "_last_state_reset_counter",
    "_initial_state",
    "_baseline_snapshot_reference",
    "_forcing_cursor",
    "_baseline_config_reference",
    "_baseline_snapshot_path",
)
# Fast-group splitting controls of RELICSimulator (CR-A-04): runtime field ->
# (simulator setting, computation.scheduling entry). The resolved value of each
# control is saved -- configured, or taken from the parameter database -- so an
# exact restart continues with the controls the saved run executed. A
# checkpoint written before a control existed has no field for it and restores
# with the documented default of its entry (ParameterSpec.default, not the
# current database value): lie, warn, forward, no sub-step overrides and no
# multirate exchanges, the behaviour of the scheduler that wrote it.
_SPLITTING_CONTROL_FIELDS = {
    "_splitting_mode": ("splitting_mode", "computation.scheduling.splitting_mode"),
    "_splitting_policy": ("splitting_policy", "computation.scheduling.splitting_policy"),
    "_fast_order_mode": ("fast_order", "computation.scheduling.fast_order"),
    "_substep_overrides": ("substep_overrides", "computation.scheduling.substep_overrides"),
    "_multirate_exchanges": ("multirate_exchanges", "computation.scheduling.multirate_exchanges"),
}
_STATE_MANAGER_FIELDS = {
    "state",
    "update_counter",
    "state_reset_counter",
    "diagnostics",
    "history",
    "conservation_history",
    "last_summary",
    "clip_loss_diagnostics",
    "auto_report_enabled",
    "provenance_transaction_active",
    "pending_provenance_events",
}


def _runtime_field_names() -> tuple[str, ...]:
    """Reuse the accepted-step transaction owner and add restart controls."""
    from .execution_loop import _STEP_TRANSACTION_ATTRIBUTES

    return tuple(
        dict.fromkeys(
            (
                *_STEP_TRANSACTION_ATTRIBUTES,
                *_CORE_RUNTIME_FIELDS,
                *_ADDITIONAL_RUNTIME_FIELDS,
            )
        )
    )


def _encode_checkpoint_value(value: Any, *, path: str) -> Any:
    """Convert runtime values to the strict weights-only checkpoint grammar."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        if _TYPE_KEY in value:
            raise ValueError(
                f"Checkpoint mapping at {path} uses reserved key '{_TYPE_KEY}'."
            )
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Unsupported checkpoint mapping key at {path}: "
                    f"{type(key).__name__}."
                )
            encoded[key] = _encode_checkpoint_value(
                item,
                path=f"{path}.{key}",
            )
        return encoded
    if isinstance(value, list):
        return [
            _encode_checkpoint_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return {
            _TYPE_KEY: "tuple",
            "items": [
                _encode_checkpoint_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, set):
        ordered = sorted(value, key=repr)
        return {
            _TYPE_KEY: "set",
            "items": [
                _encode_checkpoint_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(ordered)
            ],
        }
    if isinstance(value, UpdateMode):
        return {
            _TYPE_KEY: "UpdateMode",
            "value": value.value,
        }
    if isinstance(value, (ConservationSummary, ElementConservationReport)):
        return {
            _TYPE_KEY: type(value).__name__,
            "fields": {
                field.name: _encode_checkpoint_value(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
                for field in fields(value)
            },
        }
    if is_dataclass(value):
        raise TypeError(
            f"Unsupported checkpoint dataclass at {path}: "
            f"{type(value).__name__}."
        )
    raise TypeError(
        f"Unsupported checkpoint value at {path}: {type(value).__name__}."
    )


def _decode_checkpoint_value(value: Any, *, path: str) -> Any:
    """Decode only the explicit RELIC checkpoint grammar."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [
            _decode_checkpoint_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        raise TypeError(
            f"Invalid checkpoint value at {path}: {type(value).__name__}."
        )

    encoded_type = value.get(_TYPE_KEY)
    if encoded_type is None:
        return {
            key: _decode_checkpoint_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if set(value) != {_TYPE_KEY, "items"} and encoded_type in {"tuple", "set"}:
        raise ValueError(f"Invalid {encoded_type} payload at {path}.")
    if encoded_type == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Invalid tuple payload at {path}.")
        return tuple(
            _decode_checkpoint_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(items)
        )
    if encoded_type == "set":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Invalid set payload at {path}.")
        return {
            _decode_checkpoint_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(items)
        }
    if encoded_type == "UpdateMode":
        if set(value) != {_TYPE_KEY, "value"}:
            raise ValueError(f"Invalid UpdateMode payload at {path}.")
        try:
            return UpdateMode(value["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid UpdateMode value at {path}.") from exc
    if encoded_type in {"ConservationSummary", "ElementConservationReport"}:
        if set(value) != {_TYPE_KEY, "fields"}:
            raise ValueError(f"Invalid dataclass payload at {path}.")
        raw_fields = value.get("fields")
        if not isinstance(raw_fields, dict):
            raise ValueError(f"Invalid dataclass fields at {path}.")
        decoded_fields = {
            key: _decode_checkpoint_value(item, path=f"{path}.{key}")
            for key, item in raw_fields.items()
        }
        expected_type = (
            ConservationSummary
            if encoded_type == "ConservationSummary"
            else ElementConservationReport
        )
        expected_fields = {field.name for field in fields(expected_type)}
        if set(decoded_fields) != expected_fields:
            raise ValueError(
                f"Invalid {encoded_type} field set at {path}."
            )
        return expected_type(**decoded_fields)
    raise ValueError(
        f"Unsupported checkpoint type marker at {path}: {encoded_type!r}."
    )


def _update_payload_digest(digest: Any, value: Any) -> None:
    """Hash the strict checkpoint grammar deterministically."""
    if value is None:
        digest.update(b"none;")
        return
    if type(value) is bool:
        digest.update(b"bool:1;" if value else b"bool:0;")
        return
    if type(value) is int:
        digest.update(f"int:{value};".encode("ascii"))
        return
    if type(value) is float:
        digest.update(f"float:{value.hex()};".encode("ascii"))
        return
    if type(value) is str:
        encoded = value.encode("utf-8")
        digest.update(f"str:{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
        digest.update(b";")
        return
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"tensor:{tensor.dtype}:{list(tensor.shape)}:".encode("ascii"))
        try:
            raw = tensor.numpy().tobytes(order="C")
        except (TypeError, RuntimeError):
            raw = tensor.view(torch.int16).numpy().tobytes(order="C")
        digest.update(raw)
        digest.update(b";")
        return
    if isinstance(value, list):
        digest.update(f"list:{len(value)}:[".encode("ascii"))
        for item in value:
            _update_payload_digest(digest, item)
        digest.update(b"];")
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Checkpoint integrity mappings require string keys.")
        digest.update(f"dict:{len(value)}:{{".encode("ascii"))
        for key in sorted(value):
            _update_payload_digest(digest, key)
            _update_payload_digest(digest, value[key])
        digest.update(b"};")
        return
    raise TypeError(
        "Checkpoint integrity received unsupported value "
        f"{type(value).__name__}."
    )


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scientific_signature_value(value: Any, *, path: str) -> Any:
    """Normalize scientific contracts without serializing live objects."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Enum):
        return {
            "__relic_scientific_type__": "enum",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _scientific_signature_value(
                value.value,
                path=f"{path}.value",
            ),
        }
    if isinstance(value, Path):
        return {
            "__relic_scientific_type__": "path",
            "value": str(value),
        }
    if isinstance(value, torch.device):
        return {
            "__relic_scientific_type__": "device",
            "value": str(value),
        }
    if isinstance(value, torch.dtype):
        return {
            "__relic_scientific_type__": "dtype",
            "value": str(value),
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value, key=str):
            if not isinstance(key, str):
                raise TypeError(
                    f"Scientific contract mapping key at {path} is not text."
                )
            normalized[key] = _scientific_signature_value(
                value[key],
                path=f"{path}.{key}",
            )
        return normalized
    if isinstance(value, list):
        return [
            _scientific_signature_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return {
            "__relic_scientific_type__": "tuple",
            "items": [
                _scientific_signature_value(
                    item,
                    path=f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, (set, frozenset)):
        normalized_items = [
            _scientific_signature_value(item, path=f"{path}[]")
            for item in value
        ]
        return {
            "__relic_scientific_type__": "set",
            "items": sorted(normalized_items, key=repr),
        }
    if is_dataclass(value):
        return {
            "__relic_scientific_type__": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _scientific_signature_value(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
                for field in fields(value)
            },
        }
    if isinstance(value, type):
        return {
            "__relic_scientific_type__": "class",
            "value": f"{value.__module__}.{value.__qualname__}",
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return {
            "__relic_scientific_type__": "to_dict",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _scientific_signature_value(
                to_dict(),
                path=f"{path}.to_dict",
            ),
        }
    raise TypeError(
        f"Unsupported scientific contract value at {path}: "
        f"{type(value).__name__}."
    )


def _functor_scientific_contract(functor: Any) -> dict[str, Any]:
    metadata = getattr(functor, "metadata", None)
    metadata_value = (
        metadata.to_dict()
        if metadata is not None and callable(getattr(metadata, "to_dict", None))
        else metadata
    )
    namespace = None
    namespace_getter = getattr(functor, "_default_parameter_namespace", None)
    if callable(namespace_getter):
        namespace = namespace_getter()
    return {
        "kernel_identity": _scientific_signature_value(
            functor.get_kernel_identity(),
            path=f"functors.{functor.name}.kernel_identity",
        ),
        "config": _scientific_signature_value(
            getattr(functor, "_original_config", {}),
            path=f"functors.{functor.name}.config",
        ),
        "metadata": _scientific_signature_value(
            metadata_value,
            path=f"functors.{functor.name}.metadata",
        ),
        "parameter_values": _scientific_signature_value(
            getattr(functor, "parameter_values", {}),
            path=f"functors.{functor.name}.parameter_values",
        ),
        "parameter_namespace": (
            str(namespace) if namespace is not None else None
        ),
    }


def _collect_parameter_managers(
    functors: Mapping[str, Any],
) -> list[Any]:
    managers: list[Any] = []
    seen: set[int] = set()
    for functor in functors.values():
        manager = getattr(functor, "_parameter_manager", None)
        if manager is None:
            getter = getattr(functor, "get_parameter_manager", None)
            if callable(getter):
                # None means no manager is configured; retrieval failures propagate.
                manager = getter()
        if manager is not None and id(manager) not in seen:
            managers.append(manager)
            seen.add(id(manager))
    return managers


def _accessed_parameter_contract(
    functors: Mapping[str, Any],
    *,
    parameter_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    managers = _collect_parameter_managers(functors)
    if parameter_keys is None:
        keys = sorted(
            {
                str(key)
                for manager in managers
                for key in getattr(manager, "_access_history", set())
            }
        )
    else:
        keys = sorted(set(parameter_keys))
    contracts: dict[str, Any] = {}
    for key in keys:
        resolved_contract = None
        for manager in managers:
            getter = getattr(manager, "get_parameter", None)
            if not callable(getter):
                continue
            spec = getter(key)
            if spec is None:
                continue
            normalized = _scientific_signature_value(
                spec,
                path=f"parameters.{key}",
            )
            if (
                resolved_contract is not None
                and normalized != resolved_contract
            ):
                raise ValueError(
                    f"Conflicting ParameterManager specs for '{key}'."
                )
            resolved_contract = normalized
        contracts[key] = (
            resolved_contract
            if resolved_contract is not None
            else {"__relic_scientific_type__": "missing_parameter"}
        )
    return contracts


def _build_topology(
    simulator: Any,
    *,
    parameter_keys: Sequence[str] | None = None,
    runtime_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    functors = getattr(simulator, "functors", {})
    functor_types = {
        name: (
            f"{type(functor).__module__}.{type(functor).__qualname__}"
        )
        for name, functor in functors.items()
    }
    raw_groups = getattr(simulator, "_groups", {})
    groups = {
        str(group): list(members)
        for group, members in raw_groups.items()
    }
    dependency_graph = getattr(simulator, "dependency_graph", None)
    dependency_nodes: dict[str, Any] = {}
    dependency_edges: list[dict[str, Any]] = []
    if dependency_graph is not None:
        optional_dependencies = getattr(
            dependency_graph,
            "_optional_deps",
            {},
        )
        for name, node in dependency_graph.nodes.items():
            optional = set(optional_dependencies.get(name, set()))
            dependency_nodes[name] = {
                "required": sorted(set(node.dependencies).difference(optional)),
                "optional": sorted(optional),
                "lagged": sorted(set(node.lagged_dependencies)),
            }
        dependency_edges = sorted(
            (
                {
                    "source": str(edge.source),
                    "target": str(edge.target),
                    "dependency_type": str(edge.dependency_type.value),
                    "weight": float(edge.weight),
                    "required": bool(edge.required),
                }
                for edge in dependency_graph.edges
            ),
            key=lambda item: (
                item["source"],
                item["target"],
                item["dependency_type"],
                item["weight"],
                item["required"],
            ),
        )
    state_manager = getattr(simulator, "state_manager", None)
    state_manager_contract = {
        "type": (
            f"{type(state_manager).__module__}."
            f"{type(state_manager).__qualname__}"
            if state_manager is not None
            else None
        ),
        "schema": _scientific_signature_value(
            getattr(state_manager, "_schema", {}),
            path="state_manager.schema",
        ),
    }
    from .registration import _current_provider_view

    provider_view = _current_provider_view(simulator)
    raw_config = getattr(simulator, "config", {})
    scientific_config = (
        {
            key: value
            for key, value in raw_config.items()
            if key not in _DELIVERY_CONFIG_KEYS
        }
        if isinstance(raw_config, Mapping)
        else raw_config
    )
    scientific_contract = {
        "simulator_config": _scientific_signature_value(
            scientific_config,
            path="simulator.config",
        ),
        "state_manager": state_manager_contract,
        "functors": {
            name: dict(
                _functor_scientific_contract(functor),
                provider_identity=provider_view.provider(name).digest_record(),
            )
            for name, functor in sorted(functors.items())
        },
        "accessed_parameter_specs": _accessed_parameter_contract(
            functors,
            parameter_keys=parameter_keys,
        ),
    }
    return {
        "functor_names": sorted(functor_types),
        "functor_types": functor_types,
        "execution_order": list(getattr(simulator, "execution_order", [])),
        "groups": groups,
        "dependency_nodes": dependency_nodes,
        "dependency_edges": dependency_edges,
        "runtime_environment": (
            dict(runtime_environment)
            if runtime_environment is not None
            else _runtime_environment_signature(simulator)
        ),
        "scientific_contract": scientific_contract,
    }


# ---------------------------------------------------------------------------
# F03: compatible restart under an explicit compatibility certificate.
#
# Exact restart (no certificate) keeps whole-topology digest equality. A
# compatible restart is a separate, narrower claim: the caller supplies a
# certificate that names the exact source and target topology digests, lists
# the contract leaf paths that may differ, bounds numeric parameter drift,
# and maps renamed state variables with identical units, shape, and dtype.
# Implementation identity, graph structure, runtime environment, parameter
# specifications, and hidden/RNG state are never waivable.
# ---------------------------------------------------------------------------

COMPATIBILITY_CERTIFICATE_KIND = "relic.checkpoint.compatibility"
COMPATIBILITY_CERTIFICATE_VERSION = 1
_COMPATIBILITY_REJECTED_SECTIONS = frozenset({
    "functor_names",
    "functor_types",
    "execution_order",
    "groups",
    "dependency_nodes",
    "dependency_edges",
    "runtime_environment",
})
_COMPATIBILITY_REJECTED_PATH_TOKENS = (
    ".kernel_identity",
    ".provider_identity.implementation_digest",
    ".provider_identity.strict_identity",
    ".provider_identity.identity_rejection",
    "scientific_contract.state_manager.type",
    "scientific_contract.accessed_parameter_specs",
)


@dataclass(frozen=True)
class RestartCompatibilityCertificate:
    """Explicit kappa for one source checkpoint -> one target simulator.

    Fields:
        source_topology_sha256: digest of the saved checkpoint topology.
        target_topology_sha256: digest of the live simulator topology.
        allowed_contract_paths: dotted leaf paths (or prefixes) of the
            scientific contract that may differ, for example
            ``scientific_contract.functors.writer.parameter_values.gain`` or
            ``scientific_contract.simulator_config.output_frequency``.
        parameter_relative_tolerance: bound on |saved - current| relative to
            max(|saved|, 1e-12) for every differing numeric parameter value.
        state_variable_map: saved state name -> target state name; mapped
            variables must keep units, shape, and dtype.
        forcing_cursor_policy: ``"exact"`` restores the saved cursor;
            ``"clear"`` restores without a bound forcing sequence.
    Hidden Functor state and RNG state are always restored exactly.
    """

    source_topology_sha256: str
    target_topology_sha256: str
    allowed_contract_paths: tuple[str, ...] = ()
    parameter_relative_tolerance: float = 0.0
    state_variable_map: Mapping[str, str] = field(default_factory=dict)
    forcing_cursor_policy: str = field(kw_only=True)
    certificate_kind: str = COMPATIBILITY_CERTIFICATE_KIND
    version: int = COMPATIBILITY_CERTIFICATE_VERSION

    def __post_init__(self) -> None:
        if self.certificate_kind != COMPATIBILITY_CERTIFICATE_KIND:
            raise ValueError("Unknown compatibility certificate kind.")
        if self.version != COMPATIBILITY_CERTIFICATE_VERSION:
            raise ValueError("Unsupported compatibility certificate version.")
        for label, digest in (
            ("source", self.source_topology_sha256),
            ("target", self.target_topology_sha256),
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Certificate {label} topology digest must be 64 hex characters.")
        if not isinstance(self.allowed_contract_paths, tuple) or any(
            not isinstance(item, str) or not item for item in self.allowed_contract_paths
        ):
            raise TypeError("allowed_contract_paths must be a tuple of nonempty strings.")
        tolerance = self.parameter_relative_tolerance
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("parameter_relative_tolerance must be numeric.")
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
            raise ValueError("parameter_relative_tolerance must be finite and nonnegative.")
        if not isinstance(self.state_variable_map, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str) or not key or not value
            for key, value in self.state_variable_map.items()
        ):
            raise TypeError("state_variable_map must map nonempty names to nonempty names.")
        targets = list(self.state_variable_map.values())
        if len(set(targets)) != len(targets):
            raise ValueError("state_variable_map must not map two saved names to one target.")
        if self.forcing_cursor_policy not in ("exact", "clear"):
            raise ValueError("forcing_cursor_policy must be 'exact' or 'clear'.")

    def sha256(self) -> str:
        return _payload_sha256({
            "certificate_kind": self.certificate_kind,
            "version": self.version,
            "source_topology_sha256": self.source_topology_sha256,
            "target_topology_sha256": self.target_topology_sha256,
            "allowed_contract_paths": list(self.allowed_contract_paths),
            "parameter_relative_tolerance": float(self.parameter_relative_tolerance),
            "state_variable_map": dict(self.state_variable_map),
            "forcing_cursor_policy": self.forcing_cursor_policy,
        })


def _topology_sha256(encoded_topology: Mapping[str, Any]) -> str:
    return _payload_sha256({"topology": encoded_topology})


def _topology_leaf_differences(
    saved: Any,
    current: Any,
    *,
    path: str,
    out: list[tuple[str, Any, Any]],
) -> None:
    """Collect (path, saved leaf, current leaf) for every differing leaf."""
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        if _TYPE_KEY in saved or _TYPE_KEY in current:
            if _payload_sha256({"leaf": saved}) != _payload_sha256({"leaf": current}):
                out.append((path, saved, current))
            return
        for key in sorted(set(saved) | set(current)):
            child = f"{path}.{key}" if path else str(key)
            if key not in saved or key not in current:
                out.append((child, saved.get(key), current.get(key)))
                continue
            _topology_leaf_differences(saved[key], current[key], path=child, out=out)
        return
    if isinstance(saved, list) and isinstance(current, list):
        if len(saved) != len(current):
            out.append((path, saved, current))
            return
        for index, (left, right) in enumerate(zip(saved, current)):
            _topology_leaf_differences(left, right, path=f"{path}[{index}]", out=out)
        return
    if _payload_sha256({"leaf": saved}) != _payload_sha256({"leaf": current}):
        out.append((path, saved, current))


def _path_is_allowed(path: str, allowed: Sequence[str]) -> bool:
    return any(path == item or path.startswith(item + ".") or path.startswith(item + "[")
               for item in allowed)


def _derived_digest_path(path: str) -> Optional[tuple[str, str]]:
    """Map a derived provider digest path to (functor path, derivation)."""
    for suffix, derivation in (
        (".provider_identity.binding_digest", "binding"),
        (".provider_identity.contract_digest", "contract"),
    ):
        if path.endswith(suffix):
            return path[: -len(suffix)], derivation
    return None


def _is_numeric_leaf(encoded: Any) -> bool:
    value = _decode_checkpoint_value(encoded, path="certificate.leaf")
    if isinstance(value, torch.Tensor):
        return bool(value.is_floating_point() or not value.is_complex())
    return type(value) in (int, float)


def _numeric_within_tolerance(saved: Any, current: Any, tolerance: float) -> bool:
    saved_value = _decode_checkpoint_value(saved, path="certificate.saved")
    current_value = _decode_checkpoint_value(current, path="certificate.current")
    if isinstance(saved_value, torch.Tensor) and isinstance(current_value, torch.Tensor):
        if saved_value.shape != current_value.shape or saved_value.dtype != current_value.dtype:
            return False
        saved_f = saved_value.detach().cpu().to(torch.float64)
        current_f = current_value.detach().cpu().to(torch.float64)
        scale = torch.clamp(saved_f.abs(), min=1e-12)
        return bool(((saved_f - current_f).abs() <= tolerance * scale).all())
    if (
        type(saved_value) in (int, float)
        and type(current_value) in (int, float)
    ):
        return abs(float(saved_value) - float(current_value)) <= tolerance * max(abs(float(saved_value)), 1e-12)
    return False


def _audit_compatible_topology(
    certificate: RestartCompatibilityCertificate,
    saved_encoded: Mapping[str, Any],
    current_encoded: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the accepted difference audit or raise on any unwaivable drift."""
    saved_sha = _topology_sha256(saved_encoded)
    current_sha = _topology_sha256(current_encoded)
    if certificate.source_topology_sha256 != saved_sha:
        raise ValueError(
            "Compatibility certificate source topology digest does not match the checkpoint."
        )
    if certificate.target_topology_sha256 != current_sha:
        raise ValueError(
            "Compatibility certificate target topology digest does not match the active simulator."
        )
    differences: list[tuple[str, Any, Any]] = []
    _topology_leaf_differences(saved_encoded, current_encoded, path="", out=differences)
    rejected: list[str] = []
    accepted: list[dict[str, Any]] = []
    parameter_leaves: list[tuple[str, Any, Any]] = []
    for path, saved_leaf, current_leaf in differences:
        section = path.split(".", 1)[0].split("[", 1)[0]
        if section in _COMPATIBILITY_REJECTED_SECTIONS or any(
            token in path for token in _COMPATIBILITY_REJECTED_PATH_TOKENS
        ):
            rejected.append(path)
            continue
        derived = _derived_digest_path(path)
        if derived is not None:
            accepted.append({"path": path, "reason": f"derived_{derived[1]}_digest"})
            continue
        if not _path_is_allowed(path, certificate.allowed_contract_paths):
            rejected.append(path)
            continue
        if ".parameter_values." in path:
            parameter_leaves.append((path, saved_leaf, current_leaf))
        accepted.append({"path": path, "reason": "certified"})
    if rejected:
        raise ValueError(
            "Compatible restart rejected: uncertified or unwaivable contract "
            f"differences at {', '.join(sorted(rejected))}."
        )
    # A derived digest may only differ when the differences it derives from
    # were certified for the same functor.
    for item in accepted:
        if item["reason"].startswith("derived_"):
            prefix = item["path"].rsplit(".provider_identity.", 1)[0]
            if not any(
                other.startswith(prefix + ".") for other in (
                    entry["path"] for entry in accepted if entry["reason"] == "certified"
                )
            ):
                raise ValueError(
                    f"Compatible restart rejected: {item['path']} changed without a "
                    "certified underlying contract difference."
                )
    for path, saved_leaf, current_leaf in parameter_leaves:
        if _is_numeric_leaf(saved_leaf) and _is_numeric_leaf(current_leaf):
            if not _numeric_within_tolerance(
                saved_leaf, current_leaf, float(certificate.parameter_relative_tolerance),
            ):
                raise ValueError(
                    f"Compatible restart rejected: parameter drift at {path} exceeds the "
                    f"certified relative tolerance {certificate.parameter_relative_tolerance!r}."
                )
        elif path not in certificate.allowed_contract_paths:
            # A non-numeric parameter value has no tolerance; only an exact
            # path certification (not a prefix) can accept its change.
            raise ValueError(
                f"Compatible restart rejected: non-numeric parameter change at {path} "
                "requires that exact path in allowed_contract_paths."
            )
    return {
        "certificate_sha256": certificate.sha256(),
        "source_topology_sha256": saved_sha,
        "target_topology_sha256": current_sha,
        "accepted_differences": accepted,
    }


def _saved_schema_units(saved_topology: Mapping[str, Any], name: str) -> Optional[str]:
    schema = saved_topology.get("scientific_contract", {}).get("state_manager", {}).get("schema")
    if not isinstance(schema, Mapping) or name not in schema:
        return None
    entry = schema[name]
    if isinstance(entry, Mapping):
        fields_map = entry.get("fields")
        if isinstance(fields_map, Mapping) and fields_map.get("units") is not None:
            return str(fields_map["units"])
    return None


def _apply_state_variable_map(
    certificate: RestartCompatibilityCertificate,
    simulator: Any,
    saved_topology: Mapping[str, Any],
    decoded_state_manager: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rename saved state variables per the certificate with unit/shape checks."""
    mapping = dict(certificate.state_variable_map)
    if not mapping:
        return []
    state = dict(decoded_state_manager["state"])
    current_state = simulator.state_manager.get_state()
    current_schema = getattr(simulator.state_manager, "_schema", {})
    applied: list[dict[str, Any]] = []
    for saved_name, target_name in mapping.items():
        if saved_name not in state:
            raise ValueError(
                f"Compatibility certificate maps '{saved_name}', which the checkpoint does not carry."
            )
        if target_name in state and target_name != saved_name:
            raise ValueError(
                f"Compatibility certificate target '{target_name}' collides with a saved variable."
            )
        saved_units = _saved_schema_units(saved_topology, saved_name)
        target_spec = current_schema.get(target_name)
        target_units = None if target_spec is None or target_spec.units is None else str(target_spec.units)
        if saved_units is not None and target_units is not None and saved_units != target_units:
            raise ValueError(
                f"Compatibility certificate cannot map '{saved_name}' [{saved_units}] to "
                f"'{target_name}' [{target_units}]: units differ."
            )
        tensor = state[saved_name]
        current = current_state.get(target_name)
        if isinstance(current, torch.Tensor) and isinstance(tensor, torch.Tensor):
            if tuple(current.shape) != tuple(tensor.shape) or current.dtype != tensor.dtype:
                raise ValueError(
                    f"Compatibility certificate cannot map '{saved_name}' to '{target_name}': "
                    "shape or dtype differs from the live variable."
                )
        state[target_name] = state.pop(saved_name)
        applied.append({
            "saved": saved_name, "target": target_name,
            "units": target_units if target_units is not None else saved_units,
        })
    decoded_state_manager["state"] = state
    return applied


def compute_restart_topology_digests(
    simulator: Any,
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, str]:
    """Digests a certificate author binds: saved checkpoint vs live simulator.

    Both digests are computed exactly as ``restore_checkpoint`` computes them,
    so a certificate minted from these values names real artifacts rather
    than a self-declared identity.
    """
    source = Path(checkpoint_path).resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    payload_sha = _validate_checkpoint_payload(checkpoint)
    saved_topology = _decode_checkpoint_value(checkpoint["topology"], path="topology")
    parameter_specs = saved_topology["scientific_contract"]["accessed_parameter_specs"]
    current_topology = _build_topology(
        simulator,
        parameter_keys=tuple(parameter_specs),
        runtime_environment=_runtime_environment_signature(simulator),
    )
    return {
        "payload_sha256": payload_sha,
        "source_topology_sha256": _topology_sha256(
            _encode_checkpoint_value(saved_topology, path="topology")
        ),
        "target_topology_sha256": _topology_sha256(
            _encode_checkpoint_value(current_topology, path="topology")
        ),
    }


# ---------------------------------------------------------------------------
# F04: cross-graph migration replay under an explicit migration certificate.
#
# A migration maps one saved graph onto one target graph inside a restricted
# family: mapped Functors keep their implementation identity, type, group,
# and pairwise dependencies on the common domain; retired and new Functors
# and dropped state variables must be declared. History (time series and
# provenance of retired variables) is preserved, never disposed. The saved
# payload is transformed into the target namespace, audited with the
# compatible-restart machinery, validated, and committed atomically.
# ---------------------------------------------------------------------------

GRAPH_MIGRATION_CERTIFICATE_KIND = "relic.checkpoint.graph_migration"
GRAPH_MIGRATION_CERTIFICATE_VERSION = 1


def _require_name_map(mapping: Any, label: str, duplicate_message: str) -> list[str]:
    """Validate a nonempty-name to nonempty-name mapping and return its targets."""
    if not isinstance(mapping, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) or not key or not value
        for key, value in mapping.items()
    ):
        raise TypeError(f"{label} must map nonempty names to nonempty names.")
    targets = list(mapping.values())
    if len(set(targets)) != len(targets):
        raise ValueError(duplicate_message)
    return targets


def _require_name_tuple(names: Any, label: str) -> None:
    """Validate a tuple of nonempty strings."""
    if not isinstance(names, tuple) or any(not isinstance(n, str) or not n for n in names):
        raise TypeError(f"{label} must be a tuple of nonempty strings.")


def _require_relative_tolerance(tolerance: Any) -> None:
    """Validate a finite, nonnegative numeric relative tolerance."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise TypeError("parameter_relative_tolerance must be numeric.")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("parameter_relative_tolerance must be finite and nonnegative.")


@dataclass(frozen=True)
class GraphMigrationCertificate:
    """Explicit r_* binding one saved graph to one target graph.

    Fields:
        source_topology_sha256 / target_topology_sha256: real artifacts.
        functor_map: saved Functor name -> target Functor name (injective).
        retired_functors: saved Functors without a target (state discarded,
            history preserved).
        new_functors: target Functors without a source (start from their
            live initialization; their substreams are fresh).
        state_variable_map: saved state name -> target state name.
        retired_state_variables: saved state variables dropped from the
            live state (their time series are preserved).
        allowed_contract_paths / parameter_relative_tolerance: as in
            RestartCompatibilityCertificate, evaluated on the target names.
        replay_mode: "switch" continues forward on the target graph;
            "counterfactual" replays from a historical checkpoint and
            requires the saved history to be complete.
    """

    source_topology_sha256: str
    target_topology_sha256: str
    functor_map: Mapping[str, str]
    retired_functors: tuple[str, ...] = ()
    new_functors: tuple[str, ...] = ()
    state_variable_map: Mapping[str, str] = field(default_factory=dict)
    retired_state_variables: tuple[str, ...] = ()
    allowed_contract_paths: tuple[str, ...] = ()
    parameter_relative_tolerance: float = 0.0
    replay_mode: str = field(kw_only=True)
    new_functor_rng: str = field(kw_only=True)
    forcing_cursor_policy: str = field(kw_only=True)
    certificate_kind: str = GRAPH_MIGRATION_CERTIFICATE_KIND
    version: int = GRAPH_MIGRATION_CERTIFICATE_VERSION

    def __post_init__(self) -> None:
        if self.certificate_kind != GRAPH_MIGRATION_CERTIFICATE_KIND:
            raise ValueError("Unknown graph migration certificate kind.")
        if self.version != GRAPH_MIGRATION_CERTIFICATE_VERSION:
            raise ValueError("Unsupported graph migration certificate version.")
        for label, digest in (
            ("source", self.source_topology_sha256),
            ("target", self.target_topology_sha256),
        ):
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Certificate {label} topology digest must be 64 hex characters.")
        targets = _require_name_map(
            self.functor_map,
            "functor_map",
            "functor_map must not map two saved Functors to one target (double writer).",
        )
        for label, names in (
            ("retired_functors", self.retired_functors),
            ("new_functors", self.new_functors),
            ("retired_state_variables", self.retired_state_variables),
            ("allowed_contract_paths", self.allowed_contract_paths),
        ):
            _require_name_tuple(names, label)
        if set(self.retired_functors) & set(self.functor_map):
            raise ValueError("A saved Functor cannot be both mapped and retired.")
        if set(self.new_functors) & set(targets):
            raise ValueError("A target Functor cannot be both new and a mapping target.")
        _require_name_map(
            self.state_variable_map,
            "state_variable_map",
            "state_variable_map must not map two saved names to one target.",
        )
        if set(self.retired_state_variables) & set(self.state_variable_map):
            raise ValueError("A saved state variable cannot be both mapped and retired.")
        _require_relative_tolerance(self.parameter_relative_tolerance)
        if self.replay_mode not in ("switch", "counterfactual"):
            raise ValueError("replay_mode must be 'switch' or 'counterfactual'.")
        if self.new_functor_rng != "fresh_substream":
            raise ValueError("new_functor_rng must be 'fresh_substream'; RNG coupling of new Functors is explicit.")
        if self.forcing_cursor_policy not in ("exact", "clear"):
            raise ValueError("forcing_cursor_policy must be 'exact' or 'clear'.")

    def sha256(self) -> str:
        return _payload_sha256({
            "certificate_kind": self.certificate_kind,
            "version": self.version,
            "source_topology_sha256": self.source_topology_sha256,
            "target_topology_sha256": self.target_topology_sha256,
            "functor_map": dict(self.functor_map),
            "retired_functors": list(self.retired_functors),
            "new_functors": list(self.new_functors),
            "state_variable_map": dict(self.state_variable_map),
            "retired_state_variables": list(self.retired_state_variables),
            "allowed_contract_paths": list(self.allowed_contract_paths),
            "parameter_relative_tolerance": float(self.parameter_relative_tolerance),
            "replay_mode": self.replay_mode,
            "new_functor_rng": self.new_functor_rng,
            "forcing_cursor_policy": self.forcing_cursor_policy,
        })


def _rename_keys(mapping: Mapping[str, Any], names: Mapping[str, str], drop: Collection[str] = ()) -> dict[str, Any]:
    dropped = set(drop)
    return {
        names.get(key, key): value
        for key, value in mapping.items()
        if key not in dropped
    }


def _check_migration_common_domain(
    certificate: GraphMigrationCertificate,
    saved_topology: Mapping[str, Any],
    current_topology: Mapping[str, Any],
) -> None:
    """Enforce the graph family on the common domain of mapped Functors."""
    saved_names = set(saved_topology["functor_names"])
    current_names = set(current_topology["functor_names"])
    mapped = dict(certificate.functor_map)
    unknown_sources = sorted(set(mapped) - saved_names)
    if unknown_sources:
        raise ValueError(f"Migration maps saved Functors the checkpoint does not carry: {unknown_sources}.")
    unknown_targets = sorted(set(mapped.values()) - current_names)
    if unknown_targets:
        raise ValueError(f"Migration maps onto Functors the target graph does not register: {unknown_targets}.")
    undeclared_retired = sorted(saved_names - set(mapped) - set(certificate.retired_functors))
    if undeclared_retired:
        raise ValueError(
            "Migration leaves saved Functors without a target or a retirement "
            f"declaration: {undeclared_retired}."
        )
    unknown_retired = sorted(set(certificate.retired_functors) - saved_names)
    if unknown_retired:
        raise ValueError(f"Migration retires Functors the checkpoint does not carry: {unknown_retired}.")
    undeclared_new = sorted(current_names - set(mapped.values()) - set(certificate.new_functors))
    if undeclared_new:
        raise ValueError(
            "Target graph registers Functors the migration neither maps nor "
            f"declares new: {undeclared_new}."
        )
    unknown_new = sorted(set(certificate.new_functors) - current_names)
    if unknown_new:
        raise ValueError(f"Migration declares new Functors the target graph does not register: {unknown_new}.")

    saved_types = saved_topology["functor_types"]
    current_types = current_topology["functor_types"]
    saved_groups = {
        member: group for group, members in saved_topology["groups"].items() for member in members
    }
    current_groups = {
        member: group for group, members in current_topology["groups"].items() for member in members
    }
    saved_nodes = saved_topology["dependency_nodes"]
    current_nodes = current_topology["dependency_nodes"]
    targets = set(mapped.values())
    for source_name, target_name in mapped.items():
        if saved_types.get(source_name) != current_types.get(target_name):
            raise ValueError(
                f"Migration leaves the graph family: '{source_name}' ({saved_types.get(source_name)}) "
                f"cannot map onto '{target_name}' ({current_types.get(target_name)})."
            )
        if saved_groups.get(source_name) != current_groups.get(target_name):
            raise ValueError(
                f"Migration changes the execution group of '{source_name}' -> '{target_name}'."
            )
        for relation in ("required", "optional", "lagged"):
            saved_relation = {
                mapped[dep] for dep in saved_nodes.get(source_name, {}).get(relation, []) if dep in mapped
            }
            current_relation = {
                dep for dep in current_nodes.get(target_name, {}).get(relation, []) if dep in targets
            }
            if saved_relation != current_relation:
                raise ValueError(
                    f"Migration breaks the {relation} dependency relation of "
                    f"'{source_name}' -> '{target_name}' on the common domain."
                )


def _migrate_saved_payload(
    certificate: GraphMigrationCertificate,
    simulator: Any,
    *,
    saved_topology: Mapping[str, Any],
    current_topology: Mapping[str, Any],
    decoded_runtime: Mapping[str, Any],
    decoded_state_manager: Mapping[str, Any],
    encoded_functors: Mapping[str, Any],
) -> dict[str, Any]:
    """Transform the saved payload into the target namespace (validate first)."""
    _check_migration_common_domain(certificate, saved_topology, current_topology)
    functor_map = dict(certificate.functor_map)
    state_map = dict(certificate.state_variable_map)
    retired_state = set(certificate.retired_state_variables)
    saved_state = decoded_state_manager["state"]
    live_state = simulator.state_manager.get_state()
    unknown_state = sorted((set(state_map) | retired_state) - set(saved_state))
    if unknown_state:
        raise ValueError(f"Migration names saved state variables the checkpoint does not carry: {unknown_state}.")
    mapped_state: dict[str, str] = {
        name: str(state_map.get(name) or name) for name in saved_state if name not in retired_state
    }
    lost = sorted(name for name, target in mapped_state.items() if target not in live_state)
    if lost:
        raise ValueError(
            "Migration would silently drop saved state variables without a "
            f"target or a retirement declaration: {lost}."
        )
    collisions = sorted(
        target for target in mapped_state.values()
        if list(mapped_state.values()).count(target) > 1
    )
    if collisions:
        raise ValueError(f"Migration maps two saved state variables onto one target: {sorted(set(collisions))}.")
    current_schema = getattr(simulator.state_manager, "_schema", {})
    for saved_name, target_name in mapped_state.items():
        saved_units = _saved_schema_units(saved_topology, saved_name)
        spec = current_schema.get(target_name)
        target_units = None if spec is None or spec.units is None else str(spec.units)
        if saved_units is not None and target_units is not None and saved_units != target_units:
            raise ValueError(
                f"Migration cannot map '{saved_name}' [{saved_units}] onto "
                f"'{target_name}' [{target_units}]: units differ."
            )
        saved_tensor = saved_state[saved_name]
        live_tensor = live_state[target_name]
        if isinstance(saved_tensor, torch.Tensor) and isinstance(live_tensor, torch.Tensor):
            if tuple(saved_tensor.shape) != tuple(live_tensor.shape) or saved_tensor.dtype != live_tensor.dtype:
                raise ValueError(
                    f"Migration cannot map '{saved_name}' onto '{target_name}': "
                    "shape or dtype differs from the live variable."
                )
    if certificate.replay_mode == "counterfactual":
        time_series = decoded_runtime.get("time_series", {})
        missing_history = sorted(
            name for name in mapped_state
            if not time_series.get(name)
        )
        if int(decoded_runtime.get("step_count", 0)) <= 0 or missing_history:
            raise ValueError(
                "Migration cannot claim a complete counterfactual replay: the "
                f"checkpoint carries no history for {missing_history or 'any step'}."
            )

    # Topology in the target namespace: structure and new Functors come from
    # the live target; mapped Functor contracts come from the checkpoint so
    # every remaining difference is audited under the certificate.
    transformed_topology = deepcopy(dict(current_topology))
    saved_contract = saved_topology["scientific_contract"]
    transformed_functors_contract = dict(transformed_topology["scientific_contract"]["functors"])
    for source_name, target_name in functor_map.items():
        entry = deepcopy(dict(saved_contract["functors"][source_name]))
        if source_name != target_name:
            # A certified rename changes exactly the declared name and the
            # two name-bearing provider digests; every other contract field
            # (kernel identity, implementation digest, config, parameters,
            # remaining metadata) is still audited against the checkpoint.
            current_entry = transformed_functors_contract[target_name]
            metadata = entry.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("name") == source_name:
                entry["metadata"] = dict(metadata, name=target_name)
            identity = dict(entry.get("provider_identity") or {})
            current_identity = current_entry.get("provider_identity") or {}
            for digest_name in ("contract_digest", "binding_digest"):
                if digest_name in current_identity:
                    identity[digest_name] = current_identity[digest_name]
            entry["provider_identity"] = identity
        transformed_functors_contract[target_name] = entry
    transformed_topology["scientific_contract"] = dict(
        transformed_topology["scientific_contract"],
        functors=transformed_functors_contract,
        simulator_config=saved_contract["simulator_config"],
    )

    # Functor state: renamed for mapped, discarded for retired, fresh (live)
    # for new Functors so the commit loads them unchanged.
    transformed_functors: dict[str, Any] = {}
    for source_name, target_name in functor_map.items():
        transformed_functors[target_name] = encoded_functors[source_name]
    for name in certificate.new_functors:
        transformed_functors[name] = _capture_functor_entry(name, simulator.functors[name])

    # StateManager snapshot: renamed/dropped state, live values for new
    # variables, keys renamed inside diagnostics and history snapshots.
    transformed_state = {
        target: saved_state[source] for source, target in mapped_state.items()
    }
    for name, tensor in live_state.items():
        transformed_state.setdefault(name, tensor.detach().clone())
    transformed_snapshot = dict(decoded_state_manager)
    transformed_snapshot["state"] = transformed_state
    for field_name in ("diagnostics", "clip_loss_diagnostics"):
        value = transformed_snapshot.get(field_name)
        if isinstance(value, dict):
            transformed_snapshot[field_name] = _rename_keys(value, state_map)
    history = transformed_snapshot.get("history")
    if isinstance(history, list):
        transformed_snapshot["history"] = [
            _rename_keys(entry, state_map) if isinstance(entry, dict) else entry
            for entry in history
        ]

    # Runtime: time series are renamed but never disposed (retired variables
    # keep their history); per-step bookkeeping follows the maps.
    transformed_runtime = dict(decoded_runtime)
    time_series = transformed_runtime.get("time_series")
    if isinstance(time_series, dict):
        transformed_runtime["time_series"] = _rename_keys(time_series, state_map)
    sources = transformed_runtime.get("_step_update_sources")
    if isinstance(sources, dict):
        transformed_runtime["_step_update_sources"] = {
            state_map.get(var, var): (
                (functor_map.get(record[0], record[0]), *record[1:])
                if isinstance(record, tuple) and record else record
            )
            for var, record in sources.items()
        }
    increments = transformed_runtime.get("_step_functor_increments")
    if isinstance(increments, dict):
        transformed_runtime["_step_functor_increments"] = {
            state_map.get(var, var): (
                {functor_map.get(fn, fn): inc for fn, inc in record.items()}
                if isinstance(record, dict) else record
            )
            for var, record in increments.items()
        }
    evicted_counts = transformed_runtime.get("_provenance_evicted_counts")
    if isinstance(evicted_counts, dict):
        transformed_runtime["_provenance_evicted_counts"] = _rename_keys(evicted_counts, state_map)
    detail = transformed_runtime.get("_step_provenance_detail")
    if isinstance(detail, dict):
        transformed_runtime["_step_provenance_detail"] = {
            state_map.get(var, var): [
                (functor_map.get(entry[0], entry[0]), *entry[1:])
                if isinstance(entry, tuple) and entry else entry
                for entry in records
            ] if isinstance(records, list) else records
            for var, records in detail.items()
        }
    initial_state = transformed_runtime.get("_initial_state")
    if isinstance(initial_state, dict):
        transformed_runtime["_initial_state"] = _rename_keys(initial_state, state_map, retired_state)
    if certificate.forcing_cursor_policy == "clear":
        transformed_runtime["_forcing_cursor"] = None

    return {
        "topology": transformed_topology,
        "functors": transformed_functors,
        "state_manager": transformed_snapshot,
        "runtime": transformed_runtime,
        "mapped_state_variables": [
            {"saved": source, "target": target} for source, target in mapped_state.items() if source != target
        ],
        "preserved_history_variables": sorted(
            name for name in (decoded_runtime.get("time_series") or {}) if name in retired_state
        ),
    }


def _mismatched_topology_sections(
    saved_topology: Mapping[str, Any],
    current_topology: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return deterministic top-level checkpoint topology differences."""

    mismatched: list[str] = []
    for section in sorted(set(saved_topology) | set(current_topology)):
        if section not in saved_topology or section not in current_topology:
            mismatched.append(str(section))
            continue
        saved_sha = _payload_sha256(
            {"topology_section": saved_topology[section]}
        )
        current_sha = _payload_sha256(
            {"topology_section": current_topology[section]}
        )
        if saved_sha != current_sha:
            mismatched.append(str(section))
    return tuple(mismatched)


def _device_signature(value: Any, *, path: str) -> dict[str, Any]:
    """Normalize one resolved device without re-reading raw configuration."""
    try:
        device = value if isinstance(value, torch.device) else torch.device(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"Resolved device at '{path}' is invalid.") from exc
    return {
        "type": device.type,
        "index": device.index,
    }


def _runtime_environment_signature(simulator: Any) -> dict[str, Any]:
    """Capture the canonical resolved device and distributed environment."""
    get_info = getattr(simulator, "get_device_info", None)
    if not callable(get_info):
        raise AttributeError(
            "Simulator does not expose the canonical get_device_info API."
        )
    info = get_info()
    if not isinstance(info, Mapping):
        raise TypeError("Simulator device information must be a mapping.")
    required = {
        "device",
        "device_type",
        "device_index",
        "distributed_enabled",
        "rank",
        "world_size",
    }
    missing = sorted(required.difference(info))
    if missing:
        raise ValueError(
            "Simulator device information is missing fields: "
            + ", ".join(missing)
        )
    resolved_device = _device_signature(
        info["device"],
        path="runtime_environment.resolved_device",
    )
    if (
        info["device_type"] != resolved_device["type"]
        or info["device_index"] != resolved_device["index"]
    ):
        raise ValueError(
            "Simulator resolved device information is internally inconsistent."
        )
    distributed_enabled = info["distributed_enabled"]
    if type(distributed_enabled) is not bool:
        raise TypeError("distributed_enabled must be a bool.")
    rank = info["rank"]
    world_size = info["world_size"]
    for name, value in (("rank", rank), ("world_size", world_size)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise TypeError(f"Distributed {name} must be an integer or None.")
    if (rank is None) != (world_size is None):
        raise ValueError(
            "Distributed rank and world_size must be present together."
        )
    distributed_active = rank is not None
    if distributed_active:
        if not distributed_enabled:
            raise ValueError(
                "Active distributed runtime is not marked as enabled."
            )
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("Distributed rank/world_size values are invalid.")

    effective_device = getattr(simulator, "get_effective_device", None)
    functor_devices: dict[str, Any] = {}
    for name in sorted(getattr(simulator, "functors", {})):
        resolved = (
            effective_device(name)
            if callable(effective_device)
            else getattr(simulator.functors[name], "device", info["device"])
        )
        functor_devices[name] = _device_signature(
            resolved,
            path=f"runtime_environment.functor_devices.{name}",
        )
    module_devices = {
        str(name): _device_signature(
            device,
            path=f"runtime_environment.module_device_overrides.{name}",
        )
        for name, device in sorted(get_device_overrides().items())
    }
    return {
        "resolved_device": resolved_device,
        "functor_devices": functor_devices,
        "module_device_overrides": module_devices,
        "distributed_enabled": distributed_enabled,
        "distributed_active": distributed_active,
        "rank": rank,
        "world_size": world_size,
    }


def _require_single_process_checkpoint(
    runtime_environment: Mapping[str, Any],
) -> None:
    """Reject multi-rank use until the canonical format gains coordination."""
    if (
        runtime_environment["distributed_active"]
        and runtime_environment["world_size"] > 1
    ):
        raise RuntimeError(
            "The checkpoint format currently supports single-process "
            "runtime only."
        )


def _uses_cuda_runtime(simulator: Any) -> bool:
    global_device = getattr(simulator, "device", torch.device("cpu"))
    if _device_signature(
        global_device,
        path="runtime_environment.resolved_device",
    )["type"] == "cuda":
        return True
    effective_device = getattr(simulator, "get_effective_device", None)
    for name, functor in getattr(simulator, "functors", {}).items():
        resolved = (
            effective_device(name)
            if callable(effective_device)
            else getattr(functor, "device", global_device)
        )
        if _device_signature(
            resolved,
            path=f"runtime_environment.functor_devices.{name}",
        )["type"] == "cuda":
            return True
    for name, resolved in get_device_overrides().items():
        if _device_signature(
            resolved,
            path=f"runtime_environment.module_device_overrides.{name}",
        )["type"] == "cuda":
            return True
    state_manager = getattr(simulator, "state_manager", None)
    if state_manager is not None:
        for value in state_manager.get_state(clone=False).values():
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                return True
    for functor in getattr(simulator, "functors", {}).values():
        for tensor in functor.state_dict().values():
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
                return True
    return False


def _capture_rng_state(simulator: Any) -> dict[str, Any]:
    rng_state: dict[str, Any] = {
        "python": _encode_checkpoint_value(
            random.getstate(),
            path="rng.python",
        ),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": [],
        "numpy": None,
    }
    if _uses_cuda_runtime(simulator):
        if not is_cuda_available():
            raise RuntimeError(
                "Active CUDA runtime is unavailable during checkpoint capture."
            )
        rng_state["torch_cuda"] = [
            state.detach().cpu().clone()
            for state in torch.cuda.get_rng_state_all()
        ]
    try:
        import numpy as np

        numpy_state = cast(
            tuple[Any, Any, Any, Any, Any],
            np.random.get_state(),
        )
        rng_state["numpy"] = {
            "algorithm": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        }
    except ImportError:
        rng_state["numpy"] = None
    return rng_state


def _restore_rng_state(
    rng_state: Mapping[str, Any],
    simulator: Any,
) -> None:
    """Restore all RNG streams with validate-before-commit semantics."""
    required = {"python", "torch_cpu", "torch_cuda", "numpy"}
    if set(rng_state) != required:
        raise ValueError("Checkpoint RNG state has an invalid field set.")
    python_state = _decode_checkpoint_value(
        rng_state["python"],
        path="rng.python",
    )
    torch_cpu = rng_state["torch_cpu"]
    torch_cuda = rng_state["torch_cuda"]
    if not isinstance(torch_cpu, torch.Tensor):
        raise TypeError("Checkpoint torch CPU RNG state must be a tensor.")
    if not isinstance(torch_cuda, list) or any(
        not isinstance(item, torch.Tensor) for item in torch_cuda
    ):
        raise TypeError("Checkpoint torch CUDA RNG state must be a tensor list.")
    runtime_uses_cuda = _uses_cuda_runtime(simulator)
    if bool(torch_cuda) != runtime_uses_cuda:
        raise RuntimeError(
            "Checkpoint CUDA RNG state presence does not match the "
            "resolved runtime device environment."
        )
    python_probe = random.Random()
    python_probe.setstate(python_state)
    torch_cpu_state = torch_cpu.detach().cpu().clone()
    torch.Generator(device="cpu").set_state(torch_cpu_state)

    torch_cuda_states: list[torch.Tensor] = []
    if torch_cuda:
        if not is_cuda_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable."
            )
        if len(torch_cuda) != torch.cuda.device_count():
            raise RuntimeError(
                "Checkpoint CUDA RNG device count does not match runtime."
            )
        torch_cuda_states = [
            item.detach().cpu().clone() for item in torch_cuda
        ]
        for device_index, state in enumerate(torch_cuda_states):
            torch.Generator(device=f"cuda:{device_index}").set_state(state)

    numpy_state = rng_state["numpy"]
    np = None
    validated_numpy_state = None
    if numpy_state is not None:
        if not isinstance(numpy_state, dict):
            raise TypeError("Checkpoint NumPy RNG state must be a mapping.")
        expected_numpy_fields = {
            "algorithm",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
        if set(numpy_state) != expected_numpy_fields:
            raise ValueError("Checkpoint NumPy RNG state has an invalid field set.")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Checkpoint contains NumPy RNG state but NumPy is unavailable."
            ) from exc
        keys = numpy_state.get("keys")
        if not isinstance(keys, torch.Tensor):
            raise TypeError("Checkpoint NumPy RNG keys must be a tensor.")
        if not isinstance(numpy_state["algorithm"], str):
            raise TypeError("Checkpoint NumPy RNG algorithm must be a string.")
        if type(numpy_state["position"]) is not int:
            raise TypeError("Checkpoint NumPy RNG position must be an integer.")
        if type(numpy_state["has_gauss"]) is not int:
            raise TypeError("Checkpoint NumPy RNG Gaussian flag must be an integer.")
        if type(numpy_state["cached_gaussian"]) not in (int, float):
            raise TypeError("Checkpoint NumPy cached Gaussian must be numeric.")
        validated_numpy_state = (
            numpy_state["algorithm"],
            keys.detach().cpu().numpy().copy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            float(numpy_state["cached_gaussian"]),
        )
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(validated_numpy_state)

    previous_python = random.getstate()
    previous_torch_cpu = torch.get_rng_state().detach().cpu().clone()
    previous_torch_cuda = (
        [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch_cuda_states
        else []
    )
    previous_numpy = (
        cast(Any, np).random.get_state()
        if validated_numpy_state is not None
        else None
    )
    try:
        random.setstate(python_state)
        torch.set_rng_state(torch_cpu_state)
        if torch_cuda_states:
            torch.cuda.set_rng_state_all(torch_cuda_states)
        if validated_numpy_state is not None:
            cast(Any, np).random.set_state(validated_numpy_state)
    except Exception:
        random.setstate(previous_python)
        torch.set_rng_state(previous_torch_cpu)
        if previous_torch_cuda:
            torch.cuda.set_rng_state_all(previous_torch_cuda)
        if previous_numpy is not None:
            cast(Any, np).random.set_state(previous_numpy)
        raise


def _validate_state_manager_snapshot(
    snapshot: Any,
    *,
    require_stable_boundary: bool,
    require_notification_context: bool = False,
) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise TypeError("Checkpoint StateManager payload must be a mapping.")
    if set(snapshot) != _STATE_MANAGER_FIELDS:
        missing = sorted(_STATE_MANAGER_FIELDS.difference(snapshot))
        unknown = sorted(set(snapshot).difference(_STATE_MANAGER_FIELDS))
        raise ValueError(
            "Checkpoint StateManager field mismatch: "
            f"missing={missing}, unknown={unknown}."
        )
    if not isinstance(snapshot["state"], dict) or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in snapshot["state"].items()
    ):
        raise TypeError("Checkpoint StateManager state must map names to tensors.")
    _validate_state_manager_collections(snapshot)
    if require_stable_boundary and snapshot["provenance_transaction_active"]:
        raise RuntimeError(
            "A stable checkpoint boundary requires a closed provenance "
            "transaction."
        )
    _validate_provenance_notification_events(
        snapshot["pending_provenance_events"],
        require_notification_context=require_notification_context,
    )
    return snapshot


def _validate_state_manager_collections(snapshot: Mapping[str, Any]) -> None:
    """Type-check the counters, mappings, lists, and flags of a StateManager payload."""
    for field_name in ("update_counter", "state_reset_counter"):
        if (
            type(snapshot[field_name]) is not int
            or snapshot[field_name] < 0
        ):
            raise ValueError(
                f"Checkpoint StateManager {field_name} must be nonnegative."
            )
    for field_name in ("diagnostics", "clip_loss_diagnostics"):
        if not isinstance(snapshot[field_name], dict):
            raise TypeError(
                f"Checkpoint StateManager {field_name} must be a mapping."
            )
    for field_name in (
        "history",
        "conservation_history",
        "pending_provenance_events",
    ):
        if not isinstance(snapshot[field_name], list):
            raise TypeError(
                f"Checkpoint StateManager {field_name} must be a list."
            )
    if any(
        not isinstance(entry, dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, torch.Tensor)
            for key, value in entry.items()
        )
        for entry in snapshot["history"]
    ):
        raise TypeError(
            "Checkpoint StateManager history must contain tensor mappings."
        )
    if any(
        not isinstance(entry, ConservationSummary)
        for entry in snapshot["conservation_history"]
    ):
        raise TypeError(
            "Checkpoint StateManager conservation_history must contain "
            "ConservationSummary values."
        )
    if any(
        not isinstance(entry, dict)
        for entry in snapshot["pending_provenance_events"]
    ):
        raise TypeError(
            "Checkpoint StateManager pending provenance events must be mappings."
        )
    last_summary = snapshot["last_summary"]
    if last_summary is not None and not isinstance(
        last_summary,
        ConservationSummary,
    ):
        raise TypeError(
            "Checkpoint StateManager last_summary has an invalid type."
        )
    if type(snapshot["auto_report_enabled"]) is not bool:
        raise TypeError(
            "Checkpoint StateManager auto_report_enabled must be a bool."
        )
    if type(snapshot["provenance_transaction_active"]) is not bool:
        raise TypeError(
            "Checkpoint StateManager provenance transaction flag must be a bool."
        )


def _validate_provenance_notification_events(
    events: Sequence[Mapping[str, Any]],
    *,
    require_notification_context: bool,
) -> None:
    """Validate pending provenance notification records of a StateManager payload.

    An event may carry ``notification_context`` (persistent destination) and
    ``commit_context`` (StateManager.apply_updates, CR-A-04 D7), each optional.
    """
    event_fields = {"step", "variable", "pre_value", "post_value", "functor_increments", "unit"}
    optional_fields = {"notification_context", "commit_context"}
    for event in events:
        if not event_fields <= set(event) or set(event) - event_fields - optional_fields:
            raise ValueError("Checkpoint provenance notification fields are invalid.")
        if "commit_context" in event:
            _validate_recorded_commit_context(event["commit_context"])
        if type(event["step"]) is not int or event["step"] < 0:
            raise ValueError("Checkpoint provenance notification step must be a nonnegative integer.")
        if any(not isinstance(event[key], str) or not event[key].strip() for key in ("variable", "unit")):
            raise ValueError("Checkpoint provenance notification variable and unit must be nonempty strings.")
        if any(not isinstance(event[key], torch.Tensor) for key in ("pre_value", "post_value")):
            raise TypeError("Checkpoint provenance notification state values must be tensors.")
        increments = event["functor_increments"]
        if not isinstance(increments, dict) or any(
            not isinstance(key, str) or not key
            or not (isinstance(value, torch.Tensor) or type(value) in (int, float, bool))
            for key, value in increments.items()
        ):
            raise TypeError("Checkpoint provenance notification increments are invalid.")
        context = event.get("notification_context")
        if require_notification_context or context is not None:
            if (
                not isinstance(context, dict)
                or set(context) != {"run_id", "output_dir"}
                or any(not isinstance(value, str) or not value.strip() for value in context.values())
                or not Path(context["output_dir"]).is_absolute()
            ):
                raise ValueError("Checkpoint provenance notification requires a persistent run and absolute destination.")


def _validate_recorded_commit_context(recorded: Any) -> None:
    """Validate the commit context stored in one pending provenance event.

    References: design 20261019_cr_a04_design_01, D7 and section 2;
        StateManager.apply_updates (the recorded copy) and
        relic.core.state.manager.validate_commit_context (the caller's part).
    Mathematical expression: none (a grammar check).
    Parameter meanings: recorded is the caller's context plus
        conservation_checked (bool) and element_net_change (a mapping of element
        name to float, or None with element_net_change_error, a string).
    Units: as recorded (element net changes in g E m-2 or kg m-2).
    Domain/constraints: anything else raises TypeError or ValueError before a
        checkpoint is written or restored; nothing is coerced.
    Sign convention: not applicable.
    """
    if not isinstance(recorded, dict):
        raise TypeError("Checkpoint provenance notification commit context must be a mapping.")
    validate_commit_context(
        {key: value for key, value in recorded.items() if key not in COMMIT_CONTEXT_RESERVED_KEYS}
    )
    if type(recorded.get("conservation_checked")) is not bool or "element_net_change" not in recorded:
        raise ValueError(
            "Checkpoint provenance notification commit context needs conservation_checked "
            "and element_net_change."
        )
    net_change = recorded["element_net_change"]
    if net_change is None:
        if type(recorded.get("element_net_change_error")) is not str:
            raise ValueError(
                "Checkpoint provenance notification commit context without an element net "
                "change must state element_net_change_error."
            )
        return
    if "element_net_change_error" in recorded or not isinstance(net_change, dict) or any(
        not isinstance(element, str) or not element or type(value) is not float
        for element, value in net_change.items()
    ):
        raise ValueError(
            "Checkpoint provenance notification commit context: element_net_change must map "
            "element names to floats, without element_net_change_error."
        )


def _capture_runtime(simulator: Any) -> dict[str, Any]:
    if getattr(getattr(simulator, "state_manager", None), "_provenance_delivery_active", False):
        raise RuntimeError("Cannot capture a checkpoint during provenance notification delivery.")
    runtime: dict[str, Any] = {}
    for field_name in _runtime_field_names():
        if not hasattr(simulator, field_name):
            continue
        value = getattr(simulator, field_name)
        if field_name == "state":
            value = getattr(value, "value", value)
        runtime[field_name] = _encode_checkpoint_value(
            value,
            path=f"runtime.{field_name}",
        )
    missing = sorted(set(_CORE_RUNTIME_FIELDS).difference(runtime))
    if missing:
        raise AttributeError(
            "Simulator is missing checkpoint runtime fields: "
            + ", ".join(missing)
        )
    return runtime


def _checkpoint_grammar_sha256(value: Any, *, path: str) -> str:
    """Digest of one value in the checkpoint grammar (distinguishes tuple/list, int/float)."""
    return _payload_sha256({"value": _encode_checkpoint_value(value, path=path)})


def _validate_splitting_controls(decoded: dict[str, Any], simulator: Any) -> None:
    """Check restored splitting controls in place, before any mutation (D1 at restore).

    Each control the simulator owns is checked by the simulator's own validator,
    the check applied when it is built: a value outside the allowed values,
    malformed, or not exactly the canonical value the simulator writes raises
    ValueError naming the runtime field; a well-formed control the scheduler
    does not execute yet raises NotImplementedError. An absent field takes the
    documented default of its scheduling entry (see _SPLITTING_CONTROL_FIELDS).
    Whether the restored composition is admissible (S1, S3, S5) is resolved
    when the restore publishes its provider view.
    """
    manager = get_global_manager()
    for field_name, (setting, key) in _SPLITTING_CONTROL_FIELDS.items():
        if not hasattr(simulator, field_name):
            continue
        present = field_name in decoded
        if present:
            raw = decoded[field_name]
        else:
            spec = manager.get_parameter(key)
            if spec is None:
                raise KeyError(f"Required parameter '{key}' not found")
            raw = spec.default
        try:
            if setting == "substep_overrides":
                value = simulator._validate_substep_overrides(raw)
            elif setting == "multirate_exchanges":
                value = simulator._validate_multirate_exchanges(raw)
            else:
                value = simulator._validate_integration_control(setting, key, raw)
        except NotImplementedError as exc:
            raise NotImplementedError(f"Checkpoint runtime field '{field_name}': {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint runtime field '{field_name}' is not admissible: {exc}") from exc
        path = f"runtime.{field_name}"
        if present and _checkpoint_grammar_sha256(raw, path=path) != _checkpoint_grammar_sha256(value, path=path):
            raise ValueError(
                f"Checkpoint runtime field '{field_name}' is not admissible: {raw!r} is not the "
                f"canonical value {value!r} the simulator writes."
            )
        decoded[field_name] = value


def _validate_runtime(runtime: Any, simulator: Any) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise TypeError("Checkpoint runtime must be a mapping.")
    allowed = set(_runtime_field_names())
    unknown = sorted(set(runtime).difference(allowed))
    missing = sorted(set(_CORE_RUNTIME_FIELDS).difference(runtime))
    if unknown or missing:
        raise ValueError(
            "Checkpoint runtime field mismatch: "
            f"missing={missing}, unknown={unknown}."
        )
    decoded = {
        name: _decode_checkpoint_value(
            value,
            path=f"runtime.{name}",
        )
        for name, value in runtime.items()
    }
    if not isinstance(decoded["run_id"], str) or not decoded["run_id"].strip():
        raise ValueError("Checkpoint run_id must be a nonempty string.")
    for field_name in (
        "current_time",
        "time_step",
        "total_execution_time",
        "last_checkpoint_time",
    ):
        if type(decoded[field_name]) not in (int, float):
            raise TypeError(
                f"Checkpoint runtime field '{field_name}' must be numeric."
            )
        if not math.isfinite(float(decoded[field_name])):
            raise ValueError(
                f"Checkpoint runtime field '{field_name}' must be finite."
            )
    if float(decoded["time_step"]) <= 0.0:
        raise ValueError("Checkpoint time_step must be positive.")
    if type(decoded["step_count"]) is not int or decoded["step_count"] < 0:
        raise ValueError("Checkpoint step_count must be a nonnegative integer.")
    cursor = decoded.get("_forcing_cursor")
    if cursor is not None:
        if (
            not isinstance(cursor, dict)
            or set(cursor) != {"sequence_sha256", "length", "consumed"}
            or not isinstance(cursor["sequence_sha256"], str)
            or len(cursor["sequence_sha256"]) != 64
            or type(cursor["length"]) is not int
            or type(cursor["consumed"]) is not int
            or cursor["length"] <= 0
            or not 0 <= cursor["consumed"] <= cursor["length"]
        ):
            raise ValueError(
                "Checkpoint forcing cursor must carry sequence_sha256, a "
                "positive length, and consumed within [0, length]."
            )
    time_series = decoded["time_series"]
    if (
        not isinstance(time_series, dict)
        or any(
            not isinstance(name, str)
            or not isinstance(values, list)
            or any(not isinstance(value, torch.Tensor) for value in values)
            for name, values in time_series.items()
        )
    ):
        raise TypeError(
            "Checkpoint time_series must map names to tensor lists."
        )
    _validate_splitting_controls(decoded, simulator)
    state_type = type(simulator.state)
    try:
        decoded["state"] = state_type(decoded["state"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint simulator state is invalid.") from exc
    return decoded


def _validate_state_dict_value(
    saved: Any,
    current: Any,
    *,
    path: str,
) -> None:
    """Validate tensor geometry and safe extra-state structure."""
    if isinstance(current, torch.Tensor):
        if not isinstance(saved, torch.Tensor):
            raise TypeError(f"Checkpoint functor tensor '{path}' is invalid.")
        if saved.shape != current.shape:
            raise ValueError(
                f"Checkpoint functor tensor '{path}' shape mismatch."
            )
        if saved.dtype != current.dtype:
            raise ValueError(
                f"Checkpoint functor tensor '{path}' dtype mismatch."
            )
        return
    if isinstance(saved, torch.Tensor):
        raise TypeError(
            f"Checkpoint functor extra state '{path}' changed type."
        )
    if current is None or type(current) in (bool, int, float, str):
        if type(saved) is not type(current):
            raise TypeError(
                f"Checkpoint functor extra state '{path}' changed type."
            )
        return
    if isinstance(current, Mapping):
        if not isinstance(saved, Mapping) or set(saved) != set(current):
            raise ValueError(
                f"Checkpoint functor extra state '{path}' keys mismatch."
            )
        for key in current:
            if not isinstance(key, str):
                raise TypeError(
                    f"Checkpoint functor extra state '{path}' has "
                    "a non-text key."
                )
            _validate_state_dict_value(
                saved[key],
                current[key],
                path=f"{path}.{key}",
            )
        return
    if isinstance(current, list):
        if not isinstance(saved, list) or len(saved) != len(current):
            raise ValueError(
                f"Checkpoint functor extra state '{path}' list mismatch."
            )
        for index, (saved_item, current_item) in enumerate(
            zip(saved, current, strict=True)
        ):
            _validate_state_dict_value(
                saved_item,
                current_item,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(current, tuple):
        if not isinstance(saved, tuple) or len(saved) != len(current):
            raise ValueError(
                f"Checkpoint functor extra state '{path}' tuple mismatch."
            )
        for index, (saved_item, current_item) in enumerate(
            zip(saved, current, strict=True)
        ):
            _validate_state_dict_value(
                saved_item,
                current_item,
                path=f"{path}[{index}]",
            )
        return
    if isinstance(current, set):
        if not isinstance(saved, set) or len(saved) != len(current):
            raise ValueError(
                f"Checkpoint functor extra state '{path}' set mismatch."
            )
        return
    raise TypeError(
        f"Unsupported live functor state at '{path}': "
        f"{type(current).__name__}."
    )


def _validate_state_dict_mapping(
    name: str,
    raw_state: Any,
    current_state: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw_state, dict):
        raise TypeError(
            f"Checkpoint functor '{name}' state_dict must be a mapping."
        )
    if set(raw_state) != set(current_state):
        raise ValueError(
            f"Checkpoint functor '{name}' state_dict keys do not match."
        )
    for key, value in raw_state.items():
        comparison_value = value if key == "_extra_state" else current_state[key]
        _validate_state_dict_value(
            value,
            comparison_value,
            path=f"{name}.{key}",
        )
    return raw_state


def _capture_functor_state_dict(
    name: str,
    functor: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_state = functor.state_dict()
    encoded = _encode_checkpoint_value(
        current_state,
        path=f"functors.{name}.state_dict",
    )
    decoded = _decode_checkpoint_value(
        encoded,
        path=f"functors.{name}.state_dict",
    )
    validated = _validate_state_dict_mapping(
        name,
        decoded,
        current_state,
    )
    return encoded, validated


def _held_parameter_bindings(functor: Any) -> dict[str, tuple[Any, torch.nn.Parameter]]:
    """ParameterManager bindings held by the Functor or a nested Functor.

    Maps each manager key to (owning manager, binding). A held binding is no
    module state (it is absent from state_dict); its live value is saved and
    restored through the manager that owns it. Two different bindings under
    one key inside one Functor raise ValueError.
    """
    held: dict[str, tuple[Any, torch.nn.Parameter]] = {}
    for module in functor.modules():
        if not isinstance(module, FunctorBase):
            continue
        module_held = module.held_manager_parameters()
        if not module_held:
            continue
        manager = module.get_parameter_manager()
        for key, binding in module_held.items():
            previous = held.get(key)
            if previous is not None and previous[1] is not binding:
                raise ValueError(
                    f"Functor '{functor.name}' holds two different ParameterManager "
                    f"bindings under '{key}'."
                )
            held[key] = (manager, binding)
    return held


def _held_binding_paths(functor: Any) -> dict[str, str]:
    """state_dict path -> manager key of every held binding (the version-2 layout)."""
    paths: dict[str, str] = {}
    for prefix, module in functor.named_modules():
        if not isinstance(module, FunctorBase):
            continue
        for attribute, value in vars(module).items():
            key = module.manager_binding_key(value)
            if key is not None:
                paths[f"{prefix}.{attribute}" if prefix else attribute] = key
    return paths


def _capture_functor_entry(name: str, functor: Any) -> dict[str, Any]:
    """Encode one Functor: its module state and its held manager bindings."""
    encoded_state, _ = _capture_functor_state_dict(name, functor)
    keys_by_manager: dict[int, tuple[Any, list[str]]] = {}
    for key, (manager, _binding) in _held_parameter_bindings(functor).items():
        keys_by_manager.setdefault(id(manager), (manager, []))[1].append(key)
    values: dict[str, torch.Tensor] = {}
    for manager, keys in keys_by_manager.values():
        values.update(capture_binding_values(manager, keys))
    return {
        "state_dict": encoded_state,
        "parameter_bindings": _encode_checkpoint_value(
            values,
            path=f"functors.{name}.parameter_bindings",
        ),
    }


def _validate_held_bindings(
    name: str,
    raw_bindings: Any,
    held: Mapping[str, tuple[Any, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    """Require one saved tensor per held binding, with the binding's shape and dtype."""
    if not isinstance(raw_bindings, dict):
        raise TypeError(
            f"Checkpoint functor '{name}' parameter_bindings must be a mapping."
        )
    if set(raw_bindings) != set(held):
        raise ValueError(
            f"Checkpoint functor '{name}' held ParameterManager bindings do not "
            f"match: saved {sorted(raw_bindings)}, target {sorted(held)}."
        )
    for key, value in raw_bindings.items():
        binding = held[key][1]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != binding.shape
            or value.dtype != binding.dtype
        ):
            raise ValueError(
                f"Checkpoint functor '{name}' binding '{key}' does not match the "
                "shape and dtype of the live binding."
            )
    return raw_bindings


def _write_held_bindings(functor: Any, values: Mapping[str, torch.Tensor]) -> None:
    """Write saved live values back through the managers that own the bindings."""
    held = _held_parameter_bindings(functor)
    values_by_manager: dict[int, tuple[Any, dict[str, torch.Tensor]]] = {}
    for key, value in values.items():
        manager = held[key][0]
        values_by_manager.setdefault(id(manager), (manager, {}))[1][key] = value
    for manager, manager_values in values_by_manager.values():
        restore_binding_values(manager, manager_values)


def _validate_functor_states(
    simulator: Any,
    encoded_functors: Any,
) -> dict[str, dict[str, Any]]:
    if not isinstance(encoded_functors, dict):
        raise TypeError("Checkpoint functors must be a mapping.")
    current_functors = getattr(simulator, "functors", {})
    if set(encoded_functors) != set(current_functors):
        raise ValueError(
            "Checkpoint functor topology does not match the active simulator."
        )
    decoded: dict[str, dict[str, Any]] = {}
    for name, raw_entry in encoded_functors.items():
        if not isinstance(raw_entry, dict) or set(raw_entry) != {"state_dict", "parameter_bindings"}:
            raise ValueError(
                f"Checkpoint functor '{name}' has an invalid payload."
            )
        raw_state = _decode_checkpoint_value(
            raw_entry["state_dict"],
            path=f"functors.{name}.state_dict",
        )
        functor = current_functors[name]
        raw_bindings = _decode_checkpoint_value(
            raw_entry["parameter_bindings"],
            path=f"functors.{name}.parameter_bindings",
        )
        decoded[name] = {
            "state_dict": _validate_state_dict_mapping(
                name,
                raw_state,
                functor.state_dict(),
            ),
            "parameter_bindings": _validate_held_bindings(
                name,
                raw_bindings,
                _held_parameter_bindings(functor),
            ),
        }
    return decoded


def _upgrade_version_2_body(
    body: Mapping[str, Any],
    simulator: Any,
    functor_map: Mapping[str, str],
    moved: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Rewrite one version-2 body into the current Functor entry layout.

    A state_dict entry whose path is an attribute of the target Functor that
    holds a ParameterManager binding becomes that binding's saved value; the
    target is the active Functor of the same name (or of its mapped name under
    a migration certificate). Any other entry must be module state of the
    target; one that is neither cannot be migrated and raises ValueError.
    """
    functors: dict[str, Any] = {}
    for name, entry in body["functors"].items():
        if not isinstance(entry, dict) or set(entry) != {"state_dict"}:
            raise ValueError(f"Version-2 checkpoint functor '{name}' has an invalid payload.")
        state = _decode_checkpoint_value(entry["state_dict"], path=f"functors.{name}.state_dict")
        if not isinstance(state, dict):
            raise TypeError(f"Version-2 checkpoint functor '{name}' state_dict must be a mapping.")
        state = dict(state)
        bindings: dict[str, torch.Tensor] = {}
        target_name = functor_map.get(name, name)
        active = getattr(simulator, "functors", {})
        target = active[target_name] if target_name in active else None
        if target is not None:
            for path, key in _held_binding_paths(target).items():
                if path not in state:
                    continue
                value = state.pop(path)
                if key in bindings and not torch.equal(bindings[key], value):
                    raise ValueError(
                        f"Version-2 checkpoint functor '{name}' stores two values for the "
                        f"ParameterManager binding '{key}'."
                    )
                bindings[key] = value
                moved.setdefault(name, {})[path] = key
            leftover = sorted(set(state) - set(target.state_dict()))
            if leftover:
                raise ValueError(
                    f"Version-2 checkpoint functor '{name}' entries {leftover} are neither "
                    "module state of the target Functor nor a ParameterManager binding it "
                    "holds; they cannot be migrated."
                )
        functors[name] = {
            "state_dict": _encode_checkpoint_value(state, path=f"functors.{name}.state_dict"),
            "parameter_bindings": _encode_checkpoint_value(
                bindings, path=f"functors.{name}.parameter_bindings",
            ),
        }
    return dict(body, functors=functors, format_version=CHECKPOINT_FORMAT_VERSION)


def _upgrade_version_2_payload(
    checkpoint: Mapping[str, Any],
    simulator: Any,
    *,
    functor_map: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Upgrade a validated version-2 payload (root and delivery nodes) and reseal it.

    Returns the upgraded payload and the upgrade record for the receipt.
    """
    moved: dict[str, dict[str, str]] = {}
    body = _upgrade_version_2_body(
        {key: checkpoint[key] for key in _CHECKPOINT_BODY_FIELDS}, simulator, functor_map, moved,
    )
    state = checkpoint["delivery_state"]
    nodes = {
        node_id: dict(node, body=_upgrade_version_2_body(node["body"], simulator, functor_map, {}))
        for node_id, node in state["nodes"].items()
    }
    upgraded = _seal_checkpoint(body, dict(state, nodes=nodes))
    _validate_checkpoint_payload(upgraded)
    return upgraded, {
        "from_format_version": 2,
        "to_format_version": CHECKPOINT_FORMAT_VERSION,
        "moved_bindings": moved,
    }


def capture_runtime_transaction(
    simulator: Any, *, for_rollback: bool = False,
) -> dict[str, Any]:
    """Capture one process-local simulator transaction snapshot.

    This is the in-memory transaction entry point for callers that must restore
    a simulator after an evaluation. It reuses the durable checkpoint owners
    without defining another checkpoint format, persistence path, or restart
    schema.

    Ordinary local transactions require confirmed notifications. The durable
    restore owner may include pending notifications only for its rollback
    image; public local restore must not replay already confirmed observers.
    """
    if type(for_rollback) is not bool:
        raise TypeError("for_rollback must be a bool.")
    state_manager = getattr(simulator, "state_manager", None)
    if state_manager is None or not hasattr(state_manager, "checkpoint"):
        raise RuntimeError("Simulator has no checkpoint-capable StateManager.")

    state_manager_snapshot = _validate_state_manager_snapshot(
        state_manager.checkpoint(),
        require_stable_boundary=True,
    )
    if (state_manager_snapshot["pending_provenance_events"] or simulator._pending_external_deliveries) and not for_rollback:
        raise RuntimeError("Local transaction capture requires confirmed effects, not pending work.")
    encoded_functors: dict[str, dict[str, Any]] = {
        name: _capture_functor_entry(name, functor)
        for name, functor in simulator.functors.items()
    }
    return {
        "runtime": _capture_runtime(simulator),
        "state_manager": _encode_checkpoint_value(
            state_manager_snapshot,
            path="state_manager",
        ),
        "functors": encoded_functors,
        "rng": _capture_rng_state(simulator),
    }


def _validate_runtime_transaction(
    simulator: Any,
    transaction: Any,
    *, for_rollback: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    Mapping[str, Any],
]:
    """Validate and decode an in-memory transaction before mutation."""
    required_fields = {"runtime", "state_manager", "functors", "rng"}
    if not isinstance(transaction, Mapping):
        raise TypeError("Runtime transaction must be a mapping.")
    if set(transaction) != required_fields:
        raise ValueError("Runtime transaction field set is invalid.")

    decoded_runtime = _validate_runtime(transaction["runtime"], simulator)
    decoded_state_manager = _decode_checkpoint_value(
        transaction["state_manager"],
        path="state_manager",
    )
    decoded_state_manager = _validate_state_manager_snapshot(
        decoded_state_manager,
        require_stable_boundary=True,
    )
    if decoded_state_manager["pending_provenance_events"] and not for_rollback:
        raise RuntimeError("Local transaction restore requires confirmed notifications, not pending events.")
    decoded_functors = _validate_functor_states(
        simulator,
        transaction["functors"],
    )
    rng_state = transaction["rng"]
    if not isinstance(rng_state, Mapping):
        raise TypeError("Runtime transaction RNG payload must be a mapping.")
    return (
        decoded_runtime,
        decoded_state_manager,
        decoded_functors,
        rng_state,
    )


def _apply_runtime_fields(
    simulator: Any,
    decoded_runtime: Mapping[str, Any],
    state_manager: Any,
) -> None:
    """Apply runtime fields and remove optional fields absent at capture."""
    simulator_fields = getattr(simulator, "__dict__", {})
    for field_name in _runtime_field_names():
        if field_name in decoded_runtime:
            setattr(simulator, field_name, decoded_runtime[field_name])
        elif (
            field_name not in _CORE_RUNTIME_FIELDS
            and field_name in simulator_fields
        ):
            delattr(simulator, field_name)
    simulator._last_state_reset_counter = getattr(
        state_manager,
        "_state_reset_counter",
        decoded_runtime.get("_last_state_reset_counter", 0),
    )


def _commit_runtime_transaction(
    simulator: Any,
    decoded_runtime: Mapping[str, Any],
    decoded_state_manager: Mapping[str, Any],
    decoded_functors: Mapping[str, Mapping[str, Any]],
    rng_state: Mapping[str, Any],
    *, delivery_state: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> None:
    """Commit validated components using the canonical mutable owners."""
    state_manager = getattr(simulator, "state_manager", None)
    if state_manager is None or not hasattr(state_manager, "restore"):
        raise RuntimeError("Simulator has no restorable StateManager.")
    state_manager.restore(dict(decoded_state_manager))
    for name, entry in decoded_functors.items():
        functor = simulator.functors[name]
        functor.load_state_dict(dict(entry["state_dict"]), strict=True)
        _write_held_bindings(functor, entry["parameter_bindings"])
    _apply_runtime_fields(simulator, decoded_runtime, state_manager)
    _restore_rng_state(rng_state, simulator)
    if delivery_state is not None:
        simulator._pending_external_deliveries, simulator._last_external_delivery_receipts = delivery_state


def _restore_runtime_transaction_components(
    simulator: Any,
    decoded_runtime: Mapping[str, Any],
    decoded_state_manager: Mapping[str, Any],
    decoded_functors: Mapping[str, Mapping[str, Any]],
    rng_state: Mapping[str, Any],
    *, delivery_state: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
    publish_provider_view: bool = False,
) -> Any:
    """Restore components, compensating only before a new view is visible.

    Durable restore uses the existing registration lock and metadata snapshot
    around the single runtime rollback image. Local evaluation restore does
    not publish an epoch. Cleanup after publication cannot undo a world that
    readers may already have observed.
    """
    from relic.core.provenance import ProvenanceRecorder
    from .registration import _RegistrationSnapshot, _publish_provider_view

    lifecycle = simulator._provider_lifecycle if publish_provider_view else None
    with lifecycle.lock if lifecycle is not None else nullcontext():
        previous_transaction = capture_runtime_transaction(simulator, for_rollback=True)
        previous_components = _validate_runtime_transaction(
            simulator, previous_transaction, for_rollback=True,
        )
        previous_deliveries = (simulator._pending_external_deliveries, simulator._last_external_delivery_receipts)
        registration_snapshot = _RegistrationSnapshot.capture(simulator) if lifecycle is not None else None
        previous_view = lifecycle.committed_view if lifecycle is not None else None
        previous_epoch = lifecycle.epoch if lifecycle is not None else None
        previous_reason = lifecycle.pending_publication_reason if lifecycle is not None else None
        state_manager = simulator.state_manager
        previous_recorder = state_manager._provenance_recorder
        pending_events = decoded_state_manager["pending_provenance_events"]
        contextual_events = any("notification_context" in event for event in pending_events)
        if contextual_events:
            if previous_recorder is not None and not isinstance(previous_recorder, ProvenanceRecorder):
                raise TypeError("Persistent provenance notifications require the native ProvenanceRecorder.")
        try:
            if pending_events and previous_recorder is None:
                if not contextual_events or any("notification_context" not in event for event in pending_events):
                    raise ValueError("Restoring notifications without an observer requires persistent context.")
                state_manager.set_provenance_recorder(ProvenanceRecorder(
                    Path(pending_events[0]["notification_context"]["output_dir"]),
                    run_id=decoded_runtime["run_id"],
                ))
            _commit_runtime_transaction(
                simulator, decoded_runtime, decoded_state_manager, decoded_functors,
                rng_state, delivery_state=delivery_state,
            )
            if lifecycle is not None:
                # The restored splitting controls may differ from the live ones
                # (a restored forward order over a published reverse order), so
                # the group orders are recomputed from the dependency graph
                # before the restored plan is resolved and published; on a
                # refusal the registration snapshot restores them.
                simulator._groups_dirty = True
                restored_view = _publish_provider_view(
                    simulator, expected_epoch=previous_epoch, operation="restore",
                    provider_uid="checkpoint", new_uids={},
                )
                if restored_view is None:
                    raise RuntimeError("Checkpoint restore could not publish a committed provider view.")
                return restored_view
        except BaseException as cause:
            if lifecycle is not None and (
                lifecycle.committed_view is not previous_view or lifecycle.epoch != previous_epoch
            ):
                cause.add_note(f"Checkpoint restore already published provider epoch {lifecycle.epoch}; state remains committed.")
                raise
            # Preserve the observer reference without delivering accepted work.
            state_manager._provenance_recorder = previous_recorder
            compensation_errors: list[BaseException] = []
            if registration_snapshot is not None:
                try:
                    registration_snapshot.restore(simulator, cause)
                except BaseException as error:
                    compensation_errors.append(error)
                lifecycle.pending_publication_reason = previous_reason
            try:
                _commit_runtime_transaction(simulator, *previous_components, delivery_state=previous_deliveries)
            except BaseException as error:
                compensation_errors.append(error)
            if compensation_errors:
                raise BaseExceptionGroup(
                    "Checkpoint restore failed and compensation was incomplete",
                    [cause, *compensation_errors],
                ) from cause
            raise
    return None


def restore_runtime_transaction(
    simulator: Any,
    transaction: Mapping[str, Any],
) -> None:
    """Atomically restore a process-local simulator transaction snapshot.

    A local restore publishes no provider view, so it refuses splitting
    controls that differ from the live ones (the published plan was resolved
    from those); only the durable restore, which republishes, may change them.
    """
    decoded_components = _validate_runtime_transaction(simulator, transaction)
    if simulator.state_manager._pending_provenance_events or simulator._pending_external_deliveries:
        raise RuntimeError("Local transaction restore requires confirmed live effects, not pending work.")
    decoded_runtime = decoded_components[0]
    changed = sorted(
        field_name
        for field_name in _SPLITTING_CONTROL_FIELDS
        if field_name in decoded_runtime
        and hasattr(simulator, field_name)
        and _checkpoint_grammar_sha256(decoded_runtime[field_name], path=f"runtime.{field_name}")
        != _checkpoint_grammar_sha256(getattr(simulator, field_name), path=f"runtime.{field_name}")
    )
    if changed:
        raise RuntimeError(
            "Local transaction restore cannot change the splitting controls of the published "
            f"provider view ({changed}); use the durable checkpoint restore, which republishes."
        )
    _restore_runtime_transaction_components(simulator, *decoded_components)


def _seal_checkpoint(body: Mapping[str, Any], delivery_state: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one root and its finite delivery closure with the existing hash."""
    payload = dict(body, delivery_state=dict(delivery_state))
    payload["integrity"] = {"algorithm": "sha256", "payload_sha256": _payload_sha256(payload)}
    return payload


def _checkpoint_delivery_root(payload: Mapping[str, Any], root_id: str) -> dict[str, Any]:
    """Select a frozen root without recapturing science or embedding sub-tables."""
    state = payload["delivery_state"]
    nodes = dict(state["nodes"])
    nodes[state["root_id"]] = {
        "body": {key: payload[key] for key in _CHECKPOINT_BODY_FIELDS},
        "pending": state["pending"], "receipts": state["receipts"],
    }
    pending_nodes = [root_id]
    visited = set()
    job_keys = set()
    while pending_nodes:
        node_id = pending_nodes.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        for progress in nodes[node_id]["pending"]:
            key = progress["key"]
            job_keys.add(key)
            job = state["jobs"][key]
            if job["operation"] == "checkpoint":
                pending_nodes.append(job["snapshot_ref"])
    root = nodes[root_id]
    return _seal_checkpoint(root["body"], {
        "root_id": root_id, "pending": root["pending"], "receipts": root["receipts"],
        "jobs": {key: state["jobs"][key] for key in sorted(job_keys)},
        "nodes": {key: nodes[key] for key in sorted(visited - {root_id})},
    })


def checkpoint_delivery_snapshot(payload: Mapping[str, Any], delivery_key: str) -> dict[str, Any]:
    """Return an independent copy of a request's frozen recovery root."""
    _validate_checkpoint_payload(payload)
    job = payload["delivery_state"]["jobs"][delivery_key]
    if job["operation"] != "checkpoint":
        raise ValueError("Requested delivery is not a checkpoint.")
    return deepcopy(_checkpoint_delivery_root(payload, job["snapshot_ref"]))


def _capture_delivery_state(
    simulator: Any, body: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]],
    *, preparing: bool,
) -> dict[str, Any]:
    """Flatten frozen requests; references can cycle, Python values cannot."""
    state: dict[str, Any] = {
        "root_id": uuid4().hex, "pending": [],
        "receipts": _encode_checkpoint_value(simulator._last_external_delivery_receipts, path="delivery.receipts"),
        "jobs": {}, "nodes": {},
    }
    copy_memo: dict[int, Any] = {}

    def merge(table: dict[str, Any], key: str, value: Any) -> None:
        if key in table:
            if _payload_sha256(table[key]) != _payload_sha256(value):
                raise ValueError(f"Frozen delivery definition conflicts: {key}")
            return
        table[key] = deepcopy(value, copy_memo)

    new_checkpoint_nodes = []
    for job in jobs:
        key = job["key"]
        if any(progress["key"] == key for progress in state["pending"]):
            raise ValueError(f"Duplicate pending delivery: {key}")
        state["pending"].append({"key": key, "attempts": job["attempts"], "error": job["error"]})
        definition = _encode_checkpoint_value({field: job[field] for field in _DELIVERY_FIELDS}, path="delivery.job")
        if job["operation"] == "checkpoint":
            snapshot = job["snapshot"]
            if snapshot is None:
                if not preparing:
                    raise ValueError("An accepted checkpoint delivery has no frozen snapshot.")
                definition["snapshot_ref"] = key
                new_checkpoint_nodes.append((key, job))
            else:
                _validate_checkpoint_payload(snapshot)
                nested = snapshot["delivery_state"]
                for nested_key, value in nested["jobs"].items():
                    merge(state["jobs"], nested_key, value)
                for node_id, value in nested["nodes"].items():
                    merge(state["nodes"], node_id, value)
                merge(state["nodes"], nested["root_id"], {
                    "body": {field: snapshot[field] for field in _CHECKPOINT_BODY_FIELDS},
                    "pending": nested["pending"], "receipts": nested["receipts"],
                })
                definition["snapshot_ref"] = nested["root_id"]
        else:
            definition["snapshot"] = _encode_checkpoint_value(job["snapshot"], path="delivery.diagnostic")
        merge(state["jobs"], key, definition)
    for node_id, job in new_checkpoint_nodes:
        # Scientific values are already frozen; each root owns only its
        # bookkeeping projection and shares the immutable captured tensors.
        node_body = dict(body)
        node_body["runtime"] = dict(body["runtime"])
        if job["automatic"]:
            node_body["runtime"]["last_checkpoint_time"] = job["accepted_time"]
        state["nodes"][node_id] = {
            "body": node_body, "pending": deepcopy(state["pending"]),
            "receipts": deepcopy(state["receipts"]),
        }
    return state


def _validate_delivery_progress(node: Mapping[str, Any], jobs: Mapping[str, Any]) -> None:
    """Validate one node's pending progress rows against the job table."""
    seen = set()
    for progress in node["pending"]:
        if not isinstance(progress, dict) or set(progress) != {"key", "attempts", "error"}:
            raise ValueError("Checkpoint delivery progress fields are invalid.")
        key = progress["key"]
        if not isinstance(key, str) or key not in jobs or key in seen:
            raise ValueError("Checkpoint delivery progress has a missing or duplicate job reference.")
        seen.add(key)
        if type(progress["attempts"]) is not int or progress["attempts"] < 0 or (progress["error"] is not None and not isinstance(progress["error"], str)):
            raise ValueError("Checkpoint delivery progress values are invalid.")


def _validate_delivery_receipts(encoded_receipts: Any) -> None:
    """Validate one node's receipt rows (identity, attempts, artifact shape)."""
    receipts = _decode_checkpoint_value(encoded_receipts, path="delivery.receipts")
    receipt_keys = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != {
            "key", "run_id", "accepted_step", "accepted_time", "operation", "sources", "attempts", "receipt",
        }:
            raise ValueError("Checkpoint delivery receipt fields are invalid.")
        if not isinstance(receipt["key"], str) or not receipt["key"] or receipt["key"] in receipt_keys:
            raise ValueError("Checkpoint delivery receipt identity is invalid or duplicated.")
        receipt_keys.add(receipt["key"])
        if type(receipt["attempts"]) is not int or receipt["attempts"] < 0:
            raise ValueError("Checkpoint delivery receipt attempts are invalid.")
        artifact = receipt["receipt"]
        if receipt["operation"] == "topology":
            topology_fields = {"transition", "receipt_id", "epoch_before", "epoch_after", "status"}
            if not isinstance(artifact, dict) or set(artifact) != topology_fields or not isinstance(
                artifact["transition"], str
            ):
                raise ValueError("Checkpoint delivery receipt artifact is invalid.")
        elif not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError("Checkpoint delivery receipt artifact is invalid.")


def _validate_delivery_node(node_id: Any, node: Any, *, root_id: str, jobs: Mapping[str, Any]) -> None:
    """Validate one delivery node (root or child) of the checkpoint graph."""
    if not isinstance(node_id, str) or not node_id or not isinstance(node, dict) or set(node) != {"body", "pending", "receipts"}:
        raise ValueError("Checkpoint delivery node fields are invalid.")
    if not isinstance(node["body"], dict) or set(node["body"]) != _CHECKPOINT_BODY_FIELDS:
        raise ValueError("Checkpoint delivery node body is invalid.")
    if not isinstance(node["pending"], list) or not isinstance(node["receipts"], list):
        raise TypeError("Checkpoint delivery progress and receipts must be lists.")
    _validate_delivery_progress(node, jobs)
    _validate_delivery_receipts(node["receipts"])
    if node_id != root_id:
        # Child bodies reuse the same format validation, with no nested graph.
        empty = {"root_id": node_id, "pending": [], "receipts": [], "jobs": {}, "nodes": {}}
        _validate_checkpoint_payload(_seal_checkpoint(node["body"], empty))


def _validate_delivery_job(key: Any, encoded_job: Any, roots: Mapping[str, Any]) -> None:
    """Validate one delivery definition and its snapshot binding."""
    if not isinstance(key, str) or not key or not isinstance(encoded_job, dict):
        raise TypeError("Checkpoint delivery definitions must map keys to records.")
    operation = encoded_job.get("operation")
    snapshot_key = "snapshot_ref" if operation == "checkpoint" else "snapshot"
    if operation not in {"checkpoint", "diagnostic", "topology"} or set(encoded_job) != _DELIVERY_FIELDS | {snapshot_key}:
        raise ValueError("Checkpoint delivery definition fields are invalid.")
    job = _decode_checkpoint_value(encoded_job, path="delivery.job")
    if not isinstance(job["run_id"], str) or not job["run_id"].strip() or not isinstance(job["resource"], str) or not Path(job["resource"]).is_absolute():
        raise ValueError("Checkpoint delivery identity or resource is invalid.")
    if type(job["accepted_step"]) is not int or job["accepted_step"] < 0 or type(job["accepted_time"]) not in (int, float) or not math.isfinite(job["accepted_time"]):
        raise ValueError("Checkpoint delivery accepted time or step is invalid.")
    if type(job["automatic"]) is not bool or not isinstance(job["parameters"], dict) or not isinstance(job["sources"], list) or not job["sources"] or any(not isinstance(source, dict) for source in job["sources"]):
        raise ValueError("Checkpoint delivery options or sources are invalid.")
    _validate_delivery_snapshot_binding(key, job, operation, roots)


def _validate_delivery_snapshot_binding(
    key: str, job: Mapping[str, Any], operation: Any, roots: Mapping[str, Any]
) -> None:
    """Bind a checkpoint job to its snapshot root or a diagnostic job to its prepared output."""
    if operation == "checkpoint":
        if not isinstance(job["snapshot_ref"], str) or job["snapshot_ref"] not in roots:
            raise ValueError("Checkpoint delivery references a missing snapshot.")
        runtime = roots[job["snapshot_ref"]]["body"]["runtime"]
        if (runtime["run_id"], runtime["step_count"], runtime["current_time"]) != (job["run_id"], job["accepted_step"], job["accepted_time"]):
            raise ValueError("Checkpoint delivery snapshot does not match its accepted run and step.")
    elif not isinstance(job["snapshot"], dict) or not job["snapshot"]:
        raise ValueError("Checkpoint diagnostic delivery has no prepared snapshot.")
    else:
        result = job["snapshot"].get("result")
        if not isinstance(result, dict) or not isinstance(result.get("output_path"), str):
            raise ValueError("Checkpoint diagnostic delivery has no prepared output path.")
        if os.path.normcase(str(Path(result["output_path"]).resolve())) != os.path.normcase(str(Path(job["resource"]).resolve())):
            raise ValueError("Checkpoint diagnostic snapshot conflicts with its delivery resource.")
        if result.get("delivery_key") not in (None, key):
            raise ValueError("Checkpoint diagnostic snapshot conflicts with its delivery identity.")


def _validate_delivery_state(payload: Mapping[str, Any]) -> None:
    """Validate the file-local reference graph without publishing any effect."""
    state = payload["delivery_state"]
    if not isinstance(state, dict) or set(state) != {"root_id", "pending", "receipts", "jobs", "nodes"}:
        raise ValueError("Checkpoint delivery_state fields are invalid.")
    root_id = state["root_id"]
    if not isinstance(root_id, str) or not root_id or not isinstance(state["jobs"], dict) or not isinstance(state["nodes"], dict):
        raise ValueError("Checkpoint delivery tables and root identity are invalid.")
    if root_id in state["nodes"]:
        raise ValueError("Checkpoint delivery root must not be duplicated in nodes.")
    roots = {root_id: {"body": {key: payload[key] for key in _CHECKPOINT_BODY_FIELDS},
                       "pending": state["pending"], "receipts": state["receipts"]}, **state["nodes"]}
    for node_id, node in roots.items():
        _validate_delivery_node(node_id, node, root_id=root_id, jobs=state["jobs"])
    if any(node["body"]["format_version"] != payload["format_version"] for node in state["nodes"].values()):
        raise ValueError("Checkpoint delivery nodes mix format versions.")
    for key, encoded_job in state["jobs"].items():
        _validate_delivery_job(key, encoded_job, roots)
    canonical = _checkpoint_delivery_root(payload, root_id)["delivery_state"]
    if set(canonical["nodes"]) != set(state["nodes"]) or set(canonical["jobs"]) != set(state["jobs"]):
        raise ValueError("Checkpoint delivery tables contain unreachable definitions.")


def _restore_delivery_state(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize runtime jobs from the already validated frozen value graph."""
    state = payload["delivery_state"]
    jobs = []
    for progress in state["pending"]:
        key = progress["key"]
        job = _decode_checkpoint_value(state["jobs"][key], path="delivery.job")
        if job["operation"] == "checkpoint":
            job["snapshot"] = _checkpoint_delivery_root(payload, job.pop("snapshot_ref"))
        job.update(progress)
        jobs.append(job)
    return jobs, _decode_checkpoint_value(state["receipts"], path="delivery.receipts")


def capture_checkpoint_payload(
    simulator: Any, *, accepted_checkpoint_time: float | None = None,
    pending_commit: bool = False,
    accepted_auto_report_enabled: bool | None = None,
    pending_deliveries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Freeze the existing checkpoint format without performing file I/O.

    The step owner may prepare a candidate during its provenance transaction.
    Only the independent snapshot receives the projected accepted flags; the
    owner must accept that step before publishing the candidate. Ordinary
    captures continue to require a stable, closed transaction boundary.
    """
    if type(pending_commit) is not bool:
        raise TypeError("pending_commit must be a bool.")
    if pending_commit:
        if type(accepted_auto_report_enabled) is not bool:
            raise TypeError("pending_commit requires accepted_auto_report_enabled as a bool.")
    elif accepted_auto_report_enabled is not None:
        raise ValueError("accepted_auto_report_enabled requires pending_commit.")
    if pending_deliveries is not None and not pending_commit:
        raise ValueError("A new delivery batch requires pending_commit.")
    runtime_environment = _runtime_environment_signature(simulator)
    _require_single_process_checkpoint(runtime_environment)
    state_manager = getattr(simulator, "state_manager", None)
    if state_manager is None or not hasattr(state_manager, "checkpoint"):
        raise RuntimeError("Simulator has no checkpoint-capable StateManager.")

    runtime = _capture_runtime(simulator)
    _validate_runtime(runtime, simulator)
    if accepted_checkpoint_time is not None:
        if (
            type(accepted_checkpoint_time) not in (int, float)
            or not math.isfinite(accepted_checkpoint_time)
            or accepted_checkpoint_time != runtime["current_time"]
        ):
            raise ValueError("Accepted checkpoint time must match the captured current time.")
        # Stage the successful automatic-checkpoint bookkeeping in its payload.
        # The live timestamp changes only after the delivery receipt succeeds.
        runtime["last_checkpoint_time"] = float(accepted_checkpoint_time)
    state_manager_snapshot = _validate_state_manager_snapshot(
        state_manager.checkpoint(),
        require_stable_boundary=not pending_commit,
        require_notification_context=True,
    )
    if pending_commit:
        if not state_manager_snapshot["provenance_transaction_active"]:
            raise RuntimeError("Checkpoint preparation requires an active provenance transaction.")
        state_manager_snapshot["provenance_transaction_active"] = False
        state_manager_snapshot["auto_report_enabled"] = accepted_auto_report_enabled
        runtime["_step_update_sources"] = None
    encoded_state_manager = _encode_checkpoint_value(
        state_manager_snapshot,
        path="state_manager",
    )
    encoded_functors = {
        name: _capture_functor_entry(name, functor)
        for name, functor in simulator.functors.items()
    }
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "runtime": runtime,
        "state_manager": encoded_state_manager,
        "functors": encoded_functors,
        "topology": _encode_checkpoint_value(
            _build_topology(
                simulator,
                runtime_environment=runtime_environment,
            ),
            path="topology",
        ),
        "rng": _capture_rng_state(simulator),
    }
    deliveries = simulator._pending_external_deliveries if pending_deliveries is None else pending_deliveries
    state = _capture_delivery_state(simulator, payload, deliveries, preparing=pending_deliveries is not None)
    result = _seal_checkpoint(payload, state)
    _validate_checkpoint_payload(result)
    return result


def create_checkpoint(
    simulator: Any,
    checkpoint_path: str | os.PathLike[str],
    *,
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a live or frozen checkpoint; retry frozen content idempotently.

    A snapshot was captured at an accepted boundary and is never rebuilt from
    live state during delivery. Existing targets must match its verified
    payload before a receipt can be replayed. The direct API retains its
    exclusive-create behavior when no snapshot is provided.
    """
    target = Path(checkpoint_path).resolve()
    if snapshot is None and target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    payload = capture_checkpoint_payload(simulator) if snapshot is None else dict(snapshot)
    payload_sha = _validate_checkpoint_payload(payload)
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Only the current checkpoint format version is written: "
            f"{payload['format_version']!r} != {CHECKPOINT_FORMAT_VERSION}."
        )
    if target.exists():
        if not target.is_file() or target.stat().st_size == 0:
            raise FileExistsError(f"Existing checkpoint target is not a nonempty file: {target}")
        try:
            stored = torch.load(target, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, UnpicklingError, ValueError, IndexError) as exc:
            raise ValueError(f"Existing checkpoint target is unreadable: {target}") from exc
        stored_sha = _validate_checkpoint_payload(stored)
        if stored_sha != payload_sha:
            raise FileExistsError(f"Existing checkpoint conflicts with frozen payload: {target}")
        return _checkpoint_receipt(target, payload_sha)

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    receipt = _checkpoint_receipt(target, payload_sha)
    logger.info("Checkpoint saved: %s", target)
    return receipt


def _checkpoint_receipt(
    path: Path, payload_sha: str, *, provider_epoch: Optional[int] = None,
) -> dict[str, Any]:
    """Describe the actual checkpoint artifact after successful publication."""
    return {
        "path": str(path),
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "payload_sha256": payload_sha,
        "provider_epoch": provider_epoch,
    }


def _validate_checkpoint_payload(checkpoint: Any) -> str:
    """Validate the one format/header/integrity contract for write and restore."""
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint payload must be a mapping.")
    required_fields = _CHECKPOINT_BODY_FIELDS | {"integrity", "delivery_state"}
    if set(checkpoint) != required_fields:
        raise ValueError("Checkpoint top-level field set is invalid.")
    if checkpoint["format"] != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint format identifier is invalid.")
    if type(checkpoint["format_version"]) is not int or checkpoint["format_version"] not in _READABLE_FORMAT_VERSIONS:
        raise ValueError(
            "Unsupported checkpoint version: "
            f"{checkpoint['format_version']!r}."
        )
    created_utc = checkpoint["created_utc"]
    if not isinstance(created_utc, str) or not created_utc.endswith("Z"):
        raise ValueError("Checkpoint created_utc timestamp is invalid.")
    try:
        datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Checkpoint created_utc timestamp is invalid.") from exc
    integrity = checkpoint["integrity"]
    if (
        not isinstance(integrity, dict)
        or set(integrity) != {"algorithm", "payload_sha256"}
        or integrity["algorithm"] != "sha256"
        or not isinstance(integrity["payload_sha256"], str)
    ):
        raise ValueError("Checkpoint integrity record is invalid.")
    content = {
        key: value
        for key, value in checkpoint.items()
        if key != "integrity"
    }
    actual_payload_sha = _payload_sha256(content)
    if actual_payload_sha != integrity["payload_sha256"]:
        raise ValueError("Checkpoint integrity verification failed.")
    _validate_state_manager_snapshot(
        _decode_checkpoint_value(checkpoint["state_manager"], path="state_manager"),
        require_stable_boundary=True,
        require_notification_context=True,
    )
    _validate_delivery_state(checkpoint)
    return actual_payload_sha


def restore_checkpoint(
    simulator: Any,
    checkpoint_path: str | os.PathLike[str],
    *,
    compatibility_certificate: RestartCompatibilityCertificate | None = None,
    migration_certificate: GraphMigrationCertificate | None = None,
) -> dict[str, Any]:
    """Validate and restore one strict checkpoint without unsafe fallback.

    Without a certificate the restore is exact: the whole topology digest
    must match. With a :class:`RestartCompatibilityCertificate` the restore
    is compatible: only the certified contract differences, bounded
    parameter drift, and certified state renames are accepted, and the
    receipt records the accepted differences.
    """
    if compatibility_certificate is not None and not isinstance(
        compatibility_certificate, RestartCompatibilityCertificate
    ):
        raise TypeError(
            "compatibility_certificate must be a RestartCompatibilityCertificate."
        )
    if migration_certificate is not None and not isinstance(
        migration_certificate, GraphMigrationCertificate
    ):
        raise TypeError("migration_certificate must be a GraphMigrationCertificate.")
    if migration_certificate is not None and compatibility_certificate is not None:
        raise ValueError(
            "A migration certificate carries its own compatibility terms; "
            "pass only one certificate."
        )
    source = Path(checkpoint_path).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Checkpoint file is missing or empty: {source}")
    runtime_environment = _runtime_environment_signature(simulator)
    _require_single_process_checkpoint(runtime_environment)
    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
    )
    actual_payload_sha = _validate_checkpoint_payload(checkpoint)
    format_upgrade: dict[str, Any] | None = None
    if checkpoint["format_version"] != CHECKPOINT_FORMAT_VERSION:
        checkpoint, format_upgrade = _upgrade_version_2_payload(
            checkpoint,
            simulator,
            functor_map=(
                dict(migration_certificate.functor_map)
                if migration_certificate is not None
                else {}
            ),
        )

    saved_topology = _decode_checkpoint_value(
        checkpoint["topology"],
        path="topology",
    )
    if not isinstance(saved_topology, Mapping):
        raise TypeError("Checkpoint topology must be a mapping.")
    scientific_contract = saved_topology.get("scientific_contract")
    if not isinstance(scientific_contract, Mapping):
        raise ValueError("Checkpoint scientific contract is missing.")
    parameter_specs = scientific_contract.get("accessed_parameter_specs")
    if not isinstance(parameter_specs, Mapping) or any(
        not isinstance(key, str) for key in parameter_specs
    ):
        raise ValueError(
            "Checkpoint accessed parameter contract is invalid."
        )
    current_topology = _build_topology(
        simulator,
        parameter_keys=tuple(parameter_specs),
        runtime_environment=runtime_environment,
    )
    saved_topology_encoded = _encode_checkpoint_value(
        saved_topology,
        path="topology",
    )
    current_topology_encoded = _encode_checkpoint_value(
        current_topology,
        path="topology",
    )
    functors_payload: Any = checkpoint["functors"]
    migration: dict[str, Any] | None = None
    if migration_certificate is not None:
        saved_sha = _topology_sha256(saved_topology_encoded)
        current_sha = _topology_sha256(current_topology_encoded)
        if migration_certificate.source_topology_sha256 != saved_sha:
            raise ValueError(
                "Migration certificate source topology digest does not match the checkpoint."
            )
        if migration_certificate.target_topology_sha256 != current_sha:
            raise ValueError(
                "Migration certificate target topology digest does not match the active simulator."
            )
        migration_runtime = _validate_runtime(checkpoint["runtime"], simulator)
        migration_state_manager = _validate_state_manager_snapshot(
            _decode_checkpoint_value(checkpoint["state_manager"], path="state_manager"),
            require_stable_boundary=True,
            require_notification_context=True,
        )
        migration = _migrate_saved_payload(
            migration_certificate,
            simulator,
            saved_topology=saved_topology,
            current_topology=current_topology,
            decoded_runtime=migration_runtime,
            decoded_state_manager=migration_state_manager,
            encoded_functors=functors_payload,
        )
        saved_topology = migration["topology"]
        saved_topology_encoded = _encode_checkpoint_value(saved_topology, path="topology")
        functors_payload = migration["functors"]
        compatibility_certificate = RestartCompatibilityCertificate(
            source_topology_sha256=_topology_sha256(saved_topology_encoded),
            target_topology_sha256=current_sha,
            allowed_contract_paths=tuple(migration_certificate.allowed_contract_paths),
            parameter_relative_tolerance=float(migration_certificate.parameter_relative_tolerance),
            forcing_cursor_policy=migration_certificate.forcing_cursor_policy,
        )
    compatibility_audit: dict[str, Any] | None = None
    if _payload_sha256(
        {"topology": saved_topology_encoded}
    ) != _payload_sha256({"topology": current_topology_encoded}):
        if compatibility_certificate is None:
            mismatched_sections = _mismatched_topology_sections(
                saved_topology_encoded,
                current_topology_encoded,
            )
            raise ValueError(
                "Checkpoint functor topology, scientific contract, or runtime "
                "environment does not match the active simulator; mismatched "
                f"sections: {', '.join(mismatched_sections)}."
            )
        compatibility_audit = _audit_compatible_topology(
            compatibility_certificate,
            saved_topology_encoded,
            current_topology_encoded,
        )
    elif compatibility_certificate is not None:
        compatibility_audit = _audit_compatible_topology(
            compatibility_certificate,
            saved_topology_encoded,
            current_topology_encoded,
        )
    decoded_runtime = _validate_runtime(checkpoint["runtime"], simulator)
    if migration is not None:
        decoded_runtime = dict(migration["runtime"])
    if compatibility_certificate is not None and compatibility_certificate.forcing_cursor_policy == "clear":
        decoded_runtime = dict(decoded_runtime, _forcing_cursor=None)
    if migration is not None:
        decoded_state_manager = dict(migration["state_manager"])
    else:
        decoded_state_manager = _decode_checkpoint_value(
            checkpoint["state_manager"],
            path="state_manager",
        )
    decoded_state_manager = _validate_state_manager_snapshot(
        decoded_state_manager,
        require_stable_boundary=True,
        require_notification_context=True,
    )
    decoded_functors = _validate_functor_states(
        simulator,
        functors_payload,
    )
    mapped_variables: list[dict[str, Any]] = []
    if compatibility_certificate is not None:
        decoded_state_manager = dict(decoded_state_manager)
        mapped_variables = _apply_state_variable_map(
            compatibility_certificate,
            simulator,
            saved_topology,
            decoded_state_manager,
        )
    rng_state = checkpoint["rng"]
    if not isinstance(rng_state, dict):
        raise TypeError("Checkpoint RNG payload must be a mapping.")

    # A restore must not erase notifications already accepted by this target.
    # Identity uses the same run, variable, update index, and destination as
    # the native recorder. Content comparison reuses the checkpoint grammar.
    pending_events = []
    event_digests = {}
    for event in (
        *simulator.state_manager._pending_provenance_events,
        *decoded_state_manager["pending_provenance_events"],
    ):
        context = event.get("notification_context", {})
        destination = context.get("output_dir")
        if destination is not None:
            destination = os.path.normcase(str(Path(destination).resolve()))
        identity = (context.get("run_id", simulator.run_id), destination, event["step"], event["variable"])
        encoded_event = _encode_checkpoint_value(event, path="provenance_notification")
        comparison_event = dict(encoded_event)
        if "notification_context" in comparison_event:
            comparison_event["notification_context"] = dict(context, output_dir=destination)
        digest = _payload_sha256(comparison_event)
        if identity in event_digests:
            if event_digests[identity] != digest:
                raise ValueError("Restored provenance notification conflicts with an accepted notification.")
            continue
        event_digests[identity] = digest
        pending_events.append(_decode_checkpoint_value(encoded_event, path="provenance_notification"))
    decoded_state_manager["pending_provenance_events"] = pending_events
    incoming_jobs, incoming_receipts = _restore_delivery_state(checkpoint)
    merged_jobs = []
    job_digests = {}
    job_positions = {}
    for job in [*simulator._pending_external_deliveries, *incoming_jobs]:
        key = job["key"]
        content = _encode_checkpoint_value({field: job[field] for field in _DELIVERY_FIELDS}, path="delivery.job")
        content["snapshot"] = job["snapshot"] if job["operation"] == "checkpoint" else _encode_checkpoint_value(job["snapshot"], path="delivery.diagnostic")
        digest = _payload_sha256(content)
        if key in job_digests:
            if job_digests[key] != digest:
                raise ValueError(f"Restored delivery conflicts with an accepted delivery: {key}")
            index = job_positions[key]
            previous = merged_jobs[index]
            if job["attempts"] > previous["attempts"]:
                merged_jobs[index] = dict(previous, attempts=job["attempts"], error=job["error"])
            continue
        job_digests[key] = digest
        job_positions[key] = len(merged_jobs)
        merged_jobs.append(job)
    receipts = {}
    for record in [*incoming_receipts, *simulator._last_external_delivery_receipts]:
        previous = receipts.get(record["key"])
        if previous is None or record["attempts"] >= previous["attempts"]:
            receipts[record["key"]] = record
    restored_view = _restore_runtime_transaction_components(
        simulator,
        decoded_runtime,
        decoded_state_manager,
        decoded_functors,
        rng_state,
        delivery_state=(merged_jobs, list(receipts.values())),
        publish_provider_view=True,
    )
    # Receipt I/O follows publication and cannot rewind an observable epoch.
    try:
        receipt = _checkpoint_receipt(
            source, actual_payload_sha, provider_epoch=restored_view.epoch,
        )
    except BaseException as error:
        error.add_note(f"Checkpoint restore already published provider epoch {restored_view.epoch}; receipt creation failed.")
        raise
    receipt["restart_mode"] = "exact" if compatibility_audit is None else "compatible"
    if compatibility_audit is not None:
        receipt["compatibility"] = dict(
            compatibility_audit, mapped_state_variables=mapped_variables,
        )
    if migration is not None and migration_certificate is not None:
        receipt["restart_mode"] = "migration"
        receipt["migration"] = {
            "certificate_sha256": migration_certificate.sha256(),
            "replay_mode": migration_certificate.replay_mode,
            "functor_map": dict(migration_certificate.functor_map),
            "retired_functors": list(migration_certificate.retired_functors),
            "new_functors": list(migration_certificate.new_functors),
            "mapped_state_variables": migration["mapped_state_variables"],
            "retired_state_variables": list(migration_certificate.retired_state_variables),
            "preserved_history_variables": migration["preserved_history_variables"],
            "new_functor_rng": migration_certificate.new_functor_rng,
        }

    if format_upgrade is not None:
        receipt["format_upgrade"] = format_upgrade

    logger.info("Checkpoint restored from: %s", source)
    return receipt


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "GraphMigrationCertificate",
    "RestartCompatibilityCertificate",
    "compute_restart_topology_digests",
    "capture_checkpoint_payload",
    "capture_runtime_transaction",
    "create_checkpoint",
    "restore_checkpoint",
    "restore_runtime_transaction",
]
