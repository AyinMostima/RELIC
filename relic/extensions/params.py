"""
Extension parameter bridge using existing UnifiedParameterManager APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import yaml

from relic.parameters.spec import ParameterSpec

from .discovery import ExtensionInfo

logger = logging.getLogger(__name__)


def _normalize_namespace_prefix(prefix: Optional[str]) -> Optional[str]:
    if prefix is None:
        return None
    text = prefix.strip().strip(".")
    return text or None


def _safe_extension_key(name: str) -> str:
    cleaned = name.strip().replace("-", "_").replace(" ", "_")
    normalized = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_").strip("_")
    return normalized.lower() or "extension"


def _iter_parameter_sources(extension_info: ExtensionInfo) -> Iterable[tuple[Path, Optional[str]]]:
    for declaration in extension_info.manifest.parameters:
        source_path = (extension_info.root_path / declaration.path).resolve()
        yield source_path, _normalize_namespace_prefix(declaration.namespace_prefix)


def _load_specs_from_source(manager: Any, source_path: Path) -> Dict[str, ParameterSpec]:
    source = Path(source_path)
    if source.is_dir():
        return manager.database._load_from_directory(source)
    if source.suffix.lower() in {".yaml", ".yml"}:
        return manager.database._load_yaml_file(source, base_parameters=manager.database.cache)
    if source.suffix.lower() == ".json":
        return manager.database._load_json_file(source)
    raise ValueError(f"Unsupported parameter source type: {source}")


def _clone_with_namespace_prefix(spec: ParameterSpec, namespace_prefix: Optional[str]) -> ParameterSpec:
    prefix = _normalize_namespace_prefix(namespace_prefix)
    if prefix is None:
        return spec

    if spec.key == prefix or spec.key.startswith(f"{prefix}."):
        return spec

    payload = spec.to_dict()
    payload["key"] = f"{prefix}.{spec.key}"
    provenance = dict(payload.get("provenance") or {})
    provenance["namespace_prefix"] = prefix
    payload["provenance"] = provenance
    return ParameterSpec.from_dict(payload)


def register_extension_parameter_source(
    manager: Any,
    extension_root: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Register an extension parameter source via existing database.add_source().
    """

    source_path = Path(extension_root).resolve()
    manager.database.add_source(source_path)
    manager.database.cache_dirty = True

    if output_dir is not None:
        output_path = Path(output_dir).resolve()
        mapping = getattr(manager, "_extension_output_dirs", None)
        if mapping is None:
            mapping = {}
            setattr(manager, "_extension_output_dirs", mapping)
        mapping[str(source_path)] = str(output_path)

    return source_path


def load_extension_parameters(manager: Any, extension_info: ExtensionInfo) -> Dict[str, Any]:
    """
    Load extension parameter files into the existing manager/database system.
    """

    before_keys = set(getattr(manager, "parameters", {}).keys())
    registered_keys: Set[str] = set()
    source_paths: List[str] = []

    for source_path, namespace_prefix in _iter_parameter_sources(extension_info):
        if not source_path.exists():
            raise FileNotFoundError(f"Extension parameter source not found: {source_path}")

        register_extension_parameter_source(manager, source_path)
        source_paths.append(str(source_path))

        specs = _load_specs_from_source(manager, source_path)
        for spec in specs.values():
            effective_spec = _clone_with_namespace_prefix(spec, namespace_prefix)
            manager.register_parameter(effective_spec)
            registered_keys.add(effective_spec.key)

    manager.database.cache_dirty = True

    after_keys = set(getattr(manager, "parameters", {}).keys())
    loaded_count = len(after_keys - before_keys)
    return {
        "loaded_count": loaded_count,
        "source_paths": source_paths,
        "registered_keys": sorted(registered_keys),
    }


def _extract_namespace_from_file(path: Path) -> Optional[str]:
    # None means the file declares no namespace (or is not a YAML/JSON
    # source). A declared parameter file that cannot be read or parsed raises
    # its original error: dropping its namespace would silently change which
    # parameters belong to the extension.
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        return None

    if not isinstance(payload, dict):
        return None
    namespace = payload.get("namespace")
    if not isinstance(namespace, str):
        return None
    namespace = namespace.strip().strip(".")
    return namespace or None


def _collect_extension_prefixes(extension_info: ExtensionInfo) -> List[str]:
    prefixes: List[str] = []
    for source_path, namespace_prefix in _iter_parameter_sources(extension_info):
        if namespace_prefix:
            prefixes.append(namespace_prefix)
        elif source_path.exists() and source_path.is_file():
            file_namespace = _extract_namespace_from_file(source_path)
            if file_namespace:
                prefixes.append(file_namespace)
    default_prefix = f"extensions.{_safe_extension_key(extension_info.name)}"
    prefixes.append(default_prefix)

    # Preserve order and remove duplicates.
    unique: List[str] = []
    for prefix in prefixes:
        if prefix not in unique:
            unique.append(prefix)
    return unique


def _resolve_extension_output_dir(
    manager: Any,
    extension_info: ExtensionInfo,
    output_dir: Optional[Path],
) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()

    manifest_output = extension_info.manifest.calibrated_output
    if manifest_output:
        candidate = Path(manifest_output)
        if candidate.is_absolute():
            return candidate.resolve()
        return (extension_info.root_path / candidate).resolve()

    mapping = getattr(manager, "_extension_output_dirs", None)
    if isinstance(mapping, dict):
        mapped_path = mapping.get(str(extension_info.root_path.resolve()))
        if mapped_path:
            return Path(mapped_path).resolve()

    return (extension_info.root_path / "calibrated").resolve()


def _select_keys_for_extension(
    manager: Any,
    extension_info: ExtensionInfo,
    keys: Optional[Sequence[str]],
) -> List[str]:
    if keys is not None:
        return [str(key) for key in keys]

    prefixes = _collect_extension_prefixes(extension_info)
    selected: List[str] = []
    for key in getattr(manager, "parameters", {}).keys():
        if any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes):
            selected.append(key)
    return sorted(selected)


def save_extension_parameters(
    manager: Any,
    extension_info: ExtensionInfo,
    keys: Optional[Sequence[str]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save extension parameters to the extension-local calibrated directory.
    """

    selected_keys = _select_keys_for_extension(manager, extension_info, keys)
    if not selected_keys:
        raise ValueError(
            f"No extension parameters selected for '{extension_info.name}'."
        )

    parameter_payload: Dict[str, Dict[str, Any]] = {}
    for key in selected_keys:
        spec = manager.parameters.get(key)
        if spec is None:
            spec = manager.get_parameter(key)
        if spec is None:
            continue
        parameter_payload[key] = spec.to_dict()

    if not parameter_payload:
        raise ValueError(
            f"No available ParameterSpec objects to save for '{extension_info.name}'."
        )

    target_dir = _resolve_extension_output_dir(manager, extension_info, output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    extension_key = _safe_extension_key(extension_info.name)
    output_path = target_dir / f"{extension_key}_optimized_parameters.yaml"

    prefixes = _collect_extension_prefixes(extension_info)
    namespace = prefixes[0] if prefixes else f"extensions.{extension_key}"
    timestamp = datetime.now(timezone.utc).isoformat()

    data = {
        "schema_version": "1.0.0",
        "tier": "extension",
        "namespace": namespace,
        "description": f"Extension parameter snapshot for {extension_info.name}",
        "metadata": {
            "created_by": "relic.extensions.params.save_extension_parameters",
            "extension_name": extension_info.name,
            "last_modified": timestamp,
            "source": "extension_runtime_snapshot",
        },
        "parameters": parameter_payload,
    }

    output_path.write_text(
        yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return output_path
