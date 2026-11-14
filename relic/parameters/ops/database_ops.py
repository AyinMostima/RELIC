"""Database load/save operations for the ParameterManager."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..spec import ParameterSpec
from ..database_manager import ParameterDatabase
from ..serialization import load, export

logger = logging.getLogger(__name__)


def _load_namespaces(manager: Any, namespaces: List[str]) -> int:
    """Load declared namespaces through the index and register their keys."""
    for namespace in namespaces:
        manager.database.load_namespace(namespace)
    loaded = 0
    for key, spec in manager.database.cache.items():
        if any(key == namespace or key.startswith(f"{namespace}.") for namespace in namespaces):
            manager.register_parameter(spec)
            loaded += 1
    return loaded


def _load_directory_alias(manager: Any, directory: Path) -> int:
    """Load the namespaces declared by the participating files inside ``directory``."""
    return _load_namespaces(manager, manager.database.declared_namespaces(directory))


def _resolve_calibrated_yaml_base_parameters(
    manager: Any,
    candidate_path: Path,
) -> Optional[Dict[str, ParameterSpec]]:
    if not manager.database._is_calibrated_path(candidate_path):
        return None
    if not manager.database.cache:
        # The base values a calibrated file overrides are those of its declared
        # namespace; a file without a declared namespace needs the full load.
        namespace = manager.database._declared_namespace(candidate_path)
        if namespace:
            manager.database.load_namespace(namespace)
        else:
            manager.database.load_all()
    return manager.database.cache


def _load_alias_file_inside_db_root(
    manager: Any,
    candidate_path: Path,
) -> int:
    loaded_specs: Dict[str, ParameterSpec] = {}
    suffix = candidate_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        base_parameters = _resolve_calibrated_yaml_base_parameters(
            manager=manager,
            candidate_path=candidate_path,
        )
        loaded_specs = manager.database._load_yaml_file(
            candidate_path, base_parameters=base_parameters
        )
    elif suffix == ".json":
        loaded_specs = manager.database._load_json_file(candidate_path)

    for spec in loaded_specs.values():
        manager.register_parameter(spec)
    return len(loaded_specs)


def _try_load_single_segment_namespace_alias(
    manager: Any,
    is_absolute: bool,
    normalized_parts: List[str],
    tier_prefixes: Tuple[str, ...],
) -> Optional[int]:
    if (
        not is_absolute
        and len(normalized_parts) == 1
        and not Path(normalized_parts[0]).suffix
        and normalized_parts[0] not in tier_prefixes
    ):
        namespace = normalized_parts[0]
        if manager.database.declares_namespace(namespace):
            return _load_namespaces(manager, [namespace])
    return None


def _try_load_alias_inside_database_root(
    manager: Any,
    db_path_str: str,
    tier_prefixes: Tuple[str, ...],
) -> Optional[int]:
    try:
        candidate_path = Path(db_path_str).resolve()
        db_root = Path(manager.database_path).resolve()
        relative = candidate_path.relative_to(db_root)
    except (ValueError, OSError, RuntimeError):
        relative = None
        candidate_path = None

    if relative is None or not getattr(relative, "parts", None):
        return None

    segments = [seg for seg in relative.parts if seg]
    if not segments or segments[0] not in tier_prefixes:
        return None
    if candidate_path is None:
        return None

    if candidate_path.is_file():
        return _load_alias_file_inside_db_root(
            manager=manager,
            candidate_path=candidate_path,
        )

    if candidate_path.is_dir() and len(segments) >= 2:
        return _load_directory_alias(manager, candidate_path)

    return None


def _try_resolve_relative_tier_alias(
    manager: Any,
    db_path_str: str,
    db_path_normalized: str,
    is_absolute: bool,
    tier_prefixes: Tuple[str, ...],
) -> Tuple[Optional[int], str]:
    if (
        not is_absolute
        and db_path_normalized
        and any(db_path_normalized == p or db_path_normalized.startswith(f"{p}/") for p in tier_prefixes)
        and "relic/parameters/database" not in db_path_normalized
    ):
        segments = [seg for seg in db_path_normalized.split("/") if seg]
        if segments:
            suffix = Path(segments[-1]).suffix.lower()
            if suffix in {".yaml", ".yml", ".json"}:
                candidate = (manager.database_path / Path(*segments)).resolve()
                if candidate.is_file():
                    return None, str(candidate)
                raise FileNotFoundError(
                    f"Parameter database file alias not found: {db_path_str} -> {candidate}"
                )
            if len(segments) >= 2:
                directory = manager.database_path / Path(*segments)
                return _load_directory_alias(manager, directory), db_path_str
    return None, db_path_str


# Metadata a single-file parameter entry must declare itself; none is injected.
_SINGLE_FILE_REQUIRED_FIELDS = ("category", "units", "bounds")


def _build_spec_from_mapping(key: str, param_data: Dict[str, Any], source: Path) -> ParameterSpec:
    missing = [name for name in _SINGLE_FILE_REQUIRED_FIELDS if name not in param_data]
    if missing:
        raise ValueError(
            f"Parameter '{key}' in {source} does not declare {missing}; "
            f"a parameter file entry states {list(_SINGLE_FILE_REQUIRED_FIELDS)}"
        )
    return ParameterSpec.from_dict({"key": key, **param_data})


def _load_single_parameter_file(manager: Any, db_path_obj: Path) -> int:
    data = load(db_path_obj)
    params_dict = data["parameters"] if "parameters" in data else data

    for key, param_data in params_dict.items():
        if key.startswith("_") or key == "metadata" or "._metadata" in key:
            continue
        if not isinstance(param_data, dict):
            raise ValueError(
                f"Parameter '{key}' in {db_path_obj} is not a parameter specification mapping "
                f"(got {type(param_data).__name__}); scalar shorthand entries are not supported"
            )
        spec = _build_spec_from_mapping(key, param_data, db_path_obj)
        manager.register_parameter(spec)

    return len(params_dict)


def _load_parameter_directory(
    manager: Any,
    db_path_obj: Path,
    source_priority: Optional[List[str]],
) -> int:
    manager.database_path = db_path_obj
    manager.database = ParameterDatabase(manager.database_path, validator=manager.validator)
    if hasattr(manager, "database"):
        if source_priority:
            params = manager.database.load_all_with_priority(source_priority)
        else:
            params = manager.database.load_all()
        for spec in params.values():
            manager.register_parameter(spec)
        return len(params)
    return 0


def _load_from_default_database(manager: Any, source_priority: Optional[List[str]]) -> int:
    if hasattr(manager, "database"):
        manager.database.cache_dirty = True
        if source_priority:
            params = manager.database.load_all_with_priority(source_priority)
        else:
            params = manager.database.load_all()
        for key, spec in params.items():
            if not key.startswith("_"):
                manager.register_parameter(spec)
        return len([k for k in params.keys() if not k.startswith("_")])
    return 0


def load_from_database(manager: Any, db_path: Optional[str] = None, source_priority: Optional[List[str]] = None) -> int:
    """Load parameters from database files."""
    if db_path:
        db_path_str = str(db_path)
        db_path_normalized = db_path_str.replace("\\", "/").strip()
        is_absolute = Path(db_path_str).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", db_path_str))
        tier_prefixes = ("relic", "calibrated")
        normalized_parts = [seg for seg in db_path_normalized.split("/") if seg]

        loaded_single_root_alias = _try_load_single_segment_namespace_alias(
            manager=manager,
            is_absolute=is_absolute,
            normalized_parts=normalized_parts,
            tier_prefixes=tier_prefixes,
        )
        if loaded_single_root_alias is not None:
            return loaded_single_root_alias

        loaded_inside_root_alias = _try_load_alias_inside_database_root(
            manager=manager,
            db_path_str=db_path_str,
            tier_prefixes=tier_prefixes,
        )
        if loaded_inside_root_alias is not None:
            return loaded_inside_root_alias

        loaded_relative_alias, resolved_db_path = _try_resolve_relative_tier_alias(
            manager=manager,
            db_path_str=db_path_str,
            db_path_normalized=db_path_normalized,
            is_absolute=is_absolute,
            tier_prefixes=tier_prefixes,
        )
        if loaded_relative_alias is not None:
            return loaded_relative_alias
        db_path = resolved_db_path

        db_path_obj = Path(db_path)

        if db_path_obj.suffix and db_path_obj.is_file():
            return _load_single_parameter_file(manager, db_path_obj)

        return _load_parameter_directory(
            manager=manager,
            db_path_obj=db_path_obj,
            source_priority=source_priority,
        )
    else:
        return _load_from_default_database(manager, source_priority)


def save_to_database(manager: Any, db_path: Optional[str] = None, format: str = "yaml") -> int:
    """Save parameters to database files."""
    if db_path:
        db_path_obj = Path(db_path)
        # Check if it's a file path (has extension) or directory path
        if db_path_obj.suffix:
            # It's a file path - save directly
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)
            export(manager.parameters, format=format, output_path=db_path_obj)
        else:
            # It's a directory path - create calibrated subdir
            output_path = db_path_obj / "calibrated"
            output_path.mkdir(parents=True, exist_ok=True)
            export(manager.parameters, format=format, output_path=output_path / f"parameters.{format}")
    else:
        output_path = manager.database_path / "calibrated"
        output_path.mkdir(parents=True, exist_ok=True)
        export(manager.parameters, format=format, output_path=output_path / f"parameters.{format}")

    return len(manager.parameters)
