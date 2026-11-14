"""Parameter database loader for the RELIC parameter system.

This module houses the ParameterDatabase implementation, keeping file
loading and tier precedence logic separate from the unified manager.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .types import ParameterCategory, ValidationLevel, ValidationContext
from .category_aliases import normalize_category
from .spec import ParameterSpec
from .ops import file_loaders as _file_loaders
from .validation import ParameterValidator
from .serialization import YAML_AVAILABLE

logger = logging.getLogger(__name__)

import yaml

# Top-level ``namespace:`` declaration of a YAML parameter file. A block
# mapping key at column 0 is a top-level key, so the declaration is read from
# the raw bytes without parsing the file.
_NAMESPACE_HEADER = re.compile(rb"^namespace:[ \t]*['\"]?([A-Za-z0-9_.]+)['\"]?[ \t]*\r?$", re.MULTILINE)
YAMLError = yaml.YAMLError

class ParameterDatabase:
    """
    Two-tier parameter database manager.
    
    Manages parameter storage and retrieval across two database layers:
    - relic/: Native RELIC parameters
    - calibrated/: Calibration results and optimized parameters
    
    Each layer maintains full provenance and version tracking.
    """
    
    def __init__(self, base_path: Path, validator: Optional["ParameterValidator"] = None):
        """Initialize parameter database."""
        self.base_path = Path(base_path)
        self.validator = validator or ParameterValidator()
        fail_env = os.getenv("RELIC_FAIL_ON_PARAMETER_VALIDATION")
        if fail_env is None:
            ci_flag = os.getenv("CI", "").lower() in {"1", "true", "yes"}
            relic_env = os.getenv("RELIC_ENV", "").lower()
            self.fail_on_validation_errors = ci_flag or relic_env in {"ci", "prod", "production"}
        else:
            self.fail_on_validation_errors = fail_env.lower() in {"1", "true", "yes"}
        self.validation_warnings: List[str] = []

        # Create directory structure
        self.relic_dir = self.base_path / "relic"
        self.calibrated_dir = self.base_path / "calibrated"

        for dir_path in [self.relic_dir, self.calibrated_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # In-memory parameter cache
        self.cache: Dict[str, ParameterSpec] = {}
        self.cache_dirty = True
        self.additional_sources: List[Path] = []
        # Declared namespace per YAML file, keyed by path and invalidated by
        # the file's modification stamp (see _declared_namespace).
        self._header_cache: Dict[Path, Tuple[Tuple[int, int], Optional[str]]] = {}
        self._file_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_lock = threading.RLock()
        self._last_lazy_failure_reason: Optional[str] = None

        logger.info(f"Parameter database initialized at {self.base_path}")

    # Tiers of the participating parameter files (see _load_plan). Calibrated
    # files and additional source files are compared against the values read
    # before them (redundant-override detection); additional source
    # directories are not.
    _BASE_TIER = "base"
    _CALIBRATED_TIER = "calibrated"
    _SOURCE_FILE_TIER = "source_file"
    _SOURCE_DIR_TIER = "source_dir"

    @staticmethod
    def _parameter_files(directory: Path, *, recursive: bool = True) -> List[Path]:
        """YAML then JSON parameter files of ``directory`` in sorted order.

        Backup directories are skipped. A missing directory has no files.
        """
        if not directory.is_dir():
            return []
        ignore_dirs = {"backup_before_fix", ".backup"}
        finder = directory.rglob if recursive else directory.glob
        files: List[Path] = []
        for pattern in ("*.yaml", "*.json"):
            files.extend(
                path
                for path in sorted(finder(pattern))
                if not any(part in ignore_dirs for part in path.parts)
            )
        return files

    @staticmethod
    def _include_calibrated_snapshots() -> bool:
        """Whether top-level calibrated snapshot files take part in loads."""
        return os.environ.get("RELIC_INCLUDE_CALIBRATED_SNAPSHOTS", "0").lower() in {"1", "true", "yes"}

    def _load_plan(self) -> List[Tuple[Path, str, Optional[str]]]:
        """Participating files as ``(path, tier, declared namespace)`` in load order.

        This is the one description of what the database reads; the full load,
        lazy key loads and namespace loads all use it, so every path sees the
        same files in the same precedence. Order: native ``relic`` files (base
        tier), then the enabled calibrated scopes (the top-level calibrated
        snapshots only with RELIC_INCLUDE_CALIBRATED_SNAPSHOTS=1), then the
        additional sources. Later entries take precedence over earlier ones.
        The declared namespace comes from the YAML header line; JSON files and
        YAML files without that line have none and are read by the full load
        only. File and directory names play no role.
        """
        entries: List[Tuple[Path, str]] = [
            (path, self._BASE_TIER) for path in self._parameter_files(self.relic_dir)
        ]
        if not self._calibrated_tier_disabled():
            include_snapshots = self._include_calibrated_snapshots()
            scope_dirs = self._get_calibrated_scope_dirs()
            if scope_dirs:
                if include_snapshots:
                    entries.extend(
                        (path, self._CALIBRATED_TIER)
                        for path in self._parameter_files(self.calibrated_dir, recursive=False)
                    )
                for scope_dir in scope_dirs:
                    entries.extend(
                        (path, self._CALIBRATED_TIER) for path in self._parameter_files(scope_dir)
                    )
            elif include_snapshots:
                entries.extend(
                    (path, self._CALIBRATED_TIER) for path in self._parameter_files(self.calibrated_dir)
                )
        for raw_source in self.additional_sources:
            source = Path(raw_source)
            if source.is_dir():
                entries.extend((path, self._SOURCE_DIR_TIER) for path in self._parameter_files(source))
            elif source.suffix.lower() in {".yaml", ".yml", ".json"}:
                entries.append((source, self._SOURCE_FILE_TIER))
        return [
            (
                path,
                tier,
                self._declared_namespace(path) if path.suffix.lower() in {".yaml", ".yml"} else None,
            )
            for path, tier in entries
        ]

    def _declared_namespace(self, file_path: Path) -> Optional[str]:
        """Return the namespace a YAML file declares on its header line, if any.

        The header is read from the raw bytes (no YAML parse), so an
        unparseable file cannot break planning; the result is cached until the
        file's modification stamp changes.
        """
        stat = file_path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self._header_cache.get(file_path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        content = file_path.read_bytes()
        if content.startswith(b"\xef\xbb\xbf"):
            content = content[3:]
        match = _NAMESPACE_HEADER.search(content)
        namespace = match.group(1).decode("ascii") if match else None
        self._header_cache[file_path] = (stamp, namespace)
        return namespace

    @staticmethod
    def _namespaces_related(first: str, second: str) -> bool:
        """True when one namespace equals, contains or lies under the other."""
        return first == second or first.startswith(second + ".") or second.startswith(first + ".")

    def _closure_entries(
        self,
        plan: List[Tuple[Path, str, Optional[str]]],
        namespaces: Iterable[str],
    ) -> List[Tuple[Path, str]]:
        """Plan entries of ``namespaces`` and of every declared namespace related to them.

        Every key of a file lies under the file's declared namespace, so any
        other file that may define the same key declares a related namespace
        (an ancestor or a descendant). Closing over relatedness therefore reads
        all owners of every key the selected files define, in plan order: a
        partially loaded cache never holds a key whose other owners (or whose
        calibrated override) have not been read.
        """
        declared = {namespace for _, _, namespace in plan if namespace is not None}
        closure: Set[str] = builtins.set(namespaces)
        frontier = list(closure)
        while frontier:
            current = frontier.pop()
            for candidate in declared:
                if candidate not in closure and self._namespaces_related(candidate, current):
                    closure.add(candidate)
                    frontier.append(candidate)
        return [(path, tier) for path, tier, namespace in plan if namespace in closure]

    def _read_plan_file(
        self,
        file_path: Path,
        base_parameters: Optional[Dict[str, ParameterSpec]],
    ) -> Dict[str, ParameterSpec]:
        """Read one participating file.

        An unreadable or invalid file is recorded in ``validation_warnings``
        and skipped; with ``fail_on_validation_errors`` it raises.
        """
        try:
            if file_path.suffix.lower() == ".json":
                return self._load_json_file(file_path)
            return self._load_yaml_file(file_path, base_parameters=base_parameters)
        except (OSError, ValueError, YAMLError) as exc:
            message = f"Failed to load parameter file {file_path}: {exc}"
            self.validation_warnings.append(message)
            logger.error(message)
            if self.fail_on_validation_errors:
                raise
            return {}

    def _read_entries(
        self,
        entries: List[Tuple[Path, str]],
        parameters: Dict[str, ParameterSpec],
        *,
        keep_existing_base: bool,
    ) -> None:
        """Read plan entries into ``parameters`` in order.

        Base-tier files may not define the same key twice: that is a database
        error naming both files. Overlay tiers replace earlier values in plan
        order. ``keep_existing_base`` keeps values already present (a partially
        loaded cache can hold overlay values that a re-read base file must not
        replace).
        """
        base_files: Dict[str, Path] = {}
        for file_path, tier in entries:
            if tier == self._BASE_TIER:
                loaded = self._read_plan_file(file_path, None)
                for key, spec in loaded.items():
                    first = base_files.setdefault(key, file_path)
                    if first != file_path:
                        raise ValueError(
                            f"Parameter '{key}' is defined in more than one file: {first}, {file_path}"
                        )
                    if keep_existing_base:
                        parameters.setdefault(key, spec)
                    else:
                        parameters[key] = spec
            else:
                compare_to = parameters if tier in {self._CALIBRATED_TIER, self._SOURCE_FILE_TIER} else None
                parameters.update(self._read_plan_file(file_path, compare_to))

    def declared_namespaces(self, directory: Optional[Path] = None) -> List[str]:
        """Declared namespaces of the participating files, in load order.

        With ``directory``, only files located in that directory (recursively).
        """
        root = Path(directory).resolve() if directory is not None else None
        namespaces: List[str] = []
        for file_path, _, namespace in self._load_plan():
            if namespace is None or namespace in namespaces:
                continue
            if root is not None and root not in file_path.resolve().parents:
                continue
            namespaces.append(namespace)
        return namespaces

    def declares_namespace(self, namespace: str) -> bool:
        """True when a participating file declares ``namespace``, an ancestor or a descendant of it."""
        return any(
            declared is not None and self._namespaces_related(declared, namespace)
            for _, _, declared in self._load_plan()
        )

    def load_namespace(self, namespace: str) -> None:
        """Load the files whose declared namespace is related to ``namespace`` into the cache.

        Reads the owners of ``namespace`` (declared dot prefixes), the files
        declaring it and its descendants, closed over relatedness (see
        _closure_entries), in the precedence of the full load. After a full
        load the cache already holds every participating file.
        """
        if not self.cache_dirty:
            return
        plan = self._load_plan()
        seeds = {
            declared
            for _, _, declared in plan
            if declared is not None and self._namespaces_related(declared, namespace)
        }
        if not seeds:
            return
        with self._load_lock:
            self._read_entries(self._closure_entries(plan, seeds), self.cache, keep_existing_base=True)

    def _ensure_cache(self) -> Dict[str, ParameterSpec]:
        """Ensure the in-memory cache is populated before introspection."""
        if self.cache_dirty:
            self.load_all()
        return self.cache

    def __len__(self) -> int:
        """Return number of parameters across all tiers."""
        return len(self._ensure_cache())

    def __iter__(self):
        """Iterate over ParameterSpec instances."""
        return iter(self._ensure_cache().values())

    def __contains__(self, key: str) -> bool:
        """Check whether a parameter key exists in the database."""
        return key in self._ensure_cache()

    def load_all(self) -> Dict[str, ParameterSpec]:
        """Load every participating file (see _load_plan) with tier precedence.

        A key defined in two base-tier files raises ValueError naming both.
        """
        if not self.cache_dirty:
            return self.cache.copy()

        # Reset metadata cache for migration checks
        self._file_metadata = {}
        if (
            not self._calibrated_tier_disabled()
            and not self._get_calibrated_scope_dirs()
            and not self._include_calibrated_snapshots()
        ):
            logger.warning(
                "No calibrated scope directories found under %s; "
                "top-level calibrated files are ignored unless "
                "RELIC_INCLUDE_CALIBRATED_SNAPSHOTS=1.",
                self.calibrated_dir,
            )
        for raw_source in self.additional_sources:
            source = Path(raw_source)
            if not source.is_dir() and source.suffix.lower() not in {".yaml", ".yml", ".json"}:
                logger.warning(f"Unsupported parameter source type: {source}")

        plan = self._load_plan()
        parameters: Dict[str, ParameterSpec] = {}
        with self._load_lock:
            self._read_entries(
                [(file_path, tier) for file_path, tier, _ in plan],
                parameters,
                keep_existing_base=False,
            )
            self.cache = parameters
            self.cache_dirty = False
        logger.info(f"Loaded {len(parameters)} parameters from database")
        return parameters.copy()

    def _get_calibrated_scopes(self) -> List[str]:
        """Return ordered calibrated scope names based on environment settings."""
        default_scopes = ["global", "regional", "site_specific"]
        raw = os.environ.get("RELIC_CALIBRATED_SCALES", ",".join(default_scopes))
        requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
        ordered: List[str] = []
        for scope in default_scopes:
            if scope in requested and scope not in ordered:
                ordered.append(scope)
        return ordered if ordered else default_scopes

    def _get_calibrated_scope_dirs(self) -> List[Path]:
        """Return existing calibrated scope directories in precedence order."""
        if not self.calibrated_dir.exists():
            return []
        scope_dirs: List[Path] = []
        for scope in self._get_calibrated_scopes():
            candidate = self.calibrated_dir / scope
            if candidate.is_dir():
                scope_dirs.append(candidate)
        return scope_dirs

    def _is_calibrated_path(self, path: Path) -> bool:
        """Return True when path is within the calibrated parameter tier."""
        try:
            return path == self.calibrated_dir or self.calibrated_dir in path.parents
        except (OSError, ValueError):
            return False

    @staticmethod
    def _calibrated_tier_disabled() -> bool:
        """Return whether calibrated parameters are excluded for this process."""
        return os.environ.get(
            "RELIC_SKIP_CALIBRATED",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _lazy_load_parameter(self, key: str) -> Optional[ParameterSpec]:
        """Load ``key`` from the files whose declared namespace owns it.

        Owners are the participating files whose declared namespace is a dot
        prefix of ``key`` (never guessed from file names). They are read with
        every related file (see _closure_entries) in the precedence of the
        full load, so a key defined in two base-tier files raises naming both.
        Returns None when no owning file defines the key; ``get_parameter``
        then performs the full load, which also reads files without a
        declared namespace.
        """
        self._last_lazy_failure_reason = None
        plan = self._load_plan()
        owners = {
            declared
            for _, _, declared in plan
            if declared is not None and key.startswith(declared + ".")
        }
        if not owners:
            self._last_lazy_failure_reason = (
                f"No parameter file declares a namespace owning '{key}'"
            )
            return None
        with self._load_lock:
            self._read_entries(self._closure_entries(plan, owners), self.cache, keep_existing_base=True)
        if key in self.cache:
            return self.cache[key]
        self._last_lazy_failure_reason = (
            f"No file declaring a namespace that owns '{key}' defines it"
        )
        return None

    def load_all_with_priority(self, source_priority: List[str]) -> Dict[str, ParameterSpec]:
        """Load the named tiers of the load plan with a custom precedence.

        ``source_priority`` lists tiers, highest precedence first: ``relic``
        (the native files) and ``calibrated`` (the calibrated files the plan
        selects: the enabled scopes, top-level snapshots only when enabled,
        none when the calibrated tier is disabled). Additional sources are not
        part of a custom-priority load. An unknown tier raises ValueError; a key
        defined in two native files raises as in ``load_all``. The database
        cache is not changed.
        """
        tiers = {"relic": self._BASE_TIER, "calibrated": self._CALIBRATED_TIER}
        unknown = [source for source in source_priority if source not in tiers]
        if unknown:
            raise ValueError(
                f"Unknown parameter sources {unknown}; supported sources: {sorted(tiers)}."
            )
        plan = self._load_plan()
        entries: List[Tuple[Path, str]] = []
        # Lowest precedence first, so later tiers override earlier ones.
        for source in reversed(list(dict.fromkeys(source_priority))):
            entries.extend(
                (file_path, tier) for file_path, tier, _ in plan if tier == tiers[source]
            )
        parameters: Dict[str, ParameterSpec] = {}
        with self._load_lock:
            self._read_entries(entries, parameters, keep_existing_base=False)
        logger.info(f"Loaded {len(parameters)} parameters with custom priority: {source_priority}")
        return parameters

    def _load_from_directory(
        self,
        directory: Path,
        base_parameters: Optional[Dict[str, ParameterSpec]] = None,
    ) -> Dict[str, ParameterSpec]:
        """Load parameters from a specific directory."""
        parameters = {}
        
        if not directory.exists():
            return parameters
        
        # Scan for parameter files
        IGNORE_DIRS = ['backup_before_fix', '.backup']
        for file_path in sorted(directory.rglob("*.yaml")):
            # ARCHITECTURAL FIX (2025-10-31): Exclude backup directories from loading.
            if any(part in IGNORE_DIRS for part in file_path.parts):
                logger.debug(f"Skipping parameter file in ignored directory: {file_path}")
                continue
            try:
                file_params = self._load_yaml_file(file_path, base_parameters=base_parameters)
                parameters.update(file_params)
            except (OSError, ValueError, YAMLError) as e:
                message = f"Failed to load parameter file {file_path}: {e}"
                self.validation_warnings.append(message)
                logger.error(message)
                if self.fail_on_validation_errors:
                    raise
        
        for file_path in sorted(directory.rglob("*.json")):
            # ARCHITECTURAL FIX (2025-10-31): Exclude backup directories from loading.
            if any(part in IGNORE_DIRS for part in file_path.parts):
                logger.debug(f"Skipping parameter file in ignored directory: {file_path}")
                continue
            try:
                file_params = self._load_json_file(file_path)
                parameters.update(file_params)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                message = f"Failed to load parameter file {file_path}: {e}"
                self.validation_warnings.append(message)
                logger.error(message)
                if self.fail_on_validation_errors:
                    raise

        return parameters

    def _validate_yaml_schema(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Validate minimal YAML schema before parsing."""
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML structure in {file_path}: expected mapping at root")
        required_fields = {"schema_version", "tier", "namespace"}
        missing = required_fields - builtins.set(data.keys())
        if missing:
            logger.warning(f"Missing required fields {missing} in {file_path}; proceeding with best-effort parse.")
        has_parameters = isinstance(data.get("parameters"), dict)
        has_overrides = isinstance(data.get("overrides"), dict)
        if not has_parameters and not has_overrides:
            logger.warning(
                f"Missing 'parameters' or 'overrides' mapping in {file_path}; using root mapping as parameter source."
            )
        if "parameters" in data and not isinstance(data.get("parameters"), dict):
            logger.warning(f"'parameters' is not a mapping in {file_path}; using root mapping as parameter source.")
        if "overrides" in data and not isinstance(data.get("overrides"), dict):
            logger.warning(f"'overrides' is not a mapping in {file_path}; ignoring overrides.")
        return True

    def _validate_spec(self, spec: ParameterSpec) -> None:
        """Validate a ParameterSpec using the attached validator."""
        if self.validator is None:
            return
        strict_mode = False
        try:
            strict_mode = self.validator.validation_level == ValidationLevel.STRICT
        except (AttributeError, TypeError):
            strict_mode = False
        context = ValidationContext(strict_mode=strict_mode)
        result = self.validator.validate_value(
            spec.key,
            spec.value,
            spec,
            getattr(self.validator, "validation_level", ValidationLevel.STANDARD),
            context,
        )
        if not result.is_valid:
            message = f"Parameter validation failed for {spec.key}: {result.errors}"
            self.validation_warnings.append(message)
            if self.fail_on_validation_errors:
                raise ValueError(message)
            logger.warning(message)

    def _load_yaml_file(
        self,
        file_path: Path,
        base_parameters: Optional[Dict[str, ParameterSpec]] = None,
    ) -> Dict[str, ParameterSpec]:
        """Load parameters from YAML file."""
        return _file_loaders.load_yaml_file(self, file_path, base_parameters=base_parameters)

    def _load_json_file(self, file_path: Path) -> Dict[str, ParameterSpec]:
        """Load parameters from JSON file."""
        return _file_loaders.load_json_file(self, file_path)

    def _parse_parameter_data(
        self,
        data: Dict[str, Any],
        source_file: Path,
        base_parameters: Optional[Dict[str, ParameterSpec]] = None,
    ) -> Dict[str, ParameterSpec]:
        """Parse parameter data from file content, supporting nested structures.

        Every entry declares its metadata: a parameter is a mapping with at
        least ``value`` and ``units``; a bare value, an entry without units, a
        non-mapping root and a non-mapping ``parameters`` or ``overrides``
        section raise ValueError naming the file and the key. No unit or
        description is invented (a missing description stays empty). An empty
        file (None) declares no parameters.
        """
        parameters: Dict[str, ParameterSpec] = {}
        if data is None:
            return parameters
        if not isinstance(data, dict):
            raise ValueError(
                f"{source_file}: a parameter file holds a mapping at its root, got {type(data).__name__}"
            )
        for section in ("parameters", "overrides"):
            if section in data and data[section] is not None and not isinstance(data[section], dict):
                raise ValueError(
                    f"{source_file}: '{section}' must be a mapping of parameter entries, "
                    f"got {type(data[section]).__name__}"
                )

        param_dict = data.get('parameters')
        overrides_dict = data.get('overrides')
        namespace_prefix = data.get('namespace')
        has_parameters = isinstance(param_dict, dict)
        has_overrides = isinstance(overrides_dict, dict)
        if not has_parameters:
            # No parameters section: the root holds the entries, unless the file
            # only declares overrides.
            param_dict = {} if has_overrides else data

        def _provenance_mapping(raw: Any, spec_key: str) -> Dict[str, Any]:
            """Return a copy of a declared provenance as a mapping.

            A mapping is kept; a string is its description; absence is empty.
            Any other type is malformed metadata and raises instead of being dropped.
            """
            if raw is None:
                return {}
            if isinstance(raw, dict):
                return dict(raw)
            if isinstance(raw, str):
                return {'description': raw}
            raise TypeError(
                f"{spec_key} in {source_file}: provenance must be a mapping or a string, "
                f"got {type(raw).__name__}"
            )

        def _is_redundant_override(spec_key: str, spec_obj: ParameterSpec) -> bool:
            if not base_parameters or spec_key not in base_parameters:
                return False
            base_spec = base_parameters[spec_key]

            def _canonical_spec_payload(spec: ParameterSpec) -> Dict[str, Any]:
                payload = spec.to_dict()
                payload.pop("provenance", None)
                if "tags" in payload and payload["tags"] is not None:
                    payload["tags"] = sorted(payload["tags"])
                if "references" in payload and payload["references"] is not None:
                    payload["references"] = sorted(payload["references"])
                return payload

            return _canonical_spec_payload(spec_obj) == _canonical_spec_payload(base_spec)

        def _require_declared_fields(spec_data: Dict[str, Any], spec_key: str) -> None:
            # A parameter entry declares its metadata; nothing is inferred or assumed.
            for required_field in ('units', 'dtype', 'bounds', 'category'):
                if required_field not in spec_data:
                    raise ValueError(
                        f"{source_file}: parameter '{spec_key}' declares no '{required_field}'"
                    )
            if str(spec_data['dtype']).strip().lower() in ('', 'none'):
                raise ValueError(f"{source_file}: parameter '{spec_key}' declares an empty 'dtype'")

        def process_node(prefix: str, node: Any) -> None:
            if isinstance(node, dict):
                if 'value' in node:
                    # If we have a value, this is a parameter definition
                    spec_key = prefix
                    if namespace_prefix and spec_key and not spec_key.startswith(f"{namespace_prefix}."):
                        spec_key = f"{namespace_prefix}.{spec_key}"
                    if not spec_key:
                        logger.warning(f"Skipping parameter without key in {source_file}")
                        return
                    spec_data = node.copy()
                    spec_data['key'] = spec_key
                    _require_declared_fields(spec_data, spec_key)
                    provenance = _provenance_mapping(spec_data.get('provenance'), spec_key)
                    provenance['source_file'] = str(source_file)
                    provenance.setdefault('loaded_at', None)
                    spec_data['provenance'] = provenance
                    if 'category' in spec_data and isinstance(spec_data['category'], str):
                        spec_data['category'] = normalize_category(spec_data['category'], logger=logger)
                    try:
                        spec = ParameterSpec.from_dict(spec_data)
                        skip_validation = (
                            str(getattr(spec, "category", "")) == ParameterCategory.PROCESS_COUPLING.value
                            and str(spec.dtype).lower() in {"str", "string"}
                        )
                        if skip_validation:
                            # Avoid numeric validation for interface mapping metadata
                            pass
                        else:
                            self._validate_spec(spec)
                        if _is_redundant_override(spec_key, spec):
                            logger.debug("Skipping redundant calibrated parameter %s from %s", spec_key, source_file)
                        else:
                            parameters[spec_key] = spec
                    except (ValueError, TypeError) as spec_error:
                        logger.error(
                            f"Failed to create ParameterSpec for {spec_key} in {source_file}: {spec_error}"
                        )
                        if self.fail_on_validation_errors:
                            raise
                else:
                    for child_key, child_value in node.items():
                        child_prefix = f"{prefix}.{child_key}" if prefix else child_key
                        process_node(child_prefix, child_value)
            else:
                # A leaf that is not an entry mapping carries no declared units,
                # dtype or description; it is rejected, never auto-loaded.
                raise ValueError(
                    f"{source_file}: '{prefix}' is a bare value; a parameter entry is a mapping "
                    "with at least 'value' and 'units'"
                )

        def _merge_override_spec(
            spec_key: str,
            override_data: Dict[str, Any],
            base_specs: Optional[Dict[str, ParameterSpec]],
        ) -> Dict[str, Any]:
            if base_specs and spec_key in base_specs:
                base_spec = base_specs[spec_key]
                spec_data = base_spec.to_dict()
            else:
                spec_data = {}

            override = override_data.copy()

            if "tags" in override:
                base_tags = builtins.set(spec_data.get("tags", []) or [])
                override_tags = builtins.set(override.get("tags", []) or [])
                spec_data["tags"] = list(base_tags | override_tags)
                override.pop("tags", None)

            if "references" in override:
                base_refs = list(spec_data.get("references", []) or [])
                override_refs = list(override.get("references", []) or [])
                merged_refs = base_refs[:]
                for ref in override_refs:
                    if ref not in merged_refs:
                        merged_refs.append(ref)
                spec_data["references"] = merged_refs
                override.pop("references", None)

            if "provenance" in override:
                base_prov = _provenance_mapping(spec_data.get("provenance"), spec_key)
                base_prov.update(_provenance_mapping(override.pop("provenance"), spec_key))
                spec_data["provenance"] = base_prov

            spec_data.update(override)
            spec_data["key"] = spec_key
            provenance = _provenance_mapping(spec_data.get("provenance"), spec_key)
            provenance["source_file"] = str(source_file)
            if base_specs and spec_key in base_specs:
                base_source = getattr(base_specs[spec_key], "provenance", {}).get("source_file")
                if base_source:
                    provenance["override_base_source"] = base_source
            spec_data["provenance"] = provenance
            return spec_data

        if isinstance(overrides_dict, dict):
            merged_overrides: Dict[str, Any] = {}
            for key, override_value in overrides_dict.items():
                if not isinstance(override_value, dict):
                    raise ValueError(
                        f"override entry '{key}' in {source_file} must be a mapping of spec "
                        f"fields, got {type(override_value).__name__}"
                    )
                spec_key = key
                if namespace_prefix and spec_key and not spec_key.startswith(f"{namespace_prefix}."):
                    spec_key = f"{namespace_prefix}.{spec_key}"
                merged_overrides[spec_key] = _merge_override_spec(
                    spec_key,
                    override_value,
                    base_parameters,
                )
            param_dict = {**param_dict, **merged_overrides}

        metadata_keys = {
            "metadata",
            "_metadata",
            "schema_version",
            "tier",
            "namespace",
            "description",
            "calibration_scale",
            "calibration_info",
            "parameters",
            "overrides",
        }

        # For top-level YAML files where keys are full parameter paths
        for key, value in param_dict.items():
            if key in metadata_keys:
                continue
            if isinstance(value, dict) and 'value' in value:
                # This is a parameter definition with full key path
                spec_key = key
                if namespace_prefix and spec_key and not spec_key.startswith(f"{namespace_prefix}."):
                    spec_key = f"{namespace_prefix}.{spec_key}"
                spec_data = value.copy()
                spec_data['key'] = spec_key
                _require_declared_fields(spec_data, spec_key)
                provenance = _provenance_mapping(spec_data.get('provenance'), spec_key)
                provenance['source_file'] = str(source_file)
                provenance.setdefault('loaded_at', None)
                spec_data['provenance'] = provenance
                if 'category' in spec_data and isinstance(spec_data['category'], str):
                    spec_data['category'] = normalize_category(spec_data['category'], logger=logger)
                try:
                    spec = ParameterSpec.from_dict(spec_data)
                    skip_validation = (
                        str(getattr(spec, "category", "")) == ParameterCategory.PROCESS_COUPLING.value
                        and str(spec.dtype).lower() in {"str", "string"}
                    )
                    if not skip_validation:
                        self._validate_spec(spec)
                    if _is_redundant_override(spec_key, spec):
                        logger.debug("Skipping redundant calibrated parameter %s from %s", spec_key, source_file)
                    else:
                        parameters[spec_key] = spec
                except (ValueError, TypeError) as spec_error:
                    logger.error(
                        f"Failed to create ParameterSpec for {spec_key} in {source_file}: {spec_error}"
                    )
                    if self.fail_on_validation_errors:
                        raise
            else:
                # Recursively process nested structures
                process_node('', {key: value})
        return parameters
    
    def save_parameters(self, parameters: Dict[str, ParameterSpec], 
                       tier: str = 'calibrated',
                       filename: str = 'saved_parameters.yaml') -> None:
        """Save parameters to specific database tier."""
        if tier == 'relic':
            target_dir = self.relic_dir
        elif tier == 'calibrated':
            target_dir = self.calibrated_dir
        else:
            raise ValueError(f"Invalid database tier: {tier}")
        
        target_file = target_dir / filename
        
        # Prepare data for serialization
        timestamp = datetime.now(timezone.utc).isoformat()
        param_data = {
            "schema_version": "1.0.0",
            "tier": tier,
            "namespace": f"{tier}.saved_parameters",
            "description": "Saved parameters snapshot (auto-generated)",
            "metadata": {
                "created_by": "ParameterManager.save_parameters",
                "last_modified": timestamp,
                "source": "runtime_snapshot",
            },
            "parameters": {},
        }
        for key, spec in parameters.items():
            missing_metadata = []
            if not str(spec.description).strip():
                missing_metadata.append("description")
            if not spec.references:
                missing_metadata.append("references")
            if not str(spec.dimension).strip():
                missing_metadata.append("dimension")
            if missing_metadata:
                missing_fields = ", ".join(missing_metadata)
                raise ValueError(
                    f"Parameter '{key}' has incomplete persistence metadata: "
                    f"{missing_fields}"
                )
            param_data['parameters'][key] = spec.to_dict()
        
        # Save as YAML (preferred) or JSON
        if YAML_AVAILABLE and filename.endswith('.yaml'):
            with open(target_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(param_data, f, default_flow_style=False, indent=2, sort_keys=False)
        else:
            # Default to JSON
            json_filename = filename.replace('.yaml', '.json')
            target_file = target_dir / json_filename
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(param_data, f, indent=2)
        
        logger.info(f"Saved {len(parameters)} parameters to {target_file}")
        
        # Invalidate cache
        self.cache_dirty = True

    def add_source(self, source_path: Path) -> None:
        """Register an additional parameter source (directory or file)."""
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Parameter source does not exist: {path}")
        if path not in self.additional_sources:
            self.additional_sources.append(path)
            self.cache_dirty = True
    
    def get_parameter(self, key: str) -> Optional[ParameterSpec]:
        """
        Internal database-level ParameterSpec lookup by key.

        Notes:
            This method bypasses higher-level manager convenience behavior and
            should not be used as a direct scientific functor access path.
        """
        if key in self.cache:
            return self.cache.get(key)

        if self.cache_dirty:
            spec = self._lazy_load_parameter(key)
            if spec is not None:
                return spec

            if self.fail_on_validation_errors:
                reason = self._last_lazy_failure_reason or f"Lazy load failed for '{key}' with strict validation enabled"
                raise KeyError(reason)

            if self._last_lazy_failure_reason:
                logger.info(
                    "Lazy load default to full database scan for key '%s' (%s)",
                    key,
                    self._last_lazy_failure_reason,
                )
            else:
                logger.info("Lazy load default to full database scan for key '%s'", key)
            # Default to full load when lazy path cannot resolve the key
            self.load_all()

        return self.cache.get(key)
    
    def set_parameter(self, spec: ParameterSpec, tier: str = 'calibrated') -> None:
        """Set parameter in specific database tier."""
        # Update cache
        if self.cache_dirty:
            self.load_all()
        
        self.cache[spec.key] = spec
        
        # Save to database
        self.save_parameters({spec.key: spec}, tier)


