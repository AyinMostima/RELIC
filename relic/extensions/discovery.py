"""
Extension discovery engine for RELIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import importlib.metadata as importlib_metadata
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .protocol import MANIFEST_FILENAME, ExtensionManifest, ManifestError, parse_manifest

logger = logging.getLogger(__name__)

_EXTENSION_ENTRY_POINT_GROUP = "relic.extensions"
_ENV_EXTENSION_DIRS = "RELIC_EXTENSION_DIRS"


@dataclass(frozen=True)
class ExtensionInfo:
    """Discovered extension metadata."""

    name: str
    root_path: Path
    manifest: ExtensionManifest
    source: str  # "directory" | "entry_point"


def build_extension_info(root_path: Path, source: str) -> ExtensionInfo:
    """Create ExtensionInfo from an extension root path."""

    root = Path(root_path).resolve()
    manifest = parse_manifest(root)
    return ExtensionInfo(
        name=manifest.name,
        root_path=root,
        manifest=manifest,
        source=source,
    )


def _iter_candidate_roots(base_dir: Path) -> Iterable[Path]:
    """
    Yield possible extension roots under a base directory.

    A directory is considered an extension root when it contains
    ``relic_extension.yaml``.
    """

    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        return

    manifest_in_base = base / MANIFEST_FILENAME
    if manifest_in_base.exists():
        yield base

    for child in base.iterdir():
        if not child.is_dir():
            continue
        if (child / MANIFEST_FILENAME).exists():
            yield child


def _parse_env_dirs(raw: str) -> List[Path]:
    """Parse RELIC_EXTENSION_DIRS (semicolon-separated paths)."""

    if not raw:
        return []

    normalized = raw.replace(",", ";")
    tokens = [token.strip() for token in normalized.split(";")]
    dirs: List[Path] = []
    for token in tokens:
        if not token:
            continue
        dirs.append(Path(token).expanduser())
    return dirs


def _discover_from_directories(
    directories: Sequence[Path],
    source: str,
) -> List[ExtensionInfo]:
    infos: List[ExtensionInfo] = []
    for directory in directories:
        for root in _iter_candidate_roots(Path(directory)):
            # parse_manifest names the manifest path and the defect in its
            # ManifestError/OSError; an invalid extension is never skipped.
            infos.append(build_extension_info(root, source=source))
    return infos


def _get_entry_points() -> List[importlib_metadata.EntryPoint]:
    """Return entry points for the RELIC extension group.

    A failure to read installed distribution metadata propagates: discovery
    must not report "no entry-point extensions" when the query itself failed.
    """

    return list(importlib_metadata.entry_points(group=_EXTENSION_ENTRY_POINT_GROUP))


def _resolve_entry_point_root(entry_point: importlib_metadata.EntryPoint) -> Path:
    """
    Resolve extension root path from an entry point.

    Supported entry point targets:
    - A string or Path with the extension directory.
    - A module object (root inferred from module __file__/__path__).
    - Any object whose class module can be resolved to a package path.

    An entry point that cannot be loaded, or whose owner module cannot be
    imported, raises the original error; a target from which no extension
    root can be located raises ValueError naming the entry point. A declared
    extension is never silently skipped.
    """

    loaded = entry_point.load()

    if isinstance(loaded, (str, Path)):
        path = Path(loaded)
        if path.is_file():
            return path.parent.resolve()
        return path.resolve()

    module_file = getattr(loaded, "__file__", None)
    if module_file:
        return Path(module_file).resolve().parent

    module_paths = getattr(loaded, "__path__", None)
    if module_paths is not None:
        first = next(iter(module_paths), None)
        if first is None:
            raise ValueError(
                f"RELIC extension entry point '{entry_point.name}' resolves to a "
                "package with an empty __path__; its extension root cannot be located."
            )
        return Path(first).resolve()

    module_name = getattr(loaded, "__module__", None)
    if module_name:
        module = import_module(module_name)
        file_path = getattr(module, "__file__", None)
        if file_path:
            return Path(file_path).resolve().parent

    raise ValueError(
        f"RELIC extension entry point '{entry_point.name}' target {loaded!r} is not a "
        "path, a module, or an object whose module has a file; its extension root "
        "cannot be located."
    )


def _discover_from_entry_points() -> List[ExtensionInfo]:
    infos: List[ExtensionInfo] = []
    for entry_point in _get_entry_points():
        root = _resolve_entry_point_root(entry_point)
        # The manifest defect is re-raised with the entry point that declared
        # it; the original error (naming the manifest path) is the cause.
        try:
            infos.append(build_extension_info(root, source="entry_point"))
        except (ManifestError, OSError) as exc:
            raise ManifestError(
                f"RELIC extension entry point '{entry_point.name}' at {root}: {exc}"
            ) from exc
    return infos


def discover_extensions(
    extra_dirs: Optional[List[Path]] = None,
    use_entry_points: bool = True,
    use_env_dirs: bool = True,
) -> List[ExtensionInfo]:
    """
    Discover all available extensions.

    Discovery order:
    1. ``extra_dirs`` (caller-provided paths)
    2. ``RELIC_EXTENSION_DIRS`` (semicolon-separated paths)
    3. Python entry points in group ``relic.extensions``
    """

    discovered: List[ExtensionInfo] = []

    if extra_dirs:
        discovered.extend(_discover_from_directories(extra_dirs, source="directory"))

    if use_env_dirs:
        env_dirs = _parse_env_dirs(os.environ.get(_ENV_EXTENSION_DIRS, ""))
        discovered.extend(_discover_from_directories(env_dirs, source="directory"))

    if use_entry_points:
        discovered.extend(_discover_from_entry_points())

    by_name: dict[str, ExtensionInfo] = {}
    for info in discovered:
        if info.name in by_name:
            continue
        by_name[info.name] = info

    return list(by_name.values())

