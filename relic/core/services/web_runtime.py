"""Web interface runtime helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional
import os
from pathlib import Path


def load_interface_config(config_path: Optional[str], namespace: str) -> Optional[Dict[str, Any]]:
    """
    Load an interface configuration file using the parameter system.

    Parameters
    ----------
    config_path : Optional[str]
        Path to a YAML configuration file.
    namespace : str
        Namespace to cache the config under.

    Returns
    -------
    Optional[Dict[str, Any]]
        Parsed configuration mapping or None when no path is provided.
    """
    if not config_path:
        return None

    from relic.parameters import get_global_manager

    param_manager = get_global_manager()
    loaded = param_manager.load_yaml_file(
        config_path,
        namespace=namespace,
    )
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Interface configuration {config_path} must be a mapping, "
            f"got {type(loaded).__name__}."
        )
    return loaded


def load_web_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Load a web configuration file using the parameter system.
    """
    return load_interface_config(config_path, "interfaces.web.app")


def resolve_web_experimental_flag(enable_experimental: Optional[bool]) -> bool:
    """Resolve experimental flag for web interface."""
    if enable_experimental is None:
        return os.getenv('RELIC_EXPERIMENTAL_WEB', '').lower() in ('1', 'true', 'yes', 'on')
    return bool(enable_experimental)


def build_health_payload(
    health_checker_cls: type,
    config: Mapping[str, Any],
    detailed: bool = True,
) -> Dict[str, Any]:
    """
    Build a health payload for the web interface without embedding business logic in routes.

    Parameters
    ----------
    health_checker_cls : type
        HealthChecker class to instantiate with ``config``.
    config : Mapping[str, Any]
        Health-check configuration supplied explicitly by the caller (the web
        app passes its own configuration mapping).
    detailed : bool
        Whether to request detailed health information.

    Returns
    -------
    Dict[str, Any]
        Health payload for templates or API responses.

    Construction and check failures propagate to the caller (the web boundary
    answers them with HTTP 503); a non-mapping payload raises TypeError.
    """
    checker = health_checker_cls(dict(config))
    payload = checker.check_all(detailed=detailed)
    if not isinstance(payload, dict):
        raise TypeError(
            f"{health_checker_cls.__name__}.check_all must return a mapping, "
            f"got {type(payload).__name__}."
        )
    return payload


def ensure_web_assets(
    templates_dir: Path,
    static_dir: Path,
) -> None:
    """Verify the packaged web templates and static directories exist.

    Both directories are package data (pyproject ``interfaces/web/templates``
    and ``interfaces/web/static``); a missing one means an incomplete
    installation and raises FileNotFoundError naming it. Nothing is created:
    the web interface writes no runtime output directory.
    """
    for label, directory in (("templates", templates_dir), ("static", static_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Packaged web {label} directory is missing: {directory} "
                "(incomplete RELIC installation)."
            )
