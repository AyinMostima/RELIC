"""External configuration loading helpers for ParameterManager."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from omegaconf import OmegaConf

from .type_utils import caller_owned_value


def load_yaml_file(
    manager: Any,
    file_path: Union[str, Path],
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a YAML configuration file using safe parsing.

    This helper is for non-parameter configuration files. If a namespace is
    provided, the loaded mapping is cached under that key. An empty document
    declares no settings ({}); a document whose top level is not a mapping
    (a list or a scalar) raises TypeError naming the file. The content is parsed
    as written: a Python-specific tag (!!python/...) is rejected by the safe
    loader and raises ValueError naming the file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as handle:
        raw_content = handle.read()

    try:
        data = yaml.safe_load(raw_content)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Configuration file {path} must contain a mapping at the top level, "
            f"got {type(data).__name__}"
        )

    if namespace:
        manager._external_configs[namespace] = data
        # The cache keeps the loaded mapping; the caller receives its own copy
        # at every depth (caller_owned_value, architecture overview Section 6.1).
        return caller_owned_value(data)

    return data


def load_omegaconf_config(
    manager: Any,
    config: Any,
    *,
    namespace: str,
) -> Dict[str, Any]:
    """
    Load an OmegaConf configuration into a plain mapping.

    The resolved config is cached under the provided namespace. A config whose
    top level is not a mapping raises TypeError naming the namespace.
    """
    if config is None:
        data: Dict[str, Any] = {}
    else:
        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError(
                f"Configuration for namespace '{namespace}' must be a mapping at the top "
                f"level, got {type(container).__name__}"
            )
        data = container

    if namespace:
        manager._external_configs[namespace] = data
        # The cache keeps the resolved mapping; the caller receives its own copy.
        return caller_owned_value(data)

    return data


def get_namespace_dict(manager: Any, namespace: str) -> Dict[str, Any]:
    """Return a copy at every depth of the cached external configuration of a namespace.

    A namespace that was never ingested has no external configuration: the
    result is the explicit empty mapping, and callers merge the parameter
    database values into it.
    """
    cached = manager._external_configs.get(namespace, {})
    # A nested part of a top-level copy would still be the cached object;
    # callers merge into the result (caller_owned_value, Section 6.1).
    return caller_owned_value(dict(cached))
