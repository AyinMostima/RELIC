"""Parameter export helpers for the unified parameter system."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..serialization import YAML_AVAILABLE, TOML_AVAILABLE

import yaml
import toml  # type: ignore[import-untyped]


def export_parameters(manager: Any, output_path: str, format: str = "yaml") -> None:
    """
    Export parameters to YAML/JSON/TOML.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    output_path : str
        File path to write export output.
    format : str
        Export format (yaml, json, toml).
    """
    export_namespace(
        manager=manager,
        namespace=None,
        format=format,
        output_path=Path(output_path),
    )


def export_namespace(
    manager: Any,
    namespace: Optional[str] = None,
    format: str = "yaml",
    output_path: Optional[Path] = None,
) -> Optional[str]:
    """
    Export parameters under an optional namespace to string or file.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    namespace : Optional[str]
        Namespace prefix to filter parameters.
    format : str
        Export format (yaml, json, toml).
    output_path : Optional[Path]
        If provided, write to file and return None.

    Returns
    -------
    Optional[str]
        Exported content when output_path is None, otherwise None.
    """
    if namespace:
        export_params = {
            key: spec for key, spec in manager.parameters.items() if key.startswith(namespace)
        }
    else:
        export_params = manager.parameters.copy()

    export_data: dict[str, dict[str, Any]] = {"parameters": {}}
    for key, spec in export_params.items():
        if hasattr(spec, "to_dict"):
            export_data["parameters"][key] = spec.to_dict()
        elif hasattr(spec, "model_dump"):
            export_data["parameters"][key] = spec.model_dump()
        elif hasattr(spec, "__dict__"):
            export_data["parameters"][key] = dict(spec.__dict__)
        else:
            export_data["parameters"][key] = spec

    format_lower = format.lower()
    if format_lower == "yaml":
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML export")
        formatted_data = yaml.dump(export_data, default_flow_style=False, indent=2)
    elif format_lower == "json":
        formatted_data = json.dumps(export_data, indent=2)
    elif format_lower == "toml":
        if not TOML_AVAILABLE:
            raise ImportError("toml required for TOML export")
        formatted_data = toml.dumps(export_data)
    else:
        raise ValueError(f"Unsupported export format: {format}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(formatted_data)
        return None

    return formatted_data
