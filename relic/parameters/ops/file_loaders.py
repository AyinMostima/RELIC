from __future__ import annotations

import builtins
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

from ..serialization import YAML_AVAILABLE
import yaml
from yaml import YAMLError

from ..spec import ParameterSpec

logger = logging.getLogger(__name__)


def load_yaml_file(
    db: Any,
    file_path: Path,
    base_parameters: Optional[Dict[str, ParameterSpec]] = None,
) -> Dict[str, ParameterSpec]:
    """Load parameters from YAML file using the database instance."""
    if not YAML_AVAILABLE:
        message = f"PyYAML not available; cannot load YAML file: {file_path}"
        logger.warning(message)
        db.validation_warnings.append(message)
        return {}
    if not file_path.exists():
        logger.warning("YAML file not found: %s", file_path)
        return {}
    if not file_path.is_file():
        logger.warning("Skipping non-file YAML path: %s", file_path)
        return {}

    def _contains_python_tuple(node: Any) -> bool:
        if isinstance(node, tuple):
            return True
        if isinstance(node, dict):
            return any(_contains_python_tuple(v) for v in node.values())
        if isinstance(node, list):
            return any(_contains_python_tuple(v) for v in node)
        if isinstance(node, builtins.set):
            return any(_contains_python_tuple(v) for v in node)
        return False

    def _replace_tuples(node: Any) -> Any:
        if isinstance(node, tuple):
            return [_replace_tuples(v) for v in node]
        if isinstance(node, list):
            return [_replace_tuples(v) for v in node]
        if isinstance(node, dict):
            return {k: _replace_tuples(v) for k, v in node.items()}
        return node

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            raw_content = f.read()
        python_tag_in_text = "!!python/" in raw_content
        sanitized_content = raw_content
        already_warned = False
        if python_tag_in_text:
            message = (
                f"Unsupported python-specific YAML tag found in {file_path}; "
                "python tags are not portable across environments."
            )
            db.validation_warnings.append(message)
            logger.warning(message)
            already_warned = True
            if db.fail_on_validation_errors:
                raise ValueError(message)
            sanitized_content = sanitized_content.replace("!!python/tuple", "")
            sanitized_content = sanitized_content.replace("!!python/object", "")

        try:
            data = yaml.safe_load(sanitized_content)
        except YAMLError as exc:
            if python_tag_in_text and not db.fail_on_validation_errors:
                default_content = sanitized_content.replace("!!python/", "!")
                data = yaml.safe_load(default_content)
            else:
                raise exc
        python_tag_detected = python_tag_in_text or _contains_python_tuple(data)
        if python_tag_detected and not already_warned:
            message = (
                f"Unsupported python-specific YAML tag found in {file_path}; "
                "python tags are not portable across environments."
            )
            db.validation_warnings.append(message)
            logger.warning(message)
            if db.fail_on_validation_errors:
                raise ValueError(message)
            data = _replace_tuples(data)
        db._file_metadata[str(file_path)] = data.get("metadata", {}) if isinstance(data, dict) else {}
        db._validate_yaml_schema(data, file_path)
        parsed = db._parse_parameter_data(data, file_path, base_parameters=base_parameters)
        if isinstance(parsed, dict):
            return cast(Dict[str, ParameterSpec], parsed)
        return {}
    except YAMLError as exc:
        logger.error("Error loading YAML file %s: %s", file_path, exc)
        raise
    except ValueError as exc:
        logger.error("Parameter validation error in %s: %s", file_path, exc)
        raise
    except (OSError, TypeError, RuntimeError) as exc:
        logger.error("Unexpected error loading YAML file %s: %s", file_path, exc, exc_info=True)
        raise


def load_json_file(db: Any, file_path: Path) -> Dict[str, ParameterSpec]:
    """Load parameters from JSON file using the database instance."""
    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        parsed = db._parse_parameter_data(data, file_path)
        if isinstance(parsed, dict):
            return cast(Dict[str, ParameterSpec], parsed)
        return {}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Error loading JSON file %s: %s", file_path, exc)
        return {}

