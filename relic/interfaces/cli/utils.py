"""Common CLI helpers shared by command modules."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

# PyYAML is a RELIC dependency (the parameter database is YAML); a YAML output
# request is always written as YAML, never silently as JSON.
import yaml


class Colors:
    """ANSI color constants (kept empty for cross-platform safe defaults)."""

    RESET = ""
    RED = ""
    GREEN = ""
    YELLOW = ""
    CYAN = ""


def colorize(text: str, color: str) -> str:
    """Apply terminal color formatting when enabled."""
    if not color:
        return text
    return f"{color}{text}{Colors.RESET}"


def _print_prefixed(level: str, message: str) -> None:
    """Print a normalized status line."""
    print(f"[{level}] {message}")


def print_success(message: str) -> None:
    """Print success message."""
    _print_prefixed("OK", message)


def print_error(message: str) -> None:
    """Print error message."""
    _print_prefixed("ERROR", message)


def print_warning(message: str) -> None:
    """Print warning message."""
    _print_prefixed("WARN", message)


def print_info(message: str) -> None:
    """Print informational message."""
    _print_prefixed("INFO", message)


def print_header(title: str) -> None:
    """Print section header."""
    line = "=" * max(len(title), 8)
    print("")
    print(title)
    print(line)


def split_names_csv(value: Optional[str]) -> List[str]:
    """Split comma-delimited names into normalized tokens."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def format_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Render a simple ASCII table from row dictionaries."""
    if not rows:
        return "(no data)"

    columns = list(rows[0].keys())
    widths: Dict[str, int] = {col: len(str(col)) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    header = " | ".join(f"{col:<{widths[col]}}" for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
    body = [
        " | ".join(f"{str(row.get(col, '')):<{widths[col]}}" for col in columns)
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _json_default(value: Any) -> Any:
    """JSON representation for values the json module cannot encode.

    Arrays, tensors, and numpy scalars are written as their values through
    ``tolist()`` (nested lists, or a Python scalar for 0-d values). Any other
    object has no JSON representation and raises TypeError (the json module
    contract); it is never written as its string repr.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_output(data: Any, output_path: str, output_format: str = "json") -> Path:
    """Persist command output to JSON/YAML/text file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (output_format or "json").lower()

    if fmt == "json":
        path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
        return path

    if fmt == "yaml":
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
        return path

    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, indent=2, default=_json_default)
    path.write_text(text, encoding="utf-8")
    return path


def check_experimental_feature(name: str, env_var: str) -> bool:
    """Check whether an experimental feature flag is enabled."""
    value = os.getenv(env_var, "").strip().lower()
    enabled = value in {"1", "true", "yes", "on"}
    if not enabled:
        print_info(f"{name} is disabled. Set {env_var}=1 to enable.")
    return enabled


@dataclass
class ProgressReporter:
    """Simple progress reporter for CLI workflows."""

    total: Optional[int] = None
    description: str = "Progress"
    _current: int = 0
    _start_time: float = field(default_factory=time.time)

    def __enter__(self) -> "ProgressReporter":
        print_info(f"{self.description} started")
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> Literal[False]:
        elapsed = time.time() - self._start_time
        if exc_type is not None:
            print_error(f"{self.description} failed after {elapsed:.2f}s")
            return False
        print_success(f"{self.description} completed in {elapsed:.2f}s")
        return False

    def update(self, step: int = 1, label: Optional[str] = None) -> None:
        """Advance progress by step and optionally print stage label."""
        self._current += int(step)
        if self.total and self.total > 0:
            prefix = f"{self._current}/{self.total}"
        else:
            prefix = str(self._current)
        if label:
            print_info(f"{self.description} [{prefix}] {label}")
        else:
            print_info(f"{self.description} [{prefix}]")


__all__ = [
    "Colors",
    "ProgressReporter",
    "check_experimental_feature",
    "colorize",
    "format_table",
    "print_error",
    "print_header",
    "print_info",
    "print_success",
    "print_warning",
    "save_output",
    "split_names_csv",
]
