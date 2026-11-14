"""
Audit gate for known unit-dimension mapping mismatches from v3 review.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from relic.data.standards.dimensionality import is_declared_dimension_compatible


_CASES = (
    (
        Path("relic/parameters/database/relic/isotopes/isotope_parameters.yaml"),
        "isotopes.heartbeat.health_check_interval_seconds",
        "s",
        "time",
    ),
    (
        Path("relic/parameters/database/relic/isotopes/isotope_parameters.yaml"),
        "isotopes.heartbeat.interval_seconds",
        "s",
        "time",
    ),
    (
        Path("relic/parameters/database/relic/isotopes/isotope_parameters.yaml"),
        "isotopes.heartbeat.aggregation_tolerance_seconds",
        "s",
        "time",
    ),
    (
        Path("relic/parameters/database/relic/isotopes/isotope_parameters.yaml"),
        "isotopes.heartbeat.service_timeout_seconds",
        "s",
        "time",
    ),
)


_CANONICAL_FILES = tuple(
    sorted(Path("relic/parameters/database/relic").rglob("*.yaml"))
)


def _load_parameters(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    return doc.get("parameters", {})


def test_known_unit_dimension_mismatches_are_corrected() -> None:
    """
    References:
    - docs/source/architecture_overview.md section 6.7.2 (dimension mapping policy).
    - CF-1.8 metadata conventions for units and dimensions.

    Mathematical expression:
    - [x] == units(x), with consistent dimension labels.

    Parameter meanings:
    - units: physical unit symbol in YAML.
    - dimension: semantic dimension tag used by validation and tooling.
    """
    for file_path, key, expected_units, expected_dimension in _CASES:
        params = _load_parameters(file_path)
        assert key in params, f"Missing parameter {key} in {file_path.as_posix()}"
        entry = params[key]
        assert entry.get("units") == expected_units
        assert entry.get("dimension") == expected_dimension


def test_parameter_yaml_dimensions_match_canonical_unit_vectors() -> None:
    """Every selected YAML dimension must match its canonical SI unit vector."""
    mismatches = []
    for file_path in _CANONICAL_FILES:
        for key, entry in _load_parameters(file_path).items():
            units = str(entry.get("units", "")).strip()
            dimension = str(entry.get("dimension", "")).strip()
            if not is_declared_dimension_compatible(units, dimension):
                mismatches.append(
                    f"{file_path.as_posix()}::{key}: units={units!r}, dimension={dimension!r}"
                )

    assert not mismatches, "\n".join(mismatches)
