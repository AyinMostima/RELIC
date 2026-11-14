"""Hydrosphere YAML dimension labels must match physical unit semantics."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
HYDRO_PARAM_ROOT = REPO_ROOT / "relic" / "parameters" / "database" / "relic" / "hydrosphere"
NON_PHYSICAL_UNITS = {
    "",
    "-",
    "none",
    "n/a",
    "dimensionless",
    "bool",
    "boolean",
    "string",
    "enum",
    "category",
}


def _is_physical_units(units: str) -> bool:
    unit_text = str(units).strip().lower()
    if unit_text in NON_PHYSICAL_UNITS:
        return False
    if "dimensionless" in unit_text:
        return False
    return True


def test_hydrosphere_physical_parameters_not_labeled_configuration_dimension() -> None:
    violations: list[str] = []

    for yaml_path in HYDRO_PARAM_ROOT.glob("*.yaml"):
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        params = data.get("parameters", {}) if isinstance(data, dict) else {}
        if not isinstance(params, dict):
            continue

        for key, spec in params.items():
            if not isinstance(spec, dict):
                continue
            units = str(spec.get("units", "")).strip()
            dimension = str(spec.get("dimension", "")).strip().lower()
            if dimension != "configuration":
                continue
            if not _is_physical_units(units):
                continue
            violations.append(f"{yaml_path.name}:{key} units={units}")

    assert not violations, (
        "Found hydrosphere parameters using physical units but labeled as "
        "dimension=configuration:\n" + "\n".join(sorted(violations))
    )

