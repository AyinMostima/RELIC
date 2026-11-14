from __future__ import annotations

from pathlib import Path

import yaml


def test_case_parameters_file_has_no_top_level_namespace_prefix() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    case_yaml = repo_root / "tutorial" / "case" / "config" / "case_parameters.yaml"
    payload = yaml.safe_load(case_yaml.read_text(encoding="utf-8-sig"))
    assert isinstance(payload, dict)
    assert payload.get("namespace") == ""
    assert payload.get("schema_version") == "1.0.0"
    assert payload.get("tier") == "calibrated"

    parameters = payload.get("parameters")
    assert isinstance(parameters, dict)
    assert "biosphere.stress.ozone.ozone_detox_capacity" in parameters
    assert "hydrosphere.soil_moisture.field_capacity" in parameters


def _case_parameters() -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    case_yaml = repo_root / "tutorial" / "case" / "config" / "case_parameters.yaml"
    payload = yaml.safe_load(case_yaml.read_text(encoding="utf-8-sig"))
    return payload["parameters"]


REQUIRED_ENTRY_FIELDS = (
    "value",
    "units",
    "dtype",
    "bounds",
    "category",
    "description",
    "references",
    "tags",
    "requires_grad",
    "provenance",
)


def test_case_parameter_entries_follow_the_parameter_rules() -> None:
    """Every case entry carries the full metadata, states numbers as YAML numbers
    (PyYAML follows YAML 1.1: 1.0e12 is text, 1.0e+12 is a float) and passes the
    one bounds rule and the ParameterSpec construction the database loader uses;
    an entry that fails is dropped by the loader instead of overriding the key."""
    from relic.parameters.spec import (
        ParameterSpec,
        parameter_bounds_error,
        parameter_dtype_is_numeric,
    )

    failures = []
    for key, entry in _case_parameters().items():
        missing = [name for name in REQUIRED_ENTRY_FIELDS if name not in entry]
        if missing:
            failures.append(f"{key}: missing {missing}")
            continue
        error = parameter_bounds_error(entry["bounds"], entry["dtype"])
        if error is not None:
            failures.append(f"{key}: {error}")
        if parameter_dtype_is_numeric(entry["dtype"]) and isinstance(entry["value"], str):
            failures.append(f"{key}: numeric value stated as text {entry['value']!r}")
        try:
            ParameterSpec.from_dict(dict(entry, key=key))
        except (TypeError, ValueError) as exc:
            failures.append(f"{key}: {exc}")
    assert failures == []
    assert _case_parameters()["case.site.representative_area_m2"]["bounds"] == [1.0, 1.0e12]


def test_case_parameter_entries_declare_a_dimension_consistent_with_their_units(
    fresh_global_parameter_manager,
) -> None:
    """Every entry declares the dimension field of the parameter YAML schema
    (relic.parameters.schemas.parameter_schema.ParameterSpec); the declared dimension
    matches the SI vector of the units (relic.data.standards.dimensionality), and an
    override of a packaged key declares a dimension with the SI vector of the
    database entry (the tiers of one key may name the same vector differently, e.g.
    amount_flux and molar_flux)."""
    from relic.data.standards.dimensionality import is_declared_dimension_compatible
    from relic.parameters import get_global_manager
    from relic.parameters.schemas.parameter_schema import ParameterSpec as SchemaSpec

    manager = get_global_manager()
    failures = []
    for key, entry in _case_parameters().items():
        try:
            SchemaSpec(**entry)
        except ValueError as exc:
            failures.append(f"{key}: schema {exc}")
            continue
        if not is_declared_dimension_compatible(entry["units"], entry["dimension"]):
            failures.append(f"{key}: dimension {entry['dimension']!r} vs units {entry['units']!r}")
        spec = manager.get_parameter(key)
        if spec is not None and not (
            is_declared_dimension_compatible(spec.units, entry["dimension"])
            and is_declared_dimension_compatible(entry["units"], spec.dimension)
        ):
            failures.append(f"{key}: dimension {entry['dimension']!r} database {spec.dimension!r}")
    assert failures == []


def test_case_overrides_keep_the_database_units_and_dtype(fresh_global_parameter_manager) -> None:
    """A case entry for a packaged key changes the calibration choices (value,
    bounds, tags, requires_grad) only; units and dtype stay those of the
    database entry, so an annotated unit such as 'umol CO2 m-2 s-1' is read
    without a conversion and a restored binding keeps its dtype. Keys under a
    packaged namespace that the database does not declare (retired keys and
    compatibility aliases) are not declared by the case either."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    mismatches = []
    unknown = []
    for key, entry in _case_parameters().items():
        if key.startswith(("case.", "tutorial.case.")):
            continue
        spec = manager.get_parameter(key)
        if spec is None:
            unknown.append(key)
            continue
        for field in ("units", "dtype"):
            if str(entry[field]) != str(getattr(spec, field)):
                mismatches.append(
                    f"{key}.{field}: case {entry[field]!r} database {getattr(spec, field)!r}"
                )
    assert unknown == []
    assert mismatches == []

    parameters = _case_parameters()
    assert parameters["biosphere.photosynthesis.vcmax25"]["units"] == "umol CO2 m-2 s-1"
    assert parameters["biosphere.photosynthesis.jmax25"]["units"] == "umol e- m-2 s-1"
    assert parameters["biosphere.photosynthesis.rd25"]["units"] == "umol CO2 m-2 s-1"
