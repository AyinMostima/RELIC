from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from relic.data.standards.dimensionality import (
    is_declared_dimension_compatible,
)
from relic.parameters.schemas import ParameterDatabase

REQUIRED_PARAMETER_FIELDS = {
    "value",
    "units",
    "dtype",
    "bounds",
    "category",
    "description",
    "references",
    "tags",
    "dimension",
    "requires_grad",
    "provenance",
}


def _iter_parameter_specs(node, path=()):
    if not isinstance(node, dict):
        return
    if "value" in node and "units" in node:
        yield ".".join(path), node
        return
    for key, value in node.items():
        if isinstance(value, dict):
            yield from _iter_parameter_specs(value, path + (str(key),))


def test_all_hydrosphere_yaml_files_valid():
    yaml_dir = Path("relic/parameters/database/relic/hydrosphere")
    yaml_files = sorted(yaml_dir.glob("*.yaml"))
    assert yaml_files, "No YAML files found in hydrosphere parameter directory"

    for yaml_file in yaml_files:
        with open(yaml_file, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        db = ParameterDatabase(**data)
        assert db.schema_version == "2.0.0"


def test_all_parameter_entries_include_required_metadata():
    yaml_roots = (
        Path("relic/parameters/database/relic"),
        Path("relic/parameters/database/calibrated"),
    )
    missing = []

    for yaml_root in yaml_roots:
        for yaml_file in sorted(yaml_root.rglob("*.yaml")):
            with yaml_file.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            for key, spec in _iter_parameter_specs(data):
                absent = sorted(REQUIRED_PARAMETER_FIELDS - set(spec))
                if absent:
                    missing.append(
                        f"{yaml_file.as_posix()}::{key}: missing={','.join(absent)}"
                    )
                tags = spec.get("tags")
                if isinstance(tags, list):
                    non_text_tags = [
                        repr(tag) for tag in tags if not isinstance(tag, str)
                    ]
                else:
                    non_text_tags = [repr(tags)]
                if non_text_tags:
                    missing.append(
                        f"{yaml_file.as_posix()}::{key}: "
                        f"non_text_tags={','.join(non_text_tags)}"
                    )
                for field in ("description", "references"):
                    if not spec.get(field):
                        missing.append(
                            f"{yaml_file.as_posix()}::{key}: empty={field}"
                        )
                units = str(spec.get("units", "")).strip()
                dimension = str(spec.get("dimension", "")).strip()
                if not is_declared_dimension_compatible(units, dimension):
                    missing.append(
                        f"{yaml_file.as_posix()}::{key}: units={units!r}, dimension={dimension!r}"
                    )

    assert not missing, "\n".join(missing)


def test_every_database_entry_follows_the_bounds_rule_of_its_dtype():
    """bounds None only for a non-numeric dtype; a numeric entry writes -inf/inf when unbounded."""
    from relic.parameters.spec import parameter_bounds_error

    violations = []
    for yaml_root in (
        Path("relic/parameters/database/relic"),
        Path("relic/parameters/database/calibrated"),
    ):
        for yaml_file in sorted(yaml_root.rglob("*.yaml")):
            with yaml_file.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            for key, spec in _iter_parameter_specs(data):
                if "bounds" not in spec:
                    continue
                reason = parameter_bounds_error(spec["bounds"], spec.get("dtype"))
                if reason is not None:
                    violations.append(f"{yaml_file.as_posix()}::{key}: {reason}")

    assert not violations, "\n".join(violations)


def _numeric_leaves(value):
    """Scalar leaves of a value, default or bounds; a Tensor wire mapping yields its data."""
    if isinstance(value, dict) and value.get("type") == "tensor":
        yield from _numeric_leaves(value.get("data"))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numeric_leaves(item)
    else:
        yield value


def test_numeric_entries_state_numbers_not_strings():
    """A numeric dtype's value, default and bounds endpoints load as YAML numbers.

    PyYAML follows YAML 1.1, where a float needs a decimal point and a signed
    exponent: 1.0e7 and 1e-6 load as strings, 1.0e+7 and 1.0e-6 as floats.
    """
    from relic.parameters.spec import parameter_dtype_is_numeric

    offenders = []
    for yaml_root in (
        Path("relic/parameters/database/relic"),
        Path("relic/parameters/database/calibrated"),
    ):
        for yaml_file in sorted(yaml_root.rglob("*.yaml")):
            with yaml_file.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            for key, spec in _iter_parameter_specs(data):
                if not parameter_dtype_is_numeric(spec.get("dtype")):
                    continue
                for field in ("value", "default", "bounds"):
                    if field not in spec or (field == "default" and spec[field] is None):
                        continue
                    offenders.extend(
                        f"{yaml_file.as_posix()}::{key}.{field}: {leaf!r}"
                        for leaf in _numeric_leaves(spec[field])
                        if isinstance(leaf, bool) or not isinstance(leaf, (int, float))
                    )

    assert not offenders, "\n".join(offenders)


def _schema_payload(dtype, bounds, value):
    return {
        "schema_version": "2.0.0",
        "tier": "relic",
        "namespace": "test",
        "description": "bounds rule fixture",
        "parameters": {
            "test.entry": {
                "value": value,
                "units": "1",
                "dtype": dtype,
                "bounds": bounds,
                "category": "model_parameter",
                "description": "fixture entry",
                "references": ["unit-test"],
                "tags": ["test"],
                "dimension": "dimensionless",
                "requires_grad": False,
                "provenance": {"source": "unit_test"},
            }
        },
    }


def test_yaml_schema_applies_the_bounds_rule_of_the_dtype():
    ParameterDatabase(**_schema_payload("string", None, "horton"))
    ParameterDatabase(**_schema_payload("float32", [float("-inf"), float("inf")], 1.0))
    with pytest.raises(ValidationError, match="bounds"):
        ParameterDatabase(**_schema_payload("float32", None, 1.0))
    with pytest.raises(ValidationError, match="inf"):
        ParameterDatabase(**_schema_payload("float64", [0.0, None], 1.0))
    with pytest.raises(ValidationError, match="text"):
        ParameterDatabase(**_schema_payload("float32", ["1.0e7", 5.0e7], 2.0e7))
