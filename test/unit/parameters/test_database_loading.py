import logging
import re
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from relic.parameters import ParameterDatabase
from relic.parameters.validation import ParameterValidator
from relic.parameters.types import ValidationLevel


def _make_database(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, *, fail_on: bool = False,
                   level: ValidationLevel = ValidationLevel.STANDARD) -> ParameterDatabase:
    """Helper to create a ParameterDatabase rooted at an accessible temporary path."""
    if fail_on:
        monkeypatch.setenv("RELIC_FAIL_ON_PARAMETER_VALIDATION", "1")
    else:
        monkeypatch.delenv("RELIC_FAIL_ON_PARAMETER_VALIDATION", raising=False)
    base_root = Path.cwd() / "output" / "tmp_parameters"
    base_root.mkdir(parents=True, exist_ok=True)
    base = base_root / f"run_{uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    validator = ParameterValidator(level=level)
    request.addfinalizer(lambda: shutil.rmtree(base, ignore_errors=True))
    db = ParameterDatabase(base, validator=validator)
    db.fail_on_validation_errors = fail_on
    return db


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"overrides": {"test.override": 2.0}}, "override entry"),
        ({"overrides": {"test.override": {"value": 2.0, "provenance": ["not", "a", "mapping"]}}}, "provenance"),
        ({"parameters": {"test.top": {"value": 1.0, "units": "1", "dtype": "float64", "bounds": [0.0, 2.0],
                                        "category": "general", "provenance": 7}}}, "provenance"),
    ],
)
def test_malformed_override_or_provenance_is_rejected(request: pytest.FixtureRequest,
                                                      monkeypatch: pytest.MonkeyPatch, data, match):
    """A malformed entry raises; it is neither skipped nor given empty provenance."""
    from relic.parameters import ParameterSpec

    db = _make_database(request, monkeypatch, fail_on=False)
    base = {"test.override": ParameterSpec(key="test.override", value=1.0, units="1", dtype="float64", bounds=(-float("inf"), float("inf")), category="model_parameter")}

    with pytest.raises((TypeError, ValueError), match=match):
        db._parse_parameter_data(data, source_file=Path("probe.yaml"), base_parameters=base)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"probe": {"bare": 3.0}}, "probe.bare"),
        ({"parameters": {"probe.no_units": {"value": 1.0, "dtype": "float64"}}}, "probe.no_units"),
        ({"probe.no_units": {"value": 1.0, "dtype": "float64"}}, "probe.no_units"),
        ({"parameters": {"probe.no_bounds": {"value": 1.0, "units": "1", "dtype": "float64"}}},
         "probe.no_bounds.*'bounds'"),
        ({"probe.no_bounds": {"value": 1.0, "units": "1", "dtype": "float64"}}, "probe.no_bounds.*'bounds'"),
        ({"parameters": {"probe.no_dtype": {"value": 1.0, "units": "1", "bounds": [0.0, 2.0],
                                          "category": "general"}}}, "probe.no_dtype.*'dtype'"),
        ({"probe.empty_dtype": {"value": 1.0, "units": "1", "dtype": None, "bounds": [0.0, 2.0],
                                "category": "general"}}, "probe.empty_dtype.*'dtype'"),
        ({"parameters": {"probe.no_category": {"value": 1.0, "units": "1", "dtype": "float64",
                                             "bounds": [0.0, 2.0]}}}, "probe.no_category.*'category'"),
        ({"parameters": [{"key": "probe.x", "value": 1.0}]}, "'parameters'"),
        ({"overrides": ["probe.x"]}, "'overrides'"),
    ],
)
def test_entries_without_declared_metadata_are_rejected_with_file_and_key(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, data, match
):
    """No unit or description is invented, and no section is silently ignored."""
    db = _make_database(request, monkeypatch, fail_on=False)
    with pytest.raises(ValueError, match=match) as error:
        db._parse_parameter_data(data, source_file=Path("probe_source.yaml"))
    assert "probe_source.yaml" in str(error.value)


def test_non_mapping_parameter_file_is_rejected(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    db = _make_database(request, monkeypatch, fail_on=False)
    with pytest.raises(ValueError, match="probe_list.yaml"):
        db._parse_parameter_data([1, 2, 3], source_file=Path("probe_list.yaml"))
    assert db._parse_parameter_data(None, source_file=Path("empty.yaml")) == {}


def test_missing_description_stays_empty_instead_of_auto_loaded(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    db = _make_database(request, monkeypatch, fail_on=False)
    parsed = db._parse_parameter_data(
        {"parameters": {"probe.plain": {"value": 1.0, "units": "1", "dtype": "float64", "bounds": [0.0, 2.0],
                                        "category": "general"}}},
        source_file=Path("probe.yaml"),
    )
    assert parsed["probe.plain"].description == ""


def test_string_provenance_is_kept_as_its_description(request: pytest.FixtureRequest,
                                                      monkeypatch: pytest.MonkeyPatch):
    db = _make_database(request, monkeypatch, fail_on=False)
    parsed = db._parse_parameter_data(
        {"parameters": {"test.top": {"value": 1.0, "units": "1", "dtype": "float64", "bounds": [0.0, 2.0],
                                    "category": "general", "provenance": "field survey"}}},
        source_file=Path("probe.yaml"),
    )
    assert parsed["test.top"].provenance["description"] == "field survey"
    assert parsed["test.top"].provenance["source_file"] == "probe.yaml"


def test_invalid_yaml_tag_warns_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
                                           caplog: pytest.LogCaptureFixture):
    """Invalid YAML tags should be recorded but not raise when fail-on is disabled."""
    db = _make_database(request, monkeypatch, fail_on=False)
    bad_file = Path(db.calibrated_dir) / "bad.yaml"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text(
        "parameters:\n"
        "  test.param:\n"
        "    bounds: !!python/tuple [1.0, 2.0]\n"
        "    value: 1.0\n"
        "    units: dimensionless\n"
        "    dtype: float32\n"
        "    category: general\n",
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING, logger="relic.parameters.manager")
    params = db._load_yaml_file(bad_file)
    assert any("python-specific YAML tag" in msg for msg in db.validation_warnings)
    # Parameter should still be parsed after sanitization
    assert "test.param" in params


def test_invalid_yaml_tag_fails_when_strict(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Strict mode should fail fast on unsupported YAML tags."""
    db = _make_database(request, monkeypatch, fail_on=True)
    bad_file = Path(db.calibrated_dir) / "bad.yaml"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text(
        "parameters:\n"
        "  test.param:\n"
        "    bounds: !!python/tuple [1.0, 2.0]\n"
        "    value: 1.0\n"
        "    units: dimensionless\n"
        "    dtype: float32\n"
        "    category: general\n",
        encoding="utf-8",
    )

    assert db.fail_on_validation_errors is True
    with pytest.raises(ValueError):
        db._load_yaml_file(bad_file)


def test_missing_dtype_raises_in_strict_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Missing dtype should raise when strict validation or fail-on is enabled."""
    db = _make_database(request, monkeypatch, fail_on=True, level=ValidationLevel.STRICT)
    file_path = Path(db.relic_dir) / "strict.yaml"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "schema_version: '1.0.0'\n"
        "tier: 'relic'\n"
        "namespace: 'unregistered.test'\n"
        "parameters:\n"
        "  unregistered.test.alpha:\n"
        "    value: 1.0\n"
        "    units: dimensionless\n"
        "    bounds: [0.0, 10.0]\n"
        "    category: general\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        db.load_all()


def test_missing_dtype_is_not_inferred_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
                                                  caplog: pytest.LogCaptureFixture):
    """Without fail-on, a file with an entry lacking dtype is rejected (recorded, not loaded); no dtype is inferred."""
    db = _make_database(request, monkeypatch, fail_on=False)
    file_path = Path(db.relic_dir) / "default.yaml"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "schema_version: '1.0.0'\n"
        "tier: 'relic'\n"
        "namespace: 'unregistered.test'\n"
        "parameters:\n"
        "  unregistered.test.beta:\n"
        "    value: 2.0\n"
        "    units: dimensionless\n"
        "    bounds: [0.0, 10.0]\n"
        "    category: general\n",
        encoding="utf-8",
    )

    caplog.set_level(logging.WARNING)
    params = db.load_all()
    assert params.get("unregistered.test.beta") is None
    assert any("unregistered.test.beta" in msg and "'dtype'" in msg for msg in db.validation_warnings)


def test_lazy_load_without_declared_owner_respects_fail_on(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """A key no file declares a namespace for fails fast in strict mode and is found by the full load otherwise."""
    file_content = (
        "schema_version: '1.0.0'\n"
        "tier: 'relic'\n"
        "parameters:\n"
        "  unregistered.test.gamma:\n"
        "    value: 3.0\n"
        "    units: dimensionless\n"
        "    bounds: [0.0, 10.0]\n"
        "    dtype: float32\n"
        "    category: general\n"
        "    description: Test parameter\n"
    )

    # Strict mode raises instead of falling back to the full database scan
    strict_db = _make_database(request, monkeypatch, fail_on=True)
    strict_file = Path(strict_db.relic_dir) / "lazy_strict.yaml"
    strict_file.parent.mkdir(parents=True, exist_ok=True)
    strict_file.write_text(file_content, encoding="utf-8")
    with pytest.raises(KeyError):
        strict_db.get_parameter("unregistered.test.gamma")

    # Default mode falls back to the full load, which reads files without a namespace
    default_db = _make_database(request, monkeypatch, fail_on=False)
    default_file = Path(default_db.relic_dir) / "lazy_default.yaml"
    default_file.parent.mkdir(parents=True, exist_ok=True)
    default_file.write_text(file_content, encoding="utf-8")
    spec = default_db.get_parameter("unregistered.test.gamma")
    assert spec is not None
    assert spec.key == "unregistered.test.gamma"
    assert default_db.cache_dirty is False


def test_lazy_load_declared_namespace_resolves_in_strict_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """A key under a declared namespace loads lazily from its owner file even in strict mode."""
    strict_db = _make_database(request, monkeypatch, fail_on=True)
    owner = Path(strict_db.relic_dir) / "any_file_name.yaml"
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text(
        "schema_version: '1.0.0'\n"
        "tier: 'relic'\n"
        "namespace: 'declared.test'\n"
        "parameters:\n"
        "  declared.test.gamma:\n"
        "    value: 3.0\n"
        "    units: dimensionless\n"
        "    bounds: [0.0, 10.0]\n"
        "    dtype: float32\n"
        "    category: general\n"
        "    description: Test parameter\n",
        encoding="utf-8",
    )
    spec = strict_db.get_parameter("declared.test.gamma")
    assert spec is not None
    assert float(spec.value) == 3.0
    assert strict_db.cache_dirty is True




def _single_file_manager(tmp_path: Path, content: str):
    from relic.parameters import UnifiedParameterManager

    path = tmp_path / "single_parameter_file.yaml"
    path.write_text(content, encoding="utf-8")
    return UnifiedParameterManager(), path


def test_single_file_entry_without_category_is_rejected_with_file_and_key(tmp_path: Path):
    manager, path = _single_file_manager(
        tmp_path,
        "parameters:\n"
        "  test.single.rate:\n"
        "    value: 1.0\n"
        "    units: s-1\n"
        "    dtype: float64\n"
        "    bounds: [0.0, 2.0]\n"
        "    description: rate\n",
    )
    with pytest.raises(ValueError, match=r"test\.single\.rate.*category|category.*test\.single\.rate"):
        manager.load_from_database(str(path))
    assert not manager.has_parameter("test.single.rate")


def test_single_file_scalar_shorthand_is_rejected_with_file_and_key(tmp_path: Path):
    manager, path = _single_file_manager(tmp_path, "parameters:\n  test.single.shorthand: 2.5\n")
    with pytest.raises(ValueError, match="test.single.shorthand") as excinfo:
        manager.load_from_database(str(path))
    assert path.name in str(excinfo.value)
    assert not manager.has_parameter("test.single.shorthand")


def test_list_value_outside_its_componentwise_bounds_is_reported_at_load(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    """Loading judges a list value against its one [min, max] pair per component."""
    data = {"parameters": {"probe.temperature_domain": {
        "value": [200.0, 323.15], "units": "K", "dtype": "list",
        "bounds": [[223.15, 273.15], [323.15, 373.15]], "category": "general",
    }}}
    message = r"probe\.temperature_domain component 0 below minimum: 200\.0 < 223\.15"

    db = _make_database(request, monkeypatch, fail_on=False)
    db._parse_parameter_data(data, source_file=Path("probe.yaml"))
    assert len(db.validation_warnings) == 1
    assert re.search(message, db.validation_warnings[0])

    strict = _make_database(request, monkeypatch, fail_on=True)
    with pytest.raises(ValueError, match=message):
        strict._parse_parameter_data(data, source_file=Path("probe.yaml"))

    data["parameters"]["probe.temperature_domain"]["value"] = [223.15, 373.15]
    accepted = strict._parse_parameter_data(data, source_file=Path("probe.yaml"))
    assert accepted["probe.temperature_domain"].value == [223.15, 373.15]
