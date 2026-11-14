"""Parameter loading by declared namespace ownership (no file-name or directory guessing).

Every database file declares one ``namespace``; a key is loaded lazily from
the files whose declared namespace owns it (a dot prefix of the key), a
namespace load reads the files whose declared namespace is related to it, and
the full load reads the same files in the same precedence. A key defined in
two base-tier files is a database error on every path; an unknown key
resolves to nothing, so the manager raises KeyError as before.
"""

from pathlib import Path

import pytest
import yaml

from relic.parameters import UnifiedParameterManager
from relic.parameters.database_manager import ParameterDatabase

REPO_DATABASE = Path(__file__).resolve().parents[2] / "relic" / "parameters" / "database"
CLAMP_KEY = "validation.numerical.clamp_min_temperature"


def _parameter(value: float) -> dict:
    return {
        "value": value,
        "units": "K",
        "dtype": "float64",
        "bounds": [0.0, 1000.0],
        "category": "computational",
        "description": "Lazy resolution probe",
        "references": ["RELIC lazy resolution test"],
        "tags": ["probe"],
        "requires_grad": False,
        "provenance": {"source": "test"},
    }


def _write(path: Path, namespace: str, parameters: dict, tier: str = "relic") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0.0",
        "tier": tier,
        "namespace": namespace,
        "description": "Lazy resolution probe file",
        "parameters": parameters,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_key_loads_from_its_declared_namespace_owner_without_ambiguity(caplog):
    caplog.set_level("DEBUG", logger="relic.parameters.database_manager")
    database = ParameterDatabase(REPO_DATABASE)

    spec = database.get_parameter(CLAMP_KEY)

    source = yaml.safe_load(
        (REPO_DATABASE / "relic" / "validation_numerical.yaml").read_text(encoding="utf-8")
    )
    assert spec is not None
    assert float(spec.value) == float(source["parameters"][CLAMP_KEY]["value"])
    messages = [record.getMessage() for record in caplog.records]
    assert not [message for message in messages if "ambiguous" in message]
    assert not [message for message in messages if "full database scan" in message]
    assert database.cache_dirty is True  # resolved without a full load


def test_file_names_play_no_role_in_resolution(tmp_path):
    _write(tmp_path / "relic" / "zzz_unrelated_name.yaml", "probe.named", {"probe.named.value": _parameter(2.5)})
    _write(tmp_path / "relic" / "named" / "named.yaml", "probe.other", {"probe.other.value": _parameter(7.0)})
    database = ParameterDatabase(tmp_path)

    assert float(database.get_parameter("probe.named.value").value) == 2.5
    assert float(database.get_parameter("probe.other.value").value) == 7.0
    assert database.cache_dirty is True


def test_unparseable_file_without_header_does_not_break_the_index(tmp_path):
    _write(tmp_path / "relic" / "probe.yaml", "probe.ok", {"probe.ok.value": _parameter(4.0)})
    corrupted = tmp_path / "calibrated" / "parameters.yaml"
    corrupted.parent.mkdir(parents=True, exist_ok=True)
    corrupted.write_text("invalid: yaml: content: {[}", encoding="utf-8")

    database = ParameterDatabase(tmp_path)

    assert float(database.get_parameter("probe.ok.value").value) == 4.0
    assert database.cache_dirty is True


def test_key_defined_in_two_files_is_a_database_error(tmp_path):
    first = tmp_path / "relic" / "a" / "probe.yaml"
    second = tmp_path / "relic" / "b" / "probe.yaml"
    _write(first, "probe.dup", {"probe.dup.value": _parameter(1.0)})
    _write(second, "probe.dup", {"probe.dup.value": _parameter(2.0)})
    database = ParameterDatabase(tmp_path)

    with pytest.raises(ValueError, match="more than one file") as error:
        database.get_parameter("probe.dup.value")
    assert str(first) in str(error.value) and str(second) in str(error.value)


def test_unknown_key_resolves_to_nothing_and_the_manager_raises_keyerror():
    database = ParameterDatabase(REPO_DATABASE)
    assert database.get_parameter("validation.numerical.no_such_probe_key") is None

    manager = UnifiedParameterManager()
    with pytest.raises(KeyError):
        manager.get("validation.numerical.no_such_probe_key", required=True)


def test_load_all_rejects_a_key_defined_in_two_base_files(tmp_path):
    first = tmp_path / "relic" / "a" / "probe.yaml"
    second = tmp_path / "relic" / "b" / "probe.yaml"
    _write(first, "probe.dup", {"probe.dup.value": _parameter(1.0)})
    _write(second, "probe.dup", {"probe.dup.value": _parameter(2.0)})
    database = ParameterDatabase(tmp_path)

    with pytest.raises(ValueError, match="more than one file") as error:
        database.load_all()
    assert str(first) in str(error.value) and str(second) in str(error.value)


def test_load_namespace_reads_declared_owners_and_descendants_only(tmp_path):
    _write(tmp_path / "relic" / "x" / "one.yaml", "probe.tree", {"probe.tree.a": _parameter(1.0)})
    _write(tmp_path / "relic" / "y" / "two.yaml", "probe.tree.sub", {"probe.tree.sub.b": _parameter(2.0)})
    # A directory named like the namespace root does not make its files part of it.
    _write(tmp_path / "relic" / "probe" / "three.yaml", "other.ns", {"other.ns.c": _parameter(3.0)})
    database = ParameterDatabase(tmp_path)

    database.load_namespace("probe.tree")

    assert set(database.cache) == {"probe.tree.a", "probe.tree.sub.b"}
    assert database.cache_dirty is True


def test_load_namespace_applies_calibrated_overlays_after_base(tmp_path, monkeypatch):
    # test/conftest.py disables the calibrated tier by default; this test needs it.
    monkeypatch.delenv("RELIC_SKIP_CALIBRATED", raising=False)
    _write(tmp_path / "relic" / "p.yaml", "probe.cal", {"probe.cal.v": _parameter(1.0)})
    _write(
        tmp_path / "calibrated" / "global" / "p.yaml",
        "probe.cal",
        {"probe.cal.v": _parameter(5.0)},
        tier="calibrated",
    )
    database = ParameterDatabase(tmp_path)

    database.load_namespace("probe")

    assert float(database.cache["probe.cal.v"].value) == 5.0


def test_lazy_and_full_loads_ignore_calibrated_snapshots_alike(tmp_path, monkeypatch):
    monkeypatch.delenv("RELIC_INCLUDE_CALIBRATED_SNAPSHOTS", raising=False)
    monkeypatch.delenv("RELIC_SKIP_CALIBRATED", raising=False)
    _write(tmp_path / "relic" / "p.yaml", "probe.snap", {"probe.snap.v": _parameter(1.0)})
    _write(
        tmp_path / "calibrated" / "global" / "other.yaml",
        "probe.other",
        {"probe.other.v": _parameter(2.0)},
        tier="calibrated",
    )
    _write(
        tmp_path / "calibrated" / "snapshot.yaml",
        "probe.snap",
        {"probe.snap.v": _parameter(9.0)},
        tier="calibrated",
    )

    lazy_value = float(ParameterDatabase(tmp_path).get_parameter("probe.snap.v").value)
    full_value = float(ParameterDatabase(tmp_path).load_all()["probe.snap.v"].value)

    assert lazy_value == full_value == 1.0


def test_a_cached_key_is_never_served_before_all_its_owners_are_loaded(tmp_path):
    _write(tmp_path / "relic" / "a" / "root.yaml", "probe.root", {
        "probe.root.x": _parameter(1.0),
        "probe.root.sub.k": _parameter(2.0),
    })
    _write(tmp_path / "relic" / "b" / "sub.yaml", "probe.root.sub", {"probe.root.sub.k": _parameter(3.0)})
    database = ParameterDatabase(tmp_path)

    # Loading probe.root.x reads the file that also defines probe.root.sub.k; that
    # key has a second owner, so the duplicate is reported instead of a cached value.
    with pytest.raises(ValueError, match="more than one file"):
        database.get_parameter("probe.root.x")


def test_declared_namespace_predicate_uses_the_index(tmp_path):
    _write(tmp_path / "relic" / "zzz" / "f.yaml", "probe.decl.sub", {"probe.decl.sub.v": _parameter(1.0)})
    database = ParameterDatabase(tmp_path)

    assert database.declares_namespace("probe.decl.sub")
    assert database.declares_namespace("probe.decl")
    assert database.declares_namespace("probe.decl.sub.deeper")
    assert not database.declares_namespace("zzz")
    assert not database.declares_namespace("probe.other")


def test_is_declared_namespace_consults_the_given_manager(tmp_path):
    from relic.parameters.namespaces import is_declared_namespace

    _write(tmp_path / "relic" / "zzz" / "f.yaml", "probe.decl.sub", {"probe.decl.sub.v": _parameter(1.0)})
    manager = UnifiedParameterManager(database_path=tmp_path)

    assert is_declared_namespace("probe.decl.sub", manager)
    assert not is_declared_namespace("maintenance", manager)
    assert is_declared_namespace("validation.numerical")
    assert not is_declared_namespace("maintenance")


def test_custom_priority_load_reads_the_plan_tiers(tmp_path, monkeypatch):
    monkeypatch.delenv("RELIC_SKIP_CALIBRATED", raising=False)
    monkeypatch.delenv("RELIC_INCLUDE_CALIBRATED_SNAPSHOTS", raising=False)
    _write(tmp_path / "relic" / "p.yaml", "probe.prio", {"probe.prio.v": _parameter(1.0)})
    _write(
        tmp_path / "calibrated" / "global" / "p.yaml",
        "probe.prio",
        {"probe.prio.v": _parameter(5.0)},
        tier="calibrated",
    )
    # A top-level calibrated snapshot is not part of the plan unless enabled.
    _write(
        tmp_path / "calibrated" / "snapshot.yaml",
        "probe.prio",
        {"probe.prio.v": _parameter(9.0)},
        tier="calibrated",
    )
    database = ParameterDatabase(tmp_path)

    calibrated_first = database.load_all_with_priority(["calibrated", "relic"])
    relic_first = database.load_all_with_priority(["relic", "calibrated"])
    monkeypatch.setenv("RELIC_SKIP_CALIBRATED", "1")
    calibrated_disabled = database.load_all_with_priority(["calibrated", "relic"])

    assert float(calibrated_first["probe.prio.v"].value) == 5.0
    assert float(relic_first["probe.prio.v"].value) == 1.0
    assert float(calibrated_disabled["probe.prio.v"].value) == 1.0


def test_custom_priority_load_rejects_unknown_tiers_and_duplicate_base_keys(tmp_path):
    first = tmp_path / "relic" / "a" / "probe.yaml"
    second = tmp_path / "relic" / "b" / "probe.yaml"
    _write(first, "probe.dup", {"probe.dup.value": _parameter(1.0)})
    _write(second, "probe.dup", {"probe.dup.value": _parameter(2.0)})
    database = ParameterDatabase(tmp_path)

    with pytest.raises(ValueError, match="no_such_tier"):
        database.load_all_with_priority(["no_such_tier"])
    with pytest.raises(ValueError, match="more than one file"):
        database.load_all_with_priority(["relic"])


def test_the_declared_namespace_index_is_the_only_namespace_predicate():
    import relic.parameters.namespaces as namespaces

    assert not hasattr(namespaces, "is_registered_namespace")
    assert not hasattr(namespaces, "_discover_database_roots")
