from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from relic.extensions import discovery
from relic.extensions.protocol import MANIFEST_FILENAME, ManifestError


def _write_manifest(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "version": "0.1.0",
        "author": "tester",
        "description": "demo",
        "min_relic_version": "1.0.0",
        "functors": [
            {
                "class": f"{name}.functor.DemoFunctor",
                "bridge": "TransportMechanismFunctor",
                "category": "transport_mechanism",
                "execution_group": "fast",
            }
        ],
        "parameters": [{"path": "parameters.yaml"}],
        "dependencies": {"required": [], "optional": []},
    }
    (root / MANIFEST_FILENAME).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    (root / "parameters.yaml").write_text("schema_version: 2.0.0\nparameters: {}\n", encoding="utf-8")


def test_discover_from_extra_dirs(tmp_path: Path) -> None:
    ext_root = tmp_path / "ext_alpha"
    _write_manifest(ext_root, "ext_alpha")

    infos = discovery.discover_extensions(
        extra_dirs=[tmp_path],
        use_entry_points=False,
        use_env_dirs=False,
    )

    assert len(infos) == 1
    assert infos[0].name == "ext_alpha"
    assert infos[0].source == "directory"


def test_discover_from_env_dirs(tmp_path: Path, monkeypatch) -> None:
    ext_root = tmp_path / "ext_beta"
    _write_manifest(ext_root, "ext_beta")

    monkeypatch.setenv("RELIC_EXTENSION_DIRS", str(tmp_path))
    infos = discovery.discover_extensions(
        extra_dirs=[],
        use_entry_points=False,
        use_env_dirs=True,
    )

    assert len(infos) == 1
    assert infos[0].name == "ext_beta"


def test_discover_empty_directory_no_crash(tmp_path: Path) -> None:
    infos = discovery.discover_extensions(
        extra_dirs=[tmp_path],
        use_entry_points=False,
        use_env_dirs=False,
    )
    assert infos == []


def test_invalid_manifest_raises_naming_the_manifest(tmp_path: Path) -> None:
    bad_ext = tmp_path / "broken_ext"
    bad_ext.mkdir(parents=True, exist_ok=True)
    (bad_ext / MANIFEST_FILENAME).write_text("not: [valid", encoding="utf-8")

    with pytest.raises(ManifestError, match="Invalid YAML") as excinfo:
        discovery.discover_extensions(
            extra_dirs=[tmp_path],
            use_entry_points=False,
            use_env_dirs=False,
        )

    assert str((bad_ext / MANIFEST_FILENAME).resolve()) in str(excinfo.value)


def test_deduplicate_by_name_with_priority(tmp_path: Path, monkeypatch) -> None:
    extra_dir = tmp_path / "extra"
    env_dir = tmp_path / "env"
    _write_manifest(extra_dir / "same_ext", "same_ext")
    _write_manifest(env_dir / "same_ext", "same_ext")

    monkeypatch.setenv("RELIC_EXTENSION_DIRS", str(env_dir))
    infos = discovery.discover_extensions(
        extra_dirs=[extra_dir],
        use_entry_points=False,
        use_env_dirs=True,
    )

    assert len(infos) == 1
    assert infos[0].name == "same_ext"
    assert infos[0].root_path == (extra_dir / "same_ext").resolve()


def test_discover_from_entry_points(tmp_path: Path, monkeypatch) -> None:
    ext_root = tmp_path / "entry_ext"
    _write_manifest(ext_root, "entry_ext")

    class _FakeEntryPoint:
        name = "entry_ext"

        def load(self):
            return str(ext_root)

    _patch_entry_points(monkeypatch, [_FakeEntryPoint()])

    infos = discovery.discover_extensions(
        extra_dirs=[],
        use_entry_points=True,
        use_env_dirs=False,
    )

    assert len(infos) == 1
    assert infos[0].name == "entry_ext"
    assert infos[0].source == "entry_point"


def _patch_entry_points(monkeypatch, entry_points: list) -> None:
    """Serve ``entry_points(group=...)`` for the RELIC extension group only."""

    def _fake_entry_points(*, group: str) -> list:
        return list(entry_points) if group == "relic.extensions" else []

    monkeypatch.setattr(discovery.importlib_metadata, "entry_points", _fake_entry_points)


def test_entry_point_metadata_query_failure_propagates(monkeypatch) -> None:
    """A failed metadata query must not read as "no entry-point extensions"."""

    def _broken_entry_points(**_kwargs):
        raise OSError("unreadable distribution metadata")

    monkeypatch.setattr(discovery.importlib_metadata, "entry_points", _broken_entry_points)

    with pytest.raises(OSError, match="unreadable distribution metadata"):
        discovery.discover_extensions(
            extra_dirs=[],
            use_entry_points=True,
            use_env_dirs=False,
        )


def test_entry_point_load_failure_propagates(monkeypatch) -> None:
    """A declared entry point whose target cannot be imported is not skipped."""

    class _BrokenEntryPoint:
        name = "broken_ext"

        def load(self):
            raise ImportError("extension dependency 'missing_dep' is not installed")

    _patch_entry_points(monkeypatch, [_BrokenEntryPoint()])

    with pytest.raises(ImportError, match="missing_dep"):
        discovery.discover_extensions(
            extra_dirs=[],
            use_entry_points=True,
            use_env_dirs=False,
        )


def test_entry_point_owner_module_import_failure_propagates(monkeypatch) -> None:
    """The owner module of a loaded entry-point object must import or raise."""

    class _DetachedTarget:
        pass

    _DetachedTarget.__module__ = "relic_test_missing_extension_owner_module"

    class _DetachedEntryPoint:
        name = "detached_ext"

        def load(self):
            return _DetachedTarget()

    _patch_entry_points(monkeypatch, [_DetachedEntryPoint()])

    with pytest.raises(ModuleNotFoundError, match="relic_test_missing_extension_owner_module"):
        discovery.discover_extensions(
            extra_dirs=[],
            use_entry_points=True,
            use_env_dirs=False,
        )



def test_entry_point_with_invalid_manifest_raises_naming_the_entry_point(tmp_path: Path, monkeypatch) -> None:
    bad_root = tmp_path / "bad_entry_ext"
    bad_root.mkdir(parents=True, exist_ok=True)
    (bad_root / MANIFEST_FILENAME).write_text("name: [unterminated", encoding="utf-8")

    class _EntryPoint:
        name = "bad_entry_ext"

        def load(self):
            return str(bad_root)

    _patch_entry_points(monkeypatch, [_EntryPoint()])

    with pytest.raises(ManifestError, match="bad_entry_ext") as excinfo:
        discovery.discover_extensions(extra_dirs=[], use_entry_points=True, use_env_dirs=False)

    assert "Invalid YAML" in str(excinfo.value)


def test_entry_point_package_without_path_entries_raises(monkeypatch) -> None:
    class _EmptyNamespacePackage:
        __path__: list[str] = []

    class _EntryPoint:
        name = "empty_namespace_ext"

        def load(self):
            return _EmptyNamespacePackage()

    _patch_entry_points(monkeypatch, [_EntryPoint()])

    with pytest.raises(ValueError, match="empty_namespace_ext.*empty __path__"):
        discovery.discover_extensions(extra_dirs=[], use_entry_points=True, use_env_dirs=False)


def test_entry_point_target_without_locatable_root_raises(monkeypatch) -> None:
    class _EntryPoint:
        name = "integer_target_ext"

        def load(self):
            return 42

    _patch_entry_points(monkeypatch, [_EntryPoint()])

    with pytest.raises(ValueError, match="integer_target_ext.*extension root"):
        discovery.discover_extensions(extra_dirs=[], use_entry_points=True, use_env_dirs=False)


def test_build_extension_info_requires_the_discovery_source(tmp_path: Path) -> None:
    ext_root = tmp_path / "ext_source"
    _write_manifest(ext_root, "ext_source")

    with pytest.raises(TypeError, match="source"):
        discovery.build_extension_info(ext_root)
    assert discovery.build_extension_info(ext_root, source="entry_point").source == "entry_point"
