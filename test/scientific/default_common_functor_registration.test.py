"""Functor name registry: one import-time bootstrap, fail-fast lookups, single binding per name."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = PROJECT_ROOT / "relic" / "scientific"
BUCKET_PATH = "relic.scientific.hydrosphere.soil_moisture.bucket:BucketSoilMoisture"
AUDIT_CLI_MODULE = "relic.scientific.isotopes.audit_cli"


def test_registry_is_complete_at_import_in_a_fresh_process():
    """Name resolution does not depend on any earlier scan in the process."""
    probe = (
        "import relic.scientific as s\n"
        "print(s.get_functor_path('bucket_soil_moisture'))\n"
        "print(s.get_functor_path('BucketSoilMoisture'))\n"
    )
    env = {key: value for key, value in os.environ.items()}
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines[-2:] == [BUCKET_PATH, BUCKET_PATH]


def test_lookup_miss_raises_keyerror_naming_the_functor():
    scientific = importlib.import_module("relic.scientific")
    before = dict(scientific.FUNCTOR_REGISTRY)

    with pytest.raises(KeyError, match="no_such_functor_probe"):
        scientific.get_functor_path("no_such_functor_probe")

    assert scientific.FUNCTOR_REGISTRY == before


def test_rebinding_a_name_to_another_class_is_rejected():
    scientific = importlib.import_module("relic.scientific")
    before = dict(scientific.FUNCTOR_REGISTRY)

    class ConflictingBucketName:
        FUNCTOR_NAME = "bucket_soil_moisture"

    try:
        with pytest.raises(ValueError, match="bucket_soil_moisture"):
            scientific.register_functor(ConflictingBucketName)
        assert scientific.FUNCTOR_REGISTRY == before
    finally:
        scientific.FUNCTOR_REGISTRY.clear()
        scientific.FUNCTOR_REGISTRY.update(before)


def test_common_exports_bind_only_functor_classes():
    """Every class exported by relic.scientific is a FunctorBase subclass: no
    helper is exported under a Functor alias such as GuentherAlgorithmFunctor."""
    import inspect

    from relic.core.functor import FunctorBase

    scientific = importlib.import_module("relic.scientific")
    assert "GuentherAlgorithmFunctor" not in scientific.__all__
    assert not hasattr(scientific, "GuentherAlgorithmFunctor")
    for name in scientific.__all__:
        value = getattr(scientific, name)
        if inspect.isclass(value):
            assert issubclass(value, FunctorBase), name


def test_registering_the_same_class_again_is_idempotent():
    scientific = importlib.import_module("relic.scientific")
    before = dict(scientific.FUNCTOR_REGISTRY)
    bucket_module = importlib.import_module("relic.scientific.hydrosphere.soil_moisture.bucket")

    scientific.register_functor(bucket_module.BucketSoilMoisture)
    scientific.register_all_functors()

    assert scientific.FUNCTOR_REGISTRY == before


def test_single_writer_defaults_and_exports():
    scientific = importlib.import_module("relic.scientific")

    assert scientific.DEFAULT_ET_FUNCTOR.__name__ == "PenmanMonteithET"
    assert scientific.DEFAULT_SOIL_MOISTURE_FUNCTOR.__name__ == "RichardsSoilMoisture"
    exported_symbols = set(getattr(scientific, "__all__", []))
    assert "PriestleyTaylorET" not in exported_symbols
    assert "BucketSoilMoisture" not in exported_symbols


def _fresh_python(arguments: list[str]) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items()}
    env["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _cli_entry_modules_by_file_name() -> list[str]:
    """Module names of the files that follow the CLI naming rule ``*_cli.py``."""
    return sorted(
        ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        for path in SCIENTIFIC_ROOT.rglob("*_cli.py")
    )


def test_cli_entry_module_runs_without_a_runpy_double_import_warning():
    """`python -m` finds no earlier copy of a CLI module: the registry walk never imports it."""
    result = _fresh_python(["-W", "always::RuntimeWarning", "-m", AUDIT_CLI_MODULE, "--help"])

    assert result.returncode == 0, result.stderr[-2000:]
    assert "RuntimeWarning" not in result.stderr, result.stderr[-2000:]


def test_registry_walk_skips_exactly_the_cli_entry_modules_and_loses_no_functor():
    """Skipped modules are the *_cli.py files, and scanning them adds no registry name."""
    probe = (
        "import importlib, inspect, json, pkgutil, sys\n"
        "import relic.scientific as s\n"
        "from relic.core.functor import FunctorBase\n"
        "before = dict(s.FUNCTOR_REGISTRY)\n"
        "skipped = sorted(\n"
        "    name for _, name, _ in pkgutil.walk_packages(s.__path__, s.__name__ + '.')\n"
        "    if name not in sys.modules\n"
        ")\n"
        "for name in skipped:\n"
        "    for value in vars(importlib.import_module(name)).values():\n"
        "        if inspect.isclass(value) and issubclass(value, FunctorBase) and value is not FunctorBase:\n"
        "            s.register_functor(value)\n"
        "print(json.dumps({'skipped': skipped, 'names_before': len(before),\n"
        "                  'names_after': len(s.FUNCTOR_REGISTRY),\n"
        "                  'unchanged': before == s.FUNCTOR_REGISTRY}))\n"
    )
    result = _fresh_python(["-c", probe])

    assert result.returncode == 0, result.stderr[-2000:]
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["skipped"] == _cli_entry_modules_by_file_name()
    assert AUDIT_CLI_MODULE in observed["skipped"]
    assert observed["unchanged"], observed
    assert observed["names_after"] == observed["names_before"]
