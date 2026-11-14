import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Prelude that makes executing the RELIC subpackage relic.monitoring raise.
_INJECTED_FAILURE_PRELUDE = """
import importlib.abc
import importlib.machinery
import sys


class _InjectedFailureLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise RuntimeError("injected failure while executing relic.monitoring")


class _InjectedFailureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "relic.monitoring":
            return importlib.machinery.ModuleSpec(fullname, _InjectedFailureLoader())
        return None


sys.meta_path.insert(0, _InjectedFailureFinder())
"""

# Prelude that makes the RELIC subpackage relic.monitoring unimportable.
_HALTED_SUBMODULE_PRELUDE = """
import sys

sys.modules["relic.monitoring"] = None
"""


def _run_import_probe(lazy_enabled: bool) -> dict[str, object]:
    """Probe one import mode in an isolated interpreter."""
    probe = """
import json
import sys
import types

import relic

module_name = "extensions"
qualified_name = f"relic.{module_name}"
loaded_before_access = qualified_name in sys.modules
module = getattr(relic, module_name)
print(json.dumps({
    "lazy_enabled": bool(relic._lazy_import_enabled),
    "loaded_before_access": loaded_before_access,
    "loaded_after_access": qualified_name in sys.modules,
    "is_module": isinstance(module, types.ModuleType),
}))
"""
    environment = dict(os.environ)
    environment["RELIC_LAZY_IMPORT"] = "1" if lazy_enabled else "0"
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert output_lines, completed.stderr
    return json.loads(output_lines[-1])


def _run_eager_import_with_prelude(prelude: str) -> subprocess.CompletedProcess:
    """Import RELIC in eager mode after a prelude that breaks one RELIC subpackage."""
    probe = prelude + """
import relic

print("EAGER_IMPORT_COMPLETED", repr(relic.__dict__.get("monitoring", "<absent>")))
"""
    environment = dict(os.environ)
    environment["RELIC_LAZY_IMPORT"] = "0"
    return subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _module_identity_snapshot() -> dict[str, object]:
    """Capture parent-process module identities that probes must not mutate."""
    return {
        name: sys.modules.get(name)
        for name in ("relic", "relic.core.functor", "relic.parameters")
    }


def test_lazy_import_enabled():
    """Lazy mode resolves an unloaded optional package on first access."""
    before = _module_identity_snapshot()
    result = _run_import_probe(lazy_enabled=True)

    assert result == {
        "lazy_enabled": True,
        "loaded_before_access": False,
        "loaded_after_access": True,
        "is_module": True,
    }
    assert _module_identity_snapshot() == before


def test_lazy_import_disabled():
    """Eager mode imports the optional package during root initialization."""
    before = _module_identity_snapshot()
    result = _run_import_probe(lazy_enabled=False)

    assert result == {
        "lazy_enabled": False,
        "loaded_before_access": True,
        "loaded_after_access": True,
        "is_module": True,
    }
    assert _module_identity_snapshot() == before


@pytest.mark.parametrize(
    ("prelude", "expected_error"),
    [
        (
            _INJECTED_FAILURE_PRELUDE,
            "RuntimeError: injected failure while executing relic.monitoring",
        ),
        (
            _HALTED_SUBMODULE_PRELUDE,
            "ModuleNotFoundError: import of relic.monitoring halted; None in sys.modules",
        ),
    ],
    ids=["subpackage_raises", "subpackage_unimportable"],
)
def test_eager_import_propagates_relic_subpackage_failure(prelude, expected_error):
    """Eager mode must not replace a failing RELIC subpackage by None."""
    before = _module_identity_snapshot()
    completed = _run_eager_import_with_prelude(prelude)

    assert completed.returncode != 0, completed.stdout
    assert "EAGER_IMPORT_COMPLETED" not in completed.stdout
    assert expected_error in completed.stderr
    assert _module_identity_snapshot() == before

