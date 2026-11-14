"""
L1 Validation: Coupling modules should not initialize ParameterManager during import.

This test guards against import-time side effects that would create global
parameter manager instances, ensuring lazy resolution and easier test isolation.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


TARGET_MODULES = [
    "relic.coupling.capability_registry",
    "relic.coupling.internal",
    "relic.coupling.core.audit",
    "relic.coupling.scales.scale_manager",
]


def _run_import_probe(module_name: str) -> dict[str, object]:
    """Reload one target in an isolated interpreter with a guarded manager."""
    probe = r'''
import importlib
import json
import sys

import relic.parameters as parameters
import relic.parameters.manager as parameter_manager_module

module_name = sys.argv[1]
calls = []

def guarded_get_global_manager(*args, **kwargs):
    calls.append(module_name)
    raise AssertionError("get_global_manager should not be called during import")

parameters.get_global_manager = guarded_get_global_manager
parameter_manager_module.get_global_manager = guarded_get_global_manager
for loaded_name in list(sys.modules):
    if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
        sys.modules.pop(loaded_name, None)
importlib.invalidate_caches()
importlib.import_module(module_name)
print(json.dumps({"calls": calls, "module_loaded": module_name in sys.modules}))
'''
    environment = dict(os.environ)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe, module_name],
        cwd=REPO_ROOT,
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


@pytest.mark.parametrize("module_name", TARGET_MODULES)
def test_import_does_not_instantiate_parameter_manager(module_name: str):
    """Importing coupling modules must not call get_global_manager."""
    result = _run_import_probe(module_name)
    assert result == {"calls": [], "module_loaded": True}


