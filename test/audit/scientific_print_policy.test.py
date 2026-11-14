"""Audit gate: console output of scientific modules stays in explicit CLI entry points.

print() is forbidden outside CLI modules, and importing the scientific package
(which imports every scientific module to fill FUNCTOR_REGISTRY) must not
configure the process root logger.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from relic.scientific import is_cli_entry_module


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = ROOT / "relic" / "scientific"

# Keep print() only in explicit command-line entry scripts.
ALLOWED_PRINT_FILES = {
    SCIENTIFIC_ROOT / "isotopes" / "audit_cli.py",
}


def _is_print_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "print"


def _is_cli_file(path: Path) -> bool:
    """The CLI naming rule shared with the registry walk (relic.scientific.is_cli_entry_module)."""
    if path in ALLOWED_PRINT_FILES:
        return True
    return is_cli_entry_module(path.stem)


def test_scientific_modules_do_not_use_print_outside_cli() -> None:
    offenders: list[str] = []
    for path in sorted(SCIENTIFIC_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig", errors="ignore")
        tree = ast.parse(source)
        if _is_cli_file(path):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_print_call(node):
                rel = path.relative_to(ROOT)
                offenders.append(f"{rel}:{getattr(node, 'lineno', -1)}")

    assert not offenders, (
        "print() is only allowed in explicit CLI modules under relic/scientific.\n"
        "Use logging APIs instead in library/runtime modules.\n"
        + "\n".join(offenders)
    )


def test_importing_scientific_package_leaves_root_logging_unconfigured() -> None:
    """Logging configuration belongs to the application or a CLI main(), never to import."""
    probe = "\n".join(
        (
            "import json, logging",
            "root = logging.getLogger()",
            "before = [root.level, len(root.handlers)]",
            "import relic.scientific",
            "after = [root.level, len(root.handlers)]",
            "print(json.dumps({'before': before, 'after': after}))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["after"] == observed["before"], (
        "Importing relic.scientific changed the root logger [level, handler count] from "
        f"{observed['before']} to {observed['after']}; a scientific module configures "
        "logging at import time.\nstderr:\n" + result.stderr
    )
