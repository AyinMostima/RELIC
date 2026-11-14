"""Regression tests for the canonical scientific static-audit report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from test.audit import scientific_static_audit as audit


def _write_fixture(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")


def test_build_report_owns_source_policy_metrics(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path / "governed.py",
        """
class GovernedFunctor(FunctorBase):
    def forward(self, state):
        # References: Example (2026).
        # Mathematical expression: y = f(x).
        # Parameter meanings: x is the input and y is the output.
        manager = get_global_manager()
        return {"result": manager}
""",
    )
    _write_fixture(
        tmp_path / "policy_gaps.py",
        """
LABEL = "caf\u00e9"

def forward(state):
    return state
""",
    )

    report = audit.build_report(tmp_path)

    assert report["files_scanned"] == 2
    assert report["metrics"]["global_manager_calls_functor"] == 1
    assert report["metrics"]["global_manager_files_functor"] == 1
    assert report["metrics"]["non_ascii_files"] == 1
    assert report["metrics"]["forward_missing_template"] == 1
    assert len(report["non_ascii_files"]) == 1
    assert report["non_ascii_files"][0].endswith("policy_gaps.py")
    assert len(report["forward_template_missing"]) == 1
    assert report["forward_template_missing"][0]["file"].endswith("policy_gaps.py")


def test_direct_path_import_is_side_effect_free(tmp_path: Path) -> None:
    script_path = Path(audit.__file__).resolve()
    probe = f"import runpy; runpy.run_path({str(script_path)!r}, run_name='static_audit_probe')"

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "analysis_results").exists()
