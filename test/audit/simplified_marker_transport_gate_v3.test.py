"""
Audit gate for placeholder markers in hydrosphere transport production modules.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSPORT_ROOT = REPO_ROOT / "relic" / "scientific" / "hydrosphere" / "transport"
MARKER_PATTERN = re.compile(r"\b(simplified|fallback|placeholder|todo)\b", re.IGNORECASE)


def test_hydrosphere_transport_has_no_placeholder_markers_v3() -> None:
    """
    References:
    - RELIC architecture policy on production scientific code quality gates.

    Mathematical expression:
    - N/A (static text audit).

    Parameter meanings:
    - marker: placeholder token indicating incomplete or downgraded logic.
    """
    findings: list[str] = []
    for path in TRANSPORT_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if MARKER_PATTERN.search(line):
                findings.append(f"{path.relative_to(REPO_ROOT)}:{lineno}:{line.strip()}")

    assert not findings, (
        "Found placeholder/simplified markers in hydrosphere transport modules. "
        f"Examples: {findings[:20]}"
    )
