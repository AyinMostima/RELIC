"""Quality checks for transport boundary_conditions module header encoding/docstring."""

from __future__ import annotations

from pathlib import Path


def test_boundary_conditions_header_is_standard_docstring_and_ascii_clean() -> None:
    path = Path("relic/scientific/transport/boundary_conditions.py")
    text = path.read_text(encoding="utf-8")

    assert text.startswith('"""')
    assert "Oez???k" not in text
    assert "\\nBoundary Conditions Module" not in text
