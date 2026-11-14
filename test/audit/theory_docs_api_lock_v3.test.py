"""
Audit gate for theory docs API lock and encoding hygiene.
"""

from __future__ import annotations

from pathlib import Path


_THEORY_ROOT = Path("docs/source/theory")
_FORBIDDEN_TERMS = (
    "get_parameter_by_path",
    "_resolve_parameter",
)
_MOJIBAKE_MARKERS = (
    "鏈",
    "妗ユ帴灞",
    "鍙傛暟绯荤粺",
    "鈫?",
)


def test_theory_docs_must_not_reference_removed_parameter_apis() -> None:
    """
    References:
    - docs/source/architecture_overview.md (2026-02 parameter API hard lock).

    Mathematical expression:
    - N/A (documentation contract gate).

    Parameter meanings:
    - Removed APIs are legacy access paths superseded by SSOT entry points.
    """
    violations: list[str] = []
    for doc in _THEORY_ROOT.rglob("*.md"):
        content = doc.read_text(encoding="utf-8")
        for token in _FORBIDDEN_TERMS:
            if token in content:
                violations.append(f"{doc.as_posix()} contains forbidden token: {token}")
    assert not violations, "\n".join(violations)


def test_scale_coupling_doc_is_utf8_and_has_no_mojibake_markers() -> None:
    """
    References:
    - UTF-8 documentation requirement for architecture SSOT readability.

    Mathematical expression:
    - N/A (documentation quality gate).

    Parameter meanings:
    - Mojibake markers represent decoding corruption in theory docs.
    """
    path = _THEORY_ROOT / "scale_coupling.md"
    content = path.read_text(encoding="utf-8")

    assert "\ufffd" not in content, "scale_coupling.md contains replacement characters"
    for marker in _MOJIBAKE_MARKERS:
        assert marker not in content, f"scale_coupling.md contains mojibake marker: {marker}"

