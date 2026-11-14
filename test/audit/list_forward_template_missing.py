"""
List forward() implementations missing the mandatory scientific comment template.

Template markers (case-insensitive, within a short window after the forward docstring):
  - References:
  - Mathematical expression:
  - Parameter meanings:

This script is non-invasive and reads source files only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MARKERS = ("references:", "mathematical expression:", "parameter meanings:")
SIX_ELEMENT_MARKERS = MARKERS + (
    "units:",
    "domain/constraints:",
    "sign convention:",
)


@dataclass(frozen=True)
class Finding:
    file_path: str
    line_no: int


def _skip_forward_docstring(lines: list[str], start_index: int) -> int:
    """Return the first line index after the optional forward() docstring."""
    j = start_index
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    if j >= len(lines):
        return j

    stripped = lines[j].lstrip()
    if not (stripped.startswith('"""') or stripped.startswith("'''")):
        return j

    quote = '"""' if stripped.startswith('"""') else "'''"
    # Single-line docstring: """text"""
    if stripped.count(quote) >= 2 and stripped.strip() != quote:
        return j + 1

    # Multi-line docstring: consume until closing quote.
    j += 1
    while j < len(lines) and quote not in lines[j]:
        j += 1
    if j < len(lines):
        j += 1
    return j


def function_template_present(
    lines: list[str],
    index: int,
    markers: tuple[str, ...] = MARKERS,
) -> bool:
    """Return whether one function has the requested comment markers."""
    signature_end = index
    parenthesis_depth = 0
    signature_started = False
    for signature_end in range(index, len(lines)):
        code = lines[signature_end].split("#", 1)[0]
        if "(" in code:
            signature_started = True
        parenthesis_depth += code.count("(") - code.count(")")
        if (
            signature_started
            and parenthesis_depth <= 0
            and code.rstrip().endswith(":")
        ):
            break

    signature_comments = lines[index + 1 : signature_end + 1]
    start = _skip_forward_docstring(lines, signature_end + 1)
    window = "\n".join(signature_comments + lines[start : start + 25]).lower()
    return all(marker in window for marker in markers)


def forward_template_present(lines: list[str], index: int) -> bool:
    """Return whether one forward method has the legacy three markers."""
    return function_template_present(lines, index, MARKERS)


def list_missing_forward_templates(root: Path) -> list[Finding]:
    missing: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, line in enumerate(lines):
            if not line.lstrip().startswith("def forward("):
                continue
            if not forward_template_present(lines, i):
                missing.append(
                    Finding(
                        file_path=str(path.as_posix()).replace("/", "\\"),
                        line_no=i + 1,
                    )
                )
    return missing


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root / "relic" / "coupling"
    missing = list_missing_forward_templates(root)
    print(f"[OK] missing_forward_template={len(missing)}")
    for item in missing:
        print(f"{item.file_path}:{item.line_no}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
