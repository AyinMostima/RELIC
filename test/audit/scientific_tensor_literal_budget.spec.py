"""
Stage budget gate for scientific torch.tensor numeric literals.

This gate prevents regression while legacy literals are incrementally removed.
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_ROOT = REPO_ROOT / "relic" / "scientific"
TORCH_TENSOR_LITERAL = re.compile(r"torch\.tensor\(\s*[-+]?\d+(?:\.\d+)?")

# Stage budget should only move downward over time.
# Baseline captured on 2026-02-13.
MAX_ALLOWED_TORCH_TENSOR_LITERAL = 63


def _count_torch_tensor_numeric_literals() -> int:
    total = 0
    for path in SCIENTIFIC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        total += len(TORCH_TENSOR_LITERAL.findall(text))
    return total


def test_scientific_torch_tensor_literal_stage_budget() -> None:
    count = _count_torch_tensor_numeric_literals()
    assert count <= MAX_ALLOWED_TORCH_TENSOR_LITERAL, (
        "Scientific torch.tensor numeric literals exceed stage budget: "
        f"{count} > {MAX_ALLOWED_TORCH_TENSOR_LITERAL}. "
        "Replace literals with parameter-driven or tensor-shape-driven construction."
    )
