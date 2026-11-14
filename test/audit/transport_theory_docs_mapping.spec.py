"""Completeness checks for transport theory documentation mapping."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THEORY_DIR = PROJECT_ROOT / "docs" / "source" / "theory"


def test_transport_theory_documents_exist() -> None:
    required_files = [
        "transport.md",
        "transport_linear_advection.md",
        "transport_fickian_diffusion.md",
        "transport_advection_diffusion.md",
    ]
    for name in required_files:
        assert (THEORY_DIR / name).exists(), f"Missing theory document: {name}"


def test_transport_module_theory_contains_stability_and_splitting_constraints() -> None:
    text = (THEORY_DIR / "transport.md").read_text(encoding="utf-8")
    required_markers = [
        "dc/dt + div(v * c) = S",
        "dc/dt = div(D * grad(c)) + S",
        "CFL = |u| * dt / dx <= 1",
        "r = D * dt / dx^2 <= 0.5",
        "Sequential splitting",
        "Symmetric (Strang) splitting",
    ]
    for marker in required_markers:
        assert marker in text, f"Missing marker in transport.md: {marker}"
