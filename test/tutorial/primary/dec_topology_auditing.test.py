"""Contract tests for the NB14 topology-auditing tutorial."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = PROJECT_ROOT / "tutorial" / "primary" / "14_dec_topology_auditing.ipynb"


def _notebook_sources() -> list[str]:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def test_nb14_materializes_incidence_tensor_before_first_use() -> None:
    incidence_cell = next(
        source for source in _notebook_sources() if "build_incidence_matrix(graph)" in source
    )
    tree = ast.parse(incidence_cell)
    stores = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "B_t"
        and isinstance(node.ctx, ast.Store)
    ]
    loads = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == "B_t"
        and isinstance(node.ctx, ast.Load)
    ]

    assert stores
    assert loads
    assert min(stores) < min(loads)
    assert '"B_t" in dir()' not in incidence_cell


def test_nb14_reuses_one_incidence_tensor_for_budget_audit() -> None:
    all_source = "\n".join(_notebook_sources())

    assert all_source.count("B_t = torch.as_tensor(B, dtype=torch.float32)") == 1
    assert "B_tensor =" not in all_source
    assert "du = B_t @ f" in all_source
    assert "du_bad = B_t @ f_bad" in all_source


def test_nb14_does_not_claim_unexecuted_timestep_control() -> None:
    all_source = "\n".join(_notebook_sources())

    assert "This triggers adaptive timestep reduction in RELIC" not in all_source
    assert "no timestep controller is executed in this notebook" in all_source
    assert "# Units:" in all_source
    assert "# Sign convention:" in all_source
    assert "# Open-system route:" in all_source
