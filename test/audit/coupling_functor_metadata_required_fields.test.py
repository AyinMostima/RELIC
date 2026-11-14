"""Audit gate for required coupling FunctorMetadata fields."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

TARGETS: dict[str, set[str]] = {
    "relic/coupling/external/common/base_adapter.py": {"ExternalAdapter"},
    "relic/coupling/scales/adapters.py": {
        "SpatialAggregation",
        "ConservativeRemap",
    },
    "relic/coupling/scales/hierarchical/aggregator.py": {"HierarchicalAggregator"},
    "relic/coupling/scales/spatial/factory.py": {"AutoRegridder"},
    "relic/coupling/scales/spatial/regrid_conservative.py": {"ConservativeRegridder"},
    "relic/coupling/scales/spatial/regrid_bilinear.py": {"BilinearRegridder"},
    "relic/coupling/scales/spatial/regrid_nearest.py": {"NearestNeighborRegridder"},
    "relic/coupling/scales/temporal/aggregator.py": {"TemporalAggregator"},
    "relic/coupling/scales/temporal/disaggregator.py": {"TemporalDisaggregator"},
    "relic/coupling/scales/temporal/temporal_scaling.py": {"TemporalScaling"},
}

REQUIRED_FIELDS = {"spatial_scale", "variable_kind", "input_variables", "output_variables"}

# Declared-empty metadata lists that are the true declaration of the class, with the reason.
EMPTY_LISTS_DECLARED: dict[tuple[str, str], str] = {
    ("ExternalAdapter", "input_variables"): (
        "an external adapter is a source Functor: forward reads neither the state nor dt; its "
        "outputs are the provided catalog variables (ExternalAdapter.declared_outputs)"
    ),
}


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _class_metadata_keywords(class_node: ast.ClassDef) -> set[str]:
    fields: set[str] = set()
    for fn in [n for n in class_node.body if isinstance(n, ast.FunctionDef)]:
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            if _base_name(call.func) != "FunctorMetadata":
                continue
            fields.update({kw.arg for kw in call.keywords if kw.arg})
    return fields


def _class_has_empty_io_literal(class_node: ast.ClassDef) -> bool:
    for fn in [n for n in class_node.body if isinstance(n, ast.FunctionDef)]:
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            if _base_name(call.func) != "FunctorMetadata":
                continue
            for kw in call.keywords:
                if kw.arg not in {"input_variables", "output_variables"}:
                    continue
                if (class_node.name, kw.arg) in EMPTY_LISTS_DECLARED:
                    continue
                if isinstance(kw.value, ast.List) and len(kw.value.elts) == 0:
                    return True
    return False


def test_coupling_functor_metadata_required_fields() -> None:
    violations: list[str] = []

    for relative_path, class_names in TARGETS.items():
        path = REPO_ROOT / relative_path
        source = path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
        tree = ast.parse(source)
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

        for class_name in class_names:
            if class_name not in classes:
                violations.append(f"{relative_path}::{class_name} not found")
                continue
            cls = classes[class_name]
            fields = _class_metadata_keywords(cls)
            missing = sorted(REQUIRED_FIELDS - fields)
            if missing:
                violations.append(
                    f"{relative_path}::{class_name} missing metadata fields: {', '.join(missing)}"
                )
            if _class_has_empty_io_literal(cls):
                violations.append(
                    f"{relative_path}::{class_name} uses empty input/output metadata lists"
                )

    assert not violations, "Coupling metadata violations found:\n" + "\n".join(sorted(violations))
