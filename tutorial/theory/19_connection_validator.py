r"""Tutorial 19 -- Connection Validator and Adapter Requirement.

Demonstrates
------------
This tutorial covers Theorem 3.6.2 in docs/source/coertheory.md:
connection validation as a predicate

    V(p_s, p_t; strict) = (ok, A, W)

with:
  p = (tau, u, T, s, K)
  tau: datatype
  u: units
  T: optional tags (quantity, element, cf standard name)
  s: semantics class
  K: context requirements

Two figures are produced:
  1) 19_math_connection_validator.png
     Pure mathematical condition matrix and predicate summary.
  2) 19_relic_connection_validator.png
     Empirical RELIC validator output from can_connect(...).

References
----------
1. docs/source/coertheory.md, Theorem 3.6.2
2. relic/orchestration/validator.py, can_connect
3. relic/orchestration/port_spec.py, PortSpec
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE,
    DOUBLE_COL_WIDTH,
    add_panel_label,
    add_text_box,
    apply_nature_style,
    savefig,
)

from relic.data.standards.unit_converter import UnitConverter
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.validator import can_connect


@dataclass
class Scenario:
    """One source-target connection scenario for theorem validation."""

    name: str
    source: PortSpec
    target: PortSpec
    context: Optional[Dict[str, Any]]


def _semantic_compatibility(source_sem: str, target_sem: str) -> bool:
    """Check semantic compatibility condition C4 in theorem 3.6.2."""
    # References:
    # - docs/source/coertheory.md (Theorem 3.6.2, semantics compatibility)
    # - relic/orchestration/validator.py::_validate_semantics
    # Math expression:
    #   C4 = 1 if semantic mapping is valid; else 0.
    # Parameter meaning:
    #   source_sem: source port semantics class.
    #   target_sem: target port semantics class.
    if not source_sem or not target_sem:
        return True
    if target_sem in {"state", "diagnostic"}:
        return source_sem in {"state", "diagnostic", "flux"}
    if target_sem == "increment":
        return source_sem in {"increment", "flux"}
    if target_sem == "flux":
        return source_sem == "flux"
    if target_sem == "parameter":
        return source_sem == "parameter"
    return True


def _optional_tag_match(source: PortSpec, target: PortSpec) -> bool:
    """Check theorem condition C2 for optional tags."""
    # References:
    # - docs/source/coertheory.md (Theorem 3.6.2, optional tag consistency)
    # Math expression:
    #   C2 = prod_{tag in T} I(tag_s == tag_t when both declared)
    # Parameter meaning:
    #   source, target: source and target ports.
    pairs = [
        (source.quantity, target.quantity),
        (source.element, target.element),
        (
            (source.cf_standard_name or source.standard_name),
            (target.cf_standard_name or target.standard_name),
        ),
    ]
    for src_tag, tgt_tag in pairs:
        if src_tag and tgt_tag and str(src_tag).strip().lower() != str(tgt_tag).strip().lower():
            return False
    return True


def _unit_closed(source: PortSpec, target: PortSpec, unit_converter: UnitConverter) -> bool:
    """Check theorem condition C3 for unit closure."""
    # References:
    # - docs/source/coertheory.md (Theorem 3.6.2, unit closure)
    # - relic/orchestration/validator.py::_validate_units_compatibility
    # Math expression:
    #   C3 = I(u_s and u_t exist) * I(u_s == u_t or Convertible(u_s, u_t))
    # Parameter meaning:
    #   source.units, target.units: unit declarations.
    if not source.units or not target.units:
        return False
    if source.units == target.units:
        return True
    return bool(
        unit_converter.can_convert(source.units, target.units, source.standard_name)
        or unit_converter.can_convert(source.units, target.units, "generic")
    )


def _context_satisfied(target: PortSpec, context: Optional[Dict[str, Any]]) -> bool:
    """Check theorem condition C5 for context requirements."""
    # References:
    # - docs/source/coertheory.md (Theorem 3.6.2, context requirement)
    # - relic/orchestration/validator.py::_validate_context_requirements
    # Math expression:
    #   C5 = I(K_target subseteq K_provided)
    # Parameter meaning:
    #   target.context_requirements: required keys.
    #   context: provided context mapping.
    if not target.context_requirements:
        return True
    context = context or {}
    for key in target.context_requirements.keys():
        if context.get(key) is None:
            return False
    return True


def _theorem_conditions(
    source: PortSpec,
    target: PortSpec,
    context: Optional[Dict[str, Any]],
    unit_converter: UnitConverter,
) -> Dict[str, bool]:
    """Evaluate theorem 3.6.2 conditions C1..C5."""
    c1 = (not source.datatype or not target.datatype) or (source.datatype == target.datatype)
    c2 = _optional_tag_match(source, target)
    c3 = _unit_closed(source, target, unit_converter)
    c4 = _semantic_compatibility(
        (source.semantics or "").strip().lower(),
        (target.semantics or "").strip().lower(),
    )
    c5 = _context_satisfied(target, context)
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "OK": all([c1, c2, c3, c4, c5])}


def _make_scenarios() -> List[Scenario]:
    """Construct diverse theorem 3.6.2 scenarios."""
    common = dict(standard_name="water_depth", quantity="length", datatype="tensor")
    scenarios: List[Scenario] = [
        Scenario(
            name="identity",
            source=PortSpec(name="src_identity", units="m", semantics="state", **common),
            target=PortSpec(name="tgt_identity", units="m", semantics="state", **common),
            context={},
        ),
        Scenario(
            name="unit_adapter",
            source=PortSpec(name="src_unit", units="m", semantics="state", **common),
            target=PortSpec(name="tgt_unit", units="cm", semantics="state", **common),
            context={},
        ),
        Scenario(
            name="semantic_adapter",
            source=PortSpec(name="src_sem", units="m", semantics="flux", **common),
            target=PortSpec(name="tgt_sem", units="m", semantics="state", **common),
            context={},
        ),
        Scenario(
            name="unit_plus_semantic",
            source=PortSpec(name="src_both", units="m", semantics="flux", **common),
            target=PortSpec(name="tgt_both", units="cm", semantics="state", **common),
            context={},
        ),
        Scenario(
            name="datatype_mismatch",
            source=PortSpec(
                name="src_dtype",
                standard_name="air_temperature",
                units="K",
                quantity="temperature",
                semantics="state",
                datatype="scalar",
            ),
            target=PortSpec(
                name="tgt_dtype",
                standard_name="air_temperature",
                units="K",
                quantity="temperature",
                semantics="state",
                datatype="tensor",
            ),
            context={},
        ),
        Scenario(
            name="missing_context",
            source=PortSpec(name="src_ctx", units="m", semantics="state", **common),
            target=PortSpec(
                name="tgt_ctx",
                units="m",
                semantics="state",
                context_requirements={"air_density": "kg m-3"},
                **common,
            ),
            context={},
        ),
    ]
    return scenarios


def _classify_decision(decision: Any) -> str:
    if not decision.ok:
        return "reject"
    if decision.requires_adapter:
        return "adapter"
    return "identity"


def main() -> None:
    print("=== Tutorial 19: Connection validator theorem and RELIC evidence ===")
    unit_converter = UnitConverter()
    scenarios = _make_scenarios()

    theorem_rows: List[Dict[str, bool]] = []
    decisions: List[Any] = []
    for scenario in scenarios:
        theorem_result = _theorem_conditions(
            scenario.source,
            scenario.target,
            scenario.context,
            unit_converter,
        )
        decision = can_connect(
            scenario.source,
            scenario.target,
            unit_converter,
            context=scenario.context,
            strict=True,
        )
        theorem_rows.append(theorem_result)
        decisions.append(decision)
        print(
            f"  {scenario.name:18s} | "
            f"theorem_ok={theorem_result['OK']} | "
            f"validator_ok={decision.ok} | "
            f"adapter={decision.adapter_type} | "
            f"errors={[e.code for e in decision.errors]}"
        )

    # ------------------------------------------------------------------
    # Figure 1: pure mathematics
    # ------------------------------------------------------------------
    apply_nature_style()
    fig1, axes1 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    cond_order = ["C1", "C2", "C3", "C4", "C5", "OK"]
    heat = np.array([[1.0 if row[c] else 0.0 for c in cond_order] for row in theorem_rows], dtype=float)
    ax = axes1[0]
    im = ax.imshow(heat, vmin=0.0, vmax=1.0, cmap="YlGn")
    ax.set_xticks(np.arange(len(cond_order)))
    ax.set_xticklabels(cond_order, rotation=0)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([s.name for s in scenarios], fontsize=7)
    ax.set_title("Condition Matrix for V(ps, pt)", fontsize=9)
    ax.set_xlabel("Theorem 3.6.2 Conditions")
    ax.set_ylabel("Scenarios")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, "1" if heat[i, j] > 0.5 else "0", ha="center", va="center", fontsize=7)
    add_panel_label(ax, "a")
    cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Condition truth value", fontsize=7)

    # Panel b: stacked bar chart showing condition pass count per scenario
    ax = axes1[1]
    scenario_names = [s.name for s in scenarios]
    cond_keys = ["C1", "C2", "C3", "C4", "C5"]
    cond_labels = ["C1:dtype", "C2:tags", "C3:units", "C4:sem", "C5:ctx"]
    y_pos = np.arange(len(scenarios))
    left = np.zeros(len(scenarios))
    cond_colors = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[3], CB_PALETTE[4], CB_PALETTE[5]]
    for j, key in enumerate(cond_keys):
        vals = np.array([1 if row[key] else 0 for row in theorem_rows], dtype=float)
        ax.barh(y_pos, vals, left=left, height=0.6, color=cond_colors[j],
                label=cond_labels[j], alpha=0.85)
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenario_names, fontsize=7)
    ax.set_xlabel("Conditions satisfied")
    ax.set_title("Per-Scenario Condition Pass Count", fontsize=9)
    ax.set_xlim(0, 5.5)
    ax.legend(fontsize=5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, prop={"weight": "bold"})
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig1.suptitle("Pure Mathematics: Connection Predicate and Conditions", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig1, "19_math_connection_validator")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: RELIC empirical validator outputs
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    cls_map = {"identity": 0, "adapter": 1, "reject": 2}
    cls_names = ["identity", "adapter", "reject"]
    cls_values = [_classify_decision(decision) for decision in decisions]
    bar_colors = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[1]]

    ax = axes2[0]
    y = np.arange(len(scenarios))
    x = [cls_map[name] for name in cls_values]
    ax.barh(y, x, color=[bar_colors[v] for v in x], alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels([s.name for s in scenarios], fontsize=7)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(cls_names, fontsize=7)
    ax.set_xlabel("Validator classification")
    ax.set_title("RELIC can_connect Classification", fontsize=9)
    add_panel_label(ax, "a")

    # Panel b: heatmap of decision properties per scenario
    ax = axes2[1]
    dec_keys = ["ok", "needs_A", "has_adapter", "has_warnings", "has_errors"]
    dec_labels = ["ok", "needs\nadapter", "adapter\ntype", "warnings", "errors"]
    dec_data = []
    for decision in decisions:
        dec_data.append([
            1 if decision.ok else 0,
            1 if decision.requires_adapter else 0,
            1 if decision.adapter_type else 0,
            1 if decision.warnings else 0,
            1 if decision.errors else 0,
        ])
    dec_arr = np.array(dec_data, dtype=float)
    # Use diverging: green=1(pass), red=0(fail) for ok; reversed for errors
    im2 = ax.imshow(dec_arr, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(dec_keys)))
    ax.set_xticklabels(dec_labels, fontsize=6, ha="center")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels([s.name for s in scenarios], fontsize=7)
    ax.set_title("Decision Property Matrix", fontsize=9)
    for i in range(dec_arr.shape[0]):
        for j in range(dec_arr.shape[1]):
            val_str = "Y" if dec_arr[i, j] > 0.5 else "N"
            ax.text(j, i, val_str, ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white" if dec_arr[i, j] > 0.5 else "#333")
    cbar2 = fig2.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Decision status", fontsize=7)
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig2.suptitle("RELIC Empirical Evidence: can_connect(ps, pt)", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig2, "19_relic_connection_validator")
    plt.close(fig2)

    print("[OK] Tutorial 19 finished.")


if __name__ == "__main__":
    main()

