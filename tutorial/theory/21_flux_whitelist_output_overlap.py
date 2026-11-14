r"""Tutorial 21 -- Flux Whitelist and Output Overlap Governance.

Demonstrates
------------
1) Flux whitelist governance (spatiotemporal_scale_topology.md, Section 8.3)
2) Output overlap governance (Section 8.4)
   - aggregated
   - mutually_exclusive
   (REPLACE conflicts are rejected under every policy)

This tutorial uses real runtime helpers in:
  - relic/core/simulator/execution_group.py
  - relic/core/simulator/output_overlap_governance.py

Two figures are produced:
  1) 21_math_governance_policies.png
     Pure policy algebra and policy-set distribution.
  2) 21_relic_governance_evidence.png
     Empirical execution outcomes and whitelist warnings.

References
----------
1. docs/source/theory/spatiotemporal_scale_topology.md, 8.3 and 8.4
2. relic/core/simulator/execution_group.py
3. relic/core/simulator/output_overlap_governance.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch

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

from relic.core.simulator.execution_group import (
    _check_flux_whitelist,
    _resolve_overlap_and_maybe_skip,
)
from relic.core.simulator.output_overlap_governance import (
    load_output_overlap_policies,
)
from relic.core.state.manager import UpdateItem, UpdateMode


from relic.core.simulator.main import RELICSimulator
from relic.contracts.contract_loader import ContractRegistry

def _run_overlap_scenarios() -> Dict[str, str]:
    """Run overlap governance scenarios against the runtime helper.

    The helper is a pure function of the variable, the writing functor, the
    update item and the step-level source ledger; it returns False when the
    write may proceed and raises when the governance policy forbids it.
    """
    outcomes: Dict[str, str] = {}

    # Scenario A: aggregated policy allows a second ACCUMULATE writer.
    sources_a = {"nh4_nitrogen": ("writer_A", UpdateMode.ACCUMULATE)}
    skip_a = _resolve_overlap_and_maybe_skip(
        var_name="nh4_nitrogen",
        functor_name="writer_B",
        update_item=UpdateItem(torch.tensor([0.1]), UpdateMode.ACCUMULATE),
        step_update_sources=sources_a,
    )
    outcomes["aggregated_allows_accumulate"] = "pass" if not skip_a else "fail"

    # Scenario B: the same writer re-emitting its own variable is never an overlap.
    sources_b = {"nh4_nitrogen": ("writer_A", UpdateMode.ACCUMULATE)}
    skip_b = _resolve_overlap_and_maybe_skip(
        var_name="nh4_nitrogen",
        functor_name="writer_A",
        update_item=UpdateItem(torch.tensor([0.4]), UpdateMode.ACCUMULATE),
        step_update_sources=sources_b,
    )
    outcomes["same_writer_not_overlap"] = "pass" if not skip_b else "fail"

    # Scenario C: a REPLACE conflict is rejected even under the aggregated policy.
    sources_c = {"nh4_nitrogen": ("writer_A", UpdateMode.REPLACE)}
    try:
        _resolve_overlap_and_maybe_skip(
            var_name="nh4_nitrogen",
            functor_name="writer_B",
            update_item=UpdateItem(torch.tensor([0.6]), UpdateMode.ACCUMULATE),
            step_update_sources=sources_c,
        )
        outcomes["replace_conflict_raises"] = "fail"
    except RuntimeError:
        outcomes["replace_conflict_raises"] = "pass"

    # Scenario D: mutually_exclusive raises for a second writer.
    sources_d = {"evapotranspiration_flux": ("model_A", UpdateMode.ACCUMULATE)}
    try:
        _resolve_overlap_and_maybe_skip(
            var_name="evapotranspiration_flux",
            functor_name="model_B",
            update_item=UpdateItem(torch.tensor([0.2]), UpdateMode.ACCUMULATE),
            step_update_sources=sources_d,
        )
        outcomes["mutually_exclusive_raises"] = "fail"
    except RuntimeError:
        outcomes["mutually_exclusive_raises"] = "pass"

    return outcomes


def _run_flux_whitelist_demo() -> Dict[str, int]:
    """Run whitelist check and count warning records."""
    logger = logging.getLogger("relic.core.simulator.execution_group")
    records: List[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "is not authorized for" in record.getMessage() and "flux direction(s) of" in record.getMessage():
                records.append(record.getMessage())

    handler = _Capture()
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        whitelist = {
            "soil_moisture": {
                "flux_in": ["HydrologyIn"],
                "flux_out": ["HydrologyOut"],
            }
        }
        registry = ContractRegistry()
        registry._flux_whitelist = whitelist

        from relic.core.state.manager import StateManager
        state_manager = StateManager(schema={}, autoload_contract_registry=False)
        state_manager._contract_registry = registry
        
        sim = RELICSimulator({"time_step": 3600.0})
        sim.state_manager = state_manager

        # Positive update from unauthorized writer -> warning against flux_in whitelist.
        _check_flux_whitelist(
            self=sim,
            functor_name="UnauthorizedIn",
            adjusted_updates={
                "soil_moisture": UpdateItem(torch.tensor([0.3, 0.2]), UpdateMode.ACCUMULATE)
            },
        )
        # Negative update from unauthorized writer -> warning against flux_out whitelist.
        _check_flux_whitelist(
            self=sim,
            functor_name="UnauthorizedOut",
            adjusted_updates={
                "soil_moisture": UpdateItem(torch.tensor([-0.4, -0.1]), UpdateMode.ACCUMULATE)
            },
        )
        assert records == [
            "Functor UnauthorizedIn is not authorized for input flux direction(s) of soil_moisture",
            "Functor UnauthorizedOut is not authorized for output flux direction(s) of soil_moisture",
        ], records
        warning_count_before_controls = len(records)
        # Authorized input/output and zero flow must not produce direction warnings.
        for writer, values in [
            ("HydrologyIn", [0.3, 0.2]),
            ("HydrologyOut", [-0.4, -0.1]),
            ("UnauthorizedZero", [0.0, 0.0]),
        ]:
            _check_flux_whitelist(
                self=sim, functor_name=writer,
                adjusted_updates={"soil_moisture": UpdateItem(torch.tensor(values), UpdateMode.ACCUMULATE)},
            )
        assert len(records) == warning_count_before_controls, records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    return {
        "warning_count": len(records),
        "input_warning_count": sum("input flux direction(s)" in message for message in records),
        "output_warning_count": sum("output flux direction(s)" in message for message in records),
        "control_warning_count": len(records) - warning_count_before_controls,
    }


def main() -> None:
    print("=== Tutorial 21: Flux whitelist and overlap governance ===")

    policies = load_output_overlap_policies()
    policy_keys = ["aggregated", "mutually_exclusive"]
    policy_counts = {key: 0 for key in policy_keys}
    for entry in policies.values():
        p = entry.get("policy")
        if p in policy_counts:
            policy_counts[p] += 1

    overlap_outcomes = _run_overlap_scenarios()
    whitelist_stats = _run_flux_whitelist_demo()
    assert len(overlap_outcomes) == 4 and all(value == "pass" for value in overlap_outcomes.values()), overlap_outcomes

    print(f"  Loaded policies: {len(policies)}")
    print(f"  Policy counts: {policy_counts}")
    print(f"  Overlap outcomes: {overlap_outcomes}")
    print(f"  Flux whitelist warning counts: {whitelist_stats}")

    # ------------------------------------------------------------------
    # Figure 1: pure policy mathematics
    # ------------------------------------------------------------------
    apply_nature_style()
    fig1, axes1 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    ax = axes1[0]
    x = np.arange(len(policy_keys))
    y = [policy_counts[k] for k in policy_keys]
    ax.bar(x, y, color=[CB_PALETTE[0], CB_PALETTE[1]], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(policy_keys, rotation=15, fontsize=7)
    ax.set_ylabel("Variable count")
    ax.set_title("Governance Policy Distribution", fontsize=9)
    add_panel_label(ax, "a")

    # Panel b: policy action matrix as heatmap
    ax = axes1[1]
    # Rows: policy; columns: accept a second ACCUMULATE writer, reject a
    # REPLACE conflict, reject any second writer.
    action_matrix = np.array([
        [1, 1, 0],  # aggregated: accumulate allowed, REPLACE conflict rejected
        [0, 1, 1],  # mutually_exclusive: any second writer rejected
    ], dtype=float)
    action_labels = ["accept accumulate", "reject REPLACE", "reject 2nd writer"]
    im_pol = ax.imshow(action_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(action_labels, fontsize=7)
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(["aggregated", "mutually_exclusive"], fontsize=7)
    ax.set_xlabel("Action on overlap")
    ax.set_title("Policy Action Matrix", fontsize=9)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, "Y" if action_matrix[i, j] > 0.5 else "-",
                    ha="center", va="center", fontsize=8, fontweight="bold")
    cbar_pol = fig1.colorbar(im_pol, ax=ax, fraction=0.046, pad=0.04)
    cbar_pol.set_label("Active", fontsize=7)
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig1.suptitle("Pure Mathematics: Governance Operators on Overlapping Outputs", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig1, "21_math_governance_policies")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: RELIC empirical evidence
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    ax = axes2[0]
    names = list(overlap_outcomes.keys())
    vals = [1.0 if overlap_outcomes[name] == "pass" else 0.0 for name in names]
    ax.barh(names, vals, color=[CB_PALETTE[2] if v > 0.5 else CB_PALETTE[1] for v in vals], alpha=0.85)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Scenario pass (1=yes)")
    ax.set_title("Overlap Runtime Scenarios", fontsize=9)
    add_panel_label(ax, "a")

    # Panel b: grouped bar showing scenario outcome (pass/fail) + warning count
    ax = axes2[1]
    scenario_names_r = list(overlap_outcomes.keys()) + ["whitelist\nwarnings"]
    scenario_vals_r = [1 if overlap_outcomes[k] == "pass" else 0 for k in overlap_outcomes]
    # Normalize warning count: 2 warnings -> shown as bar height
    wc = whitelist_stats["warning_count"]
    scenario_vals_r.append(wc)
    colors_r = [CB_PALETTE[2] if v > 0.5 else CB_PALETTE[1] for v in scenario_vals_r[:-1]]
    colors_r.append(CB_PALETTE[4])  # warning count in different color
    ax.bar(np.arange(len(scenario_names_r)), scenario_vals_r, color=colors_r, alpha=0.85)
    ax.set_xticks(np.arange(len(scenario_names_r)))
    ax.set_xticklabels(scenario_names_r, fontsize=5, rotation=25, ha="right")
    ax.set_ylabel("pass=1 / count")
    ax.set_title("Governance Test Results", fontsize=9)
    add_panel_label(ax, "b", x=-0.06, y=1.10)

    fig2.suptitle("RELIC Evidence: Whitelist and Overlap Governance", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig2, "21_relic_governance_evidence")
    plt.close(fig2)

    print("[OK] Tutorial 21 finished.")


if __name__ == "__main__":
    main()

