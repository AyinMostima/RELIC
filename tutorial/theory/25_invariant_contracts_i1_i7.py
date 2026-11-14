r"""Tutorial 25 -- Invariant Definitions and an Archived Report Viewer.

The first figure illustrates seven contract families and their premises; its
identity matrix is a layout device, not an independence or coverage proof.
The second figure reads the existing test/invariants/invariant_report.json.
It does not execute tests, regenerate that report, or certify the current tree.
Only values actually present in the archived file are displayed; numeric
verification residuals and a universal machine-epsilon threshold are not invented.

References: the unified theoretical framework Section 5.4, the architecture
SSOT's state and evidence boundaries, and the existing invariant test contracts.
"""

from __future__ import annotations

import json
import hashlib
import textwrap
import os
import sys
from pathlib import Path
from typing import Dict, List

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


def _load_invariant_report() -> Dict[str, object]:
    """Read an archived artifact without running tests or changing the report."""
    report_path = _REPO / "test" / "invariants" / "invariant_report.json"
    content = report_path.read_bytes()
    data = json.loads(content.decode("utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("invariants"), dict):
        raise TypeError("The archived report must contain an invariants mapping")
    data = dict(data)
    data["artifact_sha256"] = hashlib.sha256(content).hexdigest()
    data["artifact_path"] = str(report_path)
    return data


def _invariant_math_catalog() -> List[Dict[str, str]]:
    """Return mathematical statement for each invariant contract."""
    # References:
    # - unified framework Section 5.4
    # - spatiotemporal topology Section 8.2
    return [
        {"id": "I1", "name": "No side effects", "expr": "S_after_read = S_before"},
        {"id": "I2", "name": "Unit consistency", "expr": "units(delta) == units(target)"},
        {
            "id": "I3",
            "name": "Conservation",
            "expr": "|sum(Delta_S) - sum(F_ext)*dt| <= tol",
        },
        {
            "id": "I4",
            "name": "Reversibility",
            "expr": "U(-delta, U(delta, S)) = S",
        },
        {
            "id": "I5",
            "name": "Commutativity",
            "expr": "U(delta_b, U(delta_a, S)) = U(delta_a, U(delta_b, S))",
        },
        {"id": "I6", "name": "dt scaling", "expr": "delta(dt2) = delta(dt1) * dt2/dt1"},
        {
            "id": "I7",
            "name": "Bounds clipping",
            "expr": "S_new = clamp(S + delta, L, U), clip_loss = unclamped - clamped",
        },
    ]


def main() -> None:
    print("=== Tutorial 25: Invariant definitions and archived labels; no tests run ===")

    report = _load_invariant_report()
    invariants = report.get("invariants", {})
    assert isinstance(invariants, dict)
    catalog = _invariant_math_catalog()

    pass_flags: List[float] = []
    recorded_statuses: List[str] = []
    for key in [
        "I1_no_side_effects",
        "I2_unit_consistency",
        "I3_conservation",
        "I4_reversibility",
        "I5_commutativity",
        "I6_dt_scaling",
        "I7_bounds_clipping",
    ]:
        status = str(invariants.get(key, "UNKNOWN")).upper()
        recorded_statuses.append(status)
        pass_flags.append(1.0 if status == "PASS" else 0.0)

    print(f"  Archived report timestamp: {report.get('timestamp', 'unknown')}")
    print(f"  Archived report summary: {report.get('summary', 'no summary')}")
    print(f"  Archived labels encoded for display (not new test results): {pass_flags}")

    # ------------------------------------------------------------------
    # Figure 1: definition layout and premises; no coverage or complexity measurement
    # ------------------------------------------------------------------
    apply_nature_style()
    fig1, axes1 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.44))

    ax = axes1[0]
    matrix = np.eye(7, dtype=float)
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.set_xticklabels([item["id"] for item in catalog], fontsize=7)
    ax.set_yticklabels([item["id"] for item in catalog], fontsize=7)
    ax.set_title("Contract Definition Slots (Not a Proof)", fontsize=9)
    cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Layout indicator only", fontsize=7)
    add_panel_label(ax, "a")

    # Panel b states premises instead of assigning invented complexity scores.
    ax = axes1[1]
    ax.axis("off")
    premises = [
        ["I1", "No mutation of declared inputs"],
        ["I2", "Declared units and conversions"],
        ["I3", "Closed ledger or explicit exchange"],
        ["I4", "Reversible, unclipped updates only"],
        ["I5", "Independent compatible increments"],
        ["I6", "Integrate a rate exactly once"],
        ["I7", "Disclose signed clipping residual"],
    ]
    table = ax.table(cellText=premises, colLabels=["Family", "Premise / obligation"],
                     loc="center", cellLoc="left", colWidths=[0.17, 0.83])
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.2)
    ax.set_title("Definition Reminders, Not Measured Complexity", fontsize=8)
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig1.suptitle("Invariant Definitions and Premises: No Test Execution", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig1, "25_math_invariants_i1_i7")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: historical report contents; not current regression evidence
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.44))

    ax = axes2[0]
    ids = [item["id"] for item in catalog]
    colors = [CB_PALETTE[2] if status == "PASS" else "grey" for status in recorded_statuses]
    ax.bar(ids, pass_flags, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0.0, 1.25)
    for index, (height, status) in enumerate(zip(pass_flags, recorded_statuses)):
        ax.text(index, height + 0.03, status, ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("Archived flag (0=other or unknown)")
    ax.set_title("Recorded Status Labels (Not Rerun)", fontsize=9)
    add_panel_label(ax, "a")

    # Panel b displays only recorded fields and the artifact identity.
    ax = axes2[1]
    ax.axis("off")
    timestamp = str(report.get("timestamp", "not recorded"))
    summary = str(report.get("summary", "not recorded")).strip("= ")
    i2_status = str(invariants.get("I2_unit_consistency", "UNKNOWN"))
    annotation = "\n".join([
        "ARCHIVED ARTIFACT ONLY",
        "Timestamp: " + timestamp[:10],
        "Reported summary:", textwrap.fill(summary, width=37),
        "I2 recorded label: " + i2_status,
        "SHA256: " + str(report["artifact_sha256"])[:16] + "...",
        "", "No numeric residuals are recorded here.",
        "No fresh tests are run by this script.",
        "No current-tree verification is inferred.",
    ])
    ax.text(0.03, 0.95, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=7,
            linespacing=1.35)
    ax.set_title("Artifact Contents and Evidence Limits", fontsize=8)
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig2.suptitle("Archived I1-I7 Report: Not a Fresh Regression Run", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig2, "25_relic_invariants_i1_i7")
    plt.close(fig2)

    print(f"  Archived artifact: {report['artifact_path']}")
    print(f"  Archived SHA256: {report['artifact_sha256']}")
    print("[OK] Definitions and archived fields rendered; no invariant tests were run.")


if __name__ == "__main__":
    main()
