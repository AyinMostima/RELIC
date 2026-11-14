r"""Tutorial 07 -- Multi-Rate Execution (Fast/Medium/Slow Groups).

Demonstrates
------------
This tutorial uses a hand-chosen demo schedule (fixed skips) to visualise
multi-rate savings. It is not the RELIC runtime assignment: in RELIC,
HOUR/SUB_DAILY/DAY map to fast, WEEK to medium, MONTH/SEASON/YEAR to slow
(after explicit execution_group, the Functor attribute and
characteristic_time_days take precedence), and medium/slow fire on
interval-index or calendar-month boundaries. The script prints the runtime
temporal_scale mapping next to each process for comparison.

  Demo Fast   (dt):    radiation, energy balance, stomatal response
  Demo Medium (N*dt):  photosynthesis, soil water, respiration, soil heat
  Demo Slow   (M*dt):  phenology, soil carbon, nutrient cycling

This tutorial produces TWO figures:
  Fig 1  (07_math_multirate.png): Multi-rate scheduling theory
  Fig 2  (07_relic_multirate.png): RELIC TemporalScale classification

References
----------
1. coertheory.md, Corollary 3.5.2 (multi-rate decomposition)
2. spatiotemporal_scale_topology.md, Section 5 (temporal groups)

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.functor.functor_types.TemporalScale
- relic.core.functor.functor_types.FunctorMetadata
- relic.core.functor.functor_types.ProcessCategory
- relic.core.simulator.dependency_ops (runtime temporal_scale group mapping)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE, FILL_PALETTE, DOUBLE_COL_WIDTH,
    add_panel_label, add_text_box, add_legend_outside,
    apply_nature_style, savefig,
)

# === RELIC imports ===
from types import SimpleNamespace

from relic.core.functor.functor_types import (
    TemporalScale, ProcessCategory, FunctorMetadata,
)
from relic.core.simulator.dependency_ops import _resolve_temporal_scale_group

# -----------------------------------------------------------------------
# 1. Define processes with RELIC TemporalScale classification
# -----------------------------------------------------------------------
process_table = [
    {"name": "Radiation",      "scale": TemporalScale.SECOND,
     "tau_hr": 0.1,  "cat": ProcessCategory.PHYSICAL_FIELD},
    {"name": "Energy Balance", "scale": TemporalScale.MINUTE,
     "tau_hr": 0.5,  "cat": ProcessCategory.PHYSICAL_FIELD},
    {"name": "Stomatal Cond",  "scale": TemporalScale.MINUTE,
     "tau_hr": 0.5,  "cat": ProcessCategory.BIOLOGICAL_PROCESS},
    {"name": "Photosynthesis", "scale": TemporalScale.HOUR,
     "tau_hr": 1.0,  "cat": ProcessCategory.BIOLOGICAL_PROCESS},
    {"name": "Soil Water",     "scale": TemporalScale.HOUR,
     "tau_hr": 6.0,  "cat": ProcessCategory.TRANSPORT_MECHANISM},
    {"name": "Respiration",    "scale": TemporalScale.SUB_DAILY,
     "tau_hr": 12.0, "cat": ProcessCategory.BIOLOGICAL_PROCESS},
    {"name": "Soil Heat",      "scale": TemporalScale.DAY,
     "tau_hr": 24.0, "cat": ProcessCategory.PHYSICAL_FIELD},
    {"name": "Phenology",      "scale": TemporalScale.WEEK,
     "tau_hr": 168,  "cat": ProcessCategory.BIOLOGICAL_PROCESS},
    {"name": "Soil Carbon",    "scale": TemporalScale.SEASON,
     "tau_hr": 2160, "cat": ProcessCategory.BIOLOGICAL_PROCESS},
    {"name": "Nutrient Cycle", "scale": TemporalScale.YEAR,
     "tau_hr": 8760, "cat": ProcessCategory.BIOLOGICAL_PROCESS},
]

# Demo schedule only; not dependency_ops._resolve_temporal_scale_group
dt_base = 0.5  # hr
group_defs = [
    {"label": "Fast (every dt)", "skip": 1,
     "scales": {TemporalScale.SECOND, TemporalScale.MINUTE}},
    {"label": "Medium (every N*dt)", "skip": 6,
     "scales": {TemporalScale.HOUR, TemporalScale.SUB_DAILY, TemporalScale.DAY}},
    {"label": "Slow (every M*dt)", "skip": 48,
     "scales": {TemporalScale.WEEK, TemporalScale.SEASON, TemporalScale.YEAR}},
]

groups = {g["label"]: [] for g in group_defs}
for p in process_table:
    for g in group_defs:
        if p["scale"] in g["scales"]:
            groups[g["label"]].append(p)
            break

# Use RELIC TemporalScale.to_characteristic_time_days()
print("=== RELIC TemporalScale Classification ===")
# Schedule-dependent scales (sub-daily, weekly, seasonal) carry an explicit
# period in days; fixed scales resolve from the time-constant SSOT.
DECLARED_PERIOD_DAYS = {
    TemporalScale.SUB_DAILY: 0.25,
    TemporalScale.WEEK: 7.0,
    TemporalScale.MONTH: 30.4375,
    TemporalScale.SEASON: 91.3125,
    TemporalScale.YEAR: 365.25,
}
for p in process_table:
    period = DECLARED_PERIOD_DAYS.get(p["scale"])
    char_days = p["scale"].to_characteristic_time_days(period_days=period)
    # Runtime group from the temporal_scale branch (priority 4) of the RELIC
    # assignment chain; explicit execution_group, the Functor attribute and
    # characteristic_time_days would take precedence if declared.
    runtime_group = _resolve_temporal_scale_group(
        SimpleNamespace(metadata=SimpleNamespace(temporal_scale=p["scale"]))
    )
    print(f"  {p['name']:18s}  {p['scale'].name:16s}  "
          f"tau={char_days:.2f} days  [{p['cat'].value}]  runtime={runtime_group}")
print("  (runtime = RELIC temporal_scale mapping; the demo schedule uses fixed skips)")

# Execution count simulation
n_total_steps = 480  # total base steps (10 days if dt=0.5hr)
exec_counts = {}
for g in group_defs:
    for p in groups[g["label"]]:
        exec_counts[p["name"]] = n_total_steps // g["skip"]

flops_saved = 1.0 - sum(exec_counts.values()) / (n_total_steps * len(process_table))
print(f"\n[OK] Multi-rate saves {flops_saved:.1%} of computation vs uniform stepping")

# =====================================================================
# FIGURE 1: Mathematical - Multi-rate scheduling
# =====================================================================
apply_nature_style()
group_colors = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[4]]

fig1 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.4))
gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.4, width_ratios=[1.3, 1.0])

# Panel (a): Gantt-style execution timeline
ax = fig1.add_subplot(gs1[0, 0])
t_show = min(n_total_steps, 96)
t_axis = np.arange(t_show) * dt_base

y_idx = 0
y_labels = []
for g_idx, g in enumerate(group_defs):
    for p in groups[g["label"]]:
        exec_times = np.arange(0, t_show, g["skip"])
        ax.scatter(exec_times * dt_base, np.ones_like(exec_times) * y_idx,
                   s=3, color=group_colors[g_idx], alpha=0.8, marker="|")
        y_labels.append(p["name"])
        y_idx += 1

ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=5.5)
ax.set_xlabel("Time [hr]")
ax.set_title("Multi-Rate Execution Timeline", fontsize=9, pad=6)
ax.invert_yaxis()

# Colored labels for groups
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=group_colors[i], label=g["label"])
           for i, g in enumerate(group_defs)]
ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.15),
          fontsize=6, ncol=3, frameon=False, prop={"weight": "bold"})
add_panel_label(ax, "a", x=-0.2, y=1.05)

# Panel (b): Execution count comparison
ax = fig1.add_subplot(gs1[0, 1])
names = list(exec_counts.keys())
counts = list(exec_counts.values())
uniform_counts = [n_total_steps] * len(names)

x_pos = np.arange(len(names))
width = 0.35
ax.barh(x_pos - width/2, uniform_counts, width, color=FILL_PALETTE[0],
        edgecolor=CB_PALETTE[0], linewidth=0.5, label="Uniform")
ax.barh(x_pos + width/2, counts, width, color=CB_PALETTE[0],
        edgecolor="white", linewidth=0.5, label="Multi-rate")
ax.set_yticks(x_pos)
ax.set_yticklabels(names, fontsize=5.5)
ax.set_xlabel("Executions per simulation")
ax.set_title("Computation Savings", fontsize=9, pad=6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6,
          frameon=False, prop={"weight": "bold"}, ncol=2)
ax.invert_yaxis()
add_panel_label(ax, "b", x=-0.2, y=1.05)

fig1.suptitle("Multi-Rate Execution Demo: Fixed-Skip Schedule (not RELIC runtime groups)",
              fontsize=10, y=1.03, fontweight="bold")
savefig(fig1, "07_math_multirate")
plt.close(fig1)

# =====================================================================
# FIGURE 2: RELIC TemporalScale enum
# =====================================================================
fig2 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.35))
gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.4)

# Panel (a): TemporalScale log-scale bar
ax = fig2.add_subplot(gs2[0, 0])
# UNKNOWN has no characteristic period by contract; schedule-dependent
# scales use the declared example periods from above.
ts_all = [ts for ts in TemporalScale if ts is not TemporalScale.UNKNOWN]
ts_names = [ts.name for ts in ts_all]
ts_days = [ts.to_characteristic_time_days(period_days=DECLARED_PERIOD_DAYS.get(ts)) for ts in ts_all]

# Log-scale horizontal bars
bars = ax.barh(ts_names, [max(d, 0.001) for d in ts_days],
               color=[CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(ts_all))],
               edgecolor="white", height=0.5)
ax.set_xscale("log")
ax.set_xlabel("Characteristic Time [days]")
ax.set_title("RELIC TemporalScale Lattice", fontsize=9, pad=6)
for bar, d in zip(bars, ts_days):
    ax.text(max(d, 0.001) * 1.2, bar.get_y() + bar.get_height()/2,
            f"{d:.2f}d", va="center", fontsize=5.5)
add_panel_label(ax, "a", x=-0.2, y=1.05)

# Panel (b): Process-to-scale mapping summary
ax = fig2.add_subplot(gs2[0, 1])
ax.axis("off")
ax.set_title("Declared Process-Scale Mapping", fontsize=9, pad=6)

info = "Process                Scale           Category\n"
info += "-" * 55 + "\n"
for p in process_table:
    info += (f"{p['name']:20s} {p['scale'].name:16s} "
             f"{p['cat'].value}\n")
info += f"\nSavings: {flops_saved:.1%} vs uniform"

ax.text(0.03, 0.95, info, transform=ax.transAxes,
        fontsize=5.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8",
                  edgecolor="#B0C4DE", alpha=0.95))
add_panel_label(ax, "b", x=-0.05, y=1.05)

fig2.suptitle("TemporalScale Lattice and Demo Process-Scale Mapping (not runtime groups)",
              fontsize=10, y=1.03, fontweight="bold")
savefig(fig2, "07_relic_multirate")
plt.close(fig2)

print("\n[OK] Tutorial 07 completed successfully.")
print("  Fig 1: Mathematical multi-rate scheduling")
print("  Fig 2: RELIC TemporalScale classification")
