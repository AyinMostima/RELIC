r"""Tutorial 03 -- DAG Compilation and Deterministic Topological Sort.

Demonstrates
------------
RELIC compiles the process dependency graph using Kahn's algorithm.
This tutorial uses RELIC's actual DependencyGraphResolver for both
the primary DAG sort and cycle detection.

This tutorial produces TWO figures:
  Fig 1  (03_math_dag.png): Kahn's algorithm level decomposition
  Fig 2  (03_relic_dag.png): RELIC DependencyGraphResolver output

References
----------
1. Kahn, A.B. (1962). Topological sorting of large networks. Comm. ACM.
2. Cormen, T.H. et al. (2009). Introduction to Algorithms, Ch.22.
3. coertheory.md, Theorem 3.6.1; spatiotemporal_scale_topology.md, Section 4

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.dependency_graph.DependencyGraphResolver
- relic.core.dependency_graph.CyclicDependencyError
- relic.core.functor.functor_types.FunctorMetadata, ProcessCategory
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
from relic.core.dependency_graph import DependencyGraphResolver, CyclicDependencyError
from relic.core.functor.functor_types import ProcessCategory, FunctorMetadata

# -----------------------------------------------------------------------
# Helper: Lightweight functor for RELIC's resolver
# -----------------------------------------------------------------------
from relic.core.functor.functor_base import FunctorBase

class TutorialFunctor(FunctorBase):
    """Tutorial functor compatible with DependencyGraphResolver."""
    def __init__(self, name, deps=None, cat=ProcessCategory.BIOLOGICAL_PROCESS):
        metadata = FunctorMetadata(
            name=name,
            category=cat,
            description="Tutorial stub functor for DAG topological sort demo.",
            references=[
                "docs/source/coertheory.md (DAG compilation and deterministic topological sort)",
            ],
            input_variables=[f"{name}_in"],
            output_variables=[f"{name}_out"],
        )
        # Declared predecessors are the instance declaration owned by FunctorBase.
        super().__init__(name=name, config={"dependencies": list(deps or [])}, metadata=metadata)

    def get_dependencies(self):
        return self._merge_declared_dependencies()

    def forward(self, state, **kwargs):
        return {}

# -----------------------------------------------------------------------
# 1. Define 8 processes with dependencies
# -----------------------------------------------------------------------
process_defs = [
    ("Radiation",        [],                  ProcessCategory.PHYSICAL_FIELD),
    ("Energy_Balance",   ["Radiation"],       ProcessCategory.PHYSICAL_FIELD),
    ("Photosynthesis",   ["Radiation"],       ProcessCategory.BIOLOGICAL_PROCESS),
    ("Stomatal_Cond",    ["Photosynthesis"],  ProcessCategory.BIOLOGICAL_PROCESS),
    ("Soil_Water",       [],                  ProcessCategory.TRANSPORT_MECHANISM),
    ("Transpiration",    ["Stomatal_Cond", "Soil_Water"], ProcessCategory.TRANSPORT_MECHANISM),
    ("Decomposition",    ["Soil_Water"],      ProcessCategory.BIOLOGICAL_PROCESS),
    ("Nutrient_Release", ["Decomposition"],   ProcessCategory.BIOLOGICAL_PROCESS),
]

# -----------------------------------------------------------------------
# 2. Use RELIC's DependencyGraphResolver for topological sort
# -----------------------------------------------------------------------
resolver = DependencyGraphResolver()
for fname, deps, cat in process_defs:
    resolver.add_functor(TutorialFunctor(fname, deps, cat))

relic_order = resolver.topological_sort()
relic_groups = resolver.get_parallel_groups()

print("[OK] RELIC topological sort succeeded")
print(f"  Order: {relic_order}")
print(f"  Parallel groups: {relic_groups}")

# -----------------------------------------------------------------------
# 3. Demonstrate RELIC cycle detection
# -----------------------------------------------------------------------
bad_resolver = DependencyGraphResolver()
bad_resolver.add_functor(TutorialFunctor("Radiation", ["Transpiration"]))
bad_resolver.add_functor(TutorialFunctor("Transpiration", ["Radiation"]))
cycle_detected = False
try:
    bad_resolver.topological_sort()
except (CyclicDependencyError, Exception):
    cycle_detected = True
    print("[OK] RELIC CyclicDependencyError raised for cyclic graph")

# -----------------------------------------------------------------------
# 4. Compute levels for layout
# -----------------------------------------------------------------------
name_to_level = {}
for lvl_idx, group in enumerate(relic_groups):
    for name in group:
        name_to_level[name] = lvl_idx

max_level = max(name_to_level.values()) if name_to_level else 0

# =====================================================================
# FIGURE 1: Pure Mathematical DAG + Levels
# =====================================================================
apply_nature_style()

fig1 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.38))
gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.45, width_ratios=[1.3, 1.0])

# --- Panel (a): DAG visualization ---
ax = fig1.add_subplot(gs1[0, 0])
level_colors = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[4], CB_PALETTE[1], CB_PALETTE[3]]

positions = {}
for lvl_idx, group in enumerate(relic_groups):
    n_in_group = len(group)
    for idx, name in enumerate(group):
        x = lvl_idx * 1.5
        y = (idx - (n_in_group - 1) / 2) * 1.0
        positions[name] = (x, y)

# Draw edges
for fname, deps, _ in process_defs:
    for dep_name in deps:
        if dep_name in positions and fname in positions:
            x0, y0 = positions[dep_name]
            x1, y1 = positions[fname]
            ax.annotate("", xy=(x1 - 0.15, y1),
                         xytext=(x0 + 0.15, y0),
                         arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                         color="#888",
                                         connectionstyle="arc3,rad=0.08"))

# Draw nodes
for name, (x, y) in positions.items():
    lvl = name_to_level[name]
    color = level_colors[lvl % len(level_colors)]
    circle = plt.Circle((x, y), 0.14, facecolor=color,
                         edgecolor="#aaaaaa", lw=0.8, alpha=0.88)
    ax.add_patch(circle)
    short = name[:5]
    ax.text(x, y, short, ha="center", va="center",
            fontsize=5, fontweight="bold", color="#333333")

if positions:
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
    ax.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Process DAG (Kahn's Level Decomposition)", fontsize=9, pad=6)
add_panel_label(ax, "a", x=-0.05, y=1.05)

# --- Panel (b): Gantt-like parallel groups ---
ax = fig1.add_subplot(gs1[0, 1])
all_names = [n for g in relic_groups for n in g]
y_map = {n: i for i, n in enumerate(all_names)}

for lvl_idx, group in enumerate(relic_groups):
    for name in group:
        color = level_colors[lvl_idx % len(level_colors)]
        fill = FILL_PALETTE[lvl_idx % len(FILL_PALETTE)]
        y_pos = y_map[name]
        ax.barh(y_pos, 0.8, left=lvl_idx, height=0.5,
                color=color, edgecolor="white", lw=0.5, alpha=0.85)
        ax.text(lvl_idx + 0.4, y_pos, name,
                ha="center", va="center", fontsize=5, color="#333333",
                fontweight="bold")

ax.set_xlabel("Execution Level", fontsize=8)
ax.set_yticks(range(len(all_names)))
ax.set_yticklabels(all_names, fontsize=6)
ax.set_title("Dependency Levels (Parallel Candidates)", fontsize=9, pad=6)
ax.invert_yaxis()
add_panel_label(ax, "b", x=-0.2, y=1.05)

fig1.suptitle("DAG Compilation: Deterministic Topological Sort (Kahn 1962)",
              fontsize=10, y=1.05, fontweight="bold")
savefig(fig1, "03_math_dag")
plt.close(fig1)

# =====================================================================
# FIGURE 2: RELIC System - Cycle Detection & Category Summary
# =====================================================================
fig2 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.32))
gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.4)

# Panel (a): Cycle detection visualization
ax = fig2.add_subplot(gs2[0, 0])
ax.axis("off")
ax.set_title("RELIC Cycle Detection", fontsize=9, pad=6)

info = (
    "DependencyGraphResolver Analysis\n"
    "=================================\n\n"
    f"Registered functors: {len(process_defs)}\n"
    f"Topological order:   {len(relic_order)} nodes\n"
    f"Parallel groups:     {len(relic_groups)} levels\n\n"
    "Cycle Test:\n"
    f"  A -> B, B -> A   =>  CyclicDependencyError\n"
    f"  Detected: {'YES' if cycle_detected else 'NO'}\n\n"
    "Process Categories:\n"
)
for fname, _, cat in process_defs:
    info += f"  {fname:<18s} [{cat.value}]\n"

ax.text(0.05, 0.95, info, transform=ax.transAxes,
        fontsize=6, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8",
                  edgecolor="#B0C4DE", alpha=0.95))
add_panel_label(ax, "a", x=-0.05, y=1.05)

# Panel (b): Level stats
ax = fig2.add_subplot(gs2[0, 1])
level_sizes = [len(g) for g in relic_groups]
ax.bar(range(len(level_sizes)), level_sizes,
       color=[level_colors[i % len(level_colors)] for i in range(len(level_sizes))],
       edgecolor="white", linewidth=0.5)
for i, s in enumerate(level_sizes):
    ax.text(i, s + 0.05, str(s), ha="center", fontsize=7, fontweight="bold")
ax.set_xlabel("Execution Level")
ax.set_ylabel("Functors per Level")
ax.set_title("Parallel Candidates per Level", fontsize=9, pad=6)
add_panel_label(ax, "b", x=-0.15, y=1.05)

fig2.suptitle("RELIC DependencyGraphResolver: DAG Validation & Scheduling",
              fontsize=10, y=1.03, fontweight="bold")
savefig(fig2, "03_relic_dag")
plt.close(fig2)

print("\n[OK] Tutorial 03 completed successfully.")
print("  Fig 1: Mathematical DAG + level decomposition")
print("  Fig 2: RELIC DependencyGraphResolver analysis")
