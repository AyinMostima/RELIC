r"""Tutorial 02 -- Topological Fingerprint and Betti Numbers.

Demonstrates
------------
The topological fingerprint sigma(G) = (n, m, rank(B), kappa, beta_1)
is a parameter-free structural invariant of the process graph.

  kappa = dim ker(B^T)   -- connected components
  beta_1 = m - rank(B)   -- cycle rank (feedback complexity)

This tutorial produces TWO figures:
  Fig 1  (02_math_fingerprint.png): Pure topological invariant comparison
  Fig 2  (02_relic_fingerprint.png): RELIC DependencyGraphResolver analysis

References
----------
1. Godsil, C. & Royle, G. (2001). Algebraic Graph Theory. Ch 8.
2. coertheory.md, Sections 2.3-2.5, Corollaries 2.5.1-2.5.5

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.dependency_graph.DependencyGraphResolver
- relic.core.functor.functor_types.ProcessCategory, FunctorMetadata
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
    add_panel_label, add_text_box, apply_nature_style, savefig,
)

# === RELIC imports ===
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.functor.functor_types import ProcessCategory, FunctorMetadata


def build_incidence(n_nodes, edge_list):
    """Build incidence matrix B from (source, sink) edge list.

    Reference: coertheory.md Section 2.1
    Math: B[src,j] = -1, B[snk,j] = +1 for each edge j
    """
    m = len(edge_list)
    B = np.zeros((n_nodes, m), dtype=np.float64)
    for j, (src, snk) in enumerate(edge_list):
        B[src, j] = -1
        B[snk, j] = +1
    return B


def compute_fingerprint(B):
    """Compute topological fingerprint sigma(G).

    Reference: coertheory.md Section 2.3
    Math: sigma(G) = (n, m, rank(B), kappa, beta_1)
          kappa = n - rank(B)
          beta_1 = m - rank(B)
    """
    n, m = B.shape
    r = np.linalg.matrix_rank(B)
    return {"n": n, "m": m, "rank": r, "kappa": n - r, "beta1": m - r}


# -----------------------------------------------------------------------
# 1. Define 4 graph configurations
# -----------------------------------------------------------------------
configs = [
    {"name": "Tree",          "n": 4, "edges": [(0,1), (0,2), (2,3)]},
    {"name": "Single cycle",  "n": 4, "edges": [(0,1), (1,2), (2,3), (3,0)]},
    {"name": "Multi-cycle",   "n": 4, "edges": [(0,1), (1,2), (2,3), (3,0), (0,2)]},
    {"name": "Two components","n": 6, "edges": [(0,1), (1,2), (3,4), (4,5)]},
]

results = []
for cfg in configs:
    B = build_incidence(cfg["n"], cfg["edges"])
    fp = compute_fingerprint(B)
    fp.update(name=cfg["name"], edges=cfg["edges"], n_nodes=cfg["n"])
    results.append(fp)
    print(f"  {cfg['name']:18s}  sigma = ({fp['n']},{fp['m']},{fp['rank']},{fp['kappa']},{fp['beta1']})")

# -----------------------------------------------------------------------
# 2. RELIC DependencyGraphResolver for each configuration
# -----------------------------------------------------------------------
from relic.core.functor.functor_base import FunctorBase

class TutorialFunctor(FunctorBase):
    """Tutorial functor class compatible with DependencyGraphResolver."""
    def __init__(self, name, deps=None):
        metadata = FunctorMetadata(
            name=name,
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            description="Tutorial stub functor for topology fingerprint DAG analysis.",
            references=[
                "docs/source/coertheory.md (topological fingerprint sigma(G))",
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

relic_results = []
for cfg in configs:
    resolver = DependencyGraphResolver()
    node_names = [f"P{i}" for i in range(cfg["n"])]
    # Build dependency lists from edges (skip cycles)
    adj = {i: [] for i in range(cfg["n"])}
    for src, snk in cfg["edges"]:
        adj[snk].append(node_names[src])  # snk depends on src

    for i in range(cfg["n"]):
        resolver.add_functor(TutorialFunctor(node_names[i], adj[i]))

    try:
        order = resolver.topological_sort()
        groups = resolver.get_parallel_groups()
        relic_results.append({"name": cfg["name"], "order": order, "groups": groups, "acyclic": True})
    except Exception:
        relic_results.append({"name": cfg["name"], "order": None, "groups": None, "acyclic": False})

print("\n=== RELIC Analysis ===")
for rr in relic_results:
    status = "DAG" if rr["acyclic"] else "CYCLE (rejected)"
    print(f"  {rr['name']:18s}  {status}  groups={rr['groups']}")

# =====================================================================
# FIGURE 1: Pure Mathematical Demonstration
# =====================================================================
apply_nature_style()

fig1 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.50))
gs1 = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.35, wspace=0.3,
                        height_ratios=[1.2, 1.0])

graph_layouts = [
    {0: (0.2,0.8), 1: (0.8,0.8), 2: (0.2,0.2), 3: (0.8,0.2)},
    {0: (0.2,0.8), 1: (0.8,0.8), 2: (0.8,0.2), 3: (0.2,0.2)},
    {0: (0.2,0.8), 1: (0.8,0.8), 2: (0.8,0.2), 3: (0.2,0.2)},
    {0: (0.1,0.8), 1: (0.35,0.5), 2: (0.1,0.2),
     3: (0.65,0.8), 4: (0.9,0.5), 5: (0.65,0.2)},
]

panels = ["a", "b", "c", "d"]
for col, (res, layout) in enumerate(zip(results, graph_layouts)):
    ax = fig1.add_subplot(gs1[0, col])
    for nid, (x, y) in layout.items():
        circle = plt.Circle((x, y), 0.1,
                             facecolor=CB_PALETTE[nid % len(CB_PALETTE)],
                             edgecolor="white", lw=1.0, alpha=0.85)
        ax.add_patch(circle)
        ax.text(x, y, str(nid), ha="center", va="center",
                fontsize=7, fontweight="bold", color="#333333")
    for src, snk in res["edges"]:
        x0, y0 = layout[src]
        x1, y1 = layout[snk]
        dx, dy = x1-x0, y1-y0
        length = np.sqrt(dx**2 + dy**2)
        sh = 0.12/length if length > 0 else 0
        ax.annotate("", xy=(x1-dx*sh, y1-dy*sh),
                     xytext=(x0+dx*sh, y0+dy*sh),
                     arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#555"))
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    fp_str = f"$\\sigma$=({res['n']},{res['m']},{res['rank']},{res['kappa']},{res['beta1']})"
    ax.set_title(f"{res['name']}\n{fp_str}", fontsize=7, pad=4)
    add_panel_label(ax, panels[col], x=-0.08, y=1.06)

# Bottom row: bar chart
ax_bar = fig1.add_subplot(gs1[1, :])
x_pos = np.arange(len(results))
width = 0.28
bars_k = ax_bar.bar(x_pos - width/2, [r["kappa"] for r in results],
                     width, color=CB_PALETTE[0], edgecolor="white",
                     linewidth=0.5, label="$\\kappa$ (components)")
bars_b = ax_bar.bar(x_pos + width/2, [r["beta1"] for r in results],
                     width, color=CB_PALETTE[1], edgecolor="white",
                     linewidth=0.5, label="$\\beta_1$ (cycle rank)")
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([r["name"] for r in results], fontsize=7)
ax_bar.set_ylabel("Count")
ax_bar.set_title("Topological Invariants Comparison", fontsize=9, pad=6)
ax_bar.legend(loc="upper left", fontsize=7)
for bar in list(bars_k) + list(bars_b):
    h = bar.get_height()
    if h > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.06,
                    str(int(h)), ha="center", va="bottom", fontsize=7,
                    fontweight="bold")
add_panel_label(ax_bar, "e", x=-0.04, y=1.06)

fig1.suptitle("Topological Fingerprint: $\\sigma(\\mathcal{G}) = "
              "(n, m, \\mathrm{rank}(\\mathbf{B}), \\kappa, \\beta_1)$",
              fontsize=10, y=1.05, fontweight="bold")
savefig(fig1, "02_math_fingerprint")
plt.close(fig1)

# =====================================================================
# FIGURE 2: RELIC System Integration
# =====================================================================
fig2 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.35))
gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)

# Panel (a): DAG vs cycle classification by RELIC
ax = fig2.add_subplot(gs2[0, 0])
labels = [r["name"] for r in relic_results]
colors_bar = [CB_PALETTE[2] if r["acyclic"] else CB_PALETTE[1] for r in relic_results]
n_groups = [len(r["groups"]) if r["groups"] else 0 for r in relic_results]
bars = ax.barh(labels, n_groups, color=colors_bar, edgecolor="white", height=0.5)
for bar, rr in zip(bars, relic_results):
    status = "DAG" if rr["acyclic"] else "CYCLE"
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
            status, va="center", fontsize=6.5,
            color=CB_PALETTE[2] if rr["acyclic"] else CB_PALETTE[1],
            fontweight="bold")
ax.set_xlabel("Number of Parallel Groups")
ax.set_title("RELIC DependencyGraphResolver\nDAG Classification", fontsize=8, pad=10)
add_panel_label(ax, "a", x=-0.15, y=1.10)

# Panel (b): Summary table
ax = fig2.add_subplot(gs2[0, 1])
ax.axis("off")
ax.set_title("RELIC + Math Comparison", fontsize=8, pad=10)

header = f"{'Config':<16} {'sigma(G)':<18} {'RELIC DAG?':>10}\n"
header += "-" * 46 + "\n"
rows = ""
for res, rr in zip(results, relic_results):
    fp = f"({res['n']},{res['m']},{res['rank']},{res['kappa']},{res['beta1']})"
    dag = "YES" if rr["acyclic"] else "NO (cycle)"
    rows += f"{res['name']:<16} {fp:<18} {dag:>10}\n"

ax.text(0.05, 0.9, header + rows, transform=ax.transAxes,
        fontsize=6.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8",
                  edgecolor="#B0C4DE", alpha=0.95))
add_panel_label(ax, "b", x=-0.05, y=1.10)

fig2.suptitle("RELIC Verification: DependencyGraphResolver Validates Graph Acyclicity",
              fontsize=10, y=1.08, fontweight="bold")
savefig(fig2, "02_relic_fingerprint")
plt.close(fig2)

print("\n[OK] Tutorial 02 completed successfully.")
print("  Fig 1: Mathematical topological fingerprint comparison")
print("  Fig 2: RELIC DependencyGraphResolver DAG classification")
