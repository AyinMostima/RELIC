r"""Tutorial 16 -- Hodge Decomposition on the Orchestration Complex.

Demonstrates
------------
The 1-dimensional Hodge decomposition on a biogeochemical process graph.

For a graph with incidence matrix B in R^{n x m}, the edge space R^m
admits a unique orthogonal direct-sum decomposition (coertheory.md, Thm 3.3):

    R^m  =  im(B^T)  (+)  ker(B)

where:
  im(B^T) -- "gradient flows" driven by node-potential differences
  ker(B)  -- "cycle flows" with zero divergence (B z = 0)

Physical meaning (Corollary 2.5.2-2.5.3):
  - Cycle flows are *unobservable* from stock data alone.
  - The minimum-norm (Moore-Penrose) flux f* = B^T (B B^T)^+ g
    lives entirely in im(B^T), i.e., zero cyclic component.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Given feasibility condition:  exists f s.t. g = B f
Any solution decomposes as:
    f = f_grad + f_cycle
    f_grad in im(B^T),  f_cycle in ker(B)
    f_grad . f_cycle = 0   (orthogonal)

Minimum-norm flux (Corollary 2.5.3):
    f* = B^T (B B^T)^+ g
    ||f*|| <= ||f||  for all feasible f
    f*_cycle = 0

This tutorial produces TWO figures:
  Fig 1 (16_math_hodge.png):  Pure mathematical Hodge decomposition
  Fig 2 (16_relic_hodge.png): RELIC structural analysis integration

References
----------
1. Hirani, A.N. (2003). Discrete Exterior Calculus. PhD Thesis, Caltech.
2. Godsil, C. & Royle, G. (2001). Algebraic Graph Theory. Cambridge.
3. Strang, G. (1988). Linear Algebra and Its Applications (3rd ed.).
4. coertheory.md, Sections 2.4-2.5 (Theorem 3.3, Corollaries 2.5.1-2.5.3)

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.dependency_graph.DependencyGraphResolver
- relic.core.functor.functor_types.ProcessCategory
- relic.core.functor.functor_types.VariableKind
- relic.core.functor.functor_types.FunctorMetadata
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Ensure repo root is on path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE, FILL_PALETTE, DOUBLE_COL_WIDTH,
    add_panel_label, add_text_box, add_legend_outside,
    apply_nature_style, savefig,
)

# === RELIC module imports ===
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.functor.functor_types import (
    VariableKind, ProcessCategory, FunctorMetadata,
)

# -----------------------------------------------------------------------
# 1. Build a biogeochemical cycle graph with cycles (6 pools, 8 fluxes)
# -----------------------------------------------------------------------
# Pools: 0:Leaf, 1:Wood, 2:Root, 3:Litter, 4:SOM_fast, 5:SOM_slow
# Fluxes (edges):
#   0: Leaf -> Litter      (litterfall)
#   1: Wood -> Litter      (wood turnover)
#   2: Root -> Litter      (root turnover)
#   3: Litter -> SOM_fast  (fast decomp input)
#   4: Litter -> SOM_slow  (slow decomp input)
#   5: SOM_fast -> Litter  (microbial recycling)  ** creates cycle **
#   6: SOM_fast -> SOM_slow (humification)
#   7: SOM_slow -> Litter  (slow recycling)       ** creates 2nd cycle **

POOL_NAMES = ["Leaf", "Wood", "Root", "Litter", "SOM_fast", "SOM_slow"]
FLUX_NAMES = [
    "Litterfall", "Wood turn.", "Root turn.",
    "Fast decomp.", "Slow decomp.", "Recycling",
    "Humification", "Slow recyc.",
]
n_pools = len(POOL_NAMES)
n_fluxes = len(FLUX_NAMES)

# Incidence matrix B: B[i,j] = +1 if edge j enters node i, -1 if leaves
# Reference: coertheory.md Section 2.1, Eq. B_{i,j}
B = np.zeros((n_pools, n_fluxes), dtype=np.float64)
edges = [
    (0, 3),  # Leaf -> Litter
    (1, 3),  # Wood -> Litter
    (2, 3),  # Root -> Litter
    (3, 4),  # Litter -> SOM_fast
    (3, 5),  # Litter -> SOM_slow
    (4, 3),  # SOM_fast -> Litter (recycling cycle)
    (4, 5),  # SOM_fast -> SOM_slow
    (5, 3),  # SOM_slow -> Litter (slow recycling cycle)
]
for j, (src, snk) in enumerate(edges):
    B[src, j] = -1
    B[snk, j] = +1

# -----------------------------------------------------------------------
# 2. Topological fingerprint
# -----------------------------------------------------------------------
# Reference: coertheory.md Corollary 2.5.5
# Math: sigma(G) = (n, m, rank(B), kappa, beta_1)
# where kappa = dim ker(B^T) (connected components)
#       beta_1 = dim ker(B)  (independent cycles)

rank_B = np.linalg.matrix_rank(B)
kappa = n_pools - rank_B
beta_1 = n_fluxes - rank_B
print(f"Topological fingerprint: sigma(G) = ({n_pools}, {n_fluxes}, "
      f"{rank_B}, {kappa}, {beta_1})")
print(f"  rank(B) = {rank_B}")
print(f"  kappa (connected components) = {kappa}")
print(f"  beta_1 (independent cycles) = {beta_1}")

# -----------------------------------------------------------------------
# 3. Hodge Decomposition: R^m = im(B^T) (+) ker(B)
# -----------------------------------------------------------------------
# Reference: coertheory.md Theorem 3.3
# Math: For any flux f in R^m:
#   f = f_grad + f_cycle
#   f_grad = B^T (B B^T)^+ B f   (projection onto im(B^T))
#   f_cycle = f - f_grad          (projection onto ker(B))
#   f_grad . f_cycle = 0          (orthogonal)

# Create a sample flux vector (an arbitrary feasible flux)
# Use donor-controlled kinetics: f_j = k_j * u_{src_j}
# Math: f_j = k_j * u_{src_j} [g C m^-2 day^-1]
k_rates = np.array([0.02, 0.005, 0.01, 0.03, 0.005, 0.01, 0.002, 0.001])
u0 = np.array([200.0, 300.0, 100.0, 150.0, 120.0, 130.0])
sources = [e[0] for e in edges]
f_total = k_rates * u0[sources]

# Compute projection matrices
# P_grad = B^T (B B^T)^+ B     -- projects onto im(B^T)
# P_cycle = I - P_grad          -- projects onto ker(B)
BBT = B @ B.T
BBT_pinv = np.linalg.pinv(BBT)
P_grad = B.T @ BBT_pinv @ B
P_cycle = np.eye(n_fluxes) - P_grad

f_grad = P_grad @ f_total    # gradient component
f_cycle = P_cycle @ f_total  # cyclic component

# Verification: orthogonality
dot_product = np.dot(f_grad, f_cycle)
print(f"\n[OK] Hodge Decomposition verified:")
print(f"  ||f||      = {np.linalg.norm(f_total):.4f}")
print(f"  ||f_grad|| = {np.linalg.norm(f_grad):.4f}")
print(f"  ||f_cycle||= {np.linalg.norm(f_cycle):.4f}")
print(f"  f_grad . f_cycle = {dot_product:.2e}  (should be ~0)")
print(f"  ||f||^2 = ||f_grad||^2 + ||f_cycle||^2:  "
      f"{np.linalg.norm(f_total)**2:.4f} = "
      f"{np.linalg.norm(f_grad)**2 + np.linalg.norm(f_cycle)**2:.4f}")

# Verify B @ f_cycle = 0 (cycle flows have zero divergence)
div_cycle = B @ f_cycle
print(f"  max|B f_cycle| = {np.abs(div_cycle).max():.2e}  (should be ~0)")

# -----------------------------------------------------------------------
# 4. Minimum-Norm Flux (Corollary 2.5.3)
# -----------------------------------------------------------------------
# Reference: coertheory.md Corollary 2.5.3
# Math: f* = B^T (B B^T)^+ g
# where g = B f (the observed node tendency)
# f* is the unique flux with minimum 2-norm that reproduces g

g = B @ f_total  # observed tendency
f_star = B.T @ BBT_pinv @ g  # minimum-norm flux

# f_star should equal f_grad (since f_star has zero cycle component)
err_fstar_fgrad = np.linalg.norm(f_star - f_grad)
print(f"\n[OK] Minimum-norm flux (Corollary 2.5.3):")
print(f"  ||f*|| = {np.linalg.norm(f_star):.4f}")
print(f"  ||f* - f_grad|| = {err_fstar_fgrad:.2e}  (should be ~0)")
print(f"  ||f*|| <= ||f||: {np.linalg.norm(f_star):.4f} <= "
      f"{np.linalg.norm(f_total):.4f}")

# -----------------------------------------------------------------------
# 5. Ker(B) basis -- independent cycle flows
# -----------------------------------------------------------------------
# Reference: coertheory.md Corollary 2.5.4
# Math: dim ker(B) = beta_1 = number of independent cycles

from scipy.linalg import null_space
Z = null_space(B)
print(f"\n  ker(B) basis vectors: {Z.shape[1]} (should be beta_1={beta_1})")
assert Z.shape[1] == beta_1, f"ker(B) dimension mismatch: {Z.shape[1]} != {beta_1}"

# -----------------------------------------------------------------------
# 6. RELIC DependencyGraphResolver integration
# -----------------------------------------------------------------------
from relic.core.functor.functor_base import FunctorBase

class TutorialFunctor(FunctorBase):
    """Lightweight functor stub for DependencyGraphResolver."""
    def __init__(self, name, deps=None, cat=ProcessCategory.BIOLOGICAL_PROCESS):
        metadata = FunctorMetadata(
            name=name,
            category=cat,
            description="Tutorial stub functor for DependencyGraphResolver demo.",
            references=[
                "docs/source/coertheory.md (DAG compilation and structural analysis)",
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

resolver = DependencyGraphResolver()
process_defs = [
    ("Radiation",      [],                ProcessCategory.PHYSICAL_FIELD),
    ("Photosynthesis", ["Radiation"],     ProcessCategory.BIOLOGICAL_PROCESS),
    ("Litterfall",     ["Photosynthesis"], ProcessCategory.BIOLOGICAL_PROCESS),
    ("Decomposition",  ["Litterfall"],    ProcessCategory.BIOLOGICAL_PROCESS),
    ("Humification",   ["Decomposition"], ProcessCategory.BIOLOGICAL_PROCESS),
]
for fname, deps, cat in process_defs:
    resolver.add_functor(TutorialFunctor(fname, deps, cat))

relic_order = resolver.topological_sort()
relic_groups = resolver.get_parallel_groups()
print(f"\n=== RELIC DependencyGraphResolver ===")
print(f"  Topological order: {relic_order}")
print(f"  Parallel groups: {relic_groups}")

# =====================================================================
# FIGURE 1: Pure Mathematical Hodge Decomposition
# =====================================================================
apply_nature_style()

fig1 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.80))
gs1 = gridspec.GridSpec(2, 2, figure=fig1, wspace=0.40, hspace=0.50)

# --- Panel (a): Process graph with cycles highlighted ---
ax = fig1.add_subplot(gs1[0, 0])
positions = {
    0: (0.2, 0.88), 1: (0.5, 0.88), 2: (0.8, 0.88),
    3: (0.5, 0.52), 4: (0.2, 0.16), 5: (0.8, 0.16),
}
# Draw nodes
for idx, (x, y) in positions.items():
    circle = plt.Circle((x, y), 0.09, facecolor=CB_PALETTE[idx],
                         edgecolor="white", linewidth=1.2, alpha=0.9)
    ax.add_patch(circle)
    ax.text(x, y, POOL_NAMES[idx], ha="center", va="center",
            fontsize=5.5, fontweight="bold", color="#333333")

# Draw edges with cycle highlighting
cycle_edges = {5, 7}  # recycling edges forming cycles
for j, (src, snk) in enumerate(edges):
    x0, y0 = positions[src]
    x1, y1 = positions[snk]
    dx, dy = x1 - x0, y1 - y0
    length = np.sqrt(dx**2 + dy**2)
    sh = 0.10 / length if length > 0 else 0
    color = "#E74C3C" if j in cycle_edges else "#555"
    lw = 1.4 if j in cycle_edges else 0.8
    ax.annotate("", xy=(x1 - dx*sh, y1 - dy*sh),
                xytext=(x0 + dx*sh, y0 + dy*sh),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color))

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Graph with Cycles", fontsize=9, pad=8)
ax.text(0.5, -0.06, f"$\\beta_1 = {beta_1}$ indep. cycles",
        ha="center", fontsize=7, color="#C44E52",
        transform=ax.transAxes)
add_panel_label(ax, "a", x=-0.08, y=1.05)

# --- Panel (b): Flux decomposition bar chart ---
ax = fig1.add_subplot(gs1[0, 1])
x_pos = np.arange(n_fluxes)
width = 0.26
bars1 = ax.bar(x_pos - width, f_total, width, color=CB_PALETTE[0],
               alpha=0.85, label="$f$ (total)")
bars2 = ax.bar(x_pos, f_grad, width, color=CB_PALETTE[2],
               alpha=0.85, label="$f_{\\mathrm{grad}}$")
bars3 = ax.bar(x_pos + width, f_cycle, width, color=CB_PALETTE[1],
               alpha=0.85, label="$f_{\\mathrm{cycle}}$")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"$f_{{{j+1}}}$" for j in range(n_fluxes)], fontsize=6)
ax.set_ylabel("Flux magnitude", fontsize=8)
ax.set_title("Hodge Decomposition", fontsize=9, pad=8)
ax.legend(fontsize=5.5, ncol=3, loc="upper right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "b", x=-0.20, y=1.05)

# --- Panel (c): Orthogonality visualization ---
ax = fig1.add_subplot(gs1[1, 0])
# Show f_grad and f_cycle as vectors projected onto first 2 PCA components
# Use SVD to project f_grad and f_cycle for 2D visualization
from numpy.linalg import svd
F_mat = np.column_stack([f_grad, f_cycle, f_total])
U, S, Vt = svd(F_mat - F_mat.mean(axis=1, keepdims=True), full_matrices=False)
proj_grad = U[:, :2].T @ f_grad
proj_cycle = U[:, :2].T @ f_cycle
proj_total = U[:, :2].T @ f_total

origin = np.array([0.0, 0.0])
ax.quiver(*origin, *proj_grad, angles='xy', scale_units='xy', scale=1,
          color=CB_PALETTE[2], width=0.03, label="$f_{\\mathrm{grad}}$")
ax.quiver(*origin, *proj_cycle, angles='xy', scale_units='xy', scale=1,
          color=CB_PALETTE[1], width=0.03, label="$f_{\\mathrm{cycle}}$")
ax.quiver(*origin, *proj_total, angles='xy', scale_units='xy', scale=1,
          color=CB_PALETTE[0], width=0.02, alpha=0.6, label="$f$ (total)")
# Show right angle marker
angle_size = max(np.abs(proj_grad).max(), np.abs(proj_cycle).max()) * 0.15
g_hat = proj_grad / (np.linalg.norm(proj_grad) + 1e-12) * angle_size
c_hat = proj_cycle / (np.linalg.norm(proj_cycle) + 1e-12) * angle_size
corner = g_hat + c_hat
ax.plot([g_hat[0], corner[0]], [g_hat[1], corner[1]], 'k-', lw=0.6)
ax.plot([c_hat[0], corner[0]], [c_hat[1], corner[1]], 'k-', lw=0.6)

pad = max(np.abs(proj_total).max(), np.abs(proj_grad).max(),
          np.abs(proj_cycle).max()) * 1.4
ax.set_xlim(-pad, pad)
ax.set_ylim(-pad, pad)
ax.set_aspect("equal")
ax.axhline(0, color="#ccc", lw=0.5, zorder=0)
ax.axvline(0, color="#ccc", lw=0.5, zorder=0)
ax.set_xlabel("PC1", fontsize=7)
ax.set_ylabel("PC2", fontsize=7)
ax.set_title("Orthogonality", fontsize=9, pad=8)
add_text_box(ax, f"$f_{{grad}} \\cdot f_{{cycle}}$\n$= {dot_product:.1e}$",
             loc="lower right", fontsize=6)
ax.legend(fontsize=5.5, ncol=1, loc="center right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3,
          bbox_to_anchor=(1.35, 0.5))
add_panel_label(ax, "c", x=-0.20, y=1.05)

# --- Panel (d): Minimum-norm vs total flux ---
ax = fig1.add_subplot(gs1[1, 1])
x_pos2 = np.arange(n_fluxes)
ax.bar(x_pos2 - 0.15, np.abs(f_total), 0.3, color=CB_PALETTE[0],
       alpha=0.7, label=f"$|f|$, $\\|f\\|={np.linalg.norm(f_total):.2f}$")
ax.bar(x_pos2 + 0.15, np.abs(f_star), 0.3, color=CB_PALETTE[2],
       alpha=0.7, label=f"$|f^*|$, $\\|f^*\\|={np.linalg.norm(f_star):.2f}$")
ax.set_xticks(x_pos2)
ax.set_xticklabels([f"$f_{{{j+1}}}$" for j in range(n_fluxes)], fontsize=6)
ax.set_ylabel("|Flux|", fontsize=8)
ax.set_title("Min-Norm Flux $f^*$", fontsize=9, pad=8)
ax.legend(fontsize=5, ncol=1, loc="upper right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "d", x=-0.20, y=1.05)

fig1.suptitle(
    r"Hodge Decomposition: $\mathbb{R}^m = \mathrm{im}(\mathbf{B}^\top)"
    r"\;\oplus\;\ker(\mathbf{B})$",
    fontsize=10, y=1.05, fontweight="bold",
)
savefig(fig1, "16_math_hodge")
plt.close(fig1)

# =====================================================================
# FIGURE 2: RELIC System Integration & Structural Analysis
# =====================================================================
fig2 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))
gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.45,
                        width_ratios=[1.0, 1.0, 1.2])

# --- Panel (a): Cycle basis vectors ---
ax = fig2.add_subplot(gs2[0, 0])
for k in range(Z.shape[1]):
    ax.barh(np.arange(n_fluxes) + k * 0.35, np.abs(Z[:, k]), 0.3,
            color=CB_PALETTE[k + 1], alpha=0.8,
            label=f"$z_{{{k+1}}}$ (cycle basis)")
ax.set_yticks(np.arange(n_fluxes))
ax.set_yticklabels([f"$f_{{{j+1}}}$" for j in range(n_fluxes)], fontsize=6)
ax.set_xlabel("|Coefficient|", fontsize=8)
ax.set_title("ker(B) Basis Vectors", fontsize=8, pad=6)
ax.legend(fontsize=5, ncol=1, loc="lower right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "a", x=-0.22, y=1.05)

# --- Panel (b): Norm comparison bar chart ---
ax = fig2.add_subplot(gs2[0, 1])
norm_labels = ["$\\|f\\|$", "$\\|f_{grad}\\|$", "$\\|f_{cyc}\\|$", "$\\|f^*\\|$"]
norm_vals = [
    np.linalg.norm(f_total),
    np.linalg.norm(f_grad),
    np.linalg.norm(f_cycle),
    np.linalg.norm(f_star),
]
colors_norm = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[1], CB_PALETTE[4]]
bars = ax.bar(range(4), norm_vals, color=colors_norm, alpha=0.85, width=0.55)
for bar, val in zip(bars, norm_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
            f"{val:.2f}", ha="center", fontsize=5.5, fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(norm_labels, fontsize=7)
ax.set_ylabel("2-Norm", fontsize=7)
ax.set_title("Flux Norm Comparison", fontsize=8, pad=6)
# Annotate key relationship
ax.annotate("$\\leq$", xy=(3, norm_vals[3] + 0.3),
            xytext=(0, norm_vals[0] + 0.3),
            fontsize=10, ha="center", color=CB_PALETTE[2],
            arrowprops=dict(arrowstyle="->", lw=1.2, color=CB_PALETTE[2]))
add_panel_label(ax, "b", x=-0.18, y=1.05)

# --- Panel (c): Topological fingerprint components + observability ---
ax = fig2.add_subplot(gs2[0, 2])
# Two stacked sub-visualizations: fingerprint bar + observability gauge
fp_labels = ["$n$\n(pools)", "$m$\n(fluxes)", "rank\n($\\mathbf{B}$)",
             "$\\kappa$\n(comp.)", "$\\beta_1$\n(cycles)"]
fp_vals = [n_pools, n_fluxes, rank_B, kappa, beta_1]
colors_fp = [CB_PALETTE[0], CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[4], CB_PALETTE[1]]
bars = ax.bar(range(5), fp_vals, color=colors_fp, alpha=0.8, width=0.55)
for bar, val in zip(bars, fp_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15,
            str(val), ha="center", fontsize=6, fontweight="bold")
ax.set_xticks(range(5))
ax.set_xticklabels(fp_labels, fontsize=5.5)
ax.set_ylabel("Count", fontsize=7)
ax.set_title(f"$\\sigma(\\mathcal{{G}}) = ({n_pools},{n_fluxes},"
             f"{rank_B},{kappa},{beta_1})$", fontsize=8, pad=6)
# Add observability annotation
obs_frac = rank_B / n_fluxes
ax.axhline(rank_B, color="#ccc", ls=":", lw=0.7, zorder=0)
ax.text(4.4, rank_B + 0.2,
        f"Observable: {rank_B}/{n_fluxes}\n= {obs_frac:.0%}",
        fontsize=5.5, color=CB_PALETTE[2], fontweight="bold")
add_panel_label(ax, "c", x=-0.18, y=1.05)

fig2.suptitle(
    "RELIC Structural Analysis: Hodge Decomposition & Cycle Observability",
    fontsize=10, y=1.03, fontweight="bold",
)
savefig(fig2, "16_relic_hodge")
plt.close(fig2)

# -----------------------------------------------------------------------
# Final assertions
# -----------------------------------------------------------------------
assert np.abs(dot_product) < 1e-10, "Orthogonality violated"
assert np.abs(div_cycle).max() < 1e-10, "Cycle divergence non-zero"
assert np.linalg.norm(f_star) <= np.linalg.norm(f_total) + 1e-10, \
    "Min-norm property violated"
assert Z.shape[1] == beta_1, "ker(B) dimension mismatch"

print("\n[OK] Tutorial 16 completed successfully.")
print(f"  Fig 1: Hodge decomposition (im(B^T) + ker(B))")
print(f"  Fig 2: RELIC structural analysis")
print(f"  sigma(G) = ({n_pools}, {n_fluxes}, {rank_B}, {kappa}, {beta_1})")
print(f"  beta_1 = {beta_1} independent cycles (unobservable from stocks)")
