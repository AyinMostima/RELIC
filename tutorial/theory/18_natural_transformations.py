r"""Tutorial 18 -- Natural Transformations: Naturality Square as a Coupling Diagnostic.

Demonstrates
------------
This standalone NumPy script uses the naturality square as a diagnostic: it
measures whether a process commutes with spatial aggregation. RELIC couplings
are not natural transformations in general, and RELIC does not check
naturality at runtime (only metadata enums are imported below). A coupling
would be a natural transformation (spatiotemporal_scale_topology.md,
Definition 3.3)

    eta: F => G

consisting of a family of morphisms eta_{(s,t)}: F(s,t) -> G(s,t)
if for every scale transition alpha: (s1,t1) -> (s2,t2)
the naturality square commuted:

        F(s1,t1) ---eta_{(s1,t1)}---> G(s1,t1)
           |                        |
         F(alpha)               G(alpha)
           |                        |
           v                        v
        F(s2,t2) ---eta_{(s2,t2)}---> G(s2,t2)

Physical interpretation:
  For the coupling between photosynthesis (F) and stomatal conductance (G),
  compare computing at patch scale and then aggregating with aggregating the
  inputs first and then computing. The two paths agree for affine processes
  with normalized weights and generally differ for nonlinear processes
  (Jensen gap; aggregation bias).

  Path 1: Compute at fine scale, then aggregate   -- Agg(F(x))
  Path 2: Aggregate first, then compute at coarse -- F(Agg(x))
  Naturality: Agg(F(x)) = F(Agg(x))  (exact for linear processes)

For nonlinear processes, the commutativity error (Jensen gap) is:
  epsilon = ||Agg(F(x)) - F(Agg(x))||
which quantifies the "sub-grid variability bias."

This tutorial produces TWO figures:
  Fig 1 (18_math_natural.png):  Mathematical commutative diagram
  Fig 2 (18_relic_natural.png): RELIC coupling & nonlinearity analysis

References
----------
1. Mac Lane, S. (1971). Categories for the Working Mathematician. Springer.
2. spatiotemporal_scale_topology.md, Section 3.4 (Definition 3.3)
3. Jensen, J.L.W.V. (1906). Sur les fonctions convexes et les inegalites
   entre les valeurs moyennes. Acta Math. 30, 175-193.

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.functor.functor_types.SpatialScale
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
import matplotlib.patches as mpatches
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
from relic.core.functor.functor_types import (
    SpatialScale, ProcessCategory, VariableKind, FunctorMetadata,
)

# -----------------------------------------------------------------------
# 1. Define scientific processes as functors
# -----------------------------------------------------------------------
# F: Photosynthesis (nonlinear: saturating light response)
# G: Stomatal Conductance (coupled to photosynthesis output)
# Aggregation operator: Agg (area-weighted mean)

# Reference: Farquhar et al. (1980), Ball et al. (1987)
# Math:
#   A_n = V_cmax * (C_i - Gamma*) / (C_i + K_c*(1 + O/K_o)) - R_d
#   Simplified light-limited form:
#     A_n(PAR) = A_max * PAR / (PAR + K_PAR)   [Michaelis-Menten]
#
#   g_s(A_n, RH) = g0 + g1 * A_n * RH / C_s   [Ball-Berry]

# Parameters
A_MAX = 20.0     # umol CO2 m-2 s-1  (maximum assimilation)
K_PAR = 400.0    # umol photons m-2 s-1  (half-saturation PAR)
R_D = 1.0        # umol CO2 m-2 s-1  (dark respiration)
G0 = 0.01        # mol m-2 s-1  (residual conductance)
G1 = 10.0        # dimensionless (Ball-Berry slope)
RH = 0.65        # relative humidity
C_S = 380.0      # umol mol-1  (surface CO2)


def photosynthesis(par):
    """Michaelis-Menten photosynthesis: A_n(PAR).

    Reference: Farquhar et al. (1980), light-limited simplification
    Math: A_n = A_max * PAR / (PAR + K_PAR) - R_d
    Parameters:
      A_max: maximum assimilation rate [umol CO2 m-2 s-1]
      K_PAR: half-saturation PAR [umol photons m-2 s-1]
      R_d:   dark respiration [umol CO2 m-2 s-1]
    """
    return A_MAX * par / (par + K_PAR) - R_D


def stomatal_conductance(a_n):
    """Ball-Berry stomatal conductance: g_s(A_n).

    Reference: Ball et al. (1987)
    Math: g_s = g0 + g1 * A_n * RH / C_s
    Parameters:
      g0: residual conductance [mol m-2 s-1]
      g1: Ball-Berry slope [dimensionless]
      RH: relative humidity [0-1]
      C_s: surface CO2 concentration [umol mol-1]
    """
    return G0 + G1 * np.maximum(a_n, 0) * RH / C_S


def aggregate(values, weights):
    """Area-weighted aggregation operator.

    Reference: Jones (1999), conservative remapping
    Math: Agg(x) = sum(w_i * x_i) / sum(w_i)
    """
    return np.sum(values * weights) / np.sum(weights)


# -----------------------------------------------------------------------
# 2. Generate patch-level data (heterogeneous PAR)
# -----------------------------------------------------------------------
np.random.seed(42)
n_patches = 20
# Heterogeneous PAR across patches (e.g., sunlit/shaded leaves)
PAR_patches = np.random.gamma(shape=3.0, scale=200.0, size=n_patches)
patch_areas = np.random.dirichlet(np.ones(n_patches))

print("=== Naturality Test: Photosynthesis x Aggregation ===")
print(f"  n_patches = {n_patches}")
print(f"  PAR range: [{PAR_patches.min():.0f}, {PAR_patches.max():.0f}] "
      f"umol m-2 s-1")

# -----------------------------------------------------------------------
# 3. Test commutativity of the naturality square
# -----------------------------------------------------------------------
# Path 1: Compute at fine scale, then aggregate
#   A_n(PAR_i) for each patch, then Agg(A_n)
A_patches = photosynthesis(PAR_patches)
gs_patches = stomatal_conductance(A_patches)
path1_A = aggregate(A_patches, patch_areas)    # Agg(F(x))
path1_gs = aggregate(gs_patches, patch_areas)  # Agg(G(F(x)))

# Path 2: Aggregate first, then compute at coarse scale
#   Agg(PAR_i), then A_n(Agg(PAR))
PAR_agg = aggregate(PAR_patches, patch_areas)  # Agg(x)
path2_A = photosynthesis(PAR_agg)              # F(Agg(x))
path2_gs = stomatal_conductance(path2_A)       # G(F(Agg(x)))

# Commutativity error (Jensen gap for concave function)
# Reference: Jensen (1906), for concave f: E[f(X)] <= f(E[X])
eps_A = abs(path1_A - path2_A)
eps_gs = abs(path1_gs - path2_gs)
rel_eps_A = eps_A / (abs(path1_A) + 1e-12)
rel_eps_gs = eps_gs / (abs(path1_gs) + 1e-12)

print(f"\nPath 1 (fine -> aggregate):  A={path1_A:.4f}, gs={path1_gs:.4f}")
print(f"Path 2 (aggregate -> coarse): A={path2_A:.4f}, gs={path2_gs:.4f}")
print(f"Jensen gap (A):  {eps_A:.4f}  ({rel_eps_A:.2%} relative)")
print(f"Jensen gap (gs): {eps_gs:.4f}  ({rel_eps_gs:.2%} relative)")

# Jensen inequality direction for concave Michaelis-Menten:
# E[f(X)] <= f(E[X]) for concave f
# So path1_A <= path2_A (confirmed below)
print(f"Jensen direction: Agg(F(x)) {'<=' if path1_A <= path2_A else '>'} "
      f"F(Agg(x))  (concave function)")

# -----------------------------------------------------------------------
# 4. Scan across heterogeneity levels
# -----------------------------------------------------------------------
# Math: As CV(PAR) increases, Jensen gap grows
# This quantifies the "sub-grid variability bias"
cv_values = np.linspace(0.01, 1.5, 30)
jensen_gaps = []
for cv in cv_values:
    # Generate PAR with specified CV
    mean_par = 500.0
    std_par = cv * mean_par
    par_scan = np.maximum(np.random.normal(mean_par, std_par, 50), 0)
    w_scan = np.ones(50) / 50
    # Path 1
    a1 = aggregate(photosynthesis(par_scan), w_scan)
    # Path 2
    a2 = photosynthesis(aggregate(par_scan, w_scan))
    jensen_gaps.append(abs(a1 - a2))
jensen_gaps = np.array(jensen_gaps)

# -----------------------------------------------------------------------
# 5. Linear process test (should commute exactly)
# -----------------------------------------------------------------------
# Linear function: f(x) = a*x + b commutes with aggregation exactly
# Math: Agg(a*x + b) = a*Agg(x) + b = f(Agg(x))  (exact naturality)

def linear_process(par, a=0.02, b=1.0):
    """Linear radiation conversion (exact naturality)."""
    return a * par + b

linear_path1 = aggregate(linear_process(PAR_patches), patch_areas)
linear_path2 = linear_process(aggregate(PAR_patches, patch_areas))
linear_eps = abs(linear_path1 - linear_path2)
print(f"\nLinear process: eps = {linear_eps:.2e}  (should be ~0)")

# =====================================================================
# FIGURE 1: Mathematical Commutative Diagram
# =====================================================================
apply_nature_style()

fig1 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.48))
gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.4,
                        width_ratios=[1.0, 1.0, 1.0])

# --- Panel (a): Commutative square diagram ---
ax = fig1.add_subplot(gs1[0, 0])
ax.axis("off")
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)

# Four corners of the square
corners = {
    "TL": (0.12, 0.82), "TR": (0.88, 0.82),
    "BL": (0.12, 0.22), "BR": (0.88, 0.22),
}
labels = {
    "TL": "$F(s_1,t_1)$\n(Patch Photo.)",
    "TR": "$G(s_1,t_1)$\n(Patch $g_s$)",
    "BL": "$F(s_2,t_2)$\n(Landsc. Photo.)",
    "BR": "$G(s_2,t_2)$\n(Landsc. $g_s$)",
}
colors = {
    "TL": CB_PALETTE[0], "TR": CB_PALETTE[2],
    "BL": CB_PALETTE[0], "BR": CB_PALETTE[2],
}

# Draw nodes
for key, (x, y) in corners.items():
    fill = FILL_PALETTE[0] if "L" in key else FILL_PALETTE[2]
    rect = mpatches.FancyBboxPatch(
        (x - 0.11, y - 0.07), 0.22, 0.14,
        boxstyle="round,pad=0.02", facecolor=fill,
        edgecolor=colors[key], linewidth=1.2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, y, labels[key], ha="center", va="center",
            fontsize=5.5, color="#333")

# Horizontal arrows (eta coupling)
for y_key in [("TL", "TR"), ("BL", "BR")]:
    x0, y0 = corners[y_key[0]]
    x1, y1 = corners[y_key[1]]
    ax.annotate("", xy=(x1 - 0.13, y1), xytext=(x0 + 0.13, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                color=CB_PALETTE[3]))
ax.text(0.5, 0.88, "$\\eta_{(s_1,t_1)}$ (coupling)", ha="center",
        fontsize=6.5, color=CB_PALETTE[3], fontweight="bold")
ax.text(0.5, 0.14, "$\\eta_{(s_2,t_2)}$ (coupling)", ha="center",
        fontsize=6.5, color=CB_PALETTE[3], fontweight="bold")

# Vertical arrows (scale transition F(alpha), G(alpha))
for x_key in [("TL", "BL"), ("TR", "BR")]:
    x0, y0 = corners[x_key[0]]
    x1, y1 = corners[x_key[1]]
    ax.annotate("", xy=(x1, y1 + 0.09), xytext=(x0, y0 - 0.09),
                arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                color=CB_PALETTE[4]))
ax.text(0.02, 0.52, "$F(\\alpha)$\n(Agg)", ha="center",
        fontsize=6.5, color=CB_PALETTE[4], rotation=90)
ax.text(0.98, 0.52, "$G(\\alpha)$\n(Agg)", ha="center",
        fontsize=6.5, color=CB_PALETTE[4], rotation=-90)

# Commutative check symbol
ax.text(0.5, 0.52, "Commutes?\n$\\epsilon < \\delta$",
        ha="center", va="center", fontsize=7, fontweight="bold",
        color=CB_PALETTE[1],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F0",
                  edgecolor=CB_PALETTE[1], alpha=0.9))

ax.set_title("Naturality Square", fontsize=9, pad=8)
add_panel_label(ax, "a", x=-0.05, y=1.05)

# --- Panel (b): Two paths comparison ---
ax = fig1.add_subplot(gs1[0, 1])
par_range = np.linspace(10, 1500, 200)
an_curve = photosynthesis(par_range)
ax.plot(par_range, an_curve, color=CB_PALETTE[0], lw=1.8,
        label="$A_n$(PAR)")

# Show fine-scale points and their aggregate
ax.scatter(PAR_patches, A_patches, s=12, color=CB_PALETTE[2],
           alpha=0.6, zorder=5, label="Patch values")

# Path 1 result
ax.axhline(path1_A, color=CB_PALETTE[1], lw=1.0, ls="--",
           label=f"Agg($A_n$) = {path1_A:.2f}")
# Path 2 result
ax.axhline(path2_A, color=CB_PALETTE[3], lw=1.0, ls=":",
           label=f"$A_n$(Agg) = {path2_A:.2f}")
# Aggregate PAR
ax.axvline(PAR_agg, color="#aaa", lw=0.6, ls="--", alpha=0.5)

# Jensen gap annotation
mid_A = (path1_A + path2_A) / 2
ax.annotate("", xy=(PAR_agg + 30, path1_A),
            xytext=(PAR_agg + 30, path2_A),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color=CB_PALETTE[1]))
ax.text(PAR_agg + 40, mid_A, f"$\\epsilon={eps_A:.2f}$",
        fontsize=6.5, color=CB_PALETTE[1])

ax.set_xlabel("PAR [$\\mu$mol m$^{-2}$ s$^{-1}$]", fontsize=7)
ax.set_ylabel("$A_n$ [$\\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]", fontsize=7)
ax.set_title("Nonlinear Process\n(Jensen Gap)", fontsize=9, pad=8)
ax.legend(fontsize=5, ncol=2, loc="upper left", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "b", x=-0.20, y=1.05)

# --- Panel (c): Jensen gap vs heterogeneity ---
ax = fig1.add_subplot(gs1[0, 2])
ax.plot(cv_values, jensen_gaps, color=CB_PALETTE[0], lw=1.5,
        label="$|\\mathrm{Agg}(F) - F(\\mathrm{Agg})|$")
ax.fill_between(cv_values, 0, jensen_gaps, color=FILL_PALETTE[0],
                alpha=0.4)
ax.set_xlabel("CV(PAR) [heterogeneity]", fontsize=7)
ax.set_ylabel("Jensen Gap [$\\mu$mol CO$_2$]", fontsize=7)
ax.set_title("Sub-Grid Bias\nvs. Heterogeneity", fontsize=9, pad=8)
add_text_box(ax,
             "Concave $F$:\n$\\mathrm{Agg}(F) \\leq F(\\mathrm{Agg})$\n"
             "(Jensen ineq.)",
             loc="upper left", fontsize=5.5)
add_legend_outside(ax, fontsize=5.5)
add_panel_label(ax, "c", x=-0.20, y=1.05)

fig1.suptitle(
    r"Naturality Square as a Diagnostic: Does $\eta\!: F \Rightarrow G$ "
    r"Commute with Aggregation?",
    fontsize=10, y=1.03, fontweight="bold",
)
savefig(fig1, "18_math_natural")
plt.close(fig1)

# =====================================================================
# FIGURE 2: RELIC Integration & Nonlinearity Error Budget
# =====================================================================
fig2 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45))
gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.45,
                        width_ratios=[1.0, 1.0, 1.2])

# --- Panel (a): Linear vs nonlinear commutativity ---
ax = fig2.add_subplot(gs2[0, 0])
bar_labels = ["Linear\n$f=ax+b$", "Michaelis-\nMenten", "Ball-\nBerry"]
bar_vals = [linear_eps, eps_A, eps_gs]
bar_colors = [CB_PALETTE[2], CB_PALETTE[1], CB_PALETTE[3]]
bars = ax.bar(range(3), bar_vals, color=bar_colors, alpha=0.8, width=0.5)
for bar, val in zip(bars, bar_vals):
    label = f"{val:.2e}" if val < 0.01 else f"{val:.3f}"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
            label, ha="center", fontsize=5.5)
ax.set_xticks(range(3))
ax.set_xticklabels(bar_labels, fontsize=6.5)
ax.set_ylabel("Commutativity Error $\\epsilon$", fontsize=7)
ax.set_title("Naturality Error Budget", fontsize=9, pad=6)
ax.set_yscale("symlog", linthresh=1e-10)
add_panel_label(ax, "a", x=-0.18, y=1.05)

# --- Panel (b): Path comparison heatmap ---
ax = fig2.add_subplot(gs2[0, 1])
# Create a 2D scan: vary mean PAR and CV
mean_pars = np.linspace(100, 1200, 20)
cvs = np.linspace(0.05, 1.2, 20)
error_grid = np.zeros((len(cvs), len(mean_pars)))

np.random.seed(123)
for i, cv in enumerate(cvs):
    for j, mp in enumerate(mean_pars):
        par_samp = np.maximum(np.random.normal(mp, cv * mp, 30), 0)
        w = np.ones(30) / 30
        p1 = aggregate(photosynthesis(par_samp), w)
        p2 = photosynthesis(aggregate(par_samp, w))
        error_grid[i, j] = abs(p1 - p2)

im = ax.pcolormesh(mean_pars, cvs, error_grid, cmap="YlOrRd",
                   shading="auto")
cbar = fig2.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("$|\\mathrm{Agg}(F) - F(\\mathrm{Agg})|$", fontsize=6)
cbar.ax.tick_params(labelsize=6)
ax.set_xlabel("Mean PAR [$\\mu$mol m$^{-2}$ s$^{-1}$]", fontsize=7)
ax.set_ylabel("CV(PAR)", fontsize=7)
ax.set_title("Commutativity Error Map", fontsize=9, pad=6)
add_panel_label(ax, "b", x=-0.18, y=1.05)

# --- Panel (c): Path comparison bar chart (replaces text) ---
ax = fig2.add_subplot(gs2[0, 2])

# Grouped bar: Path 1 (Agg(F(x))) vs Path 2 (F(Agg(x))) for both F and G
x_pos = np.arange(2)
width = 0.3
p1_vals = [path1_A, path1_gs * 1000]  # scale gs to mmol for visibility
p2_vals = [path2_A, path2_gs * 1000]
bars1 = ax.bar(x_pos - width/2, p1_vals, width, color=CB_PALETTE[0],
               alpha=0.8, label="Path 1: Agg(F(x))")
bars2 = ax.bar(x_pos + width/2, p2_vals, width, color=CB_PALETTE[2],
               alpha=0.8, label="Path 2: F(Agg(x))")

# Annotate Jensen gap
for i, (v1, v2) in enumerate(zip(p1_vals, p2_vals)):
    gap = abs(v1 - v2)
    gap_pct = gap / max(abs(v1), 1e-12) * 100
    ax.annotate(f"$\\epsilon$={gap:.2f}\n({gap_pct:.1f}%)",
                xy=(i, max(v1, v2) + 0.3),
                ha="center", fontsize=5.5, color=CB_PALETTE[1],
                fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels([f"$A_n$\n[umol CO$_2$ m$^{{-2}}$ s$^{{-1}}$]", f"$g_s$\n[mmol m$^{{-2}}$ s$^{{-1}}$]"],
                    fontsize=6)
ax.set_ylabel("Process Output", fontsize=7)
ax.set_title("Two-Path Comparison", fontsize=8, pad=6)
add_legend_outside(ax, fontsize=5)
add_panel_label(ax, "c", x=-0.18, y=1.05)

fig2.suptitle(
    "Coupling Commutation Analysis: Naturality Defect & Jensen Gap (standalone NumPy)",
    fontsize=10, y=1.03, fontweight="bold",
)
savefig(fig2, "18_relic_natural")
plt.close(fig2)

# -----------------------------------------------------------------------
# Final assertions
# -----------------------------------------------------------------------
assert linear_eps < 1e-10, f"Linear naturality violated: eps={linear_eps}"
assert path1_A <= path2_A + 1e-10, "Jensen inequality direction wrong"
assert eps_A > 0.01, "Expected nonzero Jensen gap for concave function"

print(f"\n[OK] Tutorial 18 completed successfully.")
print(f"  Fig 1: Naturality square + Jensen gap analysis")
print(f"  Fig 2: coupling commutation analysis + error map")
print(f"  Linear commutativity: eps = {linear_eps:.2e}")
print(f"  Nonlinear Jensen gap: eps = {eps_A:.4f} ({rel_eps_A:.2%})")
print(f"  Jensen direction: Agg(F) <= F(Agg) for concave F")
