r"""Tutorial 11 -- Variable Provenance and Increment Attribution.

Demonstrates
------------
RELIC records per-step per-variable increment decomposition, enabling
attribution of state changes to specific processes.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Increment decomposition (coertheory.md Corollary 3.6.5):

    x^{n+1} - x^n = Sum_{i in W(x)} Delta x_i + rho_x   ... (1)

where W(x) = set of writers for variable x,
      rho_x = system correction (clipping, constraint enforcement).

For linear diagnostics J(u) = w^T u:
    J(u^{n+1}) - J(u^n) = Sum_j w_j Sum_{i in W(u_j)} Delta u_{j,i} + w^T rho

References
----------
1. coertheory.md, Corollary 3.6.5 (provenance attribution)
2. spatiotemporal_scale_topology.md, Section 10.1 (Def. 10.1)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    add_legend_outside,
    CB_PALETTE, DOUBLE_COL_WIDTH, add_panel_label,
    apply_nature_style, savefig,
)

# -----------------------------------------------------------------------
# 1. Simulate 4 processes contributing to soil_carbon
# -----------------------------------------------------------------------
# Reference: coertheory.md Corollary 3.6.5
# Math: Delta x = Sum_i Delta x_i + rho_x
#
# Processes:
#   1. Litter_Input: adds carbon from litter decomposition (positive)
#   2. Root_Exudate: adds carbon from root exudation (positive)
#   3. Microbial_Decomp: removes carbon via decomposition (negative)
#   4. Leaching: removes dissolved carbon (negative)

process_names = ["Litter Input", "Root Exudate", "Microbial Decomp", "Leaching"]
n_processes = len(process_names)
n_steps = 100

rng = np.random.default_rng(42)
soil_c = 500.0  # initial soil carbon

# Per-step per-process increments
increments = np.zeros((n_steps, n_processes))
soil_c_hist = np.zeros(n_steps + 1)
soil_c_hist[0] = soil_c
total_increment = np.zeros(n_steps)

for step in range(n_steps):
    t = step / n_steps  # normalized time [0,1]

    # Reference: first-order kinetics with seasonal modulation
    # Math: Delta x_litter = k_litter * C_litter * f(T)
    #       Delta x_root = k_root * NPP * f(season)
    #       Delta x_decomp = -k_decomp * C_soil * f(T, theta)
    #       Delta x_leach = -k_leach * C_dissolved * precip

    seasonal = 0.5 + 0.5 * np.sin(2 * np.pi * t)

    increments[step, 0] = 3.0 * seasonal + rng.normal(0, 0.3)         # Litter Input
    increments[step, 1] = 1.0 + 0.5 * seasonal + rng.normal(0, 0.2)   # Root Exudate
    increments[step, 2] = -(2.5 * seasonal + 1.0) + rng.normal(0, 0.3)  # Decomp
    increments[step, 3] = -(0.5 + 0.3 * seasonal) + rng.normal(0, 0.1)  # Leaching

    total_increment[step] = increments[step].sum()
    soil_c_hist[step + 1] = soil_c_hist[step] + total_increment[step]

# Build provenance records
cumulative = np.cumsum(increments, axis=0)
total_per_process = increments.sum(axis=0)
fractions = np.abs(total_per_process) / np.abs(total_per_process).sum()

print("[OK] Provenance tracking completed")
print(f"  Soil C: {soil_c_hist[0]:.1f} -> {soil_c_hist[-1]:.1f} g C m^-2")
for i, name in enumerate(process_names):
    print(f"  {name:20s}: total = {total_per_process[i]:+.1f}  "
          f"fraction = {fractions[i]:.1%}")

# -----------------------------------------------------------------------
# 2. Plot: 3-panel figure
# -----------------------------------------------------------------------
apply_nature_style()

fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.70))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40,
                       height_ratios=[1.0, 1.0])
time = np.arange(n_steps)

# --- Panel (a): Stacked area of per-process contributions ---
ax = fig.add_subplot(gs[0, 0])
# Stack positive and negative contributions separately
pos_mask = increments > 0
neg_mask = increments < 0

pos_data = np.where(pos_mask, increments, 0)
neg_data = np.where(neg_mask, increments, 0)

# Stacked area for positive contributions
ax.stackplot(time, pos_data.T, labels=process_names,
             colors=[CB_PALETTE[i] for i in range(n_processes)], alpha=0.7)
ax.plot(time, total_increment, "k-", linewidth=1.0, label="Net increment")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Step")
ax.set_ylabel("Increment [g C m$^{-2}$ step$^{-1}$]", fontsize=7)
ax.set_title("Per-Process Contributions", fontsize=9, pad=6)
ax.legend(fontsize=5, ncol=2, loc="upper right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "a", x=-0.18, y=1.05)

# --- Panel (b): Pie chart of total contribution fractions ---
ax = fig.add_subplot(gs[0, 1])
abs_totals = np.abs(total_per_process)
wedges, texts, autotexts = ax.pie(
    abs_totals, labels=process_names,
    colors=[CB_PALETTE[i] for i in range(n_processes)],
    autopct="%1.1f%%", startangle=90, pctdistance=0.75,
    textprops={"fontsize": 6},
)
for at in autotexts:
    at.set_fontsize(5.5)
ax.set_title("Attribution Fractions\n(absolute)", fontsize=9)
add_panel_label(ax, "b", x=-0.15, y=1.05)

# --- Panel (c): Soil carbon trajectory with provenance annotation ---
ax = fig.add_subplot(gs[1, :])
ax.plot(np.arange(n_steps + 1), soil_c_hist, color=CB_PALETTE[0], linewidth=1.5)
ax.fill_between(np.arange(n_steps + 1), soil_c_hist[0], soil_c_hist,
                alpha=0.1, color=CB_PALETTE[0])

# Annotate a specific step with provenance detail
step_annotate = 50
ax.axvline(step_annotate, color="gray", linestyle="--", linewidth=0.7)

detail_text = f"Step {step_annotate} provenance:\n"
for i, name in enumerate(process_names):
    detail_text += f"  {name}: {increments[step_annotate, i]:+.2f}\n"
detail_text += f"  Net: {total_increment[step_annotate]:+.2f}"

ax.annotate(detail_text, xy=(step_annotate, soil_c_hist[step_annotate]),
            xytext=(0.75, 0.85), textcoords="axes fraction",
            fontsize=5.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ccc"),
            arrowprops=dict(arrowstyle="->", color="grey", lw=0.6))

ax.set_xlabel("Step")
ax.set_ylabel("Soil Carbon [g C m$^{-2}$]", fontsize=7)
ax.set_title("State Trajectory + Provenance", fontsize=9, pad=6)
add_panel_label(ax, "c", x=-0.08, y=1.05)

fig.suptitle(
    "Variable Provenance: $\\Delta x = \\sum_{i \\in \\mathcal{W}(x)} "
    "\\Delta x_i + \\rho_x$",
    fontsize=10, y=1.03,
)
# GridSpec spacing is set explicitly; avoid tight_layout warnings with pie charts.
savefig(fig, "11_variable_provenance")
plt.close(fig)

print("\n[OK] Tutorial 11 completed successfully.")
print("  Demonstrated: per-step increment decomposition and attribution")
