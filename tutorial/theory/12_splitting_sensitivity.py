r"""Tutorial 12 -- Splitting Sensitivity (Commutativity Diagnostic).

Demonstrates
------------
When process operators do not commute, execution order affects results.
The splitting sensitivity S(v) quantifies this non-commutativity.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Splitting sensitivity (coertheory.md Corollary 3.6.4):

    S(v) = std_sigma(|v_sigma|) / mean_sigma(|v_sigma|)   ... (1)
    chi_comm = max_v S(v)                                   ... (2)

where sigma ranges over permutations of execution order.

When operators commute: S(v) = 0 for all v.
When they don't: S(v) > 0 indicates order-dependent sensitivity.

References
----------
1. coertheory.md, Corollary 3.6.4 (splitting sensitivity)
2. spatiotemporal_scale_topology.md, Theorem 6.1 (order sensitivity)
3. Sportisse, B. (2000). Operator splitting. JCP 161, 140-168.
"""

from __future__ import annotations

import os
import itertools
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE, DOUBLE_COL_WIDTH, add_panel_label,
    apply_nature_style, savefig,
)

# -----------------------------------------------------------------------
# 1. Define three "process" operators
# -----------------------------------------------------------------------
# Reference: coertheory.md Corollary 3.6.4
# Math: For each permutation sigma, compute v_sigma = result of
#       applying operators in order sigma

# State: [C_leaf, C_soil, T_leaf]
state_names = ["$C_{leaf}$", "$C_{soil}$", "$T_{leaf}$"]
x0 = np.array([200.0, 500.0, 298.0])  # [g C/m2, g C/m2, K]
dt = 1.0

# Operator A: Photosynthesis (increases C_leaf, depends on T_leaf)
# Reference: Farquhar et al. (1980) simplified
# Math: Delta C_leaf = Anet(T) * dt, where Anet depends on T nonlinearly
def op_photosynthesis(x, dt):
    """Photosynthesis: adds carbon to leaf, temperature-dependent."""
    T = x[2]
    # Non-linear T response: Gaussian centered at 298 K
    # Reference: Medlyn et al. (2002), T optimum curve
    A_net = 15.0 * np.exp(-0.5 * ((T - 298.0) / 10.0)**2)
    delta = np.array([A_net * dt, 0.0, 0.0])
    return x + delta

# Operator B: Decomposition (transfers C from leaf litter to soil, modifies T)
# Reference: Parton et al. (1987)
# Math: Delta C_leaf = -k1*C_leaf*dt, Delta C_soil = k1*C_leaf*dt
#       Delta T = heat_release * k1 * C_leaf * dt  (exothermic)
def op_decomposition(x, dt):
    """Decomposition: moves carbon, releases heat."""
    k1 = 0.05
    flux = k1 * x[0] * dt
    heat = 0.01 * flux  # small heat release
    delta = np.array([-flux, flux, heat])
    return x + delta

# Operator C: Energy balance (adjusts T, weakly depends on C_leaf)
# Math: Delta T = -lambda*(T - T_env) * dt + radiative_effect(C_leaf)
def op_energy_balance(x, dt):
    """Energy balance: adjusts temperature."""
    T_env = 293.0
    lam = 0.1
    # Leaf C affects albedo -> temperature
    # Reference: Bonan (2019), surface energy balance
    rad_effect = 0.002 * x[0] * dt
    delta_T = -lam * (x[2] - T_env) * dt + rad_effect
    delta = np.array([0.0, 0.0, delta_T])
    return x + delta

operators = [op_photosynthesis, op_decomposition, op_energy_balance]
op_names = ["Photosynthesis", "Decomposition", "Energy Balance"]

# -----------------------------------------------------------------------
# 2. Compute results for all 6 permutations (3! = 6)
# -----------------------------------------------------------------------
# Reference: coertheory.md Corollary 3.6.4
# Math: For each sigma in Perm(3), compute v_sigma

permutations = list(itertools.permutations(range(3)))
results = np.zeros((len(permutations), 3))

for p_idx, perm in enumerate(permutations):
    x = x0.copy()
    for op_idx in perm:
        x = operators[op_idx](x, dt)
    results[p_idx] = x

# Compute splitting sensitivity S(v) for each variable
# Reference: coertheory.md Corollary 3.6.4
# Math: S(v) = std(|v_sigma|) / mean(|v_sigma|)
S_values = np.std(np.abs(results), axis=0) / np.mean(np.abs(results), axis=0)
chi_comm = np.max(S_values)

print("[OK] Splitting sensitivity computed over 6 permutations")
for i, name in enumerate(state_names):
    vals = results[:, i]
    print(f"  {name}: range = [{vals.min():.4f}, {vals.max():.4f}], S = {S_values[i]:.6f}")
print(f"  chi_comm = max S(v) = {chi_comm:.6f}")

# -----------------------------------------------------------------------
# 3. Plot: 3-panel figure
# -----------------------------------------------------------------------
apply_nature_style()

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.35))

# --- Panel (a): Bar chart of S(v) per variable ---
ax = axes[0]
x_pos = np.arange(3)
bars = ax.bar(x_pos, S_values, color=[CB_PALETTE[i] for i in range(3)],
              edgecolor="black", linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(state_names, fontsize=7)
ax.set_ylabel("$\\mathcal{S}(v)$")
ax.set_title("Splitting Sensitivity per Variable", fontsize=9)
for bar, sv in zip(bars, S_values):
    ax.text(bar.get_x() + bar.get_width()/2, sv + 0.0001,
            f"{sv:.5f}", ha="center", fontsize=6)
ax.axhline(0, color="black", linewidth=0.5)
add_panel_label(ax, "a")

# --- Panel (b): Scatter of values under different orderings ---
ax = axes[1]
for i in range(3):
    ax.scatter(np.ones(len(permutations)) * i + np.random.uniform(-0.15, 0.15, len(permutations)),
               results[:, i], color=CB_PALETTE[i], s=30, alpha=0.7,
               label=state_names[i], edgecolor="black", linewidth=0.3)
    ax.errorbar(i, np.mean(results[:, i]), yerr=np.std(results[:, i]),
                color="black", capsize=3, linewidth=1.5, zorder=5)

ax.set_xticks(range(3))
ax.set_xticklabels(state_names, fontsize=7)
ax.set_ylabel("Value under permutation $\\sigma$")
ax.set_title("Order-Dependent Spread", fontsize=9)
ax.legend(fontsize=6, frameon=False, prop={"weight": "bold"})
add_panel_label(ax, "b")

# --- Panel (c): Heatmap of deviations from mean ---
ax = axes[2]
mean_vals = results.mean(axis=0)
deviations = results - mean_vals  # shape (6, 3)

# Normalize deviations for visualization
norm_dev = deviations / (np.abs(mean_vals) + 1e-10) * 100  # percentage

im = ax.imshow(norm_dev, cmap="RdBu_r", aspect="auto",
               vmin=-np.abs(norm_dev).max(), vmax=np.abs(norm_dev).max())
ax.set_xticks(range(3))
ax.set_xticklabels(state_names, fontsize=6)
ax.set_yticks(range(6))
perm_labels = ["-".join(op_names[i][:3] for i in p) for p in permutations]
ax.set_yticklabels(perm_labels, fontsize=5)
ax.set_xlabel("Variable")
ax.set_ylabel("Permutation $\\sigma$")
ax.set_title("Deviation from Mean [%]", fontsize=9)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(labelsize=6)
cbar.set_label("%", fontsize=7)

# Annotate cells
for i in range(6):
    for j in range(3):
        ax.text(j, i, f"{norm_dev[i,j]:.3f}", ha="center", va="center",
                fontsize=4.5, color="black" if abs(norm_dev[i,j]) < 0.5*np.abs(norm_dev).max() else "white")

add_panel_label(ax, "c")

fig.suptitle(
    "Splitting Sensitivity: "
    "$\\mathcal{S}(v) = \\mathrm{std}_\\sigma(|v_\\sigma|)"
    " / \\mathrm{mean}_\\sigma(|v_\\sigma|)$",
    fontsize=10, y=1.05,
)
plt.tight_layout()
savefig(fig, "12_splitting_sensitivity")
plt.close(fig)

print("\n[OK] Tutorial 12 completed successfully.")
print(f"  Demonstrated: splitting sensitivity S(v) and chi_comm = {chi_comm:.6f}")
