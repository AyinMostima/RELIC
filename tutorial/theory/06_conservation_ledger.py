r"""Tutorial 06 -- Conservation Ledger and Budget Residual.

Demonstrates
------------
RELIC tracks per-domain budget residuals at every time step:

    r = |Delta S - (Sum F_in - Sum F_out) * dt|

When r ~ machine epsilon, the budget is closed.  When r drifts,
a flux pathway is missing or misconfigured.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Budget closure check (coertheory.md Section 3.3):

    Delta S = (Sum F_in - Sum F_out) * dt             ... (1)
    residual r = |Delta S - (Sum F_in - Sum F_out)*dt| ... (2)

For a closed system with incidence matrix B:
    d/dt (1^T u) = 0  =>  Sum Delta u_i = 0           ... (3)

References
----------
1. coertheory.md, Section 3.3 (Conservation ledger)
2. Manzoni, S. & Porporato, A. (2009). Soil carbon and nitrogen
   mineralization. Soil Biol. Biochem. 41, 1355-1379.
3. Parton, W.J. et al. (1987). CENTURY model. SSSA J. 51, 1173-1179.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
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
# 1. Four-pool carbon cycle
# -----------------------------------------------------------------------
# Pools: Leaf, Litter, SOM_fast, SOM_slow
# Reference: Parton et al. (1987)
pool_names = ["Leaf", "Litter", "SOM\\_fast", "SOM\\_slow"]
n = 4

# Incidence matrix B for fluxes:
#   f0: Leaf -> Litter  (litterfall, rate k0=0.02)
#   f1: Litter -> SOM_fast (decomp, k1=0.04)
#   f2: Litter -> SOM_slow (slow decomp, k2=0.005)
#   f3: SOM_fast -> SOM_slow (humification, k3=0.01)
edges = [(0,1), (1,2), (1,3), (2,3)]
m = len(edges)

B = np.zeros((n, m))
for j, (s, t) in enumerate(edges):
    B[s, j] = -1
    B[t, j] = +1

k = np.array([0.02, 0.04, 0.005, 0.01])
dt = 0.5
n_steps = 1000

def simulate(B, k, u0, dt, n_steps, break_flux=None):
    """Simulate du/dt = B f(u) with Forward Euler.

    Parameters
    ----------
    break_flux : int or None
        If set, break edge j=break_flux by removing its sink contribution in B,
        simulating a missing pathway (a one-sided leak).

    Returns
    -------
    U : array (n_steps+1, n_pools)
    residuals : array (n_steps,) -- per-step budget residual
    """
    edges_arr = np.array(edges)
    U = np.zeros((n_steps + 1, n))
    U[0] = u0.copy()
    residuals = np.zeros(n_steps)

    for step in range(n_steps):
        u = U[step]
        sources = edges_arr[:, 0]
        f = k * u[sources]

        B_eff = B
        if break_flux is not None:
            # Break the sink endpoint in the incidence column (leak): column sum != 0.
            # This models a misconfigured pathway where mass leaves the source but is
            # not credited to the intended sink.
            B_eff = B.copy()
            sink_rows = np.where(B_eff[:, break_flux] > 0.0)[0]
            if sink_rows.size > 0:
                B_eff[sink_rows[0], break_flux] = 0.0

        dudt = B_eff @ f
        U[step+1] = u + dt * dudt

        # Budget residual: r = |Delta S - (Sum F_in - Sum F_out)*dt|.
        # For a closed internal-exchange system, Sum F_in = Sum F_out, so r = |Delta S|.
        # Reference: docs/source/coertheory.md Section 3.3 (budget residual form).
        delta_S = (U[step+1] - u).sum()
        residuals[step] = abs(delta_S)

    return U, residuals


u0 = np.array([300.0, 200.0, 100.0, 100.0])

# Scenario 1: All fluxes intact (conservation holds)
U1, r1 = simulate(B, k, u0, dt, n_steps, break_flux=None)

# Scenario 2: Broken Litter -> SOM_fast pathway (one-sided leak)
U2, r2 = simulate(B, k, u0, dt, n_steps, break_flux=1)

time = np.arange(n_steps + 1) * dt
time_r = np.arange(n_steps) * dt

total_C1 = U1.sum(axis=1)
total_C2 = U2.sum(axis=1)

print(f"[OK] Scenario 1 (intact): max residual = {r1.max():.2e}")
print(f"  Total C range: [{total_C1.min():.4f}, {total_C1.max():.4f}]")
print(f"[OK] Scenario 2 (broken): max residual = {r2.max():.2e}")
print(f"  Total C drift: {total_C2[-1] - total_C2[0]:.2f} g C m^-2")

# -----------------------------------------------------------------------
# 2. Plot: 3-panel figure
# -----------------------------------------------------------------------
apply_nature_style()

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.32))

# --- Panel (a): Pool dynamics - conserving ---
ax = axes[0]
for i in range(n):
    ax.plot(time, U1[:, i], label=pool_names[i], linewidth=1.2)
ax.plot(time, total_C1, "k--", linewidth=1.8, label="Total C", alpha=0.8)
ax.set_xlabel("Time")
ax.set_ylabel("Carbon [g C m$^{-2}$]")
ax.set_title("Intact Fluxes (Conserving)", fontsize=9)
ax.legend(fontsize=5, ncol=2, loc="upper left", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "a", x=-0.02, y=1.08)

# --- Panel (b): Pool dynamics - broken ---
ax = axes[1]
for i in range(n):
    ax.plot(time, U2[:, i], label=pool_names[i], linewidth=1.2)
ax.plot(time, total_C2, "k--", linewidth=1.8, label="Total C", alpha=0.8)
ax.set_xlabel("Time")
ax.set_ylabel("Carbon [g C m$^{-2}$]")
ax.set_title("Missing Flux (Non-Conserving)", fontsize=9)
ax.legend(fontsize=5, ncol=2, loc="upper left", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "b", x=-0.02, y=1.08)

# --- Panel (c): Residual comparison ---
ax = axes[2]
ax.semilogy(time_r, r1 + 1e-20, label="Intact (r = |Delta S|)",
            color=CB_PALETTE[2], linewidth=1.2)
ax.semilogy(time_r, r2 + 1e-20,
            label="Broken (r = |Delta S|, leak)",
            color=CB_PALETTE[1], linewidth=1.2)
ax.set_xlabel("Time")
ax.set_ylabel("Budget Residual")
ax.set_title("Conservation Diagnostic", fontsize=9)
ax.legend(fontsize=5, ncol=1, loc="upper left", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
ax.set_ylim(1e-18, 1e2)
add_panel_label(ax, "c")

fig.suptitle(
    "Conservation Ledger: $r = |\\Delta S - (\\sum F_{in} - \\sum F_{out})\\Delta t|$",
    fontsize=10, y=1.05,
)
plt.tight_layout()
savefig(fig, "06_conservation_ledger")
plt.close(fig)

print("\n[OK] Tutorial 06 completed successfully.")
print("  Demonstrated: budget residual detects missing flux pathways")
