r"""Tutorial 13 -- End-to-End Differentiability (Adjoint Sensitivity).

Demonstrates
------------
RELIC builds all processes on torch.Tensor, enabling gradient-based
parameter calibration via PyTorch autograd. This tutorial shows
automatic differentiation through a simplified photosynthesis model.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Adjoint sensitivity (00_unified_theoretical_framework.md Section 1.3):

    dL/d_theta = dL/dS_T * Prod_{t=1}^{T} dS_t/dS_{t-1} * dS_0/d_theta

Farquhar model (simplified):
    A_c = Vcmax * (Ci - Gamma*) / (Ci + Kc*(1 + O/Ko))
    A_j = J * (Ci - Gamma*) / (4*Ci + 8*Gamma*)
    A = min(A_c, A_j) - Rd

Loss function:
    L(theta) = 1/N * Sum (A_pred(theta) - A_obs)^2

References
----------
1. 00_unified_theoretical_framework.md, Section 1.3
2. Farquhar, G.D., von Caemmerer, S., Berry, J.A. (1980).
   A biochemical model of photosynthetic CO2 assimilation in
   leaves of C3 species. Planta, 149, 78-90.
3. Bonan, G.B. (2019). Climate Change and Terrestrial Ecosystem
   Modeling. Ch. 11.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE, DOUBLE_COL_WIDTH, add_panel_label,
    apply_nature_style, savefig,
)


def farquhar_model(Ci, Vcmax, Jmax):
    """Simplified Farquhar photosynthesis model (C3).

    Reference: Farquhar et al. (1980), Planta 149, 78-90.
    Math:
      A_c = Vcmax * (Ci - Gamma_star) / (Ci + Kc*(1 + O/Ko))   ... (Rubisco-limited)
      J = Jmax * PAR / (PAR + 2.1*Jmax)                         ... (electron transport)
      A_j = J * (Ci - Gamma_star) / (4*Ci + 8*Gamma_star)       ... (RuBP-limited)
      A = min(A_c, A_j) - Rd                                    ... (net assimilation)

    Parameters
    ----------
    Ci : torch.Tensor -- intercellular CO2 [umol mol^-1]
    Vcmax : torch.Tensor -- max carboxylation rate [umol m^-2 s^-1]
    Jmax : torch.Tensor -- max electron transport [umol m^-2 s^-1]

    Returns
    -------
    A : torch.Tensor -- net assimilation rate [umol m^-2 s^-1]
    """
    Gamma_star = torch.tensor(42.75)    # CO2 compensation point [umol/mol]
    Kc = torch.tensor(404.9)            # Michaelis for CO2 [umol/mol]
    Ko = torch.tensor(278.4e3)          # Michaelis for O2 [umol/mol]
    O = torch.tensor(210.0e3)           # O2 concentration [umol/mol]
    PAR = torch.tensor(1000.0)          # PAR [umol m^-2 s^-1]
    Rd = torch.tensor(1.5)              # dark respiration [umol m^-2 s^-1]

    A_c = Vcmax * (Ci - Gamma_star) / (Ci + Kc * (1.0 + O / Ko))
    J = Jmax * PAR / (PAR + 2.1 * Jmax)
    A_j = J * (Ci - Gamma_star) / (4.0 * Ci + 8.0 * Gamma_star)
    A = torch.min(A_c, A_j) - Rd
    return A


# -----------------------------------------------------------------------
# 1. Generate synthetic observations
# -----------------------------------------------------------------------
Vcmax_true = 65.0
Jmax_true = 130.0

Ci_values = torch.linspace(100.0, 600.0, 30)
A_obs = farquhar_model(Ci_values, torch.tensor(Vcmax_true), torch.tensor(Jmax_true))
# Add noise
torch.manual_seed(42)
A_obs = A_obs + torch.randn_like(A_obs) * 1.5

# -----------------------------------------------------------------------
# 2. Gradient-based parameter calibration
# -----------------------------------------------------------------------
# Reference: 00_unified_theoretical_framework.md Section 1.3
# Math: theta_{n+1} = theta_n - lr * dL/d_theta

Vcmax_param = torch.tensor(40.0, requires_grad=True)
Jmax_param = torch.tensor(80.0, requires_grad=True)
lr = 0.5

n_iters = 200
loss_history = []
vcmax_history = [Vcmax_param.item()]
jmax_history = [Jmax_param.item()]

optimizer = torch.optim.Adam([Vcmax_param, Jmax_param], lr=lr)

for iteration in range(n_iters):
    optimizer.zero_grad()
    A_pred = farquhar_model(Ci_values, Vcmax_param, Jmax_param)
    loss = torch.mean((A_pred - A_obs) ** 2)
    loss.backward()
    optimizer.step()

    # Ensure physical bounds
    with torch.no_grad():
        Vcmax_param.clamp_(10.0, 150.0)
        Jmax_param.clamp_(20.0, 300.0)

    loss_history.append(loss.item())
    vcmax_history.append(Vcmax_param.item())
    jmax_history.append(Jmax_param.item())

print(f"[OK] Calibration completed {n_iters} iterations (fixed budget; no convergence test)")
print(f"  Vcmax: {vcmax_history[0]:.1f} -> {vcmax_history[-1]:.1f} (true = {Vcmax_true})")
print(f"  Jmax:  {jmax_history[0]:.1f} -> {jmax_history[-1]:.1f} (true = {Jmax_true})")
print(f"  Final recorded loss: {loss_history[-1]:.4f}")

# -----------------------------------------------------------------------
# 3. Compute loss landscape for contour plot
# -----------------------------------------------------------------------
vcmax_range = np.linspace(30, 100, 50)
jmax_range = np.linspace(60, 200, 50)
VV, JJ = np.meshgrid(vcmax_range, jmax_range)
LL = np.zeros_like(VV)

for i in range(50):
    for j in range(50):
        A_p = farquhar_model(Ci_values, torch.tensor(VV[i, j]), torch.tensor(JJ[i, j]))
        LL[i, j] = torch.mean((A_p - A_obs) ** 2).item()

# -----------------------------------------------------------------------
# 4. Plot: 3-panel figure
# -----------------------------------------------------------------------
apply_nature_style()

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.35))

# --- Panel (a): Loss landscape with optimization trajectory ---
ax = axes[0]
levels = np.logspace(np.log10(LL.min() + 0.1), np.log10(LL.max()), 20)
cs = ax.contourf(VV, JJ, LL, levels=levels, cmap="viridis_r", alpha=0.8)
ax.contour(VV, JJ, LL, levels=levels, colors="white", linewidths=0.3, alpha=0.5)
ax.plot(vcmax_history, jmax_history, "o-", color="red", markersize=1.5,
        linewidth=0.8, label="Optimization path")
ax.plot(vcmax_history[0], jmax_history[0], "^", color="white", markersize=8,
        markeredgecolor="red", zorder=5, label="Start")
ax.plot(Vcmax_true, Jmax_true, "*", color="yellow", markersize=10,
        markeredgecolor="black", zorder=5, label="True")
ax.set_xlabel("$V_{cmax}$ [$\\mu$mol m$^{-2}$ s$^{-1}$]")
ax.set_ylabel("$J_{max}$ [$\\mu$mol m$^{-2}$ s$^{-1}$]")
ax.set_title("Loss Landscape $\\mathcal{L}(\\theta)$", fontsize=9)
ax.legend(fontsize=6, loc="upper right", frameon=False, prop={"weight": "bold"})
cbar = fig.colorbar(cs, ax=ax, shrink=0.7)
cbar.set_label("MSE Loss", fontsize=6)
cbar.ax.tick_params(labelsize=5)
add_panel_label(ax, "a")

# --- Panel (b): Loss convergence ---
ax = axes[1]
ax.semilogy(loss_history, color=CB_PALETTE[0], linewidth=1.2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean Squared Error")
ax.set_title("Loss Convergence", fontsize=9)
ax.text(0.5, 0.85, f"Final loss = {loss_history[-1]:.3f}",
        transform=ax.transAxes, fontsize=7, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ccc"))
add_panel_label(ax, "b")

# --- Panel (c): Prediction comparison ---
ax = axes[2]
Ci_np = Ci_values.numpy()
# Predictions before calibration
A_before = farquhar_model(Ci_values, torch.tensor(vcmax_history[0]),
                           torch.tensor(jmax_history[0])).detach().numpy()
A_after = farquhar_model(Ci_values, torch.tensor(vcmax_history[-1]),
                          torch.tensor(jmax_history[-1])).detach().numpy()

ax.scatter(Ci_np, A_obs.numpy(), s=15, color="black", alpha=0.6,
           label="Observations", zorder=3)
ax.plot(Ci_np, A_before, "--", color=CB_PALETTE[1], linewidth=1.2,
        label=f"Before ($V_{{cmax}}$={vcmax_history[0]:.0f})")
ax.plot(Ci_np, A_after, "-", color=CB_PALETTE[2], linewidth=1.5,
        label=f"After ($V_{{cmax}}$={vcmax_history[-1]:.1f})")
ax.set_xlabel("$C_i$ [$\\mu$mol mol$^{-1}$]")
ax.set_ylabel("$A_{net}$ [$\\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]")
ax.set_title("A/Ci Curve Fit", fontsize=9)
ax.legend(fontsize=6, frameon=False, prop={"weight": "bold"})
add_panel_label(ax, "c")

fig.suptitle(
    "End-to-End Differentiability: Gradient-Based Parameter Calibration",
    fontsize=10, y=1.03,
)
plt.tight_layout()
savefig(fig, "13_differentiability")
plt.close(fig)

print("\n[OK] Tutorial 13 completed successfully.")
print("  Demonstrated: PyTorch autograd through Farquhar model for parameter calibration")
