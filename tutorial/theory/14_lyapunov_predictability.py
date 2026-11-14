r"""Tutorial 14 -- Lyapunov Exponent and Predictability Horizon.

Demonstrates
------------
This independent Lorenz experiment fits log separation over a finite window
of two trajectories without tangent-vector renormalization. The fitted slope
is not an asymptotic Lyapunov exponent or a RELIC validation result.
VariableKind labels describe data roles and do not determine chaos.

This tutorial produces TWO figures:
  Fig 1  (14_math_lyapunov.png): Lorenz attractor + divergence
  Fig 2  (14_relic_lyapunov.png): RELIC implications summary

References
----------
1. coertheory.md, Section 1.2
2. Lorenz (1963), J. Atmos. Sci. 20, 130-141.
3. Strogatz (2015), Nonlinear Dynamics and Chaos, Ch.9.

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.functor.functor_types.VariableKind (STATE vs DIAGNOSTIC)
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE, DOUBLE_COL_WIDTH,
    add_panel_label, add_text_box, apply_nature_style, savefig,
)
from relic.core.functor.functor_types import VariableKind


def lorenz_rhs(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Lorenz-63 RHS. Ref: Lorenz (1963)."""
    x, y, z = state[0], state[1], state[2]
    return torch.stack([sigma*(y-x), x*(rho-z)-y, x*y-beta*z])


def rk4_step(f, state, dt):
    """RK4 integrator step."""
    k1 = f(state)
    k2 = f(state + dt*0.5*k1)
    k3 = f(state + dt*0.5*k2)
    k4 = f(state + dt*k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


dt = 0.01
n_steps = 5000
x0 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
eps = 1e-9
x0_p = x0 + torch.tensor([eps, 0, 0], dtype=torch.float64)

traj1 = torch.zeros(n_steps+1, 3, dtype=torch.float64)
traj2 = torch.zeros(n_steps+1, 3, dtype=torch.float64)
traj1[0], traj2[0] = x0, x0_p
for i in range(n_steps):
    traj1[i+1] = rk4_step(lorenz_rhs, traj1[i], dt)
    traj2[i+1] = rk4_step(lorenz_rhs, traj2[i], dt)

delta = (traj1 - traj2).numpy()
div = np.linalg.norm(delta, axis=1)
div[div == 0] = 1e-20
time = np.arange(n_steps+1)*dt
log_div = np.log(div/div[0])

mask = (time > 1) & (time < 25) & (log_div > 0) & np.isfinite(log_div)
if mask.sum() <= 10:
    raise ValueError("The selected window has too few finite separation samples")
# References: Lorenz (1963); NumPy least-squares fit of this trajectory pair.
# Mathematical expression: log(d(t)/d(0)) = slope * t + intercept in the chosen window.
# Parameter meanings: d is trajectory distance; mask selects finite positive separation.
# Units: slope is inverse model-time; the Lorenz coordinates are dimensionless.
# Domain/constraints: no tangent renormalization; this finite-window fit is not asymptotic.
# Sign convention: positive slope means increasing separation in this window.
lam_max, intercept = np.polyfit(time[mask], log_div[mask], 1)
if not np.isfinite(lam_max) or lam_max <= 0:
    raise ValueError("A finite positive window slope is required for this horizon illustration")

Delta_max = 10.0
pert_sizes = np.logspace(-12, -2, 20)
T_p = (1/lam_max)*np.log(Delta_max/pert_sizes)

print(f"[INFO] finite-window log-separation slope = {lam_max:.3f}")
print("  Reference ~0.9 is asymptotic; this fit is not a numerical PASS against it")
print(f"  T_p for eps=1e-9: {(1/lam_max)*np.log(Delta_max/1e-9):.1f} time units")

# RELIC relevance
print(f"\n  RELIC VariableKind.STATE = {VariableKind.STATE.value}")
print(f"  RELIC VariableKind.DIAGNOSTIC = {VariableKind.DIAGNOSTIC.value}")
print("  -> Derived separation metrics are DIAGNOSTIC; role labels do not certify chaos")

# =====================================================================
# FIGURE 1: Mathematical
# =====================================================================
apply_nature_style()

fig1 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH*0.38))
gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.4, width_ratios=[1.2, 1.0, 1.0])

ax = fig1.add_subplot(gs1[0, 0], projection="3d")
tb = traj1.numpy()
s = 5
ax.plot(tb[::s,0], tb[::s,1], tb[::s,2], color=CB_PALETTE[0], lw=0.3, alpha=0.7)
tp = traj2.numpy()
ax.plot(tp[::s,0], tp[::s,1], tp[::s,2], color=CB_PALETTE[1], lw=0.3, alpha=0.7)
ax.set_xlabel("x", fontsize=6, labelpad=-4)
ax.set_ylabel("y", fontsize=6, labelpad=-4)
ax.set_zlabel("z", fontsize=6, labelpad=-4)
ax.tick_params(labelsize=4, pad=-2)
ax.set_title("Lorenz Attractor", fontsize=9, pad=0)
fig1.text(0.02, 0.95, "a", fontsize=12, fontweight="bold", color="#222")

ax = fig1.add_subplot(gs1[0, 1])
ax.plot(time, log_div, color=CB_PALETTE[2], lw=0.8)
t_fit = np.linspace(1, 25, 100)
ax.plot(t_fit, lam_max*t_fit+intercept, "--", color=CB_PALETTE[1], lw=1.2,
        label=f"$\\lambda_{{fit}}={lam_max:.3f}$")
ax.set_xlabel("Time $t$")
ax.set_ylabel("$\\ln(\\|\\delta\\mathbf{x}\\|/\\|\\delta\\mathbf{x}_0\\|)$")
ax.set_title("Finite-window Separation", fontsize=9, pad=6)
ax.legend(fontsize=6, frameon=False, prop={"weight": "bold"})
ax.set_xlim(0, 40)
add_panel_label(ax, "b")

ax = fig1.add_subplot(gs1[0, 2])
ax.semilogx(pert_sizes, T_p, color=CB_PALETTE[0], lw=1.5)
ax.fill_between(pert_sizes, 0, T_p, alpha=0.08, color=CB_PALETTE[0])
ax.set_xlabel("$\\|\\delta\\mathbf{x}_0\\|$")
ax.set_ylabel("$T_p$ [time units]")
ax.set_title("Conditional Horizon", fontsize=9, pad=6)
add_panel_label(ax, "c")

fig1.suptitle("Lorenz Pair: Finite-window Divergence and Conditional Extrapolation",
              fontsize=10, y=1.03, fontweight="bold")
savefig(fig1, "14_math_lyapunov")
plt.close(fig1)

# =====================================================================
# FIGURE 2: RELIC context -- ACTUAL CHARTS (not text-only)
# =====================================================================
from tutorial.theory._common import FILL_PALETTE, add_legend_outside

fig2 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.38))
gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.45,
                        width_ratios=[1.0, 1.0, 1.0])

# --- Panel (a): actual trajectory-distance summaries ---
ax = fig2.add_subplot(gs2[0, 0])
# References: Lorenz (1963); the two RK4 trajectories computed above.
# Mathematical expression: d0=d(0), dmax=max(d), drms=sqrt(mean(d**2)).
# Parameter meanings: d is the Euclidean separation of the three state coordinates.
# Units: dimensionless Lorenz coordinates; no physical mass or energy budget is inferred.
# Domain/constraints: finite distances; logarithmic display requires positive values.
# Sign convention: distances are non-negative and carry no flux direction.
distance_metrics = [float(div[0]), float(div.max()), float(np.sqrt(np.mean(div**2)))]
assert np.isfinite(distance_metrics).all() and min(distance_metrics) > 0
ax.barh(range(3), distance_metrics, color=CB_PALETTE[:3], alpha=0.8)
ax.set_xscale("log")
ax.set_yticks(range(3))
ax.set_yticklabels(["Initial distance", "Maximum distance", "RMS distance"], fontsize=6)
ax.set_xlabel("Observed distance (log)", fontsize=7)
ax.set_title("Trajectory Pair Summaries", fontsize=8, pad=6)
add_panel_label(ax, "a", x=-0.25, y=1.05)

# --- Panel (b): Predictability horizon vs perturbation size ---
ax = fig2.add_subplot(gs2[0, 1])
eps_vals = np.logspace(-12, -2, 50)
T_p_curve = (1 / lam_max) * np.log(Delta_max / eps_vals)
ax.semilogx(eps_vals, T_p_curve, color=CB_PALETTE[0], lw=1.5)
ax.fill_between(eps_vals, 0, T_p_curve, alpha=0.12, color=CB_PALETTE[0])
# Mark illustrative perturbation sizes; neither line is machine epsilon.
ax.axvline(1e-9, color=CB_PALETTE[1], ls="--", lw=0.8)
ax.text(1.5e-9, T_p_curve.max() * 0.9, "$\\epsilon=10^{-9}$",
        fontsize=5.5, color=CB_PALETTE[1])
ax.axvline(1e-3, color=CB_PALETTE[3], ls="--", lw=0.8)
ax.text(1.5e-3, T_p_curve.max() * 0.7, "$\\epsilon=10^{-3}$",
        fontsize=5.5, color=CB_PALETTE[3])
ax.set_xlabel("Perturbation $\\|\\delta x_0\\|$", fontsize=7)
ax.set_ylabel("$T_p$ [time units]", fontsize=7)
ax.set_title("$T_p = \\frac{1}{\\lambda_{fit}} "
             "\\ln\\frac{\\Delta_{max}}{\\epsilon}$ (fit)", fontsize=8, pad=6)
add_panel_label(ax, "b", x=-0.18, y=1.05)

# --- Panel (c): declared roles in this standalone experiment ---
ax = fig2.add_subplot(gs2[0, 2])
vk_names = ["STATE", "DIAGNOSTIC", "FLUX", "FORCING"]
# Three state coordinates and three explicitly displayed derived quantities:
# separation, the finite-window slope, and the conditional horizon curve.
role_counts = [int(traj1.shape[1]), 3, 0, 0]
ax.bar(range(4), role_counts, color=CB_PALETTE[:4], alpha=0.8, width=0.55)
ax.set_xticks(range(4))
ax.set_xticklabels(vk_names, fontsize=6, rotation=25, ha="right")
ax.set_ylabel("Declared quantity count", fontsize=7)
ax.set_title("Demo Roles, Not Chaos Scores", fontsize=8, pad=6)
ax.set_ylim(0, 3.6)
add_panel_label(ax, "c", x=-0.18, y=1.05)

fig2.suptitle("Independent Lorenz Experiment: Observed Distances and Declared Roles",
              fontsize=10, y=1.03, fontweight="bold")
savefig(fig2, "14_relic_lyapunov")
plt.close(fig2)

print("\n[OK] Tutorial 14 completed successfully.")
print("  Fig 1: Lorenz attractor + finite-window separation fit")
print("  Fig 2: measured trajectory summaries and illustrative role declarations")
