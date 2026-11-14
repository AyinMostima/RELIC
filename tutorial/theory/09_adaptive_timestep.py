r"""Tutorial 09 -- Adaptive Time Stepping with Error Control.

Demonstrates
------------
This demo uses a safety-factor (I controller) rule with a step-doubling
error estimate and retries rejected steps until accepted. RELIC's runtime
rule differs: Err measures the StateManager committed change over the fast
group, including rate integration, clipping and reduction. A stored steady
boundary rate has zero change. A rejected attempt longer than dt_min retries with
dt*max(0.1, 0.9/sqrt(Err)) (floored at dt_min, always shorter than the
attempt) and that retry is accepted without a second check, while an
attempt at or below dt_min is accepted without rollback with a WARNING;
accepted steps
set the next dt to dt*min(5, max(0.1, 0.9/sqrt(Err))) (x2 if Err == 0),
clipped to [dt_min, dt_max]. This script does not call the RELIC controller.
Note: both rules are I-controller style (they depend on the current Err
only). A true PI controller would incorporate the previous-step error as well.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Error norm (spatiotemporal_scale_topology.md Def. 7.1):

    Err = sqrt(1/M * Sum_k (delta_k / tol_k)^2)     ... (1)
    tol_k = atol + rtol * |y_k|                       ... (2)

Step size control (Theorem 7.1):
    dt_new = dt * min(f_max, max(f_min, f_s / Err^{1/(p+1)}))  ... (3)

where f_s = 0.9 (safety), f_min = 0.1, f_max = 5.0.

References
----------
1. spatiotemporal_scale_topology.md, Section 7.1-7.3
2. Hairer, E. & Wanner, G. (1996). Solving ODEs II. Springer.
3. coertheory.md, Theorem 3.6.8 (adaptive step with rollback)
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
    CB_PALETTE, DOUBLE_COL_WIDTH, add_panel_label,
    apply_nature_style, savefig,
)

# -----------------------------------------------------------------------
# 1. Stiff ODE system (fast + slow dynamics)
# -----------------------------------------------------------------------
# Reference: Hairer & Wanner (1996), Ch. IV
# Math:
#   dy1/dt = -100*(y1 - cos(t))   (fast, stiffness ratio ~100)
#   dy2/dt = -0.5 * y2 + sin(t)   (slow)
#
# Exact solution for y1 (quasi-steady): y1 ~ cos(t) + sin(t)/100
# Exact solution for y2: exponential decay + forced oscillation


def rhs(y, t):
    """Right-hand side of the stiff ODE system."""
    dy1 = -100.0 * (y[0] - np.cos(t))
    dy2 = -0.5 * y[1] + np.sin(t)
    return np.array([dy1, dy2])


def euler_step(y, t, dt):
    """Forward Euler step: y_{n+1} = y_n + dt * f(y_n, t_n)."""
    return y + dt * rhs(y, t)


def compute_error_norm(y_full, y_half, atol=1e-4, rtol=1e-3):
    """Compute error norm using Richardson extrapolation.

    Reference: spatiotemporal_scale_topology.md Def. 7.1
    Math: Err = sqrt(1/M * Sum (delta_k / tol_k)^2)
    where delta_k = |y_full - y_half| (error estimate)
          tol_k = atol + rtol * |y_k|
    """
    delta = np.abs(y_full - y_half)
    tol = atol + rtol * np.abs(y_half)
    return np.sqrt(np.mean((delta / tol) ** 2))


# -----------------------------------------------------------------------
# 2. Adaptive integration with I controller
# -----------------------------------------------------------------------
# Reference: spatiotemporal_scale_topology.md Theorem 7.1
# Math: dt_new = dt * min(f_max, max(f_min, f_s / Err^{1/(p+1)}))

y0 = np.array([0.5, 1.0])
T_final = 10.0
dt_init = 0.1
dt_min, dt_max = 1e-6, 1.0
f_s = 0.9       # safety factor
f_min = 0.1
f_max = 5.0
p_order = 1     # Euler is first-order

# Storage
t_hist = [0.0]
y_hist = [y0.copy()]
dt_hist = []
err_hist = []
rejected = []

t = 0.0
y = y0.copy()
dt = dt_init

while t < T_final:
    dt = min(dt, T_final - t)

    # Full step
    y_full = euler_step(y, t, dt)

    # Two half steps (for error estimation)
    y_mid = euler_step(y, t, dt / 2)
    y_half = euler_step(y_mid, t + dt / 2, dt / 2)

    err = compute_error_norm(y_full, y_half)
    dt_hist.append(dt)
    err_hist.append(err)

    if err > 1.0:
        # Reject: compute retry dt
        # Demo rule (no 0.1 floor, unlimited retries); RELIC runtime retries once
        # with dt_retry = max(dt_min, dt * max(0.1, 0.9 / sqrt(Err))) only when
        # dt > dt_min, and accepts an attempt at or below dt_min with a WARNING.
        rejected.append(t)
        dt = max(dt_min, dt * f_s / np.sqrt(err))
    else:
        # Accept: advance and predict next dt
        # Math: dt_new = dt * min(f_max, max(f_min, f_s / Err^{1/(p+1)}))
        y = y_half  # use more accurate half-step result
        t += dt_hist[-1]
        t_hist.append(t)
        y_hist.append(y.copy())

        if err > 0:
            dt = dt_hist[-1] * min(f_max, max(f_min, f_s / err ** (1.0 / (p_order + 1))))
        else:
            dt = dt_hist[-1] * f_max
        dt = np.clip(dt, dt_min, dt_max)

y_hist = np.array(y_hist)
t_hist = np.array(t_hist)

# Fixed-step comparison
n_fixed = 200
dt_fixed = T_final / n_fixed
t_fixed = np.linspace(0, T_final, n_fixed + 1)
y_fixed = np.zeros((n_fixed + 1, 2))
y_fixed[0] = y0
for i in range(n_fixed):
    y_fixed[i + 1] = euler_step(y_fixed[i], t_fixed[i], dt_fixed)

# Exact-ish reference (very small dt)
n_ref = 10000
dt_ref = T_final / n_ref
t_ref = np.linspace(0, T_final, n_ref + 1)
y_ref = np.zeros((n_ref + 1, 2))
y_ref[0] = y0
for i in range(n_ref):
    y_ref[i + 1] = euler_step(y_ref[i], t_ref[i], dt_ref)

print(f"[OK] Adaptive stepping: {len(t_hist)-1} accepted steps, {len(rejected)} rejected")
print(f"  dt range: [{min(dt_hist):.6f}, {max(dt_hist):.4f}]")

# -----------------------------------------------------------------------
# 3. Plot: 3-panel figure
# -----------------------------------------------------------------------
apply_nature_style()

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.33))

# --- Panel (a): Step size evolution ---
ax = axes[0]
accepted_times = [dt_hist[i] for i in range(len(dt_hist)) if i < len(err_hist) and err_hist[i] <= 1.0]
ax.semilogy(range(len(dt_hist)), dt_hist, "-", color=CB_PALETTE[0],
            linewidth=0.8, alpha=0.7)
for rt in rejected:
    idx = len([t for t in t_hist if t <= rt]) - 1
    if 0 <= idx < len(dt_hist):
        ax.semilogy(idx, dt_hist[idx] if idx < len(dt_hist) else dt_hist[-1],
                    "x", color="red", markersize=5, zorder=5)

ax.set_xlabel("Step Index")
ax.set_ylabel("$\\Delta t$")
ax.set_title("Step Size Evolution", fontsize=9)
# Add rejection marker to legend
ax.plot([], [], "x", color="red", markersize=5, label="Rejected")
ax.plot([], [], "-", color=CB_PALETTE[0], label="Accepted $\\Delta t$")
ax.legend(fontsize=6, frameon=False, prop={"weight": "bold"})
add_panel_label(ax, "a")

# --- Panel (b): Error norm ---
ax = axes[1]
ax.semilogy(range(len(err_hist)), err_hist, ".", color=CB_PALETTE[2],
            markersize=2, alpha=0.6)
ax.axhline(1.0, color="red", linewidth=1.0, linestyle="--",
           label="Acceptance threshold")
ax.set_xlabel("Step Index")
ax.set_ylabel("Error Norm $\\mathcal{E}$")
ax.set_title("Error Norm per Step", fontsize=9)
ax.legend(fontsize=6, frameon=False, prop={"weight": "bold"})
add_panel_label(ax, "b")

# --- Panel (c): Solution comparison ---
ax = axes[2]
ax.plot(t_ref, y_ref[:, 0], "k-", linewidth=1.0, alpha=0.8, label="Reference $y_1$")
ax.plot(t_hist, y_hist[:, 0], "o-", color=CB_PALETTE[0], markersize=2,
        linewidth=0.8, label="Adaptive $y_1$")
# Forward Euler with dt = 0.05 violates |1 - 100 dt| < 1 for y1 and diverges;
# values outside the reference range are hidden so the adaptive result stays visible.
y_pad = 0.25 * float(np.ptp(y_ref[:, 0]))
y_lo = float(y_ref[:, 0].min()) - y_pad
y_hi = float(y_ref[:, 0].max()) + y_pad
y_fixed_shown = np.where((y_fixed[:, 0] >= y_lo) & (y_fixed[:, 0] <= y_hi), y_fixed[:, 0], np.nan)
ax.plot(t_fixed, y_fixed_shown, "--", color=CB_PALETTE[1], linewidth=0.8,
        alpha=0.7, label=f"Fixed ($\\Delta t={dt_fixed:.3f}$, diverges) $y_1$")
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel("Time")
ax.set_ylabel("$y_1$ (fast component)")
ax.set_title("Solution Accuracy", fontsize=9)
ax.legend(fontsize=6, loc="lower right", frameon=False, prop={"weight": "bold"})
add_panel_label(ax, "c")

fig.suptitle(
    "Adaptive Time Stepping Demo: I Controller, Retry Until Accepted (RELIC runtime retries once)",
    fontsize=10, y=1.03,
)
plt.tight_layout()
savefig(fig, "09_adaptive_timestep")
plt.close(fig)

print("\n[OK] Tutorial 09 completed successfully.")
print("  Demonstrated: adaptive I controller with error norm and rollback")
