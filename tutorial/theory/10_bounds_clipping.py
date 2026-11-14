r"""Tutorial 10 -- Bounds Clipping and Clip-Loss Diagnostics (I7).

Demonstrates
------------
RELIC enforces physical bounds via projection P_{[L,U]} and records
the clip-loss diagnostic for auditability (Invariant I7).

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Bounds projection (coertheory.md Theorem 3.6.3):
    x^{n+1} = clamp(x^n + Sum Delta x_i, L, U)        ... (1)

Clip-loss diagnostic:
    Delta x_clip = x_unclamped - x^{n+1}                ... (2, signed)
    clip_loss = |Delta x_clip|                          ... (3)

Cumulative clip-loss serves as a proxy for numerical error magnitude.

References
----------
1. coertheory.md, Theorem 3.6.3
2. spatiotemporal_scale_topology.md, Section 8.2 (Invariant I7)
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
# 1. Simulate decomposition with large dt (causing negative pools)
# -----------------------------------------------------------------------
# Reference: Parton et al. (1987), first-order decay model
# Math: dC/dt = -k * C,  analytical: C(t) = C0 * exp(-k*t)
# With Forward Euler at large dt: C_{n+1} = C_n * (1 - k*dt)
# When k*dt > 1, C goes negative -> clipping needed

C0 = 100.0        # Initial soil carbon [g C m^-2]
k = 0.15           # Decomposition rate [step^-1]
L, U = 0.0, 1e6   # Physical bounds

n_steps = 100
dt = 1.0           # large enough that k*dt = 0.15 (safe initially)

# Analytical solution
t_exact = np.linspace(0, n_steps * dt, 500)
C_exact = C0 * np.exp(-k * t_exact)

# --- Scenario 1: Without clipping ---
C_nocl = np.zeros(n_steps + 1)
C_nocl[0] = C0
for i in range(n_steps):
    # Forward Euler: C_{n+1} = C_n + dt * (-k * C_n) = C_n * (1 - k*dt)
    C_nocl[i + 1] = C_nocl[i] * (1 - k * dt)

# --- Scenario 2: With clipping + clip_loss tracking ---
# Using aggressive dt to force clipping
dt_aggressive = 8.0  # k * dt_aggressive = 1.2 > 1 -> violates non-negativity
n_steps_agg = 40

C_clipped = np.zeros(n_steps_agg + 1)
C_unclip = np.zeros(n_steps_agg + 1)
clip_loss = np.zeros(n_steps_agg)
C_clipped[0] = C0
C_unclip[0] = C0

for i in range(n_steps_agg):
    proposed = C_clipped[i] * (1 - k * dt_aggressive)
    C_unclip[i + 1] = proposed
    # Apply bounds projection P_{[L,U]}
    # Reference: coertheory.md Theorem 3.6.3
    # Math: x^{n+1} = clamp(x_proposed, L, U)
    clamped = np.clip(proposed, L, U)
    delta_clip = proposed - clamped
    clip_loss[i] = abs(delta_clip)
    C_clipped[i + 1] = clamped

time_agg = np.arange(n_steps_agg + 1) * dt_aggressive
time_steps = np.arange(n_steps + 1) * dt
cum_clip = np.cumsum(clip_loss)

n_clipped = np.sum(clip_loss > 0)
print(f"[OK] Aggressive dt={dt_aggressive}: {n_clipped} steps required clipping")
print(f"  Total clip-loss = {cum_clip[-1]:.2f} g C m^-2")
print(f"  k*dt = {k*dt_aggressive:.1f} (>1 -> violates non-negativity)")

# -----------------------------------------------------------------------
# 2. Plot: 3-panel figure
# -----------------------------------------------------------------------
apply_nature_style()

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.33))

# --- Panel (a): Unclamped vs clamped trajectories ---
ax = axes[0]
ax.plot(t_exact, C_exact, "k-", linewidth=1.5, label="Exact solution", alpha=0.7)
ax.plot(time_agg, C_unclip, "o--", color=CB_PALETTE[1], markersize=3,
        linewidth=0.8, label="Unclamped ($k\\Delta t > 1$)")
ax.plot(time_agg, C_clipped, "s-", color=CB_PALETTE[2], markersize=3,
        linewidth=0.8, label="Clamped $P_{[0,\\infty)}$")
ax.axhline(0, color="red", linewidth=0.5, linestyle=":")
ax.set_xlabel("Time")
ax.set_ylabel("Soil Carbon [g C m$^{-2}$]")
ax.set_title("Bounds Projection Effect", fontsize=9)
ax.legend(fontsize=6, frameon=False, prop={"weight": "bold"})
ax.set_ylim(-50, 120)
add_panel_label(ax, "a")

# --- Panel (b): Clip-loss per step ---
ax = axes[1]
colors = [CB_PALETTE[1] if cl > 0 else CB_PALETTE[5] for cl in clip_loss]
ax.bar(range(n_steps_agg), clip_loss, color=colors, edgecolor="none", width=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("Clip-Loss [g C m$^{-2}$]")
ax.set_title("Per-Step Clip-Loss Diagnostic", fontsize=9)
# Annotate clipping events
if n_clipped > 0:
    first_clip = np.where(clip_loss > 0)[0][0]
    ax.annotate(f"First clip at step {first_clip}",
                xy=(first_clip, clip_loss[first_clip]),
                xytext=(first_clip + 5, clip_loss[first_clip] * 0.8),
                fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.7))
add_panel_label(ax, "b")

# --- Panel (c): Cumulative clip-loss ---
ax = axes[2]
ax.plot(range(n_steps_agg), cum_clip, "-", color=CB_PALETTE[3], linewidth=1.5)
ax.fill_between(range(n_steps_agg), 0, cum_clip, alpha=0.15, color=CB_PALETTE[3])
ax.set_xlabel("Step")
ax.set_ylabel("Cumulative Clip-Loss [g C m$^{-2}$]")
ax.set_title("Accumulated Numerical Error Proxy", fontsize=9)
ax.text(0.5, 0.85, f"Total: {cum_clip[-1]:.1f} g C m$^{{-2}}$",
        transform=ax.transAxes, fontsize=7, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ccc"))
add_panel_label(ax, "c")

fig.suptitle(
    "Bounds Clipping (I7): $x^{n+1} = P_{[L,U]}(x^n + \\sum \\Delta x_i)$, "
    "clip\\_loss = $|x_{uncl} - x_{cl}|$",
    fontsize=9, y=1.05,
)
plt.tight_layout()
savefig(fig, "10_bounds_clipping")
plt.close(fig)

print("\n[OK] Tutorial 10 completed successfully.")
print("  Demonstrated: I7 (bounds clipping with clip_loss diagnostic)")
