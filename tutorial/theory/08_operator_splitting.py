r"""Tutorial 08 -- Operator Splitting: Lie vs Strang.

Demonstrates
------------
RELIC supports Lie and Strang operator splitting (Strang for the fast group
only). With exact sub-flows, as in this demo (matrix exponentials), Lie is
1st order and Strang 2nd order. In RELIC every Strang sub-execution receives
the interval it covers (first half-steps and the middle step start at t, the
second half-steps at t + dt/2), so non-autonomous sub-flows get the correct
stage times. With numerical sub-integrators the effective order of one step
is q = min(s, min_i p_i) (s = 1 Lie, s = 2 Strang; p_i from
FunctorMetadata.sub_flow_accuracy, "unverified" if any Functor is
UNDECLARED), reported by RELICSimulator.get_splitting_order_report(). The
convergence order is verified by measuring the error vs dt on a log-log plot.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Lie splitting (Definition 6.1, spatiotemporal_scale_topology.md):
    Phi_Lie^{dt} = Phi_{A2}^{dt} o Phi_{A1}^{dt}   (A1 applied first)
    Local error O(dt^2); for linear operators with exact sub-flows
    exp(dt*A_k): log(Phi_Lie) = dt*(A1 + A2) + dt^2/2 * [A2, A1] + O(dt^3)

Strang splitting (Definition 6.2):
    Phi_Strang^{dt} = Phi_{A1}^{dt/2} o Phi_{A2}^{dt} o Phi_{A1}^{dt/2}
    Local error O(dt^3); for linear operators with exact sub-flows:
    log(Phi_Strang) = dt*(A1 + A2)
        + dt^3/24 * (2 [A2,[A2,A1]] - [A1,[A1,A2]]) + O(dt^5)

Commutator (Lie bracket):
    [A, B] = AB - BA

References
----------
1. spatiotemporal_scale_topology.md, Section 6.1-6.2
2. Sportisse, B. (2000). An analysis of operator splitting techniques.
   J. Comp. Phys. 161(1), 140-168.
3. Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of
   Time-Dependent Advection-Diffusion-Reaction Equations. Springer.
4. Strang, G. (1968). On the construction and comparison of difference
   schemes. SIAM J. Numer. Anal. 5(3), 506-517.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    add_legend_outside,
    CB_PALETTE, DOUBLE_COL_WIDTH, add_panel_label,
    apply_nature_style, savefig,
)

# -----------------------------------------------------------------------
# 1. Define coupled linear ODE: du/dt = (A1 + A2) u
# -----------------------------------------------------------------------
# Two-component system representing coupled processes:
#   Process A1 (e.g., source/sink):  du/dt = A1 u
#   Process A2 (e.g., transport):    du/dt = A2 u
#
# Reference: Hundsdorfer & Verwer (2003), Section 1.2
# Math:
#   A1 = [[-a,  b], [a, -b]]   (reaction)
#   A2 = [[-c,  0], [0, -d]]   (decay)

a, b, c, d = 0.5, 0.3, 0.1, 0.2

A1 = np.array([[-a, b], [a, -b]])
A2 = np.array([[-c, 0.0], [0.0, -d]])
A_total = A1 + A2

# Exact solution: u(t) = expm((A1+A2)*t) @ u0
# Reference: Matrix exponential for linear ODE, Strang (1988)
u0 = np.array([1.0, 0.5])
T_final = 5.0


def exact_solution(t, u0):
    """Exact solution via matrix exponential.

    Math: u(t) = exp((A1+A2)*t) * u0
    """
    return expm(A_total * t) @ u0


def lie_step(u, dt):
    """Lie splitting: Phi_{A2}^{dt} o Phi_{A1}^{dt}.

    Reference: spatiotemporal_scale_topology.md Def. 6.1
    Math: u -> exp(A1*dt) u -> exp(A2*dt) (exp(A1*dt) u)
    """
    u1 = expm(A1 * dt) @ u
    u2 = expm(A2 * dt) @ u1
    return u2


def strang_step(u, dt):
    """Strang splitting: Phi_{A1}^{dt/2} o Phi_{A2}^{dt} o Phi_{A1}^{dt/2}.

    Reference: spatiotemporal_scale_topology.md Def. 6.2
    Math: u -> exp(A1*dt/2) u -> exp(A2*dt) ... -> exp(A1*dt/2) ...
    """
    u1 = expm(A1 * dt / 2) @ u
    u2 = expm(A2 * dt) @ u1
    u3 = expm(A1 * dt / 2) @ u2
    return u3


def simulate_split(step_func, dt):
    """Run simulation with given splitting step function."""
    n_steps = int(T_final / dt)
    u = u0.copy()
    trajectory = [u.copy()]
    for _ in range(n_steps):
        u = step_func(u, dt)
        trajectory.append(u.copy())
    return np.array(trajectory)


# -----------------------------------------------------------------------
# 2. Convergence study: error vs dt
# -----------------------------------------------------------------------
dt_values = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
u_exact_final = exact_solution(T_final, u0)

lie_errors = []
strang_errors = []

for dt in dt_values:
    traj_lie = simulate_split(lie_step, dt)
    traj_strang = simulate_split(strang_step, dt)
    lie_errors.append(np.linalg.norm(traj_lie[-1] - u_exact_final))
    strang_errors.append(np.linalg.norm(traj_strang[-1] - u_exact_final))

lie_errors = np.array(lie_errors)
strang_errors = np.array(strang_errors)

# Measure convergence rates
# Reference: linear regression on log-log scale gives order p
lie_rate = np.polyfit(np.log(dt_values[:4]), np.log(lie_errors[:4] + 1e-20), 1)[0]
strang_rate = np.polyfit(np.log(dt_values[:4]), np.log(strang_errors[:4] + 1e-20), 1)[0]
print(f"[OK] Lie splitting convergence rate: {lie_rate:.2f} (expected ~1.0)")
print(f"[OK] Strang splitting convergence rate: {strang_rate:.2f} (expected ~2.0)")

# Compute commutator norm
# Reference: spatiotemporal_scale_topology.md Eq. for e_Lie
# Math: [A1, A2] = A1*A2 - A2*A1
commutator = A1 @ A2 - A2 @ A1
comm_norm = np.linalg.norm(commutator, "fro")
print(f"  Commutator ||[A1,A2]||_F = {comm_norm:.4f}")

# -----------------------------------------------------------------------
# 3. Full trajectory comparison at dt = 1.0
# -----------------------------------------------------------------------
dt_demo = 1.0
n_demo = int(T_final / dt_demo)
t_demo = np.linspace(0, T_final, 200)
t_split = np.arange(n_demo + 1) * dt_demo

exact_traj = np.array([exact_solution(t, u0) for t in t_demo])
lie_traj = simulate_split(lie_step, dt_demo)
strang_traj = simulate_split(strang_step, dt_demo)

# -----------------------------------------------------------------------
# 4. Plot: 3-panel figure
# -----------------------------------------------------------------------
apply_nature_style()

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.33))

# --- Panel (a): Solution trajectories ---
# Only plot u1 for Lie/Strang per documentation spec
ax = axes[0]
ax.plot(t_demo, exact_traj[:, 0], "k-", linewidth=1.5, label="Exact $u_1$")
ax.plot(t_demo, exact_traj[:, 1], "k--", linewidth=1.5, label="Exact $u_2$")
ax.plot(t_split, lie_traj[:, 0], "o-", color=CB_PALETTE[0], markersize=4,
        linewidth=0.8, label="Lie $u_1$")
ax.plot(t_split, strang_traj[:, 0], "s-", color=CB_PALETTE[1], markersize=4,
        linewidth=0.8, label="Strang $u_1$")
ax.set_xlabel("Time")
ax.set_ylabel("State")
ax.set_title("Solution Trajectories ($\\Delta t = 1.0$)", fontsize=9)
ax.legend(fontsize=5, ncol=2, loc="upper right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
add_panel_label(ax, "a")

# --- Panel (b): Convergence plot (log-log) ---
ax = axes[1]
ax.loglog(dt_values, lie_errors, "o-", color=CB_PALETTE[0],
          linewidth=1.2, markersize=4, label="Lie splitting")
ax.loglog(dt_values, strang_errors, "s-", color=CB_PALETTE[1],
          linewidth=1.2, markersize=4, label="Strang splitting")

# Reference slopes
dt_ref = np.array([dt_values[0], dt_values[-1]])
ax.loglog(dt_ref, lie_errors[0] * (dt_ref / dt_ref[0])**1,
          "--", color=CB_PALETTE[0], alpha=0.4, linewidth=0.8,
          label="$O(\\Delta t)$")
ax.loglog(dt_ref, strang_errors[0] * (dt_ref / dt_ref[0])**2,
          "--", color=CB_PALETTE[1], alpha=0.4, linewidth=0.8,
          label="$O(\\Delta t^2)$")
ax.set_xlabel("$\\Delta t$")
ax.set_ylabel("Error $\\|u_{split} - u_{exact}\\|$")
ax.set_title("Convergence Order", fontsize=9)
ax.legend(fontsize=5, ncol=2, loc="upper right", frameon=False,
          prop={"weight": "bold"}, borderaxespad=0.3)
ax.grid(True, alpha=0.2, which="both")
add_panel_label(ax, "b")

# --- Panel (c): Commutator information ---
ax = axes[2]
ax.axis("off")
text = (
    "Operator Splitting Theory\n"
    "========================\n\n"
    f"$A_1$ (reaction):\n"
    f"  $[[-{a:.1f},  {b:.1f}], [{a:.1f}, -{b:.1f}]]$\n\n"
    f"$A_2$ (decay):\n"
    f"  $[[-{c:.1f},  0], [0, -{d:.1f}]]$\n\n"
    f"Commutator $\\|[A_1, A_2]\\|_F = {comm_norm:.4f}$\n\n"
    f"Lie rate: {lie_rate:.2f}  (theory: 1.0)\n"
    f"Strang rate: {strang_rate:.2f}  (theory: 2.0)\n\n"
    "Leading error terms (exact sub-flows,\n"
    "$A_1$ first / on the half steps):\n"
    "  Lie:   $\\frac{{\\Delta t^2}}{{2}} [A_2, A_1]$\n"
    "  Strang: $\\frac{{\\Delta t^3}}{{24}}(2[A_2,[A_2,A_1]]$\n"
    "          $-\\,[A_1,[A_1,A_2]])$"
)
ax.text(0.05, 0.95, text, transform=ax.transAxes,
        fontsize=6.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                  edgecolor="#ddd"))
ax.set_title("Summary", fontsize=9)
add_panel_label(ax, "c")

fig.suptitle(
    "Operator Splitting with Exact Sub-flows: Lie $O(\\Delta t)$ vs Strang $O(\\Delta t^2)$",
    fontsize=10, y=1.05,
)
plt.tight_layout()
savefig(fig, "08_operator_splitting")
plt.close(fig)

print("\n[OK] Tutorial 08 completed successfully.")
print(f"  Demonstrated: Lie (rate={lie_rate:.2f}) vs Strang (rate={strang_rate:.2f}) convergence")
