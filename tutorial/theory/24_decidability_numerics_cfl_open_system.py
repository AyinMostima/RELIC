r"""Tutorial 24 -- Decidability Boundary, Floating-Point Non-Associativity, CFL Diagnostics, Open-System Closure.

Demonstrates
------------
This tutorial closes remaining coertheory gaps:
  - Floating-point non-associativity and reduction-order sensitivity
  - Rice-type decidability boundary (structural checks vs general semantic properties)
  - CFL and diffusion stability diagnostics (not applied by the RELIC step controller)
  - Open-system closure by adding environment node

Two figures are produced:
  1) 24_math_decidability_open_system.png
     Open-system closure residual and decidability boundary summary.
  2) 24_relic_numerics_cfl.png
     Floating-point order sensitivity and CFL/diffusion diagnostics from RELIC operators.

References
----------
1. docs/source/coertheory.md (Sections 1.3, 1.4, 3.4, 2.5.6)
2. Higham (2002). Accuracy and Stability of Numerical Algorithms
3. Rice (1953). Classes of recursively enumerable sets...
4. relic/computation/operators/functional.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE,
    DOUBLE_COL_WIDTH,
    add_panel_label,
    add_text_box,
    apply_nature_style,
    savefig,
)

from relic.computation.operators.functional import (
    compute_cfl_number,
    compute_diffusion_stability,
)


def _open_system_closure_demo() -> Dict[str, float]:
    """Compute budget residual before and after adding environment node."""
    # References:
    # - coertheory pushforward on open-system closure (Section 2.5.6)
    # Math expression:
    #   dS = B f
    # Parameter meaning:
    #   B: incidence matrix, f: edge flux vector.
    #   residual = sum(dS_internal)
    #
    # Open system (A,B only). Edge e2 leaves A to external world (no +1 endpoint).
    b_open = np.array([
        [-1.0, -1.0],  # A
        [1.0,  0.0],   # B
    ])
    f = np.array([5.0, 2.0])  # e1: A->B, e2: A->outside
    d_open = b_open @ f
    residual_open = float(d_open.sum())

    # Closed extension with environment node E.
    b_closed = np.array([
        [-1.0, -1.0],  # A
        [1.0,  0.0],   # B
        [0.0,  1.0],   # E
    ])
    d_closed = b_closed @ f
    residual_closed = float(d_closed.sum())
    return {
        "residual_open": residual_open,
        "residual_closed": residual_closed,
    }


def _floating_point_order_demo(seed: int = 7) -> Dict[str, float]:
    """Measure summation-order sensitivity in floating-point arithmetic."""
    # References:
    # - coertheory Section 1.3
    # - Higham (2002)
    # Math expression:
    #   fl(sum x_i) depends on reduction order in finite precision.
    # Parameter meaning:
    #   x_i: input sequence with mixed magnitudes.
    rng = np.random.default_rng(seed)
    small = rng.normal(0.0, 1e-9, size=60000)
    large = rng.normal(0.0, 1e3, size=3000)
    x = np.concatenate([small, large]).astype(np.float64)

    sum_original = float(np.sum(x))
    sum_asc = float(np.sum(x[np.argsort(np.abs(x))]))
    sum_desc = float(np.sum(x[np.argsort(np.abs(x))[::-1]]))

    # Simulate two-level parallel reduction.
    chunks = np.array_split(x, 32)
    partials = np.array([np.sum(chunk) for chunk in chunks], dtype=np.float64)
    sum_parallel_like = float(np.sum(partials))

    return {
        "original": sum_original,
        "ascending": sum_asc,
        "descending": sum_desc,
        "parallel_like": sum_parallel_like,
        "max_diff": max(
            abs(sum_original - sum_asc),
            abs(sum_original - sum_desc),
            abs(sum_original - sum_parallel_like),
        ),
    }


def _cfl_diffusion_gating_demo() -> Dict[str, np.ndarray]:
    """Compute CFL and diffusion stability metrics over dt scan."""
    # References:
    # - coertheory Section 3.4 stability gate
    # - relic/computation/operators/functional.py
    # Math expression:
    #   CFL = |v| dt / dx <= 1
    #   sigma = D dt / dx^2 <= 0.5
    # Parameter meaning:
    #   v: velocity, D: diffusivity, dt: time step, dx: grid spacing.
    velocity = torch.tensor([1.8, 2.1, 1.6, 2.4], dtype=torch.float32)  # m s^-1
    diffusivity = torch.tensor([0.55, 0.70, 0.62, 0.58], dtype=torch.float32)  # m2 s^-1
    dx = 10.0  # m
    dt_values = np.linspace(1.0, 20.0, 40)

    max_cfl: List[float] = []
    max_sigma: List[float] = []
    stable_flag: List[float] = []
    for dt in dt_values:
        cfl = compute_cfl_number(velocity, dx, float(dt))
        sigma = compute_diffusion_stability(diffusivity, dx, float(dt))
        cfl_max = float(torch.max(cfl).item())
        sigma_max = float(torch.max(sigma).item())
        max_cfl.append(cfl_max)
        max_sigma.append(sigma_max)
        stable_flag.append(1.0 if (cfl_max <= 1.0 and sigma_max <= 0.5) else 0.0)

    return {
        "dt_values": dt_values,
        "max_cfl": np.array(max_cfl),
        "max_sigma": np.array(max_sigma),
        "stable_flag": np.array(stable_flag),
    }


def main() -> None:
    print("=== Tutorial 24: Decidability and numerical stability boundaries ===")

    closure = _open_system_closure_demo()
    fp = _floating_point_order_demo()
    gate = _cfl_diffusion_gating_demo()

    print(f"  Open-system residual before closure: {closure['residual_open']:.6f}")
    print(f"  Residual after adding environment node: {closure['residual_closed']:.6f}")
    print(f"  Floating-point max order difference: {fp['max_diff']:.6e}")
    print(
        f"  Stable dt count (CFL<=1 and sigma<=0.5): "
        f"{int(np.sum(gate['stable_flag']))}/{len(gate['stable_flag'])}"
    )

    # ------------------------------------------------------------------
    # Figure 1: pure mathematics (open-system closure + decidability map)
    # ------------------------------------------------------------------
    apply_nature_style()
    fig1, axes1 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    ax = axes1[0]
    residual_vals = [closure["residual_open"], closure["residual_closed"]]
    labels = ["open graph", "with env node"]
    ax.bar(labels, residual_vals, color=[CB_PALETTE[1], CB_PALETTE[2]], edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Budget residual sum(dS)")
    ax.set_title("Open-System Closure via Environment Node", fontsize=9)
    add_panel_label(ax, "a")

    # Panel b: decidable vs undecidable property classification
    ax = axes1[1]
    property_names = ["DAG\nacyclicity", "Unit\nconvert.", "Port-type\ncompat.", "Semantic\nhalting"]
    property_decidable = [1, 1, 1, 0]  # 1=decidable, 0=undecidable
    bar_colors_d = [CB_PALETTE[2] if d else CB_PALETTE[1] for d in property_decidable]
    ax.barh(property_names, property_decidable, color=bar_colors_d, alpha=0.85, height=0.6)
    # Add annotations
    for i, (name, dec) in enumerate(zip(property_names, property_decidable)):
        label = "Decidable" if dec else "Undecidable"
        ax.text(0.5 if dec else 0.02, i, label, va="center", ha="center" if dec else "left",
                fontsize=7, fontweight="bold", color="white" if dec else "black")
    ax.set_xlim(-0.1, 1.2)
    ax.set_xlabel("Decidability (1=yes)")
    ax.set_title("Rice-type Decidability Boundary", fontsize=9)
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig1.suptitle("Pure Mathematics: Decidability Boundary and Open-System Closure", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig1, "24_math_decidability_open_system")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: RELIC empirical numerics (floating-point + CFL gate)
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    ax = axes2[0]
    method_names = ["original", "ascending", "descending", "parallel_like"]
    method_vals = [fp[name] for name in method_names]
    ax.bar(method_names, method_vals, color=CB_PALETTE[:4], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Summation result")
    ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    ax.set_title("Floating-Point Reduction Order Sensitivity", fontsize=9)
    add_text_box(ax, f"max diff = {fp['max_diff']:.3e}", loc="upper right", fontsize=7)
    add_panel_label(ax, "a")

    ax = axes2[1]
    dt_values = gate["dt_values"]
    ax.plot(dt_values, gate["max_cfl"], color=CB_PALETTE[0], linewidth=1.4, label="max CFL")
    ax.plot(dt_values, gate["max_sigma"], color=CB_PALETTE[2], linewidth=1.4, label="max sigma")
    ax.axhline(1.0, color=CB_PALETTE[0], linestyle="--", linewidth=1.0, label="CFL limit")
    ax.axhline(0.5, color=CB_PALETTE[2], linestyle="--", linewidth=1.0, label="sigma limit")
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("Stability metrics")
    ax.set_title("CFL and Diffusion Diagnostics", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", frameon=False, prop={"weight": "bold"})
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig2.suptitle("RELIC Operators: Floating-Point Order and CFL/Diffusion Diagnostics", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig2, "24_relic_numerics_cfl")
    plt.close(fig2)

    print("[OK] Tutorial 24 finished.")


if __name__ == "__main__":
    main()
