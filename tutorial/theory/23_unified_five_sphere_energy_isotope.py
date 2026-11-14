r"""Tutorial 23 -- Synthetic Five-Compartment Amount and Tracer Bookkeeping.

The sphere names label an illustrative closed transfer graph. All edges carry
one generic quantity in the same arbitrary amount unit per step; they do not
mix measured carbon, water, nutrients, or energy into a physical ledger.
T is already a per-step amount and is never multiplied by an extra dt.

The separate prescribed energy identity uses W m-2. The tracer fractions are
amount fractions, not heavy/light isotope ratios; S*f is a tracer inventory.
This script runs no RELIC Functor, simulator, isotope runtime or invariant test.
Its two figures show newly computed arithmetic for explicit synthetic inputs,
not empirical validation or measured comparisons with Earth-system models.

References: the existing unified theoretical framework, Sections 2 and 5;
Coplen (2011), Pure Appl. Chem. 83:1873-1935 for isotope quantity definitions;
Higham (2002), Accuracy and Stability of Numerical Algorithms for roundoff.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

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


def _build_transfer_matrix() -> Tuple[List[str], np.ndarray]:
    """Create synthetic per-step amounts T[i,j] directed from i to j."""
    # References: unified framework Sections 2.1-2.2; synthetic teaching fixture.
    # Mathematical expression: Delta S_i = sum_j T[j,i] - sum_j T[i,j].
    # Parameter meanings: T[i,j] is an amount moved once in the illustrative step.
    # Units: every T entry uses the same generic arbitrary amount unit (a.u.).
    # Domain/constraints: non-negative finite amounts, zero diagonal; not observed sphere fluxes.
    # Sign convention: positive T[i,j] removes from i and adds to j.
    spheres = ["Atmosphere", "Biosphere", "Hydrosphere", "Pedosphere", "Anthroposphere"]
    t = np.zeros((5, 5), dtype=float)

    # Assigned group A (fast label only; no inferred timescale)
    t[0, 1] = 3.2   # Atmosphere -> Biosphere (synthetic generic amount)
    t[1, 0] = 2.4   # Biosphere -> Atmosphere (synthetic generic amount)
    t[0, 2] = 1.8   # Atmosphere -> Hydrosphere (synthetic generic amount)
    t[2, 0] = 1.5   # Hydrosphere -> Atmosphere (synthetic generic amount)

    # Assigned group B (medium label only; no inferred timescale)
    t[1, 3] = 1.6   # Biosphere -> Pedosphere (synthetic generic amount)
    t[3, 1] = 0.7   # Pedosphere -> Biosphere (synthetic generic amount)
    t[2, 3] = 1.2   # Hydrosphere -> Pedosphere (synthetic generic amount)
    t[3, 2] = 0.9   # Pedosphere -> Hydrosphere (synthetic generic amount)

    # Assigned group C (slow label only; no inferred timescale)
    t[4, 0] = 1.1   # Anthroposphere -> Atmosphere (synthetic generic amount)
    t[4, 3] = 0.8   # Anthroposphere -> Pedosphere (synthetic generic amount)
    t[1, 4] = 0.4   # Biosphere -> Anthroposphere (synthetic generic amount)
    return spheres, t


def _compute_storage_update(transfer: np.ndarray) -> np.ndarray:
    """Compute one closed bookkeeping increment from per-step amounts."""
    # References: unified framework Section 5.1, internal-transfer cancellation.
    # Mathematical expression: Delta S = T.T*1 - T*1, hence sum(Delta S)=0.
    # Parameter meanings: transfer is T[i,j], an amount rather than a per-second rate.
    # Units: T and Delta S share one arbitrary amount unit.
    # Domain/constraints: finite non-negative square matrix; no extra dt factor.
    # Sign convention: inflow positive, outflow negative.
    assert transfer.ndim == 2 and transfer.shape[0] == transfer.shape[1]
    assert np.isfinite(transfer).all() and np.all(transfer >= 0.0)
    inflow = transfer.sum(axis=0)
    outflow = transfer.sum(axis=1)
    return inflow - outflow


def _compute_energy_closure() -> Dict[str, float]:
    """Check an independently prescribed energy-rate identity, not graph coupling."""
    # References: unified framework Section 5.2, surface energy balance notation.
    # Mathematical expression: R_net = H + LE + G + dE_canopy/dt.
    # Parameter meanings: R_net is incoming net radiation; H, LE, G are outgoing fluxes.
    # Units: every term, including the canopy storage tendency, is W m-2.
    # Domain/constraints: prescribed synthetic terms; independent of the amount-transfer matrix.
    # Sign convention: positive canopy tendency stores energy; residual is input minus uses.
    r_net = 525.0
    h = 130.0
    le = 330.0
    g = 40.0
    delta_s_canopy = 25.0
    residual = r_net - (h + le + g + delta_s_canopy)
    return {
        "R_net": r_net,
        "H": h,
        "LE": le,
        "G": g,
        "Delta_S_canopy": delta_s_canopy,
        "residual": residual,
    }


def _compute_isotope_closure(
    storage_before: np.ndarray,
    transfer: np.ndarray,
) -> Dict[str, float]:
    """Legacy helper name: compute closed synthetic tracer-fraction bookkeeping."""
    # References: Coplen (2011), isotope ratio versus amount fraction definitions.
    # Mathematical expression: M_i=S_i*f_i; Q_ij=T_ij*f_i*alpha_ij;
    #   M_after_i=M_before_i-sum_j Q_ij+sum_j Q_ji.
    # Parameter meanings: f_i is the tracer amount fraction; alpha_ij is an illustrative enrichment.
    # Units: S, T, M, Q [a.u.]; fractions and enrichment factors [1].
    # Domain/constraints: 0<=f_i*alpha_ij<=1; source inventories suffice; no isotope runtime claim.
    # Sign convention: each positive tracer transfer is subtracted and added exactly once.
    # A true heavy/light amount ratio R requires M=S*R/(1+R), not S*R.
    ratios = np.array([0.0108, 0.0112, 0.0105, 0.0115, 0.0109], dtype=float)
    alpha = np.ones_like(transfer)
    alpha[1, 3] = 1.001
    alpha[3, 1] = 0.999
    alpha[4, 0] = 1.002
    alpha[0, 2] = 0.998

    assert np.all((ratios[:, None] * alpha >= 0.0) & (ratios[:, None] * alpha <= 1.0))
    heavy_before = storage_before * ratios
    heavy_after = heavy_before.copy()
    n = transfer.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or transfer[i, j] <= 0.0:
                continue
            heavy_flux = transfer[i, j] * ratios[i] * alpha[i, j]
            heavy_after[i] -= heavy_flux
            heavy_after[j] += heavy_flux

    storage_after = storage_before + _compute_storage_update(transfer)
    assert np.all(heavy_after >= 0.0) and np.all(heavy_after <= storage_after)
    residual = float(heavy_after.sum() - heavy_before.sum())
    return {
        "total_before": float(heavy_before.sum()),
        "total_after": float(heavy_after.sum()),
        "residual": residual,
    }


def _coupling_group_strengths(transfer: np.ndarray) -> Dict[str, float]:
    """Sum synthetic amounts in assigned edge groups; do not infer physical timescales."""
    # References: unified framework Section 2.3, illustrative grouping only.
    # Mathematical expression: amount_group = sum_(i,j in group) T[i,j].
    # Parameter meanings: edge groups are stated fixture labels, not measured coupling speeds.
    # Units: per-step amounts [a.u.], identical for every group.
    # Domain/constraints: each listed edge belongs to exactly one stated group.
    # Sign convention: group totals sum positive directed transfer amounts.
    fast_edges = [(0, 1), (1, 0), (0, 2), (2, 0)]
    medium_edges = [(1, 3), (3, 1), (2, 3), (3, 2)]
    slow_edges = [(4, 0), (4, 3), (1, 4)]

    def total(edges: List[Tuple[int, int]]) -> float:
        return float(sum(transfer[i, j] for i, j in edges))

    return {
        "fast": total(fast_edges),
        "medium": total(medium_edges),
        "slow": total(slow_edges),
    }


def main() -> None:
    print("=== Tutorial 23: Synthetic amount/tracer arithmetic, not runtime validation ===")

    spheres, transfer = _build_transfer_matrix()
    storage0 = np.array([860.0, 740.0, 980.0, 1240.0, 520.0], dtype=float)
    d_storage = _compute_storage_update(transfer)
    storage1 = storage0 + d_storage
    mass_residual = float(d_storage.sum())

    energy = _compute_energy_closure()
    isotope = _compute_isotope_closure(storage0, transfer)
    strengths = _coupling_group_strengths(transfer)
    # References: Higham (2002), finite floating-point reduction error.
    # Mathematical expression: abs(residual) <= 32*u*activity for each separate identity.
    # Parameter meanings: u is dtype epsilon; activity is the relevant same-unit magnitude.
    # Units: amount/tracer checks [a.u.]; energy check [W m-2].
    # Domain/constraints: finite positive scales and feasible donor inventories.
    # Sign convention: absolute residuals are non-negative numerical discrepancies.
    eps = np.finfo(storage0.dtype).eps
    assert np.all(transfer.sum(axis=1) <= storage0) and np.all(storage1 >= 0.0)
    assert abs(mass_residual) <= 32.0 * eps * transfer.sum()
    assert abs(storage1.sum() - storage0.sum()) <= 32.0 * eps * storage0.sum()
    assert abs(energy["residual"]) <= 32.0 * eps * abs(energy["R_net"])
    assert abs(isotope["residual"]) <= 32.0 * eps * isotope["total_before"]
    assert np.isclose(sum(strengths.values()), transfer.sum(), rtol=0.0, atol=32.0 * eps * transfer.sum())

    print(f"  Synthetic amount residual [a.u.] (sum dS): {mass_residual:.3e}")
    print(f"  Independent prescribed energy identity residual: {energy['residual']:.3e} W m-2")
    print(f"  Synthetic tracer residual [a.u.]: {isotope['residual']:.3e}")
    print(
        f"  Assigned edge-group totals [a.u. per step]: "
        f"{{'fast': {strengths['fast']:.1f}, "
        f"'medium': {strengths['medium']:.1f}, "
        f"'slow': {strengths['slow']:.1f}}}"
    )

    # ------------------------------------------------------------------
    # Figure 1: pure mathematics
    # ------------------------------------------------------------------
    apply_nature_style()
    fig1, axes1 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.44))

    ax = axes1[0]
    net_exchange = transfer - transfer.T
    im = ax.imshow(net_exchange, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(spheres)))
    ax.set_yticks(np.arange(len(spheres)))
    ax.set_xticklabels(spheres, rotation=25, ha="right", fontsize=7)
    ax.set_yticklabels(spheres, fontsize=7)
    ax.set_title("Net Exchange Matrix (T - T^T)", fontsize=9)
    cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Net transfer per step [a.u.]", fontsize=7)
    add_panel_label(ax, "a")

    # Panel b reports actual sums of the stated synthetic matrix, not model scores.
    ax = axes1[1]
    group_names = list(strengths)
    group_amounts = [strengths[name] for name in group_names]
    ax.bar(group_names, group_amounts, color=CB_PALETTE[:3], alpha=0.85)
    ax.set_ylabel("Transferred amount per step [a.u.]")
    ax.set_title("Computed Synthetic Group Totals", fontsize=9)
    for index, amount in enumerate(group_amounts):
        ax.text(index, amount, f"{amount:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0.0, max(group_amounts) * 1.22)
    ax.text(0.02, 0.98, "Assigned labels, not measured timescales", transform=ax.transAxes,
            fontsize=6, ha="left", va="top")
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig1.suptitle("Synthetic Transfer Inputs: Not Measured Earth-System Exchanges", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig1, "23_math_five_sphere_matrix")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: separate synthetic bookkeeping identities, no RELIC runtime evidence
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.44))

    ax = axes2[0]
    x = np.arange(len(spheres))
    width = 0.36
    ax.bar(x - width / 2.0, storage0, width=width, color=CB_PALETTE[0], alpha=0.75, label="Before")
    ax.bar(x + width / 2.0, storage1, width=width, color=CB_PALETTE[2], alpha=0.75, label="After one step")
    ax.set_xticks(x)
    ax.set_xticklabels(spheres, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Generic amount [a.u.]")
    ax.set_title("Synthetic Closed Amount Update", fontsize=9)
    ax.legend(fontsize=6, loc="upper right", frameon=False, prop={"weight": "bold"})
    add_panel_label(ax, "a")

    ax = axes2[1]
    # Keep dimensional residuals in separate table rows; there is no common numeric axis.
    ax.axis("off")
    rows = [
        ["Amount", f"{mass_residual:.3e}", "a.u."],
        ["Energy rate", f"{energy['residual']:.3e}", "W m-2"],
        ["Tracer amount", f"{isotope['residual']:.3e}", "a.u."],
    ]
    table = ax.table(cellText=rows, colLabels=["Identity", "Residual", "Unit"],
                     loc="center", cellLoc="center", colWidths=[0.40, 0.35, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.6)
    ax.set_title("Separate Synthetic Arithmetic Checks", fontsize=9)
    ax.text(0.5, 0.16, "Energy inputs are independent of T.\nTracer fraction is not a heavy/light ratio.",
            transform=ax.transAxes, ha="center", va="center", fontsize=6)
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig2.suptitle("Synthetic Bookkeeping Checks: No Runtime or Empirical Certification", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig2, "23_relic_five_sphere_conservation")
    plt.close(fig2)

    print("[OK] Synthetic arithmetic assertions passed; no RELIC runtime tests were run.")


if __name__ == "__main__":
    main()

