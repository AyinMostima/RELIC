r"""Tutorial 15 -- Scale Coupling: Aggregation, Broadcast, and Information Loss.

Demonstrates
------------
Standalone NumPy demonstration of aggregation (area-weighted mean) and
broadcast disaggregation on nested, equal-area cells. Only the RELIC
SpatialScale enum is imported; no RELIC scale operator is called. With
area-weighted inner products the broadcast is the adjoint of the
area-weighted mean (R = M_c^{-1} P^T M_f). Aggregation conserves the area
integral on nested, fully covered cells (here by construction), while
disaggregation cannot recover sub-cell variability.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Spatial scale lattice (spatiotemporal_scale_topology.md Def. 1.1):
    The preference function p ranks scales from finest to coarsest.

Aggregation (upscaling) for intensive variables:
    phi_coarse = Sum_i(w_i * phi_i) / Sum_i(w_i)       ... (1)

Aggregation for extensive variables (phi_i an area density on nested cells):
    Phi_coarse = Sum_i(phi_i * A_i)                     ... (2)
    (ConservativeRemap expects per-cell totals X_i = phi_i * A_i;
     HierarchicalAggregator expects area densities.)

Conservation guarantee (Corollary 2.1):
    Sum_j Phi_j^t = Sum_i Phi_i^s                       ... (3)

Disaggregation (downscaling):
    phi_fine_i = phi_coarse_{J(i)}  (uniform distribution)

Weighted adjoint (finite-dimensional; M_f = diag(A_i), M_c = diag(A_J)):
    <P c, f>_{M_f} = <c, R f>_{M_c},  R = M_c^{-1} P^T M_f,  R P = I
    (P = broadcast, R = area-weighted mean)

Information loss (P R != I):
    ||phi_f - P(R(phi_f))|| > 0  unless phi_f is constant within every coarse cell

References
----------
1. spatiotemporal_scale_topology.md, Sections 1.1, 2.1-2.3
2. Jones, P.W. (1999). Conservative remapping schemes.
   JCP 154(2), 283-308.
3. Wood, A.W. et al. (2004). Hydrologic implications of downscaling.
   Climatic Change, 62, 189-216.

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.functor.functor_types.SpatialScale (scale lattice labels only)
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
    apply_nature_style, savefig, mm_to_inches,
)

# -----------------------------------------------------------------------
# 0. Import RELIC SpatialScale enum to demonstrate the scale lattice
# -----------------------------------------------------------------------
from relic.core.functor.functor_types import SpatialScale

print("RELIC SpatialScale lattice members:")
for scale in SpatialScale:
    print(f"  {scale.name} = {scale.value}")

# -----------------------------------------------------------------------
# 1. Create inhomogeneous fine-scale field (10x10 grid)
# -----------------------------------------------------------------------
# Synthetic soil carbon field with realistic spatial heterogeneity
# Reference: Manzoni & Porporato (2009)
rng = np.random.default_rng(42)

nx_fine, ny_fine = 10, 10
cell_area_fine = 100.0  # m^2 per cell

# Generate heterogeneous soil C field [g C m^-2]
# Reference: Log-normal distribution is common for soil properties
# Math: C_soil ~ LogNormal(mu, sigma)
mu_logC, sigma_logC = np.log(500), 0.5
soil_c_fine = rng.lognormal(mu_logC, sigma_logC, (ny_fine, nx_fine))

# -----------------------------------------------------------------------
# 2. Upscaling: fine (10x10) -> coarse (5x5) grid
# -----------------------------------------------------------------------
nx_coarse, ny_coarse = 5, 5
block_y, block_x = ny_fine // ny_coarse, nx_fine // nx_coarse

# Intensive variable (concentration): area-weighted mean
# Reference: spatiotemporal_scale_topology.md Theorem 2.1
# Math: phi_coarse = Sum(w_i * phi_i) / Sum(w_i)
soil_c_coarse = np.zeros((ny_coarse, nx_coarse))
for i in range(ny_coarse):
    for j in range(nx_coarse):
        block = soil_c_fine[i*block_y:(i+1)*block_y, j*block_x:(j+1)*block_x]
        # Area-weighted mean (uniform area -> simple mean)
        soil_c_coarse[i, j] = block.mean()

# Extensive variable: total C per coarse cell = mean * coarse_area
cell_area_coarse = cell_area_fine * block_x * block_y
total_C_fine = soil_c_fine.sum() * cell_area_fine
total_C_coarse = soil_c_coarse.sum() * cell_area_coarse

# -----------------------------------------------------------------------
# 3. Downscaling: coarse -> fine (uniform disaggregation)
# -----------------------------------------------------------------------
# Reference: spatiotemporal_scale_topology.md Theorem 2.1
# Math: phi_fine_i = phi_coarse_{J(i)} (uniform distribution)
soil_c_disagg = np.zeros_like(soil_c_fine)
for i in range(ny_coarse):
    for j in range(nx_coarse):
        soil_c_disagg[i*block_y:(i+1)*block_y, j*block_x:(j+1)*block_x] = soil_c_coarse[i, j]

# Information loss
# Reference: spatiotemporal_scale_topology.md Def. 2.3
# Math: info_loss = ||phi_f - Dis(Agg(phi_f))|| / ||phi_f||
info_loss_abs = np.abs(soil_c_fine - soil_c_disagg)
info_loss_relative = np.linalg.norm(info_loss_abs) / np.linalg.norm(soil_c_fine)

# Conservation check
conservation_error = abs(total_C_fine - total_C_coarse) / total_C_fine

print(f"\n[OK] Upscaling: 10x10 -> 5x5")
print(f"  Total C (fine):   {total_C_fine:.2f} g C")
print(f"  Total C (coarse): {total_C_coarse:.2f} g C")
print(f"  Conservation error: {conservation_error:.2e} ({conservation_error*100:.4f}%)")
print(f"  Information loss (relative L2): {info_loss_relative:.4f} ({info_loss_relative*100:.2f}%)")

# -----------------------------------------------------------------------
# 4. Plot: 4-panel figure (2x2)
# -----------------------------------------------------------------------
apply_nature_style()

fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.7))

vmin = min(soil_c_fine.min(), soil_c_disagg.min()) * 0.9
vmax = max(soil_c_fine.max(), soil_c_disagg.max()) * 1.1

# --- Panel (a): Fine-scale field ---
ax = axes[0, 0]
im = ax.imshow(soil_c_fine, cmap="YlOrBr", vmin=vmin, vmax=vmax, origin="lower")
ax.set_title("Fine-Scale Field (10$\\times$10)", fontsize=9)
ax.set_xlabel("x cell")
ax.set_ylabel("y cell")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("[g C m$^{-2}$]", fontsize=6)
cbar.ax.tick_params(labelsize=5)
add_panel_label(ax, "a")

# --- Panel (b): Coarse-scale aggregated field ---
ax = axes[0, 1]
im = ax.imshow(soil_c_coarse, cmap="YlOrBr", vmin=vmin, vmax=vmax, origin="lower")
ax.set_title("Aggregated Field (5$\\times$5)", fontsize=9)
ax.set_xlabel("x cell")
ax.set_ylabel("y cell")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("[g C m$^{-2}$]", fontsize=6)
cbar.ax.tick_params(labelsize=5)
add_panel_label(ax, "b")

# --- Panel (c): Disaggregated field ---
ax = axes[1, 0]
im = ax.imshow(soil_c_disagg, cmap="YlOrBr", vmin=vmin, vmax=vmax, origin="lower")
ax.set_title("Disaggregated (Uniform)", fontsize=9)
ax.set_xlabel("x cell")
ax.set_ylabel("y cell")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("[g C m$^{-2}$]", fontsize=6)
cbar.ax.tick_params(labelsize=5)
add_panel_label(ax, "c")

# --- Panel (d): Conservation bar chart + info loss ---
ax = axes[1, 1]
bar_x = [0, 1]
bar_vals = [total_C_fine, total_C_coarse]
bar_labels = ["Fine\n(10x10)", "Coarse\n(5x5)"]
bars = ax.bar(bar_x, bar_vals, color=[CB_PALETTE[0], CB_PALETTE[2]],
              edgecolor="black", linewidth=0.5, width=0.5)
ax.set_xticks(bar_x)
ax.set_xticklabels(bar_labels, fontsize=7)
ax.set_ylabel("Total Carbon [g C]")
ax.set_title("Conservation Verification", fontsize=9)

# Annotate with values
for bar, val in zip(bars, bar_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + val*0.01,
            f"{val:.0f}", ha="center", fontsize=6)

# Info loss annotation
ax.text(0.95, 0.5,
        f"Conservation err:\n{conservation_error:.2e}\n\n"
        f"Info loss (L2):\n{info_loss_relative:.1%}",
        transform=ax.transAxes, fontsize=6, ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ccc"))
add_panel_label(ax, "d")

fig.suptitle(
    "Scale Coupling on Nested Equal-Area Cells: $\\mathbf{Agg}$ Conserves the Area Integral, "
    "$\\mathbf{Dis}$ Loses Information",
    fontsize=10, y=1.03,
)
plt.tight_layout()
savefig(fig, "15_scale_coupling")
plt.close(fig)

print("\n[OK] Tutorial 15 completed successfully.")
print("  Demonstrated: upscaling/downscaling with conservation and information loss")
print(f"  Used RELIC SpatialScale enum: {[s.name for s in SpatialScale]}")
