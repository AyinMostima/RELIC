r"""Tutorial 17 -- Fiber Bundle Structure of Multi-Scale State Spaces.

Demonstrates
------------
This standalone NumPy illustration groups synthetic state dimensions by
SpatialScale. The fiber-bundle terminology is a design analogy, not a proof
of local triviality or a certified RELIC connection:

    pi: E -> B

where:
  B = S (SpatialScale lattice)     -- base space
  E_s = pi^{-1}(s)                 -- fiber over scale s
  E_s = tensor_{v in V_s} R^{d_v}  -- state tensors at scale s
  pi maps each variable to its declared SpatialScale

The example computes nested, disjoint weighted sums with fixed partitions.
Their equality follows from associativity. General conservative remapping
need not be invertible or exactly transitive; mass conservation alone does
not establish a differential-geometric connection or zero curvature.

This tutorial produces TWO figures:
  Fig 1 (17_math_fiber.png):  Mathematical fiber bundle visualization
  Fig 2 (17_relic_fiber.png): RELIC SpatialScale integration

References
----------
1. Husemoller, D. (1993). Fibre Bundles (3rd ed.). Springer.
2. Jones, P.W. (1999). First- and second-order conservative remapping
   schemes for grids in spherical coordinates. MWR 127(9), 2204-2210.
3. spatiotemporal_scale_topology.md, Sections 2.3, 2.4

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.functor.functor_types.SpatialScale
- relic.core.functor.functor_types.TemporalScale
- relic.core.functor.functor_types.VariableKind
- relic.core.functor.functor_types.FunctorMetadata
- relic.core.functor.functor_types.ProcessCategory
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# Ensure repo root is on path
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE, FILL_PALETTE, DOUBLE_COL_WIDTH,
    add_panel_label, add_text_box, add_legend_outside,
    apply_nature_style, savefig,
)

# === RELIC module imports ===
from relic.core.functor.functor_types import (
    SpatialScale, TemporalScale, VariableKind,
    ProcessCategory, FunctorMetadata,
)

# -----------------------------------------------------------------------
# 1. Define the RELIC spatial scale lattice (base space B = S)
# -----------------------------------------------------------------------
# Reference: spatiotemporal_scale_topology.md, Definition 2.1
# Math: B = S = {POINT, PATCH, GRID_CELL, HILLSLOPE, BASIN, LANDSCAPE,
#                GRID_2D, REGIONAL, GLOBAL}

# Ordering: finer -> coarser (partial order)
SCALE_ORDER = [
    SpatialScale.POINT,
    SpatialScale.PATCH,
    SpatialScale.GRID_CELL,
    SpatialScale.HILLSLOPE,
    SpatialScale.BASIN,
    SpatialScale.LANDSCAPE,
    SpatialScale.REGIONAL,
    SpatialScale.GLOBAL,
]
SCALE_NAMES = [s.value for s in SCALE_ORDER]
N_SCALES = len(SCALE_ORDER)

print("=== Selected SpatialScale labels in an illustrative order ===")
for i, s in enumerate(SCALE_ORDER):
    print(f"  Level {i}: {s.value}")

# -----------------------------------------------------------------------
# 2. Define fibers: state variables at each scale
# -----------------------------------------------------------------------
# Reference: spatiotemporal_scale_topology.md, Eq. E_s = tensor_{v in V_s} R^{d_v}
# Each fiber E_s contains the state variables declared at scale s

# Synthetic variable declarations (mimicking RELIC FunctorMetadata)
FIBER_VARS = {
    SpatialScale.POINT: [
        ("air_temp", 1, VariableKind.FORCING),
        ("PAR", 1, VariableKind.FORCING),
    ],
    SpatialScale.PATCH: [
        ("leaf_C", 1, VariableKind.STATE),
        ("root_C", 1, VariableKind.STATE),
        ("stomatal_cond", 1, VariableKind.DIAGNOSTIC),
    ],
    SpatialScale.GRID_CELL: [
        ("soil_C", 3, VariableKind.STATE),   # 3 layers
        ("soil_water", 3, VariableKind.STATE),
    ],
    SpatialScale.HILLSLOPE: [
        ("runoff", 1, VariableKind.FLUX),
        ("erosion", 1, VariableKind.FLUX),
    ],
    SpatialScale.BASIN: [
        ("streamflow", 1, VariableKind.FLUX),
        ("total_ET", 1, VariableKind.DIAGNOSTIC),
    ],
    SpatialScale.LANDSCAPE: [
        ("GPP_landscape", 1, VariableKind.DIAGNOSTIC),
    ],
    SpatialScale.REGIONAL: [
        ("atm_CO2", 1, VariableKind.FORCING),
    ],
    SpatialScale.GLOBAL: [
        ("global_C_budget", 1, VariableKind.DIAGNOSTIC),
    ],
}

print("\n=== Fiber Structure ===")
for scale, variables in FIBER_VARS.items():
    dims = sum(d for _, d, _ in variables)
    print(f"  E_{{{scale.value}}}: dim={dims}, vars={[v[0] for v in variables]}")

# -----------------------------------------------------------------------
# 3. Nested weighted-sum example (not a general connection)
# -----------------------------------------------------------------------
# Reference: spatiotemporal_scale_topology.md, Definition 2.3
# Math: R_{s1->s2}: E_{s1} -> E_{s2}
# References: Jones (1999) for actual remapping; associativity for this example.
# Mathematical expression: sum_k sum_{i in k} a_i u_i = sum_i a_i u_i.
# Parameter meanings: a_i is area fraction; u_i is patch density; partitions are disjoint.
# Units: contributions are g C per square metre of basin area, not cell mean density.
# Domain/constraints: positive normalized weights; no general regrid composition claim.
# Sign convention: non-negative carbon-density contributions; no flux is advanced.

# Demonstrate conservative aggregation: patch -> grid_cell -> basin
np.random.seed(42)
n_patches = 4
n_cells = 2
# Fine-scale data: leaf carbon at 4 patches
patch_leaf_C = np.array([150.0, 180.0, 120.0, 200.0])  # g C m^-2
patch_areas = np.array([0.25, 0.30, 0.20, 0.25])        # fractional area

# Aggregation: patch -> grid_cell (2 cells, each 2 patches)
# Math: u_cell_k = sum_{patches in cell k} area_i * u_i
cell_leaf_C = np.zeros(n_cells)
cell_leaf_C[0] = patch_areas[0]*patch_leaf_C[0] + patch_areas[1]*patch_leaf_C[1]
cell_leaf_C[1] = patch_areas[2]*patch_leaf_C[2] + patch_areas[3]*patch_leaf_C[3]

# Aggregation: grid_cell -> basin
# Math: u_basin = sum_k u_cell_k
basin_leaf_C = cell_leaf_C.sum()

# Direct aggregation: patch -> basin
# Math: u_basin_direct = sum_i area_i * u_i
basin_direct = (patch_areas * patch_leaf_C).sum()

# Conservation check
# This equality tests this fixed nested partition, not every conservative regridder.
transitivity_error = abs(basin_leaf_C - basin_direct)
print("\n[OK] Nested weighted-sum equality for this synthetic partition:")
print(f"  Patch leaf_C: {patch_leaf_C}")
print(f"  Cell leaf_C:  {cell_leaf_C}")
print(f"  Basin leaf_C (two-step): {basin_leaf_C:.4f}")
print(f"  Basin leaf_C (direct):   {basin_direct:.4f}")
print(f"  Transitivity error: {transitivity_error:.2e}  "
      f"(expected ~0 for this fixed partition)")

# Disaggregation: basin -> patch (information loss)
# Math: u_patch_i = u_basin / n_patches (uniform distribution assumption)
disagg_leaf_C = np.full(n_patches, basin_leaf_C / n_patches)
info_loss = np.linalg.norm(disagg_leaf_C - patch_leaf_C * patch_areas)
print(f"  Disagg. info loss (L2): {info_loss:.4f}")

# =====================================================================
# FIGURE 1: Mathematical Fiber Bundle Visualization
# =====================================================================
apply_nature_style()

fig1 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.52))
gs1 = gridspec.GridSpec(1, 3, figure=fig1, wspace=0.4,
                        width_ratios=[1.2, 1.0, 1.0])

# --- Panel (a): Fiber bundle schematic ---
ax = fig1.add_subplot(gs1[0, 0])
ax.axis("off")
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.16, 1.08)

# Draw base space (horizontal axis) with scale nodes
base_y = 0.10
scales_to_show = [
    (0, "Point"), (1, "Patch"), (2, "Grid Cell"),
    (3, "Hillslope"), (4, "Basin"),
]
fiber_heights = [0.7, 0.65, 0.55, 0.45, 0.35]
x_positions = [0.05, 0.25, 0.50, 0.72, 0.95]

# Draw base line
ax.plot([0.0, 1.0], [base_y, base_y], color="#333", lw=1.5, zorder=5)
ax.text(0.5, -0.10, "Base Space $B = \\mathcal{S}$ (Scale Lattice)",
        ha="center", fontsize=5.5, style="italic")

# Draw fibers as vertical lines with data points
for i, (idx, name) in enumerate(scales_to_show):
    x = x_positions[i]
    h = fiber_heights[i]
    color = CB_PALETTE[i]
    fill = FILL_PALETTE[i]

    # Fiber line (vertical)
    ax.plot([x, x], [base_y, base_y + h], color=color, lw=1.8,
            alpha=0.7, zorder=3)

    # Fiber region (shaded)
    rect = mpatches.FancyBboxPatch(
        (x - 0.06, base_y), 0.12, h,
        boxstyle="round,pad=0.02", facecolor=fill,
        edgecolor=color, alpha=0.35, linewidth=0.6, zorder=2)
    ax.add_patch(rect)

    # Scale node on base
    ax.plot(x, base_y, 'o', color=color, markersize=6, zorder=6)
    ax.text(x, base_y - 0.06, name, ha="center", fontsize=4,
            rotation=0, color="#333")

    # State variables (dots in fiber)
    n_vars = len(FIBER_VARS[SCALE_ORDER[idx]])
    for v_idx in range(n_vars):
        vy = base_y + h * (v_idx + 1) / (n_vars + 1)
        ax.plot(x, vy, 's', color=color, markersize=3.5, zorder=7)

    # Fiber label
    n_dim = sum(d for _, d, _ in FIBER_VARS[SCALE_ORDER[idx]])
    ax.text(x, base_y + h + 0.03, f"$E_s$\n$d={n_dim}$",
            ha="center", fontsize=4, color=color)

# Draw connection arrows (parallel transport)
for i in range(len(scales_to_show) - 1):
    x0, x1 = x_positions[i], x_positions[i + 1]
    mid_y0 = base_y + fiber_heights[i] * 0.5
    mid_y1 = base_y + fiber_heights[i + 1] * 0.5
    ax.annotate("", xy=(x1 - 0.07, mid_y1),
                xytext=(x0 + 0.07, mid_y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                color="#999", linestyle="--"))

ax.text(0.55, 0.78, "$R_{s_1 \\to s_2}$ (regridding)",
        ha="center", fontsize=5.5, color="#777", style="italic")
ax.set_title("Scale-indexed State Analogy", fontsize=9, pad=8)
add_panel_label(ax, "a", x=-0.05, y=1.05)

# --- Panel (b): Conservative aggregation ---
ax = fig1.add_subplot(gs1[0, 1])
stages = ["Patch\n(4 units)", "Grid Cell\n(2 units)", "Basin\n(1 unit)"]
values = [
    patch_leaf_C * patch_areas,    # extensive values at patch
    cell_leaf_C,
    np.array([basin_leaf_C])
]
y_positions_agg = [0.85, 0.50, 0.15]

for s_idx, (stage, vals, y_pos) in enumerate(zip(stages, values, y_positions_agg)):
    color = CB_PALETTE[s_idx + 1]
    # Draw boxes for each value
    n_v = len(vals)
    x_start = 0.5 - n_v * 0.08
    for v_idx, val in enumerate(vals):
        x = x_start + v_idx * 0.16
        rect = mpatches.FancyBboxPatch(
            (x, y_pos - 0.04), 0.12, 0.08,
            boxstyle="round,pad=0.01", facecolor=FILL_PALETTE[s_idx + 1],
            edgecolor=color, linewidth=0.8, alpha=0.9,
            transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.06, y_pos, f"{val:.1f}",
                ha="center", va="center", fontsize=5.5, color="#333",
                transform=ax.transAxes)
    # Stage label
    ax.text(0.12, y_pos, stage, ha="right", va="center",
            fontsize=6, color=color, fontweight="bold",
            transform=ax.transAxes)

# Arrows between stages
for i in range(2):
    ax.annotate("", xy=(0.5, y_positions_agg[i + 1] + 0.06),
                xytext=(0.5, y_positions_agg[i] - 0.06),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#777"))
    ax.text(0.62, (y_positions_agg[i] + y_positions_agg[i + 1]) / 2,
            "Agg", fontsize=6, color="#777", style="italic",
            transform=ax.transAxes)

# Conservation annotation
total_patch = (patch_leaf_C * patch_areas).sum()
ax.text(0.5, 0.02,
        f"Weighted density: {total_patch:.1f}",
        ha="center", fontsize=6.5, color=CB_PALETTE[2],
        fontweight="bold", transform=ax.transAxes)

ax.axis("off")
ax.set_title("Conservative Aggregation", fontsize=9, pad=8)
add_panel_label(ax, "b", x=-0.08, y=1.05)

# --- Panel (c): Transitivity and info loss ---
ax = fig1.add_subplot(gs1[0, 2])
labels = ["Two-step\n(P->C->B)", "Direct\n(P->B)"]
vals_compare = [basin_leaf_C, basin_direct]
bars = ax.bar([0, 1], vals_compare, color=[CB_PALETTE[0], CB_PALETTE[2]],
              alpha=0.8, width=0.5)
for bar, val in zip(bars, vals_compare):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.2f}", ha="center", fontsize=6.5)
ax.set_ylabel("Basin Leaf C [g C m$^{-2}$]", fontsize=7)
ax.set_xticks([0, 1])
ax.set_xticklabels(labels, fontsize=7)
ax.set_title("Nested Partition Equality", fontsize=9, pad=8)
add_text_box(ax, f"$\\epsilon_{{regrid}} = {transitivity_error:.1e}$\n"
             "(this partition only)",
             loc="lower right", fontsize=5.5)
add_panel_label(ax, "c", x=-0.18, y=1.05)

fig1.suptitle(
    "Synthetic State Family and Nested Weighted Contributions",
    fontsize=10, y=1.03, fontweight="bold",
)
savefig(fig1, "17_math_fiber")
plt.close(fig1)

# =====================================================================
# FIGURE 2: RELIC SpatialScale Integration
# =====================================================================
fig2 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45))
gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.45,
                        width_ratios=[1.0, 1.2, 1.0])

# --- Panel (a): RELIC SpatialScale lattice ---
ax = fig2.add_subplot(gs2[0, 0])
y_pos_list = np.linspace(0.92, 0.08, N_SCALES)
for i, (scale, y_pos) in enumerate(zip(SCALE_ORDER, y_pos_list)):
    color = CB_PALETTE[i % len(CB_PALETTE)]
    fill = FILL_PALETTE[i % len(FILL_PALETTE)]
    rect = plt.Rectangle((0.05, y_pos - 0.03), 0.6, 0.06,
                          facecolor=fill, edgecolor=color,
                          linewidth=0.8, alpha=0.85,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(0.35, y_pos, scale.value, ha="center", va="center",
            fontsize=6, fontweight="bold", color="#333",
            transform=ax.transAxes)
    # Dimension info
    if scale in FIBER_VARS:
        dim = sum(d for _, d, _ in FIBER_VARS[scale])
        ax.text(0.72, y_pos, f"dim={dim}", ha="left", va="center",
                fontsize=5.5, color="#777", transform=ax.transAxes)

# Arrows between adjacent scales
for i in range(N_SCALES - 1):
    ax.annotate("", xy=(0.35, y_pos_list[i + 1] + 0.035),
                xytext=(0.35, y_pos_list[i] - 0.035),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#aaa"))

ax.axis("off")
ax.set_title("RELIC SpatialScale Enum", fontsize=9, pad=6)
add_panel_label(ax, "a", x=-0.05, y=1.05)

# --- Panel (b): Fiber dimension per scale (stacked bar chart) ---
ax = fig2.add_subplot(gs2[0, 1])

# Compute dimension per VariableKind at each scale
kind_colors = {
    VariableKind.FORCING: CB_PALETTE[4],
    VariableKind.STATE: CB_PALETTE[0],
    VariableKind.FLUX: CB_PALETTE[2],
    VariableKind.DIAGNOSTIC: CB_PALETTE[1],
}
kind_labels = [VariableKind.FORCING, VariableKind.STATE,
               VariableKind.FLUX, VariableKind.DIAGNOSTIC]
y_idx = np.arange(N_SCALES)
bottom = np.zeros(N_SCALES)

for vk in kind_labels:
    dims = []
    for scale in SCALE_ORDER:
        if scale in FIBER_VARS:
            d = sum(dim for _, dim, kind in FIBER_VARS[scale] if kind == vk)
        else:
            d = 0
        dims.append(d)
    dims = np.array(dims)
    ax.barh(y_idx, dims, left=bottom, height=0.5,
            color=kind_colors[vk], alpha=0.8, label=vk.value)
    bottom += dims

ax.set_yticks(y_idx)
ax.set_yticklabels([s.value for s in SCALE_ORDER], fontsize=5.5)
ax.set_xlabel("Synthetic state dimension $d_s$", fontsize=7)
ax.set_title("$\\pi$: Variable-Scale Projection", fontsize=8, pad=6)
ax.invert_yaxis()
add_legend_outside(ax, fontsize=5, ncol=2)
add_panel_label(ax, "b", x=-0.18, y=1.05)

# --- Panel (c): chosen redistribution loses information; this is not curvature ---
ax = fig2.add_subplot(gs2[0, 2])

# Compare aggregation -> disaggregation vs identity
original = patch_leaf_C * patch_areas
disagg = np.full_like(original, original.sum() / len(original))
x_pos = np.arange(len(original))
ax.bar(x_pos - 0.15, original, 0.3, color=CB_PALETTE[0], alpha=0.8,
       label="Original $u_s$")
ax.bar(x_pos + 0.15, disagg, 0.3, color=CB_PALETTE[1], alpha=0.8,
       label="Dis(Agg($u_s$))")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"P{i+1}" for i in range(len(original))], fontsize=7)
ax.set_ylabel("Leaf C [g C m$^{-2}$]", fontsize=7)
ax.set_title("Chosen Redistribution Loss", fontsize=9, pad=6)
rel_err = np.linalg.norm(disagg - original) / (np.linalg.norm(original) + 1e-12)
add_text_box(ax, f"L2 loss = {rel_err:.1%}", loc="upper right", fontsize=6)
add_legend_outside(ax, fontsize=5.5)
add_panel_label(ax, "c", x=-0.18, y=1.05)

fig2.suptitle(
    "SpatialScale Labels, Synthetic Dimensions and Redistribution Loss",
    fontsize=10, y=1.03, fontweight="bold",
)
savefig(fig2, "17_relic_fiber")
plt.close(fig2)

# -----------------------------------------------------------------------
# Final verification
# -----------------------------------------------------------------------
assert transitivity_error < 1e-10, "Nested weighted sums disagree for this fixed partition"
print(f"\n[OK] Tutorial 17 completed successfully.")
print("  Fig 1: synthetic scale family and fixed-partition weighted sums")
print("  Fig 2: real enum labels with illustrative dimensions and redistribution")
print(f"  Nested-sum equality error = {transitivity_error:.2e} (this partition only)")
print(f"  Info loss (Dis o Agg != Id): {rel_err:.2%}")
