r"""Tutorial 04 -- Semantic Single Source of Truth (SSOT).

Demonstrates
------------
RELIC enforces two semantic mappings as the single source of truth:

  Pi : K -> (Theta, U, [L,U], meta)   -- parameter mapping
  Gamma : V -> (U, [L,U], writer, tags)   -- variable contract

This tutorial produces TWO figures:
  Fig 1  (04_math_ssot.png): Parameter/variable contract structure
  Fig 2  (04_relic_ssot.png): RELIC VariableKind + ProcessCategory

References
----------
1. coertheory.md, Section 3.2 (Semantic SSOT)
2. architecture_overview.md, Parameter System section

RELIC modules used
~~~~~~~~~~~~~~~~~~
- relic.core.functor.functor_types.VariableKind
- relic.core.functor.functor_types.ProcessCategory
- relic.core.functor.functor_types.FunctorMetadata
- relic.core.functor.functor_types.SpatialScale
- relic.core.functor.functor_types.TemporalScale
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tutorial.theory._common import (
    CB_PALETTE, FILL_PALETTE, DOUBLE_COL_WIDTH,
    add_panel_label, add_text_box, apply_nature_style, savefig,
)

# === RELIC imports ===
from relic.core.functor.functor_types import (
    VariableKind, ProcessCategory, FunctorMetadata,
    SpatialScale, TemporalScale, ExecutionPhase,
)

# -----------------------------------------------------------------------
# 1. Synthetic parameter database (Pi mapping)
# -----------------------------------------------------------------------
parameters = {
    "Vcmax_25": {"value": 60.0, "unit": "umol m-2 s-1",
                 "bounds": [10,150], "ref": "Farquhar (1980)"},
    "Jmax_25":  {"value": 120.0, "unit": "umol m-2 s-1",
                 "bounds": [20,300], "ref": "Wullschleger (1993)"},
    "k_decomp": {"value": 0.02, "unit": "day-1",
                 "bounds": [0.001,0.1], "ref": "Parton (1987)"},
    "theta_sat": {"value": 0.45, "unit": "m3 m-3",
                  "bounds": [0.3,0.6], "ref": "Clapp (1978)"},
    "Q10":      {"value": 2.0, "unit": "-",
                 "bounds": [1,5],   "ref": "Lloyd (1994)"},
}

# -----------------------------------------------------------------------
# 2. Variable contracts (Gamma mapping) using RELIC VariableKind
# -----------------------------------------------------------------------
variables = [
    {"name": "soil_carbon",     "kind": VariableKind.STATE,      "unit": "g C m-2",  "bounds": [0,50000]},
    {"name": "leaf_temperature","kind": VariableKind.STATE,      "unit": "K",        "bounds": [200,340]},
    {"name": "net_photo",       "kind": VariableKind.FLUX,       "unit": "umol m-2s-1","bounds": [-5,50]},
    {"name": "friction_vel",    "kind": VariableKind.DIAGNOSTIC, "unit": "m s-1",    "bounds": [0,5]},
    {"name": "air_temp",        "kind": VariableKind.FORCING,    "unit": "K",        "bounds": [200,350]},
]

# -----------------------------------------------------------------------
# 3. RELIC enum showcase
# -----------------------------------------------------------------------
print("=== RELIC Type System (Semantic SSOT) ===")
print("\nVariableKind:")
for vk in VariableKind:
    print(f"  {vk.name:12s} = {vk.value}")
print("\nProcessCategory:")
for pc in ProcessCategory:
    print(f"  {pc.name:22s} = {pc.value}")
print("\nSpatialScale:")
for ss in SpatialScale:
    print(f"  {ss.name:10s} = {ss.value}")
print("\nTemporalScale:")
# Calendar- or schedule-dependent scales have no fixed period; the SSOT
# requires the caller to declare one (example periods in days below).
EXAMPLE_PERIOD_DAYS = {
    TemporalScale.SUB_DAILY: 0.25,
    TemporalScale.WEEK: 7.0,
    TemporalScale.MONTH: 30.4375,
    TemporalScale.SEASON: 91.3125,
    TemporalScale.YEAR: 365.25,
}
for ts in TemporalScale:
    if ts is TemporalScale.UNKNOWN:
        print(f"  {ts.name:12s} = {ts.value:16s}  tau = (undeclared; no characteristic period)")
        continue
    if ts in EXAMPLE_PERIOD_DAYS:
        char_time = ts.to_characteristic_time_days(period_days=EXAMPLE_PERIOD_DAYS[ts])
        print(f"  {ts.name:12s} = {ts.value:16s}  tau = {char_time:.2f} days (declared period)")
    else:
        char_time = ts.to_characteristic_time_days()
        print(f"  {ts.name:12s} = {ts.value:16s}  tau = {char_time:.2f} days")

# Build FunctorMetadata example
meta = FunctorMetadata(
    name="Farquhar_Photosynthesis",
    category=ProcessCategory.BIOLOGICAL_PROCESS,
    spatial_scale=SpatialScale.PATCH,
    description="FvCB C3 photosynthesis model",
    references=[
        "Farquhar et al. (1980) C3 photosynthesis model",
        "docs/source/theory/photosynthesis.md (RELIC theory SSOT)",
    ],
    input_variables=["air_temp", "CO2", "PAR"],
    output_variables=["net_photo", "stomatal_cond"],
)
print(f"\n[OK] FunctorMetadata created: {meta.name}")
print(f"  Category: {meta.category.value}")
print(f"  Scale:    {meta.spatial_scale.value}")

# =====================================================================
# FIGURE 1: Mathematical SSOT Structure
# =====================================================================
apply_nature_style()

fig1 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.4))
gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.4)

# Panel (a): Parameter mapping Pi
ax = fig1.add_subplot(gs1[0, 0])
ax.axis("off")
ax.set_title("Parameter Mapping $\\Pi: \\mathcal{K} \\to (\\Theta, \\mathcal{U}, [L,U])$",
             fontsize=9, pad=8)

y = 0.92
for key, info in parameters.items():
    bounds_s = f"[{info['bounds'][0]}, {info['bounds'][1]}]"
    ax.text(0.03, y, key, fontsize=7, fontweight="bold",
            color=CB_PALETTE[0], transform=ax.transAxes)
    ax.text(0.03, y-0.05,
            f"  = {info['value']}  {info['unit']}  {bounds_s}",
            fontsize=5.5, color="#444", transform=ax.transAxes)
    ax.text(0.03, y-0.09, f"  [{info['ref']}]",
            fontsize=4.5, color="#999", fontstyle="italic",
            transform=ax.transAxes)
    y -= 0.17
add_panel_label(ax, "a", x=-0.05, y=1.05)

# Panel (b): Variable contracts Gamma
ax = fig1.add_subplot(gs1[0, 1])
ax.axis("off")
ax.set_title("Variable Contract $\\Gamma: \\mathcal{V} \\to (\\mathcal{U}, [L,U], \\mathrm{kind})$",
             fontsize=9, pad=8)

kind_colors = {
    VariableKind.STATE:      CB_PALETTE[0],
    VariableKind.FLUX:       CB_PALETTE[1],
    VariableKind.DIAGNOSTIC: CB_PALETTE[2],
    VariableKind.FORCING:    CB_PALETTE[4],
}

y = 0.92
for v in variables:
    kc = kind_colors.get(v["kind"], "#333")
    ax.text(0.03, y, v["name"], fontsize=7, fontweight="bold",
            color=kc, transform=ax.transAxes)
    ax.text(0.03, y-0.05,
            f"  [{v['kind'].value}]  {v['unit']}  bounds={v['bounds']}",
            fontsize=5.5, color="#444", transform=ax.transAxes)
    y -= 0.16
add_panel_label(ax, "b", x=-0.05, y=1.05)

fig1.suptitle("Mathematical Foundation: Semantic Single Source of Truth",
              fontsize=10, y=1.03, fontweight="bold")
savefig(fig1, "04_math_ssot")
plt.close(fig1)

# =====================================================================
# FIGURE 2: RELIC Type System
# =====================================================================
fig2 = plt.figure(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45))
gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.35, width_ratios=[1, 1, 1.2])

# Panel (a): VariableKind bar chart
ax = fig2.add_subplot(gs2[0, 0])
vk_names = [vk.name for vk in VariableKind]
vk_counts = [sum(1 for v in variables if v["kind"] == vk) for vk in VariableKind]
colors = [kind_colors.get(vk, "#999") for vk in VariableKind]
bars = ax.barh(vk_names, vk_counts, color=colors, edgecolor="white", height=0.5)
ax.set_xlabel("Count in Contract $\\Gamma$")
ax.set_title("RELIC VariableKind", fontsize=9, pad=6)
for bar, c in zip(bars, vk_counts):
    if c > 0:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(c), va="center", fontsize=7, fontweight="bold")
add_panel_label(ax, "a", x=-0.2, y=1.05)

# Panel (b): TemporalScale cascade
ax = fig2.add_subplot(gs2[0, 1])
ts_names = []
ts_times = []
for ts in TemporalScale:
    if ts is TemporalScale.UNKNOWN:
        continue  # undeclared scale has no characteristic period by contract
    ts_names.append(ts.name)
    ts_times.append(ts.to_characteristic_time_days(period_days=EXAMPLE_PERIOD_DAYS.get(ts)))
ax.barh(ts_names, ts_times, color=CB_PALETTE[3], edgecolor="white", height=0.4)
ax.set_xlabel("Characteristic Time [days]")
ax.set_title("RELIC TemporalScale", fontsize=9, pad=6)
ax.set_xscale("symlog", linthresh=0.1)
for i, t in enumerate(ts_times):
    ax.text(max(t, 0.01) + 0.02, i, f"{t:.1f}d", va="center", fontsize=5.5)
add_panel_label(ax, "b", x=-0.2, y=1.05)

# Panel (c): FunctorMetadata example
ax = fig2.add_subplot(gs2[0, 2])
ax.axis("off")
ax.set_title("RELIC FunctorMetadata", fontsize=9, pad=6)

meta_text = (
    f"name:     {meta.name}\n"
    f"category: {meta.category.value}\n"
    f"scale:    {meta.spatial_scale.value}\n"
    f"inputs:   {meta.input_variables}\n"
    f"outputs:  {meta.output_variables}\n"
    f"descr:    {meta.description}\n"
)
ax.text(0.05, 0.9, meta_text, transform=ax.transAxes,
        fontsize=6.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8",
                  edgecolor="#B0C4DE", alpha=0.95))
add_panel_label(ax, "c", x=-0.05, y=1.05)

fig2.suptitle("RELIC Type System: VariableKind + TemporalScale + FunctorMetadata",
              fontsize=10, y=1.03, fontweight="bold")
savefig(fig2, "04_relic_ssot")
plt.close(fig2)

print("\n[OK] Tutorial 04 completed successfully.")
print("  Fig 1: Mathematical SSOT (Pi + Gamma)")
print("  Fig 2: RELIC type system (VariableKind, TemporalScale, FunctorMetadata)")
