r"""Tutorial 22 -- Scale Consistency, NUOPC Clock Negotiation, Capability Registry.

Demonstrates
------------
This tutorial closes the protocol/governance gaps in spatiotemporal theory:
  - Section 9.2: scale consistency protocol (metadata-first detection)
  - Section 10.2: functor summary and introspection
  - Section 11.1: NUOPC-compatible clock negotiation
  - Section 11.2: capability registry

Two figures are produced:
  1) 22_math_scale_nuopc.png
     Pure mathematics for scale selection and clock negotiation.
  2) 22_relic_scale_nuopc_capability.png
     RELIC empirical evidence from runtime APIs.

References
----------
1. docs/source/theory/spatiotemporal_scale_topology.md (9.2, 10.2, 11.1, 11.2)
2. relic/core/simulator/scale_ops.py
3. relic/coupling/protocol/time_stagger.py
4. relic/coupling/capability_registry.py
5. relic/core/simulator/main.py (get_functor_summary, get_variable_provenance)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

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

from relic.core.functor.functor_types import ProcessCategory, SpatialScale, VariableKind
from relic.core.simulator import scale_ops
from relic.core.simulator.main import RELICSimulator
from relic.core.state.manager import UpdateMode
from relic.coupling.capability_registry import CouplingCapabilityRegistry
from relic.coupling.protocol.time_stagger import ClockConfig, TimeStagger, TimeUnit


@dataclass
class _ScaleMeta:
    """Minimal metadata for scale consistency demo."""

    spatial_scale: Any
    output_variables: List[str]


class _ScaleFunctor:
    """Minimal functor wrapper for scale metadata."""

    def __init__(self, spatial_scale: Any, outputs: List[str]) -> None:
        self.metadata = _ScaleMeta(spatial_scale=spatial_scale, output_variables=outputs)


@dataclass
class _SummaryMeta:
    """Minimal metadata for summary/provenance demo."""

    category: Any
    input_variables: List[str]
    output_variables: List[str]
    spatial_scale: Any
    variable_kind: Any
    version: str
    description: str


class _SummaryFunctor:
    """Minimal functor wrapper for summary API."""

    def __init__(self, meta: _SummaryMeta) -> None:
        self.metadata = meta


def _build_scale_simulator() -> Any:
    """Construct minimal simulator state for check_scale_consistency."""
    sim = SimpleNamespace()
    sim.enable_auto_scale = True
    sim.scale_manager = object()
    sim.strict_scale_detection = False
    sim.current_time = 86400.0
    sim._scale_check_history = []
    sim.functors = {
        "PatchThermalFunctor": _ScaleFunctor(SpatialScale.PATCH, ["soil_patch_temp"]),
        "GridHydroFunctor": _ScaleFunctor(SpatialScale.GRID_2D, ["soil_water_grid"]),
    }
    return sim


def _run_scale_consistency() -> Dict[str, Any]:
    """Run metadata-first scale consistency check."""
    # References:
    # - scale_ops.check_scale_consistency
    # Math expression:
    #   target_scale = argmax_{s in U} p(s)
    # Parameter meaning:
    #   detected scales come from metadata first, heuristic fallback second.
    sim = _build_scale_simulator()
    state = {
        # Shape suggests grid_2d, but metadata explicitly sets patch.
        "soil_patch_temp": torch.ones((64, 64), dtype=torch.float32),
        "soil_water_grid": torch.ones((32, 32), dtype=torch.float32),
        "aux_scalar": torch.tensor(1.0),
    }
    return scale_ops.check_scale_consistency(sim, state, force=False)


def _run_nuopc_negotiation() -> Dict[str, Any]:
    """Run TimeStagger negotiation and subcycle checks."""
    # References:
    # - coupling/protocol/time_stagger.py
    # Math expression:
    #   dt_coupled = LCM(dt_1, dt_2, ..., dt_n)
    # Parameter meaning:
    #   dt_i are component clocks in seconds.
    ts = TimeStagger(lag_seconds=300.0, lead_seconds=60.0)
    ts.register_clock("atmosphere", ClockConfig(900.0, unit=TimeUnit.SECONDS))
    ts.register_clock("land", ClockConfig(1800.0, unit=TimeUnit.SECONDS))
    ts.register_clock("ocean", ClockConfig(2700.0, unit=TimeUnit.SECONDS))
    dt_coupled = ts.negotiate(strategy="lcm")
    ratios = {
        "atmosphere": ts.get_subcycle_ratio("atmosphere"),
        "land": ts.get_subcycle_ratio("land"),
        "ocean": ts.get_subcycle_ratio("ocean"),
    }
    warnings = ts.validate_temporal_consistency()
    return {"dt_coupled": dt_coupled, "ratios": ratios, "warnings": warnings}


def _run_summary_and_provenance() -> Dict[str, Any]:
    """Run get_functor_summary and get_variable_provenance on dummy simulator object."""
    summary_sim = SimpleNamespace()
    summary_sim.functors = {
        "FastHydro": _SummaryFunctor(
            _SummaryMeta(
                category=ProcessCategory.PHYSICAL_FIELD,
                input_variables=["precip"],
                output_variables=["soil_moisture"],
                spatial_scale=SpatialScale.GRID_2D,
                variable_kind=VariableKind.STATE,
                version="1.0",
                description="Hydrology fast process",
            )
        ),
        "SlowBiogeochem": _SummaryFunctor(
            _SummaryMeta(
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                input_variables=["soil_moisture"],
                output_variables=["soil_respiration"],
                spatial_scale=SpatialScale.PATCH,
                variable_kind=VariableKind.FLUX,
                version="1.0",
                description="Biogeochemical slow process",
            )
        ),
    }
    summary_sim._groups = {
        "fast": ["FastHydro"],
        "medium": [],
        "slow": ["SlowBiogeochem"],
    }
    summary_sim._step_provenance_detail = {
        "soil_moisture": [
            ("FastHydro", UpdateMode.ACCUMULATE, 42, "writer"),
        ]
    }
    summary_sim._step_functor_increments = {
        "soil_moisture": {
            "FastHydro": torch.tensor([0.3, 0.2]),
        }
    }
    summary_sim._step_update_sources = {
        "soil_moisture": ("FastHydro", UpdateMode.ACCUMULATE),
    }

    functor_summary = RELICSimulator.get_functor_summary(summary_sim)
    provenance = RELICSimulator.get_variable_provenance(summary_sim, "soil_moisture")
    return {"functor_summary": functor_summary, "provenance": provenance}


def main() -> None:
    print("=== Tutorial 22: Scale consistency, NUOPC, capability registry ===")

    scale_check = _run_scale_consistency()
    nuopc = _run_nuopc_negotiation()
    summary_data = _run_summary_and_provenance()
    capability_status = CouplingCapabilityRegistry.get_implementation_status()

    print(f"  Scale check needs_regrid: {scale_check.get('needs_regrid')}")
    print(f"  Detected scales: {scale_check.get('detected_scales')}")
    print(f"  Target scale: {scale_check.get('target_scale')}")
    print(f"  NUOPC negotiated dt: {nuopc['dt_coupled']} s")
    print(f"  NUOPC subcycle ratios: {nuopc['ratios']}")
    print(
        "  Capability counts: "
        f"implemented={len(capability_status.get('implemented', []))}, "
        f"partial={len(capability_status.get('partial', []))}, "
        f"not_implemented={len(capability_status.get('not_implemented', []))}"
    )
    print(f"  Functor summary keys: {list(summary_data['functor_summary'].keys())}")

    # ------------------------------------------------------------------
    # Figure 1: pure mathematics
    # ------------------------------------------------------------------
    apply_nature_style()
    fig1, axes1 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    ax = axes1[0]
    detected = scale_check.get("detected_scales", {})
    scale_values = list(detected.values())
    prefs = [scale_ops._SCALE_PREFERENCE.get(s, -999) for s in scale_values]
    vars_list = list(detected.keys())
    ax.bar(vars_list, prefs, color=CB_PALETTE[0], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("scale preference p(s)")
    ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    ax.set_title("Scale Selection argmax p(s)", fontsize=9)
    add_panel_label(ax, "a")

    # Panel b: NUOPC clock negotiation bar chart
    ax = axes1[1]
    comp_names = ["atmosphere", "land", "ocean", "LCM\n(dt_coupled)"]
    comp_dts = [900.0, 1800.0, 2700.0, nuopc["dt_coupled"]]
    comp_colors = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[3], CB_PALETTE[4]]
    bars_nuopc = ax.bar(comp_names, comp_dts, color=comp_colors, alpha=0.85, edgecolor="none")
    for bar_n, val in zip(bars_nuopc, comp_dts):
        ax.text(bar_n.get_x() + bar_n.get_width() / 2, bar_n.get_height() + 50,
                f"{val:.0f}s", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.set_ylabel("Timestep [s]")
    ax.set_title("NUOPC Clock Negotiation", fontsize=9)
    # Add subcycle ratios as annotation
    for i, (comp, ratio) in enumerate(nuopc["ratios"].items()):
        ax.text(i, comp_dts[i] / 2, f"x{ratio}", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig1.suptitle("Pure Mathematics: Scale Protocol and Clock Negotiation", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig1, "22_math_scale_nuopc")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: RELIC empirical evidence
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.44))

    # Panel a: Scale consistency detection -- demonstrate argmax p(s)
    # Shows how RELIC detects spatial scales per-variable and resolves target_scale.
    ax = axes2[0]
    detected = scale_check.get("detected_scales", {})
    target = scale_check.get("target_scale", "unknown")
    needs_regrid = scale_check.get("needs_regrid", False)
    det_vars = list(detected.keys())
    det_scales = [str(detected[v]) for v in det_vars]
    det_prefs = [scale_ops._SCALE_PREFERENCE.get(detected[v], 0) for v in det_vars]
    bar_colors_sc = [CB_PALETTE[2] if str(detected[v]) == str(target) else CB_PALETTE[0]
                     for v in det_vars]
    bars_sc = ax.bar(det_vars, det_prefs, color=bar_colors_sc, alpha=0.85, edgecolor="none")
    # Annotate detected scale name on each bar
    for bar_sc, scale_name in zip(bars_sc, det_scales):
        ax.text(bar_sc.get_x() + bar_sc.get_width() / 2, bar_sc.get_height() + 0.1,
                scale_name.split(".")[-1], ha="center", va="bottom", fontsize=5.5,
                fontweight="bold", rotation=15)
    ax.set_ylabel("Scale preference p(s)")
    ax.tick_params(axis="x", labelrotation=20, labelsize=6)
    ax.set_title("RELIC Scale Detection: argmax$\\to$" + str(target).split(".")[-1], fontsize=8)
    # Add regrid flag
    regrid_text = "needs_regrid = True" if needs_regrid else "needs_regrid = False"
    ax.text(0.02, 0.98, regrid_text, transform=ax.transAxes, fontsize=6,
            fontweight="bold", va="top", color=CB_PALETTE[1] if needs_regrid else CB_PALETTE[2])
    add_panel_label(ax, "a")

    # Panel b: NUOPC clock negotiation -- demonstrate dt_coupled = LCM(dt_i)
    # Shows the same LCM principle from math figure, now executed via RELIC TimeStagger.
    ax = axes2[1]
    comp_names_r = list(nuopc["ratios"].keys())
    comp_dts_r = [900.0, 1800.0, 2700.0]  # registered clocks
    dt_coupled_r = nuopc["dt_coupled"]
    subcycle_ratios_r = [nuopc["ratios"][c] for c in comp_names_r]
    x_comp = np.arange(len(comp_names_r))
    width_comp = 0.38
    ax.bar(x_comp - width_comp / 2, comp_dts_r, width_comp, color=CB_PALETTE[0],
           alpha=0.85, label="Component $dt_i$")
    ax.bar(x_comp + width_comp / 2, [dt_coupled_r] * len(comp_names_r), width_comp,
           color=CB_PALETTE[4], alpha=0.70, label=f"LCM = {dt_coupled_r:.0f}s")
    # Annotate subcycle ratios
    for i, ratio in enumerate(subcycle_ratios_r):
        ax.text(i, comp_dts_r[i] + 40, f"x{ratio}", ha="center", va="bottom",
                fontsize=7, fontweight="bold", color=CB_PALETTE[0])
    ax.set_xticks(x_comp)
    ax.set_xticklabels(comp_names_r, fontsize=7)
    ax.set_ylabel("Timestep [s]")
    ax.set_title("RELIC TimeStagger: LCM Negotiation", fontsize=9)
    n_warn = len(nuopc["warnings"])
    ax.text(0.98, 0.98, f"warnings: {n_warn}", transform=ax.transAxes, fontsize=6,
            ha="right", va="top", fontweight="bold",
            color=CB_PALETTE[2] if n_warn == 0 else CB_PALETTE[1])
    ax.legend(fontsize=5, ncol=1, loc="upper left", frameon=False, prop={"weight": "bold"})
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig2.suptitle("RELIC Evidence: Scale Detection and Clock Negotiation", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig2, "22_relic_scale_nuopc_capability")
    plt.close(fig2)

    print("[OK] Tutorial 22 finished.")


if __name__ == "__main__":
    main()
