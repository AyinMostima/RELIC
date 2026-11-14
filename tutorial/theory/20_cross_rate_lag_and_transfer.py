r"""Tutorial 20 -- Cross-Rate Lag Bound and Cross-Zone Error Transfer.

Demonstrates
------------
1) Theorem 3.6.7 in docs/source/coertheory.md:
     lag_max <= Delta_t_M^(k),
     max(0, tau_M - dt) <= Delta_t_M^(k) <= tau_M + dt
   (fixed dt, interval-index triggers only; event, bootstrap/reset-first and
   adaptive-retry executions are outside the bound)
2) RELIC runtime checks:
   - _warn_cross_group_dependencies (cross-group lag warnings at every
     provider-view publication: "Cross-group sampling lag" when a slower group
     reads a faster group's output, "Cross-group hold lag" when a faster group
     reads a slower group's output; this script counts the sampling-lag
     messages, and its fixture has no hold-lag read)
   - _check_cross_zone_error_transfer (cross-zone imbalance monitor)

Two figures are produced:
  1) 20_math_cross_rate_lag.png
     Pure mathematical lag-bound verification.
  2) 20_relic_cross_rate_transfer.png
     RELIC empirical warning and transfer-ratio evidence.

References
----------
1. docs/source/coertheory.md, Theorem 3.6.7
2. docs/source/theory/spatiotemporal_scale_topology.md, Sections 5.2 and 5.3
3. relic/core/simulator/execution_loop.py
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
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

from relic.core.simulator import execution_loop
from relic.core.simulator.time_ops import compute_interval_index
from relic.core.state.manager import UpdateItem, UpdateMode


from relic.core.functor.functor_base import FunctorBase
from relic.core.functor.functor_types import FunctorMetadata, ProcessCategory
from relic.core.simulator.main import RELICSimulator

def _build_real_simulator_for_test() -> RELICSimulator:
    """Build a real RELICSimulator populated with real FunctorBase instances."""
    sim = RELICSimulator({"time_step": 3600.0})
    sim._resolve_cross_group_deps = False
    sim._order_slow = ["SlowBiogeochem"]
    sim._order_medium = ["MediumTransport"]
    sim._groups = {
        "fast": ["FastHydro", "FastAtmo"],
        "medium": ["MediumTransport"],
        "slow": ["SlowBiogeochem"],
    }
    
    class TutorialFunctor(FunctorBase):
        def __init__(self, name, inputs, outputs):
            meta = FunctorMetadata(
                name=name,
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                description="Test real functor",
                references=[],
                input_variables=inputs,
                output_variables=outputs,
            )
            super().__init__(name=name, config={}, metadata=meta)
            
        def get_dependencies(self) -> list:
            return self._merge_declared_dependencies()
            
        def forward(self, state, **kwargs):
            return {}
            
    import torch.nn as nn
    sim.functors = nn.ModuleDict({
        "FastHydro": TutorialFunctor("FastHydro", ["precip"], ["soil_moisture"]),
        "FastAtmo": TutorialFunctor("FastAtmo", ["radiation"], ["surface_temp"]),
        "MediumTransport": TutorialFunctor("MediumTransport", ["soil_moisture"], ["humidity_tendency"]),
        "SlowBiogeochem": TutorialFunctor("SlowBiogeochem", ["soil_moisture", "surface_temp"], ["soil_respiration"]),
    })
    
    sim._step_update_sources = {
        "soil_moisture": ("FastHydro", UpdateMode.ACCUMULATE),
        "soil_respiration": ("SlowBiogeochem", UpdateMode.ACCUMULATE),
    }
    sim._step_functor_increments = {
        "soil_moisture": {
            "FastHydro": torch.tensor([0.90, 0.72]),
            "MediumTransport": torch.tensor([0.12, 0.09]),
            "SlowBiogeochem": torch.tensor([0.06, 0.04]),
        },
        "soil_respiration": {
            "SlowBiogeochem": torch.tensor([0.80, 0.76]),
            "FastAtmo": torch.tensor([0.18, 0.15]),
        },
    }
    return sim


def _compute_medium_trigger_intervals(
    dt_seconds: float,
    tau_m_seconds: float,
    horizon_seconds: float,
) -> np.ndarray:
    """Compute effective medium trigger intervals Delta_t_M^(k)."""
    # References:
    # - docs/source/coertheory.md (Theorem 3.6.7)
    # - relic/core/simulator/time_ops.py::compute_interval_index
    # Math expression:
    #   Delta_t_M^(k) = t_k - t_{k-1}
    # Parameter meaning:
    #   dt_seconds: global step size.
    #   tau_m_seconds: configured medium interval.
    #   horizon_seconds: total simulated horizon.
    prev_idx = compute_interval_index(0.0, tau_m_seconds)
    trigger_times: List[float] = []
    t = 0.0
    while t < horizon_seconds:
        t += dt_seconds
        idx = compute_interval_index(t, tau_m_seconds)
        if idx > prev_idx:
            trigger_times.append(t)
            prev_idx = idx
    if not trigger_times:
        return np.array([], dtype=float)
    with_t0 = np.array([0.0] + trigger_times, dtype=float)
    return np.diff(with_t0)


def _capture_stale_read_warnings(sim: RELICSimulator) -> Dict[str, int]:
    """Run _warn_cross_group_dependencies and count sampling-lag warnings by variable."""
    logger = logging.getLogger("relic.core.simulator.execution_loop")
    records: List[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "Cross-group sampling lag" in message:
                records.append(message)

    handler = _Capture()
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        execution_loop._warn_cross_group_dependencies(sim)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    counter: Dict[str, int] = {}
    pattern = re.compile(r"reads '([^']+)' produced")
    for msg in records:
        match = pattern.search(msg)
        key = match.group(1) if match else "unknown"
        counter[key] = counter.get(key, 0) + 1
    return counter


def _run_cross_zone_transfer(sim: RELICSimulator) -> Dict[str, Any]:
    """Run _check_cross_zone_error_transfer with explicit monitoring enabled."""
    # References:
    # - relic/core/simulator/execution_loop.py::_check_cross_zone_error_transfer
    # Math expression:
    #   ratio = max(zone_contrib) / sum(zone_contrib)
    # Parameter meaning:
    #   threshold: warning threshold for dominant-zone transfer.
    all_updates = {
        "soil_moisture": UpdateItem(torch.tensor([1.08, 0.85]), UpdateMode.ACCUMULATE),
        "soil_respiration": UpdateItem(torch.tensor([0.98, 0.91]), UpdateMode.ACCUMULATE),
    }
    old_get_parameter = execution_loop._get_temporal_zone_parameter
    try:
        # _get_temporal_zone_parameter(key) reads a required YAML parameter; the
        # tutorial enables monitoring with threshold 0.72 and delegates any other key.
        execution_loop._get_temporal_zone_parameter = lambda key: (
            True
            if key == "computation.temporal_zones.monitoring_enabled"
            else 0.72
            if key == "computation.temporal_zones.cross_transfer_threshold"
            else old_get_parameter(key)
        )
        result = execution_loop._check_cross_zone_error_transfer(sim, all_updates, dt=3600.0)
    finally:
        execution_loop._get_temporal_zone_parameter = old_get_parameter
    return result


def main() -> None:
    print("=== Tutorial 20: Cross-rate lag and cross-zone transfer ===")

    # ------------------------------------------------------------------
    # Math-side lag bound verification
    # ------------------------------------------------------------------
    dt = 4000.0
    tau_m = 21600.0
    intervals = _compute_medium_trigger_intervals(
        dt_seconds=dt,
        tau_m_seconds=tau_m,
        horizon_seconds=5.0 * 86400.0,
    )
    lower = max(0.0, tau_m - dt)
    upper = tau_m + dt
    lag_upper_ok = bool(np.all((intervals >= lower - 1e-9) & (intervals <= upper + 1e-9)))
    lag_max = float(intervals.max()) if intervals.size > 0 else 0.0
    print(
        f"  Medium interval bounds: [{lower:.1f}, {upper:.1f}] s | "
        f"observed max Delta_t_M={lag_max:.1f} s | within bound={lag_upper_ok}"
    )

    # ------------------------------------------------------------------
    # RELIC-side cross-group sampling-lag and cross-zone diagnostics
    # ------------------------------------------------------------------
    sim = _build_real_simulator_for_test()
    stale_counter = _capture_stale_read_warnings(sim)
    transfer_result = _run_cross_zone_transfer(sim)
    violations = transfer_result.get("violations", []) if isinstance(transfer_result, dict) else []
    print(f"  Cross-group sampling-lag warning variables: {stale_counter}")
    print(f"  Cross-zone violations: {len(violations)}")

    # ------------------------------------------------------------------
    # Figure 1: pure mathematics
    # ------------------------------------------------------------------
    apply_nature_style()
    fig1, axes1 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    ax = axes1[0]
    x = np.arange(1, intervals.size + 1)
    ax.plot(x, intervals, marker="o", color=CB_PALETTE[0], linewidth=1.4, label="Delta_t_M^(k)")
    ax.axhline(lower, color=CB_PALETTE[1], linestyle="--", linewidth=1.0, label="lower bound")
    ax.axhline(upper, color=CB_PALETTE[2], linestyle="--", linewidth=1.0, label="upper bound")
    ax.set_xlabel("Trigger index k")
    ax.set_ylabel("Effective interval [s]")
    ax.set_title("Theorem 3.6.7 Interval Bounds", fontsize=9)
    ax.legend(fontsize=6, loc="lower right", frameon=False, prop={"weight": "bold"})
    add_panel_label(ax, "a")

    # Panel b: bar chart comparing timing parameters and bounds
    ax = axes1[1]
    param_names = [r"$\tau_M$", r"$\Delta t$", "lower\nbound", "upper\nbound", "observed\nlag_max"]
    param_vals = [tau_m, dt, lower, upper, lag_max]
    bar_colors = [CB_PALETTE[0], CB_PALETTE[2], CB_PALETTE[1], CB_PALETTE[1], CB_PALETTE[4]]
    bars = ax.bar(param_names, param_vals, color=bar_colors, alpha=0.85, edgecolor="none")
    # Annotate values
    for bar, val in zip(bars, param_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:.0f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.set_ylabel("Time [s]")
    ax.set_title("Lag Bound Parameters", fontsize=9)
    # Add pass/fail annotation
    status_color = CB_PALETTE[2] if lag_upper_ok else CB_PALETTE[1]
    status_text = "PASS" if lag_upper_ok else "FAIL"
    ax.text(0.50, 0.86, status_text, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="center", va="top", color=status_color)
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig1.suptitle("Pure Mathematics: Cross-Rate Lag Upper Bound", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig1, "20_math_cross_rate_lag")
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Figure 2: RELIC empirical diagnostics
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.42))

    ax = axes2[0]
    if stale_counter:
        names = list(stale_counter.keys())
        vals = [stale_counter[name] for name in names]
        ax.bar(names, vals, color=CB_PALETTE[3], edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Warning count")
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    else:
        ax.text(0.5, 0.5, "No sampling-lag warning", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("_warn_cross_group_dependencies", fontsize=9)
    add_panel_label(ax, "a")

    ax = axes2[1]
    if violations:
        vars_v = [item["variable"] for item in violations]
        ratios = [float(item["cross_transfer_ratio"]) for item in violations]
        threshold = float(violations[0]["threshold"])
        ax.bar(vars_v, ratios, color=CB_PALETTE[0], alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(threshold, color=CB_PALETTE[1], linestyle="--", linewidth=1.0, label="threshold")
        ax.set_ylabel("cross_transfer_ratio")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=6, loc="upper right", frameon=False, prop={"weight": "bold"})
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    else:
        ax.text(0.5, 0.5, "No cross-zone violation", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("_check_cross_zone_error_transfer", fontsize=9)
    add_panel_label(ax, "b", x=-0.06, y=1.02)

    fig2.suptitle("RELIC Evidence: Cross-Group Sampling-Lag Warnings and Cross-Zone Transfer Alerts", fontsize=10, y=1.03)
    plt.tight_layout()
    savefig(fig2, "20_relic_cross_rate_transfer")
    plt.close(fig2)

    print("[OK] Tutorial 20 finished.")


if __name__ == "__main__":
    main()
