#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone script to perform post-processing conservation audits on RELIC outputs.
Summarizes recorded canonical bucket diagnostics; missing budgets remain unevaluated.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tutorial.case.utils.time_alignment import align_frame_to_reference_time


def setup_matplotlib():
    """Setup matplotlib style for professional scientific plotting."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def audit_run_directory(
    run_dir: Path,
    output_dir: Path,
    *,
    epoch: int | None = None,
    phase_id: int | None = None,
) -> dict:
    """Summarize recorded bucket diagnostics; do not certify model conservation."""
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("Use a separate evidence output directory outside the source run")
    ts_dir = run_dir / "timeseries"
    state_path = ts_dir / "state_hourly_physics.nc"
    diag_path = ts_dir / "diagnostics_hourly.nc"
    config_path = run_dir / "input_snapshot" / "simulation.yaml"
    for path in (state_path, diag_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required recorded input missing: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("simulation"), dict):
        raise ValueError("Snapshot simulation.yaml requires a simulation mapping")
    simulation = config["simulation"]
    dt = float(simulation["dt_seconds"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Snapshot dt_seconds must be finite and positive")
    with xr.open_dataset(state_path) as source:
        state = source.load()
    with xr.open_dataset(diag_path) as source:
        diagnostics = source.load()
    for name, ds in (("state", state), ("diagnostics", diagnostics)):
        for coordinate in ("time", "step", "epoch", "phase_id"):
            if coordinate not in ds.coords or ds[coordinate].dims != ("sample",):
                raise ValueError(f"{name} requires one-dimensional {coordinate} records")
        if not np.issubdtype(ds["time"].dtype, np.datetime64):
            raise ValueError(f"{name} time must decode as datetime64")
    mask = np.ones(state.sizes["sample"], dtype=bool)
    if epoch is not None:
        mask &= state["epoch"].values == epoch
    if phase_id is not None:
        mask &= state["phase_id"].values == phase_id
    state = state.isel(sample=np.flatnonzero(mask))
    groups = np.unique(np.column_stack([state["epoch"].values, state["phase_id"].values]), axis=0)
    if len(groups) != 1:
        raise ValueError("Select exactly one nonempty epoch/phase with --epoch and --phase-id")
    selected_epoch, selected_phase = (int(value) for value in groups[0])
    diagnostics = diagnostics.isel(sample=np.flatnonzero(
        (diagnostics["epoch"].values == selected_epoch)
        & (diagnostics["phase_id"].values == selected_phase)
    ))
    state_frame = pd.DataFrame(
        {key: state[key].values for key in state.data_vars},
        index=pd.DatetimeIndex(state["time"].values),
    )
    diagnostic_frame = pd.DataFrame(
        {key: diagnostics[key].values for key in diagnostics.data_vars},
        index=pd.DatetimeIndex(diagnostics["time"].values),
    )
    diagnostic_frame = align_frame_to_reference_time(
        state_frame, diagnostic_frame, frame_name="recorded diagnostics", require_complete=True,
    )
    diagnostic_steps = pd.Series(diagnostics["step"].values, index=pd.DatetimeIndex(diagnostics["time"].values))
    steps = state["step"].values
    if not np.array_equal(diagnostic_steps.reindex(state_frame.index).values, steps):
        raise ValueError("State and diagnostic step indices differ")
    if len(steps) > 1:
        if not np.all(np.diff(steps) == 1):
            raise ValueError("Selected records must have consecutive step indices")
        elapsed = np.diff(state_frame.index.asi8) / pd.Timedelta(seconds=1).value
        if not np.allclose(elapsed, dt, rtol=0, atol=1e-6):
            raise ValueError("Recorded intervals do not match snapshot dt_seconds")

    owner = "relic/scientific/hydrosphere/soil_moisture/bucket.py::BucketSoilMoisture.forward"
    prefix = "diagnostics.soil_moisture_bucket."
    required = {
        "storage": (diagnostic_frame, diagnostics, prefix + "soil_water_storage", "kg m-2"),
        "actual_et": (diagnostic_frame, diagnostics, prefix + "adjusted_evapotranspiration", "kg m-2 s-1"),
        "input": (state_frame, state, "infiltration_flux", "kg m-2 s-1"),
        "runoff": (state_frame, state, "surface_runoff_flux", "kg m-2 s-1"),
        "drainage": (state_frame, state, "drainage_flux", "kg m-2 s-1"),
    }
    report = {
        "scope": "recorded single-bucket diagnostic consistency; no whole-model certification",
        "run_dir": str(run_dir),
        "epoch": selected_epoch,
        "phase_id": selected_phase,
        "record_count": len(steps),
        "dt_seconds": dt,
        "water": {"status": "NOT_EVALUATED", "reasons": []},
        "carbon": {"status": "NOT_EVALUATED", "reason": "No complete carbon stock and routed boundary budget is reconstructed."},
        "runtime_unit_coverage_missing": [],
        "limitations": [
            "run_case.build_simulator sets conservation_checks=False and clears Functor conservation_laws; this postprocessing does not restore those checks.",
            "Bucket storage is a recorded process diagnostic, not an independent reconstruction of the complete committed StateManager inventory.",
            "Records are post-step values stamped with loader step labels. The first record is the reference; its preceding interval is excluded.",
            "SurfaceRunoff external pre-partition and canopy interception are outside the bucket boundary; neither gross rain nor external runoff is inserted into this bucket ledger.",
            "Missing boundary terms are not replaced by zero; reported residuals alone do not prove physical closure or identify a leak.",
        ],
        "source_contract": owner,
    }
    values = {}
    for label, (frame, ds, key, expected_unit) in required.items():
        if key not in frame:
            report["water"]["reasons"].append(f"Missing {key}")
            continue
        data = frame[key].to_numpy(dtype=float)
        if not np.isfinite(data).all():
            report["water"]["reasons"].append(f"Nonfinite {key}")
            continue
        unit = ds[key].attrs.get("units")
        if unit is None and key.startswith(prefix):
            report["runtime_unit_coverage_missing"].append({
                "variable": key, "unit_from_owner": expected_unit, "source": owner,
            })
        elif unit != expected_unit:
            report["water"]["reasons"].append(f"Unsupported/missing recorded units for {key}: {unit!r}; requires {expected_unit}")
            continue
        values[label] = data
    if len(steps) < 2:
        report["water"]["reasons"].append("At least two post-step storage records are needed")
    if simulation.get("adaptive_dt_enabled", False) or simulation.get("splitting_mode", "lie") != "lie":
        report["water"]["reasons"].append("This case postprocessor requires recorded fixed-step Lie intervals")

    # References: Manabe (1969); canonical BucketSoilMoisture.forward above.
    # Mathematical expression: delta_W[j]=W[j]-W[0]; A[j]=sum(i=1..j)(I-Ea-Q-D)[i]*dt;
    # R[j]=delta_W[j]-A[j]. Index 0 is the first post-step reference, not initial W.
    # Parameter meanings: W is recorded bucket water mass; I infiltration, Ea actual ET,
    # Q bucket overflow and D drainage; dt is the snapshotted interval.
    # Units: W, A, R [kg m-2]; I, Ea, Q, D [kg m-2 s-1]; dt [s].
    # Domain/constraints: one point, one continuous phase/epoch, complete finite terms,
    # fixed intervals and canonical diagnostic units; no inferred soil depth or density.
    # Sign convention: infiltration enters; Ea/Q/D leave; positive R is excess recorded
    # storage relative to the integrated net input. No boundary-route claim is inferred.
    if not report["water"]["reasons"]:
        net_rate = values["input"] - values["actual_et"] - values["runoff"] - values["drainage"]
        integrated = np.concatenate(([0.0], np.cumsum(net_rate[1:] * dt)))
        storage_change = values["storage"] - values["storage"][0]
        residual = storage_change - integrated
        report["water"].update({
            "status": "EVALUATED_DIAGNOSTIC_ONLY", "unit": "kg m-2",
            "interval_count": len(steps) - 1,
            "max_abs_residual": float(np.max(np.abs(residual))),
            "final_residual": float(residual[-1]),
            "storage_change": float(storage_change[-1]),
            "integrated_net_input": float(integrated[-1]),
        })
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "record_label_time": state_frame.index,
            "step": steps,
            "storage_kg_m2": values["storage"],
            "net_rate_kg_m2_s": net_rate,
            "storage_change_kg_m2": storage_change,
            "integrated_net_input_kg_m2": integrated,
            "residual_kg_m2": residual,
        }).to_csv(output_dir / "bucket_recorded_budget.csv", index=False)
        setup_matplotlib()
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(state_frame.index, integrated, label="Integrated net input (excludes first interval)")
        axes[0].plot(state_frame.index, storage_change, label="Recorded bucket storage change", linestyle="--")
        axes[0].set_ylabel("Water mass [kg m-2]")
        axes[0].legend()
        axes[1].plot(state_frame.index, residual, label="Recorded diagnostic residual")
        axes[1].axhline(0, color="black", linestyle="--", linewidth=0.5)
        axes[1].set_ylabel("Residual [kg m-2]")
        axes[1].set_xlabel("Post-step record label time")
        fig.suptitle("Recorded bucket budget; not whole-model conservation certification")
        fig.tight_layout()
        fig.savefig(output_dir / "post_process_conservation_audit.png", bbox_inches="tight")
        plt.close(fig)

    # References: BucketSoilMoisture.forward and check_conservation_laws.
    # Mathematical expression: max_abs_c=max_i(abs(c_i)); no new budget law is applied.
    # Parameter meanings: c is the owner's clipped_mass_residual, recorded per step.
    # Units: kg m-2 s-1 from the canonical owner when NetCDF omits the units attribute.
    # Domain/constraints: summarize finite recorded samples only; missing is not zero.
    # Sign convention: retain the owner's signed residual; it is a disclosed diagnostic,
    # not an external mass route and not the sign convention of R reconstructed above.
    clip_key = prefix + "clipped_mass_residual"
    if clip_key in diagnostic_frame:
        clip = diagnostic_frame[clip_key].to_numpy(dtype=float)
        clip_unit = diagnostics[clip_key].attrs.get("units")
        if clip_unit is None:
            report["runtime_unit_coverage_missing"].append({"variable": clip_key, "unit_from_owner": "kg m-2 s-1", "source": owner})
        if np.isfinite(clip).all() and clip_unit in (None, "kg m-2 s-1"):
            report["recorded_canonical_clipped_mass_residual"] = {"unit": "kg m-2 s-1", "max_abs": float(np.max(np.abs(clip))), "min": float(np.min(clip)), "max": float(np.max(clip))}
        else:
            report["recorded_canonical_clipped_mass_residual"] = {"status": "NOT_EVALUATED", "reason": "Nonfinite values or incompatible recorded units"}
    else:
        report["recorded_canonical_clipped_mass_residual"] = {"status": "NOT_EVALUATED", "reason": f"Missing {clip_key}"}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recorded_budget_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[INFO] Water: {report['water']['status']}")
    for reason in report["water"]["reasons"]:
        print(f"[INFO] Water not evaluated: {reason}")
    print("[INFO] Carbon: NOT_EVALUATED; no complete recorded stock/boundary budget")
    print(f"[OK] Recorded diagnostic report written: {output_dir}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize recorded tutorial bucket budgets without conservation certification")
    parser.add_argument("--run-dir", type=Path, required=True, help="Source case run directory (read only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate evidence directory outside the source run")
    parser.add_argument("--epoch", type=int, help="Select one recorded epoch when multiple epochs exist")
    parser.add_argument("--phase-id", type=int, help="Select one recorded phase when multiple phases exist")
    args = parser.parse_args()
    audit_run_directory(args.run_dir, args.output_dir, epoch=args.epoch, phase_id=args.phase_id)
