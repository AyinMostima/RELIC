"""One-click pipeline runner for tutorial/case with progress and sanity checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import xarray as xr

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full tutorial/case pipeline in one command.")
    parser.add_argument(
        "--case-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Case root directory (default: tutorial/case).",
    )
    parser.add_argument(
        "--run-id-prefix",
        type=str,
        default="full_pipeline",
        help="Prefix for generated run ids.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip external data download stage.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip NetCDF preparation stage.",
    )
    parser.add_argument(
        "--skip-simulate",
        action="store_true",
        help="Skip 2-day sanity simulation stage.",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip optimization stage.",
    )
    parser.add_argument(
        "--opt-phases",
        type=str,
        default=None,
        help="Optional comma-separated optimization phases passed to run_case.py.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def run_command(command: List[str], cwd: Path, emit: Callable[[str], None]) -> None:
    emit(f"[INFO] cmd: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        emit(line.rstrip("\n"))
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def check_reasonableness(run_dir: Path, emit: Callable[[str], None]) -> None:
    state_path = run_dir / "timeseries" / "state_hourly.nc"
    if not state_path.exists():
        emit(f"[WARN] missing state file: {state_path}")
        return

    ds = xr.load_dataset(state_path)
    time = pd.to_datetime(ds["time"].values)
    hour = time.hour

    checks = []
    for key in [
        "soil_moisture",
        "net_photosynthesis",
        "temperature_stress_factor",
        "ozone_stress_factor",
        "water_stress_factor",
        "isoprene_emission",
    ]:
        if key not in ds:
            checks.append((f"{key}_exists", False, "missing variable"))
            continue
        values = np.asarray(ds[key].values, dtype=float)
        finite = bool(np.isfinite(values).all())
        checks.append((f"{key}_finite", finite, f"nan={int(np.isnan(values).sum())}, inf={int(np.isinf(values).sum())}"))

    if "isoprene_emission" in ds:
        iso = np.asarray(ds["isoprene_emission"].values, dtype=float)
        day_mask = (hour >= 10) & (hour <= 16)
        night_mask = (hour <= 5) | (hour >= 20)
        day_mean = float(np.nanmean(iso[day_mask])) if bool(day_mask.any()) else float("nan")
        night_mean = float(np.nanmean(iso[night_mask])) if bool(night_mask.any()) else float("nan")
        ratio = day_mean / max(night_mean, 1.0e-12)
        checks.append(("isoprene_day_night_ratio_gt_1", bool(ratio > 1.0), f"ratio={ratio:.3f}"))

    for key in ["temperature_stress_factor", "ozone_stress_factor", "water_stress_factor"]:
        if key in ds:
            values = np.asarray(ds[key].values, dtype=float)
            within = bool((np.nanmin(values) >= 0.0) and (np.nanmax(values) <= 1.05))
            checks.append((f"{key}_in_0_1", within, f"min={np.nanmin(values):.4f}, max={np.nanmax(values):.4f}"))

    if "net_photosynthesis" in ds:
        values = np.asarray(ds["net_photosynthesis"].values, dtype=float)
        checks.append(("net_photosynthesis_peak_lt_50", bool(np.nanmax(values) < 50.0), f"max={np.nanmax(values):.4f}"))

    for name, ok, detail in checks:
        tag = "[OK]" if ok else "[WARN]"
        emit(f"{tag} {name}: {detail}")


def main() -> None:
    args = parse_args()
    case_root = args.case_root.resolve()
    data_dir = case_root / "data"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    simulate_run_id = f"{args.run_id_prefix}_simulate_{timestamp}"
    optimize_run_id = f"{args.run_id_prefix}_optimize_{timestamp}"
    log_dir = case_root / "output" / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.run_id_prefix}_{timestamp}.txt"

    log_handle = log_path.open("w", encoding="utf-8")

    def emit(message: str) -> None:
        text = str(message)
        print(text)
        log_handle.write(text + "\n")
        log_handle.flush()

    stages = []
    if not args.skip_download:
        stages.append("download")
    if not args.skip_prepare:
        stages.append("prepare")
    if not args.skip_simulate:
        stages.append("simulate")
    if not args.skip_optimize:
        stages.append("optimize")

    if len(stages) == 0:
        print("[WARN] nothing to run")
        return

    progress = None
    if not args.no_progress and tqdm is not None:
        progress = tqdm(total=len(stages), desc="pipeline", dynamic_ncols=True)

    total_stages = len(stages)
    stage_index = 0

    success = False
    try:
        if "download" in stages:
            stage_index += 1
            stage_start = time.perf_counter()
            emit(f"[PROGRESS] {stage_index}/{total_stages} stage=download status=start")
            emit("[INFO] stage download: external real data")
            run_command(
                [sys.executable, str(data_dir / "download_external_data.py"), "--case-root", str(case_root)],
                cwd=case_root,
                emit=emit,
            )
            emit(f"[PROGRESS] {stage_index}/{total_stages} stage=download status=done elapsed={time.perf_counter() - stage_start:.2f}s")
            if progress is not None:
                progress.update(1)

        if "prepare" in stages:
            stage_index += 1
            stage_start = time.perf_counter()
            emit(f"[PROGRESS] {stage_index}/{total_stages} stage=prepare status=start")
            emit("[INFO] stage prepare: rebuild processed NetCDF from latest raw data")
            run_command(
                [sys.executable, str(data_dir / "prepare_netcdf.py"), "--case-root", str(case_root)],
                cwd=case_root,
                emit=emit,
            )
            emit(f"[PROGRESS] {stage_index}/{total_stages} stage=prepare status=done elapsed={time.perf_counter() - stage_start:.2f}s")
            if progress is not None:
                progress.update(1)

        if "simulate" in stages:
            stage_index += 1
            stage_start = time.perf_counter()
            emit(f"[PROGRESS] {stage_index}/{total_stages} stage=simulate status=start")
            emit("[INFO] stage simulate: 2-day sanity run")
            run_command(
                [
                    sys.executable,
                    str(case_root / "run_case.py"),
                    "--case-root",
                    str(case_root),
                    "--mode",
                    "simulate",
                    "--days",
                    "2",
                    "--run-id",
                    simulate_run_id,
                ],
                cwd=case_root,
                emit=emit,
            )
            check_reasonableness(case_root / "output" / "runs" / simulate_run_id, emit)
            emit(f"[PROGRESS] {stage_index}/{total_stages} stage=simulate status=done elapsed={time.perf_counter() - stage_start:.2f}s")
            if progress is not None:
                progress.update(1)

        if "optimize" in stages:
            stage_index += 1
            stage_start = time.perf_counter()
            emit(f"[PROGRESS] {stage_index}/{total_stages} stage=optimize status=start")
            emit("[INFO] stage optimize: full optimization")
            cmd = [
                sys.executable,
                str(case_root / "run_case.py"),
                "--case-root",
                str(case_root),
                "--mode",
                "optimize",
                "--run-id",
                optimize_run_id,
            ]
            if args.opt_phases:
                cmd.extend(["--opt-phases", args.opt_phases])
            if args.no_progress:
                cmd.append("--no-progress")
            run_command(cmd, cwd=case_root, emit=emit)
            check_reasonableness(case_root / "output" / "runs" / optimize_run_id, emit)
            emit(f"[PROGRESS] {stage_index}/{total_stages} stage=optimize status=done elapsed={time.perf_counter() - stage_start:.2f}s")
            if progress is not None:
                progress.update(1)
        success = True

    finally:
        if progress is not None:
            progress.close()
        emit(f"[INFO] pipeline log file: {log_path}")
        if success:
            emit("[OK] pipeline complete")
            emit(f"[OK] simulate run id: {simulate_run_id}")
            emit(f"[OK] optimize run id: {optimize_run_id}")
        log_handle.close()


if __name__ == "__main__":
    main()
