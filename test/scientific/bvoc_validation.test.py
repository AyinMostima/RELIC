"""
BVOC validation pipeline smoke test.

This test uses synthetic data to verify that:
  - prepare_inputs.py produces a standardized drivers parquet
  - run_validation.py executes end-to-end without errors
  - metrics CSV is generated and contains expected keys

All comments use ASCII only.
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np


def test_bvoc_validation_smoke(tmp_path: Path):
    repo = Path(__file__).resolve().parents[2]
    scripts_dir = repo / "scripts" / "bvoc_validation"
    drivers_csv = tmp_path / "ameriflux_like.csv"
    obs_csv = tmp_path / "obs.csv"
    drivers_parquet = tmp_path / "drivers.parquet"
    outdir = tmp_path / "out"

    # Build synthetic drivers (hourly for 48 hours)
    ts = pd.date_range("2020-06-01", periods=48, freq="h")
    sw_in = np.clip(600.0 * np.sin(np.linspace(0, 2 * np.pi, len(ts))) + 200.0, 0.0, None)
    ta = 20.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, len(ts)))
    swc = 0.25 + 0.02 * np.cos(np.linspace(0, 2 * np.pi, len(ts)))

    df = pd.DataFrame({
        "TIMESTAMP_START": ts,
        "SW_IN": sw_in,
        "TA": ta,
        "SWC": swc,
        "LAI": np.full(len(ts), 3.0),
        "CO2": np.full(len(ts), 410.0),
    })
    df.to_csv(drivers_csv, index=False)

    # Run prepare_inputs
    prep_cmd = [
        sys.executable,
        str(scripts_dir / "prepare_inputs.py"),
        "--input",
        str(drivers_csv),
        "--output",
        str(drivers_parquet),
        "--swc-column",
        "SWC",
        "--lai-column",
        "LAI",
        "--co2-column",
        "CO2",
    ]
    r1 = subprocess.run(prep_cmd, capture_output=True, text=True)
    assert r1.returncode == 0, f"prepare_inputs failed: {r1.stderr}\n{r1.stdout}"
    assert not r1.stderr.strip(), f"prepare_inputs emitted stderr: {r1.stderr}"
    assert drivers_parquet.exists()

    # Create synthetic observations proportional to a simple diurnal curve
    drv = pd.read_parquet(drivers_parquet)
    assert {
        "timestamp",
        "air_temperature",
        "ppfd",
        "leaf_area_index",
        "atmospheric_co2_ppm",
    }.issubset(drv.columns)
    assert "temperature" not in drv.columns
    assert "par" not in drv.columns
    assert "foliar_density" not in drv.columns

    # Proxy predicted emission: linear in PPFD and exponential in temperature.
    y_base = (
        drv["ppfd"].values * 0.05
        + np.exp((drv["air_temperature"].values - 298.15) / 10.0)
    )
    y_obs = y_base * 1.2  # scale factor
    obs = pd.DataFrame({
        "timestamp": drv["timestamp"],
        "isoprene_flux_ug_m2_s": y_obs,
    })
    obs.to_csv(obs_csv, index=False)

    # Run validation with calibration
    outdir.mkdir(parents=True, exist_ok=True)
    val_cmd = [
        sys.executable,
        str(scripts_dir / "run_validation.py"),
        "--drivers",
        str(drivers_parquet),
        "--obs",
        str(obs_csv),
        "--outdir",
        str(outdir),
        "--emission-pathway",
        "standard",
        "--calibrate",
    ]
    r2 = subprocess.run(val_cmd, capture_output=True, text=True)
    assert r2.returncode == 0, f"run_validation failed: {r2.stderr}\n{r2.stdout}"
    assert not r2.stderr.strip(), f"run_validation emitted stderr: {r2.stderr}"
    metrics_csv = outdir / "metrics_summary.csv"
    assert metrics_csv.exists(), "metrics_summary.csv not found"
    m = pd.read_csv(metrics_csv)
    # Check keys present
    for k in [
        "base",
        "calibrated",
        "calibration_scale",
        "n",
        "emission_pathway",
        "device",
    ]:
        assert k in m.columns, f"Missing column {k} in metrics_summary.csv"


