"""Diagnostics and artifact writer for Tapajos case runs."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from relic.core.units import UnitSystem


MANDATORY_LOG_KEYS = [
    "evapotranspiration_flux",
    "soil_moisture",
    "water_stress_factor",
    "temperature_stress_factor",
    "ozone_stress_factor",
    "co2_concentration",
    "co2_intercellular",
    "atmospheric_co2_ppm",
    "gross_photosynthesis",
    "gross_primary_productivity",
    "electron_transport_rate",
    "net_photosynthesis",
    "isoprene_emission",
    "stomatal_conductance",
    "drainage_flux",
    "precipitation_flux_effective",
    "precipitation_flux_gross",
    "throughfall_fraction",
    "canopy_interception_fraction",
    "canopy_interception_flux",
    "surface_runoff_flux",
    "surface_runoff_flux_external",
    "infiltration_flux",
    "runoff_coefficient",
]


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    input_snapshot_dir: Path
    timeseries_dir: Path
    optimization_dir: Path
    aggregate_dir: Path
    checkpoint_dir: Path


def _to_scalar(value: object) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().reshape(-1)[0].tolist())
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _serialize_checkpoint_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_serialize_checkpoint_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _serialize_checkpoint_value(item)
            for key, item in value.items()
        }
    return value


class DiagnosticsWriter:
    """Write run outputs required by the tutorial acceptance criteria."""

    def __init__(self, run_root: Path) -> None:
        self.layout = RunLayout(
            run_dir=run_root,
            input_snapshot_dir=run_root / "input_snapshot",
            timeseries_dir=run_root / "timeseries",
            optimization_dir=run_root / "optimization",
            aggregate_dir=run_root / "aggregate",
            checkpoint_dir=run_root / "checkpoints",
        )
        self.layout.input_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.layout.timeseries_dir.mkdir(parents=True, exist_ok=True)
        self.layout.optimization_dir.mkdir(parents=True, exist_ok=True)
        self.layout.aggregate_dir.mkdir(parents=True, exist_ok=True)
        self.layout.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.layout.aggregate_dir / "progress_log.txt"
        try:
            self.log_path.write_text("", encoding="utf-8")
        except PermissionError:
            fallback_name = datetime.now(timezone.utc).strftime("progress_log_%Y%m%d_%H%M%S.txt")
            self.log_path = self.layout.aggregate_dir / fallback_name
            self.log_path.write_text("", encoding="utf-8")
        self._rows: List[Dict[str, object]] = []
        self._phase_to_id: Dict[str, int] = {}
        self._next_phase_id = 0

    def append_log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"[{timestamp}] {message}\n"
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        except PermissionError:
            fallback_name = datetime.now(timezone.utc).strftime("progress_log_%Y%m%d_%H%M%S.txt")
            fallback_path = self.layout.aggregate_dir / fallback_name
            with fallback_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            self.log_path = fallback_path

    def _encode_phase_id(self, phase_name: str) -> int:
        key = str(phase_name)
        if key not in self._phase_to_id:
            self._phase_to_id[key] = self._next_phase_id
            self._next_phase_id += 1
        return int(self._phase_to_id[key])

    def get_phase_id_map(self) -> Dict[str, int]:
        return dict(self._phase_to_id)

    def snapshot_inputs(self, files: Iterable[Path]) -> None:
        for path in files:
            if not path.exists():
                continue
            target = self.layout.input_snapshot_dir / path.name
            shutil.copy2(path, target)

    def reset_step_logs(self) -> None:
        self._rows = []

    def log_step(
        self,
        *,
        epoch: int,
        step: int,
        time: pd.Timestamp,
        state: Dict[str, object],
        phase_name: str = "default",
        extra_keys: Optional[Iterable[str]] = None,
    ) -> None:
        phase_id = self._encode_phase_id(phase_name)
        row: Dict[str, object] = {
            "epoch": int(epoch),
            "step": int(step),
            "phase_id": phase_id,
            "time": pd.Timestamp(time).isoformat(),
        }
        keys = list(MANDATORY_LOG_KEYS)
        if extra_keys is not None:
            for key in extra_keys:
                if key not in keys:
                    keys.append(key)
        for key in keys:
            row[key] = _to_scalar(state.get(key))
        self._rows.append(row)

    def _rows_to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["epoch", "step", "phase_id", "time", *MANDATORY_LOG_KEYS])
        frame = pd.DataFrame(self._rows)
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        return frame

    def _build_dataset(
        self,
        *,
        frame: pd.DataFrame,
        columns: List[str],
    ) -> xr.Dataset:
        coords = {
            "sample": np.arange(frame.shape[0], dtype=np.int64),
            "time": ("sample", frame["time"].to_numpy(dtype="datetime64[ns]")),
            "epoch": ("sample", frame["epoch"].to_numpy(dtype=np.int64)),
            "step": ("sample", frame["step"].to_numpy(dtype=np.int64)),
            "phase_id": ("sample", frame["phase_id"].to_numpy(dtype=np.int64)),
        }
        if columns:
            dataset = xr.Dataset(
                data_vars={
                    col: ("sample", frame[col].to_numpy(dtype=np.float64))
                    for col in columns
                },
                coords=coords,
                attrs={"Conventions": "CF-1.8"},
            )
            for column in columns:
                contract_name = column[:-3] if column.endswith("_ML") else column
                unit = UnitSystem.get_standard_unit(contract_name)
                if unit != "unknown":
                    dataset[column].attrs["units"] = unit
            return dataset
        return xr.Dataset(coords=coords, attrs={"Conventions": "CF-1.8"})

    def write_timeseries(self) -> None:
        frame = self._rows_to_dataframe()
        if frame.empty:
            return

        state_columns = [
            key
            for key in frame.columns
            if key not in {"epoch", "step", "phase_id", "time"} and not key.startswith("diagnostics.")
        ]
        diag_columns = [
            key
            for key in frame.columns
            if key.startswith("diagnostics.")
            or key
            in {
                "drainage_flux",
                "precipitation_flux_effective",
                "precipitation_flux_gross",
                "throughfall_fraction",
                "canopy_interception_fraction",
                "canopy_interception_flux",
                "surface_runoff_flux",
                "surface_runoff_flux_external",
                "infiltration_flux",
                "runoff_coefficient",
            }
        ]
        physics_state_columns = [col for col in state_columns if not str(col).endswith("_ML")]
        ml_state_columns = [col for col in state_columns if str(col).endswith("_ML")]

        state_ds = self._build_dataset(frame=frame, columns=state_columns)
        state_ds.to_netcdf(self.layout.timeseries_dir / "state_hourly.nc")
        self._build_dataset(frame=frame, columns=physics_state_columns).to_netcdf(
            self.layout.timeseries_dir / "state_hourly_physics.nc"
        )
        self._build_dataset(frame=frame, columns=ml_state_columns).to_netcdf(
            self.layout.timeseries_dir / "state_hourly_ML.nc"
        )

        diag_ds = self._build_dataset(frame=frame, columns=diag_columns)
        diag_ds.to_netcdf(self.layout.timeseries_dir / "diagnostics_hourly.nc")

    def write_process_monitor(self) -> Dict[str, Path]:
        frame = self._rows_to_dataframe()
        outputs: Dict[str, Path] = {}
        if frame.empty:
            return outputs

        physics_columns = [
            col for col in frame.columns if col in {"epoch", "step", "phase_id", "time"} or not str(col).endswith("_ML")
        ]
        ml_columns = [
            col for col in frame.columns if col in {"epoch", "step", "phase_id", "time"} or str(col).endswith("_ML")
        ]

        combined_path = self.layout.aggregate_dir / "process_monitor.csv"
        physics_path = self.layout.aggregate_dir / "process_monitor_physics.csv"
        ml_path = self.layout.aggregate_dir / "process_monitor_ML.csv"

        frame.to_csv(combined_path, index=False)
        frame.loc[:, physics_columns].to_csv(physics_path, index=False)
        frame.loc[:, ml_columns].to_csv(ml_path, index=False)

        outputs["combined"] = combined_path
        outputs["physics"] = physics_path
        outputs["ml"] = ml_path
        return outputs

    def write_simulation_checkpoint(
        self,
        *,
        epoch: int,
        step: int,
        time: pd.Timestamp,
        state: Dict[str, object],
        phase_name: str,
    ) -> Path:
        physics_state = {
            str(key): _serialize_checkpoint_value(value)
            for key, value in state.items()
            if not str(key).endswith("_ML")
        }
        ml_state = {
            str(key): _serialize_checkpoint_value(value)
            for key, value in state.items()
            if str(key).endswith("_ML")
        }
        payload = {
            "kind": "simulate_runtime_checkpoint",
            "phase_name": str(phase_name),
            "epoch": int(epoch),
            "step": int(step),
            "time": pd.Timestamp(time).isoformat(),
            "written_at": datetime.now(timezone.utc).isoformat(),
            "state_physics": physics_state,
            "state_ml": ml_state,
        }
        step_path = self.layout.checkpoint_dir / f"simulate_{phase_name}_step{int(step):05d}.pt"
        latest_path = self.layout.checkpoint_dir / f"simulate_{phase_name}_latest.pt"
        torch.save(payload, step_path)
        torch.save(payload, latest_path)
        return step_path

    def write_ml_training_summary(self, summary: Dict[str, object]) -> Optional[Path]:
        if not summary:
            return None
        out_path = self.layout.aggregate_dir / "ml_training_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return out_path

    def archive_ml_surrogate_checkpoints(self, checkpoint_sources: Dict[str, Path]) -> Dict[str, str]:
        archive_dir = self.layout.checkpoint_dir / "ml_surrogates"
        archive_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, str] = {}
        for functor_name, source_path in checkpoint_sources.items():
            source = Path(source_path)
            if not source.exists():
                continue
            target = archive_dir / source.name
            shutil.copy2(source, target)
            manifest[str(functor_name)] = str(target.resolve())
        if manifest:
            manifest_path = archive_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def write_phase_loss(self, phase_name: str, rows: List[Dict[str, object]]) -> Path:
        out_path = self.layout.optimization_dir / f"{phase_name}_loss.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return out_path

    def write_gradient_norms(self, rows: List[Dict[str, object]]) -> Path:
        out_path = self.layout.optimization_dir / "gradient_norms.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        return out_path

    def write_initial_trainable_parameters(self, params: Dict[str, float]) -> Path:
        out_path = self.layout.optimization_dir / "initial_trainable_parameters.yaml"
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(params, handle, sort_keys=True)
        return out_path

    def write_optimized_parameters(self, params: Dict[str, float]) -> Path:
        out_path = self.layout.optimization_dir / "optimized_parameters.yaml"
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(params, handle, sort_keys=True)
        # Keep a compatibility alias for workflows that expect best_params.yaml.
        best_path = self.layout.optimization_dir / "best_params.yaml"
        with best_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(params, handle, sort_keys=True)
        return out_path

    def write_run_summary(self, summary: Dict[str, object]) -> Path:
        out_path = self.layout.aggregate_dir / "run_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return out_path
