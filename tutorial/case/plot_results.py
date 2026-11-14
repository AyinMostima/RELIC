"""Generate publication-style figures from a tutorial/case run directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

if os.name == "nt":
    # Work around duplicate Intel OpenMP runtime loading on Windows when
    # PyTorch and MKL-linked scientific packages are imported in the same
    # plotting process.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sys
import torch
import xarray as xr
import yaml
from pandas.errors import EmptyDataError
import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relic.utils.physical_constants import get_physical_constant
from tutorial.case.utils.time_alignment import (
    align_frame_to_reference_time,
    validate_model_time_coverage,
)

def _setup_matplotlib() -> None:
    # Nature-style figure defaults based on the official figure guide.
    # References:
    # - Nature Research Figure Guide: use Arial/Helvetica, 5-7 pt text, 8 pt bold panel labels.
    # - Nature submission guidance: white background, minimal boxing, lower-case bold panel labels, line width >= 1 pt.
    # Mathematical expression:
    # - Not applicable; this block sets publication-style plotting parameters.
    # Parameter meanings:
    # - rcParams entries control typography, stroke width, spacing, and export behavior.
    matplotlib.use("Agg", force=True)
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["axes.titlesize"] = 7
    matplotlib.rcParams["axes.labelsize"] = 7
    matplotlib.rcParams["xtick.labelsize"] = 6
    matplotlib.rcParams["ytick.labelsize"] = 6
    matplotlib.rcParams["legend.fontsize"] = 6
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["lines.linewidth"] = 1.0
    matplotlib.rcParams["xtick.major.width"] = 0.8
    matplotlib.rcParams["ytick.major.width"] = 0.8
    matplotlib.rcParams["xtick.major.size"] = 3.0
    matplotlib.rcParams["ytick.major.size"] = 3.0
    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["legend.frameon"] = False


_setup_matplotlib()
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# Colors (colorblind-safe, Nature-style)
C_OBS = "#2C2C2C"
C_MODEL = "#E64B35"
C_ML = "#7B68EE"
C_PHASE = ["#4DBBD5", "#00A087", "#E64B35", "#3C5488"]
C_FORCE = "#7E6148"


def _get_physical_constant(name: str) -> float:
    return float(get_physical_constant(name))


def _get_isoprene_molecular_weight_g_mol() -> float:
    return float(
        get_physical_constant(
            "relic.physical.constants.molecular_weights.isoprene"
        )
    )


def _read_case_value(case_params: Dict[str, object], key: str) -> float:
    mapping = case_params.get("parameters")
    if isinstance(mapping, dict) and key in mapping and isinstance(mapping[key], dict):
        value = mapping[key].get("value")
        if value is None:
            raise KeyError(f"Missing value for {key}")
        return float(value)
    raw = case_params.get(key)
    if isinstance(raw, dict):
        value = raw.get("value")
        if value is None:
            raise KeyError(f"Missing value for {key}")
        return float(value)
    raise KeyError(f"Missing parameter entry {key}")


def _load_yaml(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _load_parameter_change_context(
    run_dir: Path,
    params: Sequence[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    init_path = run_dir / "input_snapshot" / "case_parameters.yaml"
    opt_path = run_dir / "optimization" / "optimized_parameters.yaml"
    actual_init_path = run_dir / "optimization" / "initial_trainable_parameters.yaml"
    if not init_path.exists():
        raise FileNotFoundError(f"Missing initial case_parameters.yaml: {init_path}")
    if not opt_path.exists():
        raise FileNotFoundError(f"Missing optimized_parameters.yaml: {opt_path}")

    init_cfg = _load_yaml(init_path)
    optimized = _load_yaml(opt_path)
    actual_initial: Dict[str, object] = {}
    if actual_init_path.exists():
        payload = _load_yaml(actual_init_path)
        if isinstance(payload, dict):
            actual_initial = payload
    if not actual_initial:
        summary_path = run_dir / "aggregate" / "run_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                summary = {}
            if isinstance(summary, dict):
                init_from_run = summary.get("init_from_run")
                init_from_phase = summary.get("init_from_phase")
                if isinstance(init_from_run, str) and init_from_run.strip():
                    source_run_dir = run_dir.parent / init_from_run.strip()
                    if isinstance(init_from_phase, str) and init_from_phase.strip():
                        checkpoint_candidates = sorted(
                            (source_run_dir / "checkpoints").glob(
                                f"checkpoint_{init_from_phase.strip()}_epoch*.pt"
                            )
                        )
                        if checkpoint_candidates:
                            checkpoint_payload = torch.load(
                                checkpoint_candidates[-1],
                                map_location="cpu",
                                weights_only=True,
                            )
                            trainable_payload = checkpoint_payload.get("trainable_params", {})
                            if isinstance(trainable_payload, dict):
                                actual_initial = {
                                    str(key): float(
                                        value.detach().reshape(-1)[0].item()
                                        if isinstance(value, torch.Tensor)
                                        else value
                                    )
                                    for key, value in trainable_payload.items()
                                }
                    if not actual_initial:
                        source_opt_path = source_run_dir / "optimization" / "optimized_parameters.yaml"
                        if source_opt_path.exists():
                            payload = _load_yaml(source_opt_path)
                            if isinstance(payload, dict):
                                actual_initial = payload

    source_label = "start" if actual_initial else "initial"
    init_vals: List[float] = []
    opt_vals: List[float] = []
    units: List[str] = []
    param_mapping = init_cfg.get("parameters")
    for key, _alias in params:
        init_entry = param_mapping.get(key) if isinstance(param_mapping, dict) else None
        if not isinstance(init_entry, dict):
            raise KeyError(f"Missing parameter in case_parameters.yaml: {key}")
        default_init_val = float(init_entry["value"])
        init_val = float(actual_initial.get(key, default_init_val))
        unit = str(init_entry.get("units", ""))
        opt_val = float(optimized.get(key, init_val))
        init_vals.append(init_val)
        opt_vals.append(opt_val)
        units.append(unit)

    return (
        np.asarray(init_vals, dtype=float),
        np.asarray(opt_vals, dtype=float),
        units,
        source_label,
    )


def _resolve_case_root(case_root: Optional[Path]) -> Path:
    if case_root is not None:
        return case_root.resolve()
    return Path(__file__).resolve().parent


def _resolve_run_dir(case_root: Path, run_id: str) -> Path:
    run_dir = (case_root / "output" / "runs" / run_id).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def _select_final_forward_sample(ds: xr.Dataset) -> xr.Dataset:
    if "epoch" in ds:
        epoch_values = np.asarray(ds["epoch"].values)
        if np.any(epoch_values == -1):
            return ds.where(ds["epoch"] == -1, drop=True)
        max_epoch = int(np.nanmax(epoch_values))
        return ds.where(ds["epoch"] == max_epoch, drop=True)
    return ds


def _dataset_to_frame(ds: xr.Dataset, keys: Sequence[str]) -> pd.DataFrame:
    if "time" not in ds.coords:
        raise KeyError("Dataset missing time coordinate.")
    time_index = pd.to_datetime(ds["time"].values)
    payload: Dict[str, np.ndarray] = {}
    for key in keys:
        if key not in ds:
            continue
        payload[key] = np.asarray(ds[key].values, dtype=float).reshape(-1)
    frame = pd.DataFrame(payload, index=time_index)
    frame.index.name = "time"
    return frame.sort_index()


def _detect_ml_variables(state_ds: xr.Dataset) -> Dict[str, str]:
    """Detect ML path variables recorded with the _ML suffix."""
    ml_map: Dict[str, str] = {}
    for var_name in state_ds.data_vars:
        if not str(var_name).endswith("_ML"):
            continue
        base_name = str(var_name)[:-3]
        if base_name in state_ds.data_vars:
            ml_map[base_name] = str(var_name)
    return ml_map


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _compute_metrics(model: pd.Series, obs: pd.Series) -> Dict[str, float]:
    # References:
    # - Nash & Sutcliffe (1970), NSE = 1 - sum((m-o)^2)/sum((o-mean(o))^2).
    # - Willmott (1981), IoA = 1 - sum((m-o)^2)/sum((|m-mean(o)|+|o-mean(o)|)^2).
    # Mathematical expression:
    # - NSE in [-inf, 1], IoA in [0, 1], RMSE = sqrt(mean((m-o)^2)).
    # Parameter meanings:
    # - m: model, o: observation, n: sample size.
    aligned = pd.concat([model.rename("m"), obs.rename("o")], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return {"n": 0.0, "r": float("nan"), "rmse": float("nan"), "bias": float("nan"),
                "slope": float("nan"), "nse": float("nan"), "ioa": float("nan")}

    m = aligned["m"].to_numpy(dtype=float)
    o = aligned["o"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((m - o) ** 2)))
    bias = float(np.mean(m - o))
    r = _pearson_r(m, o)

    # References: numpy.polyfit, degree-one least-squares fit.
    # Mathematical expression: model = slope * observation + intercept.
    # Parameter meanings: o and m are paired observation and model samples.
    # Units: slope is dimensionless; both arrays share the variable unit.
    # Domain/constraints: at least two finite pairs and nonconstant observations.
    # Sign convention: model is the response and observation is the predictor.
    slope = float("nan")
    if o.size >= 2 and np.any(o != o[0]):
        # The finite-pair filter above defines the input domain.
        # Numerical failures for nondegenerate input must propagate.
        slope = float(np.polyfit(o, m, 1)[0])

    # NSE
    ss_res = np.sum((m - o) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    nse = float(1.0 - ss_res / max(ss_tot, 1e-12))

    # IoA (Willmott 1981)
    o_mean = np.mean(o)
    denom = np.sum((np.abs(m - o_mean) + np.abs(o - o_mean)) ** 2)
    ioa = float(1.0 - ss_res / max(denom, 1e-12))

    return {"n": float(o.size), "r": r, "rmse": rmse, "bias": bias,
            "slope": slope, "nse": nse, "ioa": ioa}


def _panel_label(ax: plt.Axes, label: str) -> None:
    clean_label = str(label).strip().strip("()").lower()
    ax.text(
        0.01,
        0.98,
        clean_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=8,
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.0, width=0.8, pad=2)


def _add_path_badge(fig: plt.Figure, *, use_ml: bool) -> None:
    label = "ML path" if use_ml else "Physics path"
    color = C_ML if use_ml else C_MODEL
    fig.text(
        0.01,
        0.995,
        label,
        ha="left",
        va="top",
        fontsize=6,
        color=color,
    )


def _add_figure_legend(
    fig: plt.Figure,
    *,
    handles: Sequence[object],
    labels: Sequence[str],
    ncol: int = 2,
) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.2,
    )


def _metric_box(ax: plt.Axes, metrics: Dict[str, float], include_slope: bool = False) -> None:
    lines = [
        f"R = {metrics['r']:.2f}",
        f"NSE = {metrics.get('nse', float('nan')):.3f}",
        f"RMSE = {metrics['rmse']:.2f}",
        f"IoA = {metrics.get('ioa', float('nan')):.3f}",
        f"bias = {metrics['bias']:.2f}",
        f"n = {int(metrics['n'])}",
    ]
    if include_slope:
        lines.insert(3, f"slope = {metrics['slope']:.2f}")
    text = "\n".join(lines)
    ax.text(
        0.98,
        0.93,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "square,pad=0.15"},
    )


def _metric_box_dual(
    ax: plt.Axes,
    physics_metrics: Dict[str, float],
    ml_metrics: Optional[Dict[str, float]] = None,
    *,
    include_slope: bool = False,
) -> None:
    lines = [
        f"Phys R = {physics_metrics['r']:.2f}",
        f"Phys NSE = {physics_metrics.get('nse', float('nan')):.3f}",
        f"Phys RMSE = {physics_metrics['rmse']:.2f}",
    ]
    if include_slope:
        lines.append(f"Phys slope = {physics_metrics['slope']:.2f}")
    if ml_metrics is not None:
        lines.extend(
            [
                f"ML R = {ml_metrics['r']:.2f}",
                f"ML NSE = {ml_metrics.get('nse', float('nan')):.3f}",
                f"ML RMSE = {ml_metrics['rmse']:.2f}",
            ]
        )
        if include_slope:
            lines.append(f"ML slope = {ml_metrics['slope']:.2f}")
    lines.append(f"n = {int(physics_metrics['n'])}")
    ax.text(
        0.98,
        0.93,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        bbox={"facecolor": "white", "edgecolor": "0.85", "boxstyle": "square,pad=0.15"},
    )


def _ensure_figures_dir(run_dir: Path) -> Path:
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(pad=0.7, rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")


def _resolve_path_column(state: pd.DataFrame, base_key: str, *, use_ml: bool) -> str:
    if use_ml:
        ml_key = f"{base_key}_ML"
        if ml_key not in state.columns:
            raise KeyError(f"Missing ML variable for plotting: {ml_key}")
        return ml_key
    if base_key not in state.columns:
        raise KeyError(f"Missing physics variable for plotting: {base_key}")
    return base_key


def _load_inputs(
    run_dir: Path,
    *,
    allow_partial: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    state_path = run_dir / "timeseries" / "state_hourly.nc"
    obs_path = run_dir / "input_snapshot" / "real_obs.nc"
    forcing_path = run_dir / "input_snapshot" / "site_forcing.nc"
    case_params_path = run_dir / "input_snapshot" / "case_parameters.yaml"

    if not state_path.exists():
        raise FileNotFoundError(f"Missing state_hourly.nc: {state_path}")
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing real_obs.nc: {obs_path}")
    if not forcing_path.exists():
        raise FileNotFoundError(f"Missing site_forcing.nc: {forcing_path}")
    if not case_params_path.exists():
        raise FileNotFoundError(f"Missing case_parameters.yaml: {case_params_path}")

    state_ds = xr.load_dataset(state_path)
    state_ds = _select_final_forward_sample(state_ds)
    ml_map = _detect_ml_variables(state_ds)
    state_keys = [
        "soil_moisture",
        "net_photosynthesis",
        "gpp",
        "gross_photosynthesis",
        "net_assimilation",
        "photosynthesis_rate",
        "isoprene_emission",
        "evapotranspiration_flux",
        "drainage_flux",
        "temperature_stress_factor",
        "water_stress_factor",
        "water_stress",
        "ozone_stress_factor",
        "ozone_uptake",
        "antioxidant_depletion",
        "stomatal_conductance",
    ]
    for ml_key in ml_map.values():
        if ml_key not in state_keys:
            state_keys.append(ml_key)
    for var_name in state_ds.data_vars:
        if str(var_name).endswith("_ML") and str(var_name) not in state_keys:
            state_keys.append(str(var_name))
    state = _dataset_to_frame(state_ds, state_keys)

    obs_ds = xr.load_dataset(obs_path)
    obs = _dataset_to_frame(obs_ds, ["soil_moisture", "net_photosynthesis", "isoprene_emission"])
    obs = align_frame_to_reference_time(
        state,
        obs,
        frame_name="observations",
        require_complete=True,
    )

    forcing_ds = xr.load_dataset(forcing_path)
    forcing = _dataset_to_frame(
        forcing_ds,
        [
            "air_temperature",
            "ppfd",
            "vapor_pressure_deficit",
            "precipitation_flux",
            "co2_concentration",
            "relative_humidity",
            "wind_speed",
            "net_radiation",
            "o3_mixing_ratio",
        ],
    )
    validate_model_time_coverage(
        state,
        forcing,
        allow_partial=allow_partial,
    )
    forcing = align_frame_to_reference_time(
        state,
        forcing,
        frame_name="forcing",
        require_complete=True,
    )

    case_params = _load_yaml(case_params_path)
    return state, obs, forcing, case_params


def _convert_isoprene_obs_to_ug_per_s(
    obs_nmol_per_s: pd.Series,
    case_params: Dict[str, object],
) -> pd.Series:
    # References:
    # - Guenther et al. (2012), isoprene observations may be reported as molar flux.
    # - SI: 1 mol = 1e6 umol, and 1 umol has mass MW [ug] when MW is [g mol-1].
    # Mathematical expression:
    # - E_obs_ug_s = E_obs_nmol_s * MW_ug_umol / (nmol_per_umol)
    # Parameter meanings:
    # - MW_ug_umol: isoprene molecular weight [ug umol-1], numerically MW [g mol-1]
    # - nmol_per_umol: conversion factor [nmol umol-1]
    mw_g_per_mol = _get_isoprene_molecular_weight_g_mol()
    nmol_per_umol = float(_read_case_value(case_params, "case.conversions.umol_to_nmol"))
    scale = mw_g_per_mol / nmol_per_umol
    return obs_nmol_per_s.astype(float) * scale


def _convert_isoprene_model_to_ug_per_s(model_mol_per_s: pd.Series) -> pd.Series:
    # References:
    # - Guenther et al. (2012), MEGAN species emissions are molar fluxes.
    # Mathematical expression:
    # - E_model_ug_s = E_model_mol_s * MW / c_ug_to_g
    # Parameter meanings:
    # - MW: isoprene molecular weight [g mol-1]
    # - c_ug_to_g: micrograms-to-grams conversion [g ug-1]
    mw_g_per_mol = _get_isoprene_molecular_weight_g_mol()
    micrograms_to_grams = _get_physical_constant(
        "relic.physical.constants.conversions.micrograms_to_grams"
    )
    return model_mol_per_s.astype(float) * (mw_g_per_mol / micrograms_to_grams)


def fig1_calibration_timeseries(
    *,
    run_dir: Path,
    state: pd.DataFrame,
    obs: pd.DataFrame,
    case_params: Dict[str, object],
    dpi: int,
    use_ml: bool = False,
) -> None:
    seconds_per_hour = _get_physical_constant("seconds_per_hour")
    double_col_width = 183.0 / 25.4
    height_in = 120.0 / 25.4

    fig, axes = plt.subplots(3, 1, figsize=(double_col_width, height_in), sharex=True)
    path_label = "ML path" if use_ml else "Physics path"
    _add_path_badge(fig, use_ml=use_ml)

    # Panel (a) Soil moisture
    sm_ax = axes[0]
    sm_model = state[_resolve_path_column(state, "soil_moisture", use_ml=use_ml)]
    sm_obs = obs["soil_moisture"]
    sm_line, = sm_ax.plot(
        sm_model.index,
        sm_model.values,
        color=C_ML if use_ml else C_MODEL,
        lw=1.1,
        label=path_label,
    )
    obs_points = sm_ax.plot(
        sm_obs.index,
        sm_obs.values,
        linestyle="None",
        marker="o",
        ms=2.2,
        color=C_OBS,
        alpha=0.75,
        label="Observation",
    )[0]
    sm_ax.set_ylabel("Soil moisture (m3 m-3)")
    _panel_label(sm_ax, "a")
    _metric_box(sm_ax, _compute_metrics(sm_model, sm_obs), include_slope=False)

    # Panel (b) Net photosynthesis
    np_ax = axes[1]
    np_model = state[_resolve_path_column(state, "net_photosynthesis", use_ml=use_ml)]
    np_obs = obs["net_photosynthesis"]
    np_ax.plot(
        np_model.index,
        np_model.values,
        color=C_ML if use_ml else C_MODEL,
        lw=1.1,
    )
    np_ax.plot(np_obs.index, np_obs.values, linestyle="None", marker="o", ms=2.2, color=C_OBS, alpha=0.75)
    np_ax.set_ylabel("Net photosynthesis (umol m-2 s-1)")
    _panel_label(np_ax, "b")
    _metric_box(np_ax, _compute_metrics(np_model, np_obs), include_slope=False)

    # Panel (c) Isoprene emission (ug m-2 h-1)
    iso_ax = axes[2]
    iso_model_ug_s = _convert_isoprene_model_to_ug_per_s(
        state[_resolve_path_column(state, "isoprene_emission", use_ml=use_ml)]
    )
    iso_obs_ug_s = _convert_isoprene_obs_to_ug_per_s(obs["isoprene_emission"], case_params)
    iso_model_ug_h = iso_model_ug_s * seconds_per_hour
    iso_obs_ug_h = iso_obs_ug_s * seconds_per_hour
    iso_ax.plot(
        iso_model_ug_h.index,
        iso_model_ug_h.values,
        color=C_ML if use_ml else C_MODEL,
        lw=1.1,
    )
    iso_ax.plot(iso_obs_ug_h.index, iso_obs_ug_h.values, linestyle="None", marker="o", ms=2.2, color=C_OBS, alpha=0.75)
    iso_ax.set_ylabel("Isoprene (ug m-2 h-1)")
    _panel_label(iso_ax, "c")
    _metric_box(iso_ax, _compute_metrics(iso_model_ug_h, iso_obs_ug_h), include_slope=False)

    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for ax in axes:
        _style_axes(ax)
    _add_figure_legend(fig, handles=[sm_line, obs_points], labels=[path_label, "Observation"], ncol=2)

    out_dir = _ensure_figures_dir(run_dir)
    stem = "fig1_calibration_timeseries_ML" if use_ml else "fig1_calibration_timeseries"
    _save_figure(fig, out_dir, stem, dpi=dpi)
    plt.close(fig)


def fig2_scatter(
    *,
    run_dir: Path,
    state: pd.DataFrame,
    obs: pd.DataFrame,
    case_params: Dict[str, object],
    dpi: int,
    use_ml: bool = False,
) -> None:
    seconds_per_hour = _get_physical_constant("seconds_per_hour")
    double_col_width = 183.0 / 25.4
    height_in = 78.0 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(double_col_width, height_in))
    _add_path_badge(fig, use_ml=use_ml)

    target_label = "GPP-derived target"
    provenance_path = run_dir / "input_snapshot" / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        target_quality = provenance.get("processing", {}).get("observation_quality", {}).get("net_photosynthesis", {})
        if str(target_quality.get("class", "")).startswith("proxy_"):
            target_label = "GPP proxy target"

    panels: List[Tuple[str, pd.Series, pd.Series, str]] = []
    panels.append(
        (
            "Soil moisture (m3 m-3)",
            state[_resolve_path_column(state, "soil_moisture", use_ml=use_ml)],
            obs["soil_moisture"],
            C_ML if use_ml else C_PHASE[0],
        )
    )
    panels.append(
        (
            "Net photosynthesis\n(umol CO2 m-2 s-1)",
            state[_resolve_path_column(state, "net_photosynthesis", use_ml=use_ml)],
            obs["net_photosynthesis"],
            C_ML if use_ml else C_PHASE[1],
        )
    )

    iso_model_ug_s = _convert_isoprene_model_to_ug_per_s(
        state[_resolve_path_column(state, "isoprene_emission", use_ml=use_ml)]
    )
    iso_obs_ug_s = _convert_isoprene_obs_to_ug_per_s(obs["isoprene_emission"], case_params)
    iso_model_ug_h = iso_model_ug_s * seconds_per_hour
    iso_obs_ug_h = iso_obs_ug_s * seconds_per_hour
    panels.append(("Isoprene (ug m-2 h-1)", iso_model_ug_h, iso_obs_ug_h, C_ML if use_ml else C_PHASE[2]))

    for idx, (title, model_series, obs_series, color) in enumerate(panels):
        ax = axes[idx]
        aligned = pd.concat([obs_series.rename("o"), model_series.rename("m")], axis=1)
        aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
        o = aligned["o"].to_numpy(dtype=float)
        m = aligned["m"].to_numpy(dtype=float)
        ax.scatter(o, m, s=10, color=color, alpha=0.4, edgecolors="none")

        if o.size > 0:
            vmin = float(np.nanmin(np.concatenate([o, m])))
            vmax = float(np.nanmax(np.concatenate([o, m])))
            pad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
            lo, hi = vmin - pad, vmax + pad
            ax.plot([lo, hi], [lo, hi], ls="--", color="0.6", lw=0.8)
            # References: numpy.polyfit, degree-one least-squares fit.
            # Mathematical expression: model = slope * observation + intercept.
            # Parameter meanings: o and m are finite paired samples for this panel.
            # Units: slope is dimensionless; intercept has the panel variable unit.
            # Domain/constraints: n >= 2 and nonconstant observations.
            # Sign convention: model is the response and observation the predictor.
            if o.size >= 2 and np.any(o != o[0]):
                slope, intercept = np.polyfit(o, m, 1)
                ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept], color=color, lw=1.0)
            else:
                print(f"[INFO] {title}: regression line unavailable for fewer than 2 or constant observations")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        path_metrics = _compute_metrics(model_series, obs_series)
        _metric_box(ax, path_metrics, include_slope=True)
        ax.set_title(title)
        ax.set_xlabel(target_label if idx == 1 else "Obs")
        if idx == 1:
            ax.set_ylabel("Model net photosynthesis")
        if idx == 0:
            ax.set_ylabel("ML" if use_ml else "Physics")
        _style_axes(ax)

    out_dir = _ensure_figures_dir(run_dir)
    stem = "fig2_scatter_nse_ML" if use_ml else "fig2_scatter_nse"
    _save_figure(fig, out_dir, stem, dpi=dpi)
    plt.close(fig)


def fig3_intermediate_and_forcing(
    *,
    run_dir: Path,
    state: pd.DataFrame,
    forcing: pd.DataFrame,
    dpi: int,
    use_ml: bool = False,
) -> None:
    seconds_per_hour = _get_physical_constant("seconds_per_hour")
    c_to_k = _get_physical_constant("celsius_to_kelvin")
    pa_to_kpa = _get_physical_constant("relic.physical.constants.conversions.pa_to_kpa")

    double_col_width = 183.0 / 25.4
    height_in = 180.0 / 25.4
    fig, axes = plt.subplots(4, 2, figsize=(double_col_width, height_in), sharex=True)
    axes = axes.reshape(4, 2)
    _add_path_badge(fig, use_ml=use_ml)

    def _plot_line(ax: plt.Axes, x: pd.DatetimeIndex, y: pd.Series, color: str, label: Optional[str] = None) -> None:
        ax.plot(x, y.values, color=color, lw=1.0, label=label)

    t = forcing.index

    # (a) PAR
    ax = axes[0, 0]
    _plot_line(ax, t, forcing["ppfd"], C_FORCE)
    ax.set_ylabel("PPFD (umol m-2 s-1)")
    _panel_label(ax, "(a)")

    # (b) Air temperature (degC)
    ax = axes[0, 1]
    air_c = forcing["air_temperature"].astype(float) - c_to_k
    _plot_line(ax, t, air_c, C_FORCE)
    ax.set_ylabel("Air temperature (deg C)")
    _panel_label(ax, "b")

    # (c) VPD (kPa)
    ax = axes[1, 0]
    vpd_kpa = forcing["vapor_pressure_deficit"].astype(float) * pa_to_kpa
    _plot_line(ax, t, vpd_kpa, C_FORCE)
    ax.set_ylabel("VPD (kPa)")
    _panel_label(ax, "c")

    # (d) Precipitation (mm h-1)
    ax = axes[1, 1]
    precip_mm_h = forcing["precipitation_flux"].astype(float) * seconds_per_hour
    xnum = mdates.date2num(t.to_pydatetime())
    ax.bar(xnum, precip_mm_h.values, width=0.03, color="#2E59A8", alpha=0.8)
    ax.set_ylabel("Precip (mm h-1)")
    _panel_label(ax, "d")

    # (e) Soil moisture
    ax = axes[2, 0]
    soil_key = _resolve_path_column(state, "soil_moisture", use_ml=use_ml)
    _plot_line(ax, state.index, state[soil_key], C_ML if use_ml else C_PHASE[0], label="ML path" if use_ml else "Physics path")
    ax.set_ylabel("Soil moisture (m3 m-3)")
    _panel_label(ax, "e")

    # (f) Water stress factor
    ax = axes[2, 1]
    water_key = _resolve_path_column(state, "water_stress_factor", use_ml=use_ml)
    _plot_line(ax, state.index, state[water_key], C_ML if use_ml else C_PHASE[1], label="ML path" if use_ml else "Physics path")
    ax.set_ylabel("Water stress (-)")
    _panel_label(ax, "f")

    # (g) Ozone stress factor
    ax = axes[3, 0]
    ozone_key = _resolve_path_column(state, "ozone_stress_factor", use_ml=use_ml)
    _plot_line(ax, state.index, state[ozone_key], C_ML if use_ml else C_PHASE[2], label="ML path" if use_ml else "Physics path")
    ax.set_ylabel("Ozone stress (-)")
    _panel_label(ax, "g")

    # (h) Stomatal conductance and residual correction
    ax = axes[3, 1]
    stomatal_key = _resolve_path_column(state, "stomatal_conductance", use_ml=use_ml)
    stomatal_handle = ax.plot(
        state.index,
        state[stomatal_key].values,
        color=C_ML if use_ml else C_MODEL,
        lw=1.1,
        label="ML path" if use_ml else "Physics path",
    )[0]
    residual_handle = None
    if use_ml and "isoprene_correction_factor_ML" in state.columns:
        residual_axis = ax.twinx()
        residual_handle = residual_axis.plot(
            state.index,
            state["isoprene_correction_factor_ML"].values,
            color="0.35",
            lw=1.0,
            ls="--",
            label="Residual factor",
        )[0]
        residual_axis.set_ylabel("Residual factor (1)")
        residual_axis.spines["top"].set_visible(False)
        residual_axis.tick_params(direction="out", length=3.0, width=0.8, pad=2)
    ax.set_ylabel("g_s (mol m-2 s-1)")
    _panel_label(ax, "h")

    for ax in axes.reshape(-1):
        _style_axes(ax)

    axes[3, 0].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[3, 1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    if residual_handle is not None:
        _add_figure_legend(fig, handles=[stomatal_handle, residual_handle], labels=["ML path", "Residual factor"], ncol=2)

    out_dir = _ensure_figures_dir(run_dir)
    stem = "fig3_intermediate_and_forcing_ML" if use_ml else "fig3_intermediate_and_forcing"
    _save_figure(fig, out_dir, stem, dpi=dpi)
    plt.close(fig)


def _plot_phase_loss(ax: plt.Axes, loss_csv: Path, color: str, label: str) -> None:
    try:
        frame = pd.read_csv(loss_csv)
    except EmptyDataError:
        ax.set_title(label)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
        
    if frame.empty:
        ax.set_title(label)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce").astype("Int64")
    frame["window"] = pd.to_numeric(frame["window"], errors="coerce").astype("Int64")
    frame["loss"] = pd.to_numeric(frame["loss"], errors="coerce")
    frame = frame.dropna(subset=["epoch", "window", "loss"])
    if frame.empty:
        ax.set_title(label)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    epochs = sorted(int(v) for v in frame["epoch"].unique())
    windows_per_epoch = int(frame[frame["epoch"] == epochs[0]]["window"].max() + 1)
    alphas = np.linspace(0.4, 1.0, num=len(epochs)) if len(epochs) > 1 else np.array([1.0])

    for alpha, epoch in zip(alphas, epochs):
        sub = frame[frame["epoch"] == epoch].sort_values("window")
        x = epoch * windows_per_epoch + sub["window"].to_numpy(dtype=float)
        y = sub["loss"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, lw=1.0, alpha=float(alpha))

    # Epoch boundaries
    for eidx in range(1, len(epochs)):
        ax.axvline(eidx * windows_per_epoch - 0.5, color="0.7", ls="--", lw=0.6)

    ymin = float(np.nanmin(frame["loss"].to_numpy(dtype=float)))
    ymax = float(np.nanmax(frame["loss"].to_numpy(dtype=float)))
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > 10.0 * max(ymin, 1e-12):
        ax.set_yscale("log")

    ax.set_title(label)
    _style_axes(ax)
    final_window = float(frame.sort_values(["epoch", "window"]).iloc[-1]["loss"])
    ax.text(0.98, 0.92, f"final={final_window:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=7)


def _has_phase_loss_data(run_dir: Path) -> bool:
    opt_dir = run_dir / "optimization"
    return any((opt_dir / f"phase{phase}_loss.csv").is_file() for phase in range(1, 5))


def fig4_loss_convergence(*, run_dir: Path, dpi: int) -> None:
    single_col_width = 89.0 / 25.4
    height_in = 100.0 / 25.4
    fig, axes = plt.subplots(4, 1, figsize=(single_col_width, height_in), sharex=True)

    opt_dir = run_dir / "optimization"
    loss_files = [
        ("phase1_loss.csv", C_PHASE[0], "Phase 1"),
        ("phase2_loss.csv", C_PHASE[1], "Phase 2"),
        ("phase3_loss.csv", C_PHASE[2], "Phase 3"),
        ("phase4_loss.csv", C_PHASE[3], "Phase 4"),
    ]

    for idx, (fname, color, title) in enumerate(loss_files):
        ax = axes[idx]
        _panel_label(ax, f"({chr(ord('a') + idx)})")
        loss_path = opt_dir / fname
        if not loss_path.exists():
            ax.set_title(title)
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            continue
        _plot_phase_loss(ax, loss_path, color, title)

    axes[-1].set_xlabel("Window index")
    out_dir = _ensure_figures_dir(run_dir)
    _save_figure(fig, out_dir, "fig4_loss_convergence", dpi=dpi)
    plt.close(fig)


def fig5_parameter_changes(*, run_dir: Path, dpi: int) -> None:
    single_col_width = 183.0 / 25.4
    height_in = 100.0 / 25.4
    fig, (ax, ax_table) = plt.subplots(
        1,
        2,
        figsize=(single_col_width, height_in),
        sharey=True,
        gridspec_kw={"width_ratios": [0.66, 0.34], "wspace": 0.05},
    )
    ax_table.axis("off")

    # Parameter display order and aliases.
    params: List[Tuple[str, str]] = [
        ("hydrosphere.soil_moisture.field_capacity", "FC"),
        ("hydrosphere.soil_moisture.porosity", "Porosity"),
        ("biosphere.photosynthesis.vcmax25", "Vcmax25"),
        ("biosphere.photosynthesis.jmax25", "Jmax25"),
        ("biosphere.bvoc.default.isoprene", "IsopEF"),
        ("biosphere.bvoc.megan.alpha_light", "alpha"),
        ("biosphere.bvoc.megan.cl_light", "cl"),
    ]

    y = np.arange(len(params), dtype=float)
    init_vals_arr, opt_vals_arr, units, source_label = _load_parameter_change_context(
        run_dir,
        params,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        opt_pct = np.where(init_vals_arr != 0.0, opt_vals_arr / init_vals_arr * 100.0, np.nan)
    init_pct = np.full_like(opt_pct, 100.0)
    delta = opt_pct - init_pct

    reference_label = "Run start" if source_label == "start" else "Input snapshot"
    result_label = "Run final"
    ax.barh(y - 0.15, init_pct, height=0.3, color="0.75", label=reference_label)
    ax.barh(y + 0.15, opt_pct, height=0.3, color=C_MODEL, label=result_label)

    x_max = float(np.nanmax(np.concatenate([opt_pct[np.isfinite(opt_pct)], np.asarray([100.0])])))
    ax.set_xlim(0.0, max(120.0, x_max * 1.15))

    # Arrows and labels.
    for yi, d, init_v, opt_v, unit in zip(y, delta, init_vals_arr, opt_vals_arr, units):
        if not np.isfinite(d):
            continue
        arrow_color = "#2E8B57" if d >= 0.0 else "#B22222"
        opt_pct_value = float(opt_v / init_v * 100.0) if init_v != 0.0 else 100.0
        ax.annotate(
            "",
            xy=(opt_pct_value, float(yi)),
            xytext=(100.0, float(yi)),
            arrowprops={"arrowstyle": "->", "color": arrow_color, "lw": 0.8},
        )
        x_right = float(ax.get_xlim()[1])
        x_anchor = max(100.0, opt_pct_value)
        x_label = x_anchor + 1.0
        ha = "left"
        if x_anchor > 0.80 * x_right:
            x_label = x_anchor - 1.0
            ha = "right"
        ax.text(
            x_label,
            float(yi),
            f"{d:+.1f}%",
            va="center",
            ha=ha,
            fontsize=7,
            color=arrow_color,
            clip_on=True,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([alias for _k, alias in params])
    ax.invert_yaxis()
    ax.set_xlabel(f"Value (% of {reference_label})")
    handles, labels = ax.get_legend_handles_labels()
    _add_figure_legend(fig, handles=handles, labels=labels, ncol=2)
    summary_path = run_dir / "aggregate" / "run_summary.json"
    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if run_summary.get("mode") == "simulate":
            fig.text(0.5, 0.935, "Forward simulation: physical parameters were not optimized.",
                     ha="center", va="top", fontsize=6)
    _style_axes(ax)

    ax_table.text(
        0.0,
        1.02,
        f"{reference_label} -> {result_label} (units)",
        transform=ax_table.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
    )
    for yi, init_v, opt_v, unit in zip(y, init_vals_arr, opt_vals_arr, units):
        unit_clean = str(unit).strip()
        if unit_clean:
            label = f"{init_v:.3g}->{opt_v:.3g} ({unit_clean})"
        else:
            label = f"{init_v:.3g}->{opt_v:.3g}"
        ax_table.text(
            0.0,
            float(yi),
            label,
            va="center",
            ha="left",
            fontsize=6.5,
            color="0.2",
            transform=ax_table.get_yaxis_transform(),
        )

    out_dir = _ensure_figures_dir(run_dir)
    _save_figure(fig, out_dir, "fig5_parameter_changes", dpi=dpi)
    plt.close(fig)


def fig6_ml_comparison(
    *,
    run_dir: Path,
    state: pd.DataFrame,
    dpi: int,
) -> None:
    """Write a policy notice figure instead of a mixed physics-ML comparison."""
    width_in = 183.0 / 25.4
    height_in = 60.0 / 25.4
    fig, ax = plt.subplots(1, 1, figsize=(width_in, height_in))
    ax.axis("off")
    ax.text(
        0.5,
        0.62,
        "Mixed physics-ML comparison figures are disabled.",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.38,
        "Use physics figures and _ML-suffixed figures separately.",
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    out_dir = _ensure_figures_dir(run_dir)
    _save_figure(fig, out_dir, "fig6_ml_comparison", dpi=dpi)
    plt.close(fig)


def _parse_fig_list(raw: Optional[str]) -> List[int]:
    if raw is None:
        return []
    items: List[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot publication-style figures for a tutorial/case run.")
    parser.add_argument("--case-root", type=Path, default=None, help="Case root (default: script directory).")
    parser.add_argument("--run-id", type=str, default="final_verify", help="Run id under output/runs/.")
    parser.add_argument("--fig", type=str, default=None, help="Comma-separated figure ids (1-6). Default: all.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default: 300).")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow an explicitly shortened run and crop all inputs to its model timeline.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    case_root = _resolve_case_root(args.case_root)
    run_dir = _resolve_run_dir(case_root, args.run_id)
    fig_ids = _parse_fig_list(args.fig)
    if not fig_ids:
        fig_ids = [1, 2, 3, 4, 5, 6]

    state, obs, forcing, case_params = _load_inputs(
        run_dir,
        allow_partial=args.allow_partial,
    )
    out_dir = _ensure_figures_dir(run_dir)
    _ = out_dir
    has_ml = any(column.endswith("_ML") for column in state.columns)

    jobs: List[Tuple[int, str]] = [
        (1, "fig1_calibration_timeseries"),
        (2, "fig2_scatter_r"),
        (3, "fig3_intermediate_and_forcing"),
        (4, "fig4_loss_convergence"),
        (5, "fig5_parameter_changes"),
        (6, "fig6_ml_comparison"),
    ]
    job_map = {jid: name for jid, name in jobs}
    selected = [jid for jid in fig_ids if jid in job_map]
    if not selected:
        raise ValueError(f"No valid --fig values in {fig_ids}; expected 1-6.")

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] figures_dir={run_dir / 'figures'}")

    for jid in selected:
        name = job_map[jid]
        print(f"[INFO] build figure {jid}: {name}")
        if jid == 1:
            fig1_calibration_timeseries(
                run_dir=run_dir,
                state=state,
                obs=obs,
                case_params=case_params,
                dpi=args.dpi,
                use_ml=False,
            )
            if has_ml:
                fig1_calibration_timeseries(
                    run_dir=run_dir,
                    state=state,
                    obs=obs,
                    case_params=case_params,
                    dpi=args.dpi,
                    use_ml=True,
                )
        elif jid == 2:
            fig2_scatter(
                run_dir=run_dir,
                state=state,
                obs=obs,
                case_params=case_params,
                dpi=args.dpi,
                use_ml=False,
            )
            if has_ml:
                fig2_scatter(
                    run_dir=run_dir,
                    state=state,
                    obs=obs,
                    case_params=case_params,
                    dpi=args.dpi,
                    use_ml=True,
                )
        elif jid == 3:
            fig3_intermediate_and_forcing(run_dir=run_dir, state=state, forcing=forcing, dpi=args.dpi, use_ml=False)
            if has_ml:
                fig3_intermediate_and_forcing(run_dir=run_dir, state=state, forcing=forcing, dpi=args.dpi, use_ml=True)
        elif jid == 4:
            if _has_phase_loss_data(run_dir):
                fig4_loss_convergence(run_dir=run_dir, dpi=args.dpi)
            else:
                print("[SKIP] figure 4 requires optimization loss outputs")
        elif jid == 5:
            fig5_parameter_changes(run_dir=run_dir, dpi=args.dpi)
        elif jid == 6:
            fig6_ml_comparison(run_dir=run_dir, state=state, dpi=args.dpi)

    print("[OK] figures written.")


if __name__ == "__main__":
    main()
