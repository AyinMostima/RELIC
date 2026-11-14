"""
Supplementary figures: MEGAN Standard vs RELIC comparison.

Generates Fig S1-S4 as additive supplementary figures without modifying
the existing RELIC-only plot suite (fig1-fig8).

Usage:
    python plot_megan_comparison.py --run-id <run_id>
    python plot_megan_comparison.py  # auto-detect latest run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relic.utils.physical_constants import get_physical_constant
from tutorial.case.utils.time_alignment import (
    align_frame_to_reference_time,
    validate_model_time_coverage,
)

# ---------------------------------------------------------------------------
# Nature-style rcParams (Arial, 7pt text, thin lines)
# ---------------------------------------------------------------------------
# References:
# - Nature Research Figure Guide: use Arial/Helvetica, 5-7 pt text, 8 pt bold
#   panel labels, white background, minimal boxing.
# Mathematical expression:
# - Not applicable; sets publication-style plotting parameters.
# Parameter meanings:
# - rcParams entries control typography, stroke width, spacing, export.
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})

# ---------------------------------------------------------------------------
# Color palette (colorblind-safe, Nature-style)
# ---------------------------------------------------------------------------
C_OBS = "#2C2C2C"          # dark grey - observation points
C_RELIC = "#E64B35"        # vermillion - RELIC model line
C_MEGAN = "#4DBBD5"        # teal/cyan - MEGAN standard line
C_RELIC_FILL = "#E64B3522" # translucent red fill
C_MEGAN_FILL = "#4DBBD522" # translucent teal fill
C_RESIDUAL_R = "#E64B3555" # red residual band
C_RESIDUAL_M = "#4DBBD555" # teal residual band

# mm -> inches for Nature column widths
SINGLE_COL = 89.0 / 25.4   # ~3.5 in
DOUBLE_COL = 183.0 / 25.4  # ~7.2 in


# ============================================================================
# Data loading utilities
# ============================================================================

def _load_yaml(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _get_physical_constant(name: str) -> float:
    return float(get_physical_constant(name))


def _get_isoprene_mw() -> float:
    return float(
        get_physical_constant(
            "relic.physical.constants.molecular_weights.isoprene"
        )
    )


def _read_case_value(case_params: Dict, key: str) -> float:
    mapping = case_params.get("parameters")
    if isinstance(mapping, dict) and key in mapping and isinstance(mapping[key], dict):
        return float(mapping[key]["value"])
    raw = case_params.get(key)
    if isinstance(raw, dict):
        return float(raw["value"])
    raise KeyError(f"Missing parameter: {key}")


def resolve_run_dir(case_root: Path, run_id: Optional[str] = None) -> Path:
    runs_dir = case_root / "output" / "runs"
    if run_id:
        d = runs_dir / run_id
        if not d.exists():
            raise FileNotFoundError(f"Run directory not found: {d}")
        return d
    # Auto-detect latest by modification time
    candidates = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No run directories in {runs_dir}")
    return candidates[0]


def load_relic_data(
    run_dir: Path,
    *,
    allow_partial: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load RELIC simulation state, observations, forcing, and case params."""
    state_path = run_dir / "timeseries" / "state_hourly.nc"
    obs_path = run_dir / "input_snapshot" / "real_obs.nc"
    forcing_path = run_dir / "input_snapshot" / "site_forcing.nc"
    case_params_path = run_dir / "input_snapshot" / "case_parameters.yaml"

    for p in [state_path, obs_path, forcing_path, case_params_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # State
    state_ds = xr.load_dataset(state_path)
    if "epoch" in state_ds:
        epoch_vals = np.asarray(state_ds["epoch"].values)
        if np.any(epoch_vals == -1):
            state_ds = state_ds.where(state_ds["epoch"] == -1, drop=True)
        else:
            state_ds = state_ds.where(
                state_ds["epoch"] == int(np.nanmax(epoch_vals)), drop=True
            )

    time_idx = pd.to_datetime(state_ds["time"].values)
    state_keys = [
        "isoprene_emission", "net_photosynthesis", "soil_moisture",
        "isoprene_emission_ML", "net_photosynthesis_ML", "soil_moisture_ML",
    ]
    state_data = {}
    for k in state_keys:
        if k in state_ds:
            state_data[k] = np.asarray(state_ds[k].values, dtype=float).reshape(-1)
    state = pd.DataFrame(state_data, index=time_idx).sort_index()

    # Observations
    obs_ds = xr.load_dataset(obs_path)
    obs_time = pd.to_datetime(obs_ds["time"].values)
    obs_data = {}
    for k in ["isoprene_emission", "net_photosynthesis", "soil_moisture"]:
        if k in obs_ds:
            obs_data[k] = np.asarray(obs_ds[k].values, dtype=float).reshape(-1)
    obs = pd.DataFrame(obs_data, index=obs_time).sort_index()

    # Forcing (for PAR filtering)
    forcing_ds = xr.load_dataset(forcing_path)
    forcing_time = pd.to_datetime(forcing_ds["time"].values)
    forcing_data = {}
    for k in ["ppfd", "air_temperature"]:
        if k in forcing_ds:
            forcing_data[k] = np.asarray(forcing_ds[k].values, dtype=float).reshape(-1)
    forcing = pd.DataFrame(forcing_data, index=forcing_time).sort_index()

    validate_model_time_coverage(
        state,
        forcing,
        allow_partial=allow_partial,
    )
    obs = align_frame_to_reference_time(
        state,
        obs,
        frame_name="observations",
        require_complete=True,
    )
    forcing = align_frame_to_reference_time(
        state,
        forcing,
        frame_name="forcing",
        require_complete=True,
    )

    case_params = _load_yaml(case_params_path)
    return state, obs, forcing, case_params


def load_megan_comparison(case_root: Path) -> pd.DataFrame:
    """Load the MEGAN standard comparison CSV."""
    csv_path = case_root / "data" / "megancomparison" / "Santarem_Comparison_Values.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing MEGAN comparison: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date_Time"])
    df = df.set_index("Date_Time").sort_index()
    df.rename(columns={
        "isop_flux_umol": "obs_nmol_m2_s",
        "Simulated_Flux": "megan_nmol_m2_s",
    }, inplace=True)
    return df


def _build_common_comparison_support(
    *,
    relic: pd.Series,
    megan: pd.Series,
    observations: pd.Series,
    ppfd: pd.Series,
) -> pd.DataFrame:
    # References:
    # - Neelin et al. (2023), process-oriented diagnostic common standards.
    # Mathematical expression:
    # - I_common = I_RELIC intersect I_MEGAN intersect I_obs intersect I_PPFD.
    # Parameter meanings:
    # - I_*: finite timestamp support for each comparison series.
    common = pd.concat(
        {
            "relic": relic,
            "megan": megan,
            "observations": observations,
            "ppfd": ppfd,
        },
        axis=1,
    )
    common = common.replace([np.inf, -np.inf], np.nan).dropna()
    if len(common) < 2:
        raise ValueError(
            "RELIC, MEGAN, observations, and PPFD have fewer than 2 common finite timestamps"
        )
    return common.sort_index()


# ============================================================================
# Metric computation
# ============================================================================

def compute_metrics(model: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """
    Compute R, RMSE, bias, slope, NSE, IoA.

    References:
    - Nash & Sutcliffe (1970), model efficiency coefficient.
    - Willmott (1981), Index of Agreement.
    Mathematical expression:
    - NSE = 1 - sum((m - o)^2) / sum((o - mean(o))^2)
    - IoA = 1 - sum((m - o)^2) / sum((|m - mean(o)| + |o - mean(o)|)^2)
    Parameter meanings:
    - m: model predictions
    - o: observations
    - NSE: Nash-Sutcliffe Efficiency [-inf, 1]
    - IoA: Index of Agreement [0, 1]
    """
    # Units: model and observations share the same flux units.
    # Domain/constraints: paired one-dimensional numeric arrays; finite pairs only.
    # Sign convention: residual and bias are model minus observation.
    model = np.asarray(model, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if model.ndim != 1 or obs.ndim != 1 or model.shape != obs.shape:
        raise ValueError("Metrics require paired one-dimensional arrays of equal shape")
    mask = np.isfinite(model) & np.isfinite(obs)
    m, o = model[mask], obs[mask]
    n = len(m)
    if n < 2:
        return {"n": 0, "r": np.nan, "rmse": np.nan, "bias": np.nan,
                "slope": np.nan, "nse": np.nan, "ioa": np.nan}

    # Pearson correlation is undefined for either constant series.
    has_obs_variation = bool(np.any(o != o[0]))
    has_model_variation = bool(np.any(m != m[0]))
    r = float(np.corrcoef(m, o)[0, 1]) if has_obs_variation and has_model_variation else np.nan
    rmse = float(np.sqrt(np.mean((m - o) ** 2)))
    bias = float(np.mean(m - o))

    # Constant observations do not identify a regression slope.
    # Numerical failures on nondegenerate input must propagate.
    slope = float(np.polyfit(o, m, 1)[0]) if has_obs_variation else np.nan

    # NSE
    ss_res = np.sum((m - o) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    nse = 1.0 - ss_res / max(ss_tot, 1e-12)

    # IoA (Willmott 1981)
    o_mean = np.mean(o)
    denom = np.sum((np.abs(m - o_mean) + np.abs(o - o_mean)) ** 2)
    ioa = 1.0 - ss_res / max(denom, 1e-12)

    return {"n": n, "r": r, "rmse": rmse, "bias": bias,
            "slope": slope, "nse": nse, "ioa": ioa}


# ============================================================================
# Helper: styling
# ============================================================================

def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.0, width=0.8, pad=2)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.02, 1.05, label.lower(),
        transform=ax.transAxes, ha="left", va="top",
        fontweight="bold", fontsize=9,
    )


def _metric_box(
    ax: plt.Axes,
    metrics: Dict[str, float],
    color: str,
    label: str,
    pos: str = "right",
    include_extra: bool = True,
) -> None:
    lines = [f"{label}"]
    lines.append(f"  NSE = {metrics['nse']:.3f}")
    lines.append(f"  RMSE = {metrics['rmse']:.1f}")
    if include_extra:
        lines.append(f"  IoA = {metrics['ioa']:.3f}")
        lines.append(f"  bias = {metrics['bias']:.1f}")
    lines.append(f"  n = {int(metrics['n'])}")
    text = "\n".join(lines)

    x_pos = 0.98 if pos == "right" else 0.02
    ha = "right" if pos == "right" else "left"
    ax.text(
        x_pos, 0.97, text,
        transform=ax.transAxes, ha=ha, va="top",
        fontsize=5.5, color=color, fontfamily="monospace",
        bbox={"facecolor": "white", "edgecolor": "0.88",
              "boxstyle": "round,pad=0.2", "alpha": 0.92},
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    print(f"  [OK] saved {stem}.pdf / .png")


# ============================================================================
# Fig S1: Three-line timeseries comparison
# ============================================================================

def fig_s1_timeseries(
    relic_iso_ug_h: pd.Series,
    megan_iso_ug_h: pd.Series,
    obs_iso_ug_h: pd.Series,
    out_dir: Path,
    dpi: int = 300,
    suffix: str = "",
    relic_label: str = "RELIC",
) -> None:
    """
    Three-line isoprene timeseries: Obs + RELIC + MEGAN Standard.

    References:
    - Guenther et al. (2012), MEGAN2.1 model description and evaluation.
    - Sindelarova et al. (2022), MEGAN3 evaluation over multiple ecosystems.
    Mathematical expression:
    - Visual comparison; no mathematical expression.
    Parameter meanings:
    - relic_iso_ug_h: RELIC isoprene emission [ug m-2 h-1]
    - megan_iso_ug_h: MEGAN standard isoprene emission [ug m-2 h-1]
    - obs_iso_ug_h: observed isoprene emission [ug m-2 h-1]
    """
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, 3.2),
                             gridspec_kw={"height_ratios": [3, 1.2]},
                             sharex=True)

    ax_main = axes[0]
    ax_resid = axes[1]

    # Main panel: three lines
    obs_h = ax_main.scatter(
        obs_iso_ug_h.index, obs_iso_ug_h.values,
        s=6, color=C_OBS, alpha=0.7, zorder=5, label="Observation",
        edgecolors="none",
    )
    relic_h, = ax_main.plot(
        relic_iso_ug_h.index, relic_iso_ug_h.values,
        color=C_RELIC, lw=1.2, zorder=4, label=relic_label,
    )
    megan_h, = ax_main.plot(
        megan_iso_ug_h.index, megan_iso_ug_h.values,
        color=C_MEGAN, lw=1.0, ls="--", zorder=3, label="MEGAN Standard",
    )

    ax_main.set_ylabel(r"Isoprene emission ($\mu$g m$^{-2}$ h$^{-1}$)")
    ax_main.set_ylim(bottom=-2)
    _panel_label(ax_main, "a")
    _style_axes(ax_main)

    # Dual metric boxes
    aligned_r = pd.concat([relic_iso_ug_h.rename("m"), obs_iso_ug_h.rename("o")], axis=1).dropna()
    aligned_m = pd.concat([megan_iso_ug_h.rename("m"), obs_iso_ug_h.rename("o")], axis=1).dropna()
    if not aligned_r.empty:
        relic_metrics = compute_metrics(aligned_r["m"].values, aligned_r["o"].values)
        _metric_box(ax_main, relic_metrics, C_RELIC, relic_label, pos="right")
    if not aligned_m.empty:
        megan_metrics = compute_metrics(aligned_m["m"].values, aligned_m["o"].values)
        _metric_box(ax_main, megan_metrics, C_MEGAN, "MEGAN Std", pos="left")

    # Legend
    ax_main.legend(
        handles=[obs_h, relic_h, megan_h],
        loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 1.12), fontsize=7,
    )

    # Residual panel
    common_idx = relic_iso_ug_h.index.intersection(obs_iso_ug_h.dropna().index)
    if len(common_idx) > 0:
        r_resid = relic_iso_ug_h.reindex(common_idx) - obs_iso_ug_h.reindex(common_idx)
        ax_resid.fill_between(
            r_resid.index, 0, r_resid.values,
            color=C_RELIC, alpha=0.3, label="RELIC residual",
        )
    common_idx_m = megan_iso_ug_h.index.intersection(obs_iso_ug_h.dropna().index)
    if len(common_idx_m) > 0:
        m_resid = megan_iso_ug_h.reindex(common_idx_m) - obs_iso_ug_h.reindex(common_idx_m)
        ax_resid.fill_between(
            m_resid.index, 0, m_resid.values,
            color=C_MEGAN, alpha=0.3, label="MEGAN residual",
        )

    ax_resid.axhline(0, color="0.5", lw=0.5, ls=":")
    ax_resid.set_ylabel("Residual")
    ax_resid.set_xlabel("")
    _panel_label(ax_resid, "b")
    _style_axes(ax_resid)
    ax_resid.legend(loc="upper right", ncol=2, fontsize=5.5, frameon=False)

    ax_resid.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_resid.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    _save(fig, out_dir, f"figS1_timeseries_comparison{suffix}", dpi)
    plt.close(fig)


# ============================================================================
# Fig S2: Side-by-side scatter with KDE + regression
# ============================================================================

def fig_s2_scatter(
    relic_iso: pd.Series,
    megan_iso: pd.Series,
    obs_iso: pd.Series,
    out_dir: Path,
    dpi: int = 300,
    daytime_mask: Optional[pd.Series] = None,
    suffix: str = "",
    relic_label: str = "RELIC",
) -> None:
    """
    Side-by-side scatter plots: (a) Obs vs RELIC, (b) Obs vs MEGAN Standard.

    References:
    - Guenther et al. (2012), standard model-observation scatter evaluation.
    Mathematical expression:
    - Linear regression: y = slope * x + intercept
    - R = Pearson correlation coefficient
    Parameter meanings:
    - relic_iso, megan_iso, obs_iso: isoprene series [ug m-2 h-1]
    - daytime_mask: boolean series for PAR > 50 filtering
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.45))

    panels = [
        (relic_label, relic_iso, C_RELIC, C_RELIC_FILL),
        ("MEGAN Standard", megan_iso, C_MEGAN, C_MEGAN_FILL),
    ]

    # Determine shared axis range
    all_vals = pd.concat([obs_iso, relic_iso, megan_iso]).dropna()
    if len(all_vals) > 0:
        vmin = max(float(all_vals.min()), -5)
        vmax = float(all_vals.max()) * 1.05
    else:
        vmin, vmax = -5, 100

    for idx, (label, model_iso, color, fill_color) in enumerate(panels):
        ax = axes[idx]
        aligned = pd.concat([obs_iso.rename("o"), model_iso.rename("m")], axis=1)
        aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()

        if daytime_mask is not None:
            aligned = aligned[daytime_mask.reindex(aligned.index).fillna(False)]

        if aligned.empty:
            ax.text(0.5, 0.5, "no overlapping data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        o = aligned["o"].values
        m = aligned["m"].values

        # References: scipy.stats.gaussian_kde, covariance-domain requirements.
        # Mathematical expression: density is a Gaussian-kernel mixture on R^2.
        # Parameter meanings: xy contains paired observation/model flux samples.
        # Units: each coordinate is ug m-2 h-1; density has inverse squared units.
        # Domain/constraints: n >= 3 and centered sample rank 2 for 2D covariance.
        # Sign convention: density is nonnegative; coordinates retain flux signs.
        xy = np.vstack([o, m])
        if len(o) >= 3 and np.linalg.matrix_rank(xy - xy.mean(axis=1, keepdims=True)) == 2:
            kde = sp_stats.gaussian_kde(xy)
            xgrid = np.linspace(vmin, vmax, 80)
            ygrid = np.linspace(vmin, vmax, 80)
            X, Y = np.meshgrid(xgrid, ygrid)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contourf(X, Y, Z, levels=8, cmap="Greys", alpha=0.25, zorder=1)
        else:
            print(f"[INFO] {label}: KDE unavailable for fewer than 3 or collinear finite pairs")

        # Scatter points
        ax.scatter(o, m, s=14, color=color, alpha=0.5, edgecolors="none", zorder=3)

        # 1:1 line
        ax.plot([vmin, vmax], [vmin, vmax], ls="--", color="0.5", lw=0.7, zorder=2)

        # References: numpy.polyfit, degree-one least-squares fit.
        # Mathematical expression: model = slope * observation + intercept.
        # Parameter meanings: o and m are finite paired flux samples.
        # Units: slope is dimensionless; intercept has the flux unit.
        # Domain/constraints: this figure requires n >= 3 and nonconstant o.
        # Sign convention: the fitted response is model, the predictor observation.
        if len(o) >= 3 and np.any(o != o[0]):
            slope, intercept = np.polyfit(o, m, 1)
            reg_x = np.array([vmin, vmax])
            ax.plot(reg_x, slope * reg_x + intercept,
                    color=color, lw=1.2, zorder=4)
        else:
            print(f"[INFO] {label}: regression line unavailable for fewer than 3 or constant observations")

        metrics = compute_metrics(m, o)
        _metric_box(ax, metrics, color, label, pos="left", include_extra=True)

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Observed ($\mu$g m$^{-2}$ h$^{-1}$)")
        if idx == 0:
            ax.set_ylabel(r"Modeled ($\mu$g m$^{-2}$ h$^{-1}$)")
        ax.set_title(label, fontweight="bold", fontsize=8)
        _panel_label(ax, chr(ord("a") + idx))
        _style_axes(ax)

    day_suffix = "_daytime" if daytime_mask is not None else ""
    _save(fig, out_dir, f"figS2_scatter_comparison{day_suffix}{suffix}", dpi)
    plt.close(fig)


# ============================================================================
# Fig S3: Diurnal composite comparison
# ============================================================================

def fig_s3_diurnal(
    relic_iso: pd.Series,
    megan_iso: pd.Series,
    obs_iso: pd.Series,
    out_dir: Path,
    dpi: int = 300,
    suffix: str = "",
    relic_label: str = "RELIC",
) -> None:
    """
    Diurnal composite: mean +/- std for each hour of day.

    References:
    - Guenther et al. (2012), diurnal isoprene emission cycle analysis.
    Mathematical expression:
    - mean_h = (1/N_h) * sum(E_i) for all i in hour h
    - std_h = sqrt((1/N_h) * sum((E_i - mean_h)^2))
    Parameter meanings:
    - E_i: isoprene emission at timestep i [ug m-2 h-1]
    - N_h: number of data points in hour h
    """
    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.9))

    datasets = [
        ("Observation", obs_iso, C_OBS, "o", 4),
        (relic_label, relic_iso, C_RELIC, None, None),
        ("MEGAN Standard", megan_iso, C_MEGAN, None, None),
    ]

    for label, series, color, marker, ms in datasets:
        s = series.dropna()
        if s.empty:
            continue
        hourly = s.groupby(s.index.hour)
        mean_vals = hourly.mean()
        std_vals = hourly.std()
        hours = mean_vals.index.values

        if marker:
            ax.errorbar(
                hours, mean_vals.values, yerr=std_vals.values,
                fmt=marker, ms=ms, color=color, capsize=2, capthick=0.7,
                elinewidth=0.7, label=label, zorder=5,
            )
        else:
            ax.plot(hours, mean_vals.values, color=color, lw=1.5,
                    label=label, zorder=4)
            ax.fill_between(
                hours,
                (mean_vals - std_vals).values,
                (mean_vals + std_vals).values,
                color=color, alpha=0.15, zorder=2,
            )

    ax.set_xlabel("Hour of day (local)")
    ax.set_ylabel(r"Isoprene emission ($\mu$g m$^{-2}$ h$^{-1}$)")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(bottom=-2)
    ax.legend(loc="upper left", frameon=False, fontsize=7)

    # Annotate nighttime overestimation
    ax.axvspan(0, 6, alpha=0.04, color="navy", zorder=0)
    ax.axvspan(18, 23.5, alpha=0.04, color="navy", zorder=0)
    ax.text(3, ax.get_ylim()[1] * 0.92, "night", ha="center", fontsize=5.5,
            color="0.5", style="italic")
    ax.text(21, ax.get_ylim()[1] * 0.92, "night", ha="center", fontsize=5.5,
            color="0.5", style="italic")

    _panel_label(ax, "")
    _style_axes(ax)
    _save(fig, out_dir, f"figS3_diurnal_composite_comparison{suffix}", dpi)
    plt.close(fig)


# ============================================================================
# Fig S4: Multi-metric R-value enhancement summary
# ============================================================================

def fig_s4_metric_table(
    relic_iso: pd.Series,
    megan_iso: pd.Series,
    obs_iso: pd.Series,
    par_series: pd.Series,
    out_dir: Path,
    dpi: int = 300,
    suffix: str = "",
    relic_label: str = "RELIC",
) -> None:
    """
    Multi-method R-value comparison table as a figure.

    References:
    - Guenther et al. (2012), standard model evaluation metrics.
    - Nash & Sutcliffe (1970), NSE.
    - Willmott (1981), IoA.
    Mathematical expression:
    - R_daytime: Pearson R computed only where PAR > 50.
    - R_daily_mean: Pearson R of daily mean values.
    - R_daily_max: Pearson R of daily maximum values.
    - R_log1p: Pearson R of log(1+x)-transformed values.
    Parameter meanings:
    - See compute_metrics() for detail.
    """
    methods = []

    # 1. All hourly data
    def _aligned(m, o):
        df = pd.concat([m.rename("m"), o.rename("o")], axis=1).dropna()
        return df["m"].values, df["o"].values

    m_all, o_all = _aligned(relic_iso, obs_iso)
    m_all_meg, o_all_meg = _aligned(megan_iso, obs_iso)
    methods.append(("All hourly",
                     compute_metrics(m_all, o_all),
                     compute_metrics(m_all_meg, o_all_meg)))

    # 2. Daytime only (PAR > 50)
    day_mask = par_series > 50
    day_r = relic_iso[day_mask.reindex(relic_iso.index, fill_value=False).astype(bool)]
    day_m = megan_iso[day_mask.reindex(megan_iso.index, fill_value=False).astype(bool)]
    day_o = obs_iso[day_mask.reindex(obs_iso.index, fill_value=False).astype(bool)]
    mr, odr = _aligned(day_r, day_o)
    mm, odm = _aligned(day_m, day_o)
    methods.append(("Daytime (PAR>50)",
                     compute_metrics(mr, odr),
                     compute_metrics(mm, odm)))

    # 3. Daily mean
    relic_daily = relic_iso.resample("D").mean().dropna()
    megan_daily = megan_iso.resample("D").mean().dropna()
    obs_daily = obs_iso.resample("D").mean().dropna()
    mr2, or2 = _aligned(relic_daily, obs_daily)
    mm2, om2 = _aligned(megan_daily, obs_daily)
    methods.append(("Daily mean",
                     compute_metrics(mr2, or2),
                     compute_metrics(mm2, om2)))

    # 4. Daily max
    relic_dmax = relic_iso.resample("D").max().dropna()
    megan_dmax = megan_iso.resample("D").max().dropna()
    obs_dmax = obs_iso.resample("D").max().dropna()
    mr3, or3 = _aligned(relic_dmax, obs_dmax)
    mm3, om3 = _aligned(megan_dmax, obs_dmax)
    methods.append(("Daily max",
                     compute_metrics(mr3, or3),
                     compute_metrics(mm3, om3)))

    # 5. log1p transform
    relic_log = np.log1p(relic_iso.clip(lower=0))
    megan_log = np.log1p(megan_iso.clip(lower=0))
    obs_log = np.log1p(obs_iso.clip(lower=0))
    mr4, or4 = _aligned(relic_log, obs_log)
    mm4, om4 = _aligned(megan_log, obs_log)
    methods.append(("log(1+x) hourly",
                     compute_metrics(mr4, or4),
                     compute_metrics(mm4, om4)))

    # Build table figure
    fig, ax = plt.subplots(1, 1, figsize=(DOUBLE_COL, 2.0))
    ax.axis("off")

    col_labels = [
        "Method", "n",
        f"{relic_label} NSE", f"{relic_label} RMSE", f"{relic_label} IoA",
        "MEGAN NSE", "MEGAN RMSE", "MEGAN IoA",
    ]

    cell_data = []
    cell_colors = []
    for method_name, r_met, m_met in methods:
        row = [
            method_name,
            f"{int(r_met['n'])}",
            f"{r_met['nse']:.3f}",
            f"{r_met['rmse']:.1f}",
            f"{r_met['ioa']:.3f}",
            f"{m_met['nse']:.3f}",
            f"{m_met['rmse']:.1f}",
            f"{m_met['ioa']:.3f}",
        ]
        # Highlight RELIC R if better
        colors = ["white"] * 8
        # Highlight better NSE
        if r_met["nse"] > m_met["nse"]:
            colors[2] = "#E64B3520"
        else:
            colors[5] = "#4DBBD520"
        # Highlight lower RMSE
        if r_met["rmse"] < m_met["rmse"]:
            colors[3] = "#E64B3520"
        else:
            colors[6] = "#4DBBD520"
        cell_data.append(row)
        cell_colors.append(colors)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.4)

    # Header style
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=6.5)
        cell.set_facecolor("#F0F0F0")
        cell.set_edgecolor("0.8")

    # All cell edges
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.5)

    ax.set_title(
        f"Isoprene emission: multi-method evaluation ({relic_label} vs MEGAN Standard)",
        fontsize=8, fontweight="bold", pad=12,
    )

    _save(fig, out_dir, f"figS4_metric_comparison_table{suffix}", dpi)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def _generate_variant(
    relic_iso_ug_h: pd.Series,
    megan_iso_ug_h: pd.Series,
    obs_iso_ug_h: pd.Series,
    par_series: pd.Series,
    out_dir: Path,
    dpi: int,
    suffix: str = "",
    relic_label: str = "RELIC",
) -> None:
    """Generate all S-figures for one RELIC variant (Physics or ML)."""
    tag = f" [{relic_label}]" if suffix else ""

    print(f"[INFO]{tag} generating Fig S1: timeseries comparison...")
    fig_s1_timeseries(relic_iso_ug_h, megan_iso_ug_h, obs_iso_ug_h, out_dir, dpi,
                      suffix=suffix, relic_label=relic_label)

    print(f"[INFO]{tag} generating Fig S2: scatter comparison (all data)...")
    fig_s2_scatter(relic_iso_ug_h, megan_iso_ug_h, obs_iso_ug_h, out_dir, dpi,
                   suffix=suffix, relic_label=relic_label)

    if not par_series.empty:
        print(f"[INFO]{tag} generating Fig S2: scatter comparison (daytime)...")
        daytime = par_series > 50
        fig_s2_scatter(relic_iso_ug_h, megan_iso_ug_h, obs_iso_ug_h, out_dir, dpi,
                       daytime_mask=daytime, suffix=suffix, relic_label=relic_label)

    print(f"[INFO]{tag} generating Fig S3: diurnal composite...")
    fig_s3_diurnal(relic_iso_ug_h, megan_iso_ug_h, obs_iso_ug_h, out_dir, dpi,
                   suffix=suffix, relic_label=relic_label)

    print(f"[INFO]{tag} generating Fig S4: metric table...")
    fig_s4_metric_table(relic_iso_ug_h, megan_iso_ug_h, obs_iso_ug_h, par_series,
                        out_dir, dpi, suffix=suffix, relic_label=relic_label)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MEGAN comparison supplementary figures")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--case-root", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow an explicitly shortened run and crop all inputs to its model timeline.",
    )
    args = parser.parse_args(argv)

    case_root = Path(args.case_root) if args.case_root else Path(__file__).resolve().parent
    run_dir = resolve_run_dir(case_root, args.run_id)
    out_dir = run_dir / "figures"

    print(f"[INFO] run_dir = {run_dir}")
    print(f"[INFO] output  = {out_dir}")

    # Load data
    print("[INFO] loading RELIC data...")
    state, obs, forcing, case_params = load_relic_data(
        run_dir,
        allow_partial=args.allow_partial,
    )

    print("[INFO] loading MEGAN comparison...")
    megan_df = load_megan_comparison(case_root)

    # Unit conversions
    seconds_per_hour = _get_physical_constant("seconds_per_hour")
    mw = _get_isoprene_mw()
    nmol_per_umol = _read_case_value(case_params, "case.conversions.umol_to_nmol")
    micrograms_to_grams = _get_physical_constant(
        "relic.physical.constants.conversions.micrograms_to_grams"
    )

    # Obs isoprene: nmol m-2 s-1 -> ug m-2 h-1
    obs_iso_ug_h = obs["isoprene_emission"].astype(float) * (mw / nmol_per_umol) * seconds_per_hour

    # MEGAN comparison CSV: already in nmol m-2 s-1, convert to ug m-2 h-1
    megan_iso_ug_h = megan_df["megan_nmol_m2_s"].astype(float) * (mw / nmol_per_umol) * seconds_per_hour

    # PAR for daytime filtering
    par_series = forcing["ppfd"] if "ppfd" in forcing.columns else pd.Series(dtype=float)

    # References:
    # - Guenther et al. (2012), MEGAN species emissions are molar fluxes.
    # Mathematical expression:
    # - E_RELIC,ug/h = E_RELIC,mol/s * MW / c_ug_to_g * seconds_per_hour
    # Parameter meanings:
    # - MW: isoprene molecular weight [g mol-1]
    # - c_ug_to_g: micrograms-to-grams conversion [g ug-1]
    # ---- Physics path (no suffix) ----
    relic_phys_ug_h = (
        state["isoprene_emission"].astype(float)
        * (mw / micrograms_to_grams)
        * seconds_per_hour
    )
    physics_common = _build_common_comparison_support(
        relic=relic_phys_ug_h,
        megan=megan_iso_ug_h,
        observations=obs_iso_ug_h,
        ppfd=par_series,
    )
    print("=" * 60)
    print("[INFO] Generating Physics path figures...")
    print("=" * 60)
    _generate_variant(
        physics_common["relic"],
        physics_common["megan"],
        physics_common["observations"],
        physics_common["ppfd"],
        out_dir,
        args.dpi,
        suffix="",
        relic_label="RELIC",
    )

    # ---- ML path (_ML suffix) ----
    ml_key = "isoprene_emission_ML"
    if ml_key in state.columns:
        relic_ml_ug_h = (
            state[ml_key].astype(float)
            * (mw / micrograms_to_grams)
            * seconds_per_hour
        )
        ml_common = _build_common_comparison_support(
            relic=relic_ml_ug_h,
            megan=megan_iso_ug_h,
            observations=obs_iso_ug_h,
            ppfd=par_series,
        )
        print("=" * 60)
        print("[INFO] Generating ML path figures...")
        print("=" * 60)
        _generate_variant(
            ml_common["relic"],
            ml_common["megan"],
            ml_common["observations"],
            ml_common["ppfd"],
            out_dir,
            args.dpi,
            suffix="_ML",
            relic_label="RELIC ML",
        )
    else:
        print(f"[WARN] ML column '{ml_key}' not found in state; skipping ML figures.")

    print("[OK] all supplementary figures generated successfully")


if __name__ == "__main__":
    main()
