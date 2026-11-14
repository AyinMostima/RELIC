"""Analyze a tutorial/case run and print key diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relic.utils.physical_constants import get_physical_constant
from tutorial.case.utils.time_alignment import (
    align_frame_to_reference_time,
    validate_model_time_coverage,
)


def _load_yaml(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _get_isoprene_molecular_weight() -> float:
    return float(
        get_physical_constant(
            "relic.physical.constants.molecular_weights.isoprene"
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze run outputs for tutorial/case.")
    parser.add_argument("--case-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow an explicitly shortened run and crop all inputs to its model timeline.",
    )
    return parser.parse_args()


def compute_metrics(model: pd.Series, obs: pd.Series) -> Dict[str, float]:
    aligned = pd.concat([model.rename("m"), obs.rename("o")], axis=1)
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return {"R": float("nan"), "RMSE": float("nan"), "bias": float("nan"), "n": 0.0}
    m = aligned["m"].to_numpy(dtype=float)
    o = aligned["o"].to_numpy(dtype=float)
    if m.size >= 2 and float(np.std(m)) > 0.0 and float(np.std(o)) > 0.0:
        r = float(np.corrcoef(m, o)[0, 1])
    else:
        r = float("nan")
    rmse = float(np.sqrt(np.mean((m - o) ** 2)))
    bias = float(np.mean(m - o))
    return {"R": r, "RMSE": rmse, "bias": bias, "n": float(m.size)}


def _load_analysis_inputs(
    run_dir: Path,
    *,
    allow_partial: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_path = run_dir / "timeseries" / "state_hourly.nc"
    obs_path = run_dir / "input_snapshot" / "real_obs.nc"
    forcing_path = run_dir / "input_snapshot" / "site_forcing.nc"
    for path in (state_path, obs_path, forcing_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing analysis input: {path}")

    state_ds = xr.load_dataset(state_path)
    obs_ds = xr.load_dataset(obs_path)
    forcing_ds = xr.load_dataset(forcing_path)

    if "epoch" in state_ds.coords:
        epoch_values = np.asarray(state_ds["epoch"].values)
        if np.any(epoch_values == -1):
            state_ds = state_ds.where(state_ds["epoch"] == -1, drop=True)
        else:
            state_ds = state_ds.where(
                state_ds["epoch"] == int(np.nanmax(epoch_values)),
                drop=True,
            )

    requested_state_variables = [
        "soil_moisture",
        "net_photosynthesis",
        "gross_primary_productivity",
        "isoprene_emission",
        "water_stress_factor",
        "soil_moisture_ML",
        "net_photosynthesis_ML",
        "isoprene_emission_ML",
    ]
    available_state_variables = [
        name for name in requested_state_variables if name in state_ds.data_vars
    ]
    state = state_ds[available_state_variables].to_dataframe().reset_index()
    state["time"] = pd.to_datetime(state["time"])
    state = state.set_index("time").sort_index()

    obs = obs_ds[
        ["soil_moisture", "net_photosynthesis", "isoprene_emission"]
    ].to_dataframe().reset_index()
    obs["time"] = pd.to_datetime(obs["time"])
    obs = obs.set_index("time").sort_index()

    forcing = forcing_ds[
        [
            "ppfd",
            "air_temperature",
            "relative_humidity",
            "vapor_pressure_deficit",
            "precipitation_flux",
            "leaf_area_index",
        ]
    ].to_dataframe().reset_index()
    forcing["time"] = pd.to_datetime(forcing["time"])
    forcing = forcing.set_index("time").sort_index()

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
    return state, obs, forcing


def main() -> None:
    args = parse_args()
    run_dir = (args.case_root / "output" / "runs" / args.run_id).resolve()
    state, obs, forcing = _load_analysis_inputs(
        run_dir,
        allow_partial=args.allow_partial,
    )

    case_cfg = _load_yaml(run_dir / "input_snapshot" / "case_parameters.yaml")
    conv = float(case_cfg["parameters"]["case.conversions.umol_to_nmol"]["value"])
    iso_mw = _get_isoprene_molecular_weight()
    seconds_per_hour = float(get_physical_constant("seconds_per_hour"))
    micrograms_to_grams = float(
        get_physical_constant(
            "relic.physical.constants.conversions.micrograms_to_grams"
        )
    )

    # References:
    # - Guenther et al. (2012), MEGAN species emissions are molar fluxes.
    # Mathematical expression:
    # - E_model,ug/h = E_model,mol/s * MW / c_ug_to_g * seconds_per_hour
    # - E_obs,ug/h = E_obs,nmol/s * MW / nmol_per_umol * seconds_per_hour
    # Parameter meanings:
    # - MW: isoprene molecular weight [g mol-1]
    # - c_ug_to_g: micrograms-to-grams conversion [g ug-1]
    iso_model_ug_h = (
        state.get("isoprene_emission", pd.Series(dtype=float)).astype(float)
        * (iso_mw / micrograms_to_grams)
        * seconds_per_hour
    )
    iso_obs_ug_h = obs["isoprene_emission"].astype(float) * (iso_mw / conv) * seconds_per_hour
    iso_ml_ug_h = pd.Series(dtype=float)
    if "isoprene_emission_ML" in state:
        iso_ml_ug_h = (
            state["isoprene_emission_ML"].astype(float)
            * (iso_mw / micrograms_to_grams)
            * seconds_per_hour
        )

    if not state.empty:
        print("METRIC_SOIL", compute_metrics(state["soil_moisture"], obs["soil_moisture"]))
        print("METRIC_GPP", compute_metrics(state["net_photosynthesis"], obs["net_photosynthesis"]))
        print("METRIC_ISOPRENE", compute_metrics(iso_model_ug_h, iso_obs_ug_h))
        if "soil_moisture_ML" in state:
            print("METRIC_SOIL_ML", compute_metrics(state["soil_moisture_ML"], obs["soil_moisture"]))
        if "net_photosynthesis_ML" in state:
            print(
                "METRIC_GPP_ML",
                compute_metrics(state["net_photosynthesis_ML"], obs["net_photosynthesis"]),
            )
        if not iso_ml_ug_h.empty:
            print("METRIC_ISOPRENE_ML", compute_metrics(iso_ml_ug_h, iso_obs_ug_h))
    else:
        print("METRIC_SOIL", {"R": float("nan"), "RMSE": float("nan"), "bias": float("nan"), "n": 0.0})
        print("METRIC_GPP", {"R": float("nan"), "RMSE": float("nan"), "bias": float("nan"), "n": 0.0})
        print("METRIC_ISOPRENE", {"R": float("nan"), "RMSE": float("nan"), "bias": float("nan"), "n": 0.0})

    if not state.empty:
        iso = pd.concat(
            [iso_model_ug_h.rename("m"), iso_obs_ug_h.rename("o"), forcing["ppfd"]],
            axis=1,
        ).dropna(subset=["m", "o"])
        obs_zero = iso["o"] <= 1.0e-9
        model_zero = iso["m"] <= 1.0e-9
        obs_positive = ~obs_zero
        print(
            "ZERO_INFLATION",
            {
                "obs_valid": int(len(iso)),
                "obs_zero_ratio": float(obs_zero.mean()),
                "model_zero_ratio": float(model_zero.mean()),
                "TP": int((obs_positive & ~model_zero).sum()),
                "TN": int((obs_zero & model_zero).sum()),
                "FP": int((obs_zero & ~model_zero).sum()),
                "FN": int((obs_positive & model_zero).sum()),
                "positive_only": compute_metrics(iso.loc[obs_positive, "m"], iso.loc[obs_positive, "o"]),
                "daytime_only": compute_metrics(iso.loc[iso["ppfd"] > 10.0, "m"], iso.loc[iso["ppfd"] > 10.0, "o"]),
                "nighttime_only": compute_metrics(iso.loc[iso["ppfd"] <= 10.0, "m"], iso.loc[iso["ppfd"] <= 10.0, "o"]),
            },
        )
    else:
        print("ZERO_INFLATION", {"obs_valid": 0})

    case_parameters = case_cfg["parameters"]
    if not isinstance(case_parameters, dict):
        raise ValueError("Invalid case parameter mapping.")
    kelvin_offset = float(get_physical_constant("celsius_to_kelvin"))
    saturation_coeff_kpa = float(
        case_parameters["case.atmosphere.vpd_formula.saturation_coeff_kpa"]["value"]
    )
    exponent_numerator = float(
        case_parameters["case.atmosphere.vpd_formula.exponent_numerator"]["value"]
    )
    exponent_denominator_offset_c = float(
        case_parameters[
            "case.atmosphere.vpd_formula.exponent_denominator_offset_c"
        ]["value"]
    )

    # References: Allen et al. (1998), FAO-56 saturation vapor pressure relation.
    # Mathematical expression: e_s = a * exp(b*T_c/(T_c+c)); VPD = e_s*(1-RH).
    # Parameter meanings: a [kPa], b [1], c [degC], T_c [degC], RH [1], VPD [Pa].
    temp_c = forcing["air_temperature"].astype(float) - kelvin_offset
    rh = forcing["relative_humidity"].astype(float)
    es = saturation_coeff_kpa * np.exp(
        (exponent_numerator * temp_c)
        / (temp_c + exponent_denominator_offset_c)
    ) * 1000.0
    vpd_calc = np.maximum(0.0, es * (1.0 - rh))
    vpd_error = np.abs(forcing["vapor_pressure_deficit"].astype(float) - vpd_calc)
    print(
        "FORCING_CHECK",
        {
            "ppfd_zero_ratio": float((forcing["ppfd"] <= 1.0e-6).mean()),
            "ppfd_max": float(forcing["ppfd"].max()),
            "precip_mean_mm_h": float((forcing["precipitation_flux"] * seconds_per_hour).mean()),
            "lai_min": float(forcing["leaf_area_index"].min()),
            "lai_max": float(forcing["leaf_area_index"].max()),
            "vpd_abs_err_mean_pa": float(vpd_error.mean()),
            "vpd_abs_err_max_pa": float(vpd_error.max()),
        },
    )

    for phase_id in [1, 2, 3, 4]:
        loss_path = run_dir / "optimization" / f"phase{phase_id}_loss.csv"
        if not loss_path.exists():
            continue
        loss_df = pd.read_csv(loss_path)
        print(
            f"LOSS_PHASE{phase_id}",
            {
                "first": float(loss_df["loss"].iloc[0]),
                "last": float(loss_df["loss"].iloc[-1]),
                "min": float(loss_df["loss"].min()),
                "max": float(loss_df["loss"].max()),
                "n": int(loss_df.shape[0]),
            },
        )

    grad_path = run_dir / "optimization" / "gradient_norms.csv"
    if grad_path.exists():
        grads = pd.read_csv(grad_path)
        grad_nonfinite = int((~np.isfinite(grads["grad_norm"].to_numpy(dtype=float))).sum())
        print("GRAD_NONFINITE", grad_nonfinite)
        print("GRAD_STATS", grads.groupby("phase")["grad_norm"].agg(["min", "max", "mean"]).to_dict())
    else:
        print("GRAD_NONFINITE", 0)
        print("GRAD_STATS", {})

    params_path = run_dir / "optimization" / "optimized_parameters.yaml"
    if not params_path.exists():
        params_path = run_dir / "optimization" / "best_params.yaml"
    if params_path.exists():
        opt_params = _load_yaml(params_path)
        print("OPTIMIZED_PARAMETERS", opt_params)
    else:
        print("OPTIMIZED_PARAMETERS", {})


if __name__ == "__main__":
    main()
