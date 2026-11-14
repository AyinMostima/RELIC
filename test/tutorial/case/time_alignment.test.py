from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tutorial.case.analyze_results import _load_analysis_inputs
from tutorial.case.plot_results import _has_phase_loss_data, _load_inputs
from tutorial.case.plot_megan_comparison import (
    _build_common_comparison_support,
    compute_metrics,
    load_relic_data,
)
from tutorial.case.utils.time_alignment import align_frame_to_reference_time


def test_align_frame_to_reference_time_trims_longer_observation_window() -> None:
    reference_index = pd.date_range("2014-06-01", periods=48, freq="h")
    observation_index = pd.date_range("2014-06-01", periods=375, freq="h")
    reference = pd.DataFrame({"model": range(48)}, index=reference_index)
    observations = pd.DataFrame(
        {"observation": range(375)},
        index=observation_index,
    )

    aligned = align_frame_to_reference_time(
        reference,
        observations,
        frame_name="observations",
        require_complete=False,
    )

    assert aligned.index.equals(reference.index)
    assert len(aligned) == 48
    assert aligned["observation"].iloc[-1] == 47


def test_align_frame_to_reference_time_rejects_missing_required_timestep() -> None:
    reference_index = pd.date_range("2014-06-01", periods=3, freq="h")
    reference = pd.DataFrame({"model": [1.0, 2.0, 3.0]}, index=reference_index)
    forcing = pd.DataFrame(
        {"ppfd": [0.0, 100.0]},
        index=reference_index.delete(1),
    )

    with pytest.raises(ValueError, match="forcing is missing 1 required timestamps"):
        align_frame_to_reference_time(
            reference,
            forcing,
            frame_name="forcing",
            require_complete=True,
        )


def test_align_frame_to_reference_time_rejects_non_overlapping_windows() -> None:
    reference = pd.DataFrame(
        {"model": [1.0]},
        index=pd.DatetimeIndex(["2014-06-01"]),
    )
    observations = pd.DataFrame(
        {"observation": [1.0]},
        index=pd.DatetimeIndex(["2015-06-01"]),
    )

    with pytest.raises(ValueError, match="does not overlap the model timeline"):
        align_frame_to_reference_time(
            reference,
            observations,
            frame_name="observations",
            require_complete=False,
        )


def test_align_frame_to_reference_time_rejects_duplicate_timestamps() -> None:
    duplicate_index = pd.DatetimeIndex(["2014-06-01", "2014-06-01"])
    reference = pd.DataFrame({"model": [1.0]}, index=pd.DatetimeIndex(["2014-06-01"]))
    observations = pd.DataFrame(
        {"observation": [1.0, 2.0]},
        index=duplicate_index,
    )

    with pytest.raises(ValueError, match="observations contains duplicate timestamps"):
        align_frame_to_reference_time(
            reference,
            observations,
            frame_name="observations",
            require_complete=False,
        )


def _write_plot_input_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    timeseries_dir = run_dir / "timeseries"
    snapshot_dir = run_dir / "input_snapshot"
    timeseries_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)

    full_time = pd.date_range("2014-06-01", periods=4, freq="h")
    model_time = full_time[:2]
    xr.Dataset(
        {
            "soil_moisture": ("time", [0.45, 0.44]),
            "net_photosynthesis": ("time", [0.0, 1.0]),
            "isoprene_emission": ("time", [0.0, 1.0e-9]),
        },
        coords={"time": model_time},
    ).to_netcdf(timeseries_dir / "state_hourly.nc")
    xr.Dataset(
        {
            "soil_moisture": ("time", [0.45, 0.44, 0.43, 0.42]),
            "net_photosynthesis": ("time", [0.0, 1.0, 2.0, 3.0]),
            "isoprene_emission": ("time", [0.0, 1.0, 2.0, 3.0]),
        },
        coords={"time": full_time},
    ).to_netcdf(snapshot_dir / "real_obs.nc")
    xr.Dataset(
        {
            "ppfd": ("time", [0.0, 100.0, 200.0, 300.0]),
            "air_temperature": ("time", [298.0, 299.0, 300.0, 301.0]),
            "relative_humidity": ("time", [0.8, 0.8, 0.7, 0.7]),
            "vapor_pressure_deficit": ("time", [100.0, 110.0, 120.0, 130.0]),
            "precipitation_flux": ("time", [0.0, 0.0, 0.0, 0.0]),
            "leaf_area_index": ("time", [5.0, 5.0, 5.0, 5.0]),
        },
        coords={"time": full_time},
    ).to_netcdf(snapshot_dir / "site_forcing.nc")
    (snapshot_dir / "case_parameters.yaml").write_text(
        "parameters: {}\n",
        encoding="utf-8",
    )
    return run_dir


def test_plot_results_inputs_reject_incomplete_model_timeline(tmp_path) -> None:
    run_dir = _write_plot_input_run(tmp_path)

    with pytest.raises(ValueError, match="incomplete model timeline: 2/4 timestamps"):
        _load_inputs(run_dir)


def test_plot_results_inputs_use_exact_model_timeline_when_partial_is_explicit(
    tmp_path,
) -> None:
    run_dir = _write_plot_input_run(tmp_path)

    state, observations, forcing, _ = _load_inputs(run_dir, allow_partial=True)

    assert observations.index.equals(state.index)
    assert forcing.index.equals(state.index)


def test_megan_plot_inputs_reject_incomplete_model_timeline(tmp_path) -> None:
    run_dir = _write_plot_input_run(tmp_path)

    with pytest.raises(ValueError, match="incomplete model timeline: 2/4 timestamps"):
        load_relic_data(run_dir)


def test_megan_plot_inputs_use_exact_model_timeline_when_partial_is_explicit(
    tmp_path,
) -> None:
    run_dir = _write_plot_input_run(tmp_path)

    state, observations, forcing, _ = load_relic_data(
        run_dir,
        allow_partial=True,
    )

    assert observations.index.equals(state.index)
    assert forcing.index.equals(state.index)


def test_megan_comparison_uses_one_common_finite_time_support() -> None:
    index = pd.date_range("2014-06-01", periods=4, freq="h")
    relic = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    observations = pd.Series([1.1, 2.1, 3.1, 4.1], index=index)
    megan = pd.Series([0.9, 2.9, 3.9], index=index.delete(1))
    ppfd = pd.Series([0.0, 100.0, 200.0, 300.0], index=index)

    common = _build_common_comparison_support(
        relic=relic,
        megan=megan,
        observations=observations,
        ppfd=ppfd,
    )

    assert common.index.equals(index.delete(1))
    assert list(common.columns) == ["relic", "megan", "observations", "ppfd"]
    assert common.notna().all().all()


def test_megan_metrics_perfect_prediction_contract() -> None:
    observations = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = compute_metrics(observations.copy(), observations)

    assert set(metrics) == {"n", "r", "rmse", "bias", "slope", "nse", "ioa"}
    assert metrics["n"] == 4
    assert metrics["r"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["slope"] == pytest.approx(1.0)
    assert metrics["nse"] == pytest.approx(1.0)
    assert metrics["ioa"] == pytest.approx(1.0)


def test_megan_metrics_use_pairwise_finite_support_for_imperfect_prediction() -> None:
    model = np.array([1.0, 2.5, np.nan, 5.0, 8.0])
    observations = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    expected_model = np.array([1.0, 2.5, 5.0])
    expected_observations = np.array([1.0, 2.0, 4.0])

    metrics = compute_metrics(model, observations)

    assert metrics["n"] == 3
    assert metrics["r"] == pytest.approx(
        np.corrcoef(expected_model, expected_observations)[0, 1]
    )
    assert metrics["rmse"] > 0.0
    assert metrics["nse"] < 1.0
    assert all(np.isfinite(metrics[key]) for key in metrics if key != "n")


def test_analysis_inputs_reject_incomplete_model_timeline(tmp_path) -> None:
    run_dir = _write_plot_input_run(tmp_path)

    with pytest.raises(ValueError, match="incomplete model timeline: 2/4 timestamps"):
        _load_analysis_inputs(run_dir)


def test_analysis_inputs_use_exact_model_timeline_when_partial_is_explicit(
    tmp_path,
) -> None:
    run_dir = _write_plot_input_run(tmp_path)

    state, observations, forcing = _load_analysis_inputs(
        run_dir,
        allow_partial=True,
    )

    assert observations.index.equals(state.index)
    assert forcing.index.equals(state.index)


def test_loss_convergence_is_unavailable_without_optimization_outputs(tmp_path) -> None:
    run_dir = tmp_path / "simulation_only"
    run_dir.mkdir()

    assert not _has_phase_loss_data(run_dir)

    optimization_dir = run_dir / "optimization"
    optimization_dir.mkdir()
    (optimization_dir / "phase2_loss.csv").write_text(
        "epoch,window,loss\n0,0,1.0\n",
        encoding="utf-8",
    )

    assert _has_phase_loss_data(run_dir)
