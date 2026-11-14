from __future__ import annotations

import pandas as pd

from tutorial.case.data.prepare_netcdf import classify_observation_quality


def test_classify_observation_quality_marks_proxy_targets() -> None:
    index = pd.date_range("2014-06-01", periods=48, freq="1h")
    # Repeated diurnal template across days.
    gpp = pd.Series([float(ts.hour) for ts in index], index=index, dtype="float64")

    quality = classify_observation_quality(
        sa1_gpp_raw_ratio=0.0,
        sa3_swc_raw_ratio=0.0,
        isoprene_obs_ratio=0.95,
        soil_moisture_mode="external_primary",
        gpp_obs_series=gpp,
    )

    assert quality["net_photosynthesis"]["class"] == "proxy_climatology_template"
    assert quality["net_photosynthesis"]["recommended_weight_multiplier"] == 0.25
    assert quality["soil_moisture"]["class"] == "proxy_external_reanalysis"
    assert quality["soil_moisture"]["recommended_weight_multiplier"] == 0.35
    assert quality["isoprene_emission"]["class"] == "real_observation"


def test_classify_observation_quality_keeps_real_targets() -> None:
    index = pd.date_range("2014-06-01", periods=72, freq="1h")
    # Day-varying signal to avoid template detection.
    gpp_values = [float(ts.hour) + float(ts.day) * 0.1 for ts in index]
    gpp = pd.Series(gpp_values, index=index, dtype="float64")

    quality = classify_observation_quality(
        sa1_gpp_raw_ratio=0.9,
        sa3_swc_raw_ratio=0.85,
        isoprene_obs_ratio=0.7,
        soil_moisture_mode="sa3_climatology_fill",
        gpp_obs_series=gpp,
    )

    assert quality["net_photosynthesis"]["class"] == "real_observation"
    assert quality["net_photosynthesis"]["recommended_weight_multiplier"] == 1.0
    assert quality["soil_moisture"]["class"] == "real_observation"
    assert quality["soil_moisture"]["recommended_weight_multiplier"] == 1.0
    assert quality["isoprene_emission"]["class"] == "real_observation_sparse"
