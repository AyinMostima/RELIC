"""GDD phenology onset event diagnostics against finite differences (ledger D73).

References: McMaster and Wilhelm (1997); RELIC theory pages phenology.md and
hybrid_events.md Section 4.
Mathematical expression: tau = t + dt (GDD_start - GDD_n)/(GDD_{n+1} - GDD_n);
d tau / d GDD_start = t_day / GDD_d; LAI jump = -LAI_max / dGDD_full.
Parameter meanings: GDD_d daily thermal time, t_day seconds per day, LAI_max the
morphology maximum, dGDD_full the full-growth offset.
Units: GDD [degC day], tau and dt [s], sensitivities [s degC-1 day-1] and
[m2 m-2 degC-1 day-1].
Domain/constraints: a single onset crossing in one daily step at constant temperature;
the owner evaluates its diagnostics in float32, so finite-difference agreement is 1e-4 relative.
Sign convention: raising GDD_start delays onset (positive tau sensitivity) and lowers
expansion-phase LAI (negative jump).
"""

from __future__ import annotations

import pytest
import torch

from relic.scientific.biosphere.phenology.models import GDDPhenology
from relic.utils.physical_constants import get_physical_constant

SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))
F64 = torch.float64


def _inputs(day: float, accumulated: float, previous_day: float) -> dict:
    """Declared GDD inputs of one daily step (the step length is the dt keyword)."""
    return {
        "day_of_year": torch.tensor(day, dtype=F64),
        "air_temperature": torch.tensor(293.15, dtype=F64),
        "air_temperature_min": torch.tensor(288.15, dtype=F64),
        "air_temperature_max": torch.tensor(298.15, dtype=F64),
        "phenology_gdd_accumulated": torch.tensor(accumulated, dtype=F64),
        "phenology_gdd_previous_day_of_year": torch.tensor(previous_day, dtype=F64),
    }


def _step(model: GDDPhenology, day: float, accumulated: float):
    return model.compute_phenological_state(
        _inputs(day, accumulated, day - 1.0), dt=SECONDS_PER_DAY
    )


def _daily_gdd(model: GDDPhenology) -> float:
    output = _step(model, 120.0, 0.0)
    return float(output.additional_outputs["phenology_gdd_accumulated"])


def test_onset_crossing_step_emits_event_time_and_sensitivities():
    model = GDDPhenology()
    gdd_start = float(model.phenology_params.gdd_threshold_start)
    daily = _daily_gdd(model)
    assert daily > 0.0
    # Start the crossing step 40 percent of one day below the threshold.
    before = gdd_start - 0.4 * daily
    output = _step(model, 130.0, before)
    extras = output.additional_outputs
    assert float(extras["diagnostics.phenology.onset_event_flag"]) == 1.0
    offset = float(extras["diagnostics.phenology.onset_event_time_offset"])
    assert offset == pytest.approx(0.4 * SECONDS_PER_DAY, rel=1e-6)
    tau_theta = float(extras["diagnostics.phenology.onset_event_time_sensitivity_gdd_start"])
    assert tau_theta == pytest.approx(SECONDS_PER_DAY / daily, rel=1e-6)
    full_offset = float(model.get_parameter_value("biosphere.phenology.gdd.full_growth_offset"))
    jump = float(extras["diagnostics.phenology.lai_sensitivity_jump_gdd_start"])
    assert jump == pytest.approx(-float(model.phenology_params.max_lai) / full_offset, rel=1e-6)


def test_no_crossing_steps_emit_zero_flag_and_zero_sensitivities():
    model = GDDPhenology()
    gdd_start = float(model.phenology_params.gdd_threshold_start)
    daily = _daily_gdd(model)
    for accumulated in (0.0, gdd_start - 3.0 * daily, gdd_start + 2.0 * daily):
        extras = _step(model, 140.0, accumulated).additional_outputs
        assert float(extras["diagnostics.phenology.onset_event_flag"]) == 0.0
        assert float(extras["diagnostics.phenology.onset_event_time_offset"]) == 0.0
        assert float(extras["diagnostics.phenology.onset_event_time_sensitivity_gdd_start"]) == 0.0
        assert float(extras["diagnostics.phenology.lai_sensitivity_jump_gdd_start"]) == 0.0


def test_event_time_sensitivity_matches_central_finite_differences_of_the_production_onset():
    model = GDDPhenology()
    gdd_start = float(model.phenology_params.gdd_threshold_start)
    daily = _daily_gdd(model)
    before = gdd_start - 0.5 * daily
    h = 1e-3 * daily

    def onset_offset(threshold: float) -> float:
        model.phenology_params.gdd_threshold_start = threshold
        return float(_step(model, 131.0, before).additional_outputs["diagnostics.phenology.onset_event_time_offset"])

    try:
        finite_difference = (onset_offset(gdd_start + h) - onset_offset(gdd_start - h)) / (2.0 * h)
        model.phenology_params.gdd_threshold_start = gdd_start
        analytic = float(_step(model, 131.0, before).additional_outputs["diagnostics.phenology.onset_event_time_sensitivity_gdd_start"])
    finally:
        model.phenology_params.gdd_threshold_start = gdd_start
    # The owner evaluates GDD diagnostics in float32; central differences with h = 1e-3 GDD_d
    # therefore agree to about 1e-4 relative.
    assert finite_difference == pytest.approx(analytic, rel=2e-4)


def test_lai_sensitivity_jump_matches_finite_differences_of_expansion_phase_lai():
    model = GDDPhenology()
    gdd_start = float(model.phenology_params.gdd_threshold_start)
    daily = _daily_gdd(model)
    full_offset = float(model.get_parameter_value("biosphere.phenology.gdd.full_growth_offset"))
    # A step that ends inside the expansion phase (after onset, before full growth).
    inside = gdd_start + 0.3 * full_offset
    h = 1e-3 * daily

    def lai(threshold: float) -> float:
        model.phenology_params.gdd_threshold_start = threshold
        return float(_step(model, 150.0, inside).leaf_area_index)

    try:
        finite_difference = (lai(gdd_start + h) - lai(gdd_start - h)) / (2.0 * h)
        model.phenology_params.gdd_threshold_start = gdd_start
        crossing = _step(model, 130.0, gdd_start - 0.4 * daily).additional_outputs
        jump = float(crossing["diagnostics.phenology.lai_sensitivity_jump_gdd_start"])
    finally:
        model.phenology_params.gdd_threshold_start = gdd_start
    # float32 owner arithmetic bounds the central-difference agreement to about 1e-4 relative.
    assert finite_difference == pytest.approx(jump, rel=2e-4)
