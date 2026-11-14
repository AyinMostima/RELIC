from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml

from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.coupling.scales.temporal.core import (
    FREQUENCY_HIERARCHY,
    CFTimeProcessor,
    PeriodIndexer,
    TimeCoordinate,
    TimeWeightGenerator,
    ValidationError,
    get_calendar_definitions,
)
from relic.parameters import get_global_manager
from relic.utils.time_constants import SECONDS_PER_DAY, SECONDS_PER_MINUTE


def test_calendar_handler_reads_days_per_year_from_parameter_manager():
    """CalendarHandler should respect ParameterManager calendar constants."""
    pm = get_global_manager()
    key = "coupling.scales.temporal.calendars.gregorian.days_per_year"
    original_value = pm.get(key, units="days", required=True)
    updated_value = float(original_value) + 1.0

    try:
        pm.set_value(key, updated_value, units="days", validate=True, notify=False)
        handler = CalendarHandler(calendar_type="gregorian")
        assert handler.days_per_year == pytest.approx(updated_value)
    finally:
        pm.set_value(key, original_value, units="days", validate=True, notify=False)


_UNUSED_CALENDAR_THRESHOLDS = (
    "min_overlap_threshold",
    "coverage_threshold",
    "calendar_overlap_minimum",
    "calendar_coverage_minimum",
    "calendar_coverage_maximum",
)


class _RecordingManager:
    """Delegate to the canonical manager while recording scalar lookups."""

    def __init__(self):
        self._canonical = get_global_manager()
        self.keys = []

    def get(self, key, *args, **kwargs):
        self.keys.append(key)
        return self._canonical.get(key, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._canonical, name)


def test_calendar_handler_reads_no_unconsumed_threshold_parameters():
    """Weights and coverage use no threshold, so none is read or stored."""
    manager = _RecordingManager()

    handler = CalendarHandler(calendar_type="noleap", parameter_manager=manager)

    read = {key.rsplit(".", 1)[-1] for key in manager.keys}
    assert read.isdisjoint(_UNUSED_CALENDAR_THRESHOLDS)
    for attribute in (
        "min_overlap_threshold",
        "coverage_threshold",
        "overlap_minimum",
        "coverage_minimum",
        "coverage_maximum",
    ):
        assert not hasattr(handler, attribute)


def test_unreferenced_temporal_scale_parameters_are_absent_from_yaml():
    path = (
        Path(__file__).resolve().parents[5]
        / "relic"
        / "parameters"
        / "database"
        / "relic"
        / "coupling"
        / "scales_parameters.yaml"
    )
    parameters = yaml.safe_load(path.read_text(encoding="utf-8-sig"))["parameters"]

    for key in (
        "coupling.scales.temporal.percentile_q",
        "coupling.scales.temporal.min_overlap_threshold",
        "coupling.scales.temporal.calendar_overlap_minimum",
        "coupling.scales.temporal.calendar_coverage_minimum",
        "coupling.scales.temporal.calendar_coverage_maximum",
        "coupling.scales.temporal.disaggregation.smooth_transitions",
        "coupling.scales.temporal.disaggregation.stochastic_noise_level",
    ):
        assert key not in parameters
    # The shared acceptance threshold stays; TemporalAggregator and
    # QualityConfig read it.
    assert "coupling.scales.temporal.coverage_threshold" in parameters


def test_calendar_definitions_use_injected_parameter_manager():
    """Calendar definitions must use the explicitly injected manager."""
    manager = Mock()
    manager.get_namespace_parameters.return_value = {
        "coupling.scales.temporal.calendars.test_calendar.days_per_year": 360.0,
        "coupling.scales.temporal.calendars.test_calendar.months_per_year": 12,
    }

    definitions = get_calendar_definitions(manager)

    assert definitions == {
        "test_calendar": {
            "days_per_year": 360.0,
            "months_per_year": 12,
        }
    }
    manager.get_namespace_parameters.assert_called_once_with(
        namespace="coupling.scales.temporal.calendars",
        include_subnamespaces=True,
        return_specs=False,
    )


def test_calendar_frequencies_do_not_publish_fixed_seconds():
    """Monthly and yearly periods must remain calendar dependent."""
    for frequency in ("M", "Q", "S", "Y"):
        metadata = FREQUENCY_HIERARCHY[frequency]
        assert "seconds" not in metadata
        assert metadata["calendar_dependent"] is True


def test_time_components_use_injected_parameter_manager():
    """CF time components must preserve explicit manager injection."""
    manager = Mock()
    device = torch.device("cpu")

    processor = CFTimeProcessor(device=device, parameter_manager=manager)
    generator = TimeWeightGenerator(device=device, parameter_manager=manager)
    indexer = PeriodIndexer(device=device, parameter_manager=manager)

    assert processor.parameter_manager is manager
    assert generator.parameter_manager is manager
    assert generator.cf_processor.parameter_manager is manager
    assert indexer.parameter_manager is manager
    assert indexer.cf_processor.parameter_manager is manager


def test_cf_time_processor_rejects_non_increasing_coordinates():
    """Scientific time coordinates must be strictly increasing."""
    processor = CFTimeProcessor(device=torch.device("cpu"))

    with pytest.raises(ValidationError, match="strictly increasing"):
        processor.parse_cf_time(
            torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64),
            units="days since 2000-01-01",
            calendar="noleap",
        )


def test_cf_standard_calendar_normalizes_to_mixed_gregorian():
    """CF standard is the mixed Julian/Gregorian calendar, not proleptic Gregorian."""
    processor = CFTimeProcessor(device=torch.device("cpu"))

    assert processor._normalize_calendar("standard") == "gregorian"


def test_period_indexer_applies_julian_century_leap_rule():
    """The Julian calendar treats every year divisible by four as leap."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    coordinate = indexer.create_period_index(
        start_time="1900-01-01",
        end_time="1901-01-01",
        frequency="Y",
        calendar="julian",
        units="days since 1900-01-01",
    )

    expected = torch.tensor([[0.0, 366.0]], dtype=torch.float64)
    assert torch.equal(coordinate.bounds.cpu(), expected)


def test_period_indexer_applies_cf_mixed_gregorian_cutover():
    """In the mixed calendar, 1582-10-04 is followed by 1582-10-15."""
    indexer = PeriodIndexer(device=torch.device("cpu"))
    last_julian_day = (1582, 10, 4, 0, 0, 0.0)

    next_day = indexer._increment_datetime(last_julian_day, "D", "gregorian")

    assert next_day == (1582, 10, 15, 0, 0, 0.0)
    with pytest.raises(ValidationError, match="calendar reform gap"):
        indexer._datetime_to_day_index((1582, 10, 10, 0, 0, 0.0), "gregorian")


def test_proleptic_gregorian_remains_continuous_across_1582_cutover_dates():
    """The proleptic Gregorian calendar must not inherit the reform gap."""
    indexer = PeriodIndexer(device=torch.device("cpu"))
    october_4 = indexer._datetime_to_day_index(
        (1582, 10, 4, 0, 0, 0.0),
        "proleptic_gregorian",
    )
    october_15 = indexer._datetime_to_day_index(
        (1582, 10, 15, 0, 0, 0.0),
        "proleptic_gregorian",
    )

    assert october_15 - october_4 == pytest.approx(11.0)
    assert indexer._increment_datetime(
        (1582, 10, 4, 0, 0, 0.0),
        "D",
        "proleptic_gregorian",
    ) == (1582, 10, 5, 0, 0, 0.0)


def _decoder_sample_day_indices():
    """Day indices around leap, century and cutover years plus a fixed stride."""
    indices = set(range(0, 1500))
    for year in (100, 101, 400, 401, 1582, 1583, 1600, 1900, 2000, 2020, 2024, 2100, 2400):
        centre = (year - 1) * 365 + (year - 1) // 4
        indices.update(range(centre - 500, centre + 900))
    indices.update(range(0, 900000, 7919))
    return sorted(index for index in indices if index >= 0)


@pytest.mark.parametrize(
    ("calendar", "cf_calendar"),
    [
        ("noleap", "noleap"),
        ("all_leap", "all_leap"),
        ("360_day", "360_day"),
        ("julian", "julian"),
        ("proleptic_gregorian", "proleptic_gregorian"),
        ("gregorian", "standard"),
    ],
)
def test_day_index_decoder_matches_independent_cftime_dates(calendar, cf_calendar):
    """The absolute day decoder agrees with cftime for every sampled day."""
    import cftime

    indexer = PeriodIndexer(device=torch.device("cpu"))
    # References: CF Conventions 1.8, Section 4.4.1; cftime.num2date reference API.
    # Mathematical expression: (y, m, d) = decode_C(D), D whole days since
    #   0001-01-01T00:00 in calendar C.
    # Parameter meanings: D is the absolute day index, C the calendar, (y, m, d)
    #   the civil year, month and day.
    # Units: D [day]; y, m, d are dimensionless integers.
    # Domain/constraints: D >= 0 and year >= 1; samples cover leap, century,
    #   400-year and 1582 cutover neighbourhoods and years 1 to about 2460.
    # Sign convention: D increases forward in time; no quantity is transferred.
    samples = _decoder_sample_day_indices()
    reference_dates = cftime.num2date(
        samples, "days since 0001-01-01 00:00:00", calendar=cf_calendar
    )
    for day_index, reference in zip(samples, reference_dates):
        decoded = indexer._day_index_to_datetime(float(day_index), calendar)
        assert decoded[:3] == (reference.year, reference.month, reference.day), day_index
        assert decoded[3:] == (0, 0, 0.0)
        assert indexer._datetime_to_day_index(decoded, calendar) == float(day_index)


@pytest.mark.parametrize("calendar", ["julian", "proleptic_gregorian", "gregorian"])
def test_day_index_decoder_does_not_enumerate_years(monkeypatch, calendar):
    """Decoding a recent date needs a bounded number of leap-rule evaluations."""
    indexer = PeriodIndexer(device=torch.device("cpu"))
    day_index = indexer._datetime_to_day_index((2020, 3, 1, 0, 0, 0.0), calendar)
    original = indexer._is_leap_year
    calls = []

    def counting(year, calendar_name):
        calls.append(year)
        return original(year, calendar_name)

    monkeypatch.setattr(indexer, "_is_leap_year", counting)

    assert indexer._day_index_to_datetime(day_index, calendar) == (2020, 3, 1, 0, 0, 0.0)
    assert len(calls) <= 16


def test_period_indexer_builds_one_half_open_daily_interval():
    """A one-day domain produces exactly one half-open interval [0, 1)."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    coordinate = indexer.create_period_index(
        start_time="2001-01-01",
        end_time="2001-01-02",
        frequency="D",
        calendar="noleap",
        units="days since 2001-01-01",
    )

    expected = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    assert torch.equal(coordinate.bounds.cpu(), expected)


@pytest.mark.parametrize(
    ("calendar", "elapsed_days", "expected_month_index"),
    [
        ("noleap", 0.0, 0),
        ("noleap", 31.0, 1),
        ("noleap", 59.0, 2),
        ("noleap", 365.0, 12),
        ("360_day", 0.0, 0),
        ("360_day", 30.0, 1),
        ("360_day", 60.0, 2),
        ("360_day", 360.0, 12),
    ],
)
def test_period_indexer_computes_monotone_elapsed_month_index(
    calendar,
    elapsed_days,
    expected_month_index,
):
    """Elapsed month indices must use the canonical CF calendar owner."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    actual = indexer.elapsed_month_index(
        torch.tensor(elapsed_days * 86400.0, dtype=torch.float64),
        calendar=calendar,
    )

    assert actual == expected_month_index


@pytest.mark.parametrize("elapsed_seconds", [-1.0, float("inf"), float("nan")])
def test_period_indexer_rejects_invalid_elapsed_time(elapsed_seconds):
    """Calendar scheduling must fail closed for negative or non-finite time."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    with pytest.raises(ValidationError, match="elapsed_seconds"):
        indexer.elapsed_month_index(elapsed_seconds, calendar="noleap")


@pytest.mark.parametrize(
    "calendar",
    ["standard", "gregorian", "proleptic_gregorian", "noleap", "365_day",
     "all_leap", "366_day", "360_day", "julian"],
)
def test_elapsed_month_index_preserves_absolute_origin_and_year_crossing(calendar):
    """A real epoch changes the decoded date, not the absolute month ordinal."""
    import cftime

    indexer = PeriodIndexer(device=torch.device("cpu"))
    origin = "1999-12-30 23:30:00" if calendar == "360_day" else "1999-12-31 23:30:00"
    # References: CF Conventions 1.8, Section 4.4; cftime.num2date reference API.
    # Mathematical expression: d=CF_decode_C(t; seconds since o);
    #   m=12*(year(d)-1)+month(d)-1; h=30*seconds_per_minute.
    # Parameter meanings: C is the source calendar, o its reference instant,
    #   t elapsed time, d the independent decoded date, m the ordinal, h half-hour.
    #   seconds_per_minute is the SSOT duration of one minute.
    # Units: t,h [s]; o,d are calendar timestamps; m is dimensionless.
    # Domain/constraints: valid year>=1 origins and times around a year boundary;
    #   expected dates come from cftime, not the production decoder.
    # Sign convention: positive t advances time; m cannot decrease; no flux.
    half_hour = float(SECONDS_PER_MINUTE) * 30
    for elapsed in (0.0, half_hour - 0.001, half_hour, half_hour + 0.001):
        reference = cftime.num2date(elapsed, f"seconds since {origin}", calendar=calendar)
        expected = 12 * (reference.year - 1) + reference.month - 1
        actual = indexer.elapsed_month_index(elapsed, calendar=calendar, time_origin=origin)
        assert actual == expected


def test_elapsed_month_index_keeps_all_96_half_hour_leap_day_coordinates():
    """Two leap-year grids retain 96 half-hour coordinates before the next month."""
    import cftime

    indexer = PeriodIndexer(device=torch.device("cpu"))
    # References: CF Conventions 1.8, Section 4.4; cftime.num2date reference API.
    # Mathematical expression: t_i=i*h, h=30*seconds_per_minute, i=0,...,47;
    #   m_i=12*(y-1)+1 on February 29; m(day_end)=12*(y-1)+2.
    # Parameter meanings: y is the selected leap year, i the half-hour position,
    #   t_i elapsed time, h its spacing, m_i the absolute month ordinal.
    #   day_end is one day after the origin; seconds_per_minute is the SSOT scale.
    # Units: t_i,h,day_end [s]; y,i,m_i are dimensionless.
    # Domain/constraints: two legal proleptic Gregorian leap-day grids; all 96
    #   coordinates must retain their calendar date and ordinal without filtering.
    # Sign convention: elapsed time advances toward March; no state or flux change.
    half_hour = float(SECONDS_PER_MINUTE) * 30
    checked = 0
    for year in (2016, 2020):
        origin = f"{year}-02-29T00:00:00"
        for offset in range(48):
            elapsed = offset * half_hour
            reference = cftime.num2date(elapsed, f"seconds since {origin}", calendar="proleptic_gregorian")
            assert reference.month == 2 and reference.day == 29
            actual = indexer.elapsed_month_index(
                elapsed, calendar="proleptic_gregorian", time_origin=origin,
            )
            assert actual == 12 * (year - 1) + 1
            checked += 1
        assert indexer.elapsed_month_index(
            float(SECONDS_PER_DAY), calendar="proleptic_gregorian", time_origin=origin,
        ) == 12 * (year - 1) + 2
    assert checked == 96


def test_elapsed_month_index_accepts_the_actual_profile005_origin_format():
    """The recorded producer uses a naive ISO timestamp, without timezone removal."""
    indexer = PeriodIndexer(device=torch.device("cpu"))
    assert indexer.elapsed_month_index(
        0.0, calendar="proleptic_gregorian", time_origin="2020-01-02T00:00:00",
    ) == 12 * 2019


@pytest.mark.parametrize(
    ("calendar", "origin", "elapsed_days"),
    [("gregorian", "1582-10-04", 18.0),
     ("standard", "1582-10-04", 18.0),
     ("proleptic_gregorian", "1582-10-04", 28.0),
     ("noleap", "2000-02-28", 1.0),
     ("360_day", "2000-02-30", 1.0),
     ("all_leap", "2001-02-29", 1.0),
     ("julian", "1900-02-29", 1.0)],
)
def test_elapsed_month_index_reuses_each_calendar_reference_domain(calendar, origin, elapsed_days):
    """The existing mixed-calendar, leap and artificial-month rules stay distinct."""
    import cftime

    indexer = PeriodIndexer(device=torch.device("cpu"))
    # References: CF Conventions 1.8, Section 4.4; cftime.num2date reference API.
    # Mathematical expression: t=D*seconds_per_day; d=CF_decode_C(t;o);
    #   m=12*(year(d)-1)+month(d)-1.
    # Parameter meanings: D is elapsed days, C the declared calendar, o the
    #   reference instant, d the independently decoded date, m its month ordinal.
    #   seconds_per_day is the SSOT duration of one day.
    # Units: t [s], D [day]; d,o timestamps; m dimensionless.
    # Domain/constraints: valid civil or artificial dates; preserve each calendar's
    #   leap and reform rules rather than substituting another calendar.
    # Sign convention: positive t moves to the next month; no quantity transfer.
    elapsed = elapsed_days * float(SECONDS_PER_DAY)
    reference = cftime.num2date(elapsed, f"seconds since {origin}", calendar=calendar)
    assert reference.day == 1
    assert indexer.elapsed_month_index(elapsed, calendar=calendar, time_origin=origin) == (
        12 * (reference.year - 1) + reference.month - 1
    )


@pytest.mark.parametrize("origin", ["", " ", 123, False, "not-a-date", "2000-13-01", "0000-01-01"])
def test_elapsed_month_index_rejects_invalid_origin_without_epoch_fallback(origin):
    indexer = PeriodIndexer(device=torch.device("cpu"))
    with pytest.raises(ValidationError):
        indexer.elapsed_month_index(0.0, calendar="proleptic_gregorian", time_origin=origin)


@pytest.mark.parametrize(
    ("calendar", "origin"),
    [("gregorian", "1582-10-10"), ("standard", "1582-10-10"),
     ("noleap", "2020-02-29"), ("360_day", "2020-01-31")],
)
def test_elapsed_month_index_rejects_origin_outside_its_calendar(calendar, origin):
    indexer = PeriodIndexer(device=torch.device("cpu"))
    with pytest.raises(ValidationError):
        indexer.elapsed_month_index(0.0, calendar=calendar, time_origin=origin)


@pytest.mark.parametrize("elapsed", [-1.0, float("nan"), float("inf"), torch.tensor([0.0, 1.0])])
def test_elapsed_month_index_keeps_elapsed_validation_with_real_origin(elapsed):
    indexer = PeriodIndexer(device=torch.device("cpu"))
    with pytest.raises(ValidationError, match="elapsed_seconds"):
        indexer.elapsed_month_index(elapsed, calendar="gregorian", time_origin="2020-01-01")


def test_elapsed_month_index_delegates_real_origin_to_existing_decoder(monkeypatch):
    """Scheduling must call the one period decoder, without a relative-month owner."""
    indexer = PeriodIndexer(device=torch.device("cpu"))
    calls = []

    def decode(value, units, calendar):
        calls.append((value, units, calendar))
        return (2020, 5, 1, 0, 0, 0.0)

    monkeypatch.setattr(indexer, "_units_value_to_datetime", decode)
    assert indexer.elapsed_month_index(1800.0, "standard", time_origin="2020-05-01") == 12 * 2019 + 4
    assert calls == [(1800.0, "seconds since 2020-05-01", "gregorian")]


@pytest.mark.parametrize(
    ("frequency", "calendar", "values", "expected_bounds"),
    [
        ("M", "gregorian", [15.5, 45.5], [[0.0, 31.0], [31.0, 60.0]]),
        ("M", "noleap", [15.5, 45.0], [[0.0, 31.0], [31.0, 59.0]]),
        ("M", "360_day", [15.0, 45.0], [[0.0, 30.0], [30.0, 60.0]]),
        ("Y", "gregorian", [183.0], [[0.0, 366.0]]),
        ("Y", "noleap", [182.5], [[0.0, 365.0]]),
        ("Y", "360_day", [180.0], [[0.0, 360.0]]),
    ],
)
def test_cf_time_processor_builds_exact_calendar_period_bounds(
    frequency,
    calendar,
    values,
    expected_bounds,
):
    """Monthly and yearly bounds must follow the selected calendar."""
    processor = CFTimeProcessor(device=torch.device("cpu"))
    coordinate = TimeCoordinate(
        values=torch.tensor(values, dtype=torch.float64),
        units="days since 2000-01-01",
        calendar=calendar,
    )

    actual = processor.create_time_bounds(coordinate, frequency)

    expected = torch.tensor(expected_bounds, dtype=torch.float64)
    assert torch.allclose(actual.cpu(), expected, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("frequency", "calendar", "expected_bounds"),
    [
        ("QS-MAR", "noleap", [[59.0, 151.0]]),
        ("qs-mar", "360_day", [[60.0, 150.0]]),
        ("SEASON-MAM", "noleap", [[59.0, 151.0]]),
        ("season-mam", "360_day", [[60.0, 150.0]]),
    ],
)
def test_period_indexer_builds_calendar_aware_anchored_mam_bounds(
    frequency,
    calendar,
    expected_bounds,
):
    """MAM is 92 noleap days but 90 days in a 360-day calendar."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    coordinate = indexer.create_period_index(
        start_time="2001-03-01",
        end_time="2001-06-01",
        frequency=frequency,
        calendar=calendar,
        units="days since 2001-01-01",
    )

    expected = torch.tensor(expected_bounds, dtype=torch.float64)
    torch.testing.assert_close(coordinate.bounds.cpu(), expected, rtol=0.0, atol=0.0)


def test_period_indexer_builds_djf_across_year_boundary():
    """A December-anchored quarter must cross the year without fixed-day logic."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    coordinate = indexer.create_period_index(
        start_time="2000-12-01",
        end_time="2001-03-01",
        frequency="SEASON-DJF",
        calendar="noleap",
        units="days since 2000-12-01",
    )

    expected = torch.tensor([[0.0, 90.0]], dtype=torch.float64)
    torch.testing.assert_close(coordinate.bounds.cpu(), expected, rtol=0.0, atol=0.0)


def test_period_indexer_clips_partial_domain_to_real_quarter_anchors():
    """A partial domain must clip Jan 15-Apr 1, not drift to Apr 15."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    coordinate = indexer.create_period_index(
        start_time="2001-01-15",
        end_time="2001-05-01",
        frequency="QS-JAN",
        calendar="noleap",
        units="days since 2001-01-01",
    )

    expected = torch.tensor(
        [[14.0, 90.0], [90.0, 120.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(coordinate.bounds.cpu(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("frequency", ["Q", "quarterly", "S", "seasonal", "QS-FOO"])
def test_period_indexer_rejects_unanchored_or_unknown_quarter_tokens(frequency):
    """Quarter and season periods require an explicit controlled anchor."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    with pytest.raises(ValidationError, match="anchor|Unsupported"):
        indexer.create_period_index(
            start_time="2001-01-01",
            end_time="2001-04-01",
            frequency=frequency,
            calendar="noleap",
            units="days since 2001-01-01",
        )


@pytest.mark.parametrize(
    ("calendar", "expected_bounds"),
    [
        ("noleap", [[59.0, 151.0]]),
        ("360_day", [[60.0, 150.0]]),
    ],
)
def test_cf_time_processor_delegates_anchored_quarter_bounds_to_period_indexer(
    calendar,
    expected_bounds,
):
    """CFTimeProcessor must reuse PeriodIndexer instead of owning season math."""
    processor = CFTimeProcessor(device=torch.device("cpu"))
    coordinate = TimeCoordinate(
        values=torch.tensor([105.0], dtype=torch.float64),
        units="days since 2001-01-01",
        calendar=calendar,
    )

    actual = processor.create_time_bounds(coordinate, "QS-MAR")

    expected = torch.tensor(expected_bounds, dtype=torch.float64)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)


def test_align_periods_uses_same_quarter_anchor_owner():
    """Alignment and period creation must share one quarter normalization owner."""
    indexer = PeriodIndexer(device=torch.device("cpu"))
    source = TimeCoordinate(
        values=torch.tensor([67.0], dtype=torch.float64),
        bounds=torch.tensor([[14.0, 120.0]], dtype=torch.float64),
        units="days since 2001-01-01",
        calendar="noleap",
    )

    aligned = indexer.align_periods(source, "QS-JAN")

    expected = torch.tensor(
        [[14.0, 90.0], [90.0, 120.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(aligned.bounds.cpu(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "frequency",
    ["Q", "S", "quarterly", "seasonal", "not-a-frequency"],
)
def test_irregular_bounds_mode_cannot_bypass_frequency_validation(frequency):
    """Midpoint inference must not make an invalid or unanchored token executable."""
    processor = CFTimeProcessor(device=torch.device("cpu"))
    coordinate = TimeCoordinate(
        values=torch.tensor([0.0, 1.0], dtype=torch.float64),
        units="days since 2001-01-01",
        calendar="noleap",
    )

    with pytest.raises(ValidationError, match="anchor|Unsupported"):
        processor.create_time_bounds(
            coordinate,
            frequency,
            irregular_intervals=True,
        )


def test_calendar_dependent_frequency_rejects_irregular_midpoint_inference():
    """An anchored quarter must use calendar bounds, never midpoint inference."""
    processor = CFTimeProcessor(device=torch.device("cpu"))
    coordinate = TimeCoordinate(
        values=torch.tensor([15.0, 45.0], dtype=torch.float64),
        units="days since 2001-01-01",
        calendar="noleap",
    )

    with pytest.raises(ValidationError, match="calendar-dependent"):
        processor.create_time_bounds(
            coordinate,
            "QS-JAN",
            irregular_intervals=True,
        )


def test_epoch_partial_quarter_clips_before_first_representable_anchor():
    """Year-one data must clip the theoretical year-zero quarter at the epoch."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    created = indexer.create_period_index(
        start_time="0001-01-15",
        end_time="0001-05-01",
        frequency="QS-FEB",
        calendar="noleap",
        units="days since 0001-01-01",
    )
    source = TimeCoordinate(
        values=torch.tensor([67.0], dtype=torch.float64),
        bounds=torch.tensor([[14.0, 120.0]], dtype=torch.float64),
        units="days since 0001-01-01",
        calendar="noleap",
    )
    aligned = indexer.align_periods(source, "QS-FEB")

    expected = torch.tensor(
        [[14.0, 31.0], [31.0, 120.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(created.bounds.cpu(), expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(aligned.bounds.cpu(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("frequency", "first_boundary_day"),
    [
        ("QS-JAN", 90.0),
        ("QS-FEB", 31.0),
        ("QS-MAR", 59.0),
        ("QS-APR", 90.0),
        ("QS-MAY", 31.0),
        ("QS-JUN", 59.0),
        ("QS-JUL", 90.0),
        ("QS-AUG", 31.0),
        ("QS-SEP", 59.0),
        ("QS-OCT", 90.0),
        ("QS-NOV", 31.0),
        ("QS-DEC", 59.0),
        ("SEASON-DJF", 59.0),
    ],
)
def test_epoch_quarter_phase_uses_first_representable_congruent_anchor(
    frequency,
    first_boundary_day,
):
    """Epoch clipping must retain the anchor's three-month congruence class."""
    indexer = PeriodIndexer(device=torch.device("cpu"))

    coordinate = indexer.create_period_index(
        start_time="0001-01-15",
        end_time="0001-05-01",
        frequency=frequency,
        calendar="noleap",
        units="days since 0001-01-01",
    )

    expected = torch.tensor(
        [[14.0, first_boundary_day], [first_boundary_day, 120.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(coordinate.bounds.cpu(), expected, rtol=0.0, atol=0.0)
