import pytest
import torch

from relic.coupling.scales.temporal import calendar as calendar_module
from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.coupling.scales.temporal.core import TimeWeightGenerator, ValidationError


def test_calendar_module_has_no_duplicate_or_placeholder_public_owners():
    """Calendar contracts must stay with CalendarHandler and TimeCoordinate."""
    obsolete_names = {
        "TimeInterval",
        "parse_time_bounds",
        "validate_calendar_type",
    }

    assert obsolete_names.isdisjoint(calendar_module.__all__)
    for name in obsolete_names:
        assert not hasattr(calendar_module, name)


def test_calendar_metadata_does_not_claim_dataset_cf_conformance():
    handler = object.__new__(CalendarHandler)
    handler.calendar_type = "noleap"
    handler.time_units = "days since 2000-01-01"
    handler.reference_date = "2000-01-01"
    handler.days_per_year = 365.0
    handler.months_per_year = 12
    handler._calendar_definitions = {
        "noleap": {
            "description": "365-day calendar",
            "leap_rule": "none",
        }
    }

    metadata = handler.get_calendar_metadata()

    assert "cf_compliant" not in metadata
    assert metadata["cf_time_coordinate_metadata_serialized"] is True
    assert metadata["dataset_conformance_checked"] is False


@pytest.mark.parametrize(
    ("calendar_type", "elapsed_days", "expected_day"),
    [
        ("noleap", 0, 1),
        ("noleap", 364, 365),
        ("noleap", 365, 1),
        ("360_day", 0, 1),
        ("360_day", 359, 360),
        ("360_day", 360, 1),
    ],
)
def test_day_of_year_respects_simulator_calendar_year_boundary(
    calendar_type,
    elapsed_days,
    expected_day,
):
    seconds_per_day = 86400.0

    assert CalendarHandler.day_of_year_from_elapsed_seconds(
        elapsed_days * seconds_per_day,
        calendar_type,
    ) == expected_day


def test_day_of_year_rejects_unsupported_simulator_calendar():
    with pytest.raises(ValueError, match="noleap or 360_day"):
        CalendarHandler.day_of_year_from_elapsed_seconds(0.0, "gregorian")


@pytest.mark.parametrize(
    "elapsed_seconds",
    [
        True,
        torch.tensor(True),
        torch.tensor([], dtype=torch.float64),
        torch.tensor([0.0, 86400.0], dtype=torch.float64),
        float("nan"),
        float("inf"),
        -1.0,
    ],
)
def test_day_of_year_rejects_non_scalar_or_invalid_elapsed_time(elapsed_seconds):
    with pytest.raises(ValueError, match="finite non-negative numeric scalar"):
        CalendarHandler.day_of_year_from_elapsed_seconds(
            elapsed_seconds,
            "noleap",
        )


@pytest.mark.parametrize(
    ("calendar_type", "day"),
    [
        ("noleap", True),
        ("noleap", torch.tensor([1, 2])),
        ("noleap", 1.5),
        ("noleap", float("nan")),
        ("noleap", 0),
        ("noleap", 366),
        ("360_day", 361),
    ],
)
def test_simulator_day_validation_rejects_non_integer_or_out_of_domain(
    calendar_type,
    day,
):
    with pytest.raises(ValueError, match="integer numeric scalar"):
        CalendarHandler.validate_simulator_day(day, calendar_type)


@pytest.mark.parametrize(
    ("calendar_type", "day"),
    [
        ("noleap", 1),
        ("noleap", torch.tensor(365)),
        ("noleap", torch.tensor([1.0])),
        ("360_day", 360.0),
    ],
)
def test_simulator_day_validation_accepts_exact_in_domain_scalar(
    calendar_type,
    day,
):
    expected = (
        int(day.reshape(()).item())
        if isinstance(day, torch.Tensor)
        else int(day)
    )
    assert CalendarHandler.validate_simulator_day(day, calendar_type) == expected


def test_simulator_year_length_comes_from_calendar_definitions():
    assert CalendarHandler.days_per_simulator_year("noleap") == 365
    assert CalendarHandler.days_per_simulator_year("360_day") == 360


def test_simulator_calendar_resolution_is_explicit_and_conflict_safe():
    assert CalendarHandler.resolve_simulator_calendar(
        configured_calendar=None,
        runtime_calendar="360_day",
        owner_name="test_owner",
    ) == "360_day"
    assert CalendarHandler.resolve_simulator_calendar(
        configured_calendar="noleap",
        runtime_calendar=None,
        owner_name="test_owner",
    ) == "noleap"

    with pytest.raises(ValueError, match="requires explicit calendar_type"):
        CalendarHandler.resolve_simulator_calendar(
            configured_calendar=None,
            runtime_calendar=None,
            owner_name="test_owner",
        )
    with pytest.raises(ValueError, match="conflicts with the configured"):
        CalendarHandler.resolve_simulator_calendar(
            configured_calendar="noleap",
            runtime_calendar="360_day",
            owner_name="test_owner",
        )
    with pytest.raises(ValueError, match="noleap or 360_day"):
        CalendarHandler.resolve_simulator_calendar(
            configured_calendar="gregorian",
            runtime_calendar=None,
            owner_name="test_owner",
        )


def test_calendar_handler_rejects_malformed_cf_time_units():
    """Calendar construction must not hide malformed units behind a fallback date."""
    with pytest.raises(ValidationError, match="Invalid CF units string"):
        CalendarHandler(
            calendar_type="noleap",
            time_units="not a CF time unit",
        )


def test_calendar_handler_calendar_query_keeps_boolean_contract():
    """The adapter query must not leak the canonical owner's exception type."""
    handler = CalendarHandler(calendar_type="noleap")

    assert handler.validate_calendar_type("365_day") is True
    assert handler.validate_calendar_type("not_a_calendar") is False


def test_calendar_handler_delegates_mixed_gregorian_arithmetic():
    """The adapter must preserve the canonical mixed-calendar leap rule."""
    mixed = CalendarHandler(calendar_type="gregorian")
    proleptic = CalendarHandler(calendar_type="proleptic_gregorian")

    assert mixed.is_leap_year(1500) is True
    assert mixed.days_in_month(2, 1500) == 29
    assert mixed.days_in_year(1500) == 366
    assert proleptic.is_leap_year(1500) is False
    assert proleptic.days_in_month(2, 1500) == 28


@pytest.mark.parametrize(
    ("calendar_type", "frequency", "coordinates", "expected"),
    [
        (
            "noleap",
            "monthly",
            [14.0, 45.0],
            [[0.0, 31.0], [31.0, 59.0]],
        ),
        (
            "360_day",
            "yearly",
            [100.0],
            [[0.0, 360.0]],
        ),
    ],
)
def test_calendar_handler_delegates_exact_calendar_bounds(
    calendar_type,
    frequency,
    coordinates,
    expected,
):
    """Monthly and yearly bounds must come from the canonical calendar owner."""
    handler = CalendarHandler(
        calendar_type=calendar_type,
        time_units="days since 2001-01-01",
    )
    time_coords = torch.tensor(coordinates, dtype=torch.float64)

    actual = handler.create_time_bounds(
        time_coords,
        frequency=frequency,
        method="centered",
    )

    assert torch.equal(actual.cpu(), torch.tensor(expected, dtype=torch.float64))


def test_calendar_handler_rejects_implicit_bound_inference():
    """Unknown frequency or alignment must fail instead of inventing a one-day bound."""
    handler = CalendarHandler(
        calendar_type="noleap",
        time_units="days since 2001-01-01",
    )
    coordinate = torch.tensor([0.0], dtype=torch.float64)

    with pytest.raises(ValidationError, match="Unsupported frequency"):
        handler.create_time_bounds(coordinate, frequency="unknown")
    with pytest.raises(ValueError, match="method must be centered"):
        handler.create_time_bounds(coordinate, frequency="daily", method="start")


def test_calendar_handler_uses_time_weight_generator():
    """CalendarHandler should delegate weight computation to TimeWeightGenerator."""
    handler = CalendarHandler(calendar_type="gregorian")
    assert isinstance(handler._weight_generator, TimeWeightGenerator)  # pylint: disable=protected-access

    source_time = torch.tensor([0.0, 1.0], dtype=torch.float64)
    source_bounds = torch.tensor([[0.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    target_time = torch.tensor([0.0], dtype=torch.float64)
    target_bounds = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

    weights, coverage = handler.compute_time_weights(
        source_time=source_time,
        source_bounds=source_bounds,
        target_time=target_time,
        target_bounds=target_bounds,
    )

    # Coverage should be full since bounds fully overlap
    assert torch.allclose(coverage, torch.tensor([1.0], dtype=torch.float64, device=coverage.device))
    # Total weight mass should be conserved
    assert torch.allclose(
        torch.sum(weights),
        torch.tensor(1.0, dtype=torch.float64, device=weights.device),
    )


@pytest.mark.parametrize(
    ("calendar_type", "frequency", "coordinate", "expected"),
    [
        ("noleap", "QS-MAR", [105.0], [[59.0, 151.0]]),
        ("360_day", "qs-mar", [105.0], [[60.0, 150.0]]),
        ("noleap", "SEASON-MAM", [105.0], [[59.0, 151.0]]),
        ("360_day", "season-mam", [105.0], [[60.0, 150.0]]),
    ],
)
def test_calendar_handler_delegates_anchored_period_aliases(
    calendar_type,
    frequency,
    coordinate,
    expected,
):
    """Public anchored aliases must retain PeriodIndexer calendar arithmetic."""
    handler = CalendarHandler(
        calendar_type=calendar_type,
        time_units="days since 2001-01-01",
    )

    actual = handler.create_time_bounds(
        torch.tensor(coordinate, dtype=torch.float64),
        frequency=frequency,
    )

    torch.testing.assert_close(
        actual.cpu(),
        torch.tensor(expected, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("frequency", ["Q", "quarterly", "S", "seasonal"])
def test_calendar_handler_rejects_unanchored_period_aliases(frequency):
    """The public adapter must preserve the canonical explicit-anchor gate."""
    handler = CalendarHandler(
        calendar_type="noleap",
        time_units="days since 2001-01-01",
    )

    with pytest.raises(ValidationError, match="anchor"):
        handler.create_time_bounds(
            torch.tensor([45.0], dtype=torch.float64),
            frequency=frequency,
        )


@pytest.mark.parametrize(
    ("frequency", "coordinate", "expected"),
    [
        ("weekly", [3.5], [[0.0, 7.0]]),
        ("annual", [182.5], [[0.0, 365.0]]),
        ("M", [15.5], [[0.0, 31.0]]),
    ],
)
def test_calendar_handler_preserves_public_and_canonical_frequency_aliases(
    frequency,
    coordinate,
    expected,
):
    """Simple aliases may normalize, but all arithmetic remains in core owners."""
    handler = CalendarHandler(
        calendar_type="noleap",
        time_units="days since 2001-01-01",
    )

    actual = handler.create_time_bounds(
        torch.tensor(coordinate, dtype=torch.float64),
        frequency=frequency,
    )

    torch.testing.assert_close(
        actual.cpu(),
        torch.tensor(expected, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_calendar_handler_is_a_thin_period_bounds_adapter(monkeypatch):
    """The wrapper must forward the controlled token without local date math."""
    handler = CalendarHandler(
        calendar_type="noleap",
        time_units="days since 2001-01-01",
        device=torch.device("cpu"),
    )
    captured = {}
    expected = torch.tensor([[10.0, 20.0]], dtype=torch.float64)

    def fake_create_time_bounds(time_coord, frequency, irregular_intervals=False):
        captured["time_coord"] = time_coord
        captured["frequency"] = frequency
        captured["irregular_intervals"] = irregular_intervals
        return expected

    monkeypatch.setattr(
        handler._cf_processor,
        "create_time_bounds",
        fake_create_time_bounds,
    )

    actual = handler.create_time_bounds(
        torch.tensor([15.0], dtype=torch.float64),
        frequency="season-mam",
    )

    assert captured["frequency"] == "season-mam"
    assert captured["irregular_intervals"] is False
    assert captured["time_coord"].calendar == "noleap"
    assert actual is expected


@pytest.mark.parametrize(
    "time_coords",
    [
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([], dtype=torch.float64),
        torch.tensor([float("nan")], dtype=torch.float64),
        torch.tensor([float("inf")], dtype=torch.float64),
        torch.sparse_coo_tensor(
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([1.0], dtype=torch.float64),
            size=(1,),
            check_invariants=False,
        ),
        torch.empty(1, device="meta"),
    ],
)
def test_calendar_handler_rejects_invalid_public_time_coordinate_tensors(time_coords):
    """The adapter must reject malformed tensors before device migration or math."""
    handler = CalendarHandler(
        calendar_type="noleap",
        time_units="days since 2001-01-01",
        device=torch.device("cpu"),
    )

    with pytest.raises(
        ValidationError,
        match="materialized|one-dimensional|non-empty|float32|finite",
    ):
        handler.create_time_bounds(time_coords, frequency="daily")


def test_calendar_handler_rejects_non_tensor_time_coordinates():
    """A Python sequence is not an implicit temporal tensor conversion contract."""
    handler = CalendarHandler(
        calendar_type="noleap",
        time_units="days since 2001-01-01",
        device=torch.device("cpu"),
    )

    with pytest.raises(TypeError, match="torch.Tensor"):
        handler.create_time_bounds([1.0], frequency="daily")


def test_calendar_handler_keeps_leap_metadata_with_calendar_definition_owner():
    """The adapter must not own leap lists or a fixed days-per-month field."""
    noleap = CalendarHandler(calendar_type="noleap")
    all_leap = CalendarHandler(calendar_type="all_leap")
    day_360 = CalendarHandler(calendar_type="360_day")

    assert not hasattr(noleap, "supports_leap_years")
    assert not hasattr(noleap, "days_per_month")
    assert noleap.get_calendar_metadata()["leap_year_support"] is False
    assert day_360.get_calendar_metadata()["leap_year_support"] is False
    assert all_leap.get_calendar_metadata()["leap_year_support"] is True
    assert all_leap.get_calendar_metadata()["leap_rule"] == "all"
