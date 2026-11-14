"""Unit tests for NUOPC-style mediator initialization phases."""

import pytest

from relic.coupling.protocol import Field, State
from relic.coupling.protocol.mediator import InitPhase, InitPhaseError, Mediator


def _build_mediator() -> Mediator:
    import_state = State("import_state")
    export_state = State("export_state")
    import_state.add_field("temp", Field("temp", standard_name="air_temperature"), "import")
    export_state.add_field(
        "precip", Field("precip", standard_name="precipitation_flux"), "export"
    )
    return Mediator(import_state, export_state)


def test_initial_phase_is_unconfigured() -> None:
    mediator = _build_mediator()
    assert mediator.phase == InitPhase.UNCONFIGURED


def test_advertise_transitions_to_advertised() -> None:
    mediator = _build_mediator()
    mediator.advertise()
    assert mediator.phase == InitPhase.ADVERTISED
    assert "temp" in mediator._advertised_fields
    assert "precip" in mediator._advertised_fields


def test_advertise_fails_if_not_unconfigured() -> None:
    mediator = _build_mediator()
    mediator.advertise()
    with pytest.raises(InitPhaseError):
        mediator.advertise()


def test_realize_requires_advertised() -> None:
    mediator = _build_mediator()
    with pytest.raises(InitPhaseError):
        mediator.realize()


def test_realize_validates_advertised_fields() -> None:
    mediator = _build_mediator()
    mediator.advertise(field_names={"temp", "missing"})
    with pytest.raises(InitPhaseError):
        mediator.realize(validate_all=True)


def test_realize_transitions_to_realized() -> None:
    mediator = _build_mediator()
    mediator.advertise()
    mediator.realize()
    assert mediator.phase == InitPhase.REALIZED


def test_data_initialize_transitions() -> None:
    mediator = _build_mediator()
    mediator.advertise()
    mediator.realize()
    mediator.data_initialize({"temp": 300.0})
    assert mediator.phase == InitPhase.DATA_INITIALIZED


def test_complete_initialization_reaches_ready() -> None:
    mediator = _build_mediator()
    mediator.advertise()
    mediator.realize()
    mediator.data_initialize()
    mediator.complete_initialization()
    assert mediator.phase == InitPhase.READY


def test_historical_initialize_completes_all_phases() -> None:
    mediator = _build_mediator()
    mediator.initialize()
    assert mediator.phase == InitPhase.READY
    assert mediator._initialized is True


def test_advance_requires_ready_phase() -> None:
    mediator = _build_mediator()
    with pytest.raises(RuntimeError):
        mediator.advance()


def test_finalize_resets_phase() -> None:
    mediator = _build_mediator()
    mediator.initialize()
    mediator.finalize()
    assert mediator.phase == InitPhase.UNCONFIGURED
    assert mediator._initialized is False

