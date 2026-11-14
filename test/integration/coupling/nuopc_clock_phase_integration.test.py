"""Integration test for clock negotiation and mediator phases."""

from relic.coupling.protocol import ClockConfig, InitPhase, Mediator, State, TimeStagger, TimeUnit
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog


def test_clock_negotiation_with_mediator_phase() -> None:
    stagger = TimeStagger()
    stagger.register_clock("fast", ClockConfig(time_step=60.0, unit=TimeUnit.SECONDS))
    stagger.register_clock("slow", ClockConfig(time_step=300.0, unit=TimeUnit.SECONDS))
    negotiated = stagger.negotiate(strategy="min")
    assert negotiated == 60.0
    assert stagger.validate_temporal_consistency() == []

    catalog = SSOTCatalog.from_parameter_manager()
    import_state = State(name="import_state", ssot_catalog=catalog, strict=True)
    export_state = State(name="export_state", ssot_catalog=catalog, strict=True)

    import_state.add_port_spec(
        PortSpec(
            name="air_temperature",
            standard_name="air_temperature",
            units="K",
            semantics="state",
        ),
        direction="import",
        data=300.0,
    )
    export_state.add_port_spec(
        PortSpec(
            name="surface_upward_sensible_heat_flux",
            standard_name="surface_upward_sensible_heat_flux",
            units="W m-2",
            semantics="flux",
        ),
        direction="export",
        data=25.0,
    )

    mediator = Mediator(import_state=import_state, export_state=export_state)
    mediator.advertise()
    mediator.realize()
    mediator.data_initialize({"air_temperature": 300.0})
    mediator.complete_initialization()
    assert mediator.phase == InitPhase.READY

    shifted_time = stagger.shift(0.0)
    assert shifted_time == 0.0
    export_payload = mediator.advance(dt=negotiated)
    assert export_payload["surface_upward_sensible_heat_flux"] == 25.0
