"""Integration test for NUOPC-style mediator initialization sequence."""

from relic.coupling.protocol import InitPhase, Mediator, State
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog


def test_nuopc_init_sequence_roundtrip() -> None:
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
    mediator.data_initialize({"air_temperature": 305.0})
    mediator.complete_initialization()

    assert mediator.phase == InitPhase.READY

    export_payload = mediator.advance(dt=60.0)
    assert export_payload["surface_upward_sensible_heat_flux"] == 25.0
