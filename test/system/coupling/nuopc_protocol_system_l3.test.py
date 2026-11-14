from relic.coupling.protocol import Mediator, State, TimeStagger
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog


def test_nuopc_protocol_system_l3():
    catalog = SSOTCatalog.from_parameter_manager()
    import_state = State(name="import_state", ssot_catalog=catalog, strict=False)
    export_state = State(name="export_state", ssot_catalog=catalog, strict=False)

    import_state.add_port_spec(
        PortSpec(
            name="precipitation_flux",
            standard_name="precipitation_flux",
            units="kg m-2 s-1",
            semantics="flux",
        ),
        direction="import",
        data=0.001,
    )
    export_state.add_port_spec(
        PortSpec(
            name="soil_moisture",
            standard_name="soil_moisture",
            units="m3 m-3",
            semantics="state",
        ),
        direction="export",
        data=0.25,
    )

    stagger = TimeStagger(lag_seconds=0.0, lead_seconds=0.0)
    mediator = Mediator(import_state=import_state, export_state=export_state)
    mediator.initialize()
    output = mediator.advance(dt=300.0)
    mediator.finalize()

    assert stagger.shift(100.0) == 100.0
    assert output["soil_moisture"] == 0.25
    assert import_state.manifest()[0]["direction"] == "import"
