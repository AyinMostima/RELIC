from relic.coupling.protocol import Field, State
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog


def test_state_adds_field_and_exports():
    catalog = SSOTCatalog.from_parameter_manager()
    state = State(name="import_state", ssot_catalog=catalog, strict=True)
    port = PortSpec(
        name="air_temperature",
        standard_name="air_temperature",
        units="K",
        semantics="state",
    )
    field = Field.from_port_spec(port, data=300.0)
    state.add_field("air_temperature", field, direction="import")
    assert state.export_state() == {}
    state.import_state({"air_temperature": 305.0})
    assert field.data == 305.0


def test_state_manifest_includes_direction():
    catalog = SSOTCatalog.from_parameter_manager()
    state = State(name="export_state", ssot_catalog=catalog, strict=False)
    port = PortSpec(
        name="precipitation_flux",
        standard_name="precipitation_flux",
        units="kg m-2 s-1",
        semantics="flux",
    )
    state.add_port_spec(port, direction="export", data=1.0)
    manifest = state.manifest()
    assert manifest[0]["direction"] == "export"
