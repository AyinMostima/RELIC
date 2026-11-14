"""Tests for binding DEC form metadata through existing SSOT and PortSpec.

Graft static edges do not cover ParameterManager keys or metadata-driven port
construction; dynamic catalog-to-NodeRegistry assertions are authoritative.
"""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from relic.parameters import get_global_manager
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import PORT_SUPPORT_FIELDS, PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog, SSOTEntry
from test.fixtures.dummy_functor import DummyIncrementFunctor


ROOT = Path(__file__).resolve().parents[3]


class _OptionalInputFunctor:
    def __init__(self, required_inputs=None):
        self.metadata = SimpleNamespace(
            input_variables=["air_temperature", "water_stress_factor"],
            output_variables=["net_photosynthesis"],
            to_dict=lambda: {"name": "optional_input_functor"},
        )
        self._required_inputs = (
            ["air_temperature"] if required_inputs is None else required_inputs
        )

    def get_required_inputs(self):
        return self._required_inputs


def test_portspec_serializes_support_metadata():
    port = PortSpec(
        name="isoprene_emission",
        standard_name="isoprene_emission",
        units="kg m-2 s-1",
        spatial_kind="grid_cell",
        area_basis="ground_area",
        temporal_basis="interval_mean",
        vertical_basis="layer_integral",
    )

    payload = port.to_dict()

    assert payload["spatial_kind"] == "grid_cell"
    assert payload["area_basis"] == "ground_area"
    assert payload["temporal_basis"] == "interval_mean"
    assert payload["vertical_basis"] == "layer_integral"
    assert payload["declared_support_fields"] == sorted(
        PORT_SUPPORT_FIELDS[1:]
    )


def test_portspec_distinguishes_explicit_null_from_undeclared_support():
    undeclared = PortSpec(
        name="undeclared",
        standard_name="undeclared",
    )
    explicit_null = PortSpec(
        name="explicit_null",
        standard_name="explicit_null",
        declared_support_fields=frozenset(PORT_SUPPORT_FIELDS),
    )

    assert undeclared.missing_support_declarations() == PORT_SUPPORT_FIELDS
    assert explicit_null.missing_support_declarations() == ()
    assert all(
        explicit_null.is_support_declared(field_name)
        for field_name in PORT_SUPPORT_FIELDS
    )
    assert explicit_null.to_dict()["declared_support_fields"] == sorted(
        PORT_SUPPORT_FIELDS
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("spatial_kind", "point"),
        ("area_basis", "projected_area"),
        ("temporal_basis", "mean_over_timestep"),
        ("vertical_basis", "canopy_column"),
    ],
)
def test_portspec_rejects_support_values_outside_frozen_vocabulary(
    field_name,
    invalid_value,
):
    kwargs = {
        "name": "test_port",
        "standard_name": "test_port",
        field_name: invalid_value,
    }

    with pytest.raises(ValueError, match=field_name):
        PortSpec(**kwargs)


def test_node_registry_populates_port_form_metadata_from_ssot_raw():
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="soil_water_storage",
            standard_name="soil_water_storage",
            units="kg m-2",
            shape={"dimensions": ["cell"]},
            raw={
                "form_degree": 2,
                "primality": "primal",
                "extensivity": "extensive",
                "complex_id": "soil_column",
                "physical_domain": "hydrosphere",
                "orientation_convention": "outward_positive",
                "boundary_role": "interior",
                "support": {
                    "spatial_kind": "grid_cell",
                    "area_basis": "ground_area",
                    "temporal_basis": "instantaneous",
                    "vertical_basis": "layer_integral",
                },
            },
        )
    )
    registry = NodeRegistry(catalog)

    definition = registry.build_from_node_spec(
        "Functor:SoilWaterTransport",
        input_names=[],
        output_names=["soil_water_storage"],
    )

    port = definition.outputs["soil_water_storage"]
    assert port.form_degree == 2
    assert port.primality == "primal"
    assert port.extensivity == "extensive"
    assert port.complex_id == "soil_column"
    assert port.physical_domain == "hydrosphere"
    assert port.orientation_convention == "outward_positive"
    assert port.boundary_role == "interior"
    assert port.shape == {"dimensions": ["cell"]}
    assert port.spatial_kind == "grid_cell"
    assert port.area_basis == "ground_area"
    assert port.temporal_basis == "instantaneous"
    assert port.vertical_basis == "layer_integral"
    assert port.to_dict()["form_degree"] == 2
    assert port.to_dict()["complex_id"] == "soil_column"
    assert port.to_dict()["spatial_kind"] == "grid_cell"


def test_node_registry_preserves_explicit_null_support_declarations():
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="scalar_diagnostic",
            standard_name="scalar_diagnostic",
            units="1",
            shape={"dimensions": []},
            raw={
                "support": {
                    "spatial_kind": None,
                    "area_basis": None,
                    "temporal_basis": None,
                    "vertical_basis": None,
                }
            },
        )
    )
    registry = NodeRegistry(catalog)

    port = registry.build_from_node_spec(
        "Functor:ScalarDiagnostic",
        input_names=[],
        output_names=["scalar_diagnostic"],
    ).outputs["scalar_diagnostic"]

    assert port.missing_support_declarations() == ()
    assert port.shape == {"dimensions": []}
    assert port.spatial_kind is None
    assert port.area_basis is None
    assert port.temporal_basis is None
    assert port.vertical_basis is None


def test_node_registry_does_not_apply_state_write_policy_to_input_ports():
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="air_temperature",
            standard_name="air_temperature",
            units="K",
            semantics="state",
            update_mode="replace",
            write_mode="immutable",
        )
    )
    registry = NodeRegistry(catalog)

    definition = registry.build_from_node_spec(
        "Functor:TemperatureReaderWriter",
        input_names=["air_temperature"],
        output_names=["air_temperature"],
    )

    input_port = definition.inputs["air_temperature"]
    output_port = definition.outputs["air_temperature"]
    assert input_port.direction == "in"
    assert input_port.write_mode is None
    assert input_port.update_mode == "replace"
    assert output_port.direction == "out"
    assert output_port.write_mode == "immutable"


def test_registered_functor_alias_resolves_the_real_runtime_class():
    registry = NodeRegistry(SSOTCatalog())
    alias = "Functor:adapter.test.increment"

    definition = registry.register_functor_instance(
        DummyIncrementFunctor(),
        node_type_id=alias,
    )

    assert definition.metadata["runtime_class"] == (
        "test.fixtures.dummy_functor.DummyIncrementFunctor"
    )
    assert registry.resolve_class_type(alias) is DummyIncrementFunctor


def test_registered_functor_uses_runtime_required_input_contract():
    registry = NodeRegistry(SSOTCatalog())

    definition = registry.register_functor_instance(_OptionalInputFunctor())

    assert definition.inputs["air_temperature"].required is True
    assert definition.inputs["water_stress_factor"].required is False


def test_registered_functor_rejects_required_input_missing_from_metadata():
    registry = NodeRegistry(SSOTCatalog())

    with pytest.raises(
        ValueError,
        match="required input.*canopy_temperature.*not declared",
    ):
        registry.register_functor_instance(
            _OptionalInputFunctor(required_inputs=["canopy_temperature"])
        )


def test_parameter_backed_ssot_catalog_merges_canonical_variable_contracts():
    catalog = SSOTCatalog.from_parameter_manager()

    expected_units = {
        "ppfd": "umol photons m-2 s-1",
        "electron_transport_rate": "umol e- m-2 s-1",
        "isoprene_emission": "mol m-2 s-1",
        "bvoc_carbon_allocation_flux": "kg C m-2 s-1",
    }
    for variable, units in expected_units.items():
        entry = catalog.get(variable)
        assert entry is not None
        assert entry.units == units
        assert entry.raw["variable"] == variable


def test_parameter_catalog_sources_flow_into_node_registry_ports():
    """Both ParameterManager variable catalogs must drive real PortSpecs."""
    manager = get_global_manager()
    external_variables = manager.get(
        "coupling.external.variables.variable_definitions",
        expected_type=dict,
        required=True,
    )
    scale_variables = manager.get(
        "coupling.scales.defaults.variables",
        expected_type=dict,
        required=True,
    )
    catalog = SSOTCatalog.from_parameter_manager(manager)
    registry = NodeRegistry(catalog)
    definition = registry.build_from_node_spec(
        "Q3:DualParameterCatalogPorts",
        input_names=["eastward_wind"],
        output_names=["gross_photosynthesis"],
    )

    cases = (
        (
            "eastward_wind",
            external_variables["eastward_wind"],
            definition.inputs["eastward_wind"],
            "in",
        ),
        (
            "gross_photosynthesis",
            scale_variables["gross_photosynthesis"],
            definition.outputs["gross_photosynthesis"],
            "out",
        ),
    )
    for variable, source_payload, port, direction in cases:
        entry = catalog.get(variable)
        assert entry is not None
        assert entry.units == source_payload["units"]
        assert entry.semantics == source_payload["semantics"]
        assert port.standard_name == entry.standard_name
        assert port.units == entry.units
        assert port.semantics == entry.semantics
        assert port.direction == direction

    gross_port = definition.outputs["gross_photosynthesis"]
    assert gross_port.write_mode == "derived"
    assert gross_port.spatial_kind == "canopy"
    assert gross_port.area_basis == "ground_area"


def test_ssot_catalog_merges_partial_support_without_losing_contract_fields():
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="ppfd",
            standard_name="ppfd",
            units="umol photons m-2 s-1",
            raw={
                "support": {
                    "temporal_basis": "instantaneous",
                }
            },
        )
    )

    catalog._load_contract_entries(
        ROOT / "relic" / "contracts" / "variables"
    )

    entry = catalog.get("ppfd")
    assert entry is not None
    assert entry.raw["support"] == {
        "spatial_kind": "grid_cell",
        "area_basis": "ground_area",
        "temporal_basis": "instantaneous",
        "vertical_basis": "none",
        "shape": {"dimensions": ["cell"]},
    }


def test_scale_variable_units_match_selected_scm_contracts():
    path = (
        ROOT
        / "relic"
        / "parameters"
        / "database"
        / "relic"
        / "coupling"
        / "scale_variables.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    variables = payload["parameters"][
        "coupling.scales.defaults.variables"
    ]["value"]

    assert variables["gross_photosynthesis"]["units"] == (
        "umol CO2 m-2 s-1"
    )
    assert variables["net_photosynthesis"]["units"] == (
        "umol CO2 m-2 s-1"
    )
    assert variables["leaf_area_index"]["units"] == "m2 m-2"


def test_scale_catalogs_declare_no_retired_net_assimilation_alias():
    """net_assimilation was an alias entry for the molar net photosynthesis An.

    The SCM contract and the Farquhar port for An is net_photosynthesis; the bare
    net_assimilation contract is retired and no Functor reads or writes that name,
    so no scale catalog declares it and the SSOT catalog cannot resolve it.
    """
    path = (
        ROOT
        / "relic"
        / "parameters"
        / "database"
        / "relic"
        / "coupling"
        / "scale_variables.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    declaring = sorted(
        key
        for key, entry in payload["parameters"].items()
        if key.endswith(".variables")
        and isinstance(entry.get("value"), dict)
        and "net_assimilation" in entry["value"]
    )
    assert declaring == []

    catalog = SSOTCatalog.from_parameter_manager(get_global_manager())
    assert catalog.get("net_assimilation") is None
    assert catalog.get("net_photosynthesis") is not None


def test_external_lai_unit_matches_selected_scm_contract():
    path = (
        ROOT
        / "relic"
        / "parameters"
        / "database"
        / "relic"
        / "coupling"
        / "external_variables.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    variables = payload["parameters"][
        "coupling.external.variables.variable_definitions"
    ]["value"]

    assert variables["leaf_area_index"]["units"] == "m2 m-2"


def _contract(variable: str, cf_standard_name):
    return SimpleNamespace(
        variable=variable,
        kind="flux",
        udunits="kg m-2 s-1",
        cf_standard_name=cf_standard_name,
        aliases=[],
        raw={
            "variable": variable,
            "dimension": {
                "udunits": "kg m-2 s-1",
                "cf_standard_name": cf_standard_name,
            },
        },
    )


def test_contract_entry_preserves_missing_cf_as_none():
    entry = SSOTCatalog._entry_from_variable_contract(
        _contract("local_flux", None)
    )

    assert entry.standard_name == "local_flux"
    assert entry.cf_standard_name is None
    assert asdict(entry)["cf_standard_name"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"standard_name": "local_name"},
        {"standard_name": "local_name", "cf_standard_name": None},
    ],
)
def test_parameter_entry_preserves_missing_or_null_cf_as_none(payload):
    entry = SSOTCatalog()._build_entry("local_key", payload)

    assert entry is not None
    assert entry.standard_name == "local_name"
    assert entry.cf_standard_name is None


@pytest.mark.parametrize(
    "invalid_cf",
    ["", "   ", " air_temperature", "air_temperature ", True, 7, [], {}],
)
def test_ssot_builders_reject_invalid_cf_values(invalid_cf):
    with pytest.raises((TypeError, ValueError), match="cf_standard_name"):
        SSOTCatalog()._build_entry(
            "local_key",
            {
                "standard_name": "local_name",
                "cf_standard_name": invalid_cf,
            },
        )

    with pytest.raises((TypeError, ValueError), match="cf_standard_name"):
        SSOTCatalog._entry_from_variable_contract(
            _contract("local_flux", invalid_cf)
        )


def test_explicit_cf_name_is_preserved_without_rewrite():
    expected = "air_temperature"
    parameter_entry = SSOTCatalog()._build_entry(
        "temperature",
        {
            "standard_name": "local_temperature",
            "cf_standard_name": expected,
        },
    )
    contract_entry = SSOTCatalog._entry_from_variable_contract(
        _contract("temperature_flux", expected)
    )

    assert parameter_entry is not None
    assert parameter_entry.standard_name == "local_temperature"
    assert parameter_entry.cf_standard_name == expected
    assert contract_entry.cf_standard_name == expected


def test_parameter_backed_catalog_preserves_scm_null_cf_values():
    expected_none = (
        "anthropogenic_nox_flux",
        "anthropogenic_pm25_flux",
        "anthropogenic_voc_flux",
        "sector_emission_flux",
    )

    for catalog in (
        SSOTCatalog.from_parameter_manager(),
        SSOTCatalog.from_parameter_manager(),
    ):
        for variable in expected_none:
            entry = catalog.get(variable)
            assert entry is not None
            assert entry.standard_name == variable
            assert entry.cf_standard_name is None

        co_entry = catalog.get("anthropogenic_co_flux")
        assert co_entry is not None
        assert co_entry.cf_standard_name == (
            "tendency_of_atmosphere_mass_content_of_carbon_monoxide_due_to_emission"
        )


def test_identical_explicit_cf_sources_merge_stably():
    expected = (
        "tendency_of_atmosphere_mass_content_of_carbon_monoxide_due_to_emission"
    )
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="anthropogenic_co_flux",
            standard_name="anthropogenic_co_flux",
            units="kg m-2 s-1",
            cf_standard_name=expected,
        )
    )

    catalog._load_contract_entries(
        ROOT / "relic" / "contracts" / "variables"
    )

    assert catalog.get("anthropogenic_co_flux").cf_standard_name == expected


def test_different_explicit_cf_sources_fail_closed():
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="anthropogenic_co_flux",
            standard_name="anthropogenic_co_flux",
            units="kg m-2 s-1",
            cf_standard_name="air_temperature",
        )
    )

    with pytest.raises(
        ValueError,
        match="anthropogenic_co_flux.*air_temperature.*carbon_monoxide",
    ):
        catalog._load_contract_entries(
            ROOT / "relic" / "contracts" / "variables"
        )


def test_contract_null_conflicts_with_existing_nonnull_cf():
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="sector_emission_flux",
            standard_name="sector_emission_flux",
            units="kg m-2 s-1",
            cf_standard_name="air_temperature",
        )
    )

    with pytest.raises(
        ValueError,
        match="sector_emission_flux.*air_temperature.*None",
    ):
        catalog._load_contract_entries(
            ROOT / "relic" / "contracts" / "variables"
        )


def test_node_registry_preserves_manual_ssot_cf_none_for_both_directions():
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="local_flux",
            standard_name="local_flux",
            units="kg m-2 s-1",
            cf_standard_name=None,
            semantics="flux",
            shape={"dimensions": ["cell"]},
            raw={
                "support": {
                    "spatial_kind": "grid_cell",
                    "area_basis": "ground_area",
                    "temporal_basis": "interval_mean",
                    "vertical_basis": "none",
                }
            },
        )
    )
    registry = NodeRegistry(catalog)

    definition = registry.build_from_node_spec(
        "Functor:LocalFlux",
        input_names=["local_flux"],
        output_names=["local_flux"],
    )

    for port, direction in (
        (definition.inputs["local_flux"], "in"),
        (definition.outputs["local_flux"], "out"),
    ):
        assert port.standard_name == "local_flux"
        assert port.cf_standard_name is None
        assert port.units == "kg m-2 s-1"
        assert port.semantics == "flux"
        assert port.shape == {"dimensions": ["cell"]}
        assert port.spatial_kind == "grid_cell"
        assert port.area_basis == "ground_area"
        assert port.temporal_basis == "interval_mean"
        assert port.vertical_basis == "none"
        assert port.direction == direction
        assert port.to_dict()["cf_standard_name"] is None


def test_node_registry_unknown_ports_do_not_invent_cf_names():
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.build_from_node_spec(
        "Functor:UnknownPorts",
        input_names=["unknown_input"],
        output_names=["unknown_output"],
    )

    input_port = definition.inputs["unknown_input"]
    output_port = definition.outputs["unknown_output"]
    assert input_port.standard_name == "unknown_input"
    assert input_port.cf_standard_name is None
    assert input_port.to_dict()["cf_standard_name"] is None
    assert output_port.standard_name == "unknown_output"
    assert output_port.cf_standard_name is None
    assert output_port.to_dict()["cf_standard_name"] is None


def test_node_registry_preserves_catalog_cf_none_and_explicit_cf():
    catalog = SSOTCatalog.from_parameter_manager()
    registry = NodeRegistry(catalog)
    variables_without_cf = (
        "anthropogenic_nox_flux",
        "anthropogenic_pm25_flux",
        "anthropogenic_voc_flux",
        "sector_emission_flux",
    )

    definition = registry.build_from_node_spec(
        "Functor:EmissionPorts",
        input_names=list(variables_without_cf),
        output_names=["anthropogenic_co_flux"],
    )

    for variable in variables_without_cf:
        port = definition.inputs[variable]
        assert port.standard_name == variable
        assert port.cf_standard_name is None
        assert port.units == "kg m-2 s-1"
        assert port.to_dict()["cf_standard_name"] is None

    co_port = definition.outputs["anthropogenic_co_flux"]
    assert co_port.standard_name == "anthropogenic_co_flux"
    assert co_port.cf_standard_name == (
        "tendency_of_atmosphere_mass_content_of_carbon_monoxide_due_to_emission"
    )


@pytest.mark.parametrize(
    "invalid_cf",
    ["", "   ", " air_temperature", "air_temperature ", True, 7, [], {}],
)
def test_node_registry_rejects_malformed_manual_cf_metadata(invalid_cf):
    catalog = SSOTCatalog()
    catalog.add_entry(
        SSOTEntry(
            name="manual_entry",
            standard_name="manual_entry",
            units="1",
            cf_standard_name=invalid_cf,
        )
    )

    with pytest.raises((TypeError, ValueError), match="cf_standard_name"):
        NodeRegistry(catalog).build_from_node_spec(
            "Functor:ManualEntry",
            input_names=["manual_entry"],
            output_names=[],
        )


def test_node_registry_reuses_the_single_ssot_cf_invariant_owner():
    source = (
        ROOT / "relic" / "orchestration" / "node_registry.py"
    ).read_text(encoding="utf-8")

    assert "_validate_optional_cf_standard_name" not in NodeRegistry.__dict__
    assert "SSOTCatalog._normalize_optional_cf_standard_name" in source


def test_ssot_entry_carries_the_spatial_basis_not_a_variable_type_chain():
    from relic.coupling.scales.types import SpatialQuantityBasis

    fields = SSOTEntry.__dataclass_fields__
    assert "variable_type" not in fields
    assert fields["spatial_basis"].default is None

    source = (ROOT / "relic" / "orchestration" / "ssot_catalog.py").read_text(encoding="utf-8")
    assert 'value.get("dimension")' not in source
    assert "resolve_spatial_basis" in source

    catalog = SSOTCatalog()
    assert catalog._build_entry(
        "litter_mass", {"units": "kg", "spatial_basis": "cell_total"}
    ).spatial_basis is SpatialQuantityBasis.CELL_TOTAL
    assert catalog._build_entry(
        "relative_humidity", {"units": "1", "variable_type": "intensive"}
    ).spatial_basis is SpatialQuantityBasis.INTENSIVE
    # Neither spatial_basis nor variable_type: the unrelated dimension/category
    # keys are not read as a basis.
    assert catalog._build_entry(
        "runoff", {"units": "kg m-2 s-1", "dimension": "extensive", "category": "FLUX"}
    ).spatial_basis is None


@pytest.mark.parametrize(
    "value, message",
    [
        ({"units": "kg m-2", "variable_type": "extensive"}, "ambiguous"),
        ({"units": "kg m-2", "spatial_basis": "extensive"}, "ambiguous"),
        ({"units": "kg m-2", "spatial_basis": "per_cell"}, "Unknown spatial quantity basis"),
        ({"units": "K", "variable_type": "extensive", "spatial_basis": "intensive"}, "inconsistent"),
    ],
)
def test_ssot_entry_spatial_basis_goes_through_the_single_owner(value, message):
    with pytest.raises(ValueError, match=message):
        SSOTCatalog()._build_entry("probe_variable", value)


def test_parameter_catalog_resolves_declared_spatial_bases():
    from relic.coupling.scales.types import SpatialQuantityBasis

    catalog = SSOTCatalog.from_parameter_manager(get_global_manager())

    assert catalog.get("precipitation_flux").spatial_basis is SpatialQuantityBasis.AREA_DENSITY
    assert catalog.get("air_temperature").spatial_basis is SpatialQuantityBasis.INTENSIVE
    assert catalog.get("relative_humidity").spatial_basis is SpatialQuantityBasis.INTENSIVE
    undeclared = [
        entry
        for entry in (catalog.get(name) for name in catalog._entries)
        if "spatial_basis" not in entry.raw and "variable_type" not in entry.raw
    ]
    assert undeclared
    assert all(entry.spatial_basis is None for entry in undeclared)


def test_daily_air_temperature_extrema_share_the_cf_standard_name():
    """CF-1.8 7.3: tasmax/tasmin are air_temperature told apart by cell_methods."""
    catalog = SSOTCatalog.from_parameter_manager(get_global_manager())

    maximum = catalog.get("daily_maximum_air_temperature")
    minimum = catalog.get("daily_minimum_air_temperature")
    assert maximum is not None and minimum is not None
    assert maximum.standard_name == minimum.standard_name == "air_temperature"
    assert maximum.raw["cell_methods"] == "time: maximum"
    assert minimum.raw["cell_methods"] == "time: minimum"
    assert catalog.get("air_temperature").name == "air_temperature"
    assert not hasattr(catalog, "conflicts")

    # The qualifiers are part of the identity: no cell_methods selects the
    # unqualified entry; each extremum is selected by its own cell_methods. The
    # external names are contract aliases of the phenology forcings, which hold
    # the qualified identities.
    assert catalog.get_by_standard_name("air_temperature").name == "air_temperature"
    assert catalog.get_by_standard_name(
        "air_temperature", cell_methods="time:  maximum"
    ) is maximum
    assert maximum.name == "air_temperature_max"
    assert catalog.get_by_standard_name(
        "air_temperature", cell_methods="time: minimum"
    ) is minimum
    assert minimum.name == "air_temperature_min"
    assert catalog.get_by_standard_name("air_temperature", cell_methods="time: mean") is None
    assert catalog.get_by_standard_name("no_such_standard_name") is None


def _entry(name, standard_name, units="K", semantics="state", **raw):
    return SSOTEntry(
        name=name,
        standard_name=standard_name,
        units=units,
        semantics=semantics,
        raw={"standard_name": standard_name, **raw},
    )


def test_shared_standard_name_with_the_same_qualifiers_raises():
    catalog = SSOTCatalog()
    catalog.add_entry(_entry("tas", "air_temperature"))

    with pytest.raises(ValueError, match=r"'tas'.*'tas_celsius'"):
        catalog.add_entry(_entry("tas_celsius", "air_temperature", units="degC"))
    with pytest.raises(ValueError, match=r"'tas'.*'tas_flux'"):
        catalog.add_entry(_entry("tas_flux", "air_temperature", semantics="flux"))

    catalog.add_entry(_entry("tasmax", "air_temperature", cell_methods="time: maximum"))
    assert catalog.get("tasmax").name == "tasmax"


def test_same_variable_name_merges_agreeing_declarations_and_rejects_contradictions():
    catalog = SSOTCatalog()
    catalog.add_entry(_entry("net_radiation", "net_downward_radiative_flux_at_surface", units="W m-2"))
    implicit = SSOTEntry(
        name="net_radiation",
        standard_name="net_radiation",
        units="W m-2",
        semantics="state",
        quantity="energy_flux",
        raw={"description": "scale default"},
    )

    catalog.add_entry(implicit)

    merged = catalog.get("net_radiation")
    assert merged.standard_name == "net_downward_radiative_flux_at_surface"
    assert merged.quantity == "energy_flux"
    assert merged.raw["description"] == "scale default"
    with pytest.raises(ValueError, match=r"'net_radiation'.*units"):
        catalog.add_entry(_entry("net_radiation", "net_downward_radiative_flux_at_surface", units="K"))


def test_dimensionless_unit_spellings_are_one_unit_not_a_conflict():
    catalog = SSOTCatalog()
    catalog.add_entry(_entry("water_stress_factor", "water_stress", units="1"))

    catalog.add_entry(_entry("water_stress_factor", "water_stress", units="dimensionless"))

    assert catalog.get("water_stress_factor").units == "1"


def _cf_units_match(standard_name, units):
    from relic.data.standards.cf_conventions import CFConventions

    label = "1" if str(units).strip() == "dimensionless" else units
    return CFConventions.are_standard_name_units_equivalent(standard_name, label)


def test_catalog_standard_names_agree_with_the_offline_cf_table():
    """Every CF claim of the catalog is a CF table name with its canonical units."""
    from relic.data.standards.cf_conventions import CFConventions

    catalog = SSOTCatalog.from_parameter_manager(get_global_manager())
    problems = []
    for entry in catalog._entries.values():
        declared = "standard_name" in entry.raw
        declared_cf = declared and CFConventions.is_valid_standard_name(entry.standard_name)
        provenance = entry.raw.get("provenance")
        if declared and isinstance(provenance, dict) and provenance.get("source") == "cf_standard":
            if not declared_cf:
                problems.append(f"{entry.name}: {entry.standard_name!r} is not a CF standard name")
        claims = {name for name in (entry.cf_standard_name,) if name}
        if declared_cf:
            claims.add(entry.standard_name)
        if len(claims) > 1:
            problems.append(f"{entry.name}: standard_name and cf_standard_name differ {sorted(claims)}")
        for claim in claims:
            if entry.units and not _cf_units_match(claim, entry.units):
                problems.append(f"{entry.name}: {entry.units!r} is not the CF unit of {claim!r}")
    assert problems == []


def test_every_catalog_entry_is_the_unique_holder_of_its_standard_name_and_qualifiers():
    catalog = SSOTCatalog.from_parameter_manager(get_global_manager())

    for entry in catalog._entries.values():
        resolved = catalog.get_by_standard_name(
            entry.standard_name, cell_methods=entry.raw.get("cell_methods")
        )
        assert resolved is entry, entry.name


def test_catalog_data_fixes_for_pressure_runoff_radiation_and_leaf_temperature():
    catalog = SSOTCatalog.from_parameter_manager(get_global_manager())

    # Surface air pressure forcing: one canonical variable, the external name is its alias.
    pressure = catalog.get("atmospheric_pressure")
    assert "air_pressure" not in catalog._entries
    assert catalog.get("air_pressure") is pressure
    assert pressure.standard_name == pressure.cf_standard_name == "surface_air_pressure"
    assert pressure.units == "Pa"
    assert catalog.get("runoff_flux").standard_name == "runoff_flux"
    assert catalog.get("net_radiation").standard_name == "surface_net_downward_radiative_flux"
    leaf = catalog.get("leaf_temperature")
    assert leaf.standard_name == "leaf_temperature"
    assert leaf.cf_standard_name is None
    assert catalog.get_by_standard_name("temperature").name == "temperature"


def test_alias_conflicts_raise_with_both_names():
    catalog = SSOTCatalog()
    catalog.add_entry(_entry("rainfall_flux", "rainfall_flux", units="kg m-2 s-1", semantics="flux"))
    catalog.add_entry(_entry("snowfall_flux", "snowfall_flux", units="kg m-2 s-1", semantics="flux"))

    # An alias that names another existing variable.
    with pytest.raises(ValueError, match=r"'snowfall_flux'.*'rainfall_flux'"):
        catalog._load_aliases({"snowfall_flux": "rainfall_flux"})
    catalog._load_aliases({"rain": "rainfall_flux"})
    catalog._load_aliases({"rain": "rainfall_flux"})
    assert catalog.get("rain").name == "rainfall_flux"
    # One alias claimed by two variables.
    with pytest.raises(ValueError, match=r"'rain'.*'rainfall_flux'.*'snowfall_flux'"):
        catalog._load_aliases({"rain": "snowfall_flux"})
    with pytest.raises(ValueError, match="unknown variable 'hail_flux'"):
        catalog._load_aliases({"hail": "hail_flux"})
    with pytest.raises(ValueError, match="not a name pair"):
        catalog._load_aliases({"sleet": 3})
    # A new variable named like an existing alias.
    with pytest.raises(ValueError, match=r"'rain'.*alias of 'rainfall_flux'"):
        catalog.add_entry(_entry("rain", "rain", units="kg m-2 s-1", semantics="flux"))
    assert not hasattr(catalog, "conflicts")


def test_node_registry_does_not_swallow_variable_alias_refresh_errors(monkeypatch):
    """_build_port resolves names through the refreshed alias map; a failed refresh raises."""
    import relic.orchestration.node_registry as node_registry_module

    def failing_refresh(*_args, **_kwargs):
        raise ValueError("alias 'tas' maps to both 'air_temperature' and 'surface_temperature'")

    monkeypatch.setattr(
        node_registry_module.variable_name_mapper,
        "refresh_from_parameter_manager",
        failing_refresh,
    )

    with pytest.raises(ValueError, match="alias 'tas' maps to both"):
        NodeRegistry(SSOTCatalog())


def test_catalog_and_variable_mapper_share_one_identity_rule_owner():
    ssot_source = (ROOT / "relic" / "orchestration" / "ssot_catalog.py").read_text(encoding="utf-8")
    mapper_source = (ROOT / "relic" / "core" / "variable_mapper.py").read_text(encoding="utf-8")

    for source in (ssot_source, mapper_source):
        assert "cf_variable_identity" in source
        assert "register_cf_identity_holder" in source
        assert "merge_cf_variable_identity" in source
        assert '" ".join(' not in source
    assert "_STANDARD_NAME_QUALIFIERS" not in ssot_source


def test_one_identity_declared_by_two_names_raises_even_when_units_agree():
    catalog = SSOTCatalog()
    catalog.add_entry(_entry("canopy_t", "canopy_temperature"))

    with pytest.raises(ValueError, match=r"'canopy_t'.*'big_leaf_t'"):
        catalog.add_entry(_entry("big_leaf_t", "canopy_temperature"))


def _write_cf_contract(directory, variable, cf_standard_name, cell_methods=None, units="K"):
    lines = [
        f"variable: {variable}",
        "kind: forcing",
        "dimension:",
        f'  udunits: "{units}"',
        f"  cf_standard_name: {cf_standard_name}",
    ]
    if cell_methods is not None:
        lines.append(f'  cell_methods: "{cell_methods}"')
    lines += ["owner: null", "aliases: []", "incrementable: false"]
    (directory / f"{variable}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_two_contracts_declaring_one_cf_identity_raise(tmp_path):
    """The contract cf_standard_name enters the one-holder rule of the catalog."""
    _write_cf_contract(tmp_path, "ozone_a", "mole_fraction_of_ozone_in_air", units="mol mol-1")
    _write_cf_contract(tmp_path, "ozone_b", "mole_fraction_of_ozone_in_air", units="mol mol-1")

    with pytest.raises(ValueError, match=r"'ozone_a'.*'ozone_b'.*mole_fraction_of_ozone_in_air"):
        SSOTCatalog()._load_contract_entries(tmp_path)


def test_contract_cf_identity_collides_with_a_catalog_standard_name(tmp_path):
    catalog = SSOTCatalog()
    catalog.add_entry(_entry("tas", "air_temperature"))
    _write_cf_contract(tmp_path, "surface_air_t", "air_temperature")

    with pytest.raises(ValueError, match=r"'tas'.*'surface_air_t'"):
        catalog._load_contract_entries(tmp_path)


def test_contract_cell_methods_qualify_its_cf_identity(tmp_path):
    """CF-1.8 7.3: one standard_name with different cell_methods is two identities."""
    _write_cf_contract(tmp_path, "tas", "air_temperature")
    _write_cf_contract(tmp_path, "tasmax", "air_temperature", cell_methods="time: maximum")
    catalog = SSOTCatalog()
    catalog._load_contract_entries(tmp_path)

    assert catalog.get_by_standard_name("air_temperature").name == "tas"
    assert catalog.get_by_standard_name(
        "air_temperature", cell_methods="time:  maximum"
    ).name == "tasmax"
    assert catalog.get_by_standard_name("air_temperature", cell_methods="time: minimum") is None


def test_every_packaged_contract_cf_identity_has_its_contract_as_the_one_holder():
    from relic.contracts import load_contract_registry

    catalog = SSOTCatalog.from_parameter_manager(get_global_manager())
    registry = load_contract_registry()
    unresolved = []
    for variable in registry.list_variables():
        contract = registry.get_contract(variable)
        if contract.cf_standard_name is None:
            continue
        holder = catalog.get_by_standard_name(
            contract.cf_standard_name, cell_methods=contract.cell_methods
        )
        if holder is None or holder.name != variable:
            unresolved.append(
                (variable, contract.cf_standard_name, contract.cell_methods, holder and holder.name)
            )
    assert unresolved == []


def test_contract_cf_conflicts_have_one_holder_per_identity():
    """The conflicts exposed by the contract CF check, each with its decided holder."""
    catalog = SSOTCatalog.from_parameter_manager(get_global_manager())

    # Daily extrema: the phenology forcings hold the qualified identities; the
    # external coupling names are their contract aliases, merged into them.
    for name, alias, method in (
        ("air_temperature_max", "daily_maximum_air_temperature", "time: maximum"),
        ("air_temperature_min", "daily_minimum_air_temperature", "time: minimum"),
    ):
        assert alias not in catalog._entries
        assert catalog.get(alias).name == name
        assert catalog.get_by_standard_name("air_temperature", cell_methods=method).name == name
    assert catalog.get_by_standard_name("air_temperature").name == "air_temperature"
    # Height: the vertical layer coordinate holds it; the tower receptor height declares none.
    assert catalog.get_by_standard_name("height").name == "layer_center_height"
    assert catalog.get("measurement_height").cf_standard_name is None
    # Phase-partitioned precipitation: the snow_pack fluxes hold the CF names; the
    # external coupling names are their contract aliases.
    for name, alias in (("rain_flux", "rainfall_flux"), ("snowfall", "snowfall_flux")):
        assert alias not in catalog._entries
        assert catalog.get(alias).name == name
        assert catalog.get_by_standard_name(alias).name == name
