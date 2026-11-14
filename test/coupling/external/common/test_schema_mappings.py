from relic.coupling.external.common.datasource import DataSource
from relic.coupling.external.common.schema import InternalSchema


def test_wrf_default_mapping_includes_radiation_and_pressure():
    """Ensure InternalSchema mappings provide required radiation and pressure variables."""
    schema = InternalSchema()
    mapping = schema.create_mapping(
        DataSource.WRF,
        [
            "shortwave_down",
            "longwave_down",
            "atmospheric_pressure",
        ],
    )

    assert mapping.get("SWDOWN") == "shortwave_down"
    assert mapping.get("PSFC") == "atmospheric_pressure"
    # Accept either LWDOWN or GLW depending on availability order
    assert any(
        mapping.get(ext) == "longwave_down"
        for ext in ("LWDOWN", "GLW")
    )


def test_create_mapping_raises_naming_every_unavailable_variable():
    """create_mapping returns the complete external -> internal mapping of the requested variables
    or raises KeyError naming every requested variable the source does not provide; it never warns
    and returns a partial mapping."""
    import warnings

    import pytest

    schema = InternalSchema()
    with pytest.raises(KeyError) as info:
        schema.create_mapping(
            DataSource.WRF, ["shortwave_down", "net_primary_productivity", "not_a_catalog_variable"]
        )
    message = str(info.value)
    assert "net_primary_productivity" in message and "not_a_catalog_variable" in message
    assert "WRF" in message.upper()
    assert "['net_primary_productivity', 'not_a_catalog_variable']" in message
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert schema.create_mapping(DataSource.WRF, ["shortwave_down"]) == {"SWDOWN": "shortwave_down"}


def test_source_mapping_is_served_by_the_schema_owner():
    """Adapters read the full external -> internal mapping from InternalSchema."""
    import pytest

    import relic.coupling.external.common.schema as schema_module

    schema = InternalSchema()
    wrf = schema.get_source_mapping(DataSource.WRF)
    assert wrf["T2"] == "air_temperature"
    assert wrf["RH2"] == "relative_humidity"
    cmip6 = schema.get_source_mapping(DataSource.CMIP6)
    assert cmip6["pr"] == "precipitation_flux"
    # The returned mapping is a copy; the owner's table is not mutated by callers.
    wrf["T2"] = "changed"
    assert schema.get_source_mapping(DataSource.WRF)["T2"] == "air_temperature"
    with pytest.raises(KeyError, match="UNKNOWN"):
        schema.get_source_mapping(DataSource.UNKNOWN)
    # No second module-level accessor exists next to the owner.
    assert not hasattr(schema_module, "get_variable_mappings")
    assert not hasattr(schema_module, "get_internal_schema")


def test_positive_definite_follows_the_catalog_valid_min():
    """A field is positive definite when its catalog valid_min is non-negative."""
    import pytest

    from relic.coupling.external.common.schema import VariableCategory, VariableDefinition
    from relic.coupling.scales.types import SpatialQuantityBasis

    schema = InternalSchema()
    expected = {
        "precipitation_flux": True,  # valid_min 0
        "shortwave_down": True,  # valid_min 0
        "air_temperature": True,  # valid_min 180 K
        "eastward_wind": False,  # valid_min -100 m s-1
        "sensible_heat_flux": False,  # valid_min -200 W m-2 (CF surface_upward_sensible_heat_flux)
        "net_radiation": False,  # valid_min -400 W m-2
    }
    for name, positive in expected.items():
        assert schema.get_variable(name).positive_definite is positive, name
    undeclared = VariableDefinition(
        standard_name="x",
        long_name="x",
        units="1",
        spatial_basis=SpatialQuantityBasis.INTENSIVE,
        category=VariableCategory.ATMOSPHERIC_STATE,
    )
    with pytest.raises(ValueError, match="valid_min"):
        undeclared.positive_definite


def test_many_to_one_inversion_needs_a_declared_canonical_source():
    """Inverting external -> internal mappings never keeps an arbitrary (last) external name."""
    import importlib
    import inspect

    import pytest

    schema = InternalSchema()
    # CMIP6: pr is total precipitation; prc is its convective part with its own CF name.
    assert schema.get_source_mapping(DataSource.CMIP6)["prc"] == "convective_precipitation_flux"
    assert schema.resolve_external_names(DataSource.CMIP6, ["precipitation_flux"]) == {
        "precipitation_flux": "pr"
    }
    # tas / tasmax / tasmin -> air_temperature: tas is the declared canonical source.
    assert schema.resolve_external_names(DataSource.CMIP6, ["air_temperature"]) == {"air_temperature": "tas"}
    # WRF convective and grid-scale accumulations are separate internal variables.
    wrf = schema.get_source_mapping(DataSource.WRF)
    assert wrf["RAINC"] == "convective_precipitation_flux"
    assert wrf["RAINNC"] == "large_scale_precipitation_flux"
    assert schema.resolve_external_names(DataSource.WRF, ["longwave_down"]) == {
        "longwave_down": "GLW"
    }
    # VIC offers only daily extrema (TMAX, TMIN): no source for the mean air temperature.
    with pytest.raises(KeyError, match="air_temperature"):
        schema.resolve_external_names(DataSource.VIC, ["air_temperature"])
    with pytest.raises(KeyError, match="leaf_area_index"):
        schema.resolve_external_names(DataSource.CMIP6, ["leaf_area_index"])
    # Every declared canonical source maps to its internal variable.
    for source, table in schema.canonical_sources().items():
        mapping = schema.get_source_mapping(source)
        for internal, external in table.items():
            assert mapping[external] == internal, (source, internal, external)
    for module_name, class_name in (
        ("relic.coupling.external.atmospheric_models.wrf_adapter", "WRFAdapter"),
        ("relic.coupling.external.atmospheric_models.cesm_adapter", "CESMAdapter"),
        ("relic.coupling.external.climate_models.cmip6_adapter", "CMIP6Adapter"),
        ("relic.coupling.external.hydrological_models.vic_adapter", "VICAdapter"),
        ("relic.coupling.external.hydrological_models.swat_adapter", "SWATAdapter"),
    ):
        source = inspect.getsource(getattr(importlib.import_module(module_name), class_name).read)
        assert "for k, v in" not in source, class_name
        assert "resolve_external_names" in source, class_name


def test_no_alias_maps_two_physical_quantities_onto_one_internal_variable():
    """Daily extrema, lowest-level and exposed-LAI fields are not the near-surface means."""
    schema = InternalSchema()
    extrema = {
        DataSource.CMIP6: ("tasmax", "tasmin"),
        DataSource.VIC: ("TMAX", "TMIN"),
    }
    from relic.contracts.contract_loader import load_contract_registry

    registry = load_contract_registry()
    for source, (maximum, minimum) in extrema.items():
        mapping = schema.get_source_mapping(source)
        # The canonical ContractRegistry names; daily_maximum/minimum_air_temperature
        # are their aliases, which StateManager refuses to write.
        assert mapping[maximum] == "air_temperature_max", source
        assert mapping[minimum] == "air_temperature_min", source
    for name, method in (("air_temperature_max", "time: maximum"),
                         ("air_temperature_min", "time: minimum")):
        definition = schema.get_variable(name)
        assert definition.standard_name == "air_temperature"
        assert definition.cell_methods == method
        assert definition.units == "K"
        assert not registry.is_alias(name)
        assert registry.get_contract(name).cell_methods == method
    assert schema.get_variable("daily_maximum_air_temperature") is None
    assert schema.get_variable("daily_minimum_air_temperature") is None
    cesm = schema.get_source_mapping(DataSource.CESM)
    assert "TBOT" not in cesm and "ELAI" not in cesm
    # Every remaining many-to-one set is one physical quantity with a declared canonical source.
    for source in (DataSource.WRF, DataSource.CESM, DataSource.CMIP6, DataSource.VIC, DataSource.SWAT):
        resolved = schema.resolve_external_names(source)
        assert "air_temperature_max" in resolved or source in (
            DataSource.WRF, DataSource.CESM, DataSource.SWAT
        )
    # SWAT output.hru TMP_MX/TMP_MN are daily only for a daily print: they are not mapped.
    swat = schema.get_source_mapping(DataSource.SWAT)
    assert "TMP_MX" not in swat and "TMP_MN" not in swat
    assert schema.canonical_sources()[DataSource.CESM] == {"air_temperature": "TREFHT"}


def test_soil_water_fields_are_layer_water_mass_with_declared_thickness():
    """Per-layer soil water maps to mass_content_of_water_in_soil_layer through a declared transform."""
    schema = InternalSchema()
    assert schema.get_source_mapping(DataSource.CESM)["H2OSOI"] == "mass_content_of_water_in_soil_layer"
    cesm = schema.get_source_transform(DataSource.CESM, "H2OSOI")
    assert cesm["kind"] == "layered"
    assert cesm["content_units"] == "mm3/mm3"
    assert cesm["thickness_variable"] == "DZSOI"
    assert cesm["thickness_layer_dimension"] == "levgrnd"
    assert cesm["configured_thickness"] == "soil_layer_thickness"
    assert schema.get_source_mapping(DataSource.VIC)["OUT_SOIL_MOIST"] == "mass_content_of_water_in_soil_layer"
    vic = schema.get_source_transform(DataSource.VIC, "OUT_SOIL_MOIST")
    assert vic == {"kind": "water_depth", "layer_dimension": "nlayer"}
    # SWAT SW is the profile water above the wilting point, not the soil water content.
    assert "SW" not in schema.get_source_mapping(DataSource.SWAT)
    for source in (DataSource.CESM, DataSource.VIC, DataSource.SWAT):
        assert "soil_moisture_content" not in schema.get_source_mapping(source).values(), source


def test_water_fluxes_snow_and_lowest_level_fields_are_mapped_by_meaning():
    """A water depth is never an energy flux; SWE is a snow amount; lowest-level fields are unmapped."""
    schema = InternalSchema()
    vic = schema.get_source_mapping(DataSource.VIC)
    swat = schema.get_source_mapping(DataSource.SWAT)
    assert vic["OUT_EVAP"] == "evapotranspiration_flux"
    assert swat["ET"] == "evapotranspiration_flux"
    assert vic["OUT_SWE"] == "surface_snow_amount"
    for source in (DataSource.VIC, DataSource.SWAT):
        assert "latent_heat_flux" not in schema.get_source_mapping(source).values()
    cesm = schema.get_source_mapping(DataSource.CESM)
    for name in ("PBOT", "QBOT", "UBOT", "VBOT", "TBOT"):
        assert name not in cesm
    evapotranspiration = schema.get_variable("evapotranspiration_flux")
    assert (evapotranspiration.standard_name, evapotranspiration.units) == (
        "water_evapotranspiration_flux",
        "kg m-2 s-1",
    )
    snow = schema.get_variable("surface_snow_amount")
    assert (snow.standard_name, snow.units) == ("surface_snow_amount", "kg m-2")
    assert schema.get_source_transform(DataSource.VIC, "OUT_EVAP") == {"kind": "water_depth_flux"}
    assert schema.get_source_transform(DataSource.SWAT, "ET") == {"kind": "water_depth_flux"}
    assert schema.get_source_transform(DataSource.VIC, "OUT_SWE") == {"kind": "water_depth", "layer_dimension": None}


def test_surface_sensible_heat_sources_route_to_the_canonical_variable():
    """WRF HFX, CLM FSH and CMIP6 hfss feed sensible_heat_flux, the one RELIC holder
    of CF surface_upward_sensible_heat_flux (written by atmospheric_turbulence and
    read by atmospheric transport); no second internal name carries that identity."""
    from relic.contracts.contract_loader import load_contract_registry
    from relic.orchestration.ssot_catalog import SSOTCatalog

    schema = InternalSchema()
    for source, external in (
        (DataSource.WRF, "HFX"),
        (DataSource.CESM, "FSH"),
        (DataSource.CMIP6, "hfss"),
    ):
        assert schema.get_source_mapping(source)[external] == "sensible_heat_flux", source
    definition = schema.get_variable("sensible_heat_flux")
    assert (definition.standard_name, definition.units) == (
        "surface_upward_sensible_heat_flux",
        "W m-2",
    )
    assert schema.get_variable("surface_upward_sensible_heat_flux") is None
    assert (
        load_contract_registry().get_contract("sensible_heat_flux").cf_standard_name
        == "surface_upward_sensible_heat_flux"
    )
    # Loading the catalog runs the CF identity uniqueness check; the identity
    # resolves to the canonical variable.
    catalog = SSOTCatalog.from_parameter_manager()
    holder = catalog.get_by_standard_name("surface_upward_sensible_heat_flux")
    assert holder.name == "sensible_heat_flux"
    assert holder.units == "W m-2"


def test_precipitation_phase_fields_map_to_their_canonical_phase_variables():
    """Snowfall and rain are read only from fields that hold exactly that phase.

    CMIP6 prsn (snowfall_flux, solid phase at the surface) is the RELIC variable
    snowfall. CAM PRECL is the large-scale (stable) precipitation rate of liquid
    plus ice (cam_diagnostics addfld 'PRECL'), so it is large-scale precipitation,
    not rain; CAM writes the snow rates PRECSL (large-scale) and PRECSC
    (convective) and no field named PRECS. Surface rain would be PRECT - PRECSC -
    PRECSL, a combination of three fields, which a one-to-one mapping cannot
    express, so no CESM field maps to rain_flux or snowfall.
    """
    from relic.contracts.contract_loader import load_contract_registry

    registry = load_contract_registry()
    schema = InternalSchema()

    cmip6 = schema.get_source_mapping(DataSource.CMIP6)
    assert cmip6["prsn"] == "snowfall"
    assert cmip6["pr"] == "precipitation_flux"
    assert cmip6["prc"] == "convective_precipitation_flux"
    snowfall = schema.get_variable("snowfall")
    assert (snowfall.standard_name, snowfall.units) == ("snowfall_flux", "kg m-2 s-1")
    assert not registry.is_alias("snowfall")
    assert registry.get_contract("snowfall").cf_standard_name == "snowfall_flux"
    assert schema.get_variable("snowfall_flux") is None

    cesm = schema.get_source_mapping(DataSource.CESM)
    assert cesm["PRECT"] == "precipitation_flux"
    assert cesm["PRECL"] == "large_scale_precipitation_flux"
    assert "PRECS" not in cesm
    assert not {"rain_flux", "rainfall_flux", "snowfall", "snowfall_flux"} & set(cesm.values())

    # WRF RAINC/RAINNC accumulate the total (liquid plus frozen) precipitation of
    # the cumulus and the grid-scale schemes.
    wrf = schema.get_source_mapping(DataSource.WRF)
    assert wrf["RAINC"] == "convective_precipitation_flux"
    assert wrf["RAINNC"] == "large_scale_precipitation_flux"
    for source in (DataSource.WRF, DataSource.CESM, DataSource.CMIP6, DataSource.VIC, DataSource.SWAT):
        mapped = set(schema.get_source_mapping(source).values())
        assert not {"rainfall_flux", "snowfall_flux"} & mapped, source


def test_no_mapping_target_is_a_contract_alias():
    """Every mapped internal name is a canonical name: StateManager refuses to write an alias.

    Surface pressure (WRF PSFC, CESM PS, CMIP6 ps) maps to the contract name atmospheric_pressure;
    air_pressure is that contract's alias.
    """
    from relic.contracts.contract_loader import load_contract_registry

    registry = load_contract_registry()
    schema = InternalSchema()
    aliased = {}
    for source in (DataSource.WRF, DataSource.CESM, DataSource.CMIP6, DataSource.VIC, DataSource.SWAT):
        for external, internal in schema.get_source_mapping(source).items():
            if registry.is_alias(internal):
                aliased[f"{source.value}:{external}"] = (internal, registry.resolve_alias(internal))
    assert aliased == {}
    assert schema.get_source_mapping(DataSource.WRF)["PSFC"] == "atmospheric_pressure"
    assert schema.get_source_mapping(DataSource.CESM)["PS"] == "atmospheric_pressure"
    assert schema.get_source_mapping(DataSource.CMIP6)["ps"] == "atmospheric_pressure"


_ADAPTER_NAMES = {
    "wrf": "WRFAdapter",
    "cesm": "CESMAdapter",
    "cmip6": "CMIP6Adapter",
    "vic": "VICAdapter",
    "swat": "SWATAdapter",
}
_SOURCES = (DataSource.WRF, DataSource.CESM, DataSource.CMIP6, DataSource.VIC, DataSource.SWAT)


def test_every_mapping_target_has_a_contract_that_authorizes_exactly_its_adapters():
    """Every variable an adapter writes has a variable contract (kind forcing when no RELIC
    process writes it) whose writers include the adapter Functor of each source mapping it, in
    the catalog units; conversely every adapter writer of a contract comes from a mapping."""
    from relic.contracts.contract_loader import load_contract_registry
    from relic.parameters import get_global_manager

    registry = load_contract_registry()
    schema = InternalSchema()
    catalog = get_global_manager().get(
        "coupling.external.variables.variable_definitions", expected_type=dict, required=True
    )
    expected = {}
    for source in _SOURCES:
        adapter = _ADAPTER_NAMES[source.value]
        for internal in schema.get_source_mapping(source).values():
            expected.setdefault(internal, set()).add(adapter)
    problems = []
    for internal, adapters in sorted(expected.items()):
        contract = registry.get_contract(internal)
        if contract is None:
            problems.append(f"{internal}: no contract")
            continue
        for adapter in sorted(adapters):
            if not registry.can_write(internal, adapter):
                problems.append(f"{internal}: {adapter} is not a writer")
        if contract.udunits != catalog[internal]["units"]:
            problems.append(f"{internal}: contract {contract.udunits!r} vs catalog {catalog[internal]['units']!r}")
        if contract.cf_standard_name != catalog[internal]["standard_name"]:
            problems.append(
                f"{internal}: contract CF {contract.cf_standard_name!r} vs catalog {catalog[internal]['standard_name']!r}"
            )
    for variable in registry.list_variables():
        declared = {name for name in registry.get_contract(variable).writers if name.endswith("Adapter")}
        if declared - expected.get(variable, set()):
            problems.append(f"{variable}: adapter writers {sorted(declared - expected.get(variable, set()))} map nothing")
    assert problems == []


def test_one_cf_identity_is_held_by_one_variable_and_mapped_to_it():
    """The external catalog holds no duplicate of a contract variable's CF identity: surface
    radiation, latent heat and evapotranspiration map to the contract names, which hold the CF
    names, and the duplicate catalog keys are gone."""
    from relic.contracts.contract_loader import load_contract_registry
    from relic.orchestration.ssot_catalog import SSOTCatalog
    from relic.parameters import get_global_manager
    from relic.scientific.hydrosphere.interface_mapping import FLUX_MAPPINGS

    schema = InternalSchema()
    canonical = {
        "shortwave_down": ("surface_downwelling_shortwave_flux_in_air", {"wrf": ["SWDOWN"], "cesm": ["FSDS"], "cmip6": ["rsds"]}),
        "longwave_down": ("surface_downwelling_longwave_flux_in_air", {"wrf": ["LWDOWN", "GLW"], "cesm": ["FLDS"], "cmip6": ["rlds"]}),
        "latent_heat_flux": ("surface_upward_latent_heat_flux", {"wrf": ["LH"], "cesm": ["EFLX_LH_TOT"], "cmip6": ["hfls"]}),
        "evapotranspiration_flux": ("water_evapotranspiration_flux", {"vic": ["OUT_EVAP"], "swat": ["ET"]}),
    }
    registry = load_contract_registry()
    ssot = SSOTCatalog.from_parameter_manager()
    catalog = get_global_manager().get(
        "coupling.external.variables.variable_definitions", expected_type=dict, required=True
    )
    for variable, (cf_name, sources) in canonical.items():
        for source in _SOURCES:
            for external in sources.get(source.value, []):
                assert schema.get_source_mapping(source)[external] == variable, (source, external)
        assert registry.get_contract(variable).cf_standard_name == cf_name
        assert ssot.get_by_standard_name(cf_name).name == variable
        assert catalog[variable]["standard_name"] == cf_name
    for duplicate in (
        "surface_downwelling_shortwave_flux",
        "surface_downwelling_longwave_flux",
        "surface_upward_latent_heat_flux",
        "water_evapotranspiration_flux",
    ):
        assert duplicate not in catalog
        assert ssot.get(duplicate) is None or ssot.get(duplicate).name != duplicate
    et = next(mapping for mapping in FLUX_MAPPINGS if mapping.standard_name == "evapotranspiration_flux")
    assert et.cf_standard_name == "water_evapotranspiration_flux"
    assert "external catalog" not in et.description
    # The contract's signed convention is the catalog's: dew and condensation are negative.
    assert catalog["latent_heat_flux"]["valid_min"] < 0.0
    assert catalog["evapotranspiration_flux"]["valid_min"] < 0.0


def test_ten_metre_wind_and_transport_grid_wind_are_distinct_variables():
    """WRF U10/V10 and CMIP6 uas/vas are the 10 m wind (height coordinate 10 m) and map to
    eastward_wind/northward_wind, which hold the CF names; wind_u/wind_v are the wind components
    of the tracer-transport grid (AtmosphericAdvectionDiffusionFunctor) and hold no CF name."""
    from relic.contracts.contract_loader import load_contract_registry
    from relic.orchestration.ssot_catalog import SSOTCatalog

    schema = InternalSchema()
    registry = load_contract_registry()
    ssot = SSOTCatalog.from_parameter_manager()
    for external, internal in (("U10", "eastward_wind"), ("V10", "northward_wind")):
        assert schema.get_source_mapping(DataSource.WRF)[external] == internal
    assert schema.get_source_mapping(DataSource.CMIP6)["uas"] == "eastward_wind"
    assert schema.get_source_mapping(DataSource.CMIP6)["vas"] == "northward_wind"
    for ten_metre, grid in (("eastward_wind", "wind_u"), ("northward_wind", "wind_v")):
        contract = registry.get_contract(ten_metre)
        assert contract.kind == "forcing"
        assert contract.cf_standard_name == ten_metre
        assert contract.raw["support"]["vertical_basis"] == "height_level"
        assert "10 m" in contract.raw["description"]
        assert ssot.get_by_standard_name(ten_metre).name == ten_metre
        transport = registry.get_contract(grid)
        assert transport.cf_standard_name is None
        assert ten_metre in transport.raw["description"]
        assert not registry.is_alias(grid) and not registry.is_alias(ten_metre)


def _declared_ranges(variable):
    """Every catalog declaration of the valid range of one variable, keyed by its location.

    The locations are the external catalog entry (valid_min/valid_max and bounds) and the scale
    catalogs coupling.scales.{defaults,daily}.variables; a catalog that does not list the variable
    contributes nothing. An external entry declares both its valid range and its bounds.
    """
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    ranges = {}
    external = manager.get(
        "coupling.external.variables.variable_definitions", expected_type=dict, required=True
    ).get(variable)
    if external is not None:
        ranges["external.valid"] = (external["valid_min"], external["valid_max"])
        ranges["external.bounds"] = tuple(external["bounds"])
    for scale in ("defaults", "daily"):
        entry = manager.get(f"coupling.scales.{scale}.variables", expected_type=dict, required=True).get(variable)
        if entry is not None:
            ranges[f"scales.{scale}"] = tuple(entry["bounds"])
    return {location: tuple(float(value) for value in pair) for location, pair in ranges.items()}


def _catalog_entries(variable):
    """The external and scale catalog entries of one variable (absent catalogs are skipped)."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    entries = {}
    external = manager.get(
        "coupling.external.variables.variable_definitions", expected_type=dict, required=True
    ).get(variable)
    if external is not None:
        entries["external"] = external
    for scale in ("defaults", "daily"):
        entry = manager.get(f"coupling.scales.{scale}.variables", expected_type=dict, required=True).get(variable)
        if entry is not None:
            entries[f"scales.{scale}"] = entry
    return entries


def _coupling_variable_bound_keys():
    """The keys of coupling.variables.*.bounds declared in variable_bounds.yaml."""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[4] / "relic/parameters/database/relic/coupling/variable_bounds.yaml"
    return set(yaml.safe_load(path.read_text(encoding="utf-8-sig"))["parameters"])


def test_evapotranspiration_has_one_signed_valid_range_and_latent_heat_follows_it():
    """evapotranspiration_flux (CF water_evapotranspiration_flux: all water vapour fluxes into the
    atmosphere, so condensation and dew are negative) has one valid range, declared identically in
    every catalog that holds one; latent_heat_flux = lambda * evapotranspiration_flux has the range
    lambda times it. The contracts keep no clipping bounds (the quantity is signed and unbounded
    in the model; the range is a plausibility range). The upper bound is derived from the energy
    supply; the lysimeter observations of Tolk et al. (2006) are its observational support."""
    import pytest

    from relic.contracts.contract_loader import load_contract_registry
    from relic.utils.physical_constants import get_physical_constant

    et = _declared_ranges("evapotranspiration_flux")
    assert set(et) == {"external.valid", "external.bounds", "scales.defaults", "scales.daily"}, et
    assert set(et.values()) == {(-1.0e-4, 6.0e-4)}, et
    lam = float(get_physical_constant("latent_heat_vaporization"))
    le = _declared_ranges("latent_heat_flux")
    assert set(le) == {"external.valid", "external.bounds", "scales.defaults", "scales.daily"}, le
    for location, (lower, upper) in le.items():
        assert lower == pytest.approx(lam * -1.0e-4), location
        assert upper == pytest.approx(lam * 6.0e-4), location
    registry = load_contract_registry()
    for variable in ("evapotranspiration_flux", "latent_heat_flux"):
        numerics = registry.get_contract(variable).raw["numerics"]
        assert numerics["lower_bound"] is None and numerics["upper_bound"] is None, variable
    entry = _catalog_entries("evapotranspiration_flux")["external"]
    references = " ".join(entry["references"])
    for cited in ("water_evapotranspiration_flux", "Kopp and Lean (2011)", "dew"):
        assert cited in references, cited
    # Observational support: Tolk, Evett and Howell (2006) Agron. J. 98(6):1646-1654.
    for cited in ("Tolk, J.A., Evett, S.R. and Howell, T.A. (2006)", "98(6):1646-1654", "10.2134/agronj2006.0031"):
        assert cited in references, cited
    assert "coupling.variables" not in references
    provenance = entry["provenance"]
    assert "1470 W m-2" in provenance["derived_bound"] and "Kopp and Lean (2011)" in provenance["derived_bound"]
    assert "Tolk et al. (2006)" in provenance["observational_support"]
    assert "11.3 mm d-1" in provenance["observational_support"]


# The one valid range of each variable of MA-21 item 186: units, [lower, upper], and the source
# or convention label its references must cite. Every catalog that lists the variable declares
# exactly this range; an SCM contract of the variable declares the same units and, where it
# declares clipping bounds, bounds inside this range.
SINGLE_RANGES = {
    "canopy_conductance": ("m s-1", (0.0, 0.05), ("Kelliher", "convention")),
    # Since 2026-10-15 the state contract and both scale catalogs declare [0, inf): the former 2.0
    # guard was an empirical convention, not a bound of the Medlyn law (Medlyn et al. 2011; CTSM
    # tech note Eq. 2.9.1); Murray et al. (2019) is cited for why observed maxima are no bound.
    "stomatal_conductance": ("mol m-2 s-1", (0.0, float("inf")), ("Murray", "Medlyn")),
    "sensible_heat_flux": ("W m-2", (-200.0, 1000.0), ("surface_upward_sensible_heat_flux", "convention")),
    "net_radiation": ("W m-2", (-400.0, 2200.0), ("surface_net_downward_radiative_flux", "Cordero", "2177")),
    "shortwave_down": ("W m-2", (0.0, 2200.0), ("surface_downwelling_shortwave_flux_in_air", "Cordero", "2177")),
    "longwave_down": ("W m-2", (0.0, 700.0), ("surface_downwelling_longwave_flux_in_air", "Stefan-Boltzmann", "56.7")),
    "precipitation_flux": ("kg m-2 s-1", (0.0, 0.52), ("31.2 mm", "Unionville")),
    "runoff_flux": ("kg m-2 s-1", (0.0, 0.52), ("precipitation_flux", "convention")),
    "net_primary_productivity": ("kg C m-2 s-1", (-1.0e-6, 1.0e-6), ("Suyker", "GPP", "convention")),
    "soil_moisture": ("m3 m-3", (0.0, 1.0), ("volume fraction",)),
}

# Former coupling.variables.<name>.bounds declarations of these quantities. register_coupling_variables
# reads only the bounds of its registered outputs, so these repeated or contradicting ranges (in
# mol m-2 s-1 for the conductances, kg C for heterotrophic respiration) had no reader.
UNREAD_COUPLING_BOUNDS = (
    "coupling.variables.stomatal_conductance.bounds",
    "coupling.variables.canopy_conductance.bounds",
    "coupling.variables.sensible_heat_flux.bounds",
    "coupling.variables.heterotrophic_respiration.bounds",
    "coupling.variables.npp.bounds",
    "coupling.variables.evapotranspiration_flux.bounds",
    "coupling.variables.latent_heat_flux.bounds",
)


def test_each_quantity_has_one_cited_valid_range_across_the_catalogs():
    """Every catalog that lists one of these variables declares the one range, in the one unit of
    its contract and readers, and cites its source (primary literature or an observed extreme) or
    labels it a convention; no unread coupling.variables bounds repeat or contradict it."""
    from relic.contracts.contract_loader import load_contract_registry

    registry = load_contract_registry()
    for variable, (units, expected, cited) in SINGLE_RANGES.items():
        ranges = _declared_ranges(variable)
        assert ranges, variable
        assert set(ranges.values()) == {expected}, (variable, ranges)
        entries = _catalog_entries(variable)
        assert {entry["units"] for entry in entries.values()} == {units}, variable
        references = " ".join(str(item) for entry in entries.values() for item in entry["references"])
        for item in cited:
            assert item in references, (variable, item)
        contract = registry.get_contract(variable)
        if contract is not None:
            assert contract.udunits == units, variable
            numerics = contract.raw["numerics"]
            for key, index in (("lower_bound", 0), ("upper_bound", 1)):
                if numerics[key] is not None:
                    assert float(numerics[key]) == expected[index], (variable, key)
    assert not set(UNREAD_COUPLING_BOUNDS) & _coupling_variable_bound_keys()


def test_surface_radiation_bounds_are_derived_and_observed():
    """The downwelling shortwave bound is the greatest measured surface global irradiance
    (2177 W m-2, Cordero et al. 2023) rounded up, and bounds the net radiation (R_n <= S_down);
    the downwelling longwave bound is sigma T^4 of the greatest measured air temperature
    (56.7 degC, WMO), about 671 W m-2, rounded up. Each entry records how its bound is derived
    and what observation supports it."""
    import pytest

    from relic.utils.physical_constants import get_physical_constant

    shortwave = _catalog_entries("shortwave_down")["external"]
    assert "2177 W m-2" in shortwave["provenance"]["observational_support"]
    assert "Cordero et al. (2023)" in shortwave["provenance"]["observational_support"]
    assert "net_radiation" in shortwave["provenance"]["derived_bound"]
    assert shortwave["valid_max"] == _catalog_entries("net_radiation")["external"]["valid_max"]
    longwave = _catalog_entries("longwave_down")["external"]
    sigma = float(get_physical_constant("relic.physical.constants.universal.stefan_boltzmann_constant"))
    limit = sigma * (56.7 + 273.15) ** 4
    assert limit == pytest.approx(671.2, abs=0.1)
    assert limit <= longwave["valid_max"] < 1.1 * limit
    assert "671 W m-2" in longwave["provenance"]["derived_bound"]
    assert "WMO" in longwave["provenance"]["observational_support"]


def test_signed_and_one_signed_carbon_fluxes_follow_their_definitions():
    """NPP = GPP - R_a (CF net_primary_productivity_of_biomass_expressed_as_carbon; autotrophic
    respiration exceeds GPP at night) is signed; heterotrophic respiration is a one-signed release
    (first_order_decomposition and pools_matrix raise on a negative rate, microbial_enzyme forms it
    from non-negative consumption rates, and it accumulates into the carbon.total boundary ledger,
    which accepts only non-negative magnitudes) and keeps no catalog range."""
    from relic.contracts.contract_loader import load_contract_registry

    lower, upper = SINGLE_RANGES["net_primary_productivity"][1]
    assert lower < 0.0 < upper
    for variable in ("heterotrophic_respiration",):
        assert _declared_ranges(variable) == {}
        contract = load_contract_registry().get_contract(variable)
        assert contract.udunits == "g C m-2 s-1"
        assert contract.raw["accumulates"] == "carbon.total"
        assert contract.raw["numerics"]["lower_bound"] == 0.0
        assert contract.raw["numerics"]["upper_bound"] is None
        assert contract.raw["provenance"]["sign_convention"] == "non_negative_release_to_atmosphere"
        assert "non-negative" in contract.raw["description"]


class _CatalogManager:
    """Parameter manager proxy serving a modified copy of the external variable catalog."""

    def __init__(self, catalog):
        self._catalog = catalog

    def get(self, key, *args, **kwargs):
        if str(key) == "variable_definitions":
            return self._catalog
        from relic.parameters import get_global_manager

        return get_global_manager().get(key, *args, **kwargs)


def test_catalog_entries_declare_every_required_field_and_their_category():
    """A catalog entry without standard_name, long_name, units or category, an unknown category
    and a non-mapping entry raise; no entry is skipped and no category is assumed."""
    import copy
    import dataclasses

    import pytest

    from relic.coupling.external.common.schema import VariableCategory, VariableDefinition
    from relic.coupling.scales.types import SpatialQuantityBasis
    from relic.parameters import get_global_manager

    catalog = get_global_manager().get(
        "coupling.external.variables.variable_definitions", expected_type=dict, required=True
    )
    for field_name in ("standard_name", "long_name", "units", "category"):
        broken = copy.deepcopy(catalog)
        del broken["air_temperature"][field_name]
        with pytest.raises(KeyError, match=field_name):
            InternalSchema(parameter_manager=_CatalogManager(broken))
    unknown = copy.deepcopy(catalog)
    unknown["air_temperature"]["category"] = "NOT_A_CATEGORY"
    with pytest.raises(ValueError, match="NOT_A_CATEGORY"):
        InternalSchema(parameter_manager=_CatalogManager(unknown))
    not_mapping = copy.deepcopy(catalog)
    not_mapping["air_temperature"] = "K"
    with pytest.raises(TypeError, match="air_temperature"):
        InternalSchema(parameter_manager=_CatalogManager(not_mapping))
    # The definition itself has no default category and no assumed temporal method.
    fields = {field.name: field for field in dataclasses.fields(VariableDefinition)}
    assert fields["category"].default is dataclasses.MISSING
    assert "temporal_method" not in fields
    with pytest.raises(TypeError):
        VariableDefinition(
            standard_name="x", long_name="x", units="1", spatial_basis=SpatialQuantityBasis.INTENSIVE
        )
    declared = InternalSchema().get_variable("air_temperature")
    assert declared.category is VariableCategory[catalog["air_temperature"]["category"]]
