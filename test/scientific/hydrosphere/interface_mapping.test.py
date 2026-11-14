import inspect

import pytest

from relic.core.functor import FunctorBase
from relic.scientific.hydrosphere import interface_mapping
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
from test.utils.functor_minimal_construction import (
    construct_minimal,
    minimal_arguments,
    relic_functor_class_paths,
    resolve_class,
)


def _iter_functor_classes():
    """Delegate discovery to the existing construction owner, restricted to hydrosphere."""
    prefix = "relic.scientific.hydrosphere"
    for path in relic_functor_class_paths():
        module_name = path.split(":", 1)[0]
        if module_name != prefix and not module_name.startswith(prefix + "."):
            continue
        cls = resolve_class(path)
        assert issubclass(cls, FunctorBase), path
        if cls is not FunctorBase and not inspect.isabstract(cls):
            yield cls


def _instantiate_functor(cls, canopy_resistance_source=None):
    """Use the documented construction once; unexpected errors propagate unchanged."""
    path = f"{cls.__module__}:{cls.__qualname__}"
    if canopy_resistance_source is None:
        return construct_minimal(path)
    assert cls is PenmanMonteithET
    arguments = minimal_arguments(path)
    arguments["config"] = dict(arguments["config"], canopy_resistance_source=canopy_resistance_source)
    return cls(**arguments)


def _mapping_coverage_cases():
    classes = tuple(_iter_functor_classes())
    assert classes, "No concrete hydrosphere Functors were discovered"
    assert len(classes) == len(set(classes)), "Duplicate concrete class discovery"
    assert PenmanMonteithET in classes, "Penman-Monteith must be covered"
    return tuple(
        (cls, mode)
        for cls in classes
        for mode in (PenmanMonteithET.CANOPY_RESISTANCE_SOURCES if cls is PenmanMonteithET else (None,))
    )


_MAPPING_COVERAGE_CASES = _mapping_coverage_cases()


@pytest.mark.parametrize(
    ("cls", "canopy_resistance_source"),
    _MAPPING_COVERAGE_CASES,
    ids=[f"{cls.__module__}:{cls.__qualname__}[{mode or 'minimal'}]" for cls, mode in _MAPPING_COVERAGE_CASES],
)
def test_hydrosphere_interface_mapping_coverage(cls, canopy_resistance_source):
    mapper = interface_mapping.hydrosphere_mapper
    functor = _instantiate_functor(cls, canopy_resistance_source)
    metadata = getattr(functor, "metadata", None)
    assert metadata is not None, f"{cls.__name__} has no metadata"
    declared = set(metadata.input_variables or []) | set(metadata.output_variables or [])
    assert declared, f"{cls.__name__} declares no input or output variables"
    assert all(isinstance(name, str) and name for name in declared), cls.__name__
    vars_to_check = {name for name in declared if not name.startswith("diagnostics.")}
    assert vars_to_check, f"{cls.__name__} has no scientific interface variables to check"
    if cls is PenmanMonteithET:
        assert functor.canopy_resistance_source == canopy_resistance_source
    missing = []
    for var in sorted(vars_to_check):
        try:
            mapper.get_standard_name(var)
        except ValueError:
            missing.append(var)
    assert not missing, f"Missing interface mappings for {cls.__name__}[{canopy_resistance_source or 'minimal'}]: {', '.join(missing)}"


def test_hydrosphere_coverage_cases_are_complete_and_cannot_be_empty(monkeypatch):
    discovered = set(_iter_functor_classes())
    assert discovered
    assert {cls for cls, _ in _MAPPING_COVERAGE_CASES} == discovered
    assert len(_MAPPING_COVERAGE_CASES) == len(discovered) + 1
    assert {mode for cls, mode in _MAPPING_COVERAGE_CASES if cls is PenmanMonteithET} == {
        "state", "fao56_reference_surface",
    }
    assert len(_MAPPING_COVERAGE_CASES) == len(set(_MAPPING_COVERAGE_CASES))
    monkeypatch.setitem(globals(), "_iter_functor_classes", lambda: iter(()))
    with pytest.raises(AssertionError, match="No concrete hydrosphere Functors"):
        _mapping_coverage_cases()


def test_standard_name_strictness():
    mapper = interface_mapping.hydrosphere_mapper
    assert mapper.get_standard_name("precipitation_flux") == "precipitation_flux"
    with pytest.raises(ValueError):
        mapper.get_standard_name("precipitation")
    assert (
        mapper.get_standard_name("evapotranspiration_flux")
        == "evapotranspiration_flux"
    )
    assert mapper.get_units("evapotranspiration_flux") == "kg m-2 s-1"
    with pytest.raises(ValueError):
        mapper.get_standard_name("vapor_flux")


def test_water_table_depth_has_one_canonical_mapping():
    mapper = interface_mapping.hydrosphere_mapper
    assert mapper.get_standard_name("water_table_depth") == "water_table_depth"
    assert mapper.get_units("water_table_depth") == "m"
    with pytest.raises(ValueError):
        mapper.get_standard_name("groundwater_depth")
    assert all(
        "groundwater_depth" not in mapping.aliases
        for mapping in interface_mapping.ALL_MAPPINGS
    )


def test_soil_matric_potential_has_one_canonical_mapping():
    mapper = interface_mapping.hydrosphere_mapper
    assert mapper.get_standard_name("soil_matric_potential") == "soil_matric_potential"
    assert mapper.get_units("soil_matric_potential") == "Pa"
    for obsolete_name in ("matric_potential", "soil_water_matric_potential"):
        with pytest.raises(ValueError):
            mapper.get_standard_name(obsolete_name)


def test_groundwater_boundary_fluxes_have_one_mass_rate_semantics():
    mapper = interface_mapping.hydrosphere_mapper
    for name in (
        "drainage_flux",
        "capillary_rise_flux",
        "lateral_inflow",
        "groundwater_pumping",
        "baseflow_flux",
    ):
        assert mapper.get_standard_name(name) == name
        assert mapper.get_units(name) == "kg m-2 s-1"

    for obsolete_name in (
        "soil_drainage",
        "baseflow_contribution",
        "capillary_flux",
        "lateral_flow",
        "gw_pumping",
        "gw_baseflow",
    ):
        with pytest.raises(ValueError):
            mapper.get_standard_name(obsolete_name)


def test_runtime_residual_mappings_preserve_physical_units():
    mapper = interface_mapping.hydrosphere_mapper
    expected_units = {
        "diagnostics.evapotranspiration.available_energy_residual": "W m-2",
        "diagnostics.evapotranspiration.canopy_storage_residual": "mm",
        "diagnostics.evapotranspiration.negative_et_projection_residual": (
            "kg m-2 s-1"
        ),
        "diagnostics.groundwater.storage_balance_residual": "mm",
        "diagnostics.groundwater.water_table_bounds_residual": "m",
        "diagnostics.groundwater.groundwater_storage_negative_residual": "mm",
        "diagnostics.subsurface_runoff.water_table_depth_clipping_residual": "m",
        "diagnostics.surface_runoff.precipitation_limit_residual": "kg m-2 s-1",
        "diagnostics.surface_runoff.runoff_nonnegative_residual": "kg m-2 s-1",
        "diagnostics.soil_moisture_bucket.clipped_mass_residual": "kg m-2 s-1",
        "diagnostics.soil_moisture_bucket.drainage_limiter_residual": "m s-1",
        "diagnostics.soil_moisture_richards.mass_conservation_error": (
            "dimensionless"
        ),
        "diagnostics.soil_moisture_richards.moisture_clipping_residual": (
            "m3 m-3"
        ),
        "diagnostics.capillary_rise.conductivity_clipping_residual": (
            "kg m-2 s-1"
        ),
        "diagnostics.capillary_rise.saturation_constraint_residual": (
            "kg m-2 s-1"
        ),
    }

    for name, units in expected_units.items():
        assert mapper.get_standard_name(name) == name
        assert mapper.get_units(name) == units

    for stale_name in (
        "diagnostics.subsurface_runoff.storage_capacity_residual",
        "diagnostics.subsurface_runoff.storage_depletion_residual",
        "diagnostics.percolation.clipped_mass_residual",
        "diagnostics.percolation.drainage_flux_upper_residual",
        "diagnostics.percolation.drainage_threshold_residual",
    ):
        with pytest.raises(ValueError):
            mapper.get_standard_name(stale_name)


def test_declared_cf_standard_names_are_cf_entries_with_their_canonical_units():
    """Every cf_standard_name is a CF v94 entry (not an alias) whose unit matches."""
    import relic.scientific.hydrosphere.interface_mapping as mapping_module
    from relic.data.standards.cf_conventions import CFConventions

    problems = []
    for mapping in mapping_module.ALL_MAPPINGS:
        cf_name = mapping.cf_standard_name
        if cf_name is None:
            continue
        if not CFConventions.is_valid_standard_name(cf_name):
            problems.append(f"{mapping.standard_name}: {cf_name!r} is not in the CF table")
            continue
        info = CFConventions.get_standard_name_info(cf_name)
        if info["membership_kind"] != "entry":
            problems.append(f"{mapping.standard_name}: {cf_name!r} is a CF {info['membership_kind']}")
        units = "1" if mapping.units in ("dimensionless", "m3 m-3") else mapping.units
        if not CFConventions.are_standard_name_units_equivalent(cf_name, units):
            problems.append(f"{mapping.standard_name}: {mapping.units!r} is not the unit of {cf_name!r}")
    assert problems == []


def test_liquid_soil_water_and_rates_without_a_cf_name_state_why():
    import relic.scientific.hydrosphere.interface_mapping as mapping_module

    by_name = {mapping.standard_name: mapping for mapping in mapping_module.ALL_MAPPINGS}
    for name in ("soil_moisture", "infiltration_flux"):
        assert by_name[name].cf_standard_name is None, name
        assert "CF" in by_name[name].description, name
    assert by_name["snow_depth"].cf_standard_name == "surface_snow_thickness"


RETIRED_BARE_NAMES = (
    "drainage_input_rate",
    "water_flux_profile",
    "vertical_flux",
    "interflow_flux",
    "total_subsurface_rate",
    "diagnostics.percolation",
    "percolation_diagnostics",
    "saturation_fraction",
    "storage_deficit",
    "capillary_availability",
    "aerodynamic_conductance",
    "infiltration_capacity",
    "hydraulic_conductivity",
    "runoff_coefficient",
    "precipitation_rate_mm_h",
    "runoff_rate_mm_h",
    "scs_retention",
    "initial_abstraction",
    "storage_fraction",
    "recession_time_constant",
    "total_water_content",
    "average_soil_moisture",
    "surface_soil_moisture",
    "mass_conservation_error",
    "solver_converged",
    "solver_iterations",
)


def test_catalog_lists_no_retired_bare_names():
    """Names whose SCM contracts were retired have no runtime producer or consumer
    (the executed values live under diagnostics.* keys or are parameters), so the
    catalog does not register them as hydrosphere variables."""
    mapper = interface_mapping.hydrosphere_mapper
    listed = [name for name in RETIRED_BARE_NAMES if name in mapper.get_all_names()["standard"]]
    assert listed == []
    for mapping in interface_mapping.ALL_MAPPINGS:
        assert not set(mapping.aliases) & set(RETIRED_BARE_NAMES), mapping.standard_name


def test_snowpack_and_sorption_labels_follow_the_scm_contracts():
    """Category, units and CF name of these catalog entries equal their contracts."""
    from relic.contracts import load_contract_registry

    registry = load_contract_registry()
    by_name = {mapping.standard_name: mapping for mapping in interface_mapping.ALL_MAPPINGS}
    for name in (
        "snowfall",
        "rain_flux",
        "snow_melt_flux",
        "snow_water_equivalent",
        "snow_depth",
        "soil_moisture",
    ):
        contract = registry.get_contract(name)
        mapping = by_name[name]
        assert mapping.category.value == contract.kind, name
        assert mapping.units == contract.udunits, name
        assert mapping.cf_standard_name == contract.cf_standard_name, name
    # The sorption water content is the prognostic soil_moisture state: one entry, no alias.
    assert "volume_fraction_of_water_in_soil" not in by_name
    assert "volume_fraction_of_water_in_soil" not in by_name["soil_moisture"].aliases


# aerodynamic_resistance keeps its catalog labels while the G7 publication freeze of
# its contract support holds; every other contracted entry must match its contract.
CONTRACT_ALIGNMENT_EXCEPTIONS = {"aerodynamic_resistance"}


def test_catalog_entries_follow_their_scm_contracts():
    """Category, CF name and units of each contracted catalog entry equal the SCM
    contract of the same variable ("1" and "dimensionless" are one unit)."""
    from relic.contracts import load_contract_registry

    def normalized(units):
        return "dimensionless" if units in ("", "1", "dimensionless") else units

    registry = load_contract_registry()
    disagreements = []
    for mapping in interface_mapping.ALL_MAPPINGS:
        name = mapping.standard_name
        contract = registry.get_contract(name)
        if contract is None or name in CONTRACT_ALIGNMENT_EXCEPTIONS:
            continue
        if mapping.category.value != contract.kind:
            disagreements.append((name, "category", mapping.category.value, contract.kind))
        if (mapping.cf_standard_name or None) != (contract.cf_standard_name or None):
            disagreements.append((name, "cf", mapping.cf_standard_name, contract.cf_standard_name))
        if normalized(mapping.units) != normalized(contract.udunits):
            disagreements.append((name, "units", mapping.units, contract.udunits))
    assert disagreements == []


def test_one_cf_identity_is_held_by_one_hydrosphere_entry():
    """The entries 'precipitation' and 'precipitation_flux' were one quantity (the surface
    precipitation mass flux, kg m-2 s-1, CF precipitation_flux); the catalog keeps one entry per CF
    identity. The bare name 'precipitation' (an amount or a flux) is refused with its canonical
    replacement, and an alias names one quantity: rainfall is the liquid part (rain_flux)."""
    from collections import defaultdict

    holders = defaultdict(list)
    for mapping in interface_mapping.ALL_MAPPINGS:
        if mapping.cf_standard_name:
            holders[mapping.cf_standard_name].append(mapping.standard_name)
    assert {cf: names for cf, names in holders.items() if len(names) > 1} == {}
    mapper = interface_mapping.hydrosphere_mapper
    assert "precipitation" not in mapper.get_all_names()["standard"]
    by_name = {mapping.standard_name: mapping for mapping in interface_mapping.ALL_MAPPINGS}
    precipitation = by_name["precipitation_flux"]
    assert precipitation.cf_standard_name == "precipitation_flux"
    assert precipitation.units == "kg m-2 s-1"
    # 'precip' is refused as an amount-or-flux short name (coupling.scales.variable_catalog).
    assert "precip" not in precipitation.aliases
    assert "rainfall_rate" not in precipitation.aliases
    assert "rainfall_rate" in by_name["rain_flux"].aliases
    with pytest.raises(ValueError, match="precipitation_flux"):
        mapper.get_standard_name("precipitation")


def test_conductance_entries_use_the_units_of_their_catalog_declaration():
    """stomatal_conductance is the leaf-level conductance in mol m-2 s-1 read by ozone_stress and
    oxygen_18 (its scale catalog declaration); canopy_conductance is the bulk canopy conductance in
    m s-1 of its SCM contract. The hydrosphere catalog declares the same units."""
    from relic.parameters import get_global_manager

    scale = get_global_manager().get("coupling.scales.defaults.variables", expected_type=dict, required=True)
    by_name = {mapping.standard_name: mapping for mapping in interface_mapping.ALL_MAPPINGS}
    for name in ("stomatal_conductance", "canopy_conductance"):
        assert by_name[name].units == scale[name]["units"], name
    assert by_name["stomatal_conductance"].units == "mol m-2 s-1"


def test_each_alias_names_exactly_one_catalog_entry():
    """An alias names one variable: it belongs to one entry, is not the standard name of any
    entry (itself included), and is not the name of an SCM contract variable. theta and
    soil_water_content were listed on both volume_fraction_of_water_in_soil and soil_moisture;
    soil_water_content is itself the contracted layered soil water input (m3 m-3) read by root
    nutrient uptake, solute advection-diffusion and Priestley-Taylor, and temperature is the
    contracted generic temperature, not air_temperature."""
    from collections import defaultdict

    from relic.contracts import load_contract_registry

    registry = load_contract_registry()
    names = {mapping.standard_name for mapping in interface_mapping.ALL_MAPPINGS}
    holders = defaultdict(list)
    problems = []
    for mapping in interface_mapping.ALL_MAPPINGS:
        for alias in mapping.aliases:
            holders[alias].append(mapping.standard_name)
            if alias in names:
                problems.append((mapping.standard_name, alias, "is a catalog entry"))
            if registry.get_contract(alias) is not None:
                problems.append((mapping.standard_name, alias, "is a contract variable"))
    problems += [(owners, alias, "is shared") for alias, owners in holders.items() if len(owners) > 1]
    assert problems == []
    by_name = {mapping.standard_name: mapping for mapping in interface_mapping.ALL_MAPPINGS}
    assert by_name["soil_moisture"].aliases == ["theta"]


# Catalog entries that named no variable of their own (MA-21 item 199): the liquid water
# density is the physical constant water_density (physical constants SSOT), not a hydrosphere
# variable; soil_moisture_change_rate is a key of the bucket diagnostics mapping, reported under
# diagnostics.soil_moisture_bucket (and by Richards as
# diagnostics.soil_moisture_richards.soil_moisture_change_rate); surface_water_flux is the flow
# label of the soil-water-transport surface port, the net volumetric boundary flux
# (infiltration_flux - evapotranspiration_flux) / rho_w [m s-1], not a state variable; and
# subsurface_drainage had no reader (the drained mass flux is drainage_flux).
NON_VARIABLE_ENTRIES = (
    "water_density_reference",
    "water.density_liquid",
    "density_liquid",
    "soil_moisture_change_rate",
    "surface_water_flux",
    "subsurface_drainage",
)


def test_catalog_lists_only_variables_not_constants_dict_keys_or_port_labels():
    from relic.contracts import load_contract_registry
    from relic.utils.physical_constants import get_physical_constant

    mapper = interface_mapping.hydrosphere_mapper
    names = mapper.get_all_names()["standard"]
    assert [name for name in NON_VARIABLE_ENTRIES if name in names] == []
    for mapping in interface_mapping.ALL_MAPPINGS:
        assert not set(mapping.aliases) & set(NON_VARIABLE_ENTRIES), mapping.standard_name
    assert float(get_physical_constant("water_density")) > 0.0
    # The diagnostics groups that carry the soil water content rate stay listed.
    for group in ("diagnostics.soil_moisture_bucket", "diagnostics.soil_moisture_richards"):
        assert mapper.get_category(group).value == "diagnostic", group
    # The canonical boundary fluxes remain the contracted variables.
    registry = load_contract_registry()
    for flux in ("infiltration_flux", "drainage_flux"):
        assert flux in names and registry.get_contract(flux).udunits == "kg m-2 s-1", flux


def test_soil_moisture_change_is_retired():
    """MA-21 item 206: the bare soil_moisture_change had no writer and no reader (the soil water
    models report the rate as diagnostics.soil_moisture_bucket.soil_moisture_change_rate and
    diagnostics.soil_moisture_richards.soil_moisture_change_rate), so its contract, catalog entries
    and entry of the runtime overlap policy (relic/core/simulator/output_overlap_whitelist.json, the
    file the simulator reads) are gone."""
    import json
    from pathlib import Path

    from relic.contracts import load_contract_registry
    from relic.parameters import get_global_manager

    registry = load_contract_registry()
    assert registry.get_contract("soil_moisture_change") is None
    assert not registry.is_alias("soil_moisture_change")
    assert "soil_moisture_change" not in interface_mapping.hydrosphere_mapper.get_all_names()["standard"]
    manager = get_global_manager()
    for scale in ("defaults", "daily"):
        catalog = manager.get(f"coupling.scales.{scale}.variables", expected_type=dict, required=True)
        assert "soil_moisture_change" not in catalog, scale
    from relic.core.simulator.output_overlap_governance import load_output_overlap_policies

    # The policies the simulator applies (loaded by their one owner) and the file they come from.
    assert "soil_moisture_change" not in load_output_overlap_policies()
    root = Path(__file__).resolve().parents[3]
    runtime_policy = root / "relic/core/simulator/output_overlap_whitelist.json"
    assert "soil_moisture_change" not in json.loads(runtime_policy.read_text(encoding="utf-8"))


def test_catalog_lists_no_physical_or_time_constant():
    """MA-21 item 208: time constants (seconds_per_day and the like) and physical constants come
    from the Time Constants SSOT (relic/utils/time_constants.py) and the physical constants SSOT
    (get_physical_constant); the hydrosphere catalog lists variables only."""
    import relic.utils.physical_constants as physical_constants
    import relic.utils.time_constants as time_constants

    constant_names = set(physical_constants.CONSTANT_MAPPINGS)
    constant_names |= {name.lower() for name in time_constants.get_time_constants()}
    assert "seconds_per_day" in constant_names
    names = set(interface_mapping.hydrosphere_mapper.get_all_names()["standard"])
    aliases = {alias for mapping in interface_mapping.ALL_MAPPINGS for alias in mapping.aliases}
    assert sorted((names | aliases) & constant_names) == []
    assert "sec_per_day" not in aliases


def test_supported_penman_construction_is_not_converted_to_skip():
    from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET
    from test.utils.functor_minimal_construction import minimal_arguments

    path = f"{PenmanMonteithET.__module__}:{PenmanMonteithET.__qualname__}"
    expected_mode = minimal_arguments(path)["config"]["canopy_resistance_source"]
    try:
        functor = _instantiate_functor(PenmanMonteithET)
    except pytest.skip.Exception as exc:
        pytest.fail(f"A supported documented construction must not skip coverage: {exc}")
    assert functor.canopy_resistance_source == expected_mode
    assert functor.metadata is not None


def test_unexpected_constructor_error_propagates_without_retry_or_skip(monkeypatch):
    from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import PenmanMonteithET

    calls = []
    def broken_constructor(self, *args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("injected supported constructor failure")
    monkeypatch.setattr(PenmanMonteithET, "__init__", broken_constructor)
    with pytest.raises(RuntimeError, match="injected supported constructor failure"):
        try:
            _instantiate_functor(PenmanMonteithET)
        except pytest.skip.Exception as exc:
            pytest.fail(f"An unexpected constructor error was hidden as a skip: {exc}")
    assert len(calls) == 1
