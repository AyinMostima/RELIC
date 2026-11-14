"""L1 tests for SCM contract loader."""

from pathlib import Path

import pytest

from relic.contracts.contract_loader import ContractRegistry
from relic.core.state.types import StateVariableType


def _write_contract(path: Path) -> Path:
    content = (
        "variable: biosphere.leaf_carbon\n"
        "title: \"Leaf Carbon\"\n"
        "kind: state\n"
        "dimension:\n"
        "  udunits: \"kg m-2\"\n"
        "  cf_standard_name: leaf_carbon\n"
        "owner: Allocation\n"
        "aliases: [C_leaf]\n"
        "incrementable: true\n"
        "accumulates: carbon.total\n"
        "writers:\n"
        "  increments:\n"
        "    - name: Photosynthesis\n"
        "    - name: Respiration\n"
        "  tendencies:\n"
        "    - name: SoilWaterFlux\n"
        "flux_contract:\n"
        "  allowed_in: [Photosynthesis]\n"
        "  allowed_out: [Respiration]\n"
        "numerics:\n"
        "  atol: 1.0e-12\n"
        "  rtol: 1.0e-9\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def test_contract_registry_basic(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path / "leaf_carbon.yaml")
    registry = ContractRegistry(strict=False)
    registry.load_contract_file(contract_path)

    assert registry.get_owner("biosphere.leaf_carbon") == "Allocation"
    assert registry.is_alias("C_leaf")
    assert registry.resolve_alias("C_leaf") == "biosphere.leaf_carbon"

    assert registry.can_write("biosphere.leaf_carbon", "Allocation")
    assert registry.can_write("biosphere.leaf_carbon", "Photosynthesis")
    assert registry.can_write("biosphere.leaf_carbon", "SoilWaterFlux")
    assert not registry.can_write("biosphere.leaf_carbon", "OtherFunctor")
    assert registry.can_increment("biosphere.leaf_carbon", "SoilWaterFlux")

    whitelist = registry.get_flux_whitelist("biosphere.leaf_carbon")
    assert whitelist["flux_in"] == ["Photosynthesis"]
    assert whitelist["flux_out"] == ["Respiration"]

    members = registry.get_domain_members("carbon.total")
    assert members.count("biosphere.leaf_carbon") == 1

    spec = registry.to_state_variable_spec("biosphere.leaf_carbon")
    assert spec is not None
    assert spec.conserved is True
    assert spec.accumulates == "carbon.total"


def test_declared_external_forcing_rejects_functor_writers(tmp_path: Path) -> None:
    """A declared forcing with no writers is closed to scientific Functors."""

    contract_path = tmp_path / "external_forcing.yaml"
    contract_path.write_text(
        "variable: air_temperature\n"
        "title: Air Temperature\n"
        "kind: forcing\n"
        "dimension:\n"
        "  udunits: K\n"
        "owner: null\n"
        "aliases: []\n"
        "incrementable: false\n"
        "writers:\n"
        "  increments: []\n"
        "  tendencies: []\n",
        encoding="utf-8",
    )
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(contract_path)

    assert not registry.can_write("air_temperature", "environmental_forcing")
    assert registry.can_write("uncontracted_runtime_value", "test_writer")
    spec = registry.to_state_variable_spec("air_temperature")
    assert spec is not None
    assert spec.incrementable is False


def test_duplicate_variable_is_recorded_without_overwriting_first_contract(
    tmp_path: Path,
) -> None:
    first = _write_contract(tmp_path / "first.yaml")
    second = tmp_path / "second.yaml"
    second.write_text(
        first.read_text(encoding="utf-8").replace(
            "owner: Allocation",
            "owner: OtherAllocation",
        ),
        encoding="utf-8",
    )
    registry = ContractRegistry(strict=False)

    registry.load_contract_file(first)
    registry.load_contract_file(second)

    assert registry.get_owner("biosphere.leaf_carbon") == "Allocation"
    assert len(registry.conflicts) == 1
    assert registry.conflicts[0].startswith(
        "duplicate_variable:biosphere.leaf_carbon:"
    )


def _write_layer_contract(path: Path, layer_line: str) -> Path:
    path.write_text(
        "variable: soil_moisture\n"
        "title: Soil Moisture\n"
        "kind: state\n"
        "dimension:\n"
        "  udunits: \"m3 m-3\"\n"
        f"{layer_line}"
        "owner: null\n"
        "aliases: []\n"
        "incrementable: false\n",
        encoding="utf-8",
    )
    return path


def test_layer_thickness_variable_is_parsed_from_the_contract(tmp_path: Path) -> None:
    """The one optional layer-thickness field names the thickness state variable."""
    declared = _write_layer_contract(
        tmp_path / "declared.yaml", "layer_thickness_variable: soil_depth\n"
    )
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(declared)
    assert registry.get_contract("soil_moisture").layer_thickness_variable == "soil_depth"

    undeclared = _write_layer_contract(tmp_path / "undeclared.yaml", "")
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(undeclared)
    assert registry.get_contract("soil_moisture").layer_thickness_variable is None


@pytest.mark.parametrize(
    "layer_line",
    ["layer_thickness_variable: 3\n", "layer_thickness_variable: \"\"\n"],
)
def test_invalid_layer_thickness_variable_fails_fast(tmp_path: Path, layer_line: str) -> None:
    contract_path = _write_layer_contract(tmp_path / "invalid.yaml", layer_line)
    with pytest.raises(ValueError, match="layer_thickness_variable"):
        ContractRegistry(strict=False).load_contract_file(contract_path)


def _write_cf_contract(path: Path, variable: str, dimension_lines: str) -> Path:
    path.write_text(
        f"variable: {variable}\n"
        "kind: forcing\n"
        "dimension:\n"
        "  udunits: \"K\"\n"
        f"{dimension_lines}"
        "owner: null\n"
        "aliases: []\n"
        "incrementable: false\n",
        encoding="utf-8",
    )
    return path


def test_cell_methods_is_parsed_from_the_contract_dimension(tmp_path: Path) -> None:
    """CF-1.8 section 7.3: cell_methods qualify the CF identity declared by a contract."""
    declared = _write_cf_contract(
        tmp_path / "declared.yaml",
        "tasmax",
        "  cf_standard_name: air_temperature\n  cell_methods: \"time: maximum\"\n",
    )
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(declared)
    assert registry.get_contract("tasmax").cell_methods == "time: maximum"

    undeclared = _write_cf_contract(
        tmp_path / "undeclared.yaml", "tas", "  cf_standard_name: air_temperature\n"
    )
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(undeclared)
    assert registry.get_contract("tas").cell_methods is None


@pytest.mark.parametrize("cell_methods_line", ["  cell_methods: 3\n", "  cell_methods: \"\"\n"])
def test_invalid_cell_methods_fails_fast(tmp_path: Path, cell_methods_line: str) -> None:
    contract_path = _write_cf_contract(
        tmp_path / "invalid.yaml",
        "tasmax",
        "  cf_standard_name: air_temperature\n" + cell_methods_line,
    )
    with pytest.raises(ValueError, match=r"cell_methods|is not of type"):
        ContractRegistry(strict=False).load_contract_file(contract_path)


def test_repository_soil_moisture_contract_declares_its_layer_thickness() -> None:
    """The column soil-moisture state describes the effective soil layer of
    thickness ``soil_depth`` [m]; the transport bridge reads it from here."""
    contracts_dir = Path(__file__).resolve().parents[3] / "relic" / "contracts" / "variables"
    registry = ContractRegistry(strict=True)
    registry.load_contracts_dir(contracts_dir)
    assert registry.get_contract("soil_moisture").layer_thickness_variable == "soil_depth"


@pytest.mark.parametrize(
    "kind,storage_role",
    [
        ("state", StateVariableType.GRID),
        ("diagnostic", StateVariableType.GRID),
        ("flux", StateVariableType.GRID),
        ("forcing", StateVariableType.GRID),
        ("parameter", StateVariableType.PARAMETER),
    ],
)
def test_contract_kind_projects_to_spec_kind_and_storage_role(
    tmp_path: Path, kind: str, storage_role: StateVariableType
) -> None:
    """The contract kind is carried as the spec kind; the storage role is not
    derived from it (no extensive/intensive reading of prognostic/diagnostic)."""
    contract_path = tmp_path / f"{kind}.yaml"
    contract_path.write_text(
        "variable: soil_temperature\n"
        f"kind: {kind}\n"
        "dimension:\n"
        "  udunits: K\n"
        "owner: null\n"
        "aliases: []\n",
        encoding="utf-8",
    )
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(contract_path)
    spec = registry.to_state_variable_spec("soil_temperature")
    assert spec.kind == kind
    assert spec.variable_type is storage_role


def test_duplicate_variable_fails_fast_in_strict_mode(tmp_path: Path) -> None:
    first = _write_contract(tmp_path / "first.yaml")
    second = _write_contract(tmp_path / "second.yaml")
    registry = ContractRegistry(strict=True)

    registry.load_contract_file(first)
    with pytest.raises(ValueError, match="Duplicate SCM variable contract"):
        registry.load_contract_file(second)


# --- Single owner of reading contract directories (D33) ---------------------

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_state_contract(directory: Path, name: str, units: str = "K") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.yaml").write_text(
        f"variable: {name}\nkind: state\ndimension:\n  udunits: {units}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "numerics,expected",
    [
        ("  lower_bound: 0.0\n", (0.0, float("inf"))),
        ("  upper_bound: 0.0\n", (float("-inf"), 0.0)),
        ("  lower_bound: -1.0\n  upper_bound: 2.0\n", (-1.0, 2.0)),
        ("  atol: 1.0e-12\n", None),
    ],
)
def test_contract_bounds_project_exactly_as_declared(
    tmp_path: Path, numerics: str, expected
) -> None:
    """lower only -> [lb, +inf), upper only -> (-inf, ub], none -> not bounded."""
    contract_path = tmp_path / "bounded_state.yaml"
    contract_path.write_text(
        "variable: bounded_state\nkind: state\ndimension:\n  udunits: kg m-2\n"
        f"numerics:\n{numerics}",
        encoding="utf-8",
    )
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(contract_path)
    assert registry.to_state_variable_spec("bounded_state").bounds == expected


@pytest.mark.parametrize(
    "variable,lower,upper",
    [
        # CF surface_upward_latent_heat_flux / water_evapotranspiration_flux are
        # upward-positive and negative for dew or condensation.
        ("latent_heat_flux", None, None),
        ("evapotranspiration_flux", None, None),
        # Budget and audit residuals carry either sign.
        ("diagnostics.surface_ponding.water_balance_residual", None, None),
        ("diagnostics.topology_transition.ledger_residual", None, None),
        # Declared ledger change Pi = -h <= 0 (water leaving by infiltration).
        ("diagnostics.topology_transition.ledger_change", None, 0.0),
        # Matric potential is a suction (non-positive).
        ("soil_matric_potential", None, 0.0),
        # Storages and directional magnitudes stay non-negative.
        ("surface_runoff_flux", 0.0, None),
        # Heterotrophic respiration is a release to the atmosphere (CF
        # surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_
        # heterotrophic_respiration, upward positive): its writers raise on a
        # negative rate and the carbon.total boundary ledger takes magnitudes.
        ("heterotrophic_respiration", 0.0, None),
        ("soil_water_storage", 0.0, None),
        # The surface sensible heat flux of the Obukhov length is signed
        # (CF surface_upward_sensible_heat_flux, upward positive).
        ("sensible_heat_flux", None, None),
        # Absolute temperature, gross uptake, photon flux density, vapour
        # pressure deficit and wind speed magnitude are non-negative.
        ("temperature", 0.0, None),
        ("gpp", 0.0, None),
        ("par", 0.0, None),
        ("wind_speed", 0.0, None),
        # Atmospheric transport: the tracer tendencies and the stability
        # parameter are signed; the Courant and diffusion stability numbers
        # and the eddy diffusivity are non-negative.
        ("diagnostics.advection_rate", None, None),
        ("diagnostics.diffusion_rate", None, None),
        ("diagnostics.stability_parameter", None, None),
        ("diagnostics.cfl_number", 0.0, None),
        ("diagnostics.stability_number", 0.0, None),
        ("diagnostics.eddy_diffusivity", 0.0, None),
    ],
)
def test_packaged_contract_bounds_follow_the_declared_sign_convention(
    variable: str, lower, upper
) -> None:
    from relic.contracts.contract_loader import load_contract_registry

    numerics = load_contract_registry().get_contract(variable).raw.get("numerics", {})
    assert numerics.get("lower_bound") == lower
    assert numerics.get("upper_bound") == upper


@pytest.mark.parametrize(
    "variable,units",
    [
        # Read as a photon flux density by the carbon-13 light correction
        # (light_par_saturation [umol photons m-2 s-1]) and the BVOC light
        # response (PPFD); written from ppfd by the tutorial forcing loader.
        ("par", "umol photons m-2 s-1"),
        # Volumetric water content for every reader (root nutrient uptake,
        # solute advection-diffusion, Priestley-Taylor, vertical percolation).
        ("soil_water_content", "m3 m-3"),
        # Kelvin for every reader of the generic name (atmospheric
        # advection-diffusion Obukhov length, fire and tillage management,
        # sesquiterpene synthesis, thermal stress after air_temperature).
        ("temperature", "K"),
        # H of the Obukhov length L = -rho cp T u*^3 / (kappa g H).
        ("sensible_heat_flux", "W m-2"),
        # Horizontal wind magnitude read by turbulence and Penman-Monteith.
        ("wind_speed", "m s-1"),
        # Atmospheric tracer tendencies of tracer_concentration [kg m-3] per
        # second, the K-theory eddy diffusivity and dimensionless numbers.
        ("diagnostics.advection_rate", "kg m-3 s-1"),
        ("diagnostics.diffusion_rate", "kg m-3 s-1"),
        ("diagnostics.eddy_diffusivity", "m2 s-1"),
        ("diagnostics.cfl_number", "1"),
        ("diagnostics.stability_number", "1"),
        ("diagnostics.stability_parameter", "1"),
    ],
)
def test_packaged_contract_units_are_the_units_their_readers_use(
    variable: str, units: str
) -> None:
    from relic.contracts.contract_loader import load_contract_registry

    assert load_contract_registry().get_contract(variable).udunits == units


@pytest.mark.parametrize(
    "variable,cf_standard_name",
    [
        # H of the Obukhov length: upward-positive surface sensible heat flux.
        ("sensible_heat_flux", "surface_upward_sensible_heat_flux"),
        # D = e_s(T_a) - e_a in Pa, the CF canonical unit.
        ("vapor_pressure_deficit", "water_vapor_saturation_deficit_in_air"),
    ],
)
def test_packaged_contract_cf_names_are_exact_for_quantity_and_units(
    variable: str, cf_standard_name: str
) -> None:
    from relic.data.standards.cf_conventions import CFConventions

    contract = _packaged_registry().get_contract(variable)

    assert contract.cf_standard_name == cf_standard_name
    assert CFConventions.are_standard_name_units_equivalent(cf_standard_name, contract.udunits)


@pytest.mark.parametrize(
    "cf_standard_name,variable",
    [
        # The atmospheric_turbulence output read by atmospheric transport; the
        # reader-side duplicate surface_heat_flux is removed.
        ("surface_upward_sensible_heat_flux", "sensible_heat_flux"),
        # The Pa variable written by environmental_forcing; the kPa companion vpd
        # read by the carbon-13 correction declares no CF name.
        ("water_vapor_saturation_deficit_in_air", "vapor_pressure_deficit"),
        # The atmospheric chemistry ozone state read by OzoneStress; the former
        # forcing name ozone_concentration is its contract alias.
        ("mole_fraction_of_ozone_in_air", "o3_mixing_ratio"),
    ],
)
def test_one_contract_holds_each_cf_identity(cf_standard_name: str, variable: str) -> None:
    """A CF identity without qualifying cell_methods belongs to one RELIC variable."""
    registry = _packaged_registry()

    holders = sorted(
        name for name in registry.list_variables()
        if registry.get_contract(name).cf_standard_name == cf_standard_name
    )
    assert holders == [variable]


# Placeholder contracts removed by the audit of the anthroposphere, atmosphere
# and transport domains: nothing in relic reads or writes these names as state
# (FunctorMetadata inputs and outputs, UpdateItem keys, config readers,
# contract aliases). A quantity that exists is carried under another name, for
# example diagnostics.energy_conservation_report,
# diagnostics.chemistry.photolysis_rate_no2, relative_humidity (for humidity),
# the UnitSystem energy-flux type
# radiation [W m-2] and diagnostics.land_use_transitions.dominant_type_index.
_AUDIT_REMOVED_PLACEHOLDERS = (
    # anthroposphere
    "anthropogenic_bc_flux", "anthropogenic_ch4_flux", "anthropogenic_co2_flux",
    "anthropogenic_emission_flux", "anthropogenic_n2o_flux", "anthropogenic_nmvoc_flux",
    "anthropogenic_oc_flux", "anthropogenic_pm10_flux", "cf_metadata",
    "delta_land_use_fraction", "fertilizer_k_flux", "hydro_params",
    "land_use_type_index", "management_schedule", "mass_conservation",
    "mechanism_species", "mixing_ratio_changes", "molecular_conversions",
    "parameter_modifiers", "resource_allocation", "sector_contributions",
    "soil_params", "speciated_emissions", "speciation_summary",
    "vegetation_params", "voc_speciation", "volatilization_flux",
    # atmosphere
    "chemistry_diagnostics", "delta_isoprene_mixing_ratio", "delta_no", "delta_no2",
    "delta_no2_mixing_ratio", "delta_no_mixing_ratio", "delta_o3",
    "delta_o3_mixing_ratio", "delta_so2_mixing_ratio", "deposition_diagnostics",
    "energy_conservation_report", "humidity", "photolysis_rate_no2", "radiation",
    "transport_diagnostics",
)


@pytest.mark.parametrize("variable", _AUDIT_REMOVED_PLACEHOLDERS)
def test_placeholder_contracts_removed_by_the_audit_stay_absent(variable: str) -> None:
    registry = _packaged_registry()

    assert registry.get_contract(variable) is None
    assert not registry.is_alias(variable)


def test_load_contracts_dir_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="absent"):
        ContractRegistry(strict=True).load_contracts_dir(tmp_path / "absent")


def test_packaged_contracts_are_loaded_once_per_process_path() -> None:
    from relic.contracts.contract_loader import (
        PACKAGED_CONTRACTS_DIR,
        load_contract_registry,
    )

    first = load_contract_registry()
    assert first is load_contract_registry()
    assert first is load_contract_registry(PACKAGED_CONTRACTS_DIR)
    assert first.get_contract("soil_moisture").layer_thickness_variable == "soil_depth"


def test_contract_directory_loader_is_strict(tmp_path: Path) -> None:
    from relic.contracts.contract_loader import load_contract_registry

    with pytest.raises(FileNotFoundError, match="absent"):
        load_contract_registry(tmp_path / "absent")

    broken = tmp_path / "broken"
    _write_state_contract(broken, "good_state")
    (broken / "broken_variable.yaml").write_text(
        "variable: broken_variable\nkind: [unclosed\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="broken_variable.yaml"):
        load_contract_registry(broken)


def test_contract_consumers_share_the_single_loader(tmp_path: Path, monkeypatch) -> None:
    """StateManager, the transport bridge, UnitSystem and SSOTCatalog read the
    packaged contracts only through load_contract_registry: a relocated
    packaged directory reaches all of them, and a missing one raises in each."""
    import relic.contracts.contract_loader as contract_loader
    from relic.core.functor import FunctorMetadata, ProcessCategory, TransportMechanismFunctor
    from relic.core.state import StateManager
    from relic.core.units import UnitSystem
    from relic.orchestration.ssot_catalog import SSOTCatalog

    relocated = tmp_path / "variables"
    _write_state_contract(relocated, "relocated_state", units="kg m-2")
    monkeypatch.setattr(contract_loader, "PACKAGED_CONTRACTS_DIR", relocated)
    monkeypatch.setattr(UnitSystem, "_contract_unit_map", None)
    monkeypatch.setattr(UnitSystem, "_contract_unit_map_lower", None)

    class _Transport(TransportMechanismFunctor):
        TRANSPORT_TYPE = "diffusion"

        def forward(self, state, time, dt, **kwargs):
            return {}

    transport = _Transport(
        name="loader_probe",
        metadata=FunctorMetadata(
            name="loader_probe",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            description="Contract loader probe",
            references=["RELIC contract loader test"],
            output_variables=["relocated_state"],
        ),
    )
    shared = contract_loader.load_contract_registry()
    assert StateManager()._contract_registry is shared
    assert transport._declared_target_rate_units("relocated_state", None)[1] == "kg m-2"
    assert UnitSystem.get_contract_unit("relocated_state") == "kg m-2"

    monkeypatch.setattr(contract_loader, "PACKAGED_CONTRACTS_DIR", tmp_path / "absent")
    monkeypatch.setattr(UnitSystem, "_contract_unit_map", None)
    monkeypatch.setattr(UnitSystem, "_contract_unit_map_lower", None)
    with pytest.raises(FileNotFoundError):
        UnitSystem.get_contract_unit("relocated_state")
    with pytest.raises(FileNotFoundError):
        SSOTCatalog()._load_contract_entries(tmp_path / "absent")


def test_no_module_reads_a_contract_directory_outside_the_loader() -> None:
    """Only relic/contracts/contract_loader.py may call load_contracts_dir."""
    offenders = []
    for path in sorted((REPO_ROOT / "relic").rglob("*.py")):
        if path.name == "contract_loader.py" and path.parent.name == "contracts":
            continue
        if "load_contracts_dir(" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


_CF_NAMED_CONTRACTS = (
    "aerodynamic_resistance",
    "leaf_area_index",
    "surface_runoff_flux",
    "subsurface_runoff_flux",
    "water_table_depth",
    "wind_speed",
)


def _packaged_registry():
    from relic.contracts.contract_loader import PACKAGED_CONTRACTS_DIR, load_contract_registry

    return load_contract_registry(PACKAGED_CONTRACTS_DIR)


@pytest.mark.parametrize("variable", _CF_NAMED_CONTRACTS)
def test_contracts_named_like_their_cf_quantity_declare_it(variable: str) -> None:
    from relic.data.standards.cf_conventions import CFConventions

    contract = _packaged_registry().get_contract(variable)

    assert contract.cf_standard_name == variable
    units = "1" if contract.udunits == "dimensionless" else contract.udunits
    assert CFConventions.are_standard_name_units_equivalent(variable, units)


def test_ozone_concentration_is_a_contract_alias_of_o3_mixing_ratio() -> None:
    """One ozone mole fraction: CF v94 mole_fraction_of_ozone_in_air (unit 1 = mol mol-1)
    is held by o3_mixing_ratio, and the former OzoneStress forcing name is its alias."""
    from relic.data.standards.cf_conventions import CFConventions

    registry = _packaged_registry()
    contract = registry.get_contract("o3_mixing_ratio")

    assert CFConventions.is_valid_standard_name("mole_fraction_of_ozone_in_air")
    assert CFConventions.are_standard_name_units_equivalent(
        "mole_fraction_of_ozone_in_air", contract.udunits
    )
    assert contract.cf_standard_name == "mole_fraction_of_ozone_in_air"
    assert registry.get_contract("ozone_concentration") is None
    assert registry.resolve_alias("ozone_concentration") == "o3_mixing_ratio"
    assert "ozone_concentration" in contract.raw["description"]


def test_soil_water_volume_fraction_records_why_it_has_no_cf_name() -> None:
    """soil_moisture is the one volumetric liquid soil water content (the sorption forcing
    volume_fraction_of_water_in_soil was the same quantity and is merged into it)."""
    registry = _packaged_registry()
    assert registry.get_contract("volume_fraction_of_water_in_soil") is None
    assert not registry.is_alias("volume_fraction_of_water_in_soil")
    contract = registry.get_contract("soil_moisture")

    assert contract.cf_standard_name is None
    assert contract.udunits == "m3 m-3"
    description = contract.raw["description"]
    assert "volume_fraction_of_condensed_water_in_soil" in description
    assert "ice" in description


# Writers of configuration-dependent outputs (not produced by the default instance):
# atmospheric_radiation writes fpar/apar_layer only with the canopy layer enabled,
# atmospheric_chemistry writes the closed-box states only with the closed_constant_volume
# boundary. Graph adapters, observation operators, coupling Functors, stochastic
# Functors requiring a seed and topology events are not default-constructible
# scientific Functors and are not audited here.
_CONFIGURATION_DEPENDENT_WRITERS = {"atmospheric_radiation", "atmospheric_chemistry"}
# Outputs that a writer declares only under a configuration option: MEGAN writes
# the T24/T240 temperature-history states with temperature_history (B4).
_CONFIGURATION_DEPENDENT_OUTPUTS = {
    ("air_temperature_24h_mean", "megan_bvoc_model"),
    ("air_temperature_240h_mean", "megan_bvoc_model"),
    # With temperature_source leaf_temperature the history states are the leaf means.
    ("leaf_temperature_24h_mean", "megan_bvoc_model"),
    ("leaf_temperature_240h_mean", "megan_bvoc_model"),
    # Per-sample ports of temporal_support daylight_gauss (absent from the default supports).
    ("net_photosynthesis_daylight_samples", "farquhar_photosynthesis"),
    ("stomatal_conductance_daylight_samples", "medlyn_stomatal_conductance"),
    ("canopy_conductance_daylight_samples", "medlyn_stomatal_conductance"),
    ("leaf_temperature_daylight_samples", "leaf_energy_balance"),
    ("ozone_stress_factor_daylight_samples", "ozone_stress"),
}


def _default_scientific_writers():
    import importlib
    import warnings

    from relic.scientific import FUNCTOR_REGISTRY
    from relic.scientific.isotopes.functor import IsotopeHeartbeatError

    outputs = {}
    for path in sorted(set(FUNCTOR_REGISTRY.values())):
        module_name, class_name = path.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                instance = cls()
            except (TypeError, ValueError, RuntimeError, KeyError, IsotopeHeartbeatError):
                continue
        outputs.setdefault(instance.name, set()).update(instance.metadata.output_variables or ())
    return outputs


def test_contract_writers_match_the_outputs_of_default_scientific_functors() -> None:
    registry = _packaged_registry()
    outputs = _default_scientific_writers()
    stale, missing = [], []
    for variable in registry.list_variables():
        contract = registry.get_contract(variable)
        declared = set(contract.writers or ())
        if contract.owner:
            declared.add(contract.owner)
        actual = {name for name, produced in outputs.items() if variable in produced}
        stale.extend(
            (variable, writer)
            for writer in declared - actual
            if writer in outputs and writer not in _CONFIGURATION_DEPENDENT_WRITERS
            and (variable, writer) not in _CONFIGURATION_DEPENDENT_OUTPUTS
        )
        missing.extend((variable, writer) for writer in actual - declared)
    assert sorted(stale) == []
    assert sorted(missing) == []


def test_corrected_writer_lists() -> None:
    registry = _packaged_registry()

    # External adapters that map a field to a variable are its alternative writers.
    assert registry.get_contract("subsurface_runoff_flux").writers == [
        "subsurface_runoff", "SWATAdapter", "VICAdapter"
    ]
    for variable in ("total_bvoc_carbon_flux", "bvoc_monoterpene_carbon_loss",
                     "soil_water_storage"):
        contract = registry.get_contract(variable)
        assert contract.writers == [] and contract.owner is None, variable
    for variable in ("rain_flux", "snow_depth", "snow_melt_flux", "snow_water_equivalent",
                     "diagnostics.snow_pack"):
        assert registry.get_contract(variable).writers == ["snow_pack"], variable
    assert registry.get_contract("snowfall").writers == ["snow_pack", "CMIP6Adapter"]
    assert registry.get_contract("diagnostics.snow_albedo").writers == ["snow_albedo"]
    assert registry.get_contract("stress_response_factor").writers == ["sesquiterpene_stress_model"]
    # One name per Courant-number definition: the finite-volume outward-face
    # sum of the generic transport Functors, and the cell-centred Courant sum
    # of atmospheric transport under its own name.
    cfl = registry.get_contract("diagnostics.cfl_number")
    assert cfl.owner is None
    assert set(cfl.writers) == {"advection_diffusion", "linear_advection"}
    cell_courant = registry.get_contract("diagnostics.atmospheric_advection_diffusion.cfl_number")
    assert cell_courant.owner == "atmospheric_advection_diffusion"
    assert cell_courant.writers == ["atmospheric_advection_diffusion"]
    assert cell_courant.udunits == "1" and cell_courant.kind == "diagnostic"
    assert registry.get_contract("diagnostics.peclet_number").writers == ["advection_diffusion"]


def test_soil_depth_contract_names_its_state_readers() -> None:
    """The soil_depth state variable is read by root nutrient uptake, nitrate
    leaching and the pedosphere interface helpers. The bucket model reads the
    parameter hydrosphere.soil_moisture.soil_depth, not this state variable."""
    from relic.contracts.contract_loader import load_contract_registry

    references = " ".join(load_contract_registry().get_contract("soil_depth").raw["references"])
    assert "bucket.py" not in references
    for module in ("root_nutrient_uptake.py", "nitrogen_leaching.py", "pedosphere/interface_mapping.py"):
        assert module in references, module
