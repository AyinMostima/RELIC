import re
from pathlib import Path

import pytest
import yaml

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.contracts.contract_loader import ContractRegistry, load_contract_registry
from relic.core.functor import ScaleCouplingFunctor
from relic.parameters import get_global_manager
from relic.scientific.anthroposphere import inventory as inventory_module
from relic.scientific.anthroposphere.inventory import EmissionInventory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BLOCKER = (
    "[N] runtime_domain_unbound: global HTAPv3.2/EDGAR annual totals "
    "lack an executable sector/species target-domain fraction or verified "
    "global target-grid binding"
)
DEAD_INVENTORY_PARAMETER_KEYS = (
    "anthroposphere.inventory.base_year",
    "anthroposphere.inventory.default_emission_factor",
    "anthroposphere.inventory.gdp_weight",
    "anthroposphere.inventory.grid_spacing_default",
    "anthroposphere.inventory.landuse_weight",
    "anthroposphere.inventory.lat_points_default",
    "anthroposphere.inventory.lon_points_default",
    "anthroposphere.inventory.population_weight",
    "anthroposphere.inventory.quality_control.mass_balance_tolerance",
    "anthroposphere.inventory.quality_control.max_emission_threshold",
    "anthroposphere.inventory.quality_control.min_emission_threshold",
    "anthroposphere.inventory.time_steps_per_year",
    "anthroposphere.inventory.quality_control.min_records_per_period",
    "anthroposphere.inventory.quality_control.max_missing_fraction",
    "anthroposphere.inventory.quality_control.max_outlier_fraction",
)
DEAD_EMISSIONS_INVENTORY_PARAMETER_KEYS = (
    "anthroposphere.emissions.inventory_database.scaling_factor",
    "anthroposphere.emissions.inventory_database.uncertainty_factor",
    "anthroposphere.emissions.inventory_database.power_sector.emissions.co2_annual",
    "anthroposphere.emissions.inventory_database.power_sector.nox_total",
    "anthroposphere.emissions.inventory_database.power_sector.so2_total",
    "anthroposphere.emissions.inventory_database.power_sector.voc_total",
    "anthroposphere.emissions.inventory_database.power_sector.pm25_total",
    "anthroposphere.emissions.inventory_database.transport_sector.co2_total",
    "anthroposphere.emissions.inventory_database.transport_sector.nox_total",
    "anthroposphere.emissions.inventory_database.transport_sector.co_total",
    "anthroposphere.emissions.inventory_database.transport_sector.voc_total",
    "anthroposphere.emissions.inventory_database.transport_sector.pm25_total",
    "anthroposphere.emissions.inventory_database.industry_sector.co2_total",
    "anthroposphere.emissions.inventory_database.industry_sector.so2_total",
    "anthroposphere.emissions.inventory_database.industry_sector.voc_total",
    "anthroposphere.emissions.inventory_database.industry_sector.pm25_total",
    "anthroposphere.emissions.inventory_database.buildings_sector.co2_total",
    "anthroposphere.emissions.inventory_database.buildings_sector.ch4_total",
    "anthroposphere.emissions.inventory_database.buildings_sector.voc_total",
    "anthroposphere.emissions.inventory_database.buildings_sector.pm25_total",
    "anthroposphere.emissions.inventory_database.agriculture_sector.ch4_total",
    "anthroposphere.emissions.inventory_database.agriculture_sector.n2o_total",
    "anthroposphere.emissions.inventory_database.agriculture_sector.nh3_total",
    "anthroposphere.emissions.inventory_database.agriculture_sector.voc_total",
    "anthroposphere.emissions.inventory_database.agriculture_sector.pm25_total",
    "anthroposphere.emissions.inventory_database.waste_sector.voc_total",
    "anthroposphere.emissions.inventory_database.waste_sector.pm25_total",
    "anthroposphere.emissions.base_year",
    "anthroposphere.emissions.inventory.spatial_resolution",
    "anthroposphere.emissions.spatial_allocation.population_decay_factor",
    "anthroposphere.emissions.spatial_allocation.economic_decay_factor",
    "anthroposphere.emissions.spatial_allocation.power_plant_weight",
    "anthroposphere.emissions.spatial_allocation.industrial_point_weight",
    "anthroposphere.emissions.temporal_profiles.power_demand.hourly_factors",
    "anthroposphere.emissions.temporal_profiles.traffic.hourly_factors",
    "anthroposphere.emissions.temporal_profiles.industrial.hourly_factors",
    "anthroposphere.emissions.temporal_profiles.residential.hourly_factors",
    "anthroposphere.emissions.temporal_profiles.agricultural.hourly_factors",
    "anthroposphere.emissions.temporal_profiles.flat.hourly_factors",
)
DEAD_ADJACENT_INVENTORY_PARAMETER_KEYS = (
    "anthroposphere.emissions.inventory.base_year",
    "anthroposphere.emissions.temporal_profiles.smoothing_factor",
    "anthroposphere.emissions.spatial_allocation.population_exponent_scale",
    "anthroposphere.emissions.spatial_allocation.gdp_decay_factor",
    "anthroposphere.emissions.spatial_allocation.road_side_weight",
    "anthroposphere.emissions.spatial_allocation.mixed_industrial_weight",
)


class _ExplosiveState(dict):
    def __contains__(self, key):
        raise AssertionError(f"state membership was accessed: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"state item was accessed: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"state get was accessed: {key}")

    def items(self):
        raise AssertionError("state items were accessed")


def test_emission_inventory_remains_the_scale_coupling_identity():
    assert issubclass(EmissionInventory, ScaleCouplingFunctor)
    assert inventory_module.__all__ == ["EmissionInventory"]


def test_constructor_rejects_before_functor_or_parameter_effects(monkeypatch):
    base_calls = []

    def _reject_base_init(*args, **kwargs):
        base_calls.append((args, kwargs))
        raise AssertionError("FunctorBase initialization was reached")

    monkeypatch.setattr(ScaleCouplingFunctor, "__init__", _reject_base_init)

    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        EmissionInventory(name="forbidden", config={"calendar_type": "noleap"})

    assert base_calls == []


@pytest.mark.parametrize(
    "args,kwargs",
    [
        pytest.param((), {}, id="default"),
        pytest.param(({"calendar_type": "360_day"},), {}, id="dict-name"),
        pytest.param((), {"name": "inventory", "config": {}}, id="keywords"),
    ],
)
def test_constructor_has_one_deterministic_runtime_blocker(args, kwargs):
    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        EmissionInventory(*args, **kwargs)


def test_unsafe_constructor_bypass_cannot_reach_state_or_emit_updates():
    inventory = object.__new__(EmissionInventory)
    state = _ExplosiveState()

    with pytest.raises(RuntimeError, match=re.escape(RUNTIME_BLOCKER)):
        EmissionInventory.forward(
            inventory,
            state,
            time=0.0,
            dt=3600.0,
            calendar_type="noleap",
        )

    assert dict(state) == {}
    assert not hasattr(inventory, "_parameters")
    assert not hasattr(inventory, "spatial_proxies")
    assert not hasattr(inventory, "cumulative_emissions")
    assert not hasattr(inventory, "inventory_records")


def test_inventory_exposes_no_formula_facade_or_false_structure():
    forbidden_class_api = {
        "compute",
        "declared_forms",
        "build_hodge_contract",
        "build_hodge_terms",
        "energy_terms",
        "entropy_terms",
        "reversible_structure",
        "dissipative_structure",
        "ports",
        "update_emission_data",
        "get_extra_state",
        "set_extra_state",
        "PHASE8_RUNTIME_REPLACEMENT_STATUS",
    }
    forbidden_module_api = {
        "AnthroposphereInventory",
        "EmissionData",
        "EmissionSpecies",
        "EmissionSector",
        "SpatialProxy",
        "InventoryRecord",
        "AggregatedInventory",
        "AggregationPeriod",
        "InventoryCategory",
        "ReportingStandard",
    }

    assert forbidden_class_api.isdisjoint(EmissionInventory.__dict__)
    assert not [name for name in forbidden_module_api if hasattr(inventory_module, name)]


def test_inventory_source_contains_only_the_fail_closed_boundary():
    source = (
        PROJECT_ROOT / "relic" / "scientific" / "anthroposphere" / "inventory.py"
    ).read_text(encoding="utf-8")
    forbidden_tokens = (
        "CochainSpaceSpec",
        "HodgeStarSpec",
        "PortSpecIPHS",
        "canonical_iphs_runtime",
        "runtime_path",
        "seconds_per_year",
        "_initialize_default_proxies",
        "_resize_proxy",
        "F.interpolate",
        "cumulative_emissions",
        "inventory_records",
        "emission_database",
        "UpdateItem(",
    )

    assert "[N] runtime_domain_unbound" in source
    assert "sector/species target-domain fraction" in source
    for token in forbidden_tokens:
        assert token not in source


def _assert_unbound_inventory_contract(variable: str) -> dict:
    path = (
        PROJECT_ROOT
        / "relic"
        / "contracts"
        / "variables"
        / "anthroposphere"
        / f"{variable}.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["owner"] is None
    assert payload["writers"] == {"increments": [], "tendencies": []}
    assert payload["flux_contract"]["allowed_in"] == []
    assert payload["provenance"]["runtime_writer"] is None
    assert payload["provenance"]["runtime_role"] == "runtime_domain_unbound"
    return payload


def test_anthropogenic_nox_contract_has_no_unbound_inventory_writer():
    payload = _assert_unbound_inventory_contract("anthropogenic_nox_flux")
    assert payload["dimension"]["cf_standard_name"] is None


def test_anthropogenic_co_contract_has_no_unbound_inventory_writer():
    _assert_unbound_inventory_contract("anthropogenic_co_flux")


def test_anthropogenic_so2_contract_has_no_unbound_inventory_writer():
    _assert_unbound_inventory_contract("anthropogenic_so2_flux")


def test_anthropogenic_voc_contract_has_no_unbound_inventory_writer():
    payload = _assert_unbound_inventory_contract("anthropogenic_voc_flux")
    assert payload["dimension"]["cf_standard_name"] is None


def test_anthropogenic_nh3_contract_has_no_unbound_inventory_writer():
    _assert_unbound_inventory_contract("anthropogenic_nh3_flux")


def test_anthropogenic_pm25_contract_has_no_unbound_inventory_writer():
    payload = _assert_unbound_inventory_contract("anthropogenic_pm25_flux")
    assert payload["dimension"]["cf_standard_name"] is None


def test_contract_registry_exposes_no_inventory_writer_for_planned_fluxes():
    variables = (
        "anthropogenic_nox_flux",
        "anthropogenic_co_flux",
        "anthropogenic_so2_flux",
        "anthropogenic_voc_flux",
        "anthropogenic_nh3_flux",
        "anthropogenic_pm25_flux",
    )
    root = PROJECT_ROOT / "relic" / "contracts" / "variables" / "anthroposphere"
    registry = ContractRegistry(strict=True)

    for variable in variables:
        registry.load_contract_file(root / f"{variable}.yaml")
        contract = registry.get_contract(variable)
        assert contract is not None
        assert contract.kind == "flux"
        assert contract.udunits == "kg m-2 s-1"
        assert contract.owner is None
        assert not registry.can_write(variable, "emission_inventory")


# Legacy emission SCM placeholders removed with the user's per-file
# authorization (2026-09-24): no relic Functor, UpdateItem key, config reader
# or contract alias reads or writes any of them as state.
_LEGACY_EMISSION_PLACEHOLDERS = (
    "emission_co", "emission_nh3", "emission_nox", "emission_pm10", "emission_pm25",
    "emission_so2", "emission_voc", "emission_uncertainty",
    "emissions.BC", "emissions.CH4", "emissions.CO", "emissions.CO2", "emissions.N2O",
    "emissions.NH3", "emissions.NOx", "emissions.OC", "emissions.PM10", "emissions.PM25",
    "emissions.SO2", "emissions.VOC",
)

# The six central runtime_domain_unbound anthropogenic flux contracts, as declared.
_CENTRAL_ANTHROPOGENIC_FLUXES = {
    "anthropogenic_co_flux": (
        "CO", "tendency_of_atmosphere_mass_content_of_carbon_monoxide_due_to_emission"),
    "anthropogenic_nh3_flux": (
        "NH3", "tendency_of_atmosphere_mass_content_of_ammonia_due_to_emission"),
    "anthropogenic_nox_flux": ("NOx", None),
    "anthropogenic_pm25_flux": ("PM2.5", None),
    "anthropogenic_so2_flux": (
        "SO2", "tendency_of_atmosphere_mass_content_of_sulfur_dioxide_due_to_emission"),
    "anthropogenic_voc_flux": ("VOC", None),
}


def test_legacy_emission_placeholders_are_removed_and_central_fluxes_are_kept():
    registry = load_contract_registry()

    for variable in _LEGACY_EMISSION_PLACEHOLDERS:
        assert registry.get_contract(variable) is None, variable
        assert not registry.is_alias(variable), variable
    for variable, (species, cf_standard_name) in _CENTRAL_ANTHROPOGENIC_FLUXES.items():
        contract = registry.get_contract(variable)
        assert contract is not None, variable
        assert contract.raw == {
            "variable": variable,
            "title": f"Anthropogenic {species} Emission Flux",
            "kind": "flux",
            "dimension": {"udunits": "kg m-2 s-1", "cf_standard_name": cf_standard_name},
            "owner": None,
            "aliases": [],
            "incrementable": False,
            "accumulates": None,
            "numerics": {"lower_bound": 0.0, "upper_bound": None, "atol": 1.0e-12, "rtol": 1.0e-9},
            "writers": {"increments": [], "tendencies": []},
            "flux_contract": {
                "allowed_in": [],
                "allowed_out": [],
                "budget_check": {"guardian": None, "tolerance": {"atol": 1.0e-10, "rtol": 1.0e-8}},
            },
            "provenance": {
                "source_file": "relic/scientific/anthroposphere/inventory.py",
                "runtime_role": "runtime_domain_unbound",
                "runtime_writer": None,
            },
        }, variable


def test_inventory_certificate_is_level_zero_and_non_operational():
    path = (
        PROJECT_ROOT
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / "anthroposphere_inventory_emission_inventory.yaml"
    )
    certificate = load_iphs_equivalence_certificate(path)
    audit = certificate.validate()
    source = path.read_text(encoding="utf-8")

    assert certificate.iphs_level == 0
    assert audit["iphs_level_name"] == "ordinary_empirical"
    assert audit["passed"] is False
    assert audit["runtime_verified_level"] == 0
    assert audit["runtime_evidence_status"] == "descriptive_only"
    assert certificate.input_variables == ()
    assert certificate.output_variables == ()
    assert certificate.composition_ports == ()
    assert certificate.runtime_replacement_map["runtime_uses_iphs"] is False
    assert certificate.runtime_replacement_map["legacy_removed"] is False
    assert "[N] runtime_domain_unbound" in source
    for token in (
        "star_inventory",
        "Hodge",
        "PortSpec",
        "canonical_iphs_runtime",
        "matter port",
        "source allocation operator",
    ):
        assert token not in source


def test_obsolete_unified_inventory_positive_suite_is_removed():
    assert not (PROJECT_ROOT / "test" / "anthroposphere" / "test_unified_inventory.py").exists()


def test_obsolete_inventory_formula_and_reporting_suites_are_removed():
    obsolete = (
        "test_inventory_unified_l1_improved.py",
        "test_inventory_unified_l1.py",
        "test_inventory.py",
        "test_inventory_emissions_integration_l2.py",
    )
    test_root = PROJECT_ROOT / "test" / "anthroposphere"

    assert not [name for name in obsolete if (test_root / name).exists()]


def test_obsolete_unbound_anthroposphere_system_profile_is_removed():
    path = PROJECT_ROOT / "test" / "anthroposphere" / "test_system_l3.py"
    assert not path.exists()


def test_orphan_inventory_parameter_namespace_is_not_active():
    path = (
        PROJECT_ROOT
        / "relic"
        / "parameters"
        / "database"
        / "relic"
        / "anthroposphere"
        / "inventory_parameters.yaml"
    )
    manager = get_global_manager()

    assert not path.exists()
    for key in DEAD_INVENTORY_PARAMETER_KEYS:
        assert manager.get_parameter(key) is None
        with pytest.raises(KeyError, match=re.escape(key)):
            manager.get(key, required=True)

    for key in (
        "anthroposphere.emissions.vertical_distribution.layer_thickness",
        "anthroposphere.emissions.sector_allocation.proxy_floor",
        "anthroposphere.emissions.partition_factors.no_fraction",
    ):
        assert manager.get_parameter(key) is not None


def test_emissions_yaml_does_not_load_the_unbound_inventory_parameter_set():
    manager = get_global_manager()

    for key in DEAD_EMISSIONS_INVENTORY_PARAMETER_KEYS:
        assert manager.get_parameter(key) is None
        with pytest.raises(KeyError, match=re.escape(key)):
            manager.get(key, required=True)

    for key in (
        "anthroposphere.emissions.vertical_distribution.layer_thickness",
        "anthroposphere.emissions.sector_allocation.proxy_floor",
        "anthroposphere.emissions.partition_factors.no_fraction",
    ):
        assert manager.get_parameter(key) is not None


def test_emissions_yaml_does_not_load_adjacent_inventory_or_proxy_orphans():
    manager = get_global_manager()

    for key in DEAD_ADJACENT_INVENTORY_PARAMETER_KEYS:
        assert manager.get_parameter(key) is None
        with pytest.raises(KeyError, match=re.escape(key)):
            manager.get(key, required=True)

    for key in (
        "anthroposphere.emissions.vertical_distribution.layer_thickness",
        "anthroposphere.emissions.sector_allocation.proxy_floor",
        "anthroposphere.emissions.partition_factors.no_fraction",
    ):
        assert manager.get_parameter(key) is not None
