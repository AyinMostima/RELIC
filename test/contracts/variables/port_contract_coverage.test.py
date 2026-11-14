"""SCM contracts of the Functor ports of the five scientific domains (MA-21).

Coverage: every non-diagnostic input or output port of a registered Functor has an SCM
contract (or is a contract alias). The scan builds the default instance of every Functor in
FUNCTOR_REGISTRY and one instance per documented configuration of the Functors that require
one; the Functors that cannot be built by design are listed with their reason.

Each contract states the unit its reader or writer uses and the kind of its runtime
role: forcings have no RELIC writer; tillage outputs are owned by TillageManagement; the
tillage soil carbon loss joins the carbon ledger once (element carbon) and its CO2-mass
form joins none.
"""
import functools
import importlib
from typing import Dict, Set

import pytest
import torch

from relic.contracts.contract_loader import PACKAGED_CONTRACTS_DIR, load_contract_registry

# variable: (kind, udunits, owner, accumulates, cf_standard_name)
PORT_CONTRACTS = {
    # FireManagement optional overrides and the coordinator nutrient observation.
    "fuel_load": ("forcing", "kg DM m-2", None, None, None),
    "burn_fraction": ("forcing", "1", None, None, None),
    "soil_nitrogen": ("forcing", "kg N m-2", None, None, None),
    # TillageManagement outputs.
    "roughness_modifier": ("state", "1", "tillage_management", None, None),
    "tillage_intensity_factor": ("diagnostic", "1", "tillage_management", None, None),
    "time_decay_factor": ("diagnostic", "1", "tillage_management", None, None),
    "cumulative_operations": ("diagnostic", "1", "tillage_management", None, None),
    "soil_disturbance_rate": ("diagnostic", "1", "tillage_management", None, None),
    "residue_incorporation_rate": ("diagnostic", "1", "tillage_management", None, None),
    "carbon_loss_rate": ("flux", "kg C m-2 s-1", "tillage_management", "carbon.total", None),
    "soil_co2_flux": ("flux", "kg CO2 m-2 s-1", "tillage_management", None, None),
    # Atmospheric transport, chemistry and radiation inputs.
    "wind_u": ("forcing", "m s-1", None, None, None),
    "wind_v": ("forcing", "m s-1", None, None, None),
    "wind_w": ("forcing", "m s-1", None, None, "upward_air_velocity"),
    "co_mixing_ratio": (
        "forcing", "mol mol-1", None, None, "mole_fraction_of_carbon_monoxide_in_air"
    ),
    "ch4_mixing_ratio": ("forcing", "mol mol-1", None, None, "mole_fraction_of_methane_in_air"),
    "longwave_up": ("forcing", "W m-2", None, None, "surface_upwelling_longwave_flux_in_air"),
    "grid_coordinates": ("forcing", "m", None, None, None),
    "source_coordinates": ("forcing", "m", None, None, None),
    # Biosphere: forcings without a RELIC writer and the ozone_stress uptake flux.
    # The big-leaf energy balance writes the leaf temperature (diagnostic, one owner).
    "leaf_temperature": ("diagnostic", "K", "leaf_energy_balance", None, None),
    # Leaf stomatal conductance to water vapour in its readers' unit (ozone_stress,
    # oxygen_18); a m s-1 source converts with g = g_m P / (R T) from state T and P.
    "stomatal_conductance": ("forcing", "mol m-2 s-1", None, None, None),
    "daylength": ("forcing", "s", None, None, None),
    "vpmax_scaling": ("forcing", "1", None, None, None),
    "stress_level": ("forcing", "1", None, None, None),
    "ozone_uptake": ("flux", "mol O3 m-2 s-1", "ozone_stress", None, None),
    "antioxidant_depletion": ("diagnostic", "s-1", "ozone_stress", None, None),
    # Hydrosphere snow inputs without a RELIC writer: the BATS snow skin temperature is an
    # energy-balance state, sublimation an external boundary flux, the zenith angle a forcing.
    "snow_surface_temperature": ("state", "K", None, None, None),
    "solar_zenith_angle": ("forcing", "rad", None, None, "solar_zenith_angle"),
    "sublimation_flux": ("flux", "kg m-2 s-1", None, None, "surface_snow_sublimation_flux"),
    # Volumetric liquid water content per soil layer (not the layer mass of C2a's
    # mass_content_of_water_in_soil_layer, kg m-2).
    "soil_moisture_profile": ("state", "m3 m-3", None, None, None),
    # Pedosphere transport, carbon-pool and methane forcings.
    "darcy_flux": ("forcing", "m s-1", None, None, None),
    "environmental_modifier": ("forcing", "1", None, None, None),
    "fast_som_input": ("forcing", "g C m-2 s-1", None, None, None),
    "slow_som_input": ("forcing", "g C m-2 s-1", None, None, None),
    "passive_som_input": ("forcing", "g C m-2 s-1", None, None, None),
    "land_surface.wetland_fraction": ("forcing", "1", None, None, None),
    "methane_data_score": ("forcing", "1", None, None, None),
    # An activity fraction in [0, 1] (bounds and readers: test_methane_chain_contracts_*).
    "microbial_activity_index": ("forcing", "1", None, None, None),
    "redox_potential": ("forcing", "mV", None, None, None),
    "soil_o2_concentration": ("forcing", "mol m-3", None, None, None),
}
TILLAGE_OUTPUTS = sorted(
    name for name, spec in PORT_CONTRACTS.items() if spec[2] == "tillage_management"
)
# Alternative declared leaf-conductance providers share one variable contract;
# their runtime composition is exclusive, not an additive combination.
PORT_TENDENCY_WRITERS = {
    "stomatal_conductance": [{
        "name": "stomatal_proxy",
        "module": "tutorial.case.functors.stomatal_proxy",
        "class": "StomatalProxyFunctor",
    }, {
        "name": "medlyn_stomatal_conductance",
        "module": "relic.scientific.biosphere.photosynthesis.stomatal",
        "class": "MedlynStomatalConductance",
        "config": {"surface_conditions": "leaf_surface_equals_canopy_air"},
    }],
}


def _registry():
    return load_contract_registry(PACKAGED_CONTRACTS_DIR)


@pytest.mark.parametrize("variable", sorted(PORT_CONTRACTS))
def test_port_contract_states_its_runtime_role(variable):
    kind, udunits, owner, accumulates, cf_name = PORT_CONTRACTS[variable]
    contract = _registry().get_contract(variable)
    assert contract is not None, variable
    assert contract.kind == kind
    assert contract.udunits == udunits
    assert contract.owner == owner
    assert contract.accumulates == accumulates
    assert contract.cf_standard_name == cf_name
    tendency_writer = PORT_TENDENCY_WRITERS.get(variable)
    if tendency_writer is None:
        assert contract.writers == ([] if owner is None else [owner])
    else:
        assert contract.writers == [writer["name"] for writer in tendency_writer]


@pytest.mark.parametrize("variable", sorted(PORT_TENDENCY_WRITERS))
def test_port_tendency_writer_is_the_declared_functor(variable):
    """The declared tendency writer exists, writes the variable in the contract unit
    as a REPLACE (non-incrementable) alternative provider."""
    writers = PORT_TENDENCY_WRITERS[variable]
    registry = _registry()
    contract = registry.get_contract(variable)
    assert contract.tendency_writers == [writer["name"] for writer in writers]
    assert contract.increment_writers == []
    assert contract.incrementable is False
    assert not registry.can_write(variable, "ozone_stress")
    assert contract.raw["provenance"]["runtime_writer"] == [writer["name"] for writer in writers]
    for entry, writer in zip(contract.raw["writers"]["tendencies"], writers, strict=True):
        assert entry["module"] == writer["module"]
        assert entry["provides"] == [variable]
        assert entry["unit"] == contract.udunits
        assert registry.can_write(variable, writer["name"])
        assert not registry.can_increment(variable, writer["name"])
        functor_class = getattr(importlib.import_module(writer["module"]), writer["class"])
        functor = functor_class(config=writer.get("config", {}))
        assert functor.name == writer["name"]
        assert variable in functor.metadata.output_variables


def test_medlyn_commits_its_absolute_leaf_conductance_through_state_manager():
    from relic.core.state import StateManager
    from relic.scientific.biosphere.photosynthesis.stomatal import MedlynStomatalConductance

    functor = MedlynStomatalConductance(config={"surface_conditions": "leaf_surface_equals_canopy_air"})
    state = {key: torch.tensor(value, dtype=torch.float64) for key, value in {
        "net_photosynthesis": 100.0, "leaf_area_index": 2.0, "ppfd": 500.0,
        "atmospheric_co2_ppm": 400.0, "vapor_pressure_deficit": 50.0,
        "stomatal_conductance": 0.3,
    }.items()}
    manager = StateManager(device=torch.device("cpu"))
    manager.initialize_state(state)
    output = functor(manager.get_state(), 0.0, 1800.0)
    update = output["stomatal_conductance"]
    assert float(update.value) > 2.0
    manager.apply_updates({"stomatal_conductance": update}, dt=1800.0,
                          update_sources={"stomatal_conductance": functor.name})
    torch.testing.assert_close(manager.get_state()["stomatal_conductance"], update.value, rtol=0, atol=0)
    contract = _registry().get_contract("stomatal_conductance")
    assert contract.raw["numerics"]["upper_bound"] is None


@pytest.mark.parametrize("medlyn_first", [True, False])
def test_medlyn_and_tutorial_proxy_cannot_both_register(medlyn_first):
    from relic.core.simulator import RELICSimulator
    from relic.scientific.biosphere.photosynthesis.stomatal import MedlynStomatalConductance
    from tutorial.case.functors.stomatal_proxy import StomatalProxyFunctor

    simulator = RELICSimulator({"device": "cpu", "strict_dependencies": False})
    medlyn = MedlynStomatalConductance(config={"surface_conditions": "leaf_surface_equals_canopy_air"})
    proxy = StomatalProxyFunctor()
    ordered = [medlyn, proxy] if medlyn_first else [proxy, medlyn]
    simulator.register_functor(ordered[0])
    with pytest.raises(ValueError, match="stomatal_conductance|overlap|exclusive"):
        simulator.register_functor(ordered[1])


def test_leaf_conductance_lower_bound_remains_enforced():
    from relic.core.state import StateManager

    manager = StateManager(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="stomatal_conductance.*bounds"):
        manager.initialize_state({"stomatal_conductance": torch.tensor(-0.01)})
    assert _registry().get_contract("stomatal_conductance").raw["numerics"]["lower_bound"] == 0.0


def test_conditional_medlyn_runs_and_commits_in_the_existing_simulator():
    from relic.core.simulator import RELICSimulator
    from relic.scientific.biosphere.photosynthesis.stomatal import MedlynStomatalConductance

    simulator = RELICSimulator({"device": "cpu", "time_step": 1800.0})
    initial = {key: torch.tensor(value, dtype=torch.float64) for key, value in {
        "net_photosynthesis": 100.0, "leaf_area_index": 2.0, "ppfd": 500.0,
        "atmospheric_co2_ppm": 400.0, "vapor_pressure_deficit": 50.0,
        "stomatal_conductance": 0.3,
    }.items()}
    simulator.initialize_state(initial)
    functor = MedlynStomatalConductance(config={"surface_conditions": "leaf_surface_equals_canopy_air"})
    simulator.register_functor(functor, execution_group="fast")
    expected = functor(initial, 0.0, 1800.0)["stomatal_conductance"].value
    simulator.step(1800.0)
    actual = simulator.state_manager.get_state()["stomatal_conductance"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual > 2.0
    torch.testing.assert_close(simulator.state_manager.get_state()["net_photosynthesis"], initial["net_photosynthesis"])


def test_wind_components_leave_the_cf_identity_to_the_external_catalog():
    """eastward_wind and northward_wind are held by the external variables of the same
    names; one CF identity belongs to one RELIC variable."""
    from relic.orchestration.ssot_catalog import SSOTCatalog

    catalog = SSOTCatalog.from_parameter_manager()
    for variable in ("wind_u", "wind_v"):
        assert catalog.get(variable).units == "m s-1"
    assert catalog.get("eastward_wind").standard_name == "eastward_wind"
    assert catalog.get("northward_wind").standard_name == "northward_wind"


def test_tillage_outputs_carry_their_contract_units():
    """The emitted UpdateItem units equal the contract units; the carbon loss carries the
    carbon element of its ledger domain and the CO2-mass form carries none."""
    from relic.scientific.anthroposphere.management.tillage import TillageManagement

    state = {
        "soil_moisture": torch.tensor([0.3]),
        "soil_temperature": torch.tensor([288.15]),
        "soil_organic_carbon": torch.tensor([5.0]),
        "day_of_year": torch.tensor(1.0),
        "days_since_tillage": torch.tensor([365.0]),
        "residue_cover": torch.tensor([0.4]),
    }
    outputs = TillageManagement(config={"calendar_type": "noleap"}).forward(state, 0.0, 86400.0)
    registry = _registry()
    for name in TILLAGE_OUTPUTS:
        assert outputs[name].unit == registry.get_contract(name).udunits, name
    assert outputs["carbon_loss_rate"].element == "carbon"
    assert outputs["soil_co2_flux"].element is None


DOMAINS = ("anthroposphere", "atmosphere", "biosphere", "hydrosphere", "pedosphere")
_PHENOLOGY_OWNERS = ("static_phenology", "gdd_phenology", "gsi_phenology", "crop_phenology")
# Functors whose constructor requires a configuration: one instance per documented choice.
CONFIGURED = {
    "MedlynStomatalConductance": [dict(config={"surface_conditions": "leaf_surface_equals_canopy_air"})],
    "PhenologyGrowth": [dict(phenology_model_owner=owner) for owner in _PHENOLOGY_OWNERS],
    "PhenologyFunctor": [dict(config={"canonical_model_owner": owner}) for owner in _PHENOLOGY_OWNERS],
    "PenmanMonteithET": [
        dict(config={"canopy_resistance_source": source})
        for source in ("state", "fao56_reference_surface")
    ],
    "FireManagement": [dict(config={"seed": 0})],
    "MarkedPoissonRainfallFunctor": [dict(config={"seed": 0})],
    "OrnsteinUhlenbeckTemperatureFunctor": [dict(config={"seed": 0})],
    # Required selections (round 47): the open mechanism is the former default instance.
    "ChemistryFunctor": [dict(physical_boundary="open_mechanism")],
    "AtmosphericAdvectionDiffusionFunctor": [dict(spatial_dims=3, stability_scheme="monin-obukhov")],
    # temporal_support has no default: both declared supports.
    "LeafEnergyBalance": [dict(config={"temporal_support": support}) for support in ("daily", "instantaneous")],
}
# Registered classes that cannot be built by design, with the reason.
UNBUILDABLE = {
    "relic.core.functor.transport_bridge:TransportMechanismFunctor": "bridge base without TRANSPORT_TYPE",
    "relic.core.functor.physical_field_bridge:PhysicalFieldFunctor": (
        "abstract bridge base (forward and get_value_at); concrete fields are probed"
    ),
    "relic.core.functor.scale_coupling_bridge:ScaleCouplingFunctor": (
        "bridge base: requires explicit source_scale/target_scale/coupling_type"
    ),
    "relic.scientific.biosphere.phenology.base:PhenologyBase": "abstract base",
    "relic.scientific.anthroposphere.emissions.coupling_to_atmosphere:AtmosphereCoupling": "runtime dependency unbound",
    "relic.scientific.anthroposphere.interface_mapping:AnthroposphereInterfaceMapping": "runtime metadata domain unbound",
    "relic.scientific.anthroposphere.inventory:EmissionInventory": "runtime domain unbound",
    "relic.scientific.isotopes.functor:IsotopeHeartbeatFunctor": "isotope runtime unbound",
}


@functools.lru_cache(maxsize=None)
def _scan():
    """Ports per domain of every buildable registered Functor and the classes that failed."""
    from relic.scientific import FUNCTOR_REGISTRY

    ports: Dict[str, Set[str]] = {domain: set() for domain in DOMAINS}
    failed = {}
    for path in sorted(set(FUNCTOR_REGISTRY.values())):
        module_name, class_name = path.split(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        try:
            instances = [cls(**kwargs) for kwargs in CONFIGURED.get(class_name, [dict()])]
        except Exception as exc:  # the classes that cannot be built are checked below
            failed[path] = f"{type(exc).__name__}: {exc}"
            continue
        parts = module_name.split(".")
        domain = parts[2] if parts[:2] == ["relic", "scientific"] else None
        if domain not in ports:
            continue
        for instance in instances:
            meta = instance.metadata
            for name in [*meta.input_variables, *meta.output_variables]:
                if not name.startswith("diagnostics."):
                    ports[domain].add(name)
    return ports, failed


def test_only_the_declared_functors_cannot_be_built():
    """The scan covers every registered Functor except the declared unbuildable ones."""
    _ports, failed = _scan()
    assert sorted(failed) == sorted(UNBUILDABLE), failed


# Ports without a contract whose contract waits for a stated decision (MA-21 round 10). A
# port leaves this table when its contract is written; any other uncontracted port fails.
PENDING = {
    # pedosphere: dissolved_organic_carbon and hydrogen_concentration left the
    # methane_production ports (C2b, 2026-09-26), so nothing is pending there.
}


def _methane_inputs():
    from relic.scientific.pedosphere.methane.oxidation import MethaneOxidationFunctor
    from relic.scientific.pedosphere.methane.production import MethaneProductionFunctor

    return (
        set(MethaneProductionFunctor().metadata.input_variables),
        set(MethaneOxidationFunctor().metadata.input_variables),
    )


def test_methane_chain_contracts_follow_the_executed_readers():
    """MA-21 round 13 (C2b round 51 follow-up): the methane forcing and control
    contracts name the readers of the current production and oxidation code:
    redox_potential and methane_gating_weight only methane_production; porosity and
    water_filled_pore_space also methane_oxidation."""
    production, oxidation = _methane_inputs()
    assert {"redox_potential", "methane_gating_weight", "microbial_activity_index"} <= production
    assert {"redox_potential", "methane_gating_weight"}.isdisjoint(oxidation)
    assert {"soil_o2_concentration", "porosity", "water_filled_pore_space"} <= oxidation
    registry = _registry()

    redox = registry.get_contract("redox_potential").raw
    text = redox["description"] + " " + redox["provenance"]["note"]
    assert "piecewise-linear" in text and "Zhu" in text
    assert "logistic" not in text and "MethaneOxidationFunctor" not in text

    gating = registry.get_contract("methane_gating_weight").raw
    assert "Only reader: MethaneProductionFunctor" in gating["provenance"]["note"]

    for variable in ("porosity", "water_filled_pore_space"):
        note = registry.get_contract(variable).raw["provenance"]["note"]
        assert "MethaneOxidationFunctor" in note and "Eq. 19.1" in note, variable


def test_methane_chain_contracts_state_the_activity_fraction_and_oxygen_basis():
    """microbial_activity_index is an activity fraction in [0, 1] multiplying the
    calibrated production directly (DNDC95 reference code, no journal source);
    soil_o2_concentration is per water volume (CLM4.5 Technical Note Eq. 19.1) and
    enters O2 / (K_O2 + O2) with the ParameterManager K_O2 (Riley et al. 2011, Table 1)."""
    from relic.parameters import get_global_manager

    registry = _registry()
    activity = registry.get_contract("microbial_activity_index").raw
    assert activity["numerics"]["lower_bound"] == 0.0
    assert activity["numerics"]["upper_bound"] == 1.0
    assert "activity fraction" in activity["description"]
    assert "reference_microbial_activity" not in activity["description"]
    references = " ".join(activity["references"])
    assert "Dndc_main.cpp" in references and "Dndc_CH4.cpp" in references
    assert "Li, C." not in references
    assert "no journal" in activity["provenance"]["note"]

    oxygen = registry.get_contract("soil_o2_concentration").raw
    text = oxygen["description"] + " " + oxygen["provenance"]["note"]
    assert "per water volume" in text and "Eq. 19.1" in text
    assert "O2 / (K_O2 + O2)" in text
    k_o2 = float(get_global_manager().get("pedosphere.methane.oxidation.half_saturation_o2"))
    assert f"K_O2 = {k_o2:g} mol m-3" in text
    assert any("Riley" in reference for reference in oxygen["references"])


def test_readerless_vpd_contract_is_retired():
    """vpd (the kPa companion of vapor_pressure_deficit) lost its only relic reader with
    the carbon-13 VPD multiplier (D37); its contract is retired (MA-21 12.4)."""
    assert _registry().get_contract("vpd") is None
    assert not (PACKAGED_CONTRACTS_DIR / "atmosphere" / "vpd.yaml").exists()


def test_daylength_contract_replaces_the_state_variable_bounds_entry():
    """state_init takes a contracted variable from its contract, so the former
    validation.numerical.state_variable_bounds daylength entry is retired and the
    contract carries its units and bounds [0, 86400] s."""
    from relic.parameters import get_global_manager

    registry = _registry()
    contract = registry.get_contract("daylength")
    assert contract is not None
    assert contract.udunits == "s"
    bounds = get_global_manager().get("validation.numerical.state_variable_bounds")
    assert "daylength" not in bounds


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_non_diagnostic_port_has_a_contract(domain):
    """Every non-diagnostic port has a contract, except the pending ports of the domain."""
    registry = _registry()
    ports, _failed = _scan()
    missing = sorted(
        name for name in ports[domain]
        if registry.get_contract(name) is None and not registry.is_alias(name)
    )
    assert missing == sorted(PENDING.get(domain, {}))
