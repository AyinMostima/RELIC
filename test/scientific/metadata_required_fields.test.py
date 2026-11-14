"""Metadata required field regression tests for targeted scientific functors."""

import fnmatch
import importlib
import re
import sys
from pathlib import Path

import pytest
import yaml

from relic.core.functor import ExecutionPhase
from relic.scientific.atmosphere.environmental_forcing import (
    EnvironmentalForcingFunctor,
)
from relic.scientific.biosphere.photosynthesis.farquhar import (
    FarquharPhotosynthesis,
)
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress


def _assert_required_metadata(functor) -> None:
    metadata = functor.metadata
    assert metadata is not None, f"{functor.name} metadata missing"
    assert isinstance(metadata.version, str) and metadata.version.strip(), (
        f"{functor.name} metadata.version is required"
    )
    assert isinstance(metadata.execution_phase, ExecutionPhase), (
        f"{functor.name} metadata.execution_phase must be ExecutionPhase enum"
    )
    assert isinstance(metadata.input_variables, list) and metadata.input_variables, (
        f"{functor.name} metadata.input_variables must be non-empty list"
    )
    assert isinstance(metadata.output_variables, list) and metadata.output_variables, (
        f"{functor.name} metadata.output_variables must be non-empty list"
    )


def test_targeted_functors_have_required_metadata_fields():
    functors = [
        EnvironmentalForcingFunctor(),
        FarquharPhotosynthesis(enable_gpu=False),
        OzoneStress(),
    ]
    for functor in functors:
        _assert_required_metadata(functor)


def test_relic_package_does_not_export_category_theory_runtime_symbols():
    relic_pkg = importlib.import_module("relic")

    assert not hasattr(relic_pkg, "CategoryObject")
    assert not hasattr(relic_pkg, "CategoryMorphism")
    assert not hasattr(relic_pkg, "CategoryFunctor")
    assert not hasattr(relic_pkg, "EcosystemCategory")
    assert "relic.core.category_theory" not in sys.modules


# Full-key metadata contract (round 34 scope; the hydrosphere Functors carry the same
# contract in test/scientific/hydrosphere/metadata_completeness.test.py). Every
# metadata.parameters entry names full ParameterManager keys (one exact key, or one
# full-key pattern whose every match the Functor reads); every key a construction reads
# is declared; an exact entry ends with the key's database units in brackets, a pattern
# entry with its matches' common units or "[units per key]"; the IPHS certificate lists
# the same entries. Framework keys that FunctorBase and the bridges read for every
# Functor (validation.numerical.response.* of the biological bridge,
# validation.conservation.functor_* of the local conservation hook,
# validation.iphs.hodge_* of the Hodge validation) are not the Functor's own parameters.
_FULL_KEY_SCOPE = (
    "relic.scientific.atmosphere.environmental_forcing:EnvironmentalForcingFunctor",
    "relic.scientific.atmosphere.radiation:RadiationField",
    "relic.scientific.atmosphere.radiation:RadiationFunctor",
    "relic.scientific.atmosphere.radiation_energy_conservation:RadiationEnergyConservationValidator",
    "relic.scientific.atmosphere.transport.AtmosphericAdvectionDiffusionFunctor:"
    "AtmosphericAdvectionDiffusionFunctor",
    "relic.scientific.atmosphere.transport.dry_deposition_aggregator:DryDepositionAggregator",
    "relic.scientific.atmosphere.transport.eddy_diffusion_blm:EddyDiffusionBLMFunctor",
    "relic.scientific.atmosphere.turbulence:TurbulenceFunctor",
    "relic.scientific.anthroposphere.emissions.sector_allocation:SectorAllocation",
    "relic.scientific.anthroposphere.emissions.species_mapping:SpeciesMapper",
    "relic.scientific.anthroposphere.land_use.mapping:LandUseParameterMapping",
    "relic.scientific.anthroposphere.management.coordinator:ManagementCoordinator",
    "relic.scientific.anthroposphere.management.fire_management:FireManagement",
    "relic.scientific.anthroposphere.management.grazing:GrazingManagement",
    "relic.scientific.anthroposphere.management.harvest:HarvestManagement",
    "relic.scientific.anthroposphere.management.irrigation:IrrigationManagement",
    "relic.scientific.anthroposphere.management.tillage:TillageManagement",
    "relic.scientific.pedosphere.interface_mapping:PedosphereInterfaceMapper",
    "relic.scientific.pedosphere.nutrient_cycling.denitrification:DenitrificationProcess",
    "relic.scientific.pedosphere.nutrient_cycling.nitrification:NitrificationProcess",
    "relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching:NitrogenLeachingProcess",
    "relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake:RootNutrientUptake",
    "relic.scientific.pedosphere.nutrient_cycling.stoichiometry:StoichiometryConstraint",
    "relic.scientific.pedosphere.transport.SoilWaterTransportFunctor:SoilWaterTransportFunctor",
    "relic.scientific.pedosphere.transport.equilibrium_sorption:EquilibriumSorptionFunctor",
    "relic.scientific.pedosphere.transport.moving_domain_diffusion:MovingDomainDiffusionFunctor",
    "relic.scientific.pedosphere.transport.two_region_mass_transfer:TwoRegionMassTransferFunctor",
    "relic.scientific.biosphere.growth.biomass_allocation:BiomassAllocation",
    "relic.scientific.biosphere.photosynthesis.farquhar:FarquharPhotosynthesis",
    "relic.scientific.biosphere.phenology.models:CropPhenology",
    "relic.scientific.biosphere.phenology.models:GDDPhenology",
    "relic.scientific.biosphere.phenology.models:GSIPhenology",
    "relic.scientific.biosphere.phenology.models:StaticPhenology",
    "relic.scientific.biosphere.respiration.growth_respiration:GrowthRespiration",
    "relic.scientific.biosphere.respiration.maintenance_respiration:MaintenanceRespiration",
    "relic.scientific.biosphere.stress.nutrient_stress:NutrientStress",
    "relic.scientific.biosphere.stress.ozone_stress:OzoneStress",
    "relic.scientific.biosphere.surrogates.simple_photosynthesis:SimpleCanopyPhotosynthesis",
    "relic.scientific.biosphere.stress.thermal_stress:ThermalStress",
    "relic.scientific.biosphere.stress.water_stress:WaterStress",
    "relic.scientific.biosphere.bvoc.megan_model:MEGANModel",
    "relic.scientific.biosphere.bvoc.synthesis.isoprene:IsopreneSynthesis",
    "relic.scientific.biosphere.bvoc.synthesis.monoterpene:MonoterpeneSynthesis",
    "relic.scientific.biosphere.bvoc.synthesis.sesquiterpene:SesquiterpeneStressModel",
    "relic.scientific.biosphere.bvoc.synthesis.sesquiterpene:SesquiterpeneSynthesis",
    # Round 36: the short-name override classes declare full keys; their class-level
    # short names only whitelist overrides and serve runtime lookups.
    "relic.scientific.atmosphere.chemistry:ChemistryFunctor",
    "relic.scientific.atmosphere.stochastic_rainfall:MarkedPoissonRainfallFunctor",
    "relic.scientific.atmosphere.stochastic_temperature:OrnsteinUhlenbeckTemperatureFunctor",
    "relic.scientific.hydrosphere.runoff.surface_ponding:HortonianPondingEventFunctor",
    "relic.scientific.hydrosphere.runoff.surface_ponding:SurfacePondingFunctor",
)
_FRAMEWORK_KEY_PREFIXES = (
    "validation.numerical.response.",
    "validation.conservation.functor_",
    "validation.iphs.hodge_",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _database_units() -> dict:
    """Full key -> units of every entry of the YAML parameter database."""
    root = _REPO_ROOT / "relic" / "parameters" / "database" / "relic"
    units = {}
    for path in sorted(root.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("parameters"), dict):
            continue
        namespace = str(data.get("namespace") or "").strip()
        for key, spec in data["parameters"].items():
            full = key if not namespace or key.startswith(namespace + ".") else f"{namespace}.{key}"
            units[full] = str(spec.get("units")) if isinstance(spec, dict) else None
    return units


def _matches(entry: str, database: dict) -> set:
    if entry in database:
        return {entry}
    pattern = entry.replace("<pft>", "*").replace("<pft_name>", "*")
    return set(fnmatch.filter(database, pattern)) if "*" in pattern else set()


def _declared_entries(functor) -> dict:
    """Declared key -> units text, by the rule of the public science generator: the entry
    name is the key, or, with a namespace, the namespace joined with the name (a namespace
    whose last segment is the name is itself the key). A string entry states its units in a
    trailing bracket, a dict entry in its 'units' field."""
    entries = {}
    for name, entry in functor.metadata.parameters.items():
        info = entry if isinstance(entry, dict) else {}
        namespace = str(info.get("namespace", "")).strip()
        if not namespace or name.startswith(namespace + "."):
            key = name
        elif namespace.rsplit(".", 1)[-1] == name:
            key = namespace
        else:
            key = f"{namespace}.{name}"
        if key.startswith(_FRAMEWORK_KEY_PREFIXES):
            continue
        if isinstance(entry, str):
            bracket = re.search(r"\[([^\[\]]+)\]\s*$", entry)
            entries[key] = bracket.group(1) if bracket else None
        else:
            entries[key] = info.get("units")
    return entries


def _certificate_entries(functor) -> list:
    process_id = f"{type(functor).__module__}.{type(functor).__name__}"
    root = _REPO_ROOT / "docs" / "source" / "theory" / "iphs_certificates"
    for path in sorted(root.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and data.get("process_id") == process_id:
            return list(data.get("parameter_keys") or [])
    raise AssertionError(f"no certificate for {process_id}")


@pytest.mark.parametrize("path", _FULL_KEY_SCOPE)
def test_metadata_parameters_are_full_keys_the_functor_reads(path, monkeypatch):
    from relic.core.functor.functor_base import FunctorBase
    from relic.parameters.namespaces import resolve_namespace
    from test.utils.functor_minimal_construction import construct_minimal

    reads = []
    original = FunctorBase.get_parameter_value

    def recording(self, name, **kwargs):
        namespace = resolve_namespace(kwargs.get("namespace"))
        reads.append(self._resolve_parameter_lookup(name, namespace)[0])
        return original(self, name, **kwargs)

    monkeypatch.setattr(FunctorBase, "get_parameter_value", recording)
    functor = construct_minimal(path)
    database = _database_units()
    declared = _declared_entries(functor)

    assert [entry for entry in declared if not _matches(entry, database)] == []
    covered = set().union(*(_matches(entry, database) for entry in declared)) if declared else set()
    construction_reads = {key for key in reads if not key.startswith(_FRAMEWORK_KEY_PREFIXES)}
    assert sorted(construction_reads - covered) == []
    for entry, declared_units in declared.items():
        units = {database[key] for key in _matches(entry, database)}
        expected = units.pop() if len(units) == 1 else "units per key"
        assert declared_units == expected, (entry, declared_units, expected)
    # A <pft> or <pft_name> metadata entry is the '*' template key of the certificate
    # (the certificate schema admits '*' keys in a non-strict certificate).
    templated = [
        entry.replace("<pft>", "*").replace("<pft_name>", "*") for entry in declared
    ]
    assert sorted(_certificate_entries(functor)) == sorted(templated)
