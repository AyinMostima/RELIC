"""L3 system tests for canonical PFT configuration ownership.

PFT traits are immutable configuration inputs. The canonical ownership path is
ParameterManager -> PFTDefinition -> PFTRegistry; PFTs are not executable
Functors and must not enter the dependency graph as state-update processes.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

import relic.scientific.biosphere as biosphere_package
import relic.scientific.biosphere.pft as pft_package
from relic.parameters import get_global_manager
from relic.scientific.biosphere.pft.pft_registry import (
    PFTRegistry,
    PFTRegistryError,
)
from relic.scientific.biosphere.pft.pft_types import (
    PFTCategory,
    PFTDefinition,
)


CANONICAL_PFTS = frozenset(category.value for category in PFTCategory)
RETIRED_ALIASES = frozenset(
    {
        "shrub_evergreen",
        "shrub_deciduous",
        "evergreen_needleleaf",
        "deciduous_broadleaf",
        "c3_grass",
        "c4_grass",
    }
)
TRAIT_GROUPS = (
    "photosynthesis",
    "respiration",
    "allocation",
    "phenology",
    "bvoc",
    "stress",
)


@pytest.fixture(scope="module")
def registry() -> PFTRegistry:
    """Create an isolated registry and restore the process singleton later."""
    previous = PFTRegistry._instance
    PFTRegistry._instance = None
    isolated = PFTRegistry.get_instance()
    isolated.load_default_pfts()
    yield isolated
    PFTRegistry._instance = previous


def test_system_contract_does_not_import_retired_pft_functor() -> None:
    """Keep system tests on the registry-owned PFT configuration path."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "relic.scientific.biosphere.pft.pft_functor" not in imported_modules
    assert not hasattr(pft_package, "PFTFunctor")
    assert not hasattr(biosphere_package, "PFTFunctor")


def test_default_catalog_is_exact_and_lazy(registry: PFTRegistry) -> None:
    """Index exactly 15 canonical YAML identities before trait materialization."""
    indexed = set(registry.list_pfts())
    assert indexed == CANONICAL_PFTS
    assert len(indexed) == 15
    assert RETIRED_ALIASES.isdisjoint(indexed)
    assert set(registry._lightweight_pfts) == CANONICAL_PFTS

    requested_name = "broadleaf_deciduous_temperate"
    requested = registry.get_pft(requested_name)
    assert isinstance(requested, PFTDefinition)
    assert requested.photosynthesis is not None
    assert requested_name not in registry._lightweight_pfts
    assert registry._lightweight_pfts == CANONICAL_PFTS - {requested_name}


def test_unknown_and_retired_names_fail_fast(registry: PFTRegistry) -> None:
    """Reject unknown identities instead of translating aliases or defaulting."""
    for name in (*sorted(RETIRED_ALIASES), "unknown_plant_type"):
        with pytest.raises(PFTRegistryError, match="not found"):
            registry.get_pft(name)


def test_all_typed_definitions_materialize_without_runtime_surface(
    registry: PFTRegistry,
) -> None:
    """Materialize all canonical traits without creating a runtime process."""
    for pft_name in sorted(CANONICAL_PFTS):
        definition = registry.get_pft(pft_name)
        assert isinstance(definition, PFTDefinition)
        assert definition.name == pft_name
        assert definition.category is PFTCategory(pft_name)
        assert all(getattr(definition, group) is not None for group in TRAIT_GROUPS)
        assert not hasattr(definition, "forward")
        assert not hasattr(definition, "metadata")


def test_typed_traits_match_parameter_manager_values_and_specs(
    registry: PFTRegistry,
) -> None:
    """Verify the typed view against representative governed YAML entries."""
    manager = get_global_manager()
    definition = registry.get_pft("broadleaf_deciduous_temperate")
    checks = {
        "photosynthesis.vcmax25": (
            definition.photosynthesis.vcmax25,
            "umol CO2 m-2 s-1",
        ),
        "allocation.f_leaf": (definition.allocation.f_leaf, "dimensionless"),
        "phenology.temp_growth_opt": (definition.phenology.temp_growth_opt, "K"),
        "bvoc.ef_isoprene": (definition.bvoc.ef_isoprene, "ug g-1 h-1"),
        "stress.temp_stress_min": (definition.stress.temp_stress_min, "K"),
    }

    for path, (typed_value, expected_units) in checks.items():
        key = f"biosphere.pft.{definition.name}.{path}"
        spec = manager.get_parameter(key)
        managed_value = manager.get(key, required=True)
        assert math.isclose(
            float(typed_value),
            float(managed_value),
            rel_tol=1.0e-7,
            abs_tol=1.0e-9,
        )
        assert spec.units == expected_units
        assert spec.bounds is not None
        assert spec.description
        assert spec.references
        assert spec.tags
        assert spec.provenance


def test_traits_are_invariant_across_environment_reads(
    registry: PFTRegistry,
) -> None:
    """PFT traits remain configuration and do not depend on seasonal state."""
    pft_name = "broadleaf_deciduous_temperate"
    definition = registry.get_pft(pft_name)
    baseline = (
        definition.photosynthesis.vcmax25,
        definition.allocation.f_leaf,
        definition.phenology.specific_leaf_area,
    )
    seasonal_temperatures_k = (288.15, 298.15, 283.15, 273.15)

    observed = []
    for _air_temperature_k in seasonal_temperatures_k:
        repeated = registry.get_pft(pft_name)
        assert repeated is definition
        observed.append(
            (
                repeated.photosynthesis.vcmax25,
                repeated.allocation.f_leaf,
                repeated.phenology.specific_leaf_area,
            )
        )
    assert observed == [baseline] * len(seasonal_temperatures_k)


def test_fraction_weighted_ecosystem_trait_aggregation(
    registry: PFTRegistry,
) -> None:
    """Validate a closed fractional-area PFT trait aggregation.

    References: Oleson et al. (2013), CLM4.5 technical description.
    Expression: P_eff = sum_i(f_i * P_i), with sum_i(f_i) = 1.
    Parameters: f_i is PFT area fraction [1]; P_i is Vcmax25
    [umol CO2 m-2 s-1]. The calculation has no source or sink route.
    """
    composition = {
        "broadleaf_deciduous_temperate": 0.4,
        "needleleaf_evergreen_temperate": 0.3,
        "c3_grass_temperate": 0.3,
    }
    assert math.fsum(composition.values()) == pytest.approx(1.0, abs=1.0e-12)

    component_values = {
        name: registry.get_pft(name).photosynthesis.vcmax25
        for name in composition
    }
    effective_vcmax25 = math.fsum(
        composition[name] * component_values[name] for name in composition
    )

    assert math.isfinite(effective_vcmax25)
    assert min(component_values.values()) <= effective_vcmax25
    assert effective_vcmax25 <= max(component_values.values())


def test_allocation_fractions_close_carbon_ledger(
    registry: PFTRegistry,
) -> None:
    """Validate closed PFT allocation fractions for every canonical type.

    References: CASA-CNP allocation scheme; Oleson et al. (2013).
    Expression: C_leaf + C_stem + C_root = C_alloc, where
    C_j = f_j * C_alloc and sum_j(f_j) = 1.
    Parameters: f_j is a dimensionless allocation fraction; C_alloc is
    allocated carbon [kg C m-2]. This closed ledger has no boundary route.
    """
    allocated_carbon = 10.0
    for pft_name in sorted(CANONICAL_PFTS):
        allocation = registry.get_pft(pft_name).allocation
        fractions = (allocation.f_leaf, allocation.f_stem, allocation.f_root)
        assert all(value is not None and 0.0 <= value <= 1.0 for value in fractions)
        assert math.fsum(fractions) == pytest.approx(1.0, abs=1.0e-6)

        tissue_carbon = tuple(allocated_carbon * value for value in fractions)
        residual = math.fsum(tissue_carbon) - allocated_carbon
        assert residual == pytest.approx(0.0, abs=1.0e-12)


def test_all_materialized_definitions_pass_registry_validation(
    registry: PFTRegistry,
) -> None:
    """Require every canonical definition to pass the existing validator."""
    results = registry.validate_all_pfts()
    assert set(results) == CANONICAL_PFTS
    assert results == {name: [] for name in results}
