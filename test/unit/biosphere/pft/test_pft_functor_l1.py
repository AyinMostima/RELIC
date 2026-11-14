"""L1 architecture tests for canonical PFT configuration ownership.

PFT traits are configuration, not runtime scientific state. The canonical
path is ParameterManager -> PFTDefinition -> PFTRegistry.
"""

from pathlib import Path

import pytest

import relic.scientific.biosphere as biosphere_package
import relic.scientific.biosphere.pft as pft_package
from relic.parameters import get_global_manager
from relic.scientific.biosphere.pft.pft_registry import (
    PFTRegistry,
    PFTRegistryError,
    get_pft_registry,
)
from relic.scientific.biosphere.pft.pft_types import PFTDefinition


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_PFTS = (
    "needleleaf_evergreen_temperate",
    "broadleaf_deciduous_temperate",
    "c3_grass_temperate",
    "c4_grass_temperate",
)
RETIRED_PFT_NAMES = (
    "evergreen_needleleaf",
    "deciduous_broadleaf",
    "c3_grass",
    "c4_grass",
)


@pytest.fixture(scope="module")
def registry() -> PFTRegistry:
    """Return one clean registry for this architecture test module; the process
    singleton is put back after the module."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PFTRegistry, "_instance", None)
        yield PFTRegistry.get_instance()


def test_pft_packages_do_not_export_runtime_functor() -> None:
    """PFT configuration must not be exposed as a scientific runtime functor."""
    assert not hasattr(pft_package, "PFTFunctor")
    assert not hasattr(biosphere_package, "PFTFunctor")


def test_production_tree_does_not_instantiate_pft_functor() -> None:
    """No production module may instantiate the retired runtime distributor."""
    consumers: list[str] = []
    retired_source = (
        PROJECT_ROOT
        / "relic"
        / "scientific"
        / "biosphere"
        / "pft"
        / "pft_functor.py"
    )
    for source_path in (PROJECT_ROOT / "relic").rglob("*.py"):
        if source_path == retired_source:
            continue
        source = source_path.read_text(encoding="utf-8-sig")
        if "PFTFunctor(" in source:
            consumers.append(str(source_path.relative_to(PROJECT_ROOT)))
    assert consumers == []


def test_registry_is_the_single_typed_configuration_owner(
    registry: PFTRegistry,
) -> None:
    """All public registry entry points must resolve to one existing owner."""
    assert get_pft_registry() is registry
    assert PFTRegistry.get_instance() is registry


def test_typed_definition_is_not_a_runtime_process(
    registry: PFTRegistry,
) -> None:
    """A PFT definition has typed traits and no forward state-update surface."""
    definition = registry.get_pft("broadleaf_deciduous_temperate")
    assert isinstance(definition, PFTDefinition)
    assert not hasattr(definition, "forward")
    assert not hasattr(definition, "metadata")


def test_typed_definition_contains_all_parameter_groups(
    registry: PFTRegistry,
) -> None:
    """The canonical typed view must expose all six configured trait groups."""
    definition = registry.get_pft("broadleaf_deciduous_temperate")
    groups = (
        definition.photosynthesis,
        definition.respiration,
        definition.allocation,
        definition.phenology,
        definition.bvoc,
        definition.stress,
    )
    assert all(group is not None for group in groups)


def test_typed_value_matches_parameter_manager_ssot(
    registry: PFTRegistry,
) -> None:
    """The typed view must preserve the exact ParameterManager value."""
    definition = registry.get_pft("broadleaf_deciduous_temperate")
    manager = get_global_manager()
    key = "biosphere.pft.broadleaf_deciduous_temperate.photosynthesis.vcmax25"
    assert definition.photosynthesis.vcmax25 == manager.get(key, required=True)


def test_parameter_spec_has_complete_governance_metadata() -> None:
    """A representative PFT trait must satisfy the canonical YAML schema."""
    manager = get_global_manager()
    key = "biosphere.pft.broadleaf_deciduous_temperate.photosynthesis.vcmax25"
    spec = manager.get_parameter(key)
    assert spec.units == "umol CO2 m-2 s-1"
    assert spec.bounds is not None
    assert spec.category
    assert spec.description
    assert spec.references
    assert spec.tags
    assert spec.provenance


@pytest.mark.parametrize("pft_name", CANONICAL_PFTS)
def test_canonical_pft_names_resolve_typed_definitions(
    registry: PFTRegistry,
    pft_name: str,
) -> None:
    """Canonical PFT identities must resolve without alias translation."""
    definition = registry.get_pft(pft_name)
    assert definition.name == pft_name
    assert definition.photosynthesis.vcmax25 is not None


@pytest.mark.parametrize("retired_name", RETIRED_PFT_NAMES)
def test_retired_pft_names_fail_fast(
    registry: PFTRegistry,
    retired_name: str,
) -> None:
    """Retired identities must not activate a compatibility fallback."""
    with pytest.raises(PFTRegistryError, match="not found"):
        registry.get_pft(retired_name)
