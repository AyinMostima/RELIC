"""
L1 tests for ScaleManager parameter resolution and variable metadata handling.
"""

import pytest

from relic.coupling.scales.scale_manager import ScaleManager
from relic.core.state import StateManager
from relic.parameters import get_global_manager, reset_global_manager


class TestScaleManagerParameterResolution:
    """Validate ScaleManager parameter resolution and flux metadata safeguards."""

    def setup_method(self):
        reset_global_manager()

    def test_scale_metadata_uses_default_catalog(self):
        """Scale-specific metadata can use the default catalog when configured."""
        pm = get_global_manager()
        manager = ScaleManager(parameter_manager=pm, state_manager=StateManager())
        manager.initialize()
        meta = manager.get_variable_metadata("air_temperature", scale="nonexistent_scale")
        assert meta, "Default variable catalog should be returned when scale-specific metadata is absent"
        assert meta["variable_type"] == "intensive"

    def test_variable_metadata_requires_flux_definitions(self):
        """Flux-like variables must be explicitly defined; they raise the same KeyError as others."""
        pm = get_global_manager()
        manager = ScaleManager(parameter_manager=pm, state_manager=StateManager())
        manager.initialize()
        with pytest.raises(KeyError, match="custom_flux"):
            manager.get_variable_metadata("custom_flux", scale="patch")

    def test_the_generic_carbon_flux_entry_is_not_a_catalog_variable(self):
        """carbon_flux (no contract, no Functor producer or consumer) is not declared by any scale
        catalog: the facade, the hierarchical aggregator and the SSOT catalog all miss it, while the
        contracted area density net_primary_productivity resolves on the same paths."""
        from relic.contracts.contract_loader import load_contract_registry
        from relic.coupling.scales.hierarchical import HierarchicalAggregator, HierarchyIndex
        from relic.coupling.scales.hierarchical.templates import get_rhessys_template
        from relic.coupling.scales.types import SpatialQuantityBasis
        from relic.orchestration.ssot_catalog import SSOTCatalog

        pm = get_global_manager()
        for namespace in ("coupling.scales.defaults", "coupling.scales.daily"):
            assert "carbon_flux" not in pm.get("variables", namespace=namespace, required=True), namespace
        manager = ScaleManager(parameter_manager=pm, state_manager=StateManager())
        manager.initialize()
        with pytest.raises(KeyError, match="carbon_flux"):
            manager.get_variable_metadata("carbon_flux", scale="patch")
        aggregator = HierarchicalAggregator(
            HierarchyIndex(source_scale="patch", target_scale="landscape", topology=get_rhessys_template("test_hillslope")),
            weight_type="area",
            rule="conservative",
        )
        with pytest.raises((KeyError, ValueError), match="carbon_flux"):
            aggregator._variable_basis("carbon_flux")
        catalog = SSOTCatalog.from_parameter_manager()
        assert catalog.get("carbon_flux") is None
        # The contracted replacement sample: an area density per unit ground area.
        assert catalog.get("net_primary_productivity").spatial_basis is SpatialQuantityBasis.AREA_DENSITY
        assert manager.get_variable_metadata("net_primary_productivity", scale="patch")["spatial_basis"] == "area_density"
        assert aggregator._variable_basis("net_primary_productivity") is SpatialQuantityBasis.AREA_DENSITY
        assert load_contract_registry().get_owner("net_primary_productivity") == "autotrophic_respiration"

    def test_variable_metadata_requires_definition(self):
        """Variables without explicit metadata should raise for clarity."""
        pm = get_global_manager()
        manager = ScaleManager(parameter_manager=pm, state_manager=StateManager())
        manager.initialize()
        with pytest.raises(KeyError):
            manager.get_variable_metadata("custom_state", scale="patch")

    def test_public_lookup_is_the_single_catalog_implementation(self):
        """get_variable_metadata(name, scale=...) is the public entry; the private batch form
        delegates to it (kept only for callers that have not switched yet)."""
        import inspect

        pm = get_global_manager()
        manager = ScaleManager(parameter_manager=pm, state_manager=StateManager())
        manager.initialize()
        entry = manager.get_variable_metadata("precipitation_flux", scale="patch")
        assert entry["spatial_basis"] == "area_density"
        entry["spatial_basis"] = "changed"
        assert manager.get_variable_metadata("precipitation_flux", scale="patch")["spatial_basis"] == "area_density"
        private_source = inspect.getsource(ScaleManager._get_variable_metadata)
        assert "get_variable_metadata(" in private_source
        assert "parameter_manager.get(" not in private_source

    def test_one_catalog_lookup_serves_the_facade_aggregator_and_adapters(self):
        """ScaleManager, HierarchicalAggregator and the scale adapters share one catalog matcher."""
        import inspect

        import relic.coupling.scales.adapters as adapters_module
        from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator
        from relic.coupling.scales.variable_catalog import CATALOG_ALIASES, find_catalog_entry

        pm = get_global_manager()
        manager = ScaleManager(parameter_manager=pm, state_manager=StateManager())
        manager.initialize()
        # A documented alias resolves through the one matcher in every caller.
        facade_entry = manager.get_variable_metadata("lai", scale="patch")
        adapter_entry, source = adapters_module._lookup_variable_metadata("lai", pm)
        assert adapter_entry is not None and source.startswith("parameter:")
        assert facade_entry["units"] == adapter_entry["units"] == "m2 m-2"
        # Resistance is the reciprocal of conductance: no alias between them.
        assert "stomatal_resistance" not in CATALOG_ALIASES
        with pytest.raises(KeyError):
            manager.get_variable_metadata("stomatal_resistance", scale="patch")
        for owner in (
            ScaleManager.get_variable_metadata,
            HierarchicalAggregator._get_variable_metadata,
            adapters_module._lookup_variable_metadata,
        ):
            source_text = inspect.getsource(owner)
            assert "find_catalog_entry(" in source_text, owner.__qualname__
            assert "alias_map" not in source_text, owner.__qualname__
        assert callable(find_catalog_entry)

    def test_ambiguous_short_names_are_not_catalog_aliases(self):
        """A short name that can denote several quantities (amount or flux; air, soil or leaf
        temperature; relative or specific humidity; CO2 concentration or flux) is no alias."""
        from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator
        from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex
        from relic.coupling.scales.variable_catalog import CATALOG_ALIASES

        assert dict(CATALOG_ALIASES) == {"lai": "leaf_area_index", "rh": "relative_humidity"}
        pm = get_global_manager()
        manager = ScaleManager(parameter_manager=pm, state_manager=StateManager())
        manager.initialize()
        index = HierarchyIndex(
            source_scale="patch",
            target_scale="hillslope",
            topology={
                "n_source_units": 2,
                "n_target_units": 1,
                "parent_map": {0: 0, 1: 0},
                "source_areas": [1.0, 1.0],
                "target_areas": [2.0],
            },
        )
        aggregator = HierarchicalAggregator(index, weight_type="area", rule="conservative")
        for name in ("precipitation", "precip", "runoff", "et", "transpiration", "temperature", "temp", "humidity", "co2"):
            with pytest.raises(KeyError, match=name):
                manager.get_variable_metadata(name, scale="patch")
            with pytest.raises(ValueError, match=name):
                aggregator._get_variable_metadata(name)
        assert manager.get_variable_metadata("rh", scale="patch")["units"] == "1"
