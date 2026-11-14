"""
L1 Unit Test: P0-1 Verification - ProcessCategory Fixes

Tests that the three internal coupling modules have correct ProcessCategory classification.

P0-1 淇鍐呭:
- BiosphereAtmosphereCoupling: BIOLOGICAL_PROCESS -> SCALE_COUPLING
- WaterCarbonCoupling: BIOLOGICAL_PROCESS -> SCALE_COUPLING
- SoilVegetationCoupling: TRANSPORT_MECHANISM -> SCALE_COUPLING

References:
- relic/coupling/internal/biosphere_atmosphere.py:178
- relic/coupling/internal/water_carbon.py:213
- relic/coupling/internal/soil_vegetation.py:215
"""

import pytest

from relic.core.functor import ProcessCategory
from relic.coupling.internal import create_coupling_functor
from relic.coupling.internal.biosphere_atmosphere import BiosphereAtmosphereCoupling
from relic.coupling.internal.soil_vegetation import SoilVegetationCoupling


class TestP0ProcessCategoryFixes:
    """Verify P0-1 ProcessCategory corrections."""

    def test_biosphere_atmosphere_category(self):
        """
        Verify BiosphereAtmosphereCoupling has ProcessCategory.SCALE_COUPLING.
        
        Original bug: Was set to BIOLOGICAL_PROCESS
        Fix location: relic/coupling/internal/biosphere_atmosphere.py:178
        """
        config = {"upstream_turbulence_functor": "atmospheric_turbulence"}
        
        functor = BiosphereAtmosphereCoupling(name='test_bio_atm', config=config)
        
        # Assert category is SCALE_COUPLING
        assert functor.metadata.category == ProcessCategory.SCALE_COUPLING, \
            f"Expected SCALE_COUPLING, got {functor.metadata.category}"
        
        # Also verify it's not the old wrong category
        assert functor.metadata.category != ProcessCategory.BIOLOGICAL_PROCESS, \
            "Category should not be BIOLOGICAL_PROCESS (old bug)"

    def test_water_carbon_is_not_an_active_coupling_type(self):
        """The duplicate water-carbon runtime is rejected by the active factory."""
        with pytest.raises(ValueError, match="Unknown coupling type"):
            create_coupling_functor("water_carbon", "inactive_wc", {})

    def test_soil_vegetation_category(self):
        """
        Verify SoilVegetationCoupling has ProcessCategory.SCALE_COUPLING.
        
        Original bug: Was set to TRANSPORT_MECHANISM
        Fix location: relic/coupling/internal/soil_vegetation.py:215
        """
        config = {
            'name': 'test_soil_veg',
            'enabled': True,
            'root_distribution': 'exponential',
        }
        
        functor = SoilVegetationCoupling(name='test_soil_veg', config=config)
        
        assert functor.metadata.category == ProcessCategory.SCALE_COUPLING, \
            f"Expected SCALE_COUPLING, got {functor.metadata.category}"
        
        assert functor.metadata.category != ProcessCategory.TRANSPORT_MECHANISM, \
            "Category should not be TRANSPORT_MECHANISM (old bug)"

    def test_all_coupling_modules_consistent(self):
        """
        Verify all three internal coupling modules have consistent category.
        
        This ensures uniform classification across the coupling layer.
        """
        config_bio_atm = {}
        config_soil_veg = {'name': 'soil_veg', 'enabled': True, 'root_distribution': 'exponential'}
        
        bio_atm = BiosphereAtmosphereCoupling(name='bio_atm', config=config_bio_atm)
        soil_veg = SoilVegetationCoupling(name='soil_veg', config=config_soil_veg)
        
        # All should be SCALE_COUPLING
        categories = [
            bio_atm.metadata.category,
            soil_veg.metadata.category,
        ]
        
        assert all(cat == ProcessCategory.SCALE_COUPLING for cat in categories), \
            f"All coupling modules should be SCALE_COUPLING, got {categories}"

    def test_metadata_fields_populated(self):
        """
        Verify metadata is properly populated beyond just category.
        
        Ensures FunctorMetadata is complete and valid.
        """
        config = {}
        
        functor = BiosphereAtmosphereCoupling(name='test_bio_atm', config=config)
        metadata = functor.metadata
        
        # Check required metadata fields
        assert metadata.category is not None, "Category must be set"
        assert metadata.description is not None, "Description must be set"
        assert len(metadata.description) > 0, "Description must not be empty"
        assert metadata.version is not None, "Version must be set"
        
        # Verify category is correct
        assert metadata.category == ProcessCategory.SCALE_COUPLING
        assert metadata.input_variables == [
            "friction_velocity",
            "air_temperature",
            "atmospheric_pressure",
        ]
        assert metadata.output_variables == ["boundary_layer_conductance"]


# MA-21 item 202: every ScaleCouplingFunctor subclass of the coupling package passes
# source_scale, target_scale and coupling_type to the bridge itself; the bridge fabricates
# none of them (a missing one raises TypeError, as the strict bridge below does).
_BRIDGE_SCALE_KEYS = ("source_scale", "target_scale", "coupling_type")


@pytest.fixture
def strict_scale_bridge(monkeypatch):
    from relic.core.functor.scale_coupling_bridge import ScaleCouplingFunctor

    original = ScaleCouplingFunctor.__init__

    def strict_init(self, *args, **kwargs):
        missing = [key for key in _BRIDGE_SCALE_KEYS if key not in kwargs]
        if missing:
            raise TypeError(f"{type(self).__name__} passes no {missing} to ScaleCouplingFunctor")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ScaleCouplingFunctor, "__init__", strict_init)


def test_internal_couplings_declare_the_scales_they_couple(strict_scale_bridge):
    """Soil-vegetation maps the soil-layer profile to the grid-cell column; water-carbon maps
    the grid-cell soil water state to grid-cell carbon-process modifiers."""
    from relic.coupling.internal.water_carbon import WaterCarbonCoupling

    soil_vegetation = SoilVegetationCoupling(name="scale_probe_sv", config={})
    assert (soil_vegetation.source_scale, soil_vegetation.target_scale, soil_vegetation.coupling_type) == (
        "soil_layer", "grid_cell", "spatial"
    )
    water_carbon = WaterCarbonCoupling(name="scale_probe_wc", config={})
    assert (water_carbon.source_scale, water_carbon.target_scale, water_carbon.coupling_type) == (
        "grid_cell", "grid_cell", "spatial"
    )


def test_hierarchical_aggregator_declares_the_hierarchical_coupling_type(strict_scale_bridge):
    from relic.coupling.scales.hierarchical import HierarchicalAggregator, HierarchyIndex

    topology = {
        "n_source_units": 2,
        "n_target_units": 1,
        "parent_map": {0: 0, 1: 0},
        "source_areas": [1.0, 1.0],
        "target_areas": [2.0],
    }
    index = HierarchyIndex(source_scale="patch", target_scale="hillslope", topology=topology)
    aggregator = HierarchicalAggregator(index=index, weight_type="area", rule="conservative")
    assert (aggregator.source_scale, aggregator.target_scale, aggregator.coupling_type) == (
        "patch", "hillslope", "hierarchical"
    )


def test_plugin_adapter_scales_are_required_instance_configuration(strict_scale_bridge, tmp_path):
    """A plugin couples whatever scales its host model declares: the scales are required keys of
    the adapter config, never assumed."""
    from relic.coupling.plugin.adapter import PluginFunctorAdapter

    path = str(tmp_path / "unused_plugin.dll")
    scales = {"source_scale": "external_grid", "target_scale": "grid_cell", "coupling_type": "spatial"}
    adapter = PluginFunctorAdapter(name="scale_probe_plugin", plugin_path=path, config=dict(scales))
    assert (adapter.source_scale, adapter.target_scale, adapter.coupling_type) == tuple(scales.values())
    for key in scales:
        partial = {k: v for k, v in scales.items() if k != key}
        with pytest.raises((KeyError, TypeError), match=key):
            PluginFunctorAdapter(name=f"scale_probe_plugin_{key}", plugin_path=path, config=partial)

