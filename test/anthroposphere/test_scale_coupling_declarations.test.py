"""The anthroposphere ScaleCouplingFunctor subclasses declare their scales (item 120).

ScaleCouplingFunctor will no longer fabricate source_scale='fine',
target_scale='coarse' and coupling_type='spatial'; every constructible subclass passes
the three values of the coupling it performs to the bridge.
"""
import pytest

from relic.core.functor import scale_coupling_bridge
from relic.core.functor.scale_coupling_bridge import ScaleCouplingFunctor
from relic.scientific.anthroposphere.emissions.sector_allocation import SectorAllocation
from relic.scientific.anthroposphere.emissions.species_mapping import SpeciesMapper
from relic.scientific.anthroposphere.land_use.transitions import LandUseTransitions
from relic.scientific.anthroposphere.management.coordinator import ManagementCoordinator

DECLARED = {
    # Grid-cell management state aggregated to the domain coordination decisions
    # (upscale: mean over cells) and decisions broadcast back to the cells.
    ManagementCoordinator: ("grid_cell", "management_domain", "spatial"),
    # Land-use fraction cochains moved between the declared fine and coarse grid
    # supports through the bound conservative chain maps.
    LandUseTransitions: ("fine_grid", "coarse_grid", "spatial"),
    # Inventory species totals split into sector components on the same grid cell
    # (E_sector = W E_input, sum over sectors = E_input): a category hierarchy.
    SectorAllocation: ("inventory_grid", "sector_species_grid", "hierarchical"),
    SpeciesMapper: ("inventory", "mechanism", "hierarchical"),
}


def _construct(cls):
    if cls is ManagementCoordinator:
        return cls(config={"calendar_type": "noleap"})
    return cls()


@pytest.mark.parametrize("cls", list(DECLARED), ids=lambda cls: cls.__name__)
def test_scale_coupling_subclass_passes_its_scales_to_the_bridge(cls, monkeypatch):
    received = {}
    original = ScaleCouplingFunctor.__init__

    def recording_init(self, *args, **kwargs):
        for key in ("source_scale", "target_scale", "coupling_type"):
            if key in kwargs:
                received[key] = kwargs[key]
        return original(self, *args, **kwargs)

    monkeypatch.setattr(scale_coupling_bridge.ScaleCouplingFunctor, "__init__", recording_init)
    instance = _construct(cls)
    expected = dict(zip(("source_scale", "target_scale", "coupling_type"), DECLARED[cls]))
    assert received == expected
    info = instance.get_coupling_metadata()
    for key, value in expected.items():
        assert info[key] == value, key
