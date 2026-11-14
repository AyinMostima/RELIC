"""
Audit gate for conservation_laws metadata contracts in coupling functors.
"""

from __future__ import annotations

from typing import Any

import torch

from relic.coupling.external.common.base_adapter import ExternalAdapter
from relic.coupling.external.common.datasource import DataSource
from relic.coupling.scales.adapters import ConservativeRemap
from relic.coupling.scales.hierarchical.aggregator import HierarchicalAggregator
from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex
from relic.coupling.scales.spatial.base_regridder import SpatialRegridderBase
from relic.coupling.scales.spatial.grid_utils import GridDescriptor


class _DummyExternalAdapter(ExternalAdapter):
    def discover(self, source):
        return {}

    def open(self, source, config):
        return None

    def read(self, variables, time_window=None, spatial_window=None):
        return {}

    def transform_to_internal(self, data, mappings):
        return {}

    def close(self):
        return None


class _DummySpatialRegridder(SpatialRegridderBase):
    def upscale(self, data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return data

    def downscale(self, data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return data


def _make_regular_grid(nx: int, ny: int) -> GridDescriptor:
    return GridDescriptor(shape=(nx, ny), spacing=(1.0, 1.0), device=torch.device("cpu"))


def test_coupling_functor_metadata_conservation_laws_contract() -> None:
    """
    References:
    - docs/source/architecture_overview.md conservation metadata requirements.
    - Jones (1999) conservative remapping constraints.

    Mathematical expression:
    - delta_storage = (flux_in - flux_out) * dt.

    Parameter meanings:
    - preserve_mass/preserve_energy/enforce_conservation control conservation policy.
    """
    remap = ConservativeRemap(preserve_mass=True, preserve_energy=True)
    assert set(remap.metadata.conservation_laws) == {"mass", "energy"}

    remap_mass_only = ConservativeRemap(preserve_mass=True, preserve_energy=False)
    assert set(remap_mass_only.metadata.conservation_laws) == {"mass"}

    hierarchy_index = HierarchyIndex(
        source_scale="fine",
        target_scale="coarse",
        device=torch.device("cpu"),
        topology={
            "parent_map": {0: 0},
            "source_areas": [1.0],
            "target_areas": [1.0],
            "n_source_units": 1,
            "n_target_units": 1,
        },
    )
    hier = HierarchicalAggregator(index=hierarchy_index, rule="conservative", weight_type="area")
    assert set(hier.metadata.conservation_laws) == {"mass", "energy"}

    adapter = _DummyExternalAdapter(
        source=DataSource.WRF, name="dummy_external", provided_variables=["air_temperature"]
    )
    assert adapter.metadata.conservation_laws == []

    source_grid = _make_regular_grid(2, 2)
    target_grid = _make_regular_grid(3, 3)
    regridder = _DummySpatialRegridder(
        source_grid=source_grid,
        target_grid=target_grid,
        name="dummy_regridder",
        enforce_conservation=True,
        conservation_tolerance=1e-6,
    )
    assert set(regridder.metadata.conservation_laws) == {"mass"}
