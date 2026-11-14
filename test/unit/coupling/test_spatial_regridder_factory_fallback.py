import torch

import torch
from relic.parameters import get_global_manager

from relic.coupling.scales.spatial.factory import SpatialRegridderFactory
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.coupling.scales.spatial.regrid_nearest import NearestNeighborRegridder


def _build_grid(nx: int, ny: int) -> GridDescriptor:
    coords_x = torch.linspace(0.0, float(nx - 1), steps=nx)
    coords_y = torch.linspace(0.0, float(ny - 1), steps=ny)
    return GridDescriptor(
        grid_type="regular",
        coord_system="latlon",
        dims=(nx, ny),
        x_coords=coords_x,
        y_coords=coords_y,
    )


def test_factory_respects_explicit_tolerance() -> None:
    grid_a = _build_grid(2, 2)
    grid_b = _build_grid(2, 2)
    pm = get_global_manager()

    factory = SpatialRegridderFactory(conservation_tolerance=1.0e-4, parameter_manager=pm)
    assert abs(factory.conservation_tolerance - 1.0e-4) < 1e-12

    regridder = SpatialRegridderFactory.create_regridder(
        "nearest",
        grid_a,
        grid_b,
        conservation_tolerance=1.0e-4,
        parameter_manager=pm,
    )
    assert isinstance(regridder, NearestNeighborRegridder)
    assert regridder.source_grid.dims == grid_a.dims
    assert regridder.target_grid.dims == grid_b.dims
