import pytest
import torch

from relic.coupling.scales.spatial.regrid_bilinear import BilinearRegridder
from relic.coupling.scales.spatial.regrid_nearest import NearestNeighborRegridder
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.parameters import get_global_manager


def _unstructured_grid() -> GridDescriptor:
    x_coords = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    y_coords = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    vertex_coords = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    connectivity = torch.tensor(
        [
            [0, 1, 3, 2],
            [0, 1, 3, 2],
            [0, 1, 3, 2],
            [0, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    return GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(4,),
        x_coords=x_coords,
        y_coords=y_coords,
        connectivity=connectivity,
        vertex_coords=vertex_coords,
    )


def test_unstructured_grid_descriptor_accepts_connectivity():
    grid = _unstructured_grid()
    assert grid.grid_type == "unstructured"
    assert grid.dims == (4,)
    assert grid.connectivity is not None
    assert grid.vertex_coords is not None


def test_irregular_bilinear_idw_center():
    pm = get_global_manager()
    source = _unstructured_grid()
    target = GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(1,),
        x_coords=torch.tensor([0.5], dtype=torch.float32),
        y_coords=torch.tensor([0.5], dtype=torch.float32),
    )
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    regridder = BilinearRegridder(source, target, parameter_manager=pm)
    result = regridder.forward({}, time=0.0, dt=1.0, data=data)

    assert "regridded_data" in result
    assert result["regridded_data"].shape == (1,)
    assert torch.allclose(result["regridded_data"], torch.tensor([2.5], dtype=data.dtype))


def test_irregular_nearest_neighbor_center():
    pm = get_global_manager()
    source = _unstructured_grid()
    target = GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(1,),
        x_coords=torch.tensor([0.5], dtype=torch.float32),
        y_coords=torch.tensor([0.5], dtype=torch.float32),
    )
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    regridder = NearestNeighborRegridder(source, target, parameter_manager=pm)
    result = regridder.forward({}, time=0.0, dt=1.0, data=data)

    assert "regridded" in result
    assert result["regridded"].shape == (1,)
    assert result["regridded"].item() in {1.0, 2.0, 3.0, 4.0}


def _scaled_idw(scale: float, target_xy) -> torch.Tensor:
    """IDW regrid of the unit-square nodes (values 1..4) with every coordinate multiplied by scale."""
    base = _unstructured_grid()
    source = GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(4,),
        x_coords=base.x_coords * scale,
        y_coords=base.y_coords * scale,
        connectivity=base.connectivity,
        vertex_coords=base.vertex_coords * scale,
    )
    target = GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(1,),
        x_coords=torch.tensor([target_xy[0] * scale], dtype=torch.float32),
        y_coords=torch.tensor([target_xy[1] * scale], dtype=torch.float32),
    )
    regridder = BilinearRegridder(source, target, parameter_manager=get_global_manager())
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    return regridder.forward({}, time=0.0, dt=1.0, data=data)["regridded_data"]


@pytest.mark.parametrize("scale", [1.0e-6, 1.0e-12, 1.0e3])
def test_irregular_idw_weights_do_not_depend_on_the_coordinate_unit(scale):
    """w_i = d_i^-p / sum_k d_k^-p is unchanged when every coordinate is multiplied by c > 0."""
    reference = _scaled_idw(1.0, (0.25, 0.4))
    assert torch.allclose(_scaled_idw(scale, (0.25, 0.4)), reference, rtol=1.0e-5, atol=0.0)


@pytest.mark.parametrize("scale", [1.0, 1.0e-12])
def test_a_target_on_a_source_node_takes_the_node_value(scale):
    """A target coinciding with node (1, 0) takes its value 2 at any coordinate unit."""
    assert torch.equal(_scaled_idw(scale, (1.0, 0.0)), torch.tensor([2.0], dtype=torch.float32))


@pytest.mark.parametrize("scale", [1.0, 1.0e3, 1.0e-3])
def test_irregular_nearest_search_radius_does_not_depend_on_the_coordinate_unit(scale):
    """The search radius is max_search_radius times the typical source spacing (median distance
    from a node to its nearest other node), so the centre of the unit square finds a node at any
    coordinate unit instead of taking the fill value."""
    base = _unstructured_grid()
    source = GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(4,),
        x_coords=base.x_coords * scale,
        y_coords=base.y_coords * scale,
        connectivity=base.connectivity,
        vertex_coords=base.vertex_coords * scale,
    )
    target = GridDescriptor(
        grid_type="unstructured",
        coord_system="cartesian",
        dims=(1,),
        x_coords=torch.tensor([0.5 * scale], dtype=torch.float32),
        y_coords=torch.tensor([0.5 * scale], dtype=torch.float32),
    )
    regridder = NearestNeighborRegridder(source, target, parameter_manager=get_global_manager())
    result = regridder.forward({}, time=0.0, dt=1.0, data=torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert result["regridded"].item() in {1.0, 2.0, 3.0, 4.0}
