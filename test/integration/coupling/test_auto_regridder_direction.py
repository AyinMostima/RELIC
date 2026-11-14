import torch
import pytest
from relic.parameters import get_global_manager

from relic.coupling.scales.spatial.factory import AutoRegridder, SpatialRegridderFactory
from relic.coupling.scales.spatial.grid_utils import GridDescriptor


class _DummyRegridder:
    def __init__(self, source_grid, target_grid, **kwargs):
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.upscale_calls = 0
        self.downscale_calls = 0

    def upscale(self, tensor: torch.Tensor) -> torch.Tensor:
        self.upscale_calls += 1
        return tensor.new_full(self.target_grid.dims, float(tensor.mean()))

    def downscale(self, tensor: torch.Tensor) -> torch.Tensor:
        self.downscale_calls += 1
        return tensor.new_full(self.target_grid.dims, float(tensor.mean()))


def _grid(dims):
    data = torch.arange(dims[0] * dims[1], dtype=torch.float32).view(*dims)
    return GridDescriptor(
        grid_type="regular",
        coord_system="latlon",
        dims=dims,
        x_coords=data,
        y_coords=data,
    )


def test_auto_regridder_uses_target_to_source_when_input_matches_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SpatialRegridderFactory.REGRIDDER_CLASSES, "bilinear", _DummyRegridder)
    fine_grid = _grid((4, 4))
    coarse_grid = _grid((2, 2))
    pm = get_global_manager()

    regridder = AutoRegridder(
        source_grid=fine_grid,
        target_grid=coarse_grid,
        enforce_conservation=False,
        parameter_manager=pm,
    )

    coarse_field = torch.ones(coarse_grid.dims)
    outputs = regridder.forward(
        state={},
        time=0.0,
        dt=1.0,
        data=coarse_field,
        variable_metadata={
            "variable_type": "intensive",
            "regridding_method": "bilinear",
        },
    )

    assert "regridded" in outputs
    assert tuple(outputs["regridded"].shape) == fine_grid.dims
    assert outputs["_regrid_metadata"]["operation"] == "upscale"


def test_auto_regridder_dict_shape_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SpatialRegridderFactory.REGRIDDER_CLASSES, "bilinear", _DummyRegridder)
    fine_grid = _grid((4, 4))
    coarse_grid = _grid((2, 2))
    pm = get_global_manager()
    regridder = AutoRegridder(
        source_grid=fine_grid,
        target_grid=coarse_grid,
        enforce_conservation=False,
        parameter_manager=pm,
    )

    data = {
        "a": torch.ones(fine_grid.dims),
        "b": torch.ones(coarse_grid.dims),
    }
    with pytest.raises(ValueError):
        regridder.forward(
            state={},
            time=0.0,
            dt=1.0,
            data=data,
            variables=["a", "b"],
            variable_metadata={
                "variable_type": "intensive",
                "regridding_method": "bilinear",
            },
        )


def _axis_grid(x, y):
    return GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(len(y), len(x)),
        x_coords=torch.tensor(x),
        y_coords=torch.tensor(y),
    )


def test_auto_regridder_needs_a_declared_method_for_upscale_and_downscale() -> None:
    """Without variable metadata no method is assumed (no bilinear default)."""
    fine = _axis_grid([0.0, 1.0, 2.0, 3.0], [0.0, 1.0])
    coarse = _axis_grid([0.0, 2.0], [0.0])
    regridder = AutoRegridder(
        source_grid=fine, target_grid=coarse, enforce_conservation=False, parameter_manager=get_global_manager()
    )
    with pytest.raises(ValueError, match="metadata"):
        regridder.upscale(torch.ones(fine.dims))
    with pytest.raises(ValueError, match="metadata"):
        regridder.downscale(torch.ones(coarse.dims))
    assert regridder.get_current_method() is None


def test_auto_regridder_maps_each_direction_from_the_input_grid_to_the_other_grid() -> None:
    """Nearest neighbour through AutoRegridder: downscale maps target-grid data onto the source grid."""
    fine = _axis_grid([0.0, 1.0, 2.0, 3.0], [0.0, 1.0])
    coarse = _axis_grid([0.0, 2.0], [0.0, 1.0])
    regridder = AutoRegridder(
        source_grid=fine, target_grid=coarse, enforce_conservation=False, parameter_manager=get_global_manager()
    )
    regridder.set_variable_metadata({"spatial_basis": "categorical", "regridding_method": "nearest"})
    classes = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    torch.testing.assert_close(regridder.upscale(classes), torch.tensor([[1.0, 3.0], [5.0, 7.0]]))
    # Coarse values at x = 0 and x = 2 go to the nearest fine cells; x = 1 is equidistant and
    # takes the first coarse node, x = 3 the second.
    down = regridder.downscale(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
    torch.testing.assert_close(down, torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]]))
