import torch
import pytest
import torch
from relic.parameters import get_global_manager

from relic.coupling.scales.spatial.factory import AutoRegridder, SpatialRegridderFactory
from relic.coupling.scales.spatial.grid_utils import GridDescriptor


class _PassthroughRegridder:
    def __init__(self, source_grid, target_grid, **kwargs):
        self.source_grid = source_grid
        self.target_grid = target_grid

    def upscale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.new_full(self.target_grid.dims, float(tensor.mean()))

    def downscale(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.new_full(self.target_grid.dims, float(tensor.mean()))


def _grid(dims):
    coords = torch.arange(dims[0] * dims[1], dtype=torch.float32).view(*dims)
    return GridDescriptor(
        grid_type="regular",
        coord_system="latlon",
        dims=dims,
        x_coords=coords,
        y_coords=coords,
    )


def test_strict_requires_grid_aligned_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SpatialRegridderFactory.REGRIDDER_CLASSES, "bilinear", _PassthroughRegridder)
    fine_grid = _grid((4, 4))
    coarse_grid = _grid((2, 2))
    pm = get_global_manager()
    regridder = AutoRegridder(
        source_grid=fine_grid,
        target_grid=coarse_grid,
        strict_direction_inference=True,
        validation_mode="strict",
        parameter_manager=pm,
    )

    ambiguous = torch.ones((3, 3))
    with pytest.raises(ValueError):
        regridder.forward(state={}, time=0.0, dt=1.0, data=ambiguous, variable_metadata={"variable_type": "intensive"})


def test_strict_validation_rejects_operation_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SpatialRegridderFactory.REGRIDDER_CLASSES, "bilinear", _PassthroughRegridder)
    fine_grid = _grid((4, 4))
    coarse_grid = _grid((2, 2))
    pm = get_global_manager()
    regridder = AutoRegridder(
        source_grid=fine_grid,
        target_grid=coarse_grid,
        strict_direction_inference=True,
        validation_mode="strict",
        parameter_manager=pm,
    )

    coarse_field = torch.ones(coarse_grid.dims)
    with pytest.raises(ValueError):
        regridder.forward(state={}, time=0.0, dt=1.0, data=coarse_field, operation="downscale", variable_metadata={"variable_type": "intensive"})


def test_invalid_validation_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SpatialRegridderFactory.REGRIDDER_CLASSES, "bilinear", _PassthroughRegridder)
    fine_grid = _grid((4, 4))
    coarse_grid = _grid((2, 2))
    pm = get_global_manager()

    with pytest.raises(ValueError):
        AutoRegridder(
            source_grid=fine_grid,
            target_grid=coarse_grid,
            strict_direction_inference=True,
            validation_mode="lenient",
            parameter_manager=pm,
        )
