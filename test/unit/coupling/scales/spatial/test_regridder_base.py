"""L1 tests for SpatialRegridderBase shared workflow."""

from __future__ import annotations

import torch

import relic.coupling.scales.spatial.grid_utils as grid_utils_module
from relic.coupling.scales.spatial.base_regridder import SpatialRegridderBase
from relic.coupling.scales.spatial.grid_utils import GridUtils


class DummyRegridder(SpatialRegridderBase):
    """Minimal concrete regridder for exercising base forward logic."""

    def __init__(self, source_grid, target_grid, device=torch.device("cpu")) -> None:
        super().__init__(
            source_grid=source_grid,
            target_grid=target_grid,
            name="dummy_regridder",
            config={},
            metadata=None,
            device=device,
        )

    def _broadcast_fill(self, data: torch.Tensor) -> torch.Tensor:
        """Helper to broadcast mean value to target grid with optional batch."""
        tgt_ny, tgt_nx = self.target_grid.dims
        if data.dim() == 3:
            batch = data.shape[0]
            fill = data.mean(dim=(-2, -1), keepdim=True)
            return fill.expand(batch, tgt_ny, tgt_nx)
        fill_value = torch.mean(data)
        return torch.full((tgt_ny, tgt_nx), fill_value, device=self.device)

    def upscale(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._broadcast_fill(data)

    def downscale(self, data: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._broadcast_fill(data)


def _regular_grids(src_shape, tgt_shape, device=torch.device("cpu")):
    utils = GridUtils(device=device)
    src = utils.create_regular_grid(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=src_shape[1],
        ny=src_shape[0],
        coord_system="cartesian",
    )
    tgt = utils.create_regular_grid(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        nx=tgt_shape[1],
        ny=tgt_shape[0],
        coord_system="cartesian",
    )
    return src, tgt


class ToleranceRegridder(DummyRegridder):
    """DummyRegridder that forwards an explicit conservation tolerance to the base."""

    def __init__(self, source_grid, target_grid, conservation_tolerance) -> None:
        SpatialRegridderBase.__init__(
            self,
            source_grid=source_grid,
            target_grid=target_grid,
            name="tolerance_regridder",
            config={},
            metadata=None,
            device=torch.device("cpu"),
            conservation_tolerance=conservation_tolerance,
        )


def test_the_tolerance_has_one_default_and_is_validated():
    """The base resolves the one default (coupling.scales.defaults.conservation_tolerance) and
    rejects an explicit tolerance that is not finite in (0, 1), for every regridder class."""
    import pytest

    from relic.coupling.scales.spatial.regrid_conservative import ConservativeRegridder
    from relic.parameters import get_global_manager

    src, tgt = _regular_grids((2, 2), (1, 1))
    default = float(get_global_manager().get("conservation_tolerance", namespace="coupling.scales.defaults", required=True))
    assert DummyRegridder(src, tgt).conservation_tolerance == default
    assert ToleranceRegridder(src, tgt, 1.0e-6).conservation_tolerance == 1.0e-6
    for invalid in (float("nan"), float("inf"), 0.0, 1.0, -1.0e-9):
        with pytest.raises(ValueError, match="conservation_tolerance"):
            ToleranceRegridder(src, tgt, invalid)
        with pytest.raises(ValueError, match="conservation_tolerance"):
            ConservativeRegridder(src, tgt, conservation_tolerance=invalid, quantity_basis="cell_total")


def test_base_forward_detects_downscale_and_formats_output():
    source_grid, target_grid = _regular_grids((4, 4), (2, 2))
    regridder = DummyRegridder(source_grid, target_grid)

    data = torch.ones((4, 4))
    result = regridder.forward({"data": data}, time=0.0, dt=1.0)

    assert result["operation"] == "downscale"
    assert tuple(result["data"].shape) == target_grid.dims
    assert result["method"] == "DummyRegridder"
    assert result["input_shape"] == tuple(data.shape)


def test_base_forward_detects_upscale():
    source_grid, target_grid = _regular_grids((2, 2), (4, 4))
    regridder = DummyRegridder(source_grid, target_grid)

    data = torch.full((2, 2), 5.0)
    result = regridder.forward({"data": data}, time=0.0, dt=1.0)

    assert result["operation"] == "upscale"
    assert torch.allclose(result["data"], torch.full(target_grid.dims, 5.0))


def test_base_raises_for_missing_or_mismatched_data_and_kernel_errors():
    """A failed call raises; it never returns the input or an empty tensor as the result."""
    import pytest

    source_grid, target_grid = _regular_grids((2, 2), (2, 2))
    regridder = DummyRegridder(source_grid, target_grid)

    with pytest.raises(ValueError, match="data"):
        regridder.forward({}, time=0.0, dt=1.0)
    # Five values are neither one field nor a batch of fields on the 4-cell grid (no truncation).
    with pytest.raises(ValueError, match="4 cells"):
        regridder.forward({"data": torch.ones(5)}, time=0.0, dt=1.0)

    class FailingRegridder(DummyRegridder):
        def upscale(self, data, **kwargs):
            raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        FailingRegridder(source_grid, target_grid).forward(
            {"data": torch.ones(2, 2), "operation": "upscale"}, time=0.0, dt=1.0
        )
    assert not hasattr(SpatialRegridderBase, "_handle_failure")


def test_base_preserves_batch_shape_in_extraction():
    source_grid, target_grid = _regular_grids((3, 3), (2, 2))
    regridder = DummyRegridder(source_grid, target_grid)

    batch_data = torch.ones((3, 3, 3))
    result = regridder.forward({"data": batch_data}, time=0.0, dt=1.0)

    assert result["input_shape"] == (3, 3, 3)
    assert tuple(result["data"].shape) == (3, *target_grid.dims)


def test_grid_utils_physical_constants_use_ssot(monkeypatch):
    """Grid geometry constants must bypass the ordinary ParameterManager path."""

    class PhysicalNamespaceGuard:
        def get(self, *args, **kwargs):
            raise AssertionError(
                "Grid physical constants must use get_physical_constant"
            )

    requested = []
    values = {
        "earth_radius": 6371000.0,
        "relic.physical.constants.conversions.deg_to_rad": 0.017453292519943295,
        "relic.physical.constants.conversions.rad_to_deg": 57.29577951308232,
    }

    def fake_get_physical_constant(name, as_tensor=False, device=None):
        requested.append(name)
        value = values[name]
        if as_tensor:
            return torch.as_tensor(value, device=device, dtype=torch.float64)
        return value

    monkeypatch.setattr(
        grid_utils_module,
        "get_global_manager",
        lambda: PhysicalNamespaceGuard(),
    )
    monkeypatch.setattr(
        grid_utils_module,
        "get_physical_constant",
        fake_get_physical_constant,
        raising=False,
    )

    utilities = GridUtils(device=torch.device("cpu"))

    assert utilities.earth_radius == 6371000.0
    assert utilities.deg_to_rad.item() == values[
        "relic.physical.constants.conversions.deg_to_rad"
    ]
    assert utilities.rad_to_deg.item() == values[
        "relic.physical.constants.conversions.rad_to_deg"
    ]
    assert requested == [
        "earth_radius",
        "relic.physical.constants.conversions.deg_to_rad",
        "relic.physical.constants.conversions.rad_to_deg",
    ]


def test_identical_cell_counts_need_an_explicit_operation():
    """Equal cell counts do not decide a direction: the call names it (no downscale default)."""
    import pytest

    source_grid, target_grid = _regular_grids((2, 2), (2, 2))
    regridder = DummyRegridder(source_grid, target_grid)
    with pytest.raises(ValueError, match="operation"):
        regridder.forward({"data": torch.ones((2, 2))}, time=0.0, dt=1.0)
    result = regridder.forward({"data": torch.ones((2, 2)), "operation": "upscale"}, time=0.0, dt=1.0)
    assert result["operation"] == "upscale"
    assert not hasattr(regridder, "_identical_grid_size_warned")
