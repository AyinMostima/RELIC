import torch

from relic.coupling.scales.spatial.regrid_nearest import NearestNeighborRegridder
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.parameters import get_global_manager


def _regular_grid(nx: int, ny: int) -> GridDescriptor:
    return GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(ny, nx),
        x_coords=torch.linspace(0.0, float(nx - 1), steps=nx),
        y_coords=torch.linspace(0.0, float(ny - 1), steps=ny),
    )


def test_forward_upscale_returns_regridded_tensor():
    source = _regular_grid(nx=2, ny=2)
    target = _regular_grid(nx=1, ny=1)
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    regridder = NearestNeighborRegridder(source, target)
    result = regridder.forward({}, time=0.0, dt=1.0, data=data)

    assert "regridded" in result
    assert result["regridded"].shape == (1, 1)
    assert torch.allclose(result["regridded"], torch.tensor([[1.0]], dtype=data.dtype))
    assert result["_regrid_metadata"]["method"] == "nearest"


def test_forward_autodetects_downscale():
    import pytest

    source = _regular_grid(nx=2, ny=2)
    target = _regular_grid(nx=1, ny=1)
    coarse = torch.tensor([[5.0]], dtype=torch.float32)

    # The one-node target grid declares no cell bounds, so the search radius of
    # boundary_mode="constant" around it is undefined and the downscale raises.
    with pytest.raises(ValueError, match="single node"):
        NearestNeighborRegridder(source, target, boundary_mode="constant").forward(
            {}, time=0.0, dt=1.0, data=coarse
        )
    regridder = NearestNeighborRegridder(source, target, boundary_mode="nearest")
    result = regridder.forward({}, time=0.0, dt=1.0, data=coarse)

    assert "regridded" in result
    assert result["regridded"].shape == (2, 2)
    expected = torch.full((2, 2), 5.0, dtype=result["regridded"].dtype)
    assert torch.allclose(result["regridded"], expected)
    assert result["_regrid_metadata"]["operation"] == "downscale"
    # The downscale covers the four source cells (the reverse mapping).
    assert result["_regrid_metadata"]["coverage"] == {
        "target_cells": 4,
        "covered_target_cells": 4,
        "uncovered_target_cells": 0,
    }


def test_parameter_defaults_apply_when_kwargs_missing():
    source = _regular_grid(nx=2, ny=2)
    target = _regular_grid(nx=1, ny=1)
    pm = get_global_manager()
    snapshot = pm.snapshot()
    try:
        pm.set_value(
            "coupling.scales.spatial.boundary_handling_mode",
            "nearest",
            units="dimensionless",
            validate=True,
            notify=False,
        )
        regridder = NearestNeighborRegridder(
            source, target, boundary_mode=None, parameter_manager=pm
        )
        assert regridder.boundary_mode == "nearest"
    finally:
        pm.restore(snapshot)


def _grid(x, y) -> GridDescriptor:
    return GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(len(y), len(x)),
        x_coords=torch.tensor(x, dtype=torch.float32),
        y_coords=torch.tensor(y, dtype=torch.float32),
    )


def test_regular_grids_keep_the_row_major_field_layout():
    """Data on a regular grid are laid out [ny, nx]; f = x + 100 y maps point for point."""
    source = _grid([0.0, 1.0, 2.0], [0.0, 10.0])
    target = _grid([0.0, 2.0], [0.0, 10.0])
    field = torch.tensor([0.0, 1.0, 2.0])[None, :] + 100.0 * torch.tensor([0.0, 10.0])[:, None]

    regridder = NearestNeighborRegridder(source, target, boundary_mode="nearest")

    expected = torch.tensor([[0.0, 2.0], [1000.0, 1002.0]])
    torch.testing.assert_close(regridder.upscale(field), expected, rtol=0.0, atol=0.0)


def test_uncovered_targets_are_missing_and_reported_not_filled():
    """A target without a source within the search radius is NaN and counted, never filled."""
    source = _grid([0.0, 1.0], [0.0, 1.0])
    target = _grid([0.0, 1.0, 50.0], [0.0])
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    regridder = NearestNeighborRegridder(source, target, boundary_mode="constant")
    result = regridder.forward({"operation": "upscale"}, time=0.0, dt=1.0, data=data)

    regridded = result["regridded"]
    assert regridded[0, 0].item() == 1.0 and regridded[0, 1].item() == 2.0
    assert torch.isnan(regridded[0, 2])
    coverage = {"target_cells": 3, "covered_target_cells": 2, "uncovered_target_cells": 1}
    assert result["_regrid_metadata"]["coverage"] == coverage
    assert regridder.coverage_report == coverage
    assert not hasattr(regridder, "fill_value")


def test_integer_field_with_uncovered_targets_raises_through_forward():
    """An integer field cannot hold a missing value: the uncovered target raises, not 0."""
    import pytest

    source = _grid([0.0, 1.0], [0.0, 1.0])
    target = _grid([0.0, 1.0, 50.0], [0.0])
    classes = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    regridder = NearestNeighborRegridder(source, target, boundary_mode="constant")

    with pytest.raises(ValueError, match="uncovered"):
        regridder.upscale(classes)
    with pytest.raises(ValueError, match="uncovered"):
        regridder.forward({"operation": "upscale"}, time=0.0, dt=1.0, data=classes)
    # With every target covered an integer field keeps its dtype and its class codes.
    covered = NearestNeighborRegridder(source, _grid([0.0, 1.0], [0.0]), boundary_mode="constant")
    codes = covered.upscale(classes)
    assert codes.dtype == torch.int64
    assert codes.tolist() == [[1, 2]]


def test_preserving_an_amount_is_refused_not_warned():
    """Nearest-neighbour assignment conserves no amount: preserve='mass' raises."""
    import pytest

    source = _grid([0.0, 1.0], [0.0, 1.0])
    target = _grid([0.0, 1.0], [0.0])
    regridder = NearestNeighborRegridder(source, target, boundary_mode="constant")
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="preserve"):
        regridder.regrid(data, source, target, preserve="mass", method="nearest")
    torch.testing.assert_close(regridder.regrid(data, source, target, method="nearest"), torch.tensor([[1.0, 2.0]]))


def test_operation_is_explicit_when_the_data_fit_both_grids():
    """A field that fits both grids (equal dims) needs an explicit operation; no upscale default."""
    import pytest

    source = _grid([0.0, 1.0], [0.0, 1.0])
    target = _grid([0.5, 1.5], [0.5, 1.5])
    regridder = NearestNeighborRegridder(source, target, boundary_mode="nearest")
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="operation"):
        regridder.forward({}, time=0.0, dt=1.0, data=data)
    # A field that fits neither grid raises instead of being taken as a source field.
    with pytest.raises(ValueError, match="grid"):
        regridder.forward({}, time=0.0, dt=1.0, data=torch.ones(3))
    result = regridder.forward({"operation": "upscale"}, time=0.0, dt=1.0, data=data)
    assert result["_regrid_metadata"]["operation"] == "upscale"


def test_forward_refuses_another_method_or_an_amount_to_preserve():
    """forward follows the rule of regrid (and of the bilinear forward): a requested method other
    than 'nearest' and a preserve other than 'none' raise, from the state or the call keywords;
    the reported method and preserve are the regridder's own."""
    import pytest

    source = _grid([0.0, 1.0], [0.0, 1.0])
    target = _grid([0.0, 1.0], [0.0])
    regridder = NearestNeighborRegridder(source, target, boundary_mode="constant")
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    accepted = regridder.forward(
        {"method": "nearest", "preserve": "none"}, time=0.0, dt=1.0, data=data, operation="upscale"
    )
    torch.testing.assert_close(accepted["regridded"], torch.tensor([[1.0, 2.0]]))
    assert accepted["_regrid_metadata"]["method"] == "nearest"
    assert accepted["_regrid_metadata"]["preserve"] == "none"

    for method in ("bilinear", "conservative"):
        with pytest.raises(ValueError, match="method"):
            regridder.forward({"method": method}, time=0.0, dt=1.0, data=data, operation="upscale")
        with pytest.raises(ValueError, match="method"):
            regridder.forward({}, time=0.0, dt=1.0, data=data, operation="upscale", method=method)
    for preserve in ("mass", "energy"):
        with pytest.raises(ValueError, match="preserve"):
            regridder.forward({"preserve": preserve}, time=0.0, dt=1.0, data=data, operation="upscale")
        with pytest.raises(ValueError, match="preserve"):
            regridder.forward({}, time=0.0, dt=1.0, data=data, operation="upscale", preserve=preserve)
