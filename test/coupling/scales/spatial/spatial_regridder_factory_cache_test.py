import pytest
import torch

from relic.coupling.scales.spatial.factory import SpatialRegridderFactory
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.parameters import get_global_manager


def _make_grid(shape):
    return GridDescriptor.from_regular_grid(shape=shape, spacing=(1.0, 1.0))


def _strip(x_edges, cell_areas=None, *, coord_system="cartesian", x_coords=None, **fields) -> GridDescriptor:
    """One-row grid on y in [0, 1] with explicit x bounds (and cell areas in m2 when given)."""
    nx = len(x_edges) - 1
    x_bounds = torch.tensor([[x_edges[i], x_edges[i + 1]] for i in range(nx)], dtype=torch.float64)
    return GridDescriptor(
        grid_type="regular",
        coord_system=coord_system,
        dims=(1, nx),
        x_coords=x_bounds.mean(dim=1) if x_coords is None else torch.tensor(x_coords, dtype=torch.float64),
        y_coords=torch.tensor([0.5], dtype=torch.float64),
        x_bounds=x_bounds,
        y_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        cell_areas=None if cell_areas is None else torch.tensor(cell_areas, dtype=torch.float64).reshape(1, nx),
        **fields,
    )


def test_cache_key_covers_the_whole_geometry_and_every_argument():
    """A regridder is reused only for equal content: every descriptor field and every argument is
    in the key (a same-shape target of another geometry never gets the cached weights)."""
    factory = SpatialRegridderFactory(parameter_manager=get_global_manager())
    source = _strip([0.0, 1.0, 2.0], [1.0, 1.0])
    target = _strip([0.0, 2.0], [2.0])
    request = dict(method="conservative", variable_metadata={"spatial_basis": "cell_total"})
    first = factory.create_regridder(source, target, **request)
    # Equal content in new descriptor objects is the same key.
    assert factory.create_regridder(_strip([0.0, 1.0, 2.0], [1.0, 1.0]), _strip([0.0, 2.0], [2.0]), **request) is first
    variants = {
        "x_bounds": _strip([0.0, 1.0], [1.0]),
        "x_coords": _strip([0.0, 2.0], [2.0], x_coords=[0.7]),
        "cell_areas": _strip([0.0, 2.0], [3.0]),
        "mask": _strip([0.0, 2.0], [2.0], mask=torch.tensor([[True]])),
        "crs": _strip([0.0, 2.0], [2.0], crs="EPSG:3857"),
        "units": _strip([0.0, 2.0], [2.0], units={"x": "km", "y": "km"}),
    }
    for field_name, variant in variants.items():
        assert factory.create_regridder(source, variant, **request) is not first, field_name
    # One geometry per regridder: the coordinate system changes on both grids.
    projected = factory.create_regridder(
        _strip([0.0, 1.0, 2.0], [1.0, 1.0], coord_system="projected"),
        _strip([0.0, 2.0], [2.0], coord_system="projected"),
        **request,
    )
    assert projected is not first
    narrow = factory.create_regridder(source, variants["x_bounds"], **request)
    torch.testing.assert_close(
        narrow.downscale(torch.tensor([[4.0, 8.0]], dtype=torch.float64)),
        torch.tensor([[4.0]], dtype=torch.float64),
    )
    # Arguments outside any fixed list are part of the key as well.
    dense = factory.create_regridder(source, target, use_sparse_weights=False, **request)
    assert dense is not first and dense.use_sparse_weights is False
    # An argument without a canonical content encoding is rejected, never ignored.
    with pytest.raises(TypeError, match="fingerprint"):
        factory.create_regridder(source, target, name=object(), **request)


def test_factory_cache_separates_boundary_modes():
    """Factory should not reuse cached regridders when the boundary modes differ.

    Uncovered targets are missing (NaN) in every configuration, so no fill value
    is part of a regridder or of its cache key.
    """
    import pytest

    pm = get_global_manager()
    factory = SpatialRegridderFactory(parameter_manager=pm)
    source = _make_grid((2, 2))
    target = _make_grid((2, 2))

    regrid_nearest = factory.create_regridder(
        source,
        target,
        method="bilinear",
        boundary_mode="nearest",
    )
    regrid_constant = factory.create_regridder(
        source,
        target,
        method="bilinear",
        boundary_mode="constant",
    )

    assert regrid_nearest is not regrid_constant
    assert regrid_nearest.boundary_mode == "nearest"
    assert regrid_constant.boundary_mode == "constant"

    # Same configuration should hit cache
    regrid_constant_again = factory.create_regridder(
        source,
        target,
        method="bilinear",
        boundary_mode="constant",
    )
    assert regrid_constant_again is regrid_constant
    with pytest.raises(TypeError):
        factory.create_regridder(source, target, method="bilinear", fill_value=5.0)


def test_factory_passes_an_explicit_quantity_basis_to_the_conservative_regridder():
    """The conservative regridder receives one explicit basis; the cache separates bases."""
    import pytest

    from relic.coupling.scales.types import SpatialQuantityBasis

    pm = get_global_manager()
    factory = SpatialRegridderFactory(parameter_manager=pm)
    source = _make_grid((2, 2))
    target = _make_grid((2, 2))

    density = factory.create_regridder(
        source, target, method="conservative", variable_metadata={"spatial_basis": "area_density"}
    )
    totals = factory.create_regridder(
        source, target, method="conservative", variable_metadata={"spatial_basis": "cell_total"}
    )
    assert density.quantity_basis is SpatialQuantityBasis.AREA_DENSITY
    assert totals.quantity_basis is SpatialQuantityBasis.CELL_TOTAL
    assert density is not totals
    with pytest.raises(ValueError, match="ambiguous"):
        factory.create_regridder(
            source, target, method="conservative", variable_metadata={"variable_type": "extensive"}
        )
    with pytest.raises(ValueError, match="conservative method only"):
        factory.create_regridder(source, target, method="bilinear", quantity_basis="intensive")
