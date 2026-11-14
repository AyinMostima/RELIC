import numpy as np
import pytest
import torch

import relic.coupling.external.common as external_common
import relic.coupling.external.common.regrid as regrid_module
from relic.coupling.external.common.regrid import (
    RegridConfig,
    RegridMethod,
    RegridOperator,
)
from relic.coupling.scales.spatial import GridDescriptor
from relic.coupling.scales.types import SpatialQuantityBasis


def test_external_regrid_uses_the_single_spatial_basis_owner():
    """No local VariableType; the config carries one explicit SpatialQuantityBasis."""
    assert not hasattr(regrid_module, "VariableType")
    assert "VariableType" not in external_common.__all__
    config = RegridConfig(spatial_basis="cell_total", method=RegridMethod.CONSERVATIVE)
    assert config.spatial_basis is SpatialQuantityBasis.CELL_TOTAL
    with pytest.raises(ValueError, match="ambiguous"):
        RegridConfig(spatial_basis="extensive", method=RegridMethod.CONSERVATIVE)


def test_regrid_config_requires_an_explicit_spatial_basis():
    """No silent INTENSIVE default: the representation of an external field must be declared."""
    with pytest.raises(TypeError, match="spatial_basis"):
        RegridConfig()
    with pytest.raises(TypeError, match="spatial_basis"):
        RegridConfig(method=RegridMethod.CONSERVATIVE)


def test_unset_tolerance_is_the_one_regridder_default():
    """An unset RegridConfig tolerance is coupling.scales.defaults.conservation_tolerance, the
    default of every regridder built without one (validation.conservation is not consulted)."""
    from relic.parameters import get_global_manager

    pm = get_global_manager()
    default = float(pm.get("conservation_tolerance", namespace="coupling.scales.defaults", required=True))
    assert not hasattr(regrid_module, "get_conservation_tolerance")
    config = RegridConfig(spatial_basis=SpatialQuantityBasis.CELL_TOTAL, method=RegridMethod.CONSERVATIVE)
    RegridOperator(parameter_manager=pm)._apply_parameter_defaults(config)
    assert config.conservation_tolerance == default


@pytest.mark.parametrize(
    "switch", ["cache_weights", "use_parallel", "chunk_size", "preserve_extrema", "monotonic"]
)
def test_regrid_config_has_no_switch_without_an_effect_path(switch):
    """RegridOperator never acted on these settings (reuse is the content-keyed factory cache; no
    parallel or chunked path exists; the factory methods take no extrema or monotonicity option), so
    RegridConfig does not declare them and the parameter database has no coupling.external.regrid key."""
    import dataclasses

    from relic.parameters import get_global_manager

    assert switch not in {field.name for field in dataclasses.fields(RegridConfig)}
    assert get_global_manager().get(switch, namespace="coupling.external.regrid", required=False) is None


def test_regrid_requires_an_explicit_config():
    """regrid() does not build a default config (that would imply a default basis)."""
    operator = RegridOperator()
    grid = _strip_grid(1, 1.0)
    with pytest.raises(TypeError, match="config"):
        operator.regrid(torch.ones((1, 1)), grid, grid)


def _strip_grid(n_cells: int, cell_area: float) -> GridDescriptor:
    grid = GridDescriptor(
        grid_type="regular",
        coord_system="latlon",
        dims=(1, n_cells),
        x_coords=torch.zeros((1, n_cells)),
        y_coords=torch.zeros((1, n_cells)),
        cell_areas=torch.full((1, n_cells), cell_area, dtype=torch.float64),
    )
    return grid


@pytest.mark.parametrize(
    "basis, expected_error",
    [
        # cell totals: sum X_i = 12 kg on both sides
        (SpatialQuantityBasis.CELL_TOTAL, 0.0),
        # area densities: sum x_i A_i = (4 + 8) * 1 = 12 vs 12 * 2 = 24
        (SpatialQuantityBasis.AREA_DENSITY, 1.0),
    ],
)
def test_conservation_check_integrates_on_the_declared_basis(basis, expected_error):
    operator = RegridOperator()
    config = RegridConfig(spatial_basis=basis, check_conservation=True, conservation_tolerance=1e-6, method=RegridMethod.CONSERVATIVE)
    metrics = operator._compute_quality_metrics(
        torch.tensor([[4.0, 8.0]], dtype=torch.float64),
        torch.tensor([[12.0]], dtype=torch.float64),
        _strip_grid(2, 1.0),
        _strip_grid(1, 2.0),
        config,
    )
    assert metrics["conservation_error"] == pytest.approx(expected_error)


def test_regrid_auto_delegates_method_selection(monkeypatch):
    operator = RegridOperator()
    config = RegridConfig(
        method=RegridMethod.AUTO,
        spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
        check_conservation=True,
        conservation_tolerance=1e-6,
        positive_definite=False,
    )

    data = torch.ones((1, 1), dtype=torch.float32)
    source_grid = GridDescriptor(
        grid_type="regular",
        coord_system="latlon",
        dims=(1, 1),
        x_coords=torch.zeros((1, 1)),
        y_coords=torch.zeros((1, 1)),
        cell_areas=torch.ones((1, 1), dtype=torch.float64),
    )
    target_grid = GridDescriptor(
        grid_type="regular",
        coord_system="latlon",
        dims=(1, 1),
        x_coords=torch.zeros((1, 1)),
        y_coords=torch.zeros((1, 1)),
        cell_areas=torch.ones((1, 1), dtype=torch.float64),
    )

    captured = {}

    class DummyRegridder:
        def __init__(self):
            self.selected_method = "conservative"

        def upscale(self, tensor):
            captured["input_shape"] = tuple(tensor.shape)
            return tensor

    def fake_create(self, source_grid_arg, target_grid_arg, method=None, variable_metadata=None):
        captured["method"] = method
        captured["metadata"] = variable_metadata
        return DummyRegridder()

    monkeypatch.setattr(regrid_module.SpatialRegridderFactory, "create_regridder", fake_create, raising=True)

    result = operator.regrid(data, source_grid, target_grid, config=config)

    assert captured["method"] is None
    # The explicit spatial basis of the external field travels to the factory unchanged.
    assert captured["metadata"] == {
        "spatial_basis": "area_density",
        "require_conservation": True,
    }
    assert captured["input_shape"] == (1, 1)
    assert result.method_used == RegridMethod.CONSERVATIVE


def test_regrid_operator_runs_on_a_real_latlon_grid():
    """No monkeypatched grid attributes: coordinates, areas and output coords come from the grids."""
    import xarray as xr

    lat = np.array([-0.5, 0.5])
    lon = np.array([0.5, 1.5, 2.5, 3.5])
    source = xr.DataArray(
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "kg m-2 s-1"},
    )
    target = {
        "lat": np.array([0.0]),
        "lon": np.array([1.0, 3.0]),
        "lat_bounds": np.array([[-1.0, 1.0]]),
        "lon_bounds": np.array([[0.0, 2.0], [2.0, 4.0]]),
    }
    config = RegridConfig(
        spatial_basis=SpatialQuantityBasis.AREA_DENSITY,
        method=RegridMethod.CONSERVATIVE,
        check_conservation=True,
    )
    source_grid = {
        "lat": lat,
        "lon": lon,
        "lat_bounds": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "lon_bounds": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
    }
    result = RegridOperator().regrid(source, source_grid, target, config)
    # Symmetric 1-degree cells about the equator have equal spherical areas, so each 2 x 2
    # target cell is the plain mean of its four sources: (1+2+5+6)/4 and (3+4+7+8)/4.
    np.testing.assert_allclose(result.data.values, [[3.5, 5.5]], rtol=1e-6)
    np.testing.assert_allclose(result.data["lat"].values, [0.0])
    np.testing.assert_allclose(result.data["lon"].values, [1.0, 3.0])
    assert result.conservation_error < 1e-6
    assert result.method_used == RegridMethod.CONSERVATIVE


def _equator_regrid(values, *, basis, check_conservation, positive_definite):
    """Remap a 2 x 4 grid of 1-degree cells about the equator onto two 2 x 2-degree cells."""
    import xarray as xr

    lat = np.array([-0.5, 0.5])
    lon = np.array([0.5, 1.5, 2.5, 3.5])
    source = xr.DataArray(
        np.asarray(values, dtype=np.float64),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"units": "kg m-2 s-1"},
    )
    source_grid = {
        "lat": lat,
        "lon": lon,
        "lat_bounds": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "lon_bounds": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
    }
    target = {
        "lat": np.array([0.0]),
        "lon": np.array([1.0, 3.0]),
        "lat_bounds": np.array([[-1.0, 1.0]]),
        "lon_bounds": np.array([[0.0, 2.0], [2.0, 4.0]]),
    }
    config = RegridConfig(
        spatial_basis=basis,
        method=RegridMethod.CONSERVATIVE,
        check_conservation=check_conservation,
        positive_definite=positive_definite,
    )
    operator = RegridOperator()
    return operator, operator.regrid(source, source_grid, target, config)


# The second target cell is the mean of (-30, 4, 7, 8) = -2.75: the remap is negative there.
_NEGATIVE_TARGET_FIELD = [[1.0, 2.0, -30.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_a_positive_definite_clip_of_an_amount_is_disclosed_in_the_ledger():
    """The clip of an amount is returned as a ledger with the StateManager clip_loss sign:
    amount before the clip = amount after the clip + clip amount (negative: the clip adds material)."""
    operator, result = _equator_regrid(
        _NEGATIVE_TARGET_FIELD,
        basis=SpatialQuantityBasis.AREA_DENSITY,
        check_conservation=False,
        positive_definite=True,
    )
    ledger = result.positive_definite_ledger
    assert ledger["clipped_cells"] == 1
    source_areas = torch.as_tensor(np.asarray(operator._compute_grid_areas(result.source_grid)), dtype=torch.float64)
    target_areas = torch.as_tensor(np.asarray(operator._compute_grid_areas(result.target_grid)), dtype=torch.float64)
    returned = torch.as_tensor(result.data.values, dtype=torch.float64)
    assert float(returned.min()) == 0.0
    source_amount = float((torch.as_tensor(_NEGATIVE_TARGET_FIELD, dtype=torch.float64) * source_areas).sum())
    # The first-order remap covers every source cell, so the amount before the clip is the source amount.
    assert ledger["amount_before"] == pytest.approx(source_amount, rel=1e-6)
    assert ledger["amount_after"] == pytest.approx(float((returned * target_areas).sum()), rel=1e-12)
    assert ledger["clip_amount"] < 0.0
    assert ledger["amount_before"] == pytest.approx(ledger["amount_after"] + ledger["clip_amount"], rel=1e-12)


def test_a_positive_definite_clip_of_an_amount_raises_when_conservation_is_required():
    """With check_conservation the clip cannot change the amount silently: the call raises."""
    with pytest.raises(ValueError, match="positive_definite"):
        _equator_regrid(
            _NEGATIVE_TARGET_FIELD,
            basis=SpatialQuantityBasis.AREA_DENSITY,
            check_conservation=True,
            positive_definite=True,
        )


def test_a_positive_definite_clip_of_an_intensive_field_reports_its_cells():
    """An intensive field has no amount: the ledger counts the clipped cells only."""
    _operator, result = _equator_regrid(
        _NEGATIVE_TARGET_FIELD,
        basis=SpatialQuantityBasis.INTENSIVE,
        check_conservation=False,
        positive_definite=True,
    )
    assert result.positive_definite_ledger["clipped_cells"] == 1
    assert "clip_amount" not in result.positive_definite_ledger


def test_regrid_reports_the_missing_fraction_and_has_no_missing_data_switch():
    """RegridConfig has no handle_missing_data or missing_threshold (the external adapter's quality
    control owns the missing-data rule); the result reports the NaN fraction of the source field."""
    import dataclasses

    names = {field.name for field in dataclasses.fields(RegridConfig)}
    assert "handle_missing_data" not in names
    assert "missing_threshold" not in names
    values = [[1.0, np.nan, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    _operator, result = _equator_regrid(
        values, basis=SpatialQuantityBasis.AREA_DENSITY, check_conservation=False, positive_definite=False
    )
    assert result.missing_data_fraction == pytest.approx(1.0 / 8.0)
    assert result.positive_definite_ledger is None


def test_conservative_regrid_needs_declared_cell_bounds():
    """A source grid without cell bounds is not given spacing-inferred cells."""
    import xarray as xr

    lat = np.array([-0.5, 0.5])
    lon = np.array([0.5, 1.5, 2.5, 3.5])
    source = xr.DataArray(np.ones((2, 4)), dims=("lat", "lon"), attrs={"units": "kg m-2 s-1"})
    target = {
        "lat": np.array([0.0]),
        "lon": np.array([1.0, 3.0]),
        "lat_bounds": np.array([[-1.0, 1.0]]),
        "lon_bounds": np.array([[0.0, 2.0], [2.0, 4.0]]),
    }
    config = RegridConfig(spatial_basis=SpatialQuantityBasis.AREA_DENSITY, method=RegridMethod.CONSERVATIVE)
    with pytest.raises(ValueError, match="bounds"):
        RegridOperator().regrid(source, {"lat": lat, "lon": lon}, target, config)


def test_dataset_grid_coordinates_are_identified_by_cf_attributes():
    """CF units (with consistent standard_name/axis) identify latitude and longitude, not names."""
    import xarray as xr

    operator = RegridOperator()
    # 2-D curvilinear coordinates with non-standard names (e.g. NEMO nav_lat/nav_lon).
    curvilinear = xr.Dataset(
        coords={
            "nav_lat": (("y", "x"), np.array([[10.0, 10.0], [11.0, 11.0]]), {"units": "degrees_north"}),
            "nav_lon": (("y", "x"), np.array([[20.0, 21.0], [20.0, 21.0]]), {"units": "degree_east"}),
        }
    )
    grid = operator._extract_grid_from_dataset(curvilinear)
    assert grid.grid_type == "curvilinear"
    np.testing.assert_allclose(grid.x_coords.numpy(), [[20.0, 21.0], [20.0, 21.0]])
    np.testing.assert_allclose(grid.y_coords.numpy(), [[10.0, 10.0], [11.0, 11.0]])
    assert grid.x_bounds is None and grid.y_bounds is None

    # Projection coordinates named x/y (metres) are not longitude/latitude.
    projected = xr.Dataset(
        coords={
            "x": ("x", np.array([0.0, 1000.0]), {"units": "m", "axis": "X"}),
            "y": ("y", np.array([0.0, 1000.0]), {"units": "m", "axis": "Y"}),
        }
    )
    with pytest.raises(ValueError, match="CF latitude"):
        operator._extract_grid_from_dataset(projected)

    # Rectilinear coordinates: bounds come from the variables named by the bounds attribute.
    rectilinear = xr.Dataset(
        {
            "lat_bnds": (("lat", "nv"), np.array([[-1.0, 0.0], [0.0, 1.0]])),
            "lon_bnds": (("lon", "nv"), np.array([[0.0, 1.0], [1.0, 2.5], [2.5, 4.0]])),
        },
        coords={
            "lat": ("lat", np.array([-0.5, 0.5]), {"units": "degrees_north", "bounds": "lat_bnds"}),
            "lon": ("lon", np.array([0.5, 1.75, 3.25]), {"units": "degrees_east", "bounds": "lon_bnds"}),
        },
    )
    grid = operator._extract_grid_from_dataset(rectilinear)
    np.testing.assert_allclose(grid.x_bounds.numpy(), [[0.0, 1.0], [1.0, 2.5], [2.5, 4.0]])
    np.testing.assert_allclose(grid.y_bounds.numpy(), [[-1.0, 0.0], [0.0, 1.0]])
    no_bounds = xr.Dataset(
        coords={
            "latitude": ("latitude", np.array([-0.5, 0.5]), {"units": "degrees_north"}),
            "longitude": ("longitude", np.array([0.5, 1.5]), {"units": "degrees_east"}),
        }
    )
    grid = operator._extract_grid_from_dataset(no_bounds)
    assert grid.x_bounds is None and grid.y_bounds is None

    # Inconsistent CF attributes raise.
    inconsistent = xr.Dataset(
        coords={
            "lat": ("lat", np.array([0.0]), {"units": "degrees_north", "axis": "X"}),
            "lon": ("lon", np.array([0.0]), {"units": "degrees_east"}),
        }
    )
    with pytest.raises(ValueError, match="axis"):
        operator._extract_grid_from_dataset(inconsistent)
