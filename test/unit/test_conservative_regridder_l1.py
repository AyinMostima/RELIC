"""
Conservative Regridder L1 Tests

Exercises the production ConservativeRegridder on regular grids to confirm
mass/energy conservation, diagnostics, and error handling without relying on
mocked components.

Key checks:
- Upscale/downscale operations preserve integrated totals within tolerance.
- Boundary and irregular data are handled without shape errors.
- Diagnostics expose expected metadata for downstream auditing.
- CUDA execution path remains functional when a GPU is available.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Tuple

import pytest
import torch

import relic.coupling.scales.spatial.regrid_conservative as conservative_module

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from relic.coupling.scales.spatial.regrid_conservative import ConservativeRegridder
from relic.coupling.scales.spatial.grid_utils import GridDescriptor, GridUtils
from relic.utils.physical_constants import get_physical_constant


def create_regular_grids(
    source_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    *,
    device: torch.device = torch.device("cpu"),
) -> Tuple[GridDescriptor, GridDescriptor]:
    """Create simple Cartesian grids covering the same domain."""
    grid_utils = GridUtils(device=device)
    source_grid = grid_utils.create_regular_grid(
        x_min=0.0,
        x_max=1000.0,
        y_min=0.0,
        y_max=1000.0,
        nx=source_shape[1],
        ny=source_shape[0],
        coord_system="cartesian",
    )
    target_grid = grid_utils.create_regular_grid(
        x_min=0.0,
        x_max=1000.0,
        y_min=0.0,
        y_max=1000.0,
        nx=target_shape[1],
        ny=target_shape[0],
        coord_system="cartesian",
    )
    return source_grid, target_grid


def _strip_grid(x_edges, cell_areas) -> GridDescriptor:
    """One-row cartesian grid with explicit x bounds on y in [0, 1] (areas in m2)."""
    nx = len(x_edges) - 1
    x_bounds = torch.tensor([[x_edges[i], x_edges[i + 1]] for i in range(nx)], dtype=torch.float64)
    return GridDescriptor(
        grid_type="regular",
        coord_system="cartesian",
        dims=(1, nx),
        x_coords=x_bounds.mean(dim=1),
        y_coords=torch.tensor([0.5], dtype=torch.float64),
        x_bounds=x_bounds,
        y_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        cell_areas=torch.tensor(cell_areas, dtype=torch.float64).reshape(1, nx),
    )


def _partial_coverage_regridder(target_edges, target_areas, **kwargs) -> ConservativeRegridder:
    """Sources [0,1] and [1,2] (1 m2 each) mapped to one target cell."""
    return ConservativeRegridder(
        source_grid=_strip_grid([0.0, 1.0, 2.0], [1.0, 1.0]),
        target_grid=_strip_grid(target_edges, target_areas),
        use_sparse_weights=False,
        device=torch.device("cpu"),
        **kwargs,
    )


class TestConservativeRegridderExplicitBasis:

    def test_missing_cell_bounds_are_not_inferred_from_the_coordinate_spacing(self) -> None:
        """Centres 0.5, 1.5, 3.0 do not determine the cells: the regridder raises without bounds."""
        bare = GridDescriptor(
            grid_type="regular",
            coord_system="cartesian",
            dims=(1, 3),
            x_coords=torch.tensor([0.5, 1.5, 3.0], dtype=torch.float64),
            y_coords=torch.tensor([0.5], dtype=torch.float64),
        )
        with pytest.raises(ValueError, match="bounds"):
            ConservativeRegridder(
                source_grid=bare,
                target_grid=_strip_grid([0.0, 4.0], [4.0]),
                use_sparse_weights=False,
                device=torch.device("cpu"),
                quantity_basis="area_density",
            )
        assert not hasattr(ConservativeRegridder, "_compute_bounds")
    """Covered-support conservation, explicit basis, autograd ratio, per-call range clamp."""

    def test_partial_coverage_keeps_the_uncovered_amount_out_of_the_target(self) -> None:
        # Densities 4 and 8 kg m-2; target [0, 1.5] (1.5 m2) covers cell 1 fully and cell 2 half.
        regridder = _partial_coverage_regridder([0.0, 1.5], [1.5])
        density = torch.tensor([[4.0, 8.0]], dtype=torch.float64, requires_grad=True)
        target = regridder.downscale(density, quantity_basis="area_density")
        torch.testing.assert_close(target, torch.tensor([[8.0 / 1.5]], dtype=torch.float64))
        assert float((target.detach() * 1.5).sum()) == pytest.approx(8.0)
        report = regridder.last_regrid_report
        assert report["quantity_basis"] == "area_density"
        assert report["output_basis"] == "area_density"
        assert report["uncovered_source_amount"] == pytest.approx(4.0)
        assert report["relative_residual"] < 1e-12
        # Gradient of the implemented map y = (x_1 * 1 + x_2 * 0.5) / 1.5.
        target.sum().backward()
        torch.testing.assert_close(
            density.grad, torch.tensor([[1.0 / 1.5, 0.5 / 1.5]], dtype=torch.float64)
        )

    def test_cell_totals_are_split_by_overlap_fraction(self) -> None:
        regridder = _partial_coverage_regridder([0.0, 1.5], [1.5], quantity_basis="cell_total")
        totals = torch.tensor([[4.0, 8.0]], dtype=torch.float64)
        target = regridder.downscale(totals)
        torch.testing.assert_close(target, torch.tensor([[8.0]], dtype=torch.float64))
        assert regridder.last_regrid_report["output_basis"] == "cell_total"
        assert regridder.last_regrid_report["uncovered_source_amount"] == pytest.approx(4.0)

    def test_missing_or_ambiguous_basis_is_rejected(self) -> None:
        regridder = _partial_coverage_regridder([0.0, 1.5], [1.5])
        data = torch.tensor([[4.0, 8.0]], dtype=torch.float64)
        with pytest.raises(ValueError, match="quantity_basis"):
            regridder.downscale(data)
        with pytest.raises(ValueError, match="ambiguous"):
            regridder.downscale(data, quantity_basis="extensive")

    def test_range_clamp_is_a_per_call_argument_with_a_reported_residual(self) -> None:
        # Target [0, 4] (4 m2) is only half covered: y = (4 + 8) / 4 = 3 kg m-2 < min(x) = 4.
        regridder = _partial_coverage_regridder([0.0, 4.0], [4.0], quantity_basis="area_density")
        data = torch.tensor([[4.0, 8.0]], dtype=torch.float64)
        governed = regridder.forward({"data": data, "operation": "downscale", "preserve": "none"}, time=0.0, dt=1.0)
        torch.testing.assert_close(governed["regridded_data"], torch.tensor([[4.0]], dtype=torch.float64))
        # The clamp adds (4 - 3) * 4 m2 = 4 kg; StateManager clip_loss sign: -4.
        assert governed["diagnostics"]["range_clip_residual"] == pytest.approx(-4.0)
        # A later direct call is not affected by the governed call's preserve setting.
        direct = regridder.downscale(data)
        torch.testing.assert_close(direct, torch.tensor([[3.0]], dtype=torch.float64))
        assert regridder.last_regrid_report["range_clip_residual"] == 0.0
        mass = regridder.forward({"data": data, "operation": "downscale", "preserve": "mass"}, time=0.0, dt=1.0)
        torch.testing.assert_close(mass["regridded_data"], torch.tensor([[3.0]], dtype=torch.float64))
        clamped = regridder.downscale(data, range_clamp=True)
        torch.testing.assert_close(clamped, torch.tensor([[4.0]], dtype=torch.float64))
        assert regridder.last_regrid_report["range_clip_residual"] == pytest.approx(-4.0)

    def test_regrid_does_not_push_uncovered_mass_into_covered_targets(self) -> None:
        regridder = _partial_coverage_regridder(
            [0.0, 1.5], [1.5], quantity_basis="area_density", enforce_conservation=False
        )
        data = torch.tensor([[4.0, 8.0]], dtype=torch.float64)
        result = regridder.regrid(data, preserve="mass", method="conservative")
        torch.testing.assert_close(result, torch.tensor([[8.0 / 1.5]], dtype=torch.float64))
        assert regridder.last_regrid_report["uncovered_source_amount"] == pytest.approx(4.0)

    def test_alternate_grids_use_the_weights_of_their_own_geometry(self) -> None:
        """A forward call with other grids of equal dims and equal centres gets their own overlaps:
        totals 4 and 8 kg give 12 kg on [0, 2] and 6 kg on [0.5, 1.5] (half of each source)."""
        regridder = _partial_coverage_regridder([0.0, 2.0], [2.0], quantity_basis="cell_total")
        data = torch.tensor([[4.0, 8.0]], dtype=torch.float64)
        wide = regridder.forward({"data": data, "operation": "downscale"}, time=0.0, dt=1.0)
        torch.testing.assert_close(wide["regridded_data"], torch.tensor([[12.0]], dtype=torch.float64))
        narrow = regridder.forward(
            {"data": data, "operation": "downscale", "target_grid": _strip_grid([0.5, 1.5], [1.0])},
            time=0.0,
            dt=1.0,
        )
        torch.testing.assert_close(narrow["regridded_data"], torch.tensor([[6.0]], dtype=torch.float64))
        # No weight cache keyed by a part of the geometry exists on the regridder.
        for name in ("_weight_cache", "_compute_cache_key", "get_cached_weights", "cache_weights"):
            assert not hasattr(regridder, name), name


class TestCoveredSupportLedgerAcceptance:
    """The covered-support ledger is accepted by the shared rule conservation_checker.ledger_acceptance."""

    @staticmethod
    def _float32_totals(n_cells: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(7)
        return (torch.rand((1, n_cells), generator=generator, dtype=torch.float64) * 1.0e-3 + 1.0).to(torch.float32)

    @staticmethod
    def _strip_regridder(n_cells: int) -> ConservativeRegridder:
        edges = [float(index) for index in range(n_cells + 1)]
        return ConservativeRegridder(
            source_grid=_strip_grid(edges, [1.0] * n_cells),
            target_grid=_strip_grid([0.0, float(n_cells)], [float(n_cells)]),
            use_sparse_weights=False,
            device=torch.device("cpu"),
            quantity_basis="cell_total",
            enforce_conservation=True,
        )

    def test_float32_rounding_is_accepted_and_a_wrong_result_is_rejected(self, monkeypatch) -> None:
        from relic.coupling.core import conservation_checker

        calls = []
        original_rule = conservation_checker.ledger_acceptance

        def spy(**kwargs):
            calls.append(kwargs["rounding_dtype"])
            return original_rule(**kwargs)

        monkeypatch.setattr(conservation_checker, "ledger_acceptance", spy)
        regridder = self._strip_regridder(64)
        values = self._float32_totals(64)
        result = regridder.regrid(values, method="conservative")
        report = regridder.last_regrid_report
        assert result.dtype == torch.float32
        # The float32 rounding of the returned total exceeds the float64-scale tolerance ...
        assert report["relative_residual"] > regridder.conservation_tolerance
        # ... and is accepted as rounding of float32 by the one shared rule.
        assert report["accepted"] is True
        assert report["rounding_allowance"] > 0.0
        assert report["absolute_residual"] <= report["accepted_residual"]
        assert calls and all(dtype == torch.float32 for dtype in calls)
        # A wrong (doubled) result is still rejected on the same enforced path.
        original_map = ConservativeRegridder._map_field
        monkeypatch.setattr(
            ConservativeRegridder,
            "_map_field",
            lambda self, *args, **kwargs: 2.0 * original_map(self, *args, **kwargs),
        )
        with pytest.raises(RuntimeError, match="covered-support residual"):
            regridder.regrid(values, method="conservative")
        assert regridder.last_regrid_report["accepted"] is False


class TestCoveredSupportWeightCheck:
    """Construction checks the weights on the covered support; partial coverage is reported."""

    def test_partial_coverage_is_reported_not_rejected(self) -> None:
        # Target [0, 1.5] covers source [0, 1] fully and source [1, 2] half, with enforcement on.
        regridder = _partial_coverage_regridder(
            [0.0, 1.5], [1.5], quantity_basis="cell_total", enforce_conservation=True
        )
        checks = regridder.verify_conservation()
        assert checks["max_source_overcoverage"] == 0.0
        assert checks["max_target_overcoverage"] == 0.0
        coverage = regridder.coverage_report
        assert coverage["fully_covered_source_cells"] == 1
        assert coverage["partially_covered_source_cells"] == 1
        assert coverage["uncovered_source_cells"] == 0
        assert coverage["min_source_coverage"] == pytest.approx(0.5)
        result = regridder.regrid(torch.tensor([[4.0, 8.0]], dtype=torch.float64), method="conservative")
        torch.testing.assert_close(result, torch.tensor([[8.0]], dtype=torch.float64))
        assert regridder.last_regrid_report["uncovered_source_amount"] == pytest.approx(4.0)
        assert regridder.last_regrid_report["relative_residual"] < 1e-12

    def test_weights_that_exceed_a_cell_raise_at_construction(self) -> None:
        # Target cells [0, 1.5] and [0.5, 2] overlap: source [0, 1] would be counted 1.5 times.
        with pytest.raises(ValueError, match="covered fraction"):
            ConservativeRegridder(
                source_grid=_strip_grid([0.0, 1.0, 2.0], [1.0, 1.0]),
                target_grid=GridDescriptor(
                    grid_type="regular",
                    coord_system="cartesian",
                    dims=(1, 2),
                    x_coords=torch.tensor([0.75, 1.25], dtype=torch.float64),
                    y_coords=torch.tensor([0.5], dtype=torch.float64),
                    x_bounds=torch.tensor([[0.0, 1.5], [0.5, 2.0]], dtype=torch.float64),
                    y_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
                    cell_areas=torch.tensor([[1.5, 1.5]], dtype=torch.float64),
                ),
                use_sparse_weights=False,
                device=torch.device("cpu"),
                quantity_basis="cell_total",
            )


def _latlon_grid(lon_edges, lat_edges, cell_areas=None) -> GridDescriptor:
    """Regular geographic grid with explicit lon/lat cell bounds [degrees]."""
    x_bounds = torch.tensor([[lon_edges[i], lon_edges[i + 1]] for i in range(len(lon_edges) - 1)], dtype=torch.float64)
    y_bounds = torch.tensor([[lat_edges[j], lat_edges[j + 1]] for j in range(len(lat_edges) - 1)], dtype=torch.float64)
    return GridDescriptor(
        grid_type="regular",
        coord_system="latlon",
        dims=(y_bounds.shape[0], x_bounds.shape[0]),
        x_coords=x_bounds.mean(dim=1),
        y_coords=y_bounds.mean(dim=1),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        cell_areas=cell_areas,
    )


def _sphere_cell_area(lon0, lon1, lat0, lat1) -> float:
    """A = R^2 (lambda_1 - lambda_0) (sin phi_1 - sin phi_0), R from the physical-constants SSOT."""
    radius = float(get_physical_constant("earth_radius"))
    return radius**2 * math.radians(lon1 - lon0) * (math.sin(math.radians(lat1)) - math.sin(math.radians(lat0)))


class TestSphericalLatLonOverlaps:
    """Geographic lat-lon cells overlap on the sphere (Jones 1999): the overlap of two lat-lon cells
    is a lat-lon cell, A = R^2 dlambda (sin phi_2 - sin phi_1); planar grids stay planar."""

    def test_overlaps_of_latlon_cells_are_spherical_cells(self) -> None:
        source = _latlon_grid([0.0, 10.0], [0.0, 30.0, 60.0])
        target = _latlon_grid([0.0, 10.0], [15.0, 45.0])
        regridder = ConservativeRegridder(
            source_grid=source,
            target_grid=target,
            use_sparse_weights=False,
            device=torch.device("cpu"),
            quantity_basis="cell_total",
        )
        sin = lambda degrees: math.sin(math.radians(degrees))  # noqa: E731
        expected_weights = torch.tensor(
            [[(sin(30.0) - sin(15.0)) / (sin(30.0) - sin(0.0))], [(sin(45.0) - sin(30.0)) / (sin(60.0) - sin(30.0))]],
            dtype=torch.float64,
        )
        weights = regridder.weight_matrix.to_dense() if regridder.weight_matrix.is_sparse else regridder.weight_matrix
        torch.testing.assert_close(weights, expected_weights, rtol=1e-12, atol=0.0)
        source_areas = torch.tensor([_sphere_cell_area(0, 10, 0, 30), _sphere_cell_area(0, 10, 30, 60)], dtype=torch.float64)
        torch.testing.assert_close(regridder.source_areas.reshape(-1), source_areas, rtol=1e-12, atol=0.0)
        target_area = _sphere_cell_area(0, 10, 15, 45)
        torch.testing.assert_close(
            regridder.target_areas.reshape(-1), torch.tensor([target_area], dtype=torch.float64), rtol=1e-12, atol=0.0
        )
        # The covered parts of the source cells are exactly the target cell on the sphere.
        covered = float((source_areas * expected_weights.reshape(-1)).sum())
        assert covered == pytest.approx(target_area, rel=1e-12)
        # Cell totals: Y = sum_i X_i W_i; the ledger closes on the covered support.
        totals = torch.tensor([3.0, 5.0], dtype=torch.float64)
        mapped = regridder.downscale(totals)
        torch.testing.assert_close(mapped, (totals * expected_weights.reshape(-1)).sum().reshape(1))
        assert regridder.last_regrid_report["relative_residual"] < 1e-12
        # Intensive values: the spherical area-weighted mean over the covered parts.
        values = torch.tensor([1.0, 2.0], dtype=torch.float64)
        mean = regridder.downscale(values, quantity_basis="intensive")
        weights_area = source_areas * expected_weights.reshape(-1)
        torch.testing.assert_close(mean, ((values * weights_area).sum() / weights_area.sum()).reshape(1))

    def test_declared_latlon_areas_must_be_the_spherical_cell_areas(self) -> None:
        lon_edges, lat_edges = [0.0, 10.0], [0.0, 30.0, 60.0]
        spherical = GridUtils(device=torch.device("cpu")).compute_grid_areas(_latlon_grid(lon_edges, lat_edges))
        ConservativeRegridder(
            source_grid=_latlon_grid(lon_edges, lat_edges, cell_areas=spherical),
            target_grid=_latlon_grid([0.0, 10.0], [0.0, 60.0]),
            use_sparse_weights=False,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )
        degree_squared = torch.tensor([[300.0], [300.0]], dtype=torch.float64).reshape(2, 1)
        with pytest.raises(ValueError, match="spherical"):
            ConservativeRegridder(
                source_grid=_latlon_grid(lon_edges, lat_edges, cell_areas=degree_squared),
                target_grid=_latlon_grid([0.0, 10.0], [0.0, 60.0]),
                use_sparse_weights=False,
                device=torch.device("cpu"),
                quantity_basis="area_density",
            )

    def test_geometries_are_not_mixed_or_reinterpreted(self) -> None:
        with pytest.raises(ValueError, match="geometry"):
            ConservativeRegridder(
                source_grid=_latlon_grid([0.0, 10.0], [0.0, 30.0]),
                target_grid=_strip_grid([0.0, 10.0], [300.0]),
                use_sparse_weights=False,
                device=torch.device("cpu"),
                quantity_basis="cell_total",
            )
        with pytest.raises(ValueError, match="latitude"):
            ConservativeRegridder(
                source_grid=_latlon_grid([0.0, 10.0], [0.0, 60.0, 120.0]),
                target_grid=_latlon_grid([0.0, 10.0], [0.0, 120.0]),
                use_sparse_weights=False,
                device=torch.device("cpu"),
                quantity_basis="cell_total",
            )

    def test_planar_grids_stay_planar(self) -> None:
        source = GridDescriptor(
            grid_type="regular",
            coord_system="cartesian",
            dims=(2, 1),
            x_coords=torch.tensor([5.0], dtype=torch.float64),
            y_coords=torch.tensor([15.0, 45.0], dtype=torch.float64),
            x_bounds=torch.tensor([[0.0, 10.0]], dtype=torch.float64),
            y_bounds=torch.tensor([[0.0, 30.0], [30.0, 60.0]], dtype=torch.float64),
        )
        target = GridDescriptor(
            grid_type="regular",
            coord_system="cartesian",
            dims=(1, 1),
            x_coords=torch.tensor([5.0], dtype=torch.float64),
            y_coords=torch.tensor([30.0], dtype=torch.float64),
            x_bounds=torch.tensor([[0.0, 10.0]], dtype=torch.float64),
            y_bounds=torch.tensor([[15.0, 45.0]], dtype=torch.float64),
        )
        regridder = ConservativeRegridder(
            source_grid=source,
            target_grid=target,
            use_sparse_weights=False,
            device=torch.device("cpu"),
            quantity_basis="cell_total",
        )
        weights = regridder.weight_matrix.to_dense() if regridder.weight_matrix.is_sparse else regridder.weight_matrix
        torch.testing.assert_close(weights, torch.tensor([[0.5], [0.5]], dtype=torch.float64))
        torch.testing.assert_close(
            regridder.source_areas.reshape(-1), torch.tensor([300.0, 300.0], dtype=torch.float64)
        )


class TestConservativeRegridderL1:
    """Suite of L1 checks for conservative regridding."""

    def test_initialization_cpu(self) -> None:
        source_grid, target_grid = create_regular_grids((10, 10), (20, 20))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        assert regridder.source_grid.dims == (10, 10)
        assert regridder.target_grid.dims == (20, 20)
        assert regridder.device.type == "cpu"

    def test_earth_radius_uses_physical_constant_ssot(self, monkeypatch) -> None:
        """Earth radius is structural input, not an ordinary Functor parameter."""
        requested = []

        def fake_get_physical_constant(name):
            requested.append(name)
            return 6371000.0

        monkeypatch.setattr(
            conservative_module,
            "get_physical_constant",
            fake_get_physical_constant,
        )
        source_grid, target_grid = create_regular_grids((2, 2), (2, 2))

        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
        )

        assert regridder.earth_radius == 6371000.0
        assert "earth_radius" not in regridder.metadata.parameters
        assert requested == ["earth_radius"]

    def test_forward_upscaling_mass_conservation(self) -> None:
        source_grid, target_grid = create_regular_grids((10, 10), (20, 20))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.ones(10, 10)
        state = {"data": source_data, "operation": "upscale"}
        result = regridder.forward(state, time=0.0, dt=1.0)

        assert "regridded_data" in result
        assert result["regridded_data"].shape == (20, 20)
        assert result["conservation_error"] < 1e-10

    def test_forward_downscaling_mass_conservation(self) -> None:
        source_grid, target_grid = create_regular_grids((20, 20), (10, 10))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.ones(20, 20)
        state = {"data": source_data, "operation": "downscale"}
        result = regridder.forward(state, time=0.0, dt=1.0)

        assert "regridded_data" in result
        assert result["regridded_data"].shape == (10, 10)
        assert result["conservation_error"] < 1e-6

    def test_auto_detect_downscale_and_conservation(self) -> None:
        source_grid, target_grid = create_regular_grids((2, 2), (1, 1))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = regridder.forward({"data": source_data}, time=0.0, dt=1.0)

        assert result["diagnostics"]["operation"] == "downscale"
        assert result["regridded_data"].shape == (1, 1)
        source_mass = torch.sum(source_data.view(-1) * regridder.source_areas.view(-1))
        expected_value = source_mass / torch.sum(regridder.target_areas)
        assert torch.allclose(result["regridded_data"], expected_value.view(1, 1), atol=1e-10)
        assert result["conservation_error"] < regridder.conservation_tolerance

    def test_auto_detect_upscale_and_conservation(self) -> None:
        source_grid, target_grid = create_regular_grids((1, 1), (2, 2))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.tensor([[5.0]])
        result = regridder.forward({"data": source_data}, time=0.0, dt=1.0)

        assert result["diagnostics"]["operation"] == "upscale"
        assert result["regridded_data"].shape == (2, 2)
        expected_value = source_data.item()
        assert torch.allclose(result["regridded_data"], torch.full((2, 2), expected_value), atol=1e-10)
        assert result["conservation_error"] < regridder.conservation_tolerance

    def test_mass_conservation_with_gradient_data(self) -> None:
        source_grid, target_grid = create_regular_grids((10, 10), (20, 20))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.arange(0, 100, dtype=torch.float32).view(10, 10)
        state = {"data": source_data, "operation": "upscale"}
        result = regridder.forward(state, time=0.0, dt=1.0)

        assert result["conservation_error"] < 1e-10

    def test_area_weighted_interpolation_range(self) -> None:
        source_grid, target_grid = create_regular_grids((5, 5), (10, 10))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.rand(5, 5)
        result = regridder.forward({"data": source_data, "operation": "upscale"}, time=0.0, dt=1.0)

        min_source, max_source = source_data.min(), source_data.max()
        min_target, max_target = result["regridded_data"].min(), result["regridded_data"].max()
        assert min_source <= min_target + 1e-6
        assert max_source >= max_target - 1e-6

    def test_boundary_condition_handling(self) -> None:
        source_grid, target_grid = create_regular_grids((10, 10), (20, 20))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.zeros(10, 10)
        source_data[0, :] = 1.0
        source_data[-1, :] = 2.0
        source_data[:, 0] = 3.0
        source_data[:, -1] = 4.0

        result = regridder.forward({"data": source_data, "operation": "upscale"}, time=0.0, dt=1.0)
        assert result["regridded_data"].shape == (20, 20)

    def test_nan_input_propagation(self) -> None:
        source_grid, target_grid = create_regular_grids((10, 10), (20, 20))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        nan_data = torch.full((10, 10), float("nan"))
        result = regridder.forward({"data": nan_data, "operation": "upscale"}, time=0.0, dt=1.0)

        assert torch.isnan(result["regridded_data"]).all()

    def test_diagnostics_presence(self) -> None:
        source_grid, target_grid = create_regular_grids((10, 10), (20, 20))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.randn(10, 10)
        result = regridder.forward({"data": source_data, "operation": "upscale"}, time=0.0, dt=1.0)

        diagnostics = result["diagnostics"]
        expected_fields = {"source_shape", "target_shape", "conservation_error", "method", "preserve"}
        assert expected_fields.issubset(diagnostics.keys())

    def test_cuda_execution_path(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device test")

        device = torch.device("cuda")
        source_grid, target_grid = create_regular_grids((5, 5), (10, 10), device=device)
        regridder = ConservativeRegridder(
            source_grid=source_grid, target_grid=target_grid, device=device, quantity_basis="area_density"
        )

        source_data = torch.randn(5, 5, device=device)
        result = regridder.forward({"data": source_data, "operation": "upscale"}, time=0.0, dt=1.0)

        assert result["regridded_data"].device.type == "cuda"

    def test_gradient_support(self) -> None:
        source_grid, target_grid = create_regular_grids((5, 5), (10, 10))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.randn(5, 5, requires_grad=True)
        result = regridder.forward({"data": source_data, "operation": "upscale"}, time=0.0, dt=1.0)
        loss = result["regridded_data"].sum()
        loss.backward()

        assert source_data.grad is not None

    def test_multiple_variables_sequence(self) -> None:
        source_grid, target_grid = create_regular_grids((10, 10), (20, 20))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        variables: Dict[str, torch.Tensor] = {
            "temperature": torch.randn(10, 10),
            "moisture": torch.rand(10, 10),
            "carbon_flux": torch.randn(10, 10),
        }

        for var_name, data in variables.items():
            result = regridder.forward({"data": data, "operation": "upscale"}, time=0.0, dt=1.0)
            assert result["regridded_data"].shape == (20, 20)
            assert "diagnostics" in result

    def test_total_flux_sign_preserved(self) -> None:
        source_grid, target_grid = create_regular_grids((10, 10), (20, 20))
        regridder = ConservativeRegridder(
            source_grid=source_grid,
            target_grid=target_grid,
            device=torch.device("cpu"),
            quantity_basis="area_density",
        )

        source_data = torch.randn(10, 10) - 0.5
        result = regridder.forward({"data": source_data, "operation": "upscale"}, time=0.0, dt=1.0)

        source_mass = torch.sum(source_data.view(-1) * regridder.source_areas.view(-1))
        target_mass = torch.sum(result["regridded_data"].view(-1) * regridder.target_areas.view(-1))
        relative_error = torch.abs(source_mass - target_mass) / torch.abs(source_mass)
        assert relative_error < regridder.conservation_tolerance

