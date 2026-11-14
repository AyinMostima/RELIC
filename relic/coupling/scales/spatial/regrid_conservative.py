"""
Conservative Regridding Module

This module implements first-order conservative remapping for spatial scale
coupling in the RELIC framework. Based on ESMF/SCRIP methodologies with
PyTorch integration for end-to-end differentiability.

Conservative remapping preserves the amount of a field on the covered support
while transferring it between spatial grids. Every remap declares one explicit
SpatialQuantityBasis (relic.coupling.scales.types): intensive, cell_total or
area_density.

Mathematical Framework:
    Weights: W[i,J] = A_iJ / A_i (overlap fraction of source cell i), never
    renormalized, so sum_J W[i,J] = f_i <= 1 is the covered fraction of cell i.
    area_density: y_J = sum_i x_i A_i W[i,J] / A_J            (destarea)
    cell_total:   Y_J = sum_i X_i W[i,J]
    intensive:    y_J = sum_i x_i A_i W[i,J] / C_J, C_J = sum_i A_i W[i,J]  (fracarea)
    Covered-support ledger: sum_J Y_J = sum_i X_i f_i (amounts); the source
    amount outside the target support, sum_i X_i (1 - f_i), is reported and is
    never rescaled into the covered targets.

Classes:
    ConservativeRegridder: Main conservative remapping implementation

References:
    - Jones, P.W. (1999). First- and second-order conservative remapping schemes
    - ESMF Reference Manual: Regridding and interpolation
    - SCRIP (Spherical Coordinate Remapping and Interpolation Package)
    - Kritsikis, E. et al. (2017). Conservative interpolation between general spherical meshes
    - Ullrich, P.A. et al. (2009). High-order finite-volume methods for the shallow-water equations
"""

import logging
import math
from collections.abc import Mapping
from typing import Dict, Optional, Tuple, Any, Union

import torch
from torch import Tensor
import torch.sparse

from ....core.functor import ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
from ....parameters.namespaces import ParameterNamespace, resolve_namespace
from ....utils.physical_constants import get_physical_constant
from ...core import conservation_checker
from ...core.conservation_checker import (
    coarsest_floating_dtype,
    ledger_relative_scale,
    require_relative_tolerance,
)
from ..types import SpatialQuantityBasis, convert_spatial_basis

logger = logging.getLogger(__name__)
from .grid_utils import GridDescriptor, compute_cell_areas
from ....core.functor import FunctorMetadata
from .base_regridder import SpatialRegridderBase


def covered_support_acceptance(report: Mapping[str, Any], tolerance: float) -> Dict[str, Any]:
    """Accept a covered-support ledger report by the one shared rule, row by row."""
    # References: conservation_checker.ledger_acceptance (the one acceptance rule of every
    #   ledger; Higham 2002 rounding bound, Jones 1999 ledgers).
    # Mathematical expression: per row b, accepted_b iff
    #   e_b <= max(tol max(|L_cov,b|, 1e-12), gamma_k(u) a_b); accepted = all_b accepted_b;
    #   the accepted residual and the rounding allowance are summed over the rows.
    # Parameter meanings: e_b, L_cov,b, a_b the residual, covered source amount and activity of
    #   row b of the report; tol the caller's relative tolerance (the regridder's own, or the
    #   ScaleManager ledger tolerance of the call's mode); u the unit roundoff of the report's
    #   rounding dtype; k the report's term count.
    # Units: e, L, a and the accepted residual in the amount unit; tol and gamma dimensionless.
    # Domain/constraints: tol finite in (0, 1); every row is judged on its own, so a wrong row is
    #   never hidden by the amounts of the other rows.
    # Sign convention: e, a and the accepted residual are non-negative.
    tolerance = require_relative_tolerance(tolerance, "The covered-support ledger tolerance")
    rounding_dtype = getattr(torch, str(report["rounding_dtype"]))
    decisions = [
        conservation_checker.ledger_acceptance(
            absolute_error=row["absolute_residual"],
            relative_scale=ledger_relative_scale(row["covered_source_amount"]),
            activity=row["activity"],
            tolerance=tolerance,
            rounding_dtype=rounding_dtype,
            term_count=int(report["term_count"]),
        )
        for row in report["rows"]
    ]
    return {
        "accepted": all(decision["accepted"] for decision in decisions),
        "accepted_residual": sum(decision["accepted_residual"] for decision in decisions),
        "rounding_allowance": sum(decision["rounding_allowance"] for decision in decisions),
    }


class ConservativeRegridder(SpatialRegridderBase):
    """
    First-order conservative remapping between spatial grids on an explicit basis.

    Parameters
    ----------
    source_grid : GridDescriptor
        Source grid descriptor with coordinates and areas [m^2].
    target_grid : GridDescriptor
        Target grid descriptor with coordinates and areas [m^2].
    conservation_tolerance : float, optional
        Relative tolerance of the covered-support ledger and of the weight checks
        [dimensionless, finite in (0, 1)]; without it SpatialRegridderBase reads the one
        default coupling.scales.defaults.conservation_tolerance.
    enforce_conservation : bool, optional
        Whether a residual above tolerance raises in ``regrid`` (dimensionless flag).
    use_sparse_weights : bool, optional
        Use sparse storage for weight matrix (dimensionless flag).
    quantity_basis : SpatialQuantityBasis or str, optional
        Declared basis of the fields remapped by this instance (intensive,
        cell_total or area_density). A per-call ``quantity_basis`` overrides it;
        when neither is given a remap raises instead of assuming a basis.
    device : torch.device, optional
        Target computation device.

    Returns
    -------
    Dict[str, Tensor]
        ``forward`` returns ``regridded_data`` (same basis as the input), the
        covered-support ``conservation_error`` and ``diagnostics``; direct calls
        record the same diagnostics in ``last_regrid_report``.

    References
    ----------
    Jones, P.W. (1999). First- and second-order conservative remapping schemes.
    ESMF User's Guide: Conservative regridding.
    SCRIP documentation: Conservative remapping algorithms ("destarea"/"fracarea").

    Mathematical Expression
    -----------------------
    ``W[i,J] = A_iJ / A_i`` (not renormalized), ``f_i = sum_J W[i,J]``,
    ``C_J = sum_i A_i W[i,J]``.
    area_density ``y_J = sum_i x_i A_i W[i,J] / A_J``; cell_total
    ``Y_J = sum_i X_i W[i,J]``; intensive ``y_J = sum_i x_i A_i W[i,J] / C_J``.
    Covered-support ledger ``sum_J Y_J = sum_i X_i f_i``; the uncovered amount
    ``sum_i X_i (1 - f_i)`` is reported, not redistributed.
    Parameters: areas [m^2], ``W`` [dimensionless], values keep their units and basis.

    Attributes:
        weight_matrix : Tensor or SparseTensor
            Precomputed overlap fractions W[i,J]
        source_areas : Tensor
            Source grid cell areas [m^2]
        target_areas : Tensor
            Target grid cell areas [m^2]
        conservation_matrix : Tensor
            W[i,J] A_i / A_J (column sums: covered fraction of each target cell)
        coverage_report : Dict[str, float]
            Covered fractions of the source and target cells (partial coverage is reported,
            never rejected; see verify_conservation)
        last_regrid_report : Dict[str, Any] or None
            Basis, covered/mapped/uncovered amounts, residual, per-row ledger, the
            acceptance by the shared rule (accepted, accepted_residual, rounding_allowance,
            tolerance, rounding_dtype, term_count) and range-clip residual of the most
            recent remap (diagnostic record, never a control switch).

    Mask handling:
        W[i,J] = 0 if source_mask[i] = False or target_mask[J] = False.

    Methods:
        upscale / downscale: source-to-target remap (identical kernel; the names
            follow the SpatialRegridderBase dispatch on relative cell counts)
        compute_weights: Calculate conservative weight matrix
        verify_conservation: Check weight and area properties
        get_interpolation_info: Return interpolation diagnostics
    """
    
    def __init__(
        self,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        conservation_tolerance: Optional[float] = None,  # Loaded from parameter manager when not provided
        enforce_conservation: bool = False,
        use_sparse_weights: bool = True,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        metadata: Optional[FunctorMetadata] = None,
        parameter_manager: Optional[Any] = None,
        quantity_basis: Optional[Union[SpatialQuantityBasis, str]] = None,
    ) -> None:
        """
        Initialize conservative regridder with source and target grids.
        
        Mathematical Expression:
            Grid setup: Omega_source -> Omega_target
            Weight computation: W = compute_conservative_weights(Omega_s, Omega_t)
            Conservation check: verify_area_conservation(W, A_s, A_t)
        """
        meta = metadata or FunctorMetadata(
            name=name or "conservative_regridder",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            description="First-order conservative remapping between grids",
            mathematical_expression=(
                "W_iJ = A_iJ / A_i; area_density y_J = sum_i x_i A_i W_iJ / A_J; "
                "cell_total Y_J = sum_i X_i W_iJ; intensive y_J = sum_i x_i A_i W_iJ / sum_i A_i W_iJ"
            ),
            references=[
                "Jones, P.W. (1999). Conservative remapping schemes for climate models.",
                "ESMF User's Guide: Conservative regridding chapter.",
                "SCRIP documentation: Conservative remapping algorithms.",
            ],
            parameters={
                "conservation_tolerance": "Relative tolerance for conservation checks (dimensionless)",
                "clamp_min": "Lower bound applied to weight matrix entries (dimensionless)",
                "clamp_max": "Upper bound applied to weight matrix entries (dimensionless)",
            },
            input_variables=["data"],
            output_variables=["regridded_data"],
            conservation_laws=["mass", "energy"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )

        # SpatialRegridderBase resolves and validates the one tolerance (the explicit argument, else
        # coupling.scales.defaults.conservation_tolerance) as self.conservation_tolerance.
        super().__init__(
            source_grid=source_grid,
            target_grid=target_grid,
            name=meta.name,
            config={},
            metadata=meta,
            parameter_manager=parameter_manager,
            enforce_conservation=enforce_conservation,
            conservation_tolerance=conservation_tolerance,
            device=device,
        )

        self.use_sparse_weights = use_sparse_weights
        clamp_value = self.get_parameter_value(
            "clamp_min",
            namespace=resolve_namespace(ParameterNamespace.COUPLING_SCALES_CONSERVATIVE),
            units="dimensionless",
            validate=True,
        )
        if isinstance(clamp_value, torch.Tensor):
            clamp_value = float(clamp_value.detach().cpu().item())
        self.conservative_clamp_min = float(clamp_value)
        clamp_max_value = self.get_parameter_value(
            "clamp_max",
            namespace=resolve_namespace(ParameterNamespace.COUPLING_SCALES_CONSERVATIVE),
            units="dimensionless",
            validate=True,
        )
        if isinstance(clamp_max_value, torch.Tensor):
            clamp_max_value = float(clamp_max_value.detach().cpu().item())
        self.conservative_clamp_max = float(clamp_max_value)
        self.quantity_basis: Optional[SpatialQuantityBasis] = None
        if quantity_basis is not None:
            self.quantity_basis = self._validate_basis(SpatialQuantityBasis.parse(quantity_basis))
        self.last_regrid_report: Optional[Dict[str, Any]] = None
        # Per-forward handoff from _prepare_run to _execute_operation; consumed immediately
        # and never read by direct upscale/downscale/regrid calls.
        self._run_settings: Optional[Dict[str, Any]] = None
        self.earth_radius = float(get_physical_constant("earth_radius"))

        # No weight cache on the instance: the weights are computed once per geometry in
        # _prepare_geometry (reuse across calls is the content-keyed SpatialRegridderFactory cache).

        # Move grids to device
        self._move_grid_to_device(self.source_grid)
        self._move_grid_to_device(self.target_grid)
        
        # Geometry, cell areas, overlap weights and the covered-support weight check.
        self.coverage_report: Dict[str, float] = {}
        self._prepare_geometry()

    def _refresh_grids_if_needed(self, source_grid: GridDescriptor, target_grid: GridDescriptor) -> None:
        """
        Refresh cached bounds, areas, and weights when grids change.
        """
        if source_grid is self.source_grid and target_grid is self.target_grid:
            return

        self.source_grid = source_grid
        self.target_grid = target_grid
        self._move_grid_to_device(self.source_grid)
        self._move_grid_to_device(self.target_grid)
        self._prepare_geometry()

    def _resolve_geometry_coord_system(
        self,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
    ) -> str:
        """
        One geometry for overlaps and areas, from the declared coordinate systems.

        Geographic grids (coord_system "latlon") are spherical: their cells are lat-lon cells on
        the sphere. Cartesian and projected grids are planar. Grids of different systems, a
        latlon grid whose latitude bounds leave [-90, 90] and an unknown system raise; a grid is
        never reinterpreted in another geometry.
        """
        source_system, target_system = source_grid.coord_system, target_grid.coord_system
        if source_system != target_system:
            raise ValueError(
                f"Conservative regridding needs one geometry; the source grid is {source_system!r} and "
                f"the target grid is {target_system!r} (latlon cells are spherical, cartesian and "
                "projected cells planar)."
            )
        if source_system == "latlon":
            for role, grid in (("source", source_grid), ("target", target_grid)):
                if self._has_invalid_lat_bounds(grid):
                    raise ValueError(
                        f"The {role} grid is declared latlon but its latitude bounds leave [-90, 90]; "
                        "declare its planar coordinate system instead."
                    )
            return "latlon"
        if source_system in ("cartesian", "projected"):
            return source_system
        raise ValueError(
            f"Unsupported coordinate system {source_system!r} for conservative regridding; expected "
            "latlon, cartesian or projected."
        )

    @staticmethod
    def _has_invalid_lat_bounds(grid: GridDescriptor) -> bool:
        """Return True when y-bounds are outside the physical latitude interval."""
        if grid.y_bounds is None:
            return False
        y_bounds = grid.y_bounds.to(dtype=torch.float64)
        return bool(torch.any(y_bounds < -90.0) or torch.any(y_bounds > 90.0))


    def _resolve_grids(
        self,
        state: Dict[str, Any],
        **kwargs: Any,
    ) -> tuple[GridDescriptor, GridDescriptor]:
        """Refresh cached grid information when alternate grids are provided."""
        if "source_grid" in state:
            source_grid = state["source_grid"]
        else:
            source_grid = self.source_grid
        if "target_grid" in state:
            target_grid = state["target_grid"]
        else:
            target_grid = self.target_grid
        self._refresh_grids_if_needed(source_grid, target_grid)
        return self.source_grid, self.target_grid

    def _prepare_run(
        self,
        state: Dict[str, Any],
        operation: str,
        data: Tensor,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **kwargs: Any,
    ) -> None:
        """Resolve the per-call settings of one governed forward before dispatch.

        ``preserve`` and ``quantity_basis`` are read from the forward kwargs or the state.
        ``preserve`` in {mass, energy} disables the range clamp (clamping would change the
        remapped amount); ``preserve == "none"`` enables it for this call only; a call without
        ``preserve`` is a conservative remap (``mass``, no clamp). The method is this
        regridder's own (``conservative``); a call naming another method raises.
        """
        _ = (data, source_grid, target_grid)
        for source in (state, kwargs):
            if "method" in source and source["method"] != "conservative":
                raise ValueError(
                    f"ConservativeRegridder performs method 'conservative'; the call requested method "
                    f"{source['method']!r}. Create the regridder of that method instead."
                )
        self._current_method = "conservative"
        if "preserve" in kwargs:
            preserve = kwargs["preserve"]
        elif "preserve" in state:
            preserve = state["preserve"]
        else:
            preserve = "mass"
        if preserve not in ("mass", "energy", "none"):
            raise ValueError(f"preserve must be 'mass', 'energy' or 'none', got {preserve!r}.")
        self._current_preserve = preserve
        if "quantity_basis" in kwargs:
            quantity_basis = kwargs["quantity_basis"]
        elif "quantity_basis" in state:
            quantity_basis = state["quantity_basis"]
        else:
            quantity_basis = None
        self._run_settings = {
            "quantity_basis": quantity_basis,
            "range_clamp": preserve == "none",
        }

    def _execute_operation(
        self,
        data: Tensor,
        operation: str,
        **kwargs: Any,
    ) -> Tensor:
        """Dispatch one governed forward with the per-call settings from _prepare_run."""
        settings = self._run_settings
        self._run_settings = None
        if settings is None:
            raise RuntimeError("ConservativeRegridder._execute_operation requires _prepare_run first.")
        if operation == "downscale":
            return self.downscale(data, **settings)
        return self.upscale(data, **settings)

    @staticmethod
    def _validate_basis(basis: SpatialQuantityBasis) -> SpatialQuantityBasis:
        if basis is SpatialQuantityBasis.CATEGORICAL:
            raise ValueError(
                "ConservativeRegridder cannot remap categorical fields; use the nearest-neighbour regridder."
            )
        return basis

    def _resolve_call_basis(
        self, quantity_basis: Optional[Union[SpatialQuantityBasis, str]]
    ) -> SpatialQuantityBasis:
        """Per-call basis, else the declared instance basis, else a ValueError (no default)."""
        if quantity_basis is not None:
            return self._validate_basis(SpatialQuantityBasis.parse(quantity_basis))
        if self.quantity_basis is not None:
            return self.quantity_basis
        raise ValueError(
            "ConservativeRegridder requires an explicit quantity_basis (intensive, cell_total or "
            "area_density) per call or at construction; a field is never assumed to be an area density."
        )

    def _format_output(
        self,
        result: Tensor,
        operation: str,
        state: Dict[str, Any],
        *,
        input_data: Tensor,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Attach the covered-support diagnostics of the remap to the outputs."""
        report = self.last_regrid_report
        if report is None:
            raise RuntimeError("ConservativeRegridder produced no remap report for this call.")
        conservation_error = float(report["relative_residual"])
        diagnostics = {
            "operation": operation,
            "source_shape": tuple(input_data.shape),
            "target_shape": tuple(result.shape),
            "conservation_error": conservation_error,
            "method": self._current_method,
            "preserve": self._current_preserve,
            **report,
        }
        if hasattr(source_grid, "dims"):
            diagnostics["source_grid_dims"] = source_grid.dims
        if hasattr(target_grid, "dims"):
            diagnostics["target_grid_dims"] = target_grid.dims

        if not report["accepted"]:
            logger.warning(
                "The covered-support residual %.2e (relative %.2e) exceeds the accepted residual "
                "%.2e (tolerance %.2e) for %s operation",
                report["absolute_residual"],
                conservation_error,
                report["accepted_residual"],
                self.conservation_tolerance,
                operation,
            )
        return {
            "regridded_data": result,
            "conservation_error": conservation_error,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _require_cell_geometry(grid: GridDescriptor, role: str) -> None:
        """
        Raise unless the grid declares its cells; they are never inferred from the centres.

        A structured grid declares x_bounds/y_bounds (data file or explicit descriptor); an
        unstructured grid declares its cell polygons (vertex_coords + connectivity) and, since
        its areas are not built from bounds, its cell_areas.
        """
        if grid.grid_type == "unstructured" and grid.vertex_coords is not None and grid.connectivity is not None:
            if grid.cell_areas is None:
                raise ValueError(
                    f"Conservative regridding of the unstructured {role} grid needs its cell_areas."
                )
            return
        if grid.x_bounds is None or grid.y_bounds is None:
            raise ValueError(
                f"Conservative regridding needs the cell bounds of the {role} grid (x_bounds/y_bounds "
                "from the data file or an explicit grid descriptor); bounds are not inferred from "
                "the coordinate spacing."
            )

    @staticmethod
    def _is_polygon_grid(grid: GridDescriptor) -> bool:
        """Whether the grid declares its cells as polygons (vertex_coords + connectivity)."""
        return grid.grid_type == "unstructured" and grid.vertex_coords is not None and grid.connectivity is not None

    def _cell_rect_bounds(self, grid: GridDescriptor) -> Tuple[Tensor, Tensor]:
        """Per-cell x and y bounds [n_cells, 2] (lower, upper) in the flattened cell order."""
        x_bounds = grid.x_bounds.to(device=self.device, dtype=torch.float64)
        y_bounds = grid.y_bounds.to(device=self.device, dtype=torch.float64)
        if len(grid.dims) == 2:
            ny, nx = (int(dim) for dim in grid.dims)
            if x_bounds.dim() == 2 and y_bounds.dim() == 2:
                x_bounds = x_bounds.unsqueeze(0).expand(ny, nx, 2)
                y_bounds = y_bounds.unsqueeze(1).expand(ny, nx, 2)
        x_cells = x_bounds.reshape(-1, 2)
        y_cells = y_bounds.reshape(-1, 2)
        n_cells = int(math.prod(int(dim) for dim in grid.dims))
        if x_cells.shape[0] != n_cells or y_cells.shape[0] != n_cells:
            raise ValueError(
                f"Cell bounds of shape {tuple(grid.x_bounds.shape)}/{tuple(grid.y_bounds.shape)} do not "
                f"describe the {n_cells} cells of a grid with dims {tuple(grid.dims)}."
            )
        return torch.sort(x_cells, dim=-1).values, torch.sort(y_cells, dim=-1).values

    def _rect_areas(self, x_cells: Tensor, y_cells: Tensor) -> Tensor:
        """Areas of per-cell bounds [n, 2] in the resolved geometry (grid_utils.compute_cell_areas)."""
        # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210 (cell areas of lat-lon
        #   grids on the sphere); grid_utils.spherical_area / cartesian_area (the area owner).
        # Mathematical expression: latlon A = R^2 (lambda_2 - lambda_1) (sin phi_2 - sin phi_1);
        #   planar A = (x_2 - x_1) (y_2 - y_1).
        # Parameter meanings: R earth radius (physical constant earth_radius); lambda, phi cell
        #   longitude and latitude bounds; x, y planar cell bounds.
        # Units: lambda, phi [degrees, converted to radians]; R [m]; latlon A [m2]; planar A in the
        #   squared bound units.
        # Domain/constraints: bounds sorted per cell; latitudes in [-90, 90].
        # Sign convention: areas are positive.
        areas = compute_cell_areas(
            x_cells.unsqueeze(1), y_cells.unsqueeze(1), self._geometry_coord_system, self.earth_radius
        )
        return areas.reshape(-1).to(device=self.device, dtype=torch.float64)

    def _resolve_cell_areas(self, grid: GridDescriptor, role: str) -> Tensor:
        """
        Cell areas of one grid: the declared cell_areas, else the geometric areas of its bounds.

        A latlon grid's declared areas must be its spherical cell areas (the measure of the
        overlaps); a mismatch raises. A polygon grid declares its areas (required).
        """
        if self._is_polygon_grid(grid):
            return grid.cell_areas
        geometric = self._rect_areas(*self._cell_rect_bounds(grid)).reshape(tuple(int(dim) for dim in grid.dims))
        if grid.cell_areas is None:
            return geometric
        declared = grid.cell_areas.to(device=self.device)
        if self._geometry_coord_system == "latlon":
            difference = (declared.to(torch.float64).reshape(geometric.shape) - geometric).abs() / geometric.abs()
            worst = float(difference.max().detach().item())
            if worst > self.conservation_tolerance:
                raise ValueError(
                    f"The declared cell_areas of the {role} latlon grid differ from the spherical cell "
                    f"areas R^2 dlambda (sin phi2 - sin phi1) of its bounds (maximum relative difference "
                    f"{worst:.2e}); overlaps and cell areas use one measure."
                )
        return declared

    def _prepare_geometry(self) -> None:
        """Geometry, cell areas, overlap weights and the covered-support weight check."""
        self._require_cell_geometry(self.source_grid, "source")
        self._require_cell_geometry(self.target_grid, "target")
        self._geometry_coord_system = self._resolve_geometry_coord_system(self.source_grid, self.target_grid)
        self.source_areas = self._resolve_cell_areas(self.source_grid, "source")
        self.target_areas = self._resolve_cell_areas(self.target_grid, "target")
        self.weight_matrix, self.conservation_matrix = self.compute_weights()
        checks = self.verify_conservation()
        self.coverage_report = {
            key: value
            for key, value in checks.items()
            if key.endswith("_cells") or key.endswith("_coverage")
        }
        inconsistency = max(
            checks["max_source_overcoverage"], checks["max_target_overcoverage"], checks["max_negative_weight"]
        )
        if inconsistency > self.conservation_tolerance:
            raise ValueError(
                "Conservative weights are inconsistent on the covered support: a covered fraction exceeds "
                f"its cell (source {checks['max_source_overcoverage']:.2e}, target "
                f"{checks['max_target_overcoverage']:.2e} above 1) or a weight is negative "
                f"({checks['max_negative_weight']:.2e}); overlapping cells or cell_areas that disagree with "
                "the cell bounds are counted twice."
            )
        if self.coverage_report["partially_covered_source_cells"] or self.coverage_report["uncovered_source_cells"]:
            logger.info(
                "Conservative regridder covers %d source cell(s) partially and %d not at all; their "
                "uncovered amounts are reported by each remap.",
                self.coverage_report["partially_covered_source_cells"],
                self.coverage_report["uncovered_source_cells"],
            )

    def _move_grid_to_device(self, grid: GridDescriptor) -> None:
        """Move grid tensors to specified device."""
        grid.x_coords = grid.x_coords.to(self.device)
        grid.y_coords = grid.y_coords.to(self.device)
        if grid.x_bounds is not None:
            grid.x_bounds = grid.x_bounds.to(self.device)
        if grid.y_bounds is not None:
            grid.y_bounds = grid.y_bounds.to(self.device)
        if grid.cell_areas is not None:
            grid.cell_areas = grid.cell_areas.to(self.device)
        if grid.mask is not None:
            grid.mask = grid.mask.to(self.device)
    
    def compute_weights(self) -> Tuple[Tensor, Tensor]:
        """
        Overlap fractions W[i,J] = A_iJ / A_i and the matrix W[i,J] A_i / A_J.

        Regular grids with 1-D bounds use the exact separable form; other bounds-based grids use
        exact pairwise rectangle overlaps; polygon grids (planar only) use polygon intersection.

        Returns:
            Tuple[Tensor, Tensor]: (weight_matrix [n_source, n_target], conservation_matrix)
        """
        if self._is_polygon_grid(self.source_grid) or self._is_polygon_grid(self.target_grid):
            weight_matrix = self._compute_polygon_weights()
        elif self._has_separable_bounds(self.source_grid) and self._has_separable_bounds(self.target_grid):
            weight_matrix = self._compute_separable_weights()
        else:
            weight_matrix = self._compute_rectangle_weights()
        return weight_matrix, self._compute_conservation_matrix(weight_matrix)

    def _compute_conservation_matrix(self, weight_matrix: Tensor) -> Tensor:
        """W[i,J] A_i / A_J; its column sums are the covered fractions of the target cells."""
        source = self.source_areas.reshape(-1).to(device=self.device, dtype=torch.float64)
        target = self.target_areas.reshape(-1).to(device=self.device, dtype=torch.float64)
        if weight_matrix.is_sparse:
            weights = weight_matrix.coalesce()
            rows, cols = weights.indices()
            values = weights.values() * source[rows] / target[cols]
            return torch.sparse_coo_tensor(weights.indices(), values, weights.shape, check_invariants=True).coalesce()
        return weight_matrix * source.unsqueeze(1) / target.unsqueeze(0)

    @staticmethod
    def _has_separable_bounds(grid: GridDescriptor) -> bool:
        """Whether a 2-D grid declares one bound pair per column and per row (x: [nx, 2], y: [ny, 2])."""
        if len(grid.dims) != 2 or grid.x_bounds is None or grid.y_bounds is None:
            return False
        ny, nx = (int(dim) for dim in grid.dims)
        return tuple(grid.x_bounds.shape) == (nx, 2) and tuple(grid.y_bounds.shape) == (ny, 2)

    def _axis_measure(self, values: Tensor, latitude: bool) -> Tensor:
        """Measure of a coordinate along one axis: sin(phi) for latitude on the sphere, else the value."""
        if latitude and self._geometry_coord_system == "latlon":
            return torch.sin(torch.deg2rad(values))
        return values

    def _interval_overlap_fractions(self, source: Tensor, target: Tensor, latitude: bool) -> Tensor:
        """F[a, b] = |I_a intersect J_b|_m / |I_a|_m for sorted intervals [n, 2] and [m, 2]."""
        lower = torch.maximum(source[:, None, 0], target[None, :, 0])
        upper = torch.minimum(source[:, None, 1], target[None, :, 1])
        overlap = torch.where(
            upper > lower,
            self._axis_measure(upper, latitude) - self._axis_measure(lower, latitude),
            torch.zeros_like(upper),
        )
        length = self._axis_measure(source[:, 1], latitude) - self._axis_measure(source[:, 0], latitude)
        if bool((length <= 0).any()):
            raise ValueError("Conservative regridding needs cells of positive extent; a source cell has none.")
        return overlap / length[:, None]

    def _compute_separable_weights(self) -> Tensor:
        """Exact overlap fractions of two regular grids with 1-D bounds."""
        # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210 (first-order conservative
        #   remapping on the sphere; a lat-lon cell has area R^2 dlambda (sin phi_2 - sin phi_1));
        #   SCRIP/ESMF first-order weights W_iJ = A_iJ / A_i.
        # Mathematical expression: the overlap of two lat-lon cells is the lat-lon cell
        #   [max lambda_1, min lambda_2] x [max phi_1, min phi_2], so
        #   W_iJ = Fx[i_x, J_x] Fy[i_y, J_y] with Fx = |overlap in lambda| / |lambda_i| and
        #   Fy = (sin phi_hi - sin phi_lo) / (sin phi_i2 - sin phi_i1) on the sphere; planar grids
        #   use Fy = |overlap in y| / |y_i|. Flattened index i = i_y n_x + i_x (row-major).
        # Parameter meanings: lambda, phi longitude/latitude bounds of the cells; x, y planar
        #   bounds; Fx, Fy axis overlap fractions; W overlap fraction of source cell i in target
        #   cell J; R cancels in W.
        # Units: bounds in degrees (latlon) or planar units; Fx, Fy, W [1].
        # Domain/constraints: bounds sorted per cell, cells of positive extent, one geometry for
        #   both grids; masked cells carry no weight. The row sums f_i are the covered fractions
        #   (not renormalized).
        # Sign convention: weights are non-negative.
        source, target = self.source_grid, self.target_grid
        src_ny, src_nx = (int(dim) for dim in source.dims)
        tgt_ny, tgt_nx = (int(dim) for dim in target.dims)
        sort = lambda bounds: torch.sort(bounds.to(device=self.device, dtype=torch.float64), dim=-1).values  # noqa: E731
        fx = self._interval_overlap_fractions(sort(source.x_bounds), sort(target.x_bounds), latitude=False)
        fy = self._interval_overlap_fractions(sort(source.y_bounds), sort(target.y_bounds), latitude=True)
        iy, jy = torch.nonzero(fy > 0, as_tuple=True)
        ix, jx = torch.nonzero(fx > 0, as_tuple=True)
        rows = (iy[:, None] * src_nx + ix[None, :]).reshape(-1)
        cols = (jy[:, None] * tgt_nx + jx[None, :]).reshape(-1)
        values = (fy[iy, jy][:, None] * fx[ix, jx][None, :]).reshape(-1)
        return self._assemble_weights(rows, cols, values, src_ny * src_nx, tgt_ny * tgt_nx)

    def _compute_rectangle_weights(self) -> Tensor:
        """Exact overlap fractions of bounds-based cells (curvilinear or cell-list grids)."""
        # References, expression, units and constraints as in _compute_separable_weights, applied
        #   to every source/target cell pair: W_iJ = Ox Oy / (Lx_i Ly_i) with Ox the longitude or x
        #   overlap, Oy the sin(phi) (spherical) or y (planar) overlap, Lx_i, Ly_i the source cell
        #   extents in the same measures. Sign convention: non-negative weights.
        source_x, source_y = self._cell_rect_bounds(self.source_grid)
        target_x, target_y = self._cell_rect_bounds(self.target_grid)
        fx = self._interval_overlap_fractions(source_x, target_x, latitude=False)
        fy = self._interval_overlap_fractions(source_y, target_y, latitude=True)
        fractions = fx * fy
        rows, cols = torch.nonzero(fractions > 0, as_tuple=True)
        return self._assemble_weights(rows, cols, fractions[rows, cols], source_x.shape[0], target_x.shape[0])

    def _assemble_weights(self, rows: Tensor, cols: Tensor, values: Tensor, n_source: int, n_target: int) -> Tensor:
        """Weight matrix from its non-zero entries: masked cells removed, configured bounds applied."""
        keep = torch.ones_like(values, dtype=torch.bool)
        if self.source_grid.mask is not None:
            keep &= self.source_grid.mask.reshape(-1).to(self.device).bool()[rows]
        if self.target_grid.mask is not None:
            keep &= self.target_grid.mask.reshape(-1).to(self.device).bool()[cols]
        values = torch.clamp(values[keep], min=self.conservative_clamp_min, max=self.conservative_clamp_max)
        weights = torch.sparse_coo_tensor(
            torch.stack([rows[keep], cols[keep]]),
            values,
            (n_source, n_target),
            device=self.device,
            dtype=torch.float64,
            check_invariants=True,
        ).coalesce()
        return weights if self.use_sparse_weights else weights.to_dense()

    def _compute_polygon_weights(self) -> Tensor:
        """
        Overlap fractions with polygon grids by planar polygon intersection (shapely).

        Polygon grids are planar only: exact overlaps of spherical polygons are not implemented,
        so a latlon grid with polygon cells raises.
        """
        if self._geometry_coord_system == "latlon":
            raise ValueError(
                "Conservative regridding of latlon polygon cells needs spherical polygon overlaps, which "
                "are not implemented; declare the polygon grid in a projected coordinate system."
            )
        try:
            from shapely.geometry import Polygon  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Conservative regridding with polygon cells requires shapely for polygon intersection."
            ) from exc

        def _polygons(grid: GridDescriptor) -> list:
            if self._is_polygon_grid(grid):
                vertex_coords = grid.vertex_coords.detach().cpu().to(dtype=torch.float64)
                connectivity = grid.connectivity.detach().cpu()
                cells = []
                for cell_idx in range(connectivity.shape[0]):
                    indices = connectivity[cell_idx]
                    points = vertex_coords[indices[indices >= 0]].tolist()
                    if len(points) < 3:
                        raise ValueError(f"Polygon cell {cell_idx} has fewer than three vertices.")
                    polygon = Polygon(points)
                    if not polygon.is_valid or polygon.is_empty or polygon.area <= 0.0:
                        raise ValueError(f"Polygon cell {cell_idx} is not a valid polygon of positive area.")
                    cells.append(polygon)
                return cells
            x_cells, y_cells = self._cell_rect_bounds(grid)
            return [
                Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                for (x0, x1), (y0, y1) in zip(x_cells.cpu().tolist(), y_cells.cpu().tolist())
            ]

        # References: Jones (1999); ESMF/SCRIP conservative remapping.
        # Mathematical expression: W_iJ = A(P_i intersect Q_J) / A(P_i) (planar polygon areas).
        # Parameter meanings: P_i source cell polygon; Q_J target cell polygon; A planar area.
        # Units: polygon areas in the squared coordinate units; W [1].
        # Domain/constraints: planar geometry; valid polygons of positive area.
        # Sign convention: non-negative weights.
        source_polygons = _polygons(self.source_grid)
        target_polygons = _polygons(self.target_grid)
        rows, cols, values = [], [], []
        for src_idx, src_polygon in enumerate(source_polygons):
            src_area = float(src_polygon.area)
            for tgt_idx, tgt_polygon in enumerate(target_polygons):
                if not src_polygon.intersects(tgt_polygon):
                    continue
                overlap = float(src_polygon.intersection(tgt_polygon).area)
                if overlap > 0.0:
                    rows.append(src_idx)
                    cols.append(tgt_idx)
                    values.append(overlap / src_area)
        return self._assemble_weights(
            torch.tensor(rows, dtype=torch.long, device=self.device),
            torch.tensor(cols, dtype=torch.long, device=self.device),
            torch.tensor(values, dtype=torch.float64, device=self.device),
            len(source_polygons),
            len(target_polygons),
        )

    def downscale(
        self,
        fine_data: Tensor,
        *,
        quantity_basis: Optional[Union[SpatialQuantityBasis, str]] = None,
        range_clamp: bool = False,
    ) -> Tensor:
        """Public API: source-grid field -> target grid (dispatched when the target is coarser)."""
        return self._remap_to_target(fine_data, quantity_basis=quantity_basis, range_clamp=range_clamp)

    def upscale(
        self,
        coarse_data: Tensor,
        *,
        quantity_basis: Optional[Union[SpatialQuantityBasis, str]] = None,
        range_clamp: bool = False,
    ) -> Tensor:
        """Public API: source-grid field -> target grid (dispatched when the target is finer)."""
        return self._remap_to_target(coarse_data, quantity_basis=quantity_basis, range_clamp=range_clamp)

    def _flatten_source_field(self, data: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
        """Return the field as [batch, n_source] and the output shape on the target grid."""
        source_dims = tuple(int(dim) for dim in self.source_grid.dims)
        target_dims = tuple(int(dim) for dim in self.target_grid.dims)
        n_source = int(math.prod(source_dims))
        n_target = int(math.prod(target_dims))
        n_dims = len(source_dims)
        if data.dim() >= n_dims and tuple(data.shape[-n_dims:]) == source_dims:
            batch_dims = tuple(int(dim) for dim in data.shape[:-n_dims])
            return data.reshape(-1, n_source), batch_dims + target_dims
        if data.dim() == 1 and data.numel() == n_source:
            return data.reshape(1, n_source), (n_target,)
        raise ValueError(
            f"Field shape {tuple(data.shape)} does not match the source grid {source_dims}; pass "
            "[..., *source_dims] or a flat vector with one value per source cell."
        )

    def _map_field(self, spatial: Tensor, basis: SpatialQuantityBasis) -> Tensor:
        """First-order conservative map of a [batch, n_source] field in its own dtype."""
        # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210 (first-order
        #   conservative remapping); SCRIP user guide (Jones, 1998) destarea/fracarea
        #   normalizations; docs/source/theory/scale_coupling.md sections 2.3-2.3.2.
        # Mathematical expression: with W_iJ = A_iJ / A_i and C_J = sum_i A_i W_iJ:
        #   cell_total    Y_J = sum_i X_i W_iJ
        #   area_density  X_i = x_i A_i, Y_J as above, y_J = Y_J / A_J          (destarea)
        #   intensive     y_J = sum_i x_i A_i W_iJ / C_J                         (fracarea)
        # Parameter meanings: X/Y cell totals; x/y area densities or intensive values; A_i, A_J
        #   source and target areas; A_iJ overlap area; W_iJ overlap fraction; C_J covered
        #   target area.
        # Units: A [m2]; W [1]; X, Y in the amount unit; x, y keep their input units.
        # Domain/constraints: evaluated in the dtype of the field (weights and areas are cast to
        #   it); intensive targets need C_J > 0; the map is linear and autograd differentiates
        #   exactly this expression; nothing is rescaled.
        # Sign convention: values keep their sign.
        dtype = spatial.dtype
        device = spatial.device
        source_areas = self.source_areas.to(device=device, dtype=dtype).reshape(-1)
        target_areas = self.target_areas.to(device=device, dtype=dtype).reshape(-1)
        weights = self.weight_matrix.to_dense() if self.weight_matrix.is_sparse else self.weight_matrix
        weights = weights.to(device=device, dtype=dtype)
        if basis.is_extensive:
            totals = convert_spatial_basis(spatial, source_areas, basis, SpatialQuantityBasis.CELL_TOTAL)
            target_totals = torch.matmul(totals, weights)
            if basis is SpatialQuantityBasis.AREA_DENSITY:
                return convert_spatial_basis(
                    target_totals, target_areas, SpatialQuantityBasis.CELL_TOTAL, basis
                )
            return target_totals
        covered_target = torch.matmul(source_areas.unsqueeze(0), weights).reshape(-1)
        if bool((covered_target <= 0).any()):
            offenders = torch.nonzero(covered_target <= 0).reshape(-1).tolist()
            raise ValueError(
                f"Target cell(s) {offenders[:8]} receive no source overlap; an intensive field is "
                "undefined there."
            )
        return torch.matmul(spatial * source_areas, weights) / covered_target

    def _apply_range_clamp(
        self, spatial: Tensor, target: Tensor, basis: SpatialQuantityBasis
    ) -> Tuple[Tensor, Tensor]:
        """Per-call clamp of the density representation to the source range, with its residual."""
        # References: Jones (1999) (first-order remaps are convex only on fully covered targets).
        # Mathematical expression: d_J <- min(max(d_J, min_i d_i), max_i d_i); residual
        #   R = sum_J (d_J - d_J,c) a_J with a_J = A_J for amounts and C_J for intensive fields,
        #   so the amount before the clamp = the amount after the clamp + R.
        # Parameter meanings: d density representation (y for densities and intensive fields,
        #   Y / A for cell totals); d_i source densities; a_J the measure of target cell J.
        # Units: d in the density units; R in the amount unit (an area integral for intensive
        #   fields).
        # Domain/constraints: with full target coverage (C_J = A_J) every y_J already lies in
        #   [min x, max x]; the clamp binds only on partially covered targets, where it changes
        #   the remapped amount, so it runs only on request and R is reported, never rescaled away.
        # Sign convention: StateManager clip_loss sign (requested minus clamped): R < 0 when the
        #   clamp adds material, R > 0 when it removes material.
        dtype = spatial.dtype
        device = spatial.device
        source_areas = self.source_areas.to(device=device, dtype=dtype).reshape(-1)
        target_areas = self.target_areas.to(device=device, dtype=dtype).reshape(-1)
        weights = self.weight_matrix.to_dense() if self.weight_matrix.is_sparse else self.weight_matrix
        weights = weights.to(device=device, dtype=dtype)
        if basis.is_extensive:
            totals = convert_spatial_basis(spatial, source_areas, basis, SpatialQuantityBasis.CELL_TOTAL)
            source_range_values = totals / source_areas
            range_measure = target_areas
        else:
            source_range_values = spatial
            range_measure = torch.matmul(source_areas.unsqueeze(0), weights).reshape(-1)
        target_range_values = target / target_areas if basis is SpatialQuantityBasis.CELL_TOTAL else target
        lower = source_range_values.min(dim=1, keepdim=True).values
        upper = source_range_values.max(dim=1, keepdim=True).values
        clamped = torch.clamp(target_range_values, min=lower, max=upper)
        clip_residual = ((target_range_values - clamped) * range_measure).sum()
        return (clamped * target_areas if basis is SpatialQuantityBasis.CELL_TOTAL else clamped), clip_residual

    def _covered_support_ledger(
        self, spatial: Tensor, target: Tensor, basis: SpatialQuantityBasis
    ) -> Dict[str, Any]:
        """Float64 ledger of the returned field on the covered support, row by row."""
        # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210 (covered-support ledger
        #   of a first-order conservative remap).
        # Mathematical expression: per row b of the [batch, n_source] field, with f_i = sum_J W_iJ
        #   and C_J = sum_i A_i W_iJ in float64:
        #   L_cov,b = sum_i S_i f_i, L_tgt,b = sum_J T_J, U_b = sum_i S_i (1 - f_i),
        #   e_b = |L_tgt,b - L_cov,b|, a_b = sum_i |S_i| f_i; S_i = X_i (cell totals, x_i A_i for
        #   area densities) or x_i A_i (intensive); T_J = Y_J (cell totals, y_J A_J for area
        #   densities) or y_J C_J (intensive), taken from the returned field.
        # Parameter meanings: S source terms; T returned target terms; f_i covered fraction;
        #   L_cov covered source amount; L_tgt mapped amount; U unmapped source amount; e residual;
        #   a activity of the covered source terms.
        # Units: all amounts in the amount unit of the field (area integrals [value m2] for
        #   intensive fields); f [1].
        # Domain/constraints: float64 from the returned (optionally clamped) values, so a wrong
        #   result can never report a zero residual; the field's own rounding enters e and is
        #   judged by covered_support_acceptance.
        # Sign convention: signed amounts; e and a are non-negative.
        with torch.no_grad():
            device = spatial.device
            weights = self.weight_matrix.to_dense() if self.weight_matrix.is_sparse else self.weight_matrix
            weights = weights.to(device=device, dtype=torch.float64)
            source_areas = self.source_areas.to(device=device, dtype=torch.float64).reshape(-1)
            target_areas = self.target_areas.to(device=device, dtype=torch.float64).reshape(-1)
            coverage = weights.sum(dim=1)
            values = spatial.detach().to(torch.float64)
            returned = target.detach().to(torch.float64)
            if basis.is_extensive:
                source_terms = convert_spatial_basis(values, source_areas, basis, SpatialQuantityBasis.CELL_TOTAL)
                target_terms = convert_spatial_basis(returned, target_areas, basis, SpatialQuantityBasis.CELL_TOTAL)
            else:
                source_terms = values * source_areas
                target_terms = returned * torch.matmul(source_areas.unsqueeze(0), weights).reshape(-1)
            covered = (source_terms * coverage).sum(dim=1)
            mapped = target_terms.sum(dim=1)
            uncovered = (source_terms * (1.0 - coverage)).sum(dim=1)
            activity = (source_terms.abs() * coverage).sum(dim=1)
            residuals = (mapped - covered).abs()
            covered_fraction = float(((coverage * source_areas).sum() / source_areas.sum()).item())
        rows = [
            {
                "covered_source_amount": float(covered[row].item()),
                "absolute_residual": float(residuals[row].item()),
                "activity": float(activity[row].item()),
            }
            for row in range(covered.shape[0])
        ]
        return {
            "covered_source_amount": float(covered.sum().item()),
            "mapped_target_amount": float(mapped.sum().item()),
            "uncovered_source_amount": float(uncovered.sum().item()),
            "covered_source_fraction": covered_fraction,
            "absolute_residual": float(residuals.sum().item()),
            "relative_residual": max(
                row["absolute_residual"] / ledger_relative_scale(row["covered_source_amount"]) for row in rows
            ),
            "activity": float(activity.sum().item()),
            "rows": rows,
        }

    def _remap_to_target(
        self,
        data: Tensor,
        *,
        quantity_basis: Optional[Union[SpatialQuantityBasis, str]],
        range_clamp: bool,
    ) -> Tensor:
        """First-order conservative remap of one field on its explicit basis, with its ledger."""
        # References: Jones (1999); docs/source/theory/scale_coupling.md sections 2.3-2.3.2 and
        #   2.10; conservation_checker.ledger_acceptance (the one acceptance rule).
        # Mathematical expression: y = map(x) (_map_field), optionally clamped (_apply_range_clamp),
        #   ledger per row (_covered_support_ledger), accepted per row by
        #   covered_support_acceptance with k = 2 n + 8 rounded steps in the field dtype.
        # Parameter meanings: x the [batch, n_source] field on its explicit basis; y the returned
        #   field; n the number of source cells; tol this regridder's tolerance.
        # Units: as in _map_field and _covered_support_ledger.
        # Domain/constraints: the basis is explicit (per call or at construction) and never
        #   defaulted; categorical fields raise; A_i, A_J > 0 and finite. k counts, per source
        #   term, the casts of A_i, W_iJ and A_J to the field dtype, the two products, the n - 1
        #   additions and the division, doubled for the numerator and the denominator C_J of an
        #   intensive field (the largest case, used for every basis).
        # Sign convention: values keep their sign; the clip residual follows the StateManager
        #   clip_loss sign (R < 0 when the clamp adds material).
        basis = self._resolve_call_basis(quantity_basis)
        if not isinstance(data, Tensor):
            raise TypeError("ConservativeRegridder expects a tensor field.")
        spatial, output_shape = self._flatten_source_field(data)
        if not torch.is_floating_point(spatial):
            spatial = spatial.to(torch.float32)
        target = self._map_field(spatial, basis)
        clip_residual = 0.0
        if range_clamp:
            target, clip = self._apply_range_clamp(spatial, target, basis)
            clip_residual = float(clip.detach().item())
        ledger = self._covered_support_ledger(spatial, target, basis)
        report: Dict[str, Any] = {
            "quantity_basis": basis.value,
            "output_basis": basis.value,
            "normalization": "fracarea" if basis is SpatialQuantityBasis.INTENSIVE else "destarea",
            **ledger,
            "tolerance": self.conservation_tolerance,
            "rounding_dtype": str(coarsest_floating_dtype(spatial.dtype, target.dtype)).replace("torch.", ""),
            "term_count": 2 * int(spatial.shape[-1]) + 8,
            "range_clamp": bool(range_clamp),
            "range_clip_residual": clip_residual,
        }
        report.update(covered_support_acceptance(report, self.conservation_tolerance))
        self.last_regrid_report = report
        return target.reshape(output_shape)

    def verify_conservation(self) -> Dict[str, float]:
        """
        Weight consistency on the covered support and the coverage of both grids.

        Returns:
            Dict[str, float]: over-coverage and negative-weight errors (all 0 for consistent
            weights) and the coverage statistics of the source and target cells.
        """
        # References: Jones, P.W. (1999) Mon. Weather Rev. 127:2204-2210; SCRIP/ESMF first-order
        #   weights (covered fractions are not renormalized).
        # Mathematical expression: f_i = sum_J W_iJ (covered fraction of source cell i);
        #   g_J = sum_i A_i W_iJ / A_J (covered fraction of target cell J). Overlaps are parts of the
        #   cells: W_iJ >= 0, f_i <= 1, g_J <= 1. Errors: max(f_i - 1, 0), max(g_J - 1, 0),
        #   max(-W_iJ, 0); a source cell is fully covered when |f_i - 1| <= tol, partially when
        #   tol < f_i < 1 - tol and uncovered when f_i <= tol.
        # Parameter meanings: W overlap fraction; A_i, A_J resolved cell areas; tol the
        #   conservation_tolerance.
        # Units: all quantities dimensionless.
        # Domain/constraints: partial coverage is not an error (the remap reports the uncovered
        #   amount); an over-covered cell means double-counted overlaps or cell areas that
        #   disagree with the cell bounds.
        # Sign convention: errors are non-negative.
        weights = self.weight_matrix.coalesce() if self.weight_matrix.is_sparse else self.weight_matrix.to_sparse().coalesce()
        rows, cols = weights.indices()
        values = weights.values().to(dtype=torch.float64)
        n_source, n_target = (int(size) for size in weights.shape)
        source_areas = self.source_areas.reshape(-1).to(device=values.device, dtype=torch.float64)
        target_areas = self.target_areas.reshape(-1).to(device=values.device, dtype=torch.float64)
        source_coverage = torch.zeros(n_source, dtype=torch.float64, device=values.device).index_add_(0, rows, values)
        target_coverage = torch.zeros(n_target, dtype=torch.float64, device=values.device).index_add_(
            0, cols, values * source_areas[rows]
        ) / target_areas
        negative = float((-values).max().item()) if values.numel() else 0.0
        tolerance = self.conservation_tolerance

        def _count(mask: Tensor) -> int:
            return int(mask.sum().item())

        return {
            "max_source_overcoverage": max(float((source_coverage - 1.0).max().item()), 0.0),
            "max_target_overcoverage": max(float((target_coverage - 1.0).max().item()), 0.0),
            "max_negative_weight": max(negative, 0.0),
            "min_source_coverage": float(source_coverage.min().item()),
            "min_target_coverage": float(target_coverage.min().item()),
            "fully_covered_source_cells": _count((source_coverage - 1.0).abs() <= tolerance),
            "partially_covered_source_cells": _count((source_coverage > tolerance) & (source_coverage < 1.0 - tolerance)),
            "uncovered_source_cells": _count(source_coverage <= tolerance),
            "fully_covered_target_cells": _count((target_coverage - 1.0).abs() <= tolerance),
            "partially_covered_target_cells": _count((target_coverage > tolerance) & (target_coverage < 1.0 - tolerance)),
            "uncovered_target_cells": _count(target_coverage <= tolerance),
        }

    def get_interpolation_info(self) -> Dict[str, Any]:
        """
        Return information about the interpolation setup.
        
        Provides diagnostics and metadata about the regridding configuration
        useful for debugging and validation.
        
        Returns:
            Dict[str, Any]
                Interpolation configuration and statistics
        """
        info = {
            "source_grid_dims": self.source_grid.dims,
            "target_grid_dims": self.target_grid.dims,
            "source_coord_system": self.source_grid.coord_system,
            "target_coord_system": self.target_grid.coord_system,
            "conservation_tolerance": self.conservation_tolerance,
            "use_sparse_weights": self.use_sparse_weights,
            "total_source_area": torch.sum(self.source_areas).item(),
            "total_target_area": torch.sum(self.target_areas).item(),
        }
        
        # Weight matrix statistics
        weights = self.weight_matrix.to_dense() if self.weight_matrix.is_sparse else self.weight_matrix
            
        nonzero_weights = weights[weights > 0]
        if len(nonzero_weights) > 0:
            info["num_nonzero_weights"] = len(nonzero_weights)
            info["min_weight"] = torch.min(nonzero_weights).item()
            info["max_weight"] = torch.max(nonzero_weights).item()
            info["mean_weight"] = torch.mean(nonzero_weights).item()
            info["weight_sparsity"] = 1.0 - len(nonzero_weights) / weights.numel()
        else:
            info["num_nonzero_weights"] = 0
            info["weight_sparsity"] = 1.0
        
        return info
    
    def regrid(
        self,
        data: Tensor,
        source_grid: Any = None,
        target_grid: Any = None,
        *,
        method: str,
        preserve: str = "mass",
        quantity_basis: Optional[Union[SpatialQuantityBasis, str]] = None,
    ) -> Tensor:
        """
        Regrid one field from the source grid to the target grid.

        Parameters:
            data : Tensor
                Field on the source grid ([..., *source_dims] or one value per cell)
            source_grid, target_grid : Any
                Must match the grids of this instance (dimension check only)
            method : str
                Only "conservative"
            preserve : str
                "mass" or "energy": no range clamp, and a covered-support ledger that the
                shared rule rejects (covered_support_acceptance with conservation_tolerance:
                a residual above max(tol |L_cov|, gamma_k(u) a) in some row) raises when
                enforce_conservation is set (else it is logged); "none": the per-call range
                clamp is applied and reported.
            quantity_basis : SpatialQuantityBasis or str, optional
                Basis of this call; defaults to the declared instance basis, never to an
                assumed area density.

        Returns:
            Tensor: field on the target grid in the same basis; last_regrid_report carries the
            covered, mapped and uncovered amounts. No global rescaling is applied: a partially
            covered source keeps its uncovered amount out of the target.
        """
        if method != "conservative":
            raise ValueError(f"ConservativeRegridder only supports method='conservative', got '{method}'")
        if preserve not in ("mass", "energy", "none"):
            raise ValueError(f"preserve must be 'mass', 'energy' or 'none', got {preserve!r}.")

        # Validate grid consistency (basic check)
        source_grid = source_grid or self.source_grid
        target_grid = target_grid or self.target_grid
        if hasattr(source_grid, 'dims') and source_grid.dims != self.source_grid.dims:
            raise ValueError(f"Source grid dimensions mismatch: expected {self.source_grid.dims}, got {source_grid.dims}")
        if hasattr(target_grid, 'dims') and target_grid.dims != self.target_grid.dims:
            raise ValueError(f"Target grid dimensions mismatch: expected {self.target_grid.dims}, got {target_grid.dims}")

        result = self._remap_to_target(
            data,
            quantity_basis=quantity_basis,
            range_clamp=preserve == "none",
        )
        report = self.last_regrid_report
        if preserve != "none" and not report["accepted"]:
            message = (
                f"The covered-support residual {report['absolute_residual']:.2e} (relative "
                f"{report['relative_residual']:.2e}) exceeds the accepted residual "
                f"{report['accepted_residual']:.2e} (tolerance {self.conservation_tolerance:.2e})."
            )
            if self.enforce_conservation:
                raise RuntimeError(message)
            logger.warning(message)

        if torch.any(torch.isfinite(data)) and not torch.isfinite(result).all():
            raise ValueError("Conservative regrid produced non-finite values for finite input.")
        if report["covered_source_fraction"] == 0.0 and bool(torch.any(data != 0)):
            raise ValueError("Conservative regrid has no source-target overlap; nothing was remapped.")
        return result
    
