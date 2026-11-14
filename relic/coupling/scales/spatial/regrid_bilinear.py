"""
Bilinear Interpolation Regridding Module

This module implements bilinear interpolation for spatial scale coupling
in the RELIC framework. Provides smooth, non-conservative interpolation
suitable for continuous state variables like temperature, pressure, and wind.

Bilinear interpolation uses weighted averaging of the four nearest grid points
to estimate values at intermediate locations. Produces smooth fields but does
not conserve total mass/energy - appropriate for intensive quantities.

Mathematical Framework:
    Bilinear interpolation formula:
    phi(x,y) = a_0_0 + a_1_0x + a_0_1y + a_1_1xy
    
    Weight calculation:
    w_1 = (1-dx) * (1-dy)  [lower-left]
    w_2 = dx * (1-dy)      [lower-right]  
    w_3 = (1-dx) * dy      [upper-left]
    w_4 = dx * dy          [upper-right]
    
    where dx, dy = normalized distances to cell center
    
    Interpolated value:
    phi(x,y) = w_1*phi_1 + w_2*phi_2 + w_3*phi_3 + w_4*phi_4

Classes:
    BilinearRegridder: Main bilinear interpolation implementation

Functions:
    compute_bilinear_weights: Calculate interpolation weights
    find_enclosing_cells: Locate surrounding grid cells
    normalize_coordinates: Convert to normalized grid coordinates

References:
    - Press, W.H. et al. (2007). Numerical Recipes: The Art of Scientific Computing
    - Burden, R.L. & Faires, J.D. (2010). Numerical Analysis, Chapter 3
    - ESMF Reference Manual: Bilinear interpolation methods
    - Akima, H. (1970). A new method of interpolation and smooth curve fitting
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

from .grid_utils import (
    GridDescriptor,
    flatten_grid_points,
    query_knn,
    target_coverage_report,
    typical_node_spacing,
)
from ....core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
from ....parameters.namespaces import ParameterNamespace, resolve_namespace
from .base_regridder import SpatialRegridderBase


class BilinearRegridder(SpatialRegridderBase):
    """
    Bilinear interpolation between spatial grids for intensive variables.

    Parameters
    ----------
    source_grid : GridDescriptor
        Source grid descriptor (areas in m^2 used for normalization).
    target_grid : GridDescriptor
        Target grid descriptor (areas in m^2).
    boundary_mode : str
        Boundary handling of a regular (rectilinear) source grid, dimensionless selector:
        "constant" leaves a target outside the source nodes (beyond the extrapolation
        tolerance) uncovered; "nearest" extends the edge values (the target coordinates are
        clamped to the node range before interpolating). On irregular grids a target is
        covered when a source node lies within the search radius, in either mode.
        Uncovered targets are missing: NaN in the returned field and counted in
        ``coverage_report``; no fill value exists.
    use_gpu_interpolation : bool
        Whether to enable GPU-accelerated interpolation (dimensionless flag).
    device : torch.device, optional
        Target computation device.

    Returns
    -------
    Dict[str, Tensor]
        Interpolated fields keyed by variable name; units preserved (intensive).

    References
    ----------
    Burden, R.L. & Faires, J.D. (2010). Numerical Analysis.
    Press, W.H. et al. (2007). Numerical Recipes.
    ESMF User's Guide: Bilinear interpolation chapter.

    Mathematical Expression
    -----------------------
    Bilinear interpolation: ``phi(x,y) = sum_{i=1..4} w_i * phi_i``, ``sum w_i = 1``.
    Weights (normalized coordinates): ``w1=(1-u)(1-v)``, ``w2=u(1-v)``, ``w3=(1-u)v``, ``w4=uv``.
    Parameters: ``u,v`` [dimensionless], ``phi`` shares input units.
    """
    
    def __init__(
        self,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        boundary_mode: Optional[str] = None,
        use_gpu_interpolation: bool = True,
        device: Optional[torch.device] = None,
        parameter_manager: Optional[Any] = None,
    ) -> None:
        """
        Initialize bilinear regridder with source and target grids.

        Mathematical Expression:
            Grid mapping: Omega_source -> Omega_target
            Interpolation setup: compute_bilinear_weights(Omega_s, Omega_t)
            Boundary handling: covered targets by boundary_mode; uncovered targets are NaN
        """
        meta = FunctorMetadata(
            name="bilinear_regridder",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            description="Bilinear interpolation for intensive variables",
            mathematical_expression="phi(x,y)=sum_{i=1..4} w_i * phi_i; sum(w_i)=1",
            references=[
                "Press, W.H. et al. (2007). Numerical Recipes: The Art of Scientific Computing.",
                "Burden, R.L. & Faires, J.D. (2010). Numerical Analysis, Chapter 3.",
                "ESMF Reference Manual: Bilinear interpolation methods.",
            ],
            input_variables=["data"],
            output_variables=["regridded_data"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(
            source_grid=source_grid,
            target_grid=target_grid,
            name=meta.name,
            config={},
            metadata=meta,
            parameter_manager=parameter_manager,
            device=device,
            enforce_conservation=False,
        )
        self.use_gpu_interpolation = use_gpu_interpolation

        self._spatial_namespace = ParameterNamespace.COUPLING_SCALES_SPATIAL
        self.extrapolation_tolerance = float(
            self.get_parameter_value(
                "extrapolation_tolerance",
                namespace=self._spatial_namespace,
                units="dimensionless",
                validate=True,
                required=True,
            )
        )
        boundary_mode_param = str(
            self.get_parameter_value(
                "boundary_handling_mode",
                namespace=self._spatial_namespace,
                units="dimensionless",
                expected_type=str,
                validate=True,
                required=True,
            )
        )

        effective_boundary_mode = boundary_mode or boundary_mode_param
        # reflect and wrap have no implementation; they are rejected instead of running as a fill.
        valid_boundary_modes = ["constant", "nearest"]
        if effective_boundary_mode not in valid_boundary_modes:
            raise ValueError(
                f"Invalid boundary_mode: {effective_boundary_mode!r}; accepted modes are {valid_boundary_modes}."
            )
        self.boundary_mode = effective_boundary_mode

        self._irregular_namespace = f"{resolve_namespace(ParameterNamespace.COUPLING_SCALES_SPATIAL)}.irregular"
        self._irregular_neighbor_count = int(
            self.get_parameter_value(
                "neighbor_count",
                namespace=self._irregular_namespace,
                units="dimensionless",
                expected_type=int,
                validate=True,
                required=True,
            )
        )
        self._irregular_weight_power = float(
            self.get_parameter_value(
                "weight_power",
                namespace=self._irregular_namespace,
                units="dimensionless",
                expected_type=float,
                validate=True,
                required=True,
            )
        )
        self._irregular_max_radius = float(
            self.get_parameter_value(
                "max_search_radius",
                namespace=self._irregular_namespace,
                units="dimensionless",
                expected_type=float,
                validate=True,
                required=True,
            )
        )
        self._irregular_use_kdtree = bool(
            self.get_parameter_value(
                "use_kdtree",
                namespace=self._irregular_namespace,
                units="dimensionless",
                expected_type=bool,
                validate=True,
                required=True,
            )
        )
        self._irregular_kdtree_leaf_size = int(
            self.get_parameter_value(
                "kdtree_leaf_size",
                namespace=self._irregular_namespace,
                units="dimensionless",
                expected_type=int,
                validate=True,
                required=True,
            )
        )
        # The numerical threshold of the scale operators (coupling.scales.defaults.epsilon), used
        # here as the relative distance below which a target coincides with a source node.
        self._coincidence_epsilon = float(
            self.get_parameter_value(
                "epsilon",
                namespace=resolve_namespace(ParameterNamespace.COUPLING_SCALES_DEFAULTS),
                units="dimensionless",
                expected_type=float,
                validate=True,
                required=True,
            )
        )

        # Precompute interpolation weights: bilinear on two regular (rectilinear) grids, inverse
        # distance weighting on any other pair (compute_interpolation_weights).
        self.interpolation_kernel = (
            "bilinear"
            if self.source_grid.grid_type == "regular" and self.target_grid.grid_type == "regular"
            else "inverse_distance_weighting"
        )
        self.interpolation_weights = self.compute_interpolation_weights()

        # Create boundary mask and the coverage of the target cells (uncovered targets are NaN).
        self.boundary_mask = self._compute_boundary_mask()
        self.coverage_report = target_coverage_report(self.boundary_mask)

    # The one method and the one preserve setting of this regridder; a call may name them, and a
    # call naming another method or another preserve setting raises (nothing is defaulted).
    METHOD = "bilinear"
    PRESERVE = "none"

    def _prepare_run(
        self,
        state: Dict[str, Any],
        operation: str,
        data: Tensor,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **kwargs: Any,
    ) -> None:
        """Check the method and preserve requests of one call against this regridder."""
        _ = (operation, data, source_grid, target_grid)
        for source in (state, kwargs):
            if "method" in source and source["method"] != self.METHOD:
                raise ValueError(
                    f"BilinearRegridder performs method {self.METHOD!r}; the call requested method "
                    f"{source['method']!r}. Create the regridder of that method instead."
                )
            if "preserve" in source and source["preserve"] != self.PRESERVE:
                raise ValueError(
                    f"Bilinear interpolation conserves no amount; preserve={source['preserve']!r} "
                    "cannot be honoured. Use the conservative regridder for mass or energy."
                )

    def _execute_operation(
        self,
        data: Tensor,
        operation: str,
        **kwargs: Any,
    ) -> Tensor:
        """Run the kernel; the method and preserve requests were checked in _prepare_run."""
        kernel_kwargs = {key: value for key, value in kwargs.items() if key not in ("method", "preserve")}
        if operation == "downscale":
            return self.downscale(data, **kernel_kwargs)
        return self.upscale(data, **kernel_kwargs)

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
        """Attach interpolation diagnostics to bilinear outputs."""
        interpolation_info = {
            "operation": operation,
            "source_shape": tuple(input_data.shape),
            "target_shape": tuple(result.shape),
            "method": self.METHOD,
            "interpolation_kernel": self.interpolation_kernel,
            "preserve": self.PRESERVE,
            "boundary_mode": self.boundary_mode,
            "use_gpu_interpolation": self.use_gpu_interpolation,
            "coverage": dict(self.coverage_report),
        }
        if hasattr(source_grid, "dims"):
            interpolation_info["source_grid_dims"] = source_grid.dims
        if hasattr(target_grid, "dims"):
            interpolation_info["target_grid_dims"] = target_grid.dims
        if hasattr(self, "boundary_mask"):
            valid_fraction = torch.mean(self.boundary_mask.float()).item()
            interpolation_info["valid_fraction"] = valid_fraction

        logger.debug(
            "Bilinear regridding %s: %s -> %s with boundary mode %s",
            operation,
            tuple(input_data.shape),
            tuple(result.shape),
            self.boundary_mode,
        )
        return {"regridded_data": result, "interpolation_info": interpolation_info}

    def compute_interpolation_weights(self) -> Dict[str, Tensor]:
        """
        Compute bilinear interpolation weights for all target points.
        
        Precomputes interpolation weights and source cell indices for efficient
        repeated interpolation. Handles both regular and irregular grids.
        
        Returns:
            Dict[str, Tensor]
                Dictionary containing:
                - "weights": Bilinear weights [n_target, 4]
                - "indices": Source cell indices [n_target, 4, 2]
                - "valid_mask": Valid interpolation mask [n_target]
                
        Mathematical Expression:
            For each target point (x_t, y_t):
            1. Find enclosing source cell: (i, j) such that
               x_s[i] <= x_t < x_s[i+1], y_s[j] <= y_t < y_s[j+1]
            2. Compute normalized coordinates:
               u = (x_t - x_s[i]) / (x_s[i+1] - x_s[i])
               v = (y_t - y_s[j]) / (y_s[j+1] - y_s[j])
            3. Calculate bilinear weights:
               w_1 = (1-u)(1-v), w_2 = u(1-v), w_3 = (1-u)v, w_4 = uv
        """
        if self.source_grid.grid_type != "regular" or self.target_grid.grid_type != "regular":
            # For irregular grids, use general point location
            return self._compute_irregular_weights()
        else:
            # Optimized computation for regular grids
            return self._compute_regular_weights()
    
    def _axis_brackets(
        self,
        nodes: Tensor,
        targets: Tensor,
        tol: float,
        axis: str,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Bracketing source nodes, fraction and coverage of target coordinates on one axis."""
        # References: Press, W.H. et al. (2007) Numerical Recipes, 3rd ed., section 3.6 (bilinear
        #   interpolation on a rectilinear grid, bracketing by bisection); ESMF Reference Manual
        #   (bilinear regridding); scipy.ndimage mode 'nearest' (edge extension a a | a b c d | d d).
        # Mathematical expression: with ascending nodes c_0 < ... < c_(n-1) and a target t:
        #   t' = clamp(t, c_0, c_(n-1)); k = max{m : c_m <= t'} limited to [0, n-2];
        #   f = (t' - c_k) / (c_(k+1) - c_k); t is covered when c_0 - tol <= t <= c_(n-1) + tol
        #   (boundary_mode "constant") and always ("nearest", edge extension); a single node
        #   (n = 1) covers t when |t - c_0| <= tol (always for "nearest"), with f = 0.
        # Parameter meanings: c node coordinates of the source axis; t target coordinate; k lower
        #   bracketing node; f fraction towards node k + 1; tol extrapolation tolerance
        #   (coupling.scales.spatial.extrapolation_tolerance times the largest node spacing).
        # Units: c, t, t' and tol in the coordinate unit of the grid; f [1].
        # Domain/constraints: nodes strictly monotonic (a descending axis is bracketed on its
        #   reversed order and the indices are mapped back; any other order raises ValueError).
        # Sign convention: coordinates keep their sign; 0 <= f <= 1.
        nodes64 = nodes.detach().to(device=self.device, dtype=torch.float64).reshape(-1)
        points = targets.detach().to(device=self.device, dtype=torch.float64).reshape(-1)
        n_nodes = int(nodes64.numel())
        extend = self.boundary_mode == "nearest"
        if n_nodes == 1:
            zeros = torch.zeros(points.shape, dtype=torch.long, device=self.device)
            inside = (
                torch.ones(points.shape, dtype=torch.bool, device=self.device)
                if extend
                else (points - nodes64[0]).abs() <= tol
            )
            return zeros, zeros.clone(), torch.zeros_like(points), inside
        steps = nodes64[1:] - nodes64[:-1]
        if bool((steps > 0).all()):
            ascending, descending = nodes64, False
        elif bool((steps < 0).all()):
            ascending, descending = nodes64.flip(0), True
        else:
            raise ValueError(
                f"The {axis} coordinates of the regular source grid must be strictly monotonic for "
                "bilinear interpolation."
            )
        low, high = ascending[0], ascending[-1]
        inside = (
            torch.ones(points.shape, dtype=torch.bool, device=self.device)
            if extend
            else (points >= low - tol) & (points <= high + tol)
        )
        clamped = torch.minimum(torch.maximum(points, low), high)
        lower = (torch.searchsorted(ascending, clamped, right=True) - 1).clamp(0, n_nodes - 2)
        fraction = (clamped - ascending[lower]) / (ascending[lower + 1] - ascending[lower])
        upper = lower + 1
        if descending:
            lower, upper = n_nodes - 1 - lower, n_nodes - 1 - upper
        return lower, upper, fraction, inside

    def _compute_regular_weights(self) -> Dict[str, Tensor]:
        """Bilinear weights of every target node on a regular (rectilinear) source grid."""
        # References: Press, W.H. et al. (2007) Numerical Recipes, 3rd ed., section 3.6; ESMF
        #   Reference Manual (bilinear regridding).
        # Mathematical expression: y_k = (1-u)(1-v) x_(jl,il) + u(1-v) x_(jl,iu) + (1-u) v x_(ju,il)
        #   + u v x_(ju,iu), with (il, iu, u) the x brackets of the target x coordinate and
        #   (jl, ju, v) the y brackets of its y coordinate (_axis_brackets); the four weights
        #   sum to 1.
        # Parameter meanings: x source field on the [ny, nx] nodes; y_k target value k; u, v
        #   fractions along x and y; tol = coupling.scales.spatial.extrapolation_tolerance times
        #   the largest node spacing of the source grid (0 when every axis has a single node).
        # Units: y keeps the units of x; u, v and the weights [1].
        # Domain/constraints: 1-D x and y axes whose lengths match dims = (ny, nx) on both grids;
        #   fields are laid out [ny, nx] and the targets are enumerated row-major
        #   (k = j nx_t + i); a target is covered when both axes cover it, its target mask (if
        #   any) is True and no source node with a non-zero weight is masked; uncovered targets
        #   are NaN in the result and counted in coverage_report.
        # Sign convention: weights are non-negative.
        for role, grid in (("source", self.source_grid), ("target", self.target_grid)):
            if grid.x_coords.dim() != 1 or grid.y_coords.dim() != 1:
                raise ValueError(
                    f"Bilinear interpolation on a regular {role} grid needs 1-D x and y axes; "
                    "describe 2-D coordinates as a curvilinear grid."
                )
            if tuple(grid.dims) != (int(grid.y_coords.numel()), int(grid.x_coords.numel())):
                raise ValueError(
                    f"The regular {role} grid declares dims {tuple(grid.dims)} but has "
                    f"{grid.y_coords.numel()} y and {grid.x_coords.numel()} x coordinates; dims are "
                    "(ny, nx)."
                )
        src_x = self.source_grid.x_coords
        src_y = self.source_grid.y_coords
        spacings = [
            float((axis[1:] - axis[:-1]).abs().max().item()) for axis in (src_x, src_y) if axis.numel() > 1
        ]
        tol = (max(spacings) if spacings else 0.0) * self.extrapolation_tolerance

        x_lower, x_upper, u, x_inside = self._axis_brackets(src_x, self.target_grid.x_coords, tol, "x")
        y_lower, y_upper, v, y_inside = self._axis_brackets(src_y, self.target_grid.y_coords, tol, "y")
        ny_t, nx_t = int(v.numel()), int(u.numel())

        def _rows(values: Tensor) -> Tensor:
            return values.reshape(ny_t, 1).expand(ny_t, nx_t).reshape(-1)

        def _cols(values: Tensor) -> Tensor:
            return values.reshape(1, nx_t).expand(ny_t, nx_t).reshape(-1)

        uu, vv = _cols(u), _rows(v)
        weights = torch.stack([(1 - uu) * (1 - vv), uu * (1 - vv), (1 - uu) * vv, uu * vv], dim=1)
        rows = torch.stack([_rows(y_lower), _rows(y_lower), _rows(y_upper), _rows(y_upper)], dim=1)
        cols = torch.stack([_cols(x_lower), _cols(x_upper), _cols(x_lower), _cols(x_upper)], dim=1)
        indices = torch.stack([rows, cols], dim=2)
        valid_mask = (y_inside.reshape(ny_t, 1) & x_inside.reshape(1, nx_t)).reshape(-1)
        if self.target_grid.mask is not None:
            valid_mask = valid_mask & self.target_grid.mask.to(self.device).reshape(-1).bool()
        if self.source_grid.mask is not None:
            src_nx = int(src_x.numel())
            node_valid = self.source_grid.mask.to(self.device).reshape(-1).bool()[rows * src_nx + cols]
            valid_mask = valid_mask & (node_valid | (weights == 0)).all(dim=1)
        return {
            "weights": weights,
            "indices": indices,
            "valid_mask": valid_mask,
        }

    def _compute_irregular_weights(self) -> Dict[str, Tensor]:
        """
        Compute interpolation weights for irregular grids.
        
        More general but computationally intensive method for irregular grids.
        Uses spatial search algorithms to locate enclosing cells.
        """
        # References: Shepard (1968) Proc. 23rd ACM National Conference, 517-524 (inverse distance
        #   weighting); ESMF regridding guidance.
        # Mathematical expression: w_i = d_i^(-p) / sum_k d_k^(-p) over the k nearest nodes; a target
        #   with d_i <= eps * max(||p_t||, ||p_i||) coincides with node i, and the m coincident nodes
        #   share w = 1 / m.
        # Parameter meanings: d_i Euclidean distance between target point p_t and node p_i;
        #   p = weight_power; k = neighbor_count; nodes beyond max_search_radius are not neighbours;
        #   eps = coupling.scales.defaults.epsilon (relative coincidence threshold).
        # Units: coordinates and d in the coordinate unit of the grid; p, eps and w [1].
        # Domain/constraints: the weights of a valid target sum to 1 and, because both the weights
        #   and the coincidence test are ratios of distances, do not depend on the coordinate unit.
        # Sign convention: weights are non-negative.
        src_points, src_mask = flatten_grid_points(self.source_grid)
        tgt_points, tgt_mask = flatten_grid_points(self.target_grid)
        src_points = src_points.to(self.device)
        tgt_points = tgt_points.to(self.device)

        # References: ESMF Reference Manual (neighbour search on unstructured grids); Shepard (1968).
        # Mathematical expression: r = rho * s with s = median_i min_{k != i} ||p_i - p_k||.
        # Parameter meanings: r search radius; rho = coupling.scales.spatial.irregular.max_search_radius
        #   (a multiple of the typical spacing); s typical spacing of the source nodes p_i.
        # Units: p, s and r in the coordinate unit of the grid; rho [1].
        # Domain/constraints: at least two distinct source nodes (s > 0, raises otherwise); the
        #   neighbour set, and so the result, does not depend on the coordinate unit.
        # Sign convention: distances and radii are non-negative.
        search_radius = self._irregular_max_radius * typical_node_spacing(
            src_points,
            mask=src_mask,
            use_kdtree=self._irregular_use_kdtree,
            leaf_size=self._irregular_kdtree_leaf_size,
        )
        k = max(1, int(self._irregular_neighbor_count))
        distances, indices = query_knn(
            src_points,
            tgt_points,
            k=k,
            source_mask=src_mask.to(self.device) if src_mask is not None else None,
            use_kdtree=self._irregular_use_kdtree,
            leaf_size=self._irregular_kdtree_leaf_size,
            max_search_radius=search_radius,
        )

        distances = distances.to(self.device)
        indices = indices.to(self.device)
        valid_mask = torch.isfinite(distances).any(dim=1)
        if tgt_mask is not None:
            valid_mask = valid_mask & tgt_mask.to(self.device)

        weights = torch.zeros_like(distances, dtype=torch.float32, device=self.device)
        source_norm = torch.linalg.vector_norm(src_points.to(dtype=torch.float64), dim=1)
        target_norm = torch.linalg.vector_norm(tgt_points.to(dtype=torch.float64), dim=1)
        neighbour_norm = torch.where(
            indices >= 0, source_norm[indices.clamp(min=0)], torch.zeros_like(distances)
        )
        coincidence_scale = torch.maximum(target_norm.unsqueeze(1), neighbour_norm)
        zero_mask = distances <= self._coincidence_epsilon * coincidence_scale
        for idx in range(distances.shape[0]):
            if not valid_mask[idx]:
                continue
            if torch.any(zero_mask[idx]):
                count = int(zero_mask[idx].sum().item())
                weights[idx, zero_mask[idx]] = 1.0 / max(count, 1)
            else:
                inv = distances[idx] ** (-self._irregular_weight_power)
                inv = torch.where(torch.isfinite(inv), inv, torch.zeros_like(inv))
                denom = torch.sum(inv)
                if denom > 0:
                    weights[idx] = inv / denom

        return {
            "weights": weights,
            "flat_indices": indices,
            "valid_mask": valid_mask,
        }
    
    def _compute_boundary_mask(self) -> Tensor:
        """
        Compute mask indicating valid interpolation regions.
        
        Returns:
            Tensor
                Boolean mask [n_target] indicating valid interpolation points
        """
        return self.interpolation_weights["valid_mask"]

    def _apply_interpolation(self, source_data: Tensor) -> Tensor:
        """
        Apply the precomputed weights from the configured source grid to the target grid.

        Parameters
        ----------
        source_data : Tensor
            Floating field on the source grid, shape [..., src_ny, src_nx].

        Returns
        -------
        Tensor
            Interpolated field on the target grid, shape [..., tgt_ny, tgt_nx]; uncovered targets
            are NaN.
        """
        # References: Press, W.H. et al. (2007) Numerical Recipes, section 3.6 (bilinear);
        #   Shepard, D. (1968) (inverse distance weighting on irregular grids).
        # Mathematical expression: y_(b,k) = sum_m w_(k,m) x_(b, s_(k,m)) over the neighbours m of
        #   target k with a non-zero weight, for every leading batch index b; y_(b,k) = NaN for an
        #   uncovered target k.
        # Parameter meanings: x source values; y target values; w_(k,m) precomputed weights
        #   (bilinear corners or inverse-distance weights, summing to 1 for a covered target);
        #   s_(k,m) flat source index of neighbour m.
        # Units: y keeps the units of x; w [1].
        # Domain/constraints: a floating field only (integer class codes or counts raise
        #   TypeError: the weights are fractional and a missing target is NaN); the trailing
        #   axes must equal the source dims. A NaN source value with a non-zero weight makes its
        #   target NaN; a zero-weight neighbour does not contribute. The result keeps the graph
        #   of the input (autograd flows through the weighted sum).
        # Sign convention: values keep their sign; weights are non-negative.
        if not torch.is_floating_point(source_data):
            raise TypeError(
                f"Bilinear interpolation needs a floating field (its weights are fractional and an "
                f"uncovered target is NaN); got dtype {source_data.dtype}. Regrid class codes with the "
                "nearest-neighbour regridder, or cast the field explicitly."
            )
        src_dims = tuple(self.source_grid.dims)
        tgt_dims = tuple(self.target_grid.dims)
        if tuple(source_data.shape[-len(src_dims):]) != src_dims:
            raise ValueError(
                f"The field of shape {tuple(source_data.shape)} does not end with the source grid "
                f"dims {src_dims}."
            )
        batch_dims = tuple(source_data.shape[: source_data.dim() - len(src_dims)])
        src_cells = int(torch.tensor(src_dims).prod().item())
        source_flat = source_data.reshape(-1, src_cells)

        weights = self.interpolation_weights["weights"].to(device=source_data.device, dtype=source_data.dtype)
        valid_mask = self.interpolation_weights["valid_mask"].to(device=source_data.device)
        if "flat_indices" in self.interpolation_weights:
            flat_indices = self.interpolation_weights["flat_indices"].to(device=source_data.device)
        else:
            indices = self.interpolation_weights["indices"].to(device=source_data.device)
            src_nx = src_dims[-1]
            flat_indices = indices[..., 0] * src_nx + indices[..., 1]

        contributes = (flat_indices >= 0) & (weights != 0)
        gathered = source_flat[:, flat_indices.clamp(min=0)]
        terms = torch.where(contributes.unsqueeze(0), gathered * weights.unsqueeze(0), torch.zeros_like(gathered))
        values = terms.sum(dim=-1)
        missing = torch.full_like(values, float("nan"))
        target_flat = torch.where(valid_mask.unsqueeze(0), values, missing)
        return target_flat.reshape(batch_dims + tgt_dims)

    def upscale(self, coarse_data: Tensor) -> Tensor:
        """
        Bilinear upscaling from coarse to fine resolution.
        
        Performs bilinear interpolation from the source grid (coarse) to the
        target grid (fine). Provides smooth interpolation but does not conserve
        total mass/energy.
        
        Parameters:
            coarse_data : Tensor
                Data on source grid to be upscaled (coarse)
                Shape: [..., src_ny, src_nx] where src_* matches source_grid.dims
                Units: [data units]
                
        Returns:
            coarse_data : Tensor
                Interpolated data on target grid
                Shape: [..., tgt_ny, tgt_nx]
                Units: [data units] (not conserved)
                
            Mathematical Expression:
            Bilinear interpolation:
            phi_target[k] = Sigma? w[k,i] * phi_source[idx[k,i]]
            
            where w[k,i] are bilinear weights and idx[k,i] are source indices
        """
        return self._apply_interpolation(coarse_data)
    
    def downscale(self, fine_data: Tensor) -> Tensor:
        """
        Bilinear downscaling from fine to coarse resolution.

        Performs bilinear interpolation from the source grid (fine) to the
        target grid (coarse). Non-conservative but preserves units.
        """
        return self._apply_interpolation(fine_data)
    
    def validate_interpolation(self, test_data: Optional[Tensor] = None) -> Dict[str, float]:
        """
        Validate interpolation quality and accuracy.
        
        Tests interpolation accuracy using analytical functions or provided
        test data. Useful for debugging and quality assessment.
        
        Parameters:
            test_data : Optional[Tensor]
                Known test data for validation
                
        Returns:
            Dict[str, float]
                Validation metrics and error statistics
        """
        metrics = {}
        
        if test_data is None:
            # Create analytical test function: f(x,y) = x^2 + y^2
            src_x = self.source_grid.x_coords
            src_y = self.source_grid.y_coords
            src_y_2d, src_x_2d = torch.meshgrid(src_y, src_x, indexing='ij')
            test_data = src_x_2d**2 + src_y_2d**2
        
        # Interpolate test data
        interpolated = self.upscale(test_data.unsqueeze(0)).squeeze(0)
        
        # Calculate analytical values at target points
        tgt_x = self.target_grid.x_coords
        tgt_y = self.target_grid.y_coords
        tgt_y_2d, tgt_x_2d = torch.meshgrid(tgt_y, tgt_x, indexing='ij')
        analytical = tgt_x_2d**2 + tgt_y_2d**2
        
        # Compute error metrics
        valid_mask = self.boundary_mask.view(self.target_grid.dims)
        if torch.any(valid_mask):
            error = torch.abs(interpolated - analytical)
            valid_error = error[valid_mask]
            
            metrics["max_error"] = torch.max(valid_error).item()
            metrics["mean_error"] = torch.mean(valid_error).item()
            metrics["rms_error"] = torch.sqrt(torch.mean(valid_error**2)).item()
            metrics["valid_fraction"] = torch.mean(valid_mask.float()).item()
        else:
            metrics["max_error"] = float('inf')
            metrics["mean_error"] = float('inf') 
            metrics["rms_error"] = float('inf')
            metrics["valid_fraction"] = 0.0
        
        return metrics
    
    def get_interpolation_info(self) -> Dict[str, Any]:
        """
        Return information about the interpolation setup.
        
        Returns:
            Dict[str, Any]
                Interpolation configuration and statistics
        """
        info = {
            "source_grid_dims": self.source_grid.dims,
            "target_grid_dims": self.target_grid.dims,
            "source_coord_system": self.source_grid.coord_system,
            "target_coord_system": self.target_grid.coord_system,
            "boundary_mode": self.boundary_mode,
            "interpolation_kernel": self.interpolation_kernel,
            "use_gpu_interpolation": self.use_gpu_interpolation,
            "coverage": dict(self.coverage_report),
        }
        
        # Interpolation statistics
        valid_mask = self.interpolation_weights["valid_mask"]
        weights = self.interpolation_weights["weights"]
        
        info["num_target_points"] = len(valid_mask)
        info["num_valid_points"] = torch.sum(valid_mask).item()
        info["valid_fraction"] = torch.mean(valid_mask.float()).item()
        
        if torch.any(valid_mask):
            valid_weights = weights[valid_mask]
            info["min_weight"] = torch.min(valid_weights).item()
            info["max_weight"] = torch.max(valid_weights).item()
            info["mean_weight"] = torch.mean(valid_weights).item()
        
        return info
    
    def regrid(
        self,
        data: Tensor,
        source_grid: Any = None,
        target_grid: Any = None,
        *,
        method: str,
        preserve: str = "none"
    ) -> Tensor:
        """
        Regrid data between arbitrary spatial grids using bilinear interpolation.
        
        Implements the SpatialScalingBase abstract method using smooth bilinear
        interpolation that preserves continuity and smoothness of the field.
        
        Parameters:
            data : Tensor
                Data to regrid from source to target grid
                Shape matches source grid dimensions
            source_grid : Any
                Source grid specification (should match self.source_grid)
            target_grid : Any  
                Target grid specification (should match self.target_grid)
            method : str
                Regridding method (only "bilinear" supported for this class)
            preserve : str
                Quantity to preserve ("none" for bilinear - no conservation)
                
        Returns:
            Tensor: Regridded data on target grid with smooth interpolation
            
        Raises:
            ValueError: If method is not "bilinear" or grid mismatch detected
            
        Mathematical Expression:
            Bilinear interpolation:
            phi_target = Bilinear_Interp(phi_source)
            
            where Bilinear_Interp provides C^0 continuity
        """
        # Validate method and preserve: bilinear interpolation conserves no amount, so a request
        # to preserve one raises instead of being ignored.
        if method != "bilinear":
            raise ValueError(f"BilinearRegridder only supports method='bilinear', got '{method}'")
        if preserve != self.PRESERVE:
            raise ValueError(
                f"Bilinear interpolation conserves no amount; preserve={preserve!r} cannot be "
                "honoured. Use the conservative regridder for mass or energy."
            )
        
        # Validate grid consistency (basic check)
        source_grid = source_grid or self.source_grid
        target_grid = target_grid or self.target_grid
        if hasattr(source_grid, 'dims') and source_grid.dims != self.source_grid.dims:
            raise ValueError(f"Source grid dimensions mismatch: expected {self.source_grid.dims}, got {source_grid.dims}")
        if hasattr(target_grid, 'dims') and target_grid.dims != self.target_grid.dims:
            raise ValueError(f"Target grid dimensions mismatch: expected {self.target_grid.dims}, got {target_grid.dims}")
        
        # Determine operation based on grid sizes
        source_size = self.source_grid.dims[0] * self.source_grid.dims[1]  # ny * nx
        target_size = self.target_grid.dims[0] * self.target_grid.dims[1]  # ny * nx
        
        if target_size < source_size:
            # Upscaling: target has fewer cells (coarser resolution)
            result = self.upscale(data)
        elif target_size > source_size:
            # Downscaling: target has more cells (finer resolution)
            result = self.downscale(data)
        else:
            # Same size: still apply bilinear transformation to handle grid differences
            result = self.upscale(data)
        
        return result
    
