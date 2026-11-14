"""
Nearest Neighbor Interpolation Regridding Module

This module implements nearest neighbor interpolation for spatial scale coupling
in the RELIC framework. Designed for categorical variables, masks, and discrete
data where preservation of exact values is more important than smoothness.

Nearest neighbor interpolation assigns the value of the closest source grid cell
to each target location. Preserves discrete values and sharp boundaries but
produces discontinuous, blocky results. Ideal for land use types, vegetation
classes, and binary masks.

Mathematical Framework:
    Nearest neighbor assignment:
    phi(x,y) = phi(x_nearest, y_nearest)
    
    Distance metric (Euclidean):
    d = sqrt[(x-x?)^2 + (y-y?)^2]
    
    Nearest cell selection:
    (i_nn, j_nn) = argmin_{i,j} d(x,y; x?,y?)
    
    Assignment:
    phi_target = phi_source[i_nn, j_nn]

Classes:
    NearestNeighborRegridder: Main nearest neighbor implementation

Functions:
    find_nearest_neighbors: Locate nearest source cells
    compute_distance_matrix: Calculate all pairwise distances
    assign_nearest_values: Perform nearest neighbor assignment

References:
    - Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data
    - Tobler, W.R. (1970). A computer movie simulating urban growth in the Detroit region
    - Li, J. & Heap, A.D. (2008). A review of spatial interpolation methods for environmental scientists
    - ESMF Reference Manual: Nearest neighbor interpolation methods
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

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


class NearestNeighborRegridder(SpatialRegridderBase):
    """
    Nearest neighbor interpolation between spatial grids for categorical/discrete fields.

    Parameters
    ----------
    source_grid : GridDescriptor
        Source grid description with coordinates and areas [m^2].
    target_grid : GridDescriptor
        Target grid description with coordinates and areas [m^2].
    distance_metric : str
        Distance metric selector ("euclidean", "manhattan", "chebyshev"), dimensionless.
    use_kdtree : bool
        Whether to use KDTree acceleration (dimensionless flag).
    boundary_mode : str
        Boundary handling ("constant", "nearest"), dimensionless selector: "constant" covers a
        target only when a source node lies within the search radius; "nearest" gives every
        target its nearest source node. An uncovered target is missing (NaN; an integer field
        with an uncovered target raises) and is counted in ``coverage_report``; no fill value
        exists.
    device : torch.device, optional
        Target computation device.

    Returns
    -------
    Dict[str, Tensor]
        Remapped fields keyed by variable name; categorical values preserved.

    References
    ----------
    Shepard, D. (1968). Two-dimensional interpolation function.
    Li, J. & Heap, A.D. (2008). Spatial interpolation methods.
    ESMF User's Guide: Nearest neighbor interpolation.

    Mathematical Expression
    -----------------------
    ``d_{k,i} = dist(target_k, source_i)``, ``i* = argmin_i d_{k,i}``, ``phi_t(k) = phi_s(i*)``.
    Parameters: distances [m], ``phi`` shares input units (typically dimensionless classes).
    """

    # The one method this regridder performs and the one preserve it honours: nearest-neighbour
    # assignment conserves no amount. forward and regrid refuse any other request.
    METHOD = "nearest"
    PRESERVE = "none"
    
    def __init__(
        self,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        distance_metric: str = "euclidean",
        use_kdtree: Optional[bool] = None,
        boundary_mode: Optional[str] = None,
        device: Optional[torch.device] = None,
        parameter_manager: Optional[Any] = None,
    ) -> None:
        """
        Initialize nearest neighbor regridder with source and target grids.
        
        Mathematical Expression:
            Grid mapping: Omega_source -> Omega_target
            NN computation: compute_nearest_neighbors(Omega_s, Omega_t, metric)
            Boundary handling: covered targets by boundary_mode; uncovered targets are NaN
        """
        meta = FunctorMetadata(
            name="nearest_neighbor_regridder",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            description="Nearest neighbor regridding for categorical fields",
            mathematical_expression="phi_t(x_j)=phi_s(x_i*), i*=argmin_i ||x_j-x_i||",
            references=[
                "Shepard, D. (1968). Two-dimensional interpolation function for irregularly spaced data.",
                "ESMF Reference Manual: Nearest neighbor interpolation.",
            ],
            input_variables=["data"],
            output_variables=["regridded"],
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
        
        valid_metrics = ["euclidean", "manhattan", "chebyshev"]
        if distance_metric not in valid_metrics:
            raise ValueError(f"Invalid distance_metric: {distance_metric}")
        
        self.distance_metric = distance_metric
        self._irregular_namespace = f"{resolve_namespace(ParameterNamespace.COUPLING_SCALES_SPATIAL)}.irregular"
        use_kdtree_param = self.get_parameter_value(
            "use_kdtree",
            namespace=self._irregular_namespace,
            units="dimensionless",
            expected_type=bool,
            validate=True,
            required=True,
        )
        self.use_kdtree = bool(use_kdtree if use_kdtree is not None else use_kdtree_param)
        self._kdtree_leaf_size = int(
            self.get_parameter_value(
                "kdtree_leaf_size",
                namespace=self._irregular_namespace,
                units="dimensionless",
                expected_type=int,
                validate=True,
                required=True,
            )
        )
        self._max_search_radius = float(
            self.get_parameter_value(
                "max_search_radius",
                namespace=self._irregular_namespace,
                units="dimensionless",
                expected_type=float,
                validate=True,
                required=True,
            )
        )

        self._spatial_namespace = ParameterNamespace.COUPLING_SCALES_SPATIAL
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
        valid_boundary_modes = ["constant", "nearest"]
        if effective_boundary_mode not in valid_boundary_modes:
            raise ValueError(f"Invalid boundary_mode: {effective_boundary_mode}")
        self.boundary_mode = effective_boundary_mode

        self._regular_radius_diagonals = float(
            self.get_parameter_value(
                "regular_search_radius_cell_diagonals",
                namespace=self._spatial_namespace,
                units="dimensionless",
                expected_type=float,
                validate=True,
                required=True,
            )
        )

        # Nearest source node of every target node (upscale: source grid -> target grid) and its
        # coverage; the reverse mapping (downscale: target grid -> source grid) is built on first use.
        self.nearest_indices, self.boundary_mask = self.compute_nearest_neighbors(
            source_grid=self.source_grid,
            target_grid=self.target_grid,
        )
        self.coverage_report = target_coverage_report(self.boundary_mask)
        self._reverse_mapping: Optional[Tuple[Tensor, Tensor]] = None

        # Optional: compute distance matrix for diagnostics
        self.distance_matrix = None

    def _reverse_neighbors(self) -> Tuple[Tensor, Tensor]:
        """Nearest target node of every source node (the downscale mapping), built once."""
        if self._reverse_mapping is None:
            self._reverse_mapping = self.compute_nearest_neighbors(
                source_grid=self.target_grid,
                target_grid=self.source_grid,
            )
        return self._reverse_mapping

    @property
    def reverse_coverage_report(self) -> Dict[str, int]:
        """Coverage of the source cells by the downscale mapping (target grid -> source grid)."""
        return target_coverage_report(self._reverse_neighbors()[1])

    def _detect_operation(
        self,
        state: Dict[str, Any],
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **kwargs: Any,
    ) -> str:
        """The call's operation, else the grid the data fit: the source grid -> upscale, the target
        grid -> downscale; data that fit both or neither grid raise (no direction is assumed)."""
        operation = kwargs["operation"] if "operation" in kwargs else state.get("operation")
        if operation is not None:
            if operation not in {"upscale", "downscale"}:
                raise ValueError(f"Invalid operation: {operation!r}")
            return operation

        data = kwargs["data"] if "data" in kwargs else state.get("data")
        sample = next(iter(data.values())) if isinstance(data, dict) and data else data
        if not isinstance(sample, torch.Tensor):
            sample = torch.as_tensor(sample)
        src_dims = tuple(source_grid.dims)
        tgt_dims = tuple(target_grid.dims)
        trailing = tuple(sample.shape[-len(src_dims):]) if sample.dim() >= len(src_dims) else None
        fits_source = trailing == src_dims
        trailing = tuple(sample.shape[-len(tgt_dims):]) if sample.dim() >= len(tgt_dims) else None
        fits_target = trailing == tgt_dims
        if fits_source and fits_target:
            raise ValueError(
                f"The data fit both the source grid {src_dims} and the target grid {tgt_dims}; pass "
                "operation='upscale' (source -> target) or 'downscale' (target -> source)."
            )
        if fits_source:
            return "upscale"
        if fits_target:
            return "downscale"
        raise ValueError(
            f"The data of shape {tuple(sample.shape)} fit neither the source grid {src_dims} nor the "
            f"target grid {tgt_dims}."
        )

    def _extract_data(
        self,
        state: Dict[str, Any],
        source_grid: GridDescriptor,
        **kwargs: Any,
    ) -> Any:
        """Allow tensor or dict inputs for nearest neighbor regridding."""
        data = kwargs.get("data", state.get("data"))
        if data is None:
            raise ValueError("NearestNeighborRegridder requires 'data' in kwargs or state.")
        if isinstance(data, dict):
            return {
                key: (value.to(self.device) if isinstance(value, Tensor) else torch.as_tensor(value, device=self.device))
                for key, value in data.items()
            }
        if isinstance(data, Tensor):
            return data.to(self.device)
        return torch.as_tensor(data, device=self.device)

    def _prepare_run(
        self,
        state: Dict[str, Any],
        operation: str,
        data: Any,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **kwargs: Any,
    ) -> None:
        """Check the method and preserve requests of one call against this regridder."""
        _ = (operation, data, source_grid, target_grid)
        for source in (state, kwargs):
            if "method" in source and source["method"] != self.METHOD:
                raise ValueError(
                    f"NearestNeighborRegridder performs method {self.METHOD!r}; the call requested "
                    f"method {source['method']!r}. Create the regridder of that method instead."
                )
            if "preserve" in source and source["preserve"] != self.PRESERVE:
                raise ValueError(
                    f"Nearest-neighbour assignment conserves no amount; preserve={source['preserve']!r} "
                    "cannot be honoured. Use the conservative regridder for mass or energy."
                )

    def _execute_operation(
        self,
        data: Any,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        """Apply nearest neighbor mapping to tensor or dict payloads (method and preserve were
        checked in _prepare_run)."""
        def _apply(tensor: Tensor) -> Tensor:
            return self.downscale(tensor) if operation == "downscale" else self.upscale(tensor)

        if isinstance(data, dict):
            return {name: _apply(tensor) for name, tensor in data.items()}
        return _apply(data)

    def _format_output(
        self,
        result: Any,
        operation: str,
        state: Dict[str, Any],
        *,
        input_data: Any,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Structure outputs with the metadata contract."""
        metadata = {
            "operation": operation,
            "method": self.METHOD,
            "preserve": self.PRESERVE,
            "distance_metric": self.distance_metric,
            "boundary_mode": self.boundary_mode,
            "coverage": dict(
                self.reverse_coverage_report if operation == "downscale" else self.coverage_report
            ),
            "references": [
                "Shepard (1968) nearest neighbor interpolation",
                "Jones (1999) grid remapping frameworks",
            ],
        }
        if isinstance(result, dict):
            output = dict(result)
        else:
            output = {"regridded": result}
        output["_regrid_metadata"] = metadata
        return output

    def compute_nearest_neighbors(
        self,
        *,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute nearest neighbor indices for all target points.
        
        Finds the closest source grid cell for each target grid cell using
        the specified distance metric. Handles boundary conditions and masking.
        
        Returns:
            Tuple[Tensor, Tensor]
                (nearest_indices, boundary_mask)
                nearest_indices: Source cell indices [n_target, 2] (j, i format)
                boundary_mask: Valid assignment mask [n_target]
                
        Mathematical Expression:
            For each target point (x_t, y_t):
            distance[i,j] = metric(x_t, y_t; x_s[i], y_s[j])
            (i_nn, j_nn) = argmin_{i,j} distance[i,j]
            
            Distance metrics:
            - Euclidean: sqrt[(x_t-x_s)^2 + (y_t-y_s)^2]
            - Manhattan: |x_t-x_s| + |y_t-y_s|
            - Chebyshev: max(|x_t-x_s|, |y_t-y_s|)
        """
        if self.use_kdtree and self.distance_metric == "euclidean":
            return self._compute_kdtree_neighbors(source_grid=source_grid, target_grid=target_grid)
        return self._compute_brute_force_neighbors(source_grid=source_grid, target_grid=target_grid)
    
    def _axis_spacing(self, coords: Tensor, bounds: Optional[Tensor], axis: str) -> float:
        """Largest node spacing of one axis of a regular grid (cell width for a single node)."""
        if coords.numel() > 1:
            return float((coords[1:] - coords[:-1]).abs().max().item())
        if bounds is not None:
            return float((bounds.reshape(-1, 2)[0, 1] - bounds.reshape(-1, 2)[0, 0]).abs().item())
        raise ValueError(
            f"The {axis} axis of the regular source grid has a single node and no cell bounds; the "
            "search radius of boundary_mode='constant' is undefined. Declare the cell bounds or use "
            "boundary_mode='nearest'."
        )

    def _search_radius(
        self,
        source_grid: GridDescriptor,
        src_points: Tensor,
        src_mask: Optional[Tensor],
    ) -> Optional[float]:
        """Search radius of boundary_mode 'constant' (None for 'nearest': every target is covered)."""
        if self.boundary_mode != "constant":
            return None
        if source_grid.grid_type == "regular" and source_grid.x_coords.dim() == 1 and source_grid.y_coords.dim() == 1:
            # References: ESMF Reference Manual (unmapped destination points of a neighbour search);
            #   Shepard, D. (1968) (nearest-neighbour search).
            # Mathematical expression: r = q * sqrt(dx^2 + dy^2), dx = max_i |x_(i+1) - x_i|,
            #   dy = max_j |y_(j+1) - y_j| (the cell width from the bounds for a single-node axis).
            # Parameter meanings: r search radius; q = coupling.scales.spatial.
            #   regular_search_radius_cell_diagonals (radius in source cell diagonals); dx, dy
            #   largest node spacings of the source axes.
            # Units: r, dx and dy in the coordinate unit of the grid; q [1].
            # Domain/constraints: a single-node axis needs its cell bounds (raises otherwise); the
            #   same radius applies to the brute-force and the KD-tree search.
            # Sign convention: distances and radii are non-negative.
            dx = self._axis_spacing(source_grid.x_coords, source_grid.x_bounds, "x")
            dy = self._axis_spacing(source_grid.y_coords, source_grid.y_bounds, "y")
            return self._regular_radius_diagonals * math.sqrt(dx * dx + dy * dy)
        # References: ESMF Reference Manual (neighbour search on unstructured grids); Shepard (1968).
        # Mathematical expression: r = rho * s with s = median_i min_{k != i} ||p_i - p_k||.
        # Parameter meanings: r search radius; rho = coupling.scales.spatial.irregular.max_search_radius
        #   (a multiple of the typical spacing); s typical spacing of the source nodes p_i.
        # Units: p, s and r in the coordinate unit of the grid; rho [1].
        # Domain/constraints: at least two distinct source nodes (s > 0, raises otherwise); the
        #   neighbour set, and so the result, does not depend on the coordinate unit.
        # Sign convention: distances and radii are non-negative.
        return self._max_search_radius * typical_node_spacing(
            src_points, mask=src_mask, use_kdtree=self.use_kdtree, leaf_size=self._kdtree_leaf_size
        )

    def _compute_brute_force_neighbors(
        self,
        *,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute nearest neighbors using brute force distance calculation.
        
        Efficient for small to medium grids. Computes all pairwise distances
        and selects minimum for each target point.
        """
        # References: Bentley (1975) KD-tree, Shepard (1968) nearest neighbor
        # Math: d = ||x_t - x_s||, i* = argmin_i d
        # Parameters: distance_metric, max_search_radius, boundary_mode
        src_points, src_mask = flatten_grid_points(source_grid)
        tgt_points, tgt_mask = flatten_grid_points(target_grid)
        src_points = src_points.to(self.device)
        tgt_points = tgt_points.to(self.device)

        diff = tgt_points[:, None, :] - src_points[None, :, :]
        if self.distance_metric == "euclidean":
            distances = torch.norm(diff, dim=2)
        elif self.distance_metric == "manhattan":
            distances = torch.sum(torch.abs(diff), dim=2)
        else:
            distances = torch.max(torch.abs(diff), dim=2)[0]

        if src_mask is not None:
            distances = distances.masked_fill(~src_mask.to(self.device), float("inf"))

        min_dist, min_idx = torch.min(distances, dim=1)
        nearest_indices = min_idx.to(dtype=torch.long)

        boundary_mask = torch.isfinite(min_dist)
        radius = self._search_radius(source_grid, src_points, src_mask)
        if radius is not None:
            boundary_mask = boundary_mask & (min_dist <= radius)

        if tgt_mask is not None:
            boundary_mask = boundary_mask & tgt_mask.to(self.device)

        return nearest_indices, boundary_mask
    
    def _compute_kdtree_neighbors(
        self,
        *,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute nearest neighbors using KDTree for efficiency.
        
        More efficient for large grids but requires additional dependencies.
        Currently not implemented - would use scipy.spatial.cKDTree or similar.
        """
        # References: Bentley (1975) KD-tree
        # Math: nearest neighbor query in Euclidean space
        # Parameters: kdtree_leaf_size, max_search_radius
        src_points, src_mask = flatten_grid_points(source_grid)
        tgt_points, tgt_mask = flatten_grid_points(target_grid)
        src_points = src_points.to(self.device)
        tgt_points = tgt_points.to(self.device)

        search_radius = self._search_radius(source_grid, src_points, src_mask)
        distances, indices = query_knn(
            src_points,
            tgt_points,
            k=1,
            source_mask=src_mask.to(self.device) if src_mask is not None else None,
            use_kdtree=True,
            leaf_size=self._kdtree_leaf_size,
            max_search_radius=search_radius,
        )

        nearest_indices = indices[:, 0]
        min_dist = distances[:, 0]
        boundary_mask = torch.isfinite(min_dist) & (nearest_indices >= 0)
        if tgt_mask is not None:
            boundary_mask = boundary_mask & tgt_mask.to(self.device)
        return nearest_indices, boundary_mask
    
    def _assign(self, values: Tensor, indices: Tensor, covered: Tensor, in_dims: Tuple[int, ...], out_dims: Tuple[int, ...]) -> Tensor:
        """Nearest-neighbour assignment of a field; uncovered targets are missing (NaN)."""
        # References: Shepard, D. (1968); ESMF Reference Manual (nearest-neighbour regridding,
        #   unmapped destination points).
        # Mathematical expression: phi_t(b, k) = phi_s(b, i*_k) for a covered target k and
        #   phi_t(b, k) = NaN otherwise, for every leading batch index b.
        # Parameter meanings: phi_s input field; phi_t output field; i*_k nearest input node of
        #   output node k; covered_k whether k has a nearest node within the search radius.
        # Units: phi_t keeps the units of phi_s (values are copied, never averaged).
        # Domain/constraints: the trailing axes of the field must equal the input dims; a
        #   non-floating field (class codes, counts, flags) keeps its dtype when every target is
        #   covered and raises ValueError when a target is uncovered (it cannot hold NaN).
        # Sign convention: values keep their sign.
        if tuple(values.shape[-len(in_dims):]) != tuple(in_dims):
            raise ValueError(
                f"The field of shape {tuple(values.shape)} does not end with the grid dims {tuple(in_dims)}."
            )
        batch_dims = tuple(values.shape[: values.dim() - len(in_dims)])
        in_cells = int(torch.tensor(in_dims).prod().item())
        out_cells = int(torch.tensor(out_dims).prod().item())
        covered = covered.to(device=values.device)
        uncovered = int((~covered).sum().item())
        if uncovered and not torch.is_floating_point(values):
            raise ValueError(
                f"{uncovered} of {out_cells} target cells are uncovered (no source node within the "
                f"search radius) and the {values.dtype} field cannot hold a missing value; use "
                "boundary_mode='nearest', a floating field, or a target grid inside the source support."
            )
        flat = values.reshape(-1, in_cells)
        gathered = flat[:, indices.to(device=values.device).clamp(min=0)]
        if uncovered:
            gathered = torch.where(covered.unsqueeze(0), gathered, torch.full_like(gathered, float("nan")))
        return gathered.reshape(batch_dims + tuple(out_dims))

    def upscale(self, fine_data: Tensor) -> Tensor:
        """
        Nearest neighbor upscaling from fine to coarse resolution.
        
        Assigns the value of the nearest fine grid cell to each coarse grid cell.
        Preserves discrete values and sharp boundaries without smoothing.
        
        Parameters:
            fine_data : Tensor
                Data on source grid to be upscaled
                Shape: [..., src_ny, src_nx]
                Units: [data units]
                
        Returns:
            coarse_data : Tensor
                Nearest neighbor assignment on target grid
                Shape: [..., tgt_ny, tgt_nx]
                Units: [data units] (values preserved exactly)
                
        Mathematical Expression:
            Nearest neighbor assignment:
            phi_target[k] = phi_source[nn_idx[k]]
            
            where nn_idx[k] = (j_nn, i_nn) is the nearest source cell index
            
            Categorical preservation:
            unique(phi_target) subseteq unique(phi_source)
        """
        return self._assign(
            fine_data,
            self.nearest_indices,
            self.boundary_mask,
            tuple(self.source_grid.dims),
            tuple(self.target_grid.dims),
        )

    def downscale(self, coarse_data: Tensor) -> Tensor:
        """
        Nearest neighbor downscaling from coarse to fine resolution.
        
        Assigns target grid values to source grid using nearest neighbor
        lookup. Creates blocky, piecewise constant interpolation.
        
        Parameters:
            coarse_data : Tensor
                Data on target grid to be downscaled
                Shape: [..., tgt_ny, tgt_nx]
                Units: [data units]
                
        Returns:
            fine_data : Tensor
                Nearest neighbor assignment on source grid
                Shape: [..., src_ny, src_nx]
                Units: [data units] (values preserved exactly)
                
        Mathematical Expression:
            Reverse nearest neighbor assignment:
            For each source point, find nearest target point
            phi_source[k] = phi_target[nn_target_idx[k]]
            
            Creates blocky appearance with sharp boundaries
        """
        reverse_indices, reverse_mask = self._reverse_neighbors()
        return self._assign(
            coarse_data,
            reverse_indices,
            reverse_mask,
            tuple(self.target_grid.dims),
            tuple(self.source_grid.dims),
        )

    def validate_categories(self, data: Tensor) -> Dict[str, Any]:
        """
        Validate categorical data preservation in interpolation.
        
        Checks that nearest neighbor interpolation preserves discrete values
        and doesn't introduce new categories. Useful for categorical variables.
        
        Parameters:
            data : Tensor
                Input categorical data to validate
                
        Returns:
            Dict[str, Any]
                Validation results and category statistics
        """
        results = {}
        
        # Get unique values in input
        input_unique = torch.unique(data)
        results["input_categories"] = input_unique.tolist()
        results["num_input_categories"] = len(input_unique)
        
        # Test interpolation (uncovered targets are missing and carry no category)
        interpolated = self.upscale(data.unsqueeze(0)).squeeze(0)
        if torch.is_floating_point(interpolated):
            interpolated = interpolated[~torch.isnan(interpolated)]
        output_unique = torch.unique(interpolated)
        results["output_categories"] = output_unique.tolist()
        results["num_output_categories"] = len(output_unique)
        
        # Check preservation
        preserved = torch.all(torch.isin(output_unique, input_unique))
        results["categories_preserved"] = preserved.item()
        
        if not preserved:
            new_categories = output_unique[~torch.isin(output_unique, input_unique)]
            results["new_categories"] = new_categories.tolist()
        else:
            results["new_categories"] = []
        
        # Coverage statistics
        input_counts = torch.bincount(data.view(-1).long())
        output_counts = torch.bincount(interpolated.view(-1).long())
        
        results["input_distribution"] = input_counts.tolist()
        results["output_distribution"] = output_counts.tolist()
        
        return results
    
    def get_interpolation_statistics(self) -> Dict[str, Any]:
        """
        Return statistics about the nearest neighbor assignment.
        
        Returns:
            Dict[str, Any]
                Interpolation configuration and usage statistics
        """
        stats = {
            "source_grid_dims": self.source_grid.dims,
            "target_grid_dims": self.target_grid.dims,
            "source_coord_system": self.source_grid.coord_system,
            "target_coord_system": self.target_grid.coord_system,
            "distance_metric": self.distance_metric,
            "boundary_mode": self.boundary_mode,
            "use_kdtree": self.use_kdtree,
            "coverage": dict(self.coverage_report),
        }
        
        # Assignment statistics
        n_target = len(self.boundary_mask)
        n_valid = torch.sum(self.boundary_mask).item()
        
        stats["num_target_points"] = n_target
        stats["num_valid_assignments"] = n_valid
        stats["valid_assignment_fraction"] = n_valid / n_target if n_target > 0 else 0.0
        
        # Source cell usage statistics
        if n_valid > 0:
            valid_indices = self.nearest_indices[self.boundary_mask]
            unique_sources, counts = torch.unique(valid_indices, return_counts=True)
            src_cells = int(torch.tensor(self.source_grid.dims).prod().item())

            stats["num_unique_sources_used"] = len(unique_sources)
            stats["max_source_usage"] = torch.max(counts).item()
            stats["mean_source_usage"] = torch.mean(counts.float()).item()
            stats["source_usage_fraction"] = len(unique_sources) / src_cells if src_cells > 0 else 0.0
        
        return stats
    
    def get_interpolation_info(self) -> Dict[str, Any]:
        """
        Return comprehensive information about the interpolation setup.
        
        Returns:
            Dict[str, Any]
                Complete interpolation configuration and diagnostics
        """
        return self.get_interpolation_statistics()
    
    def regrid(
        self,
        data: Tensor,
        source_grid: Any,
        target_grid: Any,
        method: str,
        preserve: str = "none"
    ) -> Tensor:
        """
        Regrid data between arbitrary spatial grids using nearest neighbor interpolation.
        
        Implements the SpatialScalingBase abstract method using nearest neighbor
        assignment that preserves discrete patterns and categorical data.
        
        Parameters:
            data : Tensor
                Data to regrid from source to target grid
                Shape matches source grid dimensions
            source_grid : Any
                Source grid specification (should match self.source_grid)
            target_grid : Any  
                Target grid specification (should match self.target_grid)
            method : str
                Regridding method (only "nearest" supported for this class)
            preserve : str
                Quantity to preserve ("none" for nearest - no conservation)
                
        Returns:
            Tensor: Regridded data on target grid with nearest neighbor values
            
        Raises:
            ValueError: If method is not "nearest" or grid mismatch detected
            
        Mathematical Expression:
            Nearest neighbor interpolation:
            phi_target = Nearest_Neighbor(phi_source)
            
            where each target cell gets value from closest source cell
        """
        # Validate method and preserve: nearest-neighbour assignment conserves no amount, so a
        # request to preserve one raises instead of being ignored.
        if method != self.METHOD:
            raise ValueError(f"NearestRegridder only supports method='nearest', got '{method}'")
        if preserve != self.PRESERVE:
            raise ValueError(
                f"Nearest-neighbour assignment conserves no amount; preserve={preserve!r} cannot be "
                "honoured. Use the conservative regridder for mass or energy."
            )
        
        # Validate grid consistency (basic check)
        if hasattr(source_grid, 'dims') and source_grid.dims != self.source_grid.dims:
            raise ValueError(f"Source grid dimensions mismatch: expected {self.source_grid.dims}, got {source_grid.dims}")
        if hasattr(target_grid, 'dims') and target_grid.dims != self.target_grid.dims:
            raise ValueError(f"Target grid dimensions mismatch: expected {self.target_grid.dims}, got {target_grid.dims}")
        
        # Determine operation based on grid sizes
        source_size = int(torch.tensor(self.source_grid.dims).prod().item())
        target_size = int(torch.tensor(self.target_grid.dims).prod().item())
        
        if target_size < source_size:
            # Upscaling: target has fewer cells (coarser resolution)
            result = self.upscale(data)
        elif target_size > source_size:
            # Downscaling: target has more cells (finer resolution)
            result = self.downscale(data)
        else:
            # Same size: still apply nearest neighbor transformation to handle grid differences
            result = self.upscale(data)
        
        return result
    
