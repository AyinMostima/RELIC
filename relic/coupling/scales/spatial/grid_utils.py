"""
Grid Utilities and Area Calculation Module

This module provides grid coordinate handling and cell area computation
functionality for spatial regridding operations in the RELIC framework.
Supports both regular and irregular grids with accurate area calculations
considering Earth's curvature for latitude-longitude grids.

The GridUtils class provides essential geometric calculations for conservative
remapping while maintaining PyTorch differentiability and bounded grid metadata.

Mathematical Framework:
    Spherical cell area (lat-lon grids):
    A = R^2 * Deltalambda * (sin(phi_2) - sin(phi_1))
    where R = Earth radius, Deltalambda = longitude difference, phi_1,phi_2 = latitude bounds
    
    Cartesian cell area (projected grids):
    A = Deltax * Deltay
    where Deltax, Deltay = grid spacing in x, y directions
    
    Grid coordinate transformations:
    (i, j) <-> (x, y) for regular grids
    x = x_0 + i * dx, y = y_0 + j * dy
    
    Conservation verification:
    Total_area = Sigma A_i (must equal expected domain area)

Classes:
    GridDescriptor: Grid metadata and coordinate information
    GridUtils: Grid manipulation and area calculation utilities

Functions:
    compute_cell_areas: Standalone function for area calculation
    spherical_area: Spherical area calculation for lat-lon grids
    cartesian_area: Cartesian area calculation for projected grids

References:
    - Jones, P.W. (1999). First- and second-order conservative remapping schemes
    - Snyder, J.P. (1987). Map Projections: A Working Manual, USGS Professional Paper 1395
    - CF Conventions Version 1.8: Grid mappings and coordinate systems
    - ESMF Reference Manual: Grid and Mesh classes
    - Kritsikis, E. et al. (2017). Conservative interpolation between general spherical meshes
"""

import dataclasses
import hashlib
import math
import numbers
from collections.abc import Mapping
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
from relic.core import device_policy
import torch.nn as nn
from torch import Tensor
from scipy.spatial import cKDTree

from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace
from relic.utils.physical_constants import get_physical_constant


def _resolve_earth_radius(earth_radius: Optional[float] = None) -> float:
    """
    Resolve a finite positive Earth radius from the physical-constant SSOT.
    """
    if earth_radius is not None:
        resolved_radius = float(earth_radius)
    else:
        resolved_radius = float(get_physical_constant("earth_radius"))
    if not math.isfinite(resolved_radius) or resolved_radius <= 0.0:
        raise ValueError("Earth radius must be finite and positive")
    return resolved_radius

@dataclass(init=False)
class GridDescriptor:
    """
    Grid metadata and coordinate information.
    
    Provides comprehensive description of grid structure including
    coordinate system, dimensions, and spatial extent for use in
    regridding operations.
    
    Attributes:
        grid_type : str
            Grid type ("regular", "curvilinear", "unstructured")
        coord_system : str
            Coordinate system ("latlon", "cartesian", "projected")
        dims : Tuple[int, ...]
            Grid dimensions in data order (ny, nx) for 2D grids or (ncells,).
        x_coords : Tensor
            X coordinates (longitude for latlon grids)
        y_coords : Tensor  
            Y coordinates (latitude for latlon grids)
        connectivity : Optional[Tensor]
            Cell-to-vertex connectivity [n_cells, n_vertices] for unstructured grids
        vertex_coords : Optional[Tensor]
            Vertex coordinates [n_vertices, 2] for unstructured grids
        x_bounds : Optional[Tensor]
            X coordinate bounds (shape: [..., 2])
        y_bounds : Optional[Tensor]
            Y coordinate bounds (shape: [..., 2])
        cell_areas : Optional[Tensor]
            Precomputed cell areas [m^2]
        mask : Optional[Tensor]
            Land/ocean mask (True = valid data)
        units : Dict[str, str]
            Coordinate units ("x": "degrees_east", "y": "degrees_north")
        
    Mathematical Expression:
        Grid cell definition:
        Cell(i,j) = {(x,y) : x_bounds[i,0] <= x < x_bounds[i,1], 
                            y_bounds[j,0] <= y < y_bounds[j,1]}
        
        Regular grid coordinates:
        x[i] = x_0 + i * dx, y[j] = y_0 + j * dy
        
        Bounds consistency:
        x_bounds[i,1] = x_bounds[i+1,0] (no gaps)
        y_bounds[j,1] = y_bounds[j+1,0] (no gaps)
    """
    grid_type: str
    coord_system: str
    dims: Tuple[int, ...]
    x_coords: Tensor
    y_coords: Tensor
    connectivity: Optional[Tensor] = None
    vertex_coords: Optional[Tensor] = None
    x_bounds: Optional[Tensor] = None
    y_bounds: Optional[Tensor] = None
    cell_areas: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    units: Optional[Dict[str, str]] = None
    crs: Optional[str] = None

    def __init__(
        self,
        grid_type: Optional[str] = None,
        coord_system: Optional[str] = None,
        dims: Optional[Tuple[int, ...]] = None,
        x_coords: Optional[Tensor] = None,
        y_coords: Optional[Tensor] = None,
        connectivity: Optional[Tensor] = None,
        vertex_coords: Optional[Tensor] = None,
        x_bounds: Optional[Tensor] = None,
        y_bounds: Optional[Tensor] = None,
        cell_areas: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        units: Optional[Dict[str, str]] = None,
        crs: Optional[str] = None,
        shape: Optional[Tuple[int, int]] = None,
        spacing: Optional[Tuple[float, float]] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
        device: Optional[torch.device] = None,
    ):
        """
        Initialize a grid descriptor from explicit coordinates and metadata.

        Supports two initialization styles:
        1. Explicit style with ``grid_type``, ``coord_system``, ``dims``,
           ``x_coords``, and ``y_coords``.
        2. Compact regular-grid style with ``shape`` and ``spacing``.
        """
        if (
            shape is not None
            and spacing is not None
            and (x_coords is None or y_coords is None or dims is None)
        ):
            nx, ny = shape
            dx, dy = spacing
            x0, y0 = origin
            resolved_device = device or (
                device_policy.resolve_device(default="auto")
            )
            resolved_coord_system = coord_system or ("latlon" if str(crs).upper().endswith("4326") else "cartesian")
            resolved_grid_type = grid_type or "regular"
            x_coords = torch.linspace(
                x0 + 0.5 * dx,
                x0 + (nx - 0.5) * dx,
                nx,
                device=resolved_device,
                dtype=torch.float32,
            )
            y_coords = torch.linspace(
                y0 + 0.5 * dy,
                y0 + (ny - 0.5) * dy,
                ny,
                device=resolved_device,
                dtype=torch.float32,
            )
            x_bounds = torch.stack(
                [
                    torch.linspace(x0, x0 + (nx - 1) * dx, nx, device=resolved_device),
                    torch.linspace(x0 + dx, x0 + nx * dx, nx, device=resolved_device),
                ],
                dim=-1,
            )
            y_bounds = torch.stack(
                [
                    torch.linspace(y0, y0 + (ny - 1) * dy, ny, device=resolved_device),
                    torch.linspace(y0 + dy, y0 + ny * dy, ny, device=resolved_device),
                ],
                dim=-1,
            )
            dims = (ny, nx)
            if units is None:
                units = (
                    {"x": "degrees_east", "y": "degrees_north"}
                    if resolved_coord_system == "latlon"
                    else {"x": "m", "y": "m"}
                )
            grid_type = resolved_grid_type
            coord_system = resolved_coord_system

        if grid_type is None or coord_system is None or dims is None or x_coords is None or y_coords is None:
            raise ValueError(
                "GridDescriptor requires either explicit coordinates or (shape, spacing) regular-grid arguments."
            )
        self.grid_type = grid_type
        self.coord_system = coord_system
        self.dims = dims
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.connectivity = connectivity
        self.vertex_coords = vertex_coords
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.cell_areas = cell_areas
        self.mask = mask
        self.units = units
        self.crs = crs
        self.__post_init__()

    @classmethod
    def from_regular_grid(
        cls,
        shape: Tuple[int, int],
        spacing: Tuple[float, float],
        origin: Tuple[float, float] = (0.0, 0.0),
        grid_type: str = "regular",
        coord_system: str = "cartesian",
        units: Optional[Dict[str, str]] = None,
        device: Optional[torch.device] = None,
    ) -> "GridDescriptor":
        """
        Construct a regular grid descriptor from shape and spacing.

        Parameters:
            shape : Tuple[int, int]
                Grid shape as (nx, ny).
            spacing : Tuple[float, float]
                Grid spacing as (dx, dy).
            origin : Tuple[float, float]
                Grid origin (x0, y0).
            grid_type : str
                Grid type (default: "regular").
            coord_system : str
                Coordinate system (default: "cartesian").
            units : Optional[Dict[str, str]]
                Optional units mapping for x/y.
            device : Optional[torch.device]
                Target device for coordinate tensors.

        Returns:
            GridDescriptor
                Regular grid descriptor with coordinates and bounds.
        """
        nx, ny = shape
        dx, dy = spacing
        if dx is None or dy is None:
            raise ValueError("spacing is required when constructing a regular grid descriptor.")
        x0, y0 = origin
        resolved_device = device or (
            device_policy.resolve_device(default="auto")
        )
        x_coords = torch.linspace(
            x0 + 0.5 * dx,
            x0 + (nx - 0.5) * dx,
            nx,
            device=resolved_device,
            dtype=torch.float32,
        )
        y_coords = torch.linspace(
            y0 + 0.5 * dy,
            y0 + (ny - 0.5) * dy,
            ny,
            device=resolved_device,
            dtype=torch.float32,
        )
        x_bounds = torch.stack(
            [
                torch.linspace(x0, x0 + (nx - 1) * dx, nx, device=resolved_device),
                torch.linspace(x0 + dx, x0 + nx * dx, nx, device=resolved_device),
            ],
            dim=-1,
        )
        y_bounds = torch.stack(
            [
                torch.linspace(y0, y0 + (ny - 1) * dy, ny, device=resolved_device),
                torch.linspace(y0 + dy, y0 + ny * dy, ny, device=resolved_device),
            ],
            dim=-1,
        )
        dims = (ny, nx)
        resolved_units = units or (
            {"x": "degrees_east", "y": "degrees_north"}
            if coord_system == "latlon"
            else {"x": "m", "y": "m"}
        )
        return cls(
            grid_type=grid_type,
            coord_system=coord_system,
            dims=dims,
            x_coords=x_coords,
            y_coords=y_coords,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            units=resolved_units,
        )
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get grid shape for consistency.

        Returns:
            Tuple[int, ...]: Grid dimensions (ny, nx) for regular grids or (ncells,) for unstructured
        """
        return self.dims
    
    def __post_init__(self):
        """Validate grid descriptor after initialization."""
        if self.grid_type not in ["regular", "curvilinear", "unstructured"]:
            raise ValueError(f"Invalid grid_type: {self.grid_type}")
        if self.coord_system not in ["latlon", "cartesian", "projected"]:
            raise ValueError(f"Invalid coord_system: {self.coord_system}")

        if self.grid_type == "regular" and len(self.dims) == 2:
            # Normalize alternate (nx, ny) inputs to the internal (ny, nx) convention.
            if self.x_coords.dim() == 1 and self.y_coords.dim() == 1:
                nx = int(self.x_coords.shape[0])
                ny = int(self.y_coords.shape[0])
                if self.dims == (nx, ny):
                    self.dims = (ny, nx)
                elif self.dims != (ny, nx):
                    raise ValueError(
                        "GridDescriptor dims must match coordinate lengths. "
                        f"Got dims={self.dims}, expected {(ny, nx)} or alternate {(nx, ny)}."
                    )

        if self.grid_type == "curvilinear":
            if self.x_coords.dim() != 2 or self.y_coords.dim() != 2:
                raise ValueError("Curvilinear grids require 2D x_coords and y_coords.")
            if self.x_coords.shape != self.y_coords.shape:
                raise ValueError("Curvilinear x_coords and y_coords must have the same shape.")
            ny, nx = self.x_coords.shape
            if len(self.dims) != 2:
                raise ValueError("Curvilinear grid dims must be length 2.")
            if self.dims not in {(ny, nx), (nx, ny)}:
                raise ValueError(
                    "Curvilinear dims must match coordinate shape. "
                    f"Got dims={self.dims}, expected {(ny, nx)} or {(nx, ny)}."
                )
            if self.dims == (nx, ny):
                self.dims = (ny, nx)

        if self.grid_type == "unstructured":
            if len(self.dims) != 1:
                raise ValueError("Unstructured grid dims must be length 1.")
            if self.x_coords.dim() != 1 or self.y_coords.dim() != 1:
                raise ValueError("Unstructured grids require 1D x_coords and y_coords.")
            if self.x_coords.shape[0] != self.y_coords.shape[0]:
                raise ValueError("Unstructured x_coords and y_coords must have the same length.")
            if self.dims[0] != int(self.x_coords.shape[0]):
                raise ValueError(
                    "Unstructured dims must match coordinate length. "
                    f"Got dims={self.dims}, expected ({int(self.x_coords.shape[0])},)."
                )
            if self.connectivity is not None and self.vertex_coords is None:
                raise ValueError("Unstructured connectivity requires vertex_coords.")
            if self.vertex_coords is not None:
                if self.vertex_coords.dim() != 2 or self.vertex_coords.shape[1] != 2:
                    raise ValueError("vertex_coords must have shape [n_vertices, 2].")
                if self.connectivity is not None:
                    if self.connectivity.dim() != 2:
                        raise ValueError("connectivity must have shape [n_cells, n_vertices].")
                    if self.connectivity.shape[0] != self.dims[0]:
                        raise ValueError(
                            "connectivity first dimension must match n_cells. "
                            f"Got {self.connectivity.shape[0]}, expected {self.dims[0]}."
                        )
        
        if self.units is None:
            if self.coord_system == "latlon":
                self.units = {"x": "degrees_east", "y": "degrees_north"}
            else:
                self.units = {"x": "m", "y": "m"}


def _feed_fingerprint_tag(digest: Any, tag: bytes, text: str) -> None:
    """Feed one length-prefixed, type-tagged text token into the digest."""
    encoded = text.encode("utf-8")
    digest.update(tag + len(encoded).to_bytes(8, "little") + encoded)


def _feed_fingerprint(digest: Any, value: Any, path: str) -> None:
    """Feed one declaration value into the digest in a canonical, type-tagged encoding."""
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, Enum):
        _feed_fingerprint_tag(digest, b"E", f"{type(value).__module__}.{type(value).__qualname__}.{value.name}")
    elif isinstance(value, numbers.Integral):
        _feed_fingerprint_tag(digest, b"I", str(int(value)))
    elif isinstance(value, numbers.Real):
        _feed_fingerprint_tag(digest, b"F", float(value).hex())
    elif isinstance(value, str):
        _feed_fingerprint_tag(digest, b"S", value)
    elif isinstance(value, torch.dtype):
        _feed_fingerprint_tag(digest, b"D", str(value))
    elif isinstance(value, torch.device):
        _feed_fingerprint_tag(digest, b"V", str(value))
    elif isinstance(value, Tensor):
        tensor = value.detach()
        if tensor.is_sparse:
            tensor = tensor.to_dense()
        tensor = tensor.to("cpu").contiguous()
        _feed_fingerprint_tag(digest, b"T", f"{tensor.dtype}|{tuple(tensor.shape)}")
        if tensor.dtype == torch.bfloat16:
            payload = tensor.view(torch.int16).numpy().tobytes()
        else:
            payload = tensor.numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "little") + payload)
    elif isinstance(value, GridDescriptor):
        digest.update(b"G")
        for descriptor_field in dataclasses.fields(GridDescriptor):
            _feed_fingerprint_tag(digest, b"K", descriptor_field.name)
            _feed_fingerprint(digest, getattr(value, descriptor_field.name), f"{path}.{descriptor_field.name}")
    elif isinstance(value, (tuple, list)):
        _feed_fingerprint_tag(digest, b"L", str(len(value)))
        for index, item in enumerate(value):
            _feed_fingerprint(digest, item, f"{path}[{index}]")
    elif isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: repr(item[0]))
        _feed_fingerprint_tag(digest, b"M", str(len(items)))
        for key, item in items:
            _feed_fingerprint(digest, key, f"{path}.<key>")
            _feed_fingerprint(digest, item, f"{path}[{key!r}]")
    else:
        raise TypeError(
            f"Cannot fingerprint {path} of type {type(value).__name__}: it has no canonical content "
            "encoding, and a cache key never ignores an input."
        )


def content_fingerprint(*declarations: Any) -> str:
    """SHA-256 fingerprint of declarations in one canonical, type-tagged content encoding.

    A GridDescriptor enters with every dataclass field (grid type, coordinate system, dims,
    coordinates, connectivity, vertices, bounds, cell areas, mask, units, crs); a tensor with its
    dtype, shape and values on the CPU (its device does not change the fingerprint); numbers,
    strings, enums, dtypes, devices, sequences and mappings by value. A value without a canonical
    encoding raises TypeError. Equal fingerprints mean equal declarations; any change of one
    declaration gives another fingerprint.
    """
    digest = hashlib.sha256()
    for index, declaration in enumerate(declarations):
        _feed_fingerprint(digest, declaration, f"declaration[{index}]")
    return digest.hexdigest()


def flatten_grid_points(grid: GridDescriptor) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Flatten grid coordinates into a point list for spatial indexing.

    Returns
    -------
    Tuple[Tensor, Optional[Tensor]]
        points : Tensor of shape [n_points, 2]
        mask_flat : Optional[Tensor] of shape [n_points]

    Notes
    -----
    References:
        - ESMF Reference Manual: grid/mesh point representations
        - Snyder, J.P. (1987). Map Projections (grid coordinate handling)
    Mathematical Expression:
        p_k = (x_i, y_j) for each grid cell center, k = j * nx + i on a regular grid.
    Parameters:
        grid_type, x_coords, y_coords, mask determine point layout.
    Units:
        p in the coordinate unit of the grid.
    Domain/constraints:
        A field on a regular grid is laid out [ny, nx] (dims = (ny, nx)), so the point list of
        1-D axes runs y-major (row-major, x fastest) and point k is element k of the flattened
        field and of the flattened [ny, nx] mask; 2-D coordinates are flattened row-major too.
    Sign convention:
        Not applicable (coordinates keep their sign).
    """
    x_coords = grid.x_coords
    y_coords = grid.y_coords
    if grid.grid_type == "regular":
        if x_coords.dim() == 1 and y_coords.dim() == 1:
            y_mesh, x_mesh = torch.meshgrid(y_coords, x_coords, indexing="ij")
            points = torch.stack([x_mesh.flatten(), y_mesh.flatten()], dim=1)
        elif x_coords.dim() == 2 and y_coords.dim() == 2:
            points = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
        else:
            raise ValueError("Regular grids require 1D or 2D coordinate tensors.")
    elif grid.grid_type in {"curvilinear", "unstructured"}:
        if x_coords.dim() == 2 and y_coords.dim() == 2:
            points = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
        elif x_coords.dim() == 1 and y_coords.dim() == 1:
            points = torch.stack([x_coords, y_coords], dim=1)
        else:
            raise ValueError("Irregular grids require matching 1D or 2D coordinate tensors.")
    else:
        raise ValueError(f"Unsupported grid_type for flattening: {grid.grid_type}")

    mask_flat = grid.mask.flatten() if grid.mask is not None else None
    return points, mask_flat


def target_coverage_report(covered: Tensor) -> Dict[str, int]:
    """Coverage of the target cells of a non-conservative regridder (nearest, bilinear, IDW).

    The counterpart of the covered-support report of the conservative regridder: a target cell
    that no source value reaches (outside the search radius or the source nodes, or masked) is
    uncovered; its value is missing (NaN) and it is counted here, never filled.
    """
    # References: ESMF Reference Manual (unmapped destination points: a regrid leaves them
    #   untouched and reports them instead of inventing a value); docs/source/theory/
    #   scale_coupling.md (covered support, uncovered amounts reported, never redistributed).
    # Mathematical expression: N = |T|, C = |{k in T : covered_k}|, U = N - C.
    # Parameter meanings: T target cells; covered_k whether target k receives a source value.
    # Units: counts [1].
    # Domain/constraints: covered is a boolean vector over the target cells.
    # Sign convention: N, C, U >= 0.
    flags = covered.reshape(-1).to(dtype=torch.bool)
    total = int(flags.numel())
    covered_count = int(flags.sum().item())
    return {
        "target_cells": total,
        "covered_target_cells": covered_count,
        "uncovered_target_cells": total - covered_count,
    }


def typical_node_spacing(
    points: Tensor,
    *,
    mask: Optional[Tensor] = None,
    use_kdtree: bool = False,
    leaf_size: int = 40,
) -> float:
    """
    Median distance from every node to its nearest other node.

    Parameters
    ----------
    points : Tensor
        Node coordinates [n, 2].
    mask : Optional[Tensor]
        Valid nodes [n]; masked nodes do not count.
    use_kdtree, leaf_size
        Neighbour search options of query_knn.

    Returns
    -------
    float
        s = median_i min_{k != i} ||p_i - p_k|| in the coordinate unit (a scale for relative radii).

    Raises
    ------
    ValueError
        With fewer than two valid nodes or when the nodes share their coordinates (s = 0).
    """
    # References: ESMF Reference Manual (neighbour search on unstructured grids).
    # Mathematical expression: s = median_i min_{k != i} ||p_i - p_k||.
    # Parameter meanings: p_i valid node i; s typical spacing.
    # Units: p and s in the coordinate unit of the grid.
    # Domain/constraints: two or more valid nodes and s > 0 (raises otherwise).
    # Sign convention: distances are non-negative.
    nodes = points if mask is None else points[mask.to(device=points.device, dtype=torch.bool)]
    if nodes.shape[0] < 2:
        raise ValueError("A typical node spacing needs at least two valid nodes.")
    distances, _indices = query_knn(nodes, nodes, k=2, use_kdtree=use_kdtree, leaf_size=leaf_size)
    spacing = float(torch.median(distances[:, 1]).item())
    if not spacing > 0.0:
        raise ValueError(
            "The source nodes share their coordinates (typical spacing 0); a relative search radius "
            "is undefined."
        )
    return spacing


def query_knn(
    source_points: Tensor,
    target_points: Tensor,
    k: int,
    *,
    source_mask: Optional[Tensor] = None,
    use_kdtree: bool = False,
    leaf_size: int = 40,
    max_search_radius: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Query k-nearest neighbors for each target point.

    Parameters
    ----------
    source_points : Tensor
        Source point coordinates [n_source, 2].
    target_points : Tensor
        Target point coordinates [n_target, 2].
    k : int
        Number of neighbors.
    source_mask : Optional[Tensor]
        Optional boolean mask for valid source points [n_source].
    use_kdtree : bool
        Whether to use SciPy cKDTree when available.
    leaf_size : int
        KDTree leaf size parameter.
    max_search_radius : Optional[float]
        Maximum search radius in coordinate units. Points outside are invalid.

    Returns
    -------
    Tuple[Tensor, Tensor]
        distances : Tensor [n_target, k]
        indices : Tensor [n_target, k] with -1 for invalid neighbors

    Notes
    -----
    References:
        - Bentley, J.L. (1975). Multidimensional binary search trees.
        - Shepard, D. (1968). Two-dimensional interpolation function.
    Mathematical Expression:
        i* = argmin_i ||x_t - x_i||, generalized to k nearest neighbors.
    Parameters:
        k, leaf_size, max_search_radius control neighbor search.
    """
    if k <= 0:
        raise ValueError("k must be positive for KNN query.")

    source_points = source_points.to(dtype=torch.float64)
    target_points = target_points.to(dtype=torch.float64)
    device = target_points.device

    if source_mask is not None:
        valid_indices = torch.where(source_mask)[0]
        if valid_indices.numel() == 0:
            raise ValueError("source_mask excludes all source points.")
        source_points_valid = source_points[valid_indices]
    else:
        valid_indices = None
        source_points_valid = source_points

    k_eff = min(k, source_points_valid.shape[0])
    radius = None if max_search_radius is None else float(max_search_radius)

    if use_kdtree:
        src_np = source_points_valid.detach().cpu().numpy()
        tgt_np = target_points.detach().cpu().numpy()
        tree = cKDTree(src_np, leafsize=int(leaf_size))
        distances_np, indices_np = tree.query(
            tgt_np,
            k=k_eff,
            distance_upper_bound=radius if radius is not None else float("inf"),
        )
        if k_eff == 1:
            distances_np = distances_np[:, None]
            indices_np = indices_np[:, None]
        distances = torch.tensor(distances_np, device=device, dtype=torch.float64)
        indices = torch.tensor(indices_np, device=device, dtype=torch.long)
        invalid = indices >= source_points_valid.shape[0]
        indices[invalid] = -1
        distances[invalid] = float("inf")
    else:
        distances_full = torch.cdist(target_points, source_points_valid)
        if radius is not None:
            distances_full = torch.where(
                distances_full <= radius,
                distances_full,
                torch.tensor(float("inf"), device=distances_full.device),
            )
        distances, indices = torch.topk(distances_full, k=k_eff, dim=1, largest=False)
        invalid = ~torch.isfinite(distances)
        indices = indices.to(dtype=torch.long)
        indices[invalid] = -1

    if valid_indices is not None:
        mapped = torch.full_like(indices, -1)
        valid_indices = valid_indices.to(indices.device)
        valid_mask = indices >= 0
        mapped[valid_mask] = valid_indices[indices[valid_mask]]
        indices = mapped

    return distances.to(device=device), indices.to(device=device)


class GridUtils(nn.Module):
    """
    Grid utilities for coordinate handling and area calculation.
    
    Provides comprehensive grid manipulation functions including coordinate
    transformations, area calculations, and grid validation. Supports both
    regular and irregular grids with accurate geometric calculations.
    
    Parameters:
        earth_radius : float
            Earth radius in meters for spherical calculations
        device : Optional[torch.device]
            Computation device (CPU/GPU)
            
    Mathematical Expression:
        Coordinate transformation (regular grid):
        (i, j) = ((x - x_0)/dx, (y - y_0)/dy)
        (x, y) = (x_0 + i * dx, y_0 + j * dy)
        
        Distance calculation (spherical):
        d = R * arccos(sin(phi_1)sin(phi_2) + cos(phi_1)cos(phi_2)cos(Deltalambda))
        
        Area calculation delegated to compute_cell_areas function
    
    Methods:
        create_regular_grid: Create regular grid descriptor
        compute_grid_areas: Compute cell areas for any grid
        transform_coordinates: Convert between grid indices and coordinates
        validate_grid: Check grid consistency and validity
        get_grid_spacing: Calculate grid resolution
        interpolate_to_centers: Convert bounds to cell centers
        
    References:
        - Snyder, J.P. (1987). Map Projections: A Working Manual
        - ESMF Reference Manual: Grid class documentation  
        - CF Conventions: Grid mappings and coordinate systems
        - Jones, P.W. (1999). Conservative remapping schemes
    """
    
    def __init__(
        self,
        earth_radius: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize grid utilities with Earth geometry parameters.
        
        Mathematical Expression:
            Earth geometry: R = earth_radius
            Spherical coordinates: (lambda, phi) in radians
            Cartesian coordinates: (x, y) in meters or degrees
        """
        super(GridUtils, self).__init__()
        
        resolved_radius = _resolve_earth_radius(earth_radius)
        if resolved_radius <= 0:
            raise ValueError("Earth radius must be positive")
            
        self.earth_radius = resolved_radius
        self.device = device or (
            device_policy.resolve_device(default="auto")
        )

    def _get_conversion_tensor(self, key: str) -> Tensor:
        """
        Retrieve angular conversion factors from the physical-constant SSOT.

        Notes
        -----
        The physical-constant utility is the only runtime owner for these factors.
        """
        value = get_physical_constant(
            f"relic.physical.constants.{key}",
            as_tensor=True,
            device=self.device,
        )
        return torch.as_tensor(value, device=self.device, dtype=torch.float64)

    @property
    def deg_to_rad(self) -> Tensor:
        """Degrees to radians conversion factor (dimensionless)."""
        return self._get_conversion_tensor("conversions.deg_to_rad")

    @property
    def rad_to_deg(self) -> Tensor:
        """Radians to degrees conversion factor (dimensionless)."""
        return self._get_conversion_tensor("conversions.rad_to_deg")
        
    def create_regular_grid(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        nx: int,
        ny: int,
        coord_system: str = "latlon",
        include_bounds: bool = True,
    ) -> GridDescriptor:
        """
        Create regular grid descriptor with uniform spacing.
        
        Generates a regular grid with uniform spacing in x and y directions.
        Automatically calculates coordinate bounds and cell areas if requested.
        
        Parameters:
            x_min, x_max : float
                X coordinate range [degrees or meters]
            y_min, y_max : float
                Y coordinate range [degrees or meters]  
            nx, ny : int
                Number of grid cells in x, y directions
            coord_system : str
                Coordinate system ("latlon", "cartesian", "projected")
            include_bounds : bool
                Whether to compute coordinate bounds
                
        Returns:
            GridDescriptor
                Complete grid description with coordinates and metadata
                
        Raises:
            ValueError: If grid parameters are invalid
            
        Mathematical Expression:
            Grid spacing:
            dx = (x_max - x_min) / nx
            dy = (y_max - y_min) / ny
            
            Cell centers:
            x[i] = x_min + (i + 0.5) * dx, i = 0, ..., nx-1
            y[j] = y_min + (j + 0.5) * dy, j = 0, ..., ny-1
            
            Cell bounds:
            x_bounds[i] = [x_min + i * dx, x_min + (i+1) * dx]
            y_bounds[j] = [y_min + j * dy, y_min + (j+1) * dy]
        """
        if nx <= 0 or ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid coordinate ranges")
            
        # Calculate grid spacing
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        
        # Create coordinate arrays (cell centers)
        x_coords = torch.linspace(
            x_min + 0.5 * dx, x_max - 0.5 * dx, nx,
            device=self.device, dtype=torch.float32
        )
        y_coords = torch.linspace(
            y_min + 0.5 * dy, y_max - 0.5 * dy, ny,
            device=self.device, dtype=torch.float32
        )
        
        # Create bounds if requested
        x_bounds = None
        y_bounds = None
        if include_bounds:
            x_bounds = torch.stack([
                torch.linspace(x_min, x_max - dx, nx, device=self.device),
                torch.linspace(x_min + dx, x_max, nx, device=self.device)
            ], dim=-1)
            
            y_bounds = torch.stack([
                torch.linspace(y_min, y_max - dy, ny, device=self.device),
                torch.linspace(y_min + dy, y_max, ny, device=self.device)
            ], dim=-1)
        
        # Set units based on coordinate system
        if coord_system == "latlon":
            units = {"x": "degrees_east", "y": "degrees_north"}
        else:
            units = {"x": "m", "y": "m"}
            
        return GridDescriptor(
            grid_type="regular",
            coord_system=coord_system,
            dims=(ny, nx),
            x_coords=x_coords,
            y_coords=y_coords,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            units=units
        )
    
    def compute_grid_areas(self, grid: GridDescriptor) -> Tensor:
        """
        Compute cell areas for any grid type.
        
        Wrapper around compute_cell_areas function with grid descriptor input.
        Automatically handles different coordinate systems and grid types.
        
        Parameters:
            grid : GridDescriptor
                Grid description with coordinates and bounds
                
        Returns:
            Tensor
                Cell areas [m^2], shape matches grid dimensions
                
        Mathematical Expression:
            Delegates to compute_cell_areas function
            See compute_cell_areas docstring for detailed formulas

        Raises:
            ValueError: If the grid carries no cell bounds. Bounds come from the data file or an
                explicit grid descriptor; they are never inferred from the coordinate spacing.
        """
        if grid.x_bounds is None or grid.y_bounds is None:
            raise ValueError(
                "Cell areas need the cell bounds of the grid (x_bounds/y_bounds from the data file or "
                "an explicit grid descriptor); bounds are not inferred from the coordinate spacing."
            )
        return compute_cell_areas(
            grid.x_bounds, grid.y_bounds,
            grid.coord_system, self.earth_radius
        )
    
    def transform_coordinates(
        self,
        grid: GridDescriptor,
        indices: Optional[Tensor] = None,
        coordinates: Optional[Tensor] = None,
        to_indices: bool = True,
    ) -> Tensor:
        """
        Transform between grid indices and coordinates.
        
        Provides bidirectional transformation between grid indices (i, j)
        and physical coordinates (x, y) for regular grids.
        
        Parameters:
            grid : GridDescriptor
                Grid description
            indices : Optional[Tensor]
                Grid indices (i, j) to convert to coordinates
            coordinates : Optional[Tensor]  
                Coordinates (x, y) to convert to indices
            to_indices : bool
                Direction of transformation
                
        Returns:
            Tensor
                Transformed coordinates or indices
                
        Raises:
            ValueError: If grid is not regular or inputs are invalid
            
        Mathematical Expression:
            Index to coordinate:
            x = x_0 + i * dx, y = y_0 + j * dy
            
            Coordinate to index:
            i = (x - x_0) / dx, j = (y - y_0) / dy
        """
        if grid.grid_type != "regular":
            raise ValueError("Coordinate transformation only supported for regular grids")
            
        if to_indices:
            if coordinates is None:
                raise ValueError("coordinates required for index transformation")
            # Convert coordinates to indices
            x, y = coordinates[..., 0], coordinates[..., 1]
            x_coords, y_coords = grid.x_coords, grid.y_coords
            
            # Calculate grid spacing
            dx = x_coords[1] - x_coords[0]
            dy = y_coords[1] - y_coords[0]
            x0 = x_coords[0] - 0.5 * dx
            y0 = y_coords[0] - 0.5 * dy
            
            i = (x - x0) / dx
            j = (y - y0) / dy
            
            return torch.stack([i, j], dim=-1)
        else:
            if indices is None:
                raise ValueError("indices required for coordinate transformation")
            # Convert indices to coordinates
            i, j = indices[..., 0], indices[..., 1]
            x_coords, y_coords = grid.x_coords, grid.y_coords
            
            # Calculate grid spacing
            dx = x_coords[1] - x_coords[0]
            dy = y_coords[1] - y_coords[0]
            x0 = x_coords[0] - 0.5 * dx
            y0 = y_coords[0] - 0.5 * dy
            
            x = x0 + i * dx
            y = y0 + j * dy
            
            return torch.stack([x, y], dim=-1)
    
    def validate_grid(self, grid: GridDescriptor) -> Dict[str, bool]:
        """
        Validate grid consistency and completeness.
        
        Performs comprehensive validation of grid descriptor including
        coordinate consistency, bounds validation, and area conservation.
        
        Parameters:
            grid : GridDescriptor
                Grid to validate
                
        Returns:
            Dict[str, bool]
                Validation results for different aspects
                
        Mathematical Expression:
            Bounds consistency:
            x_bounds[i,1] = x_bounds[i+1,0] (within tolerance)
            
            Monotonicity:
            x[i+1] > x[i], y[j+1] > y[j]
            
            Area positivity:
            A[i,j] > 0 for all cells
        """
        results = {}
        
        # Check coordinate monotonicity
        if grid.grid_type == "regular":
            x_diff = torch.diff(grid.x_coords)
            y_diff = torch.diff(grid.y_coords)
            results["x_monotonic"] = torch.all(x_diff > 0).item()
            results["y_monotonic"] = torch.all(y_diff > 0).item()
        else:
            results["x_monotonic"] = True  # Assume valid for irregular grids
            results["y_monotonic"] = True
        
        # Check bounds consistency
        if grid.x_bounds is not None and grid.y_bounds is not None:
            # Check bounds are properly ordered
            x_bounds_valid = torch.all(grid.x_bounds[..., 1] > grid.x_bounds[..., 0])
            y_bounds_valid = torch.all(grid.y_bounds[..., 1] > grid.y_bounds[..., 0])
            results["bounds_ordered"] = x_bounds_valid.item() and y_bounds_valid.item()
            
            # For regular grids, check bounds continuity
            if grid.grid_type == "regular":
                param_manager = get_global_manager()
                if param_manager is None:
                    raise RuntimeError("ParameterManager is required for grid validation but is not available.")
                relative_tolerance = float(
                    param_manager.get(
                        "relative_tolerance",
                        namespace=resolve_namespace(ParameterNamespace.COUPLING_DEFAULTS),
                        units="dimensionless",
                        expected_type=float,
                        required=True,
                    )
                )
                x_continuous = torch.allclose(
                    grid.x_bounds[:-1, 1], grid.x_bounds[1:, 0], rtol=relative_tolerance
                )
                y_continuous = torch.allclose(
                    grid.y_bounds[:-1, 1], grid.y_bounds[1:, 0], rtol=relative_tolerance
                )
                results["bounds_continuous"] = bool(x_continuous) and bool(y_continuous)
            else:
                results["bounds_continuous"] = True
        else:
            results["bounds_ordered"] = True
            results["bounds_continuous"] = True
        
        # Check areas if available
        if grid.cell_areas is not None:
            results["areas_positive"] = torch.all(grid.cell_areas > 0).item()
        else:
            results["areas_positive"] = True
            
        return results
    
    def get_grid_spacing(self, grid: GridDescriptor) -> Tuple[float, float]:
        """
        Calculate grid resolution for regular grids.
        
        Parameters:
            grid : GridDescriptor
                Grid description
                
        Returns:
            Tuple[float, float]
                Grid spacing (dx, dy) in coordinate units
                
        Mathematical Expression:
            dx = (x_max - x_min) / nx
            dy = (y_max - y_min) / ny
        """
        if grid.grid_type != "regular":
            raise ValueError("Grid spacing only defined for regular grids")
            
        dx = grid.x_coords[1] - grid.x_coords[0]
        dy = grid.y_coords[1] - grid.y_coords[0]
        
        return dx.item(), dy.item()


def spherical_area(
    lon_bounds: Tensor,
    lat_bounds: Tensor,
    earth_radius: Optional[float] = None,
) -> Tensor:
    """
    Calculate spherical area for latitude-longitude grid cells.
    
    Computes exact spherical area using analytical formula for spherical
    quadrilaterals. Accounts for Earth's curvature and provides accurate
    areas for conservation calculations.
    
    Parameters:
        lon_bounds : Tensor
            Longitude bounds [degrees], shape [..., 2]
        lat_bounds : Tensor
            Latitude bounds [degrees], shape [..., 2]
        earth_radius : float
            Earth radius [m]
            
    Returns:
        Tensor
            Cell areas [m^2]
            
    Mathematical Expression:
        Spherical quadrilateral area:
        A = R^2 * Deltalambda * (sin(phi_2) - sin(phi_1))
        
        where:
        R = Earth radius [m]
        Deltalambda = lon_bounds[..., 1] - lon_bounds[..., 0] [radians]
        phi_1 = lat_bounds[..., 0] [radians]  
        phi_2 = lat_bounds[..., 1] [radians]
        
        Conversion: degrees to radians
        phi = lat_degrees * pi/180
        lambda = lon_degrees * pi/180
        
    References:
        - Bevis, M. & Cambenga, R. (1999). Calculating the area of a spherical polygon
        - Snyder, J.P. (1987). Map Projections: A Working Manual, equations 3-2, 3-3
    """
    # Handle both 1D and 2D bounds
    # If bounds are 1D (shape [n, 2]), need to expand to 2D grid
    if lon_bounds.dim() == 2 and lat_bounds.dim() == 2:
        # Bounds are 1D - need to create 2D grid
        # lon_bounds: [nx, 2], lat_bounds: [ny, 2]
        ny = len(lat_bounds)
        nx = len(lon_bounds)
        
        # Expand to 2D: lon varies along x (dim 1), lat varies along y (dim 0)
        lon_bounds_2d = lon_bounds.unsqueeze(0).expand(ny, -1, -1)  # [ny, nx, 2]
        lat_bounds_2d = lat_bounds.unsqueeze(1).expand(-1, nx, -1)  # [ny, nx, 2]
        
        lon_bounds = lon_bounds_2d
        lat_bounds = lat_bounds_2d
    
    lon_bounds = lon_bounds.to(dtype=torch.float64)
    lat_bounds = lat_bounds.to(dtype=torch.float64)
    earth_radius = _resolve_earth_radius(earth_radius)

    # Convert to radians
    lon_rad = torch.deg2rad(lon_bounds)
    lat_rad = torch.deg2rad(lat_bounds)
    
    # Calculate longitude difference
    dlon = lon_rad[..., 1] - lon_rad[..., 0]
    
    # Calculate sine difference for latitude
    sin_lat1 = torch.sin(lat_rad[..., 0])
    sin_lat2 = torch.sin(lat_rad[..., 1])
    dsin_lat = sin_lat2 - sin_lat1
    
    # Spherical area formula
    area = earth_radius**2 * dlon * dsin_lat
    
    # Handle negative areas (should not occur with proper bounds)
    area = torch.abs(area)
    
    return area


def cartesian_area(x_bounds: Tensor, y_bounds: Tensor) -> Tensor:
    """
    Calculate Cartesian area for projected grid cells.
    
    Computes rectangular cell areas for projected coordinate systems
    where grid cells are approximately rectangular.
    
    Parameters:
        x_bounds : Tensor
            X coordinate bounds [m], shape [..., 2]
        y_bounds : Tensor
            Y coordinate bounds [m], shape [..., 2]
            
    Returns:
        Tensor
            Cell areas [m^2]
            
    Mathematical Expression:
        Rectangular area:
        A = Deltax * Deltay
        
        where:
        Deltax = x_bounds[..., 1] - x_bounds[..., 0]
        Deltay = y_bounds[..., 1] - y_bounds[..., 0]
    """
    dx = x_bounds[..., 1] - x_bounds[..., 0]
    dy = y_bounds[..., 1] - y_bounds[..., 0]
    
    area = dx * dy
    
    return torch.abs(area)


def compute_cell_areas(
    x_bounds: Tensor,
    y_bounds: Tensor,
    coord_system: str = "latlon",
    earth_radius: Optional[float] = None,
) -> Tensor:
    """
    Compute grid cell areas with appropriate geometric calculation.
    
    Automatically selects appropriate area calculation method based on
    coordinate system. Provides accurate areas for conservation calculations
    in regridding operations.
    
    Parameters:
        x_bounds : Tensor
            X coordinate bounds, shape [..., 2]
        y_bounds : Tensor
            Y coordinate bounds, shape [..., 2]
        coord_system : str
            Coordinate system ("latlon", "cartesian", "projected")
        earth_radius : float
            Earth radius [m] for spherical calculations
            
    Returns:
        Tensor
            Cell areas [m^2]
            
    Raises:
        ValueError: If coordinate system is not supported
        
    Mathematical Expression:
        For latlon: Uses spherical_area function
        For cartesian/projected: Uses cartesian_area function
        
        Conservation verification:
        Total_area = Sigma?? A[i,j]
        
        References:
        - Jones, P.W. (1999). First- and second-order conservative remapping schemes
        - ESMF Reference Manual: Grid area calculation methods
    """
    # Expand 1D regular bounds to 2D so areas align with (ny, nx) grids.
    # This keeps downstream weight matrices consistent (especially when GridManager
    # precomputes areas from 1D bounds).
    if x_bounds.dim() == 2 and y_bounds.dim() == 2:
        nx = x_bounds.shape[0]
        ny = y_bounds.shape[0]
        x_bounds = x_bounds.unsqueeze(0).expand(ny, nx, -1)
        y_bounds = y_bounds.unsqueeze(1).expand(ny, nx, -1)

    resolved_radius = _resolve_earth_radius(earth_radius)

    if coord_system == "latlon":
        return spherical_area(x_bounds, y_bounds, resolved_radius)
    elif coord_system in ["cartesian", "projected"]:
        return cartesian_area(x_bounds, y_bounds)
    else:
        raise ValueError(f"Unsupported coordinate system: {coord_system}")
