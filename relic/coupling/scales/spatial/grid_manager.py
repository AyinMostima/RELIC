"""
Grid Management Module

Provides unified grid management functionality for spatial coupling operations,
including grid construction, validation, device placement, and area calculations.

This module acts as a facade over GridUtils and GridDescriptor, providing
high-level operations while maintaining separation of concerns.

Mathematical Expression:
    Grid descriptor creation and management:
    G = {coords, bounds, areas, mask, metadata}
    
    Area calculation delegation:
    A = compute_cell_areas(x_bounds, y_bounds, coord_system, R_earth)
    
    Device placement operations:
    G_device = move_tensors(G, device)

References:
    - CF Conventions Version 1.8: Grid mappings and coordinate systems
    - ESMF Reference Manual: Grid class documentation
    - PyTorch Device Management: Tensor placement and memory management
"""

import logging
from typing import Dict, List, Optional, Any

import torch
from relic.core import device_policy
from torch import Tensor

from .grid_utils import GridDescriptor, GridUtils
from ....core.managers.base import BaseManager
from ....core.managers.metadata import (
    ManagerMetadata, ManagerCapability, ManagerResource, CapabilityType,
    ResourceType
)
from ....parameters.namespaces import ParameterNamespace
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)

DEFAULTS_NS = ParameterNamespace.COUPLING_DEFAULTS.value
SCALES_SPATIAL_NS = ParameterNamespace.COUPLING_SCALES_SPATIAL.value


class GridManager(BaseManager):
    """
    Unified grid management for spatial coupling operations.
    
    Provides high-level operations for grid creation, validation, device management,
    and area calculations. Acts as a facade over lower-level grid utilities while
    ensuring consistency and performance.
    
    Parameters:
        device : Optional[torch.device]
            Default computation device
        coord_names : Optional[Dict[str, str]]
            Standard coordinate name mappings
        earth_radius : float
            Earth radius for spherical area calculations [m]
            
    Mathematical Expression:
        Grid construction workflow:
        1. Coordinate generation: x[i], y[j] from bounds/spacing
        2. Bounds: declared by the caller (data file or explicit descriptor), never inferred
        3. Area computation: A[i,j] = f(x_bounds, y_bounds, coord_system)
        4. Device placement: tensors -> target_device
        
        Grid validation checks:
        - Monotonicity: x[i+1] > x[i], y[j+1] > y[j]
        - Bounds consistency: bounds[i,1] = bounds[i+1,0]
        - Area positivity: A[i,j] > 0
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        coord_names: Optional[Dict[str, str]] = None,
        earth_radius: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        parameter_manager: Optional[Any] = None,
    ) -> None:
        """
        Initialize grid manager with default settings.
        
        Parameters:
            device : Optional[torch.device]
                Default device for grid tensors
            coord_names : Optional[Dict[str, str]]  
                Coordinate name mappings (e.g., {'x': 'lon', 'y': 'lat'})
            earth_radius : Optional[float]
                Earth radius in meters for spherical calculations
            config : Optional[Dict[str, Any]]
                Additional configuration options
            parameter_manager : Optional[Any]
                ParameterManager instance for defaults and validation
        """
        # Store initialization parameters
        self._init_device = device
        self._init_coord_names = coord_names
        self._init_earth_radius = earth_radius
        self._init_config = config or {}
        self._init_parameter_manager = parameter_manager
        
        # Initialize base class with metadata
        metadata = self._create_metadata()
        super().__init__(metadata=metadata)
        
        # The actual initialization will be done in _do_initialize()
    
    def _create_metadata(self) -> ManagerMetadata:
        """Create metadata for GridManager."""
        capabilities = [
            ManagerCapability(
                type=CapabilityType.SPATIAL_OPERATIONS,
                level="advanced",
                description="Grid construction and management"
            ),
            ManagerCapability(
                type=CapabilityType.DEVICE_MANAGEMENT,
                level="standard",
                description="Device-aware tensor operations"
            ),
            ManagerCapability(
                type=CapabilityType.CF_CONVENTIONS,
                level="standard",
                description="CF-1.8 compliant grid metadata"
            ),
        ]
        
        resources = [
            ManagerResource(
                type=ResourceType.SYSTEM_MEMORY,
                quantity=None,
                unit="bytes"
            ),
            ManagerResource(
                type=ResourceType.CACHE_STORAGE,
                quantity=None,
                unit="bytes"
            ),
        ]
        
        dependencies = []  # GridManager has no dependencies on other managers
        
        return ManagerMetadata(
            name="GridManager",
            version="2.0.0",
            description="Unified grid management for spatial coupling operations",
            capabilities=capabilities,
            resources=resources,
            dependencies=dependencies,
            tags={"thread_safe", "gpu_enabled"},
            custom_attributes={
                "thread_safe": True,
                "supports_gpu": True,
                "supports_distributed": False
            }
        )
    
    def _do_initialize(self) -> None:
        """Perform GridManager-specific initialization."""
        # Set core attributes
        self.parameter_manager = (
            self._init_parameter_manager
            if self._init_parameter_manager is not None
            else get_global_manager()
        )

        if self._init_device is None:
            if self.parameter_manager is None:
                raise RuntimeError("GridManager requires ParameterManager or explicit device.")
            device_value = self.parameter_manager.get(
                "device",
                namespace=DEFAULTS_NS,
                units="dimensionless",
                expected_type=str,
                required=True,
            )
            self.device = torch.device(str(device_value))
        else:
            self.device = (
                torch.device(str(self._init_device))
                if isinstance(self._init_device, str)
                else self._init_device
            )

        if self._init_earth_radius is None:
            self.earth_radius = float(get_physical_constant("earth_radius"))
        else:
            self.earth_radius = float(self._init_earth_radius)

        if self._init_coord_names is None:
            if self.parameter_manager is None:
                raise RuntimeError("GridManager requires ParameterManager or explicit coord_names.")
            coord_names_value = self.parameter_manager.get(
                "coord_names",
                namespace=SCALES_SPATIAL_NS,
                units="dimensionless",
                expected_type=dict,
                required=True,
            )
            if not isinstance(coord_names_value, dict):
                raise TypeError("coord_names must be a mapping of coordinate aliases to canonical names.")
            self._coord_names = dict(coord_names_value)
        else:
            self._coord_names = dict(self._init_coord_names)
        
        # Initialize grid utilities
        self._grid_utils = GridUtils(
            earth_radius=self.earth_radius,
            device=self.device
        )
        
        # Cache for computed areas
        self._area_cache: Dict[str, Tensor] = {}
        
        logger.info(f"GridManager initialized: device={self.device}, earth_radius={self.earth_radius}")
    
    def _do_configure(self, config: Dict[str, Any]) -> None:
        """Configure GridManager with runtime settings."""
        settings = dict(config or {})
        # Update device if provided
        if 'device' in settings:
            device_value = settings['device']
            if device_value is None:
                raise ValueError("device cannot be None when configuring GridManager.")
            if isinstance(device_value, str):
                self.device = torch.device(device_value)
            else:
                self.device = device_value
            # Recreate grid utilities with new device
            self._grid_utils = GridUtils(
                earth_radius=self.earth_radius,
                device=self.device
            )
            
        # Update earth radius if provided
        if 'earth_radius' in settings:
            earth_radius_value = settings['earth_radius']
            if earth_radius_value is None:
                raise ValueError("earth_radius cannot be None when configuring GridManager.")
            self.earth_radius = float(earth_radius_value)
            # Recreate grid utilities with new radius
            self._grid_utils = GridUtils(
                earth_radius=self.earth_radius,
                device=self.device
            )
            
        # Update coordinate names if provided
        if 'coord_names' in settings:
            coord_names_value = settings['coord_names']
            if coord_names_value is None:
                raise ValueError("coord_names cannot be None when configuring GridManager.")
            if not isinstance(coord_names_value, dict):
                raise TypeError("coord_names must be a mapping when configuring GridManager.")
            self._coord_names.update(coord_names_value)
            
        # Cache settings
        if 'max_cache_size' in settings:
            max_cache_size = settings['max_cache_size']
            if max_cache_size is None:
                raise ValueError("max_cache_size cannot be None when configuring GridManager.")
            self._max_cache_size = int(max_cache_size)
            # Trim cache if needed
            if len(self._area_cache) > self._max_cache_size:
                # Keep most recent entries
                keys_to_remove = list(self._area_cache.keys())[:-self._max_cache_size]
                for key in keys_to_remove:
                    del self._area_cache[key]
                    
        logger.debug(f"GridManager configured with: {config}")
    
    def _do_validate(self) -> List[str]:
        """Validate GridManager configuration and state.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check device availability
        if self.device.type == 'cuda' and not device_policy.is_cuda_available():
            errors.append(f"CUDA device {self.device} requested but not available")
            
        # Validate earth radius
        if self.earth_radius <= 0:
            errors.append(f"Earth radius {self.earth_radius} must be positive")
            
        # Check grid utilities
        if self._grid_utils is None:
            errors.append("GridUtils not initialized")
            
        # Validate coordinate names
        if not self._coord_names:
            errors.append("Coordinate name mappings not initialized")
            
        if not errors:
            logger.debug("GridManager validation completed successfully")
        else:
            logger.warning(f"GridManager validation found {len(errors)} issues")
            
        return errors
    
    def _do_cleanup(self) -> None:
        """Clean up GridManager resources."""
        # Clear area cache if it exists
        if hasattr(self, '_area_cache'):
            self.clear_cache()
        
        # Clear grid utilities if it has cleanup
        if hasattr(self, '_grid_utils') and hasattr(self._grid_utils, 'cleanup'):
            self._grid_utils.cleanup()
            
        logger.info("GridManager cleanup completed")
    
    def from_tensor(
        self,
        lat: Tensor,
        lon: Tensor,
        *,
        lat_bounds: Tensor,
        lon_bounds: Tensor,
        mask: Optional[Tensor] = None,
        device: Optional[torch.device] = None
    ) -> GridDescriptor:
        """
        Create grid descriptor from coordinate and cell-bound tensors.

        The cell bounds are declared by the caller (for example the CF lat_bnds/lon_bnds of the
        data file); they are never inferred from the coordinate spacing.

        Parameters:
            lat : Tensor
                Latitude coordinates [degrees_north], shape [nlat]
            lon : Tensor
                Longitude coordinates [degrees_east], shape [nlon]
            lat_bounds : Tensor
                Latitude cell bounds [degrees_north], shape [nlat, 2]
            lon_bounds : Tensor
                Longitude cell bounds [degrees_east], shape [nlon, 2]
            mask : Optional[Tensor]
                Land/ocean mask (True = valid data)
            device : Optional[torch.device]
                Target device for tensors

        Returns:
            GridDescriptor: Complete grid description with areas

        Raises:
            ValueError: If coordinate or bound tensors are mismatched

        Mathematical Expression:
            Grid dimensions: dims = (len(lon), len(lat))
            Cell (i, j) = [lon_bounds[i, 0], lon_bounds[i, 1]] x [lat_bounds[j, 0], lat_bounds[j, 1]]
        """
        target_device = device or self.device

        # Move inputs to target device while preserving dtype
        lat = lat.to(target_device)
        lon = lon.to(target_device)
        lat_bounds = lat_bounds.to(target_device)
        lon_bounds = lon_bounds.to(target_device)
        if mask is not None:
            mask = mask.to(target_device)

        # Validate coordinate tensors
        if lat.dim() != 1 or lon.dim() != 1:
            raise ValueError("Coordinate tensors must be 1-dimensional")
        if len(lat) == 0 or len(lon) == 0:
            raise ValueError("Coordinate tensors cannot be empty")
        if tuple(lat_bounds.shape) != (len(lat), 2) or tuple(lon_bounds.shape) != (len(lon), 2):
            raise ValueError(
                "lat_bounds/lon_bounds must have shapes (nlat, 2) and (nlon, 2); got "
                f"{tuple(lat_bounds.shape)} and {tuple(lon_bounds.shape)}"
            )

        # Create grid descriptor
        grid = GridDescriptor(
            grid_type="regular",
            coord_system="latlon",
            dims=(len(lon), len(lat)),
            x_coords=lon,
            y_coords=lat,
            connectivity=None,
            vertex_coords=None,
            x_bounds=lon_bounds,
            y_bounds=lat_bounds,
            mask=mask,
            units={"x": "degrees_east", "y": "degrees_north"}
        )

        # Compute and cache areas
        grid = self.ensure_areas(grid)

        logger.debug(f"Created grid from tensors: {grid.dims}")
        return grid

    def build_from_regular(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        nlon: int,
        nlat: int,
        device: Optional[torch.device] = None
    ) -> GridDescriptor:
        """
        Build regular grid from domain bounds and resolution.
        
        Creates a uniform regular grid with specified extent and resolution,
        automatically computing coordinate bounds and cell areas.
        
        Parameters:
            lon_min, lon_max : float
                Longitude range [degrees_east]  
            lat_min, lat_max : float
                Latitude range [degrees_north]
            nlon, nlat : int
                Number of grid cells in longitude/latitude
            device : Optional[torch.device]
                Target device for grid tensors
                
        Returns:
            GridDescriptor: Regular grid with bounds and areas
            
        Mathematical Expression:
            Grid spacing:
            dlon = (lon_max - lon_min) / nlon
            dlat = (lat_max - lat_min) / nlat
            
            Cell centers:
            lon[i] = lon_min + (i + 0.5) * dlon, i = 0, ..., nlon-1
            lat[j] = lat_min + (j + 0.5) * dlat, j = 0, ..., nlat-1
        """
        target_device = device or self.device
        
        # Delegate to GridUtils for consistent creation
        grid = self._grid_utils.create_regular_grid(
            x_min=lon_min,
            x_max=lon_max,
            y_min=lat_min,
            y_max=lat_max,
            nx=nlon,
            ny=nlat,
            coord_system="latlon",
            include_bounds=True
        )
        
        # Ensure areas are computed
        grid = self.ensure_areas(grid)
        
        # Move to target device
        grid = self.move_to_device(grid, target_device)
        
        logger.debug(f"Built regular grid: {nlon}x{nlat} on {target_device}")
        return grid
    
    def ensure_areas(self, grid: GridDescriptor) -> GridDescriptor:
        """
        Ensure grid has cell areas computed and cached.
        
        Computes cell areas if not present, using appropriate method
        for the coordinate system. Caches results for performance.
        
        Parameters:
            grid : GridDescriptor
                Grid descriptor to update
                
        Returns:
            GridDescriptor: Grid with areas attached
            
        Mathematical Expression:
            Area computation delegation:
            if coord_system == "latlon":
                A = spherical_area(lon_bounds, lat_bounds, R_earth)
            else:
                A = cartesian_area(x_bounds, y_bounds)
        """
        if grid.cell_areas is not None:
            return grid  # Already has areas
        
        # Create cache key
        cache_key = self._create_area_cache_key(grid)
        
        # Check cache
        if cache_key in self._area_cache:
            grid.cell_areas = self._area_cache[cache_key]
            logger.debug(f"Using cached areas for grid {grid.dims}")
            return grid
        
        # Compute areas using GridUtils
        try:
            areas = self._grid_utils.compute_grid_areas(grid)
            grid.cell_areas = areas
            
            # Cache for future use
            self._area_cache[cache_key] = areas
            
            logger.debug(f"Computed areas for grid {grid.dims}: total={areas.sum().item():.2e} m^2")
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Failed to compute grid areas: {e}")
            raise
        
        return grid
    
    def move_to_device(
        self,
        grid: GridDescriptor,
        device: torch.device
    ) -> GridDescriptor:
        """
        Move grid tensors to target device safely.
        
        Transfers all tensor attributes of a grid descriptor to the
        specified device, preserving data integrity and relationships.
        
        Parameters:
            grid : GridDescriptor
                Source grid descriptor
            device : torch.device
                Target device
                
        Returns:
            GridDescriptor: Grid with tensors on target device
            
        Mathematical Expression:
            Device transfer operations:
            tensor_device = tensor.to(device)
            Preserves: shape, dtype, requires_grad, data values
        """
        if grid.x_coords.device == device:
            return grid  # Already on target device
        
        # Move coordinate tensors
        x_coords = grid.x_coords.to(device)
        y_coords = grid.y_coords.to(device)
        
        # Move bounds if present
        x_bounds = grid.x_bounds.to(device) if grid.x_bounds is not None else None
        y_bounds = grid.y_bounds.to(device) if grid.y_bounds is not None else None
        
        # Move areas if present
        cell_areas = grid.cell_areas.to(device) if grid.cell_areas is not None else None
        
        # Move mask if present
        mask = grid.mask.to(device) if grid.mask is not None else None
        
        # Create new grid descriptor
        moved_grid = GridDescriptor(
            grid_type=grid.grid_type,
            coord_system=grid.coord_system,
            dims=grid.dims,
            x_coords=x_coords,
            y_coords=y_coords,
            connectivity=grid.connectivity.to(device) if grid.connectivity is not None else None,
            vertex_coords=grid.vertex_coords.to(device) if grid.vertex_coords is not None else None,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            cell_areas=cell_areas,
            mask=mask,
            units=grid.units
        )
        
        logger.debug(f"Moved grid to device {device}: {grid.dims}")
        return moved_grid
    
    def coord_names(self) -> Dict[str, str]:
        """
        Get standard coordinate name mappings.
        
        Returns standardized coordinate and dimension names for
        consistency across CF conventions and regridding operations.
        
        Returns:
            Dict[str, str]: Coordinate name mappings
        """
        return self._coord_names.copy()
    
    def validate_grid(self, grid: GridDescriptor) -> Dict[str, bool]:
        """
        Validate grid consistency and completeness.
        
        Performs comprehensive validation using GridUtils validator,
        adding GridManager-specific checks for cached areas and device placement.
        
        Parameters:
            grid : GridDescriptor
                Grid to validate
                
        Returns:
            Dict[str, bool]: Validation results
        """
        # Use GridUtils validation as base
        results = self._grid_utils.validate_grid(grid)
        
        # Add GridManager-specific validations
        results["has_areas"] = grid.cell_areas is not None
        results["areas_cached"] = self._create_area_cache_key(grid) in self._area_cache
        results["device_consistent"] = self._check_device_consistency(grid)
        
        return results
    
    def clear_cache(self):
        """Clear cached areas to free memory."""
        self._area_cache.clear()
        logger.info("Cleared grid area cache")
    
    def _create_area_cache_key(self, grid: GridDescriptor) -> str:
        """Cache key of the cell areas: coordinate system, shape, device and the cell bounds."""
        import hashlib

        digest = hashlib.sha1()
        for bounds in (grid.x_bounds, grid.y_bounds):
            if bounds is None:
                digest.update(b"no-bounds")
            else:
                digest.update(bounds.detach().to("cpu", torch.float64).numpy().tobytes())
        return f"{grid.coord_system}_{grid.dims}_{grid.x_coords.device}_{digest.hexdigest()}"
    
    def _check_device_consistency(self, grid: GridDescriptor) -> bool:
        """Check if all grid tensors are on the same device."""
        device = grid.x_coords.device
        
        tensors_to_check = [grid.y_coords]
        if grid.x_bounds is not None:
            tensors_to_check.append(grid.x_bounds)
        if grid.y_bounds is not None:
            tensors_to_check.append(grid.y_bounds)
        if grid.cell_areas is not None:
            tensors_to_check.append(grid.cell_areas)
        if grid.mask is not None:
            tensors_to_check.append(grid.mask)
        
        return all(t.device == device for t in tensors_to_check)
