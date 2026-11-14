"""
Spatial Regridding for External Model Coupling

This module provides comprehensive spatial regridding capabilities for
transforming external model data to target grids required by RELIC.
Implements conservative, bilinear, and nearest-neighbor interpolation
methods with mass/energy conservation guarantees.

Integrates with the internal relic.coupling.scales.spatial regridding
infrastructure while providing external model specific adaptations
for coordinate systems, land/ocean masks, and data quality control.

Mathematical Framework:
Spatial regridding transforms data from source grid S to target grid T:
f_T = R(f_S, W, M)

Where:
- f_S, f_T: data fields on source and target grids
- R: regridding operator (conservative, bilinear, nearest)
- W: interpolation weights (sparse matrix)
- M: land/sea mask for appropriate domain restriction

Conservation Properties:
For conservative regridding of extensive quantities:
sum(f_T * A_T) = sum(f_S * A_S)
where A_S, A_T are grid cell areas.

References:
- ESMF Regridding: https://earthsystemmodeling.org/regrid/
- SCRIP: https://github.com/SCRIP-Project/SCRIP
- Jones, P.W. (1999) First- and second-order conservative remapping schemes
- xESMF: https://xesmf.readthedocs.io/en/latest/
"""

from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import xarray as xr
import torch
from torch import Tensor

# Import internal regridding infrastructure
from relic.coupling.scales.spatial.grid_utils import GridDescriptor, GridUtils
from relic.coupling.scales.spatial.factory import SpatialRegridderFactory

from relic.parameters import get_global_manager
from relic.coupling.core.conservation_checker import require_relative_tolerance
from relic.coupling.scales.types import SpatialQuantityBasis, convert_spatial_basis
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace


# CF Conventions 1.8 sections 4.1 and 4.2: latitude and longitude coordinates are identified by
# their units; the optional standard_name (latitude/longitude) and axis (Y/X) must agree with them.
CF_LATITUDE_UNITS = frozenset(
    {"degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN"}
)
CF_LONGITUDE_UNITS = frozenset(
    {"degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE"}
)
_CF_HORIZONTAL_ROLES = {
    "lat": ("latitude", "Y", CF_LATITUDE_UNITS, "degrees_north"),
    "lon": ("longitude", "X", CF_LONGITUDE_UNITS, "degrees_east"),
}


def cf_horizontal_role(name: str, attrs: Dict[str, Any]) -> Optional[str]:
    """
    CF role of one coordinate ('lat', 'lon' or None) from its attributes, never from its name.

    Raises:
        ValueError: If the attributes contradict each other (for example latitude units with
            axis X, or standard_name latitude without latitude units).
    """
    units = attrs.get("units")
    role = None
    for candidate, (_, _, cf_units, _) in _CF_HORIZONTAL_ROLES.items():
        if units in cf_units:
            role = candidate
    standard_name = attrs.get("standard_name")
    for candidate, (cf_standard_name, _, _, canonical_units) in _CF_HORIZONTAL_ROLES.items():
        if standard_name == cf_standard_name and role != candidate:
            raise ValueError(
                f"Coordinate {name!r} has standard_name {standard_name!r} but units {units!r}; CF "
                f"{cf_standard_name} coordinates carry units such as {canonical_units!r}."
            )
    axis = attrs.get("axis")
    if role is not None and axis is not None and axis != _CF_HORIZONTAL_ROLES[role][1]:
        raise ValueError(
            f"Coordinate {name!r} has {_CF_HORIZONTAL_ROLES[role][0]} units {units!r} but axis {axis!r}."
        )
    return role


def cf_horizontal_coordinates(coords: Any) -> Dict[str, str]:
    """
    Names of the CF latitude and longitude coordinates among ``coords`` (name -> DataArray).

    Returns:
        {'lat': name, 'lon': name} restricted to the roles found

    Raises:
        ValueError: If two coordinates claim the same role or their CF attributes conflict
    """
    roles: Dict[str, str] = {}
    for name, coord in coords.items():
        role = cf_horizontal_role(str(name), dict(coord.attrs))
        if role is None:
            continue
        if role in roles:
            raise ValueError(f"Several CF {role} coordinates: {roles[role]!r} and {str(name)!r}.")
        roles[role] = str(name)
    return roles


class RegridMethod(Enum):
    """Supported spatial regridding methods."""
    CONSERVATIVE = "conservative"        # Mass/energy conserving remapping
    BILINEAR = "bilinear"               # Bilinear interpolation
    NEAREST = "nearest_neighbor"        # Nearest neighbor
    PATCH_RECOVERY = "patch_recovery"   # Higher-order patch recovery
    AUTO = "auto"                       # Automatic method selection


@dataclass
class RegridConfig:
    """
    Configuration for spatial regridding operations.

    Specifies method, quality control options, performance settings,
    and physical constraints for regridding external model data.
    ``spatial_basis`` is the single spatial representation owner
    (relic.coupling.scales.types.SpatialQuantityBasis) and has no default: every
    external field declares intensive, cell_total, area_density or categorical.
    A bare "extensive" is rejected as ambiguous.
    """
    spatial_basis: SpatialQuantityBasis
    # The regridding kernel; required, like the basis (the caller states the method it applies).
    method: RegridMethod
    
    # Quality control
    check_conservation: Optional[bool] = None      # Verify mass/energy conservation
    conservation_tolerance: Optional[float] = None # Relative conservation error threshold
    
    # Physical constraints
    positive_definite: Optional[bool] = None      # Ensure non-negative values
    
    # Metadata
    description: str = ""                # Description of regridding purpose

    def __post_init__(self) -> None:
        self.spatial_basis = SpatialQuantityBasis.parse(self.spatial_basis)


@dataclass
class RegridResult:
    """
    Results from spatial regridding operation.
    
    Contains regridded data, quality metrics, and diagnostic information
    for validation and analysis.
    """
    data: Union[xr.DataArray, Tensor]   # Regridded data
    source_grid: GridDescriptor         # Source grid specification
    target_grid: GridDescriptor         # Target grid specification
    method_used: RegridMethod           # Actual method applied
    
    # Quality metrics
    conservation_error: Optional[float] = None    # Relative conservation error
    max_interpolation_error: Optional[float] = None  # Maximum local error estimate
    missing_data_fraction: float = 0.0  # NaN fraction of the source field (measured, no policy)
    
    # Performance metrics
    regrid_time: float = 0.0            # Regridding computation time (seconds)
    memory_peak: Optional[float] = None  # Peak memory usage (MB)
    
    # Output constraint: the positive_definite clip (None when the constraint is off). Keys:
    # clipped_cells, lower_bound, and for amounts amount_before, amount_after and clip_amount with
    # amount_before = amount_after + clip_amount (StateManager clip_loss sign: negative adds material).
    positive_definite_ledger: Optional[Dict[str, float]] = None

    # Diagnostics
    warnings: List[str] = field(default_factory=list)  # Quality warnings
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info


class RegridOperator:
    """
    High-level spatial regridding interface for external model data.
    
    Provides automatic method selection, quality control, and integration
    with RELIC internal grid systems. Handles diverse coordinate systems,
    irregular grids, and specialized requirements of different external models.
    """
    
    def __init__(self, parameter_manager=None, parameter_manager_resolver=None):
        """Initialize regridding operator with default configuration."""
        self._param_manager = parameter_manager
        self._param_manager_resolver = parameter_manager_resolver
        self._factory_cache = {}
        self._positive_clamp: Optional[float] = None

    def _get_parameter_manager(self):
        """Resolve ParameterManager lazily with optional injection."""
        if self._param_manager is None and self._param_manager_resolver is not None:
            self._param_manager = self._param_manager_resolver()
        if self._param_manager is None:
            self._param_manager = get_global_manager()
        if self._param_manager is None:
            raise RuntimeError(
                "ParameterManager is required for RegridOperator. "
                "Provide parameter_manager or parameter_manager_resolver."
            )
        return self._param_manager

    def _get_positive_clamp(self) -> float:
        """Retrieve the positive clamp threshold from the parameter system."""
        if self._positive_clamp is not None:
            return self._positive_clamp
        self._positive_clamp = self._get_scalar_parameter(
            "clamp_min_zero",
            namespace=resolve_namespace(ParameterNamespace.VALIDATION_NUMERICAL),
        )
        return self._positive_clamp

    def _apply_parameter_defaults(self, config: RegridConfig) -> None:
        """Populate config fields from the ParameterManager when not explicitly provided."""
        namespace = resolve_namespace(ParameterNamespace.COUPLING_EXTERNAL_REGRID)
        
        # One regridder default: an unset tolerance is coupling.scales.defaults.conservation_tolerance,
        # the value every regridder built without one reads (SpatialRegridderBase); it is set
        # explicitly so that the quality metric of this operator and its regridders use one value.
        if config.conservation_tolerance is None:
            config.conservation_tolerance = require_relative_tolerance(
                self._get_scalar_parameter(
                    "conservation_tolerance",
                    namespace=resolve_namespace(ParameterNamespace.COUPLING_SCALES_DEFAULTS),
                ),
                "coupling.scales.defaults.conservation_tolerance",
            )
        else:
            config.conservation_tolerance = require_relative_tolerance(
                config.conservation_tolerance, "RegridConfig.conservation_tolerance"
            )
        
        # Handle boolean flags including check_conservation
        if config.check_conservation is None:
            config.check_conservation = self._get_bool_parameter(
                "check_conservation",
                namespace=namespace,
            )
        
        if config.positive_definite is None:
            config.positive_definite = self._get_bool_parameter(
                "positive_definite",
                namespace=namespace,
            )

    def _get_scalar_parameter(
        self,
        key: str,
        namespace: str,
        units: str = "dimensionless",
    ) -> float:
        manager = self._get_parameter_manager()
        try:
            value = manager.get(
                key,
                namespace=namespace,
                units=units,
                required=True,
            )
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001 - explicit audit trail
            raise ValueError(f"Missing regrid parameter {namespace}.{key}: {exc}") from exc
        if value is None:
            raise ValueError(
                f"Missing regrid parameter {namespace}.{key}"
            )
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _get_bool_parameter(self, key: str, namespace: str) -> bool:
        manager = self._get_parameter_manager()
        try:
            value = manager.get(
                key,
                namespace=namespace,
                units="dimensionless",
                expected_type=bool,
                required=True,
            )
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001 - explicit audit trail
            raise ValueError(f"Missing regrid boolean parameter {namespace}.{key}: {exc}") from exc
        if value is None:
            raise ValueError(f"Missing regrid boolean parameter {namespace}.{key}")
        return bool(value)
    
    def regrid(self,
              source_data: Union[xr.DataArray, Tensor],
              source_grid: Union[GridDescriptor, xr.Dataset, Dict],
              target_grid: Union[GridDescriptor, Dict],
              config: RegridConfig) -> RegridResult:
        """
        Regrid data from source to target grid.
        
        Parameters:
            source_data: Data to regrid (with spatial dimensions)
            source_grid: Source grid specification
            target_grid: Target grid specification  
            config: Regridding configuration with an explicit spatial_basis (required)
            
        Returns:
            Regridding results with quality metrics
            
        Raises:
            ValueError: If grids are mismatched or data shape mismatch
            RuntimeError: If regridding fails
        """
        if not isinstance(config, RegridConfig):
            raise TypeError(
                f"config must be a RegridConfig with an explicit spatial_basis, got {type(config).__name__}"
            )
        self._apply_parameter_defaults(config)
        
        import time
        start_time = time.time()
        
        # Standardize grid specifications
        source_grid_desc = self._standardize_grid(source_grid)
        target_grid_desc = self._standardize_grid(target_grid)
        
        # Validate input data and grids
        self._validate_regrid_inputs(source_data, source_grid_desc, target_grid_desc, config)
        
        # Preprocess data (handle missing values, coordinate ordering)
        processed_data, preprocessing_info = self._preprocess_data(source_data, config)
        
        # Perform regridding
        regridded_data, resolved_method = self._apply_regridding(
            processed_data, source_grid_desc, target_grid_desc, config
        )
        
        # Output constraint with its ledger (an amount clip raises when conservation is required)
        final_data, positive_definite_ledger = self._postprocess_data(regridded_data, config, target_grid_desc)
        
        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(
            source_data, final_data, source_grid_desc, target_grid_desc, config
        )
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        result = RegridResult(
            data=final_data,
            source_grid=source_grid_desc,
            target_grid=target_grid_desc,
            method_used=resolved_method,
            regrid_time=elapsed_time,
            missing_data_fraction=preprocessing_info["missing_fraction"],
            positive_definite_ledger=positive_definite_ledger,
            **quality_metrics
        )
        
        return result
    
    def _standardize_grid(self, grid_spec: Union[GridDescriptor, xr.Dataset, Dict]) -> GridDescriptor:
        """
        Convert various grid specifications to standard GridDescriptor.
        
        Parameters:
            grid_spec: Grid specification in various formats
            
        Returns:
            Standardized grid descriptor
        """
        if isinstance(grid_spec, GridDescriptor):
            return grid_spec
        
        elif isinstance(grid_spec, xr.Dataset):
            # Extract grid from xarray Dataset
            return self._extract_grid_from_dataset(grid_spec)
        
        elif isinstance(grid_spec, dict):
            # Build grid from dictionary specification
            return self._build_grid_from_dict(grid_spec)
        
        else:
            raise ValueError(f"Unsupported grid specification type: {type(grid_spec)}")
    
    def _extract_grid_from_dataset(self, dataset: xr.Dataset) -> GridDescriptor:
        """
        Extract a grid descriptor from the CF coordinates of an xarray Dataset.

        Latitude and longitude are identified by their CF attributes (units with consistent
        standard_name and axis; CF 1.8 sections 4.1-4.2), never by their names. The cell bounds
        of 1-D coordinates come from the variables named by their ``bounds`` attribute (CF 1.8
        section 7.1) and are never inferred from the coordinate spacing; a grid without declared
        bounds has x_bounds/y_bounds None, and every consumer of cell geometry then raises.
        2-D (curvilinear) coordinates carry no bounds.

        Parameters:
            dataset: xarray Dataset with CF coordinate information

        Returns:
            Grid descriptor extracted from dataset
        """
        roles = cf_horizontal_coordinates(dataset.coords)
        if set(roles) != {"lat", "lon"}:
            raise ValueError(
                "The grid dataset needs one CF latitude and one CF longitude coordinate (units "
                f"degrees_north / degrees_east, CF 1.8 sections 4.1-4.2); found {roles or 'none'}."
            )
        lat_coord = dataset.coords[roles["lat"]]
        lon_coord = dataset.coords[roles["lon"]]

        if lon_coord.ndim == 2 and lon_coord.dims == lat_coord.dims:
            # 2-D curvilinear grid (e.g. WRF XLAT/XLONG).
            x_coords = torch.as_tensor(lon_coord.values, dtype=torch.float32)
            y_coords = torch.as_tensor(lat_coord.values, dtype=torch.float32)
            return GridDescriptor(
                grid_type="curvilinear",
                coord_system="latlon",
                dims=tuple(x_coords.shape),
                x_coords=x_coords,
                y_coords=y_coords,
                units={"x": "degrees_east", "y": "degrees_north"},
            )

        if lon_coord.ndim == 1 and lat_coord.ndim == 1 and lon_coord.dims != lat_coord.dims:
            # 1-D rectilinear grid (e.g. CMIP6).
            x_coords = torch.as_tensor(lon_coord.values, dtype=torch.float32)
            y_coords = torch.as_tensor(lat_coord.values, dtype=torch.float32)
            x_bounds = self._cf_coordinate_bounds(dataset, lon_coord)
            y_bounds = self._cf_coordinate_bounds(dataset, lat_coord)
            if (x_bounds is None) != (y_bounds is None):
                raise ValueError("Latitude and longitude cell bounds must be declared together.")
            return GridDescriptor(
                grid_type="regular",
                coord_system="latlon",
                dims=(int(y_coords.numel()), int(x_coords.numel())),
                x_coords=x_coords,
                y_coords=y_coords,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                units={"x": "degrees_east", "y": "degrees_north"},
            )

        raise ValueError(
            f"Unsupported coordinate structure: latitude {lat_coord.dims}, longitude {lon_coord.dims}."
        )

    @staticmethod
    def _cf_coordinate_bounds(dataset: xr.Dataset, coord: xr.DataArray) -> Optional[Tensor]:
        """Cell bounds [n, 2] of a 1-D coordinate from its CF ``bounds`` attribute, or None."""
        bounds_name = coord.attrs.get("bounds")
        if bounds_name is None:
            return None
        if bounds_name not in dataset.variables:
            raise ValueError(
                f"Coordinate {coord.name!r} declares bounds {bounds_name!r}, which the dataset does not contain."
            )
        values = np.asarray(dataset[bounds_name].values, dtype=np.float64)
        if values.shape != (coord.size, 2):
            raise ValueError(
                f"Bounds {bounds_name!r} of {coord.name!r} must have shape {(coord.size, 2)}; got {values.shape}."
            )
        return torch.as_tensor(values, dtype=torch.float32)

    def _build_grid_from_dict(self, grid_dict: Dict) -> GridDescriptor:
        """
        Build grid descriptor from dictionary specification.
        
        Parameters:
            grid_dict: Dictionary with grid parameters
            
        Returns:
            Grid descriptor built from dictionary
        """
        required_keys = ['lon', 'lat']
        for key in required_keys:
            if key not in grid_dict:
                raise ValueError(f"Missing required grid parameter: {key}")
        
        lon = np.asarray(grid_dict['lon'])
        lat = np.asarray(grid_dict['lat'])
        
        # Ensure 2D arrays
        x_bounds = None
        y_bounds = None
        has_lon_bounds = 'lon_bounds' in grid_dict
        has_lat_bounds = 'lat_bounds' in grid_dict
        if has_lon_bounds != has_lat_bounds:
            raise ValueError("lon_bounds and lat_bounds must be given together")
        if lon.ndim == 1 and lat.ndim == 1:
            grid_type = grid_dict.get('grid_type', 'regular')
            if grid_type == 'rectilinear':
                grid_type = 'regular'
            coord_system = grid_dict.get('coord_system', 'latlon')
            x_coords = torch.as_tensor(lon, dtype=torch.float32)
            y_coords = torch.as_tensor(lat, dtype=torch.float32)
            dims = (int(y_coords.numel()), int(x_coords.numel()))
            if has_lon_bounds:
                # CF cell bounds [n, 2] of the 1-D coordinates (e.g. lat_bnds/lon_bnds).
                x_bounds = torch.as_tensor(np.asarray(grid_dict['lon_bounds']), dtype=torch.float32)
                y_bounds = torch.as_tensor(np.asarray(grid_dict['lat_bounds']), dtype=torch.float32)
                if tuple(x_bounds.shape) != (dims[1], 2) or tuple(y_bounds.shape) != (dims[0], 2):
                    raise ValueError(
                        "lon_bounds/lat_bounds must have shapes (n_lon, 2) and (n_lat, 2); got "
                        f"{tuple(x_bounds.shape)} and {tuple(y_bounds.shape)}"
                    )
        elif lon.ndim == 2 and lat.ndim == 2:
            if has_lon_bounds:
                raise ValueError("Cell bounds are supported for 1-D lat/lon coordinates only")
            grid_type = grid_dict.get('grid_type', 'curvilinear')
            if grid_type == 'rectilinear':
                grid_type = 'curvilinear'
            coord_system = grid_dict.get('coord_system', 'latlon')
            x_coords = torch.as_tensor(lon, dtype=torch.float32)
            y_coords = torch.as_tensor(lat, dtype=torch.float32)
            dims = tuple(x_coords.shape)
        else:
            raise ValueError("Longitude and latitude must be 1D or 2D arrays")
        units = grid_dict.get('units', {"x": "degrees_east", "y": "degrees_north"})

        return GridDescriptor(
            grid_type=grid_type,
            coord_system=coord_system,
            dims=dims,
            x_coords=x_coords,
            y_coords=y_coords,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            units=units,
        )

    def _validate_regrid_inputs(self, 
                               data: Union[xr.DataArray, Tensor],
                               source_grid: GridDescriptor,
                               target_grid: GridDescriptor,
                               config: RegridConfig) -> None:
        """
        Validate inputs for regridding operation.
        
        Parameters:
            data: Data to regrid
            source_grid: Source grid specification
            target_grid: Target grid specification
            config: Regridding configuration
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check data dimensionality
        if isinstance(data, xr.DataArray):
            data_shape = data.shape
        else:
            data_shape = data.shape
        
        # Data must have at least 2D spatial dimensions
        if len(data_shape) < 2:
            raise ValueError("Data must have at least 2 spatial dimensions")
        
        # Check grid shapes match data
        expected_spatial_shape = source_grid.shape
        actual_spatial_shape = data_shape[-2:]  # Last 2 dims assumed spatial
        
        if expected_spatial_shape != actual_spatial_shape:
            raise ValueError(
                f"Data spatial shape {actual_spatial_shape} does not match "
                f"source grid shape {expected_spatial_shape}"
            )
        
        # Check coordinate validity
        if not self._validate_coordinates(source_grid):
            raise ValueError("Invalid source grid coordinates")
        
        if not self._validate_coordinates(target_grid):
            raise ValueError("Invalid target grid coordinates")
    
    def _validate_coordinates(self, grid: GridDescriptor) -> bool:
        """
        Validate grid coordinate arrays.

        Parameters:
            grid: Grid descriptor to validate (x_coords = longitude, y_coords = latitude
                for lat-lon grids)

        Returns:
            True if coordinates are finite and, for lat-lon grids, within range
        """
        lon = grid.x_coords.detach().cpu()
        lat = grid.y_coords.detach().cpu()
        if not bool(torch.isfinite(lon).all()) or not bool(torch.isfinite(lat).all()):
            return False
        if grid.coord_system == "latlon":
            if bool((lon < -360).any()) or bool((lon > 360).any()):
                return False
            if bool((lat < -90).any()) or bool((lat > 90).any()):
                return False
        return True

    def _preprocess_data(self,
                        data: Union[xr.DataArray, Tensor],
                        config: RegridConfig) -> Tuple[Union[xr.DataArray, Tensor], Dict]:
        """
        Measure the missing (NaN) fraction of the source field; the field is not changed.

        Missing values are neither filled nor interpolated through: the regridders propagate NaN,
        and the missing-data rule of external fields (range check, rejection above
        coupling.external.regrid.missing_threshold) belongs to the adapter quality control.

        Parameters:
            data: Input data
            config: Regridding configuration

        Returns:
            Tuple of (data, {"missing_fraction": NaN count / value count})
        """
        values = data.values if isinstance(data, xr.DataArray) else data
        missing = torch.isnan(torch.as_tensor(np.asarray(values) if not isinstance(values, Tensor) else values))
        if missing.numel() == 0:
            raise ValueError("The source field is empty; its missing fraction is undefined.")
        return data, {"missing_fraction": float(missing.sum().item()) / float(missing.numel())}
    
    def _apply_regridding(self,
                         data: Union[xr.DataArray, Tensor],
                         source_grid: GridDescriptor,
                         target_grid: GridDescriptor,
                         config: RegridConfig) -> Union[xr.DataArray, Tensor]:
        """
        Apply the selected regridding method.
        
        Parameters:
            data: Preprocessed data
            source_grid: Source grid specification
            target_grid: Target grid specification  
        config: Regridding configuration
            
        Returns:
            Tuple of (regridded data, resolved method)
        """
        if isinstance(data, xr.DataArray):
            data_tensor = torch.as_tensor(data.values, dtype=torch.float32)
        elif isinstance(data, Tensor):
            data_tensor = data.float()
        else:
            data_tensor = torch.as_tensor(np.asarray(data), dtype=torch.float32)

        # Map RegridMethod enum to factory method names
        method_enum = config.method if isinstance(config.method, RegridMethod) else (
            RegridMethod(config.method) if config.method is not None else RegridMethod.AUTO
        )
        method_map = {
            RegridMethod.CONSERVATIVE: "conservative",
            RegridMethod.BILINEAR: "bilinear",
            RegridMethod.NEAREST: "nearest",
        }
        if method_enum == RegridMethod.PATCH_RECOVERY:
            raise ValueError("Patch recovery regridding is not available in SpatialRegridderFactory.")
        method_override = None if method_enum == RegridMethod.AUTO else method_map.get(method_enum)
        if method_enum not in method_map and method_enum != RegridMethod.AUTO:
            raise ValueError(f"Unsupported regridding method: {method_enum}")

        variable_metadata = self._build_variable_metadata(config)
        pm = self._get_parameter_manager()

        factory_key = (bool(config.check_conservation), config.conservation_tolerance)
        factory = self._factory_cache.get(factory_key)
        if factory is None:
            factory = SpatialRegridderFactory(
                enforce_conservation=bool(config.check_conservation),
                parameter_manager=pm,
                conservation_tolerance=config.conservation_tolerance,
            )
            self._factory_cache[factory_key] = factory

        regridder = factory.create_regridder(
            source_grid,
            target_grid,
            method=method_override,
            variable_metadata=variable_metadata,
        )

        regridded_tensor = regridder.upscale(data_tensor)
        resolved_method = self._resolve_method_enum(
            method_enum,
            getattr(regridder, "selected_method", None),
            method_override,
        )
        
        # Convert back to original format
        if isinstance(data, xr.DataArray):
            # Preserve xarray structure with updated coordinates
            regridded_data = self._create_regridded_dataarray(
                regridded_tensor, data, target_grid
            )
        else:
            regridded_data = regridded_tensor
        
        return regridded_data, resolved_method

    def _build_variable_metadata(self, config: RegridConfig) -> Optional[Dict[str, Any]]:
        """
        Build variable metadata payload for factory method selection.

        Parameters
        ----------
        config : RegridConfig
            Regridding configuration containing variable classification.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with the explicit spatial_basis and conservation hints.
        """
        variable_metadata: Dict[str, Any] = {"spatial_basis": config.spatial_basis.value}
        if config.check_conservation:
            variable_metadata["require_conservation"] = True
        return variable_metadata

    def _resolve_method_enum(
        self,
        requested_method: RegridMethod,
        selected_method: Optional[Union[str, RegridMethod]],
        method_override: Optional[str],
    ) -> RegridMethod:
        """
        Resolve the effective RegridMethod used for this operation.

        Parameters
        ----------
        requested_method : RegridMethod
            Method requested in the configuration (may be AUTO).
        selected_method : Optional[Union[str, RegridMethod]]
            Method chosen by the factory or regridder (string or enum).
        method_override : Optional[str]
            Explicit override passed to the factory (None when AUTO).

        Returns
        -------
        RegridMethod
            Effective method enum to record in results.
        """
        if requested_method != RegridMethod.AUTO:
            return requested_method

        if isinstance(selected_method, RegridMethod):
            return selected_method

        method_lookup = {
            "conservative": RegridMethod.CONSERVATIVE,
            "bilinear": RegridMethod.BILINEAR,
            "nearest": RegridMethod.NEAREST,
            "nearest_neighbor": RegridMethod.NEAREST,
        }

        if isinstance(selected_method, str):
            return method_lookup.get(selected_method, RegridMethod.AUTO)

        if method_override:
            return method_lookup.get(method_override, RegridMethod.AUTO)

        return RegridMethod.AUTO
    
    def _create_regridded_dataarray(self,
                                  regridded_tensor: Tensor,
                                  original_da: xr.DataArray,
                                  target_grid: GridDescriptor) -> xr.DataArray:
        """
        Create regridded xarray DataArray with the target grid coordinates.

        Parameters:
            regridded_tensor: Regridded data tensor
            original_da: Original DataArray for metadata (last two dims are spatial)
            target_grid: Target grid specification

        Returns:
            Regridded DataArray; non-spatial coordinates are kept, source coordinates on the
            spatial dimensions are replaced by the target coordinates (1-D for regular grids,
            2-D ``lat``/``lon`` auxiliary coordinates for curvilinear grids).
        """
        dims = list(original_da.dims)
        spatial_dims = dims[-2:]
        coords = {
            name: coord
            for name, coord in original_da.coords.items()
            if not set(coord.dims) & set(spatial_dims)
        }
        target_x = target_grid.x_coords.detach().cpu().numpy()
        target_y = target_grid.y_coords.detach().cpu().numpy()
        if target_grid.grid_type == "regular" and target_x.ndim == 1 and target_y.ndim == 1:
            coords[spatial_dims[0]] = (spatial_dims[0], target_y)
            coords[spatial_dims[1]] = (spatial_dims[1], target_x)
        elif target_grid.grid_type == "curvilinear":
            coords["lat"] = (tuple(spatial_dims), target_y)
            coords["lon"] = (tuple(spatial_dims), target_x)
        else:
            raise ValueError(
                f"Cannot attach coordinates of a {target_grid.grid_type!r} target grid to a 2-D field"
            )

        new_shape = list(original_da.shape)
        new_shape[-2:] = list(target_grid.shape)
        regridded_array = regridded_tensor.detach().cpu().numpy().reshape(new_shape)

        return xr.DataArray(
            regridded_array,
            coords=coords,
            dims=dims,
            attrs=original_da.attrs.copy()
        )

    def _postprocess_data(self,
                         data: Union[xr.DataArray, Tensor],
                         config: RegridConfig,
                         target_grid: GridDescriptor) -> Tuple[Union[xr.DataArray, Tensor], Optional[Dict[str, float]]]:
        """
        Apply the positive_definite output constraint and return its ledger (None when it is off).

        Parameters:
            data: Regridded data
            config: Regridding configuration
            target_grid: Target grid (cell areas of the amount ledger)

        Returns:
            Tuple of (constrained data, ledger)

        Raises:
            ValueError: If the clip changes an amount (cell_total or area_density) of a call that
                requires conservation (check_conservation)
        """
        if not config.positive_definite:
            return data, None
        lower = float(self._get_positive_clamp())
        values = torch.as_tensor(data.values) if isinstance(data, xr.DataArray) else data
        # References: CF Conventions 1.8 section 2.5.1 (valid_min >= 0 makes a field non-negative);
        #   relic/core/state/manager.py bound enforcement (clip_loss = value before - value after).
        # Mathematical expression: y_c,J = max(y_J, b); d_J = y_J - y_c,J; for amounts
        #   C = sum_J d_J a_J, so sum_J y_J a_J = sum_J y_c,J a_J + C.
        # Parameter meanings: y remapped field; b positivity bound (validation.numerical.clamp_min_zero);
        #   d per-cell clip; a_J = A_J (area density) or 1 (cell total); C clip amount.
        # Units: y, b and d in the field units; A_J [m2]; C in the amount unit of the field.
        # Domain/constraints: NaN cells are neither clipped nor counted (their amounts stay NaN); the
        #   ledger is evaluated in float64 over all leading entries; an amount clip raises when the
        #   call requires conservation, otherwise it is returned.
        # Sign convention: C < 0 when the clip adds material (negative values raised to the bound),
        #   as the StateManager clip_loss (positive removes storage, negative discloses additions).
        clipped_mask = values < lower
        constrained = torch.where(clipped_mask, torch.full_like(values, lower), values)
        ledger: Dict[str, float] = {"clipped_cells": int(clipped_mask.sum().item()), "lower_bound": lower}
        if config.spatial_basis.is_extensive:
            areas = torch.as_tensor(np.asarray(self._compute_grid_areas(target_grid)), dtype=torch.float64)
            before = convert_spatial_basis(
                values.detach().to(device="cpu", dtype=torch.float64), areas,
                config.spatial_basis, SpatialQuantityBasis.CELL_TOTAL,
            )
            after = convert_spatial_basis(
                constrained.detach().to(device="cpu", dtype=torch.float64), areas,
                config.spatial_basis, SpatialQuantityBasis.CELL_TOTAL,
            )
            ledger["amount_before"] = float(before.sum().item())
            ledger["amount_after"] = float(after.sum().item())
            ledger["clip_amount"] = float((before - after).sum().item())
            if config.check_conservation and ledger["clipped_cells"] > 0:
                raise ValueError(
                    f"positive_definite clip of a {config.spatial_basis.value} field changes its amount "
                    f"by {ledger['clip_amount']:.6e} ({ledger['clipped_cells']} cells below "
                    f"{lower}; amount before {ledger['amount_before']:.6e} = after "
                    f"{ledger['amount_after']:.6e} + clip): the call requires conservation "
                    "(check_conservation). Remove the negative values of the field, or regrid without "
                    "check_conservation to receive the clip as a disclosed ledger."
                )
        if isinstance(data, xr.DataArray):
            return data.copy(data=constrained.detach().cpu().numpy()), ledger
        return constrained, ledger
    
    def _compute_quality_metrics(self,
                                source_data: Union[xr.DataArray, Tensor],
                                regridded_data: Union[xr.DataArray, Tensor],
                                source_grid: GridDescriptor,
                                target_grid: GridDescriptor,
                                config: RegridConfig) -> Dict[str, Any]:
        """
        Compute quality metrics for regridding operation.
        
        Parameters:
            source_data: Original data
            regridded_data: Regridded data
            source_grid: Source grid specification
            target_grid: Target grid specification
            config: Regridding configuration
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Conservation check for amounts (cell totals and area densities)
        if config.check_conservation and config.spatial_basis.is_extensive:
            conservation_error = self._compute_conservation_error(
                source_data, regridded_data, source_grid, target_grid, config.spatial_basis
            )
            metrics['conservation_error'] = conservation_error
            
            if conservation_error > config.conservation_tolerance:
                metrics.setdefault('warnings', []).append(
                    f"Conservation error {conservation_error:.2e} exceeds tolerance "
                    f"{config.conservation_tolerance:.2e}"
                )
        
        return metrics
    
    def _compute_conservation_error(self,
                                  source_data: Union[xr.DataArray, Tensor],
                                  regridded_data: Union[xr.DataArray, Tensor],
                                  source_grid: GridDescriptor,
                                  target_grid: GridDescriptor,
                                  spatial_basis: SpatialQuantityBasis) -> float:
        """
        Compute relative conservation error of an amount on its declared basis.

        Parameters:
            source_data: Original data
            regridded_data: Regridded data
            source_grid: Source grid
            target_grid: Target grid
            spatial_basis: cell_total or area_density

        Returns:
            Relative conservation error
        """
        # References: Jones (1999) conservative remapping; docs/source/theory/scale_coupling.md 2.3.2.
        # Mathematical expression: amounts X = convert(values, A, basis -> cell_total)
        #   (X = x A for an area density, X unchanged for a cell total);
        #   error = max |sum_J X_J - sum_i X_i| / max |sum_i X_i| over leading batch entries.
        # Parameter meanings: x area density; X cell total; A cell areas of each grid.
        # Units: A [m2]; X in the amount unit; error [1].
        # Domain/constraints: only amount bases are checked; areas finite and positive; the last
        #   two dimensions are the spatial axes.
        # Sign convention: signed amounts; the error is non-negative.
        if not spatial_basis.is_extensive:
            raise ValueError(
                f"Conservation error is defined for cell_total or area_density fields, not {spatial_basis.value!r}."
            )
        # Compute grid cell areas
        source_areas = torch.as_tensor(np.asarray(self._compute_grid_areas(source_grid)), dtype=torch.float64)
        target_areas = torch.as_tensor(np.asarray(self._compute_grid_areas(target_grid)), dtype=torch.float64)

        def _as_tensor(values: Union[xr.DataArray, Tensor, np.ndarray]) -> Tensor:
            if isinstance(values, xr.DataArray):
                return torch.as_tensor(values.values, dtype=torch.float64)
            if isinstance(values, Tensor):
                return values.detach().to(device="cpu", dtype=torch.float64)
            return torch.as_tensor(np.asarray(values), dtype=torch.float64)

        source_amounts = convert_spatial_basis(
            _as_tensor(source_data), source_areas, spatial_basis, SpatialQuantityBasis.CELL_TOTAL
        ).numpy()
        target_amounts = convert_spatial_basis(
            _as_tensor(regridded_data), target_areas, spatial_basis, SpatialQuantityBasis.CELL_TOTAL
        ).numpy()

        # Handle multidimensional data (integrate over spatial dimensions)
        spatial_axes = (-2, -1)  # Last two dimensions assumed spatial

        source_total = np.sum(source_amounts, axis=spatial_axes)
        target_total = np.sum(target_amounts, axis=spatial_axes)
        
        # Compute relative error
        source_magnitude = np.abs(source_total).max()
        if source_magnitude > 0:
            conservation_error = np.abs(target_total - source_total).max() / source_magnitude
        else:
            conservation_error = 0.0
        
        return float(conservation_error)
    
    def _compute_grid_areas(self, grid: GridDescriptor) -> np.ndarray:
        """
        Compute grid cell areas in square meters.

        Parameters:
            grid: Grid descriptor (precomputed ``cell_areas`` are used when present)

        Returns:
            Array of grid cell areas [m2] on the host
        """
        if grid.cell_areas is not None:
            return torch.as_tensor(grid.cell_areas).detach().cpu().numpy()
        # The shared grid utility owns the spherical / Cartesian area formulas.
        return GridUtils(device=torch.device("cpu")).compute_grid_areas(grid).detach().cpu().numpy()

    def clear_cache(self) -> None:
        """Clear the regridder factories and with them their content-keyed regridder caches."""
        self._factory_cache.clear()
    
    def get_supported_methods(self) -> List[RegridMethod]:
        """Get list of supported regridding methods."""
        return [
            RegridMethod.CONSERVATIVE,
            RegridMethod.BILINEAR, 
            RegridMethod.NEAREST
        ]



