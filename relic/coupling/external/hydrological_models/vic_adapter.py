"""
VIC Model Data Adapter

This module implements the VICAdapter for accessing and processing data from 
the Variable Infiltration Capacity (VIC) hydrological model output files.
It handles VIC-specific variable naming, file formats, and coordinate systems.

The adapter supports standard VIC output formats including ASCII and NetCDF
and follows CF-1.8 conventions where possible.

References:
    - VIC Model Documentation: https://vic.readthedocs.io/
    - VIC Output Formats: https://vic.readthedocs.io/en/master/Documentation/OutputFormatting/
    - CF Conventions 1.8: http://cfconventions.org/cf-conventions/v1.8/
    
Mathematical Expressions:
    Water balance: P - ET - R - DeltaS = 0
    Infiltration: I = fc * (theta/thetas)^beta
    Evapotranspiration: ET = PET * Kc * (theta/thetas)
    Runoff: R = R_surf + R_base = (P - I) + K_base * S_base
    Soil moisture: theta = W / (rho_w * d * A)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass

import torch
import xarray as xr
import numpy as np

from ..common.base_adapter import validate_record_interval, validate_time_interpolation
from ..common import (
    ExternalAdapter, DataSource,
    TimeWindow, SpatialWindow,
    TimeHandler, RegridOperator,
    IOHandler
)
# ProcessCategory removed - ExternalAdapter handles category internally

from ....parameters.namespaces import ParameterNamespace, resolve_namespace

logger = logging.getLogger(__name__)

@dataclass
class VICConfig:
    """Configuration for VIC adapter"""
    root_path: Union[str, Path]
    file_format: str = "netcdf"  # only "netcdf" is read (see __post_init__)
    file_pattern: str = "*.nc"  # File pattern to match VIC output files
    output_type: str = "fluxes"  # "fluxes", "snow", "fdepth", "lakefrac", etc.
    time_step: str = "daily"  # "daily", "monthly", "yearly"
    spatial_agg: bool = False  # Whether files contain spatially aggregated data
    grid_file: Optional[Union[str, Path]] = None  # Grid definition file
    param_file: Optional[Union[str, Path]] = None  # Parameter file for metadata
    # Output record interval declared for water depths per record when the file has no
    # CF time bounds (e.g. 1.0 and "day" for a daily output step); never inferred.
    record_interval: Optional[float] = None
    record_interval_units: Optional[str] = None

    # Rule of a forcing read at the simulation time: 'lower' (t_k <= t < t_(k+1) holds record k)
    # or 'linear' (between the two records around t); required by forward, never assumed.
    time_interpolation: Optional[str] = None
    # Catalog variables the adapter writes (its declared outputs; each a mapping target of the
    # source in InternalSchema); None declares every mapping target of the source.
    provided_variables: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_time_interpolation(self.time_interpolation)
        validate_record_interval(self.record_interval, self.record_interval_units)
        if self.file_format != "netcdf":
            raise ValueError(
                f"VIC file_format {self.file_format!r}: only the VIC NetCDF output is read. VIC ASCII "
                "and binary output columns carry no variable names (their order is declared in the VIC "
                "global parameter file), so they are rejected instead of being parsed by column position."
            )


class VICAdapter(ExternalAdapter):
    """
    Adapter for Variable Infiltration Capacity (VIC) hydrological model output.
    
    This adapter provides access to VIC model output files, handling the
    model's specific variable naming conventions, file formats, and spatial
    organization. It can process both gridded and aggregated outputs.
    
    Key Features:
    - Multiple output format support (NetCDF, ASCII, Binary)
    - Gridded and point-based data handling
    - Water balance component processing
    - Soil layer information extraction
    - CF-1.8 compliant metadata generation
    
    Attributes:
        config: VIC configuration parameters
        discovered_files: List of discovered VIC files
        current_datasets: Currently loaded datasets
        grid_info: Spatial grid information
        variable_definitions: VIC variable definitions
    """
    
    def __init__(
        self,
        config: Optional[VICConfig] = None,
        time_window: Optional[TimeWindow] = None,
        spatial_window: Optional[SpatialWindow] = None,
        parameter_manager: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize VIC adapter.
        
        Parameters:
            config: VIC configuration
            time_window: Temporal subset window 
            spatial_window: Spatial subset window
            **kwargs: Additional arguments passed to ExternalAdapter
        """
        self._require_adapter_config(config, VICConfig, "VICAdapter")
        super().__init__(
            source=DataSource.VIC,
            name="VICAdapter",
            model_id="vic",
            parameter_manager=parameter_manager,
            provided_variables=None if config is None else config.provided_variables,
            **kwargs
        )
        
        if config is None:
            # The declared default output directory (coupling.external.vic.default_root_path); a
            # failed or missing lookup raises, and the current directory is never assumed.
            default_root = self.get_parameter_value(
                "default_root_path",
                namespace=resolve_namespace(ParameterNamespace.COUPLING_EXTERNAL_VIC),
                expected_type=str,
                validate=True,
                required=True,
            )
            config = VICConfig(root_path=Path(default_root))
        self.config = config
        self.time_window = time_window
        self.spatial_window = spatial_window
        
        # Internal state
        self.discovered_files: List[Path] = []
        self.current_datasets: Dict[str, xr.Dataset] = {}
        self.grid_info: Dict[str, Any] = {}
        self.variable_definitions: Dict[str, Dict] = {}
        
        # Initialize handlers
        self.time_handler = TimeHandler()
        self.regrid_operator = RegridOperator()
        self.io_handler = IOHandler()
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load VIC variable definitions
        self._load_vic_variable_definitions()
        
    def _load_vic_variable_definitions(self):
        """Load standard VIC variable definitions"""
        self.variable_definitions = {
            # Water balance components
            'OUT_PREC': {
                'long_name': 'Precipitation',
                'units': 'mm'
            },
            'OUT_EVAP': {
                'long_name': 'Total Evapotranspiration',
                'units': 'mm'
            },
            'OUT_RUNOFF': {
                'long_name': 'Surface Runoff',
                'units': 'mm'
            },
            'OUT_BASEFLOW': {
                'long_name': 'Baseflow',
                'units': 'mm'
            },
            'OUT_WDEW': {
                'long_name': 'Canopy Interception Storage',
                'units': 'mm'
            },
            
            # Soil moisture components
            'OUT_SOIL_LIQ': {
                'long_name': 'Soil Liquid Moisture',
                'units': 'mm'
            },
            'OUT_SOIL_ICE': {
                'long_name': 'Soil Ice Content',
                'units': 'mm'
            },
            'OUT_SOIL_MOIST': {
                'long_name': 'Total Soil Moisture',
                'units': 'mm'
            },
            'OUT_SOIL_TEMP': {
                'long_name': 'Soil Temperature',
                'units': 'C'
            },
            
            # Energy balance components
            'OUT_NET_SHORT': {
                'long_name': 'Net Shortwave Radiation',
                'units': 'W/m2'
            },
            'OUT_NET_LONG': {
                'long_name': 'Net Longwave Radiation',
                'units': 'W/m2'
            },
            'OUT_LATENT': {
                'long_name': 'Latent Heat Flux',
                'units': 'W/m2'
            },
            'OUT_SENSIBLE': {
                'long_name': 'Sensible Heat Flux',
                'units': 'W/m2'
            },
            'OUT_GRND_FLUX': {
                'long_name': 'Ground Heat Flux',
                'units': 'W/m2'
            },
            
            # Snow components
            'OUT_SWE': {
                'long_name': 'Snow Water Equivalent',
                'units': 'mm'
            },
            'OUT_SNOW_DEPTH': {
                'long_name': 'Snow Depth',
                'units': 'cm'
            },
            'OUT_SNOW_CANOPY': {
                'long_name': 'Snow Interception Storage',
                'units': 'mm'
            },
            'OUT_SNOW_MELT': {
                'long_name': 'Snow Melt',
                'units': 'mm'
            }
        }
        
    def discover(
        self, 
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover available VIC output files.
        
        Scans the specified directory for VIC output files matching the
        configuration pattern. Analyzes file formats, time coverage, and
        available variables.
        
        Parameters:
            variables: List of VIC variable names to check for
            **kwargs: Additional discovery parameters
            
        Returns:
            Dictionary containing discovered file information and metadata
            
        Raises:
            FileNotFoundError: If root path does not exist
            ValueError: If no matching files found
        """
        self.logger.info(f"Discovering VIC files in {self.config.root_path}")
        
        root_path = Path(self.config.root_path)
        if not root_path.exists():
            raise FileNotFoundError(f"VIC root path does not exist: {root_path}")
            
        # Find VIC output files
        file_pattern = self.config.file_pattern
        matching_files = list(root_path.glob(file_pattern))
        
        if not matching_files:
            # Try recursive search
            matching_files = list(root_path.rglob(file_pattern))
            
        if not matching_files:
            raise ValueError(f"No VIC files found matching pattern: {file_pattern}")
            
        # Sort files by name/timestamp
        # A deterministic listing; the records are ordered by their decoded times when opened.
        matching_files = sorted(matching_files)
        
        self.discovered_files = matching_files
        
        # Analyze file contents
        file_analysis = self._analyze_vic_files(matching_files, variables)
        
        results = {
            "source": "VIC",
            "total_files": len(matching_files),
            "file_format": file_analysis["format"],
            "time_coverage": file_analysis["time_coverage"],
            "available_variables": file_analysis["variables"],
            "spatial_info": file_analysis["spatial_info"],
            "grid_type": file_analysis["grid_type"]
        }
        
        self.grid_info = file_analysis["spatial_info"]
        
        self.logger.info(
            f"Discovered {results['total_files']} VIC files in "
            f"{results['file_format']} format"
        )
        
        return results
    
    def _analyze_vic_files(
        self, 
        files: List[Path], 
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze VIC files to extract metadata"""
        
        file_format = "unknown"
        all_variables = set()
        time_range = {'start': None, 'end': None}
        spatial_info = {}
        grid_type = "unknown"
        
        # Determine file format from first file
        if files:
            first_file = files[0]
            
            if first_file.suffix.lower() in ['.nc', '.nc4', '.netcdf']:
                file_format = "netcdf"
            elif first_file.suffix.lower() in ['.txt', '.dat', '.asc']:
                file_format = "ascii"
            else:
                file_format = "binary"
        
        # Analyze NetCDF files
        if file_format == "netcdf" and files:
            with xr.open_dataset(files[0], engine='netcdf4') as sample_ds:
                # Extract variables
                if variables:
                    available_vars = [var for var in variables if var in sample_ds.data_vars]
                    all_variables.update(available_vars)
                else:
                    all_variables.update(sample_ds.data_vars.keys())

                # Spatial information
                if 'lat' in sample_ds.coords and 'lon' in sample_ds.coords:
                    lat = sample_ds.coords['lat']
                    lon = sample_ds.coords['lon']
                    
                    spatial_info = {
                        'n_lat': len(lat) if lat.dims else 1,
                        'n_lon': len(lon) if lon.dims else 1,
                        'lat_min': float(lat.min()),
                        'lat_max': float(lat.max()),
                        'lon_min': float(lon.min()),
                        'lon_max': float(lon.max())
                    }
                    
                    # Determine grid type
                    if lat.dims and lon.dims:
                        if len(lat.dims) == 1 and len(lon.dims) == 1:
                            grid_type = "regular"
                        else:
                            grid_type = "irregular"
                    else:
                        grid_type = "point"
                
                # Time information
                if 'time' in sample_ds.coords:
                    time_coord = sample_ds.coords['time']
                    if len(time_coord) > 0:
                        time_range['start'] = str(time_coord.values[0])
                        if len(files) > 1:
                            # Check last file for end time
                            with xr.open_dataset(files[-1], engine='netcdf4') as last_ds:
                                if 'time' in last_ds.coords:
                                    last_time = last_ds.coords['time']
                                    if len(last_time) > 0:
                                        time_range['end'] = str(last_time.values[-1])
                        else:
                            time_range['end'] = str(time_coord.values[-1])
                
        
        elif files:
            raise ValueError(
                f"VIC output file {files[0].name} is not NetCDF (.nc, .nc4, .netcdf); only the VIC "
                "NetCDF output is read."
            )

        return {
            'format': file_format,
            'variables': sorted(list(all_variables)),
            'time_coverage': time_range,
            'spatial_info': spatial_info,
            'grid_type': grid_type
        }
    
    def open(self, **kwargs) -> Dict[str, Any]:
        """
        Open and load discovered VIC datasets.
        
        Opens VIC files and processes them according to configuration.
        Handles different file formats and coordinate systems.
        
        Parameters:
            **kwargs: Additional opening parameters
            
        Returns:
            Dictionary containing opened datasets and metadata
            
        Raises:
            ValueError: If no files have been discovered
            IOError: If files cannot be opened
        """
        if not self.discovered_files:
            raise ValueError("No VIC files discovered. Call discover() first.")
            
        self.logger.info(f"Opening {len(self.discovered_files)} VIC files")
        
        try:
            dataset = self._open_netcdf_files()
            
            # Apply temporal and spatial subsetting
            dataset = self._apply_subsetting(dataset)
            
            # Add standard metadata
            dataset = self._add_vic_metadata(dataset)
            
            self.current_datasets["main"] = dataset
            
            return {
                "source": "VIC",
                "datasets": ["main"],
                "temporal_coverage": self._get_temporal_coverage(),
                "variables": list(dataset.data_vars.keys())
            }
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            self.logger.error(f"Failed to open VIC datasets: {e}")
            raise IOError(f"Could not open VIC data: {e}")
    
    def _open_netcdf_files(self) -> xr.Dataset:
        """Open and concatenate NetCDF VIC files"""
        
        if len(self.discovered_files) == 1:
            # Single file
            dataset = xr.open_dataset(
                self.discovered_files[0],
                engine='netcdf4',
                decode_times=True,
                decode_cf=True
            )
        else:
            # Multiple files - concatenate along time
            datasets = []
            
            for filepath in self.discovered_files:
                ds = xr.open_dataset(
                    filepath,
                    engine='netcdf4',
                    decode_times=True,
                    decode_cf=True
                )
                datasets.append(ds)
            
            # Concatenate in the order of the decoded record times (never file names).
            dataset = self._concat_by_record_time(
                datasets, [path.name for path in self.discovered_files], dim='time'
            )
        
        return dataset
    
    def _apply_subsetting(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply temporal and spatial subsetting"""
        
        # Apply temporal subsetting
        if self.time_window and 'time' in ds.coords:
            ds = ds.sel(time=slice(self.time_window.start, self.time_window.end))
        
        # Apply spatial subsetting
        if self.spatial_window:
            subset_dict = {}
            
            if 'lat' in ds.coords:
                if ds.coords['lat'].dims:  # Gridded data
                    subset_dict['lat'] = slice(
                        self.spatial_window.lat_min,
                        self.spatial_window.lat_max
                    )
                else:  # Point data - check if within bounds
                    lat_val = float(ds.coords['lat'])
                    if not (self.spatial_window.lat_min <= lat_val <= self.spatial_window.lat_max):
                        return ds.isel(time=slice(0, 0))  # Empty dataset
            
            if 'lon' in ds.coords:
                if ds.coords['lon'].dims:  # Gridded data
                    subset_dict['lon'] = slice(
                        self.spatial_window.lon_min,
                        self.spatial_window.lon_max
                    )
                else:  # Point data - check if within bounds
                    lon_val = float(ds.coords['lon'])
                    if not (self.spatial_window.lon_min <= lon_val <= self.spatial_window.lon_max):
                        return ds.isel(time=slice(0, 0))  # Empty dataset
            
            if subset_dict:
                ds = ds.sel(**subset_dict)
        
        return ds
    
    def _add_vic_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Complete VIC metadata without overriding the file.

        Attributes the file declares are authoritative; the VIC table (VIC 5 output variable
        documentation) supplies long names, and units only where the file declares none
        (recorded as ``units_source = adapter_table``, e.g. VIC ASCII output).
        """
        filled = self._apply_adapter_attributes(ds, self.variable_definitions, "VIC hydrological model")
        if filled:
            self.logger.info("VIC units taken from the adapter table (none in the file): %s", filled)

        if 'time' in ds.coords:
            for key, value in {'standard_name': 'time', 'long_name': 'Time'}.items():
                ds['time'].attrs.setdefault(key, value)
        if 'lat' in ds.coords:
            for key, value in {'standard_name': 'latitude', 'long_name': 'Latitude', 'units': 'degrees_north'}.items():
                ds['lat'].attrs.setdefault(key, value)
        if 'lon' in ds.coords:
            for key, value in {'standard_name': 'longitude', 'long_name': 'Longitude', 'units': 'degrees_east'}.items():
                ds['lon'].attrs.setdefault(key, value)

        for key, value in {
            'source': 'VIC hydrological model',
            'model_version': 'VIC',
            'output_type': self.config.output_type,
            'time_step': self.config.time_step,
        }.items():
            ds.attrs.setdefault(key, value)
        return ds

    def _get_temporal_coverage(self) -> Dict[str, Dict]:
        """Get temporal coverage for opened datasets"""
        coverage = {}
        
        for dataset_name, ds in self.current_datasets.items():
            if 'time' in ds.coords:
                time_coord = ds.coords['time']
                coverage[dataset_name] = {
                    "start": str(time_coord.values[0]),
                    "end": str(time_coord.values[-1]),
                    "n_timesteps": len(time_coord)
                }
        
        return coverage
    
    def read(
        self, 
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Read and transform VIC data to RELIC internal format.
        
        Parameters:
            variables: List of variables to read (internal names)
            **kwargs: Additional reading parameters
            
        Returns:
            Dictionary mapping internal variable names to PyTorch tensors
        """
        if not self.current_datasets:
            raise ValueError("No datasets opened. Call open() first.")
        
        self.logger.info("Reading VIC variables")
        
        # Get variable mappings
        from ..common.schema import InternalSchema
        # internal -> VIC through the schema owner (ambiguous many-to-one mappings raise
        # unless a canonical external variable is declared).
        internal_to_vic = InternalSchema().resolve_external_names(DataSource.VIC, variables)
        
        tensors = {}
        ds = self.current_datasets["main"]
        # Mapped fields the dataset contains; a requested field it lacks raises and absent mapped
        # fields are listed in last_read_report (never skipped silently).
        present = self._present_mapped_fields({"main": ds}, internal_to_vic, variables)

        for internal_var in present["main"]:
            vic_var = internal_to_vic[internal_var]
            
            # Declared source transform (if any) and unit conversion: the one read path.
            converted_data = self._read_field(ds, vic_var, internal_var)

            # Convert to PyTorch tensor
            tensor = torch.from_numpy(converted_data.values.astype(np.float32))

            # Add metadata
            tensor_meta = {
                'shape': tensor.shape,
                'dims': converted_data.dims,
                'coords': {
                    name: torch.from_numpy(coord.values.astype(np.float32))
                    for name, coord in converted_data.coords.items()
                    if coord.dtype.kind in 'fc'
                },
                'coord_dims': {
                    name: tuple(coord.dims)
                    for name, coord in converted_data.coords.items()
                    if coord.dtype.kind in 'fc'
                },
                'variable': internal_var,
                **self._record_time_metadata(converted_data),
                'cell_bounds': self._cf_horizontal_coordinates(ds, converted_data),
                'attrs': dict(converted_data.attrs)
            }

            setattr(tensor, '_relic_meta', tensor_meta)
            tensors[internal_var] = tensor
            
        
        return tensors
    
    def transform_to_internal(
        self,
        data: Dict[str, torch.Tensor],
        target_grid: Optional[Dict] = None,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Transform VIC data to RELIC internal format (shared ExternalAdapter._transform_fields)."""
        if kwargs:
            raise TypeError(f"Unsupported transform options: {sorted(kwargs)}")
        return self._transform_fields(data, target_grid=target_grid, variable_metadata=variable_metadata)

    def close(self) -> None:
        """Close all opened datasets"""
        self.logger.info("Closing VIC adapter")
        
        # Every dataset is attempted; the state is cleared and close errors are raised.
        self._close_datasets(self.discovered_files, self.grid_info)

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        time: float,
        dt: float,
        *,
        calendar_type: Optional[str] = None,
        time_origin: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VIC data provision at the simulation time.

        Returns every catalog field of the opened VIC data at time_origin + ``time`` in the
        simulator calendar through the single adapter read path (schema mapping, declared source
        transforms, unit conversion, then the configured time_interpolation);
        ``kwargs["variables"]`` restricts the internal variable names. The simulator passes
        ``calendar_type`` and ``time_origin`` from its settings; both are required.

        Returns:
            Dict[str, UpdateItem]: replace updates keyed by internal catalog variable names
        """
        # References: RELIC coupling adapter contract (relic/coupling/external/common/base_adapter.py,
        #   ExternalAdapter.read, _read_field and _read_records_at); CDEPS/E3SM shr_tInterp_getFactors
        #   (forcing_record_weights); no VIC process equation is evaluated here.
        # Mathematical expression: y_n(t) = sum_(k, w) w X_n[record k] with X_n = read(variables)[n]
        #   (X_n = convert(transform(D[m(n)])), m the schema mapping) and
        #   (k, w) = forcing_record_weights(s, t, rule), s the record times in seconds since time_origin.
        # Parameter meanings: D opened VIC dataset; m(n) external name of internal variable n;
        #   transform the declared source transform; convert the catalog unit conversion; rule the
        #   configured time_interpolation ('lower' or 'linear').
        # Units: catalog units of every internal variable (converted by the read path); time [s].
        # Domain/constraints: undeclared fields, missing units and unconvertible units raise in the
        #   read path; a missing time_origin, calendar_type or time_interpolation, a dataset calendar
        #   other than the simulator calendar and a time outside the records raise; no read
        #   position is kept.
        # Sign convention: not applicable: external data provision, no flux direction is introduced.
        return self._read_records_at(
            time,
            calendar_type=calendar_type,
            time_origin=time_origin,
            variables=kwargs.get("variables"),
        )

