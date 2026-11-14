"""
CMIP6 Data Adapter

This module implements the CMIP6Adapter for accessing and processing data from 
the Coupled Model Intercomparison Project Phase 6 (CMIP6) archive. It handles 
the CMIP6 Data Reference Syntax (DRS) directory structure, variable mapping,
and temporal data concatenation.

The adapter supports standard CMIP6 experiments and variables following 
CF-1.8 conventions and CMIP6 data standards.

References:
    - CMIP6 Data Reference Syntax: https://goo.gl/v1drZl
    - CF Conventions 1.8: http://cfconventions.org/cf-conventions/v1.8/
    - CMIP6 Global Attributes: https://goo.gl/kh31GE
    
Mathematical Expressions:
    Time concatenation: T_combined = ?[T_i] for files i=1..N
    Variable standardization: V_internal = f_map(V_cmip6, units_cmip6)
    Spatial domain: Omega = lat * lon [degrees]
    Temporal domain: T = [start_date, end_date] in CF time units
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr

from relic.utils.time_constants import SECONDS_PER_DAY

from ...scales.temporal.calendar import CalendarHandler
from ..common.base_adapter import validate_time_interpolation
from ..common import (
    DataSource,
    ExternalAdapter,
    IOHandler,
    RegridOperator,
    SpatialWindow,
    TimeHandler,
    TimeWindow,
)

# ProcessCategory removed - ExternalAdapter handles category internally

logger = logging.getLogger(__name__)

@dataclass
class CMIP6Config:
    """Configuration for CMIP6 adapter"""
    root_path: Union[str, Path]
    activity_id: str = "ScenarioMIP"  # CMIP, DAMIP, ScenarioMIP, etc.
    institution_id: Optional[str] = None  # e.g., "NCAR", "IPSL"
    source_id: Optional[str] = None  # e.g., "CESM2", "IPSL-CM6A-LR"
    experiment_id: str = "historical"  # historical, ssp245, ssp585, etc.
    variant_label: str = "r1i1p1f1"  # realization-initialization-physics-forcing
    table_id: str = "Amon"  # Amon, Lmon, Omon, etc.
    version: str = "latest"  # version or "latest"
    grid_label: Optional[str] = None  # gn, gr, gr1, etc.
    calendar_type: Optional[str] = None  # Explicit RELIC simulator calendar
    # Rule of a forcing read at the simulation time: 'lower' (t_k <= t < t_(k+1) holds record k)
    # or 'linear' (between the two records around t); required by forward, never assumed.
    time_interpolation: Optional[str] = None
    # Catalog variables the adapter writes (its declared outputs; each a mapping target of the
    # source in InternalSchema); None declares every mapping target of the source.
    provided_variables: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_time_interpolation(self.time_interpolation)

@dataclass 
class CMIP6FileInfo:
    """Information about a CMIP6 file"""
    filepath: Path
    variable_id: str
    activity_id: str
    institution_id: str
    source_id: str
    experiment_id: str
    variant_label: str
    table_id: str
    grid_label: str
    time_range: str
    version: str

    @classmethod
    def from_filepath(cls, filepath: Path) -> 'CMIP6FileInfo':
        """Parse CMIP6 file path according to DRS"""
        # CMIP6 DRS format:
        # variable_id_table_id_source_id_experiment_id_variant_label_grid_label_time_range.nc
        filename = filepath.stem
        parts = filename.split('_')
        
        if len(parts) < 7:
            raise ValueError(f"Invalid CMIP6 filename format: {filename}")
            
        variable_id = parts[0]
        table_id = parts[1] 
        source_id = parts[2]
        experiment_id = parts[3]
        variant_label = parts[4]
        grid_label = parts[5]
        time_range = parts[6]
        
        # Extract metadata from directory structure
        # /root/activity_id/institution_id/source_id/experiment_id/variant_label/table_id/variable_id/grid_label/version/
        path_parts = filepath.parts
        if len(path_parts) >= 10:
            activity_id = path_parts[-10]
            institution_id = path_parts[-9]
            version = path_parts[-2]
        else:
            # Fallback if directory structure is non-standard
            activity_id = "unknown"
            institution_id = "unknown"
            version = "unknown"
            
        return cls(
            filepath=filepath,
            variable_id=variable_id,
            activity_id=activity_id,
            institution_id=institution_id,
            source_id=source_id,
            experiment_id=experiment_id,
            variant_label=variant_label,
            table_id=table_id,
            grid_label=grid_label,
            time_range=time_range,
            version=version
        )


class CMIP6Adapter(ExternalAdapter):
    """
    Adapter for CMIP6 climate model data following CMIP6 DRS conventions.
    
    This adapter provides access to standardized climate model output from
    CMIP6 experiments, handling variable mapping, temporal concatenation,
    and spatial regridding to RELIC internal formats.
    
    The adapter automatically discovers and processes multi-file time series,
    applies CF-1.8 compliant metadata, and validates data quality.
    
    Attributes:
        config: CMIP6 configuration parameters
        discovered_files: List of discovered CMIP6 files
        current_datasets: Currently loaded datasets
    """
    
    def __init__(
        self,
        config: Optional[CMIP6Config] = None,
        time_window: Optional[TimeWindow] = None,
        spatial_window: Optional[SpatialWindow] = None,
        **kwargs
    ):
        """
        Initialize CMIP6 adapter.
        
        Parameters:
            config: CMIP6 configuration
            time_window: Temporal subset window 
            spatial_window: Spatial subset window
            **kwargs: Additional arguments passed to ExternalAdapter
        """
        self._require_adapter_config(config, CMIP6Config, "CMIP6Adapter")
        super().__init__(
            source=DataSource.CMIP6,
            name="CMIP6Adapter",
            provided_variables=None if config is None else config.provided_variables,
            **kwargs
        )

        if config is None:
            raise ValueError(
                "CMIP6Adapter needs a CMIP6Config declaring root_path; the data directory is never "
                "assumed (no current-directory default)."
            )

        self.config = config
        self.time_window = time_window
        self.spatial_window = spatial_window
        
        # Internal state
        self.discovered_files: Dict[str, List[CMIP6FileInfo]] = {}
        self.current_datasets: Dict[str, xr.Dataset] = {}
        
        # Initialize handlers
        self.time_handler = TimeHandler()
        self.regrid_operator = RegridOperator()
        self.io_handler = IOHandler()
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def discover(
        self, 
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover available CMIP6 files matching configuration.
        
        Scans the CMIP6 archive directory structure to find files matching
        the specified criteria. Groups files by variable for time series
        concatenation.
        
        Parameters:
            variables: List of CMIP6 variable names to discover
            **kwargs: Additional discovery parameters
            
        Returns:
            Dictionary containing discovered file information and metadata
            
        Raises:
            FileNotFoundError: If root path does not exist
            ValueError: If no matching files found
        """
        self.logger.info(f"Discovering CMIP6 files in {self.config.root_path}")
        
        root_path = Path(self.config.root_path)
        if not root_path.exists():
            raise FileNotFoundError(f"CMIP6 root path does not exist: {root_path}")
            
        # Build search pattern based on configuration
        pattern_parts = []
        
        # Add fixed components
        pattern_parts.extend([
            self.config.activity_id,
            self.config.institution_id or "*",
            self.config.source_id or "*", 
            self.config.experiment_id,
            self.config.variant_label,
            self.config.table_id
        ])
        
        # Add variable wildcard if not specified
        if variables:
            var_pattern = f"({('|'.join(variables))})"
        else:
            var_pattern = "*"
        pattern_parts.append(var_pattern)
        
        # Add remaining components
        pattern_parts.extend([
            self.config.grid_label or "*",
            self.config.version if self.config.version != "latest" else "*"
        ])
        
        search_pattern = "/".join(pattern_parts) + "/*.nc"
        search_path = root_path / search_pattern
        
        self.logger.debug(f"Search pattern: {search_path}")
        
        # Find matching files
        matching_files = list(root_path.glob(search_pattern))
        
        if not matching_files:
            raise ValueError(f"No CMIP6 files found matching pattern: {search_pattern}")
            
        # Parse and group files by variable
        discovered_files = {}
        
        for filepath in matching_files:
            file_info = CMIP6FileInfo.from_filepath(filepath)

            # Filter by requested variables
            if variables and file_info.variable_id not in variables:
                continue
                
            var_id = file_info.variable_id
            if var_id not in discovered_files:
                discovered_files[var_id] = []
                
            discovered_files[var_id].append(file_info)

                
        # Sort files by time range for each variable
        for var_id in discovered_files:
            discovered_files[var_id].sort(
                key=lambda x: x.time_range
            )
            
        self.discovered_files = discovered_files
        
        # Prepare discovery results
        results = {
            "source": "CMIP6",
            "total_files": sum(len(files) for files in discovered_files.values()),
            "variables": list(discovered_files.keys()),
            "file_groups": {
                var_id: len(files) for var_id, files in discovered_files.items()
            },
            "config": {
                "activity_id": self.config.activity_id,
                "institution_id": self.config.institution_id,
                "source_id": self.config.source_id,
                "experiment_id": self.config.experiment_id,
                "variant_label": self.config.variant_label,
                "table_id": self.config.table_id
            },
            "time_coverage": self._analyze_time_coverage(),
            "spatial_coverage": self._analyze_spatial_coverage()
        }
        
        self.logger.info(
            f"Discovered {results['total_files']} CMIP6 files for "
            f"{len(results['variables'])} variables"
        )
        
        return results
    
    def _analyze_time_coverage(self) -> Dict[str, Any]:
        """Analyze temporal coverage of discovered files"""
        if not self.discovered_files:
            return {}
            
        time_info = {}
        
        for var_id, files in self.discovered_files.items():
            time_ranges = []
            for file_info in files:
                # Parse time range from filename
                time_range = file_info.time_range
                if '-' in time_range:
                    start_str, end_str = time_range.split('-')
                    try:
                        # Handle different time range formats
                        if len(start_str) == 6:  # YYYYMM
                            start_date = pd.to_datetime(start_str, format='%Y%m')
                            end_date = pd.to_datetime(end_str, format='%Y%m')
                        elif len(start_str) == 8:  # YYYYMMDD
                            start_date = pd.to_datetime(start_str, format='%Y%m%d')
                            end_date = pd.to_datetime(end_str, format='%Y%m%d')
                        else:  # YYYY
                            start_date = pd.to_datetime(start_str, format='%Y')
                            end_date = pd.to_datetime(end_str, format='%Y')
                            
                        time_ranges.append((start_date, end_date))
                        
                    except ValueError:
                        self.logger.warning(f"Could not parse time range: {time_range}")
                        continue
                        
            if time_ranges:
                overall_start = min(start for start, _ in time_ranges)
                overall_end = max(end for _, end in time_ranges)
                
                time_info[var_id] = {
                    "start": overall_start.isoformat(),
                    "end": overall_end.isoformat(),
                    "n_files": len(files)
                }
                
        return time_info
    
    def _analyze_spatial_coverage(self) -> Dict[str, Any]:
        """Analyze spatial coverage by examining first file of each variable"""
        if not self.discovered_files:
            return {}
            
        spatial_info = {}
        
        for var_id, files in self.discovered_files.items():
            if not files:
                continue
                
            # Check first file for spatial dimensions
            first_file = files[0].filepath
            with xr.open_dataset(first_file, decode_times=False) as ds:
                if var_id in ds.data_vars:
                    var_data = ds[var_id]

                    # Extract spatial coordinates
                    spatial_dims = {}
                    for coord_name in ['lat', 'latitude', 'lon', 'longitude']:
                        if coord_name in var_data.coords:
                            coord = var_data.coords[coord_name]
                            spatial_dims[coord_name] = {
                                "size": coord.size,
                                "min": float(coord.min()),
                                "max": float(coord.max())
                            }

                    spatial_info[var_id] = spatial_dims

                
        return spatial_info
    
    def open(self, **kwargs) -> Dict[str, Any]:
        """
        Open and load discovered CMIP6 datasets.
        
        Opens all discovered files for each variable and concatenates them 
        along the time dimension to create continuous time series.
        
        Parameters:
            **kwargs: Additional opening parameters
            
        Returns:
            Dictionary containing opened datasets and metadata
            
        Raises:
            ValueError: If no files have been discovered
            IOError: If files cannot be opened
        """
        if not self.discovered_files:
            raise ValueError("No CMIP6 files discovered. Call discover() first.")
            
        self.logger.info("Opening CMIP6 datasets")
        
        opened_datasets = {}
        
        for var_id, files in self.discovered_files.items():
            self.logger.debug(f"Opening {len(files)} files for variable {var_id}")
            
            try:
                # Open all files for this variable
                file_datasets = []
                
                for file_info in files:
                    ds = xr.open_dataset(
                        file_info.filepath, 
                        decode_times=True,
                        decode_cf=True,
                        **kwargs
                    )
                    
                    # Add file metadata
                    ds.attrs['source_file'] = str(file_info.filepath)
                    ds.attrs['cmip6_variant_label'] = file_info.variant_label
                    ds.attrs['cmip6_table_id'] = file_info.table_id
                    
                    file_datasets.append(ds)
                    
                # Concatenate along time dimension
                if len(file_datasets) > 1:
                    # Check for overlapping times
                    combined_ds = xr.concat(
                        file_datasets,
                        dim='time', 
                        data_vars='minimal',
                        coords='minimal',
                        combine_attrs='override'
                    )
                    
                    # Remove duplicate times
                    _, unique_indices = np.unique(
                        combined_ds.time.values, 
                        return_index=True
                    )
                    combined_ds = combined_ds.isel(time=sorted(unique_indices))
                    
                else:
                    combined_ds = file_datasets[0]
                    
                # Apply temporal subsetting if specified
                if self.time_window:
                    combined_ds = self._apply_temporal_subset(combined_ds)
                    
                # Apply spatial subsetting if specified
                if self.spatial_window:
                    combined_ds = self._apply_spatial_subset(combined_ds)
                    
                opened_datasets[var_id] = combined_ds
                
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                self.logger.error(f"Failed to open datasets for {var_id}: {e}")
                raise IOError(f"Could not open CMIP6 data for variable {var_id}: {e}")
                
        self.current_datasets = opened_datasets
        
        return {
            "source": "CMIP6",
            "datasets": list(opened_datasets.keys()),
            "temporal_coverage": self._get_temporal_coverage(opened_datasets),
            "spatial_resolution": self._get_spatial_resolution(opened_datasets)
        }
    
    def _apply_temporal_subset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply temporal subsetting based on time window"""
        if not self.time_window:
            return ds
            
        return ds.sel(time=slice(self.time_window.start, self.time_window.end))
    
    def _apply_spatial_subset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply spatial subsetting based on spatial window"""
        if not self.spatial_window:
            return ds
            
        # Handle different latitude/longitude coordinate names
        lat_coords = ['lat', 'latitude', 'y']
        lon_coords = ['lon', 'longitude', 'x']
        
        subset_dict = {}
        
        for coord_name in lat_coords:
            if coord_name in ds.coords:
                subset_dict[coord_name] = slice(
                    self.spatial_window.lat_min,
                    self.spatial_window.lat_max
                )
                break
                
        for coord_name in lon_coords:
            if coord_name in ds.coords:
                subset_dict[coord_name] = slice(
                    self.spatial_window.lon_min, 
                    self.spatial_window.lon_max
                )
                break
                
        if subset_dict:
            return ds.sel(**subset_dict)
        else:
            self.logger.warning("Could not find spatial coordinates for subsetting")
            return ds
    
    def _get_temporal_coverage(self, datasets: Dict[str, xr.Dataset]) -> Dict[str, Dict]:
        """Get temporal coverage information for opened datasets"""
        coverage = {}
        
        for var_id, ds in datasets.items():
            if 'time' in ds.coords:
                time_coord = ds.coords['time']
                coverage[var_id] = {
                    "start": str(time_coord.values[0]),
                    "end": str(time_coord.values[-1]),
                    "n_timesteps": len(time_coord),
                    "frequency": self._infer_frequency(time_coord)
                }
                
        return coverage
    
    def _get_spatial_resolution(self, datasets: Dict[str, xr.Dataset]) -> Dict[str, Dict]:
        """Get spatial resolution information for opened datasets"""
        resolution = {}
        
        for var_id, ds in datasets.items():
            spatial_info = {}
            
            # Check latitude resolution
            for lat_name in ['lat', 'latitude']:
                if lat_name in ds.coords:
                    lat = ds.coords[lat_name]
                    if lat.size > 1:
                        lat_res = abs(float(lat[1] - lat[0]))
                        spatial_info['lat_resolution'] = lat_res
                        spatial_info['n_lat'] = lat.size
                    break
                    
            # Check longitude resolution  
            for lon_name in ['lon', 'longitude']:
                if lon_name in ds.coords:
                    lon = ds.coords[lon_name]
                    if lon.size > 1:
                        lon_res = abs(float(lon[1] - lon[0]))
                        spatial_info['lon_resolution'] = lon_res
                        spatial_info['n_lon'] = lon.size
                    break
                    
            resolution[var_id] = spatial_info
            
        return resolution
    
    def _infer_frequency(self, time_coord: xr.DataArray) -> str:
        """Infer frequency using the explicit CF/simulator calendar.

        References: CF Conventions 1.8 Section 4.4 and CMIP6 time axes.
        Mathematical expression: Delta t_min = min_i(t_(i+1) - t_i), and an
        annual interval satisfies Delta t_min >= Y(C).
        Parameter meanings: C is the resolved calendar and Y(C) is its exact
        ParameterManager-backed year length in days.
        Units: coordinate differences and Y(C) are converted to SI seconds.
        Domain/constraints: C must be explicit, supported, and consistent;
        coordinate differences must be strictly positive.
        Sign convention: time increases forward and every Delta t is positive.
        """
        coordinate_calendar = (
            time_coord.attrs.get("calendar")
            or time_coord.encoding.get("calendar")
            or getattr(time_coord.to_index(), "calendar", None)
        )
        calendar_type = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self.config.calendar_type,
            runtime_calendar=coordinate_calendar,
            owner_name="CMIP6Adapter._infer_frequency",
        )
        if len(time_coord) < 2:
            return "unknown"

        time_diffs = np.diff(time_coord.values)
        if len(time_diffs) == 0:
            return "unknown"
        resolved_diff_seconds = []
        for delta in time_diffs:
            candidate: Any = pd.Timedelta(delta)
            if pd.isna(candidate):
                raise ValueError("CMIP6 time differences must be finite")
            seconds = float(candidate.total_seconds())
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise ValueError(
                    "CMIP6 time coordinates must have finite positive differences"
                )
            resolved_diff_seconds.append(seconds)

        minimum_diff_seconds = min(resolved_diff_seconds)
        seconds_per_day = float(SECONDS_PER_DAY)
        year_duration_seconds = (
            CalendarHandler.days_per_simulator_year(calendar_type)
            * seconds_per_day
        )
        if minimum_diff_seconds >= year_duration_seconds:
            return "yearly"
        if minimum_diff_seconds >= 28 * seconds_per_day:
            return "monthly"
        if minimum_diff_seconds >= seconds_per_day:
            return "daily"
        return "sub-daily"
    
    def read(
        self, 
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Read and transform CMIP6 data to RELIC internal format.
        
        Reads the requested variables from opened datasets, applies variable
        mapping, unit conversions, and returns data as PyTorch tensors.
        
        Parameters:
            variables: List of variables to read (internal names)
            **kwargs: Additional reading parameters
            
        Returns:
            Dictionary mapping internal variable names to PyTorch tensors
            
        Raises:
            ValueError: If datasets are not opened or variables not available
        """
        if not self.current_datasets:
            raise ValueError("No datasets opened. Call open() first.")
            
        # internal -> CMIP6 through the schema owner (ambiguous many-to-one mappings raise unless
        # a canonical external variable is declared); None requests every mapped internal name.
        from ..common.schema import InternalSchema
        internal_to_cmip6 = InternalSchema().resolve_external_names(DataSource.CMIP6, variables)
        # Mapped fields each opened dataset contains; a requested field that no dataset has raises
        # and absent mapped fields are listed in last_read_report (never skipped silently).
        present = self._present_mapped_fields(self.current_datasets, internal_to_cmip6, variables)
        found = list(dict.fromkeys(name for names in present.values() for name in names))

        self.logger.info(f"Reading CMIP6 variables: {found}")

        tensors = {}

        for internal_var in found:
            cmip6_var = internal_to_cmip6[internal_var]
            ds = self.current_datasets[next(key for key, names in present.items() if internal_var in names)]

            # Extract data array
            # Declared source transform (if any) and unit conversion: the one read path.
            converted_data = self._read_field(ds, cmip6_var, internal_var)

            # Convert to PyTorch tensor
            tensor = torch.from_numpy(converted_data.values.astype(np.float32))

            # Add metadata
            tensor_meta = {
                'shape': tensor.shape,
                'dims': converted_data.dims,
                'coords': {
                    name: torch.from_numpy(coord.values.astype(np.float32))
                    for name, coord in converted_data.coords.items()
                    if coord.dtype.kind in 'fc'  # numeric types only
                },
                'coord_dims': {
                    name: tuple(coord.dims)
                    for name, coord in converted_data.coords.items()
                    if coord.dtype.kind in 'fc'
                },
                'attrs': dict(converted_data.attrs),
                'units': converted_data.attrs['units'],
                'variable': internal_var,
                **self._record_time_metadata(converted_data),
                'cell_bounds': self._cf_horizontal_coordinates(ds, converted_data),
            }
            
            # Store tensor with metadata
            tensors[internal_var] = tensor
            setattr(tensor, '_relic_meta', tensor_meta)

            self.logger.debug(f"Read {internal_var}: {tensor.shape}")
            
                
        return tensors
    
    def transform_to_internal(
        self,
        data: Dict[str, torch.Tensor],
        target_grid: Optional[Dict] = None,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Transform CMIP6 data to RELIC internal format (shared ExternalAdapter._transform_fields)."""
        if kwargs:
            raise TypeError(f"Unsupported transform options: {sorted(kwargs)}")
        return self._transform_fields(data, target_grid=target_grid, variable_metadata=variable_metadata)

    def close(self) -> None:
        """Close all opened datasets and clean up resources"""
        self.logger.info("Closing CMIP6 adapter")
        
        # Every dataset is attempted; the state is cleared and close errors are raised.
        self._close_datasets(self.discovered_files)

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
        Forward pass for CMIP6 data provision at the simulation time.

        Returns every catalog field of the opened CMIP6 data at time_origin + ``time`` in the
        simulator calendar through the single adapter read path (schema mapping, declared source
        transforms, unit conversion, then the configured time_interpolation);
        ``kwargs["variables"]`` restricts the internal variable names. The simulator passes
        ``calendar_type`` and ``time_origin`` from its settings; both are required.

        Returns:
            Dict[str, UpdateItem]: replace updates keyed by internal catalog variable names
        """
        # References: RELIC coupling adapter contract (relic/coupling/external/common/base_adapter.py,
        #   ExternalAdapter.read, _read_field and _read_records_at); CDEPS/E3SM shr_tInterp_getFactors
        #   (forcing_record_weights); no CMIP6 process equation is evaluated here.
        # Mathematical expression: y_n(t) = sum_(k, w) w X_n[record k] with X_n = read(variables)[n]
        #   (X_n = convert(transform(D[m(n)])), m the schema mapping) and
        #   (k, w) = forcing_record_weights(s, t, rule), s the record times in seconds since time_origin.
        # Parameter meanings: D opened CMIP6 dataset; m(n) external name of internal variable n;
        #   transform the declared source transform; convert the catalog unit conversion; rule the
        #   configured time_interpolation ('lower' or 'linear').
        # Units: catalog units of every internal variable (converted by the read path); time [s].
        # Domain/constraints: undeclared fields, missing units and unconvertible units raise in the
        #   read path; a missing time_origin, calendar_type or time_interpolation, a dataset calendar
        #   other than the simulator calendar and a time outside the records raise; no read
        #   position is kept.
        # Sign convention: not applicable: external data provision, no flux direction is introduced.
        # The configured calendar_type and the simulator's must agree (one calendar, never converted).
        calendar = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self.config.calendar_type,
            runtime_calendar=calendar_type,
            owner_name=self.name,
        )
        return self._read_records_at(
            time,
            calendar_type=calendar,
            time_origin=time_origin,
            variables=kwargs.get("variables"),
        )
