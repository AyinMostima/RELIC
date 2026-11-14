"""
WRF Model Data Adapter

This module implements the WRFAdapter for accessing and processing data from 
the Weather Research and Forecasting (WRF) model output files. It handles 
WRF-specific variable naming, coordinate systems, and temporal organization.

The adapter supports standard WRF output formats including wrfout files
and follows CF-1.8 conventions where possible.

References:
    - WRF User's Guide: https://www2.mmm.ucar.edu/wrf/users/
    - WRF-Python Documentation: https://wrf-python.readthedocs.io/
    - CF Conventions 1.8: http://cfconventions.org/cf-conventions/v1.8/
    
Mathematical Expressions:
    The read path of the mapped fields (ExternalAdapter.mapping_expression(DataSource.WRF)): the
    schema mapping, catalog unit conversion and the declared source transforms of WRF:
    de-accumulation of RAINC, RAINNC and PRCP_ACC, layer water mass of SMOIS, specific humidity
    q = w / (1 + w) of the 2 m mixing ratio Q2, and the earth-relative rotation of U10 and V10.
    open() adds no derived fields to the opened datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
import logging
from datetime import datetime
from dataclasses import dataclass
import re

import torch
import xarray as xr
import numpy as np
import pandas as pd

from ..common.base_adapter import validate_time_interpolation
from ..common import (
    ExternalAdapter, DataSource,
    TimeWindow, SpatialWindow,
    TimeHandler, RegridOperator,
    IOHandler
)
from ....core.functor import (
    ConflictResolutionPolicy,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from ....parameters.namespaces import ParameterNamespace, resolve_namespace

logger = logging.getLogger(__name__)

@dataclass
class WRFConfig:
    """Configuration for WRF adapter"""
    root_path: Union[str, Path]
    file_pattern: str = "wrfout_d*"  # File pattern to match WRF output files
    domain: Optional[int] = None  # Specific domain number (e.g., 1, 2, 3)
    vertical_levels: Optional[List[int]] = None  # Specific vertical levels to extract
    unstagger_grids: bool = True  # Whether to unstagger C-grid variables
    # WRF timestamp format of the Times variable; None takes the declared parameter
    # coupling.external.wrf.time_format (an explicit value here is kept).
    time_format: Optional[str] = None
    # Rule of a forcing read at the simulation time: 'lower' (t_k <= t < t_(k+1) holds record k)
    # or 'linear' (between the two records around t); required by forward, never assumed.
    time_interpolation: Optional[str] = None
    # Catalog variables the adapter writes (its declared outputs; each a mapping target of the
    # source in InternalSchema); None declares every mapping target of the source.
    provided_variables: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_time_interpolation(self.time_interpolation)



class WRFAdapter(ExternalAdapter):
    # wrfout records run along the 'Time' dimension.
    TIME_DIMENSION = "Time"

    """
    Adapter for Weather Research and Forecasting (WRF) model output.
    
    This adapter provides access to WRF model output files, handling the
    model's specific coordinate system, variable naming conventions, and
    temporal organization. It automatically detects domains, processes
    staggered grids, and computes diagnostic variables.
    
    Key Features:
    - Automatic domain detection and processing
    - Staggered grid handling (Arakawa C-grid)
    - Pressure level interpolation
    - Diagnostic variable computation
    - CF-1.8 compliant metadata generation
    
    Attributes:
        config: WRF configuration parameters
        discovered_files: List of discovered WRF files
        current_datasets: Currently loaded datasets
        domain_info: Information about WRF domains
    """
    
    def __init__(
        self,
        config: Optional[WRFConfig] = None,
        time_window: Optional[TimeWindow] = None,
        spatial_window: Optional[SpatialWindow] = None,
        regrid_operator: Optional[RegridOperator] = None,
        parameter_manager: Optional[Any] = None,
        parameter_manager_resolver: Optional[Callable[[], Any]] = None,
        **kwargs
    ):
        """
        Initialize WRF adapter.
        
        Parameters:
            config: WRF configuration
            time_window: Temporal subset window 
            spatial_window: Spatial subset window
            regrid_operator: Optional injected regrid operator
            **kwargs: Additional arguments passed to ExternalAdapter
        """
        self._require_adapter_config(config, WRFConfig, "WRFAdapter")
        if parameter_manager is not None:
            pm = parameter_manager
        elif parameter_manager_resolver is not None:
            pm = parameter_manager_resolver()
        else:
            pm = None

        provided_variables = None if config is None else config.provided_variables
        metadata = FunctorMetadata(
            name="WRFAdapter",
            version="1.0.0",
            description="External adapter for WRF outputs with CF-1.8 alignment and destaggered grids",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            # The read path of the mapped fields, generated from the schema mapping and transforms.
            mathematical_expression=ExternalAdapter.mapping_expression(DataSource.WRF),
            references=[
                "WRF User's Guide (NCAR/MMM)",
                "WRF Registry.EM_COMMON (Q2 'QV at 2 M' kg kg-1; U10 'U at 10 M'; V10 'V at 10 M')",
                "UCAR MMM WRF & MPAS-A Support Forum: converting model grid-relative wind to earth-relative wind",
                "CF Conventions 1.8"
            ],
            parameters={
                "root_path": "Root path to WRF outputs (path)",
                "vertical_levels": "Selected sigma/pressure levels (index)",
                "unstagger_grids": "Flag for destaggering C-grid variables (bool)",
            },
            # An adapter reads no state; its outputs are the provided catalog variables.
            input_variables=[],
            output_variables=ExternalAdapter.declared_outputs(DataSource.WRF, provided_variables),
            conservation_laws=[],
            spatial_dimensions=3,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["external_atmosphere"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FORCING,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )

        super().__init__(
            source=DataSource.WRF,
            name="WRFAdapter",
            model_id="wrf",
            parameter_manager=pm,
            metadata=metadata,
            provided_variables=provided_variables,
            **kwargs,
        )
        
        if config is None:
            # The declared default output directory (coupling.external.wrf.default_root_path); a
            # failed or missing lookup raises.
            default_root = self.get_parameter_value(
                "default_root_path",
                namespace=resolve_namespace(ParameterNamespace.COUPLING_EXTERNAL_WRF),
                expected_type=str,
                validate=True,
                required=True,
            )
            config = WRFConfig(root_path=Path(default_root))
        self.config = config
        self.time_window = time_window
        self.spatial_window = spatial_window
        
        # Internal state
        self.discovered_files: List[Path] = []
        self.current_datasets: Dict[str, xr.Dataset] = {}
        self.domain_info: Dict[int, Dict] = {}
        
        # Initialize handlers
        self.time_handler = TimeHandler()
        self.regrid_operator = regrid_operator or RegridOperator()
        self.io_handler = IOHandler()
        self._apply_time_format_override(pm)

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Conversion-context defaults for missing metadata
        conversion_namespace = self._get_conversion_namespace("wrf")
        param_specs = {
            "wrf_west_east_count": {"units": "dimensionless", "cast_to": int},
            "wrf_south_north_count": {"units": "dimensionless", "cast_to": int},
            "wrf_bottom_top_count": {"units": "dimensionless", "cast_to": int},
        }
        params = self._load_conversion_parameters(param_specs, default_namespace=conversion_namespace)
        self.wrf_west_east_default = params["wrf_west_east_count"]
        self.wrf_south_north_default = params["wrf_south_north_count"]
        self.wrf_bottom_top_default = params["wrf_bottom_top_count"]

    def _apply_time_format_override(self, parameter_manager: Optional[Any]) -> None:
        """Resolve the Times format: the configured one, else the declared parameter (raises when absent)."""
        _ = parameter_manager
        if self.config.time_format is None:
            self.config.time_format = self.get_parameter_value(
                "time_format",
                namespace=resolve_namespace(ParameterNamespace.COUPLING_EXTERNAL_WRF),
                expected_type=str,
                validate=True,
                required=True,
            )

    def discover(
        self, 
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover available WRF output files.
        
        Scans the specified directory for WRF output files matching the
        configuration pattern. Analyzes domains, time coverage, and
        available variables.
        
        Parameters:
            variables: List of WRF variable names to check for
            **kwargs: Additional discovery parameters
            
        Returns:
            Dictionary containing discovered file information and metadata
            
        Raises:
            FileNotFoundError: If root path does not exist
            ValueError: If no matching files found
        """
        self.logger.info(f"Discovering WRF files in {self.config.root_path}")
        
        root_path = Path(self.config.root_path)
        if not root_path.exists():
            raise FileNotFoundError(f"WRF root path does not exist: {root_path}")
            
        # Find WRF output files
        file_pattern = self.config.file_pattern
        if self.config.domain:
            file_pattern = file_pattern.replace("*", f"{self.config.domain:02d}*")
            
        matching_files = list(root_path.glob(file_pattern))
        
        if not matching_files:
            raise ValueError(f"No WRF files found matching pattern: {file_pattern}")

        # Files are ordered by the record times they contain (their parsed Times), never by
        # file name or modification time.
        file_times = self._read_file_times(matching_files)
        matching_files = sorted(matching_files, key=lambda path: file_times[path][0])

        self.discovered_files = matching_files

        # Analyze file contents
        file_analysis = self._analyze_wrf_files(matching_files, file_times, variables)
        
        results = {
            "source": "WRF",
            "total_files": len(matching_files),
            "domains": list(file_analysis["domains"].keys()),
            "time_coverage": file_analysis["time_coverage"],
            "available_variables": file_analysis["variables"],
            "spatial_resolution": file_analysis["spatial_resolution"],
            "vertical_levels": file_analysis["vertical_levels"]
        }
        
        self.domain_info = file_analysis["domains"]
        
        self.logger.info(
            f"Discovered {results['total_files']} WRF files across "
            f"{len(results['domains'])} domains"
        )
        
        return results
    
    def _parse_wrf_times(self, ds: xr.Dataset, source: str) -> pd.DatetimeIndex:
        """
        Record times of one WRF output from its Times character array.

        Every entry must match ``config.time_format`` and the records must be strictly
        increasing. A missing Times variable or an unparsable entry raises; no file-name or
        modification-time substitute is used.
        """
        if 'Times' not in ds.variables:
            raise ValueError(
                f"{source}: WRF output has no Times variable; its records cannot be placed in time."
            )
        parsed = []
        for entry in np.asarray(ds['Times'].values):
            text = ''.join(np.asarray(entry).astype(str).ravel())
            try:
                parsed.append(datetime.strptime(text, self.config.time_format))
            except ValueError as exc:
                raise ValueError(
                    f"{source}: WRF Times entry {text!r} does not match {self.config.time_format!r}."
                ) from exc
        times = pd.DatetimeIndex(parsed)
        if times.size == 0:
            raise ValueError(f"{source}: WRF Times has no records.")
        if times.size > 1 and not bool((times[1:] > times[:-1]).all()):
            raise ValueError(
                f"{source}: WRF Times are not strictly increasing: {[str(t) for t in times]}."
            )
        return times

    def _read_file_times(self, files: List[Path]) -> Dict[Path, pd.DatetimeIndex]:
        """Parsed record times of every WRF file (see _parse_wrf_times)."""
        file_times: Dict[Path, pd.DatetimeIndex] = {}
        for path in files:
            with xr.open_dataset(path, engine='netcdf4') as ds:
                file_times[path] = self._parse_wrf_times(ds, path.name)
        return file_times

    def _analyze_wrf_files(
        self,
        files: List[Path],
        file_times: Dict[Path, pd.DatetimeIndex],
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze WRF files: domain and variables from the first file, time coverage from the parsed Times."""

        domains = {}
        all_variables = set()
        time_range = {'start': None, 'end': None}
        spatial_res = {}
        vertical_levels = None

        # Sample first file for detailed analysis (open failures propagate)
        if files:
            with xr.open_dataset(files[0], engine='netcdf4') as sample_ds:
                # Extract domain information
                domain_id = getattr(sample_ds, 'GRID_ID', 1)

                # Spatial resolution
                dx = getattr(sample_ds, 'DX', None)
                dy = getattr(sample_ds, 'DY', None)

                # Domain size
                west_east = sample_ds.dims.get('west_east', self.wrf_west_east_default)
                south_north = sample_ds.dims.get('south_north', self.wrf_south_north_default)
                bottom_top = sample_ds.dims.get('bottom_top', self.wrf_bottom_top_default)

                domains[domain_id] = {
                    'dx': dx,
                    'dy': dy,
                    'west_east': west_east,
                    'south_north': south_north,
                    'bottom_top': bottom_top
                }

                spatial_res[domain_id] = {
                    'dx_m': dx,
                    'dy_m': dy,
                    'nx': west_east,
                    'ny': south_north,
                    'nz': bottom_top
                }

                # Available variables
                if variables:
                    # Check if requested variables exist
                    available_vars = [var for var in variables if var in sample_ds.data_vars]
                    all_variables.update(available_vars)
                else:
                    # Get all 4D variables (time-dependent)
                    for var_name, var_data in sample_ds.data_vars.items():
                        if 'Time' in var_data.dims or 'time' in var_data.dims:
                            all_variables.add(var_name)

                # Vertical levels
                if 'bottom_top' in sample_ds.dims:
                    vertical_levels = sample_ds.dims['bottom_top']

            # Time coverage from the parsed Times of the files
            time_range['start'] = min(file_times[path][0] for path in files).isoformat()
            time_range['end'] = max(file_times[path][-1] for path in files).isoformat()

        return {
            'domains': domains,
            'variables': sorted(list(all_variables)),
            'time_coverage': time_range,
            'spatial_resolution': spatial_res,
            'vertical_levels': vertical_levels
        }

    def open(self, **kwargs) -> Dict[str, Any]:
        """
        Open and load discovered WRF datasets.
        
        Opens WRF files and processes them according to configuration.
        Handles grid unstaggering, coordinate transformation, and
        temporal concatenation.
        
        Parameters:
            **kwargs: Additional opening parameters
            
        Returns:
            Dictionary containing opened datasets and metadata
            
        Raises:
            ValueError: If no files have been discovered
            IOError: If files cannot be opened
        """
        if not self.discovered_files:
            raise ValueError("No WRF files discovered. Call discover() first.")
            
        self.logger.info(f"Opening {len(self.discovered_files)} WRF files")
        
        # Group files by domain
        domain_files = self._group_files_by_domain()
        
        opened_datasets = {}
        
        for domain_id, files in domain_files.items():
            self.logger.debug(f"Opening {len(files)} files for domain {domain_id}")
            
            try:
                # Open and concatenate files
                datasets = []
                
                for filepath in files:
                    ds = xr.open_dataset(
                        filepath,
                        engine='netcdf4',
                        **kwargs
                    )
                    
                    # Process coordinates and time
                    ds = self._process_wrf_coordinates(ds)
                    
                    # Unstagger grids if requested
                    if self.config.unstagger_grids:
                        ds = self._unstagger_grids(ds)
                        
                    datasets.append(ds)
                
                # Concatenate the records along the WRF Time dimension; the files arrive in
                # the order of their parsed Times and the records must be strictly increasing.
                if len(datasets) > 1:
                    combined_ds = xr.concat(
                        datasets,
                        dim='Time',
                        data_vars='minimal',
                        coords='minimal',)
                else:
                    combined_ds = datasets[0]
                record_times = pd.DatetimeIndex(combined_ds['time'].values)
                if record_times.size > 1 and not bool((record_times[1:] > record_times[:-1]).all()):
                    raise ValueError(
                        f"WRF domain {domain_id}: record times are not strictly increasing across "
                        f"{[path.name for path in files]}: {[str(t) for t in record_times]}."
                    )
                
                # Apply temporal and spatial subsetting
                combined_ds = self._apply_subsetting(combined_ds)

                opened_datasets[f"domain_{domain_id}"] = combined_ds
                
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                self.logger.error(f"Failed to open datasets for domain {domain_id}: {e}")
                raise IOError(f"Could not open WRF data for domain {domain_id}: {e}")
        
        self.current_datasets = opened_datasets
        
        return {
            "source": "WRF",
            "domains": list(opened_datasets.keys()),
            "temporal_coverage": self._get_temporal_coverage(),
            "variables": self._get_available_variables()
        }
    
    def _group_files_by_domain(self) -> Dict[int, List[Path]]:
        """Group files by WRF domain"""
        domain_files = {}
        
        for filepath in self.discovered_files:
            # Extract domain from filename
            filename = filepath.name
            match = re.search(r'_d(\d{2})_', filename)
            if match:
                domain_id = int(match.group(1))
            else:
                domain_id = 1  # Default domain
            
            if domain_id not in domain_files:
                domain_files[domain_id] = []
            domain_files[domain_id].append(filepath)
        
        return domain_files
    
    def _process_wrf_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        """Process WRF coordinate metadata without a dataset CF claim."""
        
        # WRF record times: the Times character array parsed with config.time_format (an
        # unparsable or missing Times raises).
        source = str(ds.encoding.get('source', 'WRF output'))
        ds = ds.assign_coords(time=('Time', self._parse_wrf_times(ds, source)))
        ds['time'].attrs.update({
            'standard_name': 'time',
            'long_name': 'Time',
            'calendar': 'gregorian'
        })
        
        # Process spatial coordinates if needed
        if 'XLAT' in ds.variables and 'XLONG' in ds.variables:
            # Add more descriptive coordinate attributes
            ds['XLAT'].attrs.update({
                'standard_name': 'latitude',
                'long_name': 'Latitude',
                'units': 'degrees_north'
            })
            
            ds['XLONG'].attrs.update({
                'standard_name': 'longitude', 
                'long_name': 'Longitude',
                'units': 'degrees_east'
            })
        
        return ds
    
    def _unstagger_grids(self, ds: xr.Dataset) -> xr.Dataset:
        """Unstagger Arakawa C-grid variables"""
        
        # Variables that are staggered in x-direction (U-points)
        u_vars = ['U', 'U10']  
        
        # Variables that are staggered in y-direction (V-points)
        v_vars = ['V', 'V10']
        
        # Variables that are staggered in z-direction (W-points)
        w_vars = ['W', 'PH', 'PHB']
        
        unstaggered_ds = ds.copy()
        
        # Unstagger U variables
        for var_name in u_vars:
            if var_name in ds.data_vars:
                var_data = ds[var_name]
                if 'west_east_stag' in var_data.dims:
                    # Average adjacent points
                    unstaggered = 0.5 * (
                        var_data.isel(west_east_stag=slice(None, -1)) + 
                        var_data.isel(west_east_stag=slice(1, None))
                    )
                    # Update coordinate
                    unstaggered = unstaggered.rename({'west_east_stag': 'west_east'})
                    unstaggered_ds[var_name] = unstaggered
        
        # Unstagger V variables
        for var_name in v_vars:
            if var_name in ds.data_vars:
                var_data = ds[var_name]
                if 'south_north_stag' in var_data.dims:
                    # Average adjacent points
                    unstaggered = 0.5 * (
                        var_data.isel(south_north_stag=slice(None, -1)) + 
                        var_data.isel(south_north_stag=slice(1, None))
                    )
                    # Update coordinate
                    unstaggered = unstaggered.rename({'south_north_stag': 'south_north'})
                    unstaggered_ds[var_name] = unstaggered
        
        # Unstagger W variables
        for var_name in w_vars:
            if var_name in ds.data_vars:
                var_data = ds[var_name]
                if 'bottom_top_stag' in var_data.dims:
                    # Average adjacent points
                    unstaggered = 0.5 * (
                        var_data.isel(bottom_top_stag=slice(None, -1)) + 
                        var_data.isel(bottom_top_stag=slice(1, None))
                    )
                    # Update coordinate
                    unstaggered = unstaggered.rename({'bottom_top_stag': 'bottom_top'})
                    unstaggered_ds[var_name] = unstaggered
        
        return unstaggered_ds
    
    def _apply_subsetting(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply temporal and spatial subsetting"""
        
        # Apply temporal subsetting
        if self.time_window and 'time' in ds.coords:
            ds = ds.sel(time=slice(self.time_window.start, self.time_window.end))
        
        # Apply spatial subsetting
        if self.spatial_window:
            # WRF uses irregular grids, so we need to find grid points within bounds
            if 'XLAT' in ds.variables and 'XLONG' in ds.variables:
                lat = ds['XLAT']
                lon = ds['XLONG']
                
                # Find indices within spatial window
                mask = (
                    (lat >= self.spatial_window.lat_min) &
                    (lat <= self.spatial_window.lat_max) &
                    (lon >= self.spatial_window.lon_min) &
                    (lon <= self.spatial_window.lon_max)
                )
                
                # Apply mask to find bounding box
                valid_indices = np.where(mask.values)
                if len(valid_indices[0]) > 0 and len(valid_indices[1]) > 0:
                    y_min, y_max = valid_indices[0].min(), valid_indices[0].max() + 1
                    x_min, x_max = valid_indices[1].min(), valid_indices[1].max() + 1
                    
                    ds = ds.isel(
                        south_north=slice(y_min, y_max),
                        west_east=slice(x_min, x_max)
                    )
        
        return ds
    
    def _get_temporal_coverage(self) -> Dict[str, Dict]:
        """Get temporal coverage for opened datasets"""
        coverage = {}
        
        for domain_name, ds in self.current_datasets.items():
            if 'time' in ds.coords:
                time_coord = ds.coords['time']
                coverage[domain_name] = {
                    "start": str(time_coord.values[0]),
                    "end": str(time_coord.values[-1]),
                    "n_timesteps": len(time_coord)
                }
        
        return coverage
    
    def _get_available_variables(self) -> Dict[str, List[str]]:
        """Get available variables for opened datasets"""
        variables = {}
        
        for domain_name, ds in self.current_datasets.items():
            variables[domain_name] = list(ds.data_vars.keys())
        
        return variables
    
    def read(
        self, 
        variables: Optional[List[str]] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Read and transform WRF data to RELIC internal format.
        
        Parameters:
            variables: List of variables to read (internal names)
            domain: Specific domain to read from
            **kwargs: Additional reading parameters
            
        Returns:
            Dictionary mapping internal variable names to PyTorch tensors
        """
        if not self.current_datasets:
            raise ValueError("No datasets opened. Call open() first.")
            
        # Select domain(s) to read from
        if domain:
            datasets_to_read = {domain: self.current_datasets[domain]}
        else:
            datasets_to_read = self.current_datasets
            
        self.logger.info(f"Reading WRF variables from {len(datasets_to_read)} domains")
        
        # Get variable mappings
        from ..common.schema import InternalSchema
        # internal -> WRF through the schema owner (ambiguous many-to-one mappings raise
        # unless a canonical external variable is declared).
        internal_to_wrf = InternalSchema().resolve_external_names(DataSource.WRF, variables)
        
        tensors = {}
        
        # Mapped fields each dataset contains; a requested field that no dataset has raises
        # and absent mapped fields are listed in last_read_report (never skipped silently).
        present = self._present_mapped_fields(datasets_to_read, internal_to_wrf, variables)
        for domain_name, ds in datasets_to_read.items():
            domain_tensors = {}
            
            for internal_var in present[domain_name]:
                wrf_var = internal_to_wrf[internal_var]
                
                # Declared source transform (if any) and unit conversion: the one read path.
                converted_data = self._read_field(ds, wrf_var, internal_var)

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
                    'attrs': dict(converted_data.attrs),
                    'domain': domain_name,
                    'variable': internal_var,
                    **self._record_time_metadata(converted_data),
                    'cell_bounds': self._cf_horizontal_coordinates(ds, converted_data),
                }

                setattr(tensor, '_relic_meta', tensor_meta)
                domain_tensors[internal_var] = tensor
                
            
            # Merge domain tensors with main collection
            if len(datasets_to_read) == 1:
                tensors.update(domain_tensors)
            else:
                for var_name, tensor in domain_tensors.items():
                    key = f"{var_name}_{domain_name}"
                    tensors[key] = tensor
        
        return tensors
    
    def transform_to_internal(
        self,
        data: Dict[str, torch.Tensor],
        target_grid: Optional[Dict] = None,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Transform WRF data to RELIC internal format (shared ExternalAdapter._transform_fields)."""
        if kwargs:
            raise TypeError(f"Unsupported transform options: {sorted(kwargs)}")
        return self._transform_fields(data, target_grid=target_grid, variable_metadata=variable_metadata)

    def close(self) -> None:
        """Close all opened datasets"""
        self.logger.info("Closing WRF adapter")
        
        # Every dataset is attempted; the state is cleared and close errors are raised.
        self._close_datasets(self.discovered_files, self.domain_info)

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
        Forward pass for WRF data provision at the simulation time.

        Returns every catalog field of the opened WRF data at time_origin + ``time`` in the
        simulator calendar through the single adapter read path (schema mapping, declared source
        transforms, unit conversion, then the configured time_interpolation);
        ``kwargs["variables"]`` restricts the internal variable names. The simulator passes
        ``calendar_type`` and ``time_origin`` from its settings; both are required.

        Returns:
            Dict[str, UpdateItem]: replace updates keyed by internal catalog variable names
        """
        # References: RELIC coupling adapter contract (relic/coupling/external/common/base_adapter.py,
        #   ExternalAdapter.read, _read_field and _read_records_at); CDEPS/E3SM shr_tInterp_getFactors
        #   (forcing_record_weights); no WRF process equation is evaluated here.
        # Mathematical expression: y_n(t) = sum_(k, w) w X_n[record k] with X_n = read(variables)[n]
        #   (X_n = convert(transform(D[m(n)])), m the schema mapping) and
        #   (k, w) = forcing_record_weights(s, t, rule), s the record times in seconds since time_origin.
        # Parameter meanings: D opened WRF dataset; m(n) external name of internal variable n;
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

