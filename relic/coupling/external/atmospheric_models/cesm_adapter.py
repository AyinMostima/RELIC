"""
CESM Model Data Adapter

This module implements the CESMAdapter for accessing and processing data from 
the Community Earth System Model (CESM) atmospheric component (CAM) output.
It handles CESM-specific variable naming, coordinate systems, and file organization.

The adapter supports standard CESM history files and follows CF-1.8 conventions.

References:
    - CESM User's Guide: https://www.cesm.ucar.edu/models/cesm2/
    - CAM Documentation: https://ncar.github.io/CAM/doc/build/html/
    - CF Conventions 1.8: http://cfconventions.org/cf-conventions/v1.8/
    
Mathematical Expressions:
    Pressure levels: P(k) = A(k) * P_0 + B(k) * P_surface
    Hybrid coordinates: sigma(k) = A(k) + B(k) * (P_surface/P_0)
    open() adds no derived fields to the opened datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
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
from ....utils.physical_constants import get_physical_constant
# ProcessCategory removed - ExternalAdapter handles category internally

logger = logging.getLogger(__name__)

@dataclass
class CESMConfig:
    """Configuration for CESM adapter"""
    root_path: Union[str, Path]
    file_pattern: str = "*.cam.h*.nc"  # File pattern for CAM history files
    history_stream: Optional[str] = None  # Specific history stream (h0, h1, etc.)
    component: str = "cam"  # CESM component (cam, clm, cice, pop, etc.)
    case_name: Optional[str] = None  # Specific case name pattern
    pressure_levels: Optional[List[float]] = None  # Specific pressure levels (Pa)
    extract_surface: bool = True  # Whether to extract surface variables
    # Soil layer thicknesses [m], top-down, one per H2OSOI layer; used only when the history
    # file does not carry DZSOI (reading H2OSOI without either raises).
    soil_layer_thickness: Optional[List[float]] = None
    # Rule of a forcing read at the simulation time: 'lower' (t_k <= t < t_(k+1) holds record k)
    # or 'linear' (between the two records around t); required by forward, never assumed.
    time_interpolation: Optional[str] = None
    # Catalog variables the adapter writes (its declared outputs; each a mapping target of the
    # source in InternalSchema); None declares every mapping target of the source.
    provided_variables: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_time_interpolation(self.time_interpolation)



class CESMAdapter(ExternalAdapter):
    """
    Adapter for Community Earth System Model (CESM) atmospheric output.
    
    This adapter provides access to CESM CAM atmospheric model output files,
    handling the model's hybrid coordinate system, variable naming conventions,
    and temporal organization. It automatically processes history streams and
    computes derived variables.
    
    Key Features:
    - Hybrid sigma-pressure coordinate processing
    - Multiple history stream support
    - Pressure level interpolation
    - Surface and atmospheric variable extraction
    - CF-1.8 compliant metadata generation
    
    Attributes:
        config: CESM configuration parameters
        discovered_files: List of discovered CESM files
        current_datasets: Currently loaded datasets
        coordinate_info: Information about CESM coordinate systems
    """
    
    def __init__(
        self,
        config: Optional[CESMConfig] = None,
        time_window: Optional[TimeWindow] = None,
        spatial_window: Optional[SpatialWindow] = None,
        **kwargs
    ):
        """
        Initialize CESM adapter.
        
        Parameters:
            config: CESM configuration
            time_window: Temporal subset window 
            spatial_window: Spatial subset window
            **kwargs: Additional arguments passed to ExternalAdapter
        """
        self._require_adapter_config(config, CESMConfig, "CESMAdapter")
        super().__init__(
            source=DataSource.CESM,
            name="CESMAdapter",
            model_id="cesm",
            provided_variables=None if config is None else config.provided_variables,
            **kwargs
        )
        
        if config is None:
            raise ValueError(
                "CESMAdapter needs a CESMConfig declaring root_path; the output directory is never "
                "assumed (no current-directory default)."
            )
        self.config = config
        self.time_window = time_window
        self.spatial_window = spatial_window
        
        # Internal state
        self.discovered_files: List[Path] = []
        self.current_datasets: Dict[str, xr.Dataset] = {}
        self.coordinate_info: Dict[str, Any] = {}
        
        # Initialize handlers
        self.time_handler = TimeHandler()
        self.regrid_operator = RegridOperator()
        self.io_handler = IOHandler()

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Reference pressure P0 of the CAM hybrid coordinate when the file declares none
        self.reference_pressure_pa = float(get_physical_constant("reference_pressure"))
        
    def discover(
        self, 
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Discover available CESM output files.
        
        Scans the specified directory for CESM output files matching the
        configuration pattern. Analyzes history streams, time coverage, and
        available variables.
        
        Parameters:
            variables: List of CESM variable names to check for
            **kwargs: Additional discovery parameters
            
        Returns:
            Dictionary containing discovered file information and metadata
            
        Raises:
            FileNotFoundError: If root path does not exist
            ValueError: If no matching files found
        """
        self.logger.info(f"Discovering CESM files in {self.config.root_path}")
        
        root_path = Path(self.config.root_path)
        if not root_path.exists():
            raise FileNotFoundError(f"CESM root path does not exist: {root_path}")
            
        # Build file pattern
        file_pattern = self.config.file_pattern
        if self.config.case_name:
            file_pattern = f"{self.config.case_name}.{file_pattern}"
        if self.config.history_stream:
            file_pattern = file_pattern.replace("h*", self.config.history_stream)
            
        matching_files = list(root_path.glob(file_pattern))
        
        if not matching_files:
            # Try recursive search
            matching_files = list(root_path.rglob(file_pattern))
            
        if not matching_files:
            raise ValueError(f"No CESM files found matching pattern: {file_pattern}")
            
        # A deterministic listing; the records are ordered by their decoded times when opened.
        matching_files = sorted(matching_files)
        
        self.discovered_files = matching_files
        
        # Analyze file contents
        file_analysis = self._analyze_cesm_files(matching_files, variables)
        
        results = {
            "source": "CESM",
            "total_files": len(matching_files),
            "history_streams": file_analysis["history_streams"],
            "time_coverage": file_analysis["time_coverage"],
            "available_variables": file_analysis["variables"],
            "spatial_resolution": file_analysis["spatial_resolution"],
            "vertical_levels": file_analysis["vertical_levels"],
            "coordinate_info": file_analysis["coordinates"]
        }
        
        self.coordinate_info = file_analysis["coordinates"]
        
        self.logger.info(
            f"Discovered {results['total_files']} CESM files with "
            f"{len(results['history_streams'])} history streams"
        )
        
        return results
    
    def _analyze_cesm_files(
        self, 
        files: List[Path], 
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze CESM files to extract metadata"""
        
        history_streams = set()
        all_variables = set()
        time_range = {'start': None, 'end': None}
        spatial_res = {}
        vertical_levels = None
        coordinates = {}
        
        # Sample first file for detailed analysis
        if files:
            with xr.open_dataset(files[0], engine='netcdf4') as sample_ds:
                # Extract history stream information
                filename = files[0].name
                stream_match = re.search(r'\.h(\d+)\.', filename)
                if stream_match:
                    history_streams.add(f"h{stream_match.group(1)}")
                else:
                    history_streams.add("unknown")

                # Spatial resolution
                if 'lat' in sample_ds.coords and 'lon' in sample_ds.coords:
                    lat = sample_ds.coords['lat']
                    lon = sample_ds.coords['lon']
                    
                    if len(lat) > 1 and len(lon) > 1:
                        lat_res = abs(float(lat[1] - lat[0]))
                        lon_res = abs(float(lon[1] - lon[0]))
                        
                        spatial_res = {
                            'lat_resolution': lat_res,
                            'lon_resolution': lon_res,
                            'n_lat': len(lat),
                            'n_lon': len(lon)
                        }

                # Vertical coordinate information
                if 'lev' in sample_ds.coords:
                    vertical_levels = len(sample_ds.coords['lev'])
                    
                    # Check for hybrid coordinate coefficients
                    if 'hyam' in sample_ds.coords and 'hybm' in sample_ds.coords:
                        coordinates['hybrid_coefficients'] = {
                            'a_coeffs': 'hyam',
                            'b_coeffs': 'hybm',
                            'reference_pressure': sample_ds.attrs.get('P0', self.reference_pressure_pa)
                        }
                    
                    # Check for pressure levels
                    if 'plev' in sample_ds.coords:
                        coordinates['pressure_levels'] = sample_ds.coords['plev'].values.tolist()

                # Available variables
                if variables:
                    available_vars = [var for var in variables if var in sample_ds.data_vars]
                    all_variables.update(available_vars)
                else:
                    # Get all time-dependent variables
                    for var_name, var_data in sample_ds.data_vars.items():
                        if 'time' in var_data.dims:
                            all_variables.add(var_name)

                # Time coordinate information
                if 'time' in sample_ds.coords:
                    time_coord = sample_ds.coords['time']
                    coordinates['time'] = {
                        'calendar': time_coord.attrs.get('calendar', 'noleap'),
                        'units': time_coord.attrs.get('units', 'days since 0001-01-01'),
                        'bounds': time_coord.attrs.get('bounds', None)
                    }

        
        # Time range over every file (decoded datetime64 or cftime records, compared in the
        # dataset calendar; a file that cannot be opened raises).
        if files and len(files) > 1:
            timestamps = []
            for filepath in files:
                with xr.open_dataset(filepath, engine='netcdf4') as ds:
                    if 'time' in ds.coords:
                        timestamps.extend(list(ds.coords['time'].values))

            def _iso(value: Any) -> str:
                return value.isoformat() if hasattr(value, 'isoformat') else pd.Timestamp(value).isoformat()

            if timestamps:
                time_range['start'] = _iso(min(timestamps))
                time_range['end'] = _iso(max(timestamps))
        
        return {
            'history_streams': sorted(list(history_streams)),
            'variables': sorted(list(all_variables)),
            'time_coverage': time_range,
            'spatial_resolution': spatial_res,
            'vertical_levels': vertical_levels,
            'coordinates': coordinates
        }
    
    def open(self, **kwargs) -> Dict[str, Any]:
        """
        Open and load discovered CESM datasets.
        
        Opens CESM files and processes them according to configuration.
        Handles coordinate transformation, temporal concatenation, and
        derived variable computation.
        
        Parameters:
            **kwargs: Additional opening parameters
            
        Returns:
            Dictionary containing opened datasets and metadata
            
        Raises:
            ValueError: If no files have been discovered
            IOError: If files cannot be opened
        """
        if not self.discovered_files:
            raise ValueError("No CESM files discovered. Call discover() first.")
            
        self.logger.info(f"Opening {len(self.discovered_files)} CESM files")
        
        # Group files by history stream
        stream_files = self._group_files_by_stream()
        
        opened_datasets = {}
        
        for stream_name, files in stream_files.items():
            self.logger.debug(f"Opening {len(files)} files for stream {stream_name}")
            
            try:
                # Open and concatenate files
                datasets = []
                
                for filepath in files:
                    ds = xr.open_dataset(
                        filepath,
                        engine='netcdf4',
                        decode_times=True,
                        decode_cf=True,
                        **kwargs
                    )
                    
                    # Process coordinates
                    ds = self._process_cesm_coordinates(ds)
                    
                    datasets.append(ds)
                
                # Concatenate in the order of the decoded record times (never file names or
                # modification times).
                combined_ds = self._concat_by_record_time(
                    datasets, [path.name for path in files], dim='time'
                )
                
                # Apply temporal and spatial subsetting
                combined_ds = self._apply_subsetting(combined_ds)
                
                # Interpolate to pressure levels if requested
                if self.config.pressure_levels:
                    combined_ds = self._interpolate_to_pressure_levels(combined_ds)
                
                opened_datasets[stream_name] = combined_ds
                
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                self.logger.error(f"Failed to open datasets for stream {stream_name}: {e}")
                raise IOError(f"Could not open CESM data for stream {stream_name}: {e}")
        
        self.current_datasets = opened_datasets
        
        return {
            "source": "CESM",
            "streams": list(opened_datasets.keys()),
            "temporal_coverage": self._get_temporal_coverage(),
            "variables": self._get_available_variables()
        }
    
    def _group_files_by_stream(self) -> Dict[str, List[Path]]:
        """Group files by CESM history stream"""
        stream_files = {}
        
        for filepath in self.discovered_files:
            # Extract stream from filename
            filename = filepath.name
            stream_match = re.search(r'\.h(\d+)\.', filename)
            if stream_match:
                stream_name = f"h{stream_match.group(1)}"
            else:
                stream_name = "h0"  # Default stream
            
            if stream_name not in stream_files:
                stream_files[stream_name] = []
            stream_files[stream_name].append(filepath)
        
        return stream_files
    
    def _process_cesm_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        """Process CESM coordinate metadata without a dataset CF claim."""
        
        # Ensure latitude coordinate has proper attributes
        if 'lat' in ds.coords:
            ds['lat'].attrs.update({
                'standard_name': 'latitude',
                'long_name': 'Latitude',
                'units': 'degrees_north'
            })
        
        # Ensure longitude coordinate has proper attributes
        if 'lon' in ds.coords:
            ds['lon'].attrs.update({
                'standard_name': 'longitude',
                'long_name': 'Longitude',
                'units': 'degrees_east'
            })
        
        # Process vertical coordinate
        if 'lev' in ds.coords:
            lev_coord = ds['lev']
            
            # Check units and standardize
            if 'units' not in lev_coord.attrs:
                # CESM typically uses hybrid sigma-pressure coordinates
                lev_coord.attrs['units'] = 'level'
                lev_coord.attrs['long_name'] = 'Hybrid level at layer midpoints'
                lev_coord.attrs['standard_name'] = 'atmosphere_hybrid_sigma_pressure_coordinate'
                
                # Add formula_terms if hybrid coefficients are available
                if 'hyam' in ds.coords and 'hybm' in ds.coords:
                    lev_coord.attrs['formula_terms'] = 'a: hyam b: hybm ps: PS p0: P0'
        
        # Process time coordinate
        if 'time' in ds.coords:
            time_coord = ds['time']
            
            # Ensure standard attributes
            if 'standard_name' not in time_coord.attrs:
                time_coord.attrs['standard_name'] = 'time'
            
            # Set calendar if not present
            if 'calendar' not in time_coord.attrs:
                time_coord.attrs['calendar'] = 'noleap'  # CESM default
        
        return ds
    
    def _apply_subsetting(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply temporal and spatial subsetting"""
        
        # Apply temporal subsetting
        if self.time_window and 'time' in ds.coords:
            ds = ds.sel(time=slice(self.time_window.start, self.time_window.end))
        
        # Apply spatial subsetting
        if self.spatial_window:
            subset_dict = {}
            
            if 'lat' in ds.coords:
                subset_dict['lat'] = slice(
                    self.spatial_window.lat_min,
                    self.spatial_window.lat_max
                )
            
            if 'lon' in ds.coords:
                subset_dict['lon'] = slice(
                    self.spatial_window.lon_min,
                    self.spatial_window.lon_max
                )
            
            if subset_dict:
                ds = ds.sel(**subset_dict)
        
        return ds
    
    def _interpolate_to_pressure_levels(self, ds: xr.Dataset) -> xr.Dataset:
        """Interpolate hybrid-level fields to configured pressure levels."""
        if not self.config.pressure_levels:
            return ds

        pressure = self._compute_hybrid_pressure(ds)
        target_pressures = np.asarray(self.config.pressure_levels, dtype=np.float64)
        if target_pressures.ndim != 1 or target_pressures.size == 0:
            raise ValueError("pressure_levels must be a non-empty one-dimensional sequence")
        if not bool(np.all(np.isfinite(target_pressures))):
            raise ValueError("pressure_levels must contain only finite values")
        if bool(np.any(target_pressures <= 0.0)):
            raise ValueError("pressure_levels must be strictly positive")
        if np.unique(target_pressures).size != target_pressures.size:
            raise ValueError("pressure_levels must not contain duplicates")

        target = xr.DataArray(
            target_pressures,
            dims=("plev",),
            coords={"plev": target_pressures},
            attrs={
                "standard_name": "air_pressure",
                "long_name": "Pressure Level",
                "units": "Pa",
            },
        )
        base_coords = {
            name: coord
            for name, coord in ds.coords.items()
            if "lev" not in coord.dims and name != "lev"
        }
        pressure_ds = xr.Dataset(
            coords={**base_coords, "plev": target},
            attrs=dict(ds.attrs),
        )

        for var_name, var_data in ds.data_vars.items():
            if var_name in {"hyam", "hybm"}:
                continue
            if "lev" not in var_data.dims:
                pressure_ds[var_name] = var_data
                continue
            if not np.issubdtype(var_data.dtype, np.floating):
                raise TypeError(
                    f"CESM variable '{var_name}' with a lev dimension must be floating-point"
                )

            output_dtype = np.result_type(var_data.dtype, np.float32)
            interpolated = xr.apply_ufunc(
                self._interpolate_pressure_column,
                var_data,
                pressure,
                target,
                input_core_dims=[["lev"], ["lev"], ["plev"]],
                output_core_dims=[["plev"]],
                exclude_dims={"lev"},
                vectorize=True,
                dask="parallelized",
                output_dtypes=[output_dtype],
                dask_gufunc_kwargs={
                    "output_sizes": {"plev": target_pressures.size}
                },
                keep_attrs=True,
            )
            desired_dims = tuple(
                "plev" if dim == "lev" else dim for dim in var_data.dims
            )
            pressure_ds[var_name] = interpolated.transpose(*desired_dims)

        pressure_ds["plev"].attrs = dict(target.attrs)
        return pressure_ds

    def _compute_hybrid_pressure(self, ds: xr.Dataset) -> xr.DataArray:
        """Return CAM mid-level pressure from the declared hybrid coordinate."""
        required = ("lev", "hyam", "hybm", "PS")
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(
                "Missing CESM hybrid-pressure inputs: " + ", ".join(missing)
            )

        hyam = ds["hyam"]
        hybm = ds["hybm"]
        surface_pressure = ds["PS"]
        if hyam.dims != ("lev",) or hybm.dims != ("lev",):
            raise ValueError("hyam and hybm must be one-dimensional over lev")
        if hyam.sizes["lev"] != ds.sizes["lev"] or hybm.sizes["lev"] != ds.sizes["lev"]:
            raise ValueError("hyam, hybm, and lev must have the same length")
        if "lev" in surface_pressure.dims:
            raise ValueError("PS must not contain the lev dimension")
        if str(surface_pressure.attrs.get("units", "")).strip() != "Pa":
            raise ValueError("CESM PS must declare units 'Pa'")

        reference_pressure = float(
            ds.attrs.get("P0", self.reference_pressure_pa)
        )
        if not np.isfinite(reference_pressure) or reference_pressure <= 0.0:
            raise ValueError("CESM P0 must be a finite positive pressure in Pa")

        # References: CESM CAM hybrid sigma-pressure coordinate definition.
        # Expression: p(k) = hyam(k) * P0 + hybm(k) * PS.
        # Parameters: hyam and hybm [1], P0 and PS [Pa].
        # Route: diagnostic vertical coordinate; no state or budget update.
        pressure = hyam * reference_pressure + hybm * surface_pressure
        pressure.attrs = {
            "standard_name": "air_pressure",
            "long_name": "Hybrid Mid-Level Pressure",
            "units": "Pa",
        }
        return pressure

    @staticmethod
    def _interpolate_pressure_column(
        values: np.ndarray,
        pressures: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate one finite atmospheric column in pressure."""
        values_array = np.asarray(values)
        pressure_array = np.asarray(pressures, dtype=np.float64)
        target_array = np.asarray(targets, dtype=np.float64)
        if values_array.ndim != 1 or pressure_array.ndim != 1:
            raise ValueError("Column values and pressures must be one-dimensional")
        if values_array.size != pressure_array.size or values_array.size < 2:
            raise ValueError("Each pressure column must contain at least two paired levels")
        if not bool(np.all(np.isfinite(values_array))) or not bool(
            np.all(np.isfinite(pressure_array))
        ):
            raise ValueError("CESM interpolation columns must contain only finite values")
        if bool(np.any(pressure_array <= 0.0)):
            raise ValueError("CESM interpolation pressure must be strictly positive")

        order = np.argsort(pressure_array)
        sorted_pressure = pressure_array[order]
        sorted_values = values_array[order]
        if bool(np.any(np.diff(sorted_pressure) <= 0.0)):
            raise ValueError("CESM interpolation pressure levels must be unique")
        if bool(np.any(target_array < sorted_pressure[0])) or bool(
            np.any(target_array > sorted_pressure[-1])
        ):
            raise ValueError(
                "Requested pressure level lies outside the model column"
            )

        # References: CESM CAM hybrid-pressure post-processing convention.
        # Expression: y(p*) = linear_interp(p*, p(k), y(k)).
        # Parameters: p(k), p* [Pa], and y in the source variable units.
        # Route: diagnostic coordinate transform; no source or sink is introduced.
        result = np.interp(target_array, sorted_pressure, sorted_values)
        return result.astype(
            np.result_type(values_array.dtype, np.float32),
            copy=False,
        )
    
    def _get_temporal_coverage(self) -> Dict[str, Dict]:
        """Get temporal coverage for opened datasets"""
        coverage = {}
        
        for stream_name, ds in self.current_datasets.items():
            if 'time' in ds.coords:
                time_coord = ds.coords['time']
                coverage[stream_name] = {
                    "start": str(time_coord.values[0]),
                    "end": str(time_coord.values[-1]),
                    "n_timesteps": len(time_coord)
                }
        
        return coverage
    
    def _get_available_variables(self) -> Dict[str, List[str]]:
        """Get available variables for opened datasets"""
        variables = {}
        
        for stream_name, ds in self.current_datasets.items():
            variables[stream_name] = list(ds.data_vars.keys())
        
        return variables
    
    def read(
        self, 
        variables: Optional[List[str]] = None,
        stream: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Read and transform CESM data to RELIC internal format.
        
        Parameters:
            variables: List of variables to read (internal names)
            stream: Specific history stream to read from
            **kwargs: Additional reading parameters
            
        Returns:
            Dictionary mapping internal variable names to PyTorch tensors
        """
        if not self.current_datasets:
            raise ValueError("No datasets opened. Call open() first.")
        
        # Select stream(s) to read from
        if stream:
            datasets_to_read = {stream: self.current_datasets[stream]}
        else:
            datasets_to_read = self.current_datasets
        
        self.logger.info(f"Reading CESM variables from {len(datasets_to_read)} streams")
        
        # Get variable mappings
        from ..common.schema import InternalSchema
        # internal -> CESM through the schema owner (ambiguous many-to-one mappings raise
        # unless a canonical external variable is declared).
        internal_to_cesm = InternalSchema().resolve_external_names(DataSource.CESM, variables)
        
        tensors = {}
        
        # Mapped fields each dataset contains; a requested field that no dataset has raises
        # and absent mapped fields are listed in last_read_report (never skipped silently).
        present = self._present_mapped_fields(datasets_to_read, internal_to_cesm, variables)
        for stream_name, ds in datasets_to_read.items():
            stream_tensors = {}
            
            for internal_var in present[stream_name]:
                cesm_var = internal_to_cesm[internal_var]
                
                # Declared source transform (if any) and unit conversion: the one read path.
                converted_data = self._read_field(ds, cesm_var, internal_var)

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
                    'attrs': dict(converted_data.attrs),
                    'stream': stream_name
                }

                setattr(tensor, '_relic_meta', tensor_meta)
                stream_tensors[internal_var] = tensor
                
            
            # Merge stream tensors with main collection
            if len(datasets_to_read) == 1:
                tensors.update(stream_tensors)
            else:
                for var_name, tensor in stream_tensors.items():
                    key = f"{var_name}_{stream_name}"
                    tensors[key] = tensor
        
        return tensors
    
    def transform_to_internal(
        self,
        data: Dict[str, torch.Tensor],
        target_grid: Optional[Dict] = None,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Transform CESM data to RELIC internal format (shared ExternalAdapter._transform_fields)."""
        if kwargs:
            raise TypeError(f"Unsupported transform options: {sorted(kwargs)}")
        return self._transform_fields(data, target_grid=target_grid, variable_metadata=variable_metadata)

    def close(self) -> None:
        """Close all opened datasets"""
        self.logger.info("Closing CESM adapter")
        
        # Every dataset is attempted; the state is cleared and close errors are raised.
        self._close_datasets(self.discovered_files, self.coordinate_info)

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
        Forward pass for CESM data provision at the simulation time.

        Returns every catalog field of the opened CESM data at time_origin + ``time`` in the
        simulator calendar through the single adapter read path (schema mapping, declared source
        transforms, unit conversion, then the configured time_interpolation);
        ``kwargs["variables"]`` restricts the internal variable names. The simulator passes
        ``calendar_type`` and ``time_origin`` from its settings; both are required.

        Returns:
            Dict[str, UpdateItem]: replace updates keyed by internal catalog variable names
        """
        # References: RELIC coupling adapter contract (relic/coupling/external/common/base_adapter.py,
        #   ExternalAdapter.read, _read_field and _read_records_at); CDEPS/E3SM shr_tInterp_getFactors
        #   (forcing_record_weights); no CESM process equation is evaluated here.
        # Mathematical expression: y_n(t) = sum_(k, w) w X_n[record k] with X_n = read(variables)[n]
        #   (X_n = convert(transform(D[m(n)])), m the schema mapping) and
        #   (k, w) = forcing_record_weights(s, t, rule), s the record times in seconds since time_origin.
        # Parameter meanings: D opened CESM dataset; m(n) external name of internal variable n;
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
