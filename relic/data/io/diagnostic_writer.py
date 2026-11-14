"""
Diagnostic Writer Module

Provides bridge layer between StateManager diagnostics (ConservationSummary) 
and NetCDF output (DataWriter). Converts in-memory diagnostic data to 
xarray Datasets with local diagnostic metadata and CF time coordinates.

Key Features:
- ConservationSummary -> xarray.Dataset conversion
- Explicit metadata registry for diagnostic variables
- Time coordinate generation from simulation time
- Seamless integration with existing DataWriter

Mathematical Framework:
Time coordinate conversion: t_cf = t_ref + (t_sim / S_d) days
where t_sim is simulation time in seconds, t_ref is reference date

References:
- CF Conventions 1.8: http://cfconventions.org/cf-conventions/cf-conventions.html
- NetCDF Best Practices: https://www.unidata.ucar.edu/software/netcdf/docs/BestPractices.html
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import xarray as xr
from torch import Tensor

# Import existing modules
from ...core.state import ConservationSummary
from ..standards.cf_conventions import CFConventions
from ..standards.unit_converter import UnitConverter

if TYPE_CHECKING:
    from .writer import WriteResult

logger = logging.getLogger(__name__)


def _budget_metadata(element: str, symbol: str) -> Dict[str, Any]:
    """Format the existing budget owner's unit, without converting its value.

    References: CF 1.13 Section 3.1; StateManager conservation unit contract.
    Expression: q_cf = q_budget; the declared element tag is stored separately.
    Parameters: element is the budget name; symbol is its ledger annotation.
    Units: storage comes only from UnitConverter.get_element_budget_units.
    Route: observation of the residual; no state, source, or sink update.
    Numerics: exact unit-token formatting, with no numeric scale inference.
    """
    storage = UnitConverter.get_element_budget_units(element)["storage"]
    return {
        "units": " ".join(token for token in storage.split() if token != symbol),
        "element": symbol,
        "long_name": f"{element.capitalize()} mass balance error",
        "comment": f"Canonical {element} budget residual from StateManager",
    }


# Local diagnostic metadata, without invented official CF standard names.
# CF Conventions 1.13, Section 3.1: element labels are separate from units;
# C and N in a UDUNITS expression would mean coulomb and newton.
# Defaults do not impose CF masking ranges on diagnostic observations. Process
# validity and tolerance checks belong to their scientific/validation owners.
DEFAULT_DIAGNOSTIC_METADATA = {
    # Conservation metrics
    'carbon_balance_error': _budget_metadata('carbon', 'C'),
    'nitrogen_balance_error': _budget_metadata('nitrogen', 'N'),
    'phosphorus_balance_error': _budget_metadata('phosphorus', 'P'),
    'water_balance_error': _budget_metadata('water', 'water'),
    
    # Conservation relative errors
    'carbon_relative_error': {
        'units': '1',
        'long_name': 'Carbon mass balance relative error',
        'comment': 'Relative error: abs(balance_error) / pool_scale'
    },
    'nitrogen_relative_error': {
        'units': '1',
        'long_name': 'Nitrogen mass balance relative error',
        'comment': 'Relative error: abs(balance_error) / pool_scale'
    },
    'phosphorus_relative_error': {
        'units': '1',
        'long_name': 'Phosphorus mass balance relative error',
        'comment': 'Relative error: abs(balance_error) / pool_scale'
    },
    
    # Photosynthesis diagnostics
    'photosynthesis.vcmax_limiting': {
        'units': '1',
        'long_name': 'Vcmax limitation factor for photosynthesis',
        'comment': 'Fraction of photosynthesis limited by carboxylation capacity'
    },
    'photosynthesis.light_limiting': {
        'units': '1',
        'long_name': 'Light limitation factor for photosynthesis',
        'comment': 'Fraction of photosynthesis limited by light availability'
    },
    'photosynthesis.rubp_limiting': {
        'units': '1',
        'long_name': 'RuBP regeneration limitation factor',
        'comment': 'Fraction of photosynthesis limited by RuBP regeneration'
    },
    
    # Soil diagnostics
    'soil.water_stress': {
        'units': '1',
        'long_name': 'Soil water stress limitation factor',
        'comment': 'Stress factor due to soil water deficit (0=max stress, 1=no stress)'
    },
    'soil.temperature_factor': {
        'units': '1',
        'long_name': 'Soil temperature response factor',
        'comment': 'Temperature response multiplier for soil processes'
    },
    'soil.nitrogen_availability': {
        'units': '1',
        'long_name': 'Soil nitrogen availability factor',
        'comment': 'Nitrogen availability limitation factor (0=no N, 1=unlimited N)'
    },
    
    # Respiration diagnostics
    'respiration.q10_factor': {
        'units': '1',
        'long_name': 'Q10 temperature response for respiration',
        'comment': 'Q10 temperature response multiplier'
    },
    
    # Conservation tolerance tracking
    'dynamic_tolerance': {
        'long_name': 'Dynamic conservation tolerance',
        'comment': 'Numerical tolerance in each ledger canonical basis; mixed ledgers have no single physical unit'
    },
    'conservation_violations': {
        'units': '1',
        'long_name': 'Conservation violation count',
        'comment': 'Number of conservation violations detected in this time step'
    },
}


class DiagnosticWriter:
    """
    Bridge between StateManager diagnostics and the existing NetCDF writer.
    
    This class converts ConservationSummary objects (in-memory diagnostics) to
    xarray Datasets with local metric metadata and CF time coordinates, enabling storage
    of diagnostic information for post-hoc analysis.
    
    Mathematical Expression:
        Time conversion: t_cf = (t_sim / S_d) days since reference_date,
        where t_sim is simulation time in seconds and S_d is seconds per day.
    
    References:
        - CF Conventions 1.8: http://cfconventions.org/cf-conventions/cf-conventions.html
        - RELIC State Manager: relic/core/state_manager.py
    """
    
    def __init__(self, metadata_registry: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize diagnostic writer with optional custom metadata registry.
        
        Parameters:
            metadata_registry: Custom metadata for diagnostic variables.
                             If None, uses DEFAULT_DIAGNOSTIC_METADATA.
                             Format: {var_name: {standard_name, units, long_name, ...}}
        """
        self.metadata_registry = (
            DEFAULT_DIAGNOSTIC_METADATA.copy()
            if metadata_registry is None
            else metadata_registry.copy()
        )
        self.cf_conventions = CFConventions()
        
        logger.debug(f"DiagnosticWriter initialized with {len(self.metadata_registry)} "
                    "registered diagnostic variables")
    
    def register_diagnostic_metadata(
        self,
        diagnostic_key: str,
        standard_name: str,
        units: str,
        long_name: str,
        valid_range: Optional[Tuple[float, float]] = None,
        comment: Optional[str] = None
    ) -> None:
        """
        Register CF-1.8 metadata for a diagnostic variable.
        
        This allows custom diagnostics to be properly documented with CF metadata
        when written to NetCDF files.
        
        Parameters:
            diagnostic_key: Diagnostic variable key (e.g., 'soil.custom_metric')
            standard_name: CF-1.8 standard name
            units: Variable units (CF-compliant)
            long_name: Human-readable variable name
            valid_range: Optional (min, max) valid value range
            comment: Optional descriptive comment
        
        Example:
            writer.register_diagnostic_metadata(
                'custom.limiting_factor',
                'custom_process_limitation_factor',
                '1',
                'Custom process limitation factor',
                valid_range=(0.0, 1.0)
            )
        """
        metadata: Dict[str, Any] = {
            'standard_name': standard_name,
            'units': units,
            'long_name': long_name
        }
        
        if valid_range is not None:
            metadata['valid_range'] = valid_range
        
        if comment is not None:
            metadata['comment'] = comment
        
        self.metadata_registry[diagnostic_key] = metadata
        logger.debug(f"Registered metadata for diagnostic '{diagnostic_key}'")
    
    def create_cf_time_coordinate(
        self,
        time_values: List[float],
        reference_date: str = "1850-01-01",
        calendar: str = "gregorian"
    ) -> xr.DataArray:
        """
        Create CF-compliant time coordinate from simulation time values.
        
        Converts simulation time (seconds since simulation start) to
        CF-1.8 compliant time coordinate (days since reference date).
        
        Parameters:
            time_values: List of simulation times in seconds
            reference_date: Reference date string (YYYY-MM-DD format)
            calendar: CF calendar type (gregorian, noleap, etc.)
        
        Returns:
            xr.DataArray with CF-compliant time coordinate
        
        References:
            CF Conventions 1.8 Section 4.4; UDUNITS time coordinates.
        Mathematical expression:
            t_cf = t_sim / S_d.
        Parameter meanings:
            t_sim is elapsed simulation time and S_d is seconds per day.
        Units:
            t_sim [s], S_d [s day-1], and t_cf [day].
        Domain/constraints:
            S_d is resolved from the physical-constant SSOT.
        Sign convention:
            Positive t_cf advances after the reference epoch.
        
        Example:
            >>> times = [0.0, 1800.0, 3600.0]  # Seconds
            >>> time_coord = writer.create_cf_time_coordinate(times)
            >>> print(time_coord.attrs['units'])
            'days since 1850-01-01'
        """
        from relic.utils.physical_constants import get_physical_constant

        seconds_per_day = float(get_physical_constant("seconds_per_day"))
        days_since_ref = np.asarray(time_values, dtype=float) / seconds_per_day
        
        # Create DataArray with CF attributes
        time_coord = xr.DataArray(
            days_since_ref,
            dims=['time'],
            attrs={
                'standard_name': 'time',
                'long_name': 'time',
                'units': f'days since {reference_date}',
                'calendar': calendar,
                'axis': 'T'
            }
        )
        
        logger.debug(f"Created CF time coordinate with {len(time_values)} values")
        return time_coord
    
    def _get_variable_metadata(self, var_name: str) -> Dict[str, Any]:
        """
        Get CF-1.8 metadata for a diagnostic variable.
        
        Parameters:
            var_name: Variable name
        
        Returns:
            Metadata dictionary with CF attributes
        """
        # Check registered metadata
        if var_name in self.metadata_registry:
            return self.metadata_registry[var_name].copy()
        
        # Check for namespaced variable (e.g., 'module.variable')
        base_name = var_name.split('.')[-1]
        if base_name in self.metadata_registry:
            return self.metadata_registry[base_name].copy()
        
        # A local metric name cannot establish physical units or CF identity.
        logger.warning("No registered metadata for '%s'; units are unspecified", var_name)
        return {
            'long_name': var_name.replace('_', ' ').replace('.', ' ').title(),
            'comment': 'Unregistered diagnostic; physical units and CF identity are unspecified'
        }
    
    def convert_conservation_summary(
        self,
        summary: ConservationSummary,
        time_value: float,
        reference_date: str = "1850-01-01"
    ) -> xr.Dataset:
        """
        Convert single ConservationSummary to xarray Dataset.
        
        Extracts diagnostic data from ConservationSummary and creates
        an xarray Dataset without a dataset-level CF conformance claim.
        
        Parameters:
            summary: ConservationSummary object from StateManager
            time_value: Simulation time in seconds
            reference_date: Reference date for time coordinate
        
        Returns:
            xr.Dataset with diagnostic variables and CF metadata
        
        Example:
            >>> summary = state_manager.get_last_conservation_summary()
            >>> dataset = writer.convert_conservation_summary(summary, 1800.0)
            >>> print(dataset)
            <xarray.Dataset>
            Dimensions:  (time: 1)
            Coordinates:
              * time     (time) float64 ...
            Data variables:
                carbon_balance_error  (time) float64 ...
                ...
        """
        # Create time coordinate
        time_coord = self.create_cf_time_coordinate([time_value], reference_date)
        
        # Initialize data dictionary
        data_vars = {}
        
        # Extract balance errors from summary
        # Map element abbreviations to full names for CF compliance
        element_name_map = {
            'C': 'carbon',
            'N': 'nitrogen',
            'P': 'phosphorus',
            'W': 'water',
            'carbon': 'carbon',
            'nitrogen': 'nitrogen',
            'phosphorus': 'phosphorus',
            'water': 'water',
        }
        
        if summary.balance_error:
            for element, error in summary.balance_error.items():
                # Map abbreviated element names to full names
                full_element_name = element_name_map.get(element, element.lower())
                var_name = f'{full_element_name}_balance_error'
                if full_element_name in ('carbon', 'nitrogen', 'phosphorus', 'water'):
                    canonical = DEFAULT_DIAGNOSTIC_METADATA[var_name]
                    metadata = self.metadata_registry.get(var_name, {}).copy()
                    for field_name in ('units', 'element'):
                        if field_name in metadata and metadata[field_name] != canonical[field_name]:
                            raise ValueError(f"Diagnostic budget unit/element contract conflicts for {var_name}")
                        metadata[field_name] = canonical[field_name]
                else:
                    metadata = self._get_variable_metadata(var_name)
                
                data_vars[var_name] = xr.DataArray(
                    [error],
                    dims=['time'],
                    attrs=metadata
                )
        
        # Extract element-specific reports
        for element_name in ['carbon', 'nitrogen', 'phosphorus']:
            report_attr = f'{element_name}_report'
            report = getattr(summary, report_attr, None)
            
            if report is not None:
                # Balance error (already added above, skip)
                pass
                
                # Relative error
                var_name = f'{element_name}_relative_error'
                metadata = self._get_variable_metadata(var_name)
                
                data_vars[var_name] = xr.DataArray(
                    [report.relative_error],
                    dims=['time'],
                    attrs=metadata
                )
        
        # Add conservation metadata
        if summary.dt is not None:
            metadata = self._get_variable_metadata('dynamic_tolerance')
            data_vars['dynamic_tolerance'] = xr.DataArray(
                [summary.tolerance_value],
                dims=['time'],
                attrs=metadata
            )
        
        # Add violation count
        metadata = self._get_variable_metadata('conservation_violations')
        data_vars['conservation_violations'] = xr.DataArray(
            [summary.total_violations],
            dims=['time'],
            attrs=metadata
        )

        # Scalar diagnostic export must not overwrite another field or silently
        # discard a value that this time-series representation cannot express.
        for key, value in summary.custom_checks.items():
            if not isinstance(key, str) or not key:
                raise TypeError("Diagnostic names must be nonempty strings")
            var_name = key.replace('.', '_')
            if var_name in data_vars:
                raise ValueError(f"Diagnostic output name collision for {key}: {var_name}")
            try:
                scalar = float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(f"Diagnostic {key} must be a real scalar for time-series export") from exc
            metadata = self._get_variable_metadata(key)
            metadata['relic_diagnostic_key'] = key
            data_vars[var_name] = xr.DataArray([scalar], dims=['time'], attrs=metadata)
        
        # Create Dataset
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={'time': time_coord}
        )
        
        # Add global attributes
        dataset.attrs = self.cf_conventions.create_global_attributes(
            title="RELIC Diagnostic Output",
            source="RELIC StateManager Conservation Report",
            comment="Diagnostic variables from simulation conservation checks and module outputs"
        )
        # Only the output owner assigns artifact creation time. Keeping the
        # converted metadata stable also permits strict cross-record checking.
        dataset.attrs.pop('creation_date', None)
        
        logger.debug(f"Converted ConservationSummary with {len(data_vars)} variables")
        return dataset
    
    def convert_summaries_to_dataset(
        self,
        summaries: List[ConservationSummary],
        time_coords: List[float],
        reference_date: str = "1850-01-01"
    ) -> xr.Dataset:
        """
        Convert list of ConservationSummary objects to time-series Dataset.
        
        Creates an xarray Dataset with CF time coordinates and time series of
        diagnostic variables from multiple simulation steps.
        
        Parameters:
            summaries: List of ConservationSummary objects
            time_coords: List of corresponding simulation times (seconds)
            reference_date: Reference date for time coordinate
        
        Returns:
            xr.Dataset with time-series diagnostic variables
        
        Example:
            >>> summaries = state_manager.conservation_history[-10:]
            >>> times = [s.timestamp for s in summaries]
            >>> dataset = writer.convert_summaries_to_dataset(summaries, times)
            >>> dataset.to_netcdf('diagnostics.nc')
        """
        if len(summaries) != len(time_coords):
            raise ValueError(
                f"Length mismatch: {len(summaries)} summaries vs {len(time_coords)} times"
            )
        
        if not summaries:
            logger.warning("Empty summaries list provided")
            return xr.Dataset()
        
        # Convert each summary and concatenate
        datasets = []
        for summary, time_val in zip(summaries, time_coords):
            ds = self.convert_conservation_summary(summary, time_val, reference_date)
            datasets.append(ds)
        
        # Concatenate along time dimension
        combined = xr.concat(datasets, dim='time', combine_attrs='identical')
        
        # Update global attributes for time series
        combined.attrs['time_coverage_start'] = f"{min(time_coords)} seconds"
        combined.attrs['time_coverage_end'] = f"{max(time_coords)} seconds"
        combined.attrs['time_coverage_duration'] = f"{max(time_coords) - min(time_coords)} seconds"
        
        logger.info(f"Created time-series dataset with {len(summaries)} time steps "
                   f"and {len(combined.data_vars)} variables")
        return combined


# Convenience functions for quick export
def export_diagnostics_to_netcdf(
    summaries: List[ConservationSummary],
    time_coords: List[float],
    output_path: Path,
    reference_date: str = "1850-01-01",
    *,
    delivery_key: Optional[str] = None,
    metadata_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> "WriteResult":
    """
    Export diagnostics through DataWriter and return its checked receipt.
    
    Parameters:
        summaries: List of ConservationSummary objects
        time_coords: List of simulation times (seconds)
        output_path: Output NetCDF file path
        reference_date: Reference date for time coordinate

    Returns:
        The canonical WriteResult after successful output validation.

    Raises:
        OSError: DataWriter returned a failed write or validation result.
    
    Example:
        >>> export_diagnostics_to_netcdf(
        ...     summaries=state_manager.conservation_history,
        ...     time_coords=[s.timestamp for s in summaries],
        ...     output_path=Path('diagnostics.nc')
        ... )
    """
    writer = DiagnosticWriter(metadata_registry=metadata_registry)
    dataset = writer.convert_summaries_to_dataset(summaries, time_coords, reference_date)
    
    # Preserve the existing writer's bounded metadata and output contracts.
    from .writer import DataWriter, WriteConfig
    # CF Conventions 1.13, Sections 7.1 and 7.3: these reports supply time
    # labels, not interval support. Preserve those labels without inventing
    # bounds, including for the first accepted diagnostic record.
    data_writer = DataWriter(WriteConfig(cf_sync=True, strict_cf=False, add_bounds=False))
    result = data_writer.write_dataset(dataset, output_path, delivery_key=delivery_key)
    if not result.validation_passed:
        raise OSError(
            f"Diagnostic export failed for {output_path}: {'; '.join(result.warnings)}"
        )
    
    logger.info(f"Exported diagnostics to {output_path}")
    return result


__all__ = [
    'DiagnosticWriter',
    'DEFAULT_DIAGNOSTIC_METADATA',
    'export_diagnostics_to_netcdf',
]
