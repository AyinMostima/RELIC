"""
Base External Model Adapter

This module provides the abstract base class ExternalAdapter that defines 
the unified interface for all external model coupling. All specific adapters
(WRF, CESM, VIC, SWAT, CMIP6) inherit from this class to ensure consistent 
API and behavior.

The design follows the PRO (Policy-Resource-Operation) pattern:
- Policy: Variable mapping, transformation strategies, quality control policies
- Resource: Memory/IO constraints, caching strategies, error handling
- Operation: Read -> Validate -> Transform -> Export workflows

Mathematical Framework:
The adapter transforms external data X_ext to internal format X_int via:
X_int = T(M(V(X_ext)))

Where:
- V: Validation operator (bounded metadata, range checks, continuity)
- M: Mapping operator (variable names, coordinate systems)  
- T: Transform operator (declared source transforms, unit conversion, optional spatial
  regridding and quality control); forward evaluates every field at the simulation time t
  (time_origin + t in the simulator calendar) by the configured time_interpolation rule
  ('lower' or 'linear', forcing_record_weights), with no temporal aggregation

References:
- CF Conventions: http://cfconventions.org/cf-conventions/cf-conventions.html
- CMIP6 Data Request: https://earthsystemcog.org/projects/wip/CMIP6DataRequest
- NetCDF-Python: https://unidata.github.io/netcdf4-python/
- xarray: http://xarray.pydata.org/en/stable/
- PyTorch: https://pytorch.org/docs/stable/
"""

import abc
import bisect
import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass

import cftime
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import xarray as xr

from ....core.functor import (
    ScaleCouplingFunctor,
    ConflictResolutionPolicy,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    FunctorMetadata,
    ProcessCategory,
)
from ....core.abstractions import StateDict
from ....core.state import UpdateMode, UpdateItem
from ....parameters.namespaces import (
    ParameterNamespace,
    resolve_namespace,
    is_declared_namespace,
)

from .datasource import DataSource
from .validation_types import ValidationReport

if TYPE_CHECKING:
    # _regrid_config imports RegridConfig when it runs: the regrid module loads the spatial
    # scale layer, so the annotation binds the name for static analysis only.
    from .regrid import RegridConfig

logger = logging.getLogger(__name__)

# The formula of every declared source transform kind (InternalSchema.source_transforms()), as
# implemented by the read path (_read_field and the transform methods it dispatches to). The adapter
# metadata expression is generated from these formulas and the schema mapping, so it describes
# only what the read path does.
TRANSFORM_EXPRESSIONS: Dict[str, str] = {
    "accumulated": "r_k = rho_w (A_k - A_(k-1)) c / (t_k - t_(k-1)), A_k = P_k + I_k B",
    "layered": "m_l = rho_w theta_l dz_l",
    "water_depth": "m = rho_w d",
    "water_depth_flux": "F = rho_w v, or F_k = rho_w d_k / (t_k,end - t_k,start)",
    "mixing_ratio": "q = w / (1 + w)",
    "earth_relative_wind": "u_e = u cos(a) - v sin(a), v_e = v cos(a) + u sin(a)",
}

# Type aliases
PathLike = Union[str, Path]
DataArray = xr.DataArray
Dataset = xr.Dataset


@dataclass
class TimeWindow:
    """
    Defines temporal subset for data extraction.
    
    Parameters:
        start: Start time (ISO format string or datetime-like)
        end: End time (ISO format string or datetime-like) 
        freq: Temporal frequency ('D', 'H', 'M', etc.)
        calendar: Calendar type ('gregorian', 'noleap', '360_day')
    """
    start: Any
    end: Any  
    freq: Optional[str] = None
    calendar: Optional[str] = "gregorian"


@dataclass
class SpatialWindow:
    """
    Defines spatial subset for data extraction.
    
    Parameters:
        lon_min: Minimum longitude [degrees_east]
        lon_max: Maximum longitude [degrees_east]
        lat_min: Minimum latitude [degrees_north] 
        lat_max: Maximum latitude [degrees_north]
        vertical_levels: Optional vertical level selection
    """
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    vertical_levels: Optional[List[int]] = None


def validate_record_interval(record_interval: Optional[float], record_interval_units: Optional[str]) -> None:
    """
    Validate a declared output record interval of an adapter configuration.

    The interval and its units are declared together (for example 1.0 and "day"); the
    interval is positive. It converts water depths per record to fluxes only when the file
    declares no time bounds.
    """
    if (record_interval is None) != (record_interval_units is None):
        raise ValueError("record_interval and record_interval_units must be declared together.")
    if record_interval is not None and not float(record_interval) > 0.0:
        raise ValueError(f"record_interval must be positive, got {record_interval!r}.")


# Time interpolation rules of a forcing read at the simulation time (declared per adapter
# configuration as time_interpolation; the CDEPS/E3SM shr_tInterp_getFactors 'lower' and 'linear').
TIME_INTERPOLATIONS = ("lower", "linear")

# CF-1.8 section 4.4.1: one calendar under two names (a synonym, not a conversion).
_CF_CALENDAR_SYNONYMS = {"365_day": "noleap", "gregorian": "standard", "366_day": "all_leap"}


def validate_time_interpolation(time_interpolation: Optional[str]) -> None:
    """Validate a declared time interpolation rule of an adapter configuration (None: undeclared)."""
    if time_interpolation is not None and time_interpolation not in TIME_INTERPOLATIONS:
        raise ValueError(
            f"time_interpolation must be one of {TIME_INTERPOLATIONS}, got {time_interpolation!r}."
        )


def canonical_cf_calendar(calendar: str) -> str:
    """The CF-1.8 name of a calendar (365_day -> noleap, gregorian -> standard, 366_day -> all_leap)."""
    name = str(calendar).strip().lower()
    return _CF_CALENDAR_SYNONYMS.get(name, name)


def parse_time_origin(time_origin: Any, calendar_type: Any) -> Any:
    """The date and time of t = 0: an ISO 8601 string read in the simulator calendar (cftime)."""
    if not isinstance(calendar_type, str) or not calendar_type.strip():
        raise ValueError(
            f"A forcing read at the simulation time needs the simulator calendar_type; got {calendar_type!r}."
        )
    if not isinstance(time_origin, str) or not time_origin.strip():
        raise ValueError(
            "A forcing read at the simulation time needs time_origin, the ISO 8601 date and time of "
            f"t = 0 in the simulator calendar (simulator setting 'time_origin'); got {time_origin!r}."
        )
    try:
        return cftime.num2date(0.0, units=f"seconds since {time_origin}", calendar=canonical_cf_calendar(calendar_type))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"time_origin {time_origin!r} is not a date and time of the {calendar_type!r} calendar: {exc}"
        ) from exc


def forcing_record_weights(record_seconds: Sequence[float], t: float, rule: str) -> List[Tuple[int, float]]:
    """Records and weights of a forcing value at elapsed simulation time t (a pure function of t)."""
    # References: CDEPS/E3SM shr_tInterp_mod, shr_tInterp_getFactors (share/util/shr_tInterp_mod.F90,
    #   retrieved from the codeliteraturedatabase E3SM process knowledge base): 'lower' sets
    #   f1 = 1 on the lower-bound record; 'linear' requires t1 <= t <= t2 and sets
    #   f1 = (t2 - t) / (t2 - t1), f2 = 1 - f1; the datm stream option tintalgo selects the rule.
    # Mathematical expression: with record times s_0 < s_1 < ... < s_(K-1) and
    #   k = max{j : s_j <= t}: lower  y(t) = x_k;
    #   linear y(t) = f1 x_k + (1 - f1) x_(k+1), f1 = (s_(k+1) - t) / (s_(k+1) - s_k), and
    #   y(t) = x_k when t = s_k.
    # Parameter meanings: s_j time of record j and t the simulation time, both in seconds since
    #   time_origin in the simulator calendar; x_j record j; f1 weight of the lower record.
    # Units: s and t in s; the weights are dimensionless; y keeps the units of x.
    # Domain/constraints: s finite and strictly increasing (else ValueError); t finite with
    #   s_0 <= t <= s_(K-1): a time outside the records raises (no extrapolation, no cycling);
    #   rule in TIME_INTERPOLATIONS; the result depends only on t (no read position is kept).
    # Sign convention: weights are non-negative and sum to 1.
    if rule not in TIME_INTERPOLATIONS:
        raise ValueError(f"time_interpolation must be one of {TIME_INTERPOLATIONS}, got {rule!r}.")
    times = [float(value) for value in record_seconds]
    if not times:
        raise ValueError("A forcing read needs at least one record.")
    if not all(math.isfinite(value) for value in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"Forcing record times must be finite and strictly increasing; got {times[:8]}.")
    t = float(t)
    if not math.isfinite(t) or t < times[0] or t > times[-1]:
        raise ValueError(
            f"The simulation time {t} s is outside the forcing records [{times[0]}, {times[-1]}] s "
            "(seconds since time_origin); the read does not extrapolate."
        )
    k = bisect.bisect_right(times, t) - 1
    if rule == "lower" or times[k] == t:
        return [(k, 1.0)]
    lower_weight = (times[k + 1] - t) / (times[k + 1] - times[k])
    return [(k, lower_weight), (k + 1, 1.0 - lower_weight)]


class ExternalAdapter(ScaleCouplingFunctor):
    """
    Abstract base class for all external model adapters.
    
    Provides the unified interface for reading, validating, and transforming
    external model data to RELIC internal formats. Handles bounded metadata,
    variable mapping, unit conversion, spatial regridding, and quality control;
    no temporal resampling is performed.

    All adapters follow the same workflow:
    1. discover() - Scan source to identify available variables and metadata
    2. open() - Establish connection/handles to data source; a time_window or
       spatial_window given at construction subsets the opened data once
    3. read() - Extract the schema-mapped fields of the opened data (every record)
    4. transform_to_internal() - Convert to RELIC tensor format
    5. close() - Clean up resources
    forward() returns every provided variable at the simulation time t (time_origin + t in the
    simulator calendar, by the configured time_interpolation); it keeps no read position
    and reads neither the simulator state nor dt.

    The provided variables are the catalog variables the adapter writes: the configuration's
    ``provided_variables`` (each a mapping target of the source) or, when None, every mapping
    target of the source. They are the Functor's declared outputs (no inputs are read), each
    written in its catalog units, and the adapter is their exclusive writer
    (ConflictResolutionPolicy.EXCLUSIVE: registration refuses a second writer of any of them).
    
    The adapter preserves declared CF metadata without certifying datasets and provides detailed
    validation reports for data quality assessment.

    Parameters
    ----------
    source : DataSource
        External data source abstraction.
    name : str, optional
        Adapter name.
    model_id : Optional[str], optional
        External model identifier.
    parameter_manager : Optional[Any]
        Parameter manager for unit/namespace validation.
    metadata : Optional[FunctorMetadata]
        Scientific metadata (math expression, references, units); its input_variables must be
        empty, its output_variables the provided variables, its variable_kind FORCING and its
        conflict_resolution_policy EXCLUSIVE.
    provided_variables : Optional[Sequence[str]]
        Catalog variables the adapter writes (mapping targets of the source); None declares
        every mapping target of the source.

    Returns
    -------
    Dict[str, Tensor]
        Transformed variables in internal units (CF-1.8 compliant) after
        spatial mapping; units preserved or converted explicitly via
        ParameterManager and CF conventions.

    References
    ----------
    CF Conventions 1.8 for variable metadata and units.
    Jones, P.W. (1999). Remapping schemes (for regridding step).
    WMO Guide to Climatological Practices (data quality).
    """
    
    def __init__(
        self,
        source: DataSource,
        name: str = "ExternalAdapter",
        model_id: Optional[str] = None,
        parameter_manager=None,
        parameter_manager_resolver: Optional[Callable[[], Any]] = None,
        metadata: Optional[Any] = None,
        provided_variables: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """
        Initialize external adapter.
        
        Parameters:
            source: Type of external data source
            name: Human-readable adapter name  
            metadata: Optional FunctorMetadata override with references/units
            **kwargs: Additional configuration parameters
        """
        # Create config dict for FunctorBase
        config = kwargs.pop('config', {})
        # update_mode: a declared UpdateMode (default REPLACE); an unknown mode raises instead of
        # silently becoming REPLACE.
        update_mode_raw = kwargs.pop('update_mode', UpdateMode.REPLACE)
        if isinstance(update_mode_raw, UpdateMode):
            self.update_mode: UpdateMode = update_mode_raw
        else:
            valid = [mode.value for mode in UpdateMode]
            if str(update_mode_raw) not in valid:
                raise ValueError(f"update_mode must be one of {valid}, got {update_mode_raw!r}.")
            self.update_mode = UpdateMode(str(update_mode_raw))
        
        # The declared outputs: the provided catalog variables of the source (none are read).
        declared_outputs = self.declared_outputs(source, provided_variables)
        # Create metadata for FunctorBase
        adapter_name = metadata.name if metadata and getattr(metadata, "name", None) else name
        if metadata is not None:
            self._check_adapter_metadata(metadata, declared_outputs)
        metadata = metadata or FunctorMetadata(
            name=adapter_name,
            category=ProcessCategory.SCALE_COUPLING,
            description=f"External data adapter for {source.value}",
            version="1.0.0",
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=self.mapping_expression(source),
            references=[
                "CF Conventions 1.8",
                "Jones, P.W. (1999). Conservative remapping schemes",
            ],
            parameters={},
            input_variables=[],
            output_variables=declared_outputs,
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FORCING,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        
        super().__init__(
            name=adapter_name,
            config=config,
            metadata=metadata,
            source_scale=f"external_{source.value}",
            target_scale="internal_native",
            coupling_type="spatial",
            **kwargs,
        )
        self.metadata.conservation_laws = []
        self.conservation_properties = list(self.metadata.conservation_laws)
        
        self.source = source
        self.model_id = model_id or source.value
        # The provided catalog variables and their catalog units (the units every forward update
        # carries; StateManager checks them against the variable contract).
        from .schema import InternalSchema

        catalog = InternalSchema()
        self.provided_variables: Tuple[str, ...] = tuple(declared_outputs)
        self.provided_units: Dict[str, str] = {
            variable: catalog.get_variable(variable).units for variable in declared_outputs
        }
        self._parameter_manager = parameter_manager
        self._parameter_manager_resolver = parameter_manager_resolver
        self._conversion_namespace_cache: Dict[str, str] = {}
        self._connected = False
        self._current_dataset = None
        self._validation_report = None

    @staticmethod
    def declared_outputs(source: DataSource, provided_variables: Optional[Sequence[str]]) -> List[str]:
        """
        The catalog variables an adapter of ``source`` writes.

        None declares every mapping target of the source (sorted); a sequence must name mapping
        targets of the source without duplicates (an unknown or unmapped name raises KeyError, a
        target with several external fields and no declared canonical field raises ValueError).
        """
        from .schema import InternalSchema

        schema = InternalSchema()
        mapping = schema.get_source_mapping(source)
        if provided_variables is None:
            return sorted(set(mapping.values()))
        if isinstance(provided_variables, str) or not isinstance(provided_variables, Sequence):
            raise TypeError(
                f"provided_variables must be a sequence of catalog variable names, got {provided_variables!r}"
            )
        names = [str(name) for name in provided_variables]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"provided_variables lists duplicate variables: {duplicates}")
        schema.resolve_external_names(source, names)
        return names

    @staticmethod
    def mapping_expression(source: DataSource) -> str:
        """
        The read path of an adapter of ``source`` as a metadata expression: the time selection, the
        schema mapping m (external -> internal) and the formula of every declared source transform,
        generated from InternalSchema (a transform kind without a formula raises KeyError).
        """
        from .schema import InternalSchema

        mapping = InternalSchema().get_source_mapping(source)
        by_kind: Dict[str, List[str]] = {}
        for external, declaration in InternalSchema.source_transforms().get(source, {}).items():
            by_kind.setdefault(declaration["kind"], []).append(external)
        transforms = "; ".join(
            f"{kind} ({', '.join(sorted(fields))}): {TRANSFORM_EXPRESSIONS[kind]}"
            for kind, fields in sorted(by_kind.items())
        )
        pairs = ", ".join(f"{external}->{internal}" for external, internal in sorted(mapping.items()))
        return (
            "y_n(t) = sum_(k, w) w X_n[record k], (k, w) = forcing_record_weights(s, t, rule); "
            "X_n = convert_catalog_units(transform(D[m(n)])); "
            f"m: {pairs}; transforms: {transforms if transforms else 'none'}"
        )

    @staticmethod
    def _check_adapter_metadata(metadata: Any, declared_outputs: Sequence[str]) -> None:
        """A caller's metadata must declare the adapter contract: no inputs, the provided variables
        as outputs, the forcing kind and the exclusive-writer policy."""
        problems = []
        if list(getattr(metadata, "input_variables", []) or []):
            problems.append(f"input_variables {list(metadata.input_variables)} (an adapter reads no state)")
        outputs = list(getattr(metadata, "output_variables", []) or [])
        if sorted(outputs) != sorted(declared_outputs) or len(outputs) != len(set(outputs)):
            problems.append(f"output_variables {outputs} (the provided variables are {list(declared_outputs)})")
        if getattr(metadata, "variable_kind", None) != VariableKind.FORCING:
            problems.append(f"variable_kind {getattr(metadata, 'variable_kind', None)} (expected FORCING)")
        if getattr(metadata, "conflict_resolution_policy", None) != ConflictResolutionPolicy.EXCLUSIVE:
            problems.append(
                f"conflict_resolution_policy {getattr(metadata, 'conflict_resolution_policy', None)} "
                "(expected EXCLUSIVE)"
            )
        if problems:
            raise ValueError("External adapter metadata contradicts the adapter contract: " + "; ".join(problems))

    @property
    def parameter_manager(self):
        """
        Lazy resolver for the parameter manager to avoid import-time side effects.

        Returns
        -------
        ParameterManager
            Active parameter manager instance.
        """
        if self._parameter_manager is None and self._parameter_manager_resolver is not None:
            self._parameter_manager = self._parameter_manager_resolver()
        pm = self._parameter_manager
        if pm is None:
            pm = self.get_parameter_manager()
        if pm is None:
            raise RuntimeError(
                "ParameterManager is required for ExternalAdapter but was not provided. "
                "Pass parameter_manager or parameter_manager_resolver to enable unit and namespace validation."
            )
        self._parameter_manager = pm
        return pm

    def _get_conversion_namespace(self, model_id: Optional[str] = None) -> str:
        """Resolve the declared conversion namespace of the adapter's model.

        Reads ``coupling.external.<model>.conversion_namespace`` (required); a missing entry
        or a namespace the parameter database does not declare raises instead of falling back to a
        default namespace.
        """
        cache_key = model_id or self.model_id
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError("A model id is required to resolve the adapter conversion_namespace.")
        if cache_key in self._conversion_namespace_cache:
            return self._conversion_namespace_cache[cache_key]
        external_namespace = resolve_namespace(f"coupling.external.{cache_key}")
        candidate = self.get_parameter_value(
            "conversion_namespace",
            namespace=external_namespace,
            validate=True,
            expected_type=str,
            required=True,
        )
        resolved = str(resolve_namespace(candidate))
        if not is_declared_namespace(resolved, self.parameter_manager):
            raise KeyError(
                f"Configured conversion_namespace '{resolved}' for {external_namespace} "
                "is not declared by the parameter database."
            )
        self._conversion_namespace_cache[cache_key] = resolved
        return resolved

    def _load_conversion_parameters(
        self,
        parameter_specs: Dict[str, Dict[str, Any]],
        *,
        default_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch load conversion parameters."""
        params: Dict[str, Any] = {}
        for key, spec in parameter_specs.items():
            ns = spec.get("namespace", default_namespace or self._get_conversion_namespace())
            raw_value = self.get_parameter_value(
                key,
                namespace=ns,
                units=spec.get("units"),
                bounds=spec.get("bounds"),
                required=True,
            )
            params[key] = self._coerce_parameter_value(
                key=key,
                value=raw_value,
                cast_to=spec.get("cast_to", float),
                namespace=ns,
            )
        return params

    @staticmethod
    def _coerce_parameter_value(
        key: str,
        value: Any,
        cast_to: Callable[[Any], Any],
        namespace: Optional[str] = None,
    ) -> Any:
        """Coerce scalar parameter values to requested Python types."""
        if value is None:
            raise ValueError(
                f"Required conversion parameter '{namespace}.{key}' resolved to None."
            )
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"Parameter '{namespace}.{key}' must be scalar for cast_to conversion."
                )
            value = value.detach().cpu().item()
        try:
            return cast_to(value)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            target_name = getattr(cast_to, "__name__", str(cast_to))
            raise TypeError(
                f"Parameter '{namespace}.{key}' cannot be cast to {target_name}: {exc}"
            ) from exc

    def _read_field(self, ds: xr.Dataset, external_var: str, internal_var: str) -> xr.DataArray:
        """
        The one read path of an external field: declared source transform, then unit conversion.

        The transform (accumulated, layered, water_depth, water_depth_flux, mixing_ratio or
        earth_relative_wind) is declared in InternalSchema.source_transforms(); fields without a
        declaration go directly to the unit conversion.
        """
        from .schema import InternalSchema

        declaration = InternalSchema().get_source_transform(self.source, external_var)
        if declaration is None:
            field = ds[external_var]
        elif declaration["kind"] == "accumulated":
            field = self._deaccumulate(ds, external_var, declaration)
        elif declaration["kind"] == "layered":
            field = self._layer_water_mass(ds, external_var, declaration)
        elif declaration["kind"] == "water_depth":
            field = self._water_depth_mass(ds, external_var, declaration)
        elif declaration["kind"] == "water_depth_flux":
            field = self._water_depth_flux(ds, external_var)
        elif declaration["kind"] == "mixing_ratio":
            field = self._specific_humidity(ds, external_var, declaration)
        elif declaration["kind"] == "earth_relative_wind":
            field = self._earth_relative_wind(ds, external_var, declaration)
        else:
            raise ValueError(f"Unknown source transform {declaration['kind']!r} for {external_var!r}.")
        return self._apply_unit_conversion(field, internal_var)

    def _deaccumulate(self, ds: xr.Dataset, external_var: str, declaration: Dict[str, Any]) -> xr.DataArray:
        """Rate over each output interval from an accumulation since simulation start."""
        from ....data.standards.unit_converter import ConversionContext, unit_converter
        from ....utils.physical_constants import get_physical_constant

        accumulation = ds[external_var]
        units = accumulation.attrs.get("units")
        if units != declaration["accumulation_units"]:
            raise ValueError(
                f"{external_var} is declared as an accumulation in {declaration['accumulation_units']!r}; "
                f"the file declares {units!r}."
            )
        time_dim = declaration["time_dimension"]
        time_coord = declaration["time_coordinate"]
        if time_dim not in accumulation.dims or time_coord not in ds.coords:
            raise ValueError(
                f"De-accumulating {external_var} needs the time dimension {time_dim!r} and the time "
                f"coordinate {time_coord!r}."
            )
        n_records = accumulation.sizes[time_dim]
        if n_records < 2:
            raise ValueError(f"De-accumulating {external_var} needs at least two time records, got {n_records}.")
        # References: WRF-ARW User's Guide (Skamarock et al. 2019, NCAR/TN-556+STR) namelist
        #   bucket_mm: RAINNC/RAINC accumulate since the simulation start and are reset by the
        #   bucket size with the counters I_RAINNC/I_RAINC.
        # Mathematical expression: A_k = P_k + I_k B (I_k B omitted when no counter is declared or
        #   present); r_k = rho_w (A_k - A_{k-1}) c / (t_k - t_{k-1}) for k = 1..K-1, assigned to
        #   the interval (t_{k-1}, t_k] at t_k; the first record only provides A_0.
        # Parameter meanings: P_k accumulated depth in the file; I_k bucket counter; B bucket size
        #   (file attribute BUCKET_MM); A_k total accumulation; t_k record time; rho_w water density
        #   (physical constant water_density); c depth conversion mm -> m of the unit converter;
        #   r_k mean precipitation mass flux over the interval.
        # Units: P, A, B [mm]; I [1]; t [s]; rho_w [kg m-3]; c [m mm-1]; r [kg m-2 s-1].
        # Domain/constraints: at least two records; strictly increasing times; A must not
        #   decrease (a decrease is a bucket reset without applied counters and raises); a counter
        #   present in the file requires a positive BUCKET_MM.
        # Sign convention: r >= 0, downward mass flux into the surface.
        total = accumulation.astype("float64")
        counter = declaration.get("bucket_counter")
        buckets_applied = False
        if counter is not None and counter in ds.data_vars:
            size_attr = declaration["bucket_size_attribute"]
            bucket_size = ds.attrs.get(size_attr)
            if bucket_size is None or float(bucket_size) <= 0.0:
                raise ValueError(
                    f"{counter} is present but the dataset attribute {size_attr} is missing or not positive."
                )
            total = total + ds[counter].astype("float64") * float(bucket_size)
            buckets_applied = True
        times = np.asarray(ds[time_coord].values)
        seconds = (times[1:] - times[:-1]) / np.timedelta64(1, "s")
        if not np.all(seconds > 0):
            raise ValueError(f"Time records of {external_var} must be strictly increasing.")
        increments = total.diff(time_dim)
        if bool((increments < 0).any()):
            raise ValueError(
                f"The accumulation {external_var} decreases between records (bucket reset); the bucket "
                f"counter {counter!r} with the dataset attribute BUCKET_MM is required to de-accumulate it."
            )
        depth_m = unit_converter.convert(
            np.asarray(increments.values), "generic", units, "m", ConversionContext()
        )
        rho_w = float(get_physical_constant("water_density"))
        interval_shape = [1] * increments.ndim
        interval_shape[increments.dims.index(time_dim)] = seconds.size
        rate = rho_w * np.asarray(depth_m) / seconds.reshape(interval_shape)
        first_time = np.datetime_as_string(times[0], unit="s")
        attrs = dict(accumulation.attrs)
        attrs["units"] = "kg m-2 s-1"
        attrs["deaccumulation"] = (
            f"first record {first_time} consumed as the initial accumulation; rates over "
            f"(t_(k-1), t_k] assigned to t_k; bucket counters applied: {buckets_applied}"
        )
        return xr.DataArray(rate, dims=increments.dims, coords=increments.coords, attrs=attrs)

    def _layer_water_mass(self, ds: xr.Dataset, external_var: str, declaration: Dict[str, Any]) -> xr.DataArray:
        """
        Water mass per unit area of each soil layer from volumetric content and layer thickness.

        The thickness is the declared thickness variable of the output. Its layer dimension may
        be a deeper column than the content (CLM levgrnd around levsoi); the content layers are
        its top layers. Output without the thickness variable needs the thickness declared in the
        adapter configuration field named by ``configured_thickness``; otherwise the read raises.
        """
        from ....utils.physical_constants import get_physical_constant

        content = ds[external_var]
        if content.attrs.get("units") != declaration["content_units"]:
            raise ValueError(
                f"{external_var} is declared in {declaration['content_units']!r}; the file declares "
                f"{content.attrs.get('units')!r}."
            )
        layer_dims = [dim for dim in declaration["layer_dimensions"] if dim in content.dims]
        if len(layer_dims) != 1:
            raise ValueError(
                f"{external_var} must carry exactly one of its declared layer dimensions "
                f"{tuple(declaration['layer_dimensions'])}; its dimensions are {content.dims}."
            )
        layer_dim = layer_dims[0]
        n_layers = content.sizes[layer_dim]
        thickness_var = declaration["thickness_variable"]
        config_field = declaration["configured_thickness"]
        if thickness_var in ds.data_vars:
            thickness = ds[thickness_var]
            if thickness.attrs.get("units") != declaration["thickness_units"]:
                raise ValueError(
                    f"{thickness_var} is declared in {declaration['thickness_units']!r}; the file declares "
                    f"{thickness.attrs.get('units')!r}."
                )
            thickness_dim = declaration["thickness_layer_dimension"]
            if thickness_dim not in thickness.dims:
                raise ValueError(f"{thickness_var} must carry the layer dimension {thickness_dim!r}.")
            if thickness.sizes[thickness_dim] < n_layers:
                raise ValueError(
                    f"{thickness_var} has {thickness.sizes[thickness_dim]} layers along {thickness_dim!r}; "
                    f"{external_var} has {n_layers} layers along {layer_dim!r}."
                )
            # The content layers are the top n_layers of the thickness column.
            thickness = thickness.isel({thickness_dim: slice(0, n_layers)})
            if thickness_dim in thickness.coords:
                thickness = thickness.drop_vars(thickness_dim)
            if thickness_dim != layer_dim:
                thickness = thickness.rename({thickness_dim: layer_dim})
            thickness_source = thickness_var
        else:
            configured = getattr(self.config, config_field) if config_field is not None else None
            if configured is None:
                declared = (
                    f"and the adapter configuration declares no {config_field!r}"
                    if config_field is not None
                    else "and no configured thickness is declared for it"
                )
                raise ValueError(
                    f"{external_var} needs its layer thickness: the file has no {thickness_var!r} {declared}."
                )
            values = np.asarray(configured, dtype=np.float64)
            if values.ndim != 1 or values.size != n_layers:
                raise ValueError(
                    f"The configured {config_field} gives {values.size} thicknesses; {external_var} has "
                    f"{n_layers} layers along {layer_dim!r}."
                )
            thickness = xr.DataArray(values, dims=(layer_dim,))
            thickness_source = f"configuration {config_field}"
        if bool((thickness <= 0).any()):
            raise ValueError(f"Layer thicknesses from {thickness_source} must be positive.")
        # References: Chen and Dudhia (2001) Mon. Weather Rev. 129:569-585 (Noah LSM soil layers
        #   with thicknesses DZS); Lawrence et al. (2019) J. Adv. Model. Earth Syst. 11:4245-4287
        #   (CLM5 soil column, H2OSOI and DZSOI history fields); CF standard name
        #   mass_content_of_water_in_soil_layer.
        # Mathematical expression: m_k = rho_w theta_k dz_k for every soil layer k.
        # Parameter meanings: theta_k volumetric water content of layer k; dz_k layer thickness
        #   (output thickness variable or configured value); rho_w water density (physical
        #   constant water_density); m_k water mass per unit area of layer k.
        # Units: theta [m3 m-3] (mm3/mm3 is the same ratio); dz [m]; rho_w [kg m-3]; m [kg m-2].
        # Domain/constraints: dz_k > 0; the content layers are the top layers of the thickness
        #   column; a configured thickness gives exactly one value per content layer; the range of
        #   m is checked by the catalog quality control.
        # Sign convention: m_k >= 0 for non-negative theta.
        rho_w = float(get_physical_constant("water_density"))
        mass = rho_w * content.astype("float64") * thickness.astype("float64")
        mass = mass.transpose(*content.dims)
        mass.attrs = dict(content.attrs)
        mass.attrs["units"] = "kg m-2"
        mass.attrs["layer_thickness_source"] = thickness_source
        return mass

    @staticmethod
    def _specific_humidity(
        ds: xr.Dataset, external_var: str, declaration: Dict[str, Any]
    ) -> xr.DataArray:
        """Specific humidity from a water vapour mixing ratio (the file declares its units)."""
        mixing_ratio = ds[external_var]
        units = mixing_ratio.attrs.get("units")
        if units != declaration["mixing_ratio_units"]:
            raise ValueError(
                f"{external_var} is declared as a water vapour mixing ratio in "
                f"{declaration['mixing_ratio_units']!r}; the file declares {units!r}."
            )
        # References: WRF module_sf_sfclay (Q2, diagnostic 2 m mixing ratio in kg/kg); AMS Glossary
        #   of Meteorology ("mixing ratio", "specific humidity"); CMAQ science documentation eq.
        #   12-98 (rho_v = r / (1 + r) rho); CF standard name specific_humidity.
        # Mathematical expression: q = w / (1 + w), from w = m_v / m_d and q = m_v / (m_v + m_d).
        # Parameter meanings: w water vapour mixing ratio (mass of water vapour per mass of dry
        #   air); q specific humidity (mass of water vapour per mass of moist air).
        # Units: w [kg kg-1] (the declared units of the file); q [kg kg-1].
        # Domain/constraints: the file declares the declared units (else ValueError); w must be
        #   finite and above -1 (the denominator); the physical range of q (w >= 0) is checked by
        #   the catalog quality control.
        # Sign convention: not applicable: q and w are non-negative mass ratios.
        values = np.asarray(mixing_ratio.values, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= -1.0):
            raise ValueError(
                f"{external_var} holds a mixing ratio that is not finite or not above -1; "
                "q = w / (1 + w) is undefined."
            )
        humidity = mixing_ratio.copy(data=values / (1.0 + values))
        humidity.attrs = dict(mixing_ratio.attrs)
        humidity.attrs["units"] = "kg kg-1"
        humidity.attrs["mixing_ratio_units"] = units
        return humidity

    @staticmethod
    def _earth_relative_wind(
        ds: xr.Dataset, external_var: str, declaration: Dict[str, Any]
    ) -> xr.DataArray:
        """One earth-relative wind component from a grid-relative component pair."""
        names = [declaration[key] for key in ("u_field", "v_field", "cos_field", "sin_field")]
        missing = [name for name in names if name not in ds.variables]
        if missing:
            raise ValueError(
                f"{external_var} is a grid-relative wind component; its earth-relative "
                f"{declaration['component']} component needs {missing} in the file (no rotation is assumed)."
            )
        u, v = ds[declaration["u_field"]], ds[declaration["v_field"]]
        cos_a, sin_a = ds[declaration["cos_field"]], ds[declaration["sin_field"]]
        if u.attrs.get("units") != v.attrs.get("units"):
            raise ValueError(
                f"The wind components {declaration['u_field']} and {declaration['v_field']} declare "
                f"different units ({u.attrs.get('units')!r}, {v.attrs.get('units')!r})."
            )
        for part in (u, v, cos_a, sin_a):
            if not np.all(np.isfinite(np.asarray(part.values, dtype=np.float64))):
                raise ValueError(f"{part.name} holds non-finite values; the wind rotation is undefined.")
        # References: UCAR MMM WRF & MPAS-A Support Forum, "Converting model grid-relative wind to
        #   earth-relative wind" (u-earth = u-model*cosalpha - v-model*sinalpha,
        #   v-earth = v-model*cosalpha + u-model*sinalpha) and "Are U & V wind components grid
        #   relative?" (U10 and V10 are grid-relative); WRF Registry: U10 "U at 10 M", V10 "V at 10 M".
        # Mathematical expression: u_e = u cos(a) - v sin(a); v_e = v cos(a) + u sin(a).
        # Parameter meanings: u, v grid-relative wind components (x and y of the model grid);
        #   cos(a), sin(a) the cosine and sine of the local rotation angle a of the model grid
        #   relative to the earth (WRF COSALPHA, SINALPHA); u_e, v_e eastward and northward components.
        # Units: u, v, u_e, v_e in the file's wind units (converted to m s-1 by the catalog
        #   conversion afterwards); cos(a), sin(a) dimensionless.
        # Domain/constraints: the four fields must be present and finite and u, v must declare one
        #   unit; a Mercator or latitude-longitude grid has cos(a) = 1, sin(a) = 0 in the file.
        # Sign convention: u_e positive eastward, v_e positive northward.
        if declaration["component"] == "eastward":
            rotated = u * cos_a - v * sin_a
            source = u
        elif declaration["component"] == "northward":
            rotated = v * cos_a + u * sin_a
            source = v
        else:
            raise ValueError(f"Unknown wind component {declaration['component']!r} for {external_var!r}.")
        component = source.copy(data=np.asarray(rotated.transpose(*source.dims).values, dtype=np.float64))
        component.attrs = dict(source.attrs)
        component.attrs["rotation"] = (
            f"earth-relative from {declaration['u_field']}, {declaration['v_field']} with "
            f"{declaration['cos_field']}, {declaration['sin_field']}"
        )
        return component

    @staticmethod
    def _declared_units(field: xr.DataArray, external_var: str) -> str:
        """Units attribute of a source field (the file's declaration is authoritative)."""
        units = field.attrs.get("units")
        if not isinstance(units, str) or not units.strip():
            raise ValueError(f"{external_var} has no units attribute; its water depth cannot be converted.")
        return units

    def _water_depth_mass(
        self, ds: xr.Dataset, external_var: str, declaration: Dict[str, Any]
    ) -> xr.DataArray:
        """Water mass per unit area from a liquid-equivalent water depth (per layer when declared)."""
        from ....data.standards.dimensionality import dimensionality_from_unit
        from ....data.standards.unit_converter import unit_converter
        from ....utils.physical_constants import get_physical_constant

        depth = ds[external_var]
        units = self._declared_units(depth, external_var)
        layer_dim = declaration["layer_dimension"]
        if layer_dim is not None and layer_dim not in depth.dims:
            raise ValueError(
                f"{external_var} is declared as a water depth per soil layer along {layer_dim!r}; "
                f"its dimensions are {depth.dims}."
            )
        dimension = dimensionality_from_unit(units)
        if dimension is None or not dimension.is_compatible(dimensionality_from_unit("m")):
            raise ValueError(f"{external_var} is declared as a water depth; its units {units!r} are not a length.")
        # References: Liang et al. (1994) J. Geophys. Res. 99(D7):14415-14428 and Hamman et al.
        #   (2018) Geosci. Model Dev. 11:3481-3496 (VIC water depths per layer and snow water
        #   equivalent in mm); CF standard names mass_content_of_water_in_soil_layer and
        #   surface_snow_amount.
        # Mathematical expression: m = rho_w d (for every layer k when a layer dimension is declared).
        # Parameter meanings: d liquid-equivalent water depth; rho_w water density (physical
        #   constant water_density); m water mass per unit area.
        # Units: d [the file's length units, converted to m by the unit converter]; rho_w [kg m-3];
        #   m [kg m-2].
        # Domain/constraints: the file declares a length unit; a declared layer dimension must be
        #   present; the range of m is checked by the catalog quality control.
        # Sign convention: m >= 0 for non-negative d.
        rho_w = float(get_physical_constant("water_density"))
        depth_m = unit_converter.convert(np.asarray(depth.values, dtype=np.float64), "generic", units, "m")
        mass = depth.copy(data=rho_w * np.asarray(depth_m, dtype=np.float64))
        mass.attrs = dict(depth.attrs)
        mass.attrs["units"] = "kg m-2"
        mass.attrs["water_depth_units"] = units
        return mass

    def _water_depth_flux(self, ds: xr.Dataset, external_var: str) -> xr.DataArray:
        """
        Water mass flux from a water depth per unit time, or from a water depth per record.

        A per-record depth (a length) is divided by each record's interval, taken from the CF
        time bounds of the file's time coordinate; without declared time bounds it raises.
        """
        from ....data.standards.dimensionality import dimensionality_from_unit
        from ....data.standards.unit_converter import unit_converter
        from ....utils.physical_constants import get_physical_constant

        depth = ds[external_var]
        units = self._declared_units(depth, external_var)
        dimension = dimensionality_from_unit(units)
        rho_w = float(get_physical_constant("water_density"))
        values = np.asarray(depth.values, dtype=np.float64)
        if dimension is not None and dimension.is_compatible(dimensionality_from_unit("m s-1")):
            # References: CF Conventions 1.8 section 3.1 (units) and the standard names
            #   precipitation_flux, water_evapotranspiration_flux, surface_runoff_flux.
            # Mathematical expression: F = rho_w v with v the declared depth rate in m s-1.
            # Parameter meanings: v water depth per unit time; rho_w water density (physical
            #   constant water_density); F water mass flux.
            # Units: v [the file's length per time units, converted to m s-1]; rho_w [kg m-3];
            #   F [kg m-2 s-1].
            # Domain/constraints: the file declares a length per time unit.
            # Sign convention: F keeps the sign of v (the direction of the catalog variable).
            rate = unit_converter.convert(values, "generic", units, "m s-1")
            flux = depth.copy(data=rho_w * np.asarray(rate, dtype=np.float64))
            interval_source = "declared rate"
        elif dimension is not None and dimension.is_compatible(dimensionality_from_unit("m")):
            time_dim = self.TIME_DIMENSION
            if time_dim not in depth.dims:
                raise ValueError(
                    f"{external_var} is a water depth per record ({units!r}); converting it to a flux "
                    f"needs the record dimension {time_dim!r}; its dimensions are {depth.dims}."
                )
            bounds_name = ds[time_dim].attrs.get("bounds") if time_dim in ds.coords else None
            if bounds_name is not None:
                if not isinstance(bounds_name, str) or bounds_name not in ds.variables:
                    raise ValueError(
                        f"The time coordinate {time_dim!r} declares bounds {bounds_name!r}, which the "
                        "file does not contain."
                    )
                bounds = ds[bounds_name].values
                if bounds.ndim != 2 or bounds.shape != (ds.sizes[time_dim], 2):
                    raise ValueError(f"Time bounds {bounds_name!r} must have shape ({ds.sizes[time_dim]}, 2).")
                if bounds.dtype.kind == "M":
                    interval_s = (bounds[:, 1] - bounds[:, 0]) / np.timedelta64(1, "s")
                else:
                    time_units = str(ds[time_dim].attrs.get("units", ""))
                    step_units = time_units.split(" since ")[0].strip()
                    interval_s = unit_converter.convert(
                        np.asarray(bounds[:, 1] - bounds[:, 0], dtype=np.float64), "generic", step_units, "s"
                    )
                interval_source = f"time bounds {bounds_name}"
            else:
                interval_value = getattr(self.config, "record_interval", None)
                interval_units = getattr(self.config, "record_interval_units", None)
                if interval_value is None or interval_units is None:
                    raise ValueError(
                        f"{external_var} is a water depth per record ({units!r}); converting it to a flux "
                        f"needs the CF time bounds of {time_dim!r} (bounds attribute) or a declared "
                        "record_interval with record_interval_units in the adapter configuration; the "
                        "file declares no bounds and the configuration declares no record interval."
                    )
                interval_s = np.full(
                    depth.sizes[time_dim],
                    float(unit_converter.convert(float(interval_value), "generic", interval_units, "s")),
                )
                interval_source = f"configuration record_interval {float(interval_value)} {interval_units}"
            interval_s = np.asarray(interval_s, dtype=np.float64)
            if not bool(np.all(interval_s > 0.0)):
                raise ValueError(f"The record intervals of {external_var} ({interval_source}) must be positive.")
            # References: CF Conventions 1.8 sections 7.1 (cell boundaries) and 7.3 (cell_methods
            #   time: sum); Liang et al. (1994) and Neitsch et al. (2011) (VIC and SWAT water
            #   depths summed over each output record).
            # Mathematical expression: F_k = rho_w d_k / (t_k,end - t_k,start) for every record k.
            # Parameter meanings: d_k water depth over record k; t_k,start and t_k,end the record's
            #   CF time bounds, or t_k,end - t_k,start = the declared record_interval when the
            #   file has no bounds; rho_w water density (physical constant water_density); F_k
            #   mean water mass flux over record k.
            # Units: d [the file's length units, converted to m]; t and record_interval [s, from
            #   the declared units]; rho_w [kg m-3]; F [kg m-2 s-1].
            # Domain/constraints: the file's CF time bounds win; the configured record_interval is
            #   used only without them; intervals are positive; no interval is inferred from the
            #   coordinate spacing and without either declaration the conversion raises.
            # Sign convention: F keeps the sign of d (the direction of the catalog variable).
            depth_m = np.asarray(unit_converter.convert(values, "generic", units, "m"), dtype=np.float64)
            interval = xr.DataArray(interval_s, dims=(time_dim,))
            flux = (depth.copy(data=depth_m) * rho_w / interval).transpose(*depth.dims)
        else:
            raise ValueError(
                f"{external_var} is declared as a water depth or depth rate; its units {units!r} are "
                "neither a length nor a length per time."
            )
        flux.attrs = dict(depth.attrs)
        flux.attrs["units"] = "kg m-2 s-1"
        flux.attrs["water_depth_units"] = units
        flux.attrs["flux_interval_source"] = interval_source
        return flux

    def _apply_unit_conversion(self, data_array: xr.DataArray, target_variable: str) -> xr.DataArray:
        """Apply schema-driven unit conversion for adapter outputs.

        The target units come from the external variable catalog (InternalSchema). A variable
        missing from the catalog, a field without a ``units`` attribute, and an unsupported
        conversion raise; data is never returned in unverified units.
        """
        from .schema import InternalSchema

        definition = InternalSchema().get_variable(target_variable)
        if definition is None:
            raise KeyError(
                f"Variable {target_variable!r} is not declared in "
                "coupling.external.variables.variable_definitions."
            )
        current_units = data_array.attrs.get("units")
        if not isinstance(current_units, str) or not current_units.strip():
            raise ValueError(
                f"Field for {target_variable!r} has no units attribute; its conversion to "
                f"{definition.units!r} cannot be verified."
            )
        target_units = definition.units
        if current_units == target_units:
            return data_array

        from ....data.standards.unit_converter import unit_converter

        # Generic unit conversion of the canonical converter; water depths and volumetric
        # contents are converted by their declared source transforms (_read_field), so no
        # soil-depth or other layer context is supplied here.
        converted_values = unit_converter.convert(
            data_array.values,
            "generic",
            current_units,
            target_units,
        )
        converted_da = data_array.copy()
        converted_da.data = converted_values
        converted_da.attrs["units"] = target_units
        converted_da.attrs["original_units"] = current_units
        return converted_da

    def _apply_quality_control(self, tensor: torch.Tensor, var_name: str) -> torch.Tensor:
        """
        Catalog quality control of one external field (shared by all adapters).

        Values outside the catalog valid range become missing (NaN), missing values are never
        filled, the missing fraction is reported in ``tensor._relic_meta['quality_control']``,
        and a field whose missing fraction exceeds ``coupling.external.regrid.missing_threshold``
        is rejected.
        """
        from .schema import InternalSchema

        definition = InternalSchema().get_variable(var_name)
        if definition is None:
            raise KeyError(
                f"Variable {var_name!r} is not declared in "
                "coupling.external.variables.variable_definitions; its valid range is unknown."
            )
        if definition.valid_min is None or definition.valid_max is None:
            raise ValueError(f"Catalog entry of {var_name!r} must declare valid_min and valid_max.")
        max_missing = float(
            self.get_parameter_value(
                "missing_threshold",
                namespace=resolve_namespace(ParameterNamespace.COUPLING_EXTERNAL_REGRID),
                units="dimensionless",
                validate=True,
                required=True,
            )
        )
        # References: CF Conventions 1.8 section 2.5.1 (values outside valid_min/valid_max are
        #   missing); coupling.external.variables.variable_definitions (valid range per variable).
        # Mathematical expression: x_i -> missing if x_i < valid_min or x_i > valid_max;
        #   f_missing = N_missing / N; reject if f_missing > f_max.
        # Parameter meanings: x_i field value; N number of values; N_missing values that are NaN
        #   after the range check; f_max = coupling.external.regrid.missing_threshold.
        # Units: x, valid_min and valid_max in the catalog units of the variable; f [1].
        # Domain/constraints: missing values stay NaN (no fill value is invented); an empty field
        #   has no defined missing fraction and raises.
        # Sign convention: not applicable (a mask, no flux).
        if tensor.numel() == 0:
            raise ValueError(f"Field {var_name!r} is empty; quality control is undefined.")
        valid_min = float(definition.valid_min)
        valid_max = float(definition.valid_max)
        out_of_range = (tensor < valid_min) | (tensor > valid_max)
        checked = torch.where(out_of_range, torch.full_like(tensor, float("nan")), tensor)
        missing_fraction = float(torch.isnan(checked).sum().item()) / float(checked.numel())
        if missing_fraction > max_missing:
            raise ValueError(
                f"Field {var_name!r} has a missing fraction {missing_fraction:.3f} (NaN or outside "
                f"[{valid_min}, {valid_max}]) above coupling.external.regrid.missing_threshold "
                f"{max_missing:.3f}."
            )
        meta = dict(getattr(tensor, "_relic_meta", {}) or {})
        meta["quality_control"] = {
            "missing_fraction": missing_fraction,
            "out_of_valid_range": int(out_of_range.sum().item()),
            "valid_range": (valid_min, valid_max),
        }
        setattr(checked, "_relic_meta", meta)
        if missing_fraction > 0.0:
            self.logger.info(
                "Variable %s: missing fraction %.3f kept as NaN (%d values outside [%s, %s])",
                var_name,
                missing_fraction,
                int(out_of_range.sum().item()),
                valid_min,
                valid_max,
            )
        return checked

    def _transform_fields(
        self,
        data: Dict[str, torch.Tensor],
        target_grid: Optional[Dict] = None,
        variable_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Shared transform of external fields to RELIC internal format.

        For every field: regrid when ``target_grid`` is given (basis from ``variable_metadata``
        or the external variable catalog), apply the catalog quality control, then the tensor
        properties. Failures propagate; a field is never dropped or passed through silently.
        """
        transformed_data: Dict[str, torch.Tensor] = {}
        for var_name, tensor in data.items():
            if target_grid:
                caller_metadata = None
                if variable_metadata is not None and var_name in variable_metadata:
                    caller_metadata = variable_metadata[var_name]
                tensor = self._apply_regridding(tensor, target_grid, variable_metadata=caller_metadata)
            tensor = self._apply_quality_control(tensor, var_name)
            transformed_data[var_name] = self._ensure_tensor_properties(tensor)
        return transformed_data

    # Dimension of the record axis in the adapters' read outputs (WRF overrides with 'Time').
    TIME_DIMENSION = "time"

    def _record_time_metadata(self, field: xr.DataArray) -> Dict[str, Any]:
        """Decoded record times and calendar of a field along TIME_DIMENSION (for its read metadata).

        The time coordinate is the coordinate named TIME_DIMENSION, else the one named 'time'
        along TIME_DIMENSION, else the only datetime coordinate along it; without one the
        entries are None (a read at the simulation time then raises). The calendar is the
        calendar of cftime values, else the declared calendar attribute or encoding, else the
        CF-1.8 default 'standard' of datetime64 values without a declared calendar.
        """
        dim = self.TIME_DIMENSION
        if dim not in field.dims:
            return {}

        def _is_time(coord: xr.DataArray) -> bool:
            if tuple(coord.dims) != (dim,):
                return False
            if coord.dtype.kind == "M":
                return True
            if coord.dtype.kind in "fiu":
                # An undecoded CF time coordinate: numbers with "<unit> since <date>" units.
                units = coord.attrs.get("units")
                return isinstance(units, str) and " since " in units
            return coord.dtype.kind == "O" and coord.size > 0 and isinstance(coord.values[0], cftime.datetime)

        coordinates = {str(name): coord for name, coord in field.coords.items() if _is_time(coord)}
        if dim in coordinates:
            coord = coordinates[dim]
        elif "time" in coordinates:
            coord = coordinates["time"]
        elif len(coordinates) == 1:
            coord = next(iter(coordinates.values()))
        else:
            return {"record_times": None, "record_calendar": None}
        values = coord.values
        if coord.dtype.kind in "fiu":
            # CF-1.8 section 4.4: decode the numbers with their units and calendar (default
            # 'standard' when none is declared); cftime returns dates of that calendar.
            calendar = coord.attrs.get("calendar") or "standard"
            values = cftime.num2date(values, units=coord.attrs["units"], calendar=canonical_cf_calendar(calendar))
            values = np.atleast_1d(values)
        elif coord.dtype.kind == "O":
            calendar = values[0].calendar
        else:
            calendar = coord.encoding.get("calendar") or coord.attrs.get("calendar") or "standard"
            values = pd.to_datetime(values).to_pydatetime()
        return {"record_times": tuple(values), "record_calendar": str(calendar)}

    def _close_datasets(self, *state: Any) -> None:
        """Close every opened dataset, clear the adapter state, and raise the collected close errors.

        Every dataset is attempted (a failure of one does not keep the others open), the opened
        datasets and the given state containers are cleared, and the close errors, if any, are
        raised together; a failed close never reads as success.
        """
        errors: List[str] = []
        first: Optional[BaseException] = None
        for key, dataset in list(self.current_datasets.items()):
            try:
                dataset.close()
            except Exception as exc:  # noqa: BLE001 - cleanup of every resource; raised below (rule 4)
                errors.append(f"{key}: {type(exc).__name__}: {exc}")
                first = first or exc
        self.current_datasets.clear()
        for container in state:
            container.clear()
        if errors:
            raise RuntimeError(
                f"{self.name}: closing the opened datasets failed: " + "; ".join(errors)
            ) from first

    @staticmethod
    def _concat_by_record_time(datasets: List[xr.Dataset], sources: List[str], dim: str = "time") -> xr.Dataset:
        """Concatenate the datasets of one output stream in the order of their decoded record times."""
        # References: CF-1.8 section 4.4 (time coordinate); xarray.concat along the record axis.
        # Mathematical expression: order the files by T_(f,0), the first decoded record time of file
        #   f, concatenate along dim, and require T_1 < T_2 < ... < T_N over the combined records.
        # Parameter meanings: T_(f,j) decoded record time j of file f (datetime64 or cftime); N the
        #   number of combined records.
        # Units: record times are dates of the dataset calendar (no unit conversion here).
        # Domain/constraints: every file needs a decoded time coordinate along dim (else ValueError);
        #   file names and modification times never order the records; overlapping or repeated
        #   records raise ValueError instead of being concatenated.
        # Sign convention: not applicable (ordering only).
        if len(datasets) == 1:
            return datasets[0]
        keyed = []
        for dataset, source in zip(datasets, sources):
            coordinate = dataset.coords.get(dim)
            if coordinate is None or coordinate.size == 0 or coordinate.dtype.kind not in "MO":
                raise ValueError(
                    f"{source}: no decoded time coordinate {dim!r}; the records of a multi-file output "
                    "are ordered by their record times, never by file names or modification times."
                )
            keyed.append((coordinate.values[0], dataset))
        keyed.sort(key=lambda item: item[0])
        combined = xr.concat([dataset for _, dataset in keyed], dim=dim, data_vars="minimal", coords="minimal")
        times = combined[dim].values
        if times.size > 1 and not bool(np.all(times[1:] > times[:-1])):
            raise ValueError(
                f"The record times of {sorted(sources)} are not strictly increasing after ordering the files "
                "by their first record: the files overlap or repeat records."
            )
        return combined

    def _declared_time_interpolation(self) -> str:
        """The configured time_interpolation of this adapter (required for a read at time t)."""
        rule = getattr(self.config, "time_interpolation", None)
        if rule is None:
            raise ValueError(
                f"{self.name}: a forcing read at the simulation time needs the configuration's "
                f"time_interpolation, one of {TIME_INTERPOLATIONS}; none is assumed."
            )
        validate_time_interpolation(rule)
        return rule

    def _read_records_at(
        self,
        time: float,
        *,
        calendar_type: Optional[str],
        time_origin: Optional[str],
        variables: Optional[List[str]] = None,
        **read_kwargs: Any,
    ) -> Dict[str, UpdateItem]:
        """
        Every catalog field at the simulation time t from the single read path, as updates.

        ``read`` applies the schema mapping, the declared source transforms and the unit
        conversion. A field with TIME_DIMENSION is evaluated at time_origin + t in the
        simulator calendar by the configured time_interpolation (forcing_record_weights); a
        field without it is returned whole. The result is a function of t alone.
        """
        # References: CDEPS/E3SM shr_tInterp_getFactors (see forcing_record_weights); CF-1.8
        #   section 4.4 (time coordinate, units "seconds since <origin>", calendar).
        # Mathematical expression: s_j = date2num(T_j, "seconds since " + time_origin, C) for the
        #   record times T_j of every field; y_n(t) = sum_(k, w) w X_n[record k] with
        #   (k, w) = forcing_record_weights(s, t, rule).
        # Parameter meanings: X_n field n from read(); T_j decoded record times; C the simulator
        #   calendar_type (the dataset calendar must be C, CF synonyms aside); time_origin the
        #   date and time of t = 0 in C; rule the configured time_interpolation.
        # Units: t and s in s; y_n keeps the catalog units of field n.
        # Domain/constraints: time_origin, calendar_type and time_interpolation are required
        #   (ValueError when absent); a dataset calendar other than C raises (no conversion); a
        #   time-dependent field without a decoded time coordinate raises; t outside the records
        #   raises; no read position is kept between calls.
        # Sign convention: not applicable: data provision, no flux direction is introduced.
        rule = self._declared_time_interpolation()
        parse_time_origin(time_origin, calendar_type)
        calendar = canonical_cf_calendar(calendar_type)
        units = f"seconds since {time_origin}"
        # The provided variables, or the requested subset of them; a request outside them raises.
        requested = list(self.provided_variables) if variables is None else [str(name) for name in variables]
        outside = sorted(set(requested) - set(self.provided_variables))
        if outside:
            raise ValueError(
                f"{self.name}: {outside} are not provided variables of this adapter "
                f"{list(self.provided_variables)}."
            )
        tensors = self.read(variables=requested, **read_kwargs)
        if sorted(tensors) != sorted(requested):
            raise ValueError(
                f"{self.name}: the read returned the fields {sorted(tensors)} for the requested variables "
                f"{sorted(requested)} (a history file set with several streams needs one declared stream)."
            )
        records: Dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            meta = getattr(tensor, "_relic_meta", None)
            if meta is None:
                raise ValueError(f"Field {name!r} carries no read metadata.")
            dims = tuple(meta["dims"])
            if self.TIME_DIMENSION not in dims:
                record = tensor.to(self.device)
                setattr(record, "_relic_meta", dict(meta))
                records[name] = record
                continue
            times = meta.get("record_times")
            if times is None:
                raise ValueError(
                    f"{self.name}: field {name!r} has no decoded time coordinate along "
                    f"{self.TIME_DIMENSION!r}; its records cannot be placed at the simulation time."
                )
            dataset_calendar = canonical_cf_calendar(meta["record_calendar"])
            if dataset_calendar != calendar:
                raise ValueError(
                    f"{self.name}: field {name!r} uses the {meta['record_calendar']!r} calendar and the "
                    f"simulator the {calendar_type!r} calendar; no calendar conversion is performed."
                )
            seconds = np.asarray(cftime.date2num(list(times), units=units, calendar=calendar), dtype=np.float64)
            weights = forcing_record_weights(seconds.tolist(), float(time), rule)
            axis = dims.index(self.TIME_DIMENSION)
            if len(weights) == 1:
                record = tensor.select(axis, weights[0][0]).to(self.device)
            else:
                record = sum(weight * tensor.select(axis, index) for index, weight in weights).to(self.device)
            record_meta = {key: value for key, value in meta.items() if key != "record_times"}
            record_meta["dims"] = dims[:axis] + dims[axis + 1:]
            record_meta["shape"] = record.shape
            record_meta["record_indices"] = [index for index, _ in weights]
            record_meta["record_weights"] = [weight for _, weight in weights]
            record_meta["time_interpolation"] = rule
            record_meta["simulation_time_seconds"] = float(time)
            setattr(record, "_relic_meta", record_meta)
            records[name] = record
        return self._wrap_updates(records)

    @staticmethod
    def _apply_adapter_attributes(
        ds: xr.Dataset, table: Dict[str, Dict[str, Any]], source: str
    ) -> List[str]:
        """
        Fill the attributes a file does not declare from an adapter table; the file's win.

        Every data variable records ``units_source``: 'file' when the file declares its units,
        'adapter_table' when the table supplied them (the variable is then listed in the return
        value). Variables without a table entry get a generic long_name and source when absent.
        """
        filled: List[str] = []
        for name, var in ds.data_vars.items():
            declared = var.attrs.get("units")
            file_units = isinstance(declared, str) and bool(declared.strip())
            entry = table.get(name)
            if entry is None:
                var.attrs.setdefault("long_name", name.replace("_", " ").title())
                var.attrs.setdefault("source", source)
            else:
                for key, value in entry.items():
                    if key == "units":
                        continue
                    var.attrs.setdefault(key, value)
                if not file_units and "units" in entry:
                    var.attrs["units"] = entry["units"]
                    filled.append(str(name))
            if file_units:
                var.attrs["units_source"] = "file"
            elif str(name) in filled:
                var.attrs["units_source"] = "adapter_table"
        return filled

    def _present_mapped_fields(
        self,
        datasets: Dict[str, Any],
        internal_to_external: Dict[str, str],
        requested: Optional[List[str]],
    ) -> Dict[str, List[str]]:
        """
        {dataset key: internal names whose mapped field the dataset contains}, in request order.

        With explicitly requested variables, a variable whose mapped field no opened dataset
        contains raises. With ``requested`` None every mapped name is looked up; the mapped
        fields a dataset lacks are never skipped silently: ``last_read_report`` lists them per
        dataset (``absent_mapped_fields``: {dataset: {internal: external}}) and they are logged.
        """
        names = list(internal_to_external) if requested is None else list(requested)
        present = {
            key: [name for name in names if internal_to_external[name] in getattr(dataset, "data_vars", {})]
            for key, dataset in datasets.items()
        }
        absent = {
            key: {name: internal_to_external[name] for name in names if name not in present[key]}
            for key in datasets
        }
        if requested is not None:
            missing = [name for name in names if not any(name in listed for listed in present.values())]
            if missing:
                raise ValueError(
                    f"{self.name}: the requested variable(s) {missing} map to "
                    f"{[internal_to_external[name] for name in missing]}, which no opened dataset "
                    f"({sorted(datasets)}) contains."
                )
        self.last_read_report = {
            "requested": None if requested is None else names,
            "absent_mapped_fields": {key: fields for key, fields in absent.items() if fields},
        }
        if self.last_read_report["absent_mapped_fields"]:
            logger.info(
                "%s: mapped fields absent from the opened datasets: %s",
                self.name,
                self.last_read_report["absent_mapped_fields"],
            )
        return present

    @staticmethod
    def _require_adapter_config(config: Any, config_type: type, adapter: str) -> None:
        """
        Reject a ``config`` that is not the adapter's own configuration dataclass.

        The adapter ``config`` is a domain object (for example CESMConfig), not the FunctorBase
        configuration mapping; a mapping such as {"dependencies": [...]} would otherwise be kept
        as the domain configuration and its dependency declaration dropped silently. Functor
        dependencies of an adapter are declared with add_dependency or set_dependencies.
        """
        if config is not None and not isinstance(config, config_type):
            raise ValueError(
                f"{adapter} config must be a {config_type.__name__} or None, got "
                f"{type(config).__name__}; Functor dependencies are declared with add_dependency "
                "or set_dependencies, not through the adapter config."
            )

    def _ensure_tensor_properties(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        float32 tensor with the ``_relic_meta`` record kept.

        The received tensor is never modified: a dtype change returns a new tensor that stays
        connected to it in the autograd graph, and gradient tracking is whatever the caller set
        (``requires_grad`` is never toggled).
        """
        if tensor.dtype == torch.float32:
            return tensor
        meta = getattr(tensor, "_relic_meta", None)
        converted = tensor.to(torch.float32)
        if meta is not None:
            setattr(converted, "_relic_meta", meta)
        return converted

    def _cf_horizontal_coordinates(self, ds: xr.Dataset, field: xr.DataArray) -> Optional[Dict[str, Any]]:
        """
        CF latitude/longitude coordinates of a field and their cell bounds.

        Coordinates are identified by their CF attributes (regrid.cf_horizontal_coordinates:
        units with consistent standard_name and axis), never by their names. 1-D (rectilinear)
        coordinates take their bounds from the variables named by their ``bounds`` attribute
        (for example lat_bnds/lon_bnds); a bounds entry is None when the file declares none.
        2-D (curvilinear) coordinates carry no bounds. Returns None when the field has no CF
        latitude and longitude coordinates on a rectilinear or curvilinear layout.
        """
        from .regrid import cf_horizontal_coordinates

        roles = cf_horizontal_coordinates(field.coords)
        if set(roles) != {"lat", "lon"}:
            return None
        lat = field.coords[roles["lat"]]
        lon = field.coords[roles["lon"]]
        rectilinear = lat.dims == (roles["lat"],) and lon.dims == (roles["lon"],)
        curvilinear = lat.ndim == 2 and lat.dims == lon.dims
        if not (rectilinear or curvilinear):
            return None
        record: Dict[str, Any] = {
            "lat_coordinate": roles["lat"],
            "lon_coordinate": roles["lon"],
        }
        for role, name in roles.items():
            bounds_name = field.coords[name].attrs.get("bounds")
            if curvilinear or bounds_name is None:
                record[f"{role}_bounds"] = None
                continue
            if bounds_name not in ds.variables:
                raise ValueError(
                    f"Coordinate {name!r} declares bounds {bounds_name!r}, which the file does not contain."
                )
            record[f"{role}_bounds"] = torch.as_tensor(np.asarray(ds[bounds_name].values, dtype=np.float64))
        return record

    def _apply_regridding(
        self,
        tensor: torch.Tensor,
        target_grid: Dict,
        variable_metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Regrid one external field with the RegridOperator (shared by all adapters).

        The source grid comes from the field's lat/lon coordinates, the RegridConfig from the
        declared spatial basis (caller metadata or the external variable catalog). Failures
        propagate.
        """
        meta = getattr(tensor, '_relic_meta', None)
        if meta is None:
            raise ValueError("Regridding requires tensor._relic_meta with grid coordinates.")
        coords = meta.get('coords', {})
        coord_dims = meta.get('coord_dims')
        if not isinstance(coord_dims, dict) or set(coords) - set(coord_dims):
            raise ValueError(
                "Regridding requires coord_dims for every coordinate in tensor metadata."
            )
        variable = meta.get('variable')
        if not isinstance(variable, str):
            raise ValueError("Regridding requires the internal variable name in tensor metadata.")

        def _numpy(value):
            return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value

        source_da = xr.DataArray(
            _numpy(tensor),
            dims=meta['dims'],
            coords={
                name: (tuple(coord_dims[name]), _numpy(coord))
                for name, coord in coords.items()
            },
            attrs=meta['attrs'],
        )
        regrid_config = self._regrid_config(variable, variable_metadata)
        from .regrid import RegridMethod

        # The source grid is the explicit descriptor of the field's CF latitude/longitude
        # coordinates (identified at read time by their CF attributes, never by name).
        horizontal = meta.get('cell_bounds')
        if horizontal is None:
            raise ValueError(
                f"Regridding {variable!r} needs its CF latitude and longitude coordinates (units "
                "degrees_north / degrees_east on a rectilinear or curvilinear grid); the field "
                "declares none, and coordinates are not identified by their names."
            )
        missing = [role for role in ("lat", "lon") if horizontal.get(f"{role}_bounds") is None]
        if regrid_config.method is RegridMethod.CONSERVATIVE and missing:
            # Conservative remapping integrates over the cells: their CF bounds must be declared
            # in the file (lat_bnds/lon_bnds); they are not inferred from the coordinate spacing.
            raise ValueError(
                f"Conservative regridding of {variable!r} needs the CF cell bounds of its "
                f"{missing} coordinates (bounds attribute, e.g. lat_bnds/lon_bnds); cell bounds "
                "are not inferred from the coordinate spacing."
            )
        source_grid: Dict[str, Any] = {
            "lat": _numpy(coords[horizontal["lat_coordinate"]]),
            "lon": _numpy(coords[horizontal["lon_coordinate"]]),
        }
        if not missing:
            source_grid["lat_bounds"] = _numpy(horizontal["lat_bounds"])
            source_grid["lon_bounds"] = _numpy(horizontal["lon_bounds"])
        regrid_result = self.regrid_operator.regrid(
            source_data=source_da,
            source_grid=source_grid,
            target_grid=target_grid,
            config=regrid_config,
        )
        regridded_da = regrid_result.data
        regridded_tensor = torch.from_numpy(regridded_da.values.astype(np.float32)).to(tensor.device)

        new_meta = meta.copy()
        new_meta['shape'] = regridded_tensor.shape
        new_meta['dims'] = regridded_da.dims
        new_meta['coords'] = {
            name: torch.from_numpy(np.asarray(coord.values).astype(np.float32))
            for name, coord in regridded_da.coords.items()
            if coord.dtype.kind in 'fc'
        }
        new_meta['coord_dims'] = {
            name: tuple(coord.dims)
            for name, coord in regridded_da.coords.items()
            if coord.dtype.kind in 'fc'
        }
        new_meta['regridded'] = True
        new_meta['regrid_method'] = regrid_config.method.value
        new_meta['spatial_basis'] = regrid_config.spatial_basis.value
        new_meta['conservation_error'] = regrid_result.conservation_error
        new_meta['target_grid'] = target_grid
        setattr(regridded_tensor, '_relic_meta', new_meta)

        self.logger.debug(
            f"Regridded {variable} ({regrid_config.spatial_basis.value}) using "
            f"{regrid_config.method.value}"
        )
        return regridded_tensor

    def _regrid_config(
        self,
        variable: str,
        variable_metadata: Optional[Dict[str, Any]] = None,
    ) -> "RegridConfig":
        """
        Build the RegridConfig of one external field from its declared spatial basis.

        The basis and the positivity (valid_min >= 0) come from ``variable_metadata`` (caller)
        or from the external variable catalog ``coupling.external.variables.variable_definitions``;
        an undeclared variable raises ``KeyError`` and caller metadata without valid_min raises
        ``ValueError``. Names, standard names and units are never used to infer either.
        """
        from .regrid import RegridConfig, RegridMethod
        from .schema import InternalSchema
        from ...scales.types import SpatialQuantityBasis, resolve_spatial_basis

        if variable_metadata is not None:
            spatial_basis = resolve_spatial_basis(variable, variable_metadata)
            if variable_metadata.get("valid_min") is None:
                raise ValueError(
                    f"Caller metadata of {variable!r} must declare valid_min; the positivity of the "
                    "regridded field is taken from it."
                )
            # References: CF Conventions 1.8 section 2.5.1 (valid_min/valid_max define the valid
            #   range of a variable); coupling.external.variables.variable_definitions.
            # Mathematical expression: positive_definite = (valid_min >= 0).
            # Parameter meanings: valid_min is the smallest physically valid value in the catalog.
            # Units: valid_min carries the units of the variable; the result is a boolean.
            # Domain/constraints: valid_min must be declared; a missing value raises.
            # Sign convention: a non-negative valid_min forbids negative values of the field.
            positive_definite = float(variable_metadata["valid_min"]) >= 0.0
        else:
            definition = InternalSchema().get_variable(variable)
            if definition is None:
                raise KeyError(
                    f"Variable {variable!r} is not declared in "
                    "coupling.external.variables.variable_definitions; pass variable_metadata "
                    "with spatial_basis and valid_min to regrid it."
                )
            spatial_basis = definition.spatial_basis
            positive_definite = definition.positive_definite

        # References: Jones (1999) Mon. Weather Rev. 127:2204-2210 (first-order conservative
        #   remapping); docs/source/theory/scale_coupling.md section 2.3.2 (spatial basis).
        # Mathematical expression: method = conservative if basis in {cell_total, area_density}
        #   (the remapped amount equals the covered source amount); nearest if basis =
        #   categorical; bilinear if basis = intensive.
        # Parameter meanings: basis is the declared SpatialQuantityBasis of the field.
        # Units: the field keeps its catalog units (for example W m-2, kg m-2 s-1, K).
        # Domain/constraints: the basis is declared by the caller or the external variable
        #   catalog; the conservation check runs for amount bases only.
        # Sign convention: remapping preserves the sign of the field values.
        if spatial_basis.is_extensive:
            method = RegridMethod.CONSERVATIVE
        elif spatial_basis is SpatialQuantityBasis.CATEGORICAL:
            method = RegridMethod.NEAREST
        else:
            method = RegridMethod.BILINEAR
        return RegridConfig(
            spatial_basis=spatial_basis,
            method=method,
            check_conservation=spatial_basis.is_extensive,
            positive_definite=positive_definite,
        )

    def upscale(self, fine_data: torch.Tensor) -> torch.Tensor:
        """External adapters do not provide generic upscale operations."""
        raise NotImplementedError("ExternalAdapter does not implement upscale; use specific regridding utilities.")

    def downscale(self, coarse_data: torch.Tensor) -> torch.Tensor:
        """External adapters do not provide generic downscale operations."""
        raise NotImplementedError("ExternalAdapter does not implement downscale; use specific regridding utilities.")
    
    # Abstract interface - must be implemented by subclasses
    
    @abc.abstractmethod
    def discover(self, source: PathLike) -> Dict[str, Any]:
        """
        Discover variables and metadata in external data source.
        
        Scans the source (file, directory, or URL) to identify:
        - Available variables and their metadata
        - Spatial and temporal coverage 
        - Grid specifications and coordinate systems
        - Dataset CF conformance status fixed to unevaluated
        - Data format and access patterns
        
        Parameters:
            source: Path or URL to external data source
            
        Returns:
            Dict containing discovered metadata:
            {
                'variables': List of available variable names,
                'dimensions': Coordinate dimension information,
                'time_range': [start_time, end_time],
                'spatial_bounds': [lon_min, lon_max, lat_min, lat_max], 
                'grid_info': Grid specification details,
                'cf_version': CF convention version,
                'source_format': File format ('netcdf4', 'hdf5', etc.),
                'estimated_size': Approximate data volume in bytes
            }
        """
        pass
    
    @abc.abstractmethod 
    def open(self, source: PathLike, config: Dict[str, Any]) -> None:
        """
        Open connection to external data source.
        
        Establishes data access handles/connections. For file-based sources,
        opens file handles with appropriate chunking strategies. For remote
        sources, establishes network connections with authentication.
        
        Parameters:
            source: Path or URL to external data source
            config: Configuration dictionary with options:
                - 'chunks': Chunking strategy for large datasets
                - 'cache_size': Local cache size limit in bytes  
                - 'parallel': Enable parallel I/O operations
                - 'auth': Authentication credentials for remote access
                
        Raises:
            IOError: If source cannot be accessed
            ValueError: If source format is not supported
        """
        pass
    
    @abc.abstractmethod
    def read(self, 
             time_window: Optional[TimeWindow] = None,
             spatial_window: Optional[SpatialWindow] = None, 
             variables: Optional[List[str]] = None,
             **kwargs) -> Union[Dataset, Dict[str, Tensor]]:
        """
        Read the requested variables of the data opened by open().

        Concrete adapters return every record of the schema-mapped fields of the
        opened data as PyTorch tensors carrying their read metadata; the data were
        subset once at open() by the time_window and spatial_window given at
        construction. The window arguments of this abstract signature select no
        data per call.

        Parameters:
            time_window: Temporal subset specification (not applied per call)
            spatial_window: Spatial subset specification (not applied per call)
            variables: List of variable names to extract
            **kwargs: Additional read options
            
        Returns:
            xarray.Dataset with explicitly preserved metadata or
            Dict[str, torch.Tensor] with standardized variable names
            
        Raises:
            KeyError: If requested variables not available
            ValueError: If time/spatial windows are invalid
        """
        pass
    
    @abc.abstractmethod
    def transform_to_internal(self,
                            data: Union[Dataset, Dict[str, Any]],
                            mapping: Dict[str, str]) -> Dict[str, Tensor]:
        """
        Transform external data to RELIC internal format.

        Concrete adapters apply the shared ``_transform_fields``: spatial regridding
        to a target grid on the declared basis when one is given, the catalog
        quality control (out-of-range values become NaN and are reported; no gap
        filling) and the tensor properties. Variable names and units are already
        internal after read() (schema mapping, declared source transforms, unit
        conversion); no temporal resampling or aggregation is performed.
        
        Parameters:
            data: External data (xarray Dataset or raw dict)
            mapping: Variable name mapping {external_name: internal_name}
            
        Returns:
            Dictionary of PyTorch tensors with internal variable names:
            {
                'air_temperature': torch.Tensor,  # [K]
                'precipitation_flux': torch.Tensor,  # [kg m-2 s-1] 
                'shortwave_down': torch.Tensor,  # [W m-2]
                ...
            }
            
        Raises:
            ValueError: If a field cannot be regridded or checked (nothing is passed
            through or filled)
        """
        pass
    
    @abc.abstractmethod
    def close(self) -> None:
        """
        Close connection and clean up resources.
        
        Properly closes file handles, network connections, and deallocates
        any cached data. Should be idempotent (safe to call multiple times).
        """
        pass
    
    # Common utility methods
    
    def validate_data(self, data: Union[Dataset, Dict[str, Any]]) -> ValidationReport:
        """
        Comprehensive data validation and quality assessment.
        
        Checks bounded metadata, performs range validation, identifies missing
        data, and assesses data quality metrics.
        
        Parameters:
            data: Data to validate
            
        Returns:
            Detailed validation report
        """
        from .validators import Validator
        validator = Validator()
        return validator.validate(data, source=self.source)
    
    def get_variable_mapping(self, target_variables: List[str]) -> Dict[str, str]:
        """
        Generate variable name mapping from external to internal names.
        
        Parameters:
            target_variables: List of desired internal variable names
            
        Returns:
            Mapping dictionary {external_name: internal_name}

        Raises:
            KeyError: Naming every requested variable the source does not provide
        """
        from .schema import InternalSchema
        schema = InternalSchema()
        return schema.create_mapping(self.source, target_variables)
    
    def estimate_memory_usage(self, 
                            time_window: TimeWindow,
                            spatial_window: SpatialWindow, 
                            variables: List[str]) -> int:
        """
        Estimate memory usage for requested data subset.
        
        Parameters:
            time_window: Temporal extent
            spatial_window: Spatial extent
            variables: Variables to load
            
        Returns:
            Estimated memory usage in bytes
        """
        raise NotImplementedError(
            "ExternalAdapter.estimate_memory_usage is abstract in strict runtime. "
            "Concrete adapter must implement source-specific memory estimation."
        )
    
    # Context manager support
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()
    
    # Property accessors
    
    @property
    def connected(self) -> bool:
        """Check if adapter is connected to data source."""
        return self._connected
    
    @property
    def validation_report(self) -> Optional[ValidationReport]:
        """Get most recent validation report."""
        return self._validation_report
    
    # Restart contract (F02) of the external source binding.
    # References: torch.nn.Module get_extra_state/set_extra_state (state_dict extra state);
    #   ExternalAdapter._read_records_at, forcing_record_weights and the forward methods of the
    #   CESM, WRF, CMIP6, VIC and SWAT adapters.
    # Mathematical expression: binding B = (source, model_id); a restore accepts B_ckpt only if
    #   B_ckpt = B_adapter. Every forward returns y_n(t) = sum_(k, w) w X_n[record k] with
    #   (k, w) = forcing_record_weights(s, t, rule), s the record times in seconds since
    #   time_origin: a function of the simulation time t alone, so no read position exists
    #   and none is checkpointed; a restored run at time t reads the records of t.
    # Parameter meanings: source the DataSource value; model_id the adapter's model
    #   identifier; X_n field n from the single read path; time_origin (simulator setting) the
    #   date and time of t = 0 in the simulator calendar_type; rule the configured
    #   time_interpolation ('lower' or 'linear').
    # Units: B is identity text (dimensionless); t and s in s; y_n keeps the catalog units of
    #   field n.
    # Domain/constraints: a checkpoint of another source or model raises ValueError on
    #   restore; connection handles and cached datasets are resources, not science state, and
    #   are re-opened by the owner after restore; the time_origin and calendar_type of the
    #   restored run come from its simulator settings; the simulator's forcing cursor
    #   (bind_forcing_cursor) belongs to supplied forcing sequences and is independent of this read.
    # Sign convention: not applicable: identity binding and data provision, no flux direction.
    EXTRA_STATE_FORMAT = "relic.external_adapter.source_binding"
    EXTRA_STATE_VERSION = 1

    def get_extra_state(self) -> Dict[str, Any]:
        """Serialize the external source binding for the restart contract."""
        return {
            "format": self.EXTRA_STATE_FORMAT,
            "version": self.EXTRA_STATE_VERSION,
            "source": str(self.source.value),
            "model_id": str(self.model_id),
        }

    def set_extra_state(self, state: Any) -> None:
        """Validate a restored source binding before accepting it.

        The binding must name this adapter's source and model; a checkpoint
        produced by an adapter bound elsewhere is refused instead of being
        re-attached to a different data source.
        """
        expected_keys = {"format", "version", "source", "model_id"}
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError(
                f"External adapter '{self.name}' extra state must carry exactly "
                f"{sorted(expected_keys)}"
            )
        if state["format"] != self.EXTRA_STATE_FORMAT:
            raise ValueError(
                f"External adapter '{self.name}' extra state format "
                f"{state['format']!r} is not {self.EXTRA_STATE_FORMAT!r}"
            )
        if state["version"] != self.EXTRA_STATE_VERSION:
            raise ValueError(
                f"External adapter '{self.name}' extra state version "
                f"{state['version']!r} is not {self.EXTRA_STATE_VERSION!r}"
            )
        bound = (str(self.source.value), str(self.model_id))
        restored = (state["source"], state["model_id"])
        if restored != bound:
            raise ValueError(
                f"External adapter '{self.name}' is bound to source/model "
                f"{bound}; the checkpoint was produced for {restored}"
            )

    def _wrap_updates(self, data: Dict[str, torch.Tensor]) -> Dict[str, UpdateItem]:
        """Updates of provided variables with the explicit update mode and the catalog units."""
        undeclared = sorted(set(data) - set(self.provided_units))
        if undeclared:
            raise ValueError(f"{self.name}: {undeclared} are not provided variables of this adapter.")
        return {
            name: UpdateItem(value=value, mode=self.update_mode, unit=self.provided_units[name])
            for name, value in data.items()
        }
    
    def forward(self,
                state: StateDict,
                time: float,
                dt: float,
                **kwargs) -> Dict[str, Tensor]:
        """
        Forward pass contract of an external data adapter (template).

        The concrete adapters (CESM, WRF, CMIP6, VIC, SWAT) override it as
        ``forward(state, time, dt, *, calendar_type=None, time_origin=None, **kwargs)`` and return
        every schema-mapped field of the opened data at the simulation time ``time`` through the
        single read path (``_read_records_at``: schema mapping, declared source transforms, unit
        conversion, then the configured time_interpolation at time_origin + time in the simulator
        calendar); ``kwargs["variables"]`` restricts the internal variable names. The simulator
        passes ``calendar_type`` and ``time_origin`` from its settings; they are required. No read
        position is kept, and state and dt are not read.

        Parameters:
            state : StateDict
                Current system state (not read)
            time : float
                Current simulation time [s] since time_origin
            dt : float
                Time step duration [s] (not read)
            **kwargs : Dict
                ``variables``: optional list of internal variable names to return

        Returns:
            Dict[str, UpdateItem]: replace updates keyed by internal catalog
            variable names

        Raises:
            NotImplementedError: always, in this template

        Mathematical Expression:
            y_n(t) = sum_(k, w) w X_n[record k], (k, w) = forcing_record_weights(s, t, rule)
        """
        # References: RELIC coupling adapter contract (this module: ExternalAdapter.read, _read_field
        #   and _read_records_at); no process equation is evaluated here.
        # Mathematical expression: no computation in this template; the concrete adapters return
        #   y_n(t) = sum_(k, w) w X_n[record k] with X_n = convert(transform(D[m(n)])) and
        #   (k, w) = forcing_record_weights(s, t, rule) for every mapped internal variable n.
        # Parameter meanings: D opened dataset; m(n) external name of internal variable n; transform
        #   the declared source transform; convert the catalog unit conversion; s record times in
        #   seconds since time_origin; rule the configured time_interpolation.
        # Units: catalog units of every internal variable; time [s] since time_origin; dt [s] is
        #   accepted by the interface and not read.
        # Domain/constraints: this template always raises NotImplementedError; the concrete
        #   adapters keep no read position (the restart contract binds only source and model_id).
        # Sign convention: not applicable: data provision, no flux direction is introduced here.
        raise NotImplementedError(
            "ExternalAdapter.forward is template-only in strict runtime. "
            "Concrete adapters must override forward()."
        )




