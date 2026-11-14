"""
RELIC Coupling Layer - Multi-Scale and Multi-Model Integration

This package provides coupling capabilities for the Recomposable Ecosystem Library for Integrated Coupling (RELIC), including:
- Internal coupling between model components
- External coupling with climate and atmospheric models
- Spatial scale coupling and regridding
- Temporal scale coupling and aggregation/disaggregation
- Bounded CF metadata propagation through the canonical data-standard owner

Architecture Components:
    - internal/: Internal coupling between RELIC components
      (biosphere-atmosphere, soil-vegetation, water-carbon cycles)
    - external/: External coupling with WRF, CESM, and other models
    - scales/: Scale coupling for temporal and spatial transformations
    - core/: Core coupling infrastructure and protocols
    - capability_registry: Feature capability tracking and error codes

Design Principles:
    1. Unified Interfaces: All couplers follow standardized protocols
    2. Conservation: Mass, energy, and momentum conservation across scales
    3. CF metadata scope: bounded variable/time metadata, not dataset conformance
    4. Modularity: Loose coupling with clear interfaces
    5. Performance: Optimized for large-scale simulations

Mathematical Foundation:
    - Conservative remapping algorithms (Jones 1999)
    - Temporal aggregation with statistical methods
    - Spatial interpolation (bilinear, conservative, nearest neighbor)
    - Scale coupling transformations

References:
    - Jones, P.W. (1999). First- and Second-Order Conservative Remapping Schemes
    - CF Conventions 1.8: http://cfconventions.org/
    - SCRIP: Spherical Coordinate Remapping and Interpolation Package
    - ESMF: Earth System Modeling Framework

Version: 1.0.0 (Phase 1 - Core Coupling Framework)
"""

# Architecture note (Section 17/19):
# This package is infrastructure code (coupling/orchestration), not scientific
# Functor forward() execution code. Parameter access via
# get_global_manager().get(...) is therefore allowed here and in coupling
# submodules. Scientific Functor methods must continue to use
# self.get_parameter_value(...).

# Standard library imports
import importlib
import importlib.util
import logging
from typing import Any, Dict, List, Optional

# Parameter namespaces (resolved once to avoid mixed enum/string usage)
from ..parameters.namespaces import ParameterNamespace, resolve_namespace

BA_NAMESPACE = ParameterNamespace.COUPLING_BIOSPHERE_ATMOSPHERE
SV_NAMESPACE = ParameterNamespace.COUPLING_SOIL_VEGETATION
WC_NAMESPACE = ParameterNamespace.COUPLING_WATER_CARBON
CONSERVATION_NAMESPACE = ParameterNamespace.COUPLING_CONSERVATION
DEFAULTS_NAMESPACE = ParameterNamespace.COUPLING_DEFAULTS
SCALES_HIERARCHY_NS = ParameterNamespace.COUPLING_SCALES_HIERARCHY
SCALES_TEMPORAL_NS = ParameterNamespace.COUPLING_SCALES_TEMPORAL
SCALES_SPATIAL_NS = ParameterNamespace.COUPLING_SCALES_SPATIAL
SCALES_DEFAULTS_NS = ParameterNamespace.COUPLING_SCALES_DEFAULTS
SCALES_PHYSICAL_NS = ParameterNamespace.COUPLING_SCALES_PHYSICAL

# String aliases for logging/configuration; prefer enum constants in code paths.
BA_NAMESPACE_STR = resolve_namespace(BA_NAMESPACE)
SV_NAMESPACE_STR = resolve_namespace(SV_NAMESPACE)
WC_NAMESPACE_STR = resolve_namespace(WC_NAMESPACE)
CONSERVATION_NAMESPACE_STR = resolve_namespace(CONSERVATION_NAMESPACE)
DEFAULTS_NAMESPACE_STR = resolve_namespace(DEFAULTS_NAMESPACE)
SCALES_HIERARCHY_NS_STR = resolve_namespace(SCALES_HIERARCHY_NS)
SCALES_TEMPORAL_NS_STR = resolve_namespace(SCALES_TEMPORAL_NS)
SCALES_SPATIAL_NS_STR = resolve_namespace(SCALES_SPATIAL_NS)
SCALES_DEFAULTS_NS_STR = resolve_namespace(SCALES_DEFAULTS_NS)
SCALES_PHYSICAL_NS_STR = resolve_namespace(SCALES_PHYSICAL_NS)

# Import internal coupling modules
from . import internal
from .internal import (
    BiosphereAtmosphereCoupling,
    RootDistributionParams,
    SoilVegetationCoupling,
    SoilWaterStressParams,
    get_coupling_execution_order,
    get_default_configs,
    register_coupling_variables,
)
from .internal import (
    create_coupling_functor as _create_internal_coupling_functor,
)
from .internal import (
    get_available_couplings as _get_internal_couplings,
)

# Import external coupling modules
logger = logging.getLogger(__name__)
_external_unavailable_reason: str = ""
from . import external

EXTERNAL_AVAILABLE = True

# Import scale coupling modules with key classes
# Import core coupling infrastructure
from . import core, scales

# Import capability registry
from .capability_registry import (
    CapabilityDeclaration,
    CouplingCapabilityRegistry,
    CouplingErrorCode,
    create_not_implemented_error,
    is_external_available,
    register_external_availability,
)
from .scales import (
    AggregationConfig,
    AggregationMethod,
    AggregationResult,
    AutoRegridder,
    BilinearRegridder,
    CalendarHandler,
    ConservationLevel,
    ConservativeRegridder,
    DisaggregationConfig,
    DisaggregationMethod,
    DisaggregationResult,
    # Spatial scaling
    GridDescriptor,
    GridUtils,
    NearestNeighborRegridder,
    QualityFlags,
    ScalingDirection,
    SpatialQuantityBasis,
    SpatialRegridderFactory,
    TemporalAggregator,
    TemporalConfig,
    TemporalDisaggregator,
    TemporalScaling,
    # Temporal scaling
    TemporalScalingBase,
    TemporalSemantics,
)

# Synchronize external availability status with capability registry
register_external_availability(
    EXTERNAL_AVAILABLE,
    reason=_external_unavailable_reason,
)


# Version and metadata
__version__ = "1.0.0"
__phase__ = "Phase 1 - Coupling Framework"
__author__ = "Yuchong Long"
__license__ = "MIT"

_NAMESPACE_EXPORTS = [
    # Namespace constants
    "BA_NAMESPACE",
    "SV_NAMESPACE",
    "WC_NAMESPACE",
    "CONSERVATION_NAMESPACE",
    "DEFAULTS_NAMESPACE",
    "SCALES_HIERARCHY_NS",
    "SCALES_TEMPORAL_NS",
    "SCALES_SPATIAL_NS",
    "SCALES_DEFAULTS_NS",
    "SCALES_PHYSICAL_NS",
    "BA_NAMESPACE_STR",
    "SV_NAMESPACE_STR",
    "WC_NAMESPACE_STR",
    "CONSERVATION_NAMESPACE_STR",
    "DEFAULTS_NAMESPACE_STR",
    "SCALES_HIERARCHY_NS_STR",
    "SCALES_TEMPORAL_NS_STR",
    "SCALES_SPATIAL_NS_STR",
    "SCALES_DEFAULTS_NS_STR",
    "SCALES_PHYSICAL_NS_STR",
]

# Helper functions for coupling management
def check_coupling_capabilities(feature: str) -> bool:
    """
    Check if a coupling feature is implemented.

    Parameters:
        feature : str
            Feature name from capability registry

    Returns:
        bool: True if feature is fully implemented
    """
    if CouplingCapabilityRegistry is None:
        return False
    return bool(CouplingCapabilityRegistry.is_implemented(feature))

def get_coupling_alternatives(feature: str) -> List[str]:
    """
    Get alternative approaches for unimplemented features.

    Parameters:
        feature : str
            Feature name from capability registry

    Returns:
        List of alternative approaches
    """
    if CouplingCapabilityRegistry is None:
        return []
    alternatives = CouplingCapabilityRegistry.get_alternatives(feature)
    if isinstance(alternatives, list):
        return [str(item) for item in alternatives]
    return []

def list_implemented_coupling_features() -> List[str]:
    """
    List all implemented coupling features.

    Returns:
        List of implemented feature names
    """
    if CouplingCapabilityRegistry is None:
        return []
    capabilities = CouplingCapabilityRegistry.list_capabilities("implemented")
    if isinstance(capabilities, list):
        return [str(item) for item in capabilities]
    return []


def check_external_coupling_available() -> bool:
    """
    Indicate whether optional external coupling adapters are importable.

    Returns
    -------
    bool
        True if external adapters are available, False otherwise.
    """
    return EXTERNAL_AVAILABLE and is_external_available()


def _detect_plugin_available() -> bool:
    """Plugin package detection without importing plugin modules.

    Only an ImportError means the plugin package is not installed; any other error
    of the probe is a defect and propagates.
    """
    try:
        return importlib.util.find_spec(f"{__name__}.plugin") is not None
    except ImportError:
        return False


PLUGIN_AVAILABLE = _detect_plugin_available()


def get_available_couplings(include_plugin: bool = True) -> Dict[str, type]:
    """
    Return available coupling functors.

    Parameters
    ----------
    include_plugin : bool, optional
        If True, include the language-agnostic plugin adapter.
    """
    couplings = dict(_get_internal_couplings())
    if include_plugin and PLUGIN_AVAILABLE:
        try:
            from .plugin import PluginFunctorAdapter
            couplings["plugin"] = PluginFunctorAdapter
        except ImportError as exc:
            # Only a missing plugin module or name means "unavailable"; other errors propagate.
            logger.warning("Plugin adapter unavailable in current environment: %s", exc)
    return couplings


def create_coupling_functor(
    coupling_type: str,
    name: str,
    config: Optional[Dict[str, Any]],
    parameter_manager: Optional[Any] = None,
):
    """
    Factory function to create coupling functors.

    Supports all built-in internal coupling types and the plugin adapter type.
    """
    config_dict = dict(config or {})
    normalized_type = str(coupling_type).strip().lower()
    if normalized_type in {"plugin", "coupler_plugin", "external_plugin"}:
        if not PLUGIN_AVAILABLE:
            raise RuntimeError("Plugin coupling is unavailable in the current environment.")
        try:
            from .plugin import PluginFunctorAdapter
        except ImportError as exc:
            raise RuntimeError(f"Plugin coupling import failed: {exc}") from exc

        plugin_path = config_dict.get("plugin_path") or config_dict.get("path")
        if not plugin_path:
            raise ValueError(
                "Plugin coupling requires 'plugin_path' (or 'path') in config."
            )
        plugin_config = dict(config_dict)
        plugin_config.setdefault("plugin_path", str(plugin_path))
        return PluginFunctorAdapter(
            name=name,
            plugin_path=str(plugin_path),
            config=plugin_config,
        )
    return _create_internal_coupling_functor(
        coupling_type=normalized_type,
        name=name,
        config=config_dict,
        parameter_manager=parameter_manager,
    )


def __getattr__(name: str):
    """
    Lazily resolve plugin symbols to avoid import cycles during core bootstrap.
    """
    if name == "plugin":
        return importlib.import_module(".plugin", __name__)
    if name in {"PluginHost", "PluginFunctorAdapter"}:
        module = importlib.import_module(".plugin", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Export all public components
_PUBLIC_EXPORTS = [
    # Subpackages
    'internal',
    'external',
    'scales',
    'core',
    'plugin',
    'BiosphereAtmosphereCoupling',
    'SoilVegetationCoupling',
    'PluginHost',
    'PluginFunctorAdapter',
    'PLUGIN_AVAILABLE',
    'RootDistributionParams',
    'SoilWaterStressParams',
    'get_available_couplings',
    'create_coupling_functor',
    'get_default_configs',
    'get_coupling_execution_order',
    'register_coupling_variables',

    # Temporal scaling classes
    'TemporalScalingBase',
    'TemporalScaling',
    'TemporalConfig',
    'TemporalSemantics',
    'ScalingDirection',
    'SpatialQuantityBasis',
    'ConservationLevel',
    'TemporalAggregator',
    'AggregationMethod',
    'AggregationConfig',
    'AggregationResult',
    'TemporalDisaggregator',
    'DisaggregationMethod',
    'DisaggregationConfig',
    'DisaggregationResult',
    'QualityFlags',
    'CalendarHandler',

    # Spatial scaling classes
    'GridDescriptor',
    'GridUtils',
    'ConservativeRegridder',
    'BilinearRegridder',
    'NearestNeighborRegridder',
    'SpatialRegridderFactory',
    'AutoRegridder',

    # Capability registry
    'CouplingErrorCode',
    'CapabilityDeclaration',
    'CouplingCapabilityRegistry',
    'create_not_implemented_error',
    'register_external_availability',
    'is_external_available',

    # Helper functions
    'check_coupling_capabilities',
    'get_coupling_alternatives',
    'list_implemented_coupling_features',
    'check_external_coupling_available',

    # Metadata
    '__version__',
    '__phase__',
    '__author__',
    '__license__'
]

# Merge namespace exports with public API and de-duplicate while preserving order
__all__ = list(dict.fromkeys(_NAMESPACE_EXPORTS + _PUBLIC_EXPORTS))

# Keep explicit re-export references to satisfy static analyzers.
_COUPLING_REEXPORTS = (
    internal,
    external,
    scales,
    core,
    BiosphereAtmosphereCoupling,
    SoilVegetationCoupling,
    RootDistributionParams,
    SoilWaterStressParams,
    get_available_couplings,
    create_coupling_functor,
    get_default_configs,
    get_coupling_execution_order,
    register_coupling_variables,
    TemporalScalingBase,
    TemporalScaling,
    TemporalConfig,
    TemporalSemantics,
    ScalingDirection,
    SpatialQuantityBasis,
    ConservationLevel,
    TemporalAggregator,
    AggregationMethod,
    AggregationConfig,
    AggregationResult,
    TemporalDisaggregator,
    DisaggregationMethod,
    DisaggregationConfig,
    DisaggregationResult,
    QualityFlags,
    CalendarHandler,
    GridDescriptor,
    GridUtils,
    ConservativeRegridder,
    BilinearRegridder,
    NearestNeighborRegridder,
    SpatialRegridderFactory,
    AutoRegridder,
    CouplingErrorCode,
    CapabilityDeclaration,
    CouplingCapabilityRegistry,
    create_not_implemented_error,
    register_external_availability,
    is_external_available,
)
del _COUPLING_REEXPORTS
