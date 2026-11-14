"""
Central registry for parameter namespaces.

This enum acts as the single source of truth for namespaced parameter keys
used across the codebase. Using enums reduces typos in string literals and
improves refactor safety while keeping consistency with existing string-based
APIs.
"""

from enum import Enum
from typing import Any, Optional, Union


class ParameterNamespace(str, Enum):
    """Canonical parameter namespaces used by the ParameterManager."""

    # Coupling defaults and internal couplings
    COUPLING_DEFAULTS = "coupling.scales.defaults"
    COUPLING_SCALES_DEFAULTS = "coupling.scales.defaults"
    COUPLING_CAPABILITIES = "coupling.capabilities"
    COUPLING_BIOSPHERE_ATMOSPHERE = "coupling.scales.biosphere_atmosphere"
    COUPLING_SOIL_VEGETATION = "coupling.scales.soil_vegetation"
    COUPLING_WATER_CARBON = "coupling.scales.water_carbon"
    COUPLING_AUDIT = "coupling.scales.audit"
    COUPLING_HIERARCHY = "coupling.scales.hierarchy"
    COUPLING_SCALES_HIERARCHY = "coupling.scales.hierarchy"
    COUPLING_TEMPORAL = "coupling.scales.temporal"
    COUPLING_SCALES_TEMPORAL = "coupling.scales.temporal"
    COUPLING_SPATIAL = "coupling.scales.spatial"
    COUPLING_SCALES_SPATIAL = "coupling.scales.spatial"
    COUPLING_SCALES_CONSERVATIVE = "coupling.scales.conservative"
    COUPLING_SCALES_PHYSICAL = "coupling.scales.physical"
    COUPLING_EXTERNAL_IO = "coupling.external.io"
    COUPLING_EXTERNAL_WRF = "coupling.external.wrf"
    COUPLING_EXTERNAL_CESM = "coupling.external.cesm"
    COUPLING_EXTERNAL_CMIP6 = "coupling.external.cmip6"
    COUPLING_EXTERNAL_SWAT = "coupling.external.swat"
    COUPLING_EXTERNAL_VIC = "coupling.external.vic"
    COUPLING_EXTERNAL_REGRID = "coupling.external.regrid"
    COUPLING_EXTERNAL_VARIABLES = "coupling.external.variables"
    COUPLING_EXTERNAL_TIME = "coupling.external.time"
    COUPLING_CONSERVATION = "validation.conservation"
    COUPLING_VARIABLES = "coupling.variables"
    VALIDATION_NUMERICAL = "validation.numerical"
    CONVERSION_CONTEXT = "conversion_context"

    # Atmosphere
    ATMOSPHERE_MOST = "atmosphere.most"
    ATMOSPHERE_TRANSPORT = "atmosphere.transport"

    # Cross-sphere parameter namespaces
    PEDOSPHERE_NUTRIENT_CYCLING = "pedosphere.nutrient_cycling"
    PEDOSPHERE_VAN_GENUCHTEN = "pedosphere.van_genuchten"
    PEDOSPHERE_STATE_DEFAULTS = "pedosphere.state_defaults"
    HYDROSPHERE_EVAPOTRANSPIRATION = "hydrosphere.evapotranspiration"
    BIOSPHERE_WATER_BALANCE_SOIL_HYDRAULIC = "biosphere.water_balance.soil.hydraulic"

    # Validation
    VALIDATION_CONSERVATION_RADIATION = "validation.conservation.radiation"

    # Transport
    TRANSPORT_NUMERICAL = "transport.numerical"

    # Physical constants
    PHYSICAL_CONSTANTS = "relic.physical.constants"


def resolve_namespace(namespace: Union[str, ParameterNamespace, None]) -> Union[str, None]:
    """
    Convert a namespace enum to its string value while keeping string inputs untouched.

    Parameters
    ----------
    namespace : Union[str, ParameterNamespace, None]
        Input namespace value.

    Returns
    -------
    Optional[str]
        Resolved namespace string or None.
    """
    if namespace is None:
        return None
    if isinstance(namespace, ParameterNamespace):
        return namespace.value
    return namespace


def is_declared_namespace(namespace: str, manager: Optional[Any] = None) -> bool:
    """
    Whether a parameter file of ``manager``'s database declares ``namespace``.

    True when a participating file declares the namespace itself, an ancestor
    or a descendant of it (``ParameterDatabase.declares_namespace``). Without
    ``manager`` the process-wide manager is consulted.
    """
    if manager is None:
        from .manager import get_global_manager

        manager = get_global_manager()
    return manager.database.declares_namespace(namespace)
