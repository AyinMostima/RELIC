"""
RELIC Scientific Modules - Domain-Specific Ecosystem Processes

This package provides scientific process implementations across all major
ecosystem spheres, following the four-pillar abstraction design and unified
functor architecture.

Scientific Domains:
    atmosphere     - Atmospheric processes (radiation, turbulence, chemistry, transport)
    biosphere      - Vegetation processes (photosynthesis, respiration, growth, phenology, BVOC)
    hydrosphere    - Water cycle (ET, soil moisture, runoff, groundwater, snow)
    pedosphere     - Soil processes (decomposition, nutrient cycling, carbon pools)
    anthroposphere - Human activities (land use, management, emissions)
    isotopes       - Isotope tracking and fractionation
    transport      - Unified transport operators and boundary conditions


Design Principles:
    - All processes inherit from appropriate bridge classes (BiologicalProcessFunctor, etc.)
    - Parameters managed through UnifiedParameterManager
    - Conservation laws explicitly declared and validated
    - Units follow CF-1.8 conventions with automatic conversion
    - Automatic differentiation enabled through PyTorch tensors

Usage:
    >>> from relic.scientific import FUNCTOR_REGISTRY, get_functor_path
    >>> from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
    >>> get_functor_path("farquhar_photosynthesis")
    'relic.scientific.biosphere.photosynthesis.farquhar:FarquharPhotosynthesis'
    >>> photosynthesis = FarquharPhotosynthesis()
    >>> updates = photosynthesis(state, time=0.0, dt=3600.0)  # inputs come from the state

References:
    - Farquhar et al. (1980). Photosynthesis model. Planta.
    - Monteith & Unsworth (2013). Principles of Environmental Physics.
    - Bonan (2019). Climate Change and Terrestrial Ecosystem Modeling.
""" 
import importlib
import logging
import os
from typing import Dict, Type

from ..utils.imports import module_available

FUNCTOR_REGISTRY: Dict[str, str] = {}
logger = logging.getLogger(__name__)

# Command-line entry modules follow one naming rule: the module (file) name
# ends with "_cli" ("<name>_cli.py"). They are run with "python -m", configure
# logging and print only inside main(), define no Functor, and are never
# imported by the registry walk: runpy warns about, and executes a second copy
# of, a module that an earlier import already put in sys.modules.
CLI_ENTRY_MODULE_SUFFIX = "_cli"


def is_cli_entry_module(module_name: str) -> bool:
    """True when ``module_name`` (a dotted module name or a file stem) is a CLI entry module."""
    return module_name.rpartition(".")[2].endswith(CLI_ENTRY_MODULE_SUFFIX)


def _import_module(module_path: str):
    """Import a required module using explicit availability checks."""
    if not module_available(module_path):
        raise ImportError(f"Required module not found: {module_path}")
    return importlib.import_module(module_path)


def _to_snake_case(name: str) -> str:
    """Convert CamelCase names to snake_case."""
    if not name:
        return ""
    output: list[str] = []
    for idx, char in enumerate(name):
        if char.isupper() and idx > 0:
            output.append("_")
        output.append(char.lower())
    return "".join(output)


def register_functor(cls: Type) -> Type:
    """Register a functor class into the registry."""
    if cls is None:
        return cls
    name = getattr(cls, "FUNCTOR_NAME", None)
    if not name:
        meta = getattr(cls, "metadata", None)
        if meta is not None:
            name = getattr(meta, "name", None)
    if not name:
        name = getattr(cls, "__name__", None)

    if not name:
        return cls

    path = f"{cls.__module__}:{cls.__name__}"
    keys = [name, name.lower()]
    snake_from_class = _to_snake_case(getattr(cls, "__name__", name))
    if snake_from_class:
        keys.append(snake_from_class)
    # One binding per name: re-registering the same class is a no-op, binding a
    # name to another class is a registry error (checked before any write).
    for key in keys:
        bound = FUNCTOR_REGISTRY.get(key)
        if bound is not None and bound != path:
            raise ValueError(
                f"Functor name '{key}' is already registered to '{bound}'; "
                f"cannot bind it to '{path}'."
            )
    for key in keys:
        FUNCTOR_REGISTRY[key] = path
    return cls


def register_all_functors() -> None:
    """Register every FunctorBase subclass found in the scientific subpackages.

    This is the registry bootstrap. It runs once at the end of this module's
    import, before any lookup; calling it again is idempotent. Command-line
    entry modules (``is_cli_entry_module``) are not imported: they define no
    Functor and are executed only by ``python -m``.
    """
    from ..core.functor import FunctorBase

    import pkgutil

    for _, module_name, _ in pkgutil.walk_packages(
        path=__path__,
        prefix=__name__ + ".",
    ):
        if is_cli_entry_module(module_name):
            continue
        module = _import_module(module_name)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not isinstance(attr, type):
                continue
            if attr is FunctorBase:
                continue
            try:
                if issubclass(attr, FunctorBase):
                    register_functor(attr)
            except TypeError:
                continue


# ============================================================================
# LAZY SPHERE IMPORTS - Main domain packages
# ============================================================================
_SCIENTIFIC_MODULES = {
    "atmosphere": "relic.scientific.atmosphere",
    "biosphere": "relic.scientific.biosphere",
    "hydrosphere": "relic.scientific.hydrosphere",
    "pedosphere": "relic.scientific.pedosphere",
    "anthroposphere": "relic.scientific.anthroposphere",
    "isotopes": "relic.scientific.isotopes",
    "transport": "relic.scientific.transport",
}

_lazy_scientific_enabled = os.environ.get("RELIC_LAZY_IMPORT", "1").lower() in {"1", "true", "yes"}

if not _lazy_scientific_enabled:
    for _name, _path in _SCIENTIFIC_MODULES.items():
        globals()[_name] = _import_module(_path)
else:
    def __getattr__(name: str):
        if name in _SCIENTIFIC_MODULES:
            module = _import_module(_SCIENTIFIC_MODULES[name])
            globals()[name] = module
            return module
        raise AttributeError(f"module 'relic.scientific' has no attribute '{name}'")



# ============================================================================
# COMMON FUNCTORS - Frequently used implementations
# ============================================================================

# Biosphere functors (most commonly used)
from .biosphere.photosynthesis import FarquharPhotosynthesis as FarquharFunctor
from .biosphere.respiration.growth_respiration import (
    GrowthRespiration as GrowthRespirationFunctor,
)
from .biosphere.respiration.maintenance_respiration import (
    MaintenanceRespiration as MaintenanceRespirationFunctor,
)
from .biosphere.bvoc import MEGANModel as BVOCEmissionFunctor

# Atmosphere functors
from .atmosphere.radiation import RadiationFunctor
from .atmosphere.turbulence import TurbulenceFunctor

# Hydrosphere functors
from .hydrosphere.evapotranspiration import PenmanMonteithET
from .hydrosphere.soil_moisture import RichardsSoilMoisture

# Pedosphere functors
from .pedosphere.decomposition import (
    FirstOrderDecomposition as CENTURYDecompositionFunctor,
)
from .pedosphere.nutrient_cycling import (
    NitrogenCycling as NitrogenCycleFunctor,
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Default single-writer hydrosphere selections used by common registration path.
DEFAULT_ET_FUNCTOR = PenmanMonteithET
DEFAULT_SOIL_MOISTURE_FUNCTOR = RichardsSoilMoisture


def get_functor_path(name: str) -> str:
    """Return the ``module:Class`` path registered for ``name``.

    The name is looked up as given, lowercased and in snake case. The registry
    is complete at import (see the bootstrap at the end of this module), so a
    miss raises KeyError instead of triggering any scan.
    """
    for key in (name, name.lower(), _to_snake_case(name)):
        path = FUNCTOR_REGISTRY.get(key)
        if path is not None:
            return path
    raise KeyError(f"No functor is registered under the name '{name}'.")


def list_available_functors() -> dict:
    """
    List all available scientific functors by domain.
    
    Returns
    -------
    dict
        Dictionary mapping domain names to lists of available functor classes
    """
    available = {
        'biosphere': [],
        'atmosphere': [],
        'hydrosphere': [],
        'pedosphere': [],
        'anthroposphere': [],
        'isotopes': [],
        'transport': [],
    }
    
    # Check biosphere
    if FarquharFunctor is not None:
        available['biosphere'].append('FarquharFunctor')
    if MaintenanceRespirationFunctor is not None:
        available['biosphere'].append('MaintenanceRespirationFunctor')
    if BVOCEmissionFunctor is not None:
        available['biosphere'].append('BVOCEmissionFunctor')
    
    # Check atmosphere
    if RadiationFunctor is not None:
        available['atmosphere'].append('RadiationFunctor')
    if TurbulenceFunctor is not None:
        available['atmosphere'].append('TurbulenceFunctor')
    
    # Check hydrosphere
    if PenmanMonteithET is not None:
        available['hydrosphere'].append('PenmanMonteithET')
    if RichardsSoilMoisture is not None:
        available['hydrosphere'].append('RichardsSoilMoisture')
    
    # Check pedosphere
    if CENTURYDecompositionFunctor is not None:
        available['pedosphere'].append('CENTURYDecompositionFunctor')
    if NitrogenCycleFunctor is not None:
        available['pedosphere'].append('NitrogenCycleFunctor')
    
    return available


def print_scientific_overview():
    """Print overview of available scientific modules."""
    overview = """
    RELIC Scientific Modules Overview
    ====================================
    
    Atmosphere:
      - Radiation: Solar and longwave radiation balance
      - Turbulence: Surface energy fluxes and momentum
      - Chemistry: Atmospheric chemistry reactions
      - Transport: Advection and diffusion in atmosphere
    
    Biosphere:
      - Photosynthesis: C3/C4 carbon assimilation (Farquhar model)
      - Respiration: Maintenance and growth respiration
      - Growth: Biomass allocation and development
      - Phenology: Seasonal cycles and leaf dynamics
      - BVOC: Biogenic volatile organic compound emissions
      - Stress: Water, temperature, ozone stress responses
    
    Hydrosphere:
      - Evapotranspiration: Penman-Monteith, Priestley-Taylor
      - Soil Moisture: Richards equation, water balance
      - Runoff: Surface and subsurface flow
      - Groundwater: Aquifer dynamics
      - Snow: Snowpack accumulation and melt
    
    Pedosphere:
      - Decomposition: CENTURY model, litter breakdown
      - Nutrient Cycling: N, P, C cycles and transformations
      - Carbon Pools: SOC dynamics and stabilization
    
    Anthroposphere:
      - Land Use: Land cover changes and transitions
      - Management: Fertilization, irrigation, harvesting
      - Emissions: Agricultural and urban emissions
    
    Isotopes:
      - Tracking: 13C, 18O, 2H isotope tracking
      - Fractionation: Kinetic and equilibrium fractionation
    
    Transport:
      - Operators: Gradient, divergence, Laplacian
      - Advection: Wind-driven transport
      - Diffusion: Turbulent and molecular diffusion
      - Boundary Conditions: Dirichlet, Neumann, Robin
    
    All modules follow unified architecture with:
    - Parameter management via ParameterManager
    - Conservation law validation
    - CF-1.8 unit conventions
    - PyTorch automatic differentiation
    """
    logger.info("%s", overview)

    logger.info("Currently Available Functors:")
    available = list_available_functors()
    for domain, functors in available.items():
        if functors:
            logger.info("  %s:", domain.capitalize())
            for functor in functors:
                logger.info("    - %s", functor)


# ============================================================================
# PUBLIC API EXPORTS
# ============================================================================

__all__ = [
    # Domain packages
    'atmosphere',
    'biosphere',
    'hydrosphere',
    'pedosphere',
    'anthroposphere',
    'isotopes',
    'transport',
    
    # Common biosphere functors
    'FarquharFunctor',
    'MaintenanceRespirationFunctor',
    'GrowthRespirationFunctor',
    'BVOCEmissionFunctor',
    
    # Common atmosphere functors
    'RadiationFunctor',
    'TurbulenceFunctor',
    
    # Common hydrosphere functors
    'PenmanMonteithET',
    'RichardsSoilMoisture',
    
    # Common pedosphere functors
    'CENTURYDecompositionFunctor',
    'NitrogenCycleFunctor',
    
    # Utility functions
    'list_available_functors',
    'print_scientific_overview',
    'register_all_functors',
]


# Registry bootstrap: every FunctorBase subclass under relic.scientific is
# registered once here, before any lookup, so name resolution never depends on
# process history. Extensions add names only through register_functor().
register_all_functors()
