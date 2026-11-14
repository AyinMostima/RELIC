"""
RELIC: Recomposable Ecosystem Library for Integrated Coupling

A unified, composable framework for ecosystem biogeochemical modeling based on
category theory principles and four-pillar abstraction design.

Core Philosophy:
    - Abstraction-Driven: Four pillars (Physical, Biological, Transport, Scaling)
    - Composition-First: Category theory for modular ecosystem components
    - End-to-End Differentiability: PyTorch-based automatic differentiation
    - Conservation-Enforced: Mass, energy, momentum validation

Architecture:
    relic.core          - Core framework (FunctorBase, Simulator, StateManager)
    relic.scientific    - Scientific process modules (bio/atmo/hydro/pedo/anthropo)
    relic.coupling      - Internal/external coupling and multi-scale integration
    relic.data          - I/O, standards, quality control, transformations
    relic.parameters    - Unified parameter management with units/validation
    relic.validation    - Multi-level testing and conservation checks
    relic.optimization  - Calibration, assimilation, sensitivity analysis
    relic.computation   - GPU/distributed/parallel computation optimization
    relic.stochastic    - Stochastic processes and uncertainty quantification
    relic.interfaces    - CLI, API, web, and notebook interfaces
    relic.extensions    - Third-party plugins and user-defined functors
    relic.monitoring    - Prometheus metrics, Grafana dashboards, execution logging
    relic.models        - Basin / patch / landscape entities, surrogate helpers
    relic.contracts     - Variable contracts and domain constraints
    relic.utils         - Physical constants, tensor helpers, response functions

Quick Start:
    >>> import relic
    >>> from relic.core import RELICSimulator, FunctorBase
    >>> from relic.parameters import get_global_manager
    >>> sim = relic.create_simulator(config_path="config.yaml")
    >>> results = sim.run(start_time=0, end_time=365, dt=1800)

References:
    - Mac Lane, S. (1998). Categories for the Working Mathematician
    - Spivak, D.I. (2014). Category Theory for the Sciences
    - PyTorch Team (2019). PyTorch Deep Learning Library

Version: 1.0.0
License: MIT
"""

# Standard library
from typing import Optional, Dict, Any
import importlib
from importlib import util as importlib_util
import os
import logging

# Version and metadata
__version__ = "1.0.0"
__author__ = "Yuchong Long"
__license__ = "MIT"
__phase__ = "Phase 1 - Core Framework Integration"

# ============================================================================
# CORE FRAMEWORK - Foundation components
# ============================================================================
from relic.core.abstractions import (
    # Four-pillar abstractions
    PhysicalField,
    BiologicalProcess,
    TransportMechanism,
    ScaleCoupling,
)
from relic.core.functor import (
    # Unified functor base (AUTHORITATIVE)
    FunctorBase,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    # Bridge classes for executable functors
    BiologicalProcessFunctor,
    TransportMechanismFunctor,
    PhysicalFieldFunctor,
    ScaleCouplingFunctor,
    # Exception hierarchy
    FunctorException,
    ConfigurationError,
    DependencyError,
    ExecutionError,
    ConservationError,
)
from relic.core.simulator import (
    # Simulation framework
    RELICSimulator,
    SimulationState,
    create_simulator,
)
from relic.core.state import (
    # State management
    StateManager,
)
from relic.core.state.types import StateVariableSpec
from relic.core.dependency_graph import (
    # Dependency resolution
    DependencyGraphResolver as DependencyGraph,
    CyclicDependencyError,
)

# ============================================================================
# PARAMETER MANAGEMENT - Unified parameter system (eager)
# ============================================================================
from relic.parameters import (
    get_global_manager,
    UnifiedParameterManager,
    ParameterSpec,
)
from relic.parameters.types import ParameterCategory

# ============================================================================
# LAZY SUBMODULE IMPORTS - Reduce import-time overhead
# ============================================================================
_LAZY_MODULES = {
    "scientific": "relic.scientific",
    "coupling": "relic.coupling",
    "data": "relic.data",
    "validation": "relic.validation",
    "optimization": "relic.optimization",
    "computation": "relic.computation",
    "stochastic": "relic.stochastic",
    "interfaces": "relic.interfaces",
    "extensions": "relic.extensions",
    "monitoring": "relic.monitoring",
    "models": "relic.models",
    "contracts": "relic.contracts",
    "utils": "relic.utils",
}

logger = logging.getLogger(__name__)

_lazy_import_enabled = os.environ.get("RELIC_LAZY_IMPORT", "1").lower() in {"1", "true", "yes"}
logger.info("RELIC lazy import: %s", "ENABLED" if _lazy_import_enabled else "DISABLED")

if not _lazy_import_enabled:
    # Eager mode imports every RELIC subpackage during root initialization.
    # These are RELIC's own subpackages, so a missing or failing one is an
    # installation or code defect: the original import error propagates and
    # no None placeholder is bound for the subpackage name.
    for _name, _path in _LAZY_MODULES.items():
        globals()[_name] = importlib.import_module(_path)
else:
    def __getattr__(name: str) -> Any:
        if name in _LAZY_MODULES:
            logger.debug("Lazy loading submodule: %s", name)
            module_path = _LAZY_MODULES[name]
            spec = importlib_util.find_spec(module_path)
            if spec is None:
                raise AttributeError(
                    f"module 'relic' has no attribute '{name}' "
                    f"(missing submodule '{module_path}')"
                )
            module = importlib.import_module(module_path)
            globals()[name] = module
            return module
        raise AttributeError(f"module 'relic' has no attribute '{name}'")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_simulator(config: Optional[Dict[str, Any]] = None) -> RELICSimulator:
    """
    Create a simulator with default configuration.
    
    Parameters
    ----------
    config : Optional[Dict[str, Any]]
        Configuration dictionary. If None, uses defaults.
        
    Returns
    -------
    RELICSimulator
        Configured simulator instance
        
    Examples
    --------
    >>> sim = relic.create_default_simulator()
    >>> sim.register_functor(my_functor)
    >>> results = sim.step(dt=1800)
    """
    if config is None:
        config = {}
    return create_simulator(config)


def get_version_info() -> Dict[str, str]:
    """
    Get version and metadata information.
    
    Returns
    -------
    Dict[str, str]
        Dictionary containing version, author, license, phase info
    """
    return {
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'phase': __phase__,
    }


def print_architecture_guide() -> None:
    """Print RELIC architecture overview."""
    guide = """
    RELIC Architecture Overview
    =============================
    
    Core Framework (relic.core):
      - FunctorBase: Unified base class for all computational modules
      - RELICSimulator: Execution engine with dependency resolution
      - StateManager: Centralized state with conservation validation
      - Four Pillars: PhysicalField, BiologicalProcess, TransportMechanism, ScaleCoupling
    
    Scientific Modules (relic.scientific):
      - Atmosphere: Radiation, turbulence, chemistry, transport
      - Biosphere: Photosynthesis, respiration, growth, phenology, BVOC
      - Hydrosphere: ET, soil moisture, runoff, groundwater, snow
      - Pedosphere: Decomposition, nutrient cycling, carbon pools
      - Anthroposphere: Land use, management, emissions
      - Isotopes: Tracking and fractionation
      - Transport: Unified operators and boundary conditions
    
    Coupling (relic.coupling):
      - Internal: Cross-domain coupling (bio-atmo, soil-veg, water-carbon)
      - External: Third-party model adapters (WRF, VIC, SWAT, etc.)
      - Scales: Spatial/temporal/hierarchical scaling transformations
    
    Data (relic.data):
      - I/O: NetCDF/Zarr readers and writers with bounded CF variable metadata support
      - Standards: Unit conversion, CF conventions, CMIP standards
      - Quality: Validation, cleaning, gap filling
      - Transformers: Regridding, aggregation, interpolation
    
    Parameter System (relic.parameters):
      - Unified manager with tier-based organization
      - Automatic unit conversion and validation
      - Integration with optimization and uncertainty tools
    
    Support Systems:
      - Validation: L1-L4 testing (unit -> integration -> benchmark -> real-world)
      - Optimization: Calibration, assimilation, sensitivity, uncertainty
      - Computation: GPU, distributed, parallel, JIT optimization
      - Stochastic: Distributions, SDE, Gillespie, weather generators
      - Interfaces: CLI, API, web, notebooks
      - Extensions: Third-party plugins and user-defined functors
      - Monitoring: Prometheus metrics, Grafana dashboards, execution logging
      - Models: Basin / patch / landscape entities, surrogate helpers
      - Contracts: Variable contracts and domain constraints
      - Utils: Physical constants, tensor helpers, response functions
    
    Usage Pattern:
      1. Define scientific functors (inherit from bridge classes)
      2. Register with simulator
      3. Configure parameters via ParameterManager
      4. Run simulation with automatic dependency resolution
      5. Validate conservation laws and outputs
    
    For detailed documentation, see docs/source/
    """
    print(guide)


# ============================================================================
# PUBLIC API EXPORTS
# ============================================================================

__all__ = [
    # Version and metadata
    '__version__',
    '__author__',
    '__license__',
    '__phase__',
    
    # Core abstractions
    'PhysicalField',
    'BiologicalProcess',
    'TransportMechanism',
    'ScaleCoupling',
    
    # Unified functor system
    'FunctorBase',
    'FunctorMetadata',
    'ProcessCategory',
    'ExecutionPhase',
    
    # Bridge classes
    'BiologicalProcessFunctor',
    'TransportMechanismFunctor',
    'PhysicalFieldFunctor',
    'ScaleCouplingFunctor',
    
    # Simulation framework
    'RELICSimulator',
    'SimulationState',
    'create_simulator',
    'create_default_simulator',
    
    # State management
    'StateManager',
    'StateVariableSpec',
    
    # Dependency management
    'DependencyGraph',
    'CyclicDependencyError',
    
    # Exceptions
    'FunctorException',
    'ConfigurationError',
    'DependencyError',
    'ExecutionError',
    'ConservationError',
    
    # Parameter management
    'get_global_manager',
    'UnifiedParameterManager',
    'ParameterSpec',
    'ParameterCategory',
    
    # Subsystems (module references)
    'scientific',
    'coupling',
    'data',
    'validation',
    'optimization',
    'computation',
    'stochastic',
    'interfaces',
    'extensions',
    'monitoring',
    'models',
    'contracts',
    'utils',
    
    # Utility functions
    'get_version_info',
    'print_architecture_guide',
]
