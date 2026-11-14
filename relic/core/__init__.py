"""
RELIC Core Framework - Unified Architecture Foundation

This package provides the core framework components for the Recomposable Ecosystem Library for
Integrated Coupling (RELIC), implementing a unified architecture based on category theory
principles and four-pillar abstraction design.

Architecture Components:
    - abstractions.py: Four-pillar abstract base classes (PhysicalField, BiologicalProcess,
      TransportMechanism, ScaleCoupling) with enhanced methods
    - functor/: Unified FunctorBase class and bridge functors
    - simulator.py: Minimalist simulation engine with clean responsibilities
    - state/: Centralized state management with validation
    - dependency_graph.py: Topological execution ordering and cycle detection
    - category_theory.py: Design blueprint only (not imported at runtime)
    - config/: Configuration system with validation and loading capabilities

Design Principles:
    1. Single Source of Truth: One authoritative implementation per concept
    2. Composability: All components follow category theory composition rules
    3. Validation: Comprehensive validation at all levels (parameters, state, conservation)
    4. Performance: Optimized for large-scale ecosystem modeling with GPU support
    5. Maintainability: Clear separation of concerns with modular architecture

Mathematical Foundation:
    The framework is built on solid mathematical foundations:
    - Category Theory: Objects, Morphisms, and Functors for structural consistency
    - Conservation Laws: Mass, energy, and momentum conservation enforcement
    - Differential Equations: Robust numerical integration with stability checking
    - Automatic Differentiation: PyTorch integration for parameter optimization

References:
    - Mac Lane, S. (1998). Categories for the Working Mathematician
    - Spivak, D.I. (2014). Category Theory for the Sciences
    - Goodfellow, I. et al. (2016). Deep Learning - PyTorch integration patterns
    - PyTorch Team (2019). PyTorch: An Imperative Style Deep Learning Library

Version: 1.0.0 (Phase 1 - Core Framework)
"""

# Standard library imports
# Import core components with careful order to handle dependencies
from .abstractions import (
    PhysicalField,
    BiologicalProcess,
    TransportMechanism,
    ScaleCoupling,
    SpatialScalingBase,
    TemporalScalingBase,
    HierarchicalScalingBase,
    StateDict,
    EnvironmentDict,
    FluxDict,
)

from .functor import (
    FunctorBase,  # AUTHORITATIVE - replaces prior implementations
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    FunctorState,
    SpatialScale,
    VariableKind,
    SubFlowAccuracy,
    FunctorException,
    ConfigurationError,
    DependencyError,
    ExecutionError,
    ConservationError,
    DependencyList,
    DependencyDeclaration,
)

# Bridge classes - unified inheritance pattern for executable functors
from .functor import (
    BiologicalProcessFunctor,  # Bridge: FunctorBase + BiologicalProcess
    TransportMechanismFunctor,  # Bridge: FunctorBase + TransportMechanism
    PhysicalFieldFunctor,  # Bridge: FunctorBase + PhysicalField
    ScaleCouplingFunctor,  # Bridge: FunctorBase + ScaleCoupling
)

from .simulator import (
    RELICSimulator,
    SimulationState,
    create_simulator
)

from .state import StateManager
from .state.types import StateVariableSpec, StateVariableType

from .dependency_graph import (
    CyclicDependencyError,
    DependencyEdge,
    DependencyGraphResolver as DependencyGraph,
    DependencyType,
    ExecutionNode,
)
from .constants import SCALE_HIERARCHY, get_scale_hierarchy, refresh_scale_hierarchy

# category_theory.py: DESIGN BLUEPRINT only - not exported at runtime.
# See relic/core/category_theory.py for the theoretical reference document.

# Configuration system imports
from .config import (
    RELICConfig,
    load_config
)


# Version and metadata
__version__ = "1.0.0"
__phase__ = "Phase 1 - Core Framework"
__author__ = "Yuchong Long"
__license__ = "MIT"

# Export all public components
__all__ = [
    # Core abstractions (enhanced with new methods)
    'PhysicalField',
    'BiologicalProcess',
    'TransportMechanism',
    'ScaleCoupling',

    # Specialized scale coupling abstractions
    'SpatialScalingBase',
    'TemporalScalingBase',
    'HierarchicalScalingBase',

    # Unified functor base (AUTHORITATIVE)
    'FunctorBase',  # MAIN REPLACEMENT for prior FunctorBase classes
    'FunctorMetadata',
    'ProcessCategory',
    'ExecutionPhase',
    'FunctorState',
    'SpatialScale',
    'VariableKind',
    'SubFlowAccuracy',  # Declared local order of a Functor's own sub-flow
    'DependencyList',  # Type alias for dependency declarations
    'DependencyDeclaration',

    # Exception hierarchy
    'FunctorException',
    'ConfigurationError',
    'DependencyError',
    'ExecutionError',
    'ConservationError',

    # Bridge classes - unified inheritance pattern (NEW)
    'BiologicalProcessFunctor',  # Bridge: FunctorBase + BiologicalProcess
    'TransportMechanismFunctor',  # Bridge: FunctorBase + TransportMechanism
    'PhysicalFieldFunctor',  # Bridge: FunctorBase + PhysicalField
    'ScaleCouplingFunctor',  # Bridge: FunctorBase + ScaleCoupling

    # Simulation framework
    'RELICSimulator',
    'SimulationState',
    'create_simulator',

    # State management
    'StateManager',
    'StateVariableSpec',
    'StateVariableType',

    # Dependency management
    'DependencyGraph',
    'DependencyType',
    'DependencyEdge',
    'ExecutionNode',
    'CyclicDependencyError',

    # category_theory.py is a DESIGN BLUEPRINT - not exported at runtime

    # Configuration system
    'RELICConfig',
    'load_config',

    # Type aliases
    'StateDict',
    'EnvironmentDict',
    'FluxDict',

    # Metadata
    '__version__',
    '__phase__',
    '__author__',
    '__license__',
    'SCALE_HIERARCHY',
    'get_scale_hierarchy',
    'refresh_scale_hierarchy',
]
