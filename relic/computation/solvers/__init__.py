"""
Solver System for RELIC Unified Computation Layer

This package provides the complete solver system including:
- SolverBase: Abstract base class for all solvers
- SolverRegistry: Central registration system
- SolverFactory: Factory methods for creating solvers
- Solver implementations (ODE, linear, tridiagonal, etc.)

Usage Example:
    >>> from relic.computation.solvers import SolverFactory, register_solver
    >>>
    >>> # List available solvers
    >>> solvers = SolverFactory.list_available_solvers()
    >>>
    >>> # Create a solver
    >>> solver = SolverFactory.create(
    ...     solver_name="backward_euler",
    ...     name="be_solver",
    ...     config={"max_iterations": 100, "rtol": 1e-6}
    ... )
    >>>
    >>> # Register a new solver
    >>> @register_solver("my_solver", family=SolverFamily.ODE_IMPLICIT)
    >>> class MySolver(SolverBase):
    ...     solver_family = SolverFamily.ODE_IMPLICIT
    ...     def solve(self, state, dt):
    ...         return SolverResult(solution=new_state, status=SolverStatus.SUCCESS)
"""

# Import base classes
from .base import (
    SolverBase,
    SolverMetadata,
    SolverResult,
    SolverFamily,
    SolverStatus,
    SolverStepError,
    TensorOrTuple,
    ConfigDict
)

# Import registry
from .registry import (
    SolverRegistry,
    SolverRegistryEntry,
    register_solver
)

# Import factory
from .factory import (
    SolverFactory,
    SolverConfig
)

# Import solver implementations (triggers registration via decorators)
from . import temporal
from . import linear
from . import nonlinear
from . import geometric

# Export commonly used solver classes/functions
from .temporal import (
    BackwardEulerSolver,
    ImplicitMidpointSolver,
    CrankNicolsonSolver,
    IMEXEulerSolver,
    RK4Solver
)
from .linear import (
    ThomasAlgorithmSolver,
    solve_tridiagonal_thomas,
    solve_tridiagonal_batch,
    check_matrix_conditioning
)
from .nonlinear import BracketedBisectionSolver
from .temporal import (
    solve_ode_system
)
from .geometric import (
    IntegrationPlan,
    IntegrationStage,
    IntegrationStageCompatibilitySpec,
    build_strang_iphs_plan,
    build_stochastic_sdae_plan,
    get_implicit_midpoint_compatibility_spec,
    transactional_implicit_midpoint_step,
)
# Version info
__version__ = "0.1.0"

# Public API
__all__ = [
    # Base classes
    'SolverBase',
    'SolverMetadata',
    'SolverResult',
    'SolverFamily',
    'SolverStatus',
    'SolverStepError',
    'TensorOrTuple',
    'ConfigDict',

    # Registry
    'SolverRegistry',
    'SolverRegistryEntry',
    'register_solver',

    # Factory
    'SolverFactory',
    'SolverConfig',
    
    # Temporal solvers
    'BackwardEulerSolver',
    'ImplicitMidpointSolver',
    'CrankNicolsonSolver',
    'IMEXEulerSolver',
    'RK4Solver',

    # Linear solvers
    'ThomasAlgorithmSolver',
    'solve_tridiagonal_thomas',
    'solve_tridiagonal_batch',
    'check_matrix_conditioning',

    # Nonlinear solvers
    'BracketedBisectionSolver',

    # ODE utilities
    'solve_ode_system',
    'IntegrationPlan',
    'IntegrationStage',
    'IntegrationStageCompatibilitySpec',
    'build_strang_iphs_plan',
    'build_stochastic_sdae_plan',
    'get_implicit_midpoint_compatibility_spec',
    'transactional_implicit_midpoint_step',

    # Solver implementation modules
    'temporal',
    'linear',
    'nonlinear',
    'geometric'
]
