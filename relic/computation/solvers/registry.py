"""
Solver Registry for RELIC Unified Computation Layer

This module provides a centralized registration system for all numerical solvers
in the RELIC framework, enabling dynamic discovery, type-safe access, and
metadata-driven documentation.

Design Philosophy:
    - Registry Pattern: Central repository for solver classes
    - Decorator Registration: Simple @register_solver decorator for automatic registration
    - Type Safety: Validates that registered classes inherit from SolverBase
    - Metadata Access: Provides query capabilities for solver discovery
    - Family Organization: Groups solvers by mathematical family (linear, ODE, nonlinear, etc.)

Usage Example:
    >>> from relic.computation.solvers.registry import register_solver, SolverRegistry
    >>> from relic.computation.solvers.base import SolverBase, SolverFamily
    >>>
    >>> @register_solver("backward_euler", family=SolverFamily.ODE_IMPLICIT)
    >>> class BackwardEulerSolver(SolverBase):
    ...     solver_family = SolverFamily.ODE_IMPLICIT
    ...     def solve(self, y0, t, dt, rhs_func):
    ...         # Implementation
    ...         return SolverResult(solution=y_new, status=SolverStatus.SUCCESS)
    >>>
    >>> # Later retrieval
    >>> solver_class = SolverRegistry.get("backward_euler")
    >>> solver = solver_class(name="be_solver", config={})

References:
    - Gamma, E. et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley
"""

from typing import Dict, Type, List, Optional, Any
from dataclasses import dataclass
import logging

# Import base class for type checking
from .base import SolverBase, SolverMetadata, SolverFamily
BASE_AVAILABLE = True

logger = logging.getLogger(__name__)


@dataclass
class SolverRegistryEntry:
    """
    Registry entry for a solver class.

    Attributes
    ----------
    name : str
        Solver identifier (e.g., "backward_euler", "thomas_algorithm")
    solver_class : Type[SolverBase]
        The solver class itself
    solver_family : SolverFamily
        Mathematical family classification
    is_implicit : bool
        Whether the solver is implicit
    requires_jacobian : bool
        Whether the solver requires Jacobian matrix
    metadata : Optional[SolverMetadata]
        Solver metadata if available
    """
    name: str
    solver_class: Type['SolverBase']  # type: ignore
    solver_family: 'SolverFamily'  # type: ignore
    is_implicit: bool = False
    requires_jacobian: bool = False
    metadata: Optional['SolverMetadata'] = None  # type: ignore


class SolverRegistry:
    """
    Central registry for all numerical solvers in RELIC.

    This class maintains a global registry of solver classes, enabling:
    - Dynamic solver discovery
    - Type-safe solver instantiation
    - Metadata-driven documentation generation
    - Organization by mathematical family

    The registry is a singleton pattern - all registration operations
    affect the global registry state.

    Class Methods
    -------------
    register(name, solver_class, family=None)
        Register a solver class
    get(name)
        Retrieve solver class by name
    list_solvers(family=None, implicit_only=False)
        List all registered solvers with optional filtering
    get_metadata(name)
        Get metadata for a specific solver
    clear()
        Clear all registrations (primarily for testing)

    Examples
    --------
    >>> # Direct registration
    >>> SolverRegistry.register("my_solver", MySolverClass)
    >>>
    >>> # Retrieve and instantiate
    >>> solver_class = SolverRegistry.get("my_solver")
    >>> solver = solver_class(name="instance1", config={})
    >>>
    >>> # Query all ODE solvers
    >>> ode_solvers = SolverRegistry.list_solvers(family=SolverFamily.ODE_IMPLICIT)
    """

    # Class-level registry storage
    _registry: Dict[str, SolverRegistryEntry] = {}
    _initialized: bool = False

    @classmethod
    def register(
        cls,
        name: str,
        solver_class: Type['SolverBase'],  # type: ignore
        family: Optional['SolverFamily'] = None,  # type: ignore
        override: bool = False
    ) -> None:
        """
        Register a solver class in the global registry.

        Parameters
        ----------
        name : str
            Solver identifier (e.g., "backward_euler")
        solver_class : Type[SolverBase]
            The solver class to register (must inherit from SolverBase)
        family : Optional[SolverFamily]
            Mathematical family classification (if None, extracted from class)
        override : bool, optional
            If True, allows overwriting existing registrations

        Raises
        ------
        TypeError
            If solver_class does not inherit from SolverBase
        ValueError
            If name is already registered and override=False

        Notes
        -----
        - Solver names should be lowercase with underscores
        - The solver_family is extracted from the class's solver_family attribute if not provided
        """
        # Type validation
        if not issubclass(solver_class, SolverBase):
            raise TypeError(
                f"Solver class {solver_class.__name__} must inherit from SolverBase"
            )

        # Check for duplicate registration
        if name in cls._registry and not override:
            raise ValueError(
                f"Solver '{name}' is already registered. "
                f"Use override=True to replace existing registration."
            )

        # The family is the explicit argument or the class declaration; an
        # undeclared family is an error, not a direct linear solver.
        if family is None:
            family = getattr(solver_class, 'solver_family', None)
        if family is None:
            raise ValueError(
                f"Solver '{name}' ({solver_class.__name__}) declares no "
                "solver_family; set the class attribute or pass family="
            )

        # Record the metadata of one instance built from the YAML defaults;
        # the implicit and Jacobian flags come only from that metadata. A
        # construction failure is a registration error; the solver is never
        # registered as an explicit solver without a Jacobian by default.
        temp_instance = solver_class(name="__temp__", config={})
        metadata = temp_instance.get_metadata()
        is_implicit = metadata.is_implicit
        requires_jacobian = metadata.requires_jacobian

        # Create registry entry
        entry = SolverRegistryEntry(
            name=name,
            solver_class=solver_class,
            solver_family=family,
            is_implicit=is_implicit,
            requires_jacobian=requires_jacobian,
            metadata=metadata
        )

        cls._registry[name] = entry
        logger.info(f"Registered solver: {name} (family={family}, implicit={is_implicit})")

    @classmethod
    def get(cls, name: str) -> Type['SolverBase']:  # type: ignore
        """
        Retrieve solver class by name.

        Parameters
        ----------
        name : str
            Solver identifier

        Returns
        -------
        Type[SolverBase]
            The registered solver class

        Raises
        ------
        KeyError
            If solver name is not registered
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Solver '{name}' not found in registry. "
                f"Available solvers: {available}"
            )

        return cls._registry[name].solver_class

    @classmethod
    def list_solvers(
        cls,
        family: Optional['SolverFamily'] = None,  # type: ignore
        implicit_only: bool = False,
        requires_jacobian: Optional[bool] = None
    ) -> List[str]:
        """
        List all registered solvers with optional filtering.

        Parameters
        ----------
        family : Optional[SolverFamily]
            Filter by solver family (LINEAR_DIRECT, ODE_IMPLICIT, etc.)
        implicit_only : bool, optional
            If True, return only implicit solvers
        requires_jacobian : Optional[bool]
            Filter by Jacobian requirement (True/False/None for any)

        Returns
        -------
        List[str]
            List of solver names matching the filters

        Examples
        --------
        >>> # List all solvers
        >>> all_solvers = SolverRegistry.list_solvers()
        >>>
        >>> # List only implicit ODE solvers
        >>> implicit_ode = SolverRegistry.list_solvers(
        ...     family=SolverFamily.ODE_IMPLICIT,
        ...     implicit_only=True
        ... )
        """
        results = []

        for name, entry in cls._registry.items():
            # Apply filters
            if family is not None and entry.solver_family != family:
                continue
            if implicit_only and not entry.is_implicit:
                continue
            if requires_jacobian is not None and entry.requires_jacobian != requires_jacobian:
                continue

            results.append(name)

        return sorted(results)

    @classmethod
    def get_metadata(cls, name: str) -> Optional['SolverMetadata']:  # type: ignore
        """
        Get metadata for a specific solver.

        Parameters
        ----------
        name : str
            Solver name

        Returns
        -------
        Optional[SolverMetadata]
            Solver metadata if available, None otherwise
        """
        if name not in cls._registry:
            raise KeyError(f"Solver '{name}' not found in registry")

        return cls._registry[name].metadata

    @classmethod
    def get_entry(cls, name: str) -> SolverRegistryEntry:
        """
        Get full registry entry for a solver.

        Parameters
        ----------
        name : str
            Solver name

        Returns
        -------
        SolverRegistryEntry
            Complete registry entry with class, family, and metadata
        """
        if name not in cls._registry:
            raise KeyError(f"Solver '{name}' not found in registry")

        return cls._registry[name]

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a solver is registered.

        Parameters
        ----------
        name : str
            Solver name to check

        Returns
        -------
        bool
            True if solver is registered, False otherwise
        """
        return name in cls._registry

    @classmethod
    def get_solvers_for_problem(
        cls,
        problem_type: str,
        implicit: Optional[bool] = None
    ) -> List[str]:
        """
        Get suitable solvers for a specific problem type.

        Parameters
        ----------
        problem_type : str
            Problem type: "linear", "ode", "nonlinear", "optimization"
        implicit : Optional[bool]
            Filter by implicit/explicit

        Returns
        -------
        List[str]
            List of suitable solver names

        Examples
        --------
        >>> # Get all ODE solvers
        >>> ode_solvers = SolverRegistry.get_solvers_for_problem("ode")
        >>>
        >>> # Get implicit ODE solvers only
        >>> implicit_ode = SolverRegistry.get_solvers_for_problem("ode", implicit=True)
        """
        # Map problem types to solver families
        family_map = {
            "linear": [SolverFamily.LINEAR_DIRECT, SolverFamily.LINEAR_ITERATIVE],
            "ode": [SolverFamily.ODE_EXPLICIT, SolverFamily.ODE_IMPLICIT],
            "nonlinear": [SolverFamily.NONLINEAR],
            "optimization": [SolverFamily.OPTIMIZATION]
        }

        if problem_type not in family_map:
            logger.warning(f"Unknown problem type: {problem_type}")
            return []

        families = family_map[problem_type]
        results = []

        for name, entry in cls._registry.items():
            if entry.solver_family not in families:
                continue
            if implicit is not None and entry.is_implicit != implicit:
                continue

            results.append(name)

        return sorted(results)

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registrations.

        Warning
        -------
        This method is primarily intended for testing purposes.
        Use with caution in production code.
        """
        cls._registry.clear()
        logger.warning("Solver registry cleared")

    @classmethod
    def get_registry_info(cls) -> Dict[str, Any]:
        """
        Get summary information about the registry.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing registry statistics and summaries
        """
        family_counts = {}
        implicit_count = 0
        jacobian_count = 0

        for entry in cls._registry.values():
            # Count by family
            family_name = entry.solver_family.value if hasattr(entry.solver_family, 'value') else str(entry.solver_family)
            family_counts[family_name] = family_counts.get(family_name, 0) + 1

            # Count characteristics
            if entry.is_implicit:
                implicit_count += 1
            if entry.requires_jacobian:
                jacobian_count += 1

        return {
            'total_solvers': len(cls._registry),
            'solver_names': sorted(cls._registry.keys()),
            'families': family_counts,
            'implicit_solvers': implicit_count,
            'jacobian_required_solvers': jacobian_count
        }


def register_solver(
    name: str,
    family: Optional['SolverFamily'] = None,  # type: ignore
    override: bool = False
):
    """
    Decorator for automatic solver registration.

    This decorator simplifies solver registration by automatically
    calling SolverRegistry.register() when the class is defined.

    Parameters
    ----------
    name : str
        Solver identifier
    family : Optional[SolverFamily]
        Mathematical family classification
    override : bool, optional
        Allow overwriting existing registrations

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    >>> @register_solver("backward_euler", family=SolverFamily.ODE_IMPLICIT)
    >>> class BackwardEulerSolver(SolverBase):
    ...     solver_family = SolverFamily.ODE_IMPLICIT
    ...     def solve(self, y0, t, dt, rhs_func):
    ...         return SolverResult(solution=y_new, status=SolverStatus.SUCCESS)

    Notes
    -----
    - This decorator should be used on all solver implementations
    - The decorated class is returned unmodified (decorator only has side effects)
    - Registration occurs at import time, not instantiation time
    """
    def decorator(solver_class: Type['SolverBase']) -> Type['SolverBase']:  # type: ignore
        # Register the class
        SolverRegistry.register(
            name=name,
            solver_class=solver_class,
            family=family,
            override=override
        )

        # Return unmodified class
        return solver_class

    return decorator


# Export main classes and functions
__all__ = [
    'SolverRegistry',
    'SolverRegistryEntry',
    'register_solver'
]
