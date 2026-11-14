"""
Solver Base Classes for RELIC Unified Computation Layer

This module defines the abstract base classes for all numerical solvers used
across the RELIC framework. Solvers transform equations/systems into solutions
through iterative or direct methods.

Design Philosophy:
    - Template Pattern: SolverBase defines solve() flow, subclasses implement strategy
    - Factory Creation: Solvers instantiated via SolverFactory based on config/name
    - Operator Composition: Solvers use OperatorBase instances for matrix operations
    - Parameter Integration: Full integration with UnifiedParameterManager

Mathematical Foundation:
    Solvers address problems of the form:
    - Linear systems: Ax = b
    - ODEs/PDEs: dy/dt = f(y, t)
    - Nonlinear systems: F(x) = 0

    Each solver declares its convergence and stability contract. Executable
    evidence is required before a concrete method can claim those properties.

References:
    - Burden, R.L. & Faires, J.D. (2010). Numerical Analysis. Brooks/Cole
    - Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential Equations II. Springer
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. SIAM
    - Trefethen, L.N. & Bau, D. (1997). Numerical Linear Algebra. SIAM
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
import torch
import torch.nn as nn
import logging

from relic.parameters import get_global_manager

logger = logging.getLogger(__name__)

# Type aliases
Tensor = torch.Tensor
TensorOrTuple = Union[Tensor, Tuple[Tensor, ...]]
ConfigDict = Dict[str, Any]

# Configuration keys that SolverBase itself resolves for every solver
# (see SolverBase.__init__); a solver accepts these plus the parameters it
# declares in SolverMetadata.parameters.
SOLVER_BASE_CONFIG_KEYS: Tuple[str, ...] = ("max_iterations", "rtol", "atol")


class SolverFamily(Enum):
    """Enumeration of solver families."""
    LINEAR_DIRECT = "linear_direct"        # Direct linear solvers (LU, Cholesky, Thomas)
    LINEAR_ITERATIVE = "linear_iterative"  # Iterative linear solvers (CG, GMRES, Jacobi)
    ODE_EXPLICIT = "ode_explicit"          # Explicit ODE solvers (Forward Euler, RK)
    ODE_IMPLICIT = "ode_implicit"          # Implicit ODE solvers (Backward Euler, BDF)
    ODE_IMEX = "ode_imex"                  # IMEX ODE solvers (implicit-explicit)
    NONLINEAR = "nonlinear"                # Nonlinear solvers (Newton, fixed-point)
    OPTIMIZATION = "optimization"          # Optimization solvers (gradient descent, BFGS)


class SolverStatus(Enum):
    """Enumeration of solver status codes."""
    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations"
    DIVERGED = "diverged"
    SINGULAR = "singular"
    INVALID_INPUT = "invalid_input"


@dataclass
class SolverMetadata:
    """
    Metadata for solver documentation and validation.

    Attributes:
        name : str
            Human-readable solver name
        solver_family : SolverFamily
            Classification (linear, ode, nonlinear, etc.)
        description : str
            Brief description of solver functionality
        mathematical_expression : str
            Mathematical formulation being solved
        order : int
            Convergence order (accuracy order for time integrators)
        stability_region : str
            Description of stability properties (A-stable, L-stable, etc.)
        is_implicit : bool
            Whether solver is implicit
        requires_jacobian : bool
            Whether solver needs Jacobian matrix
        references : list
            Scientific references
        parameters : dict
            Required/optional parameters with descriptions
    """
    name: str
    solver_family: SolverFamily
    description: str = ""
    mathematical_expression: str = ""
    order: int = field(kw_only=True)
    stability_region: str = "unknown"
    is_implicit: bool = False
    requires_jacobian: bool = False
    references: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)


@dataclass
class SolverResult:
    """
    Result container for solver output.

    Attributes:
        solution : TensorOrTuple
            Solution tensor(s)
        status : SolverStatus
            Solver exit status
        iterations : int
            Number of iterations performed
        residual_norm : Optional[float]
            Final residual norm (if applicable)
        convergence_history : Optional[list]
            Convergence history (residuals per iteration)
        metadata : dict
            Additional solver-specific information
    """
    solution: TensorOrTuple
    status: SolverStatus
    iterations: int = field(kw_only=True)
    residual_norm: Optional[float] = None
    convergence_history: Optional[list] = None
    metadata: dict = field(default_factory=dict)


class SolverStepError(RuntimeError):
    """A solver step ended with a status other than SUCCESS.

    Trajectory integration raises this error instead of using a
    non-converged (MAX_ITERATIONS) or non-finite (DIVERGED) step as the next
    state. The solver diagnostics counters are updated before it is raised.

    Attributes
    ----------
    solver_name : str
        Name of the solver instance
    status : SolverStatus
        Status returned by the step
    step_index : Optional[int]
        Zero-based index of the step within a trajectory; None for a
        single step taken outside a trajectory
    t_current : float
        Start time of the failed step
    dt : float
        Length of the failed step
    iterations : int
        Iterations (or function evaluations) reported by the step
    residual_norm : Optional[float]
        Final residual norm reported by the step
    metadata : dict
        Solver-specific metadata of the step result
    """

    def __init__(
        self,
        *,
        solver_name: str,
        status: SolverStatus,
        step_index: Optional[int],
        t_current: float,
        dt: float,
        iterations: int,
        residual_norm: Optional[float],
        metadata: Dict[str, Any],
    ) -> None:
        self.solver_name = solver_name
        self.status = status
        self.step_index = step_index
        self.t_current = t_current
        self.dt = dt
        self.iterations = iterations
        self.residual_norm = residual_norm
        self.metadata = dict(metadata)
        super().__init__(
            f"Solver '{solver_name}' step {step_index} from t={t_current!r} "
            f"with dt={dt!r} ended with status {status.value} after "
            f"{iterations} iteration(s); residual_norm={residual_norm!r}"
        )


def _require_value_within_spec(full_key: str, spec: Any, value: Any) -> None:
    """Check a configured value against its YAML ParameterSpec; violations raise ValueError.

    The canonical ParameterSpec.validate_value is the only rule: it rejects a
    NaN or +/-inf number, a value outside the declared bounds and a value not
    among the allowed values; this helper adds no check of its own.
    """
    errors = spec.validate_value(value)
    if errors:
        raise ValueError(f"Configured {full_key}={value!r} violates the YAML entry: " + "; ".join(errors))


class SolverBase(ABC, nn.Module):
    """
    Abstract base class for all numerical solvers in RELIC.

    This class defines the unified interface that all solvers must implement,
    providing consistency, composability, and integration with operators and
    the parameter management system.

    Design Principles:
        1. All solvers inherit from both ABC and nn.Module
        2. solve() method is the core interface
        3. Optional operator injection for matrix operations
        4. Configuration via constructor or ParameterManager
        5. Diagnostic tracking and convergence monitoring

    Parameters
    ----------
    name : str
        Unique identifier for this solver instance
    operator : Optional[Any]
        Associated operator (for operator-based solvers)
    config : ConfigDict
        Configuration dictionary with solver-specific parameters. Accepted
        keys are SOLVER_BASE_CONFIG_KEYS plus the keys of
        SolverMetadata.parameters; any other key raises ValueError.
    metadata : Optional[SolverMetadata]
        Solver metadata (order, stability region, accepted parameters);
        required, None raises TypeError
    device : Optional[torch.device]
        Computation device (CPU/GPU)
    dtype : torch.dtype
        Default data type for computations

    Attributes
    ----------
    solver_family : SolverFamily
        Class-level family identifier (override in subclasses)
    name : str
        Instance name
    operator : Optional[Any]
        Associated operator
    config : ConfigDict
        Configuration dictionary
    metadata : SolverMetadata
        Solver metadata

    Examples
    --------
    >>> class BackwardEulerSolver(SolverBase):
    ...     solver_family = SolverFamily.ODE_IMPLICIT
    ...     def solve(self, state, dt, rhs_func, **kwargs):
    ...         # Implementation
    ...         return SolverResult(solution=new_state, status=SolverStatus.SUCCESS)
    >>> solver = BackwardEulerSolver(name="backward_euler", config={})
    >>> result = solver.solve(y0, dt=0.1, rhs_func=lambda y, t: -y)
    """

    # Class-level family identifier. Every concrete solver declares it; the
    # base class carries no default family, so an undeclared solver is not
    # silently classified as a direct linear solver.
    solver_family: SolverFamily

    def __init__(
        self,
        name: str,
        operator: Optional[Any] = None,
        config: Optional[ConfigDict] = None,
        metadata: Optional[SolverMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        parameter_namespace: Optional[str] = None
    ) -> None:
        """Initialize solver base."""
        super().__init__()

        self.name = name
        self.operator = operator
        self.config = config if config is not None else {}
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.parameter_namespace: Optional[str] = parameter_namespace

        # Every solver declares its metadata (order, accepted parameters);
        # none is invented for it here.
        if metadata is None:
            raise TypeError(
                f"Solver '{name}' ({type(self).__name__}) must declare its "
                "SolverMetadata (order, stability region, parameters)"
            )
        self.metadata = metadata

        # Fail fast on configuration keys the solver does not declare, so a
        # stale or misspelled key is never silently ignored.
        accepted_keys = set(SOLVER_BASE_CONFIG_KEYS) | set(self.metadata.parameters)
        unknown_keys = sorted(set(self.config) - accepted_keys, key=str)
        if unknown_keys:
            raise ValueError(
                f"Unknown configuration key(s) {', '.join(repr(k) for k in unknown_keys)} "
                f"for solver '{name}' ({type(self).__name__}); accepted keys: "
                f"{', '.join(sorted(accepted_keys))}"
            )

        # No diagnostic counters: a solve call changes no solver state
        # (invariant I1); the iteration count and status of each call are
        # fields of its SolverResult.

        # Parameter manager integration: fetched by _get_parameter_manager()
        # when first needed; failures are raised, never logged and ignored.
        self._param_manager = None

        # Convergence parameters (resolved from ParameterManager or explicit config)
        self.max_iterations = self._resolve_default_parameter(
            "max_iterations",
            default_key="relic.physical.constants.numerical.max_iterations_default"
        )
        self.rtol = self._resolve_default_parameter(
            "rtol",
            default_key="relic.physical.constants.numerical.tolerance_relative"
        )
        self.atol = self._resolve_default_parameter(
            "atol",
            default_key="relic.physical.constants.numerical.tolerance_absolute"
        )

    @abstractmethod
    def solve(self, *args, **kwargs) -> SolverResult:
        """
        Solve the system/equation.

        This is the core method that all solvers must implement. It defines
        the solution strategy for the specific problem type.

        Parameters
        ----------
        *args
            Positional arguments (problem-specific)
        **kwargs
            Keyword arguments (solver-specific parameters)

        Returns
        -------
        SolverResult
            Solution result with status and diagnostics

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses

        Notes
        -----
        - Implementation should handle convergence failure gracefully
        - Use self.get_parameter() to retrieve parameters with validation
        - Report the iteration count and status in the SolverResult; a solve
          call never changes solver tensors or attributes (invariant I1)
        """

    def forward(self, *args, **kwargs) -> SolverResult:
        """
        PyTorch nn.Module forward pass - delegates to solve().

        This method is called when using the solver as a callable. It
        delegates to solve() and changes no solver state (invariant I1); the
        per-call iteration count and status are fields of the SolverResult.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to solve()

        Returns
        -------
        SolverResult
            Result from solve()
        """
        return self.solve(*args, **kwargs)

    def get_parameter(
        self,
        param_name: str,
        units: Optional[str] = None,
        namespace: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> Any:
        """
        Get parameter value from configuration or ParameterManager.

        Search order:
        1. self.config[param_name]
        2. ParameterManager.get(namespace.param_name)

        Parameters
        ----------
        param_name : str
            Name of parameter
        units : Optional[str]
            Expected units (triggers unit validation)
        namespace : Optional[str]
            Parameter namespace (e.g., "computation.solvers.backward_euler")
        bounds : Optional[Tuple[float, float]]
            Valid range for parameter

        Returns
        -------
        Any
            Parameter value

        Raises
        ------
        KeyError
            If <namespace>.<param_name> is not in the parameter database
        ValueError
            If units or bounds are provided (must be defined in parameter database),
            or a configured value violates the YAML entry (bounds, allowed values)
        Other manager errors propagate unchanged (they are not reported as a
        missing parameter).
        """
        if units is not None:
            raise ValueError(
                "Units must be defined in the parameter database; do not pass units in code."
            )
        if bounds is not None:
            raise ValueError(
                "Bounds must be defined in the parameter database; do not pass bounds in code."
            )
        manager = self._get_parameter_manager()
        resolved_namespace = namespace or self.parameter_namespace
        full_key = f"{resolved_namespace}.{param_name}" if resolved_namespace else param_name
        spec = manager.get_parameter(full_key)
        if spec is None:
            raise KeyError(f"Parameter '{full_key}' is not in the parameter database")

        # A configured value overrides the YAML value but is held to the same
        # YAML contract (bounds, allowed values) as the database value.
        if param_name in self.config:
            value = self.config[param_name]
            _require_value_within_spec(full_key, spec, value)
            return value
        return manager.get(full_key, expected_type=None, required=True)

    def _get_parameter_manager(self) -> Any:
        """
        Get the latest global parameter manager instance.

        Returns
        -------
        Any
            Unified parameter manager singleton.

        Raises
        ------
        ValueError
            If parameter manager cannot be initialized.
        """
        try:
            manager = get_global_manager(
                preload_all=True,
                warmup_cache=False,
            )
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001
            raise ValueError(
                "ParameterManager initialization failed for solver parameter resolution"
            ) from exc
        self._param_manager = manager
        return manager

    def _resolve_default_parameter(self, param_name: str, default_key: str) -> Any:
        """
        Resolve a SolverBase parameter from config or the ParameterManager.

        Resolution order: ``self.config[param_name]``; else
        ``<parameter_namespace>.<param_name>`` when the solver declares a
        namespace and the key exists; else the shared global numerical
        default ``default_key`` (a required YAML parameter). No Python
        fallback value is used.

        Parameters
        ----------
        param_name : str
            Parameter name under solver namespace (e.g., "max_iterations")
        default_key : str
            Full parameter key for global defaults

        Returns
        -------
        Any
            Parameter value resolved from config or ParameterManager

        Raises
        ------
        KeyError
            If neither the namespace key nor default_key exists.
            Validation and manager errors propagate unchanged.
        """
        manager = self._get_parameter_manager()
        resolved_key = default_key
        if self.parameter_namespace:
            namespace_key = f"{self.parameter_namespace}.{param_name}"
            if manager.get_parameter(namespace_key) is not None:
                resolved_key = namespace_key
        spec = manager.get_parameter(resolved_key)
        if spec is None:
            raise KeyError(f"Parameter '{resolved_key}' is not in the parameter database")

        # A configured value is held to the YAML entry it replaces.
        if param_name in self.config:
            value = self.config[param_name]
            _require_value_within_spec(resolved_key, spec, value)
            return value
        return manager.get(resolved_key, required=True)

    def check_convergence(
        self,
        residual: Tensor,
        reference: Optional[Tensor] = None
    ) -> bool:
        """
        Check a scaled successive-iterate increment criterion.

        For the current implicit temporal callers, ``residual`` is the
        historical parameter name for ``y_new - y_previous``. This predicate
        does not prove that the accepted iterate satisfies the implicit
        equation defect tolerance, that the fixed-point map is contractive, or
        that every component or batch converged independently.

        Parameters
        ----------
        residual : Tensor
            Successive-iterate increment. It must be a nonempty, finite, real
            floating strided tensor.
        reference : Optional[Tensor]
            New-iterate scale tensor. When present, it must have the same
            shape, dtype, device, and units as ``residual``.

        Returns
        -------
        bool
            True if converged, False otherwise

        Notes
        -----
        ``atol`` has the same units as ``residual`` and ``reference``;
        ``rtol`` is dimensionless. Unit equality is a caller-side semantic
        precondition because this low-level Tensor API has no unit metadata.
        """
        if not isinstance(residual, torch.Tensor):
            raise TypeError("residual must be a torch.Tensor")
        if not residual.is_floating_point() or residual.layout != torch.strided:
            raise ValueError("residual must be a real floating strided tensor")
        if residual.numel() == 0:
            raise ValueError("residual must contain at least one value")
        if residual.device.type == "meta":
            raise ValueError("residual must use a concrete device")
        if not bool(torch.all(torch.isfinite(residual)).item()):
            raise ValueError("residual must contain only finite values")

        if reference is not None:
            if not isinstance(reference, torch.Tensor):
                raise TypeError("reference must be a torch.Tensor")
            if (
                not reference.is_floating_point()
                or reference.layout != torch.strided
            ):
                raise ValueError(
                    "reference must be a real floating strided tensor"
                )
            if reference.numel() == 0:
                raise ValueError("reference must contain at least one value")
            if reference.shape != residual.shape:
                raise ValueError("residual and reference must have the same shape")
            if reference.device != residual.device:
                raise ValueError("residual and reference must use the same device")
            if reference.dtype != residual.dtype:
                raise ValueError("residual and reference must use the same dtype")
            if reference.device.type == "meta":
                raise ValueError("reference must use a concrete device")
            if not bool(torch.all(torch.isfinite(reference)).item()):
                raise ValueError("reference must contain only finite values")

        if isinstance(self.atol, bool) or not isinstance(self.atol, Real):
            raise TypeError("atol must be a real scalar")
        if isinstance(self.rtol, bool) or not isinstance(self.rtol, Real):
            raise TypeError("rtol must be a real scalar")
        atol = float(self.atol)
        rtol = float(self.rtol)
        if not math.isfinite(atol) or atol <= 0.0:
            raise ValueError("atol must be finite and greater than zero")
        if not math.isfinite(rtol) or rtol < 0.0:
            raise ValueError("rtol must be finite and nonnegative")

        # References:
        # - PETSc SNESConvergedReason solution-step convergence contract.
        # - SciPy linearmixing x_tol/x_rtol step convergence contract.
        # Mathematical expression:
        #   rho = ||delta_y||_inf
        #   threshold = atol, when reference is None
        #   threshold = max(atol, rtol * ||reference||_inf), otherwise
        #   converged iff rho < threshold
        # Parameter meanings:
        #   delta_y is the successive iterate increment; reference is the
        #   new-iterate scale; atol is an absolute tolerance; rtol is a
        #   dimensionless relative tolerance.
        # Units, domain, and sign convention:
        #   [delta_y] = [reference] = [atol], [rtol] = 1; tensors and all
        #   scalar intermediates are finite; norms and thresholds are positive
        #   magnitudes, so no physical flux sign convention is implied.
        residual_norm = float(torch.max(torch.abs(residual)).item())
        threshold = atol
        if reference is not None:
            reference_norm = float(torch.max(torch.abs(reference)).item())
            relative_threshold = rtol * reference_norm
            if not math.isfinite(relative_threshold):
                raise ValueError("relative threshold must be finite")
            threshold = max(atol, relative_threshold)

        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("convergence threshold must be finite and positive")
        return residual_norm < threshold

    def get_metadata(self) -> SolverMetadata:
        """
        Get solver metadata.

        Returns
        -------
        SolverMetadata
            Metadata object with solver documentation
        """
        return self.metadata

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get solver diagnostic information.

        Returns
        -------
        Dict[str, Any]
            Static solver description (name, family, device, dtype); the
            per-call diagnostics (iterations, status, residual) are in each
            SolverResult
        """
        return {
            'name': self.name,
            'solver_family': self.solver_family.value,
            'device': str(self.device),
            'dtype': str(self.dtype)
        }

    def to(self, device: torch.device) -> 'SolverBase':
        """
        Move solver to specified device.

        Parameters
        ----------
        device : torch.device
            Target device

        Returns
        -------
        SolverBase
            Self (for chaining)
        """
        super().to(device)
        self.device = device
        if self.operator is not None and hasattr(self.operator, 'to'):
            self.operator.to(device)
        return self

    def __repr__(self) -> str:
        """String representation of solver."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"family='{self.solver_family.value}', "
            f"device='{self.device}'"
            f")"
        )


# Export main classes
__all__ = [
    'SolverBase',
    'SolverMetadata',
    'SolverResult',
    'SolverFamily',
    'SolverStatus',
    'TensorOrTuple',
    'ConfigDict'
]
