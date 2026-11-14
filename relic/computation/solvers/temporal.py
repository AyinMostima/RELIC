"""
Temporal Solvers for RELIC Unified Computation Layer

Time integration schemes for ordinary differential equations (ODEs) commonly
encountered in biogeochemical modeling. This module provides unified interfaces
to temporal solvers that integrate with the RELIC computation framework.

Mathematical Foundation:
    General ODE form: dy/dt = f(y, t)
    
    Temporal discretization schemes:
    - Backward Euler (BE): Implicit first-order, A-stable
    - Implicit Midpoint (IM): Implicit second-order, symplectic
    - Crank-Nicolson (CN): Implicit second-order, A-stable
    - Runge-Kutta 4 (RK4): Explicit fourth-order, conditionally stable

Applications:
    - Stiff ODEs in biogeochemical systems
    - Coupled carbon-nitrogen dynamics
    - Soil moisture evolution
    - Canopy energy balance

References:
    - Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential Equations II. Springer
    - Burden, R.L. & Faires, J.D. (2010). Numerical Analysis. Brooks/Cole
    - Iserles, A. (2008). A First Course in the Numerical Analysis of Differential Equations. Cambridge
"""

import logging
import math
import sys
from typing import Callable, Optional, Dict, Any, Tuple

import torch
from torch import Tensor

from .base import (
    SolverBase,
    SolverMetadata,
    SolverResult,
    SolverFamily,
    SolverStatus,
    SolverStepError,
)
from .registry import register_solver

logger = logging.getLogger(__name__)

# Type alias for ODE right-hand side function
ODEFunction = Callable[[Tensor, float], Tensor]


class _ODEForwardMixin:
    """Shared fixed-step trajectory integration for the one-step ODE solvers.

    The mixin has no step-size controller. The Simulator update-magnitude
    heuristic (docs/source/architecture_overview.md Section 4.4.1) is the only
    step-size controller in RELIC.
    """

    def forward(
        self,
        ode_func: ODEFunction,
        y0: Tensor,
        t_span: Tuple[float, float],
        dt: float,
    ) -> Tuple[Tensor, Tensor]:
        """
        Integrate an ODE over a time interval with repeated fixed solver steps.

        Parameters
        ----------
        ode_func : ODEFunction
            ODE right-hand side f(y, t). It is called with exactly (y, t);
            bind any further arguments with a closure or functools.partial.
        y0 : Tensor
            Initial condition at t_start
        t_span : Tuple[float, float]
            Integration interval (t_start, t_end)
        dt : float
            Fixed time step size. When (t_end - t_start) / dt is an integer
            within floating-point rounding, exactly that many steps are
            taken; otherwise one shorter final step closes the interval. The
            final time point equals t_end exactly.

        Returns
        -------
        Tuple[Tensor, Tensor]
            (time_points, solution_trajectory); solution_trajectory[k] is the
            state at time_points[k]. time_points is float64 on y0.device for
            every state dtype; solution_trajectory keeps the dtype of y0.

        Raises
        ------
        ValueError
            If t_span or dt is not finite, dt <= 0, t_end < t_start, or the
            accumulated rounding of the time grid leaves no positive final
            step.
        SolverStepError
            If any step ends with a status other than SUCCESS; the error
            carries the status, step index, start time, step length,
            iterations and residual norm. Integration stops at that step.
        """
        t_start, t_end = t_span
        if not (math.isfinite(t_start) and math.isfinite(t_end) and math.isfinite(dt)):
            raise ValueError(
                f"t_span and dt must be finite, got t_span={t_span}, dt={dt}"
            )
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        if t_end < t_start:
            raise ValueError("t_span must satisfy t_end >= t_start")

        # References:
        #   Higham, N.J. (2002). Accuracy and Stability of Numerical
        #   Algorithms, 2nd ed. SIAM, Sect. 2.2 (standard rounding model
        #   fl(a op b) = (a op b)(1 + delta), |delta| <= u = eps/2) and
        #   Sect. 4.2 (rounding error of recursive summation).
        # Mathematical expression:
        #   q = (t_end - t_start) / dt,  T = max(|t_start|, |t_end|)
        #   delta_q = eps * (T / dt + q)
        #   N = round(q)           if |q - round(q)| <= delta_q
        #   N = floor(q) + 1       otherwise;  N >= 1 whenever t_end > t_start
        # Parameter meanings:
        #   q: computed step ratio; T: magnitude of the time values; eps:
        #   machine epsilon of the Python float used for all time arithmetic
        #   (IEEE binary64, sys.float_info.epsilon); delta_q: bound on the
        #   rounding of q, from the representation of t_start and t_end
        #   (u*T each) and the subtraction and division (u*q each); N: number
        #   of steps.
        # Units:
        #   t_start, t_end, dt, T in the time unit used by ode_func; q,
        #   delta_q, eps and N dimensionless.
        # Domain/constraints:
        #   Finite inputs, dt > 0, t_end >= t_start (checked above). A
        #   remainder shorter than delta_q * dt (the documented relative
        #   tolerance of dt) is rounding, not a step: it is merged into the
        #   final step instead of producing a near-zero extra step.
        # Sign convention:
        #   Time advances forward; q >= 0 and N >= 0.
        machine_epsilon = sys.float_info.epsilon
        time_magnitude = max(abs(t_start), abs(t_end))
        step_ratio = (t_end - t_start) / dt
        if not math.isfinite(step_ratio):
            raise ValueError(
                f"(t_end - t_start) / dt is not finite for t_span={t_span}, dt={dt}"
            )
        ratio_tolerance = machine_epsilon * (time_magnitude / dt + step_ratio)
        nearest_step_count = round(step_ratio)
        if abs(step_ratio - nearest_step_count) <= ratio_tolerance:
            step_count = nearest_step_count
        else:
            step_count = math.floor(step_ratio) + 1
        if t_end > t_start:
            step_count = max(step_count, 1)

        t_points = [t_start]
        y_trajectory = [y0.clone()]

        t_current = t_start
        y_current = y0.clone()

        # References:
        #   Hairer, E. & Wanner, G. (1996). Solving Ordinary Differential
        #   Equations II. Springer. Burden, R.L. & Faires, J.D. (2010).
        #   Numerical Analysis, Ch. 5 (one-step methods). Brooks/Cole.
        # Mathematical expression:
        #   interior steps n < N-1:  h_n = dt,  t_{n+1} = t_n + dt
        #   final step n = N-1:      h_n = dt if t_n + dt == t_end
        #                            else t_end - t_n;  t_N = t_end
        #   y_{n+1} = Phi_{h_n}(y_n, t_n),  t_0 = t_start
        # Parameter meanings:
        #   dt: requested fixed step; h_n: step actually taken; Phi_h: one
        #   step of the concrete solver (self.step); y_n: state at t_n;
        #   N: step count from the block above.
        # Units:
        #   t_n, dt, h_n in the time unit used by ode_func; y_n in state units;
        #   ode_func returns state units per time unit.
        # Domain/constraints:
        #   h_n > 0 (a non-positive final step, possible only if accumulated
        #   rounding of t_n exceeds a step, raises ValueError); there is no
        #   step-size control; the recorded time advances by exactly the h_n
        #   passed to Phi, so y_trajectory[k] belongs to t_points[k].
        # Sign convention:
        #   Time advances forward (h_n > 0); ode_func returns dy/dt, and a
        #   positive value increases y.
        for step_index in range(step_count):
            t_candidate = t_current + dt
            if step_index < step_count - 1:
                step_size = dt
                t_next = t_candidate
            else:
                step_size = dt if t_candidate == t_end else t_end - t_current
                t_next = t_end
            if not step_size > 0.0:
                raise ValueError(
                    "Accumulated time rounding left no positive final step: "
                    f"t_n={t_current!r}, t_end={t_end!r}, dt={dt!r}"
                )

            y_current = self.step(
                ode_func, y_current, t_current, step_size, step_index=step_index
            )
            t_current = t_next

            t_points.append(t_current)
            y_trajectory.append(y_current.clone())

        # The time grid is computed in Python binary64 and returned in
        # float64 whatever the state dtype, so float32 states at large |t|
        # (float32 spacing 64 at 1e9) keep distinct time points.
        time_tensor = torch.tensor(t_points, device=y0.device, dtype=torch.float64)
        return time_tensor, torch.stack(y_trajectory)

    def step(
        self,
        ode_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        *,
        step_index: Optional[int] = None,
        **kwargs
    ) -> Tensor:
        """
        Take one solver step and return the next state tensor.

        Parameters
        ----------
        ode_func : ODEFunction
            ODE right-hand side f(y, t)
        y_current : Tensor
            State at t_current
        t_current : float
            Start time of the step
        dt : float
            Step length
        step_index : Optional[int]
            Zero-based index of the step within a trajectory, reported in a
            SolverStepError; None for a single step outside a trajectory
        **kwargs
            Passed to solve() and from there to ode_func

        Returns
        -------
        Tensor
            State at t_current + dt

        Raises
        ------
        SolverStepError
            If the step status is not SUCCESS (MAX_ITERATIONS or DIVERGED).
            The diagnostics counters are updated before the error is raised,
            and the non-converged or non-finite state is never returned.
        """
        result = self.solve(ode_func, y_current, t_current, dt, **kwargs)
        if result.status is not SolverStatus.SUCCESS:
            raise SolverStepError(
                solver_name=self.name,
                status=result.status,
                step_index=step_index,
                t_current=t_current,
                dt=dt,
                iterations=result.iterations,
                residual_norm=result.residual_norm,
                metadata=result.metadata,
            )
        return result.solution


@register_solver("backward_euler", family=SolverFamily.ODE_IMPLICIT)
class BackwardEulerSolver(_ODEForwardMixin, SolverBase):
    """
    Backward Euler method for implicit time integration of ODEs.
    
    This solver implements the backward Euler (implicit Euler) method, which
    is particularly suitable for stiff ODEs commonly found in biogeochemical
    models. The method is unconditionally stable (A-stable) and first-order
    accurate in time.
    
    Mathematical Expression:
        (y^{n+1} - y^n) / dt = f(y^{n+1}, t^{n+1})
        
        Rearranged as implicit system:
        y^{n+1} - dt * f(y^{n+1}, t^{n+1}) = y^n
        
        Solved via fixed-point iteration:
        y^{n+1,(k+1)} = y^n + dt * f(y^{n+1,(k)}, t^{n+1})
        
    Stability:
        - A-stable: Stable for all dt > 0 when applied to linear test equation
        - L-stable: Excellent damping of stiff high-frequency modes
        - Unconditionally stable for dissipative systems
        
    Convergence:
        - First-order accurate: Local error O(dt^2), Global error O(dt)
        - Suitable for stiff problems where explicit methods fail
        
    Parameters:
        max_iterations : int, default 50
            Maximum fixed-point iterations for implicit solve
        rtol : float, default 1e-6
            Relative convergence tolerance
        atol : float, default 1e-9
            Absolute convergence tolerance
        damping_factor : float, default 1.0
            Damping factor for fixed-point iterations (0 < damping <= 1)

    Implicit solve:
        The implicit equation is solved only by (damped) fixed-point
        iteration; there is no Newton option. The iteration map contracts
        only when dt times the Lipschitz constant of f (scaled by the
        damping) is below 1, so the A-stability of the scheme does not carry
        over to large stiff steps. A non-converged step is returned as
        MAX_ITERATIONS by solve() and raises SolverStepError in step() and
        forward().

    Examples:
        >>> solver = SolverFactory.create(
        ...     "backward_euler",
        ...     name="be_solver",
        ...     config={"max_iterations": 100, "rtol": 1e-8}
        ... )
        >>> def ode_func(y, t):
        ...     return -10.0 * y  # Stiff decay: dy/dt = -10*y
        >>> y0 = torch.tensor([1.0])
        >>> result = solver.solve(ode_func, y0, t_current=0.0, dt=0.1)
        >>> print(result.solution)  # y(0.1) using backward Euler
        
    References:
        - Hairer, E. & Wanner, G. (1996). Solving ODEs II, Section IV.1
        - Burden, R.L. & Faires, J.D. (2010). Numerical Analysis, Chapter 5
        - Ascher, U.M. & Petzold, L.R. (1998). Computer Methods for ODEs and DAEs
    """
    
    solver_family = SolverFamily.ODE_IMPLICIT
    
    def __init__(
        self,
        name: str = "backward_euler",
        operator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[SolverMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_iter: Optional[int] = None,
        max_iterations: Optional[int] = None,
        damping_factor: Optional[float] = None
    ):
        config = {} if config is None else dict(config)
        if max_iterations is None and max_iter is not None:
            max_iterations = max_iter
        if max_iterations is not None:
            config["max_iterations"] = max_iterations
        if rtol is not None:
            config["rtol"] = rtol
        if atol is not None:
            config["atol"] = atol
        if damping_factor is not None:
            config["damping_factor"] = damping_factor
        # Create metadata if not provided
        if metadata is None:
            metadata = SolverMetadata(
                name="Backward Euler",
                solver_family=SolverFamily.ODE_IMPLICIT,
                description="First-order implicit time integrator with A-stability",
                mathematical_expression=(
                    "y^{n+1} = y^n + dt * f(y^{n+1}, t^{n+1})\n"
                    "Solved via fixed-point iteration:\n"
                    "y^{n+1,(k+1)} = y^n + dt * f(y^{n+1,(k)}, t^{n+1})"
                ),
                order=1,
                stability_region="A-stable (entire left half-plane), L-stable",
                is_implicit=True,
                requires_jacobian=False,
                references=[
                    "Hairer, E. & Wanner, G. (1996). Solving ODEs II. Springer",
                    "Burden, R.L. & Faires, J.D. (2010). Numerical Analysis. Brooks/Cole",
                    "Ascher, U.M. & Petzold, L.R. (1998). Computer Methods for ODEs and DAEs. SIAM"
                ],
                parameters={
                    "max_iterations": "Maximum iterations for implicit solve (default 50)",
                    "rtol": "Relative convergence tolerance (default 1e-6)",
                    "atol": "Absolute convergence tolerance (default 1e-9)",
                    "damping_factor": "Damping for fixed-point iterations (default 1.0)"
                }
            )
        
        super().__init__(
            name,
            operator,
            config,
            metadata,
            device,
            dtype,
            parameter_namespace="computation.solvers.backward_euler"
        )

        # Get solver-specific parameters
        self.max_iterations = int(self.max_iterations)
        self.rtol = float(self.rtol)
        self.atol = float(self.atol)
        self.damping_factor = float(self.get_parameter("damping_factor"))
        # A damping of 0 would keep the start value and report it as converged;
        # the relaxation factor must lie in (0, 1] (same contract as IMEX).
        if not (0.0 < self.damping_factor <= 1.0):
            raise ValueError(
                f"damping_factor must be in (0, 1], got {self.damping_factor}"
            )
        
    def solve(
        self,
        ode_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        **kwargs
    ) -> SolverResult:
        """
        Solve one time step using Backward Euler.
        
        Parameters:
            ode_func : Callable[[Tensor, float], Tensor]
                Right-hand side function f(y, t)
                Must return a tensor with the same shape as y
            y_current : Tensor
                Current state vector at time t_current
            t_current : float
                Current time
            dt : float
                Time step size (must be positive)
            **kwargs : Additional arguments
                Passed to ode_func if needed
                
        Returns:
            SolverResult
                - solution: State vector at t_current + dt
                - status: SUCCESS if converged, MAX_ITERATIONS if not
                - iterations: Number of iterations performed
                - residual_norm: Final residual norm
                
        Raises:
            ValueError: If dt <= 0 or inputs are invalid
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        
        # Fixed-point iteration implementation
        return self._solve_fixed_point(ode_func, y_current, t_current, dt, **kwargs)

    def _solve_fixed_point(
        self,
        ode_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        **kwargs
    ) -> SolverResult:
        """Fixed-point implementation for implicit Backward Euler."""
        t_next = t_current + dt
        y_next = y_current.clone()

        # References: Hairer and Wanner (1996), Solving ODEs II, Sect. IV.8
        #     (fixed-point iteration of implicit Runge-Kutta stages); Kelley
        #     (1995), Iterative Methods for Linear and Nonlinear Equations, ch. 4.
        # Mathematical expression: G(y) = y^n + dt f(y, t^{n+1}); convergence
        #     is judged on the fixed-point residual r_k = G(y_k) - y_k; the
        #     relaxed update is y_{k+1} = omega G(y_k) + (1 - omega) y_k.
        # Parameter meanings: y^n = state at t^n, y_k = k-th iterate of y^{n+1},
        #     omega = damping_factor, dt = time step.
        # Units: y, G(y) and r_k carry the state units; dt the time units of f;
        #     omega is dimensionless.
        # Domain/constraints: 0 < omega <= 1 (checked in __init__). The residual
        #     is the undamped r_k: the relaxed increment omega r_k would
        #     understate it by the factor omega and accept an unconverged step.
        # Sign convention: r_k points from the iterate towards G(y_k).
        for k in range(self.max_iterations):
            # Evaluate RHS at current iterate
            f_val = ode_func(y_next, t_next, **kwargs)
            y_new = y_current + dt * f_val

            residual = y_new - y_next
            if self.check_convergence(residual, y_new):
                return SolverResult(
                    solution=y_new,
                    status=SolverStatus.SUCCESS,
                    iterations=k + 1,
                    residual_norm=float(torch.max(torch.abs(residual)).item())
                )

            y_next = self.damping_factor * y_new + (1 - self.damping_factor) * y_next
        
        # Max iterations reached
        logger.warning(
            f"Backward Euler did not converge in {self.max_iterations} iterations"
        )
        return SolverResult(
            solution=y_next,
            status=SolverStatus.MAX_ITERATIONS,
            iterations=self.max_iterations,
            residual_norm=float(torch.max(torch.abs(residual)).item())
        )


@register_solver("implicit_midpoint", family=SolverFamily.ODE_IMPLICIT)
class ImplicitMidpointSolver(_ODEForwardMixin, SolverBase):
    """
    Implicit midpoint method for structure-preserving ODE integration.

    Mathematical Expression:
        y^{n+1} = y^n + dt * f((y^n + y^{n+1}) / 2, t^n + dt/2)

    Fixed-point iteration:
        y^{n+1,(k+1)} = y^n + dt * f((y^n + y^{n+1,(k)}) / 2, t^n + dt/2)

    Properties:
        - Second-order accurate in time
        - Symplectic for Hamiltonian systems
        - Preserves quadratic invariants for linear systems

    References:
        - Hairer, E., Lubich, C., and Wanner, G. (2006). Geometric Numerical Integration.
        - Leimkuhler, B., and Reich, S. (2004). Simulating Hamiltonian Dynamics.
    """

    solver_family = SolverFamily.ODE_IMPLICIT

    def __init__(
        self,
        name: str = "implicit_midpoint",
        operator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[SolverMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_iter: Optional[int] = None,
        max_iterations: Optional[int] = None
    ):
        config = {} if config is None else dict(config)
        if max_iterations is None and max_iter is not None:
            max_iterations = max_iter
        if max_iterations is not None:
            config["max_iterations"] = max_iterations
        if rtol is not None:
            config["rtol"] = rtol
        if atol is not None:
            config["atol"] = atol

        if metadata is None:
            metadata = SolverMetadata(
                name="Implicit Midpoint",
                solver_family=SolverFamily.ODE_IMPLICIT,
                description="Second-order implicit midpoint integrator",
                mathematical_expression=(
                    "y^{n+1} = y^n + dt * f((y^n + y^{n+1})/2, t^n + dt/2)\n"
                    "Fixed-point: y^{n+1,(k+1)} = y^n + dt * f((y^n + y^{n+1,(k)})/2)"
                ),
                order=2,
                stability_region="A-stable, symplectic for Hamiltonian systems",
                is_implicit=True,
                requires_jacobian=False,
                references=[
                    "Hairer, E., Lubich, C., and Wanner, G. (2006). Geometric Numerical Integration. Springer.",
                    "Leimkuhler, B., and Reich, S. (2004). Simulating Hamiltonian Dynamics. Cambridge."
                ],
                parameters={
                    "max_iterations": "Maximum iterations for implicit solve",
                    "rtol": "Relative convergence tolerance",
                    "atol": "Absolute convergence tolerance"
                }
            )

        super().__init__(
            name,
            operator,
            config,
            metadata,
            device,
            dtype,
            parameter_namespace="computation.solvers.implicit_midpoint"
        )

        self.max_iterations = int(self.max_iterations)
        self.rtol = float(self.rtol)
        self.atol = float(self.atol)

    def solve(
        self,
        ode_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        **kwargs
    ) -> SolverResult:
        """Solve one step using implicit midpoint fixed-point iteration."""
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        return self._solve_fixed_point(ode_func, y_current, t_current, dt, **kwargs)

    def _solve_fixed_point(
        self,
        ode_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        **kwargs
    ) -> SolverResult:
        """Fixed-point implementation of implicit midpoint."""
        t_mid = t_current + 0.5 * dt
        y_next = y_current.clone()

        for k in range(self.max_iterations):
            y_mid = 0.5 * (y_current + y_next)
            f_mid = ode_func(y_mid, t_mid, **kwargs)
            y_new = y_current + dt * f_mid

            residual = y_new - y_next
            if self.check_convergence(residual, y_new):
                return SolverResult(
                    solution=y_new,
                    status=SolverStatus.SUCCESS,
                    iterations=k + 1,
                    residual_norm=float(torch.max(torch.abs(residual)).item())
                )

            y_next = y_new

        logger.warning(
            "Implicit midpoint did not converge in %s iterations",
            self.max_iterations,
        )
        return SolverResult(
            solution=y_next,
            status=SolverStatus.MAX_ITERATIONS,
            iterations=self.max_iterations,
            residual_norm=float(torch.max(torch.abs(residual)).item())
        )


@register_solver("crank_nicolson", family=SolverFamily.ODE_IMPLICIT)
class CrankNicolsonSolver(_ODEForwardMixin, SolverBase):
    """
    Crank-Nicolson method for implicit time integration of ODEs.
    
    This solver implements the Crank-Nicolson (trapezoidal rule) method,
    which is second-order accurate and A-stable. It is symmetric in time
    and particularly suitable for diffusion-type problems.
    
    Mathematical Expression:
        y^{n+1} = y^n + (dt/2) * [f(y^n, t^n) + f(y^{n+1}, t^{n+1})]
        
        Rearranged as:
        y^{n+1} - (dt/2)*f(y^{n+1}, t^{n+1}) = y^n + (dt/2)*f(y^n, t^n)
        
    Theta-Method Generalization:
        y^{n+1} = y^n + dt * [theta*f(y^{n+1}, t^{n+1}) + (1-theta)*f(y^n, t^n)]
        
        theta = 0:   Forward Euler (explicit, first-order)
        theta = 0.5: Crank-Nicolson (implicit, second-order)
        theta = 1:   Backward Euler (implicit, first-order)
        
    Stability:
        - A-stable for theta >= 0.5
        - Second-order accurate only for theta = 0.5
        - Oscillatory for marginally stable problems (use damping if needed)
        
    Parameters:
        theta : float, default 0.5
            Theta parameter (0 <= theta <= 1)
            theta=0.5 gives classical Crank-Nicolson
        max_iterations : int, default 50
            Maximum iterations for implicit solve
        rtol : float, default 1e-6
            Relative convergence tolerance
        atol : float, default 1e-9
            Absolute convergence tolerance
            
    Examples:
        >>> solver = SolverFactory.create(
        ...     "crank_nicolson",
        ...     name="cn_solver",
        ...     config={"theta": 0.5, "max_iterations": 100}
        ... )
        >>> def diffusion(y, t):
        ...     # 1D diffusion: dy/dt = alpha * d2y/dx2
        ...     return alpha * laplacian(y)
        >>> result = solver.solve(diffusion, y0, t_current=0.0, dt=0.01)
        
    References:
        - Crank, J. & Nicolson, P. (1947). A practical method for numerical evaluation
          of solutions of PDEs of the heat-conduction type. Proc. Cambridge Phil. Soc. 43
        - Burden, R.L. & Faires, J.D. (2010). Numerical Analysis, Chapter 9
        - Thomas, J.W. (1995). Numerical Partial Differential Equations. Springer
    """
    
    solver_family = SolverFamily.ODE_IMPLICIT
    
    def __init__(
        self,
        name: str = "crank_nicolson",
        operator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[SolverMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        theta: Optional[float] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_iter: Optional[int] = None,
        max_iterations: Optional[int] = None
    ):
        config = {} if config is None else dict(config)
        if max_iterations is None and max_iter is not None:
            max_iterations = max_iter
        if max_iterations is not None:
            config["max_iterations"] = max_iterations
        if rtol is not None:
            config["rtol"] = rtol
        if atol is not None:
            config["atol"] = atol
        if theta is not None:
            config["theta"] = theta
        # Create metadata if not provided
        if metadata is None:
            metadata = SolverMetadata(
                name="Crank-Nicolson",
                solver_family=SolverFamily.ODE_IMPLICIT,
                description="Second-order implicit time integrator (trapezoidal rule)",
                mathematical_expression=(
                    "y^{n+1} = y^n + (dt/2) * [f(y^n, t^n) + f(y^{n+1}, t^{n+1})]\n"
                    "General theta-method:\n"
                    "y^{n+1} = y^n + dt * [theta*f(y^{n+1}) + (1-theta)*f(y^n)]"
                ),
                order=2,  # Second-order for theta=0.5
                stability_region="A-stable for theta >= 0.5",
                is_implicit=True,
                requires_jacobian=False,
                references=[
                    "Crank, J. & Nicolson, P. (1947). Proc. Cambridge Phil. Soc. 43:50-67",
                    "Burden, R.L. & Faires, J.D. (2010). Numerical Analysis. Brooks/Cole",
                    "Thomas, J.W. (1995). Numerical PDEs. Springer"
                ],
                parameters={
                    "theta": "Theta parameter (0.5 for Crank-Nicolson, default 0.5)",
                    "max_iterations": "Maximum iterations for implicit solve (default 50)",
                    "rtol": "Relative convergence tolerance (default 1e-6)",
                    "atol": "Absolute convergence tolerance (default 1e-9)"
                }
            )
        
        super().__init__(
            name,
            operator,
            config,
            metadata,
            device,
            dtype,
            parameter_namespace="computation.solvers.crank_nicolson"
        )

        # Get theta parameter
        self.max_iterations = int(self.max_iterations)
        self.rtol = float(self.rtol)
        self.atol = float(self.atol)
        self.theta = float(self.get_parameter("theta"))
        # The theta-method weight is defined on [0, 1] (0 explicit, 1 implicit
        # Euler); a configured value outside that range is rejected.
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")
        
        # Update metadata order based on theta
        if abs(self.theta - 0.5) > 1e-10:
            self.metadata.order = 1  # Only second-order for theta=0.5
        
    def solve(
        self,
        ode_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        **kwargs
    ) -> SolverResult:
        """
        Solve one time step using Crank-Nicolson (theta-method).
        
        Parameters:
            ode_func : Callable[[Tensor, float], Tensor]
                Right-hand side function f(y, t)
            y_current : Tensor
                Current state vector at time t_current
            t_current : float
                Current time
            dt : float
                Time step size (must be positive)
            **kwargs : Additional arguments
                Passed to ode_func if needed
                
        Returns:
            SolverResult
                - solution: State vector at t_current + dt
                - status: SUCCESS if converged
                - iterations: Number of iterations performed
                
        Raises:
            ValueError: If dt <= 0
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        
        # Fixed-point iteration implementation
        return self._solve_fixed_point(ode_func, y_current, t_current, dt, **kwargs)

    def _solve_fixed_point(
        self,
        ode_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        **kwargs
    ) -> SolverResult:
        """Fixed-point theta-method implementation for Crank-Nicolson."""
        t_next = t_current + dt
        
        # Compute explicit part: y_n + (1-theta)*dt*f(y_n, t_n)
        f_current = ode_func(y_current, t_current, **kwargs)
        explicit_part = y_current + (1 - self.theta) * dt * f_current
        
        # Solve implicit part using fixed-point iteration
        y_next = y_current.clone()
        
        for k in range(self.max_iterations):
            f_next = ode_func(y_next, t_next, **kwargs)
            y_new = explicit_part + self.theta * dt * f_next
            
            residual = y_new - y_next
            if self.check_convergence(residual, y_new):
                return SolverResult(
                    solution=y_new,
                    status=SolverStatus.SUCCESS,
                    iterations=k + 1,
                    residual_norm=float(torch.max(torch.abs(residual)).item())
                )
            
            y_next = y_new
        
        logger.warning(
            f"Crank-Nicolson did not converge in {self.max_iterations} iterations"
        )
        return SolverResult(
            solution=y_next,
            status=SolverStatus.MAX_ITERATIONS,
            iterations=self.max_iterations,
            residual_norm=float(torch.max(torch.abs(residual)).item())
        )


@register_solver("imex_euler", family=SolverFamily.ODE_IMEX)
class IMEXEulerSolver(SolverBase):
    """
    IMEX Euler solver for split explicit/implicit ODEs.

    This solver advances ODEs of the form:
        y' = f(y, t) + g(y, t)
    using a first-order implicit-explicit scheme. The explicit component is
    optionally substepped to emulate a multirate update.

    Parameters
    ----------
    name : str
        Solver instance name.
    operator : Optional[Any]
        Optional operator (unused for IMEX Euler).
    config : Optional[Dict[str, Any]]
        Solver configuration overrides.
    metadata : Optional[SolverMetadata]
        Metadata override.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Tensor dtype for computations.
    """

    solver_family = SolverFamily.ODE_IMEX

    def __init__(
        self,
        name: str,
        operator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[SolverMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        if metadata is None:
            metadata = SolverMetadata(
                name="IMEX Euler",
                solver_family=SolverFamily.ODE_IMEX,
                description="First-order IMEX Euler with optional explicit substepping",
                mathematical_expression=(
                    "y_{n+1} = y_n + dt * f(y_n, t_n) + dt * g(y_{n+1}, t_{n+1})"
                ),
                order=1,
                stability_region="A-stable for implicit part, conditional for explicit part",
                is_implicit=True,
                requires_jacobian=False,
                references=[
                    "Hairer, E. and Wanner, G. (1996). Solving ODEs II. Springer",
                    "Ascher, U.M., Ruuth, S.J., and Spiteri, R.J. (1997). Applied Numerical Mathematics",
                ],
                parameters={
                    "explicit_substeps": "Number of explicit substeps (>=1)",
                    "damping_factor": "Fixed-point damping factor in (0, 1]"
                }
            )

        super().__init__(
            name,
            operator,
            config,
            metadata,
            device,
            dtype,
            parameter_namespace="computation.solvers.imex_euler"
        )

        self.max_iterations = int(self.max_iterations)
        self.explicit_substeps = int(self.get_parameter("explicit_substeps"))
        self.damping_factor = float(self.get_parameter("damping_factor"))
        if self.explicit_substeps < 1:
            raise ValueError("explicit_substeps must be >= 1")
        if not (0.0 < self.damping_factor <= 1.0):
            raise ValueError("damping_factor must be in (0, 1]")

    def solve(
        self,
        explicit_func: ODEFunction,
        implicit_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        **kwargs
    ) -> SolverResult:
        """
        Advance one step using IMEX Euler splitting.

        Parameters
        ----------
        explicit_func : Callable[[Tensor, float], Tensor]
            Explicit RHS function f(y, t).
        implicit_func : Callable[[Tensor, float], Tensor]
            Implicit RHS function g(y, t).
        y_current : Tensor
            Current state at time t_current.
        t_current : float
            Current time.
        dt : float
            Time step size (must be positive).
        **kwargs
            Extra arguments forwarded to the RHS functions.

        Returns
        -------
        SolverResult
            Solution at t_current + dt with status and diagnostics.
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        if implicit_func is None or explicit_func is None:
            raise ValueError("IMEX Euler requires both explicit and implicit functions")

        # References:
        # - Hairer, E. and Wanner, G. (1996). Solving ODEs II. Springer.
        # - Ascher, U.M., Ruuth, S.J., and Spiteri, R.J. (1997). Applied Numerical Mathematics.
        # Formula:
        #   y_{n+1} = y_n + dt * f(y_n, t_n) + dt * g(y_{n+1}, t_{n+1})
        # Parameters:
        #   explicit_substeps: number of explicit substeps (>=1)
        #   damping_factor: fixed-point relaxation factor in (0, 1]
        t_next = t_current + dt

        # Explicit substepping for non-stiff component.
        dt_explicit = dt / float(self.explicit_substeps)
        y_explicit = y_current
        t_explicit = t_current
        for _ in range(self.explicit_substeps):
            f_val = explicit_func(y_explicit, t_explicit, **kwargs)
            y_explicit = y_explicit + dt_explicit * f_val
            t_explicit += dt_explicit

        # Fixed-point iteration for implicit component.
        y_next = y_explicit.clone()
        for k in range(self.max_iterations):
            g_val = implicit_func(y_next, t_next, **kwargs)
            y_new = y_explicit + dt * g_val
            residual = y_new - y_next
            if self.check_convergence(residual, y_new):
                return SolverResult(
                    solution=y_new,
                    status=SolverStatus.SUCCESS,
                    iterations=k + 1,
                    residual_norm=float(torch.max(torch.abs(residual)).item()),
                    metadata={
                        "method": "imex_euler",
                        "explicit_substeps": self.explicit_substeps,
                        "dt": dt,
                        "t_current": t_current,
                        "t_next": t_next
                    }
                )
            y_next = self.damping_factor * y_new + (1.0 - self.damping_factor) * y_next

        logger.warning(
            "IMEX Euler did not converge in %s iterations", self.max_iterations
        )
        return SolverResult(
            solution=y_next,
            status=SolverStatus.MAX_ITERATIONS,
            iterations=self.max_iterations,
            residual_norm=float(torch.max(torch.abs(residual)).item()),
            metadata={
                "method": "imex_euler",
                "explicit_substeps": self.explicit_substeps,
                "dt": dt,
                "t_current": t_current,
                "t_next": t_next
            }
        )


@register_solver("rk4", family=SolverFamily.ODE_EXPLICIT)
class RK4Solver(_ODEForwardMixin, SolverBase):
    """
    Fourth-order Runge-Kutta method for explicit time integration.
    
    This solver implements the classical RK4 method, which is fourth-order
    accurate and widely used for non-stiff ODEs. It provides excellent
    accuracy per function evaluation for smooth problems.
    
    Mathematical Expression:
        k1 = f(y_n, t_n)
        k2 = f(y_n + (dt/2)*k1, t_n + dt/2)
        k3 = f(y_n + (dt/2)*k2, t_n + dt/2)
        k4 = f(y_n + dt*k3, t_n + dt)
        
        y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
    Stability:
        - Conditionally stable (explicit method)
        - Stability region: [-2.78, 0] along real axis for linear test equation
        - Requires small dt for stiff problems
        - Not suitable for stiff ODEs (use implicit methods instead)
        
    Convergence:
        - Fourth-order accurate: Global error O(dt^4)
        - Requires 4 function evaluations per step
        - Optimal for smooth, non-stiff problems
        
    Parameters:
        None specific to RK4. The method is explicit and fixed-step: the
        step size is the dt given to each solve() call, and there is no
        step-size control (see architecture_overview.md Section 4.4.1).

    Examples:
        >>> solver = SolverFactory.create("rk4", name="rk4_solver", config={})
        >>> def oscillator(y, t):
        ...     # Harmonic oscillator: d2y/dt2 = -omega^2 * y
        ...     # Convert to first-order system: [y, dy/dt]
        ...     omega = 2.0 * 3.14159
        ...     return torch.tensor([y[1], -omega**2 * y[0]])
        >>> y0 = torch.tensor([1.0, 0.0])  # Initial: y=1, dy/dt=0
        >>> result = solver.solve(oscillator, y0, t_current=0.0, dt=0.01)
        
    References:
        - Runge, C. (1895). Ueber die numerische Aufloesung von Differentialgleichungen
        - Kutta, W. (1901). Beitrag zur naeherungsweisen Integration totaler Differentialgleichungen
        - Burden, R.L. & Faires, J.D. (2010). Numerical Analysis, Chapter 5
        - Butcher, J.C. (2008). Numerical Methods for ODEs. Wiley
    """
    
    solver_family = SolverFamily.ODE_EXPLICIT
    
    def __init__(
        self,
        name: str,
        operator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[SolverMetadata] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        # Create metadata if not provided
        if metadata is None:
            metadata = SolverMetadata(
                name="Runge-Kutta 4",
                solver_family=SolverFamily.ODE_EXPLICIT,
                description="Fourth-order explicit Runge-Kutta method (classical RK4)",
                mathematical_expression=(
                    "k1 = f(y_n, t_n)\n"
                    "k2 = f(y_n + dt*k1/2, t_n + dt/2)\n"
                    "k3 = f(y_n + dt*k2/2, t_n + dt/2)\n"
                    "k4 = f(y_n + dt*k3, t_n + dt)\n"
                    "y_{n+1} = y_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)"
                ),
                order=4,
                stability_region="Conditionally stable, stability region [-2.78, 0] on real axis",
                is_implicit=False,
                requires_jacobian=False,
                references=[
                    "Runge, C. (1895). Math. Ann. 46:167-178",
                    "Kutta, W. (1901). Z. Math. Phys. 46:435-453",
                    "Burden, R.L. & Faires, J.D. (2010). Numerical Analysis. Brooks/Cole",
                    "Butcher, J.C. (2008). Numerical Methods for ODEs. Wiley"
                ],
            )
        
        super().__init__(
            name,
            operator,
            config,
            metadata,
            device,
            dtype,
            parameter_namespace="computation.solvers.rk4"
        )

    def solve(
        self,
        ode_func: ODEFunction,
        y_current: Tensor,
        t_current: float,
        dt: float,
        **kwargs
    ) -> SolverResult:
        """
        Solve one time step using classical RK4 method.
        
        Parameters:
            ode_func : Callable[[Tensor, float], Tensor]
                Right-hand side function f(y, t)
            y_current : Tensor
                Current state vector at time t_current
            t_current : float
                Current time
            dt : float
                Time step size (must be positive)
            **kwargs : Additional arguments
                Passed to ode_func if needed
                
        Returns:
            SolverResult
                - solution: State vector at t_current + dt
                - status: SUCCESS if every element of the solution is finite;
                  DIVERGED if the computed solution contains non-finite
                  values (the non-finite solution is returned unchanged and
                  metadata["divergence_reason"] is "non_finite_solution")
                - iterations: 4 (number of RHS evaluations)

        Raises:
            ValueError: If dt <= 0
            Any exception raised by ode_func or by the tensor arithmetic
            propagates unchanged (fail fast).

        Notes:
            RK4 is an explicit method requiring 4 function evaluations per step.
            It is not suitable for stiff problems - use implicit methods instead.
        """
        if dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {dt}")

        # References:
        #   Kutta, W. (1901). Z. Math. Phys. 46:435-453. Butcher, J.C. (2008).
        #   Numerical Methods for Ordinary Differential Equations. Wiley.
        #   Burden, R.L. & Faires, J.D. (2010). Numerical Analysis, Sect. 5.4.
        # Mathematical expression:
        #   k1 = f(y_n, t_n)
        #   k2 = f(y_n + dt*k1/2, t_n + dt/2)
        #   k3 = f(y_n + dt*k2/2, t_n + dt/2)
        #   k4 = f(y_n + dt*k3, t_n + dt)
        #   y_{n+1} = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        #   status = SUCCESS if all(isfinite(y_{n+1})) else DIVERGED
        # Parameter meanings:
        #   y_n: state at t_n; f: right-hand side ode_func; k1..k4: stage
        #   slopes; dt: step size; y_{n+1}: state at t_n + dt.
        # Units:
        #   y_n and y_{n+1} in state units; t_n and dt in the time unit of f;
        #   k1..k4 in state units per time unit.
        # Domain/constraints:
        #   dt > 0 (checked above); explicit and conditionally stable (real
        #   stability interval about [-2.78, 0] for dt*lambda); no step-size
        #   control. Exceptions from f are not caught; only a computed
        #   non-finite y_{n+1} is reported as DIVERGED.
        # Sign convention:
        #   f returns dy/dt; a positive slope increases y; time advances
        #   forward from t_n to t_n + dt.
        k1 = ode_func(y_current, t_current, **kwargs)
        k2 = ode_func(y_current + 0.5 * dt * k1, t_current + 0.5 * dt, **kwargs)
        k3 = ode_func(y_current + 0.5 * dt * k2, t_current + 0.5 * dt, **kwargs)
        k4 = ode_func(y_current + dt * k3, t_current + dt, **kwargs)
        y_next = y_current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        metadata = {
            "method": "rk4",
            "dt": dt,
            "t_current": t_current,
            "t_next": t_current + dt,
            "function_evaluations": 4,
        }
        non_finite_elements = int(torch.count_nonzero(~torch.isfinite(y_next)).item())
        if non_finite_elements > 0:
            metadata["divergence_reason"] = "non_finite_solution"
            metadata["non_finite_elements"] = non_finite_elements
            logger.error(
                "RK4 step from t=%s with dt=%s produced %d non-finite element(s)",
                t_current,
                dt,
                non_finite_elements,
            )
            status = SolverStatus.DIVERGED
        else:
            status = SolverStatus.SUCCESS

        return SolverResult(
            solution=y_next,
            status=status,
            iterations=4,  # 4 function evaluations
            residual_norm=None,
            metadata=metadata,
        )


def solve_ode_system(
    ode_func: ODEFunction,
    y0: Tensor,
    t_span: Tuple[float, float],
    method: str = "backward_euler",
    dt: float = 0.01,
    **solver_kwargs
) -> Tuple[Tensor, Tensor]:
    """
    Convenience function to solve ODE system with fixed steps.

    Parameters
    ----------
    ode_func : ODEFunction
        ODE right-hand side function f(y, t)
    y0 : Tensor
        Initial condition
    t_span : Tuple[float, float]
        Integration interval
    method : str, optional
        Integration method ("backward_euler", "implicit_midpoint",
        "crank_nicolson", "rk4"), default "backward_euler"
    dt : float, optional
        Fixed time step size, default 0.01
    **solver_kwargs
        Keyword arguments of the selected solver constructor; an argument the
        constructor does not declare raises TypeError

    Returns
    -------
    Tuple[Tensor, Tensor]
        (time_points, solution_trajectory)

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported methods. No explicit Euler
        ODE solver is registered, so "forward_euler" is rejected.
    """
    solver_factories: Dict[str, Callable[[], SolverBase]] = {
        "backward_euler": lambda: BackwardEulerSolver(**solver_kwargs),
        "implicit_midpoint": lambda: ImplicitMidpointSolver(**solver_kwargs),
        "crank_nicolson": lambda: CrankNicolsonSolver(**solver_kwargs),
        "rk4": lambda: RK4Solver(name="rk4_convenience", **solver_kwargs),
    }
    if method not in solver_factories:
        raise ValueError(
            f"Unknown method {method!r}; supported methods: "
            f"{', '.join(solver_factories)}"
        )
    solver = solver_factories[method]()

    return solver.forward(ode_func, y0, t_span, dt)


# Export temporal solvers
__all__ = [
    'BackwardEulerSolver',
    'ImplicitMidpointSolver',
    'CrankNicolsonSolver',
    'IMEXEulerSolver',
    'RK4Solver',
    'solve_ode_system',
]
