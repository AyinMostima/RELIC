"""Standalone numerical solvers for explicitly declared Ito SDE operators.

This module owns numerical integration only. Scientific Functors own the
drift and diffusion operators, variables, units, parameters, and structural
evidence supplied to this solver.

Mathematical Framework:
    Diagonal Ito SDE: dX(t) = a(X,t)dt + b(X,t) odot dW(t)
    Euler-Maruyama: X_{n+1} = X_n + a_n dt + b_n odot sqrt(dt) xi_n
    where xi_n ~ N(0,1) are independent standard normal random variables

References:
    - Kloeden, P.E. & Platen, E. (1992). Numerical Solution of Stochastic 
      Differential Equations. Springer.
    - Gardiner, C. (2009). Stochastic Methods: A Handbook for the Natural
      and Social Sciences. Springer, 4th edition.
    - Higham, D.J. (2001). An algorithmic introduction to numerical simulation
      of stochastic differential equations. SIAM Rev. 43, 525-546.
      
Authors: Yuchong Long
"""

from abc import abstractmethod
from dataclasses import dataclass, field
import logging
from numbers import Real
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from .category import StochasticProcess, ProcessMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SDEParameters:
    """Fixed-step Ito SDE numerical controls.

    References:
        Kloeden and Platen (1992), Chapter 10; Higham (2001).
    Mathematical expression:
        t_n = n * dt and Delta W_n ~ N(0, dt I).
    Parameter meanings:
        dt is an explicitly supplied time increment in the caller's declared
        time unit. method (required) identifies the numerical formula; only
        'euler_maruyama' is operational. adaptive records an explicitly
        rejected [N] request until Wiener refinement is owned.
        diffusion_structure (required) declares the noise operator; only the
        implemented 'diagonal' operator is accepted.
    """

    dt: float
    method: str = field(kw_only=True)
    adaptive: bool = False
    diffusion_structure: str = field(kw_only=True)

    def __post_init__(self) -> None:
        """Reject ambiguous or non-finite fixed-step controls."""
        valid_scalar = not isinstance(self.dt, bool) and isinstance(self.dt, Real)
        dt_value = torch.as_tensor(
            float(self.dt) if valid_scalar else float("nan"),
            dtype=torch.float64,
        )
        if not valid_scalar or not bool(torch.isfinite(dt_value)) or bool(
            dt_value <= 0.0
        ):
            raise ValueError("SDE dt must be a finite positive real scalar")
        if type(self.adaptive) is not bool:
            raise ValueError("SDE adaptive must be a boolean")
        if self.diffusion_structure != "diagonal":
            raise ValueError(
                "SDE diffusion_structure must be 'diagonal'; general matrix "
                "and correlated diffusion are [N]."
            )


class SDESolver(StochasticProcess):
    """
    Base class for stochastic differential equation solvers.
    
    This abstract class defines the interface for SDE solvers that can be
    integrated into RELIC's biogeochemical modeling framework. Solvers
    must handle PyTorch tensors and support automatic differentiation.
    """
    
    def __init__(self, parameters: SDEParameters):
        """
        Initialize SDE solver.
        
        Parameters:
            parameters (SDEParameters): Solver configuration
        """
        if not isinstance(parameters, SDEParameters):
            raise TypeError("parameters must be an SDEParameters instance")
        if parameters.method in {"milstein", "predictor_corrector"}:
            raise ValueError(
                f"SDE method '{parameters.method}' is not operational; "
                "only fixed-step Euler-Maruyama has an executable contract."
            )
        if parameters.method != "euler_maruyama":
            raise ValueError(f"Unknown SDE method: {parameters.method}")
        if parameters.adaptive:
            raise ValueError(
                "The standalone Euler-Maruyama solver is fixed-step. Variable "
                "steps on one Wiener path are owned by the simulator adaptive "
                "step control together with the Functor-owned Brownian tree "
                "(FunctorSubstreamMixin.draw_wiener_increment); pass the "
                "resulting increments through 'wiener_increments'."
            )
        metadata = ProcessMetadata(
            name=type(self).__name__,
            description="Fixed-step Euler-Maruyama numerical integrator",
            units={
                "solution": "declared-by-caller",
                "time": "same-as-dt",
            },
            parameters={
                "method": parameters.method,
                "dt": parameters.dt,
                "adaptive": parameters.adaptive,
                "diffusion_structure": parameters.diffusion_structure,
            },
            references=(
                "Kloeden and Platen (1992); Higham (2001); Gardiner (2009)"
            ),
        )
        super().__init__(metadata)
        self.params = parameters
        
        logger.info(f"Initialized SDE solver: method={parameters.method}, "
                   f"dt={parameters.dt}, adaptive={parameters.adaptive}")
    
    @abstractmethod
    def drift_function(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        """
        Drift coefficient mu(X,t) in the SDE dX = mu(X,t)dt + sigma(X,t)dW.
        
        Parameters:
            x (Tensor): Current state
            t (Tensor): Current time  
            **kwargs: Additional parameters
            
        Returns:
            Tensor: Drift coefficient with same shape as x
        """
        
    @abstractmethod  
    def diffusion_function(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        """
        Diffusion coefficient sigma(X,t) in the SDE dX = mu(X,t)dt + sigma(X,t)dW.
        
        Parameters:
            x (Tensor): Current state
            t (Tensor): Current time
            **kwargs: Additional parameters
            
        Returns:
            Tensor: Diffusion coefficient with same shape as x
        """

    def sample(
        self,
        input_state: Tensor,
        num_samples: int = 1,
        seed: Optional[int] = None,
        num_steps: int = 100,
        generator: Optional[torch.Generator] = None,
        initial_time: Union[float, int, Tensor] = 0.0,
        wiener_increments: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Solve SDE using the specified numerical method.

        ``wiener_increments`` (shape ``(num_samples, num_steps, *batch)``)
        supplies path increments Delta W_n drawn by the caller, for example
        by the Functor-owned Brownian tree, so the fixed-step scheme can be
        applied on a path that survives step-size changes; when omitted the
        solver draws Delta W_n ~ N(0, dt I) from ``generator``.
        
        Parameters:
            input_state (Tensor): Initial condition X_0
            num_samples (int): Number of sample paths  
            seed (int, optional): Random seed for reproducibility
            num_steps (int): Number of integration steps
            generator (torch.Generator, optional): Caller-owned RNG substream
            initial_time (Real): Time coordinate for the first returned state
            **kwargs: Additional parameters for drift/diffusion functions
            
        Returns:
            Dict[str, Tensor]: Solution data
                'solution': Solution paths [shape: (num_samples, num_steps+1, ...)]
                'time': Time grid [shape: (num_steps+1,)]
                'dt_used': Fixed time-step history
                'drift_history': Drift values at each step
                'diffusion_history': Diffusion values at each step
        """
        if not isinstance(input_state, Tensor):
            raise TypeError("input_state must be a torch.Tensor")
        if not torch.is_floating_point(input_state):
            raise TypeError("input_state must use a floating-point dtype")
        if not torch.isfinite(input_state).all():
            raise ValueError("input_state must contain only finite values")
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")
        if seed is not None and (
            type(seed) is not int or seed < 0 or seed >= 2**63
        ):
            raise ValueError("seed must be a nonnegative integer below 2**63")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        valid_initial_time = (
            not isinstance(initial_time, bool)
            and isinstance(initial_time, Real)
        ) or (
            isinstance(initial_time, Tensor)
            and initial_time.numel() == 1
            and torch.is_floating_point(initial_time)
        )
        if not valid_initial_time:
            raise ValueError("initial_time must be a finite real scalar")

        device = input_state.device
        dtype = input_state.dtype
        batch_shape = input_state.shape
        initial_time_tensor = torch.as_tensor(
            initial_time,
            device=device,
            dtype=dtype,
        )
        if initial_time_tensor.numel() != 1 or not bool(
            torch.isfinite(initial_time_tensor)
        ):
            raise ValueError("initial_time must be a finite real scalar")
        # The seed of the local stream is recorded in the result (an explicit
        # seed, or the one Generator.seed() draws) so the run can be
        # reproduced; a caller-supplied generator carries the caller's state.
        used_seed: Optional[int] = None
        if generator is None:
            active_generator = torch.Generator(device=device)
            if seed is None:
                used_seed = int(active_generator.seed())
            else:
                used_seed = int(seed)
                active_generator.manual_seed(used_seed)
        else:
            # A CUDA generator reports its device without an index while a
            # tensor on the same accelerator reports one; compare the type and,
            # when both sides carry an index, the index.
            generator_device = torch.device(generator.device)
            same_type = generator_device.type == device.type
            same_index = (
                generator_device.index is None
                or device.index is None
                or generator_device.index == device.index
            )
            if not (same_type and same_index):
                raise ValueError(
                    "generator device must match the input_state device"
                )
            active_generator = generator

        dt_tensor = torch.as_tensor(
            self.params.dt,
            device=device,
            dtype=dtype,
        )
        
        time_grid = (
            torch.arange(num_steps + 1, device=device, dtype=dtype)
            * dt_tensor
            + initial_time_tensor
        )
        dt_history = dt_tensor.expand(num_steps).clone()
        current_state = input_state.unsqueeze(0).expand(
            num_samples,
            *batch_shape,
        )
        solution_steps = [current_state]
        drift_steps = []
        diffusion_steps = []

        # References: Gaines and Lyons (1997) Brownian tree; Kloeden and
        # Platen (1992) ch. 10.
        # Mathematical expression: supplied Delta W_n replaces the solver draw
        # sqrt(dt) xi_n in X_(n+1) = X_n + a dt + b Delta W_n.
        # Parameter meanings: one increment per sample path and step.
        # Units: sqrt(time unit of dt).
        # Domain/constraints: exact shape (num_samples, num_steps, *batch),
        # matching dtype/device, finite values.
        # Sign convention: not applicable.
        if wiener_increments is not None:
            if not isinstance(wiener_increments, Tensor):
                raise TypeError("wiener_increments must be a torch.Tensor")
            expected_shape = (num_samples, num_steps, *batch_shape)
            if tuple(wiener_increments.shape) != expected_shape:
                raise ValueError(
                    "wiener_increments must have shape "
                    f"{expected_shape}, got {tuple(wiener_increments.shape)}"
                )
            if wiener_increments.dtype != dtype or wiener_increments.device != device:
                raise ValueError(
                    "wiener_increments must match the input_state dtype and device"
                )
            if not bool(torch.isfinite(wiener_increments).all()):
                raise ValueError("wiener_increments must be finite")

        # References: Kloeden and Platen (1992); Higham (2001).
        # Mathematical expression: Delta W_n ~ N(0, dt I).
        # Parameter meanings: dt is fixed; generator owns the local RNG stream.
        for step in range(num_steps):
            current_time = time_grid[step]

            next_values, drift, diffusion = self._euler_maruyama_step(
                current_state,
                current_time,
                dt_tensor,
                generator=active_generator,
                wiener_increment=(
                    None if wiener_increments is None else wiener_increments[:, step]
                ),
                **kwargs,
            )
            solution_steps.append(next_values)
            drift_steps.append(drift)
            diffusion_steps.append(diffusion)
            current_state = next_values

        solution = torch.stack(solution_steps, dim=1)
        drift_history = torch.stack(drift_steps, dim=1)
        diffusion_history = torch.stack(diffusion_steps, dim=1)
        return {
            'solution': solution,
            'time': time_grid,
            'dt_used': dt_history,
            'drift_history': drift_history,
            'diffusion_history': diffusion_history,
            'seed': used_seed,
        }

    def _euler_maruyama_step(
        self,
        x: Tensor,
        t: Tensor,
        dt: Tensor,
        *,
        generator: torch.Generator,
        wiener_increment: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Single Euler-Maruyama integration step.
        
        References:
            Kloeden and Platen (1992), Chapter 10; Higham (2001).
        Mathematical expression:
            X_(n+1) = X_n + a(X_n,t_n) dt + b(X_n,t_n) odot Delta W_n,
            where Delta W_n ~ N(0, dt I).
        Parameter meanings:
            x is X_n, t is t_n, dt is the fixed increment, a is drift, b is
            diagonal diffusion, and generator owns the path-local substream.

        Returns:
            Tuple containing X_(n+1), a(X_n,t_n), and b(X_n,t_n).
        """
        # The leading axis is the sample-path batch. Each operator is evaluated
        # once per step to preserve a single Torch-vectorized execution path.
        drift = self.drift_function(x, t, **kwargs)
        diffusion = self.diffusion_function(x, t, **kwargs)
        
        # Generate Wiener increments: DeltaW ~ N(0, Deltat) unless the caller
        # supplied the path increment (Brownian tree refinement).
        if wiener_increment is not None:
            dW = wiener_increment
        else:
            dW = torch.randn(
                x.shape,
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            ) * torch.sqrt(dt)

        next_state = x + drift * dt + diffusion * dW
        if not torch.isfinite(next_state).all():
            raise ValueError(
                "Euler-Maruyama update produced a non-finite state"
            )
        return next_state, drift, diffusion

    def log_probability(
        self,
        input_state: Tensor,
        output_state: Tensor,
        **kwargs,
    ) -> Tensor:
        """Reject an unproved transition-density claim.

        A numerical path sampler does not by itself define a normalized
        transition-density evaluator for arbitrary declared operators.
        """
        del input_state, output_state, kwargs
        raise NotImplementedError(
            "SDE transition-density evaluation is [N] for this numeric-only "
            "owner."
        )

    def moments(
        self,
        input_state: Tensor,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Reject analytical moments without an operator-specific proof."""
        del input_state, kwargs
        raise NotImplementedError(
            "SDE analytical moments are [N] for arbitrary declared operators."
        )

    def describe(self) -> Dict[str, Any]:
        """Return metadata without claiming an unbound CF unit contract."""
        description = super().describe()
        description["cf_compliant"] = False
        description["unit_binding_status"] = "[N]"
        return description


class EulerMaruyamaSolver(SDESolver):
    """Euler-Maruyama integrator for explicit numerical operators only.

    References:
        Kloeden and Platen (1992); Higham (2001); revised foundations,
        continuous SDE contract.
    Mathematical expression:
        X_(n+1) = X_n + a(X_n,t_n) dt + b(X_n,t_n) odot Delta W_n,
        with Delta W_n distributed as N(0, dt I).
    Parameter meanings:
        a is the declared drift operator, b is the declared diagonal diffusion
        operator, dt is the fixed step, and W is a Wiener process.
    """
    
    def __init__(self, 
                 parameters: SDEParameters,
                 drift_func: Optional[Callable] = None,
                 diffusion_func: Optional[Callable] = None):
        """Initialize with explicit drift and diffusion callables."""
        if not callable(drift_func) or not callable(diffusion_func):
            raise ValueError(
                "EulerMaruyamaSolver requires explicit drift_func and "
                "diffusion_func operators."
            )
        super().__init__(parameters)
        self._drift_operator = drift_func
        self._diffusion_operator = diffusion_func

    def drift_function(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Evaluate the declared drift and enforce the numeric contract."""
        value = self._drift_operator(x, t, **kwargs)
        return self._validate_operator_output("drift", value, x)

    def diffusion_function(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        """Evaluate the declared diffusion and enforce the numeric contract."""
        value = self._diffusion_operator(x, t, **kwargs)
        return self._validate_operator_output("diffusion", value, x)

    @staticmethod
    def _validate_operator_output(
        operator_name: str,
        value: Tensor,
        reference: Tensor,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{operator_name} operator must return a torch.Tensor")
        if value.shape != reference.shape:
            raise ValueError(
                f"{operator_name} operator shape must match the state shape"
            )
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(
                f"{operator_name} operator device and dtype must match the state"
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"{operator_name} operator must return only finite values"
            )
        return value
