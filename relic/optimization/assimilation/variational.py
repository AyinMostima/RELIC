"""
Variational Data Assimilation Framework

This module provides the framework for variational data assimilation methods
including 3D-Var and 4D-Var. The implementation provides basic structures
and can be extended for full variational optimization.

Scientific Background:
    Variational methods formulate data assimilation as an optimization problem,
    minimizing a cost function that measures the distance from background
    and observations weighted by their respective error covariances.

References:
    - Talagrand, O. & Courtier, P. (1987). Variational assimilation of meteorological observations
    - Le Dimet, F.X. & Talagrand, O. (1986). Variational algorithms for analysis and assimilation
    - Courtier, P. et al. (1994). A strategy for operational implementation of 4D-Var
    - Rabier, F. et al. (2000). The ECMWF operational implementation of four-dimensional variational assimilation

Mathematical Expression:
    3D-Var cost function:
    J(x) = 1/2(x-x_b)^T B^{-1}(x-x_b) + 1/2(H(x)-y)^T R^{-1}(H(x)-y)
    
    4D-Var extends over time window [t0,T]:
    J(x0) = 1/2(x0-x_b)^T B^{-1}(x0-x_b) + 1/2sum_{t=t0}^T (H_t(x_t)-y_t)^T R_t^{-1}(H_t(x_t)-y_t)
    
    Where x_t = M_{t0?t}(x0) is the model trajectory from initial condition x0
"""

import math
from typing import Dict, Optional, Callable, Tuple
import torch
from torch import Tensor
import logging
import time
from dataclasses import dataclass

from .base import (
    AssimilationMethod,
    Assimilator,
    AssimilationConfig, 
    AssimilationResult,
    ObservationSet,
    ObservationOperator
)
from ...core.simulator import RELICSimulator
from ...core.state import StateManager
from ...parameters import UnifiedParameterManager, get_global_manager

logger = logging.getLogger(__name__)


@dataclass
class VariationalConfig(AssimilationConfig):
    """Configuration for variational data assimilation.

    The 3D-Var minimizer is steepest descent with the declared
    gradient_step_size s (x <- x - s grad J; s is halved for the step
    after an iteration whose cost increased). Line-search methods,
    preconditioning, observation-error scaling, a regularization weight
    and 4D-Var window settings are not implemented; their former fields
    were accepted and ignored and are no longer offered.
    method (required) is VARIATIONAL_3D or VARIATIONAL_4D. A setting left at
    None reads optimization.assimilation.variational.<name> from the
    parameter database.
    """
    # Optimization settings (None reads the YAML value)
    max_iterations: Optional[int] = None
    convergence_tolerance: Optional[float] = None
    gradient_step_size: Optional[float] = None

    # Cost function settings
    background_error_scale: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.method not in (AssimilationMethod.VARIATIONAL_3D, AssimilationMethod.VARIATIONAL_4D):
            raise ValueError(
                "VariationalConfig.method must be AssimilationMethod.VARIATIONAL_3D or "
                f"AssimilationMethod.VARIATIONAL_4D, got {self.method!r}"
            )
        manager = get_global_manager()
        if self.max_iterations is None:
            self.max_iterations = int(manager.get(
                "optimization.assimilation.variational.max_iterations", required=True
            ))
        for name in ("convergence_tolerance", "gradient_step_size", "background_error_scale"):
            if getattr(self, name) is None:
                setattr(self, name, float(manager.get(
                    f"optimization.assimilation.variational.{name}", required=True
                )))
        step = float(self.gradient_step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(
                f"gradient_step_size must be finite and positive, got "
                f"{self.gradient_step_size!r}"
            )


class ThreeDVar(Assimilator):
    """
    3D Variational Data Assimilation
    
    Implements the basic 3D-Var algorithm for single-time analysis.
    This is a foundation implementation that can be extended with
    more sophisticated minimization algorithms and preconditioning.
    """
    
    def __init__(
        self,
        simulator: RELICSimulator,
        state_manager: StateManager,
        parameter_manager: UnifiedParameterManager,
        config: Optional[VariationalConfig] = None
    ):
        """Initialize 3D-Var assimilator"""
        self.var_config = (
            config if config is not None
            else VariationalConfig(method=AssimilationMethod.VARIATIONAL_3D)
        )
        if self.var_config.method is not AssimilationMethod.VARIATIONAL_3D:
            raise ValueError(
                "ThreeDVar needs a VariationalConfig with method "
                f"AssimilationMethod.VARIATIONAL_3D, got {self.var_config.method!r}"
            )
        super().__init__(simulator, state_manager, parameter_manager, self.var_config)
        
        # Cost function tracking
        self.cost_history: list[float] = []
        self.gradient_norms: list[float] = []
        
        logger.info("3D-Var assimilator initialized")
    
    def assimilate(
        self,
        forecast_state: Dict[str, Tensor],
        observations: ObservationSet,
        observation_operator: ObservationOperator,
        background_covariance: Optional[Tensor] = None
    ) -> AssimilationResult:
        """
        Perform 3D-Var analysis.
        
        Parameters:
            forecast_state: Background/first-guess state
            observations: Observations to assimilate
            observation_operator: H operator
            background_covariance: Background error covariance matrix
            
        Returns:
            Analysis result
        """
        start_time = time.time()
        
        # Validate observations
        valid_observations = self.validate_observations(observations)
        if len(valid_observations) == 0:
            logger.warning("No valid observations for 3D-Var")
            return self._create_analysis_result(forecast_state, valid_observations, 0.0)
        
        try:
            template_state = self._create_template_state(forecast_state)

            # Convert state to optimization vector
            x_background = self._state_to_vector(template_state)
            
            # Set up cost function
            cost_function = self._create_cost_function(
                x_background,
                template_state,
                valid_observations,
                observation_operator,
                background_covariance,
            )
            
            # Minimize cost function
            x_analysis, final_cost, n_iterations = self._minimize_cost_function(
                cost_function, x_background
            )
            
            # Convert back to state dictionary
            analysis_state = self._vector_to_state(x_analysis, template_state)
            
            # Compute diagnostics
            analysis_dtype = next(iter(analysis_state.values())).dtype
            obs_vector = valid_observations.get_observation_vector(self.device).to(dtype=analysis_dtype)
            predicted_obs = observation_operator.forward(analysis_state)
            innovation = obs_vector - predicted_obs
            
            # Create result
            result = AssimilationResult(
                analysis_mean=analysis_state,
                forecast_mean=forecast_state,
                innovation=innovation,
                rms_innovation=torch.sqrt(torch.mean(innovation**2)).item(),
                assimilation_time=valid_observations.time,
                computational_cost=time.time() - start_time,
                observations_assimilated=len(valid_observations),
                observations_rejected=len(observations) - len(valid_observations)
            )
            
            # Add variational-specific metadata
            result.metadata.update({
                'method': '3D-Var',
                'final_cost': final_cost,
                'iterations': n_iterations,
                'convergence': n_iterations < self.var_config.max_iterations
            })
            
            self.cycle_count += 1
            self.total_observations += len(valid_observations)
            self.total_computational_cost += result.computational_cost
            
            logger.info(f"3D-Var completed: {n_iterations} iterations, final cost: {final_cost:.6f}")
            
            return result
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"3D-Var analysis failed: {e}")
            raise RuntimeError(f"3D-Var analysis failed: {e}") from e
    
    def _state_to_vector(self, state: Dict[str, Tensor]) -> Tensor:
        """Convert state dictionary to optimization vector"""
        vectors = []
        for var_name in sorted(state.keys()):
            vectors.append(state[var_name].flatten())
        return torch.cat(vectors)
    
    def _vector_to_state(self, vector: Tensor, template_state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Convert optimization vector back to state dictionary"""
        state = {}
        start_idx = 0
        
        for var_name in sorted(template_state.keys()):
            var_shape = template_state[var_name].shape
            var_size = template_state[var_name].numel()
            
            state[var_name] = vector[start_idx:start_idx + var_size].reshape(var_shape)
            start_idx += var_size
            
        return state
    
    def _create_cost_function(
        self,
        x_background: Tensor,
        template_state: Dict[str, Tensor],
        observations: ObservationSet,
        observation_operator: ObservationOperator,
        background_covariance: Optional[Tensor]
    ) -> Callable[[Tensor], Tuple[float, Tensor]]:
        """
        Create cost function and its gradient for optimization.
        
        Returns function that computes J(x) and nablaJ(x)
        """
        dtype = x_background.dtype
        obs_vector = observations.get_observation_vector(self.device).to(dtype=dtype)
        R_inv = torch.inverse(
            observations.get_error_covariance_matrix(self.device).to(dtype=dtype)
        )
        
        # Background error covariance (use identity if not provided)
        if background_covariance is not None:
            B_inv = torch.inverse(background_covariance.to(device=self.device, dtype=dtype))
        else:
            # Simple diagonal covariance
            B_inv = torch.eye(len(x_background), device=self.device, dtype=dtype)
            B_inv = B_inv / (self.var_config.background_error_scale**2)
        
        def cost_and_gradient(x: Tensor) -> Tuple[float, Tensor]:
            x = x.detach().clone().requires_grad_(True)
            
            # Background term: 0.5 * (x - x_b)^T B^{-1} (x - x_b)
            dx = x - x_background
            background_cost = 0.5 * torch.dot(dx, torch.mv(B_inv, dx))
            
            # Convert vector back to state for observation operator
            state = self._vector_to_state(x, template_state)
            
            # Observation term: 0.5 * (H(x) - y)^T R^{-1} (H(x) - y)
            predicted_obs = observation_operator.forward(state)
            innovation = predicted_obs - obs_vector
            observation_cost = 0.5 * torch.dot(innovation, torch.mv(R_inv, innovation))
            
            # Total cost
            total_cost = background_cost + observation_cost
            
            # Compute gradient
            total_cost.backward()
            gradient = x.grad.detach().clone()
            
            return total_cost.item(), gradient
        
        return cost_and_gradient
    
    def _create_template_state(self, state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Preserve the forecast state structure for vector/state conversion."""
        return {
            var_name: value.detach().clone()
            for var_name, value in state.items()
        }
    
    def _minimize_cost_function(
        self,
        cost_function: Callable[[Tensor], Tuple[float, Tensor]],
        x_initial: Tensor
    ) -> Tuple[Tensor, float, int]:
        """
        Minimize cost function using gradient descent.
        
        Returns: (optimal_x, final_cost, n_iterations)
        """
        x = x_initial.clone().requires_grad_(False)
        
        # Steepest descent with the declared step size (no line search)
        learning_rate = float(self.var_config.gradient_step_size)
        self.cost_history = []
        self.gradient_norms = []
        
        for iteration in range(self.var_config.max_iterations):
            # Compute cost and gradient
            cost, gradient = cost_function(x)
            
            self.cost_history.append(cost)
            grad_norm = torch.norm(gradient).item()
            self.gradient_norms.append(grad_norm)
            
            # Check convergence
            if grad_norm < self.var_config.convergence_tolerance:
                logger.info(f"3D-Var converged at iteration {iteration}")
                break
            
            # Simple line search
            step_size = learning_rate
            if iteration > 0 and self.cost_history[-1] > self.cost_history[-2]:
                step_size *= 0.5  # Reduce step size if cost increased
            
            # Update
            x = x - step_size * gradient
            
            if iteration % 10 == 0:
                logger.debug(f"3D-Var iteration {iteration}: cost={cost:.6f}, |grad|={grad_norm:.6f}")
        
        final_cost, _ = cost_function(x)
        return x, final_cost, len(self.cost_history)
    
    def _create_analysis_result(
        self,
        forecast_state: Dict[str, Tensor],
        observations: ObservationSet,
        computational_cost: float
    ) -> AssimilationResult:
        """Create result when analysis cannot be performed"""
        return AssimilationResult(
            analysis_mean=forecast_state,
            forecast_mean=forecast_state,
            # No observation was assimilated: the RMS innovation is undefined.
            rms_innovation=float("nan"),
            assimilation_time=observations.time,
            computational_cost=computational_cost,
            observations_assimilated=0,
            observations_rejected=len(observations)
        )


class FourDVar(Assimilator):
    """
    4D Variational Data Assimilation (Framework Shell)
    
    This provides the structure for 4D-Var implementation.
    The full implementation would require adjoint model development
    and is left for future enhancement.
    """
    
    def __init__(
        self,
        simulator: RELICSimulator,
        state_manager: StateManager,
        parameter_manager: UnifiedParameterManager,
        config: Optional[VariationalConfig] = None
    ):
        """Initialize 4D-Var framework"""
        self.var_config = (
            config if config is not None
            else VariationalConfig(method=AssimilationMethod.VARIATIONAL_4D)
        )
        if self.var_config.method is not AssimilationMethod.VARIATIONAL_4D:
            raise ValueError(
                "FourDVar needs a VariationalConfig with method "
                f"AssimilationMethod.VARIATIONAL_4D, got {self.var_config.method!r}"
            )
        super().__init__(simulator, state_manager, parameter_manager, self.var_config)
        
        logger.info("4D-Var framework initialized (skeleton implementation)")
    
    def assimilate(
        self,
        forecast_state: Dict[str, Tensor],
        observations: ObservationSet,
        observation_operator: ObservationOperator,
        background_covariance: Optional[Tensor] = None
    ) -> AssimilationResult:
        """
        4D-Var analysis (skeleton implementation).
        
        Full implementation would require:
        1. Adjoint model development
        2. Multiple observation time integration  
        3. Gradient computation over time window
        4. More sophisticated minimization
        """
        # A requested 4D-Var analysis is not replaced by a 3D-Var analysis.
        raise NotImplementedError(
            "4D-Var is not implemented (it needs the tangent-linear/adjoint "
            "model and the observation time window); use ThreeDVar explicitly "
            "for a single-time analysis"
        )
