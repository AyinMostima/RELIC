"""
Gradient-Based Parameter Calibrator

This module provides a specialized calibrator for gradient-based optimization
that integrates with UnifiedParameterManager's trainable parameter contract.
Optimization is performed on isolated candidate tensors and the accepted point
is committed once through ParameterManager's atomic batch mutation API.

Scientific Background:
    Gradient-based optimization is essential for differentiable biogeochemical
    models where parameters can be learned from observations via backpropagation.
    This enables data assimilation and parameter estimation using modern
    automatic differentiation frameworks.

References:
    - Kingma & Ba (2015). Adam: A Method for Stochastic Optimization
    - Liu & Nocedal (1989). On the Limited Memory BFGS Method
    - Goodfellow et al. (2016). Deep Learning, Chapter 8

Mathematical Expression:
    Parameter update: theta_{t+1} = theta_t - alpha grad_theta L(theta_t)
    where L is the loss function, alpha is learning rate, grad_theta is gradient
    
    Adam update:
        m_t = beta_1 m_{t-1} + (1-beta_1) g_t
        v_t = beta_2 v_{t-1} + (1-beta_2) g_t^2
        theta_t = theta_t - alpha * m_t / (sqrt(v_t) + eps)
"""

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..manager import (
    ParameterSpec,
    UnifiedParameterManager,
    get_global_manager,
)
from .calibrator import (
    CalibrationConfig,
    CalibrationMethod,
    CalibrationResult,
    ParameterBounds
)
from .results import CalibrationStatus
from .hooks import CalibrationHooks, CalibrationEvent

logger = logging.getLogger(__name__)


@dataclass
class GradientCalibrationConfig:
    """Configuration for gradient-based calibration.

    The optimizer method is a required choice. Every numeric setting that is
    not stated is read from the parameter database (optimization.calibration.*);
    the Adam settings are the Algorithm 1 defaults of Kingma & Ba (2015).
    """
    # Optimization method (required: no optimizer is chosen on the caller's behalf)
    method: CalibrationMethod

    # Hyperparameters
    learning_rate: float = field(default_factory=lambda: float(
        get_global_manager().get("optimization.calibration.gradient.learning_rate", required=True)))
    max_iterations: int = field(default_factory=lambda: int(
        get_global_manager().get("optimization.calibration.max_iterations", required=True)))
    tolerance: float = field(default_factory=lambda: float(
        get_global_manager().get("optimization.calibration.objective_tolerance", required=True)))

    # Adam-specific (only used if method=ADAM)
    beta1: float = field(default_factory=lambda: float(
        get_global_manager().get("optimization.calibration.gradient.adam_beta1", required=True)))
    beta2: float = field(default_factory=lambda: float(
        get_global_manager().get("optimization.calibration.gradient.adam_beta2", required=True)))
    epsilon: float = field(default_factory=lambda: float(
        get_global_manager().get("optimization.calibration.gradient.adam_epsilon", required=True)))
    weight_decay: float = field(default_factory=lambda: float(
        get_global_manager().get("optimization.calibration.gradient.weight_decay", required=True)))

    # L-BFGS-specific (only used if method=LBFGS)
    max_eval: Optional[int] = None
    history_size: int = field(default_factory=lambda: int(
        get_global_manager().get("optimization.calibration.gradient.lbfgs_history_size", required=True)))
    line_search_fn: Optional[str] = None

    # Convergence monitoring
    patience: int = field(default_factory=lambda: int(
        get_global_manager().get("optimization.calibration.early_stopping_patience", required=True)))
    min_delta: float = field(default_factory=lambda: float(
        get_global_manager().get("optimization.calibration.min_objective_improvement", required=True)))

    # Output
    verbose: bool = True
    save_progress: bool = True
    progress_file: Optional[Path] = None
    
    def to_calibration_config(self, 
                             target_parameters: List[str],
                             parameter_bounds: Dict[str, ParameterBounds]) -> CalibrationConfig:
        """
        Convert to base CalibrationConfig for backward consistency.
        
        Parameters
        ----------
        target_parameters : List[str]
            Parameter keys to calibrate
        parameter_bounds : Dict[str, ParameterBounds]
            Parameter bounds specification
            
        Returns
        -------
        CalibrationConfig
            consistent configuration for base calibrator
        """
        method_kwargs = {
            'learning_rate': self.learning_rate
        }
        
        if self.method == CalibrationMethod.ADAM:
            # Adam-specific kwargs would go here if needed
            pass
        elif self.method == CalibrationMethod.LBFGS:
            if self.max_eval is not None:
                method_kwargs['max_eval'] = self.max_eval
            method_kwargs['history_size'] = self.history_size
            if self.line_search_fn is not None:
                method_kwargs['line_search_fn'] = self.line_search_fn
        
        return CalibrationConfig(
            method=self.method,
            target_parameters=target_parameters,
            parameter_bounds=parameter_bounds,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            method_kwargs=method_kwargs,
            save_progress=self.save_progress,
            progress_file=self.progress_file,
            verbose=self.verbose
        )


class GradientCalibrator:
    """
    Gradient-based parameter calibrator with explicit ParameterManager integration.
    
    This class provides a high-level interface for gradient-based parameter
    calibration that automatically:
    1. Retrieves trainable parameters from UnifiedParameterManager
    2. Sets up appropriate torch.optim optimizers
    3. Keeps search candidates isolated from canonical manager state
    4. Commits the accepted parameter point in one atomic manager batch
    5. Tracks convergence and provides diagnostics
    
    Key Features:
    - Automatic trainable parameter discovery via manager.get_trainable_parameters()
    - Support for Adam and L-BFGS optimizers
    - Proper gradient flow through nn.Parameter objects
    - Finite scalar objective and per-target finite-gradient validation
    - Comprehensive convergence monitoring
    
    Units:
        All parameters follow their specified units in ParameterSpec
        Objective function values are dimensionless (loss/cost)
    
    Examples
    --------
    >>> from relic.parameters.calibration.gradient_calibrator import (
    ...     GradientCalibrator, GradientCalibrationConfig
    ... )
    >>> 
    >>> # The caller supplies the simulator/session-owned manager explicitly.
    >>> config = GradientCalibrationConfig(
    ...     method=CalibrationMethod.ADAM,
    ...     learning_rate=0.001,
    ...     max_iterations=100
    ... )
    >>> 
    >>> # Define objective
    >>> def objective(params: Dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    ...     # Run model and return loss (must be differentiable)
    ...     return loss_tensor
    >>> 
    >>> # Calibrate
    >>> calibrator = GradientCalibrator(manager, config)
    >>> result = calibrator.calibrate(objective, observations)
    >>> 
    >>> # Optimized parameters are automatically updated in manager
    >>> vcmax_optimized = manager.get('biosphere.photosynthesis.vcmax25')
    """
    
    def __init__(
        self,
        parameter_manager: Optional[UnifiedParameterManager] = None,
        config: Optional[GradientCalibrationConfig] = None,
        hooks: Optional[CalibrationHooks] = None
    ):
        """
        Initialize gradient calibrator.
        
        Parameters
        ----------
        parameter_manager : UnifiedParameterManager
            Explicit simulator/session-owned parameter manager instance
        config : GradientCalibrationConfig
            Calibration configuration (required: it declares the optimizer method)
        hooks : Optional[CalibrationHooks]
            Event hooks for monitoring
        """
        if parameter_manager is None:
            raise ValueError(
                "GradientCalibrator requires an explicit parameter_manager; "
                "implicit process-global calibration state is forbidden"
            )
        if config is None:
            raise ValueError(
                "GradientCalibrator requires an explicit GradientCalibrationConfig; "
                "the optimizer method has no default"
            )
        self.parameter_manager = parameter_manager
        self.config = config
        self.hooks = hooks or CalibrationHooks()
        
        # Calibration state
        self.trainable_params: Dict[str, nn.Parameter] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.convergence_history: List[float] = []
        self.parameter_history: List[Dict[str, Any]] = []
        
        logger.info(f"GradientCalibrator initialized with method: {self.config.method.value}")
    
    def calibrate(
        self,
        objective_function: Callable[[Dict[str, torch.Tensor], Any], torch.Tensor],
        observations: Any,
        target_parameters: Optional[List[str]] = None,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> CalibrationResult:
        """
        Perform gradient-based calibration.
        
        Parameters
        ----------
        objective_function : Callable
            Function that takes parameters and observations, returns loss Tensor.
            Must be differentiable (use torch operations).
            Signature: (params: Dict[str, torch.Tensor], obs: Any) -> torch.Tensor
            Note: params values are nn.Parameter objects, maintaining gradient flow
        observations : Any
            Observational data for comparison
        target_parameters : Optional[List[str]]
            Parameter keys to calibrate. If None, uses all trainable parameters.
        parameter_bounds : Optional[Dict[str, Tuple[float, float]]]
            Parameter bounds {key: (min, max)}. If None, uses spec bounds.
            
        Returns
        -------
        CalibrationResult
            Calibration results including optimized parameters
            
        Raises
        ------
        ValueError
            If no trainable parameters found
        RuntimeError
            If objective function is not differentiable
            
        Notes
        -----
        The objective function MUST:
        1. Accept Dict[str, torch.Tensor] parameters and observations
        2. Return a differentiable torch.Tensor (scalar loss)
        3. Use only torch operations to maintain gradient flow
        4. Parameters are nn.Parameter objects - use them directly, don't convert
        
        Example objective:
        ```python
        def objective(params: Dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
            # params contains nn.Parameter tensors with gradient tracking
            # Extract parameter values (they're already tensors)
            vcmax = params['biosphere.photosynthesis.vcmax25']
            
            # Run model with parameters
            model_output = run_model(vcmax)
            
            # Calculate loss (maintains gradient flow)
            loss = torch.mean((model_output - obs) ** 2)
            return loss
        ```
        """
        start_time = time.time()
        
        # Trigger start event
        self.hooks.trigger_event(
            event=CalibrationEvent.CALIBRATION_START,
            metadata={
                'method': self.config.method.value,
                'parameters': target_parameters,
                'iterations': self.config.max_iterations,
            },
            iteration=0,
        )
        
        initial_manager_values: Dict[str, Any] = {}
        previous_convergence_history = list(self.convergence_history)
        previous_parameter_history = deepcopy(self.parameter_history)

        try:
            # Resolve canonical targets before creating isolated search candidates.
            self._setup_trainable_parameters(target_parameters, parameter_bounds)

            if not self.trainable_params:
                raise ValueError("No trainable parameters found for calibration")

            self._validate_optimizer_contract()
            logger.info(f"Calibrating {len(self.trainable_params)} parameters: {list(self.trainable_params.keys())}")

            initial_manager_values = self._capture_manager_values()
            initial_params = self._get_current_parameter_values()
            self.trainable_params = self._create_optimization_parameters()
            self._setup_optimizer()

            # Histories contain accepted optimization points only. The initial
            # point is stored separately in CalibrationResult.initial_parameters.
            self.parameter_history = []
            self.convergence_history = []

            initial_loss = self._evaluate_objective(
                objective_function,
                observations,
            )
            best_loss = float(initial_loss.detach().item())
            best_params = initial_params.copy()
            patience_counter = 0
            termination_reason = "Max iterations reached"
            stop_status = CalibrationStatus.CONVERGENCE_FAILURE

            for iteration in range(self.config.max_iterations):
                if self.optimizer is None:
                    raise RuntimeError("Gradient optimizer was not initialized")

                if self.config.method == CalibrationMethod.LBFGS:
                    def closure() -> torch.Tensor:
                        self.optimizer.zero_grad()
                        loss_val = self._evaluate_objective(
                            objective_function,
                            observations,
                        )
                        loss_val.backward()
                        self._validate_target_gradients()
                        return loss_val

                    self.optimizer.step(closure)
                else:
                    self.optimizer.zero_grad()
                    loss = self._evaluate_objective(
                        objective_function,
                        observations,
                    )
                    loss.backward()
                    self._validate_target_gradients()
                    self.optimizer.step()

                    # Projected updates are supported only for first-order
                    # optimizers. PyTorch L-BFGS is explicitly unbounded here.
                    self._apply_bounds()

                accepted_loss = self._evaluate_objective(
                    objective_function,
                    observations,
                )
                loss_value = float(accepted_loss.detach().item())
                accepted_params = self._get_current_parameter_values()
                self.convergence_history.append(loss_value)
                self.parameter_history.append(accepted_params.copy())

                if loss_value < best_loss:
                    best_loss = loss_value
                    best_params = accepted_params.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                self.hooks.trigger_event(
                    event=CalibrationEvent.ITERATION_COMPLETE,
                    metadata={
                        'method': self.config.method.value,
                    },
                    iteration=int(iteration),
                    parameters=accepted_params,
                    objective_value=float(loss_value),
                )

                if self.config.verbose and (iteration % 10 == 0 or iteration == self.config.max_iterations - 1):
                    logger.info(f"Iteration {iteration}/{self.config.max_iterations}: Loss = {loss_value:.6e}")

                if loss_value <= self.config.tolerance:
                    termination_reason = "Objective target met"
                    break

                if len(self.convergence_history) > 1:
                    improvement = abs(self.convergence_history[-2] - self.convergence_history[-1])
                    if improvement < self.config.min_delta:
                        termination_reason = "Objective change below min_delta"
                        stop_status = CalibrationStatus.TERMINATED_EARLY
                        if self.config.verbose:
                            logger.info(f"Converged at iteration {iteration} (improvement < {self.config.min_delta})")
                        break

                if patience_counter >= self.config.patience:
                    termination_reason = "Patience exhausted"
                    stop_status = CalibrationStatus.TERMINATED_EARLY
                    if self.config.verbose:
                        logger.info(f"Early stopping at iteration {iteration} (patience={self.config.patience})")
                    break

            self._update_parameter_manager(best_params)

            # A best loss within the objective target is a success whatever stopped
            # the loop; otherwise the recorded stop decides the status.
            success = best_loss <= self.config.tolerance
            result = CalibrationResult(
                method=self.config.method.value,
                target_parameters=list(self.trainable_params.keys()),
                initial_parameters=initial_params,
                optimized_parameters=best_params,
                parameter_bounds=self._get_parameter_bounds(),
                objective_value=best_loss,
                convergence_history=self.convergence_history,
                iteration_count=len(self.convergence_history),
                elapsed_time=time.time() - start_time,
                success=success,
                status=CalibrationStatus.SUCCESS if success else stop_status,
                termination_reason=termination_reason,
                metrics=None  # Would be calculated by calling code if needed
            )
            
            # Trigger completion event
            self.hooks.trigger_event(
                event=CalibrationEvent.CALIBRATION_COMPLETE,
                metadata={
                    'success': result.success,
                    'objective_value': best_loss,
                    'iterations': result.iteration_count,
                    'method': self.config.method.value,
                },
                iteration=int(result.iteration_count),
                parameters=best_params,
                objective_value=float(best_loss),
            )
            
            logger.info(f"Calibration completed in {result.elapsed_time:.2f}s, final loss: {best_loss:.6e}")
            return result
            
        except Exception as e:
            if initial_manager_values:
                try:
                    self.parameter_manager.set_batch(
                        initial_manager_values,
                        validate=False,
                        notify=False,
                    )
                except Exception as rollback_error:
                    raise RuntimeError(
                        "Calibration failed and ParameterManager rollback also failed: "
                        f"{e}"
                    ) from rollback_error
            self.convergence_history = previous_convergence_history
            self.parameter_history = previous_parameter_history
            # Trigger error event
            self.hooks.trigger_event(
                event=CalibrationEvent.CALIBRATION_ERROR,
                metadata={
                    'error': str(e),
                    'method': self.config.method.value,
                },
            )
            logger.error(f"Calibration failed: {e}")
            raise
    
    def _setup_trainable_parameters(
        self,
        target_parameters: Optional[List[str]],
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]]
    ) -> None:
        """Setup trainable parameters from manager"""
        # Get all trainable parameters from manager
        all_trainable = self.parameter_manager.get_trainable_parameters()
        
        if target_parameters is not None:
            # Filter to requested parameters
            self.trainable_params = {
                key: param for key, param in all_trainable.items()
                if key in target_parameters
            }
            
            # Check all requested parameters are trainable
            missing = set(target_parameters) - set(self.trainable_params.keys())
            if missing:
                missing_keys = ", ".join(sorted(missing))
                raise ValueError(
                    f"Requested parameters not trainable or not registered: {missing_keys}"
                )
        else:
            # Use all trainable parameters
            self.trainable_params = all_trainable

        self._validate_requested_bounds(parameter_bounds)
        
        logger.debug(f"Setup {len(self.trainable_params)} trainable parameters")
    
    def _setup_optimizer(self) -> None:
        """Setup PyTorch optimizer"""
        if self.config.method == CalibrationMethod.ADAM:
            self.optimizer = torch.optim.Adam(
                self.trainable_params.values(),
                lr=self.config.learning_rate,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay
            )
            logger.debug("Setup Adam optimizer")
            
        elif self.config.method == CalibrationMethod.LBFGS:
            self.optimizer = torch.optim.LBFGS(
                self.trainable_params.values(),
                lr=self.config.learning_rate,
                max_iter=1,
                max_eval=self.config.max_eval,
                history_size=self.config.history_size,
                line_search_fn=self.config.line_search_fn
            )
            logger.debug("Setup L-BFGS optimizer")
            
        elif self.config.method == CalibrationMethod.GRADIENT_DESCENT:
            self.optimizer = torch.optim.SGD(
                self.trainable_params.values(),
                lr=self.config.learning_rate
            )
            logger.debug("Setup SGD optimizer")
            
        else:
            raise ValueError(f"Unsupported gradient method: {self.config.method}")

    def _validate_optimizer_contract(self) -> None:
        """Reject optimizer contracts that PyTorch cannot implement faithfully."""
        if self.config.method != CalibrationMethod.LBFGS:
            return

        bounded = []
        for key in self.trainable_params:
            spec = self._get_spec_for_key(key)
            if spec is None or spec.bounds is None:
                continue
            bounds = torch.as_tensor(spec.bounds, dtype=torch.float64)
            if bool(torch.isfinite(bounds).any()):
                bounded.append(key)
        if bounded:
            raise ValueError(
                "L-BFGS does not support bounded parameters: "
                + ", ".join(sorted(bounded))
            )

    def _create_optimization_parameters(self) -> Dict[str, nn.Parameter]:
        """Create isolated leaf tensors for the optimization transaction."""
        return {
            key: nn.Parameter(parameter.detach().clone(), requires_grad=True)
            for key, parameter in self.trainable_params.items()
        }

    def _evaluate_objective(
        self,
        objective_function: Callable[[Dict[str, torch.Tensor], Any], torch.Tensor],
        observations: Any,
    ) -> torch.Tensor:
        """Evaluate the differentiable finite-scalar objective contract."""
        loss = objective_function(self._get_parameter_tensors(), observations)
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError(
                f"Objective function must return torch.Tensor, got {type(loss)}"
            )
        if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).all()):
            raise RuntimeError(
                "Objective function must return a finite scalar torch.Tensor"
            )
        loss = loss.reshape(())
        if not loss.requires_grad:
            raise RuntimeError(
                "Objective function returned non-differentiable tensor. "
                "Ensure all operations use torch and parameters have requires_grad=True"
            )
        return loss

    def _validate_target_gradients(self) -> None:
        """Require every requested target to have a finite autograd gradient."""
        disconnected = [
            key for key, parameter in self.trainable_params.items()
            if parameter.grad is None
        ]
        if disconnected:
            raise RuntimeError(
                "Objective graph is not connected to target parameters: "
                + ", ".join(sorted(disconnected))
            )

        non_finite = [
            key for key, parameter in self.trainable_params.items()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad.detach()).all())
        ]
        if non_finite:
            raise RuntimeError(
                "Objective produced a non-finite gradient for target parameters: "
                + ", ".join(sorted(non_finite))
            )
    
    def _get_current_parameter_values(self) -> Dict[str, Any]:
        """Get current parameter values as dict"""
        return {
            key: float(param.detach().item())
            if param.numel() == 1
            else param.detach().cpu().tolist()
            for key, param in self.trainable_params.items()
        }

    def _capture_manager_values(self) -> Dict[str, Any]:
        """Capture canonical values for rollback without replacing parameters."""
        captured: Dict[str, Any] = {}
        for key in self.trainable_params:
            value = self.parameter_manager.get(key, required=True)
            if isinstance(value, torch.Tensor):
                captured[key] = value.detach().clone()
            else:
                captured[key] = deepcopy(value)
        return captured
    
    def _get_parameter_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Get current parameter tensors that maintain gradient connection.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping parameter keys to nn.Parameter tensors
        """
        return {key: param for key, param in self.trainable_params.items()}

    def _get_spec_for_key(self, key: str) -> Optional[ParameterSpec]:
        """Resolve metadata through the canonical ParameterManager API."""
        return self.parameter_manager.get_parameter(key)

    def _validate_requested_bounds(
        self,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]],
    ) -> None:
        """Treat call-site bounds only as assertions of ParameterManager SSOT."""
        if parameter_bounds is None:
            return

        unknown = set(parameter_bounds) - set(self.trainable_params)
        if unknown:
            raise ValueError(
                "Bounds were supplied for parameters outside the calibration target: "
                + ", ".join(sorted(unknown))
            )

        for key, declared_bounds in parameter_bounds.items():
            if len(declared_bounds) != 2:
                raise ValueError(f"Bounds for {key} must contain exactly two values")
            spec = self._get_spec_for_key(key)
            if spec is None or spec.bounds is None:
                raise ValueError(
                    f"ParameterManager bounds are required before calibrating {key}"
                )
            requested = torch.as_tensor(declared_bounds, dtype=torch.float64)
            canonical = torch.as_tensor(spec.bounds, dtype=torch.float64)
            if (
                requested.shape != torch.Size([2])
                or not bool(torch.isfinite(requested).all())
                or not torch.equal(requested, canonical)
            ):
                raise ValueError(
                    f"Call-site bounds for {key} must exactly match ParameterManager bounds"
                )
    
    def _get_parameter_bounds(self) -> Dict[str, ParameterBounds]:
        """Get parameter bounds from specs"""
        bounds_dict = {}
        for key in self.trainable_params.keys():
            spec = self._get_spec_for_key(key)
            if spec and spec.bounds:
                lower = float(spec.bounds[0])
                upper = float(spec.bounds[1])
                if upper < lower:
                    lower, upper = upper, lower
                raw_initial = float(spec.value) if not hasattr(spec.value, 'item') else float(spec.value.item())
                # Guard against numeric jitter at bounds (e.g., float32 0.699999988 vs 0.7).
                tol = 1.0e-6 * max(1.0, abs(lower), abs(upper))
                if raw_initial < lower and abs(raw_initial - lower) <= tol:
                    raw_initial = lower
                elif raw_initial > upper and abs(raw_initial - upper) <= tol:
                    raw_initial = upper
                elif raw_initial < lower or raw_initial > upper:
                    logger.warning(
                        "Clipping out-of-bound initial value for %s: %.12g not in [%.12g, %.12g]",
                        key,
                        raw_initial,
                        lower,
                        upper,
                    )
                    raw_initial = min(max(raw_initial, lower), upper)
                bounds_dict[key] = ParameterBounds(
                    lower=lower,
                    upper=upper,
                    # Gradient optimizers update theta itself (no search-space transform).
                    scale='linear',
                    initial=raw_initial,
                )
        return bounds_dict
    
    def _apply_bounds(self) -> None:
        """Apply parameter bounds after optimizer step"""
        for key, param in self.trainable_params.items():
            spec = self._get_spec_for_key(key)
            if spec and spec.bounds:
                with torch.no_grad():
                    param.clamp_(spec.bounds[0], spec.bounds[1])
    
    def _update_parameter_manager(self, best_params: Dict[str, float]) -> None:
        """Commit optimized values through the canonical atomic mutation API."""
        updates: Dict[str, Any] = {}
        for key, value in best_params.items():
            if key not in self.trainable_params:
                raise KeyError(f"Optimized parameter is not owned by this calibration: {key}")
            parameter = self.trainable_params[key]
            candidate = torch.as_tensor(
                value,
                dtype=parameter.dtype,
                device=parameter.device,
            )
            if candidate.numel() != parameter.numel():
                raise ValueError(
                    f"Optimized value shape for {key} does not match {tuple(parameter.shape)}"
                )
            candidate = candidate.reshape(parameter.shape)
            updates[key] = (
                float(candidate.detach().item())
                if candidate.numel() == 1
                else candidate.detach().clone()
            )

        self.parameter_manager.set_batch(updates, validate=True, notify=True)
        logger.debug(f"Updated {len(best_params)} parameters in manager")
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """
        Get convergence analysis.
        
        Returns
        -------
        Dict[str, Any]
            Convergence metrics including rate, plateaus, final improvement
        """
        if not self.convergence_history:
            return {"status": "no_data"}
        
        history = self.convergence_history
        
        return {
            "total_iterations": len(history),
            "initial_loss": history[0],
            "final_loss": history[-1],
            "best_loss": min(history),
            "improvement": history[0] - history[-1],
            "relative_improvement": (history[0] - history[-1]) / history[0] if history[0] != 0 else 0,
            "converged": history[-1] < self.config.tolerance
        }

