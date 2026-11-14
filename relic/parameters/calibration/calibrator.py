"""
Parameter Calibration Engine

This module owns derivative-free parameter calibration for the RELIC parameter
management system. Gradient descent, Adam, and L-BFGS are owned exclusively by
``GradientCalibrator`` and are rejected by ``ParameterCalibrator``.

Scientific Background:
    Parameter calibration is essential for biogeochemical models to ensure 
    accurate representation of observed ecosystem processes. The calibration 
    process involves optimizing model parameters to minimize the difference 
    between model predictions and observations.

References:
    - Luo et al. (2009). Parameter identifiability, constraint, and 
      equifinality in data assimilation with ecosystem models.
    - Trudinger et al. (2007). OptIC project: An intercomparison of 
      optimization techniques for parameter estimation in terrestrial 
      biogeochemical models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, Callable, Any, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import torch
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import time
from scipy.optimize import differential_evolution

from ..manager import get_global_manager

if TYPE_CHECKING:
    from ..manager import UnifiedParameterManager
from .results import CalibrationResult, CalibrationMetrics, CalibrationStatus
from .hooks import CalibrationHooks, CalibrationEvent

# Search-space transforms of ParameterBounds.scale: 'linear' samples theta itself,
# 'log' samples ln(theta) (requires 0 < lower).
PARAMETER_BOUND_SCALES = ("linear", "log")


class CalibrationMethod(Enum):
    """Available calibration methods"""
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    LBFGS = "lbfgs"


class ObjectiveType(Enum):
    """Available objective function types"""
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    NASH_SUTCLIFFE = "nash_sutcliffe"
    R_SQUARED = "r_squared"
    LOG_LIKELIHOOD = "log_likelihood"
    BAYESIAN_INFORMATION_CRITERION = "bic"
    AKAIKE_INFORMATION_CRITERION = "aic"


@dataclass
class ParameterBounds:
    """Parameter bounds for calibration"""
    lower: float
    upper: float
    scale: str  # search-space transform, one of PARAMETER_BOUND_SCALES; no default
    initial: Optional[float] = None

    def __post_init__(self):
        if self.scale not in PARAMETER_BOUND_SCALES:
            raise ValueError(
                f"ParameterBounds scale must be one of {PARAMETER_BOUND_SCALES}, got {self.scale!r}"
            )
        if self.lower >= self.upper:
            raise ValueError(f"Lower bound {self.lower} must be < upper bound {self.upper}")
        if self.initial is not None:
            if not (self.lower <= self.initial <= self.upper):
                raise ValueError(f"Initial value {self.initial} out of bounds [{self.lower}, {self.upper}]")


@dataclass
class CalibrationConfig:
    """Configuration for parameter calibration"""
    method: CalibrationMethod
    target_parameters: List[str]
    parameter_bounds: Dict[str, ParameterBounds]
    # Iteration cap and objective target: read from the parameter database when
    # not stated (optimization.calibration.max_iterations / .objective_tolerance).
    max_iterations: int = field(
        default_factory=lambda: int(get_global_manager().get("optimization.calibration.max_iterations", required=True))
    )
    tolerance: float = field(
        default_factory=lambda: float(
            get_global_manager().get("optimization.calibration.objective_tolerance", required=True)
        )
    )
    population_size: Optional[int] = None
    random_seed: Optional[int] = None
    parallel: bool = False
    n_workers: Optional[int] = None

    # Method-specific parameters
    method_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Validation settings
    cross_validation_folds: Optional[int] = None

    # Output settings
    save_progress: bool = True
    progress_file: Optional[Path] = None
    verbose: bool = True

    @classmethod
    def create_simple(cls,
                     method: Union[str, CalibrationMethod],
                     max_iterations: Optional[int] = None,
                     population_size: Optional[int] = None,
                     tolerance: Optional[float] = None,
                     **kwargs) -> 'CalibrationConfig':
        """
        Create a simplified CalibrationConfig for testing and quick setup.
        
        This factory method creates a basic configuration with minimal required parameters,
        suitable for testing scenarios where the full parameter specification is handled
        separately.
        
        Parameters:
            method: Optimization method name or enum (required; no method is
                chosen on the caller's behalf)
            max_iterations: Maximum optimization iterations; None reads
                optimization.calibration.max_iterations
            population_size: Population size for population-based methods
            tolerance: Objective target; None reads
                optimization.calibration.objective_tolerance
            **kwargs: Additional method-specific arguments

        Returns:
            CalibrationConfig instance with default/empty parameter specifications
        """
        if isinstance(method, str):
            method = CalibrationMethod(method)

        # Create minimal parameter specifications for testing
        target_parameters = ['test_param']
        parameter_bounds = {
            'test_param': ParameterBounds(lower=0.0, upper=1.0, scale='linear')
        }
        stated = {}
        if max_iterations is not None:
            stated['max_iterations'] = max_iterations
        if tolerance is not None:
            stated['tolerance'] = tolerance

        return cls(
            method=method,
            target_parameters=target_parameters,
            parameter_bounds=parameter_bounds,
            population_size=population_size,
            **stated,
            **kwargs
        )
    
    def __post_init__(self):
        # Validate parameter bounds
        for param_name in self.target_parameters:
            if param_name not in self.parameter_bounds:
                raise ValueError(f"Bounds required for parameter: {param_name}")
        
        # Set default population size based on method
        if self.population_size is None:
            if self.method in [CalibrationMethod.DIFFERENTIAL_EVOLUTION, 
                             CalibrationMethod.GENETIC_ALGORITHM]:
                self.population_size = max(50, len(self.target_parameters) * 10)
            elif self.method == CalibrationMethod.PARTICLE_SWARM:
                self.population_size = max(20, len(self.target_parameters) * 5)


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions"""
    
    @abstractmethod
    def __call__(self, parameters: Dict[str, float], observations: Any) -> float:
        """
        Evaluate objective function
        
        Parameters:
            parameters: Dictionary of parameter values
            observations: Observation data for comparison
            
        Returns:
            Objective function value (lower is better)
        """


class MSEObjective(ObjectiveFunction):
    """Mean Squared Error objective function"""
    
    def __init__(self, model_runner: Callable, variable_names: List[str]):
        self.model_runner = model_runner
        self.variable_names = variable_names
    
    def __call__(self, parameters: Dict[str, float], observations: torch.Tensor) -> float:
        """
        Calculate MSE between model output and observations
        
        Mathematical Expression:
            MSE = (1/n) * Sigma(model_i - obs_i)^2
            where n is the number of observations
        """
        # References:
        #   Moriasi et al. (2007), Trans. ASABE 50(3):885-900 (model evaluation statistics).
        # Mathematical expression:
        #   MSE = (1/n) * sum_i (m_i - o_i)^2
        # Parameter meanings:
        #   m_i: i-th element of the named model outputs, flattened and concatenated;
        #   o_i: the matching flattened observation; n: number of compared elements.
        # Units:
        #   m_i and o_i in the observation units u; MSE in u^2.
        # Domain/constraints:
        #   the flattened model and observation vectors must be broadcast-compatible;
        #   a model failure or a missing output variable raises (no inf penalty value).
        # Sign convention:
        #   MSE >= 0; a lower value is a better fit (minimized).
        model_output = self.model_runner(parameters)
        if isinstance(model_output, dict):
            model_values = torch.cat([
                model_output[var].flatten() for var in self.variable_names
            ])
        else:
            model_values = model_output.flatten()

        mse = torch.mean((model_values - observations.flatten()) ** 2)
        return float(mse)


class RMSEObjective(ObjectiveFunction):
    """Root Mean Squared Error objective function"""
    
    def __init__(self, model_runner: Callable, variable_names: List[str]):
        self.mse_objective = MSEObjective(model_runner, variable_names)
    
    def __call__(self, parameters: Dict[str, float], observations: torch.Tensor) -> float:
        """
        Calculate RMSE between model output and observations
        
        Mathematical Expression:
            RMSE = sqrt[(1/n) * Sigma(model_i - obs_i)^2]
        """
        mse = self.mse_objective(parameters, observations)
        return np.sqrt(mse)


class NashSutcliffeObjective(ObjectiveFunction):
    """Nash-Sutcliffe Efficiency objective function (minimization form)"""
    
    def __init__(self, model_runner: Callable, variable_names: List[str]):
        self.model_runner = model_runner
        self.variable_names = variable_names
    
    def __call__(self, parameters: Dict[str, float], observations: torch.Tensor) -> float:
        """
        Calculate 1 - Nash-Sutcliffe Efficiency (for minimization)
        
        Mathematical Expression:
            NSE = 1 - [Sigma(model_i - obs_i)^2 / Sigma(obs_i - obs_mean)^2]
            Objective = 1 - NSE (for minimization)
        """
        # References:
        #   Nash, J. E. and Sutcliffe, J. V. (1970), J. Hydrol. 10(3):282-290.
        # Mathematical expression:
        #   NSE = 1 - sum_i (m_i - o_i)^2 / sum_i (o_i - o_mean)^2; objective = 1 - NSE
        # Parameter meanings:
        #   m_i: i-th element of the named model outputs, flattened and concatenated;
        #   o_i: the matching flattened observation; o_mean: mean of the observations.
        # Units:
        #   m_i, o_i and o_mean in the observation units; NSE and the objective are dimensionless.
        # Domain/constraints:
        #   sum_i (o_i - o_mean)^2 > 0; observations without variance leave NSE undefined and
        #   raise ValueError; a model failure or a missing output variable raises.
        # Sign convention:
        #   NSE <= 1 (1 is a perfect fit); the minimized objective 1 - NSE >= 0.
        model_output = self.model_runner(parameters)
        if isinstance(model_output, dict):
            model_values = torch.cat([
                model_output[var].flatten() for var in self.variable_names
            ])
        else:
            model_values = model_output.flatten()

        obs_flat = observations.flatten()
        obs_mean = torch.mean(obs_flat)
        numerator = torch.sum((model_values - obs_flat) ** 2)
        denominator = torch.sum((obs_flat - obs_mean) ** 2)
        if denominator == 0:
            raise ValueError(
                "Nash-Sutcliffe efficiency is undefined for observations without variance"
            )

        nse = 1 - (numerator / denominator)
        return float(1 - nse)  # Return 1-NSE for minimization


class ParameterCalibrator:
    """
    Derivative-free parameter calibration engine for RELIC models.
    
    This class owns differential evolution, genetic algorithm, simulated
    annealing, particle swarm, and Bayesian optimization. Gradient methods use
    ``GradientCalibrator`` and are not executable through this class.
    """
    
    def __init__(self, 
                 parameter_manager: Optional[UnifiedParameterManager] = None,
                 hooks: Optional[CalibrationHooks] = None):
        """
        Initialize parameter calibrator
        
        Parameters:
            parameter_manager: Parameter manager instance
            hooks: Calibration hooks for event handling
        """
        self.parameter_manager = parameter_manager or get_global_manager()
        self.hooks = hooks or CalibrationHooks()
        self.logger = logging.getLogger(__name__)
        
        # Available optimization methods
        self._optimizers = {
            CalibrationMethod.DIFFERENTIAL_EVOLUTION: self._differential_evolution,
            CalibrationMethod.GENETIC_ALGORITHM: self._genetic_algorithm,
            CalibrationMethod.SIMULATED_ANNEALING: self._simulated_annealing,
            CalibrationMethod.PARTICLE_SWARM: self._particle_swarm,
            CalibrationMethod.BAYESIAN_OPTIMIZATION: self._bayesian_optimization,
        }
    
    def calibrate(self,
                  config: CalibrationConfig,
                  objective_function: ObjectiveFunction,
                  observations: torch.Tensor,
                  initial_parameters: Optional[Dict[str, float]] = None) -> CalibrationResult:
        """
        Calibrate model parameters using specified method
        
        Parameters:
            config: Calibration configuration
            objective_function: Function to minimize
            observations: Observational data for comparison
            initial_parameters: Initial parameter values
            
        Returns:
            Calibration results including optimized parameters
        """
        if config.method in (
            CalibrationMethod.GRADIENT_DESCENT,
            CalibrationMethod.ADAM,
            CalibrationMethod.LBFGS,
        ):
            raise ValueError(
                "Gradient-based calibration is owned by GradientCalibrator; "
                f"ParameterCalibrator cannot execute {config.method.value!r}"
            )

        self.logger.info(f"Starting calibration with method: {config.method.value}")
        
        # Trigger start event
        self.hooks.trigger_event(CalibrationEvent.CALIBRATION_START, {
            'method': config.method.value,
            'parameters': config.target_parameters,
            'iterations': config.max_iterations
        })
        
        start_time = time.time()
        
        try:
            # Set random seed for reproducibility
            if config.random_seed is not None:
                np.random.seed(config.random_seed)
                torch.manual_seed(config.random_seed)
            
            # Get initial parameter values
            initial_params = self._prepare_initial_parameters(
                config.target_parameters, 
                config.parameter_bounds,
                initial_parameters
            )
            
            # Run optimization
            optimizer_func = self._optimizers[config.method]
            result = optimizer_func(
                config=config,
                objective_function=objective_function,
                observations=observations,
                initial_parameters=initial_params
            )
            
            # Calculate final metrics
            final_objective = objective_function(result.optimized_parameters, observations)
            
            # Create calibration result
            calibration_result = CalibrationResult(
                method=config.method.value,
                target_parameters=config.target_parameters,
                initial_parameters=initial_params,
                optimized_parameters=result.optimized_parameters,
                parameter_bounds={name: bounds for name, bounds in config.parameter_bounds.items()},
                objective_value=final_objective,
                convergence_history=result.convergence_history,
                iteration_count=result.iteration_count,
                elapsed_time=time.time() - start_time,
                success=result.success,
                # A run that ends without meeting its objective target did not converge.
                status=(
                    CalibrationStatus.SUCCESS if result.success
                    else CalibrationStatus.CONVERGENCE_FAILURE
                ),
                termination_reason=result.termination_reason,
                metrics=self._calculate_metrics(
                    result.optimized_parameters, 
                    objective_function, 
                    observations
                )
            )
            
            # Trigger completion event
            self.hooks.trigger_event(CalibrationEvent.CALIBRATION_COMPLETE, {
                'success': result.success,
                'objective_value': final_objective,
                'iterations': result.iteration_count
            })
            
            self.logger.info(f"Calibration completed in {calibration_result.elapsed_time:.2f}s")
            return calibration_result
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            # Trigger error event
            self.hooks.trigger_event(CalibrationEvent.CALIBRATION_ERROR, {
                'error': str(e),
                'method': config.method.value
            })
            self.logger.error(f"Calibration failed: {e}")
            raise
    
    def _prepare_initial_parameters(self,
                                  target_parameters: List[str],
                                  parameter_bounds: Dict[str, ParameterBounds],
                                  initial_parameters: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Prepare initial parameter values for optimization"""
        initial_params = {}
        
        for param_name in target_parameters:
            bounds = parameter_bounds[param_name]
            
            if initial_parameters and param_name in initial_parameters:
                # Use provided initial value
                initial_params[param_name] = initial_parameters[param_name]
            elif bounds.initial is not None:
                # Use bounds-specified initial value
                initial_params[param_name] = bounds.initial
            else:
                # Use current parameter value from manager
                current_value = self.parameter_manager.get(param_name)
                if current_value is not None:
                    if isinstance(current_value, torch.Tensor):
                        initial_params[param_name] = float(current_value.item())
                    else:
                        initial_params[param_name] = float(current_value)
                else:
                    # Use middle of bounds
                    if bounds.scale == 'log':
                        initial_params[param_name] = np.exp(
                            (np.log(bounds.lower) + np.log(bounds.upper)) / 2
                        )
                    else:
                        initial_params[param_name] = (bounds.lower + bounds.upper) / 2
        
        return initial_params
    
    def _differential_evolution(self,
                              config: CalibrationConfig,
                              objective_function: ObjectiveFunction,
                              observations: torch.Tensor,
                              initial_parameters: Dict[str, float]) -> 'OptimizationResult':
        """
        Differential Evolution optimization algorithm
        
        References:
            Storn, R., & Price, K. (1997). Differential evolution-a simple and 
            efficient heuristic for global optimization over continuous spaces.
        """
        # Prepare bounds for scipy
        bounds = []
        param_names = []
        for param_name in config.target_parameters:
            param_bounds = config.parameter_bounds[param_name]
            if param_bounds.scale == 'log':
                bounds.append((np.log(param_bounds.lower), np.log(param_bounds.upper)))
            else:
                bounds.append((param_bounds.lower, param_bounds.upper))
            param_names.append(param_name)
        
        # Store convergence history
        convergence_history = []
        
        # Objective function wrapper
        def objective_wrapper(x):
            params = {}
            for i, param_name in enumerate(param_names):
                param_bounds = config.parameter_bounds[param_name]
                if param_bounds.scale == 'log':
                    params[param_name] = np.exp(x[i])
                else:
                    params[param_name] = x[i]
            obj_val = objective_function(params, observations)
            convergence_history.append(float(obj_val))
            return obj_val
        
        de_iteration = {"count": 0}

        def _de_callback(xk: np.ndarray, convergence: float) -> bool:
            params = {}
            for i, param_name in enumerate(param_names):
                param_bounds = config.parameter_bounds[param_name]
                raw_value = float(xk[i])
                if param_bounds.scale == "log":
                    params[param_name] = float(np.exp(raw_value))
                else:
                    params[param_name] = raw_value

            objective_value = float(objective_function(params, observations))

            self.hooks.trigger_event(
                event=CalibrationEvent.ITERATION_COMPLETE,
                metadata={
                    "method": config.method.value,
                    "convergence": float(convergence),
                },
                iteration=int(de_iteration["count"]),
                parameters=params,
                objective_value=objective_value,
            )
            de_iteration["count"] += 1
            return False

        method_kwargs = dict(config.method_kwargs or {})
        method_kwargs.setdefault("callback", _de_callback)

        # Resolve DE population multiplier from a user-facing absolute population size.
        n_params = max(1, len(config.target_parameters))
        if config.population_size is None:
            popsize_multiplier = 1
        else:
            popsize_multiplier = max(1, int(np.ceil(float(config.population_size) / float(n_params))))

        workers_arg = method_kwargs.get("workers", None)
        parallel_requested = bool(config.parallel) or workers_arg not in (None, 1)
        if parallel_requested:
            raise RuntimeError(
                "Parallel differential evolution requires an isolated objective "
                "factory and is not available for live calibration runtimes. "
                "Use the canonical ParallelExecutor parameter-sweep path for "
                "deterministic-pure independent evaluations."
            )

        # Run optimization
        result = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=config.max_iterations,
            popsize=popsize_multiplier,
            tol=config.tolerance,
            seed=config.random_seed,
            **method_kwargs
        )
        
        # Extract optimized parameters
        optimized_params = {}
        for i, param_name in enumerate(param_names):
            param_bounds = config.parameter_bounds[param_name]
            if param_bounds.scale == 'log':
                optimized_params[param_name] = np.exp(result.x[i])
            else:
                optimized_params[param_name] = result.x[i]
        
        return OptimizationResult(
            optimized_parameters=optimized_params,
            convergence_history=convergence_history,
            iteration_count=result.nit,
            success=result.success,
            termination_reason=result.message
        )
    
    def _genetic_algorithm(self,
                          config: CalibrationConfig,
                          objective_function: ObjectiveFunction,
                          observations: torch.Tensor,
                          initial_parameters: Dict[str, float]) -> 'OptimizationResult':
        """Simple genetic algorithm implementation"""
        
        population_size = config.population_size
        param_names = config.target_parameters
        n_params = len(param_names)
        
        # Initialize population
        population = np.random.random((population_size, n_params))
        for i, param_name in enumerate(param_names):
            bounds = config.parameter_bounds[param_name]
            if bounds.scale == 'log':
                population[:, i] = np.exp(
                    population[:, i] * (np.log(bounds.upper) - np.log(bounds.lower)) + 
                    np.log(bounds.lower)
                )
            else:
                population[:, i] = (
                    population[:, i] * (bounds.upper - bounds.lower) + bounds.lower
                )
        
        convergence_history = []
        best_params = None
        best_fitness = float('inf')
        
        for iteration in range(config.max_iterations):
            # Evaluate population
            fitness = np.zeros(population_size)
            for i in range(population_size):
                params = {param_names[j]: population[i, j] for j in range(n_params)}
                fitness[i] = objective_function(params, observations)
                
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_params = params.copy()
            
            convergence_history.append(best_fitness)
            
            # Check convergence
            if best_fitness < config.tolerance:
                break
            
            # Selection, crossover, and mutation (simplified)
            # Tournament selection
            new_population = np.zeros_like(population)
            for i in range(population_size):
                tournament_size = 3
                tournament_indices = np.random.choice(population_size, tournament_size)
                winner_idx = tournament_indices[np.argmin(fitness[tournament_indices])]
                new_population[i] = population[winner_idx]
            
            # Crossover and mutation
            for i in range(0, population_size, 2):
                if i + 1 < population_size and np.random.random() < 0.8 and n_params > 1:  # Crossover probability
                    # Single-point crossover
                    crossover_point = np.random.randint(1, n_params)
                    new_population[i, crossover_point:], new_population[i+1, crossover_point:] = \
                        new_population[i+1, crossover_point:].copy(), new_population[i, crossover_point:].copy()
                
                # Mutation
                for j in range(n_params):
                    if np.random.random() < 0.1:  # Mutation probability
                        bounds = config.parameter_bounds[param_names[j]]
                        if bounds.scale == 'log':
                            new_population[i, j] = np.exp(
                                np.random.uniform(np.log(bounds.lower), np.log(bounds.upper))
                            )
                        else:
                            new_population[i, j] = np.random.uniform(bounds.lower, bounds.upper)
            
            population = new_population
        
        return OptimizationResult(
            optimized_parameters=best_params,
            convergence_history=convergence_history,
            iteration_count=len(convergence_history),
            success=best_fitness < config.tolerance,
            termination_reason="Converged" if best_fitness < config.tolerance else "Max iterations reached"
        )
    
    def _simulated_annealing(self,
                           config: CalibrationConfig,
                           objective_function: ObjectiveFunction,
                           observations: torch.Tensor,
                           initial_parameters: Dict[str, float]) -> 'OptimizationResult':
        """Simulated annealing optimization"""
        
        current_params = initial_parameters.copy()
        current_objective = objective_function(current_params, observations)
        
        best_params = current_params.copy()
        best_objective = current_objective
        
        convergence_history = [current_objective]
        temperature = config.method_kwargs.get('initial_temperature', 1000.0)
        cooling_rate = config.method_kwargs.get('cooling_rate', 0.95)
        
        param_names = list(current_params.keys())
        
        for iteration in range(config.max_iterations):
            # Generate neighbor solution
            new_params = current_params.copy()
            param_to_modify = np.random.choice(param_names)
            bounds = config.parameter_bounds[param_to_modify]
            
            # Add random perturbation
            if bounds.scale == 'log':
                log_value = np.log(current_params[param_to_modify])
                log_bounds = (np.log(bounds.lower), np.log(bounds.upper))
                step_size = (log_bounds[1] - log_bounds[0]) * 0.1
                new_log_value = log_value + np.random.normal(0, step_size)
                new_log_value = np.clip(new_log_value, log_bounds[0], log_bounds[1])
                new_params[param_to_modify] = np.exp(new_log_value)
            else:
                step_size = (bounds.upper - bounds.lower) * 0.1
                new_value = current_params[param_to_modify] + np.random.normal(0, step_size)
                new_params[param_to_modify] = np.clip(new_value, bounds.lower, bounds.upper)
            
            # Evaluate new solution
            new_objective = objective_function(new_params, observations)
            
            # Accept or reject based on Metropolis criterion
            delta = new_objective - current_objective
            if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                current_params = new_params
                current_objective = new_objective
                
                if new_objective < best_objective:
                    best_params = new_params.copy()
                    best_objective = new_objective
            
            convergence_history.append(best_objective)
            
            # Cool down
            temperature *= cooling_rate
            
            # Check convergence
            if best_objective < config.tolerance:
                break
        
        return OptimizationResult(
            optimized_parameters=best_params,
            convergence_history=convergence_history,
            iteration_count=len(convergence_history),
            success=best_objective < config.tolerance,
            termination_reason="Converged" if best_objective < config.tolerance else "Max iterations reached"
        )
    
    def _particle_swarm(self,
                       config: CalibrationConfig,
                       objective_function: ObjectiveFunction,
                       observations: torch.Tensor,
                       initial_parameters: Dict[str, float]) -> 'OptimizationResult':
        """Particle Swarm Optimization"""
        
        # PSO parameters
        w = config.method_kwargs.get('inertia_weight', 0.7)
        c1 = config.method_kwargs.get('cognitive_weight', 1.5)
        c2 = config.method_kwargs.get('social_weight', 1.5)
        
        population_size = config.population_size
        param_names = config.target_parameters
        n_params = len(param_names)
        
        # Initialize particles
        particles = np.random.random((population_size, n_params))
        velocities = np.zeros((population_size, n_params))
        personal_best = particles.copy()
        personal_best_fitness = np.full(population_size, float('inf'))
        
        # Scale to parameter bounds
        for i, param_name in enumerate(param_names):
            bounds = config.parameter_bounds[param_name]
            if bounds.scale == 'log':
                particles[:, i] = np.exp(
                    particles[:, i] * (np.log(bounds.upper) - np.log(bounds.lower)) + 
                    np.log(bounds.lower)
                )
                personal_best[:, i] = particles[:, i]
            else:
                particles[:, i] = (
                    particles[:, i] * (bounds.upper - bounds.lower) + bounds.lower
                )
                personal_best[:, i] = particles[:, i]
        
        global_best = None
        global_best_fitness = float('inf')
        convergence_history = []
        
        for iteration in range(config.max_iterations):
            # Evaluate particles
            for i in range(population_size):
                params = {param_names[j]: particles[i, j] for j in range(n_params)}
                fitness = objective_function(params, observations)
                
                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best[i] = particles[i].copy()
                
                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()
            
            convergence_history.append(global_best_fitness)
            
            # Check convergence
            if global_best_fitness < config.tolerance:
                break
            
            # Update velocities and positions
            for i in range(population_size):
                r1, r2 = np.random.random(n_params), np.random.random(n_params)
                
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_best[i] - particles[i]) +
                               c2 * r2 * (global_best - particles[i]))
                
                particles[i] += velocities[i]
                
                # Apply bounds
                for j, param_name in enumerate(param_names):
                    bounds = config.parameter_bounds[param_name]
                    particles[i, j] = np.clip(particles[i, j], bounds.lower, bounds.upper)
        
        # Convert best solution to parameters
        best_params = {param_names[j]: global_best[j] for j in range(n_params)}
        
        return OptimizationResult(
            optimized_parameters=best_params,
            convergence_history=convergence_history,
            iteration_count=len(convergence_history),
            success=global_best_fitness < config.tolerance,
            termination_reason="Converged" if global_best_fitness < config.tolerance else "Max iterations reached"
        )
    
    def _bayesian_optimization(self,
                             config: CalibrationConfig,
                             objective_function: ObjectiveFunction,
                             observations: torch.Tensor,
                             initial_parameters: Dict[str, float]) -> 'OptimizationResult':
        """Bayesian optimization (simplified implementation)"""
        
        # This is a simplified implementation
        # For production use, consider using libraries like skopt or GPyOpt
        
        # Use initial parameters as starting point
        best_params = initial_parameters.copy()
        best_objective = objective_function(best_params, observations)
        convergence_history = [best_objective]
        
        param_names = config.target_parameters
        n_evaluations = config.method_kwargs.get('n_evaluations', 50)
        
        for iteration in range(min(n_evaluations, config.max_iterations)):
            # Simple random search (replace with GP-based acquisition in production)
            candidate_params = {}
            for param_name in param_names:
                bounds = config.parameter_bounds[param_name]
                if bounds.scale == 'log':
                    candidate_params[param_name] = np.exp(
                        np.random.uniform(np.log(bounds.lower), np.log(bounds.upper))
                    )
                else:
                    candidate_params[param_name] = np.random.uniform(bounds.lower, bounds.upper)
            
            candidate_objective = objective_function(candidate_params, observations)
            
            if candidate_objective < best_objective:
                best_params = candidate_params
                best_objective = candidate_objective
            
            convergence_history.append(best_objective)
            
            if best_objective < config.tolerance:
                break
        
        return OptimizationResult(
            optimized_parameters=best_params,
            convergence_history=convergence_history,
            iteration_count=len(convergence_history),
            success=best_objective < config.tolerance,
            termination_reason="Converged" if best_objective < config.tolerance else "Max iterations reached"
        )
    
    def _calculate_metrics(self,
                          parameters: Dict[str, float],
                          objective_function: ObjectiveFunction,
                          observations: torch.Tensor) -> Optional[CalibrationMetrics]:
        """Calculate goodness-of-fit metrics from the model output of the optimized parameters.

        Returns None when the objective function exposes no model_runner: no
        metric can then be computed and none is substituted (the calibration
        validator reports the missing metrics).
        """
        if not hasattr(objective_function, 'model_runner'):
            return None

        # References:
        #   Nash and Sutcliffe (1970), J. Hydrol. 10(3):282-290 (NSE); Moriasi et al. (2007),
        #   Trans. ASABE 50(3):885-900, p. 886-887 (MSE, RMSE, MAE, Pearson r, R2, bias, percent bias).
        # Mathematical expression:
        #   MSE = mean((m - o)^2); RMSE = sqrt(MSE); MAE = mean(|m - o|);
        #   NSE = 1 - SS_res / SS_tot, SS_res = sum((o - m)^2), SS_tot = sum((o - o_mean)^2);
        #   r = sum((m - m_mean)(o - o_mean)) / sqrt(SS_m * SS_tot), SS_m = sum((m - m_mean)^2);
        #   R2 = r^2 (Moriasi et al. 2007, p. 887: collinearity of simulated and observed data);
        #   bias = mean(m - o); percent_bias = 100 * sum(m - o) / sum(o)
        # Parameter meanings:
        #   m: flattened model outputs of every returned variable; o: flattened observations;
        #   m_mean, o_mean: means of m and o.
        # Units:
        #   m, o, MAE, RMSE and bias in the observation units u; MSE in u^2; r, R2 and NSE
        #   dimensionless; percent_bias in percent.
        # Domain/constraints:
        #   SS_tot > 0 (observations with variance) for NSE, r and R2; SS_m > 0 (model output
        #   with variance) for r and R2; sum(o) != 0 for percent_bias; otherwise ValueError, no
        #   sentinel value. r is clamped to [-1, 1] against roundoff only (as numpy.corrcoef).
        #   A model failure propagates.
        # Sign convention:
        #   a positive bias means the model overestimates the observations; r > 0 means the
        #   model and the observations vary in the same direction.
        model_output = objective_function.model_runner(parameters)

        if isinstance(model_output, dict):
            model_values = torch.cat([
                output.flatten() for output in model_output.values()
            ])
        else:
            model_values = model_output.flatten()

        obs_values = observations.flatten()

        mse = torch.mean((model_values - obs_values) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(model_values - obs_values))

        obs_mean = torch.mean(obs_values)
        ss_res = torch.sum((obs_values - model_values) ** 2)
        ss_tot = torch.sum((obs_values - obs_mean) ** 2)
        if ss_tot == 0:
            raise ValueError(
                "R-squared and Nash-Sutcliffe efficiency are undefined for observations "
                "without variance"
            )
        obs_sum = torch.sum(obs_values)
        if obs_sum == 0:
            raise ValueError("percent bias is undefined for observations that sum to zero")

        model_centered = model_values - torch.mean(model_values)
        ss_model = torch.sum(model_centered ** 2)
        if ss_model == 0:
            raise ValueError(
                "Pearson correlation (and R-squared = r^2) is undefined for a model output "
                "without variance"
            )

        nse = 1 - (ss_res / ss_tot)
        correlation = torch.clamp(
            torch.sum(model_centered * (obs_values - obs_mean)) / torch.sqrt(ss_model * ss_tot),
            -1.0,
            1.0,
        )

        return CalibrationMetrics(
            mse=float(mse),
            rmse=float(rmse),
            mae=float(mae),
            r_squared=float(correlation ** 2),
            nash_sutcliffe=float(nse),
            bias=float(torch.mean(model_values - obs_values)),
            percent_bias=float(100 * torch.sum(model_values - obs_values) / obs_sum),
            correlation=float(correlation),
        )


@dataclass
class OptimizationResult:
    """Result from optimization algorithm"""
    optimized_parameters: Dict[str, float]
    convergence_history: List[float]
    iteration_count: int
    success: bool
    termination_reason: str
