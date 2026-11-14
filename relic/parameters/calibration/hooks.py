"""
Calibration Hooks and Event System

This module provides an event-driven hook system for parameter calibration,
allowing users to register callbacks for different calibration events.
This enables monitoring, logging, visualization, and custom behaviors
during the calibration process.

Scientific Background:
    Parameter calibration is an iterative process that benefits from 
    monitoring and intervention capabilities. Hooks allow researchers to:
    - Monitor convergence in real-time
    - Implement adaptive strategies
    - Apply domain-specific constraints

References:
    - Hadley (2011). R packages: organize, test, document, and share your code.
    - Event-driven programming patterns in scientific computing
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import logging
import time
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor

from ..manager import get_global_manager

# Local alias to avoid runtime import cycle with calibrator module.
ParameterBounds = Any


class CalibrationEvent(Enum):
    """Enumeration of calibration events that can trigger hooks"""
    
    # Main calibration events
    CALIBRATION_START = "calibration_start"
    CALIBRATION_COMPLETE = "calibration_complete"
    CALIBRATION_ERROR = "calibration_error"
    CALIBRATION_ABORT = "calibration_abort"
    
    # Iteration events
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    ITERATION_PROGRESS = "iteration_progress"
    
    # Parameter events
    PARAMETER_UPDATE = "parameter_update"
    PARAMETER_BOUNDS_VIOLATION = "parameter_bounds_violation"
    PARAMETER_VALIDATION_ERROR = "parameter_validation_error"
    
    # Objective function events
    OBJECTIVE_EVALUATION = "objective_evaluation"
    OBJECTIVE_IMPROVEMENT = "objective_improvement"
    OBJECTIVE_STAGNATION = "objective_stagnation"
    
    # Convergence events
    CONVERGENCE_CHECK = "convergence_check"
    CONVERGENCE_ACHIEVED = "convergence_achieved"
    CONVERGENCE_FAILED = "convergence_failed"
    
    # Population-based algorithm events
    POPULATION_INITIALIZED = "population_initialized"
    POPULATION_EVOLVED = "population_evolved"
    ELITE_UPDATED = "elite_updated"
    
    # Custom events
    CUSTOM_WARNING = "custom_warning"
    CUSTOM_INFO = "custom_info"


@dataclass
class CalibrationEventData:
    """Data structure for calibration event information"""
    event: CalibrationEvent
    timestamp: float
    iteration: Optional[int] = None
    parameters: Optional[Dict[str, float]] = None
    objective_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event': self.event.value,
            'timestamp': self.timestamp,
            'iteration': self.iteration,
            'parameters': self.parameters,
            'objective_value': self.objective_value,
            'metadata': self.metadata
        }


class CalibrationCallback(ABC):
    """Abstract base class for calibration callbacks"""
    
    @abstractmethod
    def __call__(self, event_data: CalibrationEventData) -> Optional[Dict[str, Any]]:
        """
        Handle calibration event
        
        Parameters:
            event_data: Event data structure
            
        Returns:
            Optional dictionary with response data
        """
    
    @property
    def supported_events(self) -> List[CalibrationEvent]:
        """Return list of events this callback handles"""
        return [event for event in CalibrationEvent]


class LoggingCallback(CalibrationCallback):
    """Callback for logging calibration events"""
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 log_level: int = logging.INFO,
                 include_parameters: bool = False):
        self.logger = logger or logging.getLogger(__name__)
        self.log_level = log_level
        self.include_parameters = include_parameters
    
    def __call__(self, event_data: CalibrationEventData) -> Optional[Dict[str, Any]]:
        """Log calibration event"""
        message = f"Calibration Event: {event_data.event.value}"
        
        if event_data.iteration is not None:
            message += f" (Iteration {event_data.iteration})"
        
        if event_data.objective_value is not None:
            message += f" | Objective: {event_data.objective_value:.6e}"
        
        if self.include_parameters and event_data.parameters:
            param_str = ", ".join([
                f"{k}={v:.4f}" for k, v in event_data.parameters.items()
            ])
            message += f" | Parameters: {param_str}"
        
        if event_data.metadata:
            metadata_str = ", ".join([
                f"{k}={v}" for k, v in event_data.metadata.items()
            ])
            message += f" | Metadata: {metadata_str}"
        
        self.logger.log(self.log_level, message)
        return None


class ProgressCallback(CalibrationCallback):
    """Callback for monitoring calibration progress"""
    
    def __init__(self, print_interval: int = 10):
        self.print_interval = print_interval
        self.start_time = None
        self.last_print_iteration = 0
    
    def __call__(self, event_data: CalibrationEventData) -> Optional[Dict[str, Any]]:
        """Print calibration progress"""
        if event_data.event == CalibrationEvent.CALIBRATION_START:
            self.start_time = event_data.timestamp
            print("Calibration started...")
            
        elif event_data.event == CalibrationEvent.ITERATION_COMPLETE:
            if (event_data.iteration is not None and 
                event_data.iteration % self.print_interval == 0):
                
                elapsed = event_data.timestamp - self.start_time if self.start_time else 0
                objective_str = f"{event_data.objective_value:.6e}" if event_data.objective_value else "N/A"
                
                print(f"Iteration {event_data.iteration:4d} | "
                      f"Objective: {objective_str} | "
                      f"Elapsed: {elapsed:.1f}s")
                
                self.last_print_iteration = event_data.iteration
                
        elif event_data.event == CalibrationEvent.CALIBRATION_COMPLETE:
            elapsed = event_data.timestamp - self.start_time if self.start_time else 0
            print(f"Calibration completed in {elapsed:.1f}s")
            
        elif event_data.event == CalibrationEvent.CALIBRATION_ERROR:
            print(f"Calibration error: {event_data.metadata.get('error', 'Unknown error')}")
        
        return None


class ConvergenceCallback(CalibrationCallback):
    """Callback for monitoring convergence criteria"""
    
    def __init__(self,
                 patience: Optional[int] = None,
                 min_improvement: Optional[float] = None,
                 early_stopping: bool = True):
        # Unstated settings are read from the parameter database; the same keys
        # drive the early stopping of GradientCalibrator.
        manager = get_global_manager()
        self.patience = int(
            patience if patience is not None
            else manager.get("optimization.calibration.early_stopping_patience", required=True)
        )
        self.min_improvement = float(
            min_improvement if min_improvement is not None
            else manager.get("optimization.calibration.min_objective_improvement", required=True)
        )
        self.early_stopping = early_stopping
        self.best_objective = float('inf')
        self.patience_counter = 0
        self.convergence_history = []
    
    def __call__(self, event_data: CalibrationEventData) -> Optional[Dict[str, Any]]:
        """Monitor convergence and potentially trigger early stopping"""
        if (event_data.event == CalibrationEvent.OBJECTIVE_EVALUATION and
            event_data.objective_value is not None):
            
            current_objective = event_data.objective_value
            self.convergence_history.append(current_objective)
            
            # Check for improvement
            if current_objective < self.best_objective - self.min_improvement:
                self.best_objective = current_objective
                self.patience_counter = 0
                
                # Trigger improvement event
                return {
                    'trigger_event': CalibrationEvent.OBJECTIVE_IMPROVEMENT,
                    'best_objective': self.best_objective,
                    'improvement': self.best_objective - current_objective
                }
            else:
                self.patience_counter += 1
                
                # Check for stagnation
                if self.patience_counter >= self.patience:
                    if self.early_stopping:
                        return {
                            'trigger_event': CalibrationEvent.CONVERGENCE_FAILED,
                            'reason': 'patience_exceeded',
                            'patience_counter': self.patience_counter,
                            'request_termination': True
                        }
                    else:
                        return {
                            'trigger_event': CalibrationEvent.OBJECTIVE_STAGNATION,
                            'patience_counter': self.patience_counter
                        }
        
        return None


class ParameterBoundsCallback(CalibrationCallback):
    """Callback for monitoring parameter bounds violations"""
    
    def __init__(self, parameter_bounds: Dict[str, 'ParameterBounds']):
        self.parameter_bounds = parameter_bounds
        self.violation_counts = {}
    
    def __call__(self, event_data: CalibrationEventData) -> Optional[Dict[str, Any]]:
        """Check for parameter bounds violations"""
        if (event_data.event == CalibrationEvent.PARAMETER_UPDATE and
            event_data.parameters):
            
            violations = []
            for param_name, value in event_data.parameters.items():
                if param_name in self.parameter_bounds:
                    bounds = self.parameter_bounds[param_name]
                    
                    if value < bounds.lower or value > bounds.upper:
                        violations.append({
                            'parameter': param_name,
                            'value': value,
                            'bounds': (bounds.lower, bounds.upper)
                        })
                        
                        # Count violations
                        if param_name not in self.violation_counts:
                            self.violation_counts[param_name] = 0
                        self.violation_counts[param_name] += 1
            
            if violations:
                return {
                    'trigger_event': CalibrationEvent.PARAMETER_BOUNDS_VIOLATION,
                    'violations': violations,
                    'violation_counts': self.violation_counts
                }
        
        return None


class AdaptiveCallback(CalibrationCallback):
    """Callback for adaptive calibration strategies"""
    
    def __init__(self, 
                 adaptation_interval: Optional[int] = None,
                 population_adaptation: bool = True,
                 learning_rate_adaptation: bool = True):
        # Unstated settings and every threshold/factor are read from the parameter
        # database (optimization.calibration.adaptive.*); they are RELIC convention values.
        manager = get_global_manager()
        prefix = "optimization.calibration.adaptive."
        self.adaptation_interval = int(
            adaptation_interval if adaptation_interval is not None
            else manager.get(prefix + "adaptation_interval", required=True)
        )
        self.improvement_scale_floor = float(manager.get(prefix + "improvement_scale_floor", required=True))
        self.population_stagnation_rate = float(manager.get(prefix + "population_stagnation_rate", required=True))
        self.population_growth_factor = float(manager.get(prefix + "population_growth_factor", required=True))
        self.learning_rate_decrease_rate = float(manager.get(prefix + "learning_rate_decrease_rate", required=True))
        self.learning_rate_decrease_factor = float(manager.get(prefix + "learning_rate_decrease_factor", required=True))
        self.learning_rate_increase_rate = float(manager.get(prefix + "learning_rate_increase_rate", required=True))
        self.learning_rate_increase_factor = float(manager.get(prefix + "learning_rate_increase_factor", required=True))
        self.population_adaptation = population_adaptation
        self.learning_rate_adaptation = learning_rate_adaptation
        self.objective_history = []
        self.adaptation_count = 0
    
    def __call__(self, event_data: CalibrationEventData) -> Optional[Dict[str, Any]]:
        """Implement adaptive strategies"""
        if (event_data.event == CalibrationEvent.ITERATION_COMPLETE and
            event_data.iteration is not None and
            event_data.iteration % self.adaptation_interval == 0 and
            event_data.objective_value is not None):
            
            self.objective_history.append(event_data.objective_value)
            
            if len(self.objective_history) >= 2:
                # References: Hairer, Norsett and Wanner (1993), Solving ODEs I, Sect. II.4
                #   (mixed absolute/relative error scale sc = Atol + |y| Rtol).
                # Mathematical expression: rho = (L_prev - L_curr) / max(|L_prev|, s).
                # Parameter meanings: L_prev, L_curr the objective values at the previous and
                #   the current adaptation point; s = optimization.calibration.adaptive.
                #   improvement_scale_floor (RELIC convention value).
                # Units: L_prev, L_curr and s in the units of the objective function; rho
                #   dimensionless.
                # Domain/constraints: s > 0, so rho is defined when L_prev = 0 (it is then the
                #   improvement in units of s); relative to |L_prev| above the floor.
                # Sign convention: rho > 0 when the objective decreased (improved), rho < 0
                #   when it increased; |L_prev| keeps the sign for negative objectives.
                previous = self.objective_history[-2]
                current = self.objective_history[-1]
                recent_improvement = (previous - current) / max(abs(previous), self.improvement_scale_floor)
                
                suggestions = {}
                
                # Adapt population size for population-based methods
                if self.population_adaptation and recent_improvement < self.population_stagnation_rate:
                    suggestions['increase_population'] = True
                    suggestions['population_factor'] = self.population_growth_factor
                
                # Adapt learning rate for gradient-based methods
                if self.learning_rate_adaptation:
                    if recent_improvement < self.learning_rate_decrease_rate:
                        suggestions['decrease_learning_rate'] = True
                        suggestions['learning_rate_factor'] = self.learning_rate_decrease_factor
                    elif recent_improvement > self.learning_rate_increase_rate:
                        suggestions['increase_learning_rate'] = True
                        suggestions['learning_rate_factor'] = self.learning_rate_increase_factor
                
                if suggestions:
                    self.adaptation_count += 1
                    return {
                        'trigger_event': CalibrationEvent.CUSTOM_INFO,
                        'message': 'adaptive_suggestions',
                        'suggestions': suggestions,
                        'improvement_rate': recent_improvement,
                        'adaptation_count': self.adaptation_count
                    }
        
        return None


class CalibrationHooks:
    """
    Main hooks manager for calibration events
    
    This class manages the registration and execution of callbacks for
    different calibration events. It supports both synchronous and
    asynchronous callback execution.
    """
    
    def __init__(self, 
                 async_execution: bool = False,
                 max_workers: int = 4):
        self.callbacks: Dict[CalibrationEvent, List[CalibrationCallback]] = {}
        self.global_callbacks: List[CalibrationCallback] = []
        self.async_execution = async_execution
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Initialize executor for async callbacks
        if self.async_execution:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = None
    
    def register_callback(self, 
                         callback: CalibrationCallback,
                         events: Optional[List[CalibrationEvent]] = None):
        """
        Register a callback for specific events
        
        Parameters:
            callback: Callback function to register
            events: List of events to handle (None = all events)
        """
        with self._lock:
            if events is None:
                # Register as global callback
                self.global_callbacks.append(callback)
            else:
                # Register for specific events
                for event in events:
                    if event not in self.callbacks:
                        self.callbacks[event] = []
                    self.callbacks[event].append(callback)
    
    def unregister_callback(self, callback: CalibrationCallback):
        """Remove a callback from all events"""
        with self._lock:
            # Remove from global callbacks
            if callback in self.global_callbacks:
                self.global_callbacks.remove(callback)
            
            # Remove from specific event callbacks
            for event_callbacks in self.callbacks.values():
                if callback in event_callbacks:
                    event_callbacks.remove(callback)
    
    def trigger_event(self, 
                     event: CalibrationEvent,
                     metadata: Optional[Dict[str, Any]] = None,
                     iteration: Optional[int] = None,
                     parameters: Optional[Dict[str, float]] = None,
                     objective_value: Optional[float] = None):
        """
        Trigger a calibration event and execute associated callbacks
        
        Parameters:
            event: Event type to trigger
            metadata: Additional event metadata
            iteration: Current iteration number
            parameters: Current parameter values
            objective_value: Current objective function value
        """
        # Create event data
        event_data = CalibrationEventData(
            event=event,
            timestamp=time.time(),
            iteration=iteration,
            parameters=parameters,
            objective_value=objective_value,
            metadata=metadata or {}
        )
        
        # Get callbacks for this event
        callbacks_to_execute = []
        
        with self._lock:
            # Add global callbacks
            callbacks_to_execute.extend(self.global_callbacks)
            
            # Add event-specific callbacks
            if event in self.callbacks:
                callbacks_to_execute.extend(self.callbacks[event])
        
        # Execute callbacks. A callback failure, or an async callback that does
        # not finish within the 30 s wait, propagates to the calibration run:
        # callbacks drive early stopping, bounds monitoring and adaptation, so a
        # failed one must not be logged and skipped.
        if self.async_execution and self.executor:
            # Asynchronous execution
            futures = [
                self.executor.submit(callback, event_data)
                for callback in callbacks_to_execute
            ]

            # Collect results in registration order
            for future in futures:
                result = future.result(timeout=30)  # 30 second timeout
                if result:
                    self._handle_callback_result(result, event_data)
        else:
            # Synchronous execution
            for callback in callbacks_to_execute:
                result = callback(event_data)
                if result:
                    self._handle_callback_result(result, event_data)
    
    def _handle_callback_result(self, 
                               result: Dict[str, Any],
                               original_event_data: CalibrationEventData):
        """Handle results from callback execution"""
        if 'trigger_event' in result:
            # Callback wants to trigger another event
            new_event = result['trigger_event']
            new_metadata = {k: v for k, v in result.items() if k != 'trigger_event'}
            
            # Trigger the new event
            self.trigger_event(
                event=new_event,
                metadata=new_metadata,
                iteration=original_event_data.iteration,
                parameters=original_event_data.parameters,
                objective_value=original_event_data.objective_value
            )
    
    def add_logging(self, 
                   log_level: int = logging.INFO,
                   include_parameters: bool = False):
        """Add default logging callback"""
        logging_callback = LoggingCallback(
            log_level=log_level,
            include_parameters=include_parameters
        )
        self.register_callback(logging_callback)
    
    def add_progress_monitoring(self, print_interval: int = 10):
        """Add default progress monitoring callback"""
        progress_callback = ProgressCallback(print_interval=print_interval)
        self.register_callback(progress_callback)
    
    def add_convergence_monitoring(self,
                                  patience: Optional[int] = None,
                                  min_improvement: Optional[float] = None,
                                  early_stopping: bool = True):
        """Add convergence monitoring callback (None reads the parameter database)"""
        convergence_callback = ConvergenceCallback(
            patience=patience,
            min_improvement=min_improvement,
            early_stopping=early_stopping
        )
        self.register_callback(convergence_callback)
    
    def add_bounds_monitoring(self, parameter_bounds: Dict[str, 'ParameterBounds']):
        """Add parameter bounds monitoring callback"""
        bounds_callback = ParameterBoundsCallback(parameter_bounds)
        self.register_callback(bounds_callback)
    
    def add_adaptive_strategies(self,
                               adaptation_interval: Optional[int] = None,
                               population_adaptation: bool = True,
                               learning_rate_adaptation: bool = True):
        """Add adaptive calibration strategies callback (None reads the parameter database)"""
        adaptive_callback = AdaptiveCallback(
            adaptation_interval=adaptation_interval,
            population_adaptation=population_adaptation,
            learning_rate_adaptation=learning_rate_adaptation
        )
        self.register_callback(adaptive_callback)
    
    def cleanup(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)


# Example usage and convenience functions
def create_default_hooks(log_level: int = logging.INFO,
                        progress_interval: int = 10,
                        enable_convergence_monitoring: bool = True,
                        enable_adaptive_strategies: bool = True) -> CalibrationHooks:
    """
    Create a CalibrationHooks instance with default callbacks
    
    Parameters:
        log_level: Logging level for events
        progress_interval: Interval for progress updates
        enable_convergence_monitoring: Whether to enable convergence monitoring
        enable_adaptive_strategies: Whether to enable adaptive strategies
        
    Returns:
        Configured CalibrationHooks instance
    """
    hooks = CalibrationHooks()
    
    # Add logging
    hooks.add_logging(log_level=log_level)
    
    # Add progress monitoring
    hooks.add_progress_monitoring(print_interval=progress_interval)
    
    # Add convergence monitoring
    if enable_convergence_monitoring:
        hooks.add_convergence_monitoring()
    
    # Add adaptive strategies
    if enable_adaptive_strategies:
        hooks.add_adaptive_strategies()
    
    return hooks
