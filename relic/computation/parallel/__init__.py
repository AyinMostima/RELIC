"""
Parallel Computation Submodule

Provides parallel execution capabilities for RELIC including:
- Thread-based and process-based parallel execution
- Automatic backend selection and optimization
- Intelligent chunk sizing and load balancing  
- Integration with stochastic ensemble simulations
- Comprehensive error handling and statistics

Main Components:
    - ParallelExecutor: High-level parallel execution engine
    - ParallelConfig: Configuration management
    - Load balancing strategies for heterogeneous workloads
    - Chunk size optimization for scientific computing
"""

from dataclasses import dataclass
from typing import Any, Callable

# Import main parallel execution components
from .executor import (
    # Classes
    ParallelExecutor,
    ParallelConfig,
    ExecutionResult,
    ParallelBackend,
    ChunkStrategy,
    SimpleLoadBalancer,
    WeightedLoadBalancer,
    ChunkSizeOptimizer,
    LoadBalancer,

    # Functions
    parallel_map,
    parallel_starmap,
    parallel_context,
    get_parallel_stats,
    parallel_healthcheck,

    # Global instances
    _default_executor
)

PARALLEL_AVAILABLE = True


@dataclass(frozen=True)
class _ParameterSweepCallable:
    """Picklable adapter from ordered combinations to keyword arguments."""

    func: Callable[..., Any]
    parameter_names: tuple[str, ...]

    def __call__(self, combination: tuple[Any, ...]) -> Any:
        return self.func(**dict(zip(self.parameter_names, combination)))


# Convenience functions for scientific computing patterns
def parallel_parameter_sweep(
    func,
    parameter_ranges: dict,
    backend: str = "auto",
    max_workers=None,
    allow_sequential: bool = False,
):
    """
    Parallel parameter sweep for scientific simulations.
    
    Parameters:
        func : Callable
            Function to evaluate (should accept **kwargs)
        parameter_ranges : dict
            Dictionary of parameter names to ranges/lists
        backend : str
            Parallel backend to use
        max_workers : int
            Maximum number of workers
    
    Returns:
        List: Results for each parameter combination
    """
    if not PARALLEL_AVAILABLE:
        if not allow_sequential:
            raise RuntimeError("Parallel execution not available")
        import itertools
        param_names = list(parameter_ranges.keys())
        param_values = [parameter_ranges[name] for name in param_names]
        results = []
        
        for combo in itertools.product(*param_values):
            kwargs = dict(zip(param_names, combo))
            result = func(**kwargs)
            results.append(result)
        
        return results
    
    import itertools
    
    # Generate all parameter combinations
    param_names = list(parameter_ranges.keys())
    param_values = [parameter_ranges[name] for name in param_names]
    
    # Create all combinations
    combinations = list(itertools.product(*param_values))
    evaluate_parameters = _ParameterSweepCallable(
        func=func,
        parameter_names=tuple(param_names),
    )
    
    # Execute in parallel
    return parallel_map(
        evaluate_parameters, 
        combinations,
        backend=backend,
        max_workers=max_workers,
        complexity="high",
        allow_sequential=allow_sequential,
    )


def parallel_ensemble_run(
    simulation_func,
    parameter_sets: list,
    backend: str = "processes",  # Prefer processes for ensemble runs
    max_workers=None,
    progress_callback=None,
    allow_sequential: bool = False,
):
    """
    Run ensemble simulations in parallel.
    
    Parameters:
        simulation_func : Callable
            Simulation function
        parameter_sets : list
            List of parameter dictionaries
        backend : str
            Parallel backend
        max_workers : int
            Maximum number of workers
        progress_callback : Callable
            Progress callback function
    
    Returns:
        List: Simulation results
    """
    if not PARALLEL_AVAILABLE:
        if not allow_sequential:
            raise RuntimeError("Parallel execution not available")
        results = []
        for i, params in enumerate(parameter_sets):
            result = simulation_func(params)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(parameter_sets))
        return results
    
    config = ParallelConfig(
        backend=ParallelBackend(backend),
        max_workers=max_workers,
        progress_callback=progress_callback,
        enable_profiling=True
    )
    
    executor = ParallelExecutor(config)
    return executor.parallel_map(simulation_func, parameter_sets, complexity="high")


def parallel_batch_evaluation(
    func,
    data_batches: list,
    backend: str = "threads",  # Prefer threads for data processing
    max_workers=None,
    allow_sequential: bool = False,
):
    """
    Evaluate function on data batches in parallel.
    
    Parameters:
        func : Callable
            Function to apply to each batch
        data_batches : list
            List of data batches
        backend : str
            Parallel backend
        max_workers : int
            Maximum number of workers
    
    Returns:
        List: Results for each batch
    """
    return parallel_map(
        func, 
        data_batches,
        backend=backend,
        max_workers=max_workers,
        complexity="medium",
        allow_sequential=allow_sequential,
    )


# Export interface
__all__ = [
    # Availability flag
    'PARALLEL_AVAILABLE',
    
    # Classes
    'ParallelExecutor',
    'ParallelConfig',
    'ExecutionResult', 
    'ParallelBackend',
    'ChunkStrategy',
    'SimpleLoadBalancer',
    'WeightedLoadBalancer',
    'ChunkSizeOptimizer',
    'LoadBalancer',
    
    # Functions
    'parallel_map',
    'parallel_starmap',
    'parallel_context',
    'get_parallel_stats',
    'parallel_healthcheck',
    
    # Scientific computing patterns
    'parallel_parameter_sweep',
    'parallel_ensemble_run',
    'parallel_batch_evaluation',
    
    # Global instances
    '_default_executor'
]
