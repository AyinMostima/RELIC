"""
Distributed Computing Submodule

Provides distributed computing capabilities for RELIC including:
- Windows Gloo backend support
- Multi-node and multi-GPU distributed training/simulation
- Process group management with automatic cleanup
- Distributed ensemble simulations
- Integration with PyTorch distributed communication

Main Components:
    - ProcessGroupManager: Process group lifecycle management
    - DistributedSimulationLauncher: Distributed simulation utilities
    - Collective communication primitives (all-reduce, broadcast, etc.)
    - Windows-friendly distributed computing setup
"""

# Import distributed components
from .launcher import (
    # Classes
    ProcessGroupManager,
    DistributedConfig,
    DistributedSimulationLauncher,
    DistributedBackend,
    DistributedError,

    # Functions
    get_process_group_manager,
    is_distributed,
    get_rank,
    get_world_size,
    is_master,
    init_distributed,
    cleanup_distributed,
    distributed_context,
    barrier,
    all_reduce,
    broadcast,
    distributed_wrapper,
    distributed_healthcheck,
    setup_distributed_simulation,
    finalize_distributed_simulation,
    DISTRIBUTED_AVAILABLE as _LAUNCHER_DISTRIBUTED_AVAILABLE,
)

DISTRIBUTED_AVAILABLE = bool(_LAUNCHER_DISTRIBUTED_AVAILABLE)


# Utility functions for distributed scientific computing
def distributed_ensemble_run(
    simulation_func,
    parameter_sets: list,
    backend: str = "gloo",
    allow_single_process: bool = False,
    **kwargs,
) -> list:
    """
    Run ensemble simulation in distributed fashion.
    
    Parameters:
        simulation_func : Callable
            Simulation function to execute
        parameter_sets : list
            List of parameter dictionaries
        backend : str
            Distributed backend ("gloo" recommended for Windows)
        **kwargs : Additional distributed configuration
    
    Returns:
        list: Results from distributed simulation
    """
    if not DISTRIBUTED_AVAILABLE:
        if not allow_single_process:
            raise RuntimeError("Distributed computing not available")
        return [simulation_func(params) for params in parameter_sets]
    
    launcher = DistributedSimulationLauncher()
    return launcher.launch_ensemble_simulation(simulation_func, parameter_sets, **kwargs)


def master_only(func=None, *, allow_single_process: bool = False):
    """
    Decorator to execute function only on master process (rank 0).
    
    Parameters:
        func : Callable
            Function to wrap
    
    Returns:
        Callable: Wrapped function that only runs on master
    """
    def decorator(wrapped):
        if not DISTRIBUTED_AVAILABLE:
            if allow_single_process:
                return wrapped
            def _raise(*args, **kwargs):
                raise RuntimeError("Distributed computing not available")
            return _raise
    
        from functools import wraps
        @wraps(wrapped)
        def wrapper(*args, **kwargs):
            if is_master():
                return wrapped(*args, **kwargs)
            return None
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)


def all_ranks(func):
    """
    Decorator to explicitly execute function on all ranks.
    
    Parameters:
        func : Callable
            Function to wrap
    
    Returns:
        Callable: Function that runs on all ranks
    """
    # This is mainly for documentation purposes
    return func


def synchronize_processes():
    """Synchronize all distributed processes."""
    if is_distributed():
        barrier()


def get_distributed_info() -> dict:
    """
    Get current distributed computing information.
    
    Returns:
        dict: Distributed state information
    """
    return {
        'available': DISTRIBUTED_AVAILABLE,
        'initialized': is_distributed(),
        'rank': get_rank(),
        'world_size': get_world_size(),
        'is_master': is_master()
    }


# Export interface  
__all__ = [
    # Availability flag
    'DISTRIBUTED_AVAILABLE',
    
    # Classes
    'ProcessGroupManager',
    'DistributedConfig',
    'DistributedSimulationLauncher', 
    'DistributedBackend',
    'DistributedError',
    
    # Functions
    'is_distributed',
    'get_rank',
    'get_world_size',
    'is_master',
    'init_distributed',
    'cleanup_distributed',
    'distributed_context',
    'barrier',
    'all_reduce',
    'broadcast',
    'distributed_wrapper',
    'distributed_healthcheck',
    'setup_distributed_simulation',
    'finalize_distributed_simulation',
    'get_process_group_manager',
    
    # Utility functions
    'distributed_ensemble_run',
    'master_only',
    'all_ranks', 
    'synchronize_processes',
    'get_distributed_info',
    
]
