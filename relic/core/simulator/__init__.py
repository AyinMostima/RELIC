"""Simulator subpackage with main engine and helpers."""

from .main import (
    RELICSimulator,
    SimulationState,
    create_simulator,
    SimulatorConfig,
    TimeSeriesData,
    CheckpointData,
    DependencyGraph,
    HAS_SCALE_MANAGER,
    ScaleManager,
    HAS_DISTRIBUTED,
    ProcessGroupManager,
    get_process_group_manager,
    is_distributed,
    get_rank,
    get_world_size,
)
from .checkpoint import (
    create_checkpoint,
    restore_checkpoint,
)
from .diagnostics import (
    collect_functor_diagnostics,
    export_diagnostics,
)

__all__ = [
    "RELICSimulator",
    "SimulationState",
    "create_simulator",
    "SimulatorConfig",
    "TimeSeriesData",
    "CheckpointData",
    "DependencyGraph",
    "HAS_SCALE_MANAGER",
    "ScaleManager",
    "HAS_DISTRIBUTED",
    "ProcessGroupManager",
    "get_process_group_manager",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "create_checkpoint",
    "restore_checkpoint",
    "collect_functor_diagnostics",
    "export_diagnostics",
]
