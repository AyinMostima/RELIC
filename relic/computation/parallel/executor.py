"""
Parallel Execution Module for RELIC

This module provides parallel computation capabilities optimized for scientific
computing workloads. Supports both thread-based and process-based parallelism
with automatic backend selection, intelligent chunk sizing, and comprehensive
error handling.

Key Features:
    - Thread-safe parallel execution with multiple backends
    - Intelligent chunk size optimization for batch operations
    - Automatic load balancing and work stealing
    - Memory-efficient processing for large datasets
    - Exception handling and result aggregation
    - Integration with stochastic ensemble simulations

Mathematical Considerations:
    - Vectorization over parallel loops where possible
    - Embarrassingly parallel scientific computations
    - Parameter sweep and uncertainty quantification support
    - Numerical stability across parallel execution paths
    - Consistent random number generation with distributed seeds

References:
    - Python concurrent.futures documentation
    - Parallel computing patterns for scientific applications
    - Load balancing strategies for heterogeneous workloads
    - Memory management in parallel scientific computing
"""

import logging
import os
import time
import threading
from typing import (
    Dict, Any, Optional, Callable, List, Tuple, TypeVar, 
    Protocol, Mapping
)
from concurrent.futures import (
    ThreadPoolExecutor, ProcessPoolExecutor, 
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from contextlib import contextmanager
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
import math

# Required scientific runtime dependencies
import psutil

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
R = TypeVar('R')

# Global execution statistics
_EXECUTION_STATS = {
    'tasks_submitted': 0,
    'tasks_completed': 0,
    'tasks_failed': 0,
    'total_execution_time': 0.0,
}
_EXECUTION_STATS_LOCK = threading.Lock()


class ParallelBackend(Enum):
    """Enumeration of available parallel backends."""
    THREADS = "threads"
    PROCESSES = "processes"
    AUTO = "auto"


class ChunkStrategy(Enum):
    """Enumeration of chunking strategies for parallel execution."""
    AUTO = "auto"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    DYNAMIC = "dynamic"


class LoadBalancer(Protocol):
    """Protocol for load balancing strategies."""
    
    def distribute_work(self, items: List[T], num_workers: int) -> List[List[T]]:
        """Distribute work items among workers."""
        ...


@dataclass
class ExecutionResult:
    """Result container for parallel execution."""
    results: List[Any]
    execution_time: float
    worker_stats: Dict[str, Any]
    errors: List[Exception]
    success_rate: float


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    backend: ParallelBackend = ParallelBackend.AUTO
    max_workers: Optional[int] = None
    chunk_strategy: ChunkStrategy = ChunkStrategy.AUTO
    chunk_size: Optional[int] = None
    timeout: Optional[float] = None
    enable_profiling: bool = False
    progress_callback: Optional[Callable[[int, int], None]] = None
    allow_sequential: bool = False
    torch_threads_per_worker: Optional[int] = None
    enforce_torch_thread_budget: bool = True

    @classmethod
    def from_computation_config(cls, config: Any) -> "ParallelConfig":
        """Normalize the canonical computation policy for one execution."""
        if hasattr(config, "model_dump"):
            payload = config.model_dump()
        elif isinstance(config, Mapping):
            payload = dict(config)
        else:
            raise TypeError(
                "computation config must be a mapping or Pydantic model"
            )

        nested = payload.get("computation")
        if isinstance(nested, Mapping):
            payload = dict(nested)

        backend_value = getattr(
            payload.get("parallel_backend", "threads"),
            "value",
            payload.get("parallel_backend", "threads"),
        )
        backend_name = str(backend_value).strip().lower()
        if backend_name in {"dask", "ray"}:
            raise NotImplementedError(
                f"Parallel backend '{backend_name}' is declared but not wired "
                "to the canonical ParallelExecutor runtime."
            )
        try:
            backend = ParallelBackend(backend_name)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported computation parallel backend: {backend_name}"
            ) from exc

        chunk_name = str(payload.get("chunk_size", "auto")).strip().lower()
        chunk_strategies = {
            "auto": ChunkStrategy.AUTO,
            "small": ChunkStrategy.DYNAMIC,
            "medium": ChunkStrategy.AUTO,
            "large": ChunkStrategy.FIXED,
            "adaptive": ChunkStrategy.ADAPTIVE,
        }
        if chunk_name not in chunk_strategies:
            raise ValueError(
                f"Unsupported computation chunk_size strategy: {chunk_name}"
            )

        max_workers = payload.get("num_workers")
        if max_workers is not None:
            max_workers = int(max_workers)

        return cls(
            backend=backend,
            max_workers=max_workers,
            chunk_strategy=chunk_strategies[chunk_name],
            enable_profiling=bool(payload.get("enable_profiling", False)),
            allow_sequential=False,
        )


@dataclass
class _ChunkExecution:
    """Internal chunk result with real worker identity and thread budget."""

    results: List[Any]
    pid: int
    torch_threads: int


def _get_torch_thread_count() -> int:
    """Read the active Torch intra-op thread budget.

    Torch is a required RELIC dependency; a failing probe propagates and is
    never reported as a one-thread budget.
    """
    import torch

    return int(torch.get_num_threads())


def _initialize_worker_thread_budget(num_threads: int) -> None:
    """Apply one CPU thread budget at process-worker startup."""
    value = str(max(1, int(num_threads)))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    # The initializer runs first in a fresh spawn/forkserver worker, before
    # any parallel Torch work; a budget that cannot be applied is an error
    # that fails the pool (BrokenProcessPool), never a silently unbounded
    # worker.
    import torch

    torch.set_num_threads(int(value))
    torch.set_num_interop_threads(1)


def _worker_runtime_probe() -> Dict[str, int]:
    """Return real process identity and active Torch thread budget."""
    return {
        "pid": os.getpid(),
        "torch_threads": _get_torch_thread_count(),
    }


@dataclass(frozen=True)
class _StarredCallable:
    """Picklable adapter for starmap operations."""

    func: Callable

    def __call__(self, args_tuple: Tuple) -> Any:
        return self.func(*args_tuple)


@dataclass(frozen=True)
class _FilterCallable:
    """Picklable adapter that retains each predicate input."""

    predicate: Callable

    def __call__(self, item: T) -> Tuple[T, bool]:
        return item, self.predicate(item)


class SimpleLoadBalancer:
    """Simple round-robin load balancer."""
    
    def distribute_work(self, items: List[T], num_workers: int) -> List[List[T]]:
        """Distribute items evenly among workers."""
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if not items:
            return [[] for _ in range(num_workers)]
        
        chunks = [[] for _ in range(num_workers)]
        for i, item in enumerate(items):
            chunks[i % num_workers].append(item)
        
        return chunks


class WeightedLoadBalancer:
    """Load balancer with weighted distribution.

    The weights are required (one finite, non-negative weight per worker
    with a positive sum); a missing or mismatched weight vector is an
    error, never replaced by round-robin distribution.
    """
    
    def __init__(self, weights: List[float]):
        if weights is None:
            raise ValueError("weights are required for weighted load balancing")
        weights = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("weights must be finite and nonnegative")
        if sum(weights) <= 0:
            raise ValueError("weights must have a positive sum")
        self.weights = weights
    
    def distribute_work(self, items: List[T], num_workers: int) -> List[List[T]]:
        """Distribute items based on worker weights."""
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if not items:
            return [[] for _ in range(num_workers)]
        
        if len(self.weights) != num_workers:
            raise ValueError(
                f"weighted load balancing needs one weight per worker: "
                f"{len(self.weights)} weights for {num_workers} workers"
            )
        
        # Normalize weights
        total_weight = sum(self.weights)
        normalized_weights = [w / total_weight for w in self.weights]
        
        # Distribute items proportionally
        chunks = [[] for _ in range(num_workers)]
        items_per_worker = [int(len(items) * w) for w in normalized_weights]
        
        # Adjust for rounding errors
        remaining = len(items) - sum(items_per_worker)
        for i in range(remaining):
            items_per_worker[i] += 1
        
        # Assign items to chunks
        start_idx = 0
        for i, count in enumerate(items_per_worker):
            chunks[i] = items[start_idx:start_idx + count]
            start_idx += count
        
        return chunks


# Chunk-count multiplier per declared item complexity (scheduling heuristic:
# cheaper items are split into more chunks per worker).
_ITEM_COMPLEXITY_MULTIPLIERS = {"low": 8, "medium": 4, "high": 2}


def _require_item_complexity(item_complexity: str) -> None:
    """Reject an undeclared item complexity instead of treating it as medium."""
    if item_complexity not in _ITEM_COMPLEXITY_MULTIPLIERS:
        raise ValueError(
            f"item_complexity must be one of "
            f"{sorted(_ITEM_COMPLEXITY_MULTIPLIERS)}, got {item_complexity!r}"
        )


class ChunkSizeOptimizer:
    """Optimizer for determining optimal chunk sizes."""
    
    def __init__(self):
        self.performance_history = {}
        self._lock = threading.Lock()
    
    def estimate_chunk_size(
        self, 
        total_items: int,
        num_workers: int,
        item_complexity: str = "medium",
        strategy: ChunkStrategy = ChunkStrategy.AUTO
    ) -> int:
        """
        Estimate optimal chunk size for parallel execution.
        
        Parameters:
            total_items : int
                Total number of items to process
            num_workers : int
                Number of parallel workers
            item_complexity : str
                Complexity level ("low", "medium", "high")
            strategy : ChunkStrategy
                Chunking strategy to use
        
        Returns:
            int: Optimal chunk size

        Raises:
            ValueError: invalid work dimensions, an undeclared complexity,
                a strategy that is not a ChunkStrategy, or ADAPTIVE without
                recorded performance for this complexity and worker count
                (ADAPTIVE is never silently replaced by AUTO).
        """
        if not isinstance(strategy, ChunkStrategy):
            raise ValueError(f"strategy must be a ChunkStrategy, got {strategy!r}")
        _require_item_complexity(item_complexity)
        if total_items < 0:
            raise ValueError("total_items must be nonnegative")
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")

        if strategy == ChunkStrategy.FIXED:
            return max(1, total_items // (num_workers * 4))
        
        elif strategy == ChunkStrategy.AUTO:
            # Base chunk size on complexity and worker count
            multiplier = _ITEM_COMPLEXITY_MULTIPLIERS[item_complexity]
            
            chunk_size = max(1, total_items // (num_workers * multiplier))
            
            # Ensure reasonable bounds
            min_chunk = 1
            max_chunk = max(100, total_items // num_workers)
            
            return max(min_chunk, min(chunk_size, max_chunk))
        
        elif strategy == ChunkStrategy.ADAPTIVE:
            # ADAPTIVE is defined only by recorded performance; without a
            # record it has nothing to adapt and raises (no AUTO substitute).
            with self._lock:
                key = f"{item_complexity}_{num_workers}"
                if key not in self.performance_history:
                    raise ValueError(
                        "adaptive chunk sizing has no recorded performance for "
                        f"complexity '{item_complexity}' with {num_workers} "
                        "workers; record it first (enable_profiling with a "
                        "non-adaptive strategy) or choose another ChunkStrategy"
                    )
                history = self.performance_history[key]
                historical_optimal = max(
                    history,
                    key=lambda size: (
                        history[size]["throughput"],
                        -history[size]["execution_time"],
                    ),
                )
                # Adjust based on current workload
                scale_factor = total_items / 1000  # Normalize to 1000 items
                return max(1, int(historical_optimal * math.sqrt(scale_factor)))
        
        # ChunkStrategy.DYNAMIC: start with smaller chunks for dynamic
        # adjustment (the enum has no other member).
        return max(1, total_items // (num_workers * 8))
    
    def record_performance(
        self,
        item_complexity: str,
        num_workers: int,
        chunk_size: int,
        execution_time: float,
        throughput: float
    ):
        """Record performance data for future optimization."""
        with self._lock:
            key = f"{item_complexity}_{num_workers}"
            if key not in self.performance_history:
                self.performance_history[key] = {}
            
            self.performance_history[key][chunk_size] = {
                'execution_time': execution_time,
                'throughput': throughput,
                'timestamp': time.time()
            }


class ParallelExecutor:
    """
    High-level parallel execution engine for scientific computing.
    
    Provides unified interface for both thread-based and process-based
    parallel execution with automatic optimization and error handling.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel executor.
        
        Parameters:
            config : Optional[ParallelConfig]
                Configuration for parallel execution
        """
        self.config = config or ParallelConfig()
        self.chunk_optimizer = ChunkSizeOptimizer()
        
        # System information
        self.cpu_count = mp.cpu_count()
        self.available_memory = self._get_available_memory()
        
        # Execution state
        self._active_executors = {}
        self._execution_lock = _EXECUTION_STATS_LOCK
        self._current_torch_threads_per_worker = 1
        self._last_execution_metadata: Dict[str, Any] = {
            "requested_backend": self.config.backend.value,
            "effective_backend": "none",
            "requested_workers": self.config.max_workers,
            "effective_workers": 0,
            "torch_threads_per_worker": 0,
            "fallback": False,
            "expected_count": 0,
            "actual_count": 0,
            "worker_pids": [],
            "worker_torch_threads": [],
        }
    
    def _get_available_memory(self) -> float:
        """Get available system memory in MB.

        psutil is a required dependency; a failing probe propagates and is
        never reported as a default amount of memory.
        """
        return psutil.virtual_memory().available / (1024 * 1024)
    
    def _select_backend(self, func: Callable, items: List[Any]) -> ParallelBackend:
        """
        Automatically select optimal parallel backend.
        
        Parameters:
            func : Callable
                Function to execute
            items : List[Any]
                Items to process
        
        Returns:
            ParallelBackend: Selected backend
        """
        if self.config.backend != ParallelBackend.AUTO:
            return self.config.backend
        
        # Heuristics for backend selection
        # Use processes for CPU-intensive tasks with large datasets
        # Use threads for I/O-bound tasks or small datasets
        
        item_count = len(items)
        
        # Check if function is likely CPU-bound
        func_name = getattr(func, '__name__', str(func))
        cpu_bound_keywords = ['compute', 'calculate', 'process', 'simulate', 'optimize']
        is_likely_cpu_bound = any(keyword in func_name.lower() for keyword in cpu_bound_keywords)
        
        # Decision logic
        if is_likely_cpu_bound and item_count > 10:
            return ParallelBackend.PROCESSES
        elif item_count > 100:
            return ParallelBackend.THREADS  # Threads for large I/O-bound tasks
        else:
            return ParallelBackend.THREADS  # Default to threads for safety
    
    def _determine_worker_count(self, backend: ParallelBackend) -> int:
        """
        Determine optimal number of workers.
        
        Parameters:
            backend : ParallelBackend
                Selected parallel backend
        
        Returns:
            int: Number of workers to use
        """
        if self.config.max_workers is not None:
            if self.config.max_workers <= 0:
                raise ValueError("max_workers must be positive")
            requested_workers = min(self.config.max_workers, self.cpu_count)
        elif backend == ParallelBackend.PROCESSES:
            # For CPU-bound tasks, use all available cores
            requested_workers = self.cpu_count
        else:
            # For I/O-bound tasks, can use more threads than cores
            requested_workers = min(self.cpu_count * 2, 32)

        if (
            backend == ParallelBackend.THREADS
            and self.config.enforce_torch_thread_budget
        ):
            actual_torch_threads = max(1, _get_torch_thread_count())
            safe_workers = max(1, self.cpu_count // actual_torch_threads)
            return min(requested_workers, safe_workers)
        return requested_workers

    def _resolve_torch_threads_per_worker(self, num_workers: int) -> int:
        """Resolve a bounded intra-op budget for each outer worker."""
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        configured = self.config.torch_threads_per_worker
        if configured is not None:
            if configured <= 0:
                raise ValueError("torch_threads_per_worker must be positive")
            return min(int(configured), self.cpu_count)
        return max(1, self.cpu_count // num_workers)
    
    def _chunk_items(
        self, 
        items: List[T], 
        num_workers: int,
        complexity: str = "medium"
    ) -> List[List[T]]:
        """
        Chunk items for parallel processing.
        
        Parameters:
            items : List[T]
                Items to chunk
            num_workers : int
                Number of workers
            complexity : str
                Complexity estimate for chunk sizing
        
        Returns:
            List[List[T]]: Chunked items
        """
        if not items:
            return []
        
        if self.config.chunk_size is not None:
            # Use fixed chunk size
            chunk_size = self.config.chunk_size
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Estimate optimal chunk size
        chunk_size = self.chunk_optimizer.estimate_chunk_size(
            len(items), num_workers, complexity, self.config.chunk_strategy
        )
        
        if self.config.chunk_strategy == ChunkStrategy.DYNAMIC:
            # For dynamic chunking, create more smaller chunks
            chunk_size = max(1, chunk_size // 2)
        
        # Create chunks
        if chunk_size == 1:
            # Individual items
            return [[item] for item in items]
        else:
            # Fixed-size chunks
            return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    def _execute_with_threads(
        self, 
        func: Callable,
        chunks: List[List[T]], 
        num_workers: int
    ) -> ExecutionResult:
        """Execute using thread pool."""
        start_time = time.time()
        chunk_results_by_id = {}
        errors = []
        worker_stats = {
            'backend': 'threads',
            'num_workers': num_workers,
            'worker_pids': set(),
            'worker_torch_threads': set(),
        }
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks
            future_to_chunk = {}
            member_start = 0
            for i, chunk in enumerate(chunks):
                if chunk:  # Skip empty chunks
                    future = executor.submit(
                        self._process_chunk,
                        func,
                        chunk,
                        i,
                        member_start,
                    )
                    future_to_chunk[future] = i
                member_start += len(chunk)
            
            # Collect results with optional progress tracking
            completed_count = 0
            try:
                for future in as_completed(
                    future_to_chunk,
                    timeout=self.config.timeout,
                ):
                    try:
                        chunk_execution = future.result()
                        chunk_id = future_to_chunk[future]
                        chunk_results_by_id[chunk_id] = chunk_execution.results
                        worker_stats['worker_pids'].add(chunk_execution.pid)
                        worker_stats['worker_torch_threads'].add(
                            chunk_execution.torch_threads
                        )
                        completed_count += 1

                        if self.config.progress_callback:
                            self.config.progress_callback(
                                completed_count,
                                len(future_to_chunk),
                            )

                    except Exception as e:  # noqa: BLE001 - preserve worker failure context
                        chunk_id = future_to_chunk[future]
                        error = RuntimeError(f"chunk {chunk_id} failed: {e}")
                        error.__cause__ = e
                        errors.append(error)
                        logger.warning(
                            "Chunk %s processing failed: %s",
                            chunk_id,
                            e,
                        )
            except FuturesTimeoutError as exc:
                unfinished_chunks = sorted(
                    chunk_id
                    for future, chunk_id in future_to_chunk.items()
                    if not future.done()
                )
                cancelled_chunks = sorted(
                    future_to_chunk[future]
                    for future in future_to_chunk
                    if not future.done() and future.cancel()
                )
                worker_stats['timed_out'] = True
                worker_stats['unfinished_chunks'] = unfinished_chunks
                worker_stats['cancelled_chunks'] = cancelled_chunks
                error = RuntimeError(
                    "thread execution timed out; unfinished chunks="
                    f"{unfinished_chunks}; cancelled chunks={cancelled_chunks}"
                )
                error.__cause__ = exc
                errors.append(error)
        
        results = [
            item
            for chunk_id in sorted(chunk_results_by_id)
            for item in chunk_results_by_id[chunk_id]
        ]
        execution_time = time.time() - start_time
        success_rate = completed_count / len(future_to_chunk)
        
        worker_stats['worker_pids'] = sorted(worker_stats['worker_pids'])
        worker_stats['worker_torch_threads'] = sorted(
            worker_stats['worker_torch_threads']
        )
        return ExecutionResult(
            results=results,
            execution_time=execution_time,
            worker_stats=worker_stats,
            errors=errors,
            success_rate=success_rate
        )
    
    def _execute_with_processes(
        self, 
        func: Callable,
        chunks: List[List[T]], 
        num_workers: int
    ) -> ExecutionResult:
        """Execute using process pool."""
        start_time = time.time()
        chunk_results_by_id = {}
        errors = []
        worker_stats = {
            'backend': 'processes',
            'num_workers': num_workers,
            'worker_pids': set(),
            'worker_torch_threads': set(),
        }

        if self.config.timeout is not None and not callable(
            getattr(ProcessPoolExecutor, "terminate_workers", None)
        ):
            raise RuntimeError(
                "Process timeout cancellation requires the public "
                "ProcessPoolExecutor.terminate_workers API from Python 3.14."
            )

        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_initialize_worker_thread_budget,
            initargs=(self._current_torch_threads_per_worker,),
        )
        terminated = False
        future_to_chunk = {}
        completed_count = 0
        try:
            # Submit all chunks
            member_start = 0
            for i, chunk in enumerate(chunks):
                if chunk:  # Skip empty chunks
                    future = executor.submit(
                        self._process_chunk_static,
                        func,
                        chunk,
                        i,
                        member_start,
                    )
                    future_to_chunk[future] = i
                member_start += len(chunk)
            
            # Collect results
            try:
                for future in as_completed(
                    future_to_chunk,
                    timeout=self.config.timeout,
                ):
                    try:
                        chunk_execution = future.result()
                        chunk_id = future_to_chunk[future]
                        chunk_results_by_id[chunk_id] = chunk_execution.results
                        worker_stats['worker_pids'].add(chunk_execution.pid)
                        worker_stats['worker_torch_threads'].add(
                            chunk_execution.torch_threads
                        )
                        completed_count += 1

                        if self.config.progress_callback:
                            self.config.progress_callback(
                                completed_count,
                                len(future_to_chunk),
                            )

                    except Exception as e:  # noqa: BLE001 - preserve worker failure context
                        chunk_id = future_to_chunk[future]
                        error = RuntimeError(f"chunk {chunk_id} failed: {e}")
                        error.__cause__ = e
                        errors.append(error)
                        logger.warning(
                            "Chunk %s processing failed: %s",
                            chunk_id,
                            e,
                        )
            except FuturesTimeoutError as exc:
                unfinished_chunks = sorted(
                    chunk_id
                    for future, chunk_id in future_to_chunk.items()
                    if not future.done()
                )
                for future in future_to_chunk:
                    if not future.done():
                        future.cancel()
                worker_stats['timed_out'] = True
                worker_stats['unfinished_chunks'] = unfinished_chunks
                error = RuntimeError(
                    "process execution timed out; unfinished chunks="
                    f"{unfinished_chunks}"
                )
                error.__cause__ = exc
                errors.append(error)
                executor.terminate_workers()
                terminated = True
        finally:
            if not terminated:
                executor.shutdown(wait=True, cancel_futures=False)
        
        results = [
            item
            for chunk_id in sorted(chunk_results_by_id)
            for item in chunk_results_by_id[chunk_id]
        ]
        execution_time = time.time() - start_time
        success_rate = completed_count / len(future_to_chunk)
        
        worker_stats['worker_pids'] = sorted(worker_stats['worker_pids'])
        worker_stats['worker_torch_threads'] = sorted(
            worker_stats['worker_torch_threads']
        )
        return ExecutionResult(
            results=results,
            execution_time=execution_time,
            worker_stats=worker_stats,
            errors=errors,
            success_rate=success_rate
        )
    
    def _process_chunk(
        self,
        func: Callable,
        chunk: List[T],
        chunk_id: int,
        member_start: int,
    ) -> _ChunkExecution:
        """Process a chunk of items (for threads)."""
        results = []
        for offset, item in enumerate(chunk):
            try:
                result = func(item)
                results.append(result)
            except Exception as e:  # noqa: BLE001 - preserve member failure context
                member_id = member_start + offset
                logger.error(
                    "Error processing member %s in chunk %s: %s",
                    member_id,
                    chunk_id,
                    e,
                )
                raise RuntimeError(
                    f"member {member_id} in chunk {chunk_id} failed: {e}"
                ) from e
        return _ChunkExecution(
            results=results,
            pid=os.getpid(),
            torch_threads=_get_torch_thread_count(),
        )
    
    @staticmethod
    def _process_chunk_static(
        func: Callable,
        chunk: List[T],
        chunk_id: int,
        member_start: int,
    ) -> _ChunkExecution:
        """Static method for process-based chunk processing."""
        results = []
        for offset, item in enumerate(chunk):
            try:
                result = func(item)
                results.append(result)
            except Exception as e:  # noqa: BLE001 - preserve member failure context
                member_id = member_start + offset
                logger.error(
                    "Error processing member %s in chunk %s: %s",
                    member_id,
                    chunk_id,
                    e,
                )
                raise RuntimeError(
                    f"member {member_id} in chunk {chunk_id} failed: {e}"
                ) from e
        return _ChunkExecution(
            results=results,
            pid=os.getpid(),
            torch_threads=_get_torch_thread_count(),
        )
    
    def parallel_map(
        self, 
        func: Callable[[T], R], 
        items: List[T],
        complexity: str = "medium"
    ) -> List[R]:
        """
        Parallel map operation with automatic optimization.
        
        Parameters:
            func : Callable[[T], R]
                Function to apply to each item
            items : List[T]
                Items to process
            complexity : str
                Computational complexity hint ("low", "medium", "high")
        
        Returns:
            List[R]: Results in same order as input items
        """
        _require_item_complexity(complexity)
        requested_backend = self.config.backend.value
        requested_workers = self.config.max_workers
        if not items:
            self._last_execution_metadata = {
                "requested_backend": requested_backend,
                "effective_backend": "none",
                "requested_workers": requested_workers,
                "effective_workers": 0,
                "torch_threads_per_worker": 0,
                "fallback": False,
                "expected_count": 0,
                "actual_count": 0,
                "worker_pids": [],
                "worker_torch_threads": [],
            }
            return []

        if self.config.allow_sequential and len(items) < 4:
            results = [func(item) for item in items]
            self._last_execution_metadata = {
                "requested_backend": requested_backend,
                "effective_backend": "sequential",
                "requested_workers": requested_workers,
                "effective_workers": 1,
                "torch_threads_per_worker": self.cpu_count,
                "fallback": False,
                "expected_count": len(items),
                "actual_count": len(results),
                "worker_pids": [os.getpid()],
                "worker_torch_threads": [_get_torch_thread_count()],
            }
            return results
        
        # Select backend and configure execution
        backend = self._select_backend(func, items)
        num_workers = min(
            self._determine_worker_count(backend),
            len(items),
        )
        if backend == ParallelBackend.THREADS:
            torch_threads_per_worker = max(1, _get_torch_thread_count())
        else:
            torch_threads_per_worker = self._resolve_torch_threads_per_worker(
                num_workers
            )
        self._current_torch_threads_per_worker = torch_threads_per_worker
        
        # Create chunks
        chunks = self._chunk_items(items, num_workers, complexity)

        self._last_execution_metadata = {
            "requested_backend": requested_backend,
            "effective_backend": backend.value,
            "requested_workers": requested_workers,
            "effective_workers": num_workers,
            "torch_threads_per_worker": torch_threads_per_worker,
            "fallback": False,
            "expected_count": len(items),
            "actual_count": 0,
            "worker_pids": [],
            "worker_torch_threads": [],
        }
        
        # Execute based on backend
        if backend == ParallelBackend.PROCESSES:
            result = self._execute_with_processes(func, chunks, num_workers)
        else:
            result = self._execute_with_threads(func, chunks, num_workers)

        self._last_execution_metadata = {
            "requested_backend": requested_backend,
            "effective_backend": backend.value,
            "requested_workers": requested_workers,
            "effective_workers": num_workers,
            "torch_threads_per_worker": torch_threads_per_worker,
            "fallback": False,
            "expected_count": len(items),
            "actual_count": len(result.results),
            "worker_pids": list(result.worker_stats.get("worker_pids", [])),
            "worker_torch_threads": list(
                result.worker_stats.get("worker_torch_threads", [])
            ),
        }
        
        # Update statistics
        global _EXECUTION_STATS
        with self._execution_lock:
            _EXECUTION_STATS['tasks_submitted'] += len(items)
            _EXECUTION_STATS['tasks_completed'] += len(result.results)
            _EXECUTION_STATS['tasks_failed'] += len(result.errors)
            _EXECUTION_STATS['total_execution_time'] += result.execution_time
            # No parallel efficiency is reported: no sequential baseline is
            # measured, and an assumed per-item time would be a made-up figure.
        
        # Log performance information
        if self.config.enable_profiling:
            timing_resolution = time.get_clock_info("time").resolution
            measured_time = max(result.execution_time, timing_resolution)
            throughput = len(result.results) / measured_time
            logger.info(
                f"Parallel execution completed: {len(result.results)} items, "
                f"{result.execution_time:.2f}s, {throughput:.1f} items/s, "
                f"success_rate={result.success_rate:.2%}"
            )
            
            # Record performance for optimization
            self.chunk_optimizer.record_performance(
                complexity, num_workers, 
                len(chunks[0]) if chunks else 1,
                result.execution_time, throughput
            )
        
        if result.errors:
            # Every chunk failure was collected with its cause; the aggregate
            # is raised and chained to the first failure.
            error_messages = "; ".join(str(error) for error in result.errors)
            raise RuntimeError(
                f"Parallel execution failed: {error_messages}"
            ) from result.errors[0]
        if len(result.results) != len(items):
            raise RuntimeError(
                "Parallel execution returned an incomplete result set: "
                f"expected {len(items)}, got {len(result.results)}."
            )
        
        return result.results
    
    def parallel_starmap(
        self,
        func: Callable,
        items: List[Tuple],
        complexity: str = "medium"
    ) -> List[Any]:
        """
        Parallel starmap operation (unpacks tuples as arguments).
        
        Parameters:
            func : Callable
                Function to apply (should accept unpacked tuple arguments)
            items : List[Tuple]
                List of tuples to unpack as function arguments
            complexity : str
                Computational complexity hint
        
        Returns:
            List[Any]: Results in same order as input
        """
        return self.parallel_map(_StarredCallable(func), items, complexity)
    
    def parallel_filter(
        self,
        predicate: Callable[[T], bool],
        items: List[T],
        complexity: str = "low"
    ) -> List[T]:
        """
        Parallel filter operation.
        
        Parameters:
            predicate : Callable[[T], bool]
                Predicate function
            items : List[T]
                Items to filter
            complexity : str
                Computational complexity hint
        
        Returns:
            List[T]: Filtered items
        """
        results = self.parallel_map(_FilterCallable(predicate), items, complexity)
        return [item for item, keep in results if keep]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._execution_lock:
            stats = _EXECUTION_STATS.copy()
            stats['available_workers'] = self.cpu_count
            stats['available_memory_mb'] = self.available_memory
            return stats
    
    def get_last_execution_metadata(self) -> Dict[str, Any]:
        """Return requested and effective settings for the latest map call."""
        return dict(self._last_execution_metadata)


# Global executor instance
_default_executor = ParallelExecutor()


def _config_from_public_arguments(
    *,
    backend: str,
    max_workers: Optional[int],
    kwargs: Dict[str, Any],
) -> ParallelConfig:
    """Build one runtime snapshot from canonical or explicit policy."""
    computation_config = kwargs.pop("computation_config", None)
    if computation_config is None:
        return ParallelConfig(
            backend=ParallelBackend(backend),
            max_workers=max_workers,
            **kwargs,
        )

    config = ParallelConfig.from_computation_config(computation_config)
    if backend != "auto":
        config.backend = ParallelBackend(backend)
    if max_workers is not None:
        config.max_workers = max_workers
    for key, value in kwargs.items():
        if not hasattr(config, key):
            raise TypeError(f"Unknown ParallelConfig field: {key}")
        setattr(config, key, value)
    return config


def parallel_map(
    func: Callable[[T], R],
    items: List[T], 
    backend: str = "auto",
    max_workers: Optional[int] = None,
    **kwargs
) -> List[R]:
    """
    Convenience function for parallel map operation.
    
    Parameters:
        func : Callable[[T], R]
            Function to apply to each item
        items : List[T]
            Items to process
        backend : str
            Parallel backend ("auto", "threads", "processes")
        max_workers : Optional[int]
            Maximum number of workers
        **kwargs : Additional configuration options
    
    Returns:
        List[R]: Results in same order as input items
    """
    complexity = kwargs.pop("complexity", "medium")
    config = _config_from_public_arguments(
        backend=backend,
        max_workers=max_workers,
        kwargs=kwargs,
    )
    
    executor = ParallelExecutor(config)
    return executor.parallel_map(func, items, complexity)


def parallel_starmap(
    func: Callable,
    items: List[Tuple],
    backend: str = "auto", 
    max_workers: Optional[int] = None,
    **kwargs
) -> List[Any]:
    """
    Convenience function for parallel starmap operation.
    
    Parameters:
        func : Callable
            Function to apply (should accept unpacked arguments)
        items : List[Tuple]
            Tuples to unpack as function arguments
        backend : str
            Parallel backend
        max_workers : Optional[int]
            Maximum number of workers
        **kwargs : Additional configuration options
    
    Returns:
        List[Any]: Results in same order as input
    """
    complexity = kwargs.pop("complexity", "medium")
    config = _config_from_public_arguments(
        backend=backend,
        max_workers=max_workers,
        kwargs=kwargs,
    )
    
    executor = ParallelExecutor(config)
    return executor.parallel_starmap(func, items, complexity)


@contextmanager
def parallel_context(
    backend: str = "auto",
    max_workers: Optional[int] = None,
    **kwargs
):
    """
    Context manager for parallel execution configuration.
    
    Parameters:
        backend : str
            Parallel backend
        max_workers : Optional[int]
            Maximum number of workers
        **kwargs : Additional configuration
    
    Yields:
        ParallelExecutor: Configured executor
    """
    config = _config_from_public_arguments(
        backend=backend,
        max_workers=max_workers,
        kwargs=kwargs,
    )
    
    executor = ParallelExecutor(config)
    try:
        yield executor
    finally:
        # Cleanup if needed
        pass


def get_parallel_stats() -> Dict[str, Any]:
    """Get global parallel execution statistics."""
    return _default_executor.get_stats()


def parallel_healthcheck() -> Dict[str, Any]:
    """
    Perform parallel execution health check.
    
    Returns:
        Dict[str, Any]: Health check results
    """
    status = {
        'parallel_available': True,
        'cpu_count': mp.cpu_count(),
        'available_memory_mb': _default_executor.available_memory,
        'execution_stats': get_parallel_stats(),
        'warnings': [],
        'recommendations': []
    }
    
    # Check system resources
    if status['cpu_count'] <= 1:
        status['warnings'].append("Single-core system - limited parallel benefit")
    
    if status['available_memory_mb'] < 1024:
        status['warnings'].append("Low available memory may limit parallel performance")
        status['recommendations'].append("Consider reducing max_workers or using streaming")
    
    # Check execution statistics
    stats = status['execution_stats']
    if stats['tasks_failed'] > stats['tasks_completed'] * 0.1:
        status['warnings'].append("High task failure rate")
        status['recommendations'].append("Check error handling and resource limits")
    
    return status


# Export main classes and functions
__all__ = [
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
    
    # Global instances
    '_default_executor'
]
