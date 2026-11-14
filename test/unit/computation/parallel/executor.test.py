"""Behavioral tests for the canonical parallel executor."""

from __future__ import annotations

import math
import os
import pickle
import time

import pytest
import psutil
import torch

import relic.computation.parallel.executor as executor_module
from relic.core.config.models import ComputationConfig
from relic.computation import parallel_map as root_parallel_map
from relic.computation.parallel.executor import (
    ChunkSizeOptimizer,
    ChunkStrategy,
    ExecutionResult,
    ParallelBackend,
    ParallelConfig,
    ParallelExecutor,
    SimpleLoadBalancer,
    WeightedLoadBalancer,
    parallel_context,
    parallel_healthcheck,
)
from relic.computation.parallel.executor import (
    parallel_map as module_parallel_map,
)
from relic.computation.parallel.executor import (
    parallel_starmap as module_parallel_starmap,
)
from relic.computation.parallel import (
    parallel_batch_evaluation,
    parallel_parameter_sweep,
)
from test.fixtures.parallel_workers import parameter_record


def _add(left: int, right: int) -> int:
    return left + right


def _is_even(value: int) -> bool:
    return value % 2 == 0


@pytest.mark.parametrize(
    ("chunk_size", "expected_strategy"),
    [
        ("auto", ChunkStrategy.AUTO),
        ("small", ChunkStrategy.DYNAMIC),
        ("medium", ChunkStrategy.AUTO),
        ("large", ChunkStrategy.FIXED),
        ("adaptive", ChunkStrategy.ADAPTIVE),
    ],
)
def test_parallel_config_consumes_computation_config(
    chunk_size: str,
    expected_strategy: ChunkStrategy,
) -> None:
    computation_config = ComputationConfig(
        parallel_backend="processes",
        num_workers=4,
        chunk_size=chunk_size,
        enable_profiling=True,
    )

    runtime_config = ParallelConfig.from_computation_config(computation_config)

    assert runtime_config.backend is ParallelBackend.PROCESSES
    assert runtime_config.max_workers == 4
    assert runtime_config.chunk_strategy is expected_strategy
    assert runtime_config.enable_profiling is True
    assert runtime_config.allow_sequential is False


@pytest.mark.parametrize("backend", ["dask", "ray"])
def test_parallel_config_rejects_unwired_computation_backends(backend: str) -> None:
    with pytest.raises(NotImplementedError, match=backend):
        ParallelConfig.from_computation_config(
            ComputationConfig(parallel_backend=backend)
        )


def test_module_parallel_map_consumes_nested_computation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_parallel_map(self, func, items, complexity="medium"):
        captured["config"] = self.config
        return [func(item) for item in items]

    monkeypatch.setattr(ParallelExecutor, "parallel_map", fake_parallel_map)

    result = module_parallel_map(
        str,
        [1, 2],
        computation_config={
            "computation": {
                "parallel_backend": "threads",
                "num_workers": 2,
                "chunk_size": "small",
                "enable_profiling": True,
            }
        },
    )

    assert result == ["1", "2"]
    assert captured["config"].backend is ParallelBackend.THREADS
    assert captured["config"].max_workers == 2
    assert captured["config"].chunk_strategy is ChunkStrategy.DYNAMIC
    assert captured["config"].enable_profiling is True


@pytest.fixture(autouse=True)
def _reset_execution_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor_module,
        "_EXECUTION_STATS",
        {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
        },
    )


def test_simple_load_balancer_distributes_all_items_round_robin() -> None:
    chunks = SimpleLoadBalancer().distribute_work(list(range(7)), num_workers=3)

    assert chunks == [[0, 3, 6], [1, 4], [2, 5]]


def test_simple_load_balancer_rejects_nonpositive_worker_count() -> None:
    with pytest.raises(ValueError, match="num_workers must be positive"):
        SimpleLoadBalancer().distribute_work([1], num_workers=0)


def test_load_balancers_return_one_empty_chunk_per_worker() -> None:
    assert SimpleLoadBalancer().distribute_work([], num_workers=2) == [[], []]
    assert WeightedLoadBalancer([1.0, 1.0]).distribute_work(
        [],
        num_workers=2,
    ) == [[], []]


def test_weighted_load_balancer_respects_worker_capacity() -> None:
    chunks = WeightedLoadBalancer([3.0, 1.0]).distribute_work(
        list(range(8)),
        num_workers=2,
    )

    assert chunks == [list(range(6)), list(range(6, 8))]


def test_weighted_load_balancer_rejects_a_weight_count_mismatch() -> None:
    """One weight per worker is required; round-robin is never substituted."""
    with pytest.raises(ValueError, match="one weight per worker"):
        WeightedLoadBalancer([1.0]).distribute_work(
            list(range(5)),
            num_workers=2,
        )


def test_weighted_load_balancer_requires_explicit_weights() -> None:
    with pytest.raises(TypeError):
        WeightedLoadBalancer()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="weights"):
        WeightedLoadBalancer(None)  # type: ignore[arg-type]


def test_weighted_load_balancer_rejects_nonpositive_worker_count() -> None:
    with pytest.raises(ValueError, match="num_workers must be positive"):
        WeightedLoadBalancer([1.0]).distribute_work([1], num_workers=0)


@pytest.mark.parametrize(
    "weights",
    [
        [0.0, 0.0],
        [1.0, -1.0],
        [1.0, float("nan")],
    ],
)
def test_weighted_load_balancer_rejects_invalid_weights(
    weights: list[float],
) -> None:
    with pytest.raises(ValueError, match="weights"):
        WeightedLoadBalancer(weights).distribute_work([1, 2], num_workers=2)


@pytest.mark.parametrize(
    "strategy",
    [strategy for strategy in ChunkStrategy if strategy is not ChunkStrategy.ADAPTIVE],
)
def test_chunk_size_optimizer_always_returns_positive_size(
    strategy: ChunkStrategy,
) -> None:
    size = ChunkSizeOptimizer().estimate_chunk_size(
        total_items=100,
        num_workers=4,
        strategy=strategy,
    )

    assert isinstance(size, int)
    assert size >= 1


def test_chunk_size_optimizer_rejects_invalid_work_dimensions() -> None:
    optimizer = ChunkSizeOptimizer()

    with pytest.raises(ValueError, match="total_items must be nonnegative"):
        optimizer.estimate_chunk_size(total_items=-1, num_workers=2)
    with pytest.raises(ValueError, match="num_workers must be positive"):
        optimizer.estimate_chunk_size(total_items=10, num_workers=0)


def test_adaptive_chunking_without_recorded_performance_is_an_error() -> None:
    """ADAPTIVE needs recorded history; it is never silently replaced by AUTO."""
    with pytest.raises(ValueError, match="no recorded performance"):
        ChunkSizeOptimizer().estimate_chunk_size(
            total_items=100,
            num_workers=4,
            strategy=ChunkStrategy.ADAPTIVE,
        )


def test_adaptive_executor_without_history_fails_before_execution() -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=2,
            chunk_strategy=ChunkStrategy.ADAPTIVE,
        )
    )

    with pytest.raises(ValueError, match="no recorded performance"):
        executor.parallel_map(lambda value: value, list(range(8)))


def test_chunk_size_optimizer_rejects_unknown_complexity_and_strategy() -> None:
    optimizer = ChunkSizeOptimizer()

    with pytest.raises(ValueError, match="item_complexity"):
        optimizer.estimate_chunk_size(
            total_items=100, num_workers=4, item_complexity="extreme"
        )
    with pytest.raises(ValueError, match="strategy"):
        optimizer.estimate_chunk_size(
            total_items=100, num_workers=4, strategy="auto"  # type: ignore[arg-type]
        )


def test_parallel_map_rejects_unknown_complexity() -> None:
    executor = ParallelExecutor(
        ParallelConfig(backend=ParallelBackend.THREADS, max_workers=2)
    )

    with pytest.raises(ValueError, match="item_complexity"):
        executor.parallel_map(lambda value: value, list(range(8)), complexity="extreme")


def test_chunk_size_optimizer_uses_best_recorded_adaptive_size() -> None:
    optimizer = ChunkSizeOptimizer()
    optimizer.record_performance("medium", 4, 4, 2.0, 10.0)
    optimizer.record_performance("medium", 4, 8, 1.0, 20.0)

    assert optimizer.estimate_chunk_size(
        total_items=1000,
        num_workers=4,
        strategy=ChunkStrategy.ADAPTIVE,
    ) == 8
    assert optimizer.estimate_chunk_size(
        total_items=4000,
        num_workers=4,
        strategy=ChunkStrategy.ADAPTIVE,
    ) == 16


def test_memory_probe_failure_propagates_instead_of_a_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """psutil is a required dependency; a failed probe is not 8192 MB."""
    monkeypatch.setattr(
        executor_module.psutil,
        "virtual_memory",
        lambda: (_ for _ in ()).throw(OSError("unavailable")),
    )

    with pytest.raises(OSError, match="unavailable"):
        ParallelExecutor()


def test_torch_thread_probe_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_probe() -> int:
        raise RuntimeError("thread pool unavailable")

    monkeypatch.setattr(torch, "get_num_threads", failing_probe)

    with pytest.raises(RuntimeError, match="thread pool unavailable"):
        executor_module._get_torch_thread_count()


def test_worker_thread_budget_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker whose thread budget cannot be applied must not start silently."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)

    def failing_budget(_threads: int) -> None:
        raise RuntimeError("cannot set threads")

    monkeypatch.setattr(torch, "set_num_threads", failing_budget)

    with pytest.raises(RuntimeError, match="cannot set threads"):
        executor_module._initialize_worker_thread_budget(2)


def test_parallel_config_has_no_unenforced_memory_limit() -> None:
    """A memory limit that no code enforces is not accepted as configuration."""
    with pytest.raises(TypeError):
        ParallelConfig(memory_limit_mb=100)  # type: ignore[call-arg]


def test_executor_has_no_unused_load_balancer_hook() -> None:
    """Chunking never consulted a load balancer; the hook is not offered."""
    executor = ParallelExecutor()

    assert not hasattr(executor, "load_balancer")
    assert not hasattr(executor, "set_load_balancer")


def test_executor_resource_and_backend_selection_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module.mp, "cpu_count", lambda: 4)
    executor = ParallelExecutor()

    def compute_value(value: int) -> int:
        return value

    assert executor.available_memory > 0.0
    assert executor._select_backend(compute_value, list(range(11))) is ParallelBackend.PROCESSES
    assert executor._select_backend(str, list(range(101))) is ParallelBackend.THREADS
    assert executor._select_backend(str, [1]) is ParallelBackend.THREADS
    assert executor._determine_worker_count(ParallelBackend.PROCESSES) == 4
    assert executor._determine_worker_count(ParallelBackend.THREADS) == max(
        1,
        4 // torch.get_num_threads(),
    )


def test_executors_share_the_global_statistics_lock() -> None:
    first = ParallelExecutor()
    second = ParallelExecutor()

    assert first._execution_lock is second._execution_lock


def test_dynamic_chunking_preserves_all_items() -> None:
    executor = ParallelExecutor(
        ParallelConfig(chunk_strategy=ChunkStrategy.DYNAMIC)
    )

    chunks = executor._chunk_items(list(range(20)), num_workers=2)

    assert [item for chunk in chunks for item in chunk] == list(range(20))
    assert executor._chunk_items([], num_workers=2) == []


def test_thread_parallel_map_preserves_input_order() -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=3,
            chunk_size=2,
        )
    )

    def delayed_square(value: int) -> int:
        time.sleep((6 - value) * 0.002)
        return value * value

    values = list(range(6))
    result = executor.parallel_map(delayed_square, values)

    assert result == [value * value for value in values]


def test_parallel_map_fast_paths_do_not_create_a_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = ParallelExecutor(ParallelConfig(allow_sequential=True))
    monkeypatch.setattr(
        executor,
        "_execute_with_threads",
        lambda *_args, **_kwargs: pytest.fail("thread pool should not be used"),
    )
    monkeypatch.setattr(
        executor,
        "_execute_with_processes",
        lambda *_args, **_kwargs: pytest.fail("process pool should not be used"),
    )

    assert executor.parallel_map(lambda value: value * 2, []) == []
    assert executor.parallel_map(lambda value: value * 2, [3]) == [6]
    assert executor.parallel_map(lambda value: value * 2, [1, 2, 3]) == [2, 4, 6]


def test_parallel_map_fails_closed_with_chunk_context() -> None:
    progress: list[tuple[int, int]] = []
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=2,
            chunk_size=1,
            progress_callback=lambda complete, total: progress.append((complete, total)),
        )
    )

    def fail_on_two(value: int) -> int:
        if value == 2:
            raise ValueError("bad item")
        return value

    with pytest.raises(RuntimeError, match=r"chunk 2.*member 2.*bad item") as excinfo:
        executor.parallel_map(fail_on_two, list(range(5)))

    # The aggregate keeps the member failure in its cause chain.
    cause = excinfo.value.__cause__
    while cause is not None and not isinstance(cause, ValueError):
        cause = cause.__cause__
    assert isinstance(cause, ValueError) and str(cause) == "bad item"
    assert progress[-1] == (4, 5)
    assert executor.get_stats()["tasks_failed"] == 1
    metadata = executor.get_last_execution_metadata()
    assert metadata["requested_backend"] == "threads"
    assert metadata["effective_backend"] == "threads"
    assert metadata["fallback"] is False
    assert metadata["expected_count"] == 5
    assert metadata["actual_count"] == 4


def test_parallel_map_wraps_arbitrary_member_exception() -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=2,
            chunk_size=1,
        )
    )

    def fail_with_zero_division(value: int) -> float:
        if value == 1:
            return 1 / 0
        return float(value)

    with pytest.raises(RuntimeError, match=r"chunk 1.*member 1.*division by zero"):
        executor.parallel_map(fail_with_zero_division, list(range(4)))


def test_small_workload_uses_pool_unless_sequential_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=1,
            allow_sequential=False,
        )
    )
    called = False

    def execute_threads(func, chunks, num_workers):
        nonlocal called
        called = True
        results = [func(item) for chunk in chunks for item in chunk]
        return ExecutionResult(
            results=results,
            execution_time=0.1,
            worker_stats={
                "backend": "threads",
                "num_workers": num_workers,
                "worker_pids": [os.getpid()],
                "worker_torch_threads": [1],
            },
            errors=[],
            success_rate=1.0,
        )

    monkeypatch.setattr(executor, "_execute_with_threads", execute_threads)

    assert executor.parallel_map(str, [1, 2, 3]) == ["1", "2", "3"]
    assert called is True


def test_effective_workers_do_not_exceed_input_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module, "_get_torch_thread_count", lambda: 1)
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=8,
            chunk_size=1,
        )
    )

    def execute_threads(func, chunks, num_workers):
        assert num_workers == 2
        return ExecutionResult(
            results=[func(item) for chunk in chunks for item in chunk],
            execution_time=0.1,
            worker_stats={
                "backend": "threads",
                "num_workers": num_workers,
                "worker_pids": [os.getpid()],
                "worker_torch_threads": [torch.get_num_threads()],
            },
            errors=[],
            success_rate=1.0,
        )

    monkeypatch.setattr(executor, "_execute_with_threads", execute_threads)

    assert executor.parallel_map(str, [1, 2]) == ["1", "2"]
    assert executor.get_last_execution_metadata()["effective_workers"] == 2


def test_thread_backend_does_not_mutate_process_global_torch_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_calls: list[int] = []
    monkeypatch.setattr(
        torch,
        "set_num_threads",
        lambda value: set_calls.append(int(value)),
    )
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=2,
            chunk_size=1,
        )
    )

    assert executor.parallel_map(str, list(range(4))) == ["0", "1", "2", "3"]
    assert set_calls == []
    metadata = executor.get_last_execution_metadata()
    expected_workers = max(1, executor.cpu_count // torch.get_num_threads())
    assert metadata["effective_workers"] == min(2, expected_workers)
    assert metadata["torch_threads_per_worker"] == torch.get_num_threads()


def test_thread_backend_requires_explicit_opt_out_for_io_oversubscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module.mp, "cpu_count", lambda: 24)
    monkeypatch.setattr(executor_module, "_get_torch_thread_count", lambda: 24)
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=4,
            chunk_size=1,
            enforce_torch_thread_budget=False,
        )
    )

    assert executor.parallel_map(str, list(range(4))) == ["0", "1", "2", "3"]
    metadata = executor.get_last_execution_metadata()
    assert metadata["effective_workers"] == 4
    assert metadata["torch_threads_per_worker"] == 24


def test_backend_failure_replaces_stale_execution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=2,
            chunk_size=1,
        )
    )
    executor._last_execution_metadata["actual_count"] = 99
    monkeypatch.setattr(
        executor,
        "_execute_with_threads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("pool startup failed")
        ),
    )

    with pytest.raises(RuntimeError, match="pool startup failed"):
        executor.parallel_map(str, list(range(4)))

    metadata = executor.get_last_execution_metadata()
    assert metadata["requested_backend"] == "threads"
    assert metadata["effective_backend"] == "threads"
    assert metadata["expected_count"] == 4
    assert metadata["actual_count"] == 0
    assert metadata["fallback"] is False


def test_thread_timeout_fails_closed_and_records_cancelled_work() -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=2,
            chunk_size=1,
            timeout=0.01,
        )
    )

    def slow_identity(value: int) -> int:
        time.sleep(0.05)
        return value

    with pytest.raises(RuntimeError, match="thread execution timed out"):
        executor.parallel_map(slow_identity, list(range(4)))

    metadata = executor.get_last_execution_metadata()
    assert metadata["expected_count"] == 4
    assert metadata["actual_count"] < 4
    assert metadata["fallback"] is False
    assert executor.get_stats()["tasks_failed"] >= 1


def test_process_timeout_requires_public_worker_termination_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PoolWithoutTermination:
        pass

    monkeypatch.setattr(
        executor_module,
        "ProcessPoolExecutor",
        PoolWithoutTermination,
    )
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.PROCESSES,
            max_workers=2,
            chunk_size=1,
            timeout=0.01,
        )
    )

    with pytest.raises(RuntimeError, match="terminate_workers"):
        executor.parallel_map(str, list(range(4)))

    metadata = executor.get_last_execution_metadata()
    assert metadata["effective_backend"] == "processes"
    assert metadata["expected_count"] == 4
    assert metadata["actual_count"] == 0


def test_parallel_map_records_requested_and_effective_execution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module.mp, "cpu_count", lambda: 4)
    monkeypatch.setattr(executor_module, "_get_torch_thread_count", lambda: 2)
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=2,
            chunk_size=1,
        )
    )

    assert executor.parallel_map(str, list(range(4))) == ["0", "1", "2", "3"]

    metadata = executor.get_last_execution_metadata()
    assert metadata == {
        "requested_backend": "threads",
        "effective_backend": "threads",
        "requested_workers": 2,
        "effective_workers": 2,
        "torch_threads_per_worker": 2,
        "fallback": False,
        "expected_count": 4,
        "actual_count": 4,
        "worker_pids": [os.getpid()],
        "worker_torch_threads": [2],
    }


@pytest.mark.parametrize(
    ("workers", "expected_threads"),
    [(1, 24), (2, 12), (4, 6), (8, 3)],
)
def test_process_worker_budget_divides_available_cpu_threads(
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
    expected_threads: int,
) -> None:
    monkeypatch.setattr(executor_module.mp, "cpu_count", lambda: 24)
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.PROCESSES,
            max_workers=workers,
        )
    )

    assert executor._resolve_torch_threads_per_worker(workers) == expected_threads


@pytest.mark.serial
def test_real_process_backend_reports_worker_pid_and_torch_budget() -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.PROCESSES,
            max_workers=2,
            chunk_size=1,
            torch_threads_per_worker=2,
        )
    )

    records = executor.parallel_starmap(
        executor_module._worker_runtime_probe,
        [(), (), (), ()],
        complexity="high",
    )

    assert len(records) == 4
    worker_pids = {record["pid"] for record in records}
    assert worker_pids
    assert os.getpid() not in worker_pids
    assert all(record["torch_threads"] == 2 for record in records)
    metadata = executor.get_last_execution_metadata()
    assert set(metadata["worker_pids"]) == worker_pids
    assert metadata["worker_torch_threads"] == [2]
    deadline = time.monotonic() + 5.0
    while any(psutil.pid_exists(pid) for pid in worker_pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(psutil.pid_exists(pid) for pid in worker_pids)


def test_parallel_map_handles_zero_duration_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.THREADS,
            max_workers=1,
            chunk_size=1,
            enable_profiling=True,
        )
    )
    monkeypatch.setattr(
        executor,
        "_execute_with_threads",
        lambda *_args, **_kwargs: ExecutionResult(
            results=[0, 1, 2, 3],
            execution_time=0.0,
            worker_stats={"backend": "threads", "num_workers": 1},
            errors=[],
            success_rate=1.0,
        ),
    )

    assert executor.parallel_map(lambda value: value, list(range(4))) == list(range(4))
    history = executor.chunk_optimizer.performance_history["medium_1"]
    assert all(math.isfinite(record["throughput"]) for record in history.values())


@pytest.mark.parametrize("operation", ["starmap", "filter"])
def test_process_adapters_are_picklable(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = ParallelExecutor(
        ParallelConfig(
            backend=ParallelBackend.PROCESSES,
            max_workers=2,
            chunk_size=1,
        )
    )

    def execute_picklable(func, chunks, num_workers):
        pickle.dumps(func)
        return ExecutionResult(
            results=[func(item) for chunk in chunks for item in chunk],
            execution_time=0.1,
            worker_stats={"backend": "processes", "num_workers": num_workers},
            errors=[],
            success_rate=1.0,
        )

    monkeypatch.setattr(executor, "_execute_with_processes", execute_picklable)

    if operation == "starmap":
        assert executor.parallel_starmap(
            _add,
            [(1, 2), (3, 4), (5, 6), (7, 8)],
        ) == [3, 7, 11, 15]
    else:
        assert executor.parallel_filter(_is_even, list(range(4))) == [0, 2]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            ParallelConfig(
                backend=ParallelBackend.THREADS,
                max_workers=0,
            ),
            "max_workers must be positive",
        ),
        (
            ParallelConfig(
                backend=ParallelBackend.THREADS,
                max_workers=1,
                chunk_size=0,
            ),
            "chunk_size must be positive",
        ),
    ],
)
def test_parallel_map_rejects_invalid_execution_dimensions(
    config: ParallelConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ParallelExecutor(config).parallel_map(lambda value: value, list(range(4)))


def test_parallel_filter_and_starmap_share_map_semantics() -> None:
    executor = ParallelExecutor(
        ParallelConfig(backend=ParallelBackend.THREADS, max_workers=2)
    )

    assert executor.parallel_filter(lambda value: value % 2 == 0, list(range(6))) == [
        0,
        2,
        4,
    ]
    assert executor.parallel_starmap(
        lambda left, right: left + right,
        [(1, 2), (3, 4)],
    ) == [3, 7]


def test_module_wrappers_route_complexity_outside_parallel_config() -> None:
    assert module_parallel_map(
        lambda value: value * 2,
        [1, 2, 3],
        backend="threads",
        complexity="high",
    ) == [2, 4, 6]
    assert module_parallel_starmap(
        _add,
        [(1, 2), (3, 4)],
        backend="threads",
        complexity="low",
    ) == [3, 7]


def test_root_parallel_map_is_a_thin_canonical_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def canonical(func, items, **kwargs):
        calls.append(dict(kwargs))
        return [func(item) for item in items]

    monkeypatch.setattr(executor_module, "parallel_map", canonical)

    assert root_parallel_map(
        str,
        [1, 2, 3],
        backend="threads",
        max_workers=2,
        allow_sequential=False,
    ) == ["1", "2", "3"]
    assert calls == [
        {
            "backend": "threads",
            "max_workers": 2,
            "allow_sequential": False,
        }
    ]


def test_root_parallel_map_forwards_computation_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def canonical(func, items, **kwargs):
        calls.append(dict(kwargs))
        return [func(item) for item in items]

    monkeypatch.setattr(executor_module, "parallel_map", canonical)
    computation_config = {
        "computation": {
            "parallel_backend": "threads",
            "num_workers": 2,
            "chunk_size": "auto",
        }
    }

    assert root_parallel_map(
        str,
        [1, 2],
        computation_config=computation_config,
    ) == ["1", "2"]
    assert calls == [
        {
            "backend": "threads",
            "max_workers": None,
            "allow_sequential": False,
            "computation_config": computation_config,
        }
    ]


def test_root_parallel_map_rejects_invalid_backend_without_replay() -> None:
    with pytest.raises(ValueError, match="invalid"):
        root_parallel_map(
            str,
            [1, 2, 3],
            backend="invalid",
            allow_sequential=True,
        )


def test_parallel_batch_evaluation_accepts_explicit_fallback_policy() -> None:
    assert parallel_batch_evaluation(
        str,
        [1, 2, 3, 4],
        backend="threads",
        max_workers=2,
        allow_sequential=False,
    ) == ["1", "2", "3", "4"]


@pytest.mark.serial
def test_real_process_parameter_sweep_is_picklable_complete_and_ordered() -> None:
    results = parallel_parameter_sweep(
        parameter_record,
        {
            "alpha": [1, 2],
            "beta": [3, 4],
        },
        backend="processes",
        max_workers=2,
        allow_sequential=False,
    )

    assert [record["value"] for record in results] == [13, 14, 23, 24]
    assert len(results) == 4
    worker_pids = {record["pid"] for record in results}
    assert worker_pids
    assert os.getpid() not in worker_pids


def test_parallel_context_uses_requested_configuration() -> None:
    with parallel_context(backend="threads", max_workers=2) as executor:
        assert executor.config.backend is ParallelBackend.THREADS
        assert executor.config.max_workers == 2


def test_parallel_healthcheck_has_stable_schema() -> None:
    status = parallel_healthcheck()

    assert status["parallel_available"] is True
    assert status["cpu_count"] >= 1
    assert status["available_memory_mb"] > 0
    assert set(status["execution_stats"]) >= {
        "tasks_submitted",
        "tasks_completed",
        "tasks_failed",
        "total_execution_time",
    }
    # No sequential baseline is measured, so no efficiency is reported.
    assert "parallel_efficiency" not in status["execution_stats"]
    assert isinstance(status["warnings"], list)
    assert isinstance(status["recommendations"], list)


def test_parallel_healthcheck_reports_resource_and_failure_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module.mp, "cpu_count", lambda: 1)
    monkeypatch.setattr(executor_module._default_executor, "available_memory", 512.0)
    monkeypatch.setattr(
        executor_module,
        "get_parallel_stats",
        lambda: {
            "tasks_submitted": 10,
            "tasks_completed": 5,
            "tasks_failed": 2,
            "total_execution_time": 1.0,
        },
    )

    status = parallel_healthcheck()

    assert len(status["warnings"]) == 3
    assert len(status["recommendations"]) == 2
