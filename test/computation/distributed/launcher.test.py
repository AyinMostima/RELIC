"""Tests for distributed launcher result gathering behavior."""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import time
from typing import Any, Dict, List

import pytest

from relic.computation.distributed import launcher
from test.fixtures.distributed_gloo_worker import run_gloo_worker


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_port_rebinds(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def _run_real_workers(world_size: int, scenario: str) -> list[dict[str, Any]]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_local_port()
    processes = [
        context.Process(
            target=run_gloo_worker,
            args=(rank, world_size, port, scenario, result_queue),
        )
        for rank in range(world_size)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 60.0
        for process in processes:
            remaining = max(0.0, deadline - time.monotonic())
            process.join(remaining)
        if any(process.is_alive() for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(5.0)
            pytest.fail("Real Gloo worker exceeded the 60 second deadline")

        assert [process.exitcode for process in processes] == [0] * world_size
        records = [result_queue.get(timeout=5.0) for _ in range(world_size)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5.0)
            process.close()
        result_queue.close()
        result_queue.join_thread()
    _assert_port_rebinds(port)
    return sorted(records, key=lambda record: record["rank"])


def test_process_group_preserves_explicit_local_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = launcher.ProcessGroupManager()
    config = launcher.DistributedConfig(
        backend="gloo",
        world_size=2,
        rank=1,
        local_rank=0,
        use_cuda=False,
    )
    monkeypatch.setattr(launcher, "DISTRIBUTED_AVAILABLE", True)
    monkeypatch.setattr(manager, "_get_available_backends", lambda: ["gloo"])
    monkeypatch.setattr(launcher.dist, "init_process_group", lambda **_kwargs: None)

    try:
        assert manager.init_process_group(config) is True
        assert launcher._DISTRIBUTED_STATE["local_rank"] == 0
    finally:
        manager._initialized = False
        launcher._DISTRIBUTED_STATE.update(
            initialized=False,
            rank=0,
            world_size=1,
            backend=None,
            local_rank=0,
        )


def _reset_launcher_state(manager) -> None:
    manager._initialized = False
    launcher._DISTRIBUTED_STATE.update(
        initialized=False, rank=0, world_size=1, backend=None, local_rank=0,
    )


def test_unavailable_requested_backend_is_rejected_without_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = launcher.ProcessGroupManager()
    calls = []
    monkeypatch.setattr(launcher, "DISTRIBUTED_AVAILABLE", True)
    monkeypatch.setattr(manager, "_get_available_backends", lambda: ["gloo"])
    monkeypatch.setattr(launcher.dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    config = launcher.DistributedConfig(backend="nccl", world_size=1, rank=0, use_cuda=False)
    try:
        with pytest.raises(launcher.DistributedError, match="nccl"):
            manager.init_process_group(config)
        assert calls == [] and config.backend == "nccl"
        assert manager._initialized is False
    finally:
        _reset_launcher_state(manager)


def test_process_group_initialization_failures_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(variable, raising=False)
    manager = launcher.ProcessGroupManager()
    config = launcher.DistributedConfig(backend="gloo", world_size=1, rank=0, use_cuda=False)
    monkeypatch.setattr(manager, "_get_available_backends", lambda: ["gloo"])
    try:
        monkeypatch.setattr(launcher, "DISTRIBUTED_AVAILABLE", False)
        with pytest.raises(launcher.DistributedError, match="not available"):
            manager.init_process_group(config)
        monkeypatch.setattr(launcher, "DISTRIBUTED_AVAILABLE", True)

        def _refused(**_kwargs):
            raise RuntimeError("rendezvous refused")

        monkeypatch.setattr(launcher.dist, "init_process_group", _refused)
        with pytest.raises(RuntimeError, match="rendezvous refused"):
            manager.init_process_group(config)
        assert manager._initialized is False
        assert "MASTER_PORT" not in launcher.os.environ
    finally:
        _reset_launcher_state(manager)


def test_gpu_rank_without_a_visible_device_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(variable, raising=False)
    manager = launcher.ProcessGroupManager()
    selected = []
    monkeypatch.setattr(launcher, "DISTRIBUTED_AVAILABLE", True)
    monkeypatch.setattr(manager, "_get_available_backends", lambda: ["gloo"])
    monkeypatch.setattr(launcher.dist, "init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr(launcher.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(launcher.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(launcher.device_policy, "is_cuda_available", lambda: True)
    monkeypatch.setattr(launcher.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(launcher.torch.cuda, "set_device", selected.append)
    config = launcher.DistributedConfig(backend="gloo", world_size=4, rank=3, local_rank=3, use_cuda=True)
    try:
        with pytest.raises(launcher.DistributedError, match="local_rank"):
            manager.init_process_group(config)
        assert selected == []
        assert manager._initialized is False
    finally:
        _reset_launcher_state(manager)


def test_destroy_reports_cleanup_failures_after_releasing_the_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = launcher.ProcessGroupManager()
    destroyed = []
    monkeypatch.setattr(launcher, "DISTRIBUTED_AVAILABLE", True)
    monkeypatch.setattr(launcher.dist, "is_initialized", lambda: not destroyed)
    monkeypatch.setattr(launcher.dist, "destroy_process_group", lambda: destroyed.append(True))
    manager._initialized = True

    def _failing_callback():
        raise OSError("checkpoint flush failed")

    manager.add_cleanup_callback(_failing_callback)
    try:
        with pytest.raises(ExceptionGroup) as error:
            manager.destroy_process_group()
        assert [str(item) for item in error.value.exceptions] == ["checkpoint flush failed"]
        assert destroyed == [True] and manager._initialized is False
    finally:
        _reset_launcher_state(manager)


def test_failed_group_destruction_propagates_and_keeps_the_group_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = launcher.ProcessGroupManager()
    monkeypatch.setattr(launcher, "DISTRIBUTED_AVAILABLE", True)
    monkeypatch.setattr(launcher.dist, "is_initialized", lambda: True)

    def _stuck():
        raise RuntimeError("process group still has pending work")

    monkeypatch.setattr(launcher.dist, "destroy_process_group", _stuck)
    manager._initialized = True
    try:
        with pytest.raises(ExceptionGroup) as error:
            manager.destroy_process_group()
        assert "pending work" in str(error.value.exceptions[0])
        assert manager._initialized is True
    finally:
        _reset_launcher_state(manager)


def test_reinitialization_is_idempotent_only_for_the_same_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(variable, raising=False)
    manager = launcher.ProcessGroupManager()
    calls = []
    monkeypatch.setattr(launcher, "DISTRIBUTED_AVAILABLE", True)
    monkeypatch.setattr(manager, "_get_available_backends", lambda: ["gloo"])
    monkeypatch.setattr(launcher.dist, "init_process_group", lambda **kwargs: calls.append(kwargs))

    def _config(**changes):
        base = dict(backend="gloo", world_size=2, rank=0, master_port=29600, use_cuda=False)
        base.update(changes)
        return launcher.DistributedConfig(**base)

    try:
        assert manager.init_process_group(_config()) is True
        assert manager.init_process_group(_config()) is True
        assert len(calls) == 1
        with pytest.raises(launcher.DistributedError, match="world_size"):
            manager.init_process_group(_config(world_size=4))
        with pytest.raises(launcher.DistributedError, match="master_port"):
            manager.init_process_group(_config(master_port=29601))
        assert len(calls) == 1
    finally:
        _reset_launcher_state(manager)


def test_single_node_multi_gpu_requires_explicit_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher.device_policy,
        "is_cuda_available",
        lambda: False,
    )
    simulation_launcher = launcher.DistributedSimulationLauncher()

    with pytest.raises(RuntimeError, match="allow_cpu=True"):
        simulation_launcher.run_single_node_multi_gpu(
            lambda: "cpu",
            allow_cpu=False,
        )

    assert simulation_launcher.run_single_node_multi_gpu(
        lambda: "cpu",
        allow_cpu=True,
    ) == "cpu"


def test_distributed_config_is_the_single_owner_of_the_launch_defaults() -> None:
    import dataclasses
    import inspect

    fields = {field.name for field in dataclasses.fields(launcher.DistributedConfig)}
    assert not {"cleanup_on_exit", "retry_attempts"} & fields
    assert list(inspect.signature(launcher.ProcessGroupManager.__init__).parameters) == ["self", "config"]
    for function in (launcher.init_distributed, launcher.setup_distributed_simulation):
        parameters = inspect.signature(function).parameters.values()
        assert [parameter.kind for parameter in parameters] == [inspect.Parameter.VAR_KEYWORD], function.__name__
    manager = launcher.ProcessGroupManager()
    assert manager._config == launcher.DistributedConfig()
    configured = launcher.ProcessGroupManager(launcher.DistributedConfig(master_port=29611))
    assert configured._config.master_port == 29611


def test_gathered_ensemble_payload_missing_a_key_raises(monkeypatch) -> None:
    """The payloads are this module's own; a missing key is an internal error."""
    monkeypatch.setattr(launcher, "is_distributed", lambda: True)
    monkeypatch.setattr(launcher, "get_world_size", lambda: 2)
    monkeypatch.setattr(launcher, "get_rank", lambda: 0)
    monkeypatch.setattr(launcher, "is_master", lambda: True)

    def fake_gather_object(local_payload, object_gather_list, dst=0):
        object_gather_list[0] = local_payload
        object_gather_list[1] = {"rank": 1, "indices": [1], "results": [10]}

    monkeypatch.setattr(launcher.dist, "gather_object", fake_gather_object)
    with pytest.raises(KeyError, match="errors"):
        launcher.DistributedSimulationLauncher().launch_ensemble_simulation(
            simulation_func=lambda params: params["value"] * 10,
            parameter_sets=[{"value": 0}, {"value": 1}],
        )


def test_launch_ensemble_simulation_merges_gathered_payloads_on_master(monkeypatch) -> None:
    parameter_sets = [{"value": i} for i in range(5)]

    monkeypatch.setattr(launcher, "is_distributed", lambda: True)
    monkeypatch.setattr(launcher, "get_world_size", lambda: 3)
    monkeypatch.setattr(launcher, "get_rank", lambda: 0)
    monkeypatch.setattr(launcher, "is_master", lambda: True)

    def fake_gather_object(
        local_payload: Dict[str, Any],
        object_gather_list: List[Dict[str, Any]] | None,
        dst: int = 0,
    ) -> None:
        assert dst == 0
        assert object_gather_list is not None
        object_gather_list[0] = local_payload
        object_gather_list[1] = {"rank": 1, "indices": [1, 4], "results": [10, 40], "errors": []}
        object_gather_list[2] = {"rank": 2, "indices": [2], "results": [20], "errors": []}

    monkeypatch.setattr(launcher.dist, "gather_object", fake_gather_object)

    simulation_launcher = launcher.DistributedSimulationLauncher()
    output = simulation_launcher.launch_ensemble_simulation(
        simulation_func=lambda params: params["value"] * 10,
        parameter_sets=parameter_sets,
    )
    assert output == [0, 10, 20, 30, 40]


def test_launch_ensemble_simulation_returns_local_results_on_worker(monkeypatch) -> None:
    parameter_sets = [{"value": i} for i in range(5)]

    monkeypatch.setattr(launcher, "is_distributed", lambda: True)
    monkeypatch.setattr(launcher, "get_world_size", lambda: 3)
    monkeypatch.setattr(launcher, "get_rank", lambda: 1)
    monkeypatch.setattr(launcher, "is_master", lambda: False)

    called = {"worker_sent_payload": False}

    def fake_gather_object(local_payload: Dict[str, Any], object_gather_list, dst: int = 0) -> None:
        assert dst == 0
        assert object_gather_list is None
        called["worker_sent_payload"] = True
        assert local_payload["indices"] == [1, 4]
        assert local_payload["results"] == [10, 40]

    monkeypatch.setattr(launcher.dist, "gather_object", fake_gather_object)

    simulation_launcher = launcher.DistributedSimulationLauncher()
    output = simulation_launcher.launch_ensemble_simulation(
        simulation_func=lambda params: params["value"] * 10,
        parameter_sets=parameter_sets,
    )
    assert output == [10, 40]
    assert called["worker_sent_payload"]


def test_launch_ensemble_simulation_fails_closed_on_gathered_member_error(
    monkeypatch,
) -> None:
    parameter_sets = [{"value": i} for i in range(3)]
    monkeypatch.setattr(launcher, "is_distributed", lambda: True)
    monkeypatch.setattr(launcher, "get_world_size", lambda: 2)
    monkeypatch.setattr(launcher, "get_rank", lambda: 0)
    monkeypatch.setattr(launcher, "is_master", lambda: True)

    def fake_gather(local_payload, gathered, dst=0):
        assert dst == 0
        gathered[0] = local_payload
        gathered[1] = {
            "rank": 1,
            "indices": [1],
            "results": [None],
            "errors": [
                {
                    "member_id": 1,
                    "rank": 1,
                    "type": "ValueError",
                    "message": "bad member",
                }
            ],
        }

    monkeypatch.setattr(launcher.dist, "gather_object", fake_gather)

    with pytest.raises(RuntimeError, match=r"member 1.*ValueError.*bad member"):
        launcher.DistributedSimulationLauncher().launch_ensemble_simulation(
            simulation_func=lambda params: params["value"],
            parameter_sets=parameter_sets,
        )


def test_launch_ensemble_simulation_rejects_missing_rank_payload(monkeypatch) -> None:
    parameter_sets = [{"value": i} for i in range(2)]
    monkeypatch.setattr(launcher, "is_distributed", lambda: True)
    monkeypatch.setattr(launcher, "get_world_size", lambda: 2)
    monkeypatch.setattr(launcher, "get_rank", lambda: 0)
    monkeypatch.setattr(launcher, "is_master", lambda: True)

    def fake_gather(local_payload, gathered, dst=0):
        assert dst == 0
        gathered[0] = local_payload
        gathered[1] = None

    monkeypatch.setattr(launcher.dist, "gather_object", fake_gather)

    with pytest.raises(RuntimeError, match="missing rank payload"):
        launcher.DistributedSimulationLauncher().launch_ensemble_simulation(
            simulation_func=lambda params: params["value"],
            parameter_sets=parameter_sets,
        )


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.slow
def test_real_gloo_world_size_one_collectives_and_cleanup() -> None:
    records = _run_real_workers(world_size=1, scenario="collectives")

    assert len(records) == 1
    record = records[0]
    assert record["pid"] != os.getpid()
    assert record["init_ok"] is True
    assert record["second_init_ok"] is True
    assert record["all_reduce_value"] == pytest.approx(1.0)
    assert record["gathered"] == [{"rank": 0, "pid": record["pid"]}]
    assert [item["member_id"] for item in record["ensemble_results"]] == list(
        range(5)
    )
    assert record["error"] is None
    assert record["owner_initialized_after_cleanup"] is False
    assert record["torch_initialized_after_cleanup"] is False


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.slow
def test_real_gloo_world_size_two_shards_gathers_and_cleans_up() -> None:
    records = _run_real_workers(world_size=2, scenario="collectives")

    assert {record["rank"] for record in records} == {0, 1}
    assert len({record["pid"] for record in records}) == 2
    assert all(record["pid"] != os.getpid() for record in records)
    assert all(record["init_ok"] is True for record in records)
    assert all(record["second_init_ok"] is True for record in records)
    assert all(
        record["all_reduce_value"] == pytest.approx(3.0)
        for record in records
    )
    master = records[0]
    assert [payload["rank"] for payload in master["gathered"]] == [0, 1]
    assert [item["member_id"] for item in master["ensemble_results"]] == list(
        range(5)
    )
    assert [item["rank"] for item in master["ensemble_results"]] == [0, 1, 0, 1, 0]
    assert all(record["error"] is None for record in records)
    assert all(
        record["owner_initialized_after_cleanup"] is False
        for record in records
    )
    assert all(
        record["torch_initialized_after_cleanup"] is False
        for record in records
    )


@pytest.mark.integration
@pytest.mark.serial
@pytest.mark.slow
def test_distributed_context_cleans_up_after_exception() -> None:
    records = _run_real_workers(world_size=1, scenario="context_error")

    record = records[0]
    assert record["init_ok"] is True
    assert record["context_initialized"] is True
    assert record["expected_error"] == "expected context failure"
    assert record["error"] is None
    assert record["owner_initialized_after_cleanup"] is False
    assert record["torch_initialized_after_cleanup"] is False
