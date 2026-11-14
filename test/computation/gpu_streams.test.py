import pytest
import torch

import relic.computation.gpu as gpu_package
import relic.computation.gpu.accelerator as accelerator_module
from relic.computation.gpu.accelerator import StreamManager
from relic.core import device_policy


def test_gpu_package_does_not_export_device_manager_owner() -> None:
    assert "DeviceManager" not in gpu_package.__all__
    assert not hasattr(gpu_package, "DeviceManager")


def test_accelerator_has_no_device_manager_or_global_device_cache() -> None:
    for name in (
        "DeviceManager",
        "get_global_device_manager",
        "set_global_device_manager",
        "_global_device_manager",
        "_DEVICE_CACHE",
        "_MEMORY_STATS",
    ):
        assert not hasattr(accelerator_module, name)


def test_device_selection_uses_canonical_require_gpu_semantics(monkeypatch) -> None:
    monkeypatch.setattr(device_policy, "is_cuda_available", lambda: False)

    assert accelerator_module.select_device(
        "cuda",
        require_gpu=False,
    ) == torch.device("cpu")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        accelerator_module.select_device("cuda", require_gpu=True)


def test_cpu_device_utilities_do_not_require_a_manager(monkeypatch) -> None:
    monkeypatch.setattr(device_policy, "is_cuda_available", lambda: False)

    assert accelerator_module.is_cuda_available() is False
    assert accelerator_module.get_device_info("cpu") == {
        "type": "cpu",
        "name": "CPU",
        "available": True,
        "memory": "system",
    }
    accelerator_module.clear_gpu_cache()
    accelerator_module.synchronize_device(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_stream_creation() -> None:
    manager = StreamManager(max_streams=2)
    stream_a = manager.get_stream("stream_a")
    stream_b = manager.get_stream("stream_b")

    assert stream_a is not None
    assert stream_b is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_event_synchronization() -> None:
    manager = StreamManager(max_streams=2)
    manager.record_event("stream_a", "event_1")
    manager.wait_event("stream_b", "event_1")
    manager.synchronize_all()

    stats = manager.get_pipeline_efficiency()
    assert stats["recorded_events"] >= 1.0
    assert stats["wait_operations"] >= 1.0


def test_cpu_fallback() -> None:
    stream_manager = StreamManager(max_streams=2)
    stream_manager._cuda_available = False
    assert stream_manager.get_stream("stream_cpu") is None
    stream_manager.record_event("stream_cpu", "event_cpu")
    stream_manager.wait_event("stream_cpu", "event_cpu")
    stream_manager.synchronize_all()
    metrics = stream_manager.get_pipeline_efficiency()
    assert metrics["utilization"] == 0.0
