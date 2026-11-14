"""
Device and distributed runtime inspection helpers for RELICSimulator.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
from relic.core import device_policy


def get_device_info(
    simulator: Any,
    has_distributed: Callable[[], bool],
    is_distributed: Callable[[], bool],
    get_rank: Callable[[], int],
    get_world_size: Callable[[], int],
    logger: Any,
) -> Dict[str, Any]:
    """
    Query current device configuration and distributed computing status.
    """
    info = {
        "device": simulator.device if hasattr(simulator, "device") else torch.device("cpu"),
        "device_type": None,
        "device_index": None,
        "distributed_enabled": (
            simulator.enable_distributed
            if hasattr(simulator, "enable_distributed")
            else False
        ),
        "rank": None,
        "world_size": None,
        "cuda_available": device_policy.is_cuda_available(),
        "gpu_memory_allocated": None,
        "gpu_memory_cached": None,
    }

    if hasattr(simulator, "device"):
        info["device_type"] = simulator.device.type
        if simulator.device.type == "cuda":
            info["device_index"] = simulator.device.index

    if has_distributed() and info["distributed_enabled"] and is_distributed():
        info["rank"] = get_rank()
        info["world_size"] = get_world_size()

    if info["device_type"] == "cuda" and device_policy.is_cuda_available():
        device_idx = info["device_index"] if info["device_index"] is not None else 0
        info["gpu_memory_allocated"] = (
            torch.cuda.memory_allocated(device_idx) / 1024**2
        )
        info["gpu_memory_cached"] = (
            torch.cuda.memory_reserved(device_idx) / 1024**2
        )

    return info
