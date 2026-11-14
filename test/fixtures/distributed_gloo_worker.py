"""Importable Windows-spawn workers for real Gloo integration tests."""

from __future__ import annotations

import os
import traceback
from typing import Any

import torch


def _distributed_member(parameters: dict[str, int]) -> dict[str, int]:
    from relic.computation.distributed import launcher

    return {
        "member_id": int(parameters["member_id"]),
        "rank": int(launcher.get_rank()),
        "pid": os.getpid(),
    }


def run_gloo_worker(
    rank: int,
    world_size: int,
    master_port: int,
    scenario: str,
    result_queue: Any,
) -> None:
    """Execute one bounded Gloo scenario and always return a cleanup ledger."""
    from relic.computation.distributed import launcher

    result: dict[str, Any] = {
        "scenario": scenario,
        "pid": os.getpid(),
        "rank": rank,
        "world_size": world_size,
        "init_ok": False,
        "second_init_ok": None,
        "all_reduce_value": None,
        "gathered": None,
        "ensemble_results": None,
        "context_initialized": None,
        "expected_error": None,
        "error_type": None,
        "error": None,
        "traceback": None,
    }
    try:
        kwargs = {
            "backend": "gloo",
            "master_addr": "127.0.0.1",
            "master_port": master_port,
            "world_size": world_size,
            "rank": rank,
            "timeout_minutes": 0.5,
            "use_cuda": False,
        }
        if scenario == "context_error":
            try:
                with launcher.distributed_context(**kwargs) as initialized:
                    result["init_ok"] = bool(initialized)
                    result["context_initialized"] = launcher.is_distributed()
                    raise RuntimeError("expected context failure")
            except RuntimeError as exc:
                result["expected_error"] = str(exc)
        else:
            result["init_ok"] = launcher.init_distributed(**kwargs)
            if not result["init_ok"]:
                raise RuntimeError("Gloo initialization returned false")
            result["second_init_ok"] = launcher.init_distributed(**kwargs)

            reduced = torch.tensor(float(rank + 1), dtype=torch.float64)
            launcher.all_reduce(reduced)
            result["all_reduce_value"] = float(reduced.item())
            launcher.barrier()

            payload = {"rank": rank, "pid": os.getpid()}
            if rank == 0:
                gathered: list[dict[str, int] | None] = [None] * world_size
                launcher.dist.gather_object(payload, gathered, dst=0)
                result["gathered"] = gathered
            else:
                launcher.dist.gather_object(payload, None, dst=0)

            parameter_sets = [
                {"member_id": member_id}
                for member_id in range(5)
            ]
            result["ensemble_results"] = (
                launcher.DistributedSimulationLauncher().launch_ensemble_simulation(
                    _distributed_member,
                    parameter_sets,
                )
            )
            launcher.barrier()
    except Exception as exc:  # noqa: BLE001 - return the full child failure ledger
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            launcher.cleanup_distributed()
        except Exception as exc:  # noqa: BLE001 - cleanup must reach the parent ledger
            if result["error"] is None:
                result["error_type"] = type(exc).__name__
                result["error"] = str(exc)
                result["traceback"] = traceback.format_exc()
        result["owner_initialized_after_cleanup"] = (
            launcher.get_process_group_manager().is_initialized()
        )
        result["torch_initialized_after_cleanup"] = launcher.dist.is_initialized()
        result_queue.put(result)
