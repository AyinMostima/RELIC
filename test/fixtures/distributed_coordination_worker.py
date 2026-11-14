"""Importable Windows-spawn workers for real two-rank coordination tests (E02).

Each worker initializes a real Gloo process group, builds a RELICSimulator in
distributed mode, and drives the coordinated register/unregister protocol
through one injected scenario. The ledger it returns carries every rank's
coordination outcomes, provider epochs, registered providers, reduction
results, device, RNG identity, captured owner logs, and cleanup state.
"""

from __future__ import annotations

import hashlib
import logging
import os
import traceback
from typing import Any

import torch

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TransportMechanismFunctor,
    VariableKind,
)
from relic.core.functor.stochastic_substream import FunctorSubstreamMixin

LOGGERS = (
    "relic.computation.distributed.launcher",
    "relic.core.simulator.main",
    "relic.core.simulator.execution_loop",
)


class RankWriter(FunctorSubstreamMixin, TransportMechanismFunctor):
    """Writes rank + 1 so the distributed mean is (1 + 2) / 2 = 1.5 for two ranks."""

    TRANSPORT_TYPE = "diffusion"

    EXTRA_STATE_FORMAT = "relic.test.coordination_rank_writer"
    SUBSTREAM_LABEL = "RankWriter"

    def __init__(self, name="rank_writer", seed=7, rank=0):
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="coordination fixture writer",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            input_variables=["x"],
            output_variables=[f"{name}_out"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(name=name, config={"parameters": {}, "dependencies": []}, metadata=metadata)
        self._initialize_substream(seed)
        self.rank_value = float(rank + 1)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {f"{self.name}_out": state["x"].to(self.device) * 0.0 + self.rank_value}


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}:{record.levelname}:{record.getMessage()}")


def _tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_coordination_worker(
    rank: int,
    world_size: int,
    master_port: int,
    scenario: str,
    result_queue: Any,
) -> None:
    """Execute one coordination scenario and always return the ledger."""
    from relic.computation.distributed import launcher
    from relic.core.simulator import RELICSimulator, main as simulator_main

    capture = _LogCapture()
    for name in LOGGERS:
        logging.getLogger(name).addHandler(capture)
        logging.getLogger(name).setLevel(logging.DEBUG)

    result: dict[str, Any] = {
        "scenario": scenario,
        "pid": os.getpid(),
        "rank": rank,
        "world_size": world_size,
        "init_ok": False,
        "coordination_active": None,
        "outcomes": [],
        "epochs": [],
        "providers_after": None,
        "step_value": None,
        "step_digest": None,
        "device": None,
        "rng": None,
        "checkpoint_error": None,
        "raised": None,
        "error_type": None,
        "error": None,
        "traceback": None,
        "logs": [],
    }
    try:
        result["init_ok"] = launcher.init_distributed(
            backend="gloo",
            master_addr="127.0.0.1",
            master_port=master_port,
            world_size=world_size,
            rank=rank,
            timeout_minutes=0.1,
            use_cuda=False,
        )
        if not result["init_ok"]:
            raise RuntimeError("Gloo initialization returned false")

        # The device_mismatch scenario lets rank 0 take the GPU-first default
        # while rank 1 stays on CPU; every other scenario pins CPU explicitly.
        device = "auto" if (scenario == "device_mismatch" and rank == 0) else "cpu"
        sim = RELICSimulator({
            "device": device,
            "enable_distributed": True,
            "conservation_checks": False,
            "strict_dependencies": False,
            "adaptive_dt_enabled": False,
            # The two state variables of this fixture are pure numbers: x is a
            # placeholder input and rank_writer_out holds rank + 1, whose
            # distributed mean is 1.5. They are in no contract or catalog, so the
            # caller declares their unit (state_init infers none from names).
            "state_units": {"x": "dimensionless", "rank_writer_out": "dimensionless"},
        })
        sim.initialize_state({"x": torch.tensor(1.0), "rank_writer_out": torch.tensor(0.0)})
        result["coordination_active"] = sim._coordination_active()
        result["device"] = str(sim.device)
        result["epochs"].append(sim.provider_epoch)

        seed = 8 if (scenario == "identity_mismatch" and rank == 1) else 7
        writer = RankWriter(seed=seed, rank=rank)
        result["rng"] = writer.seed

        if scenario == "prepare_failure" and rank == 1:
            def _failing_vote(functor, execution_group, _original=sim._provider_identity_vote):
                raise RuntimeError("injected prepare failure")
            sim._provider_identity_vote = _failing_vote  # type: ignore[method-assign]
        if scenario == "apply_failure" and rank == 1:
            def _failing_apply(simulator, functor, execution_group, *, _prepare_device=True):
                raise RuntimeError("injected apply failure")
            simulator_main._register_functor_impl = _failing_apply  # type: ignore[assignment]
        if scenario == "rank_exit" and rank == 1:
            def _exit_vote(functor, execution_group):
                os._exit(3)
            sim._provider_identity_vote = _exit_vote  # type: ignore[method-assign]

        def _attempt(operation, call):
            try:
                call()
                result["raised"] = None
            except RuntimeError as exc:
                result["raised"] = str(exc)
            result["outcomes"] = [item.as_dict() for item in sim.get_coordination_outcomes()]
            result["epochs"].append(sim.provider_epoch)

        prepare_device = not (scenario == "registration_mode_mismatch" and rank == 1)
        result["prepare_device"] = prepare_device
        _attempt("register", lambda: sim.register_functor(writer, execution_group="fast", _prepare_device=prepare_device))
        result["providers_after"] = sorted(sim.functors.keys())

        if scenario == "commit":
            sim.step(dt=1.0)
            value = sim.get_state()["rank_writer_out"]
            result["step_value"] = float(value.item())
            result["step_digest"] = _tensor_digest(value)
            try:
                sim.create_checkpoint(os.path.join(os.getcwd(), f"coordination_rank{rank}.ckpt"))
            except RuntimeError as exc:
                result["checkpoint_error"] = str(exc)
            _attempt("unregister", lambda: sim.unregister_functor("rank_writer"))
            result["providers_after"] = sorted(sim.functors.keys())
        elif scenario == "apply_failure":
            # A second transaction exposes the epoch divergence left behind by
            # the compensated first one; it must abort on every rank.
            _attempt("register_again", lambda: sim.register_functor(RankWriter(name="rank_writer_2", seed=7, rank=rank), execution_group="fast"))
            result["providers_after"] = sorted(sim.functors.keys())
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
        result["owner_initialized_after_cleanup"] = launcher.get_process_group_manager().is_initialized()
        result["torch_initialized_after_cleanup"] = launcher.dist.is_initialized()
        result["logs"] = list(capture.messages)
        result_queue.put(result)
