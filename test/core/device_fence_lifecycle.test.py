"""Device fences bound to the provider lifecycle (E01).

A Python return from a kernel launch is not completion of the device work.
The registration owner records a ``DeviceFence`` (CUDA events on the current
streams of the view's devices) when a reader token is released and when a
view is retired; a retired view is disposed only after its last reader left
and every fence completed. Device migration at registration waits for the
source device's enqueued work before copying tensors and renews the
substream generator on the target device.

CPU behaviour is always verified. The CUDA cases run on a real device and are
skipped (reported as unverified, never as passed) when none is available.
"""

from __future__ import annotations

import time

import pytest
import torch

from relic.core import device_policy
from relic.core.device_policy import DeviceFence, record_device_fence
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    TransportMechanismFunctor,
    VariableKind,
)
from relic.core.functor.stochastic_substream import FunctorSubstreamMixin
from relic.core.simulator import RELICSimulator

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="real CUDA device required; GPU part unverified")
# About 0.7 s of spinning on this class of GPU; long enough to observe a
# pending fence, short enough for the suite.
SPIN_CYCLES = 2_000_000_000


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'a_out': 'dimensionless',
    'b_out': 'dimensionless',
    'substream_out': 'dimensionless',
    'x': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


def _metadata(name, inputs, outputs):
    return FunctorMetadata(
        name=name,
        version="1.0.0",
        description="device fence fixture",
        category=ProcessCategory.TRANSPORT_MECHANISM,
        execution_phase=ExecutionPhase.MAIN_PROCESSES,
        references=["test fixture"],
        input_variables=list(inputs),
        output_variables=list(outputs),
        spatial_scale=SpatialScale.GRID_CELL,
        variable_kind=VariableKind.FLUX,
    )


class _GainWriter(TransportMechanismFunctor):
    """out = gain * x on the functor device; records what it actually computed."""

    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name="a", gain=2.0, spin=False):
        super().__init__(name=name, config={}, metadata=_metadata(name, ["x"], [f"{name}_out"]))
        self.gain = torch.nn.Parameter(torch.tensor(float(gain)))
        self.spin = bool(spin)
        self.observed = {}

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        x = state["x"].to(self.device)
        if self.spin and self.device.type == "cuda":
            torch.cuda._sleep(SPIN_CYCLES)
        y = x * self.gain
        self.observed = {
            "device": y.device,
            "requires_grad": y.requires_grad,
            "has_grad_fn": y.grad_fn is not None,
        }
        return {f"{self.name}_out": y}


class _SubstreamWriter(FunctorSubstreamMixin, PhysicalFieldFunctor):
    """Substream owner whose generator must follow the functor device."""

    EXTRA_STATE_FORMAT = "relic.test.device_fence_substream"
    SUBSTREAM_LABEL = "DeviceFence"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self, seed=3):
        metadata = FunctorMetadata(
            name="substream_writer",
            version="1.0.0",
            description="substream device fixture",
            category=ProcessCategory.PHYSICAL_FIELD,
            input_variables=["x"],
            output_variables=["substream_out"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        super().__init__(name="substream_writer", config={"parameters": {}, "dependencies": []}, metadata=metadata)
        self._initialize_substream(seed)
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        return {"substream_out": state["x"].to(self.device) * self.gain}


def _simulator(device, **config):
    sim = _declared_simulator({
        "device": device,
        "conservation_checks": False,
        "strict_dependencies": False,
        "adaptive_dt_enabled": False,
        **config,
    })
    sim.initialize_state({
        "x": torch.tensor(1.5),
        "a_out": torch.tensor(0.0),
        "b_out": torch.tensor(0.0),
        "substream_out": torch.tensor(0.0),
    })
    return sim


def _receipt(sim, operation, name):
    return [r for r in sim.get_lifecycle_receipts() if r.operation == operation and r.provider_name == name][-1]


class TestCpuFence:
    def test_cpu_fence_is_vacuously_complete_and_dispose_is_immediate(self):
        fence = DeviceFence(["cpu", torch.device("cpu")])
        assert fence.devices == () and fence.completed is True
        assert fence.describe() == {"devices": [], "events": 0, "completed": True}
        assert fence.wait(timeout=0.0) is True

        sim = _simulator("cpu")
        a = _GainWriter("a")
        sim.register_functor(a, execution_group="fast")
        sim.step(dt=1.0)
        old_epoch = sim.provider_epoch
        sim.register_functor(_GainWriter("b", gain=3.0), execution_group="fast")
        lifecycle = sim._provider_lifecycle
        assert lifecycle.retired_status[old_epoch] == "disposed"
        assert lifecycle.device_fence_status(old_epoch) == {
            "epoch": old_epoch, "status": "disposed", "pending_fences": 0, "devices": [],
        }
        retired = _receipt(sim, "register", "b").managed_delta["provider_view"]["retired_view"]
        assert retired["status"] == "disposed" and retired["pending_fences"] == 0
        assert _receipt(sim, "register", "a").managed_delta["device"]["source_fence"] is None
        assert a.observed["device"].type == "cpu"
        assert sim.get_state()["a_out"].item() == 3.0

    def test_cuda_request_falls_back_to_cpu_when_cuda_is_absent(self, monkeypatch):
        monkeypatch.setattr(device_policy, "is_cuda_available", lambda: False)
        sim = _simulator("cuda")
        assert sim.device.type == "cpu"
        fence = DeviceFence([torch.device("cuda")])
        assert fence.devices == (torch.device("cuda"),)
        assert fence.describe() == {"devices": ["cuda"], "events": 0, "completed": True}
        with pytest.raises(RuntimeError, match="requires CUDA"):
            _declared_simulator({"device": "cuda", "require_gpu": True})


@requires_cuda
class TestCudaFence:
    def test_python_return_precedes_device_completion(self):
        fence_before = record_device_fence(["cuda"])
        assert fence_before.completed is True
        torch.cuda._sleep(SPIN_CYCLES)
        fence = record_device_fence([torch.device("cuda")])
        returned_at = time.perf_counter()
        assert fence.devices == (torch.device("cuda", 0),) and fence.describe()["events"] == 1
        assert fence.completed is False
        assert fence.wait(timeout=0.01) is False
        assert fence.wait(timeout=30.0) is True
        assert fence.completed is True
        assert time.perf_counter() - returned_at > 0.05

    def test_retired_view_is_disposed_only_after_the_reader_fence_completes(self):
        sim = _simulator("cuda")
        a = _GainWriter("a")
        sim.register_functor(a, execution_group="fast")
        sim.step(dt=1.0)
        assert a.observed["device"].type == "cuda"
        old_epoch = sim.provider_epoch
        lifecycle = sim._provider_lifecycle
        # A reader holds the old view across the swap and keeps enqueuing
        # device work on its stream after the new epoch was published.
        token = sim.acquire_provider_view()
        sim.register_functor(_GainWriter("b", gain=3.0), execution_group="fast")
        assert lifecycle.retired_status[old_epoch] == "pending_dispose"
        torch.cuda._sleep(SPIN_CYCLES)
        token.release()
        # The token is released, the reader count is zero, yet the view stays
        # alive: its fence has not completed.
        assert lifecycle.active_reader_count(old_epoch) == 0
        assert lifecycle.retired_status[old_epoch] == "pending_device_fence"
        assert old_epoch in lifecycle.retired_views
        status = lifecycle.device_fence_status(old_epoch)
        assert status["pending_fences"] >= 1 and status["devices"] == ["cuda:0"]
        assert lifecycle.retry_dispose(old_epoch) == "pending_device_fence"
        assert lifecycle.wait_quiescent(old_epoch, timeout=0.01) is False
        assert lifecycle.wait_quiescent(old_epoch, timeout=30.0) is True
        assert lifecycle.retired_status[old_epoch] == "disposed"
        assert old_epoch not in lifecycle.retired_views
        assert lifecycle.device_fence_status(old_epoch)["pending_fences"] == 0
        sim.step(dt=1.0)
        assert sim.get_state()["b_out"].item() == 4.5

    def test_retirement_without_readers_still_waits_for_enqueued_work(self):
        sim = _simulator("cuda")
        a = _GainWriter("a", spin=True)
        sim.register_functor(a, execution_group="fast")
        sim.step(dt=1.0)
        old_epoch = sim.provider_epoch
        lifecycle = sim._provider_lifecycle
        torch.cuda._sleep(SPIN_CYCLES)
        removed = sim.unregister_functor("a")
        assert removed is True
        receipt = _receipt(sim, "unregister", "a")
        retired = receipt.managed_delta["provider_view"]["retired_view"]
        assert retired["epoch"] == old_epoch
        assert retired["status"] == "pending_device_fence"
        assert retired["pending_fences"] == 1 and retired["devices"] == ["cuda:0"]
        assert lifecycle.retired_status[old_epoch] == "pending_device_fence"
        assert lifecycle.wait_quiescent(old_epoch, timeout=30.0) is True
        assert lifecycle.retired_status[old_epoch] == "disposed"

    def test_override_and_auto_place_actual_tensors_and_autograd_per_functor(self):
        sim = _simulator("auto", device_overrides={"b": "cpu"})
        assert sim.device.type == "cuda"
        a = _GainWriter("a")
        b = _GainWriter("b", gain=3.0)
        sim.register_functor(a, execution_group="fast")
        sim.register_functor(b, execution_group="fast")
        assert sim.get_effective_device("a").type == "cuda"
        assert sim.get_effective_device("b").type == "cpu"
        assert a.gain.device.type == "cuda" and b.gain.device.type == "cpu"
        sim.step(dt=1.0)
        assert a.observed == {"device": torch.device("cuda", 0), "requires_grad": True, "has_grad_fn": True}
        assert b.observed == {"device": torch.device("cpu"), "requires_grad": True, "has_grad_fn": True}
        state = sim.get_state()
        assert state["a_out"].item() == 3.0 and state["b_out"].item() == 4.5
        # Only the CUDA provider contributes a fence device.
        old_epoch = sim.provider_epoch
        torch.cuda._sleep(SPIN_CYCLES)
        sim.unregister_functor("b")
        retired = _receipt(sim, "unregister", "b").managed_delta["provider_view"]["retired_view"]
        assert retired["devices"] == ["cuda:0"]
        assert sim._provider_lifecycle.wait_quiescent(old_epoch, timeout=30.0) is True

    def test_migration_waits_for_source_work_and_renews_the_substream_generator(self):
        # CPU placeholder -> CUDA at registration: generator re-created on CUDA.
        sim = _simulator("cuda")
        writer = _SubstreamWriter(seed=3)
        assert torch.device(writer.generator.device).type == "cpu"
        sim.register_functor(writer, execution_group="fast")
        assert writer.device.type == "cuda"
        assert torch.device(writer.generator.device).type == "cuda"
        assert writer.draw_count == 0
        receipt = _receipt(sim, "register", "substream_writer")
        assert receipt.managed_delta["device"]["before"] == "cpu"
        assert receipt.managed_delta["device"]["after"].startswith("cuda")
        assert receipt.managed_delta["device"]["source_fence"] == {"devices": [], "events": 0, "completed": True}

        # CUDA -> CPU (override) while work is still enqueued on the source:
        # the copy observes the completed work because the fence was awaited.
        sim_cpu = _simulator("cuda", device_overrides={"a": "cpu"})
        a = _GainWriter("a")
        a.to_device("cuda")
        torch.cuda._sleep(SPIN_CYCLES)
        with torch.no_grad():
            a.gain.add_(5.0)
        started = time.perf_counter()
        sim_cpu.register_functor(a, execution_group="fast")
        assert a.device.type == "cpu" and a.gain.device.type == "cpu"
        assert a.gain.item() == 7.0
        fence = _receipt(sim_cpu, "register", "a").managed_delta["device"]["source_fence"]
        assert fence == {"devices": ["cuda:0"], "events": 1, "completed": True}
        assert time.perf_counter() - started > 0.05
