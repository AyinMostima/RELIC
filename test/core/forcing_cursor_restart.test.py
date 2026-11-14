"""Forcing cursor restart contract (F02).

The simulator owns one forcing cursor: the canonical digest of an externally
supplied forcing sequence, its length, and the number of snapshots consumed by
accepted steps. It advances only inside an accepted step, rolls back with the
step transaction, is part of the durable checkpoint runtime, and lets
``GraphExecutor.execute_forcing_sequence_from_checkpoint`` resume a
mid-sequence checkpoint at the next unconsumed snapshot.
"""

from __future__ import annotations

from collections.abc import Sequence
import weakref

import pytest
import torch

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    TransportMechanismFunctor,
)
from relic.core.simulator import RELICSimulator
from relic.core.simulator import checkpoint as checkpoint_module
from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.orchestration.graph_ir import GraphIR, NodeSpec
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.port_spec import PortSpec
from relic.orchestration.ssot_catalog import SSOTCatalog

INTEGRATOR_CLASS_TYPE = "Functor:test.forcing_integrator"
SEQUENCE_SHA = "a" * 64
OTHER_SHA = "b" * 64


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'writer_out': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


class _Writer(TransportMechanismFunctor):
    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name="writer", fail=False):
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="forcing cursor fixture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["test fixture"],
            output_variables=["writer_out"],
        )
        super().__init__(name=name, config={"fail": bool(fail)}, metadata=metadata)

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt, **kwargs):
        if self._original_config["fail"]:
            raise RuntimeError("kernel failure")
        return {"writer_out": torch.tensor(1.0)}


class _ForcingIntegrator(PhysicalFieldFunctor):
    """x_{n+1} = x_n + forcing_n * dt: the forcing enters the integrated state."""

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self, name: str = "integrator", config: dict | None = None) -> None:
        metadata = FunctorMetadata(
            name=name,
            description="Integrates the external forcing into x.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="x_{n+1} = x_n + f_n dt",
            references=["RELIC internal forcing cursor verification"],
            parameters={},
            input_variables=["forcing", "x"],
            output_variables=["x"],
        )
        super().__init__(name=name, config=config or {}, metadata=metadata)

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state, time, dt, **kwargs):
        # References: RELIC internal forcing cursor verification.
        # Mathematical expression: x_{n+1} = x_n + f_n * dt, emitted as the
        # governed ACCUMULATE increment f_n * dt.
        # Parameter meanings: f_n is the forcing snapshot applied at step n.
        # Units: dimensionless test quantities; dt in s.
        # Domain/constraints: finite tensors of one shape.
        # Sign convention: positive forcing increases x.
        return {"x": state["forcing"] * dt}


def _port(name: str) -> PortSpec:
    return PortSpec(
        name=name, standard_name=name, units="1", semantics="state",
        spatial_kind="grid_cell", area_basis="ground_area",
        temporal_basis="instantaneous", vertical_basis="none",
    )


def _build_plan():
    registry = NodeRegistry(SSOTCatalog())
    definition = registry.register_functor_instance(
        _ForcingIntegrator(), node_type_id=INTEGRATOR_CLASS_TYPE,
    )
    definition.inputs["forcing"] = _port("forcing")
    definition.inputs["x"] = _port("x")
    definition.outputs["x"] = _port("x")
    graph = GraphIR(
        schema_version="1.0.0",
        nodes={
            "forcing": NodeSpec(class_type="State:ExternalForcing", outputs=["forcing"]),
            "integrator": NodeSpec(
                class_type=INTEGRATOR_CLASS_TYPE,
                inputs={"forcing": {"state": "forcing"}, "x": {"state": "x"}},
                outputs=["x"],
                config={"execution_group": "fast"},
            ),
        },
    )
    return compile_graph(graph, registry), registry


def _simulator(**config):
    sim = _declared_simulator({
        "device": "cpu", "conservation_checks": False,
        "strict_dependencies": False, "adaptive_dt_enabled": False, **config,
    })
    sim.initialize_state({"writer_out": torch.tensor(0.0)})
    return sim


SEQUENCE = [
    {"forcing": torch.tensor([1.0, 2.0])},
    {"forcing": torch.tensor([3.0, 4.0])},
    {"forcing": torch.tensor([5.0, 6.0])},
]
INITIAL = {"forcing": torch.tensor([0.0, 0.0]), "x": torch.tensor([0.0, 0.0])}


class TestSimulatorForcingCursor:
    def test_cursor_advances_only_in_accepted_steps_and_rolls_back_on_failure(self):
        sim = _simulator()
        writer = _Writer()
        sim.register_functor(writer, execution_group="fast")
        assert sim.get_forcing_cursor() is None
        sim.step(dt=1.0)
        assert sim.get_forcing_cursor() is None
        bound = sim.bind_forcing_cursor(sequence_sha256=SEQUENCE_SHA, length=2)
        assert bound == {"sequence_sha256": SEQUENCE_SHA, "length": 2, "consumed": 0}
        sim.step(dt=1.0)
        assert sim.get_forcing_cursor()["consumed"] == 1

        writer._original_config["fail"] = True
        with pytest.raises(Exception, match="kernel failure"):
            sim.step(dt=1.0)
        assert sim.get_forcing_cursor()["consumed"] == 1
        assert sim.step_count == 2
        writer._original_config["fail"] = False
        sim.step(dt=1.0)
        assert sim.get_forcing_cursor()["consumed"] == 2

    def test_exhausted_cursor_refuses_the_step_and_leaves_state_unchanged(self):
        sim = _simulator()
        sim.register_functor(_Writer(), execution_group="fast")
        sim.bind_forcing_cursor(sequence_sha256=SEQUENCE_SHA, length=1)
        sim.step(dt=1.0)
        before = float(sim.get_state()["writer_out"])
        with pytest.raises(RuntimeError, match="exhausted"):
            sim.step(dt=1.0)
        assert sim.step_count == 1 and sim.current_time == 1.0
        assert float(sim.get_state()["writer_out"]) == before
        assert sim.get_forcing_cursor() == {
            "sequence_sha256": SEQUENCE_SHA, "length": 1, "consumed": 1,
        }
        sim.clear_forcing_cursor()
        sim.step(dt=1.0)
        assert sim.step_count == 2

    def test_binding_refuses_another_sequence_while_partially_consumed(self):
        sim = _simulator()
        sim.register_functor(_Writer(), execution_group="fast")
        sim.bind_forcing_cursor(sequence_sha256=SEQUENCE_SHA, length=3)
        sim.step(dt=1.0)
        with pytest.raises(RuntimeError, match="partially consumed"):
            sim.bind_forcing_cursor(sequence_sha256=OTHER_SHA, length=3)
        with pytest.raises(RuntimeError, match="partially consumed"):
            sim.bind_forcing_cursor(sequence_sha256=SEQUENCE_SHA, length=4)
        rebound = sim.bind_forcing_cursor(sequence_sha256=SEQUENCE_SHA, length=3)
        assert rebound["consumed"] == 1
        for bad in (
            {"sequence_sha256": "short", "length": 3},
            {"sequence_sha256": SEQUENCE_SHA, "length": 0},
            {"sequence_sha256": SEQUENCE_SHA, "length": 3, "consumed": 4},
        ):
            with pytest.raises(ValueError):
                sim.bind_forcing_cursor(**bad)

    def test_checkpoint_carries_the_cursor_and_validation_rejects_corruption(self, tmp_path):
        sim = _simulator()
        sim.register_functor(_Writer(), execution_group="fast")
        sim.bind_forcing_cursor(sequence_sha256=SEQUENCE_SHA, length=3)
        sim.step(dt=1.0)
        path = tmp_path / "cursor.pt"
        sim.create_checkpoint(path)
        payload = torch.load(path, weights_only=True)
        runtime = payload["runtime"]
        assert runtime["_forcing_cursor"] == {
            "sequence_sha256": SEQUENCE_SHA, "length": 3, "consumed": 1,
        }
        sim.step(dt=1.0)
        assert sim.get_forcing_cursor()["consumed"] == 2
        sim.restore_checkpoint(path)
        assert sim.get_forcing_cursor()["consumed"] == 1 and sim.step_count == 1

        for corrupted in (
            dict(runtime["_forcing_cursor"], consumed=4),
            dict(runtime["_forcing_cursor"], consumed=-1),
            dict(runtime["_forcing_cursor"], length=0),
            dict(runtime["_forcing_cursor"], sequence_sha256="x"),
            {"length": 3, "consumed": 1},
        ):
            with pytest.raises(ValueError, match="forcing cursor"):
                checkpoint_module._validate_runtime(
                    dict(runtime, _forcing_cursor=corrupted), sim,
                )
        decoded = checkpoint_module._validate_runtime(dict(runtime, _forcing_cursor=None), sim)
        assert decoded["_forcing_cursor"] is None


class _IndexedForcingSequence(Sequence):
    """Stable indexed fixture that creates one snapshot on demand, without a cache."""

    def __init__(self, length=32, offset=0.0, invalid_index=None):
        self.length = length
        self.offset = offset
        self.invalid_index = invalid_index
        self.references = []
        self.peak_live = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if type(index) is not int or index < 0 or index >= self.length:
            raise IndexError(index)
        value = torch.tensor([index + 1.0 + self.offset, index + 2.0 + self.offset])
        if index == self.invalid_index:
            value[0] = float("nan")
        self.references.append(weakref.ref(value))
        self.peak_live = max(self.peak_live, sum(ref() is not None for ref in self.references))
        return {"forcing": value}

class TestExecutorForcingSequenceRestart:
    def _executor(self, tmp_path=None, **extra):
        plan, registry = _build_plan()
        config = {"time_step": 30.0, "adaptive_dt_enabled": False, "device": "cpu"}
        if tmp_path is not None:
            config.update({
                "enable_checkpointing": True,
                "checkpoint_frequency": 30.0,
                "checkpoint_directory": str(tmp_path / "auto"),
            })
        config.update(extra)
        return GraphExecutor(plan, registry=registry, simulator_config=config)

    def test_indexed_forcing_stays_bounded_and_matches_eager_digest_and_state(self):
        indexed = _IndexedForcingSequence()
        eager = [{"forcing": torch.tensor([i + 1.0, i + 2.0])} for i in range(32)]
        executor = self._executor()
        final = executor.execute_forcing_sequence(INITIAL, indexed, dt=30.0)
        assert final["x"].cpu().tolist() == [15840.0, 16800.0]
        assert indexed.peak_live <= 4
        assert executor._simulator.get_forcing_cursor() == {
            "sequence_sha256": GraphExecutor._forcing_sequence_sha256(eager),
            "length": 32, "consumed": 32,
        }
        eager_executor = self._executor()
        eager_final = eager_executor.execute_forcing_sequence(INITIAL, eager, dt=30.0)
        assert torch.equal(final["x"], eager_final["x"])
        assert all(torch.equal(row["forcing"], torch.tensor([i + 1.0, i + 2.0]))
                   for i, row in enumerate(eager))

    def test_indexed_forcing_restart_and_changed_source_rejection(self, tmp_path):
        producer = self._executor(tmp_path)
        expected = producer.execute_forcing_sequence(INITIAL, _IndexedForcingSequence(3), dt=30.0)
        path = next((tmp_path / "auto").rglob("checkpoint_step_000000000001.pt"))
        restored = self._executor()
        final = restored.execute_forcing_sequence_from_checkpoint(
            INITIAL, path, _IndexedForcingSequence(3), dt=30.0,
        )
        assert torch.equal(final["x"], expected["x"])
        assert restored._simulator.get_forcing_cursor()["consumed"] == 3
        with pytest.raises(ValueError, match="different forcing sequence"):
            self._executor().execute_forcing_sequence_from_checkpoint(
                INITIAL, path, _IndexedForcingSequence(3, offset=1.0), dt=30.0,
            )

    def test_indexed_forcing_validates_the_last_sample_before_any_execution(self):
        executor = self._executor()
        with pytest.raises(ValueError, match="finite"):
            executor.execute_forcing_sequence(
                INITIAL, _IndexedForcingSequence(32, invalid_index=31), dt=30.0,
            )
        assert not executor._execution_prepared
        with pytest.raises(TypeError, match="sequence"):
            executor.execute_forcing_sequence(INITIAL, iter(SEQUENCE), dt=30.0)

    def test_split_by_mid_sequence_checkpoint_matches_uninterrupted(self, tmp_path):
        uninterrupted = self._executor()
        expected = uninterrupted.execute_forcing_sequence(INITIAL, SEQUENCE, dt=30.0)
        assert expected["x"].cpu().tolist() == [270.0, 360.0]
        assert uninterrupted._simulator.get_forcing_cursor()["consumed"] == 3

        producer = self._executor(tmp_path)
        producer.execute_forcing_sequence(INITIAL, SEQUENCE, dt=30.0)
        producer._simulator.retry_external_deliveries()
        checkpoints = sorted((tmp_path / "auto").rglob("checkpoint_step_*.pt"))
        assert [p.name for p in checkpoints] == [
            "checkpoint_step_000000000001.pt",
            "checkpoint_step_000000000002.pt",
            "checkpoint_step_000000000003.pt",
        ]
        first = torch.load(checkpoints[0], weights_only=True)
        assert first["runtime"]["_forcing_cursor"]["consumed"] == 1

        resumed = self._executor()
        final = resumed.execute_forcing_sequence_from_checkpoint(
            INITIAL, checkpoints[0], SEQUENCE, dt=30.0,
        )
        assert torch.equal(final["x"].cpu(), expected["x"].cpu())
        assert torch.equal(final["forcing"].cpu(), expected["forcing"].cpu())
        assert resumed._simulator.step_count == 3
        assert resumed._simulator.current_time == pytest.approx(90.0)
        assert resumed._simulator.get_forcing_cursor()["consumed"] == 3
        assert resumed._simulator.run_id == producer._simulator.run_id
        for name in ("x",):
            torch.testing.assert_close(
                torch.stack([t.cpu() for t in resumed._simulator.time_series[name]]),
                torch.stack([t.cpu() for t in uninterrupted._simulator.time_series[name]]),
                rtol=0.0, atol=0.0,
            )

    def test_resume_rejects_other_sequences_exhausted_cursors_and_plain_checkpoints(self, tmp_path):
        producer = self._executor(tmp_path)
        producer.execute_forcing_sequence(INITIAL, SEQUENCE, dt=30.0)
        producer._simulator.retry_external_deliveries()
        checkpoints = sorted((tmp_path / "auto").rglob("checkpoint_step_*.pt"))

        other_sequence = [dict(s, forcing=s["forcing"] * 2.0) for s in SEQUENCE]
        with pytest.raises(ValueError, match="different forcing sequence"):
            self._executor().execute_forcing_sequence_from_checkpoint(
                INITIAL, checkpoints[0], other_sequence, dt=30.0,
            )
        with pytest.raises(ValueError, match="different forcing sequence"):
            self._executor().execute_forcing_sequence_from_checkpoint(
                INITIAL, checkpoints[0], SEQUENCE[:2], dt=30.0,
            )
        with pytest.raises(ValueError, match="exhausted"):
            self._executor().execute_forcing_sequence_from_checkpoint(
                INITIAL, checkpoints[2], SEQUENCE, dt=30.0,
            )

        plain = self._executor()
        plain.execute(INITIAL, n_steps=1, dt=30.0)
        plain_path = tmp_path / "plain.pt"
        plain.create_checkpoint(plain_path)
        with pytest.raises(RuntimeError, match="no forcing cursor"):
            self._executor().execute_forcing_sequence_from_checkpoint(
                INITIAL, plain_path, SEQUENCE, dt=30.0,
            )

    def test_cursor_drift_is_refused_before_a_step(self, tmp_path):
        producer = self._executor(tmp_path)
        producer.execute_forcing_sequence(INITIAL, SEQUENCE[:1], dt=30.0)
        producer._simulator.retry_external_deliveries()
        checkpoint = next((tmp_path / "auto").rglob("checkpoint_step_000000000001.pt"))
        resumed = self._executor()
        with pytest.raises(ValueError, match="exhausted"):
            resumed.execute_forcing_sequence_from_checkpoint(
                INITIAL, checkpoint, SEQUENCE[:1], dt=30.0,
            )
        assert resumed._simulator.step_count == 1


class TestExternalAdapterForcingByTime:
    """An external adapter reads its forcing at the simulation time, independent of the cursor.

    The simulator passes its calendar_type and time_origin settings; the adapter evaluates every
    field at time_origin + t by its configured time_interpolation, so the read is a function of
    t alone: a run restored from a checkpoint reads the same records as the uninterrupted run,
    and no forcing cursor is bound or checkpointed for it.
    """

    @staticmethod
    def _adapter(root):
        import cftime
        import numpy as np
        import xarray as xr

        from relic.coupling.external.hydrological_models.vic_adapter import VICAdapter, VICConfig

        vic = VICAdapter(
            VICConfig(root_path=root, time_interpolation="lower", provided_variables=["surface_snow_amount"])
        )
        vic.current_datasets = {
            "main": xr.Dataset(
                {"OUT_SWE": (("time", "cell"), np.array([[150.0, 0.0], [140.0, 10.0], [130.0, 20.0]]), {"units": "mm"})},
                coords={"time": [cftime.DatetimeNoLeap(2000, 1, 1 + day) for day in range(3)]},
            )
        }
        # The configuration declares the one catalog variable the adapter provides (its output).
        assert vic.metadata.output_variables == ["surface_snow_amount"]
        return vic

    def _simulator(self, root, **settings):
        config = {
            "device": "cpu", "calendar_type": "noleap", "time_origin": "2000-01-01T00:00:00",
            "conservation_checks": False, "strict_dependencies": False, "adaptive_dt_enabled": False,
        }
        config.update(settings)
        sim = RELICSimulator({key: value for key, value in config.items() if value is not None})
        sim.initialize_state({"surface_snow_amount": torch.tensor([0.0, 0.0])})
        sim.register_functor(self._adapter(root), execution_group="fast")
        return sim

    def test_adapter_forcing_follows_the_simulation_time_across_a_restart(self, tmp_path):
        half_day = 43200.0
        uninterrupted = self._simulator(tmp_path)
        observed = []
        for _ in range(4):
            uninterrupted.step(dt=half_day)
            observed.append(uninterrupted.get_state()["surface_snow_amount"].tolist())
        # Steps start at t = 0, 12 h, 24 h and 36 h: records 0, 0, 1, 1 under 'lower'.
        assert observed == [[150.0, 0.0], [150.0, 0.0], [140.0, 10.0], [140.0, 10.0]]
        assert uninterrupted.get_forcing_cursor() is None

        producer = self._simulator(tmp_path)
        producer.step(dt=half_day)
        path = tmp_path / "adapter.pt"
        producer.create_checkpoint(path)
        assert torch.load(path, weights_only=True)["runtime"]["_forcing_cursor"] is None

        resumed = self._simulator(tmp_path)
        resumed.restore_checkpoint(path)
        replay = []
        for _ in range(3):
            resumed.step(dt=half_day)
            replay.append(resumed.get_state()["surface_snow_amount"].tolist())
        assert replay == observed[1:]
        assert resumed.get_forcing_cursor() is None

    def test_a_simulator_without_time_origin_refuses_the_adapter_read(self, tmp_path):
        sim = self._simulator(tmp_path, time_origin=None)
        with pytest.raises(Exception) as caught:
            sim.step(dt=43200.0)
        chain, error = [], caught.value
        while error is not None:
            chain.append(str(error))
            error = error.__cause__
        assert any("time_origin" in message for message in chain), chain
        assert sim.step_count == 0
