"""
L1 Unit Tests for RELICSimulator Core Functionality

Tests the core simulator scheduling, dependency resolution, and state management
integration without requiring complex scientific processes.

Mathematical Basis:
- Dependency Graph: DAG topological ordering ensures proper execution sequence
- State Update: x(t+dt) = x(t) + sum(updates from all functors)
- Conservation: sum(inputs) - sum(outputs) = delta(pools)

References:
- Ascher & Petzold (1998). Computer Methods for ODEs and DAEs
- Graph theory: Topological sorting for DAG execution
"""

import copy
import random

import numpy as np
import pytest
import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from relic.core.simulator import RELICSimulator
from relic.core.simulator import checkpoint as checkpoint_module
from relic.core.simulator.checkpoint import (
    capture_runtime_transaction,
    create_checkpoint,
    restore_checkpoint,
    restore_runtime_transaction,
)
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.core.state.types import StateVariableSpec
from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionError,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
)
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.state.types import ConservationSummary
from relic.parameters import ParameterSpec
from relic.scientific.anthroposphere.management.coordinator import (
    ManagementCoordinator,
)
from relic.scientific.anthroposphere.management.event_scheduler import (
    EventPriority,
    EventType,
    ManagementEvent,
)
from relic.scientific.anthroposphere.management.fire_management import (
    FireManagement,
)


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'test_pool': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


def _fresh_checkpoint_simulator(functor):
    simulator = _declared_simulator(
        {
            "time_step": 1.0,
            "max_time": 10.0,
            "enable_strict_conservation": False,
            "strict_dependencies": False,
            "device": "cpu",
        }
    )
    state_manager = StateManager()
    state_manager.initialize()
    simulator.initialize(state_manager, DependencyGraphResolver())
    simulator.register_functor(functor)
    return simulator


def _assert_checkpoint_value_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert list(left) == list(right)
        for key in left:
            _assert_checkpoint_value_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_checkpoint_value_equal(left_item, right_item)
        return
    assert left == right


class SimpleFunctor(BiologicalProcessFunctor):
    """
    Simple test functor that adds a constant value.
    
    Mathematical Expression: pool(t+dt) = pool(t) + constant
    Parameters: increment_value (default=1.0)
    References: None (algebraic test process)
    """
    def __init__(self, name="simple", increment=1.0, dependencies=None):
        meta = FunctorMetadata(
            name=name,
            description=f"Add {increment} to test_pool",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=f"pool += {increment}",
            references=["None (unit test functor)"],
            input_variables=["test_pool"],
            output_variables=["test_pool"],
        )
        super().__init__(name=name, config={"increment": increment}, metadata=meta)
        self.increment = increment
        self._dependencies = dependencies or []
        self.force_initial_execution = True
    
    def get_dependencies(self):
        return self._dependencies
    
    def forward(self, state, time, dt):
        """Return increment to test_pool."""
        return {
            "test_pool": UpdateItem(
                value=torch.tensor(self.increment, dtype=torch.float32),
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
            )
        }


class DependentFunctor(BiologicalProcessFunctor):
    """
    Test functor with dependencies.
    
    Mathematical Expression: pool(t+dt) = pool(t) + multiplier * increment
    Parameters: multiplier (default=2.0)
    References: None (algebraic test process)
    """
    def __init__(self, name="dependent", multiplier=2.0, depends_on=None):
        meta = FunctorMetadata(
            name=name,
            description=f"Multiply by {multiplier}",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=f"pool += {multiplier} * increment",
            references=["None (unit test functor)"],
            input_variables=["test_pool"],
            output_variables=["test_pool"],
        )
        super().__init__(name=name, config={"multiplier": multiplier}, metadata=meta)
        self.multiplier = multiplier
        self._depends_on = depends_on or []
    
    def get_dependencies(self):
        return self._depends_on
    
    def forward(self, state, time, dt):
        """Return multiplied increment."""
        return {
            "test_pool": UpdateItem(
                value=torch.tensor(self.multiplier, dtype=torch.float32),
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
            )
        }


class _BindingStandIn:
    """A ParameterManager stand-in holding one learnable binding (item 74)."""

    def __init__(self, value):
        self.nn_parameters = {
            "test.theta": torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
        }

    def get_parameter(self, key):
        return None

    def get(self, key, **kwargs):
        return {"validation.numerical.response.ph_optimum": 7.0}.get(key)


class _BindingHolderFunctor(BiologicalProcessFunctor):
    """Holds the stand-in's learnable binding by reference; adds nothing."""

    def __init__(self, manager, name="binding_holder"):
        meta = FunctorMetadata(
            name=name,
            description="Holds a ParameterManager binding",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            references=["None (unit test functor)"],
            input_variables=["test_pool"],
            output_variables=["test_pool"],
        )
        super().__init__(name=name, config={}, metadata=meta, parameter_manager=manager)
        self.theta = manager.nn_parameters["test.theta"]

    def get_dependencies(self):
        return []

    def forward(self, state, time, dt):
        return {
            "test_pool": UpdateItem(
                value=torch.tensor(0.0, dtype=torch.float32),
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
            )
        }


def _as_version_2(payload, attribute_of, extra_state=None):
    """The version-2 layout of a checkpoint payload: every held binding is a
    Functor state_dict entry under the attribute that holds it."""
    decode = checkpoint_module._decode_checkpoint_value
    encode = checkpoint_module._encode_checkpoint_value
    assert payload["delivery_state"]["nodes"] == {}
    body = {key: payload[key] for key in checkpoint_module._CHECKPOINT_BODY_FIELDS}
    functors = {}
    for name, entry in payload["functors"].items():
        state = dict(decode(entry["state_dict"], path="version2"))
        for key, value in decode(entry["parameter_bindings"], path="version2").items():
            state[attribute_of[key]] = value
        state.update(extra_state or {})
        functors[name] = {"state_dict": encode(state, path="version2")}
    body["functors"] = functors
    body["format_version"] = 2
    return checkpoint_module._seal_checkpoint(body, payload["delivery_state"])


def _replacement_checkpoint_kernel(self, state, time, dt):
    """Replace the kernel while preserving class, signature, and configuration."""
    return {
        "test_pool": UpdateItem(
            value=torch.tensor(2.0 * self.increment, dtype=torch.float32),
            mode=UpdateMode.ACCUMULATE,
            per_timestep=True,
        )
    }


class TestRELICSimulatorL1:
    """L1 Unit tests for RELICSimulator core functionality."""
    
    def setup_method(self):
        """Set up test simulator."""
        self.config = {
            "time_step": 1.0,
            "max_time": 10.0,
            "enable_strict_conservation": False
        }
        self.simulator = _declared_simulator(self.config)
        
        # Initialize simulator with required components
        state_manager = StateManager()
        state_manager.initialize()
        dependency_graph = DependencyGraphResolver()
        self.simulator.initialize(state_manager, dependency_graph)
    
    def test_provenance_retention_rolls_after_acceptance_and_checkpoint_replays(self, tmp_path):
        """A bounded audit window must not change accepted state or restart."""
        sim = self.simulator
        sim.config["max_provenance_records_per_variable"] = 3
        sim.output_frequency = 1000000
        sim.state_manager.max_history_length = 2
        sim.register_functor(SimpleFunctor())
        sim.initialize_state({"test_pool": torch.tensor(0.0)})
        for _ in range(12):
            sim.step(1.0)
        assert len(sim._step_provenance_detail["test_pool"]) == 3
        assert all(entry[0] == "simple" for entry in sim._step_provenance_detail["test_pool"])
        assert sim._provenance_evicted_counts == {"test_pool": 9}
        path = tmp_path / "bounded.pt"
        sim.create_checkpoint(path)
        for _ in range(7):
            sim.step(1.0)
        expected = sim.state_manager.get_state("test_pool").clone()
        expected_counts = dict(sim._provenance_evicted_counts)
        sim.restore_checkpoint(path)
        assert sim._provenance_evicted_counts == {"test_pool": 9}
        for _ in range(7):
            sim.step(1.0)
        assert torch.equal(sim.state_manager.get_state("test_pool"), expected)
        assert expected.item() == 19.0
        assert sim._provenance_evicted_counts == expected_counts == {"test_pool": 16}
        assert len(sim._step_provenance_detail["test_pool"]) == 3
        assert sim.get_variable_provenance("test_pool")[-1]["functor"] == "simple"

    def test_provenance_retention_restores_evicted_window_on_prepare_failure(self, monkeypatch):
        from relic.core.simulator import execution_loop
        sim = self.simulator
        sim.config["max_provenance_records_per_variable"] = 2
        sim.register_functor(SimpleFunctor())
        sim.initialize_state({"test_pool": torch.tensor(0.0)})
        for _ in range(4):
            sim.step(1.0)
        original = sim._step_provenance_detail["test_pool"]
        expected = list(original)
        counts = dict(sim._provenance_evicted_counts)

        def reject_after_trim(*args, **kwargs):
            assert len(sim._step_provenance_detail["test_pool"]) == 2
            assert sim._provenance_evicted_counts["test_pool"] == 3
            raise RuntimeError("injected delivery preparation failure")

        monkeypatch.setattr(execution_loop, "_prepare_external_deliveries", reject_after_trim)
        with pytest.raises(RuntimeError, match="injected delivery preparation failure"):
            sim.step(1.0)
        assert sim.step_count == 4
        assert sim.state_manager.get_state("test_pool").item() == 4.0
        assert sim._step_provenance_detail["test_pool"] is original
        assert sim._step_provenance_detail["test_pool"] == expected
        assert sim._provenance_evicted_counts == counts

    @pytest.mark.parametrize("invalid", [0, -1, True, None, 2.5, "3"])
    def test_provenance_retention_rejects_invalid_limits_before_step(self, invalid):
        self.simulator.config["max_provenance_records_per_variable"] = invalid
        with pytest.raises(ValueError, match="max_provenance_records_per_variable"):
            self.simulator.step(1.0)
        assert self.simulator.step_count == 0

    def test_default_provenance_retention_is_bounded_in_automatic_checkpoint(self, tmp_path):
        sim = self.simulator
        sim.output_frequency = 1000000
        sim.state_manager.max_history_length = 2
        sim.register_functor(SimpleFunctor())
        sim.initialize_state({"test_pool": torch.tensor(0.0)})
        for _ in range(258):
            sim.step(1.0)
        sim.enable_checkpointing = True
        sim.checkpoint_frequency = 1.0
        sim.checkpoint_directory = str(tmp_path)
        sim.step(1.0)
        path = tmp_path / sim.run_id / "checkpoint_step_000000000259.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        detail = checkpoint_module._decode_checkpoint_value(
            payload["runtime"]["_step_provenance_detail"], path="runtime._step_provenance_detail"
        )
        assert len(detail["test_pool"]) == 256
        assert payload["runtime"]["_provenance_evicted_counts"] == {"test_pool": 3}
        assert sim._provenance_evicted_counts == {"test_pool": 3}

    def test_simulator_initialization(self):
        """Test simulator initializes with correct configuration."""
        assert self.simulator is not None
        assert hasattr(self.simulator, 'state_manager')
        assert hasattr(self.simulator, 'dependency_graph')
        assert self.simulator.current_time == 0.0
        
        print("[OK] Simulator initialized successfully")

    @pytest.mark.parametrize(
        ("source_calendar", "runtime_calendar"),
        [("noleap", "noleap"), ("365_day", "noleap"), ("360_day", "360_day"),
         ("standard", "standard"), ("gregorian", "standard"),
         ("proleptic_gregorian", "proleptic_gregorian"), (" Gregorian ", "standard")],
    )
    def test_calendar_admission_preserves_source_and_uses_existing_cf_synonyms(
        self, source_calendar, runtime_calendar
    ):
        source = {"device": "cpu", "calendar_type": source_calendar,
                  "time_origin": "2020-01-02T00:00:00"}
        simulator = _declared_simulator(source)
        assert source["calendar_type"] == source_calendar
        assert simulator._get_setting("calendar_type") == source_calendar
        assert simulator._calendar_type == runtime_calendar
        assert simulator._time_origin == source["time_origin"]

    @pytest.mark.parametrize("calendar", ["standard", "gregorian", "proleptic_gregorian"])
    def test_civil_calendar_admission_validates_real_leap_origin(self, calendar):
        simulator = _declared_simulator({
            "device": "cpu", "calendar_type": calendar,
            "time_origin": "2020-02-29T23:30:00",
        })
        assert simulator._time_origin == "2020-02-29T23:30:00"

    @pytest.mark.parametrize("calendar", ["standard", "gregorian", "proleptic_gregorian"])
    def test_civil_calendar_admission_rejects_invalid_origin(self, calendar):
        with pytest.raises(ValueError, match="time_origin"):
            _declared_simulator({"device": "cpu", "calendar_type": calendar,
                                 "time_origin": "2019-02-29T00:00:00"})

    @pytest.mark.parametrize("calendar", ["standard", "gregorian"])
    def test_mixed_calendar_admission_rejects_reform_gap_origin(self, calendar):
        with pytest.raises(ValueError, match="time_origin"):
            _declared_simulator({"device": "cpu", "calendar_type": calendar,
                                 "time_origin": "1582-10-10T00:00:00"})

    @pytest.mark.parametrize("calendar", ["martian", "julian", "all_leap", "366_day", None, False])
    def test_calendar_admission_does_not_claim_unqualified_calendar_consumers(self, calendar):
        with pytest.raises(ValueError, match="calendar_type"):
            _declared_simulator({"device": "cpu", "calendar_type": calendar})

    def test_calendar_admission_reuses_the_existing_origin_validator(self, monkeypatch):
        from relic.coupling.external.common import base_adapter

        calls = []
        original = base_adapter.parse_time_origin

        def validate(origin, calendar):
            calls.append((origin, calendar))
            return original(origin, calendar)

        monkeypatch.setattr(base_adapter, "parse_time_origin", validate)
        simulator = _declared_simulator({
            "device": "cpu", "calendar_type": "gregorian",
            "time_origin": "2020-01-02T00:00:00",
        })
        assert calls == [("2020-01-02T00:00:00", "standard")]
        assert simulator._get_setting("calendar_type") == "gregorian"

    def test_simulator_owns_distinct_run_identity_and_retains_pending_delivery_on_reset(self):
        from uuid import UUID
        first_id = self.simulator.run_id
        assert UUID(first_id).version == 4
        other = _declared_simulator({"device": "cpu"})
        assert other.run_id != first_id
        self.simulator.step(dt=1.0)
        assert self.simulator.run_id == first_id
        pending = {"key": "accepted", "run_id": first_id, "snapshot": {"value": 2.0}}
        self.simulator._pending_external_deliveries.append(pending)
        self.simulator.reset()
        assert self.simulator.run_id != first_id
        assert UUID(self.simulator.run_id).version == 4
        assert self.simulator._pending_external_deliveries == [pending]

    def test_simulator_checkpoint_facade_accepts_frozen_payload(self, tmp_path):
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        self.simulator.current_time = 9.0
        path = tmp_path / "facade.pt"
        first = self.simulator.create_checkpoint(path, snapshot=payload)
        repeated = self.simulator.create_checkpoint(path, snapshot=payload)
        assert first == repeated
        assert torch.load(path, weights_only=True)["runtime"]["current_time"] == 0.0
        assert self.simulator.current_time == 9.0

    def test_checkpoint_restores_the_original_run_identity(self, tmp_path):
        from relic.core.provenance import ProvenanceRecorder
        recorder = ProvenanceRecorder(tmp_path / "provenance", run_id="standalone")
        self.simulator.state_manager.set_provenance_recorder(recorder)
        original_id = self.simulator.run_id
        assert recorder.run_id == original_id
        path = tmp_path / "run.pt"
        self.simulator.create_checkpoint(path)
        self.simulator.reset()
        assert self.simulator.run_id != original_id
        assert recorder.run_id == self.simulator.run_id
        self.simulator.restore_checkpoint(path)
        assert self.simulator.run_id == original_id
        assert recorder.run_id == original_id

    @pytest.mark.parametrize("run_id", [None, "", False])
    def test_checkpoint_rejects_invalid_run_identity_before_output(self, tmp_path, run_id):
        before = self.simulator.run_id
        with pytest.raises((TypeError, ValueError), match="run_id"):
            self.simulator.run_id = run_id
            self.simulator.create_checkpoint(tmp_path / "invalid_run.pt")
        assert self.simulator.run_id == before
        assert not (tmp_path / "invalid_run.pt").exists()

    @pytest.mark.parametrize("route", ["initialize", "initialize_state", "set_state_manager", "late"])
    def test_all_manager_attachment_routes_bind_recorder_identity(self, tmp_path, route):
        from relic.core.provenance import ProvenanceRecorder
        simulator = _declared_simulator({"device": "cpu"})
        recorder = ProvenanceRecorder(tmp_path, run_id="standalone")
        if route in ("initialize", "set_state_manager"):
            manager = StateManager()
            manager.set_provenance_recorder(recorder)
            if route == "initialize":
                simulator.initialize(manager, DependencyGraphResolver())
            else:
                simulator.set_state_manager(manager)
        else:
            if route == "late":
                simulator = self.simulator
            else:
                simulator.initialize_state({"test_pool": torch.tensor(0.0)})
            simulator.state_manager.set_provenance_recorder(recorder)
        assert recorder.run_id == simulator.run_id

    def test_failed_restore_rolls_back_the_recorders_run_identity(self, tmp_path, monkeypatch):
        from relic.core.provenance import ProvenanceRecorder
        recorder = ProvenanceRecorder(tmp_path / "provenance", run_id="standalone")
        self.simulator.state_manager.set_provenance_recorder(recorder)
        saved_id = self.simulator.run_id
        path = tmp_path / "identity.pt"
        self.simulator.create_checkpoint(path)
        self.simulator.reset()
        current_id = self.simulator.run_id
        apply = checkpoint_module._apply_runtime_fields
        def fail_after_saved_identity(simulator, runtime, state_manager):
            apply(simulator, runtime, state_manager)
            if runtime["run_id"] == saved_id:
                raise RuntimeError("injected restore identity failure")
        monkeypatch.setattr(checkpoint_module, "_apply_runtime_fields", fail_after_saved_identity)
        with pytest.raises(RuntimeError, match="identity failure"):
            self.simulator.restore_checkpoint(path)
        assert self.simulator.run_id == recorder.run_id == current_id

    @pytest.mark.parametrize("change", ["run", "manager"])
    def test_active_step_cannot_replace_its_run_or_manager_owner(self, change):
        before_id = self.simulator.run_id
        before_manager = self.simulator.state_manager
        before_manager.begin_provenance_transaction()
        with pytest.raises(RuntimeError, match="active|transaction"):
            if change == "run":
                self.simulator.run_id = "another-run"
            else:
                self.simulator.state_manager = StateManager()
        assert self.simulator.run_id == before_id
        assert self.simulator.state_manager is before_manager
        before_manager.rollback_provenance_transaction()

    def test_pending_automatic_checkpoint_time_is_frozen_without_live_commit(self):
        self.simulator.current_time = 4.0
        payload = checkpoint_module.capture_checkpoint_payload(
            self.simulator, accepted_checkpoint_time=4.0,
        )
        assert payload["runtime"]["last_checkpoint_time"] == 4.0
        assert self.simulator.last_checkpoint_time == 0.0
        assert payload["runtime"]["run_id"] == self.simulator.run_id

    @pytest.mark.parametrize("timestamp", [3.0, float("nan"), True])
    def test_pending_checkpoint_time_must_identify_the_captured_state(self, timestamp):
        self.simulator.current_time = 4.0
        with pytest.raises((TypeError, ValueError), match="checkpoint time"):
            checkpoint_module.capture_checkpoint_payload(
                self.simulator, accepted_checkpoint_time=timestamp,
            )

    @pytest.mark.parametrize("auto_report", [True, False])
    def test_checkpoint_preparation_projects_commit_without_mutating_live_state(self, auto_report, tmp_path):
        from relic.core.provenance import ProvenanceRecorder
        manager = self.simulator.state_manager
        manager.set_provenance_recorder(ProvenanceRecorder(tmp_path / "provenance"))
        manager.begin_provenance_transaction()
        manager.set_auto_report_enabled(False)
        manager.apply_updates({"test_pool": torch.tensor(1.0)}, check_conservation=False)
        self.simulator._step_update_sources = {"test_pool": ["source"]}
        before = manager.checkpoint()
        with pytest.raises(RuntimeError, match="stable checkpoint boundary"):
            checkpoint_module.capture_checkpoint_payload(self.simulator)
        prepared = checkpoint_module.capture_checkpoint_payload(
            self.simulator, pending_commit=True,
            accepted_auto_report_enabled=auto_report,
        )
        _assert_checkpoint_value_equal(before, manager.checkpoint())
        assert self.simulator._step_update_sources == {"test_pool": ["source"]}
        assert prepared["state_manager"]["provenance_transaction_active"] is False
        expected_events = copy.deepcopy(before["pending_provenance_events"])
        for event in expected_events:
            for key in ("pre_value", "post_value"):
                event[key] = event[key].cpu()
        _assert_checkpoint_value_equal(
            prepared["state_manager"]["pending_provenance_events"],
            expected_events,
        )
        assert prepared["state_manager"]["auto_report_enabled"] is auto_report
        assert prepared["runtime"]["_step_update_sources"] is None
        manager.commit_provenance_transaction()
        manager.set_auto_report_enabled(auto_report)
        self.simulator._step_update_sources = None
        accepted = checkpoint_module.capture_checkpoint_payload(self.simulator)
        for payload in (prepared, accepted):
            payload.pop("created_utc")
            payload.pop("integrity")
            payload["delivery_state"].pop("root_id")
        # The prepared snapshot owns notifications whose delivery happened later.
        prepared["state_manager"]["pending_provenance_events"] = []
        _assert_checkpoint_value_equal(prepared, accepted)

    def test_accepted_notification_checkpoint_restores_original_destination(self, tmp_path, monkeypatch):
        import json
        from relic.core.provenance import ProvenanceRecorder
        from relic.core.provenance import recorder as recorder_module

        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        self.simulator.register_functor(SimpleFunctor(name="increment"))
        destination = tmp_path / "provenance"
        self.simulator.state_manager.set_provenance_recorder(ProvenanceRecorder(destination))

        def fail_log(path, event):
            raise OSError("notification log unavailable")

        with monkeypatch.context() as local:
            local.setattr(recorder_module, "_append_runtime_event", fail_log)
            with pytest.raises(RuntimeError, match="notification log unavailable"):
                self.simulator.step(dt=1.0)
        assert self.simulator.step_count == 1
        assert len(self.simulator.state_manager._pending_provenance_events) == 1
        path = tmp_path / "accepted.pt"
        self.simulator.create_checkpoint(path)
        prov_before = (destination / "prov.jsonld").read_bytes()
        target = _declared_simulator(self.config)
        target.initialize(StateManager(), DependencyGraphResolver())
        target.register_functor(SimpleFunctor(name="increment"))
        target.restore_checkpoint(path)
        assert target.run_id == self.simulator.run_id
        assert target.state_manager._provenance_recorder is not None
        assert (destination / "prov.jsonld").read_bytes() == prov_before
        assert not (destination / "runtime.log.jsonl").exists()
        target.retry_external_deliveries()
        assert target.state_manager._pending_provenance_events == []
        assert target.step_count == 1
        assert float(target.state_manager.get_state("test_pool")) == 1.0
        target.step(dt=1.0)
        records = [json.loads(line) for line in (destination / "runtime.log.jsonl").read_text().splitlines()]
        assert len(records) == 2
        assert {event["run_id"] for event in records} == {self.simulator.run_id}
        assert float(target.state_manager.get_state("test_pool")) == 2.0

    @pytest.mark.parametrize("version", [True, float(checkpoint_module.CHECKPOINT_FORMAT_VERSION)])
    def test_checkpoint_version_requires_an_exact_integer(self, tmp_path, version):
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        payload["format_version"] = version
        payload["integrity"]["payload_sha256"] = checkpoint_module._payload_sha256({
            key: value for key, value in payload.items() if key != "integrity"
        })
        with pytest.raises((TypeError, ValueError), match="version"):
            create_checkpoint(object(), tmp_path / "invalid.pt", snapshot=payload)
        assert not (tmp_path / "invalid.pt").exists()

    def test_cancelled_restore_keeps_unbound_accepted_notifications_without_delivery(self, tmp_path, monkeypatch):
        from relic.core.provenance import ProvenanceRecorder

        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        path = tmp_path / "before.pt"
        self.simulator.create_checkpoint(path)
        manager = self.simulator.state_manager
        destination = tmp_path / "provenance"
        manager.set_provenance_recorder(ProvenanceRecorder(destination))
        manager.begin_provenance_transaction()
        manager.apply_updates({"test_pool": torch.tensor(1.0)}, check_conservation=False)
        accepted = manager.checkpoint()
        accepted["provenance_transaction_active"] = False
        manager.rollback_provenance_transaction()
        manager.set_provenance_recorder(None)
        manager.restore(accepted)
        before = capture_runtime_transaction(self.simulator, for_rollback=True)
        restore_rng = checkpoint_module._restore_rng_state
        calls = 0

        def cancel_once(rng_state, simulator):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt("cancelled restore")
            return restore_rng(rng_state, simulator)

        monkeypatch.setattr(checkpoint_module, "_restore_rng_state", cancel_once)
        with pytest.raises(KeyboardInterrupt, match="cancelled restore"):
            self.simulator.restore_checkpoint(path)
        assert manager._provenance_recorder is None
        assert not destination.exists()
        _assert_checkpoint_value_equal(
            capture_runtime_transaction(self.simulator, for_rollback=True), before,
        )

    @pytest.mark.parametrize("alias", ["dot", "case"])
    @pytest.mark.parametrize("conflict", [True, False])
    def test_restore_rejects_notification_conflict_at_same_physical_destination(self, tmp_path, alias, conflict):
        import os
        from relic.core.provenance import ProvenanceRecorder

        if alias == "case" and os.name != "nt":
            pytest.skip("Windows case-insensitive filesystem contract")
        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        manager = self.simulator.state_manager
        destination = tmp_path / "Provenance"
        manager.set_provenance_recorder(ProvenanceRecorder(destination))
        manager.begin_provenance_transaction()
        manager.apply_updates({"test_pool": torch.tensor(1.0)}, check_conservation=False)
        payload = checkpoint_module.capture_checkpoint_payload(
            self.simulator, pending_commit=True, accepted_auto_report_enabled=True,
        )
        path = tmp_path / "pending.pt"
        create_checkpoint(object(), path, snapshot=payload)
        accepted = manager.checkpoint()
        accepted["provenance_transaction_active"] = False
        event = accepted["pending_provenance_events"][0]
        if conflict:
            event["post_value"].fill_(2.0)
        event["notification_context"]["output_dir"] = (
            str(destination / ".." / destination.name) if alias == "dot"
            else str(destination).swapcase()
        )
        manager.restore(accepted)
        before = capture_runtime_transaction(self.simulator, for_rollback=True)
        if conflict:
            with pytest.raises(ValueError, match="conflicts with an accepted notification"):
                self.simulator.restore_checkpoint(path)
            _assert_checkpoint_value_equal(
                capture_runtime_transaction(self.simulator, for_rollback=True), before,
            )
        else:
            self.simulator.restore_checkpoint(path)
            assert len(manager._pending_provenance_events) == 1
            assert manager._pending_provenance_events[0]["notification_context"] == event["notification_context"]
        assert not destination.exists()

    @pytest.mark.parametrize("operation", ["capture", "restore", "restore_live"])
    def test_public_local_transaction_requires_confirmed_notifications(self, operation):
        class Observer:
            calls = 0

            def record_state_update(self, *args):
                self.calls += 1

        observer = Observer()
        manager = self.simulator.state_manager
        manager.set_provenance_recorder(observer)
        image = capture_runtime_transaction(self.simulator)
        manager.begin_provenance_transaction()
        manager.apply_updates({"test_pool": torch.tensor(1.0)}, check_conservation=False)
        accepted = manager.checkpoint()
        accepted["provenance_transaction_active"] = False
        manager.restore(accepted)
        if operation == "capture":
            with pytest.raises(RuntimeError, match="pending|confirmed"):
                capture_runtime_transaction(self.simulator)
            assert observer.calls == 0
            assert len(manager._pending_provenance_events) == 1
        elif operation == "restore_live":
            with pytest.raises(RuntimeError, match="pending|confirmed"):
                restore_runtime_transaction(self.simulator, image)
            assert len(manager._pending_provenance_events) == 1
            assert observer.calls == 0
        else:
            image["state_manager"] = checkpoint_module._encode_checkpoint_value(accepted, path="state_manager")
            manager.commit_provenance_transaction()
            assert observer.calls == 1
            with pytest.raises(RuntimeError, match="pending|confirmed"):
                restore_runtime_transaction(self.simulator, image)
            assert manager._pending_provenance_events == []
            assert observer.calls == 1

    def test_pending_notification_replays_idempotently_in_fresh_processes(self, tmp_path, monkeypatch):
        import json
        import subprocess
        import sys
        from relic.core.provenance import ProvenanceRecorder
        from relic.core.provenance import recorder as recorder_module

        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        destination = tmp_path / "provenance"
        manager = self.simulator.state_manager
        manager.set_provenance_recorder(ProvenanceRecorder(destination))

        def interrupt_log(path, event):
            raise OSError("log unavailable")

        with monkeypatch.context() as local:
            local.setattr(recorder_module, "_append_runtime_event", interrupt_log)
            with pytest.raises(RuntimeError, match="log unavailable"):
                manager.apply_updates({"test_pool": torch.tensor(1.0)}, check_conservation=False)
        path = tmp_path / "pending.pt"
        self.simulator.create_checkpoint(path)
        script = '''
import json
import sys
import torch
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.core.dependency_graph import DependencyGraphResolver
simulator = RELICSimulator(json.loads(sys.argv[2]))
manager = StateManager()
manager.initialize()
simulator.initialize(manager, DependencyGraphResolver())
simulator.initialize_state({"test_pool": torch.tensor(0.0)})
simulator.restore_checkpoint(sys.argv[1])
simulator.retry_external_deliveries()
print(json.dumps({"run_id": simulator.run_id, "value": float(manager.get_state("test_pool")), "counter": manager._update_counter, "pending": len(manager._pending_provenance_events)}))
'''
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-c", script, str(path), json.dumps(dict(self.config, state_units=_TEST_STATE_UNITS))],
                check=True, capture_output=True, text=True, encoding="utf-8", timeout=120,
            )
            output = json.loads(result.stdout.strip().splitlines()[-1])
            assert output == {"run_id": self.simulator.run_id, "value": 1.0, "counter": 1, "pending": 0}
        assert len((destination / "runtime.log.jsonl").read_text().splitlines()) == 1

    @pytest.mark.parametrize("saved_attempts,live_attempts", [(5, 1), (1, 5)])
    def test_checkpoint_restores_pending_checkpoint_delivery_without_recapture(self, tmp_path, monkeypatch, saved_attempts, live_attempts):
        config = dict(self.config, enable_checkpointing=True, checkpoint_frequency=1.0,
                      checkpoint_directory=str(tmp_path / "deliveries"))

        def make_simulator():
            simulator = _declared_simulator(config)
            manager = StateManager()
            manager.initialize({"test_pool": torch.tensor(0.0)})
            simulator.initialize(manager, DependencyGraphResolver())
            simulator.register_functor(SimpleFunctor(name="increment"))
            return simulator

        simulator = make_simulator()
        local_image = capture_runtime_transaction(simulator)
        def fail_write(*args, **kwargs):
            raise OSError("delivery unavailable")

        with monkeypatch.context() as local:
            local.setattr(checkpoint_module.torch, "save", fail_write)
            with pytest.raises(OSError, match="delivery unavailable"):
                simulator.step(dt=1.0)
        original_job = simulator._pending_external_deliveries[0]
        original_job.update(attempts=saved_attempts, error=f"failed-{saved_attempts}")
        original_digest = original_job["snapshot"]["integrity"]["payload_sha256"]
        recovery = tmp_path / "recovery.pt"
        simulator.create_checkpoint(recovery)
        restored = make_simulator()
        live_job = copy.deepcopy(original_job)
        live_job.update(attempts=live_attempts, error=f"failed-{live_attempts}")
        restored._pending_external_deliveries.append(live_job)
        restored.restore_checkpoint(recovery)
        assert len(restored._pending_external_deliveries) == 1
        restored_job = restored._pending_external_deliveries[0]
        assert restored_job["key"] == original_job["key"]
        assert restored_job["attempts"] == max(saved_attempts, live_attempts)
        assert restored_job["error"] == f"failed-{max(saved_attempts, live_attempts)}"
        assert restored_job["snapshot"]["integrity"]["payload_sha256"] == original_digest
        with pytest.raises(RuntimeError, match="pending|confirmed"):
            capture_runtime_transaction(restored)
        with pytest.raises(RuntimeError, match="pending|confirmed"):
            restore_runtime_transaction(restored, local_image)
        assert not checkpoint_module.Path(original_job["resource"]).exists()
        with monkeypatch.context() as local:
            local.setattr(checkpoint_module, "capture_checkpoint_payload", lambda *a, **k: pytest.fail("Frozen delivery must not recapture"))
            restored.retry_external_deliveries()
        delivered = torch.load(original_job["resource"], map_location="cpu", weights_only=True)
        assert delivered["integrity"]["payload_sha256"] == original_digest
        assert delivered["runtime"]["run_id"] == simulator.run_id
        assert float(delivered["state_manager"]["state"]["test_pool"]) == 1.0
        assert restored.step_count == 1
        assert float(restored.state_manager.get_state("test_pool")) == 1.0
        assert restored._pending_external_deliveries == []
        assert len(restored._last_external_delivery_receipts) == 1
        assert restored.retry_external_deliveries() == []

    def test_each_checkpoint_root_preserves_the_entire_frozen_delivery_batch(self, tmp_path):
        from relic.core.simulator.diagnostics import capture_diagnostic_export
        from relic.core.state.types import ConservationSummary

        self.simulator.state_manager._conservation_history.append(
            ConservationSummary(timestamp=0.0, balance_error={"C": 1.0}, total_violations=0, tolerance_value=1e-6),
        )
        diagnostic = capture_diagnostic_export(self.simulator, output_path=tmp_path / "diagnostic.nc")
        jobs = []
        for key in ("a", "b", "diagnostic"):
            jobs.append({
                "key": key, "run_id": self.simulator.run_id, "accepted_step": 0,
                "accepted_time": 0.0, "operation": "diagnostic" if key == "diagnostic" else "checkpoint",
                "resource": str(tmp_path / ("diagnostic.nc" if key == "diagnostic" else f"{key}.pt")),
                "parameters": {}, "sources": [{"request_id": key}], "automatic": False,
                "snapshot": diagnostic if key == "diagnostic" else None, "attempts": 0, "error": None,
            })
        manager = self.simulator.state_manager
        manager.begin_provenance_transaction()
        batch = checkpoint_module.capture_checkpoint_payload(
            self.simulator, pending_commit=True, accepted_auto_report_enabled=True,
            pending_deliveries=jobs,
        )
        manager.rollback_provenance_transaction()
        snapshots = {key: checkpoint_module.checkpoint_delivery_snapshot(batch, key) for key in ("a", "b")}
        for key, snapshot in snapshots.items():
            state = snapshot["delivery_state"]
            assert state["root_id"] == key
            assert set(state["nodes"]) == ({"a", "b"} - {key})
            assert [item["key"] for item in state["pending"]] == ["a", "b", "diagnostic"]
            assert all("delivery_state" not in node["body"] for node in state["nodes"].values())
        independent = checkpoint_module.checkpoint_delivery_snapshot(batch, "a")
        independent["state_manager"]["update_counter"] = 999
        assert snapshots["b"]["state_manager"]["update_counter"] == 0
        assert batch["state_manager"]["update_counter"] == 0
        create_checkpoint(object(), tmp_path / "a.pt", snapshot=snapshots["a"])
        for key in ("a", "b"):
            self.simulator.restore_checkpoint(tmp_path / f"{key}.pt")
            assert len(self.simulator._pending_external_deliveries) == 3
            self.simulator.retry_external_deliveries()
            assert self.simulator._pending_external_deliveries == []
            assert len(self.simulator._last_external_delivery_receipts) == 3
            assert self.simulator.step_count == 0
            for node_key in ("a", "b"):
                artifact = torch.load(tmp_path / f"{node_key}.pt", weights_only=True)
                assert artifact["integrity"] == snapshots[node_key]["integrity"]
        import xarray as xr
        with xr.open_dataset(tmp_path / "diagnostic.nc") as dataset:
            assert dataset.attrs["relic_delivery_key"] == "diagnostic"
            assert any(float(variable.max()) == 1.0 for variable in dataset.data_vars.values())

    @pytest.mark.parametrize("damage", ["receipt", "missing_job", "orphan_node", "root_duplicate"])
    def test_checkpoint_rejects_malformed_delivery_tables_before_publication(self, tmp_path, damage):
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        state = payload["delivery_state"]
        if damage == "receipt":
            state["receipts"] = [1]
        elif damage == "missing_job":
            state["pending"] = [{"key": "missing", "attempts": 0, "error": None}]
        else:
            key = state["root_id"] if damage == "root_duplicate" else "orphan"
            state["nodes"][key] = {
                "body": {field: payload[field] for field in checkpoint_module._CHECKPOINT_BODY_FIELDS},
                "pending": [], "receipts": [],
            }
        payload["integrity"]["payload_sha256"] = checkpoint_module._payload_sha256({
            key: value for key, value in payload.items() if key != "integrity"
        })
        with pytest.raises((TypeError, ValueError), match="receipt|reference|unreachable|duplicated"):
            create_checkpoint(object(), tmp_path / "invalid.pt", snapshot=payload)
        assert not (tmp_path / "invalid.pt").exists()

    @pytest.mark.parametrize("damage", ["context", "run", "destination", "step", "value", "fields"])
    def test_checkpoint_rejects_malformed_persistent_notification(self, tmp_path, damage):
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        event = {
            "step": 0, "variable": "test_pool", "unit": "1",
            "pre_value": torch.tensor(0.0), "post_value": torch.tensor(1.0),
            "functor_increments": {"increment": torch.tensor(1.0)},
            "notification_context": {"run_id": self.simulator.run_id, "output_dir": str(tmp_path)},
        }
        if damage == "context":
            event.pop("notification_context")
        elif damage == "run":
            event["notification_context"]["run_id"] = ""
        elif damage == "destination":
            event["notification_context"]["output_dir"] = "relative/path"
        elif damage == "step":
            event["step"] = True
        elif damage == "value":
            event["post_value"] = "not a tensor"
        else:
            event["unknown"] = True
        payload["state_manager"]["pending_provenance_events"] = [event]
        payload["integrity"]["payload_sha256"] = checkpoint_module._payload_sha256({
            key: value for key, value in payload.items() if key != "integrity"
        })
        with pytest.raises((TypeError, ValueError), match="provenance notification"):
            create_checkpoint(object(), tmp_path / "invalid.pt", snapshot=payload)
        assert not (tmp_path / "invalid.pt").exists()

    @pytest.mark.parametrize("options", [
        {"pending_commit": 1},
        {"pending_commit": True},
        {"pending_commit": True, "accepted_auto_report_enabled": 1},
        {"accepted_auto_report_enabled": True},
    ])
    def test_checkpoint_preparation_requires_explicit_boolean_contract(self, options):
        self.simulator.state_manager.begin_provenance_transaction()
        with pytest.raises((TypeError, ValueError), match="pending_commit|auto_report"):
            checkpoint_module.capture_checkpoint_payload(self.simulator, **options)

    def test_checkpoint_preparation_requires_active_transaction(self):
        with pytest.raises(RuntimeError, match="active provenance transaction"):
            checkpoint_module.capture_checkpoint_payload(
                self.simulator, pending_commit=True, accepted_auto_report_enabled=True,
            )

    def test_simulator_diagnostic_facade_returns_frozen_delivery_receipt(self, tmp_path):
        from relic.core.simulator.diagnostics import capture_diagnostic_export
        self.simulator.state_manager._conservation_history.append(
            ConservationSummary(timestamp=1.0, balance_error={"C": 0.0002}, total_violations=0, tolerance_value=1e-6)
        )
        path = tmp_path / "facade.nc"
        snapshot = capture_diagnostic_export(self.simulator, output_path=path)
        result = self.simulator._export_diagnostics(snapshot=snapshot, delivery_key="facade")
        assert result.validation_passed and result.delivery_key == "facade"
        assert result.output_path == path

    def test_failed_step_restores_functor_state_dict(self):
        """A failed accepted-step transaction must restore mutable buffers."""

        class WindowBufferFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="window_buffer")
                self.register_buffer("window_total", torch.zeros(()))

            def forward(self, state, time, dt):
                self.window_total.add_(1.0)
                return super().forward(state, time, dt)

        class FailingFunctor(ScaleCouplingFunctor):
            def __init__(self):
                metadata = FunctorMetadata(
                    name="fail_after_window",
                    description="Raise after the mutable buffer changes",
                    category=ProcessCategory.SCALE_COUPLING,
                    execution_phase=ExecutionPhase.POSTPROCESSING,
                    mathematical_expression="failure injection",
                    references=["None (transaction unit test)"],
                    input_variables=["test_pool"],
                    output_variables=["diagnostics.transaction.failure"],
                )
                super().__init__(
                    name="fail_after_window",
                    config={},
                    metadata=metadata,
                    source_scale="grid_cell",
                    target_scale="grid_cell",
                    coupling_type="spatial",
                )

            def get_dependencies(self):
                return ["window_buffer"]

            def forward(self, state, time, dt):
                raise RuntimeError("injected step failure")

        simulator = _declared_simulator(
            {
                "time_step": 1.0,
                "max_time": 1.0,
                "enable_strict_conservation": False,
            }
        )
        manager = StateManager()
        manager.initialize({"test_pool": torch.tensor(0.0)})
        simulator.initialize(manager, DependencyGraphResolver())
        window = WindowBufferFunctor()
        simulator.register_functor(window, execution_group="fast")
        simulator.register_functor(FailingFunctor(), execution_group="fast")

        with pytest.raises(ExecutionError, match="injected step failure"):
            simulator.step(dt=1.0)

        torch.testing.assert_close(
            window.window_total,
            torch.zeros_like(window.window_total),
        )
        restored_pool = simulator.state_manager.get_state()["test_pool"]
        torch.testing.assert_close(
            restored_pool,
            torch.zeros_like(restored_pool),
        )
        assert simulator.current_time == pytest.approx(0.0)
        assert simulator.step_count == 0

    def test_restore_checkpoint_rejects_legacy_payload(self, tmp_path):
        """Restore must not enter an unsafe multi-version compatibility path."""
        checkpoint_path = tmp_path / "legacy.pt"
        torch.save(
            {"current_time": 12.5, "dt": 2.0, "step_count": 3},
            checkpoint_path,
        )

        with pytest.raises(ValueError, match="top-level field set"):
            self.simulator.restore_checkpoint(str(checkpoint_path))

    def test_restore_checkpoint_uses_standard_functor_state_dict(self, tmp_path):
        """Simulator restore must pass state mappings to torch Module semantics."""
        class ParameterFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="checkpoint_functor")
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

        functor = ParameterFunctor()
        self.simulator.register_functor(functor)
        checkpoint_path = tmp_path / "functor_state.pt"
        with torch.no_grad():
            functor.weight.fill_(7.0)
        create_checkpoint(self.simulator, checkpoint_path)
        with torch.no_grad():
            functor.weight.fill_(1.0)

        self.simulator.restore_checkpoint(str(checkpoint_path))

        assert float(functor.weight.detach()) == pytest.approx(7.0)

    def test_checkpoint_saves_held_manager_bindings_through_the_manager(self, tmp_path):
        """Item 74: a held ParameterManager binding is no Functor state_dict entry;
        the checkpoint saves its live value through the manager (the Functor entry's
        parameter_bindings) and restore writes it back in place, so the Functor and
        the manager keep one object."""
        manager = _BindingStandIn(2.0)
        functor = _BindingHolderFunctor(manager)
        self.simulator.register_functor(functor)
        binding = manager.nn_parameters["test.theta"]
        with torch.no_grad():
            binding.fill_(3.25)
        path = tmp_path / "held.pt"
        receipt = create_checkpoint(self.simulator, path)
        assert receipt["format_version"] == checkpoint_module.CHECKPOINT_FORMAT_VERSION == 3
        entry = torch.load(path, map_location="cpu", weights_only=True)["functors"]["binding_holder"]
        assert set(entry) == {"state_dict", "parameter_bindings"}
        assert "theta" not in checkpoint_module._decode_checkpoint_value(entry["state_dict"], path="t")
        saved = checkpoint_module._decode_checkpoint_value(entry["parameter_bindings"], path="t")
        assert list(saved) == ["test.theta"]
        assert float(saved["test.theta"]) == 3.25
        with torch.no_grad():
            binding.fill_(9.0)
        receipt = self.simulator.restore_checkpoint(str(path))
        assert float(binding.detach()) == 3.25
        assert functor.theta is binding
        assert manager.nn_parameters["test.theta"] is binding
        assert "format_upgrade" not in receipt

    def test_runtime_transaction_restores_held_manager_bindings(self):
        """The in-memory transaction of an evaluation restores held bindings too."""
        manager = _BindingStandIn(2.0)
        functor = _BindingHolderFunctor(manager)
        self.simulator.register_functor(functor)
        binding = manager.nn_parameters["test.theta"]
        transaction = capture_runtime_transaction(self.simulator)
        with torch.no_grad():
            binding.fill_(9.0)
        restore_runtime_transaction(self.simulator, transaction)
        assert float(binding.detach()) == 2.0
        assert functor.theta is binding

    def test_version_2_checkpoint_moves_held_bindings_to_the_manager(self, tmp_path):
        """Data continuity: version 2 stored a held binding as a Functor state_dict
        entry. Restore reads it, writes the entry whose target attribute holds a
        ParameterManager binding into that binding through the manager, and
        records the upgrade on the receipt. Only the current version is written."""
        manager = _BindingStandIn(2.0)
        functor = _BindingHolderFunctor(manager)
        self.simulator.register_functor(functor)
        binding = manager.nn_parameters["test.theta"]
        with torch.no_grad():
            binding.fill_(3.25)
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        version_2 = _as_version_2(payload, {"test.theta": "theta"})
        path = tmp_path / "version2.pt"
        torch.save(version_2, path)
        with pytest.raises(ValueError, match="current checkpoint format version"):
            create_checkpoint(self.simulator, tmp_path / "rewrite.pt", snapshot=version_2)
        with torch.no_grad():
            binding.fill_(9.0)
        receipt = self.simulator.restore_checkpoint(str(path))
        assert float(binding.detach()) == 3.25
        assert functor.theta is binding
        assert receipt["format_upgrade"] == {
            "from_format_version": 2,
            "to_format_version": 3,
            "moved_bindings": {"binding_holder": {"theta": "test.theta"}},
        }

    def test_version_2_entry_that_cannot_be_migrated_is_rejected(self, tmp_path):
        """A version-2 state entry that is neither module state of the target nor a
        binding it holds cannot be migrated: restore raises before any change."""
        manager = _BindingStandIn(2.0)
        functor = _BindingHolderFunctor(manager)
        self.simulator.register_functor(functor)
        binding = manager.nn_parameters["test.theta"]
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        path = tmp_path / "version2_extra.pt"
        torch.save(
            _as_version_2(
                payload, {"test.theta": "theta"},
                extra_state={"retired_weight": torch.tensor(1.0)},
            ),
            path,
        )
        with torch.no_grad():
            binding.fill_(9.0)
        with pytest.raises(ValueError, match="cannot be migrated"):
            self.simulator.restore_checkpoint(str(path))
        assert float(binding.detach()) == 9.0

    def test_checkpoint_roundtrip_supports_safe_functor_extra_state(
        self,
        tmp_path,
    ):
        """PyTorch extra state must use the same safe grammar in both directions."""

        class ExtraStateFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="extra_state_functor")
                self.counter = 7
                self.local_rng = torch.Generator(device="cpu")
                self.local_rng.manual_seed(123)

            def get_extra_state(self):
                return {
                    "counter": self.counter,
                    "local_rng_state": self.local_rng.get_state(),
                }

            def set_extra_state(self, state):
                self.counter = int(state["counter"])
                self.local_rng.set_state(state["local_rng_state"])

        functor = ExtraStateFunctor()
        self.simulator.register_functor(functor)
        checkpoint_path = tmp_path / "extra_state.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        expected_draw = torch.rand(6, generator=functor.local_rng)
        functor.counter = 99
        functor.local_rng.manual_seed(999)

        restore_checkpoint(self.simulator, checkpoint_path)
        actual_draw = torch.rand(6, generator=functor.local_rng)

        assert functor.counter == 7
        assert torch.equal(actual_draw, expected_draw)

    def test_checkpoint_restores_dynamic_fire_extra_state_into_fresh_functor(
        self,
        tmp_path,
    ):
        source_functor = FireManagement(config={"seed": 123})
        source_functor.fire_events.append(
            copy.deepcopy(source_functor.fire_events[0])
        )
        source_functor.fire_events[-1].date += 1
        source_simulator = _fresh_checkpoint_simulator(source_functor)
        target_functor = FireManagement(config={"seed": 123})
        target_simulator = _fresh_checkpoint_simulator(target_functor)
        for dependency in ("aboveground_biomass", "temperature"):
            source_simulator.register_functor(SimpleFunctor(name=dependency))
            target_simulator.register_functor(SimpleFunctor(name=dependency))
        checkpoint_path = tmp_path / "dynamic_fire_extra_state.pt"
        create_checkpoint(source_simulator, checkpoint_path)
        expected = copy.deepcopy(source_functor.state_dict())

        restore_checkpoint(target_simulator, checkpoint_path)

        _assert_checkpoint_value_equal(target_functor.state_dict(), expected)

    def test_checkpoint_restores_dynamic_coordinator_events_into_fresh_functor(
        self,
        tmp_path,
    ):
        source_functor = ManagementCoordinator()
        event = ManagementEvent(
            event_id="checkpoint_irrigation",
            event_type=EventType.IRRIGATION,
            scheduled_time=3600.0,
            priority=EventPriority.NORMAL,
            duration=60.0,
            parameters={"water_amount": 1.0},
            resource_requirements={"water": 1.0},
        )
        assert source_functor.scheduler.schedule_event(event) is True
        source_simulator = _fresh_checkpoint_simulator(source_functor)
        target_functor = ManagementCoordinator()
        target_simulator = _fresh_checkpoint_simulator(target_functor)
        checkpoint_path = tmp_path / "dynamic_coordinator_extra_state.pt"
        create_checkpoint(source_simulator, checkpoint_path)
        expected = copy.deepcopy(source_functor.state_dict())

        restore_checkpoint(target_simulator, checkpoint_path)

        _assert_checkpoint_value_equal(target_functor.state_dict(), expected)

    def test_checkpoint_capture_uses_resolved_global_cuda_device(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A resolved CUDA runtime must require CUDA RNG even without tensors."""
        checkpoint_path = tmp_path / "resolved_global_cuda.pt"
        self.simulator.device = torch.device("cuda")
        monkeypatch.setattr(
            checkpoint_module,
            "is_cuda_available",
            lambda: False,
        )

        with pytest.raises(RuntimeError, match="CUDA runtime is unavailable"):
            create_checkpoint(self.simulator, checkpoint_path)

        assert not checkpoint_path.exists()

    def test_checkpoint_capture_uses_resolved_functor_cuda_device(
        self,
        tmp_path,
        monkeypatch,
    ):
        """An empty-state Functor CUDA override still requires CUDA RNG."""
        checkpoint_path = tmp_path / "resolved_functor_cuda.pt"
        self.simulator.device = torch.device("cpu")
        functor = SimpleFunctor(name="cuda_functor")
        self.simulator.register_functor(functor)
        self.simulator._device_overrides[functor.name] = torch.device("cuda")
        monkeypatch.setattr(
            checkpoint_module,
            "is_cuda_available",
            lambda: False,
        )

        with pytest.raises(RuntimeError, match="CUDA runtime is unavailable"):
            create_checkpoint(self.simulator, checkpoint_path)

        assert not checkpoint_path.exists()

    def test_checkpoint_restore_rejects_resolved_device_mismatch_before_mutation(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Resolved runtime devices are part of restart compatibility."""
        checkpoint_path = tmp_path / "resolved_device.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        self.simulator.current_time = 3.0
        source_device = self.simulator.get_device_info()["device"]
        mismatched_device = torch.device(
            "cpu" if source_device.type == "cuda" else "cuda"
        )
        monkeypatch.setattr(
            self.simulator,
            "get_device_info",
            lambda: {
                "device": mismatched_device,
                "device_type": mismatched_device.type,
                "device_index": mismatched_device.index,
                "distributed_enabled": False,
                "rank": None,
                "world_size": None,
            },
        )

        with pytest.raises(ValueError, match="runtime environment"):
            restore_checkpoint(self.simulator, checkpoint_path)

        assert self.simulator.current_time == pytest.approx(3.0)

    def test_checkpoint_create_rejects_active_multi_rank_runtime(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The single-file checkpoint format must fail closed for multi-rank."""
        checkpoint_path = tmp_path / "multi_rank_create.pt"
        monkeypatch.setattr(
            self.simulator,
            "get_device_info",
            lambda: {
                "device": torch.device("cpu"),
                "device_type": "cpu",
                "device_index": None,
                "distributed_enabled": True,
                "rank": 0,
                "world_size": 2,
            },
        )

        with pytest.raises(RuntimeError, match="single-process"):
            create_checkpoint(self.simulator, checkpoint_path)

        assert not checkpoint_path.exists()

    def test_checkpoint_restore_rejects_active_multi_rank_runtime(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Multi-rank restore must fail before changing simulator state."""
        checkpoint_path = tmp_path / "multi_rank_restore.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        self.simulator.current_time = 4.0
        monkeypatch.setattr(
            self.simulator,
            "get_device_info",
            lambda: {
                "device": torch.device("cpu"),
                "device_type": "cpu",
                "device_index": None,
                "distributed_enabled": True,
                "rank": 1,
                "world_size": 2,
            },
        )

        with pytest.raises(RuntimeError, match="single-process"):
            restore_checkpoint(self.simulator, checkpoint_path)

        assert self.simulator.current_time == pytest.approx(4.0)

    def test_rng_restore_rejects_missing_cuda_state_for_cuda_runtime(
        self,
        monkeypatch,
    ):
        """CUDA runtime and serialized CUDA RNG presence must be symmetric."""
        rng_state = checkpoint_module._capture_rng_state(self.simulator)
        mismatched_cuda_usage = not bool(rng_state["torch_cuda"])
        monkeypatch.setattr(
            checkpoint_module,
            "_uses_cuda_runtime",
            lambda simulator: mismatched_cuda_usage,
        )

        with pytest.raises(RuntimeError, match="CUDA RNG state presence"):
            checkpoint_module._restore_rng_state(
                rng_state,
                self.simulator,
            )

    def test_rng_restore_rejects_invalid_numpy_state_without_partial_mutation(
        self,
    ):
        """Invalid compatibility RNG data must not mutate any global stream."""
        import random

        np = pytest.importorskip("numpy")
        malformed = copy.deepcopy(
            checkpoint_module._capture_rng_state(self.simulator)
        )
        assert malformed["numpy"] is not None
        malformed["numpy"]["algorithm"] = "invalid_algorithm"

        random.random()
        torch.rand(4)
        np.random.random(4)
        before_attempt = checkpoint_module._capture_rng_state(self.simulator)

        with pytest.raises(ValueError):
            checkpoint_module._restore_rng_state(
                malformed,
                self.simulator,
            )

        after_attempt = checkpoint_module._capture_rng_state(self.simulator)
        _assert_checkpoint_value_equal(after_attempt, before_attempt)

    @pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
    def test_checkpoint_restore_rolls_back_safe_functor_extra_state(
        self,
        tmp_path,
        monkeypatch,
        failure,
    ):
        """A later apply failure must roll back non-tensor module extra state."""

        class ExtraStateFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="rollback_extra_state")
                self.counter = 7

            def get_extra_state(self):
                return {"counter": self.counter}

            def set_extra_state(self, state):
                self.counter = int(state["counter"])

        functor = ExtraStateFunctor()
        self.simulator.register_functor(functor)
        checkpoint_path = tmp_path / "rollback_extra_state.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        functor.counter = 99
        original_restore_rng = checkpoint_module._restore_rng_state
        calls = 0

        def fail_once(rng_state, simulator):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise failure("injected RNG restore failure")
            return original_restore_rng(rng_state, simulator)

        monkeypatch.setattr(
            checkpoint_module,
            "_restore_rng_state",
            fail_once,
        )

        with pytest.raises(
            failure,
            match="injected RNG restore failure",
        ):
            restore_checkpoint(self.simulator, checkpoint_path)

        assert functor.counter == 99

    def test_checkpoint_restore_rejects_scientific_config_mismatch(
        self,
        tmp_path,
    ):
        """Matching classes and names cannot hide a different process config."""
        source_functor = SimpleFunctor(
            name="configured_functor",
            increment=1.0,
        )
        self.simulator.register_functor(source_functor)
        checkpoint_path = tmp_path / "scientific_config.pt"
        create_checkpoint(self.simulator, checkpoint_path)

        target = _declared_simulator(self.config)
        manager = StateManager()
        manager.initialize()
        target.initialize(manager, DependencyGraphResolver())
        target.register_functor(
            SimpleFunctor(
                name="configured_functor",
                increment=2.0,
            )
        )

        with pytest.raises(
            ValueError,
            match=r"scientific contract.*mismatched sections: scientific_contract",
        ):
            restore_checkpoint(target, checkpoint_path)

    def test_checkpoint_restore_rejects_state_schema_mismatch(
        self,
        tmp_path,
    ):
        """StateManager units and bounds are part of restart compatibility."""
        source_manager = StateManager(
            schema={
                "test_pool": StateVariableSpec(
                    name="test_pool",
                    units="kg C m-2",
                    bounds=(0.0, 10.0),
                )
            }
        )
        source_manager.initialize({"test_pool": torch.tensor(1.0)})
        source = _declared_simulator(self.config)
        source.initialize(source_manager, DependencyGraphResolver())
        checkpoint_path = tmp_path / "state_schema.pt"
        create_checkpoint(source, checkpoint_path)

        target_manager = StateManager(
            schema={
                "test_pool": StateVariableSpec(
                    name="test_pool",
                    units="kg N m-2",
                    bounds=(0.0, 10.0),
                )
            }
        )
        target_manager.initialize({"test_pool": torch.tensor(1.0)})
        target = _declared_simulator(self.config)
        target.initialize(target_manager, DependencyGraphResolver())

        with pytest.raises(ValueError, match="scientific contract"):
            restore_checkpoint(target, checkpoint_path)

    def test_checkpoint_restore_rejects_accessed_parameter_mismatch(
        self,
        tmp_path,
    ):
        """Accessed ParameterManager specs must be bound to the checkpoint."""

        class ParameterManagerStub:
            def __init__(self, value):
                self._access_history = {"biosphere.test.rate"}
                self.spec = ParameterSpec(
                    key="biosphere.test.rate",
                    value=value,
                    units="s-1",
                    bounds=(0.0, 10.0),
                    description="Checkpoint parameter contract.",
                    references=["Unit test"], dtype="float32", category="model_parameter",
                )

            def get_parameter(self, key):
                return self.spec if key == self.spec.key else None

        source_functor = SimpleFunctor(name="parameter_functor")
        source_functor.parameter_manager = ParameterManagerStub(1.0)
        self.simulator.register_functor(source_functor)
        checkpoint_path = tmp_path / "parameter_contract.pt"
        create_checkpoint(self.simulator, checkpoint_path)

        target = _declared_simulator(self.config)
        manager = StateManager()
        manager.initialize()
        target.initialize(manager, DependencyGraphResolver())
        target_functor = SimpleFunctor(name="parameter_functor")
        target_functor.parameter_manager = ParameterManagerStub(2.0)
        target.register_functor(target_functor)

        with pytest.raises(ValueError, match="scientific contract"):
            restore_checkpoint(target, checkpoint_path)

    def test_checkpoint_roundtrip_preserves_complete_runtime_state(self, tmp_path):
        """Disk checkpoints must preserve the canonical runtime owners."""

        class ParameterFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="checkpoint_functor")
                self.weight = torch.nn.Parameter(torch.tensor(3.0))

        self.simulator.initialize_state({"test_pool": torch.tensor(2.0)})
        functor = ParameterFunctor()
        self.simulator.register_functor(functor)
        self.simulator.current_time = 12.0
        self.simulator.time_step = 2.0
        self.simulator.step_count = 6
        self.simulator.total_execution_time = 1.25
        self.simulator.last_checkpoint_time = 10.0
        self.simulator.time_series = {
            "test_pool": [torch.tensor(1.0), torch.tensor(2.0)]
        }
        self.simulator._previous_day_index = 3
        self.simulator._last_medium_execution_time = 8.0
        self.simulator._scale_check_history = [{"time": 8.0, "ok": True}]
        self.simulator.state_manager._current_diagnostics = {
            "writer": "checkpoint_functor",
            "magnitude": torch.tensor(2.0),
        }
        self.simulator.state_manager._history = [
            {"test_pool": torch.tensor(1.0)}
        ]
        self.simulator.state_manager._last_summary = ConservationSummary(
            total_violations=0,
            warnings=["none"],
            timestamp=4.0, tolerance_value=1e-6,
        )

        checkpoint_path = tmp_path / "runtime.pt"
        receipt = create_checkpoint(self.simulator, checkpoint_path)

        assert receipt["path"] == str(checkpoint_path.resolve())
        assert receipt["format_version"] == checkpoint_module.CHECKPOINT_FORMAT_VERSION
        assert receipt["size_bytes"] == checkpoint_path.stat().st_size
        assert len(receipt["sha256"]) == 64

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        assert payload["format"] == "relic.simulator.checkpoint"
        assert payload["format_version"] == checkpoint_module.CHECKPOINT_FORMAT_VERSION
        assert "dt" not in payload
        assert payload["runtime"]["time_step"] == pytest.approx(2.0)
        assert "metadata" not in payload["functors"]["checkpoint_functor"]

        self.simulator.current_time = 99.0
        self.simulator.time_step = 99.0
        self.simulator.step_count = 99
        self.simulator.total_execution_time = 99.0
        self.simulator.last_checkpoint_time = 99.0
        self.simulator.time_series.clear()
        self.simulator._previous_day_index = -1
        self.simulator._last_medium_execution_time = 0.0
        self.simulator._scale_check_history.clear()
        self.simulator.state_manager.set_state(
            {"test_pool": torch.tensor(99.0)},
            validate=False,
        )
        self.simulator.state_manager._current_diagnostics.clear()
        self.simulator.state_manager._history.clear()
        self.simulator.state_manager._last_summary = None
        with torch.no_grad():
            functor.weight.fill_(99.0)

        restored_receipt = restore_checkpoint(self.simulator, checkpoint_path)

        assert restored_receipt["sha256"] == receipt["sha256"]
        assert self.simulator.current_time == pytest.approx(12.0)
        assert self.simulator.time_step == pytest.approx(2.0)
        assert self.simulator.step_count == 6
        assert self.simulator.total_execution_time == pytest.approx(1.25)
        assert self.simulator.last_checkpoint_time == pytest.approx(10.0)
        assert torch.equal(
            self.simulator.time_series["test_pool"][1],
            torch.tensor(2.0),
        )
        assert self.simulator._previous_day_index == 3
        assert self.simulator._last_medium_execution_time == pytest.approx(8.0)
        assert self.simulator._scale_check_history == [{"time": 8.0, "ok": True}]
        assert torch.equal(
            self.simulator.state_manager.get_state()["test_pool"].cpu(),
            torch.tensor(2.0),
        )
        assert torch.equal(
            self.simulator.state_manager._current_diagnostics["magnitude"].cpu(),
            torch.tensor(2.0),
        )
        assert isinstance(
            self.simulator.state_manager._last_summary,
            ConservationSummary,
        )
        assert float(functor.weight.detach()) == pytest.approx(3.0)
        assert (
            self.simulator._last_state_reset_counter
            == self.simulator.state_manager._state_reset_counter
        )

    def test_runtime_transaction_restores_state_functors_runtime_and_rng(self):
        """An in-memory call transaction must restore every mutable owner."""

        class BufferFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="transaction_buffer")
                self.register_buffer("running_total", torch.tensor(3.0))

        self.simulator.initialize_state({"test_pool": torch.tensor(2.0)})
        functor = BufferFunctor()
        self.simulator.register_functor(functor)
        self.simulator.current_time = 12.0
        self.simulator.step_count = 4
        self.simulator.time_series = {"test_pool": [torch.tensor(2.0)]}
        if hasattr(self.simulator, "_baseline_snapshot_reference"):
            delattr(self.simulator, "_baseline_snapshot_reference")

        random.seed(124)
        np.random.seed(125)
        torch.manual_seed(126)
        transaction = capture_runtime_transaction(self.simulator)
        previous_epoch = self.simulator.provider_epoch
        expected_python = random.random()
        expected_numpy = float(np.random.random())
        expected_torch = torch.rand(4)

        self.simulator.current_time = 99.0
        self.simulator.step_count = 99
        self.simulator.time_series["test_pool"].append(torch.tensor(99.0))
        self.simulator._baseline_snapshot_reference = {"leaked": True}
        self.simulator.state_manager.set_state(
            {"test_pool": torch.tensor(99.0)},
            validate=False,
        )
        with torch.no_grad():
            functor.running_total.fill_(99.0)
        random.seed(991)
        np.random.seed(992)
        torch.manual_seed(993)

        restore_runtime_transaction(self.simulator, transaction)

        assert self.simulator.provider_epoch == previous_epoch
        assert self.simulator.current_time == pytest.approx(12.0)
        assert self.simulator.step_count == 4
        assert self.simulator.time_series == {
            "test_pool": [torch.tensor(2.0)]
        }
        assert not hasattr(self.simulator, "_baseline_snapshot_reference")
        restored_pool = self.simulator.state_manager.get_state()["test_pool"]
        torch.testing.assert_close(
            restored_pool,
            torch.full_like(restored_pool, 2.0),
        )
        torch.testing.assert_close(
            functor.running_total,
            torch.full_like(functor.running_total, 3.0),
        )
        assert random.random() == pytest.approx(expected_python)
        assert float(np.random.random()) == pytest.approx(expected_numpy)
        torch.testing.assert_close(torch.rand(4), expected_torch)

    def test_runtime_transaction_restore_failure_rolls_back_live_state(self):
        """A malformed transaction must not partially mutate the live runtime."""

        class BufferFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="rollback_buffer")
                self.register_buffer("running_total", torch.tensor(1.0))

        self.simulator.initialize_state({"test_pool": torch.tensor(1.0)})
        functor = BufferFunctor()
        self.simulator.register_functor(functor)
        malformed = capture_runtime_transaction(self.simulator)
        malformed["rng"]["torch_cpu"] = "invalid"

        self.simulator.current_time = 7.0
        self.simulator.step_count = 7
        self.simulator.state_manager.set_state(
            {"test_pool": torch.tensor(7.0)},
            validate=False,
        )
        with torch.no_grad():
            functor.running_total.fill_(7.0)

        random.seed(55)
        np.random.seed(55)
        torch.manual_seed(55)
        expected_python = random.Random(55).random()
        expected_numpy = float(np.random.RandomState(55).random_sample())
        expected_generator = torch.Generator(device="cpu")
        expected_generator.manual_seed(55)
        expected_torch = torch.rand(4, generator=expected_generator)

        with pytest.raises(TypeError, match="torch CPU RNG state"):
            restore_runtime_transaction(self.simulator, malformed)

        assert self.simulator.current_time == pytest.approx(7.0)
        assert self.simulator.step_count == 7
        live_pool = self.simulator.state_manager.get_state()["test_pool"]
        torch.testing.assert_close(
            live_pool,
            torch.full_like(live_pool, 7.0),
        )
        torch.testing.assert_close(
            functor.running_total,
            torch.full_like(functor.running_total, 7.0),
        )
        assert random.random() == pytest.approx(expected_python)
        assert float(np.random.random()) == pytest.approx(expected_numpy)
        torch.testing.assert_close(torch.rand(4), expected_torch)

    @pytest.mark.parametrize("failure", ["error", "cancel", "deferred"])
    def test_checkpoint_restore_publication_failure_rolls_back_all_live_owners(
        self, tmp_path, monkeypatch, failure,
    ):
        from relic.core.provenance import ProvenanceRecorder
        from relic.core.simulator import registration

        class BufferedFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="publication_buffer")
                self.register_buffer("running_total", torch.tensor(2.0))

        sim = _fresh_checkpoint_simulator(BufferedFunctor())
        sim.initialize_state({"test_pool": torch.tensor(2.0)})
        sim.current_time = 2.0
        sim.step_count = 2
        path = tmp_path / "publication.pt"
        sim.create_checkpoint(path)
        sim.current_time = 7.0
        sim.step_count = 7
        sim.functors["publication_buffer"].running_total.fill_(7.0)
        manager = sim.state_manager
        manager.set_state({"test_pool": torch.tensor(6.0)}, validate=False)
        destination = tmp_path / "pending_provenance"
        recorder = ProvenanceRecorder(destination, run_id=sim.run_id)
        manager.set_provenance_recorder(recorder)
        manager.begin_provenance_transaction()
        manager.apply_updates({"test_pool": torch.tensor(1.0)}, check_conservation=False)
        accepted = manager.checkpoint()
        accepted["provenance_transaction_active"] = False
        manager.rollback_provenance_transaction()
        manager.restore(accepted)
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        before = capture_runtime_transaction(sim, for_rollback=True)
        lifecycle = sim._provider_lifecycle
        view, epoch = lifecycle.committed_view, lifecycle.epoch
        history = list(lifecycle.publication_history)
        deliveries = sim._pending_external_deliveries
        receipts = sim._last_external_delivery_receipts
        lifecycle.pending_publication_reason = "existing pending reason"
        lifecycle.deferred_publications = int(failure == "deferred")
        backups = []
        original_capture = checkpoint_module.capture_runtime_transaction

        def capture_once(*args, **kwargs):
            backups.append(kwargs.get("for_rollback"))
            return original_capture(*args, **kwargs)

        def fail_publication(*args, **kwargs):
            if failure == "cancel":
                raise KeyboardInterrupt("publication cancelled")
            raise RuntimeError("publication failed")

        monkeypatch.setattr(checkpoint_module, "capture_runtime_transaction", capture_once)
        if failure != "deferred":
            monkeypatch.setattr(registration, "_publish_provider_view", fail_publication)
        expected_error = KeyboardInterrupt if failure == "cancel" else RuntimeError
        with pytest.raises(expected_error):
            sim.restore_checkpoint(path)
        _assert_checkpoint_value_equal(capture_runtime_transaction(sim, for_rollback=True), before)
        assert backups == [True]
        assert manager._provenance_recorder is recorder
        assert sim._pending_external_deliveries is deliveries
        assert sim._last_external_delivery_receipts is receipts
        assert lifecycle.committed_view is view and lifecycle.epoch == epoch
        assert lifecycle.publication_history == history
        assert lifecycle.pending_publication_reason == "existing pending reason"
        assert lifecycle.deferred_publications == int(failure == "deferred")
        assert not destination.exists()

    def test_checkpoint_restore_publication_preparation_restores_registration_metadata(
        self, tmp_path, monkeypatch,
    ):
        from relic.core.simulator import registration

        sim = _fresh_checkpoint_simulator(SimpleFunctor())
        sim.initialize_state({"test_pool": torch.tensor(2.0)})
        path = tmp_path / "preparation.pt"
        sim.create_checkpoint(path)
        sim.state_manager.set_state({"test_pool": torch.tensor(7.0)}, validate=False)
        lifecycle = sim._provider_lifecycle
        lifecycle.provider_uids.clear()
        sim._groups_dirty = True
        before = registration._RegistrationSnapshot.capture(sim)
        before_runtime = capture_runtime_transaction(sim)
        old_sequence = lifecycle.uid_sequence
        build = registration._build_provider_identities

        def fail_after_preparation(*args, **kwargs):
            identities = build(*args, **kwargs)
            assert identities and lifecycle.provider_uids
            assert sim._groups_dirty is False
            raise RuntimeError("identity preparation failed")

        monkeypatch.setattr(registration, "_build_provider_identities", fail_after_preparation)
        with pytest.raises(RuntimeError, match="identity preparation failed"):
            sim.restore_checkpoint(path)
        after = registration._RegistrationSnapshot.capture(sim)
        _assert_checkpoint_value_equal(after.attributes, before.attributes)
        _assert_checkpoint_value_equal(after.lifecycle_maps, before.lifecycle_maps)
        _assert_checkpoint_value_equal(capture_runtime_transaction(sim), before_runtime)
        assert lifecycle.uid_sequence > old_sequence

    @pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
    def test_checkpoint_restore_never_rewinds_a_published_epoch(
        self, tmp_path, monkeypatch, error_type,
    ):
        sim = self.simulator
        sim.initialize_state({"test_pool": torch.tensor(2.0)})
        sim.current_time = 2.0
        sim.step_count = 2
        torch.manual_seed(2)
        path = tmp_path / "published.pt"
        sim.create_checkpoint(path)
        saved_rng = torch.get_rng_state().clone()
        sim.state_manager.set_state({"test_pool": torch.tensor(7.0)}, validate=False)
        sim.current_time = 7.0
        sim.step_count = 7
        torch.manual_seed(7)
        lifecycle = sim._provider_lifecycle
        old_epoch = lifecycle.epoch
        retire = lifecycle.retire_view
        observed = []

        def fail_after_retirement(view):
            retire(view)
            with sim.acquire_provider_view() as reader:
                observed.append((reader.view.epoch, sim.current_time, float(sim.get_state()["test_pool"])))
            raise error_type("retirement interrupted after publication")

        monkeypatch.setattr(lifecycle, "retire_view", fail_after_retirement)
        with pytest.raises(error_type, match="retirement interrupted"):
            sim.restore_checkpoint(path)
        assert observed == [(old_epoch + 1, 2.0, 2.0)]
        assert lifecycle.epoch == old_epoch + 1
        assert lifecycle.committed_view.epoch == old_epoch + 1
        assert lifecycle.publication_history[-1]["epoch"] == old_epoch + 1
        assert sim.current_time == 2.0 and sim.step_count == 2
        assert float(sim.get_state()["test_pool"]) == 2.0
        assert torch.equal(torch.get_rng_state(), saved_rng)

    def test_checkpoint_restore_dispose_failure_is_observable_and_retryable(self, tmp_path, monkeypatch):
        sim = self.simulator
        sim.initialize_state({"test_pool": torch.tensor(2.0)})
        path = tmp_path / "dispose.pt"
        sim.create_checkpoint(path)
        sim.state_manager.set_state({"test_pool": torch.tensor(7.0)}, validate=False)
        lifecycle = sim._provider_lifecycle
        epoch = lifecycle.epoch
        dispose = lifecycle.dispose_view
        calls = []

        def fail_once(view):
            calls.append(view.epoch)
            if len(calls) == 1:
                raise OSError("native handle busy")
            dispose(view)

        monkeypatch.setattr(lifecycle, "dispose_view", fail_once)
        receipt = sim.restore_checkpoint(path)
        assert receipt["provider_epoch"] == epoch + 1
        assert float(sim.get_state()["test_pool"]) == 2.0
        assert lifecycle.retired_status[epoch] == "dispose_failed"
        assert "native handle busy" in lifecycle.dispose_errors[epoch]
        assert lifecycle.retry_dispose(epoch) == "disposed"
        assert calls == [epoch, epoch]

    def test_checkpoint_restore_receipt_failure_preserves_published_state(self, tmp_path, monkeypatch):
        sim = self.simulator
        sim.initialize_state({"test_pool": torch.tensor(2.0)})
        path = tmp_path / "receipt.pt"
        sim.create_checkpoint(path)
        sim.state_manager.set_state({"test_pool": torch.tensor(7.0)}, validate=False)
        epoch = sim.provider_epoch

        def fail_receipt(*args, **kwargs):
            raise OSError("receipt unavailable")

        monkeypatch.setattr(checkpoint_module, "_checkpoint_receipt", fail_receipt)
        with pytest.raises(OSError, match="receipt unavailable"):
            sim.restore_checkpoint(path)
        assert sim.provider_epoch == epoch + 1
        assert float(sim.get_state()["test_pool"]) == 2.0

    def test_checkpoint_restore_attempts_runtime_compensation_when_registration_compensation_fails(
        self, tmp_path, monkeypatch,
    ):
        from relic.core.simulator import registration

        sim = self.simulator
        sim.initialize_state({"test_pool": torch.tensor(2.0)})
        path = tmp_path / "compensation.pt"
        sim.create_checkpoint(path)
        sim.state_manager.set_state({"test_pool": torch.tensor(7.0)}, validate=False)
        before = capture_runtime_transaction(sim)

        def fail_publication(*args, **kwargs):
            raise RuntimeError("publication fault")

        def fail_compensation(*args, **kwargs):
            raise OSError("registration compensation fault")

        monkeypatch.setattr(registration, "_publish_provider_view", fail_publication)
        monkeypatch.setattr(registration._RegistrationSnapshot, "restore", fail_compensation)
        with pytest.raises(BaseExceptionGroup) as caught:
            sim.restore_checkpoint(path)
        assert [str(exc) for exc in caught.value.exceptions] == [
            "publication fault", "registration compensation fault",
        ]
        _assert_checkpoint_value_equal(capture_runtime_transaction(sim), before)

    def test_checkpoint_restore_replays_torch_rng_state(self, tmp_path):
        """Restart must reproduce stochastic draws from the checkpoint boundary."""
        checkpoint_path = tmp_path / "rng.pt"
        torch.manual_seed(1234)
        create_checkpoint(self.simulator, checkpoint_path)
        expected = torch.rand(8)

        torch.manual_seed(9876)
        restore_checkpoint(self.simulator, checkpoint_path)
        actual = torch.rand(8)

        assert torch.equal(actual, expected)

    def test_checkpoint_restore_rejects_integrity_tampering_before_mutation(
        self,
        tmp_path,
    ):
        """Integrity failure must leave the active simulator unchanged."""
        self.simulator.initialize_state({"test_pool": torch.tensor(2.0)})
        checkpoint_path = tmp_path / "tampered.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        payload["runtime"]["current_time"] = 100.0
        torch.save(payload, checkpoint_path)

        self.simulator.current_time = 7.0
        self.simulator.state_manager.set_state(
            {"test_pool": torch.tensor(7.0)},
            validate=False,
        )
        with pytest.raises(ValueError, match="integrity"):
            restore_checkpoint(self.simulator, checkpoint_path)

        assert self.simulator.current_time == pytest.approx(7.0)
        assert torch.equal(
            self.simulator.state_manager.get_state()["test_pool"].cpu(),
            torch.tensor(7.0),
        )

    def test_checkpoint_restore_rejects_unsupported_version(self, tmp_path):
        """Unknown checkpoint versions must never enter a compatibility path."""
        checkpoint_path = tmp_path / "future.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        payload["format_version"] = 999
        torch.save(payload, checkpoint_path)

        with pytest.raises(ValueError, match="Unsupported checkpoint version"):
            restore_checkpoint(self.simulator, checkpoint_path)

    def test_checkpoint_restore_requires_exact_functor_topology(self, tmp_path):
        """A checkpoint cannot be restored into a different scientific graph."""
        source_functor = SimpleFunctor(name="source_functor")
        self.simulator.register_functor(source_functor)
        checkpoint_path = tmp_path / "topology.pt"
        create_checkpoint(self.simulator, checkpoint_path)

        target = _declared_simulator(self.config)
        manager = StateManager()
        manager.initialize()
        target.initialize(manager, DependencyGraphResolver())
        target.current_time = 5.0

        with pytest.raises(ValueError, match="functor topology"):
            restore_checkpoint(target, checkpoint_path)

        assert target.current_time == pytest.approx(5.0)

    def test_checkpoint_restore_rejects_dependency_topology_mismatch(
        self,
        tmp_path,
    ):
        """Matching names and classes cannot hide different dependencies."""
        self.simulator.register_functor(SimpleFunctor(name="base"))
        self.simulator.register_functor(
            DependentFunctor(name="dependent", depends_on=["base"])
        )
        checkpoint_path = tmp_path / "dependency_topology.pt"
        create_checkpoint(self.simulator, checkpoint_path)

        target = _declared_simulator(self.config)
        manager = StateManager()
        manager.initialize()
        target.initialize(manager, DependencyGraphResolver())
        target.register_functor(SimpleFunctor(name="base"))
        target.register_functor(
            DependentFunctor(name="dependent", depends_on=[])
        )

        with pytest.raises(ValueError, match="functor topology"):
            restore_checkpoint(target, checkpoint_path)

    def test_checkpoint_rejects_replaced_kernel_before_state_or_rng_mutation(
        self, tmp_path, monkeypatch,
    ):
        source = _fresh_checkpoint_simulator(SimpleFunctor(name="same_kernel"))
        source.initialize_state({"test_pool": torch.tensor(0.0)})
        checkpoint_path = tmp_path / "original_kernel.pt"
        source.create_checkpoint(checkpoint_path)
        original_identity = source.functors["same_kernel"].get_kernel_identity()
        source.step(dt=1.0)
        assert float(source.state_manager.get_state()["test_pool"]) == 1.0

        target = _fresh_checkpoint_simulator(SimpleFunctor(name="same_kernel"))
        target.initialize_state({"test_pool": torch.tensor(9.0)})
        target.current_time = 5.0
        before = capture_runtime_transaction(target)
        kernel = SimpleFunctor.forward
        monkeypatch.setattr(kernel, "__code__", _replacement_checkpoint_kernel.__code__)
        assert SimpleFunctor.forward is kernel
        assert target.functors["same_kernel"].get_kernel_identity() != original_identity
        with pytest.raises(ValueError, match="mismatched sections: scientific_contract"):
            target.restore_checkpoint(checkpoint_path)
        _assert_checkpoint_value_equal(capture_runtime_transaction(target), before)
        target.step(dt=1.0)
        assert float(target.state_manager.get_state()["test_pool"]) == 11.0

    def test_checkpoint_binds_one_kernel_identity_and_replays_the_real_output(self, tmp_path):
        initial = {"test_pool": torch.tensor(0.0)}
        uninterrupted = _fresh_checkpoint_simulator(SimpleFunctor(name="same_kernel"))
        uninterrupted.initialize_state(initial)
        for _ in range(4):
            uninterrupted.step(dt=1.0)
        first_leg = _fresh_checkpoint_simulator(SimpleFunctor(name="same_kernel"))
        first_leg.initialize_state(initial)
        for _ in range(2):
            first_leg.step(dt=1.0)
        checkpoint_path = tmp_path / "same_kernel.pt"
        first_leg.create_checkpoint(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved = payload["topology"]["scientific_contract"]["functors"]["same_kernel"]["kernel_identity"]
        assert saved == first_leg.functors["same_kernel"].get_kernel_identity()

        resumed = _fresh_checkpoint_simulator(SimpleFunctor(name="same_kernel"))
        resumed.initialize_state(initial)
        resumed.restore_checkpoint(checkpoint_path)
        for _ in range(2):
            resumed.step(dt=1.0)
        _assert_checkpoint_value_equal(
            resumed.state_manager.get_state(), uninterrupted.state_manager.get_state(),
        )
        _assert_checkpoint_value_equal(resumed.time_series, uninterrupted.time_series)
        assert float(resumed.state_manager.get_state()["test_pool"]) == 4.0
        assert resumed.current_time == uninterrupted.current_time == 4.0
        assert resumed.step_count == uninterrupted.step_count == 4

    def test_checkpoint_rejects_missing_kernel_identity_before_mutation(self, tmp_path):
        simulator = _fresh_checkpoint_simulator(SimpleFunctor(name="same_kernel"))
        simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        checkpoint_path = tmp_path / "missing_kernel_identity.pt"
        simulator.create_checkpoint(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        payload["topology"]["scientific_contract"]["functors"]["same_kernel"].pop("kernel_identity", None)
        content = {key: value for key, value in payload.items() if key != "integrity"}
        payload["integrity"]["payload_sha256"] = checkpoint_module._payload_sha256(content)
        torch.save(payload, checkpoint_path)
        simulator.current_time = 3.0
        before = capture_runtime_transaction(simulator)
        with pytest.raises(ValueError, match="mismatched sections: scientific_contract"):
            simulator.restore_checkpoint(checkpoint_path)
        _assert_checkpoint_value_equal(capture_runtime_transaction(simulator), before)

    def test_checkpoint_rejects_mutable_kernel_closure_without_writing(self, tmp_path):
        captured = [True]

        class ClosureFunctor(SimpleFunctor):
            def forward(self, state, time, dt):
                return super().forward(state, time, dt) if captured else {}

        simulator = _fresh_checkpoint_simulator(ClosureFunctor(name="closure_kernel"))
        simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        checkpoint_path = tmp_path / "uncertified_closure.pt"
        with pytest.raises(ValueError, match="mutable or unsupported closure"):
            simulator.create_checkpoint(checkpoint_path)
        assert not checkpoint_path.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_checkpoint_restore_rejects_nonfinite_runtime_before_mutation(
        self,
        tmp_path,
    ):
        """A valid digest cannot authorize a nonfinite simulation clock."""
        checkpoint_path = tmp_path / "nonfinite.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        payload["runtime"]["current_time"] = float("nan")
        content = {
            key: value
            for key, value in payload.items()
            if key != "integrity"
        }
        payload["integrity"]["payload_sha256"] = (
            checkpoint_module._payload_sha256(content)
        )
        torch.save(payload, checkpoint_path)
        self.simulator.current_time = 3.0

        with pytest.raises(ValueError, match="finite"):
            restore_checkpoint(self.simulator, checkpoint_path)

        assert self.simulator.current_time == pytest.approx(3.0)

    def test_checkpoint_restore_rejects_invalid_state_manager_history(
        self,
        tmp_path,
    ):
        """StateManager history must remain a tensor snapshot sequence."""
        checkpoint_path = tmp_path / "invalid_history.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        payload["state_manager"]["history"] = [{"pool": "not-a-tensor"}]
        content = {
            key: value
            for key, value in payload.items()
            if key != "integrity"
        }
        payload["integrity"]["payload_sha256"] = (
            checkpoint_module._payload_sha256(content)
        )
        torch.save(payload, checkpoint_path)

        with pytest.raises(TypeError, match="history"):
            restore_checkpoint(self.simulator, checkpoint_path)

    def test_checkpoint_creation_fails_closed_for_unsafe_runtime_value(
        self,
        tmp_path,
    ):
        """Unsupported objects must not be silently pickled into checkpoints."""
        checkpoint_path = tmp_path / "unsafe.pt"
        self.simulator.state_manager._current_diagnostics = {"unsafe": object()}

        with pytest.raises(TypeError, match="Unsupported checkpoint value"):
            create_checkpoint(self.simulator, checkpoint_path)

        assert not checkpoint_path.exists()
        assert list(tmp_path.glob("unsafe.pt.*.tmp")) == []

    def test_checkpoint_creation_requires_stable_state_boundary(self, tmp_path):
        """Open provenance transactions cannot cross a disk checkpoint."""
        checkpoint_path = tmp_path / "open_transaction.pt"
        self.simulator.state_manager.begin_provenance_transaction()

        with pytest.raises(RuntimeError, match="stable checkpoint boundary"):
            create_checkpoint(self.simulator, checkpoint_path)

        assert not checkpoint_path.exists()

    def test_checkpoint_restore_validates_state_manager_schema(self, tmp_path):
        """Integrity alone cannot authorize a malformed owner snapshot."""
        checkpoint_path = tmp_path / "invalid_state_manager.pt"
        create_checkpoint(self.simulator, checkpoint_path)
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        payload["state_manager"].pop("state")
        content = {
            key: value
            for key, value in payload.items()
            if key != "integrity"
        }
        payload["integrity"]["payload_sha256"] = (
            checkpoint_module._payload_sha256(content)
        )
        torch.save(payload, checkpoint_path)

        with pytest.raises(ValueError, match="StateManager field"):
            restore_checkpoint(self.simulator, checkpoint_path)

    @pytest.mark.parametrize("boundary", ["save", "replace"])
    @pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
    def test_checkpoint_atomic_write_failure_leaves_no_artifact(
        self,
        tmp_path,
        monkeypatch,
        boundary,
        failure,
    ):
        """A failed atomic save must propagate and remove its temp file."""
        checkpoint_path = tmp_path / "failed.pt"

        def fail_save(*args, **kwargs):
            if boundary == "save":
                args[1].write_bytes(b"partial checkpoint")
            raise failure("injected save failure")

        owner = checkpoint_module.torch if boundary == "save" else checkpoint_module.os
        monkeypatch.setattr(owner, boundary, fail_save)
        with pytest.raises(failure, match="injected save failure"):
            create_checkpoint(self.simulator, checkpoint_path)

        assert not checkpoint_path.exists()
        assert list(tmp_path.glob("failed.pt.*.tmp")) == []

    def test_checkpoint_frozen_payload_does_not_recapture_live_data(self, tmp_path, monkeypatch):
        class FrozenParameterFunctor(SimpleFunctor):
            def __init__(self):
                super().__init__(name="frozen_parameter")
                self.weight = torch.nn.Parameter(torch.tensor(3.0))

        functor = FrozenParameterFunctor()
        self.simulator.initialize_state({"test_pool": torch.tensor(2.0)})
        self.simulator.register_functor(functor)
        self.simulator.current_time = 4.0
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        assert list(tmp_path.iterdir()) == []
        self.simulator.state_manager.apply_updates({
            "test_pool": UpdateItem(torch.tensor(7.0), mode=UpdateMode.REPLACE),
        })
        self.simulator.current_time = 11.0
        with torch.no_grad():
            functor.weight.fill_(9.0)
        torch.rand(5)
        live_rng = torch.get_rng_state().clone()

        def reject_recapture(*args, **kwargs):
            pytest.fail("A frozen delivery must not recapture live runtime")

        for name in ("_capture_runtime", "_capture_functor_state_dict", "_capture_rng_state", "_build_topology"):
            monkeypatch.setattr(checkpoint_module, name, reject_recapture)
        path = tmp_path / "frozen.pt"
        receipt = create_checkpoint(object(), path, snapshot=payload)
        stored = torch.load(path, weights_only=True)
        assert stored["runtime"]["current_time"] == 4.0
        assert float(stored["state_manager"]["state"]["test_pool"]) == 2.0
        assert float(stored["functors"]["frozen_parameter"]["state_dict"]["weight"]) == 3.0
        assert self.simulator.current_time == 11.0
        assert float(self.simulator.state_manager.get_state()["test_pool"]) == 7.0
        assert float(functor.weight.detach()) == 9.0
        assert torch.equal(torch.get_rng_state(), live_rng)
        assert receipt["payload_sha256"] == payload["integrity"]["payload_sha256"]

    def test_checkpoint_frozen_delivery_retries_without_rewriting(self, tmp_path, monkeypatch):
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        path = tmp_path / "retry.pt"
        first = create_checkpoint(self.simulator, path, snapshot=payload)
        before = path.read_bytes()

        def reject_write(*args, **kwargs):
            pytest.fail("Identical accepted payload must not be written twice")

        monkeypatch.setattr(checkpoint_module.torch, "save", reject_write)
        monkeypatch.setattr(checkpoint_module.os, "replace", reject_write)
        assert create_checkpoint(object(), path, snapshot=payload) == first
        assert path.read_bytes() == before
        with pytest.raises(FileExistsError):
            create_checkpoint(self.simulator, path)

    @pytest.mark.parametrize("damage", ["conflict", "tamper", "empty", "broken", "directory"])
    def test_checkpoint_retry_rejects_conflicting_or_damaged_target(self, tmp_path, damage):
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        path = tmp_path / "conflict.pt"
        if damage == "directory":
            path.mkdir()
        elif damage in ("empty", "broken"):
            path.write_bytes(b"" if damage == "empty" else b"broken checkpoint")
        else:
            stored = copy.deepcopy(payload)
            stored["runtime"]["current_time"] = 7.0
            if damage == "conflict":
                stored["integrity"]["payload_sha256"] = checkpoint_module._payload_sha256({
                    key: value for key, value in stored.items() if key != "integrity"
                })
            torch.save(stored, path)
        before = None if path.is_dir() else path.read_bytes()
        with pytest.raises((ValueError, FileExistsError)):
            create_checkpoint(object(), path, snapshot=payload)
        assert (None if path.is_dir() else path.read_bytes()) == before

    @pytest.mark.parametrize("damage", ["empty", "field", "format", "version", "timestamp", "integrity", "tensor"])
    def test_checkpoint_rejects_invalid_frozen_payload_without_output(self, tmp_path, damage):
        self.simulator.initialize_state({"test_pool": torch.tensor(2.0)})
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        if damage == "empty":
            payload = {}
        elif damage == "field":
            del payload["rng"]
        elif damage == "format":
            payload["format"] = "wrong"
        elif damage == "version":
            payload["format_version"] = 999
        elif damage == "timestamp":
            payload["created_utc"] = "invalid"
        elif damage == "integrity":
            payload["integrity"] = {}
        else:
            payload["state_manager"]["state"]["test_pool"].fill_(7.0)
        with pytest.raises((TypeError, ValueError)):
            create_checkpoint(object(), tmp_path / "invalid.pt", snapshot=payload)
        assert list(tmp_path.iterdir()) == []

    def test_checkpoint_receipt_failure_after_publish_is_retryable(self, tmp_path, monkeypatch):
        payload = checkpoint_module.capture_checkpoint_payload(self.simulator)
        path = tmp_path / "receipt.pt"
        with monkeypatch.context() as local:
            def fail_receipt(*args, **kwargs):
                raise OSError("injected receipt failure")
            local.setattr(checkpoint_module, "_file_sha256", fail_receipt)
            with pytest.raises(OSError, match="receipt failure"):
                create_checkpoint(self.simulator, path, snapshot=payload)
        before = path.read_bytes()
        def reject_save(*args, **kwargs):
            pytest.fail("Published data must not be rewritten after receipt failure")
        monkeypatch.setattr(checkpoint_module.torch, "save", reject_save)
        receipt = create_checkpoint(object(), path, snapshot=payload)
        assert receipt["sha256"] == checkpoint_module._file_sha256(path)
        assert path.read_bytes() == before

    def test_simulator_checkpoint_api_returns_explicit_receipts(self, tmp_path):
        """The public API must expose the canonical checkpoint receipt."""
        checkpoint_path = tmp_path / "public.pt"
        self.simulator.current_time = 4.0

        created = self.simulator.create_checkpoint(checkpoint_path)
        self.simulator.current_time = 9.0
        restored = self.simulator.restore_checkpoint(checkpoint_path)

        assert created["path"] == str(checkpoint_path.resolve())
        assert restored["path"] == created["path"]
        assert restored["sha256"] == created["sha256"]
        assert self.simulator.current_time == pytest.approx(4.0)

    def test_simulator_checkpointing_is_opt_in(self, tmp_path):
        """Automatic disk writes require an explicit switch and directory."""
        assert self.simulator.enable_checkpointing is False
        assert self.simulator.checkpoint_directory is None

        with pytest.raises(
            ValueError,
            match="checkpoint_directory",
        ):
            _declared_simulator(
                {
                    "time_step": 1.0,
                    "enable_checkpointing": True,
                }
            )

        enabled = _declared_simulator(
            {
                "time_step": 1.0,
                "enable_checkpointing": True,
                "checkpoint_directory": str(tmp_path),
            }
        )
        manager = StateManager()
        manager.initialize()
        enabled.initialize(manager, DependencyGraphResolver())
        enabled.step_count = 3
        receipt = enabled._create_checkpoint()

        assert enabled.enable_checkpointing is True
        assert enabled.checkpoint_directory == str(tmp_path.resolve())
        assert receipt["path"].endswith("checkpoint_step_000000000003.pt")
        assert checkpoint_module.Path(receipt["path"]).parent.name == enabled.run_id

    def test_automatic_checkpoint_uses_stable_accepted_step_boundary(
        self,
        tmp_path,
    ):
        """Automatic persistence must observe a closed provenance transaction."""
        simulator = _declared_simulator(
            {
                "time_step": 10.0,
                "checkpoint_frequency": 10.0,
                "enable_checkpointing": True,
                "checkpoint_directory": str(tmp_path),
            }
        )
        manager = StateManager()
        manager.initialize({"test_pool": torch.tensor(0.0)})
        simulator.initialize(manager, DependencyGraphResolver())
        simulator.register_functor(SimpleFunctor(name="increment"))

        simulator.step(dt=10.0)

        checkpoint_path = tmp_path / simulator.run_id / "checkpoint_step_000000000001.pt"
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        assert (
            payload["state_manager"]["provenance_transaction_active"]
            is False
        )
        assert payload["state_manager"]["pending_provenance_events"] == []
        assert payload["runtime"]["last_checkpoint_time"] == pytest.approx(
            10.0
        )
        assert simulator.last_checkpoint_time == pytest.approx(10.0)

        simulator.restore_checkpoint(checkpoint_path)
        simulator.step(dt=1.0)

        assert list(tmp_path.rglob("*.pt")) == [checkpoint_path]
        assert simulator.current_time == pytest.approx(11.0)

    def test_default_step_never_writes_implicit_checkpoint(self, tmp_path):
        """A checkpoint directory alone cannot enable automatic disk writes."""
        simulator = _declared_simulator(
            {
                "time_step": 1.0,
                "checkpoint_frequency": 1.0,
                "enable_checkpointing": False,
                "checkpoint_directory": str(tmp_path),
            }
        )
        manager = StateManager()
        manager.initialize({"test_pool": torch.tensor(0.0)})
        simulator.initialize(manager, DependencyGraphResolver())
        simulator.register_functor(SimpleFunctor(name="increment"))

        simulator.step(dt=1.0)

        assert list(tmp_path.rglob("*.pt")) == []
        assert simulator.last_checkpoint_time == pytest.approx(0.0)

    def test_automatic_checkpoint_failure_preserves_accepted_step(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Post-commit I/O failure cannot roll back committed science state."""
        simulator = _declared_simulator(
            {
                "time_step": 1.0,
                "checkpoint_frequency": 1.0,
                "enable_checkpointing": True,
                "checkpoint_directory": str(tmp_path),
            }
        )
        manager = StateManager()
        manager.initialize({"test_pool": torch.tensor(0.0)})
        simulator.initialize(manager, DependencyGraphResolver())
        simulator.register_functor(SimpleFunctor(name="increment"))

        def fail_checkpoint(active_simulator, checkpoint_path, *, snapshot):
            assert (
                active_simulator.state_manager._provenance_transaction_active
                is False
            )
            assert snapshot["runtime"]["current_time"] == 1.0
            assert snapshot["state_manager"]["state"]["test_pool"].item() == 1.0
            assert str(checkpoint_path).endswith("checkpoint_step_000000000001.pt")
            raise OSError("injected automatic checkpoint failure")

        monkeypatch.setattr(
            RELICSimulator,
            "_create_checkpoint",
            fail_checkpoint,
        )
        with pytest.raises(
            OSError,
            match="injected automatic checkpoint failure",
        ):
            simulator.step(dt=1.0)

        assert simulator.current_time == pytest.approx(1.0)
        assert simulator.step_count == 1
        assert float(
            simulator.state_manager.get_state()["test_pool"]
        ) == pytest.approx(1.0)
        assert simulator.last_checkpoint_time == pytest.approx(0.0)
        assert simulator.state_manager._provenance_transaction_active is False
        assert len(simulator._pending_external_deliveries) == 1
        assert simulator._pending_external_deliveries[0]["snapshot"] is not None
    
    def test_register_single_functor(self):
        """Test registering a single functor."""
        functor = SimpleFunctor(name="test_functor", increment=5.0)
        
        self.simulator.register_functor(functor)
        
        # Verify functor was added to dependency graph
        assert len(self.simulator.dependency_graph.nodes) == 1
        assert "test_functor" in self.simulator.dependency_graph.nodes
        
        print("[OK] Single functor registration successful")
    
    def test_dependency_resolution_simple(self):
        """Test dependency graph resolution with simple functors."""
        f1 = SimpleFunctor(name="functor_1", increment=1.0)
        f2 = SimpleFunctor(name="functor_2", increment=2.0)
        
        self.simulator.register_functor(f1)
        self.simulator.register_functor(f2)
        
        # Resolve dependencies
        execution_order = self.simulator.dependency_graph.topological_sort()
        
        assert len(execution_order) == 2
        assert "functor_1" in execution_order
        assert "functor_2" in execution_order
        
        print(f"[OK] Dependency resolution: order={execution_order}")
    
    def test_dependency_resolution_with_dependencies(self):
        """Test dependency resolution with dependent functors."""
        f1 = SimpleFunctor(name="base_functor", increment=1.0)
        f2 = DependentFunctor(name="dependent_functor", multiplier=2.0, depends_on=["base_functor"])
        
        self.simulator.register_functor(f1)
        self.simulator.register_functor(f2)
        
        # Resolve dependencies
        execution_order = self.simulator.dependency_graph.topological_sort()
        
        # base_functor must execute before dependent_functor
        assert len(execution_order) == 2
        base_idx = execution_order.index("base_functor")
        dep_idx = execution_order.index("dependent_functor")
        assert base_idx < dep_idx, "Base functor must execute before dependent functor"
        
        print(f"[OK] Dependency ordering correct: {execution_order}")
    
    def test_state_initialization(self):
        """Test state manager initialization."""
        initial_state = {
            "test_pool": torch.tensor(0.0, dtype=torch.float32),
            "secondary_pool": torch.tensor(10.0, dtype=torch.float32)
        }
        
        self.simulator.initialize_state(initial_state)
        
        current_state = self.simulator.state_manager.get_state()
        assert "test_pool" in current_state
        assert float(current_state["test_pool"]) == 0.0
        assert float(current_state["secondary_pool"]) == 10.0
        
        print("[OK] State initialized correctly")
    
    def test_single_step_execution(self):
        """Test single time step execution."""
        # Initialize state
        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        
        # Register simple functor
        functor = SimpleFunctor(name="increment_functor", increment=3.0)
        self.simulator.register_functor(functor)
        
        # Execute one step
        self.simulator.step(dt=1.0)
        
        # Check state was updated
        final_state = self.simulator.state_manager.get_state()
        assert float(final_state["test_pool"]) == pytest.approx(3.0, abs=1e-6)
        assert self.simulator.current_time == pytest.approx(1.0, abs=1e-6)
        
        print(f"[OK] Single step: pool={float(final_state['test_pool']):.2f}, time={self.simulator.current_time:.2f}")
    
    def test_multiple_step_execution(self):
        """Test multiple time step execution."""
        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        
        functor = SimpleFunctor(name="steady_increment", increment=2.0)
        self.simulator.register_functor(functor)
        
        # Execute 5 steps
        for i in range(5):
            self.simulator.step(dt=1.0)
        
        # Pool should have accumulated 5 * 2.0 = 10.0
        final_state = self.simulator.state_manager.get_state()
        assert float(final_state["test_pool"]) == pytest.approx(10.0, abs=1e-5)
        assert self.simulator.current_time == pytest.approx(5.0, abs=1e-6)
        
        print(f"[OK] Multiple steps: pool={float(final_state['test_pool']):.2f}, time={self.simulator.current_time:.2f}")
    
    def test_multiple_functors_execution(self):
        """Test execution with multiple functors updating same variable."""
        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        
        f1 = SimpleFunctor(name="adder_1", increment=1.0)
        f2 = SimpleFunctor(name="adder_2", increment=2.0)
        f3 = SimpleFunctor(name="adder_3", increment=3.0)
        
        self.simulator.register_functor(f1)
        self.simulator.register_functor(f2)
        self.simulator.register_functor(f3)
        
        # Execute one step
        self.simulator.step(dt=1.0)
        
        # Pool should accumulate all increments: 1 + 2 + 3 = 6
        final_state = self.simulator.state_manager.get_state()
        assert float(final_state["test_pool"]) == pytest.approx(6.0, abs=1e-5)
        
        print(f"[OK] Multiple functors: pool={float(final_state['test_pool']):.2f}")
    
    def test_execution_groups(self):
        """Test functor execution in different groups (fast/medium/slow)."""
        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        
        # Register functors in different execution groups
        fast_f = SimpleFunctor(name="fast_functor", increment=1.0)
        medium_f = SimpleFunctor(name="medium_functor", increment=2.0)
        slow_f = SimpleFunctor(name="slow_functor", increment=4.0)
        
        self.simulator.register_functor(fast_f, execution_group="fast")
        self.simulator.register_functor(medium_f, execution_group="medium")
        self.simulator.register_functor(slow_f, execution_group="slow")
        
        # Execute one step
        self.simulator.step(dt=1.0)
        
        # All functors should execute, order: fast -> medium -> slow
        final_state = self.simulator.state_manager.get_state()
        expected_total = 1.0 + 2.0 + 4.0
        assert float(final_state["test_pool"]) == pytest.approx(expected_total, abs=1e-5)
        
        print(f"[OK] Execution groups: pool={float(final_state['test_pool']):.2f}")
    
    def test_run_simulation(self):
        """Test running simulation for specified time period."""
        self.simulator.initialize_state({"test_pool": torch.tensor(0.0)})
        
        functor = SimpleFunctor(name="continuous_increment", increment=1.5)
        self.simulator.register_functor(functor)
        
        # Run from t=0 to t=4 with dt=1
        self.simulator.run(start_time=0.0, end_time=4.0, dt=1.0)
        
        # Should execute 4 steps: pool = 4 * 1.5 = 6.0
        final_state = self.simulator.state_manager.get_state()
        assert float(final_state["test_pool"]) == pytest.approx(6.0, abs=1e-5)
        assert self.simulator.current_time == pytest.approx(4.0, abs=1e-6)
        
        print(f"[OK] Run simulation: pool={float(final_state['test_pool']):.2f}, time={self.simulator.current_time:.2f}")
    
    def test_reset_simulation(self):
        """Test resetting simulation to initial state."""
        initial_state = {"test_pool": torch.tensor(5.0)}
        self.simulator.initialize_state(initial_state)
        
        functor = SimpleFunctor(name="modifier", increment=10.0)
        self.simulator.register_functor(functor)
        
        # Run simulation
        self.simulator.step(dt=1.0)
        assert float(self.simulator.state_manager.get_state()["test_pool"]) == pytest.approx(15.0, abs=1e-5)
        
        # Reset
        self.simulator.reset()
        
        # Should be back to initial state
        assert self.simulator.current_time == 0.0
        reset_state = self.simulator.state_manager.get_state()
        assert float(reset_state["test_pool"]) == pytest.approx(5.0, abs=1e-5)
        
        print("[OK] Simulation reset successful")


class TestDependencyGraphResolverL1:
    """L1 Unit tests for DependencyGraphResolver."""
    
    def setup_method(self):
        """Set up dependency resolver."""
        self.resolver = DependencyGraphResolver()
    
    def test_add_functor(self):
        """Test adding functor to dependency graph."""
        functor = SimpleFunctor(name="test_f")
        self.resolver.add_functor(functor, functor_id="test_f")
        
        assert "test_f" in self.resolver.nodes
        assert self.resolver.nodes["test_f"].functor == functor
        
        print("[OK] Functor added to graph")
    
    def test_resolve_no_dependencies(self):
        """Test resolving graph with no dependencies."""
        f1 = SimpleFunctor(name="f1")
        f2 = SimpleFunctor(name="f2")
        
        self.resolver.add_functor(f1, "f1")
        self.resolver.add_functor(f2, "f2")
        
        order = self.resolver.topological_sort()
        
        # Should return both functors (order may vary)
        assert len(order) == 2
        assert "f1" in order
        assert "f2" in order
        
        print(f"[OK] No dependencies: order={order}")
    
    def test_resolve_linear_dependencies(self):
        """Test resolving linear dependency chain: A -> B -> C."""
        f_a = SimpleFunctor(name="A", dependencies=[])
        f_b = DependentFunctor(name="B", depends_on=["A"])
        f_c = DependentFunctor(name="C", depends_on=["B"])
        
        self.resolver.add_functor(f_a, "A")
        self.resolver.add_functor(f_b, "B")
        self.resolver.add_functor(f_c, "C")
        
        order = self.resolver.topological_sort()
        
        # A must come before B, B before C
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")
        
        print(f"[OK] Linear dependencies: order={order}")
    
    def test_resolve_diamond_dependencies(self):
        """Test resolving diamond dependency: A -> B, A -> C, B -> D, C -> D."""
        f_a = SimpleFunctor(name="A", dependencies=[])
        f_b = DependentFunctor(name="B", depends_on=["A"])
        f_c = DependentFunctor(name="C", depends_on=["A"])
        f_d = DependentFunctor(name="D", depends_on=["B", "C"])
        
        self.resolver.add_functor(f_a, "A")
        self.resolver.add_functor(f_b, "B")
        self.resolver.add_functor(f_c, "C")
        self.resolver.add_functor(f_d, "D")
        
        order = self.resolver.topological_sort()
        
        # A must come first
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        # D must come last
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")
        
        print(f"[OK] Diamond dependencies: order={order}")
    
    def test_detect_circular_dependency(self):
        """Test detection of circular dependencies."""
        # Create circular dependency: A -> B -> A
        f_a = DependentFunctor(name="A", depends_on=["B"])
        f_b = DependentFunctor(name="B", depends_on=["A"])
        
        self.resolver.add_functor(f_a, "A")
        self.resolver.add_functor(f_b, "B")
        
        # Should raise CyclicDependencyError
        with pytest.raises(Exception) as exc_info:
            self.resolver.topological_sort()
        
        assert (
            "cyclic" in str(exc_info.value).lower()
            or "cycle" in str(exc_info.value).lower()
            or "circular" in str(exc_info.value).lower()
        )
        
        print("[OK] Circular dependency detected")
    
    def test_get_parallel_groups(self):
        """Test getting parallel execution groups."""
        # Create independent functors that can run in parallel
        f_a = SimpleFunctor(name="A")
        f_b = SimpleFunctor(name="B")
        # And one dependent functor
        f_c = DependentFunctor(name="C", depends_on=["A", "B"])
        
        self.resolver.add_functor(f_a, "A")
        self.resolver.add_functor(f_b, "B")
        self.resolver.add_functor(f_c, "C")
        
        groups = self.resolver.get_parallel_groups()
        
        group_index = {
            functor_name: index
            for index, group in enumerate(groups)
            for functor_name in group
        }
        assert group_index["A"] == group_index["B"]
        assert group_index["A"] < group_index["C"]
        
        print(f"[OK] Execution groups: {groups}")


if __name__ == "__main__":
    print("=" * 70)
    print("L1 Unit Tests: RELICSimulator Core Functionality")
    print("=" * 70)
    
    # Test Simulator
    print("\n[Testing RELICSimulator]")
    test_sim = TestRELICSimulatorL1()
    
    try:
        test_sim.setup_method()
        test_sim.test_simulator_initialization()
        
        test_sim.setup_method()
        test_sim.test_register_single_functor()
        
        test_sim.setup_method()
        test_sim.test_dependency_resolution_simple()
        
        test_sim.setup_method()
        test_sim.test_dependency_resolution_with_dependencies()
        
        test_sim.setup_method()
        test_sim.test_state_initialization()
        
        test_sim.setup_method()
        test_sim.test_single_step_execution()
        
        test_sim.setup_method()
        test_sim.test_multiple_step_execution()
        
        test_sim.setup_method()
        test_sim.test_multiple_functors_execution()
        
        test_sim.setup_method()
        test_sim.test_execution_groups()
        
        test_sim.setup_method()
        test_sim.test_run_simulation()
        
        test_sim.setup_method()
        test_sim.test_reset_simulation()
        
        print("\n[+] All RELICSimulator tests PASSED")
        
    except Exception as e:
        print(f"\n[-] RELICSimulator test FAILED: {e}")
        raise
    
    # Test Dependency Graph Resolver
    print("\n[Testing DependencyGraphResolver]")
    test_dep = TestDependencyGraphResolverL1()
    
    try:
        test_dep.setup_method()
        test_dep.test_add_functor()
        
        test_dep.setup_method()
        test_dep.test_resolve_no_dependencies()
        
        test_dep.setup_method()
        test_dep.test_resolve_linear_dependencies()
        
        test_dep.setup_method()
        test_dep.test_resolve_diamond_dependencies()
        
        test_dep.setup_method()
        test_dep.test_detect_circular_dependency()
        
        test_dep.setup_method()
        test_dep.test_get_parallel_groups()
        
        print("\n[+] All DependencyGraphResolver tests PASSED")
        
    except Exception as e:
        print(f"\n[-] DependencyGraphResolver test FAILED: {e}")
        raise
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All L1 Unit Tests for Simulator PASSED")
    print("=" * 70)
