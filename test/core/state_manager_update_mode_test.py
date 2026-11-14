import pytest
import torch

import relic.core.state as state_package
import relic.core.state.conservation as state_conservation_module
from relic.core.state import (
    StateManager,
    StateVariableSpec,
    UpdateItem,
    UpdateMode,
    StateWriteMode,
)
from relic.core.irf_specs import RunSpecs
from relic.contracts.contract_loader import ContractRegistry, VariableContract
from relic.utils.physical_constants import get_physical_constant


def _build_state_manager():
    schema = {
        "air_temperature": StateVariableSpec(
            name="air_temperature",
            units="K",
            bounds=(0.0, 400.0),
            conserved=False,
            required=False,
        )
    }
    manager = StateManager(schema=schema)
    manager.set_state({"air_temperature": torch.tensor(300.0)})
    return manager


class _CommitObserver:
    """Capture recorder payload together with committed manager state."""

    def __init__(self, manager):
        self.manager = manager
        self.events = []

    def record_state_update(
        self,
        step_index,
        variable_name,
        pre_value,
        post_value,
        increments,
        unit,
    ):
        self.events.append(
            {
                "step_index": step_index,
                "variable_name": variable_name,
                "pre_value": pre_value.clone(),
                "post_value": post_value.clone(),
                "increments": dict(increments),
                "unit": unit,
                "visible_state": self.manager.get_state(variable_name),
                "visible_counter": self.manager.get_diagnostics()["update_counter"],
            }
        )


@pytest.mark.parametrize("entrypoint", ["batch", "single"])
def test_provenance_recorder_observes_only_committed_state(entrypoint):
    """Recorder callbacks must run after the state and counter commit."""
    manager = _build_state_manager()
    # This test isolates recorder commit ordering from SCM writer policy.
    manager._contract_registry = None
    recorder = _CommitObserver(manager)
    manager.set_provenance_recorder(recorder)
    update = UpdateItem(
        torch.tensor(1.5),
        mode=UpdateMode.ACCUMULATE,
        unit="K",
        basis="per_step",
        per_timestep=True,
        source_functor="test_writer",
    )

    if entrypoint == "batch":
        manager.apply_updates(
            {"air_temperature": update},
            dt=1.0,
            check_conservation=False,
            update_sources={"air_temperature": "test_writer"},
            update_increments={
                "air_temperature": {"test_writer": torch.tensor(1.5)}
            },
        )
    else:
        manager.apply_increment("air_temperature", update, dt=1.0)

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event["step_index"] == 0
    assert event["variable_name"] == "air_temperature"
    assert event["unit"] == "K"
    assert torch.equal(event["pre_value"], torch.full_like(event["pre_value"], 300.0))
    assert torch.equal(event["post_value"], torch.full_like(event["post_value"], 301.5))
    increment = event["increments"]["test_writer"]
    assert torch.equal(increment, torch.full_like(increment, 1.5))
    assert torch.equal(
        event["visible_state"],
        torch.full_like(event["visible_state"], 301.5),
    )
    assert event["visible_counter"] == 1


def _notification_increment(manager):
    manager.apply_increment(
        "air_temperature",
        UpdateItem(torch.tensor(1.0), mode=UpdateMode.ACCUMULATE, unit="K",
                   per_timestep=True, source_functor="test_writer"),
        dt=1.0,
    )


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_committed_notifications_keep_successful_prefix_and_retry_pending_tail(monkeypatch, failure_type):
    manager = _build_state_manager()
    manager._contract_registry = None
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    record = observer.record_state_update

    def interrupt_second(step, *args):
        if step == 1:
            raise failure_type("notification interrupted")
        record(step, *args)

    monkeypatch.setattr(observer, "record_state_update", interrupt_second)
    manager.begin_provenance_transaction()
    for _ in range(3):
        _notification_increment(manager)
    assert observer.events == []
    expected = RuntimeError if failure_type is OSError else KeyboardInterrupt
    with pytest.raises(expected) as error:
        manager.commit_provenance_transaction()
    if failure_type is OSError:
        assert isinstance(error.value.__cause__, OSError)
    assert manager._provenance_transaction_active is False
    assert [event["step_index"] for event in observer.events] == [0]
    assert [event["step"] for event in manager._pending_provenance_events] == [1, 2]
    assert manager.get_state("air_temperature").item() == 303.0
    assert manager._update_counter == 3
    manager.rollback_provenance_transaction()
    assert [event["step"] for event in manager._pending_provenance_events] == [1, 2]
    monkeypatch.setattr(observer, "record_state_update", record)
    manager.commit_provenance_transaction()
    assert [event["step_index"] for event in observer.events] == [0, 1, 2]
    assert manager._pending_provenance_events == []
    assert manager.get_state("air_temperature").item() == 303.0
    assert manager._update_counter == 3


@pytest.mark.parametrize("operation", ["begin", "reset", "rebind", "rollback"])
def test_lifecycle_does_not_discard_unconfirmed_accepted_notifications(monkeypatch, operation):
    manager = _build_state_manager()
    manager._contract_registry = None
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    record = observer.record_state_update
    replacement = _CommitObserver(manager)

    def fail_record(*args):
        raise OSError("notification unavailable")

    monkeypatch.setattr(observer, "record_state_update", fail_record)
    manager.begin_provenance_transaction()
    _notification_increment(manager)
    with pytest.raises(RuntimeError, match="notification"):
        manager.commit_provenance_transaction()
    action = {
        "begin": manager.begin_provenance_transaction,
        "reset": manager.reset,
        "rebind": lambda: manager.set_provenance_recorder(replacement),
        "rollback": manager.rollback_provenance_transaction,
    }[operation]
    if operation == "rollback":
        action()
    else:
        with pytest.raises(RuntimeError, match="notification"):
            action()
    assert manager._provenance_transaction_active is False
    assert len(manager._pending_provenance_events) == 1
    assert manager.get_state("air_temperature").item() == 301.0
    assert manager._update_counter == 1
    assert manager._provenance_recorder is observer
    monkeypatch.setattr(observer, "record_state_update", record)
    manager.commit_provenance_transaction()
    assert len(observer.events) == 1
    assert manager._pending_provenance_events == []


def test_direct_batch_notification_failure_preserves_all_values_events_and_report(monkeypatch):
    manager = _build_state_manager()
    manager._schema["soil_temperature"] = StateVariableSpec(
        name="soil_temperature", units="K", bounds=(0.0, 400.0), conserved=False, required=False,
    )
    manager.set_state({"air_temperature": torch.tensor(300.0), "soil_temperature": torch.tensor(280.0)})
    manager._contract_registry = None
    manager._conservation_enabled = True
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    record = observer.record_state_update

    def fail_record(*args):
        raise OSError("batch notification unavailable")

    monkeypatch.setattr(observer, "record_state_update", fail_record)
    updates = {name: UpdateItem(torch.tensor(1.0), mode=UpdateMode.ACCUMULATE, unit="K", per_timestep=True)
               for name in ("air_temperature", "soil_temperature")}
    with pytest.raises(RuntimeError, match="notification"):
        manager.apply_updates(updates, dt=1.0, update_increments={
            name: {"writer": torch.tensor(1.0)} for name in updates
        })
    assert manager.get_state("air_temperature").item() == 301.0
    assert manager.get_state("soil_temperature").item() == 281.0
    assert manager._update_counter == 1
    assert len(manager._pending_provenance_events) == 2
    assert len(manager.conservation_history) == 1
    with pytest.raises(RuntimeError, match="notification"):
        _notification_increment(manager)
    assert manager.get_state("air_temperature").item() == 301.0
    assert manager._update_counter == 1
    monkeypatch.setattr(observer, "record_state_update", record)
    manager.commit_provenance_transaction()
    assert [event["variable_name"] for event in observer.events] == ["air_temperature", "soil_temperature"]
    assert manager._pending_provenance_events == []
    assert len(manager.conservation_history) == 1


def test_notification_freezing_failure_precedes_state_publication(monkeypatch):
    manager = _build_state_manager()
    manager._contract_registry = None
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    clone = manager._clone_transaction_value

    def reject_event(value):
        if isinstance(value, dict) and "variable" in value:
            raise TypeError("event cannot be frozen")
        return clone(value)

    monkeypatch.setattr(manager, "_clone_transaction_value", reject_event)
    with pytest.raises(TypeError, match="cannot be frozen"):
        _notification_increment(manager)
    assert manager.get_state("air_temperature").item() == 300.0
    assert manager._update_counter == 0
    assert manager._pending_provenance_events == []
    assert observer.events == []


@pytest.mark.parametrize("dt", [-1.0, 0.0, float("nan"), float("inf"), True])
def test_state_update_rejects_invalid_integration_intervals_before_mutation(dt):
    manager = _build_state_manager()
    before = manager.get_state()
    counter = manager._update_counter
    with pytest.raises((TypeError, ValueError), match="time step|interval"):
        manager.apply_updates(
            {"air_temperature": UpdateItem(torch.tensor(1.0), mode=UpdateMode.ACCUMULATE,
                                            unit="K", per_timestep=True)},
            dt=dt,
        )
    assert torch.equal(manager.get_state("air_temperature"), before["air_temperature"])
    assert manager._update_counter == counter


def test_notification_delivery_rejects_reentrant_transactions_without_reemitting(monkeypatch):
    manager = _build_state_manager()
    manager._contract_registry = None
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    attempts = 0

    def reenter(*args):
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            raise ValueError("observer was entered twice")
        manager.begin_provenance_transaction()

    monkeypatch.setattr(observer, "record_state_update", reenter)
    with pytest.raises(RuntimeError, match="reenter"):
        _notification_increment(manager)
    assert attempts == 1
    assert manager.get_state("air_temperature").item() == 301.0
    assert manager._update_counter == 1
    assert manager._provenance_transaction_active is False
    assert len(manager._pending_provenance_events) == 1


@pytest.mark.parametrize("source", ["map", "item", "unspecified"])
def test_replacement_updates_notify_actual_state_change_and_explicit_source(source):
    manager = _build_state_manager()
    manager._contract_registry = None
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    manager.apply_updates(
        {"air_temperature": UpdateItem(torch.tensor(310.0), mode=UpdateMode.REPLACE, unit="K",
                                       source_functor="forcing" if source == "item" else None)},
        dt=1.0, check_conservation=False,
        update_sources={"air_temperature": "forcing"} if source == "map" else None,
    )
    assert len(observer.events) == 1
    event = observer.events[0]
    assert event["pre_value"].item() == 300.0
    assert event["post_value"].item() == 310.0
    assert event["unit"] == "K"
    if source == "unspecified":
        assert event["increments"] == {}
    else:
        assert set(event["increments"]) == {"forcing"}
        assert event["increments"]["forcing"].item() == 10.0


def test_replacement_notification_excludes_diagnostic_only_updates():
    manager = _build_state_manager()
    manager._contract_registry = None
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    manager.apply_updates({
        "air_temperature": UpdateItem(torch.tensor(305.0), mode=UpdateMode.REPLACE),
        "diagnostics.sample": UpdateItem(torch.tensor(7.0), write_mode=StateWriteMode.DERIVED),
    }, dt=1.0, check_conservation=False)
    assert [event["variable_name"] for event in observer.events] == ["air_temperature"]
    assert "diagnostics.sample" not in manager.get_state()


def test_native_notification_keeps_its_frozen_run_and_destination(tmp_path, monkeypatch):
    import json
    import relic.core.provenance.recorder as recorder_module
    from relic.core.provenance import ProvenanceRecorder
    from relic.core.simulator.checkpoint import _encode_checkpoint_value
    manager = _build_state_manager()
    manager._contract_registry = None
    recorder = ProvenanceRecorder(tmp_path / "old", run_id="old-run")
    manager.set_provenance_recorder(recorder)
    with monkeypatch.context() as local:
        def unavailable(*args):
            raise OSError("native log unavailable")
        local.setattr(recorder_module, "_append_runtime_event", unavailable)
        with pytest.raises(RuntimeError, match="notification"):
            _notification_increment(manager)
    assert manager.get_state("air_temperature").item() == 301.0
    _encode_checkpoint_value(manager.checkpoint(), path="manager")
    recorder.run_id = "new-run"
    recorder.output_dir = tmp_path / "new"
    manager.commit_provenance_transaction()
    assert not (tmp_path / "new").exists()
    assert manager._pending_provenance_events == []
    rows = [json.loads(line) for line in (tmp_path / "old" / "runtime.log.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["run_id"] == "old-run"
    assert manager._update_counter == 1


def test_invalid_native_notification_context_is_rejected_before_state_commit(tmp_path):
    from relic.core.provenance import ProvenanceRecorder
    manager = _build_state_manager()
    manager._contract_registry = None
    recorder = ProvenanceRecorder(tmp_path / "invalid", run_id="valid")
    manager.set_provenance_recorder(recorder)
    recorder.run_id = ""
    with pytest.raises(ValueError, match="run_id"):
        _notification_increment(manager)
    assert manager.get_state("air_temperature").item() == 300.0
    assert manager._update_counter == 0
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("late_attachment", [False, True])
def test_provenance_attachment_uses_the_supplied_simulator_identity(tmp_path, late_attachment):
    from relic.core.provenance import ProvenanceRecorder
    manager = _build_state_manager()
    recorder = ProvenanceRecorder(tmp_path, run_id="standalone")
    if late_attachment:
        manager.set_provenance_recorder(None, run_id="simulator-run")
        manager.set_provenance_recorder(recorder)
    else:
        manager.set_provenance_recorder(recorder, run_id="simulator-run")
    assert recorder.run_id == "simulator-run"


def test_provenance_identity_binding_does_not_add_fields_to_plain_observers():
    manager = _build_state_manager()
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer, run_id="simulator-run")
    assert not hasattr(observer, "run_id")


def test_replace_mode_sets_absolute_value_without_double_counting():
    """Replace mode should override state instead of accumulating."""
    manager = _build_state_manager()
    updates = {"air_temperature": UpdateItem(torch.tensor(300.0), mode=UpdateMode.REPLACE)}

    manager.apply_updates(updates, dt=1.0, check_conservation=False)

    result = manager.get_state("air_temperature")
    assert torch.isclose(result, torch.tensor(300.0)), "Replace mode should not accumulate absolute forcing"


def test_replace_mode_computes_delta_from_previous_state():
    """Replace mode should update to target value even when different from base."""
    manager = _build_state_manager()
    updates = {"air_temperature": UpdateItem(torch.tensor(310.0), mode=UpdateMode.REPLACE)}

    manager.apply_updates(updates, dt=1.0, check_conservation=True)

    result = manager.get_state("air_temperature")
    assert torch.isclose(result, torch.tensor(310.0)), "Replace mode should set final state to target"


def test_accumulate_mode_preserves_delta_semantics():
    """Accumulate mode should continue to add deltas."""
    manager = _build_state_manager()
    updates = {"air_temperature": UpdateItem(torch.tensor(2.5), mode=UpdateMode.ACCUMULATE)}

    manager.apply_updates(updates, dt=1.0, check_conservation=False)

    result = manager.get_state("air_temperature")
    assert torch.isclose(result, torch.tensor(302.5)), "Accumulate mode should add deltas to state"


def test_replace_mode_normalizes_delta_for_conservation(monkeypatch):
    """Replace mode should feed conservation check with delta, not absolute value."""
    manager = _build_state_manager()
    captured = {}

    def _capture_check(self, updates, dt=None, scale_factor=None):
        captured.update(updates)
        return {}

    monkeypatch.setattr(StateManager, "_check_conservation", _capture_check, raising=True)

    updates = {"air_temperature": UpdateItem(torch.tensor(305.0), mode=UpdateMode.REPLACE)}
    manager.apply_updates(updates, dt=1.0, check_conservation=True)

    assert "air_temperature" in captured
    assert torch.isclose(captured["air_temperature"], torch.tensor(5.0))
    assert torch.isclose(manager.get_state("air_temperature"), torch.tensor(305.0))


def test_apply_updates_accepts_scalar_payloads_and_tuples():
    """Union FluxDict values (floats or tuple payloads) should be normalized safely."""
    manager = _build_state_manager()
    updates = {
        "air_temperature": (1.5, UpdateMode.ACCUMULATE),
    }

    manager.apply_updates(updates, dt=1.0, check_conservation=False)

    assert torch.isclose(manager.get_state("air_temperature"), torch.tensor(301.5))


@pytest.mark.parametrize("basis", ["per_second", "instantaneous"])
def test_instantaneous_flux_aggregate_replaces_value_between_transactions(basis):
    """Flux writers add within one transaction but do not sum historical rates."""
    manager = StateManager(
        schema={
            "n2o_flux": StateVariableSpec(
                name="n2o_flux",
                units="g N m-2 s-1",
                conserved=False,
                required=False,
            )
        },
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="n2o_flux",
            kind="flux",
            udunits="g N m-2 s-1",
            owner=None,
            incrementable=True,
            writers=["nitrification", "denitrification"],
            increment_writers=["nitrification", "denitrification"],
        )
    )
    manager._contract_registry = registry
    manager.set_state({"n2o_flux": torch.tensor(0.0)}, validate=False)

    for expected_rate in (torch.tensor(3.0e-8), torch.tensor(5.0e-8)):
        manager.apply_updates(
            {
                "n2o_flux": UpdateItem(
                    value=expected_rate,
                    mode=UpdateMode.ACCUMULATE,
                    unit="g N m-2 s-1",
                    basis=basis,
                    per_timestep=False,
                )
            },
            dt=10.0,
            check_conservation=False,
            update_sources={"n2o_flux": "denitrification"},
            update_increments={
                "n2o_flux": {
                    "nitrification": expected_rate * 0.4,
                    "denitrification": expected_rate * 0.6,
                }
            },
        )
        assert torch.allclose(manager.get_state("n2o_flux"), expected_rate)


def test_delta_updates_apply_dt_and_map_to_target():
    """delta_ keys should map to the base state variable and apply dt."""
    manager = _build_state_manager()
    updates = {
        "delta_air_temperature": torch.tensor(1.5),
    }

    manager.apply_updates(updates, dt=2.0, check_conservation=False)

    assert torch.isclose(manager.get_state("air_temperature"), torch.tensor(303.0))


def test_delta_updates_without_dt_keep_increment():
    """delta_ keys should apply raw increments when dt is None."""
    manager = _build_state_manager()
    updates = {
        "delta_air_temperature": torch.tensor(2.0),
    }

    manager.apply_updates(updates, dt=None, check_conservation=False)

    assert torch.isclose(manager.get_state("air_temperature"), torch.tensor(302.0))


def test_conservation_ledger_merges_parameter_and_canonical(monkeypatch):
    """Parameter ledger entries should merge with canonical defaults instead of replacing them."""
    calls = []

    def _fake_build_conservation_ledger(contract_registry, force_reload=False):
        calls.append((contract_registry, force_reload))
        return {"carbon": ["soil_carbon_fast", "param_pool"]}

    monkeypatch.setattr(
        state_conservation_module,
        "build_conservation_ledger",
        _fake_build_conservation_ledger,
        raising=True,
    )

    manager = StateManager(schema={})
    ledger = manager.get_conservation_ledger()
    carbon_pools = ledger.get("carbon", [])

    assert "param_pool" in carbon_pools
    assert "soil_carbon_fast" in carbon_pools
    assert len(calls) == 1


def test_state_manager_has_one_conservation_ledger_initialization_path():
    manager = StateManager(schema={})
    assert not hasattr(manager, "_load_parameterized_conservation_ledger")


def test_state_package_does_not_export_internal_ledger_builders():
    assert not hasattr(state_package, "load_parameter_ledger")
    assert not hasattr(state_package, "load_parameterized_conservation_ledger")
    assert not hasattr(state_package, "initialize_default_conservation_ledger")
    assert not hasattr(
        state_conservation_module,
        "load_parameterized_conservation_ledger",
    )


def test_apply_updates_rejects_element_hint_conflicting_with_scm():
    schema = {
        "leaf_carbon": StateVariableSpec(
            name="leaf_carbon",
            units="kg C m-2",
            conserved=True,
            required=False,
        )
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="leaf_carbon",
            kind="state",
            udunits="kg C m-2",
            accumulates="carbon.total",
        )
    )
    manager._contract_registry = registry
    manager.set_conservation_ledger({"carbon": ["leaf_carbon"]})
    manager.set_state({"leaf_carbon": torch.tensor(1.0)}, validate=False)

    with pytest.raises(ValueError, match="element.*conflicts.*carbon"):
        manager.apply_updates(
            {
                "leaf_carbon": UpdateItem(
                    value=torch.tensor(0.1),
                    mode=UpdateMode.ACCUMULATE,
                    per_timestep=True,
                    element="N",
                )
            },
            check_conservation=False,
        )

    assert torch.equal(manager.get_state("leaf_carbon"), torch.tensor(1.0))


def test_apply_updates_requires_convertible_unit_for_element_hint():
    schema = {
        "unclassified_pool": StateVariableSpec(
            name="unclassified_pool",
            units="kg",
            conserved=True,
            required=False,
        )
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    manager.set_state({"unclassified_pool": torch.tensor(1.0)}, validate=False)

    with pytest.raises(
        ValueError,
        match="Conservation unit.*unclassified_pool",
    ):
        manager.apply_updates(
            {
                "unclassified_pool": UpdateItem(
                    value=torch.tensor(0.5),
                    mode=UpdateMode.ACCUMULATE,
                    per_timestep=True,
                    element="C",
                )
            },
            check_conservation=True,
            rollback_on_failure=True,
        )

    assert torch.equal(manager.get_state("unclassified_pool"), torch.tensor(1.0))


def test_unbalanced_water_storage_update_is_not_silently_accepted():
    schema = {
        "soil_water_storage": StateVariableSpec(
            name="soil_water_storage",
            units="kg m-2",
            conserved=True,
            required=False,
        )
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="soil_water_storage",
            kind="state",
            udunits="kg m-2",
            accumulates="water.total",
        )
    )
    manager._contract_registry = registry
    manager.set_conservation_ledger({"water": ["soil_water_storage"]})
    manager.set_state({"soil_water_storage": torch.tensor(10.0)}, validate=False)

    with pytest.raises(ValueError, match="Conservation violation for water"):
        manager.apply_updates(
            {
                "soil_water_storage": UpdateItem(
                    value=torch.tensor(0.5),
                    mode=UpdateMode.ACCUMULATE,
                    per_timestep=True,
                    element="W",
                )
            },
            check_conservation=True,
            rollback_on_failure=True,
        )

    assert torch.equal(manager.get_state("soil_water_storage"), torch.tensor(10.0))


def test_unbalanced_phosphorus_storage_update_is_not_silently_accepted():
    schema = {
        "available_phosphorus": StateVariableSpec(
            name="available_phosphorus",
            units="g P m-2",
            conserved=True,
            required=False,
        )
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="available_phosphorus",
            kind="state",
            udunits="g P m-2",
            accumulates="phosphorus.total",
        )
    )
    manager._contract_registry = registry
    manager.set_conservation_ledger(
        {"phosphorus": ["available_phosphorus"]}
    )
    manager.set_state(
        {"available_phosphorus": torch.tensor(1.0)},
        validate=False,
    )

    with pytest.raises(ValueError, match="Conservation violation for phosphorus"):
        manager.apply_updates(
            {
                "available_phosphorus": UpdateItem(
                    value=torch.tensor(0.25),
                    mode=UpdateMode.ACCUMULATE,
                    per_timestep=True,
                    element="P",
                )
            },
            check_conservation=True,
            rollback_on_failure=True,
        )

    assert torch.equal(
        manager.get_state("available_phosphorus"),
        torch.tensor(1.0),
    )


def _build_open_phosphorus_budget_manager():
    schema = {
        "available_phosphorus": StateVariableSpec(
            name="available_phosphorus",
            units="g P m-2",
            conserved=True,
            required=False,
        ),
        "fertilizer_p_flux": StateVariableSpec(
            name="fertilizer_p_flux",
            units="g P m-2 s-1",
            conserved=True,
            required=False,
        ),
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="available_phosphorus",
            kind="state",
            udunits="g P m-2",
            owner=None,
            accumulates="phosphorus.total",
            flux_in=["fertilization_management"],
            flux_out=["root_nutrient_uptake"],
            writers=["fertilization_management", "root_nutrient_uptake"],
            increment_writers=[
                "fertilization_management",
                "root_nutrient_uptake",
            ],
        )
    )
    registry._register_contract(
        VariableContract(
            variable="fertilizer_p_flux",
            kind="flux",
            udunits="g P m-2 s-1",
            owner="fertilization_management",
            accumulates="phosphorus.total",
            flux_in=["fertilization_management"],
            flux_out=[],
            writers=["fertilization_management"],
            tendency_writers=["fertilization_management"],
        )
    )
    manager._contract_registry = registry
    manager.set_conservation_ledger(
        {
            "phosphorus": [
                "available_phosphorus",
                "fertilizer_p_flux",
            ]
        }
    )
    manager.set_state(
        {
            "available_phosphorus": torch.tensor(1.0),
            "fertilizer_p_flux": torch.tensor(0.0),
        },
        validate=False,
    )
    return manager


def test_open_phosphorus_transaction_balances_storage_against_input_flux():
    manager = _build_open_phosphorus_budget_manager()
    dt = 86400.0
    phosphorus_input = torch.tensor(0.872)
    phosphorus_flux = phosphorus_input / dt
    updates = {
        "available_phosphorus": UpdateItem(
            value=phosphorus_input,
            mode=UpdateMode.ACCUMULATE,
            unit="g P m-2",
            per_timestep=True,
            element="P",
            source_functor="fertilization_management",
        ),
        "fertilizer_p_flux": UpdateItem(
            value=phosphorus_flux,
            mode=UpdateMode.REPLACE,
            unit="g P m-2 s-1",
            per_timestep=False,
            element=None,
            source_functor="fertilization_management",
        ),
    }

    manager.apply_updates(
        updates,
        dt=dt,
        check_conservation=True,
        rollback_on_failure=True,
        update_sources={
            "available_phosphorus": "fertilization_management",
            "fertilizer_p_flux": "fertilization_management",
        },
        update_increments={
            "available_phosphorus": {
                "fertilization_management": phosphorus_input,
            }
        },
    )

    assert torch.allclose(
        manager.get_state("available_phosphorus"),
        torch.tensor(1.872),
    )
    assert torch.allclose(
        manager.get_state("fertilizer_p_flux"),
        phosphorus_flux,
    )


def test_open_phosphorus_transaction_rejects_negative_input_flux():
    manager = _build_open_phosphorus_budget_manager()
    dt = 86400.0
    phosphorus_input = torch.tensor(0.872)

    with pytest.raises(ValueError, match="input flux.*non-negative"):
        manager.apply_updates(
            {
                "available_phosphorus": UpdateItem(
                    value=phosphorus_input,
                    mode=UpdateMode.ACCUMULATE,
                    unit="g P m-2",
                    per_timestep=True,
                    element="P",
                    source_functor="fertilization_management",
                ),
                "fertilizer_p_flux": UpdateItem(
                    value=-(phosphorus_input / dt),
                    mode=UpdateMode.REPLACE,
                    unit="g P m-2 s-1",
                    per_timestep=False,
                    element=None,
                    source_functor="fertilization_management",
                ),
            },
            dt=dt,
            check_conservation=True,
            rollback_on_failure=True,
            update_sources={
                "available_phosphorus": "fertilization_management",
                "fertilizer_p_flux": "fertilization_management",
            },
            update_increments={
                "available_phosphorus": {
                    "fertilization_management": phosphorus_input,
                }
            },
        )


def test_open_phosphorus_transaction_rejects_flux_unit_mismatch():
    manager = _build_open_phosphorus_budget_manager()
    dt = 86400.0
    phosphorus_input = torch.tensor(0.872)

    with pytest.raises(ValueError, match="budget unit mismatch"):
        manager.apply_updates(
            {
                "available_phosphorus": UpdateItem(
                    value=phosphorus_input,
                    mode=UpdateMode.ACCUMULATE,
                    unit="g P m-2",
                    per_timestep=True,
                    element="P",
                    source_functor="fertilization_management",
                ),
                "fertilizer_p_flux": UpdateItem(
                    value=phosphorus_input / dt,
                    mode=UpdateMode.REPLACE,
                    unit="kg P m-2 s-1",
                    per_timestep=False,
                    element=None,
                    source_functor="fertilization_management",
                ),
            },
            dt=dt,
            check_conservation=True,
            rollback_on_failure=True,
            update_sources={
                "available_phosphorus": "fertilization_management",
                "fertilizer_p_flux": "fertilization_management",
            },
            update_increments={
                "available_phosphorus": {
                    "fertilization_management": phosphorus_input,
                }
            },
        )


def _build_mixed_unit_nitrogen_budget_manager():
    schema = {
        "nh4_nitrogen": StateVariableSpec(
            name="nh4_nitrogen",
            units="g N m-2",
            conserved=True,
            required=False,
        ),
        "fertilizer_n_flux": StateVariableSpec(
            name="fertilizer_n_flux",
            units="kg N m-2 s-1",
            conserved=True,
            required=False,
        ),
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="nh4_nitrogen",
            kind="state",
            udunits="g N m-2",
            owner=None,
            accumulates="nitrogen.total",
            writers=["NitrogenCycling"],
            increment_writers=["NitrogenCycling"],
        )
    )
    registry._register_contract(
        VariableContract(
            variable="fertilizer_n_flux",
            kind="flux",
            udunits="kg N m-2 s-1",
            owner="fertilization_management",
            accumulates="nitrogen.total",
            flux_in=["fertilization_management"],
            flux_out=[],
            writers=["fertilization_management"],
            tendency_writers=["fertilization_management"],
        )
    )
    manager._contract_registry = registry
    manager.set_conservation_ledger(
        {"nitrogen": ["nh4_nitrogen", "fertilizer_n_flux"]}
    )
    manager.set_state(
        {
            "nh4_nitrogen": torch.tensor(1.0),
            "fertilizer_n_flux": torch.tensor(0.0),
        },
        validate=False,
    )
    return manager


def test_open_nitrogen_budget_normalizes_kg_flux_to_g_storage():
    manager = _build_mixed_unit_nitrogen_budget_manager()
    dt = 86400.0
    nitrogen_input_g = torch.tensor(4.6)
    nitrogen_flux_kg = nitrogen_input_g / (1000.0 * dt)

    manager.apply_updates(
        {
            "nh4_nitrogen": UpdateItem(
                value=nitrogen_input_g,
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                per_timestep=True,
                element="N",
                source_functor="NitrogenCycling",
            ),
            "fertilizer_n_flux": UpdateItem(
                value=nitrogen_flux_kg,
                mode=UpdateMode.REPLACE,
                unit="kg N m-2 s-1",
                per_timestep=False,
                element=None,
                source_functor="fertilization_management",
            ),
        },
        dt=dt,
        check_conservation=True,
        rollback_on_failure=True,
        update_sources={
            "nh4_nitrogen": "NitrogenCycling",
            "fertilizer_n_flux": "fertilization_management",
        },
        update_increments={
            "nh4_nitrogen": {"NitrogenCycling": nitrogen_input_g}
        },
    )

    assert torch.allclose(manager.get_state("nh4_nitrogen"), torch.tensor(5.6))


def test_conservation_report_reuses_normalized_open_nitrogen_budget():
    manager = _build_mixed_unit_nitrogen_budget_manager()
    dt = 86400.0
    nitrogen_input_g = torch.tensor(4.6)
    updates = {
        "nh4_nitrogen": UpdateItem(
            value=nitrogen_input_g,
            mode=UpdateMode.ACCUMULATE,
            unit="g N m-2",
            per_timestep=True,
            element="N",
            source_functor="NitrogenCycling",
        ),
        "fertilizer_n_flux": UpdateItem(
            value=nitrogen_input_g / (1000.0 * dt),
            mode=UpdateMode.REPLACE,
            unit="kg N m-2 s-1",
            per_timestep=False,
            element=None,
            source_functor="fertilization_management",
        ),
    }

    summary = manager.generate_conservation_report(
        updates,
        dt=dt,
        store_history=False,
    )

    assert summary.nitrogen_report is not None
    assert summary.nitrogen_report.balance_error == pytest.approx(0.0, abs=1e-9)
    guardian = summary.custom_checks["conservation_guardian"]["nitrogen.total"]
    assert guardian["passed"] is True
    assert guardian["residual"] == pytest.approx(0.0, abs=1e-9)


def test_closed_nitrogen_transfer_normalizes_molar_plant_storage():
    nitrogen_molar_mass = float(
        get_physical_constant("relic.physical.constants.molecular_weights.nitrogen")
    )
    schema = {
        "nh4_nitrogen": StateVariableSpec(
            name="nh4_nitrogen",
            units="g N m-2",
            conserved=True,
            required=False,
        ),
        "plant_n_storage": StateVariableSpec(
            name="plant_n_storage",
            units="mol N m-2",
            conserved=True,
            required=False,
        ),
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    for variable, unit in (
        ("nh4_nitrogen", "g N m-2"),
        ("plant_n_storage", "mol N m-2"),
    ):
        registry._register_contract(
            VariableContract(
                variable=variable,
                kind="state",
                udunits=unit,
                owner="root_nutrient_uptake",
                accumulates="nitrogen.total",
                writers=["root_nutrient_uptake"],
                increment_writers=["root_nutrient_uptake"],
            )
        )
    manager._contract_registry = registry
    manager.set_conservation_ledger(
        {"nitrogen": ["nh4_nitrogen", "plant_n_storage"]}
    )
    manager.set_state(
        {
            "nh4_nitrogen": torch.tensor(100.0),
            "plant_n_storage": torch.tensor(0.0),
        },
        validate=False,
    )

    manager.apply_updates(
        {
            "nh4_nitrogen": UpdateItem(
                value=torch.tensor(-nitrogen_molar_mass),
                mode=UpdateMode.ACCUMULATE,
                unit="g N m-2",
                per_timestep=True,
                element="N",
                source_functor="root_nutrient_uptake",
            ),
            "plant_n_storage": UpdateItem(
                value=torch.tensor(1.0),
                mode=UpdateMode.ACCUMULATE,
                unit="mol N m-2",
                per_timestep=True,
                element="N",
                source_functor="root_nutrient_uptake",
            ),
        },
        dt=1.0,
        check_conservation=True,
        rollback_on_failure=True,
        update_sources={
            "nh4_nitrogen": "root_nutrient_uptake",
            "plant_n_storage": "root_nutrient_uptake",
        },
        update_increments={
            "nh4_nitrogen": {
                "root_nutrient_uptake": torch.tensor(-nitrogen_molar_mass)
            },
            "plant_n_storage": {
                "root_nutrient_uptake": torch.tensor(1.0)
            },
        },
    )

    assert manager.get_state("plant_n_storage").item() == pytest.approx(1.0)


def test_closed_carbon_transfer_normalizes_kg_and_g_storage():
    schema = {
        "leaf_carbon": StateVariableSpec(
            name="leaf_carbon",
            units="kg C m-2",
            conserved=True,
            required=False,
        ),
        "soil_carbon": StateVariableSpec(
            name="soil_carbon",
            units="g C m-2",
            conserved=True,
            required=False,
        ),
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    for variable, unit in (
        ("leaf_carbon", "kg C m-2"),
        ("soil_carbon", "g C m-2"),
    ):
        registry._register_contract(
            VariableContract(
                variable=variable,
                kind="state",
                udunits=unit,
                owner="carbon_transfer",
                accumulates="carbon.total",
                writers=["carbon_transfer"],
                increment_writers=["carbon_transfer"],
            )
        )
    manager._contract_registry = registry
    manager.set_conservation_ledger(
        {"carbon": ["leaf_carbon", "soil_carbon"]}
    )
    manager.set_state(
        {"leaf_carbon": torch.tensor(1.0), "soil_carbon": torch.tensor(0.0)},
        validate=False,
    )

    manager.apply_updates(
        {
            "leaf_carbon": UpdateItem(
                value=torch.tensor(-0.001),
                mode=UpdateMode.ACCUMULATE,
                unit="kg C m-2",
                per_timestep=True,
                element="C",
                source_functor="carbon_transfer",
            ),
            "soil_carbon": UpdateItem(
                value=torch.tensor(1.0),
                mode=UpdateMode.ACCUMULATE,
                unit="g C m-2",
                per_timestep=True,
                element="C",
                source_functor="carbon_transfer",
            ),
        },
        dt=1.0,
        check_conservation=True,
        rollback_on_failure=True,
        update_sources={
            "leaf_carbon": "carbon_transfer",
            "soil_carbon": "carbon_transfer",
        },
        update_increments={
            "leaf_carbon": {"carbon_transfer": torch.tensor(-0.001)},
            "soil_carbon": {"carbon_transfer": torch.tensor(1.0)},
        },
    )

    assert manager.get_state("soil_carbon").item() == pytest.approx(1.0)


def test_conservation_error_contributors_follow_explicit_membership():
    schema = {
        "leaf_carbon": StateVariableSpec(
            name="leaf_carbon",
            units="kg C m-2",
            conserved=True,
            required=False,
        ),
        "carbon_observation": StateVariableSpec(
            name="carbon_observation",
            units="kg C m-2",
            conserved=False,
            required=False,
        ),
    }
    manager = StateManager(
        schema=schema,
        autoload_contract_registry=False,
        device="cpu",
    )
    registry = ContractRegistry()
    registry._register_contract(
        VariableContract(
            variable="leaf_carbon",
            kind="state",
            udunits="kg C m-2",
            accumulates="carbon.total",
        )
    )
    registry._register_contract(
        VariableContract(
            variable="carbon_observation",
            kind="diagnostic",
            udunits="kg C m-2",
            accumulates=None,
        )
    )
    manager._contract_registry = registry
    manager.set_conservation_ledger({"carbon": ["leaf_carbon"]})
    manager.set_state(
        {
            "leaf_carbon": torch.tensor(1.0),
            "carbon_observation": torch.tensor(1.0),
        },
        validate=False,
    )

    with pytest.raises(ValueError, match="Conservation violation for carbon") as exc:
        manager.apply_updates(
            {
                "leaf_carbon": UpdateItem(
                    value=torch.tensor(0.5),
                    mode=UpdateMode.ACCUMULATE,
                    per_timestep=True,
                    element="C",
                ),
                "carbon_observation": UpdateItem(
                    value=torch.tensor(10.0),
                    mode=UpdateMode.REPLACE,
                    per_timestep=True,
                ),
            },
            check_conservation=True,
            rollback_on_failure=True,
        )

    assert "leaf_carbon=" in str(exc.value)
    assert "carbon_observation" not in str(exc.value)


def test_run_specs_overrides_accumulate_when_not_allowed():
    manager = _build_state_manager()
    run_specs = RunSpecs(update_rules={"soil_moisture": "accumulate"}, strict=False)
    manager.set_run_specs(run_specs)

    updates = {"air_temperature": UpdateItem(torch.tensor(310.0), mode=UpdateMode.ACCUMULATE)}
    manager.apply_updates(updates, dt=1.0, check_conservation=False)

    result = manager.get_state("air_temperature")
    assert torch.isclose(result, torch.tensor(310.0))


def test_run_specs_strict_raises_on_accumulate_without_rule():
    manager = _build_state_manager()
    run_specs = RunSpecs(update_rules={}, strict=True)
    manager.set_run_specs(run_specs)

    updates = {"air_temperature": UpdateItem(torch.tensor(310.0), mode=UpdateMode.ACCUMULATE)}
    with pytest.raises(ValueError):
        manager.apply_updates(updates, dt=1.0, check_conservation=False)


def test_derived_write_mode_skips_state_and_records_diagnostic():
    manager = _build_state_manager()
    updates = {
        "air_temperature": UpdateItem(
            torch.tensor(305.0),
            mode=UpdateMode.REPLACE,
            write_mode=StateWriteMode.DERIVED,
        )
    }
    manager.apply_updates(updates, dt=1.0, check_conservation=False)
    assert torch.isclose(manager.get_state("air_temperature"), torch.tensor(300.0))
    assert "air_temperature" in manager._current_diagnostics


def test_immutable_write_mode_blocks_updates_when_strict():
    manager = _build_state_manager()
    run_specs = RunSpecs(strict=True)
    manager.set_run_specs(run_specs)
    updates = {
        "air_temperature": UpdateItem(
            torch.tensor(310.0),
            mode=UpdateMode.REPLACE,
            write_mode=StateWriteMode.IMMUTABLE,
        )
    }
    with pytest.raises(ValueError):
        manager.apply_updates(updates, dt=1.0, check_conservation=False)


def test_external_intent_owns_an_immutable_data_descriptor():
    from dataclasses import FrozenInstanceError
    from relic.core.state.types import ExternalIntent

    descriptor = {"variables": ["air_temperature"], "options": {"enabled": True}}
    intent = ExternalIntent("write_outputs", "output_file", "snapshot", descriptor)
    descriptor["variables"].append("soil_temperature")
    descriptor["options"]["enabled"] = False

    assert intent.payload["variables"] == ("air_temperature",)
    assert intent.payload["options"]["enabled"] is True
    with pytest.raises(TypeError):
        intent.payload["new"] = 1
    with pytest.raises(TypeError):
        intent.payload["options"]["enabled"] = False
    with pytest.raises(FrozenInstanceError):
        intent.operation = "another_operation"


def test_state_package_exports_the_single_output_contract_owner():
    from relic.core.state import types

    assert state_package.ExternalIntent is types.ExternalIntent
    assert state_package.partition_outputs is types.partition_outputs


@pytest.mark.parametrize("payload", [
    {"live": torch.tensor(1.0)}, {"live": object()}, {"live": lambda: None},
    {"value": float("nan")}, {"value": float("inf")}, {1: "ambiguous_key"},
])
def test_external_intent_rejects_live_objects_and_non_data_descriptors(payload):
    from relic.core.state.types import ExternalIntent

    with pytest.raises((TypeError, ValueError)):
        ExternalIntent("write_outputs", "output_file", "snapshot", payload)


def test_external_intent_rejects_cyclic_descriptors():
    from relic.core.state.types import ExternalIntent

    payload = {}
    payload["cycle"] = payload
    with pytest.raises(ValueError, match="cyclic"):
        ExternalIntent("write_outputs", "output_file", "snapshot", payload)


@pytest.mark.parametrize("field", ["operation", "resource", "request_id"])
def test_external_intent_requires_explicit_identity_fields(field):
    from relic.core.state.types import ExternalIntent

    values = {"operation": "write_outputs", "resource": "output_file", "request_id": "snapshot"}
    values[field] = " "
    with pytest.raises(ValueError, match=field):
        ExternalIntent(**values)


def test_output_partition_preserves_state_gradients_and_raw_diagnostics():
    from relic.core.state.types import ExternalIntent, partition_outputs

    parameter = torch.tensor(2.0, requires_grad=True)
    state_update = UpdateItem(parameter * 3.0)
    residual = torch.tensor(0.0)
    derived = UpdateItem(torch.tensor(305.0), mode=UpdateMode.REPLACE, write_mode=StateWriteMode.DERIVED)
    intent = ExternalIntent("export_diagnostics", "diagnostics", "diagnostic_snapshot")
    raw = {"pool": state_update, "diagnostics.residual": residual, "air_temperature": derived, "export": intent}

    state, diagnostics, intents = partition_outputs(raw)

    assert list(state) == ["pool"]
    assert list(diagnostics) == ["diagnostics.residual", "air_temperature"]
    assert intents == {"export": intent}
    assert state["pool"] is state_update
    assert diagnostics["diagnostics.residual"] is residual
    assert diagnostics["air_temperature"] is derived
    state["pool"].value.backward()
    assert parameter.grad.item() == 3.0
    assert len(raw) == 4


def test_output_partition_uses_resolved_write_mode_without_a_second_resolver():
    from relic.core.state.types import partition_outputs

    value = torch.tensor(7.0)
    state, diagnostics, intents = partition_outputs(
        {"reported_temperature": value},
        write_modes={"reported_temperature": StateWriteMode.DERIVED},
    )
    assert state == {}
    assert diagnostics["reported_temperature"] is value
    assert intents == {}


def test_output_partition_does_not_guess_undeclared_numbers_are_diagnostics():
    from relic.core.state.types import partition_outputs

    state, diagnostics, intents = partition_outputs({"undeclared_output": 2.0})
    assert state == {"undeclared_output": 2.0}
    assert diagnostics == intents == {}
    raw_mapping = {"nested": 2.0}
    state, diagnostics, _ = partition_outputs({"undeclared_output": raw_mapping})
    assert state["undeclared_output"] is raw_mapping
    assert diagnostics == {}


def test_output_partition_defers_existing_update_payload_coercion_to_state_manager():
    from relic.core.state.types import partition_outputs

    manager = _build_state_manager()
    payload = {"value": torch.tensor(310.0), "mode": "replace", "units": "K"}
    state, diagnostics, intents = partition_outputs({"air_temperature": payload})
    assert state["air_temperature"] is payload
    assert diagnostics == intents == {}
    manager.apply_updates(state, dt=1.0, check_conservation=False)
    assert manager.get_state("air_temperature").item() == 310.0


def test_output_partition_preserves_explicit_structured_diagnostics():
    from relic.core.state.types import partition_outputs

    payload = {"count": 2, "residual": torch.tensor(0.0)}
    state, diagnostics, intents = partition_outputs({"diagnostics.balance": payload})
    assert diagnostics["diagnostics.balance"] is payload
    assert state == intents == {}


@pytest.mark.parametrize("conflict", ["declared", "diagnostic", "wrapped"])
def test_output_partition_rejects_conflicting_external_intent_markers(conflict):
    from relic.core.state.types import ExternalIntent, partition_outputs

    intent = ExternalIntent("export_diagnostics", "diagnostics", "snapshot")
    key = "diagnostics.export" if conflict == "diagnostic" else "export"
    value = UpdateItem(intent) if conflict == "wrapped" else intent
    with pytest.raises((TypeError, ValueError), match="ExternalIntent|external intent"):
        partition_outputs({key: value}, declared_outputs=("export",) if conflict == "declared" else ())


@pytest.mark.parametrize("encoding", ["direct", "wrapped", "nested_diagnostic"])
def test_state_manager_rejects_external_intents_before_any_mutation(encoding):
    from relic.core.state import ExternalIntent

    manager = _build_state_manager()
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    intent = ExternalIntent("write_outputs", "output_file", "snapshot")
    bad_name, bad_value = {
        "direct": ("export", intent),
        "wrapped": ("export", UpdateItem(intent)),
        "nested_diagnostic": ("diagnostics.export", {"request": intent}),
    }[encoding]
    counter_before = manager._update_counter
    schema_before = set(manager.schema)
    with pytest.raises(TypeError, match="ExternalIntent"):
        manager.apply_updates({
            "air_temperature": UpdateItem(torch.tensor(310.0), mode=UpdateMode.REPLACE),
            bad_name: bad_value,
        }, dt=1.0, check_conservation=False)
    assert manager.get_state("air_temperature").item() == 300.0
    assert manager._update_counter == counter_before
    assert set(manager.schema) == schema_before
    assert observer.events == []


def test_state_manager_does_not_silently_discard_invalid_state_payload():
    manager = _build_state_manager()
    counter_before = manager._update_counter
    with pytest.raises(TypeError, match="state payload"):
        manager.apply_updates({"air_temperature": "not-numeric"}, check_conservation=False)
    assert manager.get_state("air_temperature").item() == 300.0
    assert manager._update_counter == counter_before


@pytest.mark.parametrize("entrypoint", ["batch", "single"])
@pytest.mark.parametrize(
    "mode,basis,integrated,value,unit,dt,expected",
    [
        (UpdateMode.ACCUMULATE, "per_step", True, 2.0, "degC", None, 302.0),
        (UpdateMode.REPLACE, "per_step", True, 20.0, "degC", None, 293.15),
        (UpdateMode.ACCUMULATE, "per_second", False, 2.0, "K", 3.0, 306.0),
        (UpdateMode.ACCUMULATE, "per_second", True, 2.0, "K", 3.0, 302.0),
        (UpdateMode.FLUX, "per_second", False, 2.0, "K", 3.0, 306.0),
        (UpdateMode.FLUX, "per_second", True, 2.0, "K", 3.0, 302.0),
        (UpdateMode.FLUX, "per_step", False, 2.0, "K", 3.0, 306.0),
        (UpdateMode.FLUX, "per_second", False, 2.0, "K s-1", 3.0, 306.0),
        (UpdateMode.ACCUMULATE, "per_step", True, 2.0, "K s-1", 3.0, 302.0),
        (UpdateMode.REPLACE, "per_second", False, 310.0, "K", 3.0, 310.0),
    ],
)
def test_single_and_batch_share_units_and_exactly_once_time_integration(
    entrypoint, mode, basis, integrated, value, unit, dt, expected,
):
    """Absolute temperatures, differences, rates and amounts remain distinct."""
    manager = StateManager({
        "audit_temperature": StateVariableSpec(name="audit_temperature", units="K"),
    })
    manager.initialize({"audit_temperature": torch.tensor(300.0, dtype=torch.float64)})
    payload = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    item = UpdateItem(payload, mode=mode, unit=unit, basis=basis, per_timestep=integrated)
    if entrypoint == "single":
        manager.apply_increment("audit_temperature", item, dt=dt)
    else:
        manager.apply_updates({"audit_temperature": item}, dt=dt)
    actual = manager.get_state("audit_temperature")
    assert actual.dtype == torch.float64
    assert actual.item() == pytest.approx(expected, rel=0.0, abs=1.0e-12)
    actual.backward()
    expected_gradient = 3.0 if expected == 306.0 else 1.0
    assert payload.grad.item() == pytest.approx(expected_gradient, rel=0.0, abs=1.0e-12)


@pytest.mark.parametrize("entrypoint", ["batch", "single"])
def test_unit_mismatch_rejects_before_state_and_counter_change(entrypoint):
    manager = StateManager({
        "audit_temperature": StateVariableSpec(name="audit_temperature", units="K"),
    })
    manager.initialize({"audit_temperature": torch.tensor(300.0, dtype=torch.float64)})
    item = UpdateItem(torch.tensor(2.0), unit="kg", per_timestep=True)
    counter = manager._update_counter
    with pytest.raises(ValueError, match="[Uu]nit"):
        if entrypoint == "single":
            manager.apply_increment("audit_temperature", item, dt=1.0)
        else:
            manager.apply_updates({"audit_temperature": item}, dt=1.0)
    assert manager.get_state("audit_temperature").item() == 300.0
    assert manager._update_counter == counter


@pytest.mark.parametrize("entrypoint", ["batch", "single"])
@pytest.mark.parametrize("mode", [UpdateMode.ACCUMULATE, UpdateMode.FLUX])
def test_unintegrated_state_rate_requires_an_interval(entrypoint, mode):
    manager = StateManager({"audit_stock": StateVariableSpec(name="audit_stock", units="kg")})
    manager.initialize({"audit_stock": torch.tensor(1.0, dtype=torch.float64)})
    item = UpdateItem(torch.tensor(2.0), mode=mode, unit="kg", basis="per_second")
    with pytest.raises(ValueError, match="dt|interval"):
        if entrypoint == "single":
            manager.apply_increment("audit_stock", item)
        else:
            manager.apply_updates({"audit_stock": item})
    assert manager.get_state("audit_stock").item() == 1.0


@pytest.mark.parametrize("entrypoint", ["batch", "single"])
def test_scalar_update_preserves_existing_float64_precision(entrypoint):
    manager = StateManager({"audit_scalar": StateVariableSpec(name="audit_scalar", units="dimensionless")})
    manager.initialize({"audit_scalar": torch.tensor(0.0, dtype=torch.float64)})
    value = 1.0 + 2.0 ** -40
    item = UpdateItem(value, unit="dimensionless", per_timestep=True)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        if entrypoint == "single":
            manager.apply_increment("audit_scalar", item)
        else:
            manager.apply_updates({"audit_scalar": item})
    finally:
        torch.set_default_dtype(previous_dtype)
    assert manager.get_state("audit_scalar").item() == value


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_closed_transfer_checks_realized_pool_changes(dtype):
    """A lost donor decrement must not certify a created recipient amount."""
    manager = StateManager({
        name: StateVariableSpec(name=name, units="kg C m-2", conserved=True,
                                accumulates="carbon.total")
        for name in ("audit_donor", "audit_recipient")
    })
    manager.initialize({
        "audit_donor": torch.tensor(100000000.0, dtype=dtype),
        "audit_recipient": torch.tensor(0.0, dtype=dtype),
    })
    before = manager.get_state()
    counter = manager._update_counter
    updates = {
        name: UpdateItem(torch.tensor(value, dtype=dtype), unit="kg C m-2",
                         per_timestep=True, element="carbon")
        for name, value in (("audit_donor", -1.0), ("audit_recipient", 1.0))
    }
    if dtype == torch.float32:
        with pytest.raises(ValueError, match="Conservation violation for carbon"):
            manager.apply_updates(updates, dt=1.0, rollback_on_failure=True)
        assert manager._update_counter == counter
        for name, value in before.items():
            assert torch.equal(manager.get_state(name), value)
    else:
        manager.apply_updates(updates, dt=1.0, rollback_on_failure=True)
        actual = manager.get_state()
        assert actual["audit_donor"].item() == 99999999.0
        assert actual["audit_recipient"].item() == 1.0
        assert sum((actual[name].double() - before[name].double()).item() for name in before) == 0.0


def test_nonblocking_conservation_reports_realized_roundoff_residual():
    manager = StateManager({
        name: StateVariableSpec(name=name, units="kg C m-2", conserved=True,
                                accumulates="carbon.total")
        for name in ("audit_donor", "audit_recipient")
    }, conservation_checks={"raise_on_violation": False})
    manager.initialize({"audit_donor": torch.tensor(100000000.0, dtype=torch.float32),
                        "audit_recipient": torch.tensor(0.0, dtype=torch.float32)})
    manager.apply_updates({
        name: UpdateItem(torch.tensor(value, dtype=torch.float32), unit="kg C m-2", per_timestep=True,
                         element="carbon")
        for name, value in (("audit_donor", -1.0), ("audit_recipient", 1.0))
    }, dt=1.0)
    report = manager.get_last_conservation_summary()
    assert report.carbon_report.balance_error == pytest.approx(1000.0)
    assert report.total_violations > 0


def test_conservation_reduction_cannot_hide_a_realized_vector_residual():
    """Float64 reduction must retain a small term between large opposing terms."""
    manager = StateManager({
        "audit_carbon": StateVariableSpec(
            name="audit_carbon", units="g C m-2", conserved=True,
            accumulates="carbon.total", spatial_dims=1,
        ),
    }, device="cpu")
    initial = torch.tensor([0.0, 0.0, 2.0 ** 24], dtype=torch.float32)
    manager.initialize({"audit_carbon": initial})
    delta = torch.tensor([2.0 ** 24, 1.0, -(2.0 ** 24)], dtype=torch.float32)
    assert delta.sum().item() == 0.0
    assert delta.double().sum().item() == 1.0
    counter = manager._update_counter
    with pytest.raises(ValueError, match="Conservation violation for carbon"):
        manager.apply_updates({
            "audit_carbon": UpdateItem(delta, unit="g C m-2", per_timestep=True,
                                       element="carbon"),
        }, dt=1.0)
    assert torch.equal(manager.get_state("audit_carbon"), initial)
    assert manager._update_counter == counter


@pytest.mark.parametrize("entrypoint", ["batch", "single"])
@pytest.mark.parametrize("interval_shape", [(), (1,)])
def test_scalar_tensor_interval_preserves_precision_and_rate_gradients(entrypoint, interval_shape):
    """The exact q*dt integral must retain both tensor input derivatives."""
    manager = StateManager({
        "audit_time_integral": StateVariableSpec(
            name="audit_time_integral", units="kg", conserved=False,
        ),
    }, device="cpu")
    manager.initialize({"audit_time_integral": torch.tensor(0.0, dtype=torch.float64)})
    rate = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    interval = torch.full(interval_shape, 1.0 + 2.0 ** -30, dtype=torch.float64,
                          requires_grad=True)
    item = UpdateItem(rate, mode=UpdateMode.FLUX, unit="kg s-1",
                      basis="per_second", per_timestep=False)
    if entrypoint == "single":
        manager.apply_increment("audit_time_integral", item, dt=interval)
    else:
        manager.apply_updates({"audit_time_integral": item}, dt=interval)
    result = manager.get_state("audit_time_integral")
    assert result.shape == torch.Size([])
    assert result.dtype == torch.float64
    assert result.item() == 2.0 * (1.0 + 2.0 ** -30)
    rate_gradient, interval_gradient = torch.autograd.grad(result, (rate, interval))
    assert rate_gradient.item() == 1.0 + 2.0 ** -30
    torch.testing.assert_close(interval_gradient, torch.full_like(interval, 2.0), rtol=0, atol=0)


@pytest.mark.parametrize("interval", [
    torch.tensor([1.0, 2.0]), torch.tensor(float("nan")),
    torch.tensor(float("inf")), torch.tensor(True), torch.tensor(1.0 + 1.0j),
])
def test_invalid_tensor_interval_is_refused_before_state_change(interval):
    manager = _build_state_manager()
    before = manager.get_state("air_temperature").clone()
    counter = manager._update_counter
    with pytest.raises((TypeError, ValueError), match="interval|time step"):
        manager.apply_updates({
            "air_temperature": UpdateItem(torch.tensor(1.0), unit="K", per_timestep=True),
        }, dt=interval)
    assert torch.equal(manager.get_state("air_temperature"), before)
    assert manager._update_counter == counter


@pytest.mark.parametrize("mode", [UpdateMode.ACCUMULATE, UpdateMode.REPLACE])
def test_budget_subtracts_stored_values_before_narrowing_the_difference(mode):
    """An audit subtraction must retain a small pre-existing recipient stock."""
    manager = StateManager({
        name: StateVariableSpec(name=name, units="g C m-2", conserved=True,
                                accumulates="carbon.total")
        for name in ("audit_donor", "audit_recipient")
    }, device="cpu")
    initial = {"audit_donor": torch.tensor(2.0 ** 25, dtype=torch.float32),
               "audit_recipient": torch.tensor(1.0, dtype=torch.float32)}
    manager.initialize(initial)
    donor_value = -(2.0 ** 25) if mode == UpdateMode.ACCUMULATE else 0.0
    updates = {
        name: UpdateItem(torch.tensor(amount, dtype=torch.float32), mode=mode,
                         unit="g C m-2", per_timestep=True, element="carbon")
        for name, amount in (("audit_donor", donor_value), ("audit_recipient", 2.0 ** 25))
    }
    with pytest.raises(ValueError, match="Conservation violation for carbon"):
        manager.apply_updates(updates, dt=1.0, rollback_on_failure=True)
    for name, before in initial.items():
        assert torch.equal(manager.get_state(name), before)


@pytest.mark.parametrize("rate_dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("tensor_interval", [False, True])
def test_rate_integration_uses_destination_and_interval_precision(rate_dtype, tensor_interval):
    """Integer rates and narrow rates must not quantize a fractional interval."""
    manager = StateManager({
        "audit_integral": StateVariableSpec(name="audit_integral", units="kg", conserved=False),
    }, device="cpu")
    manager.initialize({"audit_integral": torch.tensor(0.0, dtype=torch.float64)})
    seconds = 0.5 if rate_dtype == torch.int64 else 1.0 + 2.0 ** -30
    interval = torch.tensor(seconds, dtype=torch.float64, requires_grad=True) if tensor_interval else seconds
    manager.apply_updates({
        "audit_integral": UpdateItem(torch.tensor(2, dtype=rate_dtype),
                                     mode=UpdateMode.FLUX, unit="kg s-1", per_timestep=False),
    }, dt=interval)
    result = manager.get_state("audit_integral")
    assert result.dtype == torch.float64
    assert result.item() == 2.0 * seconds
    if tensor_interval:
        assert torch.autograd.grad(result, interval)[0].item() == 2.0


def test_clipping_disclosure_keeps_the_small_retained_stock():
    manager = StateManager({
        "audit_clip": StateVariableSpec(name="audit_clip", units="dimensionless",
                                        conserved=False, bounds=(0.0, 1.0)),
    }, device="cpu")
    manager.initialize({"audit_clip": torch.tensor(0.0, dtype=torch.float32)})
    manager.apply_updates({
        "audit_clip": UpdateItem(torch.tensor(2.0 ** 25, dtype=torch.float32),
                                 mode=UpdateMode.REPLACE, unit="dimensionless", per_timestep=True),
    }, dt=1.0)
    assert manager.get_state("audit_clip").item() == 1.0
    assert manager._clip_loss_diagnostics["audit_clip"] == 2.0 ** 25 - 1.0


def test_automatic_provenance_increment_matches_realized_precision():
    manager = StateManager({
        name: StateVariableSpec(name=name, units="g C m-2", conserved=True,
                                accumulates="carbon.total")
        for name in ("audit_donor", "audit_recipient")
    }, conservation_checks={"raise_on_violation": False}, device="cpu")
    manager.initialize({"audit_donor": torch.tensor(2.0 ** 25, dtype=torch.float32),
                        "audit_recipient": torch.tensor(1.0, dtype=torch.float32)})
    observer = _CommitObserver(manager)
    manager.set_provenance_recorder(observer)
    manager.apply_updates({
        name: UpdateItem(torch.tensor(amount, dtype=torch.float32), unit="g C m-2",
                         per_timestep=True, element="carbon", source_functor="audit_writer")
        for name, amount in (("audit_donor", -(2.0 ** 25)), ("audit_recipient", 2.0 ** 25))
    }, dt=1.0)
    recipient_event = next(event for event in observer.events if event["variable_name"] == "audit_recipient")
    realized = recipient_event["post_value"].double() - recipient_event["pre_value"].double()
    assert realized.item() == 2.0 ** 25 - 1.0
    assert recipient_event["increments"]["audit_writer"].item() == realized.item()
    assert manager.get_last_conservation_summary().carbon_report.balance_error == pytest.approx(1.0)


@pytest.mark.parametrize("entrypoint", ["batch", "single"])
@pytest.mark.parametrize("interval", [1.0, 17.0])
def test_instantaneous_replacement_keeps_existing_point_value_semantics(entrypoint, interval):
    manager = _build_state_manager()
    update = UpdateItem(torch.tensor(295.0), mode=UpdateMode.REPLACE,
                        unit="K", basis="instantaneous", per_timestep=False)
    if entrypoint == "single":
        manager.apply_increment("air_temperature", update, dt=interval)
    else:
        manager.apply_updates({"air_temperature": update}, dt=interval)
    assert manager.get_state("air_temperature").item() == 295.0


def test_instantaneous_basis_is_not_a_stock_accumulation_rate():
    manager = _build_state_manager()
    before = manager.get_state("air_temperature").clone()
    with pytest.raises(ValueError, match="instantaneous"):
        manager.apply_updates({
            "air_temperature": UpdateItem(torch.tensor(1.0), mode=UpdateMode.ACCUMULATE,
                                           unit="K", basis="instantaneous", per_timestep=False),
        }, dt=2.0)
    assert torch.equal(manager.get_state("air_temperature"), before)
