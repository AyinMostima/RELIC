"""Provider identity, committed view, lifecycle receipts, epoch freeze, and swap.

Covers the registration-owned provider lifecycle (C03/C04) and the epoch and
quiescent-swap contracts consumed by the step transaction (D01/D02):

- three content digests (implementation/contract/binding) versus instance UIDs;
- immutable committed views that later registry mutation cannot pollute;
- receipts for committed, rolled-back, and compensation-incomplete transactions;
- one frozen view per step transaction, also when a registration lands mid-step;
- reader tokens, retire-then-dispose ordering, CAS failure, and dispose failure
  under real thread interleaving controlled by Barrier/Event.
"""

from __future__ import annotations

import threading
import types

import pytest
import torch

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
)
from relic.core.simulator import RELICSimulator
from relic.core.simulator.registration import (
    CommittedProviderView,
    LifecycleReceipt,
    ProviderIdentity,
)


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'consumer_out': 'dimensionless',
    'writer_a_out': 'dimensionless',
    'writer_b_out': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


def _metadata(name, outputs, inputs=()):
    return FunctorMetadata(
        name=name,
        version="1.0.0",
        description="provider lifecycle fixture",
        category=ProcessCategory.TRANSPORT_MECHANISM,
        execution_phase=ExecutionPhase.MAIN_PROCESSES,
        references=["test fixture"],
        input_variables=list(inputs),
        output_variables=list(outputs),
    )


class _ConstantWriter(TransportMechanismFunctor):
    """Writes one constant value; records the step epoch it observed."""

    TRANSPORT_TYPE = "diffusion"

    def __init__(self, name="writer_a", value=2.5, dependencies=(), config=None):
        metadata = _metadata(name, [f"{name}_out"])
        super().__init__(
            name=name,
            config=dict(config or {}, value=float(value)),
            metadata=metadata,
        )
        self._dependency_names = list(dependencies)
        self.parameter_values["gain"] = 1.0
        self.observed_epochs = []
        self.on_forward = None

    def get_dependencies(self):
        return list(self._dependency_names)

    def forward(self, state, time, dt, **kwargs):
        if self.on_forward is not None:
            self.on_forward(self)
        return {f"{self.name}_out": torch.tensor(self._original_config["value"])}


class _RecordingWriter(_ConstantWriter):
    """Same content as _ConstantWriter but a different kernel implementation."""

    def forward(self, state, time, dt, **kwargs):
        if self.on_forward is not None:
            self.on_forward(self)
        return {f"{self.name}_out": torch.tensor(self._original_config["value"]) * 1.0}


def _simulator(*names, **config):
    sim = _declared_simulator({
        "device": "cpu",
        "conservation_checks": False,
        "strict_dependencies": False,
        "adaptive_dt_enabled": False,
        **config,
    })
    sim.initialize_state({f"{name}_out": torch.tensor(0.0) for name in names})
    return sim


def _identity(sim, name) -> ProviderIdentity:
    return sim.get_committed_provider_view().provider(name)


def _receipts(sim, operation=None):
    receipts = sim.get_lifecycle_receipts()
    if operation is None:
        return receipts
    return tuple(r for r in receipts if r.operation == operation)


# ---------------------------------------------------------------------------
# C03: identity digests, UIDs, committed view
# ---------------------------------------------------------------------------


class TestProviderIdentity:
    def test_empty_registry_is_published_as_epoch_one(self):
        sim = _simulator()
        view = sim.get_committed_provider_view()
        assert isinstance(view, CommittedProviderView)
        assert view.epoch == 1 and sim.provider_epoch == 1
        assert dict(view.providers) == {}
        assert view.registry_signature == ()

    def test_reinstantiated_provider_shares_digests_but_never_reuses_uid(self):
        first = _simulator("writer_a")
        second = _simulator("writer_a")
        first.register_functor(_ConstantWriter(), execution_group="fast")
        second.register_functor(_ConstantWriter(), execution_group="fast")
        a = _identity(first, "writer_a")
        b = _identity(second, "writer_a")
        assert a.strict_identity and b.strict_identity
        assert a.implementation_digest == b.implementation_digest
        assert a.contract_digest == b.contract_digest
        assert a.binding_digest == b.binding_digest
        assert a.provider_uid != b.provider_uid

        first.unregister_functor("writer_a")
        assert a.provider_uid in first._provider_lifecycle.retired_uids
        first.register_functor(_ConstantWriter(), execution_group="fast")
        c = _identity(first, "writer_a")
        assert c.provider_uid != a.provider_uid
        assert c.implementation_digest == a.implementation_digest
        assert c.contract_digest == a.contract_digest
        assert c.binding_digest == a.binding_digest

    def test_binding_contract_and_implementation_digests_are_separate(self):
        base = _simulator("writer_a")
        base.register_functor(_ConstantWriter(), execution_group="fast")
        reference = _identity(base, "writer_a")

        rebound = _simulator("writer_a")
        functor = _ConstantWriter()
        functor.parameter_values["gain"] = 2.0
        rebound.register_functor(functor, execution_group="fast")
        bound = _identity(rebound, "writer_a")
        assert bound.implementation_digest == reference.implementation_digest
        assert bound.contract_digest == reference.contract_digest
        assert bound.binding_digest != reference.binding_digest

        reconfigured = _simulator("writer_a")
        reconfigured.register_functor(
            _ConstantWriter(config={"schema": "v2"}), execution_group="fast"
        )
        contract = _identity(reconfigured, "writer_a")
        assert contract.implementation_digest == reference.implementation_digest
        assert contract.contract_digest != reference.contract_digest

        regrouped = _simulator("writer_a")
        regrouped.register_functor(_ConstantWriter(), execution_group="medium")
        assert _identity(regrouped, "writer_a").binding_digest != reference.binding_digest
        assert _identity(regrouped, "writer_a").contract_digest == reference.contract_digest

        replaced = _simulator("writer_a")
        replaced.register_functor(_RecordingWriter(), execution_group="fast")
        other = _identity(replaced, "writer_a")
        assert other.implementation_digest != reference.implementation_digest
        assert other.binding_digest == reference.binding_digest

    def test_in_place_forward_replacement_changes_implementation_digest(self):
        class _Local(_ConstantWriter):
            pass

        sim = _simulator("writer_a")
        sim.register_functor(_Local(), execution_group="fast")
        before = _identity(sim, "writer_a").implementation_digest

        def replacement(self, state, time, dt, **kwargs):
            return {f"{self.name}_out": torch.tensor(9.0)}

        _Local.forward = replacement
        sim.unregister_functor("writer_a")
        sim.register_functor(_Local(), execution_group="fast")
        assert _identity(sim, "writer_a").implementation_digest != before

    def test_untraceable_kernel_narrows_identity_instead_of_repr_fallback(self):
        namespace = {}
        exec(
            "def forward(self, state, time, dt, **kwargs):\n"
            "    return {'writer_a_out': torch.tensor(1.0)}\n",
            {"torch": torch},
            namespace,
        )
        dynamic = types.new_class("_Dynamic", (_ConstantWriter,), {})
        dynamic.forward = namespace["forward"]
        sim = _simulator("writer_a")
        sim.register_functor(dynamic(), execution_group="fast")
        identity = _identity(sim, "writer_a")
        assert identity.strict_identity is False
        assert identity.implementation_digest is None
        assert identity.identity_rejection
        assert identity.contract_digest and identity.binding_digest

    def test_committed_view_is_immutable_and_isolated_from_registry_mutation(self):
        sim = _simulator("writer_a", "writer_b")
        sim.register_functor(_ConstantWriter("writer_a"), execution_group="fast")
        first = sim.get_committed_provider_view()
        sim.register_functor(_ConstantWriter("writer_b"), execution_group="fast")
        second = sim.get_committed_provider_view()
        assert second.epoch == first.epoch + 1
        assert set(first.providers) == {"writer_a"}
        assert set(second.providers) == {"writer_a", "writer_b"}
        assert second.provider("writer_a").provider_uid == first.provider("writer_a").provider_uid
        with pytest.raises(TypeError):
            first.providers["writer_b"] = second.provider("writer_b")  # type: ignore[index]
        sim._groups["fast"].append("ghost")
        assert "ghost" not in second.groups["fast"]
        assert second.order_fast == ("writer_a", "writer_b") or set(second.order_fast) == {"writer_a", "writer_b"}
        with pytest.raises(KeyError):
            first.provider("writer_b")
        sim._groups["fast"].remove("ghost")

    def test_unmet_required_dependency_is_recorded_or_refused_by_mode(self):
        # Ordinary mode: the view is published with the unresolved requirement
        # recorded (the existing "execution order remains unresolved" contract).
        sim = _simulator("writer_a", "consumer")
        sim.register_functor(_ConstantWriter("writer_a"), execution_group="fast")
        epoch = sim.provider_epoch
        sim.register_functor(
            _ConstantWriter("consumer", dependencies=["missing_provider"]),
            execution_group="fast",
        )
        receipt = _receipts(sim, "register")[-1]
        assert receipt.status == "committed" and receipt.view_published is True
        assert receipt.epoch_after == epoch + 1
        assert receipt.managed_delta["provider_view"]["unresolved_required"] == [
            "consumer -> missing_provider",
        ]
        view = sim.get_committed_provider_view()
        assert view.unresolved_required == ("consumer -> missing_provider",)
        assert "consumer" in view.providers
        sim.step(dt=1.0)
        assert sim.get_state()["consumer_out"].item() == 2.5
        sim.initialize_state({"missing_provider_out": torch.tensor(0.0)})
        sim.register_functor(_ConstantWriter("missing_provider"), execution_group="fast")
        assert sim.get_committed_provider_view().unresolved_required == ()
        assert sim.provider_epoch == epoch + 2

        # Strict mode: the registration itself is refused and rolled back, so
        # no view with an unmet requirement is ever published.
        strict = _simulator("writer_a", "consumer", strict_dependencies=True)
        strict.register_functor(_ConstantWriter("writer_a"), execution_group="fast")
        strict_view = strict.get_committed_provider_view()
        with pytest.raises(RuntimeError, match="Missing required dependencies"):
            strict.register_functor(
                _ConstantWriter("consumer", dependencies=["missing_provider"]),
                execution_group="fast",
            )
        assert strict.get_committed_provider_view() is strict_view
        assert "consumer" not in strict.functors
        assert _receipts(strict)[-1].status == "rolled_back"


# ---------------------------------------------------------------------------
# C04: lifecycle receipts and short transactions
# ---------------------------------------------------------------------------


class TestLifecycleReceipts:
    def test_committed_registration_and_retirement_receipts(self):
        sim = _simulator("writer_a")
        functor = _ConstantWriter()
        sim.register_functor(functor, execution_group="fast")
        receipt = _receipts(sim, "register")[-1]
        assert isinstance(receipt, LifecycleReceipt)
        identity = _identity(sim, "writer_a")
        assert receipt.provider_uid == identity.provider_uid
        assert receipt.implementation_digest == identity.implementation_digest
        assert receipt.contract_digest == identity.contract_digest
        assert receipt.binding_digest == identity.binding_digest
        assert receipt.epoch_before == 1 and receipt.epoch_after == 2
        assert receipt.view_published is True and receipt.status == "committed"
        assert receipt.completed_actions[0] == f"module_membership:add:{identity.provider_uid}"
        assert receipt.completed_actions[-1] == "provider_view:publish:2"
        assert any(action.startswith("device_migration:") for action in receipt.completed_actions)
        assert receipt.compensated_actions == ()
        assert receipt.managed_delta["module"] == {"added": "writer_a"}
        assert receipt.managed_delta["group"] == {"name": "fast", "added": "writer_a"}
        assert receipt.managed_delta["execution_order"]["after"] == ["writer_a"]
        assert receipt.managed_delta["device"]["after"] == "cpu"
        assert "candidate_device" in receipt.recoverable_resources

        sim.unregister_functor("writer_a")
        retirement = _receipts(sim, "unregister")[-1]
        assert retirement.provider_uid == identity.provider_uid
        assert retirement.status == "committed" and retirement.view_published
        assert retirement.epoch_before == 2 and retirement.epoch_after == 3
        assert f"provider_uid:retire:{identity.provider_uid}" in retirement.completed_actions
        assert retirement.managed_delta["provider_view"]["retired_epoch"] == 2
        assert identity.provider_uid in sim._provider_lifecycle.retired_uids
        assert sim._provider_lifecycle.retired_status[2] == "disposed"

    @pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
    def test_failed_registration_receipt_lists_compensation_and_keeps_view(self, monkeypatch, failure):
        sim = _simulator("writer_a")
        view_before = sim.get_committed_provider_view()

        def fail_ledger():
            raise failure("late failure")

        monkeypatch.setattr(sim, "_refresh_conservation_ledger", fail_ledger)
        with pytest.raises(failure):
            sim.register_functor(_ConstantWriter(), execution_group="fast")
        receipt = _receipts(sim)[-1]
        assert receipt.operation == "register" and receipt.status == "rolled_back"
        assert receipt.view_published is False and receipt.epoch_after is None
        assert receipt.compensated_actions == tuple(reversed(receipt.completed_actions))
        assert receipt.completed_actions[-1] == "execution_order:refresh"
        assert "late failure" in receipt.error
        assert sim.get_committed_provider_view() is view_before
        assert sim.provider_epoch == view_before.epoch
        assert "writer_a" not in sim.functors
        assert "writer_a" not in sim._provider_lifecycle.provider_uids
        assert sim._provider_lifecycle.unpublished_providers == {}

    def test_incomplete_compensation_is_reported_in_receipt_and_exception(self, monkeypatch):
        sim = _simulator("writer_a")

        def fail_ledger():
            raise RuntimeError("late failure")

        def fail_restore(ledger):
            raise OSError("ledger restore unavailable")

        monkeypatch.setattr(sim, "_refresh_conservation_ledger", fail_ledger)
        monkeypatch.setattr(sim.state_manager, "set_conservation_ledger", fail_restore)
        with pytest.raises(BaseExceptionGroup) as info:
            sim.register_functor(_ConstantWriter(), execution_group="fast")
        assert info.group_contains(RuntimeError) and info.group_contains(OSError)
        receipt = _receipts(sim)[-1]
        assert receipt.status == "compensation_incomplete"
        assert "ledger restore unavailable" in receipt.error

    def test_repeated_retirement_does_not_repeat_release(self):
        sim = _simulator("writer_a")
        sim.register_functor(_ConstantWriter(), execution_group="fast")
        uid = _identity(sim, "writer_a").provider_uid
        assert sim.unregister_functor("writer_a") is True
        receipts_after_first = len(_receipts(sim))
        retired_after_first = set(sim._provider_lifecycle.retired_uids)
        epoch_after_first = sim.provider_epoch
        assert sim.unregister_functor("writer_a", strict=False) is False
        with pytest.raises(KeyError):
            sim.unregister_functor("writer_a")
        assert len(_receipts(sim)) == receipts_after_first
        assert set(sim._provider_lifecycle.retired_uids) == retired_after_first == {uid}
        assert sim.provider_epoch == epoch_after_first

    def test_failed_retirement_keeps_provider_and_uid_active(self, monkeypatch):
        sim = _simulator("writer_a")
        sim.register_functor(_ConstantWriter(), execution_group="fast")
        uid = _identity(sim, "writer_a").provider_uid
        view = sim.get_committed_provider_view()

        def fail_ledger():
            raise RuntimeError("retire failure")

        monkeypatch.setattr(sim, "_refresh_conservation_ledger", fail_ledger)
        with pytest.raises(RuntimeError, match="retire failure"):
            sim.unregister_functor("writer_a")
        receipt = _receipts(sim)[-1]
        assert receipt.operation == "unregister" and receipt.status == "rolled_back"
        assert receipt.provider_uid == uid
        assert sim.get_committed_provider_view() is view
        assert sim._provider_lifecycle.provider_uids["writer_a"] == uid
        assert uid not in sim._provider_lifecycle.retired_uids
        assert "writer_a" in sim.functors and "writer_a" in sim._groups["fast"]


# ---------------------------------------------------------------------------
# D01: epoch owner and step freeze
# ---------------------------------------------------------------------------


class TestEpochFreeze:
    def test_one_epoch_is_frozen_for_every_functor_in_the_step(self):
        sim = _simulator("writer_a", "writer_b")
        a = _ConstantWriter("writer_a")
        b = _ConstantWriter("writer_b")
        for functor in (a, b):
            functor.on_forward = lambda f, s=sim: f.observed_epochs.append(s._active_step_view.epoch)
        sim.register_functor(a, execution_group="fast")
        sim.register_functor(b, execution_group="fast")
        epoch = sim.provider_epoch
        sim.step(dt=1.0)
        assert a.observed_epochs == [epoch] and b.observed_epochs == [epoch]
        assert sim._active_step_view is None
        assert sim._provider_lifecycle.active_reader_count(epoch) == 0
        records = sim.get_variable_provenance("writer_a_out")
        writer_records = [r for r in records if r["role"] == "writer"]
        assert writer_records and writer_records[0]["provider_uid"] == _identity(sim, "writer_a").provider_uid
        assert writer_records[0]["provider_epoch"] == epoch

    def test_registration_during_a_step_does_not_mix_providers(self):
        sim = _simulator("writer_a", "writer_b")
        a = _ConstantWriter("writer_a")
        b = _ConstantWriter("writer_b")
        published = {}

        def register_b(functor):
            sim.register_functor(b, execution_group="fast")
            published["epoch"] = sim.provider_epoch
            published["old_status"] = sim._provider_lifecycle.retired_status[sim._active_step_view.epoch]
            published["readers"] = sim._provider_lifecycle.active_reader_count(sim._active_step_view.epoch)

        a.on_forward = register_b
        b.on_forward = lambda f, s=sim: f.observed_epochs.append(s._active_step_view.epoch)
        sim.register_functor(a, execution_group="fast")
        old_epoch = sim.provider_epoch
        sim.step(dt=1.0)
        assert published == {"epoch": old_epoch + 1, "old_status": "pending_dispose", "readers": 1}
        assert b.observed_epochs == []
        assert sim.get_state()["writer_b_out"].item() == 0.0
        assert sim._provider_lifecycle.retired_status[old_epoch] == "disposed"
        a.on_forward = None
        sim.step(dt=1.0)
        assert b.observed_epochs == [old_epoch + 1]
        assert sim.get_state()["writer_b_out"].item() == 2.5

    def test_rejected_step_keeps_the_committed_view_and_releases_the_token(self):
        sim = _simulator("writer_a")
        a = _ConstantWriter("writer_a")

        def explode(functor):
            raise RuntimeError("kernel failure")

        a.on_forward = explode
        sim.register_functor(a, execution_group="fast")
        view = sim.get_committed_provider_view()
        with pytest.raises(Exception, match="kernel failure"):
            sim.step(dt=1.0)
        assert sim.get_committed_provider_view() is view
        assert sim.provider_epoch == view.epoch
        assert sim._active_step_view is None
        assert sim._provider_lifecycle.active_reader_count(view.epoch) == 0
        a.on_forward = None
        sim.step(dt=1.0)
        assert sim.step_count == 1

    def test_same_values_with_a_different_implementation_publish_a_new_view(self):
        sim = _simulator("writer_a")
        sim.register_functor(_ConstantWriter(), execution_group="fast")
        first = _identity(sim, "writer_a")
        epoch = sim.provider_epoch
        sim.unregister_functor("writer_a")
        sim.register_functor(_RecordingWriter(), execution_group="fast")
        second = _identity(sim, "writer_a")
        assert sim.provider_epoch == epoch + 2
        assert second.implementation_digest != first.implementation_digest
        assert second.binding_digest == first.binding_digest
        assert second.provider_uid != first.provider_uid

    def test_groups_mutated_outside_the_owner_are_refused(self):
        sim = _simulator("writer_a")
        sim.register_functor(_ConstantWriter(), execution_group="fast")
        sim._groups_dirty = True
        with pytest.raises(RuntimeError, match="outside the registration owner"):
            sim.step(dt=1.0)
        sim._groups_dirty = False
        sim.step(dt=1.0)

    def test_checkpoint_binds_provider_digests_and_restore_publishes_a_fresh_epoch(self, tmp_path):
        sim = _simulator("writer_a")
        sim.register_functor(_ConstantWriter(), execution_group="fast")
        sim.step(dt=1.0)
        path = tmp_path / "view.pt"
        sim.create_checkpoint(path)
        payload = torch.load(path, weights_only=True)
        contract = payload["topology"]["scientific_contract"]["functors"]["writer_a"]
        identity = _identity(sim, "writer_a")
        assert contract["provider_identity"]["implementation_digest"] == identity.implementation_digest
        assert contract["provider_identity"]["contract_digest"] == identity.contract_digest
        assert contract["provider_identity"]["binding_digest"] == identity.binding_digest

        epoch_before = sim.provider_epoch
        sim.step(dt=1.0)
        receipt = sim.restore_checkpoint(path)
        assert receipt["provider_epoch"] == epoch_before + 1
        assert sim.provider_epoch == epoch_before + 1
        assert sim.step_count == 1
        assert sim.get_committed_provider_view().provider("writer_a").provider_uid == identity.provider_uid

        other = _simulator("writer_a")
        other.register_functor(_RecordingWriter(), execution_group="fast")
        with pytest.raises(ValueError, match="topology"):
            other.restore_checkpoint(path)


# ---------------------------------------------------------------------------
# D02: quiescent swap, reader tokens, drain, dispose (real threads)
# ---------------------------------------------------------------------------


class TestQuiescentSwap:
    def test_step_reader_finishes_on_the_old_view_and_dispose_waits_for_it(self):
        sim = _simulator("writer_a", "writer_b")
        a = _ConstantWriter("writer_a")
        entered = threading.Barrier(2, timeout=10)
        proceed = threading.Event()
        seen = {}

        def hold(functor):
            seen["epoch"] = sim._active_step_view.epoch
            entered.wait()
            assert proceed.wait(10)
            seen["view_after_swap"] = sim._active_step_view.epoch

        a.on_forward = hold
        sim.register_functor(a, execution_group="fast")
        old_epoch = sim.provider_epoch
        errors = []

        def run_step():
            try:
                sim.step(dt=1.0)
            except BaseException as exc:  # noqa: BLE001 - surfaced through join
                errors.append(exc)

        worker = threading.Thread(target=run_step, name="step-reader")
        worker.start()
        entered.wait()
        # Swap while the reader is inside the kernel: publication is immediate,
        # disposal of the old view is deferred to the reader's release.
        sim.register_functor(_ConstantWriter("writer_b"), execution_group="fast")
        lifecycle = sim._provider_lifecycle
        assert sim.provider_epoch == old_epoch + 1
        assert lifecycle.retired_status[old_epoch] == "pending_dispose"
        assert lifecycle.active_reader_count(old_epoch) == 1
        assert lifecycle.wait_quiescent(old_epoch, timeout=0.2) is False
        proceed.set()
        worker.join(10)
        assert not worker.is_alive() and errors == []
        assert seen == {"epoch": old_epoch, "view_after_swap": old_epoch}
        assert lifecycle.wait_quiescent(old_epoch, timeout=5) is True
        assert lifecycle.retired_status[old_epoch] == "disposed"
        assert old_epoch not in lifecycle.retired_views
        assert sim.get_state()["writer_b_out"].item() == 0.0
        a.on_forward = None
        sim.step(dt=1.0)
        assert sim.get_state()["writer_b_out"].item() == 2.5

    def test_concurrent_registrations_serialize_with_monotonic_epochs(self):
        sim = _simulator("writer_a", "writer_b")
        start = threading.Barrier(3, timeout=10)
        errors = []

        def register(name):
            try:
                start.wait()
                sim.register_functor(_ConstantWriter(name), execution_group="fast")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(name,)) for name in ("writer_a", "writer_b")]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(10)
        assert errors == []
        history = [entry["epoch"] for entry in sim._provider_lifecycle.publication_history]
        assert history == [1, 2, 3]
        assert set(sim.get_committed_provider_view().providers) == {"writer_a", "writer_b"}
        receipts = _receipts(sim, "register")
        assert sorted((r.epoch_before, r.epoch_after) for r in receipts) == [(1, 2), (2, 3)]
        assert sim._provider_lifecycle.retired_status == {1: "disposed", 2: "disposed"}

    def test_cas_failure_does_not_expose_a_second_plan(self, monkeypatch):
        sim = _simulator("writer_a")
        lifecycle = sim._provider_lifecycle
        view = sim.get_committed_provider_view()
        original = sim._compute_group_execution_orders

        def out_of_band_bump():
            lifecycle.epoch += 1
            original()

        monkeypatch.setattr(sim, "_compute_group_execution_orders", out_of_band_bump)
        with pytest.raises(RuntimeError, match="compare-and-swap"):
            sim.register_functor(_ConstantWriter(), execution_group="fast")
        assert lifecycle.committed_view is view
        assert "writer_a" not in sim.functors
        assert _receipts(sim)[-1].status == "rolled_back"
        assert lifecycle.publication_history[-1]["epoch"] == view.epoch

    def test_dispose_failure_is_observable_and_retryable_without_a_mixed_world(self, monkeypatch):
        sim = _simulator("writer_a", "writer_b")
        sim.register_functor(_ConstantWriter("writer_a"), execution_group="fast")
        lifecycle = sim._provider_lifecycle
        old_epoch = sim.provider_epoch
        calls = []
        genuine = lifecycle.dispose_view

        def failing_dispose(view):
            calls.append(view.epoch)
            if len(calls) == 1:
                raise OSError("native handle busy")
            genuine(view)

        monkeypatch.setattr(lifecycle, "dispose_view", failing_dispose)
        sim.register_functor(_ConstantWriter("writer_b"), execution_group="fast")
        assert lifecycle.retired_status[old_epoch] == "dispose_failed"
        assert "native handle busy" in lifecycle.dispose_errors[old_epoch]
        assert sim.get_committed_provider_view().epoch == old_epoch + 1
        sim.step(dt=1.0)
        assert sim.get_state()["writer_b_out"].item() == 2.5
        assert lifecycle.retry_dispose(old_epoch) == "disposed"
        assert old_epoch not in lifecycle.dispose_errors and calls == [old_epoch, old_epoch]

    def test_cancellation_inside_the_step_releases_the_reader_token(self):
        sim = _simulator("writer_a", "writer_b")
        a = _ConstantWriter("writer_a")

        def cancel(functor):
            raise KeyboardInterrupt("operator cancel")

        a.on_forward = cancel
        sim.register_functor(a, execution_group="fast")
        epoch = sim.provider_epoch
        with pytest.raises(KeyboardInterrupt):
            sim.step(dt=1.0)
        lifecycle = sim._provider_lifecycle
        assert lifecycle.active_reader_count(epoch) == 0
        assert sim._active_step_view is None
        sim.register_functor(_ConstantWriter("writer_b"), execution_group="fast")
        assert lifecycle.retired_status[epoch] == "disposed"

    def test_reader_token_release_is_idempotent(self):
        sim = _simulator()
        lifecycle = sim._provider_lifecycle
        with sim.acquire_provider_view() as token:
            assert lifecycle.active_reader_count(token.view.epoch) == 1
            token.release()
            token.release()
        assert lifecycle.active_reader_count(token.view.epoch) == 0
        assert token.released
        with pytest.raises(RuntimeError, match="more readers than acquired"):
            lifecycle.release_reader(token.view.epoch)
