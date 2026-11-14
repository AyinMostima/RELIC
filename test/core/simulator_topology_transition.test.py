"""Hybrid topology transitions committed by the lifecycle owner (G09-TOPO).

A declared transition retires and connects providers, remaps the state with
one transactional write, audits the declared ledger, and publishes exactly
one provider epoch at the accepted-step boundary. Requests raised inside a
step are executed after acceptance; that step runs on the view it started
with. Restart evidence: a post-onset checkpoint restores exactly into a
post-onset simulator and continues bitwise; a pre-onset checkpoint replays
the same onset; a renamed graph migrates under the F04 certificate.
"""

from __future__ import annotations

import pytest
import torch

import relic.core.simulator.registration as registration
from relic.core.simulator import RELICSimulator
from relic.core.simulator.checkpoint import compute_restart_topology_digests
from relic.core.simulator.registration import RemapRow, TopologyTransition
from relic.core.state import ExternalIntent
from relic.scientific.atmosphere.stochastic_rainfall import MarkedPoissonRainfallFunctor
from relic.scientific.hydrosphere.runoff.surface_ponding import (
    ACTIVE_VARIABLE,
    EXCESS_VARIABLE,
    PONDED_DEPTH_VARIABLE,
    RAINFALL_DEPTH_VARIABLE,
    TRANSITION_CESSATION,
    TRANSITION_ONSET,
    HortonianPondingEventFunctor,
    SurfacePondingFunctor,
    ponding_cessation_transition,
    ponding_onset_transition,
)

COUNT = "stochastic_rainfall_event_count"
DT = 3600.0
N_CELLS = 2
RAIN_OVERRIDES = {"event_rate": 4.0, "mean_depth": 40.0}

pytestmark = pytest.mark.filterwarnings("ignore:Conservation law 'none' violated")


def _simulator(*, seed=11, event_name="ponding_event", declare=True, connected=False, ponded=0.0, excess=0.0):
    sim = RELICSimulator({
        "device": "cpu", "conservation_checks": False, "strict_dependencies": False,
        "adaptive_dt_enabled": False, "time_step": DT, "output_frequency": 1,
    })
    z = torch.zeros(N_CELLS, dtype=torch.float64)
    sim.initialize_state({
        COUNT: z.clone(), RAINFALL_DEPTH_VARIABLE: z.clone(),
        PONDED_DEPTH_VARIABLE: torch.full_like(z, float(ponded)),
        EXCESS_VARIABLE: torch.full_like(z, float(excess)),
        ACTIVE_VARIABLE: torch.full_like(z, 1.0 if connected else 0.0),
    })
    rain = MarkedPoissonRainfallFunctor(name="stochastic_rainfall", config={"seed": seed, "parameters": dict(RAIN_OVERRIDES)})
    event = HortonianPondingEventFunctor(name=event_name)
    store = SurfacePondingFunctor(config={"event_provider": event_name})
    sim.register_functor(rain, execution_group="fast")
    sim.register_functor(event, execution_group="fast")
    if declare:
        sim.declare_topology_transition(ponding_onset_transition(store, execution_group="fast"))
        sim.declare_topology_transition(ponding_cessation_transition())
    return sim, store


def _run_until(sim, predicate, max_steps=60):
    for _ in range(max_steps):
        sim.step(dt=DT)
        if predicate(sim):
            return sim.step_count
    raise AssertionError("condition not reached")


def _series(sim, name):
    return [t.detach().cpu().clone() for t in sim.time_series.get(name, [])]


class TestTransitionDeclaration:
    """audit_route and ledger_change_route are declared by every transition; the
    ledger tolerance defaults to computation.scheduling.topology_transition_tolerance."""

    def test_routes_are_required_keywords_and_the_tolerance_reads_the_database(
        self, restore_global_parameter_manager,
    ):
        import dataclasses

        fields = {field.name: field for field in dataclasses.fields(TopologyTransition)}
        for name in ("audit_route", "ledger_change_route"):
            assert fields[name].default is dataclasses.MISSING, name
            assert fields[name].kw_only, name
        assert fields["tolerance"].default is None
        store = SurfacePondingFunctor()
        declared = dict(
            name="t", event_name="t", connect=((store, "fast"),), ledger_variables=("x",),
            audit_route="a", ledger_change_route="b",
        )
        restore_global_parameter_manager.set_batch(
            {"computation.scheduling.topology_transition_tolerance": 2.0e-7}, validate=True, notify=False,
        )
        assert TopologyTransition(**declared).tolerance == pytest.approx(2.0e-7)
        assert TopologyTransition(**declared, tolerance=1.0e-3).tolerance == pytest.approx(1.0e-3)
        with pytest.raises(TypeError, match="audit_route"):
            TopologyTransition(name="t", event_name="t", connect=((store, "fast"),), ledger_change_route="b")


class TestManualCommit:
    def test_onset_commits_one_epoch_with_an_exact_ledger(self):
        sim, store = _simulator(excess=3.0)
        epoch = sim.provider_epoch
        receipts_before = len(sim.get_lifecycle_receipts())
        receipt = sim.apply_topology_transition(TRANSITION_ONSET)
        assert receipt.operation == "topology_transition" and receipt.status == "committed"
        assert receipt.epoch_before == epoch and receipt.epoch_after == epoch + 1 == sim.provider_epoch
        assert "surface_ponding" in sim.functors and sim.functors["surface_ponding"] is store
        view = sim.get_committed_provider_view()
        assert set(view.providers) == {"stochastic_rainfall", "ponding_event", "surface_ponding"}
        assert view.order_fast.index("ponding_event") < view.order_fast.index("surface_ponding")
        state = sim.get_state()
        assert state[PONDED_DEPTH_VARIABLE].tolist() == [3.0, 3.0]
        assert state[EXCESS_VARIABLE].tolist() == [0.0, 0.0]
        assert state[ACTIVE_VARIABLE].tolist() == [1.0, 1.0]
        ledger = receipt.managed_delta["ledger"]
        assert ledger["residual_max"] == 0.0 and ledger["before_max"] == 3.0 and ledger["after_max"] == 3.0
        assert receipt.completed_actions[-1] == f"provider_view:publish:{epoch + 1}"
        assert "connect:surface_ponding:fast" in receipt.completed_actions
        # The registration inside the transition did not publish on its own.
        inner = [r for r in sim.get_lifecycle_receipts()[receipts_before:] if r.operation == "register"]
        assert len(inner) == 1 and inner[0].view_published is False
        assert sim.get_topology_history()[-1]["transition"] == TRANSITION_ONSET
        diagnostics = sim.state_manager.get_diagnostics()["current"]
        assert diagnostics["topology_transition.ledger_residual"] == 0.0
        assert receipt.managed_delta["state_remap"] == {
            PONDED_DEPTH_VARIABLE: {"sources": {PONDED_DEPTH_VARIABLE: 1.0, EXCESS_VARIABLE: 1.0}, "offset": 0.0},
            EXCESS_VARIABLE: {"sources": {}, "offset": 0.0},
            ACTIVE_VARIABLE: {"sources": {}, "offset": 1.0},
        }

    def test_cessation_retires_the_store_and_declares_the_infiltrated_residual(self):
        sim, _store = _simulator(excess=2.0)
        sim.apply_topology_transition(TRANSITION_ONSET)
        epoch = sim.provider_epoch
        receipt = sim.apply_topology_transition(TRANSITION_CESSATION)
        assert receipt.status == "committed" and receipt.epoch_after == epoch + 1
        assert "surface_ponding" not in sim.functors
        state = sim.get_state()
        assert state[PONDED_DEPTH_VARIABLE].tolist() == [0.0, 0.0]
        assert state[ACTIVE_VARIABLE].tolist() == [0.0, 0.0]
        ledger = receipt.managed_delta["ledger"]
        assert ledger["declared_change_sum"] == pytest.approx(-4.0) and ledger["residual_max"] == 0.0
        assert sim.state_manager.get_diagnostics()["current"]["topology_transition.ledger_change"] == pytest.approx(-4.0)
        retired = [r for r in sim.get_lifecycle_receipts() if r.operation == "unregister"][-1]
        assert retired.provider_name == "surface_ponding" and retired.view_published is False

    def test_repeated_reset_and_undeclared_transitions_are_refused(self):
        sim, _store = _simulator()
        sim.apply_topology_transition(TRANSITION_ONSET)
        epoch = sim.provider_epoch
        with pytest.raises(RuntimeError, match="guard failed"):
            sim.apply_topology_transition(TRANSITION_ONSET)
        assert sim.provider_epoch == epoch and "surface_ponding" in sim.functors
        with pytest.raises(ValueError, match="not declared"):
            sim.apply_topology_transition("flood")
        with pytest.raises(ValueError, match="already declared with another definition"):
            sim.declare_topology_transition(ponding_onset_transition(SurfacePondingFunctor(), execution_group="fast"))

    def test_ledger_leak_is_rejected_before_any_mutation(self):
        sim, store = _simulator(declare=False, excess=2.0)
        leaking = TopologyTransition(
            name="leak", event_name="leak", connect=((store, "fast"),),
            state_remap=(RemapRow(target=PONDED_DEPTH_VARIABLE, sources={EXCESS_VARIABLE: 0.5}),),
            ledger_variables=(PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE),
            audit_route="diagnostics.topology_transition.ledger_residual",
            ledger_change_route="diagnostics.topology_transition.ledger_change",
        )
        sim.declare_topology_transition(leaking)
        epoch = sim.provider_epoch
        with pytest.raises(ValueError, match="leaks the declared ledger"):
            sim.apply_topology_transition("leak")
        assert sim.provider_epoch == epoch and "surface_ponding" not in sim.functors
        assert sim.get_state()[PONDED_DEPTH_VARIABLE].tolist() == [0.0, 0.0]
        assert not any(r.operation == "topology_transition" for r in sim.get_lifecycle_receipts())

    def test_publication_failure_compensates_registry_and_state(self, monkeypatch):
        sim, _store = _simulator(excess=1.5)
        epoch = sim.provider_epoch
        original = registration._publish_provider_view

        def failing(simulator, **kwargs):
            if kwargs.get("operation") == "topology_transition":
                raise RuntimeError("publication interrupted")
            return original(simulator, **kwargs)

        monkeypatch.setattr(registration, "_publish_provider_view", failing)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            sim.apply_topology_transition(TRANSITION_ONSET)
        monkeypatch.setattr(registration, "_publish_provider_view", original)
        assert sim.provider_epoch == epoch
        assert "surface_ponding" not in sim.functors
        state = sim.get_state()
        assert state[PONDED_DEPTH_VARIABLE].tolist() == [0.0, 0.0]
        assert state[EXCESS_VARIABLE].tolist() == [1.5, 1.5]
        assert state[ACTIVE_VARIABLE].tolist() == [0.0, 0.0]
        receipt = sim.get_lifecycle_receipts()[-1]
        assert receipt.operation == "topology_transition" and receipt.status == "rolled_back"
        assert receipt.compensated_actions[0] == "state_remap"
        # The committed view still matches the live registry: the next step runs.
        sim.step(dt=DT)
        assert sim.provider_epoch == epoch


class TestEventDrivenCommit:
    def test_request_is_committed_after_the_accepting_step(self):
        sim, _store = _simulator()
        epoch_before = sim.provider_epoch
        onset_step = _run_until(sim, lambda s: "surface_ponding" in s.functors)
        history = sim.get_topology_history()
        assert history[0]["transition"] == TRANSITION_ONSET and history[0]["accepted_step"] == onset_step
        assert history[0]["epoch_before"] == epoch_before and history[0]["epoch_after"] == epoch_before + 1
        # The accepting step ran on the old view: no ponded update was written
        # in that step, and the connected store first writes in the next one.
        provenance_after_onset = sim.get_variable_provenance(PONDED_DEPTH_VARIABLE)
        assert "surface_ponding" not in {r["functor"] for r in provenance_after_onset}
        assert sim.get_state()[ACTIVE_VARIABLE].tolist() == [1.0, 1.0]
        delivery = sim._last_external_delivery_receipts[-1]
        assert delivery["operation"] == "topology" and delivery["receipt"]["transition"] == TRANSITION_ONSET
        assert delivery["receipt"]["epoch_after"] == epoch_before + 1
        sim.step(dt=DT)
        writers = {r["functor"] for r in sim.get_variable_provenance(PONDED_DEPTH_VARIABLE)}
        assert "surface_ponding" in writers
        cessation_step = _run_until(sim, lambda s: "surface_ponding" not in s.functors)
        assert cessation_step > onset_step + 1
        assert sim.get_state()[ACTIVE_VARIABLE].tolist() == [0.0, 0.0]
        assert sim.get_state()[PONDED_DEPTH_VARIABLE].tolist() == [0.0, 0.0]
        assert [item["transition"] for item in sim.get_topology_history()] == [TRANSITION_ONSET, TRANSITION_CESSATION]
        assert all(item["ledger_residual_max"] == 0.0 for item in sim.get_topology_history())

    def test_undeclared_request_rolls_the_step_back(self):
        sim, _store = _simulator(declare=False)
        epoch = sim.provider_epoch
        _run_until(sim, lambda s: float(s.get_state()[RAINFALL_DEPTH_VARIABLE].max()) > 0.0, max_steps=20)
        before = {name: value.clone() for name, value in sim.get_state().items()}
        step_count = sim.step_count
        try:
            _run_until(sim, lambda s: False, max_steps=40)
        except ValueError as error:
            assert "Undeclared topology transition" in str(error)
        else:  # pragma: no cover - the rainfall must eventually exceed the threshold
            raise AssertionError("expected the undeclared request to be refused")
        assert sim.provider_epoch == epoch and "surface_ponding" not in sim.functors
        state = sim.get_state()
        assert state[ACTIVE_VARIABLE].tolist() == [0.0, 0.0]
        # The refused step left the committed state of the last accepted step.
        assert sim.step_count >= step_count
        assert torch.equal(state[ACTIVE_VARIABLE], before[ACTIVE_VARIABLE])

    def test_transition_inside_a_step_is_refused(self):
        sim, _store = _simulator()

        class _Intruder(HortonianPondingEventFunctor):
            def forward(self, state, time, dt, **kwargs):
                sim.apply_topology_transition(TRANSITION_ONSET)
                return super().forward(state, time, dt, **kwargs)

        sim.unregister_functor("ponding_event")
        sim.register_functor(_Intruder(name="ponding_event"), execution_group="fast")
        epoch = sim.provider_epoch
        with pytest.raises(Exception, match="inside a step transaction"):
            sim.step(dt=DT)
        assert sim.provider_epoch == epoch and "surface_ponding" not in sim.functors

    def test_intent_payload_must_name_its_resource(self):
        from relic.core.simulator.execution_loop import _prepare_external_deliveries

        sim, _store = _simulator()
        mismatched = ExternalIntent("topology", TRANSITION_ONSET, "r", {"transition": TRANSITION_CESSATION})
        record = {"functor": "x", "substep": 0, "time": 0.0, "output": "o", "intent": mismatched}
        with pytest.raises(ValueError, match="must match its resource"):
            _prepare_external_deliveries(sim, [record], accepted_auto_report_enabled=False)
        unknown = ExternalIntent("topology", "flood", "r", {"transition": "flood"})
        with pytest.raises(ValueError, match="Undeclared topology transition"):
            _prepare_external_deliveries(sim, [dict(record, intent=unknown)], accepted_auto_report_enabled=False)
        extra = ExternalIntent("topology", TRANSITION_ONSET, "r", {"transition": TRANSITION_ONSET, "force": True})
        with pytest.raises(ValueError, match="Unsupported topology delivery options"):
            _prepare_external_deliveries(sim, [dict(record, intent=extra)], accepted_auto_report_enabled=False)
        jobs = _prepare_external_deliveries(
            sim,
            [dict(record, intent=ExternalIntent("topology", TRANSITION_ONSET, "r", {"transition": TRANSITION_ONSET}))],
            accepted_auto_report_enabled=False,
        )
        assert jobs[0]["operation"] == "topology" and jobs[0]["resource"] == f"topology:{TRANSITION_ONSET}"
        assert jobs[0]["snapshot"]["transition"] == TRANSITION_ONSET


class TestRestart:
    def test_post_onset_checkpoint_restores_exactly_and_continues_bitwise(self, tmp_path):
        reference, _ = _simulator()
        onset_step = _run_until(reference, lambda s: "surface_ponding" in s.functors)
        for _ in range(2):
            reference.step(dt=DT)
        path = tmp_path / "post_onset.pt"
        reference.create_checkpoint(path)
        for _ in range(6):
            reference.step(dt=DT)

        restored, _ = _simulator()
        restored.apply_topology_transition(TRANSITION_ONSET)
        receipt = restored.restore_checkpoint(path)
        assert receipt["restart_mode"] == "exact"
        assert restored.step_count == onset_step + 2
        assert set(restored.functors) == {"stochastic_rainfall", "ponding_event", "surface_ponding"}
        for _ in range(6):
            restored.step(dt=DT)
        for name in (RAINFALL_DEPTH_VARIABLE, PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE, ACTIVE_VARIABLE):
            assert torch.equal(restored.get_state()[name], reference.get_state()[name]), name
            assert all(torch.equal(a, b) for a, b in zip(_series(restored, name), _series(reference, name))), name
        assert [i["transition"] for i in restored.get_topology_history()[1:]] == [
            i["transition"] for i in reference.get_topology_history()[1:]
        ]

    def test_pre_onset_checkpoint_replays_the_same_onset(self, tmp_path):
        reference, _ = _simulator()
        reference.step(dt=DT)
        path = tmp_path / "pre_onset.pt"
        reference.create_checkpoint(path)
        onset_step = _run_until(reference, lambda s: "surface_ponding" in s.functors)
        restored, _ = _simulator()
        restored.restore_checkpoint(path)
        assert restored.step_count == 1 and "surface_ponding" not in restored.functors
        assert _run_until(restored, lambda s: "surface_ponding" in s.functors) == onset_step
        for name in (RAINFALL_DEPTH_VARIABLE, PONDED_DEPTH_VARIABLE, ACTIVE_VARIABLE):
            assert torch.equal(restored.get_state()[name], reference.get_state()[name]), name

    def test_post_onset_checkpoint_restores_under_a_compatibility_certificate(self, tmp_path):
        from relic.core.simulator.checkpoint import RestartCompatibilityCertificate

        reference, _ = _simulator()
        _run_until(reference, lambda s: "surface_ponding" in s.functors)
        reference.step(dt=DT)
        path = tmp_path / "post_onset.pt"
        reference.create_checkpoint(path)

        # The target store drains faster: a certified parameter difference on
        # the connected provider, everything else exact.
        target, _ = _simulator()
        target.unregister_functor("ponding_event")
        target.register_functor(HortonianPondingEventFunctor(), execution_group="fast")
        faster = SurfacePondingFunctor(config={"parameters": {"ponding_drainage_time": 900.0}})
        target._provider_lifecycle.topology_transitions.clear()
        target.declare_topology_transition(ponding_onset_transition(faster, execution_group="fast"))
        target.declare_topology_transition(ponding_cessation_transition())
        target.apply_topology_transition(TRANSITION_ONSET)
        with pytest.raises(ValueError, match="does not match the active simulator"):
            target.restore_checkpoint(path)
        digests = compute_restart_topology_digests(target, path)
        certificate = RestartCompatibilityCertificate(
            source_topology_sha256=digests["source_topology_sha256"],
            target_topology_sha256=digests["target_topology_sha256"],
            allowed_contract_paths=(
                "scientific_contract.functors.surface_ponding.config.parameters",
                "scientific_contract.functors.surface_ponding.parameter_values.ponding_drainage_time",
            ),
            parameter_relative_tolerance=1.0, forcing_cursor_policy="exact",
        )
        receipt = target.restore_checkpoint(path, compatibility_certificate=certificate)
        assert receipt["restart_mode"] == "compatible"
        assert set(target.functors) == {"stochastic_rainfall", "ponding_event", "surface_ponding"}
        for name in (RAINFALL_DEPTH_VARIABLE, PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE, ACTIVE_VARIABLE):
            assert torch.equal(target.get_state()[name], reference.get_state()[name]), name
        target.step(dt=DT)
        reference.step(dt=DT)
        # Same rainfall path, faster recession: the ponded depth differs only
        # where water was stored.
        assert torch.equal(target.get_state()[RAINFALL_DEPTH_VARIABLE], reference.get_state()[RAINFALL_DEPTH_VARIABLE])
        stored = reference.get_state()[PONDED_DEPTH_VARIABLE] > 0.0
        assert bool((target.get_state()[PONDED_DEPTH_VARIABLE][stored] < reference.get_state()[PONDED_DEPTH_VARIABLE][stored]).all())
