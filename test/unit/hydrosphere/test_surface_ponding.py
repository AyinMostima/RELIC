"""L1 tests for the Hortonian ponding event detector and the ponding store.

References:
    - Horton (1933), Trans. Am. Geophys. Union 14, 446-460.
    - Beven (2012), Rainfall-Runoff Modelling: The Primer, ch. 6.
    - Dooge (1973), Linear Theory of Hydrologic Systems.
    - Goebel, Sanfelice and Teel (2012), Hybrid Dynamical Systems.
"""

from __future__ import annotations

import math

import pytest
import torch

from relic.core.simulator.registration import RemapRow, TopologyTransition
from relic.core.state import ExternalIntent, UpdateItem, UpdateMode
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
from relic.utils.time_constants import SECONDS_PER_HOUR

DT = 3600.0


def _event(**overrides):
    return HortonianPondingEventFunctor(config={"parameters": overrides} if overrides else None)


def _state(depth, active=0.0, ponded=0.0, excess=0.0):
    depth = torch.as_tensor(depth, dtype=torch.float64)
    return {
        RAINFALL_DEPTH_VARIABLE: depth,
        ACTIVE_VARIABLE: torch.full_like(depth, float(active)),
        PONDED_DEPTH_VARIABLE: torch.full_like(depth, float(ponded)),
        EXCESS_VARIABLE: torch.full_like(depth, float(excess)),
    }


def _params(functor):
    return {
        name: float(functor.get_parameter_value(name, required=True).detach())
        for name in functor.PARAMETERS
    }


class TestEventContract:
    def test_metadata_dependencies_and_topology_category(self):
        functor = _event()
        assert functor.topology_category == "topology_change"
        assert functor.metadata.input_variables == [RAINFALL_DEPTH_VARIABLE, ACTIVE_VARIABLE, PONDED_DEPTH_VARIABLE]
        assert functor.metadata.output_variables[0] == EXCESS_VARIABLE
        assert all(name.startswith("diagnostics.ponding_event.") for name in functor.metadata.output_variables[1:])
        assert functor.get_dependencies() == ["stochastic_rainfall"]
        assert _event().get_extra_state() == {
            "format": HortonianPondingEventFunctor.EXTRA_STATE_FORMAT,
            "version": 1,
            "last_rainfall_depth": None,
        }
        custom = HortonianPondingEventFunctor(config={"rainfall_provider": "rain_a"})
        assert custom.get_dependencies() == ["rain_a"]
        with pytest.raises(ValueError, match="Unknown ponding event parameter"):
            _event(bogus=1.0)
        # config["dependencies"] seeds instance names; the provider stays the
        # static declaration and the instance name is merged into required.
        forwarded = HortonianPondingEventFunctor(config={"dependencies": ["x"]})
        assert forwarded.get_dependencies() == ["stochastic_rainfall", "x"]

    def test_parameters_come_from_the_database(self):
        params = _params(_event())
        assert params["ponding_infiltration_capacity"] == pytest.approx(10.0)
        assert params["ponding_onset_depth"] == pytest.approx(2.0)
        assert params["ponding_cessation_depth"] == pytest.approx(0.1)


class TestEventKernel:
    def test_first_call_has_no_increment_and_records_the_depth(self):
        functor = _event()
        outputs = functor.forward(_state([12.0, 4.0]), 0.0, DT)
        assert outputs[EXCESS_VARIABLE].mode is UpdateMode.REPLACE
        assert outputs[EXCESS_VARIABLE].value.tolist() == [0.0, 0.0]
        assert outputs["diagnostics.ponding_event.rainfall_increment"].tolist() == [0.0, 0.0]
        assert functor.last_rainfall_depth.tolist() == [12.0, 4.0]
        assert not any(isinstance(value, ExternalIntent) for value in outputs.values())

    def test_excess_is_the_increment_above_the_capacity_depth(self):
        functor = _event()
        capacity = _params(functor)["ponding_infiltration_capacity"] * DT / float(SECONDS_PER_HOUR)
        functor.forward(_state([0.0, 0.0, 0.0]), 0.0, DT)
        outputs = functor.forward(_state([capacity + 3.5, capacity, 1.0]), DT, DT)
        # References: Horton (1933). Expression: E = max(0, dH - I dt).
        assert outputs[EXCESS_VARIABLE].value.tolist() == pytest.approx([3.5, 0.0, 0.0])
        assert outputs["diagnostics.ponding_event.infiltrated_excess"].tolist() == pytest.approx([3.5, 0.0, 0.0])
        assert functor.check_conservation_laws({}, outputs) == {"mass": 0.0}

    def test_decreasing_cumulative_depth_and_bad_flags_are_rejected(self):
        functor = _event()
        functor.forward(_state([5.0]), 0.0, DT)
        with pytest.raises(ValueError, match="decreased"):
            functor.forward(_state([4.0]), DT, DT)
        with pytest.raises(ValueError, match="uniform over cells"):
            state = _state([5.0, 5.0])
            state[ACTIVE_VARIABLE] = torch.tensor([0.0, 1.0], dtype=torch.float64)
            functor.forward(state, DT, DT)
        with pytest.raises(ValueError, match="0 or 1"):
            state = _state([5.0])
            state[ACTIVE_VARIABLE] = torch.tensor([0.5], dtype=torch.float64)
            functor.forward(state, DT, DT)
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            functor.forward(_state([5.0]), DT, 0.0)

    def test_onset_request_only_when_disconnected_and_above_threshold(self):
        functor = _event()
        params = _params(functor)
        capacity = params["ponding_infiltration_capacity"] * DT / float(SECONDS_PER_HOUR)
        functor.forward(_state([0.0, 0.0]), 0.0, DT)
        below = functor.forward(_state([capacity + 0.5 * params["ponding_onset_depth"], 0.0]), DT, DT)
        assert not any(isinstance(value, ExternalIntent) for value in below.values())
        above = functor.forward(_state([capacity + 0.5 * params["ponding_onset_depth"] + capacity + params["ponding_onset_depth"], 0.0]), 2 * DT, DT)
        intent = above[f"topology_intent_{TRANSITION_ONSET}"]
        assert isinstance(intent, ExternalIntent)
        assert intent.operation == "topology" and intent.resource == TRANSITION_ONSET
        assert dict(intent.payload) == {"transition": TRANSITION_ONSET}
        # Already connected: the same excess emits no onset request.
        connected = _event()
        connected.forward(_state([0.0, 0.0], active=1.0), 0.0, DT)
        again = connected.forward(_state([capacity + 2.0 * params["ponding_onset_depth"], 0.0], active=1.0, ponded=5.0), DT, DT)
        assert not any(isinstance(value, ExternalIntent) for value in again.values())
        assert again["diagnostics.ponding_event.infiltrated_excess"].tolist() == [0.0, 0.0]

    def test_cessation_request_requires_zero_excess_and_a_drained_store(self):
        functor = _event()
        params = _params(functor)
        functor.forward(_state([0.0, 0.0], active=1.0, ponded=0.05), 0.0, DT)
        drained = functor.forward(_state([0.0, 0.0], active=1.0, ponded=0.05), DT, DT)
        intent = drained[f"topology_intent_{TRANSITION_CESSATION}"]
        assert intent.resource == TRANSITION_CESSATION and dict(intent.payload) == {"transition": TRANSITION_CESSATION}
        still_wet = functor.forward(_state([0.0, 0.0], active=1.0, ponded=params["ponding_cessation_depth"] * 4.0), 2 * DT, DT)
        assert not any(isinstance(value, ExternalIntent) for value in still_wet.values())
        capacity = params["ponding_infiltration_capacity"] * DT / float(SECONDS_PER_HOUR)
        raining = functor.forward(_state([capacity + 1.0, 0.0], active=1.0, ponded=0.0), 3 * DT, DT)
        assert not any(isinstance(value, ExternalIntent) for value in raining.values())

    def test_extra_state_round_trip_and_rejections(self):
        functor = _event()
        functor.forward(_state([7.0, 2.0]), 0.0, DT)
        state = functor.state_dict()
        clone = _event()
        clone.load_state_dict(state, strict=True)
        assert clone.last_rainfall_depth.tolist() == [7.0, 2.0]
        outputs = clone.forward(_state([7.0 + 30.0, 2.0]), DT, DT)
        assert outputs["diagnostics.ponding_event.rainfall_increment"].tolist() == [30.0, 0.0]
        for mutation in (
            lambda s: s.__setitem__("format", "other"),
            lambda s: s.__setitem__("version", 2),
            lambda s: s.__setitem__("last_rainfall_depth", torch.tensor([1, 2])),
            lambda s: s.pop("last_rainfall_depth"),
        ):
            broken = functor.state_dict()
            mutation(broken["_extra_state"])
            with pytest.raises((ValueError, TypeError)):
                _event().load_state_dict(broken, strict=True)


class TestPondingStore:
    def test_metadata_and_parameters(self):
        functor = SurfacePondingFunctor()
        assert functor.metadata.input_variables == [PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE]
        assert functor.metadata.output_variables[0] == PONDED_DEPTH_VARIABLE
        assert functor.get_dependencies() == ["ponding_event"]
        assert functor.transport_type == "advection"
        assert float(functor.get_parameter_value("ponding_drainage_time", required=True)) == pytest.approx(1800.0)

    def test_exact_recession_and_closed_budget(self):
        functor = SurfacePondingFunctor()
        tau = float(functor.get_parameter_value("ponding_drainage_time", required=True))
        ponded = torch.tensor([4.0, 0.0, 1.0], dtype=torch.float64)
        excess = torch.tensor([2.0, 3.0, 0.0], dtype=torch.float64)
        outputs = functor.forward({PONDED_DEPTH_VARIABLE: ponded, EXCESS_VARIABLE: excess}, 0.0, DT)
        update = outputs[PONDED_DEPTH_VARIABLE]
        assert isinstance(update, UpdateItem) and update.mode is UpdateMode.ACCUMULATE and update.per_timestep
        # References: Dooge (1973). Expression: h' = (h + E) exp(-dt/tau).
        expected_end = (ponded + excess) * math.exp(-DT / tau)
        torch.testing.assert_close(ponded + update.value, expected_end, rtol=0.0, atol=1e-13)
        runoff = outputs["diagnostics.surface_ponding.runoff_depth"]
        torch.testing.assert_close(runoff, ponded + excess - expected_end, rtol=0.0, atol=1e-13)
        assert functor.check_conservation_laws({}, outputs)["mass"] <= 1e-15
        assert outputs["diagnostics.surface_ponding.water_balance_residual"].abs().max() <= 1e-15

    def test_invalid_inputs_are_rejected(self):
        functor = SurfacePondingFunctor()
        with pytest.raises(ValueError, match="nonnegative"):
            functor.forward({PONDED_DEPTH_VARIABLE: torch.tensor([-1.0]), EXCESS_VARIABLE: torch.tensor([0.0])}, 0.0, DT)
        with pytest.raises(ValueError, match="share one shape"):
            functor.forward({PONDED_DEPTH_VARIABLE: torch.tensor([1.0, 2.0]), EXCESS_VARIABLE: torch.tensor([0.0])}, 0.0, DT)
        with pytest.raises(ValueError, match="required"):
            functor.forward({PONDED_DEPTH_VARIABLE: torch.tensor([1.0])}, 0.0, DT)
        with pytest.raises(ValueError, match="Unknown surface ponding parameter"):
            SurfacePondingFunctor(config={"parameters": {"tau": 1.0}})
        forwarded = SurfacePondingFunctor(config={"dependencies": ["x"]})
        assert forwarded.get_dependencies() == ["ponding_event", "x"]


class TestDeclaredTransitions:
    ORDER = (PONDED_DEPTH_VARIABLE, EXCESS_VARIABLE, ACTIVE_VARIABLE)

    def test_onset_remap_is_ledger_preserving(self):
        transition = ponding_onset_transition(SurfacePondingFunctor(), execution_group="fast")
        assert isinstance(transition, TopologyTransition)
        assert transition.name == TRANSITION_ONSET and transition.guard == (ACTIVE_VARIABLE, 0.0)
        assert transition.remap_matrix(self.ORDER) == [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        assert transition.reset_vector(self.ORDER) == [0.0, 0.0, 1.0]
        # C = [1, 1, 0]: C P = C on every state with a^- = 0.
        ledger = [1.0, 1.0, 0.0]
        projected = [sum(ledger[i] * transition.remap_matrix(self.ORDER)[i][j] for i in range(3)) for j in range(3)]
        assert projected == ledger
        assert transition.ledger_change_source is None
        assert [functor.name for functor, _ in transition.connect] == ["surface_ponding"]

    def test_cessation_declares_the_infiltrated_residual(self):
        transition = ponding_cessation_transition()
        assert transition.retire == ("surface_ponding",) and transition.guard == (ACTIVE_VARIABLE, 1.0)
        assert transition.remap_matrix(self.ORDER) == [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        assert transition.reset_vector(self.ORDER) == [0.0, 0.0, 0.0]
        assert transition.ledger_change_source == PONDED_DEPTH_VARIABLE
        assert transition.parameter_keys == ("hydrosphere.runoff.ponding_cessation_depth",)

    def test_invalid_transition_declarations_are_rejected(self):
        store = SurfacePondingFunctor()
        with pytest.raises(ValueError, match="retire or connect"):
            TopologyTransition(name="t", event_name="t", ledger_variables=("x",), audit_route="a", ledger_change_route="b")
        with pytest.raises(TypeError, match="FunctorBase"):
            TopologyTransition(name="t", event_name="t", connect=(("not a functor", "fast"),), ledger_variables=("x",), audit_route="a", ledger_change_route="b")
        with pytest.raises(ValueError, match="unique"):
            TopologyTransition(
                name="t", event_name="t", connect=((store, "fast"),), ledger_variables=("x",),
                state_remap=(RemapRow(target="x"), RemapRow(target="x", offset=1.0)),
                audit_route="a", ledger_change_route="b",
            )
        with pytest.raises(TypeError, match="numeric coefficients"):
            RemapRow(target="x", sources={"y": "1"})
        with pytest.raises(KeyError, match="outside the ordered state"):
            ponding_onset_transition(store, execution_group="fast").remap_matrix((PONDED_DEPTH_VARIABLE, ACTIVE_VARIABLE))


def test_config_first_source_order_is_the_mixin_class_declaration():
    """Round 47: explicit config overrides precede the ParameterManager through the
    NamespaceDefaultMixin class declaration, not through a get_parameter_value override."""
    from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin

    for functor_class in (HortonianPondingEventFunctor, SurfacePondingFunctor,):
        assert issubclass(functor_class, NamespaceDefaultMixin)
        assert functor_class.parameter_lookup_source_order == ("config", "manager")
        assert "get_parameter_value" not in functor_class.__dict__


def test_ponding_onset_transition_requires_its_execution_group():
    """Round 47: the rate group of the connected store is declared by the caller."""
    import inspect

    parameter = inspect.signature(ponding_onset_transition).parameters["execution_group"]
    assert parameter.default is inspect.Parameter.empty
