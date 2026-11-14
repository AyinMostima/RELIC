"""L1 tests for the marked-Poisson stochastic rainfall Functor.

References:
    - Rodriguez-Iturbe et al. (1999), Proc. R. Soc. A 455, 3789-3805.
    - Laio et al. (2001), Adv. Water Resour. 24, 707-723.
    - Gillespie (1977), J. Phys. Chem. 81, 2340-2361 (direct method).
    - Karlin and Taylor (1981), A Second Course in Stochastic Processes,
      compound Poisson processes.
"""

import math

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.atmosphere.stochastic_rainfall import MarkedPoissonRainfallFunctor
from relic.utils.time_constants import SECONDS_PER_DAY

COUNT = "stochastic_rainfall_event_count"
DEPTH = "stochastic_rainfall_depth"
DIAG = "diagnostics.stochastic_rainfall."


def _functor(seed=7, **overrides):
    config = {"seed": seed}
    config.update(overrides)
    return MarkedPoissonRainfallFunctor(name="stochastic_rainfall", config=config)


def _coefficients(functor):
    rate = float(functor.get_parameter_value("event_rate", required=True).detach())
    depth = float(functor.get_parameter_value("mean_depth", required=True).detach())
    return rate, depth


def _state(n, device="cpu"):
    return {
        COUNT: torch.zeros(n, dtype=torch.float64, device=device),
        DEPTH: torch.zeros(n, dtype=torch.float64, device=device),
    }


class TestContract:
    def test_metadata_and_owned_substream(self):
        functor = _functor()
        assert functor.metadata.input_variables == [COUNT, DEPTH]
        assert functor.metadata.output_variables[:2] == [COUNT, DEPTH]
        assert all(name.startswith(DIAG) for name in functor.metadata.output_variables[2:])
        assert functor.get_dependencies() == []
        assert functor.stochastic_category == "marked_jump"
        assert functor.noise_classification == "external_forcing"
        assert functor.solver_name == "direct_ssa"
        assert functor.mark_law == "exponential"
        assert functor.STATE_VARIABLES == (COUNT, DEPTH)
        extra = functor.get_extra_state()
        assert extra["format"] == functor.EXTRA_STATE_FORMAT
        assert extra["draw_count"] == 0
        assert isinstance(extra["generator_state"], torch.Tensor)

    @pytest.mark.parametrize("config", [{}, {"seed": -1}, {"seed": 1.5}, {"seed": True}])
    def test_seed_is_required_and_validated(self, config):
        with pytest.raises(ValueError, match="seed"):
            MarkedPoissonRainfallFunctor(name="stochastic_rainfall", config=config)


class TestMarkedPoissonStep:
    def test_compensators_and_update_contract(self):
        # References: Rodriguez-Iturbe et al. (1999); Karlin and Taylor (1981).
        # Mathematical expression: dN ~ Poisson(lambda dt); dH = sum_{i<=dN} z_i,
        #   z_i ~ Exp(mean alpha); compensators E[dN] = lambda dt, E[dH] = lambda alpha dt.
        # Parameter meanings: lambda event rate [day-1], alpha mean depth [mm].
        # Units: dt s -> day; N count [1]; H depth [mm].
        # Domain/constraints: finite positive dt; integer-valued nonnegative count.
        # Sign convention: every event adds a nonnegative depth; no negative marks.
        functor = _functor(seed=3)
        rate, depth = _coefficients(functor)
        dt_seconds = 6.0 * 3600.0
        dt_days = dt_seconds / float(SECONDS_PER_DAY)
        result = functor.forward(_state(8), 0.0, dt_seconds)
        for name, unit in ((COUNT, "1"), (DEPTH, "mm")):
            item = result[name]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.unit == unit
            assert item.value.shape == (8,)
        count_increment = result[COUNT].value
        depth_increment = result[DEPTH].value
        assert torch.equal(count_increment, torch.round(count_increment))
        assert bool((count_increment >= 0).all()) and bool((depth_increment >= 0).all())
        assert bool(((count_increment == 0) == (depth_increment == 0)).all())
        torch.testing.assert_close(result[DIAG + "event_count_increment"].value, count_increment)
        torch.testing.assert_close(result[DIAG + "depth_increment"].value, depth_increment)
        torch.testing.assert_close(
            result[DIAG + "compensator_count"].value,
            torch.full((8,), rate * dt_days, dtype=torch.float64),
        )
        torch.testing.assert_close(
            result[DIAG + "compensator_depth"].value,
            torch.full((8,), rate * depth * dt_days, dtype=torch.float64),
        )
        assert result[DIAG + "compensator_depth"].unit == "mm"
        assert result[DIAG + "compensator_count"].unit == "1"
        assert functor.get_extra_state()["draw_count"] == 1

    def test_poisson_count_and_exponential_mark_statistics(self):
        # Poisson counts: mean = variance = lambda dt; conditional on one
        # event the depth is exponential with mean alpha.
        functor = _functor(seed=11)
        rate, depth = _coefficients(functor)
        n = 40000
        dt_seconds = float(SECONDS_PER_DAY)
        expected = rate * dt_seconds / float(SECONDS_PER_DAY)
        result = functor.forward(_state(n), 0.0, dt_seconds)
        counts = result[COUNT].value
        depths = result[DEPTH].value
        observed_mean = float(counts.mean())
        assert abs(observed_mean - expected) < 4.0 * math.sqrt(expected / n)
        observed_var = float(counts.var(unbiased=True))
        assert abs(observed_var - expected) < 4.0 * math.sqrt(2.0 * expected * expected / n + expected / n)
        single = depths[counts == 1]
        assert single.numel() > 100
        assert abs(float(single.mean()) - depth) < 4.0 * depth / math.sqrt(single.numel())
        assert abs(float(depths.mean()) - rate * depth) < 4.0 * depth * math.sqrt(2.0 * expected / n)

    def test_same_seed_reproduces_and_other_seed_differs(self):
        first = _functor(seed=5).forward(_state(64), 0.0, 3.0 * 3600.0)
        again = _functor(seed=5).forward(_state(64), 0.0, 3.0 * 3600.0)
        other = _functor(seed=6).forward(_state(64), 0.0, 3.0 * 3600.0)
        assert torch.equal(first[COUNT].value, again[COUNT].value)
        assert torch.equal(first[DEPTH].value, again[DEPTH].value)
        assert not torch.equal(first[DEPTH].value, other[DEPTH].value)

    def test_depth_gradient_with_respect_to_mean_depth(self):
        mean_depth = torch.tensor(12.0, dtype=torch.float64, requires_grad=True)
        functor = _functor(seed=9, parameters={"mean_depth": mean_depth})
        result = functor.forward(_state(512), 0.0, float(SECONDS_PER_DAY))
        depth = result[DEPTH].value
        assert depth.requires_grad
        (grad,) = torch.autograd.grad(depth.sum(), (mean_depth,))
        # dH/dalpha = sum of unit exponential marks = H / alpha.
        torch.testing.assert_close(grad, depth.sum().detach() / 12.0)

    @pytest.mark.parametrize(
        "case",
        ["missing_count", "missing_depth", "fractional_count", "negative_count",
         "nonfinite", "bad_dt", "non_tensor", "shape_mismatch"],
    )
    def test_domain_rejections(self, case):
        functor = _functor()
        state = _state(4)
        dt = 3600.0
        if case == "missing_count":
            state.pop(COUNT)
        elif case == "missing_depth":
            state.pop(DEPTH)
        elif case == "fractional_count":
            state[COUNT] = torch.full((4,), 0.5, dtype=torch.float64)
        elif case == "negative_count":
            state[COUNT] = torch.full((4,), -1.0, dtype=torch.float64)
        elif case == "nonfinite":
            state[DEPTH] = torch.full((4,), float("nan"), dtype=torch.float64)
        elif case == "bad_dt":
            dt = 0.0
        elif case == "non_tensor":
            state[DEPTH] = 0.0
        elif case == "shape_mismatch":
            state[DEPTH] = torch.zeros(3, dtype=torch.float64)
        with pytest.raises((ValueError, TypeError)):
            functor.forward(state, 0.0, dt)
        assert functor.get_extra_state()["draw_count"] == 0


class TestGovernedRuntime:
    @staticmethod
    def _simulator(seed):
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False, "enable_auto_scale": False})
        simulator.initialize_state(_state(16))
        functor = _functor(seed=seed)
        simulator.register_functor(functor, execution_group="fast")
        return simulator, functor

    def test_committed_states_accumulate_the_emitted_increments(self):
        simulator, functor = self._simulator(seed=21)
        dt = 6.0 * 3600.0
        total_count = torch.zeros(16, dtype=torch.float64)
        total_depth = torch.zeros(16, dtype=torch.float64)
        for _ in range(8):
            outputs = simulator.step(dt=dt)
            total_count = total_count + outputs[DIAG + "event_count_increment"]
            total_depth = total_depth + outputs[DIAG + "depth_increment"]
        torch.testing.assert_close(simulator.state_manager.get_state(COUNT), total_count, rtol=0.0, atol=0.0)
        torch.testing.assert_close(simulator.state_manager.get_state(DEPTH), total_depth, rtol=0.0, atol=0.0)
        assert functor.get_extra_state()["draw_count"] == 8
        assert float(total_count.sum()) > 0.0

    def test_checkpoint_restart_continues_the_same_event_stream(self, tmp_path):
        dt = 6.0 * 3600.0
        uninterrupted, _ = self._simulator(seed=33)
        for _ in range(8):
            uninterrupted.step(dt=dt)

        source, source_functor = self._simulator(seed=33)
        for _ in range(4):
            source.step(dt=dt)
        checkpoint = tmp_path / "rain.pt"
        source.create_checkpoint(checkpoint)
        assert source_functor.get_extra_state()["draw_count"] == 4

        restored, restored_functor = self._simulator(seed=99)
        restored.restore_checkpoint(checkpoint)
        assert restored_functor.get_extra_state()["draw_count"] == 4
        for _ in range(4):
            restored.step(dt=dt)
        for name in (COUNT, DEPTH):
            torch.testing.assert_close(
                restored.state_manager.get_state(name),
                uninterrupted.state_manager.get_state(name),
                rtol=0.0,
                atol=0.0,
            )

    def test_rejected_adaptive_step_rewinds_the_event_stream(self):
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({
            "device": "cpu", "conservation_checks": False, "enable_auto_scale": False,
            "adaptive_dt_enabled": True, "adaptive_atol": 1e-30, "adaptive_rtol": 1e-30,
            "adaptive_dt_min": 1.0, "adaptive_dt_max": 7200.0,
        })
        simulator.initialize_state(_state(16))
        functor = _functor(seed=31)
        simulator.register_functor(functor, execution_group="fast")
        before = functor.get_extra_state()
        simulator.step(dt=3600.0)
        after = functor.get_extra_state()
        assert after["draw_count"] == before["draw_count"] + 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA substream test requires a CUDA device")
def test_cuda_functor_owns_a_cuda_event_stream():
    functor = _functor(seed=13)
    functor.to_device("cuda")
    assert torch.device(functor.generator.device).type == "cuda"
    result = functor.forward(_state(256, device="cuda"), 0.0, float(SECONDS_PER_DAY))
    assert result[COUNT].value.device.type == "cuda"
    assert result[DEPTH].value.device.type == "cuda"
    assert functor.get_extra_state()["draw_count"] == 1
    with pytest.raises(ValueError, match="migration across device types after draws"):
        functor.to_device("cpu")



class TestAdaptiveEventPath:
    """Variable steps stay on one marked-jump path (F05).

    References: Gillespie (1977); Karlin and Taylor (1981) compound Poisson:
    the restriction of a Poisson path on [t, t+H) to [t, t+h) keeps exactly
    the events with times below h together with their marks.
    """

    def test_rejected_trial_is_pinned_and_split_at_the_kernel(self):
        functor = _functor(seed=41, parameters={"event_rate": {"value": 5.0}})  # 5 per day (upper bound)
        state = _state(64)
        snapshot = functor.get_extra_state()
        trial = functor.forward(state, 0.0, 6 * 3600.0)
        trial_counts = trial[COUNT].value.clone()
        trial_depth = trial[DEPTH].value.clone()
        assert float(trial_counts.sum()) > 0

        functor.set_extra_state(snapshot)
        functor.record_rejected_trial_path()
        pending = functor._jump_pending
        assert pending is not None and pending["width"] == pytest.approx(6 * 3600.0 / SECONDS_PER_DAY)
        assert int(pending["cell_index"].numel()) == int(trial_counts.sum())

        retry = functor.forward(state, 0.0, 2 * 3600.0)
        remainder = functor._jump_pending
        assert remainder is not None
        assert remainder["time"] == pytest.approx(2 * 3600.0 / SECONDS_PER_DAY)
        assert remainder["width"] == pytest.approx(4 * 3600.0 / SECONDS_PER_DAY)
        assert bool((remainder["event_time"] >= 0).all())
        expected_retry_counts = torch.bincount(
            pending["cell_index"][pending["event_time"] < 2 * 3600.0 / SECONDS_PER_DAY], minlength=64,
        ).to(torch.float64)
        torch.testing.assert_close(retry[COUNT].value, expected_retry_counts, rtol=0.0, atol=0.0)

        tail = functor.forward(state, 2 * 3600.0, 4 * 3600.0)
        assert functor._jump_pending is None
        torch.testing.assert_close(retry[COUNT].value + tail[COUNT].value, trial_counts, rtol=0.0, atol=0.0)
        torch.testing.assert_close(retry[DEPTH].value + tail[DEPTH].value, trial_depth, rtol=1e-12, atol=1e-12)
        assert functor.draw_count == 2

    def test_step_longer_than_the_pending_window_extends_with_fresh_events(self):
        functor = _functor(seed=43, parameters={"event_rate": {"value": 5.0}})
        state = _state(32)
        snapshot = functor.get_extra_state()
        trial = functor.forward(state, 0.0, 3600.0)
        functor.set_extra_state(snapshot)
        functor.record_rejected_trial_path()
        longer = functor.forward(state, 0.0, 3 * 3600.0)
        assert functor._jump_pending is None
        assert bool((longer[COUNT].value >= trial[COUNT].value).all())
        assert bool((longer[DEPTH].value >= trial[DEPTH].value - 1e-12).all())

    def test_pending_ledger_round_trips_and_rejects_corruption(self):
        functor = _functor(seed=45, parameters={"event_rate": {"value": 5.0}})
        state = _state(32)
        snapshot = functor.get_extra_state()
        assert snapshot["jump_pending"] is None
        trial = functor.forward(state, 0.0, 6 * 3600.0)
        assert float(trial[COUNT].value.sum()) > 0
        functor.set_extra_state(snapshot)
        functor.record_rejected_trial_path()
        pinned = functor.get_extra_state()
        assert pinned["jump_pending"] is not None and pinned["jump_pending"]["n_cells"] == 32
        other = _functor(seed=1, parameters={"event_rate": {"value": 5.0}})
        other.set_extra_state(pinned)
        assert other._jump_pending is not None
        torch.testing.assert_close(
            other._jump_pending["event_time"], functor._jump_pending["event_time"], rtol=0.0, atol=0.0,
        )
        good = pinned["jump_pending"]
        for bad in (
            {k: v for k, v in good.items() if k != "unit_mark"},
            dict(good, width=0.0),
            dict(good, unit_mark=good["unit_mark"][:0]),
            dict(good, cell_index=[0]),
        ):
            with pytest.raises((TypeError, ValueError)):
                other.set_extra_state(dict(pinned, jump_pending=bad))
        with pytest.raises(ValueError, match="jump_pending"):
            other.set_extra_state({k: v for k, v in pinned.items() if k != "jump_pending"})

    def test_governed_adaptive_rejection_keeps_the_event_path(self):
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({
            "device": "cpu", "conservation_checks": False, "enable_auto_scale": False,
            "adaptive_dt_enabled": True, "adaptive_atol": 1e-30, "adaptive_rtol": 1e-30,
            "adaptive_dt_min": 1.0, "adaptive_dt_max": 7200.0,
        })
        simulator.initialize_state(_state(64))
        functor = _functor(seed=31, parameters={"event_rate": {"value": 5.0}})
        simulator.register_functor(functor, execution_group="fast")
        probe = _functor(seed=31, parameters={"event_rate": {"value": 5.0}})
        trial = probe.forward(_state(64), 0.0, 3600.0)
        assert float(trial[COUNT].value.sum()) > 0
        simulator.step(dt=3600.0)
        accepted_dt = simulator.current_time
        assert 0.0 < accepted_dt < 3600.0
        accepted = simulator.get_state()
        pending = functor.get_extra_state()["jump_pending"]
        assert pending is not None
        assert pending["width"] == pytest.approx((3600.0 - accepted_dt) / SECONDS_PER_DAY)
        remaining_counts = torch.bincount(pending["cell_index"], minlength=64).to(torch.float64)
        torch.testing.assert_close(accepted[COUNT].to(torch.float64) + remaining_counts, trial[COUNT].value, rtol=0.0, atol=0.0)
        assert functor.draw_count == 1


def test_config_first_source_order_is_the_mixin_class_declaration():
    """Round 47: explicit config overrides precede the ParameterManager through the
    NamespaceDefaultMixin class declaration, not through a get_parameter_value override."""
    from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin

    for functor_class in (MarkedPoissonRainfallFunctor,):
        assert issubclass(functor_class, NamespaceDefaultMixin)
        assert functor_class.parameter_lookup_source_order == ("config", "manager")
        assert "get_parameter_value" not in functor_class.__dict__
