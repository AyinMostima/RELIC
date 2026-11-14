"""L1 tests for the Ornstein-Uhlenbeck stochastic air-temperature anomaly Functor.

References:
    - Hasselmann (1976), Tellus 28, 473-485.
    - Kloeden and Platen (1992), ch. 10; Higham (2001), SIAM Review 43.
"""

import math

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.atmosphere.stochastic_temperature import (
    OrnsteinUhlenbeckTemperatureFunctor,
)
from relic.utils.time_constants import SECONDS_PER_DAY

STATE = "stochastic_air_temperature_anomaly"


def _functor(seed=7, **overrides):
    config = {"seed": seed}
    config.update(overrides)
    return OrnsteinUhlenbeckTemperatureFunctor(name="stochastic_air_temperature", config=config)


def _coefficients(functor):
    alpha = float(functor.get_parameter_value("reversion_rate", required=True).detach())
    sigma = float(functor.get_parameter_value("volatility", required=True).detach())
    return alpha, sigma


class TestContract:
    def test_metadata_and_owned_substream(self):
        functor = _functor()
        assert functor.metadata.input_variables == [STATE]
        assert functor.metadata.output_variables[0] == STATE
        assert all(
            name.startswith("diagnostics.stochastic_temperature.")
            for name in functor.metadata.output_variables[1:]
        )
        assert functor.get_dependencies() == []
        assert isinstance(functor.generator, torch.Generator)
        assert functor.stochastic_category == "continuous_sde"
        assert functor.noise_classification == "external_forcing"
        extra = functor.get_extra_state()
        assert extra["format"] == functor.EXTRA_STATE_FORMAT
        assert extra["draw_count"] == 0
        assert isinstance(extra["generator_state"], torch.Tensor)

    @pytest.mark.parametrize("config", [{}, {"seed": -1}, {"seed": 1.5}, {"seed": True}])
    def test_seed_is_required_and_validated(self, config):
        with pytest.raises(ValueError, match="seed"):
            OrnsteinUhlenbeckTemperatureFunctor(name="stochastic_air_temperature", config=config)


class TestEulerMaruyamaStep:
    def test_step_matches_exact_ar1_conditional_statistics(self):
        # References: Hasselmann (1976); Kloeden and Platen (1992) ch. 10.
        # Mathematical expression: X1 = X0 - alpha X0 dt + sigma dW, dW ~ N(0, dt).
        # Parameter meanings: alpha day-1, sigma K day-1/2, dt in days.
        # Units: X K; dt s converted to days by SECONDS_PER_DAY.
        # Domain/constraints: 0 < alpha dt < 1.
        # Sign convention: positive increment warms the anomaly.
        functor = _functor(seed=11)
        alpha, sigma = _coefficients(functor)
        dt_seconds = 6.0 * 3600.0
        dt_days = dt_seconds / float(SECONDS_PER_DAY)
        x0 = torch.tensor(2.5, dtype=torch.float64)
        result = functor.forward({STATE: x0.clone()}, 0.0, dt_seconds)
        item = result[STATE]
        assert isinstance(item, UpdateItem)
        assert item.mode is UpdateMode.ACCUMULATE
        assert item.unit == "K"
        drift = result["diagnostics.stochastic_temperature.drift"].value
        diffusion = result["diagnostics.stochastic_temperature.diffusion"].value
        wiener = result["diagnostics.stochastic_temperature.wiener_increment"].value
        torch.testing.assert_close(drift, torch.tensor(-alpha * 2.5, dtype=torch.float64))
        torch.testing.assert_close(diffusion, torch.tensor(sigma, dtype=torch.float64))
        torch.testing.assert_close(item.value, drift * dt_days + diffusion * wiener, rtol=1e-12, atol=0.0)
        assert functor.get_extra_state()["draw_count"] == 1

        # Same seed reproduces the same Wiener increment; another seed does not.
        again = _functor(seed=11).forward({STATE: x0.clone()}, 0.0, dt_seconds)
        torch.testing.assert_close(again[STATE].value, item.value, rtol=0.0, atol=0.0)
        other = _functor(seed=12).forward({STATE: x0.clone()}, 0.0, dt_seconds)
        assert float(other[STATE].value.detach()) != float(item.value.detach())

    def test_stationary_variance_of_the_discrete_recursion(self):
        # Var_inf = sigma^2 dt / (1 - (1 - alpha dt)^2) for the AR(1) recursion.
        functor = _functor(seed=3)
        alpha, sigma = _coefficients(functor)
        dt_days = 0.5
        dt_seconds = dt_days * float(SECONDS_PER_DAY)
        n_boxes = 4096
        x = torch.zeros(n_boxes, dtype=torch.float64)
        for _ in range(400):
            x = x + functor.forward({STATE: x}, 0.0, dt_seconds)[STATE].value.detach()
        expected = sigma * sigma * dt_days / (1.0 - (1.0 - alpha * dt_days) ** 2)
        observed = float(x.var(unbiased=True))
        standard_error = expected * math.sqrt(2.0 / (n_boxes - 1))
        assert abs(observed - expected) < 4.0 * standard_error
        assert abs(float(x.mean())) < 4.0 * math.sqrt(expected / n_boxes)

    def test_gradients_flow_to_state_and_parameters(self):
        volatility = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        functor = _functor(seed=5, parameters={"volatility": volatility})
        alpha, _ = _coefficients(functor)
        x0 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        dt_seconds = 3600.0
        result = functor.forward({STATE: x0}, 0.0, dt_seconds)
        wiener = result["diagnostics.stochastic_temperature.wiener_increment"].value
        grads = torch.autograd.grad(result[STATE].value, (x0, volatility))
        assert float(grads[0]) == pytest.approx(-alpha * dt_seconds / float(SECONDS_PER_DAY))
        assert float(grads[1]) == pytest.approx(float(wiener))

    @pytest.mark.parametrize("case", ["missing_state", "nonfinite", "too_large_step", "non_tensor"])
    def test_domain_rejections(self, case):
        functor = _functor()
        state = {STATE: torch.tensor(0.5, dtype=torch.float64)}
        dt = 3600.0
        if case == "missing_state":
            state = {}
        elif case == "nonfinite":
            state[STATE] = torch.tensor(float("nan"), dtype=torch.float64)
        elif case == "too_large_step":
            dt = 400.0 * float(SECONDS_PER_DAY)
        elif case == "non_tensor":
            state[STATE] = 0.5
        with pytest.raises((ValueError, TypeError)):
            functor.forward(state, 0.0, dt)
        assert functor.get_extra_state()["draw_count"] == 0


class TestGovernedRuntime:
    @staticmethod
    def _simulator(seed):
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False, "enable_auto_scale": False})
        simulator.initialize_state({STATE: torch.tensor(1.0, dtype=torch.float64)})
        functor = _functor(seed=seed)
        simulator.register_functor(functor, execution_group="fast")
        return simulator, functor

    def test_checkpoint_restart_continues_the_same_substream(self, tmp_path):
        dt = 3.0 * 3600.0
        uninterrupted, _ = self._simulator(seed=21)
        for _ in range(6):
            uninterrupted.step(dt=dt)

        source, source_functor = self._simulator(seed=21)
        for _ in range(3):
            source.step(dt=dt)
        checkpoint = tmp_path / "ou.pt"
        source.create_checkpoint(checkpoint)
        assert source_functor.get_extra_state()["draw_count"] == 3

        restored, restored_functor = self._simulator(seed=99)
        restored.restore_checkpoint(checkpoint)
        assert restored_functor.get_extra_state()["draw_count"] == 3
        for _ in range(3):
            restored.step(dt=dt)
        torch.testing.assert_close(
            restored.state_manager.get_state(STATE),
            uninterrupted.state_manager.get_state(STATE),
            rtol=0.0,
            atol=0.0,
        )
        assert restored.step_count == 6

    def test_rejected_adaptive_step_rewinds_the_substream(self):
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({
            "device": "cpu", "conservation_checks": False, "enable_auto_scale": False,
            "adaptive_dt_enabled": True, "adaptive_atol": 1e-30, "adaptive_rtol": 1e-30,
            "adaptive_dt_min": 1.0, "adaptive_dt_max": 7200.0,
        })
        simulator.initialize_state({STATE: torch.tensor(1.0, dtype=torch.float64)})
        functor = _functor(seed=31)
        simulator.register_functor(functor, execution_group="fast")
        before = functor.get_extra_state()
        simulator.step(dt=3600.0)
        after = functor.get_extra_state()
        # Whatever the adaptive controller rejected, the accepted step consumed
        # exactly one draw from the substream position that was snapshotted.
        assert after["draw_count"] == before["draw_count"] + 1


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA substream test requires a CUDA device")
def test_cuda_functor_owns_a_cuda_substream_and_advances_cuda_states():
    functor = _functor(seed=13)
    functor.to_device("cuda")
    assert torch.device(functor.generator.device).type == "cuda"
    state = {STATE: torch.tensor(1.0, dtype=torch.float64, device="cuda")}
    result = functor.forward(state, 0.0, 3600.0)
    assert result[STATE].value.device.type == "cuda"
    assert functor.get_extra_state()["draw_count"] == 1
    with pytest.raises(ValueError, match="migration across device types after draws"):
        functor.to_device("cpu")



class TestAdaptiveWienerPath:
    """Variable steps stay on one Wiener path (F05).

    References: Gaines and Lyons (1997) Brownian tree; Karlin and Taylor
    (1981) ch. 9 Brownian bridge.
    """

    def _diagnostic(self, output, name):
        return output[name].value

    def test_rejected_trial_is_refined_by_the_bridge_at_the_kernel(self):
        functor = _functor(seed=17)
        state = {STATE: torch.tensor([0.5, -0.25], dtype=torch.float64)}
        snapshot = functor.get_extra_state()
        trial = functor.forward(state, 0.0, 3600.0)
        trial_dw = self._diagnostic(trial, functor.DIAGNOSTICS[2]).clone()
        # The simulator rejects the trial: rewind, then pin the path.
        functor.set_extra_state(snapshot)
        functor.record_rejected_trial_path()
        assert len(functor.wiener_pending) == 1
        torch.testing.assert_close(functor.wiener_pending[0]["increment"], trial_dw, rtol=0.0, atol=0.0)

        retry = functor.forward(state, 0.0, 900.0)
        retry_dw = self._diagnostic(retry, functor.DIAGNOSTICS[2])
        assert len(functor.wiener_pending) == 1
        remainder = functor.wiener_pending[0]
        assert remainder["width"] == pytest.approx(2700.0 / SECONDS_PER_DAY)
        assert remainder["time"] == pytest.approx(900.0 / SECONDS_PER_DAY)
        torch.testing.assert_close(retry_dw + remainder["increment"], trial_dw, rtol=1e-12, atol=1e-12)
        # A plain redraw of the short step would scale the trial by sqrt(1/4).
        assert not torch.allclose(retry_dw, trial_dw * 0.5)
        # The state increment uses exactly that path increment.
        alpha, sigma = _coefficients(functor)
        dt_days = 900.0 / SECONDS_PER_DAY
        expected = -alpha * state[STATE] * dt_days + sigma * retry_dw
        torch.testing.assert_close(retry[STATE].value, expected, rtol=1e-12, atol=1e-12)

        # Continuing over the remainder consumes the pinned piece, so the
        # two accepted pieces sum to the rejected trial's increment.
        tail = functor.forward(state, 900.0, 2700.0)
        tail_dw = self._diagnostic(tail, functor.DIAGNOSTICS[2])
        torch.testing.assert_close(retry_dw + tail_dw, trial_dw, rtol=1e-12, atol=1e-12)
        assert functor.wiener_pending == []
        # Two emitted steps after the rewind: the retry and the tail.
        assert functor.draw_count == 2

    def test_fixed_step_rewind_replays_the_same_trial(self):
        functor = _functor(seed=23)
        state = {STATE: torch.tensor(1.0, dtype=torch.float64)}
        snapshot = functor.get_extra_state()
        first = functor.forward(state, 0.0, 3600.0)
        functor.set_extra_state(snapshot)
        second = functor.forward(state, 0.0, 3600.0)
        torch.testing.assert_close(first[STATE].value, second[STATE].value, rtol=0.0, atol=0.0)
        assert functor.wiener_pending == []

    @pytest.mark.parametrize("attempt_s", [3600.0, 0.5])
    def test_governed_retry_interval_stays_inside_the_rejected_interval(self, attempt_s):
        """The retry refines the pinned rejected increment: its interval is a sub-interval.

        An attempt no longer than adaptive_dt_min cannot be shortened; it is
        accepted without rollback, so no increment is pinned and no retry drawn.
        """
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({
            "device": "cpu", "conservation_checks": False, "enable_auto_scale": False,
            "adaptive_dt_enabled": True, "adaptive_atol": 1e-30, "adaptive_rtol": 1e-30,
            "adaptive_dt_min": 1.0, "adaptive_dt_max": 7200.0,
        })
        simulator.initialize_state({STATE: torch.tensor(1.0, dtype=torch.float64)})
        functor = _functor(seed=37)
        draws = []
        original_draw = functor.draw_wiener_increment

        def recording_draw(time, width, *, like):
            draws.append((float(time), float(width)))
            return original_draw(time, width, like=like)

        functor.draw_wiener_increment = recording_draw
        simulator.register_functor(functor, execution_group="fast")

        simulator.step(dt=attempt_s)

        start, rejected_width = draws[0]
        assert rejected_width == pytest.approx(attempt_s / SECONDS_PER_DAY)
        for time, width in draws[1:]:
            assert time >= start
            assert time + width <= start + rejected_width * (1.0 + 1e-12)
        if attempt_s <= 1.0:
            assert len(draws) == 1
            assert simulator.current_time == attempt_s
            assert functor.get_extra_state()["wiener_pending"] == []
        else:
            assert len(draws) == 2
            assert 0.0 < simulator.current_time < attempt_s

    def test_governed_adaptive_rejection_keeps_the_path(self):
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({
            "device": "cpu", "conservation_checks": False, "enable_auto_scale": False,
            "adaptive_dt_enabled": True, "adaptive_atol": 1e-30, "adaptive_rtol": 1e-30,
            "adaptive_dt_min": 1.0, "adaptive_dt_max": 7200.0,
        })
        simulator.initialize_state({STATE: torch.tensor(1.0, dtype=torch.float64)})
        functor = _functor(seed=31)
        simulator.register_functor(functor, execution_group="fast")
        # Reproduce the rejected trial's increment from the same seed.
        probe = _functor(seed=31)
        trial_dw = probe.forward(
            {STATE: torch.tensor(1.0, dtype=torch.float64)}, 0.0, 3600.0,
        )[functor.DIAGNOSTICS[2]].value.clone()

        simulator.step(dt=3600.0)
        accepted_dt = simulator.current_time
        assert 0.0 < accepted_dt < 3600.0
        alpha, sigma = _coefficients(functor)
        x1 = simulator.get_state()[STATE].to(torch.float64)
        dt_days = accepted_dt / SECONDS_PER_DAY
        # X1 = X0 - alpha X0 dt_d + sigma dW1 with X0 = 1 gives the accepted dW1.
        accepted_dw = (x1 - 1.0 + alpha * 1.0 * dt_days) / sigma
        pending = functor.get_extra_state()["wiener_pending"]
        assert len(pending) == 1
        assert pending[0]["width"] == pytest.approx((3600.0 - accepted_dt) / SECONDS_PER_DAY)
        torch.testing.assert_close(
            accepted_dw.to(torch.float64) + pending[0]["increment"].to(torch.float64),
            trial_dw, rtol=1e-12, atol=1e-12,
        )
        assert functor.draw_count == 1


def test_config_first_source_order_is_the_mixin_class_declaration():
    """Round 47: explicit config overrides precede the ParameterManager through the
    NamespaceDefaultMixin class declaration, not through a get_parameter_value override."""
    from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin

    for functor_class in (OrnsteinUhlenbeckTemperatureFunctor,):
        assert issubclass(functor_class, NamespaceDefaultMixin)
        assert functor_class.parameter_lookup_source_order == ("config", "manager")
        assert "get_parameter_value" not in functor_class.__dict__
