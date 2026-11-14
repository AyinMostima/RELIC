"""G11-OBS: managed observation feedback through the governed simulator path.

References: Kalman (1960); Kalnay (2003) ch. 5; Evensen (2003).
Mathematical expression: d = y - x, K = B / (B + R), x_a = x + K d,
P_a = (1 - K) B, nu = |d| / sqrt(B + R).
Parameter meanings: B background error variance, R observation error
variance, nu_max innovation gate.
Units: soil moisture m3 m-3 (dimensionless volumetric fraction); variances
dimensionless; K and nu dimensionless.
Domain/constraints: B, R > 0; the gate refuses the whole update; the
functor is stateless.
Sign convention: d > 0 moves the analysis toward the larger observation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.coupling.observation import ObservationStateFeedbackFunctor, TowerFluxSupportOperator
from relic.core.functor.functor_types import VariableKind
from relic.parameters import get_global_manager

B_KEY = ObservationStateFeedbackFunctor.BACKGROUND_ERROR_VARIANCE
R_KEY = ObservationStateFeedbackFunctor.OBSERVATION_ERROR_VARIANCE
GATE_KEY = ObservationStateFeedbackFunctor.MAX_NORMALIZED_INNOVATION
DIAG = "observation_state_feedback.observation.state_feedback."


def _functor() -> ObservationStateFeedbackFunctor:
    return ObservationStateFeedbackFunctor(config={
        "state_variable": "soil_moisture",
        "observation_variable": "soil_moisture_observation",
        "units": "m3 m-3",
    })


def _simulator(observation: float) -> RELICSimulator:
    sim = RELICSimulator({
        "device": "cpu",
        "conservation_checks": False,
        "strict_dependencies": False,
        "adaptive_dt_enabled": False,
        # The configured observation variable carries the units of _functor().
        "state_units": {"soil_moisture_observation": "m3 m-3"},
    })
    sim.initialize_state({
        "soil_moisture": torch.tensor([0.30], dtype=torch.float64),
        "soil_moisture_observation": torch.tensor([observation], dtype=torch.float64),
    })
    sim.register_functor(_functor(), execution_group="fast")
    return sim


class TestKernel:
    def test_hand_computed_analysis(self):
        x = torch.tensor([0.30], dtype=torch.float64)
        y = torch.tensor([0.35], dtype=torch.float64)
        result = ObservationStateFeedbackFunctor.analysis(x, y, torch.tensor(0.01, dtype=torch.float64), torch.tensor(0.04, dtype=torch.float64))
        assert float(result["gain"]) == pytest.approx(0.2)
        assert float(result["innovation"]) == pytest.approx(0.05)
        assert float(result["analysis"]) == pytest.approx(0.31)
        assert float(result["analysis_variance"]) == pytest.approx(0.008)
        assert float(result["normalized_innovation"]) == pytest.approx(0.05 / (0.05 ** 0.5))

    def test_parameter_gradient_of_the_analysis_is_analytic(self):
        x = torch.tensor([0.30], dtype=torch.float64)
        y = torch.tensor([0.35], dtype=torch.float64)
        background = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
        observation = torch.tensor(0.04, dtype=torch.float64, requires_grad=True)
        analysis = ObservationStateFeedbackFunctor.analysis(x, y, background, observation)["analysis"]
        grad_b, grad_r = torch.autograd.grad(analysis.sum(), (background, observation))
        # d x_a / d R = -d B / (B + R)^2 ; d x_a / d B = d R / (B + R)^2.
        assert float(grad_r) == pytest.approx(-0.05 * 0.01 / 0.05 ** 2)
        assert float(grad_b) == pytest.approx(0.05 * 0.04 / 0.05 ** 2)

    def test_configuration_is_fail_closed(self):
        with pytest.raises(ValueError, match="requires config"):
            ObservationStateFeedbackFunctor(config={})
        with pytest.raises(ValueError, match="must differ"):
            ObservationStateFeedbackFunctor(config={"state_variable": "a", "observation_variable": "a"})


class TestGovernedFeedback:
    def test_analysis_is_committed_by_the_state_manager_with_provenance_and_diagnostics(self):
        manager = get_global_manager()
        assert float(manager.get(B_KEY, required=True)) == 0.01 and float(manager.get(R_KEY, required=True)) == 0.04
        sim = _simulator(0.35)
        sim.step(dt=3600.0)
        assert float(sim.state_manager.get_state("soil_moisture").item()) == pytest.approx(0.31)
        assert float(sim.state_manager.get_state("soil_moisture_observation").item()) == 0.35
        provenance = str(sim.get_variable_provenance("soil_moisture"))
        assert "observation_state_feedback" in provenance and "replace" in provenance.lower()
        current = sim.state_manager.get_diagnostics()["current"]
        assert float(torch.as_tensor(current[DIAG + "innovation"]).item()) == pytest.approx(0.05)
        assert float(torch.as_tensor(current[DIAG + "kalman_gain"]).item()) == pytest.approx(0.2)
        assert float(torch.as_tensor(current[DIAG + "analysis_increment"]).item()) == pytest.approx(0.01)
        assert float(torch.as_tensor(current[DIAG + "analysis_error_variance"]).item()) == pytest.approx(0.008)
        for key in ObservationStateFeedbackFunctor.DIAGNOSTICS:
            assert key not in sim.get_state()

    def test_gate_refuses_the_whole_update_and_leaves_the_state_untouched(self):
        # nu = |d| / sqrt(0.05) = 0.67 for d = 0.15; the gate floor 0.5
        # (parameter bounds) refuses it while the state stays in [0, 1].
        manager = get_global_manager()
        original = float(manager.get(GATE_KEY, required=True))
        try:
            manager.set_value(GATE_KEY, 0.5)
            assert float(manager.get(GATE_KEY, required=True)) == 0.5
            sim = _simulator(0.45)
            with pytest.raises(Exception, match="normalized innovation"):
                sim.step(dt=3600.0)
            assert float(sim.state_manager.get_state("soil_moisture").item()) == 0.30
            assert sim.current_time == 0.0 and sim.step_count == 0
        finally:
            manager.set_value(GATE_KEY, original)

    def test_restart_from_a_durable_checkpoint_reproduces_the_analysis_sequence(self, tmp_path: Path):
        sim = _simulator(0.35)
        sim.step(dt=3600.0)
        checkpoint = tmp_path / "feedback.ckpt"
        sim.create_checkpoint(checkpoint)
        sim.step(dt=3600.0)
        expected = sim.state_manager.get_state("soil_moisture").clone()
        # Second analysis: x = 0.31, d = 0.04, x_a = 0.318.
        assert float(expected.item()) == pytest.approx(0.318)
        resumed = _simulator(0.35)
        resumed.restore_checkpoint(checkpoint)
        resumed.step(dt=3600.0)
        assert torch.equal(resumed.state_manager.get_state("soil_moisture"), expected)

    def test_read_only_observation_operators_stay_read_only(self):
        tower = TowerFluxSupportOperator()
        assert tower.metadata.variable_kind == VariableKind.DIAGNOSTIC
        assert all(name.startswith("diagnostics.") for name in tower.metadata.output_variables)
        feedback = _functor()
        assert feedback.metadata.variable_kind == VariableKind.STATE
        assert "soil_moisture" in feedback.metadata.output_variables

    def test_unauthorized_target_is_refused_by_the_contract(self):
        functor = ObservationStateFeedbackFunctor(config={
            "state_variable": "nh4_nitrogen",
            "observation_variable": "nh4_observation",
            "units": "g N m-2",
        })
        sim = RELICSimulator({
            "device": "cpu",
            "conservation_checks": False,
            "strict_dependencies": False,
            # The configured observation variable carries the Functor's declared units.
            "state_units": {"nh4_observation": "g N m-2"},
        })
        sim.initialize_state({
            "nh4_nitrogen": torch.tensor([0.10], dtype=torch.float64),
            "nh4_observation": torch.tensor([0.12], dtype=torch.float64),
        })
        sim.register_functor(functor, execution_group="fast")
        with pytest.raises(Exception, match="not authorized"):
            sim.step(dt=3600.0)
        assert float(sim.state_manager.get_state("nh4_nitrogen").item()) == 0.10
