import torch

from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.optimization.assimilation import (
    EnKFConfig,
    EnsembleKalmanFilter,
    IdentityOperator,
    Observation,
    ObservationSet,
    ThreeDVar,
    VariationalConfig,
)
from relic.optimization.assimilation.base import AssimilationMethod
from relic.parameters import get_global_manager


def _build_scalar_observation(value: float, variable_name: str, time_value: float = 0.0) -> ObservationSet:
    observation = Observation(
        value=torch.tensor([value], dtype=torch.float32),
        error_variance=torch.tensor([1.5**2], dtype=torch.float32),
        coordinates=(0.0, 0.0, 0.0),
        time=time_value,
        variable_name=variable_name,
        quality_flag=0,
    )
    return ObservationSet(observations=[observation], time=time_value)


def _build_runtime_objects():
    state_manager = StateManager()
    simulator = RELICSimulator()
    simulator.set_state_manager(state_manager)
    parameter_manager = get_global_manager()
    return simulator, state_manager, parameter_manager


def test_threedvar_preserves_forecast_variable_names() -> None:
    simulator, state_manager, parameter_manager = _build_runtime_objects()
    assimilator = ThreeDVar(
        simulator,
        state_manager,
        parameter_manager,
        VariationalConfig(
            method=AssimilationMethod.VARIATIONAL_3D, max_iterations=20, convergence_tolerance=1e-6
        ),
    )
    forecast_state = {"GPP": torch.tensor([10.0], dtype=torch.float32)}
    observations = _build_scalar_observation(12.5, "GPP")
    operator = IdentityOperator(variable_names=["GPP"])

    result = assimilator.assimilate(forecast_state, observations, operator)

    assert set(result.analysis_mean.keys()) == {"GPP"}
    assert set(result.forecast_mean.keys()) == {"GPP"}
    assert result.analysis_mean["GPP"].item() > forecast_state["GPP"].item()
    assert abs(result.innovation[0].item()) < 2.5


def test_threedvar_respects_unobserved_state_components() -> None:
    simulator, state_manager, parameter_manager = _build_runtime_objects()
    assimilator = ThreeDVar(
        simulator,
        state_manager,
        parameter_manager,
        VariationalConfig(
            method=AssimilationMethod.VARIATIONAL_3D, max_iterations=20, convergence_tolerance=1e-6
        ),
    )
    forecast_state = {
        "GPP": torch.tensor([10.0], dtype=torch.float32),
        "ET": torch.tensor([2.0], dtype=torch.float32),
    }
    observations = _build_scalar_observation(12.5, "GPP")
    operator = IdentityOperator(variable_names=["GPP"])

    result = assimilator.assimilate(forecast_state, observations, operator)

    assert set(result.analysis_mean.keys()) == {"ET", "GPP"}
    assert result.analysis_mean["GPP"].item() > forecast_state["GPP"].item()
    assert torch.allclose(result.analysis_mean["ET"], forecast_state["ET"], atol=1.0e-6)


def test_enkf_identity_operator_runs_with_named_scientific_state() -> None:
    simulator, state_manager, parameter_manager = _build_runtime_objects()
    assimilator = EnsembleKalmanFilter(
        simulator,
        state_manager,
        parameter_manager,
        EnKFConfig(ensemble_size=8, initial_perturbation_std=0.1, covariance_inflation=1.05),
    )
    forecast_state = {"GPP": torch.tensor([10.0], dtype=torch.float32)}
    observations = _build_scalar_observation(12.5, "GPP")
    operator = IdentityOperator(variable_names=["GPP"])
    assimilator.initialize_ensemble(forecast_state)

    result = assimilator.assimilate(forecast_state, observations, operator)
    stats = assimilator.get_ensemble_statistics()

    assert set(result.analysis_mean.keys()) == {"GPP"}
    assert "GPP" in stats
    assert stats["GPP"]["ensemble_spread"] >= 0.0
