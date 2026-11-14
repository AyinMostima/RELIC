"""L1 tests for the Ensemble Kalman Filter (relic/optimization/assimilation/enkf.py).

References:
    Evensen (2003), Ocean Dynamics 53, 343-367; Burgers et al. (1998),
    Mon. Wea. Rev. 126, 1719-1724.
Mathematical expression:
    K = (N-1)^-1 A (HA)^T [(N-1)^-1 (HA)(HA)^T + R + r I]^-1.
Parameter meanings:
    A: state anomalies, HA: predicted-observation anomalies, R: observation
    error covariance, r: the configured regularization, N: ensemble size.

A failure of the declared analysis is raised with its cause: no member keeps
its old state as a 'forecast', no pseudo-inverse replaces a failed solve, and
no configuration option that has no implementation is accepted.
"""

from __future__ import annotations

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.optimization.assimilation import (
    EnKFConfig,
    EnsembleKalmanFilter,
    IdentityOperator,
    Observation,
    ObservationSet,
)
from relic.parameters import get_global_manager


def _observation(value: float, variable: str = "GPP") -> ObservationSet:
    observation = Observation(
        value=torch.tensor([value], dtype=torch.float32),
        error_variance=torch.tensor([1.5**2], dtype=torch.float32),
        coordinates=(0.0, 0.0, 0.0),
        time=0.0,
        variable_name=variable,
        quality_flag=0,
    )
    return ObservationSet(observations=[observation], time=0.0)


def _filter(**config) -> EnsembleKalmanFilter:
    state_manager = StateManager()
    simulator = RELICSimulator()
    simulator.set_state_manager(state_manager)
    values = {"ensemble_size": 8, "initial_perturbation_std": 0.1}
    values.update(config)
    enkf = EnsembleKalmanFilter(
        simulator, state_manager, get_global_manager(), EnKFConfig(**values)
    )
    enkf.initialize_ensemble({"GPP": torch.tensor([10.0], dtype=torch.float32)})
    return enkf


def test_ensemble_forecast_is_refused_instead_of_keeping_stale_members() -> None:
    enkf = _filter()
    before = {name: value.clone() for name, value in enkf.ensemble_states.items()}

    with pytest.raises(NotImplementedError, match="forecast"):
        enkf.forecast_ensemble(1.0)
    assert all(torch.equal(before[name], enkf.ensemble_states[name]) for name in before)


def test_failed_gain_solve_is_not_replaced_by_a_pseudo_inverse(monkeypatch) -> None:
    enkf = _filter()

    def failing_solve(*_args, **_kwargs):
        raise torch.linalg.LinAlgError("singular innovation covariance")

    monkeypatch.setattr(torch.linalg, "solve", failing_solve)

    with pytest.raises(RuntimeError, match="EnKF analysis failed") as excinfo:
        enkf.assimilate({}, _observation(12.5), IdentityOperator(["GPP"]))
    cause = excinfo.value.__cause__
    assert isinstance(cause, RuntimeError) and "innovation covariance" in str(cause)
    assert isinstance(cause.__cause__, torch.linalg.LinAlgError)


def test_analysis_failure_keeps_its_cause() -> None:
    enkf = _filter()

    class FailingOperator(IdentityOperator):
        def forward(self, state):
            raise ValueError("operator domain error")

    with pytest.raises(RuntimeError, match="operator domain error") as excinfo:
        enkf.assimilate({}, _observation(12.5), FailingOperator(["GPP"]))
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_explicit_covariance_gain_with_an_identity_jacobian_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="efficient_gain"):
        EnKFConfig(initial_perturbation_std=0.1, efficient_gain=False)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("localization_length_scale", 50.0),
        ("adaptive_inflation", True),
        ("svd_truncation", 0.99),
        ("minimum_variance", 1.0e-6),
        ("batch_observations", False),
        ("inflation_factor", 1.05),
    ],
)
def test_options_without_an_implementation_are_not_accepted(field_name, value) -> None:
    """These settings were stored but never applied; they are no longer offered."""
    with pytest.raises(TypeError):
        EnKFConfig(initial_perturbation_std=0.1, **{field_name: value})


@pytest.mark.parametrize("method", ["none", "gaspari_cohn", "cutoff"])
def test_localization_is_not_offered(method) -> None:
    """Covariance localization is not implemented, so the setting does not exist."""
    with pytest.raises(TypeError):
        EnKFConfig(initial_perturbation_std=0.1, localization_method=method)


def test_enkf_config_fixes_its_method_and_needs_the_state_perturbation() -> None:
    from relic.optimization.assimilation.base import AssimilationMethod

    config = EnKFConfig(initial_perturbation_std=0.1)
    assert config.method is AssimilationMethod.ENSEMBLE_KALMAN_FILTER
    with pytest.raises(TypeError):
        EnKFConfig(initial_perturbation_std=0.1, method=AssimilationMethod.VARIATIONAL_3D)
    with pytest.raises(TypeError):
        EnKFConfig()


def test_enkf_numerical_settings_are_read_from_the_parameter_database() -> None:
    keys = {
        "parameter_perturbation_std": "optimization.assimilation.enkf.parameter_perturbation_std",
        "covariance_inflation": "optimization.assimilation.enkf.covariance_inflation",
        "regularization": "optimization.assimilation.enkf.regularization",
    }
    expected = {"parameter_perturbation_std": 0.05, "covariance_inflation": 1.02, "regularization": 1.0e-6}
    manager = get_global_manager()
    originals = {key: manager.get(key, required=True) for key in keys.values()}
    config = EnKFConfig(initial_perturbation_std=0.1)
    for name, value in expected.items():
        assert getattr(config, name) == pytest.approx(value)
    assert EnKFConfig(initial_perturbation_std=0.1, covariance_inflation=1.5).covariance_inflation == 1.5
    try:
        manager.set_value(keys["covariance_inflation"], 1.1)
        assert EnKFConfig(initial_perturbation_std=0.1).covariance_inflation == pytest.approx(1.1)
    finally:
        for key, value in originals.items():
            manager.set_value(key, value)


def test_declared_inflation_is_applied_to_the_analysis_anomalies() -> None:
    torch.manual_seed(3)
    inflated = _filter(covariance_inflation=1.5)
    torch.manual_seed(3)
    plain = _filter(covariance_inflation=1.0)

    torch.manual_seed(5)
    inflated.assimilate({}, _observation(12.5), IdentityOperator(["GPP"]))
    torch.manual_seed(5)
    plain.assimilate({}, _observation(12.5), IdentityOperator(["GPP"]))

    spread_inflated = torch.std(inflated.ensemble_states["GPP"])
    spread_plain = torch.std(plain.ensemble_states["GPP"])
    assert float(spread_inflated) == pytest.approx(1.5 * float(spread_plain), rel=1e-5)
