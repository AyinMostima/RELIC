"""L1 tests for bootstrap uncertainty quantification.

Module under test: relic/optimization/uncertainty/bootstrap.py (BootstrapUQ,
BootstrapConfig).

References:
    Efron (1987), J. Am. Stat. Assoc. 82, 171-185 (bias-corrected intervals);
    Efron and Tibshirani (1993), An Introduction to the Bootstrap, ch. 13-14.
Mathematical expression:
    percentile CI = [q(alpha/2), q(1 - alpha/2)] of the bootstrap statistics;
    BC (a = 0): z0 = Phi^-1(#{theta*_b < theta_hat} / B),
    alpha_1,2 = Phi(2 z0 + z_{alpha/2}), Phi(2 z0 + z_{1-alpha/2}).
Parameter meanings:
    theta*_b are the bootstrap statistics, theta_hat the original statistic,
    B the number of bootstrap statistics, alpha = 1 - confidence level.

A requested method either runs as declared or raises: no percentile interval
is substituted for a failed BCa interval, no unimplemented option (acceleration,
basic or studentized intervals, paired bootstrap, write-only settings) is
accepted, and no missing or non-finite statistic is replaced or dropped.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import relic.optimization.uncertainty.bootstrap as bootstrap_module
from relic.optimization.uncertainty.base import DistributionType, UncertainParameter
from relic.optimization.uncertainty.bootstrap import BootstrapConfig, BootstrapUQ


class _DummyParameterManager:
    def exists(self, name: str) -> bool:
        return True


def _uq(**overrides) -> BootstrapUQ:
    values = {
        "n_bootstrap": 8, "random_seed": 7,
        "bootstrap_type": "parametric", "ci_method": "percentile",
    }
    values.update(overrides)
    return BootstrapUQ(_DummyParameterManager(), BootstrapConfig(**values))


def _uniform(name: str = "a") -> dict[str, UncertainParameter]:
    return {
        name: UncertainParameter(
            name=name,
            distribution=DistributionType.UNIFORM,
            lower_bound=0.0,
            upper_bound=1.0,
        )
    }


def _linear_model(parameters):
    return {"y": 2.0 * float(parameters["a"])}


# ---------------------------------------------------------------------------
# Configuration: only implemented options are accepted.
# ---------------------------------------------------------------------------


def test_acceleration_correction_is_rejected_instead_of_a_zero_placeholder() -> None:
    with pytest.raises(NotImplementedError, match="acceleration"):
        BootstrapConfig(bootstrap_type="parametric", ci_method="bca", acceleration_correction=True)


@pytest.mark.parametrize("method", ["basic", "studentized", "percentile_t"])
def test_unimplemented_interval_method_is_rejected(method: str) -> None:
    with pytest.raises(ValueError, match="ci_method"):
        BootstrapConfig(bootstrap_type="parametric", ci_method=method)


@pytest.mark.parametrize("bootstrap_type", ["paired", "block"])
def test_unimplemented_bootstrap_type_is_rejected(bootstrap_type: str) -> None:
    with pytest.raises(ValueError, match="bootstrap_type"):
        BootstrapConfig(ci_method="percentile", bootstrap_type=bootstrap_type)


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("bias_correction", False),
        ("stratified_sampling", True),
        ("minimum_sample_size", 50),
        ("outlier_handling", "remove"),
    ],
)
def test_settings_without_an_implementation_are_not_accepted(field_name: str, value) -> None:
    """These settings were stored but never read; they are no longer offered."""
    with pytest.raises(TypeError):
        BootstrapConfig(ci_method="percentile", bootstrap_type="parametric", **{field_name: value})


def test_residual_bootstrap_without_observations_is_an_error() -> None:
    uq = _uq(bootstrap_type="residual")

    with pytest.raises(ValueError, match="observations"):
        uq.quantify(_linear_model, _uniform())


def test_balanced_resampling_of_a_non_uniform_parameter_is_an_error() -> None:
    uq = _uq(balanced_resampling=True)
    parameters = {
        "a": UncertainParameter(
            name="a", distribution=DistributionType.NORMAL, mean=1.0, std=0.1
        )
    }

    with pytest.raises(RuntimeError, match="balanced") as excinfo:
        uq.quantify(_linear_model, parameters)
    assert isinstance(excinfo.value.__cause__, NotImplementedError)


# ---------------------------------------------------------------------------
# Execution: failures keep their cause; nothing is substituted.
# ---------------------------------------------------------------------------


def test_quantify_failure_keeps_its_cause() -> None:
    uq = _uq()

    def failing_model(_parameters):
        raise ValueError("model diverged")

    with pytest.raises(RuntimeError, match="Bootstrap UQ failed") as excinfo:
        uq.quantify(failing_model, _uniform())
    cause = excinfo.value.__cause__
    while cause is not None and not isinstance(cause, ValueError):
        cause = cause.__cause__
    assert isinstance(cause, ValueError) and str(cause) == "model diverged"


def test_parametric_bootstrap_runs_without_a_seed() -> None:
    result = _uq(random_seed=None).quantify(_linear_model, _uniform())

    assert set(result.confidence_intervals) == {"y"}


def test_replicate_that_drops_an_output_is_an_error() -> None:
    """The original sample defines the outputs; a replicate may not drop one."""
    calls = {"count": 0}

    def shrinking_model(parameters):
        calls["count"] += 1
        outputs = {"y": 2.0 * float(parameters["a"])}
        if calls["count"] <= 100:
            outputs["z"] = float(parameters["a"])
        return outputs

    with pytest.raises(RuntimeError, match="'z'") as excinfo:
        _uq().quantify(shrinking_model, _uniform())
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_residual_bootstrap_rejects_a_length_mismatch() -> None:
    uq = _uq(bootstrap_type="residual")

    def vector_model(_parameters):
        return {"y": np.array([1.0, 2.0, 3.0])}

    with pytest.raises(RuntimeError, match="length") as excinfo:
        uq.quantify(vector_model, _uniform(), observations={"y": np.array([1.0, 2.0])})
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_residual_bootstrap_rejects_an_observed_variable_the_model_lacks() -> None:
    uq = _uq(bootstrap_type="residual")

    with pytest.raises(RuntimeError, match="'w'") as excinfo:
        uq.quantify(
            _linear_model,
            _uniform(),
            observations={"y": np.array([1.0]), "w": np.array([2.0])},
        )
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_non_finite_bootstrap_statistics_are_an_error() -> None:
    uq = _uq()
    samples = {"y": np.array([1.0, np.nan, 2.0, 3.0])}

    with pytest.raises(ValueError, match="non-finite"):
        uq._compute_bootstrap_intervals({"y": 2.0}, samples, ["a"], computation_time=0.0)


# ---------------------------------------------------------------------------
# Interval methods.
# ---------------------------------------------------------------------------


def test_bca_failure_propagates_instead_of_a_percentile_interval(monkeypatch) -> None:
    uq = _uq(ci_method="bca")

    def failing_ppf(_value):
        raise ValueError("ppf unavailable")

    monkeypatch.setattr(bootstrap_module.stats.norm, "ppf", failing_ppf)

    with pytest.raises(ValueError, match="ppf unavailable"):
        uq._bca_ci(np.linspace(0.0, 1.0, 21), 0.5, 0.95)


@pytest.mark.parametrize("original", [-1.0, 2.0])
def test_bca_with_the_original_outside_the_bootstrap_sample_is_an_error(original: float) -> None:
    """z0 = +-infinity: the bias-corrected interval is undefined, not clipped."""
    uq = _uq(ci_method="bca")

    with pytest.raises(ValueError, match="bias correction"):
        uq._bca_ci(np.linspace(0.0, 1.0, 21), original, 0.95)


def test_bca_without_an_original_statistic_is_an_error() -> None:
    uq = _uq(ci_method="bca")

    with pytest.raises(KeyError, match="original statistic"):
        uq._compute_bootstrap_intervals({}, {"y": np.linspace(0.0, 1.0, 21)}, ["a"], computation_time=0.0)


def test_bca_equals_the_percentile_interval_for_an_unbiased_sample() -> None:
    """z0 = 0 when half the statistics lie below theta_hat: BC = percentile."""
    uq = _uq(ci_method="bca")
    sample = np.linspace(0.0, 1.0, 20)

    assert uq._bca_ci(sample, 0.5, 0.9) == pytest.approx(uq._percentile_ci(sample, 0.9))


def test_bca_matches_the_bias_corrected_formula() -> None:
    uq = _uq(ci_method="bca")
    sample = np.linspace(0.0, 1.0, 41)
    original = 0.3

    z0 = stats.norm.ppf(np.sum(sample < original) / sample.size)
    lower_q = stats.norm.cdf(2.0 * z0 + stats.norm.ppf(0.05))
    upper_q = stats.norm.cdf(2.0 * z0 + stats.norm.ppf(0.95))
    expected = (np.percentile(sample, 100 * lower_q), np.percentile(sample, 100 * upper_q))

    assert uq._bca_ci(sample, original, 0.9) == pytest.approx(expected)


def test_quantify_reports_the_declared_interval_method() -> None:
    """The reported interval is the BC interval about the original statistic."""
    uq = _uq(ci_method="bca", n_bootstrap=16, random_seed=11)
    result = uq.quantify(_linear_model, _uniform())

    original = float(np.mean(uq.original_data["outputs"]["y"]))
    expected = uq._bca_ci(uq.bootstrap_samples["y"], original, 0.95)
    assert result.confidence_intervals["y"][0.95] == pytest.approx(expected)



# ---------------------------------------------------------------------------
# Independent replicate streams (MA-21 C7 round 2).
# The original sample and every replicate draw from distinct children of one
# numpy SeedSequence; replicate 0 used the original seed and reproduced the
# original sample exactly. The global numpy generator is not reseeded.
# ---------------------------------------------------------------------------


def test_first_replicate_is_not_a_copy_of_the_original_sample() -> None:
    uq = _uq(n_bootstrap=3)
    seen = []

    def model(parameters):
        seen.append(float(parameters["a"]))
        return {"y": float(parameters["a"])}

    uq.quantify(model, _uniform())
    original_size = len(seen) // 4
    original = seen[:original_size]
    first_replicate = seen[original_size:2 * original_size]
    assert original != first_replicate


def test_bootstrap_leaves_the_numpy_global_generator_untouched() -> None:
    state = np.random.get_state()
    _uq(n_bootstrap=3).quantify(_linear_model, _uniform())
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_seeded_bootstrap_is_reproducible() -> None:
    first = _uq(n_bootstrap=4).quantify(_linear_model, _uniform())
    second = _uq(n_bootstrap=4).quantify(_linear_model, _uniform())
    assert first.confidence_intervals == second.confidence_intervals


# ---------------------------------------------------------------------------
# The parametric sample size m = max(m_floor, B // d) comes from the parameter
# database (optimization.uncertainty.bootstrap.*); the original sample and each
# of the B replicates have m members (Efron and Tibshirani 1993, ch. 6: a
# bootstrap sample has the size of the original sample).
# ---------------------------------------------------------------------------


def _model_evaluations(uq: BootstrapUQ) -> int:
    seen = []

    def model(parameters):
        seen.append(float(parameters["a"]))
        return {"y": float(parameters["a"])}

    uq.quantify(model, _uniform())
    return len(seen)


def test_parametric_sample_size_is_read_from_the_parameter_database() -> None:
    from relic.parameters import get_global_manager

    floor_key = "optimization.uncertainty.bootstrap.parametric_sample_size_floor"
    divisor_key = "optimization.uncertainty.bootstrap.parametric_sample_size_replicate_divisor"
    manager = get_global_manager()
    originals = {key: manager.get(key, required=True) for key in (floor_key, divisor_key)}
    assert {key: int(value) for key, value in originals.items()} == {floor_key: 100, divisor_key: 10}
    # B = 3: m = max(100, 3 // 10) = 100 and (B + 1) m model evaluations.
    assert _model_evaluations(_uq(n_bootstrap=3)) == 4 * 100
    try:
        manager.set_value(floor_key, 5)
        assert _model_evaluations(_uq(n_bootstrap=3)) == 4 * 5
        manager.set_value(divisor_key, 1)
        # m = max(5, 8 // 1) = 8.
        assert _model_evaluations(_uq(n_bootstrap=8)) == 9 * 8
    finally:
        for key, value in originals.items():
            manager.set_value(key, value)
    assert {key: manager.get(key, required=True) for key in originals} == originals


# ---------------------------------------------------------------------------
# Round 5: the caller chooses the bootstrap type and interval method; the
# number of replicates comes from optimization.uncertainty.bootstrap.n_bootstrap.
# ---------------------------------------------------------------------------


def test_bootstrap_config_fixes_its_method_and_needs_its_choices() -> None:
    from relic.optimization.uncertainty.base import UQMethod

    with pytest.raises(TypeError):
        BootstrapConfig(ci_method="percentile")
    with pytest.raises(TypeError):
        BootstrapConfig(bootstrap_type="parametric")
    with pytest.raises(TypeError):
        BootstrapConfig(bootstrap_type="parametric", ci_method="percentile", method=UQMethod.ENSEMBLE)
    config = BootstrapConfig(bootstrap_type="parametric", ci_method="percentile")
    assert config.method is UQMethod.BOOTSTRAP
    assert config.n_bootstrap == 1000 and config.n_samples == 1000


def test_bootstrap_quantifier_needs_an_explicit_config() -> None:
    with pytest.raises(TypeError, match="BootstrapConfig"):
        BootstrapUQ(_DummyParameterManager())


def test_residual_bootstrap_needs_a_declared_parameter_mean() -> None:
    uq = _uq(bootstrap_type="residual")
    parameters = {
        "a": UncertainParameter(
            name="a", distribution=DistributionType.BETA, alpha=2.0, beta=3.0,
            lower_bound=0.0, upper_bound=1.0,
        )
    }
    with pytest.raises(RuntimeError, match="declared mean") as excinfo:
        uq.quantify(_linear_model, parameters, observations={"y": np.array([1.0])})
    assert isinstance(excinfo.value.__cause__, ValueError)
