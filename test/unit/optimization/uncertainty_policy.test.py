import numpy as np
import pytest
import torch

from relic.optimization.uncertainty import ensemble as ensemble_module
from relic.optimization.uncertainty.base import (
    DistributionType,
    UQConfig,
    UQMethod,
    UncertainParameter,
    UncertaintyQuantifier,
)
from relic.optimization.uncertainty.ensemble import EnsembleConfig, EnsembleUQ
from relic.stochastic.ensemble_runner import (
    CallableContract,
    ExecutionMode,
    StatisticType,
)


class _DummyParameterManager:
    def exists(self, name: str) -> bool:
        return True


class _DummyUQ(UncertaintyQuantifier):
    def quantify(self, model_runner, uncertain_parameters, baseline_parameters=None, observations=None):
        raise NotImplementedError("Not used in policy tests")


def _make_quantifier(config: UQConfig) -> _DummyUQ:
    return _DummyUQ(parameter_manager=_DummyParameterManager(), config=config)


def _uniform_parameter(name: str) -> UncertainParameter:
    return UncertainParameter(
        name=name,
        distribution=DistributionType.UNIFORM,
        lower_bound=0.0,
        upper_bound=1.0,
    )


def _make_ensemble_uq(**config_overrides) -> EnsembleUQ:
    config_values = {
        "ensemble_size": 3,
        "sampling_strategy": "lhs",
        "convergence_check": False,
        "outlier_detection": False,
    }
    config_values.update(config_overrides)
    config = EnsembleConfig(**config_values)
    return EnsembleUQ(parameter_manager=_DummyParameterManager(), config=config)


@pytest.mark.parametrize("capability", ["localization", "adaptive_sampling"])
def test_ensemble_config_fails_closed_for_nonoperational_capabilities(capability):
    with pytest.raises(NotImplementedError, match=r"\[N\]"):
        EnsembleConfig(sampling_strategy="lhs", **{capability: True})


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_ensemble_config_rejects_invalid_local_rng_seed(seed):
    with pytest.raises(ValueError, match="random_seed"):
        EnsembleConfig(sampling_strategy="lhs", random_seed=seed)


def test_ensemble_config_rejects_free_string_callable_contract():
    with pytest.raises(TypeError, match="parallel_callable_contract"):
        EnsembleConfig(sampling_strategy="lhs", parallel_callable_contract="deterministic_pure")


def test_ensemble_outlier_detection_is_explicit_opt_in():
    assert EnsembleConfig(sampling_strategy="lhs").outlier_detection is False


def test_uq_raises_on_failed_sample_by_default():
    config = UQConfig(method=UQMethod.ENSEMBLE, n_samples=1)
    uq = _make_quantifier(config)
    samples = {"a": np.array([1.0])}

    def model_runner(_params):
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        uq.evaluate_model_ensemble(model_runner, samples)


def test_uq_allows_failed_sample_when_enabled():
    config = UQConfig(method=UQMethod.ENSEMBLE, n_samples=1, allow_failed_samples=True, failed_sample_value=-1.0)
    uq = _make_quantifier(config)
    samples = {"a": np.array([1.0])}

    def model_runner(_params):
        raise RuntimeError("fail")

    outputs = uq.evaluate_model_ensemble(model_runner, samples)
    assert outputs["output"][0] == -1.0


def test_uq_missing_outputs_raises_by_default():
    config = UQConfig(method=UQMethod.ENSEMBLE, n_samples=2)
    uq = _make_quantifier(config)
    samples = {"a": np.array([1.0, 2.0])}

    def model_runner(_params):
        if float(_params["a"]) < 1.5:
            return {"y": 1.0}
        return {}

    with pytest.raises(RuntimeError):
        uq.evaluate_model_ensemble(model_runner, samples)


def test_uq_missing_outputs_allowed_when_enabled():
    config = UQConfig(method=UQMethod.ENSEMBLE, n_samples=2, allow_missing_outputs=True, missing_output_value=2.5)
    uq = _make_quantifier(config)
    samples = {"a": np.array([1.0, 2.0])}

    def model_runner(_params):
        if float(_params["a"]) < 1.5:
            return {"y": 1.0}
        return {}

    outputs = uq.evaluate_model_ensemble(model_runner, samples)
    assert outputs["y"][0] == 1.0
    assert outputs["y"][1] == 2.5


def test_ensemble_parallel_bridge_uses_current_runner_contract(monkeypatch):
    captured = {}

    class _FakeRunner:
        def __init__(self, config, device, parameter_manager):
            captured["config"] = config
            captured["device"] = device
            captured["parameter_manager"] = parameter_manager

        def run_ensemble(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return type(
                "_Result",
                (),
                {"samples": {"y": torch.tensor([11.0, 12.0, 13.0])}},
            )()

    monkeypatch.setattr(ensemble_module, "StochasticEnsembleRunner", _FakeRunner)
    uq = _make_ensemble_uq(
        parallel_evaluation=True,
        parallel_callable_contract=CallableContract.DETERMINISTIC_PURE,
        n_workers=8,
    )

    outputs = uq._evaluate_ensemble(
        lambda inputs: {"y": inputs["a"] + inputs["b"]},
        {"a": np.array([1.0, 2.0, 3.0])},
        {"a": 999.0, "b": 10.0},
    )

    assert captured["config"].mode is ExecutionMode.PARALLEL_THREAD
    assert (
        captured["config"].callable_contract
        is CallableContract.DETERMINISTIC_PURE
    )
    assert captured["config"].outlier_detection is False
    assert captured["config"].convergence_check is False
    assert captured["device"] == uq.device
    assert captured["parameter_manager"] is uq.parameter_manager
    run_kwargs = captured["run_kwargs"]
    assert set(run_kwargs) == {
        "model_callable",
        "parameter_samples",
        "fixed_inputs",
        "statistics",
        "store_samples",
        "seed",
    }
    assert isinstance(run_kwargs["parameter_samples"]["a"], torch.Tensor)
    assert run_kwargs["parameter_samples"]["a"].shape == (3,)
    assert run_kwargs["fixed_inputs"] == {"b": 10.0}
    assert run_kwargs["statistics"] == [StatisticType.MEAN]
    assert run_kwargs["store_samples"] is True
    assert run_kwargs["seed"] == uq.ensemble_config.random_seed
    np.testing.assert_array_equal(outputs["y"], np.array([11.0, 12.0, 13.0]))
    assert uq.model_evaluations == 3


def test_ensemble_parallel_bridge_preserves_member_order_with_real_runner():
    uq = _make_ensemble_uq(
        parallel_evaluation=True,
        parallel_callable_contract=CallableContract.DETERMINISTIC_PURE,
        n_workers=2,
    )

    outputs = uq._evaluate_ensemble(
        lambda inputs: {"y": float(inputs["a"]) + float(inputs["b"])},
        {"a": np.array([1.0, 2.0, 3.0])},
        {"b": 10.0},
    )

    np.testing.assert_allclose(outputs["y"], np.array([11.0, 12.0, 13.0]))
    assert uq.model_evaluations == 3


def test_ensemble_parallel_failure_never_replays_sequentially(monkeypatch):
    uq = _make_ensemble_uq(
        parallel_evaluation=True,
        parallel_callable_contract=CallableContract.DETERMINISTIC_PURE,
        n_workers=1,
    )
    calls = {1.0: 0, 2.0: 0, 3.0: 0}

    def model_runner(inputs):
        value = float(inputs["a"])
        calls[value] += 1
        if value == 2.0:
            raise RuntimeError("member failure")
        return {"y": torch.as_tensor(value)}

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("sequential replay is forbidden")

    monkeypatch.setattr(uq, "evaluate_model_ensemble", forbidden_fallback)

    with pytest.raises(RuntimeError, match=r"member 1 .*failed"):
        uq._evaluate_ensemble(
            model_runner,
            {"a": np.array([1.0, 2.0, 3.0])},
        )

    assert all(count <= 1 for count in calls.values())
    assert uq.model_evaluations == 0


@pytest.mark.parametrize("policy_name", ["allow_failed_samples", "allow_missing_outputs"])
def test_ensemble_parallel_rejects_substitution_policies_before_execution(policy_name):
    uq = _make_ensemble_uq(
        parallel_evaluation=True,
        parallel_callable_contract=CallableContract.DETERMINISTIC_PURE,
        **{policy_name: True},
    )
    model_calls = 0

    def model_runner(_inputs):
        nonlocal model_calls
        model_calls += 1
        return {"y": torch.tensor(1.0)}

    with pytest.raises(ValueError, match="sequential"):
        uq._evaluate_ensemble(
            model_runner,
            {"a": np.array([1.0, 2.0, 3.0])},
        )

    assert model_calls == 0


def test_ensemble_parallel_rejects_non_scalar_member_outputs(monkeypatch):
    class _FakeRunner:
        def __init__(self, config, device, parameter_manager):
            pass

        def run_ensemble(self, **_kwargs):
            return type(
                "_Result",
                (),
                {"samples": {"y": torch.ones((3, 1))}},
            )()

    monkeypatch.setattr(ensemble_module, "StochasticEnsembleRunner", _FakeRunner)
    uq = _make_ensemble_uq(
        parallel_evaluation=True,
        parallel_callable_contract=CallableContract.DETERMINISTIC_PURE,
    )

    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        uq._evaluate_ensemble(
            lambda _inputs: {"y": torch.tensor(1.0)},
            {"a": np.array([1.0, 2.0, 3.0])},
        )


@pytest.mark.parametrize(
    "samples, error_match",
    [
        (
            {"a": np.array([1.0, 2.0]), "b": np.array([1.0])},
            "same ensemble size",
        ),
        ({"a": np.array([1.0, np.nan, 3.0])}, "finite"),
        ({}, "non-empty"),
    ],
)
def test_ensemble_parallel_validates_sample_ledger_before_execution(samples, error_match):
    uq = _make_ensemble_uq(
        parallel_evaluation=True,
        parallel_callable_contract=CallableContract.DETERMINISTIC_PURE,
    )
    model_calls = 0

    def model_runner(_inputs):
        nonlocal model_calls
        model_calls += 1
        return {"y": torch.tensor(1.0)}

    with pytest.raises(ValueError, match=error_match):
        uq._evaluate_ensemble(model_runner, samples)

    assert model_calls == 0


def test_ensemble_random_sampling_is_reproducible_independent_and_rng_local():
    parameters = {
        "a": _uniform_parameter("a"),
        "b": _uniform_parameter("b"),
    }
    config = EnsembleConfig(
        ensemble_size=32,
        sampling_strategy="random",
        random_seed=731,
        convergence_check=False,
        outlier_detection=False,
    )
    first = EnsembleUQ(_DummyParameterManager(), config)
    second = EnsembleUQ(_DummyParameterManager(), config)

    np.random.seed(991)
    state_before = np.random.get_state()
    first_samples = first._generate_parameter_ensemble(parameters)
    state_after = np.random.get_state()
    second_samples = second._generate_parameter_ensemble(parameters)

    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    np.testing.assert_array_equal(first_samples["a"], second_samples["a"])
    np.testing.assert_array_equal(first_samples["b"], second_samples["b"])
    assert not np.array_equal(first_samples["a"], first_samples["b"])


def test_ensemble_sobol_requires_power_of_two_members():
    uq = _make_ensemble_uq(sampling_strategy="sobol")

    with pytest.raises(ValueError, match="power of two"):
        uq._generate_parameter_ensemble({"a": _uniform_parameter("a")})


@pytest.mark.parametrize("strategy", ["random", "lhs", "sobol"])
def test_ensemble_distribution_parameters_must_be_explicit(strategy):
    ensemble_size = 4 if strategy == "sobol" else 3
    uq = EnsembleUQ(
        _DummyParameterManager(),
        EnsembleConfig(
            ensemble_size=ensemble_size,
            sampling_strategy=strategy,
            convergence_check=False,
            outlier_detection=False,
        ),
    )
    ambiguous_beta = UncertainParameter(
        name="beta_parameter",
        distribution=DistributionType.BETA,
        mean=0.5,
        std=0.1,
        lower_bound=0.0,
        upper_bound=1.0,
    )

    with pytest.raises(ValueError, match="alpha and beta"):
        uq._generate_parameter_ensemble({"beta_parameter": ambiguous_beta})


@pytest.mark.parametrize(
    ("strategy", "ensemble_size"),
    [("lhs", 3), ("sobol", 4)],
)
def test_ensemble_qmc_sampling_does_not_mutate_numpy_global_rng(
    strategy,
    ensemble_size,
):
    uq = EnsembleUQ(
        _DummyParameterManager(),
        EnsembleConfig(
            ensemble_size=ensemble_size,
            sampling_strategy=strategy,
            random_seed=431,
            convergence_check=False,
        ),
    )
    np.random.seed(877)
    state_before = np.random.get_state()

    uq._generate_parameter_ensemble({"a": _uniform_parameter("a")})

    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_ensemble_parameter_ledger_requires_key_name_identity():
    uq = _make_ensemble_uq()

    with pytest.raises(ValueError, match="ledger key"):
        uq._generate_parameter_ensemble({"declared": _uniform_parameter("other")})


def test_ensemble_parallel_requires_explicit_deterministic_pure_contract():
    uq = _make_ensemble_uq(parallel_evaluation=True)
    model_calls = 0

    def model_runner(_inputs):
        nonlocal model_calls
        model_calls += 1
        return {"y": torch.tensor(1.0)}

    with pytest.raises(NotImplementedError, match=r"\[N\].*deterministic_pure"):
        uq._evaluate_ensemble(model_runner, {"a": np.array([1.0])})

    assert model_calls == 0


def test_ensemble_sequential_path_uses_canonical_stochastic_runner(monkeypatch):
    captured = {}

    class _FakeRunner:
        def __init__(self, config, device, parameter_manager):
            captured["mode"] = config.mode

        def run_ensemble(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return type(
                "_Result",
                (),
                {"samples": {"y": torch.tensor([2.0, 3.0, 4.0])}},
            )()

    monkeypatch.setattr(ensemble_module, "StochasticEnsembleRunner", _FakeRunner)
    uq = _make_ensemble_uq(parallel_evaluation=False)

    outputs = uq._evaluate_ensemble(
        lambda inputs: {"y": inputs["a"] + 1.0},
        {"a": np.array([1.0, 2.0, 3.0])},
    )

    assert captured["mode"] is ExecutionMode.SEQUENTIAL
    assert captured["run_kwargs"]["store_samples"] is True
    np.testing.assert_array_equal(outputs["y"], np.array([2.0, 3.0, 4.0]))


def test_ensemble_runner_bridge_preserves_executed_parameter_dtype_and_value():
    exact_float = np.float64(1.0000000000000002)
    exact_integer = np.int64(16_777_217)
    observed = {}
    uq = _make_ensemble_uq(
        ensemble_size=1,
        parallel_evaluation=True,
        parallel_callable_contract=CallableContract.DETERMINISTIC_PURE,
        n_workers=1,
    )

    def model_runner(inputs):
        observed.update(inputs)
        return {"y": np.float64(inputs["x"]) + np.float64(inputs["count"])}

    uq._evaluate_ensemble(
        model_runner,
        {
            "x": np.array([exact_float], dtype=np.float64),
            "count": np.array([exact_integer], dtype=np.int64),
        },
    )

    assert observed["x"] == exact_float
    assert observed["count"] == exact_integer


def test_ensemble_quantify_reports_retained_members_and_attempted_evaluations(
    monkeypatch,
):
    uq = _make_ensemble_uq(
        outlier_detection=True,
        outlier_threshold=1.0,
        compute_moments=False,
        compute_correlations=False,
    )
    monkeypatch.setattr(
        uq,
        "_generate_parameter_ensemble",
        lambda _parameters: {"a": np.array([0.0, 1.0, 2.0])},
    )

    result = uq.quantify(
        lambda inputs: {
            "y": 100.0 if float(inputs["a"]) == 2.0 else 0.0,
        },
        {"a": _uniform_parameter("a")},
    )

    assert result.n_samples == 2
    assert result.n_model_evaluations == 3
    assert len(result.output_samples["y"]) == 2
    assert len(result.parameter_samples["a"]) == 2


def test_ensemble_convergence_is_not_vacuously_true_below_evidence_floor():
    uq = _make_ensemble_uq()

    assert uq._check_ensemble_convergence({"y": np.ones(99)}) is False


def test_ensemble_convergence_evidence_floor_is_read_from_the_parameter_database():
    """The minimum member count is optimization.uncertainty.ensemble.convergence_min_members."""
    from relic.parameters import get_global_manager

    key = "optimization.uncertainty.ensemble.convergence_min_members"
    manager = get_global_manager()
    original = manager.get(key, required=True)
    assert int(original) == 100
    uq = _make_ensemble_uq()
    assert uq._check_ensemble_convergence({"y": np.ones(60)}) is False
    try:
        manager.set_value(key, 50)
        assert uq._check_ensemble_convergence({"y": np.ones(60)}) is True
        assert uq._check_ensemble_convergence({"y": np.ones(49)}) is False
    finally:
        manager.set_value(key, original)
    assert manager.get(key, required=True) == original


def test_ensemble_convergence_window_constants_are_read_from_the_parameter_database():
    """First monitored prefix N // d_start and stride max(1, (N // d_start) // d_stride) come from YAML.

    Member 1 is 1 + c and the others are 1, so the prefix mean is m(n) = 1 + c / n
    and the relative drift between prefixes n and n + s is c s / ((n + s)(n + c)).
    With N = 100 and c = 30: d_start = 4, d_stride = 10 (n = 25, s = 2) gives
    0.040 <= 0.05; d_start = 10 (n = 10, s = 1) gives 0.068; d_stride = 1
    (n = 25, s = 25) gives 0.27.
    """
    from relic.parameters import get_global_manager

    start_key = "optimization.uncertainty.ensemble.convergence_first_prefix_divisor"
    stride_key = "optimization.uncertainty.ensemble.convergence_stride_divisor"
    manager = get_global_manager()
    originals = {key: manager.get(key, required=True) for key in (start_key, stride_key)}
    assert {key: int(value) for key, value in originals.items()} == {start_key: 4, stride_key: 10}
    values = np.ones(100)
    values[0] = 31.0
    uq = _make_ensemble_uq()
    assert uq._check_ensemble_convergence({"y": values}) is True
    try:
        manager.set_value(start_key, 10)
        assert uq._check_ensemble_convergence({"y": values}) is False
        manager.set_value(start_key, originals[start_key])
        manager.set_value(stride_key, 1)
        assert uq._check_ensemble_convergence({"y": values}) is False
    finally:
        for key, value in originals.items():
            manager.set_value(key, value)
    assert {key: manager.get(key, required=True) for key in originals} == originals


def test_ensemble_outlier_filter_preserves_member_alignment():
    uq = _make_ensemble_uq(outlier_detection=True, outlier_threshold=1.0)

    filtered, mask = uq._filter_outliers(
        {
            "y": np.array([0.0, 0.0, 100.0]),
            "z": np.array([10.0, 20.0, 30.0]),
        }
    )

    np.testing.assert_array_equal(mask, np.array([False, False, True]))
    np.testing.assert_array_equal(filtered["y"], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(filtered["z"], np.array([10.0, 20.0]))


def test_ensemble_outlier_filter_rejects_empty_survivor_ledger():
    uq = _make_ensemble_uq(outlier_detection=True, outlier_threshold=0.1)

    with pytest.raises(RuntimeError, match="every ensemble member"):
        uq._filter_outliers({"y": np.array([0.0, 1.0])})


# ---------------------------------------------------------------------------
# relic/optimization/uncertainty/base.py: no fabricated statistics.
# ---------------------------------------------------------------------------


def test_missing_credible_interval_is_an_error_not_zero_width():
    from relic.optimization.uncertainty.base import UQResult

    result = UQResult(
        method="probe", parameter_names=["a"], output_variables=["y"],
        n_samples=0, n_model_evaluations=0, computation_time=0.0,
    )

    with pytest.raises(KeyError, match="y"):
        result.get_credible_interval("y", 0.95)
    with pytest.raises(KeyError, match="y"):
        result.get_prediction_bounds("y", 0.95)


@pytest.mark.parametrize(
    "distribution, extra",
    [
        (DistributionType.GAMMA, {"alpha": 2.0, "beta": 1.0, "lower_bound": 0.0}),
        (DistributionType.LOGNORMAL, {"mean": 1.0, "std": 0.5, "lower_bound": 0.0}),
    ],
)
def test_unimplemented_pdf_is_an_error_not_a_constant_density(distribution, extra):
    parameter = UncertainParameter(name="p", distribution=distribution, **extra)

    with pytest.raises(NotImplementedError, match="pdf"):
        parameter.pdf(np.array([0.5, 1.0]))


def test_triangular_parameter_requires_an_explicit_mode():
    with pytest.raises(ValueError, match="mode"):
        UncertainParameter(
            name="t",
            distribution=DistributionType.TRIANGULAR,
            lower_bound=0.0,
            upper_bound=1.0,
        )


def test_beta_parameter_requires_explicit_shapes_and_finite_bounds():
    implicit = UncertainParameter(
        name="b",
        distribution=DistributionType.BETA,
        mean=5.0,
        std=1.0,
        lower_bound=0.0,
        upper_bound=10.0,
    )
    with pytest.raises(ValueError, match="alpha and beta"):
        implicit.sample(4, random_state=1)

    unbounded = UncertainParameter(
        name="b", distribution=DistributionType.BETA, alpha=2.0, beta=3.0
    )
    with pytest.raises(ValueError, match="finite bounds"):
        unbounded.sample(4, random_state=1)


def test_statistics_reject_non_finite_outputs_without_a_failure_policy():
    uq = _make_quantifier(UQConfig(method=UQMethod.ENSEMBLE, compute_correlations=False))

    with pytest.raises(ValueError, match="non-finite"):
        uq.compute_statistics({"y": np.array([1.0, np.nan, 3.0])}, {"a": np.arange(3.0)}, computation_time=0.0)


def test_statistics_reject_a_variable_without_any_finite_sample():
    uq = _make_quantifier(UQConfig(method=UQMethod.ENSEMBLE, allow_failed_samples=True, compute_correlations=False))

    with pytest.raises(ValueError, match="no finite"):
        uq.compute_statistics({"y": np.array([np.nan, np.nan])}, {"a": np.arange(2.0)}, computation_time=0.0)


def test_statistics_drop_declared_failed_samples_only_under_the_policy():
    uq = _make_quantifier(UQConfig(method=UQMethod.ENSEMBLE, allow_failed_samples=True, compute_correlations=False))

    result = uq.compute_statistics({"y": np.array([1.0, np.nan, 3.0])}, {"a": np.arange(3.0)}, computation_time=0.0)

    assert result.output_mean["y"] == pytest.approx(2.0)


def test_undefined_correlation_is_reported_as_nan_not_zero():
    uq = _make_quantifier(UQConfig(method=UQMethod.ENSEMBLE))

    matrix = uq._compute_correlations({"a": np.ones(5), "b": np.arange(5.0)})
    cross = uq._compute_cross_correlations({"a": np.ones(5)}, {"y": np.arange(5.0)})

    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[1, 0])
    assert np.isnan(cross["a"]["y"])



# ---------------------------------------------------------------------------
# Bounded sampling (MA-21 C7 round 2).
#
# References: Devroye (1986), Non-Uniform Random Variate Generation, Sect. II.2
# (inversion of a truncated distribution); Adams et al., Dakota User's Manual
# (bounded normal and bounded lognormal are truncated distributions).
# Mathematical expression: x = F^-1(F(a) + u (F(b) - F(a))), u in [0, 1].
# Parameter meanings: F is the declared CDF, [a, b] the declared bounds.
#
# Bounds truncate the declared distribution. Clamping piled the probability
# outside [a, b] onto the bounds themselves (a censored law that was not
# declared); inflation that leaves the bounds is an error, and one sampler
# owns every distribution.
# ---------------------------------------------------------------------------


def _bounded_normal():
    return UncertainParameter(
        name="n",
        distribution=DistributionType.NORMAL,
        mean=0.0,
        std=1.0,
        lower_bound=-0.1,
        upper_bound=0.1,
    )


def test_bounded_normal_samples_are_truncated_not_clamped():
    samples = np.asarray(_bounded_normal().sample(400, random_state=4), dtype=float)
    assert samples.min() >= -0.1 and samples.max() <= 0.1
    assert not np.any(np.isclose(np.abs(samples), 0.1, rtol=0.0, atol=1e-12))
    # Within +-0.1 sigma the truncated normal is almost uniform.
    assert abs(float(np.mean(samples < 0.0)) - 0.5) < 0.1


def test_ensemble_bounded_normal_is_truncated_not_clamped():
    uq = _make_ensemble_uq(ensemble_size=64, sampling_strategy="lhs", random_seed=5)
    samples = uq._generate_parameter_ensemble({"n": _bounded_normal()})["n"]
    assert not np.any(np.isclose(np.abs(samples), 0.1, rtol=0.0, atol=1e-12))


def test_gamma_parameter_requires_explicit_shape_and_scale():
    implicit = UncertainParameter(
        name="g", distribution=DistributionType.GAMMA, mean=2.0, std=1.0, lower_bound=0.0
    )
    with pytest.raises(ValueError, match="alpha and beta"):
        implicit.sample(4, random_state=1)
    assert implicit.alpha is None and implicit.beta is None


def test_parameter_sampling_leaves_the_numpy_global_generator_untouched():
    state = np.random.get_state()
    _uniform_parameter("a").sample(5, random_state=3)
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_parameters_of_one_run_draw_independent_streams():
    uq = _make_quantifier(UQConfig(method=UQMethod.ENSEMBLE, random_seed=9, compute_correlations=False))
    samples = uq.generate_parameter_samples(
        {"a": _uniform_parameter("a"), "b": _uniform_parameter("b")}, 16
    )
    assert not np.allclose(samples["a"], samples["b"])


def test_inflation_that_leaves_the_bounds_is_an_error():
    uq = _make_ensemble_uq(ensemble_size=8, ensemble_inflation=3.0, random_seed=2)
    with pytest.raises(ValueError, match="inflation"):
        uq._generate_parameter_ensemble({"a": _uniform_parameter("a")})


def test_convergence_of_a_small_magnitude_output_is_not_granted_by_a_floor():
    """A 1e-6 denominator floor made every output with |mean| << 1e-6 look converged."""
    uq = _make_ensemble_uq(ensemble_size=200, convergence_tolerance=0.05)
    rng = np.random.default_rng(0)
    # Zero-mean flux of magnitude 1e-9: its running mean drifts by far more than
    # 5% relative, but the old floor divided the drift by 1e-6.
    outputs = {"flux": 1e-9 * rng.standard_normal(200)}
    assert uq._check_ensemble_convergence(outputs) is False


# ---------------------------------------------------------------------------
# Round 5: no literal defaults in the UQ configuration. The caller chooses the
# method (fixed by EnsembleConfig / BootstrapConfig) and the sampling strategy;
# sample sizes, tolerance, inflation and outlier threshold come from YAML; a
# normal-type parameter declares mean and std; results state their counters;
# declared parameter units are checked against the parameter database.
# ---------------------------------------------------------------------------


def test_uq_method_is_required_and_numeric_settings_come_from_yaml():
    from relic.parameters import get_global_manager

    with pytest.raises(TypeError):
        UQConfig()
    config = UQConfig(method=UQMethod.ENSEMBLE)
    assert (config.n_samples, config.n_bootstrap) == (1000, 1000)
    assert config.convergence_tolerance == pytest.approx(0.05)
    assert not hasattr(config, "max_iterations")
    with pytest.raises(TypeError):
        UQConfig(method=UQMethod.ENSEMBLE, max_iterations=10)
    manager = get_global_manager()
    key = "optimization.uncertainty.n_samples"
    original = manager.get(key, required=True)
    try:
        manager.set_value(key, 64)
        assert UQConfig(method=UQMethod.ENSEMBLE).n_samples == 64
    finally:
        manager.set_value(key, original)


def test_ensemble_config_fixes_its_method_and_reads_its_settings_from_yaml():
    from relic.parameters import get_global_manager

    with pytest.raises(TypeError):
        EnsembleConfig()
    with pytest.raises(TypeError):
        EnsembleConfig(sampling_strategy="lhs", method=UQMethod.BOOTSTRAP)
    config = EnsembleConfig(sampling_strategy="lhs")
    assert config.method is UQMethod.ENSEMBLE
    assert (config.ensemble_size, config.n_samples) == (1000, 1000)
    assert config.ensemble_inflation == pytest.approx(1.0)
    assert config.outlier_threshold == pytest.approx(3.0)
    manager = get_global_manager()
    key = "stochastic.ensemble.outlier_threshold"
    original = manager.get(key, required=True)
    try:
        manager.set_value(key, 2.5)
        assert EnsembleConfig(sampling_strategy="lhs").outlier_threshold == pytest.approx(2.5)
    finally:
        manager.set_value(key, original)


def test_ensemble_quantifier_needs_an_explicit_config():
    with pytest.raises(TypeError, match="EnsembleConfig"):
        EnsembleUQ(_DummyParameterManager())


@pytest.mark.parametrize("distribution", [DistributionType.NORMAL, DistributionType.LOGNORMAL])
def test_normal_type_parameters_declare_mean_and_std(distribution):
    with pytest.raises(ValueError, match="mean and std"):
        UncertainParameter(name="p", distribution=distribution, lower_bound=0.1, upper_bound=2.0)
    with pytest.raises(ValueError, match="mean and std"):
        UncertainParameter(name="p", distribution=distribution, mean=1.0, lower_bound=0.1, upper_bound=2.0)


def test_uniform_parameter_moments_follow_from_its_bounds():
    parameter = _uniform_parameter("a")
    assert parameter.mean == pytest.approx(0.5)
    assert parameter.std == pytest.approx(1.0 / np.sqrt(12.0))


def test_uq_result_counters_are_required():
    from relic.optimization.uncertainty.base import UQResult

    fields = dict(method="probe", parameter_names=["a"], output_variables=["y"])
    for missing in ("n_samples", "n_model_evaluations", "computation_time"):
        counters = {"n_samples": 1, "n_model_evaluations": 1, "computation_time": 0.0}
        counters.pop(missing)
        with pytest.raises(TypeError):
            UQResult(**fields, **counters)


def test_compute_statistics_needs_the_computation_time():
    uq = _make_quantifier(UQConfig(method=UQMethod.ENSEMBLE, compute_correlations=False))
    with pytest.raises(TypeError):
        uq.compute_statistics({"y": np.array([1.0, 2.0])}, {"a": np.arange(2.0)})
    result = uq.compute_statistics(
        {"y": np.array([1.0, 2.0])}, {"a": np.arange(2.0)}, computation_time=1.5
    )
    assert result.computation_time == 1.5


def test_declared_parameter_units_are_checked_against_the_parameter_spec():
    class _Spec:
        units = "umol m-2 s-1"

    class _Manager:
        def exists(self, name):
            return name == "vcmax25"

        def get_parameter(self, name):
            return _Spec()

    uq = _DummyUQ(parameter_manager=_Manager(), config=UQConfig(method=UQMethod.ENSEMBLE))

    def parameter(name, units):
        return UncertainParameter(
            name=name, distribution=DistributionType.UNIFORM, lower_bound=20.0, upper_bound=80.0, units=units
        )

    uq.validate_uncertain_parameters({"vcmax25": parameter("vcmax25", "umol m-2 s-1")})
    uq.validate_uncertain_parameters({"vcmax25": parameter("vcmax25", "")})
    with pytest.raises(ValueError, match="units"):
        uq.validate_uncertain_parameters({"vcmax25": parameter("vcmax25", "mol m-2 s-1")})
    with pytest.raises(ValueError, match="units"):
        uq.validate_uncertain_parameters({"other": parameter("other", "m")})


def test_interval_levels_and_percentiles_come_from_the_parameter_database():
    """Round 6: optimization.uncertainty.confidence_levels / percentiles are the single source."""
    from relic.parameters import get_global_manager

    config = UQConfig(method=UQMethod.ENSEMBLE)
    assert config.confidence_levels == pytest.approx([0.68, 0.95])
    assert config.percentiles == pytest.approx([5.0, 25.0, 50.0, 75.0, 95.0])
    manager = get_global_manager()
    key = "optimization.uncertainty.confidence_levels"
    original = manager.get(key, required=True)
    try:
        manager.set_value(key, [0.9])
        assert UQConfig(method=UQMethod.ENSEMBLE).confidence_levels == pytest.approx([0.9])
    finally:
        manager.set_value(key, original)
    stated = UQConfig(method=UQMethod.ENSEMBLE, percentiles=[10.0, 90.0])
    assert stated.percentiles == [10.0, 90.0]
