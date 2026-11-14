import numpy as np
import pytest

from relic.optimization.sensitivity.base import SensitivityConfig, SensitivityAnalyzer


class _DummySensitivity(SensitivityAnalyzer):
    def analyze(self, model_runner, parameter_ranges):
        raise NotImplementedError("Not used in policy tests")

class _DummyParameterManager:
    def exists(self, name: str) -> bool:
        return True


def _make_analyzer(config: SensitivityConfig) -> _DummySensitivity:
    return _DummySensitivity(parameter_manager=_DummyParameterManager(), config=config)


def test_sensitivity_raises_on_failed_sample():
    config = SensitivityConfig(method=SensitivityMethod.MORRIS, aggregation_method="mean")
    analyzer = _make_analyzer(config)
    samples = np.array([[1.0], [2.0]])

    def model_runner(_params):
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        analyzer.evaluate_model_batch(model_runner, samples, ["a"])



# ---------------------------------------------------------------------------
# No silent substitution in the sensitivity analyzers (MA-21 C7).
#
# References:
#     Morris (1991), Technometrics 33, 161-174; Campolongo et al. (2007),
#     Environ. Model. Softw. 22, 1509-1518 (elementary effects on a p-level grid);
#     Saltelli et al. (2010), Comput. Phys. Commun. 181, 259-270 and Jansen (1999),
#     Comput. Phys. Commun. 117, 35-43 (A/B/AB_i estimators).
# Mathematical expression:
#     EE_i = [f(x + Delta e_i) - f(x)] / Delta, Delta = grid_jump / (p - 1) of the range;
#     S_i = 1 - E[(f_B - f_AB_i)^2] / (2 V(Y)), ST_i = E[(f_A - f_AB_i)^2] / (2 V(Y)).
# Parameter meanings:
#     p: n_levels, grid_jump: step in grid levels, f_A, f_B, f_AB_i: model outputs on
#     the base matrices and on A with column i taken from B, V(Y): output variance.
#
# A declared option runs as declared or is refused; a missing, constant or
# non-finite model output raises instead of becoming an index of 0 or 1.
# ---------------------------------------------------------------------------

import math

from relic.optimization.sensitivity.base import SensitivityMethod
from relic.optimization.sensitivity import (
    MorrisAnalyzer,
    MorrisConfig,
    ParameterRange,
    SensitivityResult,
    SobolAnalyzer,
    SobolConfig,
)


def _uniform_ranges(*names):
    return {name: ParameterRange(name, 0.0, 1.0, distribution="uniform") for name in names}


def _morris(**config):
    values = {"n_trajectories": 12, "random_seed": 5, "aggregation_method": "mean"}
    values.update(config)
    return MorrisAnalyzer(_DummyParameterManager(), MorrisConfig(**values))


def _sobol(**config):
    values = {
        "n_samples": 64, "skip_values": 0, "convergence_check": False, "random_seed": 3,
        "aggregation_method": "mean", "estimator": "saltelli",
    }
    values.update(config)
    return SobolAnalyzer(_DummyParameterManager(), SobolConfig(**values))


def _linear(params):
    return {"y": 2.0 * params["x1"] + 3.0 * params["x2"]}


def _cause_types(exc):
    types = []
    while exc is not None:
        types.append(type(exc))
        exc = exc.__cause__
    return types


class TestSensitivityConfigContracts:
    def test_unknown_aggregation_method_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="aggregation_method"):
            SensitivityConfig(method=SensitivityMethod.MORRIS, aggregation_method="median")

    def test_unknown_aggregation_method_is_rejected_at_use(self):
        analyzer = _make_analyzer(SensitivityConfig(method=SensitivityMethod.MORRIS, aggregation_method="mean"))
        with pytest.raises(ValueError, match="median"):
            analyzer.aggregate_outputs(np.ones((2, 3)), "median")

    @pytest.mark.parametrize(
        "field_name, value",
        [
            ("temporal_aggregation", "final"),
            ("parallel_execution", True),
            ("bootstrap_confidence", True),
            ("confidence_level", 0.9),
        ],
    )
    def test_settings_without_an_implementation_are_not_offered(self, field_name, value):
        with pytest.raises(TypeError):
            SensitivityConfig(method=SensitivityMethod.MORRIS, aggregation_method="mean", **{field_name: value})

    @pytest.mark.parametrize(
        "field_name",
        ["optimal_trajectories", "local_optimization", "morris_mu_star", "interaction_analysis"],
    )
    def test_morris_settings_without_an_implementation_are_not_offered(self, field_name):
        with pytest.raises(TypeError):
            MorrisConfig(aggregation_method="mean", **{field_name: True})

    def test_placeholder_bootstrap_interval_is_not_offered(self):
        assert not hasattr(SensitivityAnalyzer, "compute_bootstrap_confidence")


class TestSensitivityResultQueries:
    def _result(self):
        return SensitivityResult(
            method="Morris",
            parameter_names=["a", "b"],
            output_variables=["y"],
            first_order_indices={"y": {"a": 0.4, "b": 0.1}},
            n_model_evaluations=6,
            computation_time=0.0,
        )

    def test_unknown_index_type_is_rejected(self):
        with pytest.raises(ValueError, match="index_type"):
            self._result().get_most_sensitive_parameters("y", index_type="second")

    def test_missing_index_type_raises_instead_of_an_empty_ranking(self):
        with pytest.raises(KeyError, match="total"):
            self._result().get_most_sensitive_parameters("y", index_type="total")

    def test_interaction_strength_without_total_indices_raises(self):
        with pytest.raises(KeyError, match="total-order"):
            self._result().get_interaction_strength("y")

    def test_interaction_strength_is_undefined_for_zero_total_variance_share(self):
        result = self._result()
        result.first_order_indices = {"y": {"a": 0.0, "b": 0.0}}
        result.total_order_indices = {"y": {"a": 0.0, "b": 0.0}}
        assert math.isnan(result.get_interaction_strength("y"))


class TestParameterRangeContracts:
    @pytest.mark.parametrize("distribution", ["normal", "lognormal", "triangular"])
    def test_non_uniform_distribution_needs_its_own_parameters(self, distribution):
        with pytest.raises(ValueError, match=distribution):
            ParameterRange("a", 1.0, 2.0, distribution=distribution)

    def test_unknown_distribution_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="gamma"):
            ParameterRange("a", 1.0, 2.0, distribution="gamma")

    def test_sample_matrix_columns_are_independent_and_global_rng_is_untouched(self):
        analyzer = _make_analyzer(SensitivityConfig(method=SensitivityMethod.MORRIS, aggregation_method="mean", random_seed=11))
        state = np.random.get_state()
        matrix = analyzer.create_sample_matrix(_uniform_ranges("a", "b"), 32)
        after = np.random.get_state()
        assert not np.allclose(matrix[:, 0], matrix[:, 1])
        assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]


class TestMorrisNoSilentSubstitution:
    def test_every_elementary_effect_of_a_linear_model_is_measured(self):
        """Steps stay on the grid, so no clipped step contributes a fake zero effect."""
        result = _morris().analyze(_linear, _uniform_ranges("x1", "x2"))
        assert result.first_order_indices["y"]["x1"] == pytest.approx(2.0)
        assert result.first_order_indices["y"]["x2"] == pytest.approx(3.0)
        assert result.morris_sigma["y"]["x1"] == pytest.approx(0.0, abs=1e-12)
        assert result.morris_sigma["y"]["x2"] == pytest.approx(0.0, abs=1e-12)

    def test_grid_jump_that_cannot_stay_inside_the_grid_is_rejected(self):
        with pytest.raises(ValueError, match="grid_jump"):
            MorrisConfig(aggregation_method="mean", n_levels=4, grid_jump=3)

    def test_seeded_design_is_reproducible_without_touching_global_rng(self):
        state = np.random.get_state()
        analyzer_a, analyzer_b = _morris(), _morris()
        analyzer_a.analyze(_linear, _uniform_ranges("x1", "x2"))
        analyzer_b.analyze(_linear, _uniform_ranges("x1", "x2"))
        after = np.random.get_state()
        assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
        assert np.array_equal(analyzer_a.trajectories, analyzer_b.trajectories)

    def test_non_finite_model_output_raises(self):
        def runner(params):
            return {"y": float("nan") if params["x1"] > 0.5 else params["x1"]}

        with pytest.raises(RuntimeError, match="Morris analysis failed") as excinfo:
            _morris().analyze(runner, _uniform_ranges("x1", "x2"))
        assert ValueError in _cause_types(excinfo.value)

    def test_output_missing_from_a_later_run_raises(self):
        calls = {"n": 0}

        def runner(params):
            calls["n"] += 1
            outputs = {"y": params["x1"]}
            if calls["n"] == 1:
                outputs["z"] = params["x2"]
            return outputs

        with pytest.raises(RuntimeError) as excinfo:
            _morris().analyze(runner, _uniform_ranges("x1", "x2"))
        assert KeyError in _cause_types(excinfo.value)

    def test_declared_output_variables_select_the_analysed_outputs(self):
        def runner(params):
            return {"y": params["x1"], "z": params["x2"]}

        result = _morris(output_variables=["z"]).analyze(runner, _uniform_ranges("x1", "x2"))
        assert result.output_variables == ["z"]

    def test_declared_output_variable_missing_from_the_model_raises(self):
        with pytest.raises(RuntimeError) as excinfo:
            _morris(output_variables=["w"]).analyze(_linear, _uniform_ranges("x1", "x2"))
        assert KeyError in _cause_types(excinfo.value)

    def test_non_uniform_range_is_refused_not_sampled_as_uniform(self):
        ranges = _uniform_ranges("x1")
        ranges["x2"] = ParameterRange("x2", 0.0, 1.0, distribution="normal", mean=0.5, std=0.1)
        with pytest.raises(NotImplementedError, match="uniform"):
            _morris().analyze(_linear, ranges)

    def test_baseline_parameters_reach_the_model(self):
        seen = []

        def runner(params):
            seen.append(params["c"])
            return {"y": params["x1"] * params["c"]}

        _morris().analyze(runner, _uniform_ranges("x1"), baseline_parameters={"c": 4.0})
        assert seen and all(value == 4.0 for value in seen)


class TestSobolNoSilentSubstitution:
    @pytest.mark.parametrize("estimator", ["janon", "glen", "owen", "saltelly"])
    def test_only_the_implemented_estimator_is_offered(self, estimator):
        """Only 'saltelli' is implemented; any other name is an unknown estimator."""
        from relic.optimization.sensitivity import sobol

        assert sobol.ESTIMATORS == ("saltelli",)
        assert not hasattr(sobol, "UNIMPLEMENTED_ESTIMATORS")
        with pytest.raises(ValueError, match="estimator"):
            SobolConfig(aggregation_method="mean", estimator=estimator)

    def test_estimator_stays_required(self):
        with pytest.raises(TypeError, match="estimator"):
            SobolConfig(aggregation_method="mean")

    def test_second_order_indices_are_not_a_configuration_option(self):
        """Second-order indices (Saltelli 2002) are not implemented and not declared."""
        with pytest.raises(TypeError, match="calc_second_order"):
            SobolConfig(aggregation_method="mean", estimator="saltelli", calc_second_order=False)
        assert not hasattr(SobolConfig(aggregation_method="mean", estimator="saltelli"), "calc_second_order")

    def test_scrambled_design_uses_the_declared_seed(self):
        first = _sobol(random_seed=3).analyze(_linear, _uniform_ranges("x1", "x2"))
        second = _sobol(random_seed=3).analyze(_linear, _uniform_ranges("x1", "x2"))
        other = _sobol(random_seed=4).analyze(_linear, _uniform_ranges("x1", "x2"))
        assert first.first_order_indices == second.first_order_indices
        assert first.first_order_indices != other.first_order_indices

    def test_missing_output_raises_instead_of_zero(self):
        calls = {"n": 0}

        def runner(params):
            calls["n"] += 1
            outputs = {"y": params["x1"]}
            if calls["n"] == 1:
                outputs["z"] = params["x2"]
            return outputs

        with pytest.raises(RuntimeError) as excinfo:
            _sobol().analyze(runner, _uniform_ranges("x1", "x2"))
        assert KeyError in _cause_types(excinfo.value)

    def test_constant_output_raises_instead_of_unit_indices(self):
        with pytest.raises(RuntimeError) as excinfo:
            _sobol().analyze(lambda params: {"y": 1.0}, _uniform_ranges("x1", "x2"))
        assert ValueError in _cause_types(excinfo.value)
        assert "variance" in str(excinfo.value)

    def test_non_finite_output_raises_instead_of_zero_index(self):
        def runner(params):
            return {"y": float("nan") if params["x1"] > 0.5 else params["x1"]}

        with pytest.raises(RuntimeError) as excinfo:
            _sobol().analyze(runner, _uniform_ranges("x1", "x2"))
        assert ValueError in _cause_types(excinfo.value)

    def test_first_sample_failure_reports_its_index(self):
        def runner(_params):
            raise ValueError("model domain error")

        with pytest.raises(RuntimeError) as excinfo:
            _sobol().analyze(runner, _uniform_ranges("x1", "x2"))
        cause = excinfo.value.__cause__
        assert isinstance(cause, RuntimeError) and "sample 0" in str(cause)
        assert isinstance(cause.__cause__, ValueError)

    def test_convergence_check_covers_every_output(self):
        def runner(params):
            return {"y": params["x1"] + 2.0 * params["x2"], "z": params["x2"]}

        result = _sobol(convergence_check=True, convergence_samples=[16, 32, 64]).analyze(
            runner, _uniform_ranges("x1", "x2")
        )
        assert set(result.first_order_indices) == {"y", "z"}

    @pytest.mark.parametrize("samples", [[64], [16, 128], [0, 64]])
    def test_convergence_samples_must_define_a_check(self, samples):
        with pytest.raises(ValueError, match="convergence_samples"):
            SobolConfig(
                aggregation_method="mean", estimator="saltelli",
                n_samples=64, convergence_check=True, convergence_samples=samples,
            )

    def test_convergence_uses_absolute_index_change(self):
        analyzer = _sobol(convergence_tolerance=0.05)
        data = {
            "y": {
                "sample_sizes": [32, 64],
                "first_order_convergence": {0: [0.0, 0.01]},
                "total_order_convergence": {0: [0.0, 0.01]},
            }
        }
        assert analyzer._check_convergence(data) is True

    def test_convergence_without_two_sample_sizes_raises(self):
        analyzer = _sobol()
        data = {
            "y": {
                "sample_sizes": [64],
                "first_order_convergence": {0: [0.3]},
                "total_order_convergence": {0: [0.3]},
            }
        }
        with pytest.raises(ValueError, match="sample sizes"):
            analyzer._check_convergence(data)

    def test_negative_first_order_estimate_is_reported_unclipped(self):
        analyzer = _sobol()
        analyzer.output_order = ["y"]
        outputs = {
            "A": {"y": np.array([0.0, 1.0])},
            "B": {"y": np.array([0.0, 1.0])},
            "AB_0": {"y": np.array([1.0, 0.0])},
        }
        indices = analyzer._calculate_sobol_indices(outputs, 1)
        assert indices["y"]["first_order"][0] == pytest.approx(-0.5)

    def test_first_order_sum_above_one_is_reported_unclipped(self):
        analyzer = _sobol()
        indices = {"y": {"first_order": {0: 0.7, 1: 0.6}, "total_order": {0: 0.7, 1: 0.6}}}
        result = analyzer._create_sobol_result(
            indices, ["x1", "x2"], None, n_model_evaluations=0, computation_time=0.0
        )
        assert result.variance_explained["y"] == pytest.approx(1.3)

    def test_non_uniform_range_is_refused_not_sampled_as_uniform(self):
        ranges = _uniform_ranges("x1")
        ranges["x2"] = ParameterRange("x2", 0.1, 1.0, distribution="uniform", log_scale=True)
        with pytest.raises(NotImplementedError, match="log_scale"):
            _sobol().analyze(_linear, ranges)



# ---------------------------------------------------------------------------
# Round 5: no literal defaults in the sensitivity configuration. The caller
# chooses the method (fixed by MorrisConfig / SobolConfig), the aggregation,
# the estimator and the range distribution; the sample sizes, Morris grid and
# tolerance come from optimization.sensitivity.* (YAML); results state their
# counters; declared range units are checked against the parameter database.
# ---------------------------------------------------------------------------


class TestRound5NoLiteralDefaults:
    def test_method_and_aggregation_are_required(self):
        with pytest.raises(TypeError):
            SensitivityConfig(aggregation_method="mean")
        with pytest.raises(TypeError):
            SensitivityConfig(method=SensitivityMethod.MORRIS)

    def test_subclasses_fix_their_method(self):
        assert MorrisConfig(aggregation_method="mean").method is SensitivityMethod.MORRIS
        sobol = SobolConfig(aggregation_method="mean", estimator="saltelli")
        assert sobol.method is SensitivityMethod.SOBOL
        with pytest.raises(TypeError):
            MorrisConfig(aggregation_method="mean", method=SensitivityMethod.SOBOL)
        with pytest.raises(TypeError):
            SobolConfig(aggregation_method="mean", estimator="saltelli", method=SensitivityMethod.MORRIS)
        with pytest.raises(TypeError):
            SobolConfig(aggregation_method="mean")

    def test_analyzers_need_an_explicit_config(self):
        with pytest.raises(TypeError, match="MorrisConfig"):
            MorrisAnalyzer(_DummyParameterManager())
        with pytest.raises(TypeError, match="SobolConfig"):
            SobolAnalyzer(_DummyParameterManager())

    def test_numeric_settings_are_read_from_the_parameter_database(self):
        from relic.parameters import get_global_manager

        expected = {
            "optimization.sensitivity.morris.n_trajectories": 50,
            "optimization.sensitivity.sobol.n_samples": 1000,
            "optimization.sensitivity.convergence_tolerance": 0.05,
            "optimization.sensitivity.morris.n_levels": 4,
            "optimization.sensitivity.morris.grid_jump": 2,
            "optimization.sensitivity.sobol.skip_values": 1000,
        }
        manager = get_global_manager()
        originals = {key: manager.get(key, required=True) for key in expected}
        for key, value in expected.items():
            assert float(originals[key]) == pytest.approx(value)
        base = SensitivityConfig(method=SensitivityMethod.MORRIS, aggregation_method="mean")
        # Sample settings belong to the method: r to Morris, N to Sobol.
        assert not hasattr(base, "n_trajectories") and not hasattr(base, "n_samples")
        assert base.convergence_tolerance == pytest.approx(0.05)
        morris = MorrisConfig(aggregation_method="mean")
        assert (morris.n_trajectories, morris.n_levels, morris.grid_jump) == (50, 4, 2)
        sobol = SobolConfig(aggregation_method="mean", estimator="saltelli")
        assert (sobol.n_samples, sobol.skip_values) == (1000, 1000)
        try:
            manager.set_value("optimization.sensitivity.morris.n_levels", 6)
            manager.set_value("optimization.sensitivity.morris.grid_jump", 3)
            patched = MorrisConfig(aggregation_method="mean")
            assert (patched.n_levels, patched.grid_jump) == (6, 3)
        finally:
            for key, value in originals.items():
                manager.set_value(key, value)

    def test_the_range_distribution_is_required(self):
        with pytest.raises(TypeError):
            ParameterRange("a", 0.0, 1.0)

    def test_result_counters_are_required(self):
        fields = dict(method="Morris", parameter_names=["a"], output_variables=["y"])
        with pytest.raises(TypeError):
            SensitivityResult(**fields, computation_time=0.0)
        with pytest.raises(TypeError):
            SensitivityResult(**fields, n_model_evaluations=3)
        result = SensitivityResult(**fields, n_model_evaluations=3, computation_time=0.5)
        assert (result.n_model_evaluations, result.computation_time) == (3, 0.5)

    def test_declared_range_units_are_checked_against_the_parameter_spec(self):
        class _Spec:
            units = "umol m-2 s-1"

        class _Manager:
            def exists(self, name):
                return name == "vcmax25"

            def get_parameter(self, name):
                return _Spec()

        analyzer = _DummySensitivity(
            parameter_manager=_Manager(),
            config=SensitivityConfig(method=SensitivityMethod.MORRIS, aggregation_method="mean"),
        )
        analyzer.validate_parameter_ranges(
            {"vcmax25": ParameterRange("vcmax25", 20.0, 80.0, distribution="uniform", units="umol m-2 s-1")}
        )
        analyzer.validate_parameter_ranges(
            {"vcmax25": ParameterRange("vcmax25", 20.0, 80.0, distribution="uniform")}
        )
        with pytest.raises(ValueError, match="units"):
            analyzer.validate_parameter_ranges(
                {"vcmax25": ParameterRange("vcmax25", 20.0, 80.0, distribution="uniform", units="mol m-2 s-1")}
            )
        with pytest.raises(ValueError, match="units"):
            analyzer.validate_parameter_ranges(
                {"x1": ParameterRange("x1", 0.0, 1.0, distribution="uniform", units="m")}
            )


class TestRound6SampleSettings:
    """Round 6: r belongs to Morris, N to Sobol; no r * 10 sample count exists."""

    def test_morris_has_no_sample_count_and_sobol_no_trajectories(self):
        morris = MorrisConfig(aggregation_method="mean")
        assert not hasattr(morris, "n_samples")
        with pytest.raises(TypeError, match="n_samples"):
            MorrisConfig(aggregation_method="mean", n_samples=10)
        with pytest.raises(TypeError, match="n_trajectories"):
            SobolConfig(aggregation_method="mean", estimator="saltelli", n_trajectories=10)

    def test_sobol_convergence_sample_sizes_come_from_the_parameter_database(self):
        from relic.parameters import get_global_manager

        config = SobolConfig(aggregation_method="mean", estimator="saltelli", n_samples=1000)
        assert config.convergence_samples == [250, 500, 1000]
        manager = get_global_manager()
        key = "optimization.sensitivity.sobol.convergence_sample_divisors"
        original = manager.get(key, required=True)
        try:
            manager.set_value(key, [5, 2, 1])
            patched = SobolConfig(aggregation_method="mean", estimator="saltelli", n_samples=1000)
            assert patched.convergence_samples == [200, 500, 1000]
        finally:
            manager.set_value(key, original)
