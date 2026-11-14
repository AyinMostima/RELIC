"""L1 tests for probabilistic forecast validation metrics."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from relic.stochastic import validation as validation_module
from relic.stochastic.validation import (
    CRPSValidator,
    PITValidator,
    StochasticValidationResult,
    brier_score,
    coverage_probability,
    crps_score,
    pit_test,
    quantile_score,
)

pytestmark = pytest.mark.unit


class TestPITValidation:
    def test_ensemble_pit_matches_empirical_cdf_definition(self) -> None:
        observations = torch.tensor([0.0, 1.0])
        ensemble = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 2.0, 3.0]])

        values = PITValidator()._compute_ensemble_pit(observations, ensemble)

        expected = torch.tensor([2.0 / 3.0, 1.0 / 3.0], dtype=values.dtype)
        torch.testing.assert_close(values, expected)

    def test_parametric_normal_pit_at_location_is_one_half(self) -> None:
        observations = torch.tensor([0.0, 2.0])
        params = {
            "loc": torch.tensor([0.0, 2.0]),
            "scale": torch.tensor([1.0, 3.0]),
        }

        values = PITValidator()._compute_parametric_pit(observations, params)

        torch.testing.assert_close(values, torch.full((2,), 0.5))

    def test_parametric_uniform_pit_reports_continuous_method(self) -> None:
        observations = torch.tensor([0.25, 0.75])
        params = {
            "low": torch.zeros(2),
            "high": torch.ones(2),
        }

        result = PITValidator().compute_score(observations, params)

        assert result["pit_method"] == "parametric_cdf"
        assert result["pit_is_discrete"] is False
        assert result["ensemble_members"] is None
        assert "uniformity_test_interpretation" not in result

    def test_public_pit_result_preserves_score_and_sample_size(self) -> None:
        observations = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        ensemble = torch.tensor(
            [
                [-2.0, -1.0, 0.0],
                [-1.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
            ]
        )

        result = pit_test(observations, ensemble)

        assert isinstance(result, StochasticValidationResult)
        assert result.sample_size == 4
        assert result.score == pytest.approx(2.0 / 3.0)
        assert result.decomposition["deviation_from_uniform"] == pytest.approx(
            1.0 / 6.0
        )
        assert 0.0 <= result.pvalue <= 1.0
        assert result.metadata["pit_method"] == "empirical_ensemble_cdf"
        assert result.metadata["pit_is_discrete"] is True
        assert 0.0 <= result.metadata["cvm_pvalue"] <= 1.0

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, math.nan])
    def test_pit_rejects_invalid_significance_level(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            PITValidator(alpha=alpha)

    @pytest.mark.parametrize(
        ("observations", "ensemble", "message"),
        [
            (torch.ones(2, 1), torch.ones(2, 3), "Observations"),
            (torch.ones(2), torch.ones(2), "2D"),
            (torch.ones(2), torch.ones(3, 2), "size mismatch"),
            (torch.empty(0), torch.empty(0, 2), "non-empty"),
            (torch.ones(2), torch.empty(2, 0), "ensemble member"),
            (
                torch.tensor([1.0, math.nan]),
                torch.ones(2, 2),
                "finite",
            ),
            (
                torch.ones(2),
                torch.tensor([[1.0, math.inf], [1.0, 2.0]]),
                "finite",
            ),
        ],
    )
    def test_pit_rejects_invalid_ensemble_contract(
        self,
        observations: torch.Tensor,
        ensemble: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            PITValidator().compute_score(observations, ensemble)

    def test_pit_rejects_unknown_prediction_type(self) -> None:
        with pytest.raises(TypeError, match="Tensor.*Dict"):
            PITValidator().compute_score(torch.tensor([0.0]), [0.0])

    def test_pit_runs_histogram_and_serial_independence_diagnostics(self) -> None:
        values = torch.linspace(0.01, 0.99, 100)

        result = PITValidator()._run_consistency_tests(
            values,
            n_bins=10,
            test_independence=True,
        )

        assert 0.0 <= result["chi2_pvalue"] <= 1.0
        assert 0.0 <= result["ljungbox_pvalue"] <= 1.0

    def test_failed_serial_independence_test_propagates(self, monkeypatch) -> None:
        """A requested Ljung-Box test that fails is an error, not a missing key."""

        def failing_ljungbox(*_args, **_kwargs):
            raise ValueError("lags exceed the series length")

        monkeypatch.setattr(validation_module, "acorr_ljungbox", failing_ljungbox)

        with pytest.raises(ValueError, match="lags exceed"):
            PITValidator()._run_consistency_tests(
                torch.linspace(0.01, 0.99, 100), test_independence=True
            )

    def test_requested_serial_independence_test_needs_enough_values(self) -> None:
        with pytest.raises(ValueError, match="test_independence"):
            PITValidator()._run_consistency_tests(
                torch.linspace(0.05, 0.95, 12), test_independence=True
            )

    @pytest.mark.parametrize("n_bins", [1, 10])
    def test_requested_histogram_test_is_not_skipped_silently(self, n_bins) -> None:
        """An explicit n_bins asks for the chi-squared test; it runs or raises."""
        with pytest.raises(ValueError, match="n_bins"):
            PITValidator()._run_consistency_tests(
                torch.linspace(0.05, 0.95, 12), n_bins=n_bins
            )

    def test_default_histogram_test_is_optional_for_short_series(self) -> None:
        result = PITValidator()._run_consistency_tests(torch.linspace(0.05, 0.95, 12))
        assert "chi2_pvalue" not in result
        assert 0.0 <= result["ks_pvalue"] <= 1.0

    def test_pit_diagnostic_plot_returns_values_and_saves(
        self, monkeypatch, tmp_path
    ) -> None:
        observations = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        ensemble = observations.unsqueeze(1) + torch.tensor([-0.5, 0.0, 0.5])
        output_path = tmp_path / "pit.png"
        figure = MagicMock()
        axes = MagicMock()
        savefig = MagicMock()
        monkeypatch.setattr(
            validation_module.plt,
            "subplots",
            lambda *args, **kwargs: (figure, axes),
        )
        monkeypatch.setattr(validation_module.plt, "tight_layout", MagicMock())
        monkeypatch.setattr(validation_module.plt, "savefig", savefig)

        result = PITValidator().diagnostic_plots(
            observations,
            ensemble,
            save_path=str(output_path),
        )

        assert result["pit_values"].shape == observations.shape
        assert result["figure"] is figure
        savefig.assert_called_once_with(
            str(output_path),
            dpi=300,
            bbox_inches="tight",
        )

    def test_pit_diagnostic_plot_accepts_parametric_distribution(
        self, monkeypatch
    ) -> None:
        observations = torch.tensor([0.25, 0.75])
        params = {
            "low": torch.zeros(2),
            "high": torch.ones(2),
        }
        figure = MagicMock()
        monkeypatch.setattr(
            validation_module.plt,
            "subplots",
            lambda *args, **kwargs: (figure, MagicMock()),
        )
        monkeypatch.setattr(validation_module.plt, "tight_layout", MagicMock())

        result = PITValidator().diagnostic_plots(observations, params)

        torch.testing.assert_close(
            result["pit_values"],
            observations,
        )
        assert result["figure"] is figure


class TestCRPSValidation:
    def test_unimplemented_or_misspelled_options_are_rejected(self) -> None:
        """'weights' was documented but ignored; unknown options are errors."""
        with pytest.raises(TypeError, match="weights"):
            CRPSValidator(decompose=False).compute_score(
                torch.tensor([0.0]), torch.tensor([[-1.0, 1.0]]), weights=torch.ones(1)
            )

    def test_fair_crps_is_rejected_for_a_parametric_forecast(self) -> None:
        params = {"loc": torch.tensor([0.0]), "scale": torch.tensor([2.0])}
        with pytest.raises(ValueError, match="fair_crps"):
            CRPSValidator(decompose=False).compute_score(
                torch.tensor([0.0]), params, fair_crps=True
            )

    def test_requested_decomposition_of_a_parametric_forecast_is_flagged(self) -> None:
        params = {"loc": torch.tensor([0.0]), "scale": torch.tensor([2.0])}
        result = CRPSValidator(decompose=True).compute_score(torch.tensor([0.0]), params)
        assert result["decomposition_available"] is False
        assert "ensemble" in result["decomposition_reason"]

    def test_ensemble_crps_matches_pairwise_definition(self) -> None:
        observations = torch.tensor([0.0])
        ensemble = torch.tensor([[-1.0, 1.0]])

        result = CRPSValidator(decompose=False).compute_score(
            observations, ensemble
        )

        torch.testing.assert_close(result["crps_values"], torch.tensor([0.5]))
        assert result["crps_mean"] == pytest.approx(0.5)

    def test_deterministic_ensemble_crps_reduces_to_absolute_error(self) -> None:
        observations = torch.tensor([0.0, 2.0])
        ensemble = torch.tensor([[2.0], [-1.0]])

        result = CRPSValidator(decompose=False).compute_score(
            observations, ensemble
        )

        torch.testing.assert_close(result["crps_values"], torch.tensor([2.0, 3.0]))

    def test_uniform_crps_is_nonnegative_inside_and_outside_support(self) -> None:
        observations = torch.tensor([1.0, -1.0, 3.0])
        params = {
            "low": torch.zeros(3),
            "high": torch.full((3,), 2.0),
        }

        result = CRPSValidator(decompose=False).compute_score(
            observations, params
        )

        expected = torch.tensor([1.0 / 6.0, 5.0 / 3.0, 5.0 / 3.0])
        torch.testing.assert_close(result["crps_values"], expected)
        assert result["crps_mean"] >= 0.0

    def test_normal_crps_at_mean_matches_closed_form(self) -> None:
        observations = torch.tensor([0.0])
        params = {"loc": torch.tensor([0.0]), "scale": torch.tensor([2.0])}

        result = CRPSValidator(decompose=False).compute_score(
            observations, params
        )

        expected = 2.0 * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
        assert result["crps_mean"] == pytest.approx(expected)

    def test_fair_crps_requires_at_least_two_members(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            CRPSValidator(decompose=False).compute_score(
                torch.tensor([0.0]),
                torch.tensor([[1.0]]),
                fair_crps=True,
            )

    def test_fair_crps_uses_off_diagonal_member_pairs(self) -> None:
        result = CRPSValidator(decompose=False).compute_score(
            torch.tensor([0.0]),
            torch.tensor([[-1.0, 1.0]]),
            fair_crps=True,
        )

        torch.testing.assert_close(result["crps_values"], torch.tensor([0.0]))

    def test_fair_crps_does_not_report_standard_crps_decomposition(self) -> None:
        result = crps_score(
            torch.tensor([0.0]),
            torch.tensor([[-1.0, 1.0]]),
            fair=True,
        )

        assert result.decomposition == {}
        assert result.metadata["decomposition_available"] is False
        assert result.metadata["decomposition_reason"] == (
            "standard_empirical_cdf_decomposition_not_valid_for_fair_crps"
        )

    @pytest.mark.parametrize(
        ("observations", "ensemble", "message"),
        [
            (torch.ones(2), torch.ones(2), "2D"),
            (torch.ones(2), torch.ones(3, 2), "Size mismatch"),
            (torch.empty(0), torch.empty(0, 2), "non-empty"),
            (torch.ones(2), torch.empty(2, 0), "ensemble member"),
            (
                torch.tensor([1.0, math.nan]),
                torch.ones(2, 2),
                "finite",
            ),
        ],
    )
    def test_crps_rejects_invalid_ensemble_contract(
        self,
        observations: torch.Tensor,
        ensemble: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            CRPSValidator(decompose=False).compute_score(observations, ensemble)

    def test_crps_result_exposes_decomposition_as_mapping(self) -> None:
        observations = torch.tensor([-1.0, 0.0, 1.0])
        ensemble = torch.tensor(
            [[-1.5, -0.5], [-0.5, 0.5], [0.5, 1.5]]
        )

        result = crps_score(observations, ensemble)

        assert isinstance(result.decomposition, dict)
        assert set(result.decomposition) == {
            "reliability",
            "resolution",
            "uncertainty",
        }
        reconstructed = (
            result.decomposition["reliability"]
            - result.decomposition["resolution"]
            + result.decomposition["uncertainty"]
        )
        assert reconstructed == pytest.approx(result.score)

    def test_crps_decomposition_scales_with_observation_units(self) -> None:
        observations = torch.tensor([-1.0, 0.0, 1.0])
        ensemble = torch.tensor(
            [[-1.5, -0.5], [-0.5, 0.5], [0.5, 1.5]]
        )

        base = crps_score(observations, ensemble)
        scaled = crps_score(observations * 10.0, ensemble * 10.0)

        assert scaled.score == pytest.approx(base.score * 10.0)
        for component in ("reliability", "resolution", "uncertainty"):
            assert scaled.decomposition[component] == pytest.approx(
                base.decomposition[component] * 10.0
            )

    def test_crps_rejects_unsupported_parametric_distribution(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            CRPSValidator(decompose=False).compute_score(
                torch.tensor([0.0, 1.0]),
                {"shape": torch.ones(2)},
            )

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            (
                {"loc": torch.zeros(2), "scale": torch.tensor([1.0, 0.0])},
                "positive",
            ),
            (
                {"low": torch.zeros(2), "high": torch.tensor([1.0, 0.0])},
                "greater",
            ),
            (
                {"loc": torch.zeros(3), "scale": torch.ones(3)},
                "same shape",
            ),
        ],
    )
    def test_crps_rejects_invalid_parametric_contract(
        self,
        params: dict[str, torch.Tensor],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            CRPSValidator(decompose=False).compute_score(
                torch.tensor([0.0, 1.0]),
                params,
            )

    def test_crps_diagnostic_plot_returns_values_and_saves(
        self, monkeypatch, tmp_path
    ) -> None:
        observations = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        ensemble = observations.unsqueeze(1) + torch.tensor([-0.5, 0.0, 0.5])
        output_path = tmp_path / "crps.png"
        figure = MagicMock()
        axes = MagicMock()
        savefig = MagicMock()
        monkeypatch.setattr(
            validation_module.plt,
            "subplots",
            lambda *args, **kwargs: (figure, axes),
        )
        monkeypatch.setattr(validation_module.plt, "tight_layout", MagicMock())
        monkeypatch.setattr(validation_module.plt, "savefig", savefig)

        result = CRPSValidator().diagnostic_plots(
            observations,
            ensemble,
            save_path=str(output_path),
        )

        assert torch.isfinite(result["crps_values"]).all()
        assert result["skill_score"] is not None
        assert result["skill_score"].shape == observations.shape
        assert result["figure"] is figure
        savefig.assert_called_once_with(
            str(output_path),
            dpi=300,
            bbox_inches="tight",
        )

    def test_crps_diagnostic_plot_handles_deterministic_ensemble(
        self, monkeypatch
    ) -> None:
        observations = torch.tensor([0.0, 1.0])
        ensemble = torch.tensor([[0.5], [1.5]])
        figure = MagicMock()
        monkeypatch.setattr(
            validation_module.plt,
            "subplots",
            lambda *args, **kwargs: (figure, MagicMock()),
        )
        monkeypatch.setattr(validation_module.plt, "tight_layout", MagicMock())

        result = CRPSValidator().diagnostic_plots(observations, ensemble)

        torch.testing.assert_close(
            result["crps_values"],
            torch.tensor([0.5, 0.5]),
        )
        assert result["skill_score"] is None
        assert result["figure"] is figure

    def test_crps_diagnostic_plot_marks_zero_climatology_skill_undefined(
        self, monkeypatch
    ) -> None:
        observations = torch.zeros(3)
        ensemble = torch.tensor(
            [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]
        )
        figure = MagicMock()
        monkeypatch.setattr(
            validation_module.plt,
            "subplots",
            lambda *args, **kwargs: (figure, MagicMock()),
        )
        monkeypatch.setattr(validation_module.plt, "tight_layout", MagicMock())

        result = CRPSValidator().diagnostic_plots(observations, ensemble)

        assert result["skill_score"] is not None
        assert torch.isnan(result["skill_score"]).all()


class TestCoverageProbability:
    def test_coverage_includes_interval_endpoints(self) -> None:
        observations = torch.tensor([0.0, 1.0, 2.0, 3.0])
        lower = torch.tensor([0.0, 0.5, 2.0, 4.0])
        upper = torch.tensor([0.0, 1.5, 2.5, 5.0])

        result = coverage_probability(
            observations, lower, upper, nominal_coverage=0.75
        )

        assert result.score == pytest.approx(0.75)
        assert result.decomposition["n_covered"] == 3
        assert result.pvalue == pytest.approx(1.0)
        assert result.metadata["within_bounds"].tolist() == [True, True, True, False]

    @pytest.mark.parametrize("covered", [False, True])
    def test_extreme_coverage_has_non_degenerate_exact_interval(
        self, covered: bool
    ) -> None:
        observations = torch.arange(4.0)
        if covered:
            lower = observations - 1.0
            upper = observations + 1.0
        else:
            lower = observations + 1.0
            upper = observations + 2.0

        result = coverage_probability(observations, lower, upper)

        lower_ci, upper_ci = result.confidence_interval
        if covered:
            assert 0.0 < lower_ci < 1.0
            assert upper_ci == pytest.approx(1.0)
        else:
            assert lower_ci == pytest.approx(0.0)
            assert 0.0 < upper_ci < 1.0

    @pytest.mark.parametrize("nominal", [0.0, 1.0, -0.1, 1.1, math.nan])
    def test_coverage_rejects_invalid_nominal_probability(
        self, nominal: float
    ) -> None:
        values = torch.tensor([0.0, 1.0])
        with pytest.raises(ValueError, match="nominal_coverage"):
            coverage_probability(values, values, values, nominal)

    @pytest.mark.parametrize(
        ("observations", "lower", "upper", "message"),
        [
            (torch.empty(0), torch.empty(0), torch.empty(0), "non-empty"),
            (torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 1), "1D"),
            (torch.ones(2), torch.ones(3), torch.ones(2), "same shape"),
            (
                torch.tensor([0.0, math.nan]),
                torch.zeros(2),
                torch.ones(2),
                "finite",
            ),
            (
                torch.zeros(2),
                torch.tensor([0.0, 2.0]),
                torch.tensor([1.0, 1.0]),
                "lower",
            ),
        ],
    )
    def test_coverage_rejects_invalid_interval_contract(
        self,
        observations: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            coverage_probability(observations, lower, upper)

    def test_nominal_coverage_default_is_read_from_the_parameter_database(self) -> None:
        """Without nominal_coverage the YAML value nominal_interval_coverage applies."""
        import inspect

        from scipy import stats

        from relic.parameters import get_global_manager

        assert inspect.signature(coverage_probability).parameters["nominal_coverage"].default is None
        key = "validation.stochastic_scores.nominal_interval_coverage"
        manager = get_global_manager()
        original = manager.get(key, required=True)
        assert original == pytest.approx(0.9)
        observations = torch.arange(10.0)
        lower = observations - 0.5
        upper = torch.where(observations < 7.0, observations + 0.5, observations - 0.25)
        default = coverage_probability(observations, lower, upper)
        assert default.decomposition["nominal_coverage"] == pytest.approx(0.9)
        try:
            manager.set_value(key, 0.75)
            result = coverage_probability(observations, lower, upper)
            assert result.decomposition["nominal_coverage"] == pytest.approx(0.75)
            assert result.pvalue == pytest.approx(stats.binomtest(7, 10, 0.75).pvalue)
            explicit = coverage_probability(observations, lower, upper, nominal_coverage=0.6)
            assert explicit.decomposition["nominal_coverage"] == pytest.approx(0.6)
        finally:
            manager.set_value(key, original)
        assert manager.get(key, required=True) == original

    def test_interval_confidence_level_is_read_from_the_parameter_database(self) -> None:
        # References: Clopper and Pearson (1934), exact binomial interval.
        # Mathematical expression: [B(alpha/2; k, n-k+1), B(1-alpha/2; k+1, n-k)],
        #   alpha = 1 - validation.stochastic_scores.coverage_interval_confidence_level.
        # Parameter meanings: k covered observations out of n; B the beta quantile.
        # Units: probabilities (dimensionless).
        # Domain/constraints: 0 < k < n here, so both limits are beta quantiles.
        # Sign convention: lower <= empirical coverage <= upper.
        from scipy import stats

        from relic.parameters import get_global_manager

        key = "validation.stochastic_scores.coverage_interval_confidence_level"
        manager = get_global_manager()
        original = manager.get(key, required=True)
        assert original == pytest.approx(0.95)
        observations = torch.arange(10.0)
        lower = observations - 0.5
        upper = torch.where(observations < 7.0, observations + 0.5, observations - 0.25)

        def expected(level: float):
            alpha = 1.0 - level
            return (
                float(stats.beta.ppf(alpha / 2, 7, 4)),
                float(stats.beta.ppf(1 - alpha / 2, 8, 3)),
            )

        result = coverage_probability(observations, lower, upper, nominal_coverage=0.7)
        assert result.decomposition["n_covered"] == 7
        assert result.confidence_interval == pytest.approx(expected(0.95))
        try:
            manager.set_value(key, 0.5)
            narrow = coverage_probability(observations, lower, upper, nominal_coverage=0.7)
            assert narrow.confidence_interval == pytest.approx(expected(0.5))
        finally:
            manager.set_value(key, original)
        assert manager.get(key, required=True) == original


class TestBrierScore:
    def test_brier_score_matches_binary_squared_error(self) -> None:
        observations = torch.tensor([-1.0, 1.0])
        probabilities = torch.tensor([0.25, 0.75])

        result = brier_score(observations, probabilities, threshold=0.0)

        assert result.score == pytest.approx(0.0625)
        expected_outcomes = torch.tensor(
            [0.0, 1.0],
            dtype=result.metadata["binary_outcomes"].dtype,
        )
        torch.testing.assert_close(
            result.metadata["binary_outcomes"], expected_outcomes
        )

    def test_brier_murphy_decomposition_closes(self) -> None:
        observations = torch.tensor([-1.0, -2.0, 1.0, 2.0])
        probabilities = torch.tensor([0.2, 0.2, 0.8, 0.8])

        result = brier_score(observations, probabilities, threshold=0.0)
        components = result.decomposition

        reconstructed = (
            components["reliability"]
            - components["resolution"]
            + components["uncertainty"]
        )
        assert reconstructed == pytest.approx(result.score)

    @pytest.mark.parametrize(
        ("observations", "probabilities", "message"),
        [
            (torch.empty(0), torch.empty(0), "non-empty"),
            (torch.ones(2, 1), torch.ones(2), "1D"),
            (torch.ones(2), torch.ones(3), "same shape"),
            (torch.ones(2), torch.tensor([-0.1, 0.5]), r"\[0, 1\]"),
            (torch.ones(2), torch.tensor([0.5, 1.1]), r"\[0, 1\]"),
            (
                torch.tensor([1.0, math.inf]),
                torch.tensor([0.5, 0.5]),
                "finite",
            ),
        ],
    )
    def test_brier_rejects_invalid_probability_contract(
        self,
        observations: torch.Tensor,
        probabilities: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            brier_score(observations, probabilities, threshold=0.0)


class TestQuantileScore:
    def test_quantile_keys_do_not_collide_and_follow_the_ensemble_rule(self) -> None:
        """'.2f' keys merged 0.025 with 0.03; labels come from the shared quantile_label."""
        from relic.stochastic.ensemble_runner import quantile_label

        observations = torch.tensor([0.0, 2.0])
        forecasts = torch.tensor([[0.0, 0.0], [1.0, 3.0]])
        levels = torch.tensor([0.025, 0.03], dtype=torch.float32)

        result = quantile_score(observations, forecasts, levels)

        assert set(result.decomposition) == {"q02.5", "q03"}
        assert quantile_label(levels[0]) == "q02.5"

    def test_quantile_score_matches_pinball_loss(self) -> None:
        observations = torch.tensor([0.0, 2.0])
        forecasts = torch.tensor([[0.0, 0.0], [1.0, 3.0]])
        levels = torch.tensor([0.25, 0.75])

        result = quantile_score(observations, forecasts, levels)

        torch.testing.assert_close(
            result.metadata["mean_scores_by_level"],
            torch.tensor([0.125, 0.125]),
        )
        assert result.score == pytest.approx(0.0625)
        assert (
            result.metadata["aggregation_method"]
            == "trapezoidal_pinball_integral"
        )
        assert result.metadata["quantile_level_range"] == pytest.approx(
            (0.25, 0.75)
        )
        assert result.metadata["is_full_crps"] is False

    def test_single_quantile_score_uses_mean_pinball_loss(self) -> None:
        observations = torch.tensor([0.0, 2.0])
        forecasts = torch.tensor([[1.0], [1.0]])
        levels = torch.tensor([0.5])

        result = quantile_score(observations, forecasts, levels)

        assert result.score == pytest.approx(0.5)
        assert (
            result.metadata["aggregation_method"]
            == "single_level_pinball_loss"
        )
        assert result.metadata["is_full_crps"] is False

    @pytest.mark.parametrize(
        ("observations", "forecasts", "levels", "message"),
        [
            (torch.empty(0), torch.empty(0, 1), torch.tensor([0.5]), "non-empty"),
            (torch.ones(2, 1), torch.ones(2, 1), torch.tensor([0.5]), "1D"),
            (torch.ones(2), torch.ones(2), torch.tensor([0.5]), "2D"),
            (torch.ones(2), torch.ones(3, 1), torch.tensor([0.5]), "same"),
            (torch.ones(2), torch.ones(2, 2), torch.tensor([0.5]), "size mismatch"),
            (
                torch.ones(2),
                torch.ones(2, 2),
                torch.tensor([0.0, 0.5]),
                "strictly between",
            ),
            (
                torch.ones(2),
                torch.ones(2, 2),
                torch.tensor([0.75, 0.25]),
                "increasing",
            ),
            (
                torch.ones(2),
                torch.tensor([[1.0, 0.0], [1.0, 2.0]]),
                torch.tensor([0.25, 0.75]),
                "nondecreasing",
            ),
        ],
    )
    def test_quantile_rejects_invalid_forecast_contract(
        self,
        observations: torch.Tensor,
        forecasts: torch.Tensor,
        levels: torch.Tensor,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            quantile_score(observations, forecasts, levels)


class TestRound5Declarations:
    """Round 5: no literal significance level and no default sample size."""

    def test_pit_significance_level_is_the_parameter_database_value(self, monkeypatch) -> None:
        from relic.parameters import get_global_manager

        assert PITValidator().alpha == pytest.approx(0.05)
        seen = []
        original_init = PITValidator.__init__

        def recording_init(validator, *args, **kwargs):
            original_init(validator, *args, **kwargs)
            seen.append(validator.alpha)

        manager = get_global_manager()
        key = "validation.stochastic_scores.pit_significance_level"
        original = manager.get(key, required=True)
        try:
            manager.set_value(key, 0.1)
            assert PITValidator().alpha == pytest.approx(0.1)
            assert PITValidator(alpha=0.01).alpha == pytest.approx(0.01)
            monkeypatch.setattr(PITValidator, "__init__", recording_init)
            validation_module.pit_test(
                torch.linspace(-1.0, 1.0, 12),
                torch.linspace(-1.0, 1.0, 12).unsqueeze(1) + torch.linspace(-0.5, 0.5, 8),
            )
            assert seen == [pytest.approx(0.1)]
        finally:
            manager.set_value(key, original)

    def test_result_sample_size_is_required(self) -> None:
        with pytest.raises(TypeError):
            StochasticValidationResult(score=0.0, decomposition={})
        result = StochasticValidationResult(score=0.0, decomposition={}, sample_size=3)
        assert result.sample_size == 3
