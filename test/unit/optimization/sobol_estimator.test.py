import pytest

from relic.optimization.sensitivity import ParameterRange, SobolAnalyzer, SobolConfig


class _DummyParameterManager:
    def exists(self, name: str) -> bool:
        return True


def test_sobol_estimator_recovers_additive_reference_indices():
    analyzer = SobolAnalyzer(
        parameter_manager=_DummyParameterManager(),
        config=SobolConfig(
            aggregation_method="mean",
            estimator="saltelli",
            n_samples=1024,
            skip_values=64,
            batch_size=128,
            convergence_check=False,
            random_seed=7,
        ),
    )

    parameter_ranges = {
        "x1": ParameterRange("x1", 0.0, 1.0, distribution="uniform"),
        "x2": ParameterRange("x2", 0.0, 1.0, distribution="uniform"),
    }

    def model_runner(sampled_params):
        return {
            "y": float(sampled_params["x1"] + 2.0 * sampled_params["x2"]),
        }

    result = analyzer.analyze(model_runner, parameter_ranges)
    first_order = result.first_order_indices["y"]
    total_order = result.total_order_indices["y"]

    assert first_order["x1"] == pytest.approx(0.2, abs=0.05)
    assert total_order["x1"] == pytest.approx(0.2, abs=0.05)
    assert first_order["x2"] == pytest.approx(0.8, abs=0.05)
    assert total_order["x2"] == pytest.approx(0.8, abs=0.05)
    assert total_order["x1"] + 5e-3 >= first_order["x1"]
    assert total_order["x2"] + 5e-3 >= first_order["x2"]
