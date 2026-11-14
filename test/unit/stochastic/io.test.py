"""L1 tests for stochastic result serialization metadata (relic.stochastic.io).

References:
    - CF Metadata Conventions 1.8, standard name table (air_temperature,
      air_temperature_anomaly).
"""

import numpy as np
import pytest
import torch

from relic.stochastic.io import _convert_results_to_dataset

STATE = "stochastic_air_temperature_anomaly"


class TestUndeclaredVariablesAreRejected:
    """Units and dimensions are declared by the producer; no name rule guesses them."""

    @pytest.mark.parametrize(
        "key", ["weather_temperature", "trajectory_rainfall_event_count", "parameter_samples"]
    )
    def test_undeclared_tensor_raises(self, key):
        with pytest.raises(ValueError, match=key):
            _convert_results_to_dataset({key: torch.zeros(3)})

    def test_declared_units_without_dimensions_raise(self):
        with pytest.raises(ValueError, match="variable_dimensions"):
            _convert_results_to_dataset(
                {"flux": torch.zeros(3), "variable_units": {"flux": "kg m-2 s-1"}}
            )

    def test_dimension_count_must_match_the_tensor(self):
        with pytest.raises(ValueError, match="flux"):
            _convert_results_to_dataset(
                {
                    "flux": torch.zeros((2, 3)),
                    "variable_units": {"flux": "kg m-2 s-1"},
                    "variable_dimensions": {"flux": ["time"]},
                }
            )


class TestResultsDataset:
    def test_bound_sde_result_layout_converts_to_a_dataset(self):
        results = {
            "process": "stochastic_air_temperature",
            "seed": 7,
            "state_units": {STATE: "K"},
            "variable_units": {"time_seconds": "s"},
            "variable_dimensions": {"time_seconds": ["time"]},
            "time_seconds": torch.tensor([0.0, 3600.0, 7200.0], dtype=torch.float64),
            f"trajectory_{STATE}": torch.zeros((2, 3), dtype=torch.float64),
            f"ensemble_mean_{STATE}": torch.zeros(3, dtype=torch.float64),
            f"ensemble_variance_{STATE}": torch.zeros(3, dtype=torch.float64),
            "parameter_keys": ["stochastic.weather.temperature.reversion_rate"],
            "assertion_audit": {"structural_passed": True, "op": "stochastic_sdae_check"},
            "execution": {"device": "cpu", "step_count": 2, "execution_order": "stochastic_air_temperature"},
        }

        dataset = _convert_results_to_dataset(results)

        assert dataset[f"trajectory_{STATE}"].dims == ("sample", "time")
        assert dataset[f"trajectory_{STATE}"].attrs["units"] == "K"
        assert "standard_name" not in dataset[f"trajectory_{STATE}"].attrs
        assert dataset[f"ensemble_mean_{STATE}"].dims == ("time",)
        assert dataset[f"ensemble_variance_{STATE}"].attrs["units"] == "K2"
        # No CF "time" coordinate: the relative clock lives in time_seconds.
        assert "time" not in dataset.coords
        assert dataset["time_seconds"].dims == ("time",)
        assert dataset["time_seconds"].attrs["units"] == "s"
        assert np.array_equal(dataset["time_seconds"].values, np.array([0.0, 3600.0, 7200.0]))
        # Textual metadata is global attributes; numeric/boolean scalars are
        # 0-D variables (CF data variables never carry string payloads).
        assert dataset.attrs["parameter_keys"] == "stochastic.weather.temperature.reversion_rate"
        assert dataset.attrs["process"] == "stochastic_air_temperature"
        assert dataset.attrs["assertion_audit_op"] == "stochastic_sdae_check"
        assert dataset.attrs["execution_device"] == "cpu"
        assert bool(dataset["assertion_audit_structural_passed"].values) is True
        assert int(dataset["execution_step_count"].values) == 2
        assert int(dataset["seed"].values) == 7

    def test_declared_arrays_of_different_lengths_do_not_collide(self):
        dataset = _convert_results_to_dataset(
            {
                "short": torch.zeros(1, dtype=torch.float64),
                "long": torch.zeros(3, dtype=torch.float64),
                "variable_units": {"short": "1", "long": "m"},
                "variable_dimensions": {"short": ["short_index"], "long": ["long_index"]},
                "labels": ["a", "b"],
                "weights": [0.5, 0.25, 0.25],
            }
        )
        assert dataset["short"].dims == ("short_index",)
        assert dataset["long"].dims == ("long_index",)
        assert dataset["long"].attrs["units"] == "m"
        assert dataset.attrs["labels"] == "a,b"
        assert dataset["weights"].dims == ("weights_dim0",)


class TestDeclaredStateMetadata:
    """Bound results declare their state units (SSOT catalog); nothing is guessed.

    A name substring must not decide the metadata: the rainfall event count is
    a count in '1', not a precipitation_flux in kg m-2 s-1, and the SSOT catalog
    declares no CF standard name for these stochastic states.
    """

    def _results(self):
        return {
            "state_units": {
                "stochastic_rainfall_event_count": "1",
                "stochastic_rainfall_depth": "mm",
            },
            "trajectory_stochastic_rainfall_event_count": torch.zeros((2, 3)),
            "trajectory_stochastic_rainfall_depth": torch.zeros((2, 3)),
            "ensemble_mean_stochastic_rainfall_depth": torch.zeros(3),
            "ensemble_variance_stochastic_rainfall_depth": torch.zeros(3),
            "analytic_variance_stochastic_rainfall_event_count": torch.zeros(3),
        }

    def test_declared_units_replace_name_based_guesses(self):
        dataset = _convert_results_to_dataset(self._results())

        count = dataset["trajectory_stochastic_rainfall_event_count"]
        depth = dataset["trajectory_stochastic_rainfall_depth"]
        assert count.attrs["units"] == "1"
        assert depth.attrs["units"] == "mm"
        assert dataset["ensemble_mean_stochastic_rainfall_depth"].attrs["units"] == "mm"
        assert "standard_name" not in count.attrs
        assert "standard_name" not in depth.attrs

    def test_variance_of_a_declared_state_has_squared_units(self):
        dataset = _convert_results_to_dataset(self._results())

        assert dataset["ensemble_variance_stochastic_rainfall_depth"].attrs["units"] == "mm2"
        assert dataset["analytic_variance_stochastic_rainfall_event_count"].attrs["units"] == "1"
        assert "standard_name" not in dataset["ensemble_variance_stochastic_rainfall_depth"].attrs

    def test_compound_declared_unit_is_squared_per_factor(self):
        results = {
            "state_units": {"flux_state": "kg m-2 s-1"},
            "ensemble_variance_flux_state": torch.zeros(3),
        }
        dataset = _convert_results_to_dataset(results)
        assert dataset["ensemble_variance_flux_state"].attrs["units"] == "kg2 m-4 s-2"
