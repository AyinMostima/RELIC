"""L1 tests for the assimilation base contracts (relic/optimization/assimilation/base.py).

The identity observation operator reports only declared state values (a
missing variable is an error, never an observation of zero), and the base
configuration offers only settings that some assimilator reads.
"""

from __future__ import annotations

import pytest
import torch

from relic.optimization.assimilation import AssimilationConfig, IdentityOperator
from relic.optimization.assimilation.base import AssimilationMethod, AssimilationResult, Observation


def test_identity_operator_rejects_a_missing_state_variable() -> None:
    operator = IdentityOperator(variable_names=["GPP", "ET"])

    with pytest.raises(KeyError, match="ET"):
        operator.forward({"GPP": torch.tensor([10.0])})


def test_identity_operator_extracts_declared_variables_in_order() -> None:
    operator = IdentityOperator(variable_names=["ET", "GPP"])

    observed = operator.forward({"GPP": torch.tensor([10.0]), "ET": torch.tensor([2.0])})

    assert torch.equal(observed, torch.tensor([2.0, 10.0]))


@pytest.mark.parametrize(
    "field_name, value",
    [
        ("inflation_factor", 1.05),
        ("localization_radius", 50.0),
        ("observation_error_threshold", 4.0),
        ("innovation_threshold", 2.0),
        ("parallel_forecasts", True),
        ("save_ensemble", True),
        ("save_diagnostics", False),
    ],
)
def test_settings_that_no_assimilator_reads_are_not_accepted(field_name, value) -> None:
    with pytest.raises(TypeError):
        AssimilationConfig(method=AssimilationMethod.ENSEMBLE_KALMAN_FILTER, **{field_name: value})


# ---------------------------------------------------------------------------
# Round 5: no literal defaults. The method is chosen by the caller, the
# ensemble size comes from optimization.assimilation.ensemble_size (YAML), an
# observation declares its quality flag, and a result reports every field it
# carries (no fabricated zero).
# ---------------------------------------------------------------------------


def test_the_assimilation_method_is_required() -> None:
    with pytest.raises(TypeError):
        AssimilationConfig()
    config = AssimilationConfig(method=AssimilationMethod.VARIATIONAL_3D)
    assert config.method is AssimilationMethod.VARIATIONAL_3D


def test_the_ensemble_size_is_read_from_the_parameter_database() -> None:
    from relic.parameters import get_global_manager

    key = "optimization.assimilation.ensemble_size"
    manager = get_global_manager()
    original = manager.get(key, required=True)
    assert int(original) == 50
    assert AssimilationConfig(method=AssimilationMethod.ENSEMBLE_KALMAN_FILTER).ensemble_size == 50
    assert AssimilationConfig(method=AssimilationMethod.ENSEMBLE_KALMAN_FILTER, ensemble_size=7).ensemble_size == 7
    try:
        manager.set_value(key, 12)
        assert AssimilationConfig(method=AssimilationMethod.ENSEMBLE_KALMAN_FILTER).ensemble_size == 12
    finally:
        manager.set_value(key, original)


def _observation_fields():
    return dict(
        value=torch.tensor([1.0]),
        error_variance=torch.tensor([1.0]),
        coordinates=(0.0, 0.0, 0.0),
        time=0.0,
        variable_name="GPP",
    )


def test_an_observation_declares_its_quality_flag_and_has_no_unchecked_units() -> None:
    with pytest.raises(TypeError):
        Observation(**_observation_fields())
    assert Observation(**_observation_fields(), quality_flag=1).quality_flag == 1
    with pytest.raises(TypeError):
        Observation(**_observation_fields(), quality_flag=0, units="1")


def test_an_assimilation_result_reports_every_diagnostic_it_carries() -> None:
    fields = dict(
        analysis_mean={"GPP": torch.tensor([1.0])},
        rms_innovation=0.5,
        assimilation_time=0.0,
        computational_cost=0.1,
        observations_assimilated=1,
        observations_rejected=0,
    )
    result = AssimilationResult(**fields)
    assert result.rms_innovation == 0.5
    for name in ("rms_innovation", "assimilation_time", "computational_cost",
                 "observations_assimilated", "observations_rejected"):
        partial = dict(fields)
        partial.pop(name)
        with pytest.raises(TypeError):
            AssimilationResult(**partial)
    for removed in ("degrees_of_freedom", "analysis_error_reduction"):
        assert not hasattr(result, removed)
        with pytest.raises(TypeError):
            AssimilationResult(**fields, **{removed: 0})
