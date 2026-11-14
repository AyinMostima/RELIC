"""L1 tests for PhysicalConsistencyValidator handling of non-numeric values.

Module under test: relic/parameters/validators/range_validator.py. A value of a
checked parameter that is not numeric makes the validation fail; it is not
skipped so that the remaining values read as valid.
"""

from __future__ import annotations

import pytest
import torch

from relic.parameters.validators.range_validator import PhysicalConsistencyValidator


@pytest.fixture
def validator() -> PhysicalConsistencyValidator:
    return PhysicalConsistencyValidator()


class TestAllocationFractions:
    @pytest.mark.parametrize(
        "bad_value",
        [None, "forty percent", torch.tensor([0.2, 0.2])],
        ids=["none", "text", "multi-element-tensor"],
    )
    def test_non_numeric_fraction_fails_the_validation(self, validator, bad_value) -> None:
        result = validator.validate_allocation_fractions(
            {"allocation.leaf_fraction": bad_value, "allocation.root_fraction": 1.0}
        )

        assert result.is_valid is False
        assert any("allocation.leaf_fraction" in error for error in result.errors)

    def test_numeric_fractions_summing_to_one_are_valid(self, validator) -> None:
        result = validator.validate_allocation_fractions(
            {
                "allocation.leaf_fraction": torch.tensor(0.4),
                "allocation.root_fraction": 0.6,
                "pft.name": "temperate oak",
            }
        )

        assert result.is_valid is True
        assert result.errors == []


class TestTemperatureResponses:
    def test_non_numeric_activation_energy_fails_the_validation(self, validator) -> None:
        result = validator.validate_temperature_responses(
            {"photosynthesis.vcmax_ea": "sixty-five kJ"}
        )

        assert result.is_valid is False
        assert any("photosynthesis.vcmax_ea" in error for error in result.errors)

    def test_non_numeric_unrelated_parameter_is_not_a_finding(self, validator) -> None:
        result = validator.validate_temperature_responses(
            {"pft.name": "temperate oak", "photosynthesis.vcmax_ea": 65330.0}
        )

        assert result.is_valid is True
        assert result.errors == []
