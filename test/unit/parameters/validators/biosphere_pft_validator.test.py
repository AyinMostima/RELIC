"""L1 tests for the PFT collection validation boundary.

Module under test: relic/parameters/validators/biosphere_pft.py. A PFT whose
parameter data cannot be evaluated gets a failed PFTValidationResult; an error
of the validator implementation propagates instead of being reported as a PFT
finding.
"""

from __future__ import annotations

import pytest

from relic.parameters.validators.biosphere_pft import (
    BiospherePFTValidator,
    PFTValidationLevel,
    PFTValidationResult,
)


def _validator(monkeypatch, error: Exception) -> BiospherePFTValidator:
    validator = BiospherePFTValidator(PFTValidationLevel.MODERATE)

    def _validate_pft(pft):
        if pft == "valid_pft":
            return PFTValidationResult(True, [], [], {"photosynthesis.vcmax25": True})
        raise error

    monkeypatch.setattr(validator, "validate_pft", _validate_pft)
    return validator


@pytest.mark.parametrize(
    "error",
    [TypeError("'<' not supported between 'NoneType' and 'float'"), ValueError("could not convert")],
    ids=["type", "value"],
)
def test_unevaluable_pft_data_gives_a_failed_result(monkeypatch, error: Exception) -> None:
    validator = _validator(monkeypatch, error)

    results = validator.validate_pft_collection({"valid": "valid_pft", "broken": "broken_pft"})

    assert results["valid"].is_valid is True
    assert results["broken"].is_valid is False
    assert results["broken"].errors == [f"Validation exception: {error}"]


@pytest.mark.parametrize(
    "error",
    [KeyError("condition"), RuntimeError("validator state corrupted")],
    ids=["key", "runtime"],
)
def test_validator_implementation_error_propagates(monkeypatch, error: Exception) -> None:
    validator = _validator(monkeypatch, error)

    with pytest.raises(type(error)):
        validator.validate_pft_collection({"broken": "broken_pft"})
