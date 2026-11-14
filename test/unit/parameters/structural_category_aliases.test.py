"""Tests for structural parameter category aliases."""

import pytest

from relic.parameters.category_aliases import normalize_category
from relic.parameters.serialization import deserialize_parameter_spec
from relic.parameters.types import ParameterCategory


def test_structural_category_aliases_map_to_existing_parameter_categories():
    assert normalize_category("hodge") == ParameterCategory.PHYSICAL
    assert normalize_category("onsager") == ParameterCategory.PHYSICAL
    assert normalize_category("metric") == ParameterCategory.MODEL_CONFIGURATION
    assert normalize_category("port") == ParameterCategory.PROCESS_COUPLING
    assert normalize_category("structural_tolerance") == ParameterCategory.VALIDATION_PARAMETER


@pytest.mark.parametrize("category", ["", "unknown_category", "biophyscial", None, 3, False, [], {}])
def test_invalid_parameter_category_is_not_replaced_with_a_default(category):
    with pytest.raises((TypeError, ValueError), match="[Cc]ategory"):
        normalize_category(category)


@pytest.mark.parametrize("category", [ParameterCategory.COMPUTATIONAL, ParameterCategory.CALIBRATION])
def test_canonical_parameter_category_accepts_case_and_surrounding_whitespace(category):
    assert normalize_category("  " + category.value.upper() + "  ") is category


def test_public_spec_deserializer_rejects_unknown_category():
    with pytest.raises(ValueError, match="[Cc]ategory"):
        deserialize_parameter_spec({"key": "test.category", "value": 1.0, "category": "biophyscial"})
