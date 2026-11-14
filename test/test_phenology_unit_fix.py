"""
Test for phenology parameter unit conversion fix.

This test verifies that turnover rate parameters are correctly validated
after converting from year-1 to s-1 units.

Reference:
    - Issue: ValidationError for turnover rates due to unit mismatch
    - Fix: Adjusted Pydantic Field bounds from year-1 to s-1
"""

import pytest
from relic.scientific.biosphere.phenology.parameters import (
    PhenologyParameters,
    PhenologyParameterSets,
)
from relic.parameters import get_global_manager


def _parameters(**overrides):
    """Build a complete explicit payload for focused tests: the preset of the requested
    model (gsi unless phenology_type is given), which holds exactly that model's fields,
    with the overrides applied."""
    presets = {
        "gsi": PhenologyParameterSets.deciduous_tree_gsi,
        "static": PhenologyParameterSets.evergreen_tree_static,
        "gdd": PhenologyParameterSets.grassland_gdd,
        "crop": PhenologyParameterSets.maize_crop,
    }
    kind = overrides.get("phenology_type", "gsi")
    values = {
        name: value
        for name, value in presets[kind]().model_dump().items()
        if value is not None
    }
    values.update(overrides)
    return PhenologyParameters(**values)


class TestPhenologyUnitConversion:
    """Test suite for phenology parameter unit conversion fix."""

    def test_turnover_rate_validation_with_si_units(self):
        """
        Test that turnover rates in s-1 units pass validation.

        Mathematical Expression:
            rate_s1 = rate_year1 / 31536000
            where 31536000 = seconds per year

        Expected behavior:
            - leaf_turnover_rate: 7.927e-9 s-1 (0.25 year-1) should pass
            - froot_turnover_rate: 3.171e-8 s-1 (1.0 year-1) should pass
        """
        param_manager = get_global_manager()

        # Get parameters from database (already in s-1)
        leaf_rate = param_manager.get(
            "biosphere.phenology.turnover.leaf_turnover_rate",
            default=7.927e-9
        )
        froot_rate = param_manager.get(
            "biosphere.phenology.turnover.froot_turnover_rate",
            default=3.171e-8
        )

        # Create PhenologyParameters instance
        # This should not raise ValidationError anymore
        params = _parameters(
            leaf_turnover_rate=leaf_rate,
            froot_turnover_rate=froot_rate,
        )

        # Verify values are within SI bounds
        assert 3.171e-9 <= params.leaf_turnover_rate <= 6.342e-8, \
            f"Leaf turnover rate {params.leaf_turnover_rate} outside valid s-1 range"
        assert 1.585e-8 <= params.froot_turnover_rate <= 9.512e-8, \
            f"Fine root turnover rate {params.froot_turnover_rate} outside valid s-1 range"

    def test_unit_conversion_accuracy(self):
        """
        Test that unit conversion from year-1 to s-1 is mathematically correct.

        Mathematical Expression:
            SECONDS_PER_YEAR = 365.25 * 24 * 3600 = 31557600 (astronomical year)
            SECONDS_PER_YEAR = 365 * 24 * 3600 = 31536000 (calendar year)

        We use calendar year for consistency.
        """
        SECONDS_PER_YEAR = 31536000  # Calendar year

        # Test conversions
        conversions = [
            (0.25, 7.927e-9),   # leaf_turnover_rate
            (1.0, 3.171e-8),    # froot_turnover_rate
            (0.02, 6.342e-10),  # wood_turnover_rate
        ]

        for rate_year, rate_si in conversions:
            calculated_si = rate_year / SECONDS_PER_YEAR
            # Allow 1% tolerance for floating point
            relative_error = abs(calculated_si - rate_si) / rate_si
            assert relative_error < 0.01, \
                f"Conversion error: {rate_year} year-1 = {calculated_si} s-1, expected {rate_si} s-1"

    def test_boundary_validation(self):
        """Test that boundary values are correctly enforced in s-1 units."""
        # Test minimum boundary (should pass)
        params_min = _parameters(
            leaf_turnover_rate=3.171e-9,   # 0.1 year-1
            froot_turnover_rate=1.585e-8,  # 0.5 year-1
        )
        assert params_min.leaf_turnover_rate == 3.171e-9

        # Test maximum boundary (should pass)
        params_max = _parameters(
            leaf_turnover_rate=6.342e-8,   # 2.0 year-1
            froot_turnover_rate=9.512e-8,  # 3.0 year-1
        )
        assert params_max.froot_turnover_rate == 9.512e-8

        # Test below minimum (should fail)
        with pytest.raises(ValueError, match="greater than or equal to"):
            _parameters(
                leaf_turnover_rate=1.0e-9  # Too small
            )

        # Test above maximum (should fail)
        with pytest.raises(ValueError, match="less than or equal to"):
            _parameters(
                froot_turnover_rate=1.0e-7  # Too large
            )

    def test_default_values_from_parameter_manager(self):
        """Test that default values from ParameterManager are valid."""
        param_manager = get_global_manager()

        # Load all turnover parameters
        turnover_params = {
            "leaf": param_manager.get("biosphere.phenology.turnover.leaf_turnover_rate"),
            "froot": param_manager.get("biosphere.phenology.turnover.froot_turnover_rate"),
        }

        # All should be non-None and valid
        for name, value in turnover_params.items():
            assert value is not None, f"{name}_turnover_rate not found in ParameterManager"
            assert isinstance(value, (int, float)), f"{name}_turnover_rate is not numeric"
            assert value > 0, f"{name}_turnover_rate must be positive"

    def test_scientific_reasonableness(self):
        """
        Test that converted values are scientifically reasonable.

        References:
            - Leaf turnover: Reich et al. (1997) PNAS 94:13730-13734
            - Fine root turnover: Gill & Jackson (2000) Oecologia 122:336-342
        """
        params = _parameters()

        # Convert back to year-1 for readability
        SECONDS_PER_YEAR = 31536000

        leaf_year = params.leaf_turnover_rate * SECONDS_PER_YEAR
        froot_year = params.froot_turnover_rate * SECONDS_PER_YEAR

        # Check literature ranges
        # Leaves: 0.1-2.0 year-1 (deciduous ~0.25-1.0, evergreen ~0.1-0.3)
        assert 0.1 <= leaf_year <= 2.0, "Leaf turnover outside literature range"

        # Fine roots: 0.5-3.0 year-1 (typically 0.5-1.5)
        assert 0.5 <= froot_year <= 3.0, "Fine root turnover outside literature range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

