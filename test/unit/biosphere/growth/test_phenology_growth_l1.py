"""
L1 Unit Tests for Phenology Growth Parameter Integration

This test module verifies that PhenologyGrowth Functor correctly loads and uses
all parameters from the ParameterManager, with special focus on the P0 photoperiod
unit conversion fix.

Test Coverage:
- Photoperiod parameter loading (CRITICAL: verify seconds to hours conversion)
- GDD (growing degree days) parameters
- Growing season detection parameters
- Phenology state transition parameters
- All parameters loaded from ParameterManager (no hardcoding)
- Functor forward() integration with valid inputs

References:
- Richardson et al. (2013). Climate change, phenology, and phenological control. Agric For Meteorol 169:156-173
- Chuine (2010). Why does phenology drive species distribution? Phil Trans R Soc B 365:3149-3160
- Basler (2016). Evaluating phenological models for the prediction of leaf-out dates. Agric For Meteorol 217:10-21
"""

import pytest
import torch
import sys
import os
from pathlib import Path

# Add relic to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from relic.parameters import get_global_manager
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth


@pytest.fixture(scope="module")
def param_manager():
    """Get global parameter manager instance."""
    return get_global_manager()


def _make_functor(**kwargs):
    """Build the observer against one explicit canonical phenology owner."""
    kwargs.setdefault("phenology_model_owner", "gsi_phenology")
    return PhenologyGrowth(**kwargs)


# =============================================================================
# Photoperiod Parameters (CRITICAL: P0 Fix Verification)
# =============================================================================

class TestPhotoperiodParameters:
    """Test photoperiod parameter loading and unit conversion (P0 critical fix)."""

    def test_photoperiod_threshold_exists(self, param_manager):
        """Verify daylen_threshold parameter exists in phenology.gsi namespace."""
        value = param_manager.get("biosphere.phenology.gsi.daylen_threshold")
        assert value is not None, "daylen_threshold parameter not found"
        # Expected: 39300 seconds (10.92 hours)
        assert value == 39300.0, f"Expected 39300.0 seconds, got {value}"

    def test_photoperiod_threshold_units(self, param_manager):
        """Verify daylen_threshold has correct units (seconds)."""
        spec = param_manager.get_parameter("biosphere.phenology.gsi.daylen_threshold")
        assert spec is not None, "daylen_threshold spec not found"
        assert spec.units == "s", f"Expected units 's', got '{spec.units}'"

    def test_observer_does_not_recompute_photoperiod(self):
        """Photoperiod belongs to the selected main phenology closure."""
        functor = _make_functor()

        assert "daylength" not in functor.metadata.input_variables
        assert not hasattr(functor, "daylength_critical")


# =============================================================================
# GDD (Growing Degree Days) Parameters
# =============================================================================

class TestGDDParameters:
    """Test GDD accumulation and threshold parameters."""

    def test_base_temperature_exists(self, param_manager):
        """Verify base_temp parameter exists."""
        value = param_manager.get("biosphere.phenology.gdd.base_temp")
        assert value is not None, "base_temp parameter not found"
        # Expected: 278.15 K (5 degC)
        assert value == 278.15, f"Expected 278.15 K, got {value}"

    def test_base_temperature_units(self, param_manager):
        """Verify base_temp has correct units (Kelvin)."""
        spec = param_manager.get_parameter("biosphere.phenology.gdd.base_temp")
        assert spec is not None, "base_temp spec not found"
        assert spec.units == "K", f"Expected units 'K', got '{spec.units}'"

    def test_gdd_threshold_start_exists(self, param_manager):
        """Verify threshold_start parameter exists."""
        value = param_manager.get("biosphere.phenology.gdd.threshold_start")
        assert value is not None, "threshold_start parameter not found"
        # Expected: 200 degC*day for budburst
        assert value == 200.0, f"Expected 200.0 degC*day, got {value}"

    def test_gdd_full_growth_offset_exists(self, param_manager):
        """Verify full_growth_offset parameter exists."""
        value = param_manager.get("biosphere.phenology.gdd.full_growth_offset")
        assert value is not None, "full_growth_offset parameter not found"
        # Expected: 300 degC*day offset from start to full leaf
        assert value == 300.0, f"Expected 300.0 degC*day, got {value}"

    def test_gdd_threshold_end_exists(self, param_manager):
        """Verify threshold_end parameter exists."""
        value = param_manager.get("biosphere.phenology.gdd.threshold_end")
        assert value is not None, "threshold_end parameter not found"
        # Expected: 1200 degC*day for senescence
        assert value == 1200.0, f"Expected 1200.0 degC*day, got {value}"

    def test_all_gdd_parameters_have_correct_units(self, param_manager):
        """Verify all GDD parameters have correct units."""
        params_units = {
            "biosphere.phenology.gdd.base_temp": "K",
            "biosphere.phenology.gdd.threshold_start": "degC_day",
            "biosphere.phenology.gdd.full_growth_offset": "degC_day",
            "biosphere.phenology.gdd.threshold_end": "degC_day",
        }

        for param_key, expected_units in params_units.items():
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"Parameter spec not found for {param_key}"
            actual_units = spec.units
            assert actual_units == expected_units, \
                f"{param_key}: expected units '{expected_units}', got '{actual_units}'"


# =============================================================================
# Growing Season Detection Parameters
# =============================================================================

class TestGrowingSeasonParameters:
    """Test growing season start/end detection parameters."""

    def test_temperature_opt_exists(self, param_manager):
        """Verify temperature_opt parameter exists."""
        value = param_manager.get("biosphere.growth.allocation.temperature_opt")
        assert value is not None, "temperature_opt parameter not found"
        # Expected: 298.15 K (25 degC)
        assert value == 298.15, f"Expected 298.15 K, got {value}"

    def test_temperature_range_tolerance_exists(self, param_manager):
        """Verify temperature_range_tolerance parameter exists."""
        value = param_manager.get("biosphere.growth.defaults.temperature_range_tolerance")
        assert value is not None, "temperature_range_tolerance parameter not found"
        # Expected: 10.0 K
        assert value == 10.0, f"Expected 10.0 K, got {value}"

    def test_growing_season_parameters_have_correct_units(self, param_manager):
        """Verify all growing season parameters have correct units."""
        params_units = {
            "biosphere.growth.allocation.temperature_opt": "K",
            "biosphere.growth.defaults.temperature_range_tolerance": "K",
        }

        for param_key, expected_units in params_units.items():
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"Parameter spec not found for {param_key}"
            actual_units = spec.units
            assert actual_units == expected_units, \
                f"{param_key}: expected units '{expected_units}', got '{actual_units}'"


# =============================================================================
# Phenology State Transition Parameters
# =============================================================================

class TestPhenologyStateTransitionParameters:
    """Test phenology state transition control parameters."""

    def test_phenology_clamp_min_exists(self, param_manager):
        """Verify phenology_clamp_min parameter exists."""
        value = param_manager.get("biosphere.growth.allocation.phenology_clamp_min")
        assert value is not None, "phenology_clamp_min parameter not found"
        assert value == 0.0, f"Expected 0.0, got {value}"

    def test_phenology_clamp_max_exists(self, param_manager):
        """Verify phenology_clamp_max parameter exists."""
        value = param_manager.get("biosphere.growth.allocation.phenology_clamp_max")
        assert value is not None, "phenology_clamp_max parameter not found"
        assert value == 1.0, f"Expected 1.0, got {value}"

    def test_growth_active_threshold_exists(self, param_manager):
        """Verify growth_active_threshold parameter exists."""
        value = param_manager.get("biosphere.growth.allocation.growth_active_threshold")
        assert value is not None, "growth_active_threshold parameter not found"
        assert value == 0.1, f"Expected 0.1, got {value}"

    def test_phenology_control_parameters_have_correct_units(self, param_manager):
        """Verify all phenology control parameters have correct units."""
        params_units = {
            "biosphere.growth.allocation.phenology_clamp_min": "dimensionless",
            "biosphere.growth.allocation.phenology_clamp_max": "dimensionless",
            "biosphere.growth.allocation.growth_active_threshold": "dimensionless",
        }

        for param_key, expected_units in params_units.items():
            spec = param_manager.get_parameter(param_key)
            assert spec is not None, f"Parameter spec not found for {param_key}"
            actual_units = spec.units
            assert actual_units == expected_units, \
                f"{param_key}: expected units '{expected_units}', got '{actual_units}'"


# =============================================================================
# PhenologyGrowth Functor Integration Tests
# =============================================================================

class TestPhenologyGrowthFunctorIntegration:
    """Test PhenologyGrowth Functor loads all parameters and computes correctly."""

    def test_canonical_selected_owner_observer_contract(self):
        """The helper observes the selected main phenology instead of replacing it."""
        functor = _make_functor()

        assert functor.name == "phenology_growth"
        assert functor.metadata.input_variables == ["phenological_factor"]
        assert functor.metadata.output_variables == [
            "diagnostics.phenology.growth_active"
        ]
        assert functor.get_dependencies() == ["gsi_phenology"]

        result = functor.forward(
            {"phenological_factor": torch.tensor(0.2)},
            time=0.0,
            dt=86400.0,
        )
        assert set(result) == {"diagnostics.phenology.growth_active"}
        assert "phenological_factor" not in result
        assert "growth_active" not in result
        item = result["diagnostics.phenology.growth_active"]
        assert item.mode is UpdateMode.REPLACE
        assert item.per_timestep is False
        assert item.unit == "dimensionless"
        assert item.source_functor == "phenology_growth"
        assert item.value.dtype is torch.bool
        assert item.value.item() is True

    def test_source_is_single_selected_owner_observation_runtime(self):
        """The module contains no fifth phenology formula or compatibility branch."""
        source_path = (
            Path(__file__).parents[4]
            / "relic"
            / "scientific"
            / "biosphere"
            / "growth"
            / "phenology_growth.py"
        )
        source = source_path.read_text(encoding="utf-8")

        forbidden_markers = [
            "SKELETAL IMPLEMENTATION",
            "DO NOT use in production",
            "legacy\" + \"_mode",
            "legacy_mode",
            "accumulated_gdd",
            "gdd_budburst",
            "daylength_critical",
            "temperature_optimal",
            "_compute_rate_core",
        ]
        for marker in forbidden_markers:
            assert marker not in source

    def test_functor_initialization(self):
        """Only the canonical runtime writer name is accepted."""
        functor = _make_functor()
        assert functor.name == "phenology_growth"
        with pytest.raises(ValueError, match="canonical"):
            PhenologyGrowth(
                name="test_phenology_growth",
                phenology_model_owner="gsi_phenology",
            )

    def test_missing_main_phenology_owner_fails_fast(self):
        """The dependency owner must be selected explicitly at composition time."""
        with pytest.raises(ValueError, match="phenology_model_owner"):
            PhenologyGrowth()

    @pytest.mark.parametrize(
        "unsupported",
        [
            {"pft_name": "broadleaf_deciduous_temperate"},
            {"compatibility_mode": True},
        ],
    )
    def test_unsupported_constructor_options_fail_fast(self, unsupported):
        """Unused PFT and compatibility options cannot be silently ignored."""
        with pytest.raises(TypeError):
            PhenologyGrowth(
                phenology_model_owner="gsi_phenology",
                **unsupported,
            )

    def test_functor_loads_all_parameters(self):
        """Only the calibrated diagnostic threshold belongs to this observer."""
        functor = _make_functor()

        assert functor.growth_active_threshold == pytest.approx(0.1)
        assert not hasattr(functor, "clamp_min")
        assert not hasattr(functor, "clamp_max")

    @pytest.mark.parametrize(
        "owner",
        [
            "static_phenology",
            "gdd_phenology",
            "gsi_phenology",
            "crop_phenology",
        ],
    )
    def test_selected_main_phenology_is_the_only_dependency(self, owner):
        """The observer follows one compile-time-selected main closure."""
        functor = PhenologyGrowth(phenology_model_owner=owner)
        assert functor.get_dependencies() == [owner]

    def test_unknown_main_phenology_owner_fails_fast(self):
        """A fifth closure cannot enter through an unchecked owner string."""
        with pytest.raises(ValueError, match="phenology_model_owner"):
            PhenologyGrowth(phenology_model_owner="phenology_growth")

    def test_forward_with_canonical_inputs(self):
        """The observer exposes one namespaced Boolean diagnostic."""
        functor = _make_functor()
        state = {"phenological_factor": torch.tensor(0.2, dtype=torch.float64)}

        result = functor.forward(state, time=0.0, dt=86400.0)

        assert set(result) == {"diagnostics.phenology.growth_active"}
        assert set(result) == set(functor.metadata.output_variables)
        assert "phenological_factor" not in result
        assert "growth_active" not in result
        assert "rate" not in result
        for key, item in result.items():
            assert isinstance(item, UpdateItem), key
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is False
            assert item.unit == "dimensionless"
            assert item.source_functor == "phenology_growth"
        diagnostic = result["diagnostics.phenology.growth_active"].value
        assert diagnostic.dtype is torch.bool
        assert diagnostic.device == state["phenological_factor"].device
        assert diagnostic.item() is True

    def test_missing_required_input_fails_fast(self):
        """A missing selected-owner control must not use a hidden default."""
        functor = _make_functor()
        with pytest.raises(KeyError, match="phenological_factor"):
            functor.forward({}, time=0.0, dt=86400.0)

    @pytest.mark.parametrize("value", [0.2, True, 1])
    def test_non_tensor_input_fails_fast(self, value):
        """Scientific state input must already be a Tensor."""
        functor = _make_functor()
        with pytest.raises(TypeError, match="torch.Tensor"):
            functor.forward(
                {"phenological_factor": value},
                time=0.0,
                dt=86400.0,
            )

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (float("nan"), "finite"),
            (float("inf"), "finite"),
            (-1.0e-6, "between"),
            (1.000001, "between"),
        ],
    )
    def test_invalid_inputs_fail_fast(self, value, message):
        """Invalid selected-owner controls must not be clamped silently."""
        functor = _make_functor()
        with pytest.raises(ValueError, match=message):
            functor.forward(
                {"phenological_factor": torch.tensor(value)},
                time=0.0,
                dt=86400.0,
            )

    @pytest.mark.parametrize(
        ("time", "dt", "message"),
        [
            (float("nan"), 1.0, "time"),
            (0.0, float("inf"), "dt"),
            (0.0, 0.0, "dt"),
            (0.0, -1.0, "dt"),
        ],
    )
    def test_invalid_time_or_dt_fails_fast(self, time, dt, message):
        """Runtime controls must be finite and dt must be positive."""
        functor = _make_functor()
        with pytest.raises(ValueError, match=message):
            functor.forward(
                {"phenological_factor": torch.tensor(0.2)},
                time=time,
                dt=dt,
            )

    def test_activity_diagnostic_is_monotone_in_selected_owner_factor(self):
        """The Boolean diagnostic cannot create a second phenology trajectory."""
        functor = _make_functor()
        observed = []
        for factor in [0.0, 0.05, 0.1, 0.2, 1.0]:
            result = functor.forward(
                {"phenological_factor": torch.tensor(factor)},
                time=0.0,
                dt=86400.0,
            )
            observed.append(
                int(result["diagnostics.phenology.growth_active"].value.item())
            )

        assert observed == [0, 0, 0, 1, 1]

    def test_growth_activity_uses_strict_parameter_backed_threshold(self):
        """Equality is inactive and the next representable value is active."""
        functor = _make_functor()
        threshold = torch.tensor(functor.growth_active_threshold)
        just_above = torch.nextafter(threshold, torch.tensor(float("inf")))

        at = functor.forward(
            {"phenological_factor": threshold}, time=0.0, dt=1.0
        )["diagnostics.phenology.growth_active"].value
        above = functor.forward(
            {"phenological_factor": just_above}, time=0.0, dt=1.0
        )["diagnostics.phenology.growth_active"].value

        assert at.item() is False
        assert above.item() is True

    def test_metadata_includes_all_parameters(self):
        """Verify Functor metadata includes all parameter definitions."""
        functor = _make_functor()
        metadata = functor.metadata

        expected_params = {"growth_active_threshold"}

        process_params = {
            name
            for name, entry in metadata.parameters.items()
            if entry["namespace"].startswith("biosphere.")
        }
        assert process_params == expected_params

        for param_name in sorted(expected_params):
            assert param_name in metadata.parameters, \
                f"Parameter '{param_name}' should be documented in metadata.parameters"

            param_info = metadata.parameters[param_name]
            assert 'description' in param_info, f"{param_name} should have description"
            assert 'units' in param_info, f"{param_name} should have units"
            assert 'namespace' in param_info, f"{param_name} should have namespace"

    def test_metadata_declares_canonical_observation_operator(self):
        """Verify Phase 8 metadata is an observation map, not a reaction update."""
        functor = _make_functor()
        expression = functor.metadata.mathematical_expression

        assert "observation" in expression.lower()
        assert "phenological_factor" in expression
        assert "no state" in expression.lower()
        assert "strict IPHS" not in expression


# =============================================================================
# Scientific Validation Tests
# =============================================================================

class TestScientificValidation:
    """Test that phenology calculations match scientific literature."""

    def test_response_is_an_honest_phenomenological_diagnostic(self):
        """Metadata does not misstate the binary gate as a universal law."""
        functor = _make_functor()
        text = " ".join(
            [
                functor.metadata.description,
                functor.metadata.mathematical_expression,
            ]
        ).lower()

        assert "phenomenological" in text
        assert "selected" in text
        assert "hamiltonian" not in text
        assert "generic" not in text

    def test_selected_owner_factor_remains_the_only_phenology_trajectory(self):
        """The observer returns no reconstructed phenology factor or stage."""
        functor = _make_functor()
        result = functor.forward(
            {"phenological_factor": torch.tensor(0.8)},
            time=0.0,
            dt=86400.0,
        )

        assert set(result) == {"diagnostics.phenology.growth_active"}
        assert all("gdd" not in key for key in result)
        assert all("phase" not in key for key in result)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

