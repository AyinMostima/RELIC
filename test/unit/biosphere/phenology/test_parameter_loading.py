"""
Unit tests for phenology parameter loading from ParameterManager.

This test module validates that all phenology parameters are correctly
loaded from the centralized ParameterManager instead of hardcoded defaults,
ensuring the fixes from the parameter unification effort are working properly.

Test Coverage:
    - L1: PhenologyParameterSets preset loading without default
    - L1: PhenologyParameters default factory loading 
    - L1: Unit consistency and validation
    - L1: Parameter namespace verification
    
References:
    - CF-1.8 Convention for units
    - RELIC parameter architecture documentation
"""

import pytest
import torch
from pydantic import ValidationError
from typing import Dict, Any
import relic.scientific.biosphere.phenology.parameters as phenology_parameters
from relic.scientific.biosphere.phenology.parameters import (
    PhenologyParameters,
    PhenologyParameterSets,
)
from relic.parameters import get_global_manager


def _valid_parameter_values() -> Dict[str, Any]:
    """Return a complete explicit parameter payload for validation tests."""
    return PhenologyParameterSets.deciduous_tree_gsi().model_dump()


class TestPhenologyParameterSetsLoading:
    """Test PhenologyParameterSets loads from ParameterManager correctly."""
    
    def setup_method(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        # Ensure phenology parameters are loaded
        self.param_manager.load_from_database("biosphere")
        
    def test_deciduous_tree_gsi_preset_loading(self):
        """Test deciduous tree GSI preset loads from ParameterManager."""
        # Get preset parameters
        params = PhenologyParameterSets.deciduous_tree_gsi()
        
        # Verify basic attributes
        assert params.phenology_type == "gsi"
        
        # Verify GSI-specific parameters are loaded
        assert params.gsi_tmin_threshold is not None
        assert params.gsi_tmin_optimal is not None
        assert params.gsi_vpd_min is not None
        assert params.gsi_vpd_max is not None
        assert params.gsi_onset_threshold is not None
        assert params.gsi_offset_threshold is not None
        
        # Verify values match what's in ParameterManager
        namespace = "biosphere.phenology.presets.deciduous_tree_gsi"
        assert params.gsi_tmin_threshold == self.param_manager.get(f"{namespace}.gsi_tmin_threshold")
        assert params.gsi_tmin_optimal == self.param_manager.get(f"{namespace}.gsi_tmin_optimal")
        assert params.gsi_vpd_min == self.param_manager.get(f"{namespace}.gsi_vpd_min")
        assert params.gsi_vpd_max == self.param_manager.get(f"{namespace}.gsi_vpd_max")
        
    def test_evergreen_tree_static_preset_loading(self):
        """Test evergreen tree static preset loads from ParameterManager."""
        params = PhenologyParameterSets.evergreen_tree_static()
        
        assert params.phenology_type == "static"
        
        # Verify static parameters
        assert params.leaf_onset_doy is not None
        assert params.leaf_offset_doy is not None
        
        # Verify values match ParameterManager
        namespace = "biosphere.phenology.presets.evergreen_tree_static"
        assert params.leaf_onset_doy == self.param_manager.get(f"{namespace}.leaf_onset_doy")
        assert params.leaf_offset_doy == self.param_manager.get(f"{namespace}.leaf_offset_doy")
        
    def test_grassland_gdd_preset_loading(self):
        """Test grassland GDD preset loads from ParameterManager."""
        params = PhenologyParameterSets.grassland_gdd()
        
        assert params.phenology_type == "gdd"
        
        # Verify GDD parameters
        assert params.gdd_base_temp is not None
        assert params.gdd_threshold_start is not None
        assert params.gdd_threshold_end is not None
        
        # Verify values match ParameterManager
        namespace = "biosphere.phenology.presets.grassland_gdd"
        assert params.gdd_base_temp == self.param_manager.get(f"{namespace}.gdd_base_temp")
        assert params.gdd_threshold_start == self.param_manager.get(f"{namespace}.gdd_threshold_start")
        
    def test_winter_wheat_crop_preset_loading(self):
        """Test winter wheat crop preset loads from ParameterManager."""
        params = PhenologyParameterSets.winter_wheat_crop()
        
        assert params.phenology_type == "crop"
        assert params.crop_type == "wheat"
        
        # Verify crop parameters
        assert params.crop_phu_total is not None
        assert params.crop_hi_max is not None
        assert params.crop_sowing_doy is not None
        assert params.crop_harvest_doy is not None
        assert params.crop_vernalization_required is not None
        
        # Verify values match ParameterManager
        namespace = "biosphere.phenology.presets.winter_wheat_crop"
        assert params.crop_type == self.param_manager.get(f"{namespace}.crop_type")
        assert params.crop_phu_total == self.param_manager.get(f"{namespace}.crop_phu_total")
        assert self.param_manager.get_parameter(
            f"{namespace}.crop_phu_total"
        ).units == "degC day"
        
    def test_maize_crop_preset_loading(self):
        """Test maize crop preset loads from ParameterManager."""
        params = PhenologyParameterSets.maize_crop()
        
        assert params.phenology_type == "crop"
        assert params.crop_type == "maize"
        
        # Verify values match ParameterManager
        namespace = "biosphere.phenology.presets.maize_crop"
        assert params.crop_type == self.param_manager.get(f"{namespace}.crop_type")
        assert params.crop_phu_total == self.param_manager.get(f"{namespace}.crop_phu_total")
        assert params.crop_hi_max == self.param_manager.get(f"{namespace}.crop_hi_max")
        assert self.param_manager.get_parameter(
            f"{namespace}.crop_phu_total"
        ).units == "degC day"
        
    def test_preset_loading_with_overrides(self):
        """Test preset loading with parameter overrides."""
        # Load preset with overrides
        overrides = {"max_lai": 10.0, "min_lai": 0.5}
        params = PhenologyParameterSets._load_preset_params("deciduous_tree_gsi", overrides)

        # Verify overrides are applied
        assert params.max_lai == 10.0
        assert params.min_lai == 0.5
        
        # Verify non-overridden params still come from ParameterManager
        assert params.phenology_type == "gsi"
        
    def test_invalid_preset_raises_error(self):
        """Test that invalid preset name raises ValueError."""
        with pytest.raises(ValueError, match="not properly configured"):
            PhenologyParameterSets._load_preset_params("nonexistent_preset")


class TestPhenologyParametersConstruction:
    """Test that the value object never reads ParameterManager implicitly."""
    
    def setup_method(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database("biosphere")
        
    def test_missing_explicit_fields_fail_without_manager_access(self, monkeypatch):
        """A bare value object must fail validation before any global lookup."""
        monkeypatch.setattr(
            phenology_parameters,
            "get_global_manager",
            lambda: (_ for _ in ()).throw(
                AssertionError("PhenologyParameters accessed the global manager")
            ),
        )

        with pytest.raises(ValidationError, match="Field required"):
            PhenologyParameters()

    def test_complete_explicit_payload_needs_no_manager_access(self, monkeypatch):
        """A complete payload constructs independently of ParameterManager."""
        values = _valid_parameter_values()
        monkeypatch.setattr(
            phenology_parameters,
            "get_global_manager",
            lambda: (_ for _ in ()).throw(
                AssertionError("PhenologyParameters accessed the global manager")
            ),
        )

        params = PhenologyParameters(**values)

        assert params.model_dump() == values


class TestParameterValidation:
    """Test parameter validation and constraints."""
    
    def test_leaf_offset_after_onset_validation(self):
        """Test that leaf_offset_doy must be after leaf_onset_doy."""
        values = _valid_parameter_values()
        values.update(leaf_onset_doy=200, leaf_offset_doy=100)
        with pytest.raises(ValueError, match="leaf_offset_doy must be greater than leaf_onset_doy"):
            PhenologyParameters(**values)
            
    def test_gdd_threshold_validation(self):
        """Test GDD threshold validation."""
        values = _valid_parameter_values()
        values.update(gdd_threshold_start=1000.0, gdd_threshold_end=500.0)
        with pytest.raises(ValueError, match="gdd_threshold_end must be greater than gdd_threshold_start"):
            PhenologyParameters(**values)
            
    def test_gsi_temperature_validation(self):
        """Test GSI temperature validation."""
        values = _valid_parameter_values()
        values.update(gsi_tmin_threshold=5.0, gsi_tmin_optimal=2.0)
        with pytest.raises(ValueError, match="gsi_tmin_optimal must be greater than gsi_tmin_threshold"):
            PhenologyParameters(**values)
            
    def test_gsi_vpd_validation(self):
        """Test GSI VPD validation."""
        values = _valid_parameter_values()
        values.update(gsi_vpd_min=4000.0, gsi_vpd_max=2000.0)
        with pytest.raises(ValueError, match="gsi_vpd_max must be greater than gsi_vpd_min"):
            PhenologyParameters(**values)
            
    def test_lai_validation(self):
        """Test LAI validation."""
        values = _valid_parameter_values()
        values.update(min_lai=5.0, max_lai=10.0)
        # Test that min_lai has bounds
        with pytest.raises(ValueError, match="Input should be less than or equal to"):
            PhenologyParameters(**values)


class TestStrictParameterAccess:
    """Test that no compatibility accessor survives in the value-object module."""

    def test_default_helper_is_not_exposed(self):
        """Missing YAML parameters must never be replaced in Python."""
        assert not hasattr(phenology_parameters, "get_param_default")
    
    def test_compatibility_accessor_is_not_exposed(self):
        """ParameterManager reads must use the canonical public manager API."""
        assert not hasattr(phenology_parameters, "get_param_from_manager")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


