"""
L1 Unit Tests for Phenology Parameter Integration

This module tests the parameter system integration for the phenology module,
ensuring proper parameter loading, validation, and default mechanisms.
"""

import pytest
import torch
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from relic.parameters import get_global_manager
from relic.scientific.biosphere.phenology.parameters import (
    PhenologyParameters,
    PhenologyParameterSets,
)
from relic.scientific.biosphere.phenology.common.strategies import StaticGrowthStrategy


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


class TestParameterManagerIntegration:
    """Test ParameterManager integration with phenology module."""
    
    def setup_method(self):
        """Reset parameter manager before each test."""
        self.param_manager = get_global_manager()
        # Load phenology parameters using the manager's load_from_database method
        self.param_manager.load_from_database()
    
    def test_parameter_manager_available(self):
        """Test that ParameterManager is available and initialized."""
        assert self.param_manager is not None
        
        # Check that phenology parameters are loaded
        params = self.param_manager.list(namespace="biosphere.phenology")
        assert len(params) > 0, "No phenology parameters loaded"
    
    def test_parameter_loading_from_yaml(self):
        """Test loading parameters from phenology_unified.yaml."""
        # Test specific parameter values
        value = self.param_manager.get("biosphere.phenology.static.leaf_onset_doy")
        assert value is not None
        assert value == 120  # Default value from YAML
        
        # Test parameter with units
        value = self.param_manager.get("biosphere.phenology.gdd.base_temp")
        assert value is not None
        # Stored in Kelvin per CF conventions (5 degC)
        assert value == pytest.approx(273.15 + 5.0, rel=0, abs=1e-6)
    
    def test_parameter_categories(self):
        """Test that parameter categories are correctly mapped."""
        # Get parameter spec to check category
        spec = self.param_manager.get_parameter("biosphere.phenology.static.leaf_onset_doy")
        assert spec is not None
        category = getattr(spec.category, "value", spec.category)
        assert category in ["biological_parameter", "model_parameter"]
        
        # Check computational parameters
        spec = self.param_manager.get_parameter("biosphere.phenology.computational.numerical_tolerance")
        assert spec is not None
        category = getattr(spec.category, "value", spec.category)
        assert category in ["computational", "computational_parameter"]
    
    def test_missing_parameter_handling(self):
        """Test handling of missing parameters."""
        # Try to get a parameter that doesn't exist
        value = self.param_manager.get("biosphere.phenology.nonexistent.parameter", default=42.0)
        assert value == 42.0
        
        # Check that missing request is tracked
        missing = self.param_manager.get_missing_requests()
        assert any("nonexistent" in key for key in missing)
    
    def test_parameter_bounds_validation(self):
        """Test parameter bounds validation."""
        # Get parameter with bounds
        spec = self.param_manager.get_parameter("biosphere.phenology.gdd.base_temp")
        assert spec is not None
        assert spec.bounds is not None
        assert len(spec.bounds) == 2
        assert spec.bounds[0] <= spec.value <= spec.bounds[1]


class TestPhenologyParameters:
    """Test PhenologyParameters dataclass with ParameterManager."""
    
    def setup_method(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()
    
    def test_phenology_parameters_initialization(self):
        """Test explicit construction from a ParameterManager-backed preset."""
        params = PhenologyParameterSets.deciduous_tree_gsi()

        # The GSI preset holds the GSI fields and no field of another model
        assert params.phenology_type == "gsi"
        assert params.gsi_onset_threshold is not None
        assert params.leaf_onset_doy is None
        assert params.gdd_base_temp is None

    def test_phenology_parameters_override(self):
        """Test that an explicit payload can override preset values."""
        preset = PhenologyParameterSets.grassland_gdd()
        values = {k: v for k, v in preset.model_dump().items() if v is not None}
        values.update(
            gdd_base_temp=10.0,
            gdd_threshold_start=100.0,
        )
        params = PhenologyParameters(**values)

        assert params.phenology_type == "gdd"
        assert params.gdd_base_temp == 10.0
        assert params.gdd_threshold_start == 100.0

        # Check that non-overridden values remain sourced from the preset payload.
        assert params.gdd_threshold_end == preset.gdd_threshold_end

    def test_new_parameters_added(self):
        """Test that newly added parameters are accessible."""
        # Test parameters we added in the fix
        value = self.param_manager.get("biosphere.phenology.turnover.leaf_turnover_rate")
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        seconds_per_year = 365.25 * 24 * 3600
        assert value * seconds_per_year == pytest.approx(0.25, rel=0, abs=1e-3)
        
        value = self.param_manager.get("biosphere.phenology.turnover.froot_turnover_rate")
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        assert value * seconds_per_year == pytest.approx(1.0, rel=0, abs=1e-2)


class TestStrategyParameterAccess:
    """Test non-Functor strategy parameter access path."""
    
    def setup_method(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()
        self.accessor = StaticGrowthStrategy(
            parameter_manager=self.param_manager
        )
    
    def test_parameter_accessor_get_value(self):
        """Test getting parameter values through accessor."""
        value = self.accessor.get_parameter_value(
            "static.leaf_onset_doy",
        )
        assert value == 120  # Value from database
        
        # Test with full key
        value = self.accessor.get_parameter_value(
            "biosphere.phenology.static.leaf_offset_doy",
        )
        assert value == 300  # Value from database
    
    def test_parameter_accessor_returns_scalar(self):
        """Test getting scalar values through strategy accessor."""
        value = self.accessor.get_parameter_value(
            "gdd.base_temp",
        )
        assert isinstance(value, (int, float))
        assert value == pytest.approx(273.15 + 5.0, rel=0, abs=1e-3)
    
    def test_missing_parameter_raises_unless_declared_optional(self):
        """The accessor reads formal YAML keys: a missing key raises, a caller-supplied
        default is not accepted, and only an explicitly optional read returns None."""
        with pytest.raises(KeyError):
            self.accessor.get_parameter_value("nonexistent.parameter")
        assert self.accessor.get_parameter_value("nonexistent.parameter", required=False) is None
        with pytest.raises(TypeError):
            self.accessor.get_parameter_value("static.leaf_onset_doy", default=100)
    
    def test_parameter_accessor_required(self):
        """Test required parameter raises error when missing."""
        with pytest.raises(KeyError):
            self.accessor.get_parameter_value(
                "nonexistent.required.parameter",
                required=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


