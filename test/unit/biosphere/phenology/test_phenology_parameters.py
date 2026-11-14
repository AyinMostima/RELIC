"""Unit tests for Phenology Parameter System Integration.

This module tests the integration of the phenology module with the 
unified ParameterManager system, ensuring all parameters are correctly
loaded from the centralized database instead of hardcoded values.

Test Coverage:
    - L1: Parameter loading from ParameterManager
    - L1: PhenologyParameterSets preset loading
    - L1: Unit metadata validation
"""

import pytest
import torch

from relic.scientific.biosphere.phenology.parameters import (
    PhenologyParameters, 
    PhenologyParameterSets
)
from relic.parameters import get_global_manager
from relic.scientific.biosphere.phenology.common.strategies import (
    GrowthFactorStrategy,
)


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


class TestParameterManagerIntegration:
    """Test integration with the unified ParameterManager system."""
    
    def setup_method(self):
        """Setup test environment."""
        # Get the global parameter manager
        self.param_manager = get_global_manager()
        # Load phenology parameters
        self.param_manager.load_from_database("biosphere")
        
    def test_phenology_parameter_units_metadata(self):
        """Test that PhenologyParameters has correct unit metadata."""
        # Check specific fields have proper units (not placeholders)
        unit_checks = {
            "latitude": "degrees_north",
            "leaf_onset_doy": "day",
            "leaf_offset_doy": "day",
            "gdd_base_temp": "degC",
            "gdd_threshold_start": "degC day",
            "gsi_tmin_threshold": "degC",
            "gsi_vpd_min": "Pa",
            "gsi_daylen_threshold": "s",
            "max_lai": "m2 m-2",
            "sla": "m^2 kg^-1 C",
            "leaf_turnover_rate": "s-1"
        }
        
        # Get the model fields
        model_fields = PhenologyParameters.model_fields
        assert "gsi_swp_wilting" not in model_fields
        assert "gsi_swp_optimal" not in model_fields
        
        for field_name, expected_unit in unit_checks.items():
            if field_name in model_fields:
                field_info = model_fields[field_name]
                # In Pydantic v2, json_schema_extra is stored differently
                if hasattr(field_info, 'json_schema_extra') and field_info.json_schema_extra:
                    unit = field_info.json_schema_extra.get('unit')
                    assert unit == expected_unit, \
                        f"Field {field_name} has unit '{unit}', expected '{expected_unit}'"

    def test_gsi_parameter_database_matches_three_factor_runtime(self):
        """Reject parameters that are not consumed by the canonical GSI runtime."""
        stale_keys = {
            "biosphere.phenology.gsi.photoperiod_span",
            "biosphere.phenology.gsi.daylength_span_min_seconds",
            "biosphere.phenology.gsi.photoperiod_amplitude_seconds",
            "biosphere.phenology.gsi.photoperiod_phase_shift_doy",
            "biosphere.phenology.gsi.temp_range_min",
            "biosphere.phenology.gsi.swp_wilting",
            "biosphere.phenology.gsi.swp_optimal",
            "biosphere.phenology.gsi.default_vpd",
            "biosphere.phenology.gsi.default_latitude",
            "biosphere.phenology.gsi.default_swp",
            "biosphere.phenology.gsi.year_reset_threshold",
        }
        registered_keys = set(self.param_manager.parameters)
        assert stale_keys.isdisjoint(registered_keys)
        assert self.param_manager.get(
            "biosphere.phenology.gsi.memory_days",
            required=True,
        ) == 21
    
    def test_phenology_parameter_sets_preset_loading(self):
        """Test that PhenologyParameterSets can load presets."""
        # Test different preset types
        deciduous = PhenologyParameterSets.deciduous_tree_gsi()
        assert deciduous.phenology_type == "gsi"
        expected_decid_tmin = self.param_manager.get(
            "biosphere.phenology.presets.deciduous_tree_gsi.gsi_tmin_threshold",
            units="degC",
        )
        expected_decid_lai = self.param_manager.get(
            "biosphere.phenology.presets.deciduous_tree_gsi.max_lai"
        )
        assert deciduous.gsi_tmin_threshold == expected_decid_tmin
        assert deciduous.max_lai == expected_decid_lai
        
        evergreen = PhenologyParameterSets.evergreen_tree_static()
        assert evergreen.phenology_type == "static"
        expected_evergreen_onset = self.param_manager.get(
            "biosphere.phenology.presets.evergreen_tree_static.leaf_onset_doy"
        )
        expected_evergreen_lai = self.param_manager.get(
            "biosphere.phenology.presets.evergreen_tree_static.max_lai"
        )
        assert evergreen.leaf_onset_doy == expected_evergreen_onset
        assert evergreen.max_lai == expected_evergreen_lai
        
        grassland = PhenologyParameterSets.grassland_gdd()
        assert grassland.phenology_type == "gdd"
        expected_grassland_base = self.param_manager.get(
            "biosphere.phenology.presets.grassland_gdd.gdd_base_temp",
            units="degC",
        )
        expected_grassland_lai = self.param_manager.get(
            "biosphere.phenology.presets.grassland_gdd.max_lai"
        )
        assert grassland.gdd_base_temp == expected_grassland_base
        assert grassland.max_lai == expected_grassland_lai
        
        wheat = PhenologyParameterSets.winter_wheat_crop()
        assert wheat.phenology_type == "crop"
        assert wheat.crop_type == "wheat"
        expected_vernal = self.param_manager.get(
            "biosphere.phenology.presets.winter_wheat_crop.crop_vernalization_required"
        )
        assert bool(wheat.crop_vernalization_required) == bool(expected_vernal)
        
    def test_strategy_inputs_are_a_state_subset(self):
        """Every phenology strategy takes its drivers as one inputs mapping (a subset of
        the declared state inputs); there is no environment channel."""
        import inspect

        from relic.scientific.biosphere.phenology.common import strategies as module

        classes = [
            obj for obj in vars(module).values()
            if inspect.isclass(obj) and issubclass(obj, module.PhenologyStrategy)
        ]
        assert classes
        for cls in classes:
            params = list(inspect.signature(cls.compute).parameters)
            assert params[:2] == ["self", "inputs"], cls.__name__

    def test_strategies_have_parameter_access(self):
        """A strategy reads a formal YAML key through the injected ParameterManager; no
        caller default exists and a missing required key raises."""
        class TestGrowthStrategy(GrowthFactorStrategy):
            def compute(self, environment, **kwargs):
                clamp_min = self.get_parameter_value(
                    "clamp_min",
                    namespace="biosphere.phenology.computational",
                )
                return torch.tensor(clamp_min)

        strategy = TestGrowthStrategy(
            name="test",
            parameter_manager=self.param_manager,
        )

        expected = self.param_manager.get("biosphere.phenology.computational.clamp_min")
        assert strategy.compute({}) == torch.tensor(expected)
        with pytest.raises(KeyError):
            strategy.get_parameter_value(
                "clamp_min_absent", namespace="biosphere.phenology.computational"
            )
        assert strategy.get_parameter_value(
            "clamp_min_absent",
            namespace="biosphere.phenology.computational",
            required=False,
        ) is None
        with pytest.raises(TypeError):
            strategy.get_parameter_value(
                "clamp_min", default=0.0, namespace="biosphere.phenology.computational"
            )
        
    def test_parameter_manager_audit_phenology(self):
        """Test parameter audit specifically for phenology parameters."""
        # Run audit for phenology namespace
        audit_result = self.param_manager.run_registry_audit()
        
        # Check that phenology parameters are registered
        assert "biosphere.phenology" in str(audit_result)
        
        # Check for missing requests (should be minimal after our fixes)
        missing = self.param_manager.get_missing_requests()
        phenology_missing = [
            k for k in missing if "phenology" in k and ".presets." not in k
        ]
        
        # Log any missing parameters for debugging
        if phenology_missing:
            print(f"Missing phenology parameters: {phenology_missing}")
        
        filtered_missing = [
            k for k in phenology_missing
            if not k.startswith("biological_process.")
            and not k.startswith("phenology_type")
            and "test.nonexistent_param" not in k
        ]
        assert len(filtered_missing) < 10, \
            f"Too many missing phenology parameters: {filtered_missing}"


class TestParameterValidation:
    """Test parameter validation and bounds checking."""
    
    def setup_method(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database("biosphere")
    
    def test_parameter_bounds_validation(self):
        """Test that parameters respect their defined bounds."""
        from pydantic import ValidationError
        
        # Test LAI bounds - should raise validation error if out of bounds
        with pytest.raises(ValidationError):
            _parameters(max_lai=20.0)  # Exceeds max bound of 15.0
        
        # Test temperature bounds - should raise validation error if out of bounds  
        with pytest.raises(ValidationError):
            _parameters(phenology_type="gdd", gdd_base_temp=-15.0)  # Below min bound of -10.0
        
        # Valid parameters should work
        params = _parameters(phenology_type="gdd", max_lai=10.0, gdd_base_temp=5.0)
        assert params.max_lai == 10.0
        assert params.gdd_base_temp == 5.0
        
    def test_parameter_type_validation(self):
        """Test that parameters have correct types."""
        params = _parameters(phenology_type="static")

        # Integer parameters
        assert isinstance(params.leaf_onset_doy, int)
        assert isinstance(params.leaf_offset_doy, int)

        # Float parameters
        assert isinstance(params.max_lai, float)

        # Boolean parameters
        crop = _parameters(phenology_type="crop")
        assert isinstance(crop.crop_vernalization_required, bool)


class TestParameterPersistence:
    """Test parameter persistence and serialization."""
    
    def setup_method(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database("biosphere")
        
    def test_parameter_serialization(self):
        """Test that parameters can be serialized and deserialized."""
        params = _parameters(
            phenology_type="gsi",
            max_lai=7.5,
        )
        
        # Convert to dict
        params_dict = params.model_dump()
        assert params_dict["phenology_type"] == "gsi"
        assert params_dict["max_lai"] == 7.5
        
        # Recreate from dict
        params_restored = PhenologyParameters(**params_dict)
        assert params_restored.max_lai == 7.5
        
    def test_parameter_json_export(self):
        """Test JSON export with proper metadata."""
        params = _parameters()
        
        # Export to JSON with schema
        json_str = params.model_dump_json(indent=2)
        assert "gsi" in json_str
        
        # Check that parameters are properly serialized
        params_dict = params.model_dump()
        assert "max_lai" in params_dict
        assert params_dict["phenology_type"] == "gsi"
        
        # For Pydantic v2, check field metadata directly
        model_fields = PhenologyParameters.model_fields
        assert "max_lai" in model_fields
        
        # Check unit metadata is present in field definition
        lai_field = model_fields.get("max_lai")
        if lai_field and hasattr(lai_field, 'json_schema_extra'):
            if lai_field.json_schema_extra:
                assert lai_field.json_schema_extra.get("unit") == "m2 m-2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

