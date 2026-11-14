"""
Unit Tests for Phenology Parameters Module

Tests comprehensive parameter models for phenological processes with type safety
and validation. Covers all parameter types, PFT-specific sets, and validation logic.

Scientific Validation Focus:
- Parameter range validation for ecological realism
- Pydantic model constraints and validators
- Cross-parameter consistency checks
- PFT-specific parameter sets accuracy
- Default parameter completeness

Test Organization:
- Basic parameter validation
- Cross-validation between parameters
- PFT parameter sets validation
- Device consistency testing
- Configuration loading and validation
"""

import pytest
import torch
from pydantic import ValidationError

from relic.scientific.biosphere.phenology.parameters import (
    PHENOLOGY_PARAMETER_FIELD_KEYS,
    PhenologyParameters,
    PhenologyParameterSets,
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


class TestPhenologyParameters:
    """Test suite for PhenologyParameters model."""
    
    def test_default_initialization(self):
        """The GSI container holds the GSI fields and no field of another model."""
        params = _parameters()

        assert params.phenology_type == "gsi"
        assert params.gsi_tmin_threshold == 0.0
        assert params.gsi_tmin_optimal == 5.0
        assert params.leaf_onset_doy is None
        assert params.gdd_base_temp is None

    def test_basic_parameter_validation(self):
        """Test basic parameter validation ranges."""
        params = _parameters(
            phenology_type="static",
            leaf_onset_doy=100,
            leaf_offset_doy=300,
        )
        assert params.leaf_onset_doy == 100

        # Invalid day of year
        with pytest.raises(ValidationError):
            _parameters(phenology_type="static", leaf_onset_doy=400)  # > 366

    def test_cross_parameter_validation(self):
        """Test validators that check relationships between parameters."""
        with pytest.raises(ValidationError, match="leaf_offset_doy must be greater"):
            _parameters(phenology_type="static", leaf_onset_doy=200, leaf_offset_doy=150)

        with pytest.raises(ValidationError, match="gdd_threshold_end must be greater"):
            _parameters(
                phenology_type="gdd", gdd_threshold_start=500.0, gdd_threshold_end=300.0
            )

        with pytest.raises(ValidationError, match="gsi_tmin_optimal must be greater"):
            _parameters(gsi_tmin_threshold=10.0, gsi_tmin_optimal=5.0)

        with pytest.raises(ValidationError, match="gsi_vpd_max must be greater"):
            _parameters(gsi_vpd_min=2000.0, gsi_vpd_max=1500.0)

        with pytest.raises(ValidationError, match="crop_harvest_doy must be greater"):
            _parameters(phenology_type="crop", crop_sowing_doy=150, crop_harvest_doy=120)

        with pytest.raises(ValidationError, match="max_lai must be greater"):
            _parameters(min_lai=0.8, max_lai=0.7)

    def test_gdd_parameters(self):
        """Test GDD-specific parameter validation."""
        gdd_params = _parameters(
            phenology_type="gdd",
            gdd_base_temp=5.0,
            gdd_threshold_start=150.0,
            gdd_threshold_end=1000.0,
            gdd_max_temp_limit=30.0
        )
        
        assert gdd_params.phenology_type == "gdd"
        assert gdd_params.gdd_base_temp == 5.0
        assert gdd_params.gdd_threshold_start < gdd_params.gdd_threshold_end
        
        # Test temperature limits
        with pytest.raises(ValidationError):
            _parameters(phenology_type="gdd", gdd_base_temp=-15.0)  # Too low
            
        with pytest.raises(ValidationError):
            _parameters(phenology_type="gdd", gdd_max_temp_limit=60.0)  # Too high
    
    def test_gsi_parameters(self):
        """Test GSI-specific parameter validation."""
        gsi_params = _parameters(
            phenology_type="gsi",
            gsi_tmin_threshold=-5.0,
            gsi_tmin_optimal=10.0,
            gsi_vpd_min=500.0,
            gsi_vpd_max=3500.0,
            gsi_daylen_threshold=43200.0,  # 12 hours in seconds
            gsi_memory_days=14
        )
        
        assert gsi_params.phenology_type == "gsi"
        assert gsi_params.gsi_tmin_optimal > gsi_params.gsi_tmin_threshold
        assert gsi_params.gsi_vpd_max > gsi_params.gsi_vpd_min
        assert 1 <= gsi_params.gsi_memory_days <= 30
        assert "gsi_swp_wilting" not in PhenologyParameters.model_fields
        assert "gsi_swp_optimal" not in PhenologyParameters.model_fields
        assert "gsi_swp_wilting" not in PHENOLOGY_PARAMETER_FIELD_KEYS
        assert "gsi_swp_optimal" not in PHENOLOGY_PARAMETER_FIELD_KEYS
    
    def test_crop_parameters(self):
        """Test crop-specific parameter validation."""
        crop_params = _parameters(
            phenology_type="crop",
            crop_type="wheat",
            crop_phu_total=1500.0,
            crop_hi_max=0.50,
            crop_sowing_doy=100,
            crop_harvest_doy=250,
            crop_vernalization_required=True,
            crop_vernalization_days_required=60
        )
        
        assert crop_params.phenology_type == "crop"
        assert crop_params.crop_type == "wheat"
        assert crop_params.crop_vernalization_required is True
        assert crop_params.crop_harvest_doy > crop_params.crop_sowing_doy
        
        # Test harvest index bounds
        with pytest.raises(ValidationError):
            _parameters(phenology_type="crop", crop_hi_max=1.2)  # > 0.8
    
    def test_validation_for_phenology_type(self):
        """Test phenology type-specific validation."""
        # A crop container without crop_type is rejected at construction
        with pytest.raises(ValidationError, match="crop_type"):
            _parameters(phenology_type="crop", crop_type=None)

        # Valid crop configuration
        params = _parameters(phenology_type="crop", crop_type="maize")
        params.validate_for_phenology_type()  # Should not raise

        # Invalid GDD configuration should fail during initialization
        with pytest.raises(ValidationError):
            _parameters(
                phenology_type="gdd",
                gdd_threshold_start=1000.0,
                gdd_threshold_end=500.0,
            )

    def test_device_conversion(self):
        """Test device conversion functionality."""
        params = _parameters()
        
        # Test CPU device
        cpu_params = params.to_device(torch.device('cpu'))
        assert isinstance(cpu_params, PhenologyParameters)
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_params = params.to_device(torch.device('cuda'))
            assert isinstance(cuda_params, PhenologyParameters)
    
    def test_parameter_units_and_bounds(self):
        """Test parameter units are properly set and bounds are ecological."""
        gsi = _parameters()
        gdd = _parameters(phenology_type="gdd")

        # Check temperature bounds (in Celsius equivalent)
        assert -10.0 <= gdd.gdd_base_temp <= 25.0
        assert -20.0 <= gsi.gsi_tmin_threshold <= 10.0
        assert 0.0 <= gsi.gsi_tmin_optimal <= 15.0

        # Check VPD bounds (in Pa)
        assert 0.0 <= gsi.gsi_vpd_min <= 5000.0
        assert 1000.0 <= gsi.gsi_vpd_max <= 8000.0

        # Check LAI bounds
        assert 0.01 <= gsi.min_lai <= 1.0
        assert 1.0 <= gsi.max_lai <= 15.0

    def test_numerical_tolerance_bounds(self):
        """The numerical tolerance of the phenology checks lies in its declared bounds."""
        params = _parameters()
        assert 1e-12 <= params.numerical_tolerance <= 1e-6


class TestPhenologyParameterSets:
    """Test suite for predefined parameter sets."""
    
    def test_deciduous_tree_gsi(self):
        """Test deciduous tree GSI parameter set."""
        params = PhenologyParameterSets.deciduous_tree_gsi()
        
        assert params.phenology_type == "gsi"
        assert params.gsi_tmin_threshold == 0.0
        assert params.gsi_tmin_optimal == 5.0
        assert params.max_lai == 6.0
        assert params.leaf_turnover_rate == pytest.approx(7.927e-09, rel=0, abs=1e-12)
        
        # Validate the parameter set
        params.validate_for_phenology_type()  # Should not raise
    
    def test_evergreen_tree_static(self):
        """Test evergreen tree static parameter set."""
        params = PhenologyParameterSets.evergreen_tree_static()
        
        assert params.phenology_type == "static"
        assert params.leaf_onset_doy == 90
        assert params.leaf_offset_doy == 330
        assert params.max_lai == 4.0
        assert params.leaf_turnover_rate == pytest.approx(7.927e-09, rel=0, abs=1e-12)
    
    def test_grassland_gdd(self):
        """Test grassland GDD parameter set."""
        params = PhenologyParameterSets.grassland_gdd()
        
        assert params.phenology_type == "gdd"
        assert params.gdd_base_temp == 5.0
        assert params.gdd_threshold_start == 200.0
        assert params.gdd_threshold_end == 1100.0
        assert params.max_lai == 3.0
        assert params.leaf_turnover_rate == pytest.approx(7.927e-09, rel=0, abs=1e-12)
        
        params.validate_for_phenology_type()
    
    def test_winter_wheat_crop(self):
        """Test winter wheat crop parameter set."""
        params = PhenologyParameterSets.winter_wheat_crop()
        
        assert params.phenology_type == "crop"
        assert params.crop_type == "wheat"
        assert params.crop_phu_total == 1200.0
        assert params.crop_hi_max == 0.5
        assert params.crop_sowing_doy == 270  # Fall sowing
        assert params.crop_harvest_doy == 150  # Following year
        assert params.crop_vernalization_required is True
        assert params.crop_vernalization_days_required == 45
        
        params.validate_for_phenology_type()
    
    def test_maize_crop(self):
        """Test maize crop parameter set."""
        params = PhenologyParameterSets.maize_crop()
        
        assert params.phenology_type == "crop"
        assert params.crop_type == "maize"
        assert params.crop_phu_total == 1400.0
        assert params.crop_hi_max == 0.48
        assert params.crop_sowing_doy == 120  # Spring sowing
        assert params.crop_harvest_doy == 260  # Late summer harvest
        assert params.crop_vernalization_required is False
        assert params.gdd_base_temp == 8.0  # C4 crop
        
        params.validate_for_phenology_type()
    
    def test_all_predefined_sets_valid(self):
        """Test that all predefined parameter sets are valid."""
        parameter_sets = [
            PhenologyParameterSets.deciduous_tree_gsi(),
            PhenologyParameterSets.evergreen_tree_static(),
            PhenologyParameterSets.grassland_gdd(),
            PhenologyParameterSets.winter_wheat_crop(),
            PhenologyParameterSets.maize_crop()
        ]
        
        for params in parameter_sets:
            # All sets should pass validation
            params.validate_for_phenology_type()
            
            # All sets should have consistent cross-parameter relationships
            assert params.max_lai > params.min_lai
            if params.phenology_type == "gdd":
                assert params.gdd_threshold_end > params.gdd_threshold_start
            if params.phenology_type == "crop":
                if params.crop_harvest_doy <= params.crop_sowing_doy:
                    assert params.crop_sowing_doy >= 200 and params.crop_harvest_doy <= 200
                else:
                    assert params.crop_harvest_doy > params.crop_sowing_doy


class TestParameterTypeEnums:
    """Test parameter type enumerations."""
    
    def test_phenology_type_enum(self):
        """Test PhenologyType literal."""
        # Valid phenology types
        valid_types = ["static", "gdd", "gsi", "crop"]
        for ptype in valid_types:
            params = _parameters(phenology_type=ptype)
            assert params.phenology_type == ptype
    
    def test_crop_type_enum(self):
        """Test CropType literal."""
        valid_types = ["wheat", "maize", "soybean", "rice"]
        for ctype in valid_types:
            params = _parameters(phenology_type="crop", crop_type=ctype)
            assert params.crop_type == ctype


class TestIntegrationScenarios:
    """Test realistic parameter combinations."""
    
    def test_temperate_deciduous_forest(self):
        """Test parameters for temperate deciduous forest."""
        params = _parameters(
            phenology_type="gsi",
            gsi_tmin_threshold=-3.0,
            gsi_tmin_optimal=8.0,
            max_lai=6.0,
            leaf_turnover_rate=7.927e-09,
        )

        params.validate_for_phenology_type()
        assert params.phenology_type == "gsi"
        assert params.gsi_tmin_optimal == 8.0

    def test_boreal_evergreen_forest(self):
        """Test parameters for boreal evergreen forest."""
        params = _parameters(
            phenology_type="static",
            leaf_onset_doy=140,  # Late spring at high latitude
            leaf_offset_doy=260,  # Early fall
            max_lai=4.0,  # Lower LAI in boreal
            leaf_turnover_rate=3.5e-09,  # Long leaf lifespan
        )

        params.validate_for_phenology_type()
        assert params.leaf_onset_doy == 140
        assert params.leaf_turnover_rate == 3.5e-09

    def test_mediterranean_shrubland(self):
        """Test parameters for Mediterranean shrubland."""
        params = _parameters(
            phenology_type="static",
            leaf_onset_doy=90,  # Early spring growth
            leaf_offset_doy=320,  # Late senescence
            max_lai=3.5,
        )

        params.validate_for_phenology_type()
        assert params.leaf_offset_doy == 320

    def test_agricultural_system(self):
        """Test parameters for agricultural crop system."""
        params = _parameters(
            phenology_type="crop",
            crop_type="soybean",
            crop_phu_total=1300.0,
            crop_hi_max=0.40,
            crop_sowing_doy=140,  # Late spring
            crop_harvest_doy=280,  # Early fall
            gdd_base_temp=10.0,
            max_lai=4.5,
        )

        params.validate_for_phenology_type()
        assert params.crop_type == "soybean"
        assert params.gdd_base_temp == 10.0


class TestModelParameterSets:
    """Each phenology model owns one parameter set: the common fields plus its own."""

    def test_container_holds_exactly_the_configured_model_fields(self):
        """A container carries the common fields and the fields of its model: a
        missing model field and a field of another model both fail validation."""
        from relic.scientific.biosphere.phenology.parameters import (
            PHENOLOGY_COMMON_FIELDS,
            PHENOLOGY_MODEL_FIELDS,
        )

        gsi = PhenologyParameterSets.deciduous_tree_gsi()
        payload = {
            name: getattr(gsi, name)
            for name in (*PHENOLOGY_COMMON_FIELDS, *PHENOLOGY_MODEL_FIELDS["gsi"])
        }
        assert PhenologyParameters(**payload).phenology_type == "gsi"
        incomplete = {key: value for key, value in payload.items() if key != "gsi_memory_days"}
        with pytest.raises(ValidationError, match="gsi_memory_days"):
            PhenologyParameters(**incomplete)
        with pytest.raises(ValidationError, match="leaf_onset_doy"):
            PhenologyParameters(**payload, leaf_onset_doy=120)

    def test_presets_carry_only_their_model_fields(self):
        """Every preset container sets the common fields and its model fields only."""
        from relic.scientific.biosphere.phenology.parameters import (
            PHENOLOGY_COMMON_FIELDS,
            PHENOLOGY_MODEL_FIELDS,
        )

        for factory in (
            PhenologyParameterSets.deciduous_tree_gsi,
            PhenologyParameterSets.evergreen_tree_static,
            PhenologyParameterSets.grassland_gdd,
            PhenologyParameterSets.winter_wheat_crop,
            PhenologyParameterSets.maize_crop,
        ):
            params = factory()
            present = {name for name, value in params.model_dump().items() if value is not None}
            expected = set(PHENOLOGY_COMMON_FIELDS) | set(PHENOLOGY_MODEL_FIELDS[params.phenology_type])
            assert present == expected, factory.__name__

    def test_preset_entry_that_cannot_be_read_raises(self, monkeypatch):
        """A preset entry whose read fails stops the preset instead of silently taking
        the generic value of the field."""
        import relic.scientific.biosphere.phenology.parameters as module

        original = module._read_parameter

        def failing(key, **kwargs):
            if key == "biosphere.phenology.presets.grassland_gdd.gdd_threshold_start":
                raise ValueError("unit mismatch for gdd_threshold_start")
            return original(key, **kwargs)

        monkeypatch.setattr(module, "_read_parameter", failing)
        with pytest.raises(ValueError, match="unit mismatch"):
            PhenologyParameterSets.grassland_gdd()


