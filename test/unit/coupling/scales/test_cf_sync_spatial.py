"""
Unit Tests for CF-1.8 Spatial Metadata Synchronization

Tests the CF-1.8 compliant spatial metadata synchronization functionality
for hierarchical and spatial coupling operations in relic/coupling/scales.

Coverage Areas:
- CF-1.8 spatial regrid metadata synchronization
- Conservative/bilinear/nearest regrid method metadata updates  
- Grid mapping and coordinate reference updates
- Cell methods for spatial operations
- History tracking for spatial transformations

Test Framework:
- L1 unit tests for individual CF functions
- Isolated testing of pure functions
- Mock spatial coordinate objects
- Parameterized tests for different regrid methods

Mathematical Validation:
- Conservative regridding: area-weighted metadata
- Bilinear interpolation: smooth method metadata  
- Nearest neighbor: categorical variable metadata
- Grid transformation: coordinate system updates

References:
    - CF Conventions Version 1.8: http://cfconventions.org/cf-conventions/v1.8.0/
    - CF Cell Methods Guidelines
    - Grid Mapping Standards
"""

import inspect
import os

# Import modules under test
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import relic.data.standards.cf_conventions as cf_module
import relic.utils.physical_constants as physical_constants_module
from relic.data.standards.cf_conventions import (
    CFAttributeSynchronizer,
    CFConventions,
    conserve_check,
)
from relic.utils.time_constants import SECONDS_PER_DAY


class TestCFSpatialSync:
    """Test CF-1.8 spatial metadata synchronization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.synchronizer = CFAttributeSynchronizer(strict_mode=False)
        self.strict_synchronizer = CFAttributeSynchronizer(strict_mode=True)
        
        # Mock spatial data
        self.spatial_data = torch.randn(10, 20, 30)  # [time, lat, lon]
        
        # Source and target grid specifications
        self.source_grid = {
            'lat_name': 'lat',
            'lon_name': 'lon',
            'shape': (20, 30),
            'grid_mapping': 'latitude_longitude',
            'resolution': 0.5
        }
        
        self.target_grid = {
            'lat_name': 'latitude',
            'lon_name': 'longitude', 
            'shape': (40, 60),
            'grid_mapping': 'latitude_longitude',
            'resolution': 0.25
        }
        
        # Base metadata for testing
        self.base_metadata = {
            'variable_name': 'temperature',
            'standard_name': 'air_temperature',
            'units': 'K',
            'long_name': 'Air temperature',
            'coordinates': 'lat lon',
            'cell_methods': 'time: mean',
            'history': 'Original data processing'
        }
    
    def test_conservative_regrid_metadata_sync(self):
        """Test CF metadata sync for conservative regridding."""
        regrid_method = 'conservative'
        
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=self.base_metadata,
            regrid_method=regrid_method,
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        # Check CF-1.8 compliance
        assert 'standard_name' in result
        assert result['standard_name'] == 'air_temperature'
        assert result['units'] == 'K'
        
        # Check grid information updates
        assert result['grid_mapping'] == 'latitude_longitude'
        assert result['coordinates'] == 'latitude longitude'
        
        # Regridding provenance must not rewrite statistical cell semantics.
        assert result['cell_methods'] == 'time: mean'
        
        # Check history update
        assert 'Regridded from (20, 30) to (40, 60) using conservative method' in result['history']
        assert 'Original data processing' in result['history']
        
        # Check shape information
        assert result['shape'] == list(self.spatial_data.shape)
        
        # Check data range
        assert 'actual_range' in result
        assert len(result['actual_range']) == 2
    
    def test_bilinear_regrid_metadata_sync(self):
        """Test CF metadata sync for bilinear regridding."""
        regrid_method = 'bilinear'
        
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=self.base_metadata,
            regrid_method=regrid_method,
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        assert result['cell_methods'] == 'time: mean'
        
        # Check history contains method information
        assert 'bilinear method' in result['history']
        
        # Other attributes should be consistent with conservative test
        assert result['standard_name'] == 'air_temperature'
        assert result['coordinates'] == 'latitude longitude'
    
    def test_nearest_regrid_metadata_sync(self):
        """Test CF metadata sync for nearest neighbor regridding."""
        regrid_method = 'nearest'
        
        # Use categorical variable for nearest neighbor
        categorical_metadata = self.base_metadata.copy()
        categorical_metadata.update({
            'variable_name': 'land_cover',
            'standard_name': 'land_cover_lccs',
            'units': '1',
            'long_name': 'Land cover classification'
        })
        
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=categorical_metadata,
            regrid_method=regrid_method,
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        assert result['cell_methods'] == 'time: mean'
        
        # Check categorical variable handling
        assert result['units'] == '1'
        assert 'nearest method' in result['history']
    
    @pytest.mark.parametrize(
        "regrid_method",
        ['conservative', 'bilinear', 'nearest', 'cubic'],
    )
    def test_regrid_method_preserves_cell_methods(self, regrid_method):
        """A numerical regrid method is not a CF cell-method statistic."""
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=self.base_metadata,
            regrid_method=regrid_method,
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        assert result['cell_methods'] == 'time: mean'
    
    def test_missing_cell_methods_remains_missing(self):
        """Regridding must not invent a cell-method declaration."""
        metadata_no_cell_methods = self.base_metadata.copy()
        del metadata_no_cell_methods['cell_methods']
        
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=metadata_no_cell_methods,
            regrid_method='conservative',
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        assert 'cell_methods' not in result

    @pytest.mark.parametrize(
        "missing_key",
        ["grid_mapping", "lat_name", "lon_name", "shape"],
    )
    def test_regridding_requires_explicit_target_grid_metadata(self, missing_key):
        target_grid = self.target_grid.copy()
        del target_grid[missing_key]

        with pytest.raises(ValueError, match="explicit target_grid"):
            self.synchronizer.sync_after_regridding(
                data=self.spatial_data,
                metadata=self.base_metadata,
                regrid_method="conservative",
                source_grid=self.source_grid,
                target_grid=target_grid,
            )
    
    def test_regrid_target_grid_fields_are_one_public_constant(self):
        """The CF grid fields a regrid target declares live in one exported tuple."""
        source = Path(cf_module.__file__).read_text(encoding="utf-8-sig")

        assert cf_module.CF_REGRID_TARGET_GRID_FIELDS == ("grid_mapping", "lat_name", "lon_name")
        assert "CF_REGRID_TARGET_GRID_FIELDS" in cf_module.__all__
        assert "CF_REGRID_TARGET_GRID_FIELDS" in inspect.getsource(
            CFAttributeSynchronizer.sync_after_regridding
        )
        assert source.count('"lat_name", "lon_name"') == 1

    @pytest.mark.parametrize("field", ["grid_mapping", "lat_name", "lon_name"])
    @pytest.mark.parametrize("bad_value", ["", "   ", None, 3])
    def test_regrid_target_grid_fields_must_be_non_empty_strings(self, field, bad_value):
        target_grid = self.target_grid.copy()
        target_grid[field] = bad_value

        with pytest.raises(ValueError, match=f"explicit target_grid metadata is required: {field}"):
            self.synchronizer.sync_after_regridding(
                data=self.spatial_data,
                metadata=self.base_metadata,
                regrid_method="conservative",
                source_grid=self.source_grid,
                target_grid=target_grid,
            )

    def test_grid_mapping_update(self):
        """Test grid mapping information updates."""
        # Custom grid mapping
        target_grid_custom = self.target_grid.copy()
        target_grid_custom['grid_mapping'] = 'lambert_conformal_conic'
        
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=self.base_metadata,
            regrid_method='conservative',
            source_grid=self.source_grid,
            target_grid=target_grid_custom
        )
        
        assert result['grid_mapping'] == 'lambert_conformal_conic'
    
    def test_coordinates_update(self):
        """Test coordinates attribute updates."""
        # Custom coordinate names
        target_grid_custom = self.target_grid.copy()
        target_grid_custom.update({
            'lat_name': 'y',
            'lon_name': 'x'
        })
        
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=self.base_metadata,
            regrid_method='conservative',
            source_grid=self.source_grid,
            target_grid=target_grid_custom
        )
        
        assert result['coordinates'] == 'y x'
    
    def test_history_preservation_and_update(self):
        """Test that existing history is preserved and new entry added."""
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=self.base_metadata,
            regrid_method='conservative',
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        history = result['history']
        
        # Original history should be preserved
        assert 'Original data processing' in history
        
        # New regridding information should be added
        assert 'Regridded from (20, 30) to (40, 60)' in history
        assert 'conservative method' in history
        
        # Should be semicolon-separated
        assert ';' in history
    
    def test_missing_history_handling(self):
        """Test handling when original metadata has no history."""
        metadata_no_history = self.base_metadata.copy()
        del metadata_no_history['history']
        
        result = self.synchronizer.sync_after_regridding(
            data=self.spatial_data,
            metadata=metadata_no_history,
            regrid_method='bilinear',
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        # Should only contain new regridding information
        assert 'Regridded from (20, 30) to (40, 60) using bilinear method' in result['history']
        assert ';' not in result['history']  # No semicolon if no original history
    
    def test_data_shape_and_range_update(self):
        """Test that shape and data range are correctly updated."""
        # Create data with known range
        test_data = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])  # shape: [2, 2, 2], range: [1.0, 8.0]
        
        result = self.synchronizer.sync_after_regridding(
            data=test_data,
            metadata=self.base_metadata,
            regrid_method='conservative',
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        assert result['shape'] == [2, 2, 2]
        assert result['actual_range'] == [1.0, 8.0]
    
    def test_empty_data_handling(self):
        """Test handling of empty data tensor."""
        empty_data = torch.tensor([])
        
        result = self.synchronizer.sync_after_regridding(
            data=empty_data,
            metadata=self.base_metadata,
            regrid_method='conservative',
            source_grid=self.source_grid,
            target_grid=self.target_grid
        )
        
        assert result['shape'] == [0]
        # Should not have actual_range for empty data
        assert 'actual_range' not in result


class TestCFSpatialValidation:
    """Test CF-1.8 spatial validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.synchronizer = CFAttributeSynchronizer(strict_mode=True)
    
    def test_strict_mode_validation(self):
        """Test that strict mode validates CF compliance."""
        # Create invalid metadata (missing units)
        invalid_metadata = {
            'standard_name': 'air_temperature',
            'long_name': 'Air temperature'
            # Missing 'units'
        }
        
        spatial_data = torch.randn(5, 10, 10)
        source_grid = {'shape': (10, 10)}
        target_grid = {
            'shape': (20, 20),
            'grid_mapping': 'latitude_longitude',
            'lat_name': 'latitude',
            'lon_name': 'longitude',
        }
        
        with pytest.raises(ValueError, match="standard-name metadata"):
            self.synchronizer.sync_after_regridding(
                data=spatial_data,
                metadata=invalid_metadata,
                regrid_method='conservative',
                source_grid=source_grid,
                target_grid=target_grid
            )
    
    def test_missing_standard_name_is_not_inferred(self):
        """A local variable name must not become an official CF identity."""
        metadata = {
            'variable_name': 'temp',
            'long_name': 'Temperature'
        }
        
        spatial_data = torch.randn(5, 10, 10)
        synchronizer = CFAttributeSynchronizer(strict_mode=False)
        
        result = synchronizer.sync_after_regridding(
            data=spatial_data,
            metadata=metadata,
            regrid_method='conservative',
            source_grid={'shape': (10, 10)},
            target_grid={
                'shape': (20, 20),
                'grid_mapping': 'latitude_longitude',
                'lat_name': 'latitude',
                'lon_name': 'longitude',
            }
        )
        
        assert 'standard_name' not in result
        assert 'units' not in result
    
    def test_cf_compliance_validation(self):
        """Test the scope-limited standard-name and unit metadata subset."""
        valid_metadata = {
            'standard_name': 'air_temperature',
            'units': 'K',
            'long_name': 'Air temperature',
            'coordinates': 'lat lon',
            'cell_methods': 'time: mean'
        }
        
        result = CFConventions.validate_cf_compliance(valid_metadata)
        is_compliant, issues = result
        assert is_compliant
        assert len(issues) == 0
        assert result.validation_scope == "single_variable_standard_name_and_units"
        assert result.dataset_conformance_checked is False
        assert result.coordinates_checked is False
        assert result.cell_methods_checked is False
        
        # Test invalid metadata
        invalid_metadata = {
            'standard_name': 'air_temperature',
            # Missing units
            'cell_methods': 'invalid syntax'  # Invalid cell_methods
        }
        
        result = CFConventions.validate_cf_compliance(invalid_metadata)
        is_compliant, issues = result
        assert not is_compliant
        assert len(issues) > 0
        assert any('units' in issue for issue in issues)
        assert not any('cell_methods' in issue for issue in issues)
        assert result.cell_methods_checked is False


class TestCFElementLabelledUnits:
    """RELIC writes the element into the unit (kg C m-2 s-1); CF writes it into the name
    (expressed_as_carbon, carbon_mass_..., mass_content_of_carbon) with kg m-2 s-1. The unit
    is compatible when its element-aware SI dimension equals that of the canonical units and
    its element is the element basis of the name."""

    @pytest.mark.parametrize(
        ("standard_name", "units"),
        [
            ("gross_primary_productivity_of_biomass_expressed_as_carbon", "kg C m-2 s-1"),
            ("gross_primary_productivity_of_biomass_expressed_as_carbon", "kg m-2 s-1"),
            ("net_primary_productivity_of_biomass_expressed_as_carbon", "g C m-2 s-1"),
            (
                "surface_upward_mass_flux_of_carbon_dioxide_expressed_as_carbon_due_to_respiration_in_soil",
                "kg C m-2 s-1",
            ),
            (
                "mass_flux_of_nitrogen_compounds_expressed_as_nitrogen_out_of_vegetation_"
                "and_litter_and_soil_due_to_leaching_and_runoff",
                "kg N m-2 s-1",
            ),
            ("nitrogen_mass_transport_in_river_channel", "kg N s-1"),
            ("carbon_mass_transport_in_river_channel", "kg C s-1"),
            ("phosphorus_mass_transport_in_river_channel", "kg P s-1"),
            ("vegetation_mass_content_of_carbon", "kg C m-2"),
            ("soil_carbon_content", "kg C m-2"),
        ],
    )
    def test_element_labelled_units_match_the_element_basis_of_the_name(self, standard_name, units):
        result = CFConventions.validate_cf_compliance({"standard_name": standard_name, "units": units})
        assert result.is_valid, result.issues
        assert CFConventions.are_standard_name_units_equivalent(standard_name, units)

    @pytest.mark.parametrize(
        ("standard_name", "units"),
        [
            ("gross_primary_productivity_of_biomass_expressed_as_carbon", "kg N m-2 s-1"),
            ("carbon_mass_transport_in_river_channel", "kg N s-1"),
            ("gross_primary_productivity_of_biomass_expressed_as_carbon", "kg C m-2"),
            ("vegetation_mass_content_of_carbon", "kg C m-2 s-1"),
            ("mass_concentration_of_carbon_dioxide_in_air", "kg C m-3"),
            ("air_temperature", "kg C m-2 s-1"),
            ("gross_primary_productivity_of_biomass_expressed_as_carbon", "kg C m-2 s-1 N"),
        ],
    )
    def test_a_wrong_element_or_dimension_is_incompatible(self, standard_name, units):
        result = CFConventions.validate_cf_compliance({"standard_name": standard_name, "units": units})
        assert not result.is_valid
        assert any("not physically compatible" in issue for issue in result.issues)


class TestCFSpatialUtilities:
    """Test CF spatial utility functions."""
    
    def test_create_bounds_variable_name(self):
        """Test bounds variable name creation."""
        assert CFConventions.create_bounds_variable_name('lat') == 'lat_bounds'
        assert CFConventions.create_bounds_variable_name('longitude') == 'longitude_bounds'
    
    def test_cell_methods_update(self):
        """Test cell_methods string updates."""
        # Empty original
        result = CFConventions.update_cell_methods(
            '', 'area', 'mean', 'where land'
        )
        assert result == 'area: mean (where land)'
        
        # Existing methods
        result = CFConventions.update_cell_methods(
            'time: mean', 'area', 'sum'
        )
        assert result == 'time: mean area: sum'
        
        # Replace existing dimension
        result = CFConventions.update_cell_methods(
            'time: mean area: sum', 'area', 'mean'
        )
        assert result == 'time: mean area: mean'
    
    def test_variable_type_requires_explicit_contract(self):
        """CF names and units must not infer the regridding quantity role."""
        synchronizer = CFAttributeSynchronizer()

        for variable_type in ("extensive", "intensive", "categorical"):
            assert synchronizer.infer_variable_type(
                {"variable_type": variable_type}
            ) == variable_type

        for metadata in (
            {"standard_name": "precipitation_flux"},
            {"standard_name": "air_temperature", "units": "K"},
            {"variable_name": "land_cover_type", "units": "1"},
            {},
            {"variable_type": "not_a_role"},
            {"variable_type": 17},
        ):
            with pytest.raises(ValueError, match="explicit variable_type"):
                synchronizer.infer_variable_type(metadata)

    def test_conserve_check_requires_explicit_variable_type(self):
        dataset = xr.Dataset({"temperature": ("cell", [280.0, 281.0])})
        dataset["temperature"].attrs = {
            "standard_name": "air_temperature",
            "units": "K",
        }

        is_conserved, diagnostics = conserve_check(dataset)

        assert is_conserved is False
        assert diagnostics["passed"] is False
        assert any(
            "explicit variable_type" in violation
            for violation in diagnostics["violations"]
        )

    def test_conserve_check_default_tolerance_is_the_canonical_scale_key(self, monkeypatch):
        # Mode-less scale conservation checks read exactly one key,
        # coupling.scales.defaults.conservation_tolerance; there is no literal default.
        from relic.parameters import get_global_manager

        captured = []

        def recording(var, dataset, tolerance, cell_methods):
            captured.append(tolerance)
            return True, 0.0

        monkeypatch.setattr(cf_module, "_check_spatial_conservation", recording)
        dataset = xr.Dataset(
            {"stock": (("x",), np.ones(2), {"units": "kg", "variable_type": "extensive"})}
        )
        passed, _ = conserve_check(dataset)
        expected = float(
            get_global_manager().get("coupling.scales.defaults.conservation_tolerance", required=True)
        )
        assert passed is True
        assert captured == [expected]
        assert inspect.signature(conserve_check).parameters["tolerance"].default is None

    def test_conserve_check_reports_a_missing_units_attribute(self):
        # A missing units attribute is a finding about the dataset, not an empty unit string.
        dataset = xr.Dataset({"stock": (("cell",), np.ones(2), {"variable_type": "extensive"})})
        passed, diagnostics = conserve_check(dataset, tolerance=1e-8)
        assert passed is False
        assert any("stock" in v and "units" in v for v in diagnostics["violations"])
        assert diagnostics["variable_results"]["stock"]["conserved"] is False

        dataset["stock"].attrs["units"] = "kg"
        passed, diagnostics = conserve_check(dataset, tolerance=1e-8)
        assert passed is True and diagnostics["violations"] == []

    def test_conservation_helpers_raise_their_evaluation_errors(self):
        """The helpers raise; conserve_check records the error and fails the report."""
        class BrokenSpatialVariable:
            dims = ("x",)

            @property
            def values(self):
                raise RuntimeError("broken spatial values")

        class NoAreas:
            data_vars = {}
            coords = {}

        with pytest.raises(RuntimeError, match="broken spatial values"):
            cf_module._check_spatial_conservation(
                BrokenSpatialVariable(),
                NoAreas(),
                1e-6,
                "",
            )

        class BrokenStock:
            dims = ("time",)
            time = [0, 1]

            @property
            def values(self):
                raise RuntimeError("broken stock values")

        class Flux:
            name = "water_flux"
            dims = ("time",)
            time = [0, 1]
            values = np.array([1.0, 1.0])

        class Dataset:
            data_vars = {"water": BrokenStock()}

            def __getitem__(self, name):
                return self.data_vars[name]

        with pytest.raises(RuntimeError, match="broken stock values"):
            cf_module._check_temporal_conservation(
                Flux(),
                Dataset(),
                1e-6,
            )

    def test_temporal_conservation_uses_seconds_per_day_ssot(self, monkeypatch):
        class TimeAxis:
            values = [0.0, 1.0]

            def __len__(self):
                return len(self.values)

        class Stock:
            dims = ("time",)
            time = TimeAxis()
            values = np.array([0.0, float(SECONDS_PER_DAY)])

        class Flux:
            name = "water_flux"
            dims = ("time",)
            time = TimeAxis()
            values = np.array([1.0, 1.0])

        class Dataset:
            data_vars = {"water": Stock()}

            def __getitem__(self, name):
                return self.data_vars[name]

        monkeypatch.setattr(
            physical_constants_module,
            "get_physical_constant",
            lambda name: float(SECONDS_PER_DAY),
        )
        baseline_ok, baseline_error = cf_module._check_temporal_conservation(
            Flux(), Dataset(), 1e-12
        )

        monkeypatch.setattr(
            physical_constants_module,
            "get_physical_constant",
            lambda name: float(SECONDS_PER_DAY) / 2.0,
        )
        scaled_ok, scaled_error = cf_module._check_temporal_conservation(
            Flux(), Dataset(), 1e-12
        )

        assert bool(baseline_ok) is True
        assert baseline_error == 0.0
        assert bool(scaled_ok) is False
        assert scaled_error == pytest.approx(0.5)

    def test_temporal_conservation_has_no_local_day_literal(self):
        source = inspect.getsource(cf_module._check_temporal_conservation)

        assert "86400" not in source
        assert "get_physical_constant" in source
    
    def test_units_validation(self):
        """Test units validation."""
        # Valid CF units
        assert CFConventions.validate_units(None, 'K')
        assert CFConventions.validate_units(None, 'kg m-2 s-1')
        assert CFConventions.validate_units(None, '1')
        assert CFConventions.validate_units(None, 'degrees_north')
        
        # Invalid units (basic validation)
        assert not CFConventions.validate_units(None, 'invalid@unit')
        assert not CFConventions.validate_units(None, 'custom_unit')


class TestCFOfficialV94Catalog:
    """Executable oracles for the bundled official standard-name table."""

    def test_catalog_uses_official_artifact_without_parameter_manager(self):
        CFConventions._load_catalog(force=True)
        metadata = CFConventions.get_table_metadata()

        assert metadata == {
            "version_number": "94",
            "conventions": "CF-StandardNameTable-94",
            "last_modified": "2026-06-09T17:23:36Z",
            "entry_count": 5071,
            "alias_count": 599,
        }
        assert not hasattr(cf_module, "_get_global_manager")
        assert not hasattr(cf_module, "_read_cf_catalog_yaml")
        assert not hasattr(cf_module, "_extract_param_value")

    def test_duplicate_parameter_catalog_file_is_absent(self):
        repo_root = Path(__file__).resolve().parents[4]
        catalog_path = (
            repo_root
            / "relic"
            / "parameters"
            / "database"
            / "relic"
            / "conversions"
            / "cf_catalog.yaml"
        )

        assert not catalog_path.exists()

    @pytest.mark.parametrize(
        "name",
        [
            "air_temperature",
            "gross_primary_productivity_of_carbon",
            "air_temperature detection_minimum",
            "air_temperature number_of_observations",
            "air_temperature standard_error",
            "air_temperature status_flag",
        ],
    )
    def test_official_entry_alias_and_modifiers_are_members(self, name):
        assert CFConventions.is_valid_standard_name(name)
        assert CFConventions.get_standard_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "totally_fake_cf_name",
            "temperature",
            "Air_temperature",
            "",
            " air_temperature",
            "air_temperature ",
            "air_temperature\tstandard_error",
            "air_temperature unknown_modifier",
            "air_temperature standard_error status_flag",
            None,
            17,
        ],
    )
    def test_unknown_or_malformed_standard_names_fail_closed(self, name):
        assert not CFConventions.is_valid_standard_name(name)
        with pytest.raises((TypeError, ValueError)):
            CFConventions.get_standard_name(name)

    def test_cross_kind_collisions_resolve_entry_first_with_alias_provenance(self):
        ocean = CFConventions.get_standard_name_info("ocean_volume")
        precipitation = CFConventions.get_standard_name_info(
            "convective_precipitation_rate"
        )

        assert ocean["membership_kind"] == "entry"
        assert ocean["standard_name"] == "ocean_volume"
        assert ocean["canonical_units"] == "m3"
        assert ocean["colliding_alias_targets"] == ("sea_water_volume",)
        assert precipitation["membership_kind"] == "entry"
        assert precipitation["standard_name"] == "convective_precipitation_rate"
        assert precipitation["canonical_units"] == "m s-1"
        assert precipitation["colliding_alias_targets"] == (
            "lwe_convective_precipitation_rate",
        )

    def test_multitarget_alias_identity_is_preserved_without_target_guessing(self):
        name = "surface_carbon_dioxide_mole_flux"
        info = CFConventions.get_standard_name_info(name)

        assert CFConventions.get_standard_name(name) == name
        assert info["membership_kind"] == "alias"
        assert info["canonical_candidates"] == (
            "surface_downward_mole_flux_of_carbon_dioxide",
            "surface_upward_mole_flux_of_carbon_dioxide",
        )
        assert info["semantic_resolution"] == "ambiguous"
        assert info["canonical_units"] == "mol m-2 s-1"

    def test_undefined_canonical_unit_is_not_dimensionless_or_locally_filled(self):
        info = CFConventions.get_standard_name_info("area_type")

        assert info["membership_kind"] == "entry"
        assert info["canonical_units"] is None
        assert info["unit_contract_defined"] is False
        assert CFConventions.is_valid_standard_name("area_type")
        assert CFConventions.get_standard_units("area_type") is None
        with pytest.raises(ValueError, match="unit contract is not defined"):
            CFConventions.are_standard_name_units_equivalent("area_type", "1")

    @pytest.mark.parametrize("actual", ["dB", "decibel", "db", "dbel"])
    def test_official_db_spellings_are_a_narrow_unit_exception(self, actual):
        name = "sound_pressure_level_in_water"
        assert CFConventions.get_standard_units(name) == "dB"
        assert CFConventions.are_standard_name_units_equivalent(name, actual)

    @pytest.mark.parametrize(
        "actual",
        ["1", "dimensionless", "DB", "Decibel", "decibels", "dBZ", "dBW"],
    )
    def test_db_is_not_dimensionless_or_fuzzy(self, actual):
        assert not CFConventions.are_standard_name_units_equivalent(
            "sound_pressure_level_in_water",
            actual,
        )

    def test_canonical_unit_equivalence_uses_udunits_convertibility(self):
        assert CFConventions.are_standard_name_units_equivalent(
            "air_temperature", "K"
        )
        assert CFConventions.are_standard_name_units_equivalent(
            "air_temperature", "degree_Celsius"
        )
        assert not CFConventions.are_standard_name_units_equivalent(
            "air_temperature", "Pa"
        )
        assert not CFConventions.are_standard_name_units_equivalent(
            "runoff_amount", "kg m-2 s-1"
        )
        assert CFConventions.get_standard_units("runoff_amount") == "kg m-2"

    def test_modifier_unit_contracts_do_not_inherit_invalid_base_units(self):
        assert CFConventions.get_standard_units(
            "air_temperature number_of_observations"
        ) == "1"
        assert CFConventions.get_standard_units("air_temperature status_flag") is None
        assert CFConventions.are_standard_name_units_equivalent(
            "air_temperature number_of_observations",
            "1",
        )
        assert CFConventions.are_standard_name_units_equivalent(
            "air_temperature status_flag",
            None,
        )
        assert not CFConventions.are_standard_name_units_equivalent(
            "air_temperature status_flag",
            "K",
        )

    def test_temperature_standard_error_units_metadata_is_checked_when_present(self):
        valid = CFConventions.validate_cf_compliance(
            {
                "standard_name": "air_temperature standard_error",
                "units": "degree_Celsius",
                "units_metadata": "temperature: difference",
            }
        )
        invalid = CFConventions.validate_cf_compliance(
            {
                "standard_name": "air_temperature standard_error",
                "units": "degree_Celsius",
                "units_metadata": "temperature: on_scale",
            }
        )

        assert valid.is_valid
        assert not invalid.is_valid
        assert any("temperature: difference" in issue for issue in invalid.issues)

    def test_scope_limited_validator_rejects_fake_name_and_wrong_units(self):
        fake = CFConventions.validate_cf_compliance(
            {"standard_name": "totally_fake_cf_name", "units": "custom_unit"}
        )
        wrong_units = CFConventions.validate_cf_compliance(
            {"standard_name": "air_temperature", "units": "Pa"}
        )
        valid = CFConventions.validate_cf_compliance(
            {"standard_name": "air_temperature", "units": "degree_Celsius"}
        )

        assert not fake.is_valid
        assert not wrong_units.is_valid
        assert valid.is_valid
        assert valid.membership_checked is True
        assert valid.unit_contract_defined is True
        assert valid.unit_equivalence_checked is True
        assert valid.dataset_conformance_checked is False


class TestCFMetadataMutationBoundary:
    """P2 oracles for effect-free validation before metadata mutation."""

    def test_explicit_standard_name_and_equivalent_units_are_preserved(self):
        metadata = {
            "variable_name": "local_temperature",
            "standard_name": "air_temperature",
            "units": "degree_Celsius",
            "provenance": "source-file",
        }

        result = CFConventions.sync_variable_attributes(
            torch.tensor([20.0]),
            metadata,
        )

        assert result["standard_name"] == "air_temperature"
        assert result["units"] == "degree_Celsius"
        assert result["provenance"] == "source-file"
        assert metadata == {
            "variable_name": "local_temperature",
            "standard_name": "air_temperature",
            "units": "degree_Celsius",
            "provenance": "source-file",
        }

    def test_missing_standard_name_and_units_remain_missing(self):
        metadata = {"variable_name": "mystery_signal", "long_name": "Mystery"}

        result = CFConventions.sync_variable_attributes(
            torch.tensor([1.0]),
            metadata,
        )

        assert "standard_name" not in result
        assert "units" not in result
        assert metadata == {
            "variable_name": "mystery_signal",
            "long_name": "Mystery",
        }

    @pytest.mark.parametrize(
        "metadata",
        [
            {"standard_name": "totally_fake_cf_name", "units": "K"},
            {"standard_name": "air_temperature", "units": "Pa"},
            {"standard_name": "air_temperature"},
            {"units": "custom_unit"},
        ],
    )
    def test_invalid_explicit_metadata_fails_without_mutating_input(self, metadata):
        before = dict(metadata)

        with pytest.raises(ValueError, match="metadata"):
            CFConventions.sync_variable_attributes(
                torch.tensor([1.0]),
                metadata,
            )

        assert metadata == before

    def test_global_attribute_helper_does_not_self_declare_cf_conformance(self):
        attrs = CFConventions.create_global_attributes(title="Scoped output")

        assert "Conventions" not in attrs
        assert attrs["title"] == "Scoped output"
        assert "creation_date" in attrs


def test_data_public_cf_helper_preserves_single_variable_scope(monkeypatch):
    import relic.data as data_package

    assert not hasattr(data_package, "check_cf_compliance")
    assert "check_cf_compliance" not in data_package.__all__
    assert "check_cf_variable_metadata" in data_package.__all__

    result = data_package.check_cf_variable_metadata(
        {"standard_name": "air_temperature", "units": "K"}
    )
    assert result.is_valid is True
    assert result.validation_scope == "single_variable_standard_name_and_units"
    assert result.dataset_conformance_checked is False
    assert result.coordinates_checked is False
    assert result.cell_methods_checked is False

    monkeypatch.setattr(data_package, "_get_symbol", lambda _name: None)
    assert data_package.validate_units(1.0, "K") is False


def test_canonical_cf_synchronizer_does_not_claim_dataset_conformance():
    synchronizer = CFAttributeSynchronizer()
    metadata = synchronizer.sync_after_aggregation(
        torch.tensor([280.0], dtype=torch.float64),
        {"standard_name": "air_temperature", "units": "K"},
        aggregation_method="mean",
        dimension="time",
        source_label="hourly",
        target_label="daily",
    )

    assert callable(synchronizer.sync_after_aggregation)
    assert callable(synchronizer.sync_after_disaggregation)
    assert callable(synchronizer.sync_after_regridding)
    assert callable(synchronizer.sync_after_temporal_aggregation)
    assert metadata["standard_name"] == "air_temperature"
    assert metadata["units"] == "K"
    assert metadata["cell_methods"] == "time: mean"
    assert "cf_compliant" not in metadata
    assert "dataset_conformance_checked" not in metadata


def test_temporal_sync_requires_coverage_and_reads_the_threshold_parameter():
    """The synchronizer forwards explicit coverage; no 0.8 or 1.0 fallback."""
    from relic.coupling.scales.temporal.core import TimeCoordinate

    synchronizer = CFAttributeSynchronizer()
    time_coord = TimeCoordinate(
        values=torch.tensor([0.5], dtype=torch.float64),
        units="days since 2000-01-01",
        calendar="noleap",
        bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    )
    signature = inspect.signature(synchronizer.sync_after_temporal_aggregation)
    assert signature.parameters["coverage_stats"].default is inspect.Parameter.empty

    metadata = synchronizer.sync_after_temporal_aggregation(
        torch.tensor([280.0], dtype=torch.float64),
        {"standard_name": "air_temperature", "units": "K"},
        time_coord,
        "aggregation",
        "mean",
        {"valid_points": 18, "total_points": 24},
        source_label="hourly",
        target_label="daily",
    )

    assert metadata["coverage_fraction"] == pytest.approx(0.75)
    assert metadata["coverage_threshold"] == pytest.approx(0.8)
    assert metadata["quality_state"] == "insufficient_coverage"


class TestCFTemporalRescalingSync:
    """Temporal rescaling keeps the variable's identity; the time axis is described apart.

    A rate stays a rate and an interval accumulation keeps its accumulated unit:
    only cell_methods ("time: <method>"), history, coverage statistics and the
    attributes of the result time coordinate are written.
    """

    _RATE = {
        "standard_name": "precipitation_flux",
        "units": "kg m-2 s-1",
        "long_name": "Precipitation rate",
        "cell_methods": "area: mean",
        "history": "created",
    }
    _COVERAGE = {"valid_points": 48, "total_points": 48}

    @staticmethod
    def _time_coord(bounds=True):
        from relic.coupling.scales.temporal.core import TimeCoordinate

        return TimeCoordinate(
            values=torch.tensor([0.5, 1.5], dtype=torch.float64),
            units="days since 2000-01-01",
            calendar="noleap",
            bounds=(
                torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64) if bounds else None
            ),
        )

    def _sync(self, metadata, method="mean", operation="aggregation", strict=False, bounds=True,
              labels=("hourly", "daily")):
        return CFAttributeSynchronizer(strict_mode=strict).sync_after_temporal_aggregation(
            torch.tensor([2.0e-5, 3.0e-5], dtype=torch.float64),
            metadata,
            self._time_coord(bounds),
            operation,
            method,
            dict(self._COVERAGE),
            source_label=labels[0],
            target_label=labels[1],
        )

    def test_rate_keeps_its_standard_name_and_units(self):
        metadata = dict(self._RATE)
        before = dict(metadata)

        updated = self._sync(metadata)

        assert updated["standard_name"] == "precipitation_flux"
        assert updated["units"] == "kg m-2 s-1"
        assert updated["long_name"] == "Precipitation rate"
        assert updated["cell_methods"] == "area: mean time: mean"
        assert updated["history"] == "created; Aggregated time from hourly to daily (time: mean)"
        assert updated["time_coordinate"] == {
            "standard_name": "time",
            "long_name": "time",
            "axis": "T",
            "units": "days since 2000-01-01",
            "calendar": "noleap",
            "bounds": "time_bounds",
        }
        for coordinate_only in ("calendar", "axis", "bounds", "has_time_bounds"):
            assert coordinate_only not in updated
        assert updated["coverage_fraction"] == pytest.approx(1.0)
        assert updated["quality_state"] == "good_data"
        assert metadata == before

    def test_interval_accumulation_keeps_its_accumulated_unit(self):
        updated = self._sync({"standard_name": "precipitation_amount", "units": "kg m-2"}, method="sum")

        assert updated["standard_name"] == "precipitation_amount"
        assert updated["units"] == "kg m-2"
        assert updated["cell_methods"] == "time: sum"

    def test_time_coordinate_without_bounds_declares_no_bounds(self):
        updated = self._sync(dict(self._RATE), bounds=False)

        assert "bounds" not in updated["time_coordinate"]

    def test_temporal_disaggregation_uses_the_disaggregation_contract(self):
        updated = self._sync(dict(self._RATE), operation="disaggregation", labels=("daily", "hourly"))

        assert updated["units"] == "kg m-2 s-1"
        assert updated["history"] == "created; Disaggregated time from daily to hourly (time: mean)"
        with pytest.raises(ValueError, match="disaggregation_method"):
            self._sync(dict(self._RATE), method="maximum", operation="disaggregation")
        with pytest.raises(ValueError, match="operation"):
            self._sync(dict(self._RATE), operation="regridding")

    def test_state_disaggregated_to_instants_is_a_time_point(self):
        """state_at_boundary disaggregation yields instants: "time: point" (CF-1.8 7.3)."""
        updated = self._sync(
            {"standard_name": "air_temperature", "units": "K"},
            method="point",
            operation="disaggregation",
            labels=("daily", "hourly"),
        )

        assert updated["cell_methods"] == "time: point"
        assert updated["units"] == "K"
        with pytest.raises(ValueError, match="disaggregation_method"):
            CFAttributeSynchronizer().sync_after_disaggregation(
                torch.tensor([1.0], dtype=torch.float64),
                {"units": "kg"},
                "point",
                dimension="area",
                source_label="hillslope",
                target_label="patch",
            )
        with pytest.raises(ValueError, match="no disaggregation cell-method contract"):
            CFAttributeSynchronizer().sync_after_disaggregation(
                torch.tensor([1.0], dtype=torch.float64),
                {"units": "kg"},
                "sum",
                dimension="depth",
                source_label="layer",
                target_label="sublayer",
            )

    def test_strict_mode_accepts_a_valid_variable_and_rejects_invalid_metadata(self):
        updated = self._sync(dict(self._RATE), strict=True)
        assert updated["standard_name"] == "precipitation_flux"

        with pytest.raises(ValueError, match="CF standard-name metadata issues"):
            self._sync({"standard_name": "area_type", "units": "1"}, strict=True)

    def test_signature_requires_labels_and_has_no_history_parameters(self):
        parameters = inspect.signature(CFAttributeSynchronizer.sync_after_temporal_aggregation).parameters

        assert "parameters" not in parameters
        for label in ("source_label", "target_label"):
            assert parameters[label].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[label].default is inspect.Parameter.empty
        for removed in (
            "synchronize_cf_temporal_metadata",
            "create_time_bounds_metadata",
            "create_history_entry",
            "update_history",
            "validate_cf_compliance_temporal",
        ):
            assert not hasattr(cf_module, removed)
            assert removed not in cf_module.__all__


class TestCFRescalingSync:
    """Aggregation and disaggregation share one CF synchronization contract.

    cell_methods follow the declared spatial basis: a cell_total field is a
    per-cell sum ("area: sum"); intensive and area_density fields are per-cell
    means ("area: mean"). Each rescaling step appends one history entry that
    names the explicit source and target scale labels.
    """

    _CELL_TOTAL = {"units": "kg", "cell_methods": "area: sum", "history": "created"}
    _INTENSIVE = {"standard_name": "air_temperature", "units": "K", "cell_methods": "time: mean"}

    def test_disaggregation_of_cell_total_writes_area_sum_and_history(self):
        metadata = dict(self._CELL_TOTAL)
        before = dict(metadata)
        # Area-proportional split of [6, 4] over child areas [1, 3 | 2, 2].
        data = torch.tensor([1.5, 4.5, 2.0, 2.0], dtype=torch.float64)

        updated = CFAttributeSynchronizer().sync_after_disaggregation(
            data,
            metadata,
            disaggregation_method="sum",
            dimension="area",
            source_label="hillslope",
            target_label="patch",
        )

        assert updated["cell_methods"] == "area: sum"
        assert updated["history"] == "created; Disaggregated area from hillslope to patch (area: sum)"
        assert updated["units"] == "kg"
        assert updated["shape"] == [4]
        assert updated["actual_range"] == [1.5, 4.5]
        assert metadata == before

    def test_disaggregation_of_intensive_field_writes_area_mean(self):
        updated = CFAttributeSynchronizer().sync_after_disaggregation(
            torch.tensor([280.0, 280.0, 285.0], dtype=torch.float64),
            dict(self._INTENSIVE),
            disaggregation_method="mean",
            dimension="area",
            source_label="hillslope",
            target_label="patch",
        )

        assert updated["cell_methods"] == "time: mean area: mean"
        assert updated["history"] == "Disaggregated area from hillslope to patch (area: mean)"
        assert updated["standard_name"] == "air_temperature"

    def test_aggregation_appends_history_with_explicit_labels(self):
        updated = CFAttributeSynchronizer().sync_after_aggregation(
            torch.tensor([6.0, 4.0], dtype=torch.float64),
            dict(self._CELL_TOTAL),
            aggregation_method="sum",
            dimension="area",
            source_label="patch",
            target_label="hillslope",
        )

        assert updated["cell_methods"] == "area: sum"
        assert updated["history"] == "created; Aggregated area from patch to hillslope (area: sum)"

    def test_both_directions_share_one_contract_apart_from_the_history_verb(self):
        synchronizer = CFAttributeSynchronizer()
        data = torch.tensor([280.0, 285.0], dtype=torch.float64)
        kwargs = {"dimension": "area", "source_label": "a", "target_label": "b"}

        aggregated = synchronizer.sync_after_aggregation(data, dict(self._INTENSIVE), "mean", **kwargs)
        disaggregated = synchronizer.sync_after_disaggregation(data, dict(self._INTENSIVE), "mean", **kwargs)

        assert aggregated.pop("history") == "Aggregated area from a to b (area: mean)"
        assert disaggregated.pop("history") == "Disaggregated area from a to b (area: mean)"
        assert aggregated == disaggregated
        assert not hasattr(CFAttributeSynchronizer, "_sync_temporal_default")

    @pytest.mark.parametrize("method_name", ["sync_after_aggregation", "sync_after_disaggregation"])
    def test_scale_labels_are_required_keyword_arguments(self, method_name):
        synchronizer = CFAttributeSynchronizer()
        parameters = inspect.signature(getattr(synchronizer, method_name)).parameters
        for label in ("source_label", "target_label"):
            assert parameters[label].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[label].default is inspect.Parameter.empty

        method = getattr(synchronizer, method_name)
        data = torch.tensor([1.0], dtype=torch.float64)
        with pytest.raises(TypeError, match="source_label"):
            method(data, dict(self._CELL_TOTAL), "sum", dimension="area", target_label="patch")
        for bad in ("", "   ", None, 3):
            with pytest.raises(ValueError, match="source_label"):
                method(data, dict(self._CELL_TOTAL), "sum", dimension="area",
                       source_label=bad, target_label="patch")
            with pytest.raises(ValueError, match="target_label"):
                method(data, dict(self._CELL_TOTAL), "sum", dimension="area",
                       source_label="hillslope", target_label=bad)

    @pytest.mark.parametrize("method_name", ["sync_after_aggregation", "sync_after_disaggregation"])
    def test_rescaled_dimension_is_a_required_argument(self, method_name):
        # The dimension names the cell_methods entry that is written; neither
        # direction supplies one on the caller's behalf.
        method = getattr(CFAttributeSynchronizer(), method_name)
        assert inspect.signature(method).parameters["dimension"].default is inspect.Parameter.empty
        with pytest.raises(TypeError, match="dimension"):
            method(torch.tensor([1.0], dtype=torch.float64), dict(self._CELL_TOTAL), "sum",
                   source_label="hillslope", target_label="patch")

    @pytest.mark.parametrize("method", ["maximum", "linear", "point", ""])
    def test_disaggregation_accepts_only_sum_or_mean(self, method):
        with pytest.raises(ValueError, match="disaggregation_method"):
            CFAttributeSynchronizer().sync_after_disaggregation(
                torch.tensor([1.0], dtype=torch.float64),
                dict(self._CELL_TOTAL),
                method,
                dimension="area",
                source_label="hillslope",
                target_label="patch",
            )

    @pytest.mark.parametrize("method_name", ["sync_after_aggregation", "sync_after_disaggregation"])
    def test_strict_mode_validation_is_identical_in_both_directions(self, method_name):
        # area_type is a CF entry without a canonical unit contract: accepted
        # outside strict mode, rejected by strict-mode validation.
        metadata = {"standard_name": "area_type", "units": "1"}
        data = torch.tensor([1.0, 2.0], dtype=torch.float64)
        labels = {"dimension": "area", "source_label": "hillslope", "target_label": "patch"}

        relaxed = getattr(CFAttributeSynchronizer(), method_name)(data, dict(metadata), "sum", **labels)
        assert relaxed["cell_methods"] == "area: sum"
        with pytest.raises(
            ValueError,
            match=r"CF standard-name metadata issues: .*area_type' unit contract is not defined",
        ):
            getattr(CFAttributeSynchronizer(strict_mode=True), method_name)(
                data, dict(metadata), "sum", **labels
            )


class TestCFVariableIdentity:
    """CF-1.8 3.3/7.3: identity = standard_name + qualifiers; one owner, one holder."""

    def test_identity_defaults_the_standard_name_and_normalizes_cell_methods(self):
        identity = cf_module.cf_variable_identity(
            "tasmax", {"standard_name": "air_temperature", "cell_methods": "time:   maximum"}
        )
        implicit = cf_module.cf_variable_identity("litter_mass", {"units": "kg"})

        assert identity.key == ("air_temperature", (("cell_methods", "time: maximum"),))
        assert identity.standard_name_declared is True
        assert implicit.key == ("litter_mass", ())
        assert implicit.standard_name_declared is False
        assert cf_module.CF_STANDARD_NAME_QUALIFIERS == ("cell_methods",)

    @pytest.mark.parametrize(
        "declaration, message",
        [
            ({"standard_name": "air_temperature", "cell_methods": "  "}, "cell_methods"),
            ({"standard_name": "air_temperature", "cell_methods": 3}, "cell_methods"),
            ({"standard_name": "   "}, "standard_name"),
        ],
    )
    def test_invalid_declarations_raise_with_the_variable_named(self, declaration, message):
        with pytest.raises(ValueError, match=rf"'tas'.*{message}"):
            cf_module.cf_variable_identity("tas", declaration)

    def test_two_declarations_of_one_variable_merge_or_raise(self):
        declared = cf_module.cf_variable_identity(
            "net_radiation", {"standard_name": "surface_net_downward_radiative_flux"}
        )
        implicit = cf_module.cf_variable_identity("net_radiation", {})

        assert cf_module.merge_cf_variable_identity("net_radiation", implicit, declared) == declared
        assert cf_module.merge_cf_variable_identity("net_radiation", declared, implicit) == declared
        other = cf_module.cf_variable_identity("net_radiation", {"standard_name": "net_radiation"})
        with pytest.raises(ValueError, match=r"'net_radiation'.*surface_net_downward_radiative_flux"):
            cf_module.merge_cf_variable_identity("net_radiation", declared, other)
        qualified = cf_module.cf_variable_identity(
            "net_radiation",
            {"standard_name": "surface_net_downward_radiative_flux", "cell_methods": "time: mean"},
        )
        assert cf_module.merge_cf_variable_identity("net_radiation", declared, qualified) == qualified
        with pytest.raises(ValueError, match="cell_methods"):
            cf_module.merge_cf_variable_identity(
                "net_radiation",
                qualified,
                cf_module.cf_variable_identity(
                    "net_radiation",
                    {"standard_name": "surface_net_downward_radiative_flux", "cell_methods": "time: sum"},
                ),
            )

    def test_one_identity_has_one_holder(self):
        holders = {}
        tas = cf_module.cf_variable_identity("tas", {"standard_name": "air_temperature"})
        tasmax = cf_module.cf_variable_identity(
            "tasmax", {"standard_name": "air_temperature", "cell_methods": "time: maximum"}
        )

        cf_module.register_cf_identity_holder(holders, "tas", tas)
        cf_module.register_cf_identity_holder(holders, "tasmax", tasmax)
        cf_module.register_cf_identity_holder(holders, "tas", tas)
        with pytest.raises(ValueError, match=r"'tas'.*'tair'"):
            cf_module.register_cf_identity_holder(
                holders, "tair", cf_module.cf_variable_identity("tair", {"standard_name": "air_temperature"})
            )
        assert holders == {tas.key: "tas", tasmax.key: "tasmax"}


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])


def test_conserve_check_marks_the_report_failed_when_a_check_raises(monkeypatch):
    """A variable whose conservation check raises is recorded and fails the report."""
    import numpy as np
    import xarray as xr

    from relic.data.standards import cf_conventions

    def exploding(*args, **kwargs):
        raise ValueError("cell_measures missing")

    monkeypatch.setattr(cf_conventions, "_check_spatial_conservation", exploding)
    dataset = xr.Dataset(
        {
            "flux": (
                ("time", "x"),
                np.ones((2, 2)),
                {"units": "kg m-2 s-1", "cell_methods": "area: sum", "variable_type": "extensive"},
            )
        }
    )
    passed, diagnostics = cf_conventions.conserve_check(dataset, check_variables=["flux"])
    assert passed is False and diagnostics["passed"] is False
    assert diagnostics["variable_results"]["flux"]["checked"] is False
    assert "cell_measures missing" in diagnostics["variable_results"]["flux"]["error"]
    # A requested variable the dataset does not hold fails the report as well.
    missing_passed, _ = cf_conventions.conserve_check(dataset, check_variables=["not_there"])
    assert missing_passed is False
