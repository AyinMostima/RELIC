"""
Unit tests for PFT registry and parameter loading.
Tests the fixes for:
1. Index error in pft_registry.py  
2. Unknown PFT default handling
3. Temperature unit consistency
4. Parameter completeness
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch

from relic.scientific.biosphere.pft.pft_registry import PFTRegistry, PFTDefinition
from relic.scientific.biosphere.pft.pft_types import (
    PFTCategory,
    PhenologyParameters,
)
from relic.parameters import get_global_manager, ParameterSpec

@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _load_pft_parameter_files(param_manager) -> None:
    pft_dir = Path("relic/parameters/database/relic/biosphere/pft")
    if not pft_dir.exists():
        return
    skip = {"__init__", "split_validation_report", "phase_6_4_cleanup_report"}
    yaml_files = sorted(pft_dir.glob("*.yaml"))
    for yaml_file in yaml_files:
        if yaml_file.stem in skip:
            continue
        param_manager.load_from_database(str(yaml_file))


class TestPFTRegistry:
    """Test suite for PFT registry functionality."""
    
    @pytest.fixture
    def setup_registry(self, restore_pft_registry_singleton):
        """Setup test registry with clean state (the singleton is put back afterwards)."""
        # Clear existing registry
        PFTRegistry._instance = None
        registry = PFTRegistry.get_instance()
        registry._pft_definitions.clear()
        
        # Reset global parameter manager and load PFT parameters
        param_manager = get_global_manager()
        
        # Load the actual PFT parameter database
        _load_pft_parameter_files(param_manager)
        
        return registry
    
    def test_singleton_pattern(self):
        """Test that PFTRegistry follows singleton pattern."""
        registry1 = PFTRegistry.get_instance()
        registry2 = PFTRegistry.get_instance()
        assert registry1 is registry2
    
    def test_pft_name_extraction_from_parameters(self, setup_registry):
        """Test that PFT names are extracted correctly from parameter keys."""
        registry = setup_registry
        param_manager = get_global_manager()
        
        # Verify parameters were loaded from YAML
        all_params = param_manager.list(namespace="biosphere.pft")
        assert len(all_params) > 0, "No parameters loaded from YAML"
        
        # Extract unique PFT names from loaded parameters
        expected_pft_names = set()
        for param_key in all_params:
            if param_key.startswith("biosphere.pft."):
                parts = param_key.split(".")
                if len(parts) >= 4:
                    pft_name = parts[2]
                    expected_pft_names.add(pft_name)
        
        assert len(expected_pft_names) > 0, "No PFT names extracted from parameters"
        
        # The canonical default load materializes the PFTs of the packaged database.
        registry.load_default_pfts()
        
        # Verify PFT names were extracted and registered correctly
        pft_names = registry.list_pfts()
        
        # Should have created PFTs from YAML parameters
        assert len(pft_names) > 0, f"No PFTs created. Expected: {expected_pft_names}"
        
        # Check that expected PFTs were created
        for expected_name in ["c3_grass_temperate", "c4_grass_temperate", "broadleaf_deciduous_temperate"]:
            if expected_name in expected_pft_names:
                assert expected_name in pft_names, f"Expected PFT '{expected_name}' not found in {pft_names}"
        
        # Should NOT have parameter categories as PFT names
        assert "photosynthesis" not in pft_names
        assert "phenology" not in pft_names
        assert "allocation" not in pft_names

    def test_default_load_materializes_each_pft_once(self, setup_registry, caplog):
        """A cold default load must build one typed definition per PFT file."""
        registry = setup_registry
        registry._pft_definitions.clear()
        registry._parameter_files.clear()

        with patch.object(
            registry,
            "register_pft",
            wraps=registry.register_pft,
        ) as register_pft:
            registry.load_default_pfts()

        assert len(registry.list_pfts()) == 15
        assert register_pft.call_count == 15
        assert len(registry._parameter_files) == 16
        assert "Overwriting existing PFT definition" not in caplog.text

    def test_unknown_pft_raises_error(self, setup_registry, tmp_path):
        """Test that unknown PFT types raise ValueError instead of defaulting."""
        registry = setup_registry
        param_manager = get_global_manager()
        parameter_file = tmp_path / "unknown_pft.yaml"
        parameter_file.write_text(
            "namespace: biosphere.pft.unknown_plant_type\nparameters: {}\n",
            encoding="utf-8",
        )
        unknown_key = "biosphere.pft.unknown_plant_type.photosynthesis.vcmax25"

        with (
            patch.object(param_manager, "load_from_database", return_value=1),
            patch.object(param_manager, "list", return_value=[unknown_key]),
        ):
            registry._validation_strict = True
            with pytest.raises(ValueError) as excinfo:
                registry.load_pft_file(parameter_file, lightweight=True)

            assert "Unknown PFT type" in str(excinfo.value)
            assert "unknown_plant_type" in str(excinfo.value)
    
    def test_temperature_units_consistency(self, setup_registry):
        """Test that temperature parameters use canonical absolute units."""
        param_manager = get_global_manager()

        # Load actual PFT parameters
        _load_pft_parameter_files(param_manager)

        # Check temperature parameters
        temp_params = [
            "biosphere.pft.c3_grass_temperate.phenology.temp_growth_opt",
            "biosphere.pft.c4_grass_temperate.phenology.temp_growth_opt",
            "biosphere.pft.c4_grass_temperate.phenology.temp_growth_max"
        ]

        for param_key in temp_params:
            spec = param_manager.get_parameter(param_key)
            assert spec.units == "K", (
                f"{param_key} has units {spec.units}, expected K"
            )
            value = param_manager.get(param_key, required=True)
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert 200 < value < 350, (
                f"{param_key} value {value} outside reasonable K range"
            )
    
    def test_allocation_parameters_complete(self, setup_registry):
        """Test that all required allocation parameters are present."""
        param_manager = get_global_manager()

        # Load actual PFT parameters
        _load_pft_parameter_files(param_manager)

        # Check c3_grass_temperate has all allocation parameters
        allocation_params = ["f_root", "f_leaf", "f_stem"]

        for param in allocation_params:
            param_key = f"biosphere.pft.c3_grass_temperate.allocation.{param}"
            value = param_manager.get(param_key, required=True)
            assert value is not None, f"Missing allocation parameter: {param_key}"

        f_root = param_manager.get(
            "biosphere.pft.c3_grass_temperate.allocation.f_root",
            required=True,
        )
        f_leaf = param_manager.get(
            "biosphere.pft.c3_grass_temperate.allocation.f_leaf",
            required=True,
        )
        f_stem = param_manager.get(
            "biosphere.pft.c3_grass_temperate.allocation.f_stem",
            required=True,
        )
        if isinstance(f_root, torch.Tensor):
            f_root = f_root.item()
        if isinstance(f_leaf, torch.Tensor):
            f_leaf = f_leaf.item()
        if isinstance(f_stem, torch.Tensor):
            f_stem = f_stem.item()

        total = f_root + f_leaf + f_stem
        assert 0.95 <= total <= 1.05, (
            f"Allocation fractions sum to {total}, should be near 1.0"
        )
    
    def test_pft_definition_creation(self, setup_registry):
        """Test that PFT definitions are created correctly."""
        registry = setup_registry
        
        # Create a test PFT definition without parameter manager for tests
        pft_def = PFTDefinition(
            name="test_pft",
            category=PFTCategory.C3_GRASS_TEMPERATE,
            description="Test PFT",
            source="test",
            _use_param_manager=False,  # Important for test PFTs
            parameter_manager=get_global_manager(),
        )
        
        # Register it
        registry.register_pft(pft_def)
        
        # Verify it was registered
        assert "test_pft" in registry.list_pfts()
        retrieved = registry.get_pft("test_pft")
        assert retrieved.name == "test_pft"
        assert retrieved.category == PFTCategory.C3_GRASS_TEMPERATE
        
        # Test listing
        all_pfts = registry.list_pfts()
        assert "test_pft" in all_pfts
    
    def test_get_parameters_from_manager(self, setup_registry):
        """Test that parameters are correctly retrieved from ParameterManager."""
        registry = setup_registry
        param_manager = get_global_manager()
        
        # Register test parameters
        test_value = 25.0
        spec = ParameterSpec(
            key="biosphere.pft.test_pft.phenology.temp_growth_opt",
            value=test_value,
            units="degC",
            bounds=(20.0, 30.0),
            description="Test temperature parameter", dtype="float32", category="model_parameter"
        )
        param_manager.register_parameter(spec)
        
        # Create PhenologyParameters instance and manually test parameter fetching
        phenology_params = PhenologyParameters(
            parameter_manager=param_manager
        )
        phenology_params._pft_namespace = "biosphere.pft.test_pft"
        
        # Test direct parameter fetching without full _fetch_from_parameter_manager
        fetched_value = phenology_params.get_parameter_from_manager(
            f"{phenology_params._pft_namespace}.phenology.temp_growth_opt", 
            required=False
        )
        
        # Verify the parameter was fetched correctly
        assert fetched_value == test_value
        
        
class TestPFTRegistryFailFast:
    """The registry reads the packaged PFT database or raises; it fabricates no PFTs."""

    @pytest.fixture
    def registry(self, restore_pft_registry_singleton):
        PFTRegistry._instance = None
        registry = PFTRegistry.get_instance()
        registry._pft_definitions.clear()
        registry._parameter_files.clear()
        return registry

    def test_default_pft_fabrication_is_retired(self, registry, tmp_path):
        """A directory without PFT files raises instead of creating minimal default
        PFTs; the fallback builder and its manifest are retired."""
        from relic.parameters import get_database_root
        from relic.scientific.biosphere.pft.pft_registry import PFTRegistryError

        assert not hasattr(PFTRegistry, "_create_default_pfts")
        assert not (get_database_root() / "relic" / "biosphere" / "pft_default.yaml").exists()
        with pytest.raises(PFTRegistryError, match="PFT"):
            registry.load_default_pfts(tmp_path)
        (tmp_path / "pft").mkdir()
        with pytest.raises(PFTRegistryError, match="PFT"):
            registry.load_default_pfts(tmp_path)
        # list_pfts() would load the packaged database; the failed loads registered nothing.
        assert registry._pft_definitions == {}

    def test_default_directory_is_the_packaged_relic_database(self, registry):
        """The default directory is the relic tier of the packaged parameter database
        (relic.parameters.get_database_root()), whose pft folder holds one file per PFT."""
        from relic.parameters import get_database_root

        registry.load_default_pfts()
        pft_dir = get_database_root() / "relic" / "biosphere" / "pft"
        assert {path.parent for path in registry._parameter_files} == {pft_dir}
        assert len(registry.list_pfts()) == 15

    def test_pft_file_errors_propagate(self, registry, monkeypatch):
        """An unreadable PFT file stops the default load instead of being logged and
        skipped, whatever the validation strictness."""
        registry._validation_strict = False

        def broken(self, filepath, lightweight=False):
            raise ValueError(f"cannot read {filepath}")

        monkeypatch.setattr(PFTRegistry, "load_pft_file", broken)
        with pytest.raises(ValueError, match="cannot read"):
            registry.load_default_pfts()

    def test_pft_definition_errors_propagate(self, registry, monkeypatch):
        """A PFT whose definition cannot be built raises from load_pft_file in the
        default (non-strict) mode instead of being logged and dropped."""
        import relic.scientific.biosphere.pft.pft_registry as pft_registry
        from relic.parameters import get_database_root

        registry._validation_strict = False

        def broken_definition(*args, **kwargs):
            raise ValueError("missing required PFT parameter")

        monkeypatch.setattr(pft_registry, "PFTDefinition", broken_definition)
        path = get_database_root() / "relic" / "biosphere" / "pft" / "c3_grass_temperate.yaml"
        with pytest.raises(ValueError, match="missing required PFT parameter"):
            registry.load_pft_file(path, lightweight=True)

    def test_unknown_pft_type_raises_without_strict_mode(self, registry, tmp_path):
        """A PFT namespace outside the registered types raises in the default mode too."""
        param_manager = get_global_manager()
        parameter_file = tmp_path / "unknown_pft.yaml"
        parameter_file.write_text(
            "namespace: biosphere.pft.unknown_plant_type\nparameters: {}\n",
            encoding="utf-8",
        )
        registry._validation_strict = False
        with patch.object(param_manager, "load_from_database", return_value=1):
            with pytest.raises(ValueError, match="Unknown PFT type"):
                registry.load_pft_file(parameter_file, lightweight=True)

    def test_validation_reports_only_pft_validation_failures(self, registry, monkeypatch):
        """validate_all_pfts and _validate_pft turn only parameter-value failures into
        PFT validation errors; a defect in a check propagates."""
        registry.load_default_pfts()
        pft = registry.get_pft("c3_grass_temperate")

        def defect(self, definition):
            raise AttributeError("defect in a consistency check")

        monkeypatch.setattr(PFTRegistry, "_validate_stress_consistency", defect)
        registry.clear_cache()
        with pytest.raises(AttributeError, match="defect"):
            registry._validate_pft(pft)
        with pytest.raises(AttributeError, match="defect"):
            registry.validate_all_pfts()

    def test_set_parameter_raises_when_the_manager_rejects_the_value(self, registry, monkeypatch):
        """set_parameter writes the ParameterManager first; a rejected write raises
        instead of warning and leaving the local value and the manager apart."""
        import relic.scientific.biosphere.pft.pft_types as pft_types

        registry.load_default_pfts()
        pft = registry.get_pft("c3_grass_temperate")
        before = pft.photosynthesis.vcmax25

        def reject(key, value, **kwargs):
            raise ValueError(f"rejected {key}")

        monkeypatch.setattr(pft_types, "set_parameter_value", reject)
        with pytest.raises(ValueError, match="rejected"):
            pft.set_parameter("photosynthesis.vcmax25", before * 1.1)
        assert pft.photosynthesis.vcmax25 == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

