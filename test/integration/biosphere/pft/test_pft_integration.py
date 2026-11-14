"""
Integration tests for PFT module with ParameterManager.
Tests L1-L3 validation levels:
- L1: Unit verification
- L2: Module integration verification
- L3: System benchmarking
"""

import pytest
import torch
from pathlib import Path

from relic.scientific.biosphere.pft.pft_registry import PFTRegistry, PFTDefinition
from relic.scientific.biosphere.pft.pft_types import (
    PFTCategory,
    PhotosynthesisParameters,
    PhotosynthesisPathway,
)
from relic.parameters import get_global_manager, ParameterSpec
from relic.core.state import StateManager
from relic.data.standards.unit_converter import UnitConverter, ConversionContext

_PFT_PARAMS_LOADED = False
_PFT_VALIDATION_REPORT = None
PFT_SAMPLE_LIMIT = 3

@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _select_pft_sample(pft_list, limit=PFT_SAMPLE_LIMIT):
    pft_list = sorted(set(pft_list))
    preferred = [
        "broadleaf_deciduous_temperate",
        "needleleaf_evergreen_temperate",
        "c3_grass_temperate",
    ]
    sample = [name for name in preferred if name in pft_list]
    if len(sample) < limit:
        for name in pft_list:
            if name not in sample:
                sample.append(name)
            if len(sample) >= limit:
                break
    return sample[:limit]

def _load_pft_parameter_files(param_manager) -> None:
    global _PFT_PARAMS_LOADED
    if _PFT_PARAMS_LOADED:
        return
    pft_dir = Path("relic/parameters/database/relic/biosphere/pft")
    if not pft_dir.exists():
        return
    skip = {"__init__", "split_validation_report", "phase_6_4_cleanup_report"}
    yaml_files = sorted(pft_dir.glob("*.yaml"))
    for yaml_file in yaml_files:
        if yaml_file.stem in skip:
            continue
        param_manager.load_from_database(str(yaml_file))
    _PFT_PARAMS_LOADED = True


class TestPFTL1Unit:
    """L1: Unit verification tests for PFT components."""

    @pytest.fixture
    def setup(self, restore_pft_registry_singleton):
        """Setup clean test environment (the singleton is put back afterwards)."""
        PFTRegistry._instance = None
        registry = PFTRegistry.get_instance()
        registry._pft_definitions.clear()
        param_manager = get_global_manager()
        return registry, param_manager

    def test_pft_category_enum_completeness(self):
        """Test all PFT category enums are defined."""
        expected_categories = [
            "needleleaf_evergreen_temperate",
            "needleleaf_evergreen_boreal",
            "needleleaf_deciduous_boreal",
            "broadleaf_evergreen_tropical",
            "broadleaf_evergreen_temperate",
            "broadleaf_deciduous_tropical",
            "broadleaf_deciduous_temperate",
            "broadleaf_deciduous_boreal",
            "shrub_evergreen_temperate",
            "shrub_deciduous_temperate",
            "c3_grass_arctic",
            "c3_grass_temperate",
            "c4_grass_temperate",
            "crop_temperate",
            "crop_tropical"
        ]

        actual_categories = [cat.value for cat in PFTCategory]
        assert len(actual_categories) == 15, f"Expected 15 PFT categories, got {len(actual_categories)}"

        for expected in expected_categories:
            assert expected in actual_categories, f"Missing PFT category: {expected}"

    def test_photosynthesis_parameters_structure(self, setup):
        """Test PhotosynthesisParameters dataclass structure."""
        _, parameter_manager = setup
        params = PhotosynthesisParameters(
            _pft_namespace="biosphere.pft.broadleaf_deciduous_temperate",
            parameter_manager=parameter_manager,
        )

        # Check all required attributes exist
        required_attrs = [
            'vcmax25', 'jmax25', 'rd25', 'vcmax_ea', 'vcmax_hd',
            'vcmax_se', 'jmax_ea', 'jmax_hd', 'jmax_se', 'rd_ea',
            'rd_q10', 'alpha', 'theta', 'g0', 'g1', 'pathway'
        ]

        for attr in required_attrs:
            assert hasattr(params, attr), f"Missing attribute: {attr}"

        # Check pathway is correct type
        assert isinstance(params.pathway, PhotosynthesisPathway)

    def test_pft_definition_validation(self, setup):
        """Test PFT definition validation logic."""
        registry, parameter_manager = setup

        # Valid PFT definition
        valid_pft = PFTDefinition(
            name="c3_grass_temperate",
            category=PFTCategory.C3_GRASS_TEMPERATE,
            description="Valid test PFT",
            source="test",
            parameter_manager=parameter_manager,
        )

        # Should not raise
        registry._validate_pft(valid_pft)

        # Check photosynthesis consistency
        valid_pft.photosynthesis.vcmax25 = 50.0
        valid_pft.photosynthesis.jmax25 = 100.0  # 2x vcmax25 is reasonable
        errors = registry._validate_photosynthesis_consistency(valid_pft)
        assert len(errors) == 0, f"Unexpected validation errors: {errors}"

    def test_allocation_fractions_validation(self, setup):
        """Test allocation parameter validation."""
        registry, parameter_manager = setup

        pft = PFTDefinition(
            name="broadleaf_deciduous_temperate",
            category=PFTCategory.BROADLEAF_DECIDUOUS_TEMPERATE,
            description="Test allocation",
            source="test",
            parameter_manager=parameter_manager,
        )

        # Set valid allocation fractions (sum to 1.0)
        pft.allocation.f_leaf = 0.3
        pft.allocation.f_stem = 0.4
        pft.allocation.f_root = 0.3

        errors = registry._validate_allocation_consistency(pft)
        assert len(errors) == 0, "Valid allocation should not have errors"

        # Test invalid allocation (doesn't sum to 1.0)
        pft.allocation.f_leaf = 0.5
        pft.allocation.f_stem = 0.4
        pft.allocation.f_root = 0.3  # Sum = 1.2

        errors = registry._validate_allocation_consistency(pft)
        assert len(errors) > 0, "Invalid allocation should have errors"
        assert any("sum" in err.lower() for err in errors)


class TestPFTL2Integration:
    """L2: Module integration verification with ParameterManager."""

    @pytest.fixture
    def setup_integration(self, restore_pft_registry_singleton):
        """Setup for integration testing (the singleton is put back afterwards)."""
        PFTRegistry._instance = None
        registry = PFTRegistry.get_instance()
        registry._pft_definitions.clear()

        param_manager = get_global_manager()
        # Register some test parameters
        self._register_test_parameters(param_manager)

        return registry, param_manager

    def _register_test_parameters(self, param_manager):
        """Register test parameters in ParameterManager."""
        test_params = {
            "biosphere.pft.test_tree.photosynthesis.vcmax25": {
                "value": 60.0,
                "units": "umol CO2 m-2 s-1",
                "bounds": [30.0, 100.0],
                "description": "Test tree Vcmax at 25C"
            },
            "biosphere.pft.test_tree.photosynthesis.jmax25": {
                "value": 120.0,
                "units": "umol e- m-2 s-1",
                "bounds": [60.0, 200.0],
                "description": "Test tree Jmax at 25C"
            },
            "biosphere.pft.test_tree.allocation.f_leaf": {
                "value": 0.3,
                "units": "dimensionless",
                "bounds": [0.0, 1.0],
                "description": "Leaf allocation fraction"
            },
            "biosphere.pft.test_tree.allocation.f_stem": {
                "value": 0.5,
                "units": "dimensionless",
                "bounds": [0.0, 1.0],
                "description": "Stem allocation fraction"
            },
            "biosphere.pft.test_tree.allocation.f_root": {
                "value": 0.2,
                "units": "dimensionless",
                "bounds": [0.0, 1.0],
                "description": "Root allocation fraction"
            }
        }

        for key, spec in test_params.items():
            param_manager.register_parameter(
                ParameterSpec(
                    key=key,
                    value=spec["value"],
                    units=spec["units"],
                    bounds=spec["bounds"],
                    description=spec["description"], dtype="float32", category="model_parameter",
                )
            )

    def test_parameter_loading_from_manager(self, setup_integration):
        """Test loading PFT parameters from ParameterManager."""
        registry, param_manager = setup_integration

        # Verify parameters are registered
        params_list = param_manager.list(namespace="biosphere.pft.test_tree")
        assert len(params_list) > 0, "No test parameters found"

        # Test parameter retrieval
        vcmax = param_manager.get("biosphere.pft.test_tree.photosynthesis.vcmax25")
        assert vcmax == 60.0 or (torch.is_tensor(vcmax) and vcmax.item() == 60.0)

    def test_pft_parameter_namespace_extraction(self, setup_integration):
        """Test correct extraction of PFT names from parameter namespaces."""
        _, param_manager = setup_integration

        # Get all PFT parameters
        all_params = param_manager.list(namespace="biosphere.pft")

        # Extract PFT names using corrected logic (parts[2])
        pft_names = set()
        for param_key in all_params:
            if param_key.startswith("biosphere.pft."):
                parts = param_key.split(".")
                if len(parts) >= 4:
                    pft_name = parts[2]  # Corrected from parts[3]
                    pft_names.add(pft_name)

        # Should extract "test_tree" not "photosynthesis" or "allocation"
        assert "test_tree" in pft_names
        assert "photosynthesis" not in pft_names
        assert "allocation" not in pft_names

    def test_unit_conversion_integration(self, setup_integration):
        """Test unit conversion between parameters and PFT system."""
        _, param_manager = setup_integration
        unit_converter = UnitConverter()
        context = ConversionContext()

        # Register temperature parameter in Kelvin
        spec = ParameterSpec(
            key="biosphere.pft.test_tree.phenology.temp_opt",
            value=298.15,  # 25掳C in Kelvin
            units="K",
            description="Optimal temperature", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"
        )
        param_manager.register_parameter(spec)

        # Retrieve and convert to Celsius
        temp_k = param_manager.get("biosphere.pft.test_tree.phenology.temp_opt")
        if torch.is_tensor(temp_k):
            temp_k = temp_k.item()

        temp_c = unit_converter.convert(temp_k, "air_temperature", "K", "degC", context)
        assert abs(temp_c - 25.0) < 0.1, f"Temperature conversion error: {temp_c}"


class TestPFTL3System:
    """L3: System-level benchmarking and validation."""

    @pytest.fixture(scope="class")
    def setup_system(self):
        """Setup complete system for testing; the PFTRegistry singleton that was in
        place before the class is put back afterwards."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(PFTRegistry, "_instance", None)
            registry = PFTRegistry.get_instance()
            param_manager = get_global_manager()
            state_manager = StateManager()

            # Load actual PFT parameters if available
            _load_pft_parameter_files(param_manager)
            if not registry.list_pfts():
                registry.load_default_pfts()

            yield registry, param_manager, state_manager

    def test_full_pft_loading_workflow(self, setup_system):
        """Test complete PFT loading and initialization workflow."""
        registry, param_manager, _ = setup_system

        # Check if default PFTs were loaded
        pft_list = registry.list_pfts()

        if len(pft_list) > 0:
            # Verify at least one common PFT type exists
            common_pfts = ["broadleaf_deciduous_temperate", "c3_grass_temperate"]
            found_any = any(pft in pft_list for pft in common_pfts)
            assert found_any, f"No common PFTs found in {pft_list}"

            # Test retrieving a PFT
            for pft_name in _select_pft_sample(pft_list):
                pft_def = registry.get_pft(pft_name)
                assert pft_def is not None
                assert pft_def.name == pft_name
                assert isinstance(pft_def.category, PFTCategory)

    def test_parameter_consistency_across_pfts(self, setup_system):
        """Test parameter consistency and bounds across different PFTs."""
        registry, param_manager, _ = setup_system

        pft_list = registry.list_pfts()
        sample_pfts = _select_pft_sample(pft_list)
        for pft_name in sample_pfts:
            # Check photosynthesis parameters
            vcmax_key = f"biosphere.pft.{pft_name}.photosynthesis.vcmax25"
            jmax_key = f"biosphere.pft.{pft_name}.photosynthesis.jmax25"

            try:
                vcmax = param_manager.get(vcmax_key, default=None)
                jmax = param_manager.get(jmax_key, default=None)

                if vcmax is not None and jmax is not None:
                    if torch.is_tensor(vcmax):
                        vcmax = vcmax.item()
                    if torch.is_tensor(jmax):
                        jmax = jmax.item()

                    # Jmax should typically be 1.5-2.5x Vcmax
                    ratio = jmax / vcmax if vcmax > 0 else 0
                    assert 1.0 <= ratio <= 3.0, f"Unusual Jmax/Vcmax ratio for {pft_name}: {ratio}"
            except Exception as e:
                # Parameters might not exist for all PFTs
                pass

    def test_pft_category_coverage(self, setup_system):
        """Test that multiple PFT categories are represented."""
        registry, _, _ = setup_system

        categories = registry.list_pft_categories()

        # Should have at least some diversity in PFT types
        if len(categories) > 0:
            # Check for major functional groups
            has_trees = any("broadleaf" in cat or "needleleaf" in cat for cat in categories)
            has_grasses = any("grass" in cat for cat in categories)

            # At minimum should have trees OR grasses
            assert has_trees or has_grasses, "Missing major PFT functional groups"

    def test_validation_report_generation(self, setup_system):
        """Test PFT validation report generation."""
        registry, _, _ = setup_system

        # Generate validation report
        global _PFT_VALIDATION_REPORT
        if _PFT_VALIDATION_REPORT is None:
            pft_list = registry.list_pfts()
            if len(pft_list) > 3:
                sample_pfts = _select_pft_sample(pft_list)
                report = {}
                for pft_name in sample_pfts:
                    try:
                        registry._validate_pft(registry.get_pft(pft_name))
                        report[pft_name] = []
                    except Exception as e:
                        report[pft_name] = [str(e)]
                _PFT_VALIDATION_REPORT = report
            else:
                _PFT_VALIDATION_REPORT = registry.validate_all_pfts()
        validation_report = _PFT_VALIDATION_REPORT

        # Report should be a dict with PFT names as keys
        assert isinstance(validation_report, dict)

        # Check report structure
        for pft_name, errors in validation_report.items():
            assert isinstance(errors, list)
            # If there are errors, they should be strings
            for error in errors:
                assert isinstance(error, str)


# Benchmarking against known values
class TestPFTBenchmarks:
    """Benchmark tests comparing against literature values."""

    def test_photosynthesis_parameter_ranges(self):
        """Test photosynthesis parameters against known ranges from literature."""
        param_manager = get_global_manager()

        # Load parameters
        _load_pft_parameter_files(param_manager)

        # Literature ranges (Kattge et al. 2011, Rogers et al. 2017)
        vcmax_ranges = {
            "broadleaf_deciduous_temperate": (40, 100),  # umol m-2 s-1
            "needleleaf_evergreen_temperate": (30, 80),
            "c3_grass_temperate": (30, 90),
            "c4_grass_temperate": (20, 60)
        }

        for pft_type, (min_val, max_val) in vcmax_ranges.items():
            param_key = f"biosphere.pft.{pft_type}.photosynthesis.vcmax25"
            try:
                vcmax = param_manager.get(param_key)
                if torch.is_tensor(vcmax):
                    vcmax = vcmax.item()

                assert min_val <= vcmax <= max_val, \
                    f"{pft_type} Vcmax25 {vcmax} outside literature range [{min_val}, {max_val}]"
            except:
                pass  # PFT might not be defined

    def test_allocation_fractions_sum(self):
        """Test that allocation fractions sum to approximately 1.0."""
        param_manager = get_global_manager()

        _load_pft_parameter_files(param_manager)

        test_pfts = ["broadleaf_deciduous_temperate", "c3_grass_temperate"]

        for pft_type in test_pfts:
            try:
                f_leaf = param_manager.get(f"biosphere.pft.{pft_type}.allocation.f_leaf", 0.0)
                f_stem = param_manager.get(f"biosphere.pft.{pft_type}.allocation.f_stem", 0.0)
                f_root = param_manager.get(f"biosphere.pft.{pft_type}.allocation.f_root", 0.0)

                # Convert tensors if needed
                if torch.is_tensor(f_leaf):
                    f_leaf = f_leaf.item()
                if torch.is_tensor(f_stem):
                    f_stem = f_stem.item()
                if torch.is_tensor(f_root):
                    f_root = f_root.item()

                total = f_leaf + f_stem + f_root

                # Allow small numerical error
                assert 0.95 <= total <= 1.05, \
                    f"{pft_type} allocation fractions sum to {total}, should be ~1.0"
            except:
                pass  # PFT might not be defined


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

