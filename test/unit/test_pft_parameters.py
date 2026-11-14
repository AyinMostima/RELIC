"""
Unit tests for PFT parameter loading from the unified parameter system.

This test file verifies:
1. All PFT parameters exist in YAML database
2. PFTRegistry returns complete typed parameter definitions
3. No hardcoded defaults remain
4. Parameter ranges and units are valid
"""

import pytest
from pathlib import Path

from relic.parameters import get_global_manager
from relic.scientific.biosphere.pft.pft_registry import get_pft_registry
from relic.scientific.biosphere.pft.pft_types import (
    PFTDefinition,
    PhotosynthesisParameters, RespirationParameters,
    AllocationParameters, PhenologyParameters, 
    BVOCParameters, StressParameters
)

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


class TestPFTParameterLoading:
    """Test suite for PFT parameter loading from ParameterManager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.param_manager = get_global_manager()
        self.registry = get_pft_registry()
        
        # Load PFT parameters from per-PFT files
        _load_pft_parameter_files(self.param_manager)
            
        self.pft_types = [
            'broadleaf_deciduous_temperate',
            'needleleaf_evergreen_temperate', 
            'c3_grass_temperate',
            'c4_grass_temperate',
            'shrub_deciduous_temperate'
        ]
    
    def test_pft_registry_typed_lookup(self):
        """Test typed registry lookup for every supported PFT."""
        for pft_name in self.pft_types:
            pft = self.registry.get_pft(pft_name)
            assert isinstance(pft, PFTDefinition)
            assert pft.name == pft_name
    
    def test_photosynthesis_parameters_require_namespace(self):
        """Test that PhotosynthesisParameters requires namespace."""
        # Should raise error without namespace
        with pytest.raises(ValueError, match="PFT namespace required"):
            params = PhotosynthesisParameters(parameter_manager=self.param_manager)
    
    def test_photosynthesis_parameters_loading(self):
        """Test loading photosynthesis parameters from database."""
        for pft_name in self.pft_types:
            namespace = f"biosphere.pft.{pft_name}"
            params = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
            assert params.vcmax25 is not None, f"vcmax25 not loaded for {pft_name}"
            assert params.jmax25 is not None, f"jmax25 not loaded for {pft_name}"
            assert params.vcmax25 > 0, f"vcmax25 must be positive for {pft_name}"
            assert params.jmax25 > 0, f"jmax25 must be positive for {pft_name}"
            # Farquhar et al. (1980): Vcmax in umol CO2 and Jmax in umol electrons.
            for name, units in (
                ("vcmax25", "umol CO2 m-2 s-1"),
                ("jmax25", "umol e- m-2 s-1"),
            ):
                key = f"{namespace}.photosynthesis.{name}"
                assert getattr(params, name) == pytest.approx(
                    self.param_manager.get(key), rel=1e-6, abs=1e-12
                )
                assert self.param_manager.get_parameter(key).units == units
    
    def test_respiration_parameters_loading(self):
        """Test loading respiration parameters from database."""
        for pft_name in self.pft_types:
            namespace = f"biosphere.pft.{pft_name}"
            params = RespirationParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
            assert params.maint_resp_leaf is not None
            assert params.q10_maint_resp is not None
            assert params.maint_resp_leaf > 0
            assert params.q10_maint_resp > 1
    
    def test_allocation_parameters_loading(self):
        """Test loading allocation parameters from database."""
        for pft_name in self.pft_types:
            namespace = f"biosphere.pft.{pft_name}"
            params = AllocationParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
            assert params.f_leaf is not None
            assert params.f_stem is not None
            assert params.f_root is not None
            total = params.f_leaf + params.f_stem + params.f_root
            assert 0.9 <= total <= 1.1, f"Allocation fractions must sum to ~1, got {total}"
    
    def test_phenology_parameters_loading(self):
        """Test loading phenology parameters from database."""
        for pft_name in self.pft_types:
            namespace = f"biosphere.pft.{pft_name}"
            params = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
            assert params.gdd_min is not None
            assert params.specific_leaf_area is not None
            assert params.specific_leaf_area > 0
    
    def test_bvoc_parameters_loading(self):
        """Test loading BVOC parameters from database."""
        for pft_name in self.pft_types:
            namespace = f"biosphere.pft.{pft_name}"
            params = BVOCParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
            assert params.ef_isoprene is not None
            assert params.ef_monoterpene is not None
            assert params.ef_isoprene >= 0
            assert params.ef_monoterpene >= 0
    
    def test_stress_parameters_loading(self):
        """Test loading stress parameters from database."""
        for pft_name in self.pft_types:
            namespace = f"biosphere.pft.{pft_name}"
            params = StressParameters(parameter_manager=self.param_manager, _pft_namespace=namespace)
            assert params.temp_stress_min is not None
            assert params.temp_stress_max is not None
            assert params.temp_stress_min < params.temp_stress_max
    
    def test_pft_registry_matches_parameter_manager(self):
        """Test typed definitions against exact ParameterManager keys."""
        for pft_name in self.pft_types:
            pft = self.registry.get_pft(pft_name)
            namespace = f"biosphere.pft.{pft_name}"
            bindings = {
                f"{namespace}.photosynthesis.vcmax25": pft.photosynthesis.vcmax25,
                f"{namespace}.photosynthesis.jmax25": pft.photosynthesis.jmax25,
                f"{namespace}.allocation.f_leaf": pft.allocation.f_leaf,
                f"{namespace}.stress.water_stress_threshold": pft.stress.water_stress_threshold,
            }
            for key, typed_value in bindings.items():
                assert typed_value == pytest.approx(
                    self.param_manager.get(key), rel=1e-6, abs=1e-12
                )
                spec = self.param_manager.get_parameter(key)
                assert spec.units
                assert spec.bounds is not None
    
    def test_no_hardcoded_defaults(self):
        """Verify that no _use_defaults methods exist."""
        classes_to_check = [
            PhotosynthesisParameters,
            RespirationParameters, 
            AllocationParameters,
            PhenologyParameters,
            BVOCParameters,
            StressParameters
        ]
        
        for cls in classes_to_check:
            # Check that _use_defaults method doesn't exist
            assert not hasattr(cls, '_use_defaults'), \
                f"{cls.__name__} should not have _use_defaults method"
            
            assert cls.__name__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

