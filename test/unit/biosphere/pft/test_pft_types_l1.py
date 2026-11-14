#!/usr/bin/env python
"""
L1 Unit Tests for PFT Types
This module tests individual PFT type parameter loading and validation.
"""

import unittest
import sys
import os
from pathlib import Path
import torch

# Add relic to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from relic.scientific.biosphere.pft.pft_types import (
    PhotosynthesisParameters,
    PhenologyParameters,
    AllocationParameters,
    BVOCParameters,
    StressParameters,
    PFTDefinition
)
from relic.parameters import get_global_manager

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


class TestPhotoosynthesisParams(unittest.TestCase):
    """Test photosynthesis parameter loading from ParameterManager."""
    
    def setUp(self):
        """Initialize parameter manager and load parameters."""
        self.param_manager = get_global_manager()
        # Clear any existing parameters first
        self.param_manager.clear_parameters()
        # Load PFT parameters specifically
        _load_pft_parameter_files(self.param_manager)
        
    def test_c3_photosynthesis_params_loading(self):
        """Test loading C3 photosynthesis parameters."""
        # Create params for broadleaf deciduous (C3)
        params = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.broadleaf_deciduous_temperate')
        
        # Check key C3 parameters
        expected_vcmax = self.param_manager.get(
            "biosphere.pft.broadleaf_deciduous_temperate.photosynthesis.vcmax25"
        )
        expected_jmax = self.param_manager.get(
            "biosphere.pft.broadleaf_deciduous_temperate.photosynthesis.jmax25"
        )
        if isinstance(expected_vcmax, torch.Tensor):
            expected_vcmax = float(expected_vcmax.detach().cpu().item())
        if isinstance(expected_jmax, torch.Tensor):
            expected_jmax = float(expected_jmax.detach().cpu().item())
        self.assertEqual(params.vcmax25, expected_vcmax, "Vcmax25 should match ParameterManager value")
        self.assertEqual(params.jmax25, expected_jmax, "Jmax25 should match ParameterManager value")
        self.assertEqual(params.pathway.value, 'C3', "Pathway should be C3")
        
        # Check temperature response parameters exist
        self.assertIsNotNone(params.vcmax_ea, "Vcmax activation energy should be loaded")
        self.assertIsNotNone(params.jmax_ea, "Jmax activation energy should be loaded")
        
    def test_c4_photosynthesis_params_loading(self):
        """Test loading C4 photosynthesis parameters."""
        # Create params for C4 grass
        params = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.c4_grass_temperate')
        
        # Check C4 specific parameters
        self.assertEqual(params.pathway.value, 'C4', "Pathway should be C4")
        expected_vcmax = self.param_manager.get(
            "biosphere.pft.c4_grass_temperate.photosynthesis.vcmax25"
        )
        if isinstance(expected_vcmax, torch.Tensor):
            expected_vcmax = float(expected_vcmax.detach().cpu().item())
        self.assertEqual(params.vcmax25, expected_vcmax, "Vcmax25 should match ParameterManager value")
        self.assertLess(params.jmax25/params.vcmax25, 2.0, "C4 should have lower Jmax/Vcmax ratio")
        
    def test_stomatal_parameters(self):
        """Test stomatal conductance parameters."""
        params = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.needleleaf_evergreen_temperate')
        
        # Check Ball-Berry parameters
        self.assertIsNotNone(params.g0, "Minimum stomatal conductance should exist")
        self.assertIsNotNone(params.g1, "Stomatal slope parameter should exist")
        self.assertGreater(params.g0, 0, "g0 should be positive")
        self.assertGreater(params.g1, 0, "g1 should be positive")
        
    def test_no_hardcoded_defaults(self):
        """Verify parameters come from ParameterManager, not hardcoded."""
        # Create params with a known PFT
        params = PhotosynthesisParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.c3_grass_temperate')
        
        # These should be loaded from parameter manager
        self.assertIsNotNone(params.vcmax25)
        self.assertIsNotNone(params.jmax25)
        
        # Verify the values match what's in the database
        expected_vcmax = self.param_manager.get('biosphere.pft.c3_grass_temperate.photosynthesis.vcmax25')
        self.assertEqual(params.vcmax25, expected_vcmax, "Value should come from ParameterManager")


class TestPhenologyParameters(unittest.TestCase):
    """Test phenology parameter loading from ParameterManager."""
    
    def setUp(self):
        """Initialize parameter manager and load parameters."""
        self.param_manager = get_global_manager()
        # Clear any existing parameters first
        self.param_manager.clear_parameters()
        # Load PFT parameters specifically
        _load_pft_parameter_files(self.param_manager)
        
    def test_deciduous_phenology_params(self):
        """Test loading deciduous phenology parameters."""
        params = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.broadleaf_deciduous_temperate')
        
        # Check GDD parameters
        self.assertIsNotNone(params.gdd_min, "GDD minimum temperature should exist")
        self.assertIsNotNone(params.gdd_leaf_on, "GDD leaf onset threshold should exist")
        self.assertIsNotNone(params.gdd_leaf_off, "GDD leaf senescence threshold should exist")
        self.assertLess(params.gdd_leaf_on, params.gdd_leaf_off, 
                       "Leaf onset should require fewer GDD than senescence")
        
    def test_evergreen_phenology_params(self):
        """Test loading evergreen phenology parameters."""
        params = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.needleleaf_evergreen_temperate')
        
        # Check leaf longevity
        self.assertIsNotNone(params.leaf_longevity, "Leaf longevity should exist")
        self.assertGreater(params.leaf_longevity, 365, 
                          "Evergreen leaf longevity should be > 1 year")
        
    def test_temperature_thresholds(self):
        """Test temperature growth thresholds."""
        params = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.c3_grass_temperate')
        
        # Check temperature hierarchy
        self.assertLess(params.temp_growth_min, params.temp_growth_opt,
                       "Min temp should be less than optimal")
        self.assertLess(params.temp_growth_opt, params.temp_growth_max,
                       "Optimal temp should be less than maximum")
        
    def test_photoperiod_parameters(self):
        """Test photoperiod response parameters."""
        params = PhenologyParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.shrub_deciduous_temperate')
        
        self.assertIsNotNone(params.photoperiod_critical, 
                           "Critical photoperiod should exist")
        self.assertIsNotNone(params.photoperiod_sensitivity,
                           "Photoperiod sensitivity should exist")
        self.assertGreaterEqual(params.photoperiod_critical, 8.0,
                               "Critical photoperiod should be >= 8 hours")
        self.assertLessEqual(params.photoperiod_critical, 16.0,
                            "Critical photoperiod should be <= 16 hours")


class TestAllocationParameters(unittest.TestCase):
    """Test carbon allocation parameter loading from ParameterManager."""
    
    def setUp(self):
        """Initialize parameter manager and load parameters."""
        self.param_manager = get_global_manager()
        # Clear any existing parameters first
        self.param_manager.clear_parameters()
        # Load PFT parameters specifically
        _load_pft_parameter_files(self.param_manager)
        
    def test_allocation_fractions(self):
        """Test carbon allocation fractions sum appropriately."""
        for pft_name in ['broadleaf_deciduous_temperate', 'c3_grass_temperate']:
            params = AllocationParameters(parameter_manager=self.param_manager, _pft_namespace=f'biosphere.pft.{pft_name}')
            
            # Check individual fractions
            self.assertGreater(params.f_leaf, 0, "Leaf allocation should be positive")
            self.assertGreater(params.f_root, 0, "Root allocation should be positive")
            self.assertGreaterEqual(params.f_stem, 0, "Stem allocation should be >= 0")
            
            # Sum should be reasonable (not necessarily 1.0 due to storage)
            total = params.f_leaf + params.f_stem + params.f_root
            self.assertLessEqual(total, 1.0, "Total allocation should not exceed 1.0")
            self.assertGreater(total, 0.5, "Total allocation should be substantial")
            
    def test_grass_vs_tree_allocation(self):
        """Test differences in allocation between grasses and trees."""
        grass_params = AllocationParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.c3_grass_temperate')
        tree_params = AllocationParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.broadleaf_deciduous_temperate')
        
        # Grasses should have higher root allocation
        self.assertGreater(grass_params.f_root, tree_params.f_root,
                          "Grasses should have higher root allocation than trees")
        
        # Trees should have higher stem allocation  
        self.assertGreater(tree_params.f_stem, grass_params.f_stem,
                          "Trees should have higher stem allocation than grasses")
        
    def test_respiration_coefficients(self):
        """Test growth respiration coefficients."""
        params = AllocationParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.broadleaf_deciduous_temperate')
        
        # Check respiration coefficients are reasonable
        for field in ['growth_resp_leaf', 'growth_resp_stem', 'growth_resp_root']:
            value = getattr(params, field)
            self.assertIsNotNone(value, f"{field} should exist")
            self.assertGreater(value, 1.0, f"{field} should be > 1.0")
            self.assertLess(value, 2.0, f"{field} should be < 2.0")


class TestBVOCParameters(unittest.TestCase):
    """Test BVOC emission parameter loading from ParameterManager."""
    
    def setUp(self):
        """Initialize parameter manager and load parameters."""
        self.param_manager = get_global_manager()
        # Clear any existing parameters first
        self.param_manager.clear_parameters()
        # Load PFT parameters specifically
        _load_pft_parameter_files(self.param_manager)
        
    def test_broadleaf_bvoc_emissions(self):
        """Test BVOC emissions for broadleaf trees."""
        params = BVOCParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.broadleaf_deciduous_temperate')
        
        # Broadleaf trees are typically isoprene emitters
        self.assertIsNotNone(params.ef_isoprene, "Isoprene emission factor should exist")
        self.assertGreater(params.ef_isoprene, 0, "Broadleaf should emit isoprene")
        
    def test_conifer_bvoc_emissions(self):
        """Test BVOC emissions for conifers."""
        params = BVOCParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.needleleaf_evergreen_temperate')
        
        # Conifers are typically monoterpene emitters
        self.assertIsNotNone(params.ef_monoterpene, "Monoterpene emission factor should exist")
        self.assertGreater(params.ef_monoterpene, 0, "Conifers should emit monoterpenes")
        
        # Conifers typically have very low isoprene
        self.assertIsNotNone(params.ef_isoprene)
        self.assertLess(params.ef_isoprene, 1.0, "Conifers should have low isoprene")
        
    def test_bvoc_environmental_dependencies(self):
        """Test BVOC environmental dependency parameters."""
        params = BVOCParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.shrub_deciduous_temperate')
        
        # Check light and temperature dependencies
        self.assertIsNotNone(params.light_dep_monoterpene,
                           "Light dependency for monoterpene should exist")
        self.assertIsNotNone(params.temp_dep_monoterpene,
                           "Temperature dependency for monoterpene should exist")
        
        # Dependencies should be between 0 and 1
        self.assertGreaterEqual(params.light_dep_monoterpene, 0)
        self.assertLessEqual(params.light_dep_monoterpene, 1)


class TestStressParameters(unittest.TestCase):
    """Test stress response parameter loading from ParameterManager."""
    
    def setUp(self):
        """Initialize parameter manager and load parameters."""
        self.param_manager = get_global_manager()
        # Clear any existing parameters first
        self.param_manager.clear_parameters()
        # Load PFT parameters specifically
        _load_pft_parameter_files(self.param_manager)
        
    def test_water_stress_parameters(self):
        """Test water stress threshold parameters."""
        # C4 grasses should be more drought tolerant
        c4_params = StressParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.c4_grass_temperate')
        c3_params = StressParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.c3_grass_temperate')
        
        self.assertIsNotNone(c4_params.water_stress_threshold)
        self.assertIsNotNone(c3_params.water_stress_threshold)
        
        # Water potential is negative, less negative = less stress tolerant
        self.assertLess(c4_params.water_stress_threshold, -1.0,
                       "Water stress threshold should be negative (MPa)")
        
    def test_nitrogen_stress_parameters(self):
        """Test nitrogen stress parameters."""
        params = StressParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.shrub_deciduous_temperate')
        
        self.assertIsNotNone(params.nitrogen_stress_half_sat,
                           "Nitrogen half-saturation constant should exist")
        self.assertGreater(params.nitrogen_stress_half_sat, 0,
                          "Nitrogen half-sat should be positive")
        
    def test_stress_threshold_ranges(self):
        """Test that all stress thresholds are within reasonable ranges."""
        params = StressParameters(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.broadleaf_deciduous_temperate')
        
        # Check light stress
        self.assertIsNotNone(params.light_stress_threshold)
        self.assertGreaterEqual(params.light_stress_threshold, 0)
        
        # Check temperature stress
        self.assertIsNotNone(params.temp_stress_threshold)


class TestPFTDefinitionIntegration(unittest.TestCase):
    """Test integrated PFT parameter loading."""
    
    def setUp(self):
        """Initialize parameter manager and load parameters."""
        self.param_manager = get_global_manager()
        # Clear any existing parameters first
        self.param_manager.clear_parameters()
        # Load PFT parameters specifically
        _load_pft_parameter_files(self.param_manager)
        
    def test_complete_pft_params_loading(self):
        """Test loading complete PFT parameters for all PFT types."""
        pft_names = [
            'broadleaf_deciduous_temperate',
            'needleleaf_evergreen_temperate',
            'c3_grass_temperate',
            'c4_grass_temperate',
            'shrub_deciduous_temperate'
        ]
        
        for pft_name in pft_names:
            with self.subTest(pft_name=pft_name):
                params = PFTDefinition(parameter_manager=self.param_manager, _pft_namespace=f'biosphere.pft.{pft_name}')
                
                # Check all parameter groups are loaded
                self.assertIsNotNone(params.photosynthesis, 
                                   f"Photosynthesis params missing for {pft_name}")
                self.assertIsNotNone(params.phenology,
                                   f"Phenology params missing for {pft_name}")
                self.assertIsNotNone(params.allocation,
                                   f"Allocation params missing for {pft_name}")
                self.assertIsNotNone(params.bvoc,
                                   f"BVOC params missing for {pft_name}")
                self.assertIsNotNone(params.stress,
                                   f"Stress params missing for {pft_name}")
                
    def test_parameter_consistency(self):
        """Test parameter consistency across groups."""
        params = PFTDefinition(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.broadleaf_deciduous_temperate')
        
        # Check temperature consistency
        self.assertLessEqual(
            params.phenology.temp_growth_min,
            params.phenology.temp_growth_opt,
            "Growth min temp should be <= optimal temp"
        )
        
    def test_parameter_source_verification(self):
        """Verify parameters are loaded from parameter manager."""
        # Get a parameter directly from manager
        direct_vcmax = self.param_manager.get(
            'biosphere.pft.broadleaf_deciduous_temperate.photosynthesis.vcmax25'
        )
        
        # Get same parameter through PFT params
        params = PFTDefinition(parameter_manager=self.param_manager, _pft_namespace='biosphere.pft.broadleaf_deciduous_temperate')
        pft_vcmax = params.photosynthesis.vcmax25
        
        # They should match
        self.assertEqual(direct_vcmax, pft_vcmax,
                        "Parameter should come from ParameterManager")


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestPhotoosynthesisParams))
    suite.addTest(unittest.makeSuite(TestPhenologyParameters))
    suite.addTest(unittest.makeSuite(TestAllocationParameters))
    suite.addTest(unittest.makeSuite(TestBVOCParameters))
    suite.addTest(unittest.makeSuite(TestStressParameters))
    suite.addTest(unittest.makeSuite(TestPFTDefinitionIntegration))
    
    return suite


if __name__ == '__main__':
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

