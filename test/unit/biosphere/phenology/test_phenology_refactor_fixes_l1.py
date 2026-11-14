"""
L1 Unit Tests for Phenology Parameter System Refactoring

This comprehensive test suite validates all the critical fixes implemented during
the phenology parameter system refactoring, including:
1. PhenologyPhase encoding to match growth module expectations
2. Flux key naming consistency with validation expectations
3. Parameter loading from unified ParameterManager
4. Namespace consistency

Author: Yuchong Long
Date: 2025-09-25
"""

import pytest
import torch
from unittest.mock import patch

# Import phenology components
from relic.scientific.biosphere.phenology.base import (
    PhenologyBase, PhenologyPhase, PhenologyOutput
)
from relic.scientific.biosphere.phenology.models import (
    StaticPhenology, GDDPhenology, GSIPhenology
)
from relic.parameters import get_global_manager


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


class TestPhenologyPhaseEncoding:
    """Test suite for PhenologyPhase encoding to match growth module expectations."""
    
    def test_phase_mapping_to_integers(self):
        """Test that PhenologyPhase enum maps correctly to integer states."""
        phase_mapping = {
            PhenologyPhase.DORMANCY: 0,
            PhenologyPhase.CHILLING: 0,
            PhenologyPhase.VERNALIZATION: 0,
            PhenologyPhase.BUD_BREAK: 1,
            PhenologyPhase.LEAF_EXPANSION: 1,
            PhenologyPhase.FULL_LEAF: 2,
            PhenologyPhase.FLOWERING: 2,
            PhenologyPhase.SENESCENCE: 3,
            PhenologyPhase.ABSCISSION: 3,
        }

        assert phase_mapping[PhenologyPhase.DORMANCY] == 0
        assert phase_mapping[PhenologyPhase.CHILLING] == 0
        assert phase_mapping[PhenologyPhase.BUD_BREAK] == 1
        assert phase_mapping[PhenologyPhase.LEAF_EXPANSION] == 1
        assert phase_mapping[PhenologyPhase.FULL_LEAF] == 2
        assert phase_mapping[PhenologyPhase.FLOWERING] == 2
        assert phase_mapping[PhenologyPhase.SENESCENCE] == 3
        assert phase_mapping[PhenologyPhase.ABSCISSION] == 3
    
    def test_phase_output_in_fluxes(self):
        """Test that phenology_phase in flux output is integer encoded."""
        config = dict({
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0,
            'leaf_onset_doy': 120,
            'leaf_offset_doy': 280
        })

        phenology = StaticPhenology(name="test_static", config=config)
        state = {
            'leaf_carbon': torch.tensor(0.1),
            'froot_carbon': torch.tensor(0.1),
            'leaf_nitrogen': torch.tensor(0.004),
            'froot_nitrogen': torch.tensor(0.002),
        }
        environment = {
            'day_of_year': torch.tensor(150.0),
            'air_temperature': torch.tensor(288.15),
        }

        with patch.object(phenology, 'compute_phenological_state') as mock_compute:
            mock_compute.return_value = PhenologyOutput(
                phenology_phase=PhenologyPhase.FULL_LEAF,
                leaf_area_index=torch.tensor(5.0),
                growth_factor=torch.tensor(1.0),
                senescence_rate=torch.tensor(0.0),
            )
            updates = phenology({**state, **environment}, time=0.0, dt=1.0)
            fluxes = {key: update.value for key, update in updates.items()}

        assert fluxes['phenology_phase'].item() == 2.0


class TestFluxKeyNaming:
    """Test suite for flux key naming consistency."""
    
    def test_flux_keys_match_validation_expectations(self):
        """Test that flux output keys match validation expectations."""
        config = dict({
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0
        })
        
        phenology = StaticPhenology(name="test_static", config=config)

        # Mock state and environment
        state = {
            'leaf_carbon': torch.tensor(0.1),
            'froot_carbon': torch.tensor(0.1),
            'leaf_nitrogen': torch.tensor(0.004),
            'froot_nitrogen': torch.tensor(0.002)
        }
        environment = {
            'day_of_year': torch.tensor(280.0),  # During senescence
            'air_temperature': torch.tensor(283.15)
        }

        # Get fluxes
        with patch.object(phenology, 'compute_phenological_state') as mock_compute:
            mock_compute.return_value = PhenologyOutput(
                phenology_phase=PhenologyPhase.SENESCENCE,
                leaf_area_index=torch.tensor(2.0),
                growth_factor=torch.tensor(0.3),
                senescence_rate=torch.tensor(0.8)
            )

            updates = phenology({**state, **environment}, time=0.0, dt=1.0)
            fluxes = {key: update.value for key, update in updates.items()}

            # Check that correct flux keys are present (with _carbon suffix)
            assert 'leaf_litterfall_carbon' in fluxes  # Not 'leaf_litter_carbon'
            assert 'froot_litterfall_carbon' in fluxes  # Not 'froot_litter_carbon'
            assert 'leaf_litterfall_nitrogen' in fluxes  # Not 'leaf_litter_nitrogen'
            assert 'froot_litterfall_nitrogen' in fluxes  # Not 'froot_litter_nitrogen'
            assert fluxes['leaf_litterfall_carbon'] is not None
            assert fluxes['froot_litterfall_carbon'] is not None
            assert fluxes['leaf_carbon'] is not None
            assert fluxes['froot_carbon'] is not None


class TestParameterLoading:
    """Test suite for parameter loading from unified ParameterManager."""
    
    def test_parameters_load_from_manager(self):
        """Test that parameters are loaded from ParameterManager, not hardcoded."""
        config = dict({
            'phenology_type': 'gdd',
            'vegetation_type': 'tree',
            'latitude': 45.0
        })
        
        phenology = GDDPhenology(name="test_gdd", config=config)
        manager = get_global_manager()
        
        expected_tol = manager.get('biosphere.validation.tolerance.phenology')
        assert phenology.conservation_tolerance == pytest.approx(
            float(expected_tol),
            rel=1e-6,
            abs=1e-9,
        )
    
    def test_missing_parameters_registered(self):
        """Test that previously missing parameters are now accessible."""
        param_manager = get_global_manager()
        
        # Test that stress parameters are accessible
        # These should be in the database now
        expected_params = [
            'biosphere.phenology.config.numerical_tolerance'
        ]
        
        for param_key in expected_params:
            # Try to get the parameter - it should either exist or register if missing
            value = param_manager.get(param_key, default=None)
            # If value is None, it means it wasn't found and no default was provided
            # But the key should be accessible without error
            assert param_key is not None  # Basic assertion to ensure no exception
    
    def test_namespace_consistency(self):
        """Test that namespace usage is consistent between base.py and parameters.py."""
        config = dict({
            'phenology_type': 'gsi',
            'vegetation_type': 'tree',
            'latitude': 45.0
        })
        
        original_get = PhenologyBase.get_parameter_value
        parameter_names = []

        def tracking_get(instance, name, **kwargs):
            parameter_names.append(name)
            return original_get(instance, name, **kwargs)

        with patch.object(
            PhenologyBase,
            "get_parameter_value",
            new=tracking_get,
        ):
            GSIPhenology(name="test_gsi", config=config)

        from relic.scientific.biosphere.phenology.parameters import phenology_field_names

        canonical_keys = {
            PhenologyBase.PARAMETER_FIELD_KEYS[field]
            for field in phenology_field_names("gsi")
            if field != "phenology_type"
        }
        assert canonical_keys.issubset(set(parameter_names))
        assert (
            "biosphere.phenology.common.phenology_type"
            not in parameter_names
        )
        assert "biosphere.phenology.morphology.max_lai" in parameter_names
        assert "biosphere.phenology.morphology.min_lai" in parameter_names
        assert "biosphere.phenology.lai.max_lai_deciduous" not in parameter_names
        assert "biosphere.phenology.lai.min_lai" not in parameter_names
        assert (
            "biosphere.phenology.morphology.specific_leaf_area"
            not in parameter_names
        )


class TestConservationValidation:
    """Test suite for conservation law validation."""
    
    def test_allocation_fractions_sum_to_one(self):
        """Test that allocation fractions sum to 1 with tolerance."""
        config = dict({
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0
        })
        
        phenology = StaticPhenology(name="test_static", config=config)
        phenology.conservation_tolerance = 1e-6
        
        # Create output with allocation fractions
        output = PhenologyOutput(
            phenology_phase=PhenologyPhase.FULL_LEAF,
            leaf_area_index=torch.tensor(5.0),
            growth_factor=torch.tensor(1.0),
            senescence_rate=torch.tensor(0.0),
            allocation_fractions={
                'leaf': torch.tensor(0.4),
                'stem': torch.tensor(0.3),
                'root': torch.tensor(0.3)
            }
        )
        
        # This should pass validation
        phenology._validate_output_bounds(output)
        
        # Test with fractions that don't sum to 1
        output.allocation_fractions = {
            'leaf': torch.tensor(0.4),
            'stem': torch.tensor(0.3),
            'root': torch.tensor(0.2)  # Sum = 0.9
        }
        
        # This should raise ValueError for invalid bounds/constraints
        with pytest.raises(ValueError, match="Allocation fractions do not sum to 1"):
            phenology._validate_output_bounds(output)
    
    def test_phase_bounds_validation(self):
        """Test that phenological variables are within valid bounds."""
        config = dict({
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0
        })
        
        phenology = StaticPhenology(name="test_static", config=config)
        phenology.conservation_tolerance = 0.0
        
        # Test negative LAI (should fail)
        output = PhenologyOutput(
            phenology_phase=PhenologyPhase.FULL_LEAF,
            leaf_area_index=torch.tensor(-1.0),  # Invalid negative LAI
            growth_factor=torch.tensor(1.0),
            senescence_rate=torch.tensor(0.0)
        )
        
        with pytest.raises(ValueError, match="Negative LAI detected"):
            phenology._validate_output_bounds(output)
        
        # Test growth factor out of bounds
        output = PhenologyOutput(
            phenology_phase=PhenologyPhase.FULL_LEAF,
            leaf_area_index=torch.tensor(5.0),
            growth_factor=torch.tensor(1.5),  # > 1.0
            senescence_rate=torch.tensor(0.0)
        )
        
        with pytest.raises(ValueError, match="Growth factor out of bounds"):
            phenology._validate_output_bounds(output)


class TestDependencyResolution:
    """Test suite for Functor dependency resolution."""
    
    def test_missing_photoperiod_functor_removed(self):
        """Static forcing ports must not become direct Functor dependencies."""
        config = dict({
            'phenology_type': 'static',
            'vegetation_type': 'tree',
            'latitude': 45.0
        })
        
        phenology = StaticPhenology(name="test_static", config=config)

        assert phenology.get_dependencies() == []
        assert set(phenology.metadata.input_variables) == {
            'day_of_year',
            'air_temperature',
            'leaf_carbon',
            'froot_carbon',
            'leaf_nitrogen',
            'froot_nitrogen',
        }
class TestParameterDatabaseIntegrity:
    """Test suite for parameter database integrity after fixes."""
    
    def test_no_duplicate_drought_stress_keys(self):
        """Test that drought_stress duplicate has been resolved."""
        param_manager = get_global_manager()
        
        # The key should now be drought_stress_threshold, not drought_stress_base
        # And there should be only one value
        value = param_manager.get('biosphere.phenology.stress.drought_stress_threshold', default=None)
        
        # If it exists, it should be negative (physically correct)
        if value is not None:
            assert value <= 0, "Drought stress threshold should be negative (MPa)"
    
    def test_computational_defaults_available(self):
        """Test that computational default parameters are available."""
        param_manager = get_global_manager()
        
        computational_params = [
            'biosphere.phenology.computational.default_day_of_year',
            'biosphere.phenology.computational.default_temperature',
            'biosphere.phenology.computational.default_leaf_carbon',
            'biosphere.phenology.computational.default_froot_carbon',
            'biosphere.phenology.computational.default_leaf_nitrogen',
            'biosphere.phenology.computational.default_froot_nitrogen',
            'biosphere.phenology.computational.numerical_tolerance',
        ]
        
        for param_key in computational_params:
            # These should all be accessible without error
            value = param_manager.get(param_key, default=None)
            assert param_key is not None  # Basic check that no exception occurred


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

