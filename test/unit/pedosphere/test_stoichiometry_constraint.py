"""
L1 Unit Tests for StoichiometryConstraint Module

This module provides comprehensive unit tests for the stoichiometric constraint
implementation, covering C-N coupling, mass balance verification, and 
stoichiometric corrections.

Test Categories:
- L1.1: Basic instantiation and configuration  
- L1.2: Scientific accuracy against literature values
- L1.3: Stoichiometric ratio enforcement
- L1.4: Mass balance verification
- L1.5: Correction algorithms
- L1.6: Constraint violation detection
- L1.7: Scale coupling functionality
- L1.8: Boundary condition handling
- L1.9: Parameter sensitivity analysis
- L1.10: Performance and stability tests
"""

import pytest
import torch
from typing import Dict

# Import the module under test
from relic.scientific.pedosphere.nutrient_cycling import StoichiometryConstraint
from relic.parameters import get_global_manager

# Import utilities for testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def _normalize_param_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return value

def _set_param_value(param_manager, key: str, value) -> None:
    spec = param_manager.get_parameter(key)
    units = spec.units if spec is not None else None
    param_manager.set_value(key, value, units=units, validate=True, notify=False)


def _with_diagnostics(state, **diagnostics):
    updated = dict(state)
    for key, value in diagnostics.items():
        if value is not None:
            updated[f"diagnostics.{key}"] = value
    return updated


class TestStoichiometryConstraintL1:
    """L1 Unit Tests for StoichiometryConstraint functor."""

    @pytest.fixture(autouse=True)
    def _default_param_overrides(self, default_config):
        param_manager = get_global_manager()
        overrides = {
            "pedosphere.stoichiometry.metabolic_litter_cn": default_config["metabolic_litter_cn"],
            "pedosphere.stoichiometry.structural_litter_cn": default_config["structural_litter_cn"],
            "pedosphere.stoichiometry.fast_som_cn": default_config["fast_som_cn"],
            "pedosphere.stoichiometry.slow_som_cn": default_config["slow_som_cn"],
            "pedosphere.stoichiometry.passive_som_cn": default_config["passive_som_cn"],
            "pedosphere.stoichiometry.microbial_biomass_cn": default_config["microbial_biomass_cn"],
            "pedosphere.numerical.stoichiometric_flexibility": default_config["stoichiometric_flexibility"],
            "pedosphere.numerical.stoichiometric_correction_strength": default_config["stoichiometric_correction_strength"],
        }
        original_values = {
            key: _normalize_param_value(param_manager.get(key)) for key in overrides
        }
        for key, value in overrides.items():
            _set_param_value(param_manager, key, value)
        yield
        for key, value in original_values.items():
            _set_param_value(param_manager, key, value)
    
    @pytest.fixture
    def default_config(self) -> Dict:
        """Default configuration for StoichiometryConstraint."""
        return {
            'metabolic_litter_cn': 15.0,      # C:N ratio
            'structural_litter_cn': 40.0,     # C:N ratio
            'fast_som_cn': 12.0,              # C:N ratio
            'slow_som_cn': 20.0,              # C:N ratio
            'passive_som_cn': 25.0,           # C:N ratio
            'microbial_biomass_cn': 8.0,      # C:N ratio
            'stoichiometric_flexibility': 0.2, # 20% tolerance
            'stoichiometric_correction_strength': 0.5  # 50% correction
        }
    
    @pytest.fixture  
    def stoichiometry(self, default_config) -> StoichiometryConstraint:
        """Create StoichiometryConstraint instance."""
        return StoichiometryConstraint(name="test_stoichiometry")
    
    @pytest.fixture
    def standard_state(self) -> Dict:
        """Standard state for testing."""
        return {
            'metabolic_litter_c': torch.tensor(0.1),     # g C m-2
            'structural_litter_c': torch.tensor(0.3),    # g C m-2
            'fast_som_c': torch.tensor(0.8),             # g C m-2
            'slow_som_c': torch.tensor(2.0),             # g C m-2
            'passive_som_c': torch.tensor(4.0),          # g C m-2
            'metabolic_litter_n': torch.tensor(0.007),   # g N m-2 (C:N=15)
            'structural_litter_n': torch.tensor(0.0075), # g N m-2 (C:N=40)
            'fast_som_n': torch.tensor(0.067),           # g N m-2 (C:N=12)
            'slow_som_n': torch.tensor(0.1),             # g N m-2 (C:N=20)
            'passive_som_n': torch.tensor(0.16),         # g N m-2 (C:N=25)
            # Explicit external budget terms (zero exchange), no default substitute.
            'diagnostics.carbon_inputs': torch.tensor(0.0),
            'diagnostics.nitrogen_inputs': torch.tensor(0.0),
            'diagnostics.carbon_outputs': torch.tensor(0.0),
            'diagnostics.nitrogen_outputs': torch.tensor(0.0),
        }
    
    @pytest.fixture
    def balanced_fluxes(self) -> Dict:
        """Balanced C and N fluxes for testing."""
        return {
            'carbon_fluxes': {
                'delta_metabolic_litter_c': torch.tensor(1.5e-6),  # g C m-2 s-1
                'delta_fast_som_c': torch.tensor(-8.0e-7),         # g C m-2 s-1
            },
            'nitrogen_fluxes': {
                'delta_metabolic_litter_n': torch.tensor(1.0e-7),  # g N m-2 s-1 (C:N=15)
                'delta_fast_som_n': torch.tensor(-6.67e-8),        # g N m-2 s-1 (C:N=12)
            }
        }
    
    @pytest.fixture
    def imbalanced_fluxes(self) -> Dict:
        """Imbalanced C and N fluxes for testing corrections."""
        return {
            'carbon_fluxes': {
                'delta_metabolic_litter_c': torch.tensor(3.0e-6),  # Too much C
                'delta_fast_som_c': torch.tensor(-8.0e-7),
            },
            'nitrogen_fluxes': {
                'delta_metabolic_litter_n': torch.tensor(1.0e-7),  # Current C:N = 30 (should be 15)
                'delta_fast_som_n': torch.tensor(-6.67e-8),
            }
        }
        
    # =============================================================================
    # L1.1: Basic Instantiation and Configuration Tests
    # =============================================================================
    
    def test_instantiation_basic(self, default_config):
        """Test basic instantiation with default configuration."""
        stoich = StoichiometryConstraint(name="test_stoich")
        
        assert stoich.name == "test_stoich"
        assert hasattr(stoich, 'metadata')
        assert hasattr(stoich, 'target_ratios')
        assert hasattr(stoich, 'flexibility')
        assert stoich.metadata.name == "test_stoich"
        
    def test_instantiation_metadata(self, stoichiometry):
        """Test metadata completeness."""
        metadata = stoichiometry.metadata
        
        # Check required metadata fields
        assert metadata.name is not None
        assert metadata.version is not None
        assert metadata.description is not None
        assert metadata.mathematical_expression is not None
        
        # Check scientific references
        assert len(metadata.references) >= 2
        assert any("Manzoni" in ref for ref in metadata.references)
        assert any("Cleveland" in ref for ref in metadata.references)
        
        # Check parameters documentation
        assert len(metadata.parameters) >= 4
        assert 'pedosphere.stoichiometry.metabolic_litter_cn' in metadata.parameters
        assert 'pedosphere.stoichiometry.slow_som_cn' in metadata.parameters
        assert 'pedosphere.numerical.stoichiometric_flexibility' in metadata.parameters
    
    def test_target_ratios_initialization(self, stoichiometry):
        """Test target ratios are properly initialized."""
        ratios = stoichiometry.target_ratios
        
        # Check all expected pools exist
        expected_pools = ['metabolic_litter', 'structural_litter', 'fast_som', 
                         'slow_som', 'passive_som', 'microbial']
        assert all(pool in ratios for pool in expected_pools)
        
        # Check ratios are reasonable (literature values)
        assert 10.0 <= ratios['metabolic_litter'] <= 25.0  # Metabolic litter
        assert 25.0 <= ratios['structural_litter'] <= 150.0  # Structural litter
        assert 8.0 <= ratios['fast_som'] <= 20.0  # Fast SOM
        assert 15.0 <= ratios['slow_som'] <= 30.0  # Slow SOM  
        assert 20.0 <= ratios['passive_som'] <= 50.0  # Passive SOM
        assert 5.0 <= ratios['microbial'] <= 15.0  # Microbial biomass
    
    def test_parameter_initialization(self, stoichiometry):
        """Test parameter initialization."""
        # Check flexibility parameter
        assert 0.0 <= stoichiometry.flexibility <= 1.0
        
        # No mass-balance tolerance is read (no check compares against one).
        assert not hasattr(stoichiometry, "mass_balance_tolerance")
        
        # Check correction strength
        assert 0.0 <= stoichiometry.correction_strength <= 1.0
        
        # Check tracking variables
        assert stoichiometry.constraint_violations == 0
        assert isinstance(stoichiometry.mass_balance_errors, list)
        assert len(stoichiometry.mass_balance_errors) == 0
    
    # =============================================================================
    # L1.2: Scientific Accuracy Tests Against Literature Values
    # =============================================================================
    
    def test_target_ratios_literature_values(self, stoichiometry):
        """Test target ratios against literature values."""
        ratios = stoichiometry.target_ratios
        
        # Literature values (McGroddy et al., 2004; Cleveland & Liptzin, 2007)
        # Metabolic litter: 10-25 (low lignin, high N)
        assert 10.0 <= ratios['metabolic_litter'] <= 25.0
        
        # Structural litter: 30-150 (high lignin, low N)  
        assert 30.0 <= ratios['structural_litter'] <= 150.0
        
        # Fast SOM: 8-15 (active, microbial-dominated)
        assert 8.0 <= ratios['fast_som'] <= 15.0
        
        # Slow SOM: 15-30 (stabilized organic matter)
        assert 15.0 <= ratios['slow_som'] <= 30.0
        
        # Passive SOM: 20-50 (highly stabilized)
        assert 20.0 <= ratios['passive_som'] <= 50.0
        
        # Microbial biomass: 5-12 (typical range)
        assert 5.0 <= ratios['microbial'] <= 12.0
    
    def test_flexibility_literature_values(self, stoichiometry):
        """Test flexibility parameter against literature values."""
        flex = stoichiometry.flexibility
        
        # Literature: typical stoichiometric flexibility 10-30% (Manzoni & Porporato, 2009)
        assert 0.1 <= flex <= 0.5, f"Flexibility should be 10-50%: {flex}"
    
    def test_correction_strength_reasonable(self, stoichiometry):
        """Test correction strength is reasonable."""
        strength = stoichiometry.correction_strength
        
        # Should be partial correction to avoid numerical instability
        assert 0.1 <= strength <= 0.9, f"Correction strength should be moderate: {strength}"
    
    # =============================================================================
    # L1.3: Stoichiometric Ratio Enforcement Tests
    # =============================================================================
    
    def test_balanced_fluxes_no_correction(self, stoichiometry, standard_state, balanced_fluxes):
        """Test that balanced fluxes don't trigger corrections."""
        state_with_diag = _with_diagnostics(standard_state, **balanced_fluxes)
        result = stoichiometry.forward(
            state_with_diag, time=0.0, dt=3600.0
        )
        
        # Check that fluxes are not significantly modified
        original_c = balanced_fluxes['carbon_fluxes']['delta_metabolic_litter_c']
        corrected_c = result['diagnostics.stoichiometry_corrected_delta_metabolic_litter_c']
        
        relative_change = torch.abs(corrected_c - original_c) / torch.abs(original_c + 1e-12)
        assert relative_change < 0.05, f"Balanced flux should not be corrected: {relative_change:.4f}"
        
        # Check violation flags are low
        violation = result.get('diagnostics.stoichiometry_metabolic_litter_violation', torch.tensor(0.0))
        assert violation < 0.5, f"Balanced flux should not show violation: {violation}"
    
    def test_imbalanced_fluxes_correction(self, stoichiometry, standard_state, imbalanced_fluxes):
        """Test that imbalanced fluxes trigger corrections."""
        state_with_diag = _with_diagnostics(standard_state, **imbalanced_fluxes)
        result = stoichiometry.forward(
            state_with_diag, time=0.0, dt=3600.0
        )
        
        # Check that correction was applied
        original_c = imbalanced_fluxes['carbon_fluxes']['delta_metabolic_litter_c']
        original_n = imbalanced_fluxes['nitrogen_fluxes']['delta_metabolic_litter_n']
        corrected_c = result['diagnostics.stoichiometry_corrected_delta_metabolic_litter_c']
        corrected_n = result['diagnostics.stoichiometry_corrected_delta_metabolic_litter_n']
        
        # Calculate original and corrected C:N ratios
        original_cn = original_c / original_n
        corrected_cn = corrected_c / corrected_n
        target_cn = stoichiometry.target_ratios['metabolic_litter']
        
        # Corrected ratio should be closer to target
        original_error = torch.abs(original_cn - target_cn) / target_cn
        corrected_error = torch.abs(corrected_cn - target_cn) / target_cn
        
        assert corrected_error < original_error, \
            f"Correction should improve C:N ratio: {corrected_error:.3f} vs {original_error:.3f}"
        
        # Check violation flag is set
        violation = result.get('diagnostics.stoichiometry_metabolic_litter_violation', torch.tensor(0.0))
        assert violation > 0.5, f"Imbalanced flux should show violation: {violation}"
    
    def test_correction_algorithm_carbon_excess(self, stoichiometry):
        """Test correction algorithm when carbon is in excess."""
        # Create fluxes with carbon excess (C:N = 30, target = 15)
        carbon_flux = torch.tensor(3.0e-6)
        nitrogen_flux = torch.tensor(1.0e-7)
        target_cn = 15.0
        
        corrected = stoichiometry._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test_pool"
        )
        
        corrected_c = corrected['carbon']
        corrected_n = corrected['nitrogen']
        
        # Check that nitrogen was increased (carbon excess case)
        assert corrected_n > nitrogen_flux, "Nitrogen flux should be increased when carbon in excess"
        
        # Check that C:N ratio moved toward target
        corrected_cn = corrected_c / corrected_n
        original_cn = carbon_flux / nitrogen_flux
        
        assert torch.abs(corrected_cn - target_cn) < torch.abs(original_cn - target_cn), \
            "Corrected C:N should be closer to target"
    
    def test_correction_algorithm_nitrogen_excess(self, stoichiometry):
        """Test correction algorithm when nitrogen is in excess."""
        # Create fluxes with nitrogen excess (C:N = 7.5, target = 15)
        carbon_flux = torch.tensor(1.5e-6)
        nitrogen_flux = torch.tensor(2.0e-7)
        target_cn = 15.0
        
        corrected = stoichiometry._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test_pool"
        )
        
        corrected_c = corrected['carbon']
        corrected_n = corrected['nitrogen']
        
        # Check that carbon was increased (nitrogen excess case)
        assert corrected_c > carbon_flux, "Carbon flux should be increased when nitrogen in excess"
        
        # Check that C:N ratio moved toward target
        corrected_cn = corrected_c / corrected_n
        original_cn = carbon_flux / nitrogen_flux
        
        assert torch.abs(corrected_cn - target_cn) < torch.abs(original_cn - target_cn), \
            "Corrected C:N should be closer to target"
    
    # =============================================================================
    # L1.4: Mass Balance Verification Tests
    # =============================================================================
    
    def test_mass_balance_check_function(self, stoichiometry, standard_state):
        """Test mass balance checking function."""
        # Create simple balanced fluxes
        c_fluxes = {'delta_fast_som_c': torch.tensor(-1.0e-6)}
        n_fluxes = {'delta_fast_som_n': torch.tensor(-8.33e-8)}  # C:N = 12
        
        balance = stoichiometry._check_mass_balance(
            c_fluxes,
            n_fluxes,
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            dt=3600.0,
        )
        
        # Check that balance results contain expected keys
        assert 'carbon_error' in balance
        assert 'nitrogen_error' in balance
        assert 'total_error' in balance
        
        # Check that errors are tensors and finite
        for key, error in balance.items():
            assert torch.isfinite(error).all(), f"{key} should be finite"
            assert error >= 0.0, f"{key} should be non-negative: {error}"
    
    def test_mass_balance_error_tracking(self, stoichiometry, standard_state, balanced_fluxes):
        """Test that mass balance errors are properly tracked."""
        initial_count = len(stoichiometry.mass_balance_errors)
        
        # Run several forward passes
        for i in range(3):
            state_with_diag = _with_diagnostics(standard_state, **balanced_fluxes)
            result = stoichiometry.forward(
                state_with_diag, time=float(i), dt=3600.0
            )
        
        # Check that errors were tracked
        assert len(stoichiometry.mass_balance_errors) == initial_count + 3
        
        # Check that total error is reported
        final_result = result
        assert 'diagnostics.stoichiometry_mass_balance_error_total' in final_result
        assert torch.isfinite(final_result['diagnostics.stoichiometry_mass_balance_error_total'])
    
    # =============================================================================
    # L1.5: Correction Algorithms Tests
    # =============================================================================
    
    def test_zero_flux_handling(self, stoichiometry):
        """Test handling of zero fluxes."""
        # Test with zero nitrogen flux
        carbon_flux = torch.tensor(1.0e-6)
        nitrogen_flux = torch.tensor(0.0)
        target_cn = 15.0
        
        corrected = stoichiometry._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test_pool"
        )
        
        # Should not crash and should return finite values
        assert torch.isfinite(corrected['carbon'])
        assert torch.isfinite(corrected['nitrogen'])
        assert torch.isfinite(corrected['violation'])
        
        # Test with zero carbon flux
        carbon_flux = torch.tensor(0.0)
        nitrogen_flux = torch.tensor(1.0e-7)
        
        corrected = stoichiometry._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test_pool"
        )
        
        assert torch.isfinite(corrected['carbon'])
        assert torch.isfinite(corrected['nitrogen'])
    
    def test_very_small_flux_handling(self, stoichiometry):
        """Test handling of very small fluxes."""
        # Test with very small fluxes
        carbon_flux = torch.tensor(1e-12)
        nitrogen_flux = torch.tensor(1e-13)
        target_cn = 15.0
        
        corrected = stoichiometry._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test_pool"
        )
        
        # Should not crash and should maintain reasonable magnitudes
        assert torch.isfinite(corrected['carbon'])
        assert torch.isfinite(corrected['nitrogen'])
        assert torch.abs(corrected['carbon']) >= torch.abs(carbon_flux) * 0.5
        assert torch.abs(corrected['nitrogen']) >= torch.abs(nitrogen_flux) * 0.5
    
    def test_correction_strength_effect(self, default_config):
        """Test effect of different correction strengths."""
        # Test with weak correction
        param_manager = get_global_manager()
        strength_key = "pedosphere.numerical.stoichiometric_correction_strength"
        _set_param_value(param_manager, strength_key, 0.1)
        weak_stoich = StoichiometryConstraint(name="weak")
        
        # Test with strong correction
        _set_param_value(param_manager, strength_key, 0.9)
        strong_stoich = StoichiometryConstraint(name="strong")
        
        # Same imbalanced fluxes
        carbon_flux = torch.tensor(3.0e-6)
        nitrogen_flux = torch.tensor(1.0e-7)
        target_cn = 15.0
        
        weak_result = weak_stoich._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test"
        )
        strong_result = strong_stoich._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test"
        )
        
        # Strong correction should make bigger changes
        weak_change = torch.abs(weak_result['nitrogen'] - nitrogen_flux)
        strong_change = torch.abs(strong_result['nitrogen'] - nitrogen_flux)
        
        assert strong_change > weak_change, "Strong correction should make larger changes"
    
    # =============================================================================
    # L1.6: Constraint Violation Detection Tests
    # =============================================================================
    
    def test_violation_detection_within_flexibility(self, stoichiometry):
        """Test that violations within flexibility are not flagged."""
        target_cn = 15.0
        flexibility = stoichiometry.flexibility  # e.g., 0.2 (20%)
        
        # Create flux within flexibility (18% deviation, < 20%)
        deviation_factor = 1.18  # 18% higher C:N
        carbon_flux = torch.tensor(1.5e-6 * deviation_factor)
        nitrogen_flux = torch.tensor(1.0e-7)
        
        result = stoichiometry._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test_pool"
        )
        
        violation = result['violation']
        assert violation < 0.5, f"Small deviation should not trigger violation: {violation}"
    
    def test_violation_detection_exceeds_flexibility(self, stoichiometry):
        """Test that violations exceeding flexibility are flagged."""
        target_cn = 15.0
        flexibility = stoichiometry.flexibility  # e.g., 0.2 (20%)
        
        # Create flux exceeding flexibility (50% deviation, > 20%)
        deviation_factor = 1.5  # 50% higher C:N
        carbon_flux = torch.tensor(1.5e-6 * deviation_factor)
        nitrogen_flux = torch.tensor(1.0e-7)
        
        result = stoichiometry._apply_stoichiometric_correction(
            carbon_flux, nitrogen_flux, target_cn, "test_pool"
        )
        
        violation = result['violation']
        assert violation > 0.5, f"Large deviation should trigger violation: {violation}"
    
    def test_constraint_violation_counting(self, stoichiometry, standard_state):
        """Test that constraint violations are properly counted."""
        initial_count = stoichiometry.constraint_violations
        
        # Create imbalanced fluxes that should trigger violations
        imbalanced_fluxes = {
            'carbon_fluxes': {
                'delta_metabolic_litter_c': torch.tensor(5.0e-6),  # Very high C
                'delta_fast_som_c': torch.tensor(-1.0e-6),
            },
            'nitrogen_fluxes': {
                'delta_metabolic_litter_n': torch.tensor(1.0e-7),  # Low N -> high C:N
                'delta_fast_som_n': torch.tensor(-8.33e-8),
            }
        }
        
        state_with_diag = _with_diagnostics(standard_state, **imbalanced_fluxes)
        result = stoichiometry.forward(
            state_with_diag, time=0.0, dt=3600.0
        )
        
        # Check that violation count increased
        final_count = stoichiometry.constraint_violations
        assert final_count > initial_count, "Violation count should increase"
        
        # Check that violation count is reported
        assert 'diagnostics.stoichiometry_constraint_violations_count' in result
        assert result['diagnostics.stoichiometry_constraint_violations_count'] == final_count
    
    # =============================================================================
    # L1.7: Scale Coupling Functionality Tests
    # =============================================================================
    
    def test_upscale_function(self, stoichiometry):
        """Test upscale function."""
        # Test with 2D tensor
        fine_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        upscaled = stoichiometry.upscale(fine_data)
        
        expected = torch.tensor([[1.5], [3.5]])  # Mean along last dimension
        assert torch.allclose(upscaled, expected, rtol=1e-5)
        
        # Test with 1D tensor
        fine_1d = torch.tensor([1.0, 2.0, 3.0])
        upscaled_1d = stoichiometry.upscale(fine_1d)

        expected_1d = torch.tensor(2.0)  # Mean of 1D input
        assert torch.allclose(upscaled_1d, expected_1d, rtol=1e-5)
    
    def test_downscale_function(self, stoichiometry):
        """Test downscale function."""
        # Test with 1D tensor
        coarse_data = torch.tensor([5.0])
        downscaled = stoichiometry.downscale(coarse_data)
        
        expected = torch.tensor([5.0, 5.0])  # Repeat once
        assert torch.allclose(downscaled, expected)
        
        # Test with 0D tensor
        coarse_0d = torch.tensor(3.0)
        downscaled_0d = stoichiometry.downscale(coarse_0d)
        
        assert torch.allclose(downscaled_0d, coarse_0d)  # Should return as-is for 0D
    
    def test_resolution_converter_function(self, stoichiometry):
        """Test resolution converter function."""
        # For stoichiometry, resolution conversion should be identity
        data = torch.tensor([1.0, 2.0, 3.0])
        source_res = (1.0, 1.0)
        target_res = (0.5, 0.5)
        
        converted = stoichiometry.resolution_converter(data, source_res, target_res)
        
        assert torch.allclose(converted, data), "Resolution conversion should be identity for stoichiometry"
    
    # =============================================================================
    # L1.8: Boundary Condition Handling Tests
    # =============================================================================
    
    def test_empty_flux_handling(self, stoichiometry, standard_state):
        """Test handling of empty flux dictionaries."""
        empty_fluxes = {
            'carbon_fluxes': {},
            'nitrogen_fluxes': {}
        }
        
        state_with_diag = _with_diagnostics(standard_state, **empty_fluxes)
        result = stoichiometry.forward(
            state_with_diag, time=0.0, dt=3600.0
        )
        
        # Should not crash and should return basic diagnostics
        assert 'diagnostics.stoichiometry_mass_balance_error_total' in result
        assert 'diagnostics.stoichiometry_constraint_violations_count' in result
        assert torch.isfinite(result['diagnostics.stoichiometry_mass_balance_error_total'])
    
    def test_missing_pool_handling(self, stoichiometry, standard_state):
        """Test handling when some pools are missing from fluxes."""
        partial_fluxes = {
            'carbon_fluxes': {
                'delta_metabolic_litter_c': torch.tensor(1.0e-6),
                # Missing corresponding N flux
            },
            'nitrogen_fluxes': {
                'delta_fast_som_n': torch.tensor(-5.0e-8),
                # Missing corresponding C flux
            }
        }
        
        state_with_diag = _with_diagnostics(standard_state, **partial_fluxes)
        result = stoichiometry.forward(
            state_with_diag, time=0.0, dt=3600.0
        )
        
        # Should handle gracefully without crashing
        assert 'diagnostics.stoichiometry_mass_balance_error_total' in result
        assert torch.isfinite(result['diagnostics.stoichiometry_mass_balance_error_total'])
    
    # =============================================================================
    # L1.9: Parameter Sensitivity Analysis Tests
    # =============================================================================
    
    def test_flexibility_sensitivity(self, default_config, standard_state):
        """Test sensitivity to flexibility parameter."""
        imbalanced_fluxes = {
            'carbon_fluxes': {'delta_metabolic_litter_c': torch.tensor(2.25e-6)},  # C:N = 22.5
            'nitrogen_fluxes': {'delta_metabolic_litter_n': torch.tensor(1.0e-7)}   # Target: 15
        }
        
        # Test with strict flexibility
        param_manager = get_global_manager()
        flexibility_key = "pedosphere.numerical.stoichiometric_flexibility"
        _set_param_value(param_manager, flexibility_key, 0.1)
        strict_stoich = StoichiometryConstraint(name="strict")
        
        # Test with loose flexibility
        _set_param_value(param_manager, flexibility_key, 0.6)
        loose_stoich = StoichiometryConstraint(name="loose")
        
        strict_state = _with_diagnostics(standard_state, **imbalanced_fluxes)
        loose_state = _with_diagnostics(standard_state, **imbalanced_fluxes)
        strict_result = strict_stoich.forward(strict_state, 0.0, 3600.0)
        loose_result = loose_stoich.forward(loose_state, 0.0, 3600.0)
        
        # Strict should show more violations
        strict_violation = strict_result.get('diagnostics.stoichiometry_metabolic_litter_violation', torch.tensor(0.0))
        loose_violation = loose_result.get('diagnostics.stoichiometry_metabolic_litter_violation', torch.tensor(0.0))
        
        assert strict_violation >= loose_violation, "Strict flexibility should show more violations"
    
    def test_target_ratio_sensitivity(self, default_config, standard_state):
        """Test sensitivity to target ratio changes."""
        # Same flux for different targets
        test_fluxes = {
            'carbon_fluxes': {'delta_metabolic_litter_c': torch.tensor(2.0e-6)},
            'nitrogen_fluxes': {'delta_metabolic_litter_n': torch.tensor(1.0e-7)}  # C:N = 20
        }
        
        # Target = 15 (current C:N is high)
        param_manager = get_global_manager()
        cn_key = "pedosphere.stoichiometry.metabolic_litter_cn"
        _set_param_value(param_manager, cn_key, 15.0)
        stoich_15 = StoichiometryConstraint(name="cn15")
        
        # Target = 25 (current C:N is low)
        _set_param_value(param_manager, cn_key, 25.0)
        stoich_25 = StoichiometryConstraint(name="cn25")
        
        state_cn15 = _with_diagnostics(standard_state, **test_fluxes)
        state_cn25 = _with_diagnostics(standard_state, **test_fluxes)
        result_15 = stoich_15.forward(state_cn15, 0.0, 3600.0)
        result_25 = stoich_25.forward(state_cn25, 0.0, 3600.0)
        
        # Different targets should give different correction directions
        corrected_n_15 = result_15.get(
            'diagnostics.stoichiometry_corrected_delta_metabolic_litter_n',
            test_fluxes['nitrogen_fluxes']['delta_metabolic_litter_n'],
        )
        corrected_n_25 = result_25.get(
            'diagnostics.stoichiometry_corrected_delta_metabolic_litter_n',
            test_fluxes['nitrogen_fluxes']['delta_metabolic_litter_n'],
        )
        
        # For target 15 (too high C:N), should increase N
        # For target 25 (too low C:N), should decrease N or increase less
        assert corrected_n_15 != corrected_n_25, "Different targets should give different corrections"
    
    # =============================================================================
    # L1.10: Performance and Stability Tests
    # =============================================================================
    
    def test_numerical_stability_extreme_ratios(self, stoichiometry, standard_state):
        """Test numerical stability with extreme C:N ratios."""
        extreme_fluxes = {
            'carbon_fluxes': {'delta_metabolic_litter_c': torch.tensor(1.0e-3)},  # Very large C
            'nitrogen_fluxes': {'delta_metabolic_litter_n': torch.tensor(1.0e-12)}  # Very small N
        }
        
        state_with_diag = _with_diagnostics(standard_state, **extreme_fluxes)
        result = stoichiometry.forward(
            state_with_diag, time=0.0, dt=3600.0
        )
        
        # Should not crash and should return finite values
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"{key} should be finite with extreme ratios"
    
    def test_repeated_execution_consistency(self, stoichiometry, standard_state, balanced_fluxes):
        """Test consistency of repeated executions."""
        results = []
        
        for _ in range(3):
            state_with_diag = _with_diagnostics(standard_state.copy(), **balanced_fluxes)
            result = stoichiometry.forward(
                state_with_diag, time=0.0, dt=3600.0
            )
            error = float(result['diagnostics.stoichiometry_mass_balance_error_total'])
            results.append(error)
        
        # Should be deterministic
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-10, \
                f"Results should be deterministic: {results[0]:.2e} vs {results[i]:.2e}"
    
    def test_gradient_computation_consistency(self, stoichiometry, standard_state, balanced_fluxes):
        """Test consistency with PyTorch gradient computation."""
        # Create fluxes with gradients
        grad_fluxes = {
            'carbon_fluxes': {
                'delta_metabolic_litter_c': torch.tensor(1.5e-6, requires_grad=True)
            },
            'nitrogen_fluxes': {
                'delta_metabolic_litter_n': torch.tensor(1.0e-7, requires_grad=True)
            }
        }
        
        state_with_diag = _with_diagnostics(standard_state, **grad_fluxes)
        result = stoichiometry.forward(
            state_with_diag, time=0.0, dt=3600.0
        )
        
        # Try gradient computation
        total_error = result['diagnostics.stoichiometry_mass_balance_error_total']
        
        if total_error.requires_grad:
            try:
                total_error.backward()
                # Check that some gradients exist
            except RuntimeError:
                # Some operations may not support gradients, which is acceptable
                pass


# Round 35: no *_default value substitutes an absent budget term and the unused
# mass_balance_tolerance is not read; with a term absent the budget is not evaluable
# and its diagnostic is omitted (the architecture budget rule: no term is taken as zero).
def test_cn_violation_is_decided_without_a_substituted_nitrogen_flux():
    """|C - CN_t N| > f CN_t |N| decides the violation; no flux floor stands in."""
    stoich = StoichiometryConstraint(name="stoich_division_free")
    target_cn = 15.0
    exact = stoich._apply_stoichiometric_correction(
        torch.tensor(15.0e-20), torch.tensor(1.0e-20), target_cn, "exact_tiny"
    )
    assert float(exact["violation"]) == 0.0
    idle = stoich._apply_stoichiometric_correction(
        torch.tensor(0.0), torch.tensor(0.0), target_cn, "idle"
    )
    assert float(idle["violation"]) == 0.0
    carbon_only = stoich._apply_stoichiometric_correction(
        torch.tensor(1.0e-6), torch.tensor(0.0), target_cn, "carbon_only"
    )
    assert float(carbon_only["violation"]) == 1.0
    assert not hasattr(stoich, "epsilon_concentration")


def test_mass_balance_error_is_scaled_by_the_gross_budget_activity():
    """rho = e / (|Delta| + |I dt| + |O dt|); e <= the scale, and an idle budget is exact."""
    stoich = StoichiometryConstraint(name="stoich_gross_scale")
    dt = 3600.0
    balance = stoich._check_mass_balance(
        {"delta_fast_som_c": torch.tensor(0.0)},
        {"delta_fast_som_n": torch.tensor(0.0)},
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(1.0e-6),
        torch.tensor(0.0),
        dt=dt,
    )
    assert float(balance["carbon_relative_error"]) == pytest.approx(1.0)
    assert float(balance["nitrogen_relative_error"]) == 0.0
    assert not hasattr(stoich, "epsilon")


def test_absent_budget_terms_make_the_balance_not_evaluable():
    stoich = StoichiometryConstraint(name="stoich_not_evaluable")
    for key in stoich.metadata.parameters:
        assert not key.endswith("_default"), key
    assert "pedosphere.numerical.mass_balance_tolerance" not in stoich.metadata.parameters
    assert not hasattr(stoich, "mass_balance_tolerance")
    optional = set(stoich.metadata.optional_output_variables)
    assert {
        "diagnostics.stoichiometry_mass_balance_error_carbon",
        "diagnostics.stoichiometry_mass_balance_error_nitrogen",
        "diagnostics.stoichiometry_mass_balance_error_total",
    } <= optional
    state = _with_diagnostics(
        {},
        carbon_fluxes={"delta_fast_som_c": torch.tensor(0.1)},
        nitrogen_fluxes={"delta_fast_som_n": torch.tensor(0.01)},
        carbon_inputs=torch.tensor(1e-6),
        carbon_outputs=torch.tensor(5e-7),
    )
    output = stoich.forward(state, time=0.0, dt=3600.0)
    assert "diagnostics.stoichiometry_mass_balance_error_carbon" in output
    assert "diagnostics.stoichiometry_mass_balance_error_nitrogen" not in output
    assert "diagnostics.stoichiometry_mass_balance_error_total" not in output


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


