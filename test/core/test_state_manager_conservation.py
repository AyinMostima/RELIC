"""
Test StateManager Element-Level Conservation Checking

Tests the enhanced conservation checking functionality including:
- Element pool identification (C/N/P)
- Conservation report generation  
- Strict conservation validation
- Feature flag integration

Mathematical expressions:
- Mass balance: 危(inputs) - 危(outputs) = 螖(pools)
- Conservation error: |螖(total_pools)| should be 鈮?0 for closed system
- Relative error: |error| / |total_pools| should be << 1

References:
- Ascher, U.M. & Petzold, L.R. (1998). Computer Methods for ODEs and DAEs
- Conservation principles in biogeochemical modeling
"""

import pytest
import torch
import sys
import os

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from relic.core.state import StateManager, StateVariableSpec, StateVariableType
from relic.core.feature_flags import feature_manager


class TestStateManagerConservation:
    
    def setup_method(self):
        """Set up test state manager with C/N/P pools."""
        self.state_manager = StateManager()
        # Initialize the state manager
        self.state_manager.initialize()
        
        # Add explicit carbon, nitrogen, and phosphorus pools for testing.
        carbon_specs = [
            StateVariableSpec(
                name="soil_carbon",
                units="kg C m-2",
                variable_type=StateVariableType.SCALAR,
                bounds=(0.0, 1000.0),
                description="Soil carbon pool",
                conserved=True,
                default_value=0.0,
            ),
            StateVariableSpec(
                name="root_carbon",
                units="kg C m-2",
                variable_type=StateVariableType.SCALAR,
                bounds=(0.0, 1000.0),
                description="Root carbon pool",
                conserved=True,
                default_value=0.0,
            ),
        ]

        # Add some nitrogen pools for testing
        nitrogen_specs = [
            # A synthetic soil pool: soil_nitrogen is a packaged forcing contract (the
            # coordinator's aggregate soil N observation, no writer, no element
            # domain), which the SCM registry keeps out of the nitrogen ledger.
            StateVariableSpec(
                name="subsoil_nitrogen",
                units="g N m-2", 
                variable_type=StateVariableType.SCALAR,
                bounds=(0.0, 1000.0),
                description="Soil nitrogen pool",
                conserved=True,
                default_value=50.0
            ),
            # A synthetic shoot pool: plant_nitrogen is a packaged forcing
            # contract (read by growth_respiration, no writer, no element
            # domain), which the SCM registry keeps out of the nitrogen ledger.
            StateVariableSpec(
                name="shoot_nitrogen",
                units="g N m-2",
                variable_type=StateVariableType.SCALAR,
                bounds=(0.0, 500.0),
                description="Shoot nitrogen pool",
                conserved=True,
                default_value=25.0
            ),
            StateVariableSpec(
                name="ammonium_n",
                units="g N m-2",
                variable_type=StateVariableType.SCALAR,
                bounds=(0.0, 100.0),
                description="Ammonium nitrogen pool",
                conserved=True,
                default_value=5.0
            )
        ]
        
        # Add phosphorus pools
        phosphorus_specs = [
            StateVariableSpec(
                name="soil_phosphorus",
                units="g P m-2",
                variable_type=StateVariableType.SCALAR,
                bounds=(0.0, 100.0),
                description="Soil phosphorus pool",
                conserved=True,
                default_value=10.0
            )
        ]
        
        # Register new variables
        for spec in carbon_specs + nitrogen_specs + phosphorus_specs:
            self.state_manager.register_variable(spec)
        
        # Re-initialize element pools after adding new variables
        self.state_manager._initialize_element_pools()
    
    def test_element_pool_identification(self):
        """Test automatic identification of C/N/P pools from schema."""
        pools = self.state_manager.element_pools
        
        def contains(pool_list, suffix):
            return any(name.endswith(suffix) for name in pool_list)
        
        # Check carbon pools (from default schema)
        assert contains(pools['carbon'], 'leaf_carbon')
        assert contains(pools['carbon'], 'root_carbon')
        assert contains(pools['carbon'], 'soil_carbon')
        
        # Check nitrogen pools  
        assert contains(pools['nitrogen'], 'subsoil_nitrogen')
        assert contains(pools['nitrogen'], 'shoot_nitrogen')
        assert contains(pools['nitrogen'], 'ammonium_n')
        
        # Check phosphorus pools
        assert contains(pools['phosphorus'], 'soil_phosphorus')
        
        print(f"Identified pools: C={len(pools['carbon'])}, N={len(pools['nitrogen'])}, P={len(pools['phosphorus'])}")
        print(f"Nitrogen pools: {pools['nitrogen']}")
        print(f"Carbon pools: {pools['carbon']}")
        print(f"Phosphorus pools: {pools['phosphorus']}")
    
    def test_element_conservation_report_tracks_each_element_independently(self):
        """A balanced N transfer must not hide an independent C residual."""
        # Create balanced nitrogen update (transfer between pools)
        updates = {
            'subsoil_nitrogen': torch.tensor(-10.0),    # Soil loses 10 g N
            'shoot_nitrogen': torch.tensor(10.0),    # Plant gains 10 g N
            'leaf_carbon': torch.tensor(5.0)         # Independent carbon residual
        }
        
        summary = self.state_manager.generate_conservation_report(updates)
        n_report = summary.nitrogen_report
        assert n_report is not None
        assert pytest.approx(n_report.balance_error, abs=1e-9) == 0.0
        assert summary.carbon_report is not None
        assert summary.carbon_report.balance_error == pytest.approx(5000.0)
        assert summary.total_violations == 1
        
        print(f"Balanced nitrogen transfer: error={n_report.balance_error:.2e}")
    
    def test_element_conservation_report_violation(self):
        """Test conservation report generation with violations."""
        # Create unbalanced nitrogen update (net creation)
        updates = {
            'subsoil_nitrogen': torch.tensor(15.0),     # Soil gains 15 g N
            'shoot_nitrogen': torch.tensor(10.0),    # Plant gains 10 g N
            # Net creation of 25 g N - violation!
        }
        
        summary = self.state_manager.generate_conservation_report(updates)
        n_report = summary.nitrogen_report
        assert n_report is not None
        assert pytest.approx(n_report.balance_error, rel=1e-6) == 25.0
        assert n_report.relative_error > 0.01  # Should exceed 1% threshold
        assert summary.total_violations > 0
        assert any("nitrogen" in warning.lower() for warning in summary.warnings)
        
        print(f"Nitrogen violation: error={n_report.balance_error:.1f}, relative={n_report.relative_error:.1%}")
    
    def test_negative_pool_detection(self):
        """Test detection of updates that would create negative pools."""
        # Create update that would make plant nitrogen negative
        updates = {
            'shoot_nitrogen': torch.tensor(-100.0),  # Plant loses 100 g N (current=25)
        }
        
        summary = self.state_manager.generate_conservation_report(updates)
        assert summary.nitrogen_report is not None
        assert summary.total_violations > 0
        assert any("nitrogen" in warning.lower() for warning in summary.warnings)
        
        print(f"Negative pool violation warnings: {summary.warnings}")
    
    def test_comprehensive_conservation_report(self):
        """Test generation of comprehensive conservation summary."""
        # Create mixed updates affecting multiple elements
        updates = {
            'leaf_carbon': torch.tensor(-5.0),      # Carbon loss
            'soil_carbon': torch.tensor(3.0),       # Partial carbon recovery
            'subsoil_nitrogen': torch.tensor(-2.0),    # Nitrogen loss
            'shoot_nitrogen': torch.tensor(2.0),    # Nitrogen gain (balanced)
            'soil_phosphorus': torch.tensor(1.0),   # Phosphorus gain (violation)
        }
        
        # Generate comprehensive report
        summary = self.state_manager.generate_conservation_report(updates)
        
        assert summary.carbon_report is not None
        assert summary.nitrogen_report is not None  
        assert summary.phosphorus_report is not None
        
        # Carbon should have a 2 kg C m-2 residual in canonical g C m-2.
        assert summary.carbon_report.balance_error == 2000.0
        
        # Nitrogen should be balanced
        assert summary.nitrogen_report.balance_error == 0.0
        
        # Phosphorus should have violation (net gain of 1 g P)
        assert summary.phosphorus_report.balance_error == 1.0
        
        print(f"Summary violations: total={summary.total_violations}, warnings={len(summary.warnings)}")
    
    def test_strict_conservation_feature_flag(self):
        """Test integration with strict conservation feature flag."""
        # Test with feature disabled (default)
        feature_manager.disable('strict_conservation_enabled')
        
        updates = {
            'subsoil_nitrogen': torch.tensor(100.0),  # Major violation
        }
        
        # Should pass when feature is disabled
        result = self.state_manager.check_strict_conservation(updates)
        assert result is True  # Should pass when disabled
        
        # Enable strict conservation
        feature_manager.enable('strict_conservation_enabled')
        
        # Should fail with major violation
        result = self.state_manager.check_strict_conservation(updates)
        assert result is False  # Should fail with violation
        
        # Check that report was stored
        assert len(self.state_manager.conservation_history) > 0
        latest_report = self.state_manager.conservation_history[-1]
        assert latest_report.total_violations > 0
        
        # Clean up
        feature_manager.disable('strict_conservation_enabled')
        
        print(f"Strict conservation test: violations detected={latest_report.total_violations}")
    
    def test_apply_updates_with_conservation_checking(self):
        """Test that apply_updates integrates with conservation checking."""
        # Enable strict conservation for this test
        feature_manager.enable('strict_conservation_enabled')
        
        try:
            # Test balanced update (should succeed)
            balanced_updates = {
                'subsoil_nitrogen': torch.tensor(-5.0),
                'shoot_nitrogen': torch.tensor(5.0),
            }
            
            initial_soil_n = float(self.state_manager.state_vars['subsoil_nitrogen'])
            initial_plant_n = float(self.state_manager.state_vars['shoot_nitrogen'])
            
            # Should succeed
            self.state_manager.apply_updates(balanced_updates)
            
            # Check that update was applied
            final_soil_n = float(self.state_manager.state_vars['subsoil_nitrogen'])
            final_plant_n = float(self.state_manager.state_vars['shoot_nitrogen'])
            
            assert final_soil_n == initial_soil_n - 5.0
            assert final_plant_n == initial_plant_n + 5.0
            
            print("Balanced update applied successfully")
            
            # Test severely unbalanced update (should fail)
            severe_violation = {
                'subsoil_nitrogen': torch.tensor(1000.0),  # Major violation
            }
            
            # Should raise ValueError due to conservation violation
            with pytest.raises(ValueError, match="(?i)conservation"):
                self.state_manager.apply_updates(severe_violation)
            
            print("Severe violation correctly rejected")
            
        finally:
            # Clean up
            feature_manager.disable('strict_conservation_enabled')
    
    def test_conservation_history_tracking(self):
        """Test that conservation history is properly tracked."""
        feature_manager.enable('strict_conservation_enabled')
        
        try:
            initial_history_length = len(self.state_manager.conservation_history)
            
            updates = {
                'soil_carbon': torch.tensor(10.0),  # Some violation
            }
            
            # Trigger conservation check
            self.state_manager.check_strict_conservation(updates)
            
            # Check that history was updated
            assert len(self.state_manager.conservation_history) == initial_history_length + 1
            
            latest_report = self.state_manager.conservation_history[-1]
            assert latest_report.carbon_report is not None
            assert latest_report.timestamp > 0
            
            print(f"Conservation history: {len(self.state_manager.conservation_history)} entries")
            
        finally:
            feature_manager.disable('strict_conservation_enabled')


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestStateManagerConservation()
    test_instance.setup_method()
    
    print("Testing StateManager Element-Level Conservation...")
    
    try:
        test_instance.test_element_pool_identification()
        print("[+] Element pool identification: PASS")
        
        test_instance.test_element_conservation_report_balanced()
        print("[+] Balanced conservation report: PASS")
        
        test_instance.test_element_conservation_report_violation()
        print("[+] Conservation violation detection: PASS")
        
        test_instance.test_negative_pool_detection()
        print("[+] Negative pool detection: PASS")
        
        test_instance.test_comprehensive_conservation_report()
        print("[+] Comprehensive conservation report: PASS")
        
        test_instance.test_strict_conservation_feature_flag()
        print("[+] Feature flag integration: PASS")
        
        test_instance.test_apply_updates_with_conservation_checking()
        print("[+] Apply updates with conservation: PASS")
        
        test_instance.test_conservation_history_tracking()
        print("[+] Conservation history tracking: PASS")
        
        print("\nAll StateManager conservation tests passed successfully!")
        
    except Exception as e:
        print(f"[-] Test failed: {e}")
        raise
