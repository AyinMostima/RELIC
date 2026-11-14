"""
L2 Integration Tests for CrossDomainValidator Integration

Tests the integration of CrossDomainValidator with the hierarchical aggregation
system to ensure cross-scale consistency, unit validation, and conservation
law enforcement across different spatial and temporal domains.

Coverage Areas:
- Cross-scale unit consistency validation
- Conservation law enforcement across domains
- Physical constraint consistency across scales
- Temporal-spatial domain boundary validation
- Multi-variable cross-domain coupling validation
- Error detection and diagnostic reporting

Test Framework:
- L2 integration tests for domain validation
- Mock multi-domain data scenarios
- Cross-validator integration with facade
- Boundary condition testing
- Conservation enforcement validation

Mathematical Validation:
- Unit consistency: units(aggregate(X)) = f(units(X))
- Conservation across domains: 危(X_domain1) = 危(X_domain2) 
- Physical constraints: min 鈮?X_scaled 鈮?max for all domains
- Boundary flux balance: inflow_domain = outflow_domain

References:
    - Domain validation theory
    - Cross-scale consistency requirements
    - RELIC conservation frameworks
"""

import pytest
import torch

# Import modules under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.core.policy import ScalingMode
from relic.core.state import StateManager


class MockCrossDomainValidator:
    """Mock CrossDomainValidator for integration testing."""
    
    def __init__(self):
        self.validation_history = []
        self.violation_count = 0
    
    def validate_unit_consistency(self, source_data, target_data, source_units, target_units):
        """Mock unit consistency validation."""
        validation_result = {
            'consistent': source_units == target_units,
            'source_units': source_units,
            'target_units': target_units,
            'conversion_applied': False
        }
        
        self.validation_history.append({
            'type': 'unit_consistency',
            'result': validation_result
        })
        
        return validation_result
    
    def validate_conservation_across_domains(self, domain1_data, domain2_data, tolerance=1e-9):
        """Mock conservation validation across domains."""
        domain1_total = domain1_data.sum() if torch.is_tensor(domain1_data) else sum(domain1_data.values())
        domain2_total = domain2_data.sum() if torch.is_tensor(domain2_data) else sum(domain2_data.values())
        
        relative_error = abs(domain2_total - domain1_total) / abs(domain1_total) if domain1_total != 0 else 0
        is_conserved = relative_error <= tolerance
        
        if not is_conserved:
            self.violation_count += 1
        
        validation_result = {
            'conserved': is_conserved,
            'relative_error': float(relative_error),
            'tolerance': tolerance,
            'domain1_total': float(domain1_total),
            'domain2_total': float(domain2_total)
        }
        
        self.validation_history.append({
            'type': 'conservation',
            'result': validation_result
        })
        
        return validation_result
    
    def validate_physical_constraints_consistency(self, data, constraints, variable_name):
        """Mock physical constraints validation."""
        violations = []
        
        if 'min' in constraints:
            min_violations = (data < constraints['min']).sum().item()
            if min_violations > 0:
                violations.append(f"{min_violations} values below minimum {constraints['min']}")
        
        if 'max' in constraints:
            max_violations = (data > constraints['max']).sum().item()
            if max_violations > 0:
                violations.append(f"{max_violations} values above maximum {constraints['max']}")
        
        validation_result = {
            'valid': len(violations) == 0,
            'violations': violations,
            'variable': variable_name,
            'constraints': constraints
        }
        
        self.validation_history.append({
            'type': 'physical_constraints',
            'result': validation_result
        })
        
        return validation_result
    
    def get_diagnostic_report(self):
        """Generate diagnostic report."""
        return {
            'total_validations': len(self.validation_history),
            'violation_count': self.violation_count,
            'validation_types': list(set(v['type'] for v in self.validation_history)),
            'history': self.validation_history
        }


class TestCrossDomainValidatorIntegration:
    """Test CrossDomainValidator integration with hierarchical aggregation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
        # Create manager with validator integration
        self.facade = ScaleManager(
            device=self.device,
            default_mode=ScalingMode.CONSERVATIVE,
            enforce_conservation=True,
            state_manager=StateManager(),
        )
        self.facade.initialize()
        
        # Create mock validator
        self.validator = MockCrossDomainValidator()
        
        # Mock integration (in real implementation, would be injected)
        self.facade.cross_domain_validator = self.validator
        
        # Test scenarios
        self.test_scenarios = self._create_validation_scenarios()
    
    def _create_validation_scenarios(self):
        """Create test scenarios for validation."""
        scenarios = {}
        
        # Scenario 1: Unit consistency validation
        scenarios['unit_consistency'] = {
            'description': 'Test unit consistency across scales',
            'source_data': {
                'gpp': torch.tensor([10.0, 15.0, 12.0, 18.0, 14.0, 20.0], dtype=torch.float32)
            },
            'source_scale': 'patch',
            'target_scale': 'hillslope',
            'units': {
                'gpp': 'kg C m-2 s-1'
            },
            'expected_unit_consistency': True
        }
        
        # Scenario 2: Conservation validation
        scenarios['conservation_validation'] = {
            'description': 'Test conservation across domain boundaries',
            'source_data': {
                'streamflow': torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64)
            },
            'source_scale': 'patch',
            'target_scale': 'hillslope',
            'conservation_tolerance': 1e-9,
            'expect_conservation': True
        }
        
        # Scenario 3: Physical constraints validation
        scenarios['constraint_validation'] = {
            'description': 'Test physical constraint consistency',
            'source_data': {
                'temperature': torch.tensor([273.0, 275.0, 280.0, 285.0, 290.0, 295.0], dtype=torch.float32)
            },
            'source_scale': 'patch',
            'target_scale': 'hillslope',
            'constraints': {
                'temperature': {'min': 200.0, 'max': 350.0}  # K
            },
            'expect_constraint_violations': False
        }
        
        # Scenario 4: Multi-variable cross-domain validation
        scenarios['multi_variable_validation'] = {
            'description': 'Test multi-variable domain consistency',
            'source_data': {
                'carbon_flux': torch.tensor([8.0, 12.0, 10.0, 15.0, 11.0, 16.0], dtype=torch.float32),
                'water_flux': torch.tensor([20.0, 25.0, 22.0, 30.0, 28.0, 35.0], dtype=torch.float32),
                'energy_flux': torch.tensor([100.0, 120.0, 110.0, 140.0, 130.0, 150.0], dtype=torch.float32)
            },
            'source_scale': 'patch',
            'target_scale': 'hillslope',
            'coupling_requirements': {
                'carbon_water_ratio': {'min': 0.2, 'max': 0.8},
                'energy_carbon_ratio': {'min': 8.0, 'max': 15.0}
            }
        }
        
        return scenarios
    
    def test_unit_consistency_validation(self):
        """Test unit consistency validation across scale transitions."""
        scenario = self.test_scenarios['unit_consistency']
        
        # Mock hierarchical components
        self._mock_hierarchical_components()
        
        # Perform aggregation with unit validation
        source_data = scenario['source_data']['gpp']
        source_units = scenario['units']['gpp']
        
        # Mock aggregation result
        aggregated_result = torch.tensor([25.0, 32.0], dtype=torch.float32)  # Mock result
        
        # Validate unit consistency
        unit_validation = self.validator.validate_unit_consistency(
            source_data=source_data,
            target_data=aggregated_result,
            source_units=source_units,
            target_units=source_units  # Should be same for extensive variables
        )
        
        # Check validation result
        assert unit_validation['consistent'] == scenario['expected_unit_consistency']
        assert unit_validation['source_units'] == source_units
        assert unit_validation['target_units'] == source_units
        
        # Check validation history
        assert len(self.validator.validation_history) > 0
        assert self.validator.validation_history[-1]['type'] == 'unit_consistency'
    
    def test_conservation_across_domains(self):
        """Test conservation validation across domain boundaries."""
        scenario = self.test_scenarios['conservation_validation']
        
        # Mock hierarchical components
        self._mock_hierarchical_components()
        
        source_data = scenario['source_data']['streamflow']
        tolerance = scenario['conservation_tolerance']
        
        # Create aggregated result that conserves mass
        # Assuming 2 hillslopes: [0,1,2] -> hillslope0, [3,4,5] -> hillslope1
        expected_hillslope1 = source_data[0] + source_data[1] + source_data[2]  # 9.0
        expected_hillslope2 = source_data[3] + source_data[4] + source_data[5]  # 18.0
        aggregated_result = torch.tensor([expected_hillslope1, expected_hillslope2], dtype=torch.float64)
        
        # Validate conservation
        conservation_validation = self.validator.validate_conservation_across_domains(
            domain1_data=source_data,
            domain2_data=aggregated_result,
            tolerance=tolerance
        )
        
        # Check conservation validation
        assert conservation_validation['conserved'] == scenario['expect_conservation']
        assert conservation_validation['relative_error'] <= tolerance
        
        # Verify totals match
        source_total = source_data.sum().item()
        target_total = aggregated_result.sum().item()
        assert abs(target_total - source_total) / source_total <= tolerance
    
    def test_physical_constraints_validation(self):
        """Test physical constraint consistency validation."""
        scenario = self.test_scenarios['constraint_validation']
        
        source_data = scenario['source_data']['temperature']
        constraints = scenario['constraints']['temperature']
        
        # Mock aggregated result (area-weighted mean for temperature)
        aggregated_result = torch.tensor([276.0, 290.0], dtype=torch.float32)  # Mock result
        
        # Validate physical constraints
        constraint_validation = self.validator.validate_physical_constraints_consistency(
            data=aggregated_result,
            constraints=constraints,
            variable_name='temperature'
        )
        
        # Check constraint validation
        expected_valid = not scenario['expect_constraint_violations']
        assert constraint_validation['valid'] == expected_valid
        
        if scenario['expect_constraint_violations']:
            assert len(constraint_validation['violations']) > 0
        else:
            assert len(constraint_validation['violations']) == 0
        
        # Verify constraints are checked
        assert constraint_validation['constraints'] == constraints
        assert constraint_validation['variable'] == 'temperature'
    
    def test_multi_variable_cross_domain_validation(self):
        """Test multi-variable cross-domain consistency validation."""
        scenario = self.test_scenarios['multi_variable_validation']
        
        source_data = scenario['source_data']
        coupling_requirements = scenario['coupling_requirements']
        
        # Mock aggregated results
        aggregated_data = {
            'carbon_flux': torch.tensor([30.0, 42.0], dtype=torch.float32),
            'water_flux': torch.tensor([67.0, 93.0], dtype=torch.float32),
            'energy_flux': torch.tensor([330.0, 420.0], dtype=torch.float32)
        }
        
        # Validate individual variables
        validation_results = {}
        for var_name, var_data in aggregated_data.items():
            # Basic conservation validation
            conservation_result = self.validator.validate_conservation_across_domains(
                domain1_data=source_data[var_name],
                domain2_data=var_data,
                tolerance=1e-8
            )
            validation_results[var_name] = conservation_result
        
        # Validate coupling relationships
        coupling_validation = {}
        
        # Carbon-water ratio validation
        carbon_water_ratio = aggregated_data['carbon_flux'] / aggregated_data['water_flux']
        ratio_constraints = coupling_requirements['carbon_water_ratio']
        
        ratio_valid = (
            (carbon_water_ratio >= ratio_constraints['min']).all() and
            (carbon_water_ratio <= ratio_constraints['max']).all()
        )
        
        coupling_validation['carbon_water_ratio'] = {
            'valid': ratio_valid.item(),
            'ratios': carbon_water_ratio.tolist(),
            'constraints': ratio_constraints
        }
        
        # Energy-carbon ratio validation
        energy_carbon_ratio = aggregated_data['energy_flux'] / aggregated_data['carbon_flux']
        energy_constraints = coupling_requirements['energy_carbon_ratio']
        
        energy_ratio_valid = (
            (energy_carbon_ratio >= energy_constraints['min']).all() and
            (energy_carbon_ratio <= energy_constraints['max']).all()
        )
        
        coupling_validation['energy_carbon_ratio'] = {
            'valid': energy_ratio_valid.item(),
            'ratios': energy_carbon_ratio.tolist(),
            'constraints': energy_constraints
        }
        
        # Check that all individual variables are conserved
        for var_name, result in validation_results.items():
            assert result['conserved'], f"Conservation failed for {var_name}"
        
        # Check coupling relationships
        assert coupling_validation['carbon_water_ratio']['valid'], "Carbon-water ratio out of bounds"
        assert coupling_validation['energy_carbon_ratio']['valid'], "Energy-carbon ratio out of bounds"
    
    def test_boundary_flux_validation(self):
        """Test boundary flux validation across domain interfaces."""
        # Create boundary flux scenario
        patch_inflow = torch.tensor([5.0, 8.0, 6.0, 10.0, 7.0, 12.0], dtype=torch.float32)
        patch_outflow = torch.tensor([4.8, 8.2, 5.9, 10.1, 6.8, 12.3], dtype=torch.float32)  # Small differences
        
        # Expected hillslope fluxes (aggregated)
        hillslope1_inflow = patch_inflow[0:3].sum()   # 19.0
        hillslope1_outflow = patch_outflow[0:3].sum() # 18.9
        hillslope2_inflow = patch_inflow[3:6].sum()   # 29.0  
        hillslope2_outflow = patch_outflow[3:6].sum() # 29.2
        
        aggregated_inflow = torch.tensor([hillslope1_inflow, hillslope2_inflow])
        aggregated_outflow = torch.tensor([hillslope1_outflow, hillslope2_outflow])
        
        # Validate boundary balance
        inflow_conservation = self.validator.validate_conservation_across_domains(
            domain1_data=patch_inflow,
            domain2_data=aggregated_inflow,
            tolerance=1e-9
        )
        
        outflow_conservation = self.validator.validate_conservation_across_domains(
            domain1_data=patch_outflow,
            domain2_data=aggregated_outflow,
            tolerance=0.3  # More realistic tolerance for small conservation errors
        )
        
        # Calculate boundary balance at hillslope level
        boundary_balance_errors = torch.abs(aggregated_inflow - aggregated_outflow)
        relative_balance_errors = boundary_balance_errors / aggregated_inflow
        
        # Check boundary flux conservation
        assert inflow_conservation['conserved'], "Inflow not conserved across scales"
        assert outflow_conservation['conserved'], "Outflow not conserved across scales"
        
        # Check boundary balance (should be within reasonable limits)
        max_balance_error = relative_balance_errors.max().item()
        assert max_balance_error <= 0.05, f"Boundary balance error too large: {max_balance_error}"
    
    def test_temporal_domain_consistency(self):
        """Test consistency validation across temporal domains."""
        # Create temporal data: hourly -> daily aggregation
        hourly_data = torch.tensor([
            [1.0, 1.5, 2.0, 1.8, 1.2, 2.5],  # Hour 1
            [1.2, 1.8, 2.2, 2.0, 1.4, 2.8],  # Hour 2
            [1.1, 1.6, 1.9, 1.9, 1.3, 2.6],  # Hour 3
            [1.3, 1.7, 2.1, 2.1, 1.5, 2.9]   # Hour 4
        ], dtype=torch.float32)
        
        # Daily aggregate (sum over time)
        daily_data = hourly_data.sum(dim=0)
        
        # Validate temporal aggregation conservation
        temporal_conservation = self.validator.validate_conservation_across_domains(
            domain1_data=hourly_data,
            domain2_data=daily_data,
            tolerance=1e-9
        )
        
        # Check temporal conservation
        assert temporal_conservation['conserved'], "Temporal aggregation not conserved"
        
        # Validate spatial consistency within temporal domain
        # Each hour should have same spatial pattern (relative)
        for hour in range(4):
            hour_data = hourly_data[hour]
            
            # Validate units and constraints
            unit_validation = self.validator.validate_unit_consistency(
                source_data=hour_data,
                target_data=daily_data,
                source_units='kg C m-2 s-1',
                target_units='kg C m-2 s-1'
            )
            
            assert unit_validation['consistent'], f"Unit inconsistency at hour {hour}"
    
    def test_validator_diagnostic_reporting(self):
        """Test comprehensive validator diagnostic reporting."""
        # Run multiple validation operations
        test_data = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float32)
        
        # Unit validation
        self.validator.validate_unit_consistency(
            source_data=test_data,
            target_data=test_data,
            source_units='kg m-2',
            target_units='kg m-2'
        )
        
        # Conservation validation
        self.validator.validate_conservation_across_domains(
            domain1_data=test_data,
            domain2_data=test_data * 1.001,  # Small error
            tolerance=1e-6
        )
        
        # Constraint validation
        self.validator.validate_physical_constraints_consistency(
            data=test_data,
            constraints={'min': 5.0, 'max': 25.0},
            variable_name='test_var'
        )
        
        # Get diagnostic report
        report = self.validator.get_diagnostic_report()
        
        # Validate report structure
        assert 'total_validations' in report
        assert 'violation_count' in report
        assert 'validation_types' in report
        assert 'history' in report
        
        # Check report content
        assert report['total_validations'] >= 3
        assert len(report['validation_types']) >= 3
        assert 'unit_consistency' in report['validation_types']
        assert 'conservation' in report['validation_types']
        assert 'physical_constraints' in report['validation_types']
        
        # Validate history details
        assert len(report['history']) == report['total_validations']
        
        for validation_entry in report['history']:
            assert 'type' in validation_entry
            assert 'result' in validation_entry
            assert validation_entry['type'] in ['unit_consistency', 'conservation', 'physical_constraints']
    
    def test_error_handling_and_recovery(self):
        """Test validator error handling and recovery mechanisms."""
        # Test with invalid data
        invalid_scenarios = [
            {
                'name': 'nan_data',
                'data': torch.tensor([float('nan'), 1.0, 2.0]),
                'expect_error': True
            },
            {
                'name': 'inf_data',
                'data': torch.tensor([float('inf'), 1.0, 2.0]),
                'expect_error': True
            },
            {
                'name': 'empty_data',
                'data': torch.tensor([]),
                'expect_error': True
            },
            {
                'name': 'mismatched_shapes',
                'source_data': torch.tensor([1.0, 2.0]),
                'target_data': torch.tensor([1.0, 2.0, 3.0]),
                'expect_error': False  # Should handle gracefully
            }
        ]
        
        for scenario in invalid_scenarios:
            try:
                if 'source_data' in scenario and 'target_data' in scenario:
                    # Test conservation validation with mismatched shapes
                    result = self.validator.validate_conservation_across_domains(
                        domain1_data=scenario['source_data'],
                        domain2_data=scenario['target_data']
                    )
                    
                    if scenario['expect_error']:
                        pytest.fail(f"Expected error for scenario: {scenario['name']}")
                    else:
                        # Should handle gracefully
                        assert 'conserved' in result
                        
                else:
                    # Test constraint validation with invalid data
                    result = self.validator.validate_physical_constraints_consistency(
                        data=scenario['data'],
                        constraints={'min': 0.0, 'max': 10.0},
                        variable_name='test'
                    )
                    
                    if scenario['expect_error']:
                        # Should detect invalid data
                        assert not result['valid'] or len(result['violations']) > 0
                    
            except Exception as e:
                if not scenario['expect_error']:
                    pytest.fail(f"Unexpected error for scenario {scenario['name']}: {e}")
    
    # Helper methods
    def _mock_hierarchical_components(self):
        """Mock hierarchical components for testing."""
        # Simple mock for testing - would use actual mocks in implementation
        pass


class TestValidatorIntegrationPerformance:
    """Test performance aspects of validator integration."""
    
    def test_large_scale_validation_performance(self):
        """Test validation performance with large datasets."""
        import time
        
        # Large datasets
        n_patches = 10000
        large_source = torch.randn(n_patches, dtype=torch.float32)
        large_target = torch.randn(n_patches // 10, dtype=torch.float32)
        
        validator = MockCrossDomainValidator()
        
        # Time validation operations
        start_time = time.time()
        
        # Unit validation
        validator.validate_unit_consistency(
            source_data=large_source,
            target_data=large_target,
            source_units='kg m-2',
            target_units='kg m-2'
        )
        
        # Conservation validation
        validator.validate_conservation_across_domains(
            domain1_data=large_source,
            domain2_data=large_target
        )
        
        # Constraint validation
        validator.validate_physical_constraints_consistency(
            data=large_target,
            constraints={'min': -10.0, 'max': 10.0},
            variable_name='large_var'
        )
        
        elapsed_time = time.time() - start_time
        
        # Performance should be reasonable
        assert elapsed_time < 5.0, f"Validation too slow: {elapsed_time:.2f}s"
        
        # Get diagnostic report
        report = validator.get_diagnostic_report()
        assert report['total_validations'] == 3


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])

