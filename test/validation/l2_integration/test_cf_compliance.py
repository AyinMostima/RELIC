"""
L2 Integration Tests: CF-1.8 Compliance and Cross-Domain Integration

Tests for CF-1.8 standard compliance, variable naming consistency,
unit conversion validation, and cross-domain interface contracts.

Focus Areas:
1. CF-1.8 Standard Name Validation
2. Unit System Consistency and Conversion
3. Cross-Domain Variable Mapping
4. Interface Contract Validation
5. Metadata Compliance

Mathematical Framework:
    CF-1.8 compliance requires:
    - Standard names from CFSR vocabulary
    - Unit consistency with UDUNITS-2 library
    - Coordinate system compliance (latitude, longitude, time)
    - Cell methods for statistical operations
    
References:
    - CF-1.8 Convention: https://cfconventions.org/
    - UDUNITS-2 Reference: https://www.unidata.ucar.edu/software/udunits/
"""

import pytest

from relic.validation.framework.models import (
    ValidationLevel, ValidationCode, ValidationDetail,
    ValidationResultModel, ValidationReportModel, ValidationContext
)
from relic.validation.framework.engine import ValidationEngine
from relic.validation.framework.decorators import validated_rule


class TestCFStandardNameValidation:
    """Test CF-1.8 standard name validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
        
        # Mock variable data with CF standard names
        self.valid_variables = {
            'temperature': {
                'cf_standard_name': 'air_temperature',
                'units': 'K',
                'long_name': 'Air Temperature'
            },
            'gpp': {
                'cf_standard_name': 'gross_primary_productivity_of_carbon',
                'units': 'kg C m-2 s-1',
                'long_name': 'Gross Primary Production'
            },
            'precipitation': {
                'cf_standard_name': 'precipitation_flux',
                'units': 'kg m-2 s-1',
                'long_name': 'Precipitation Rate'
            }
        }
        
        self.invalid_variables = {
            'temp': {
                'cf_standard_name': 'invalid_temperature_name',
                'units': 'C',
                'long_name': 'Temperature'
            },
            'carbon_flux': {
                'cf_standard_name': '',  # Missing standard name
                'units': 'g C m-2 day-1',
                'long_name': 'Carbon Flux'
            }
        }
    
    def test_valid_cf_standard_names(self):
        """Test validation of valid CF standard names."""
        @validated_rule(
            rule_id="cf_standard_name_validation",
            layer="L2",
            domains=["atmosphere", "biosphere"],
            tags=["cf", "metadata"]
        )
        def validate_cf_standard_names(context):
            """Validate CF standard names for variables."""
            results = []
            variables = context.config.get('variables', {})
            
            # Known valid CF standard names (subset for testing)
            valid_cf_names = {
                'air_temperature',
                'gross_primary_productivity_of_carbon',
                'precipitation_flux',
                'soil_temperature',
                'surface_downwelling_shortwave_flux'
            }
            
            for var_name, var_info in variables.items():
                cf_name = var_info.get('cf_standard_name', '')
                
                if not cf_name:
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.CF_NAME_MISSING,
                        message=f"Variable '{var_name}' missing CF standard name",
                        layer="L2",
                        scope="cf_standard_name_validation"
                    ))
                elif cf_name not in valid_cf_names:
                    results.append(ValidationResultModel(
                        level=ValidationLevel.ERROR,
                        code=ValidationCode.CF_STANDARD_NAME_INVALID,
                        message=f"Variable '{var_name}' has invalid CF standard name: '{cf_name}'",
                        layer="L2",
                        scope="cf_standard_name_validation"
                    ))
                else:
                    results.append(ValidationResultModel(
                        level=ValidationLevel.INFO,
                        code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add CF_COMPLIANCE_OK
                        message=f"Variable '{var_name}' has valid CF standard name: '{cf_name}'",
                        layer="L2",
                        scope="cf_standard_name_validation",
                        tags=["cf", "valid"]
                    ))
            
            return results
        
        # Test valid variables
        context = ValidationContext(config={'variables': self.valid_variables})
        results = validate_cf_standard_names(context)
        
        # Should have INFO results for all valid variables
        info_results = [r for r in results if r.level == ValidationLevel.INFO]
        assert len(info_results) == 3
        
        # No ERROR or WARNING results
        error_results = [r for r in results if r.level in [ValidationLevel.ERROR, ValidationLevel.WARNING]]
        assert len(error_results) == 0
    
    def test_invalid_cf_standard_names(self):
        """Test validation of invalid CF standard names."""
        @validated_rule(
            rule_id="cf_standard_name_validation",
            layer="L2",
            domains=["atmosphere", "biosphere"],
            tags=["cf", "metadata"]
        )
        def validate_cf_standard_names(context):
            """Validate CF standard names for variables."""
            results = []
            variables = context.config.get('variables', {})
            
            # Known valid CF standard names (subset for testing)
            valid_cf_names = {
                'air_temperature',
                'gross_primary_productivity_of_carbon',
                'precipitation_flux'
            }
            
            for var_name, var_info in variables.items():
                cf_name = var_info.get('cf_standard_name', '')
                
                if not cf_name:
                    results.append(ValidationResultModel(
                        level=ValidationLevel.WARNING,
                        code=ValidationCode.CF_NAME_MISSING,
                        message=f"Variable '{var_name}' missing CF standard name",
                        layer="L2",
                        scope="cf_standard_name_validation"
                    ))
                elif cf_name not in valid_cf_names:
                    results.append(ValidationResultModel(
                        level=ValidationLevel.ERROR,
                        code=ValidationCode.CF_STANDARD_NAME_INVALID,
                        message=f"Variable '{var_name}' has invalid CF standard name: '{cf_name}'",
                        layer="L2",
                        scope="cf_standard_name_validation"
                    ))
            
            return results
        
        # Test invalid variables
        context = ValidationContext(config={'variables': self.invalid_variables})
        results = validate_cf_standard_names(context)
        
        # Should have 1 ERROR (invalid name) and 1 WARNING (missing name)
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        warning_results = [r for r in results if r.level == ValidationLevel.WARNING]
        
        assert len(error_results) == 1
        assert len(warning_results) == 1
        assert "invalid_temperature_name" in error_results[0].message
        assert "missing CF standard name" in warning_results[0].message


class TestUnitSystemConsistency:
    """Test unit system consistency and conversion validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_unit_consistency_validation(self):
        """Test validation of unit consistency across variables."""
        @validated_rule(
            rule_id="unit_consistency_validation",
            layer="L2",
            domains=["all"],
            tags=["units", "consistency"]
        )
        def validate_unit_consistency(context):
            """Validate unit consistency for related variables."""
            results = []
            variables = context.config.get('variables', {})
            
            # Unit consistency rules
            unit_consistency_rules = {
                'temperature': ['K', 'degC'],
                'carbon_flux': ['kg C m-2 s-1', 'g C m-2 s-1', 'mol C m-2 s-1'],
                'precipitation': ['kg m-2 s-1', 'mm s-1'],
                'energy_flux': ['W m-2', 'J m-2 s-1']
            }
            
            def classify_variable_type(var_name, var_info):
                """Classify variable type based on name and CF standard name."""
                cf_name = var_info.get('cf_standard_name', '').lower()
                var_name_lower = var_name.lower()
                
                if 'temperature' in cf_name or 'temperature' in var_name_lower:
                    return 'temperature'
                elif 'carbon' in cf_name or 'gpp' in var_name_lower or 'npp' in var_name_lower:
                    return 'carbon_flux'
                elif 'precipitation' in cf_name or 'precip' in var_name_lower:
                    return 'precipitation'
                elif 'flux' in cf_name and ('shortwave' in cf_name or 'longwave' in cf_name):
                    return 'energy_flux'
                else:
                    return 'unknown'
            
            for var_name, var_info in variables.items():
                units = var_info.get('units', '')
                var_type = classify_variable_type(var_name, var_info)
                
                if var_type in unit_consistency_rules:
                    valid_units = unit_consistency_rules[var_type]
                    if units not in valid_units:
                        results.append(ValidationResultModel(
                            level=ValidationLevel.WARNING,
                            code=ValidationCode.CF_UNIT_MISMATCH,
                            message=f"Variable '{var_name}' has units '{units}', expected one of: {valid_units}",
                            layer="L2",
                            scope="unit_consistency_validation"
                        ))
                    else:
                        results.append(ValidationResultModel(
                            level=ValidationLevel.INFO,
                            code=ValidationCode.GENERIC_VALIDATION_FAILED,
                            message=f"Variable '{var_name}' has consistent units: '{units}'",
                            layer="L2",
                            scope="unit_consistency_validation"
                        ))
            
            return results
        
        # Test variables with mixed unit consistency
        test_variables = {
            'air_temp': {
                'cf_standard_name': 'air_temperature',
                'units': 'K'  # Correct
            },
            'soil_temp': {
                'cf_standard_name': 'soil_temperature', 
                'units': 'F'  # Incorrect
            },
            'gpp': {
                'cf_standard_name': 'gross_primary_productivity_of_carbon',
                'units': 'kg C m-2 s-1'  # Correct
            }
        }
        
        context = ValidationContext(config={'variables': test_variables})
        results = validate_unit_consistency(context)
        
        # Should have some INFO (correct) and some WARNING (incorrect) results
        info_results = [r for r in results if r.level == ValidationLevel.INFO]
        warning_results = [r for r in results if r.level == ValidationLevel.WARNING]
        
        assert len(info_results) >= 1  # At least air_temp and gpp should be correct
        assert len(warning_results) >= 1  # soil_temp should be incorrect
        
        # Check specific warning message
        temp_warning = next((r for r in warning_results if 'soil_temp' in r.message), None)
        assert temp_warning is not None
        assert "units 'F'" in temp_warning.message


class TestCrossDomainIntegration:
    """Test cross-domain variable integration and interface contracts."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    def test_cross_domain_variable_mapping(self):
        """Test cross-domain variable mapping and interface validation."""
        @validated_rule(
            rule_id="cross_domain_variable_mapping",
            layer="L2",
            domains=["atmosphere", "biosphere", "hydrosphere"],
            tags=["cross_domain", "interface"]
        )
        def validate_cross_domain_variables(context):
            """Validate cross-domain variable mappings and interfaces."""
            results = []
            domains = context.config.get('domains', {})
            
            # Define interface variable requirements between domains
            interface_requirements = {
                ('atmosphere', 'biosphere'): [
                    'air_temperature', 'relative_humidity', 'precipitation_flux',
                    'surface_downwelling_shortwave_flux'
                ],
                ('biosphere', 'hydrosphere'): [
                    'transpiration', 'root_uptake', 'soil_moisture_content'
                ],
                ('atmosphere', 'hydrosphere'): [
                    'precipitation_flux', 'evaporation_flux'
                ]
            }
            
            for (domain1, domain2), required_vars in interface_requirements.items():
                if domain1 in domains and domain2 in domains:
                    domain1_vars = set(domains[domain1].get('variables', {}).keys())
                    domain2_vars = set(domains[domain2].get('variables', {}).keys())
                    
                    for var_name in required_vars:
                        # Check if both domains have the variable
                        in_domain1 = var_name in domain1_vars
                        in_domain2 = var_name in domain2_vars
                        
                        if not (in_domain1 and in_domain2):
                            missing_domains = []
                            if not in_domain1:
                                missing_domains.append(domain1)
                            if not in_domain2:
                                missing_domains.append(domain2)
                            
                            results.append(ValidationResultModel(
                                level=ValidationLevel.ERROR,
                                code=ValidationCode.COUPLING_CONTRACT_VIOLATION,
                                message=f"Interface variable '{var_name}' missing in domains: {missing_domains}",
                                layer="L2",
                                scope="cross_domain_variable_mapping",
                                details=[ValidationDetail(
                                    variable_name=var_name,
                                    location={'missing_domains': missing_domains}
                                )]
                            ))
                        else:
                            # Variable present in both domains
                            results.append(ValidationResultModel(
                                level=ValidationLevel.INFO,
                                code=ValidationCode.GENERIC_VALIDATION_FAILED,  # TODO: Add INTERFACE_OK
                                message=f"Interface variable '{var_name}' available in {domain1} and {domain2}",
                                layer="L2",
                                scope="cross_domain_variable_mapping"
                            ))
            
            return results
        
        # Test complete interface
        complete_domains = {
            'atmosphere': {
                'variables': {
                    'air_temperature': {'units': 'K'},
                    'precipitation_flux': {'units': 'kg m-2 s-1'},
                    'surface_downwelling_shortwave_flux': {'units': 'W m-2'},
                    'relative_humidity': {'units': '1'},
                    'evaporation_flux': {'units': 'kg m-2 s-1'}  # Added missing interface variable
                }
            },
            'biosphere': {
                'variables': {
                    'air_temperature': {'units': 'K'},
                    'precipitation_flux': {'units': 'kg m-2 s-1'},
                    'surface_downwelling_shortwave_flux': {'units': 'W m-2'},
                    'relative_humidity': {'units': '1'},
                    'transpiration': {'units': 'kg m-2 s-1'},
                    'root_uptake': {'units': 'kg m-2 s-1'},
                    'soil_moisture_content': {'units': 'kg m-2'}
                }
            },
            'hydrosphere': {
                'variables': {
                    'precipitation_flux': {'units': 'kg m-2 s-1'},
                    'transpiration': {'units': 'kg m-2 s-1'},
                    'root_uptake': {'units': 'kg m-2 s-1'},
                    'soil_moisture_content': {'units': 'kg m-2'},
                    'evaporation_flux': {'units': 'kg m-2 s-1'}
                }
            }
        }
        
        context = ValidationContext(config={'domains': complete_domains})
        results = validate_cross_domain_variables(context)
        
        # Should have mostly INFO results (interfaces complete)
        info_results = [r for r in results if r.level == ValidationLevel.INFO]
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        
        assert len(info_results) > 0
        assert len(error_results) == 0  # All interfaces should be complete
    
    def test_incomplete_cross_domain_interface(self):
        """Test detection of incomplete cross-domain interfaces."""
        @validated_rule(
            rule_id="cross_domain_variable_mapping",
            layer="L2", 
            domains=["atmosphere", "biosphere"],
            tags=["cross_domain", "interface"]
        )
        def validate_cross_domain_variables(context):
            """Validate cross-domain variable mappings."""
            results = []
            domains = context.config.get('domains', {})
            
            # Simplified interface requirements for test
            required_vars = ['air_temperature', 'precipitation_flux']
            
            if 'atmosphere' in domains and 'biosphere' in domains:
                atm_vars = set(domains['atmosphere'].get('variables', {}).keys())
                bio_vars = set(domains['biosphere'].get('variables', {}).keys())
                
                for var_name in required_vars:
                    if var_name not in atm_vars or var_name not in bio_vars:
                        missing_domains = []
                        if var_name not in atm_vars:
                            missing_domains.append('atmosphere')
                        if var_name not in bio_vars:
                            missing_domains.append('biosphere')
                        
                        results.append(ValidationResultModel(
                            level=ValidationLevel.ERROR,
                            code=ValidationCode.COUPLING_CONTRACT_VIOLATION,
                            message=f"Interface variable '{var_name}' missing in: {missing_domains}",
                            layer="L2",
                            scope="cross_domain_variable_mapping"
                        ))
            
            return results
        
        # Test incomplete interface (missing variables)
        incomplete_domains = {
            'atmosphere': {
                'variables': {
                    'air_temperature': {'units': 'K'}
                    # Missing precipitation_flux
                }
            },
            'biosphere': {
                'variables': {
                    'precipitation_flux': {'units': 'kg m-2 s-1'}
                    # Missing air_temperature
                }
            }
        }
        
        context = ValidationContext(config={'domains': incomplete_domains})
        results = validate_cross_domain_variables(context)
        
        # Should have ERROR results for missing variables
        error_results = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(error_results) == 2  # Both variables missing from one domain each
        
        # Check error messages contain expected information
        error_messages = [r.message for r in error_results]
        assert any("air_temperature" in msg and "biosphere" in msg for msg in error_messages)
        assert any("precipitation_flux" in msg and "atmosphere" in msg for msg in error_messages)


class TestL2EngineIntegration:
    """Test L2-level validation engine integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ValidationEngine()
    
    
    def test_l2_validation_execution(self):
        """Test execution of L2 validation suite."""
        # Register some L2 rules for testing
        @validated_rule(
            rule_id="test_l2_rule_1",
            layer="L2",
            domains=["atmosphere"],
            tags=["test"]
        )
        def test_rule_1(context):
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.GENERIC_VALIDATION_FAILED,
                message="Test L2 rule 1 passed",
                layer="L2",
                scope="test_l2_rule_1"
            )]
        
        @validated_rule(
            rule_id="test_l2_rule_2",
            layer="L2",
            domains=["biosphere"],
            tags=["test"]
        )
        def test_rule_2(context):
            return [ValidationResultModel(
                level=ValidationLevel.WARNING,
                code=ValidationCode.CF_UNIT_MISMATCH,
                message="Test L2 rule 2 warning",
                layer="L2", 
                scope="test_l2_rule_2"
            )]
        
        # Execute L2 validation
        context = ValidationContext(
            domain_filter=["atmosphere", "biosphere"],
            execution_layer="L2"
        )
        
        report = self.engine.run(layers="L2", context=context)
        
        # Verify results
        assert isinstance(report, ValidationReportModel)
        assert len(report.results) >= 2  # Our test rules plus any registered rules
        
        # Check that L2 rules were executed
        l2_results = [r for r in report.results if r.layer == "L2"]
        assert len(l2_results) >= 2
        
        # Check summary statistics
        assert report.summary["total"] >= 2
        assert "L2" in report.summary["by_layer"]
        assert report.summary["by_layer"]["L2"] >= 2


if __name__ == "__main__":
    pytest.main([__file__])

