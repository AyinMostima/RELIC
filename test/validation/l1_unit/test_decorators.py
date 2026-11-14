"""
L1 Unit Tests: Validation Decorators

Tests for @validated_rule decorator,
focusing on rule standardization and error handling.

Focus Areas:
1. @validated_rule: Rule registration, metadata handling, error wrapping
2. Error Handling: Exception catching, error result generation
3. Rule Signature: Parameter validation, context handling
4. Metadata Integration: Rule registration with registry

Mathematical Framework:
    Decorator D transforms function f with metadata m:
    D(f, m) -> g where g : ValidationContext -> ValidationResults
    
References:
    - Python decorator patterns
    - Function signature introspection"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from relic.validation.framework.decorators import (
    validated_rule
)
from relic.validation.framework.models import (
    ValidationResultModel, ValidationLevel, ValidationCode, ValidationContext
)
from relic.validation.framework.registry import RuleMetadata


class TestValidatedRuleDecorator:
    """Test @validated_rule decorator functionality."""
    
    def test_basic_validated_rule(self):
        """Test basic @validated_rule decoration."""
        @validated_rule(
            rule_id="test_rule",
            layer="L1",
            name="Test Rule",
            description="A test validation rule"
        )
        def test_validation_rule(context: ValidationContext) -> List[ValidationResultModel]:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_BOUND_VIOLATION,
                message="Test validation passed",
                layer="L1",
                scope="test_rule"
            )]
        
        # Test that the function is wrapped properly
        assert callable(test_validation_rule)
        assert hasattr(test_validation_rule, '__wrapped__')
        assert hasattr(test_validation_rule, '_validation_metadata')
        
        # Test metadata
        metadata = test_validation_rule._validation_metadata
        assert metadata.rule_id == "test_rule"
        assert metadata.layer == "L1"
        assert metadata.name == "Test Rule"
        assert metadata.description == "A test validation rule"
    
    def test_validated_rule_with_full_metadata(self):
        """Test @validated_rule with complete metadata."""
        @validated_rule(
            rule_id="comprehensive_rule",
            layer="L2",
            name="Comprehensive Rule",
            description="A comprehensive test rule",
            domains=["biosphere", "atmosphere"],
            tags=["conservation", "mass_balance"],
            dependencies=["prerequisite_rule"],
            references=["Smith et al. 2023", "Jones 2022"],
            units_contract={"input": "kg/m^2/s", "output": "kg/m^2/s"}
        )
        def comprehensive_rule(context: ValidationContext) -> List[ValidationResultModel]:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.MASS_BALANCE_FAIL_CARBON,
                message="Conservation check",
                layer="L2",
                scope="comprehensive_rule"
            )]
        
        metadata = comprehensive_rule._validation_metadata
        assert metadata.domains == ["biosphere", "atmosphere"]
        # Tags include both specified tags and domains (auto-added)
        assert "conservation" in metadata.tags
        assert "mass_balance" in metadata.tags
        assert "biosphere" in metadata.tags  # Auto-added from domains
        assert "atmosphere" in metadata.tags  # Auto-added from domains
        assert metadata.dependencies == ["prerequisite_rule"]
        assert metadata.references == ["Smith et al. 2023", "Jones 2022"]
        assert metadata.units_contract == {"input": "kg/m^2/s", "output": "kg/m^2/s"}
    
    def test_validated_rule_execution(self):
        """Test execution of @validated_rule decorated function."""
        @validated_rule(
            rule_id="execution_test",
            layer="L1",
            name="Execution Test"
        )
        def test_rule(context: ValidationContext) -> List[ValidationResultModel]:
            # Use context information
            domain_filter = context.domain_filter or []
            message = f"Executed for domains: {domain_filter}"
            
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_BOUND_VIOLATION,
                message=message,
                layer="L1",
                scope="execution_test"
            )]
        
        # Execute the decorated rule
        context = ValidationContext(domain_filter=["biosphere"])
        results = test_rule(context)
        
        assert len(results) == 1
        assert isinstance(results[0], ValidationResultModel)
        assert results[0].level == ValidationLevel.INFO
        assert "biosphere" in results[0].message
        assert results[0].scope == "execution_test"
    
    def test_validated_rule_error_handling(self):
        """Test error handling in @validated_rule decorated functions."""
        @validated_rule(
            rule_id="error_test",
            layer="L1",
            name="Error Test"
        )
        def error_rule(context: ValidationContext) -> List[ValidationResultModel]:
            raise ValueError("Intentional test error")
        
        context = ValidationContext()
        results = error_rule(context)
        
        # Should return error result instead of raising exception
        assert len(results) == 1
        assert results[0].level == ValidationLevel.ERROR
        assert results[0].code == ValidationCode.GENERIC_VALIDATION_FAILED
        assert "Intentional test error" in results[0].message
        assert results[0].scope == "error_test"
    
    def test_validated_rule_auto_registration(self):
        """Test automatic registry registration."""
        with patch('relic.validation.framework.decorators.get_global_registry') as mock_registry:
            mock_reg_instance = Mock()
            mock_registry.return_value = mock_reg_instance
            
            @validated_rule(
                rule_id="auto_reg_test",
                layer="L1",
                name="Auto Registration Test"
            )
            def auto_reg_rule(context: ValidationContext) -> List[ValidationResultModel]:
                return []
            
            # Verify registration was called
            mock_reg_instance.register_rule.assert_called_once()
            call_args = mock_reg_instance.register_rule.call_args
            rule_func, metadata = call_args[0]  # First two positional arguments
            assert callable(rule_func)  # rule function
            assert isinstance(metadata, RuleMetadata)  # metadata
            assert metadata.rule_id == "auto_reg_test"
    
    def test_validated_rule_return_type_validation(self):
        """Test validation of rule return types."""
        @validated_rule(
            rule_id="return_test",
            layer="L1",
            name="Return Type Test"
        )
        def wrong_return_type(context: ValidationContext) -> List[ValidationResultModel]:
            # Return wrong type - should be handled gracefully
            return "invalid return type"
        
        context = ValidationContext()
        results = wrong_return_type(context)
        
        # Should convert to proper error result  
        assert len(results) == 1
        assert results[0].level == ValidationLevel.WARNING
        assert results[0].code == ValidationCode.GENERIC_VALIDATION_FAILED
        assert "non-standard result" in results[0].message


# TestValidationRuleWrapper removed as ValidationRuleWrapper is not implemented


class TestDecoratorIntegration:
    """Test integration between decorators and other framework components."""
    
    @patch('relic.validation.framework.decorators.get_global_registry')
    def test_decorator_registry_integration(self, mock_registry):
        """Test decorator integration with validation registry."""
        mock_reg_instance = Mock()
        mock_registry.return_value = mock_reg_instance
        
        @validated_rule(
            rule_id="integration_test",
            layer="L1",
            name="Integration Test"
        )
        def integration_rule(context: ValidationContext) -> List[ValidationResultModel]:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_BOUND_VIOLATION,
                message="Integration test",
                layer="L1",
                scope="integration_test"
            )]
        
        # Verify rule was registered
        mock_reg_instance.register_rule.assert_called_once()
        
        # Verify registered metadata
        call_args = mock_reg_instance.register_rule.call_args
        rule_func, metadata = call_args[0]
        
        # The rule_id is in the metadata, not a separate argument
        assert metadata.rule_id == "integration_test"
        assert callable(rule_func)
        assert metadata.name == "Integration Test"
        assert metadata.layer == "L1"
    
    def test_decorator_metadata_inheritance(self):
        """Test that decorated functions preserve original function metadata."""
        def original_function(context: ValidationContext) -> List[ValidationResultModel]:
            """Original function docstring."""
            return []
        
        original_function.custom_attr = "test_value"
        
        decorated = validated_rule(
            rule_id="metadata_test",
            layer="L1",
            name="Metadata Test"
        )(original_function)
        
        # Check that original metadata is preserved
        assert hasattr(decorated, '__wrapped__')
        assert decorated.__wrapped__ == original_function
        assert decorated.__wrapped__.custom_attr == "test_value"
        
        # Check that validation metadata is added
        assert hasattr(decorated, '_validation_metadata')
        assert decorated._validation_metadata.rule_id == "metadata_test"
    
    def test_multiple_decorator_application(self):
        """Test applying multiple decorators to the same function."""
        # Note: This is mainly to ensure decorators don't interfere
        def base_rule(context: ValidationContext) -> List[ValidationResultModel]:
            return [ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_BOUND_VIOLATION,
                message="Multi-decorator test",
                layer="L1",
                scope="multi_test"
            )]
        
        # Apply validated_rule decorator
        decorated_rule = validated_rule(
            rule_id="multi_decorator_test",
            layer="L1",
            name="Multi Decorator Test"
        )(base_rule)
        
        # Test that decorated function works
        context = ValidationContext()
        results = decorated_rule(context)
        
        assert len(results) == 1
        assert results[0].message == "Multi-decorator test"
        assert hasattr(decorated_rule, '_validation_metadata')


if __name__ == "__main__":
    pytest.main([__file__])

