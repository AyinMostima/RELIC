"""
L1 Unit Tests: Validation Registry

Tests for ValidationRegistry functionality including rule registration,
metadata management, dependency resolution, and rule discovery.

Focus Areas:
1. Rule Registration: Adding/removing rules, metadata validation
2. Rule Discovery: Filtering by layer, domain, tags
3. Dependency Resolution: Topological sorting, cycle detection
4. Registry Integrity: Consistency checks, error handling
5. Thread Safety: Concurrent access patterns

Mathematical Framework:
    Registry R manages rules as directed acyclic graph (DAG):
    R = (V, E) where V = rules, E = dependencies
    
    Dependency resolution produces topological ordering:
    order(R) = topological_sort(DAG(R))
    
References:
    - Graph theory algorithms
    - Dependency injection patterns
"""

import pytest
import threading
import time

from relic.validation.framework.registry import (
    ValidationRegistry, RuleMetadata, get_global_registry, reset_global_registry
)
from relic.validation.framework import registry as registry_module
from relic.validation.framework.models import ValidationResultModel, ValidationLevel, ValidationCode


class TestValidationRegistry:
    """Test ValidationRegistry core functionality."""
    
    def setup_method(self):
        """Set up fresh registry for each test."""
        self.registry = ValidationRegistry()
    
    def test_registry_initialization(self):
        """Test ValidationRegistry initialization."""
        assert len(self.registry._rules) == 0
        assert len(self.registry._metadata) == 0
        assert isinstance(self.registry._rules, dict)
        assert isinstance(self.registry._metadata, dict)
    
    def test_register_simple_rule(self):
        """Test registering a basic validation rule."""
        def dummy_rule(context):
            return ValidationResultModel(
                level=ValidationLevel.INFO,
                code=ValidationCode.PARAM_BOUND_VIOLATION,
                message="Test rule",
                layer="L1",
                scope="test"
            )
        
        metadata = RuleMetadata(
            rule_id="test_rule",
            name="Test Rule",
            layer="L1",
            domains=["test"],
            description="A test validation rule"
        )
        
        self.registry.register_rule(dummy_rule, metadata)
        
        assert "test_rule" in self.registry._rules
        assert "test_rule" in self.registry._metadata
        assert self.registry._rules["test_rule"] == dummy_rule
        assert self.registry._metadata["test_rule"] == metadata
    
    def test_register_rule_with_dependencies(self):
        """Test registering rules with dependency relationships."""
        def rule_a(context):
            return []
        
        def rule_b(context):
            return []
        
        metadata_a = RuleMetadata(
            rule_id="rule_a",
            name="Rule A",
            description="Test rule A",
            layer="L1", 
            domains=["test"],
            dependencies=[]
        )
        
        metadata_b = RuleMetadata(
            rule_id="rule_b",
            name="Rule B",
            description="Test rule B",
            layer="L1",
            domains=["test"], 
            dependencies=["rule_a"]
        )
        
        self.registry.register_rule(rule_a, metadata_a)
        self.registry.register_rule(rule_b, metadata_b)
        
        # Test dependency resolution
        resolved_order = self.registry.resolve_dependencies(["rule_a", "rule_b"])
        assert resolved_order == ["rule_a", "rule_b"]
    
    def test_register_rule_overwrite_protection(self):
        """Test protection against overwriting existing rules."""
        def dummy_rule(context):
            return []
        
        metadata = RuleMetadata(
            rule_id="test_rule",
            name="Test Rule",
            description="Test rule",
            layer="L1",
            domains=["test"]
        )
        
        # Register rule first time - should succeed
        self.registry.register_rule(dummy_rule, metadata)
        
        # Try to register again without overwrite flag - should raise error
        with pytest.raises(ValueError, match="Rule 'test_rule' already exists"):
            self.registry.register_rule(dummy_rule, metadata)
        
        # Register with overwrite flag - should succeed
        self.registry.register_rule(dummy_rule, metadata, overwrite=True)
    
    def test_unregister_rule(self):
        """Test unregistering rules."""
        def dummy_rule(context):
            return []
        
        metadata = RuleMetadata(
            rule_id="test_rule",
            name="Test Rule",
            description="Test rule",
            layer="L1",
            domains=["test"]
        )
        
        self.registry.register_rule(dummy_rule, metadata)
        assert "test_rule" in self.registry._rules
        
        self.registry.unregister_rule("test_rule")
        assert "test_rule" not in self.registry._rules
        assert "test_rule" not in self.registry._metadata
    
    def test_unregister_nonexistent_rule(self):
        """Test unregistering a rule that doesn't exist."""
        # unregister_rule returns False if rule doesn't exist
        result = self.registry.unregister_rule("nonexistent")
        assert result is False
    
    def test_get_rule(self):
        """Test retrieving registered rules."""
        def dummy_rule(context):
            return []
        
        metadata = RuleMetadata(
            rule_id="test_rule",
            name="Test Rule",
            description="Test rule",
            layer="L1",
            domains=["test"]
        )
        
        self.registry.register_rule(dummy_rule, metadata)
        
        retrieved_rule = self.registry.get_rule("test_rule")
        assert retrieved_rule == dummy_rule
        
        # Test non-existent rule
        assert self.registry.get_rule("nonexistent") is None
    
    def test_get_rule_metadata(self):
        """Test retrieving rule metadata."""
        def dummy_rule(context):
            return []
        
        metadata = RuleMetadata(
            rule_id="test_rule",
            name="Test Rule",
            layer="L1",
            domains=["test"],
            description="Test description"
        )
        
        self.registry.register_rule(dummy_rule, metadata)
        
        retrieved_metadata = self.registry.get_metadata("test_rule")
        assert retrieved_metadata == metadata
        assert retrieved_metadata.name == "Test Rule"
        assert retrieved_metadata.description == "Test description"


class TestRuleFiltering:
    """Test rule discovery and filtering functionality."""
    
    def setup_method(self):
        """Set up registry with sample rules for filtering tests."""
        self.registry = ValidationRegistry()
        
        # Register multiple rules with different attributes
        rules_data = [
            ("cf_rule", "L1", ["atmosphere", "biosphere"], ["cf", "units"]),
            ("conservation_rule", "L2", ["biosphere"], ["conservation", "mass_balance"]),
            ("stochastic_rule", "L1", ["atmosphere"], ["stochastic", "scoring"]),
            ("coupling_rule", "L2", ["atmosphere", "biosphere"], ["coupling", "interfaces"]),
            ("parameter_rule", "L1", ["all"], ["parameters", "bounds"])
        ]
        
        for rule_id, layer, domains, tags in rules_data:
            def dummy_rule(context):
                return []
            
            metadata = RuleMetadata(
                rule_id=rule_id,
                name=f"{rule_id.replace('_', ' ').title()}",
                layer=layer,
                domains=domains,
                tags=tags,
                description=f"Test {rule_id}"
            )
            
            self.registry.register_rule(dummy_rule, metadata)
    
    def test_list_all_rules(self):
        """Test listing all registered rules."""
        all_rules = self.registry.list_rules()
        assert len(all_rules) == 5
        assert "cf_rule" in all_rules
        assert "conservation_rule" in all_rules
        assert "stochastic_rule" in all_rules
        assert "coupling_rule" in all_rules
        assert "parameter_rule" in all_rules
    
    def test_filter_by_layer(self):
        """Test filtering rules by validation layer."""
        l1_rules = self.registry.list_rules(layer="L1")
        assert len(l1_rules) == 3
        assert "cf_rule" in l1_rules
        assert "stochastic_rule" in l1_rules
        assert "parameter_rule" in l1_rules
        
        l2_rules = self.registry.list_rules(layer="L2")
        assert len(l2_rules) == 2
        assert "conservation_rule" in l2_rules
        assert "coupling_rule" in l2_rules
    
    def test_filter_by_domains(self):
        """Test filtering rules by scientific domains."""
        atmosphere_rules = self.registry.list_rules(domains=["atmosphere"])
        assert len(atmosphere_rules) == 3
        assert "cf_rule" in atmosphere_rules
        assert "stochastic_rule" in atmosphere_rules
        assert "coupling_rule" in atmosphere_rules
        
        biosphere_rules = self.registry.list_rules(domains=["biosphere"])
        assert len(biosphere_rules) == 3
        assert "cf_rule" in biosphere_rules
        assert "conservation_rule" in biosphere_rules
        assert "coupling_rule" in biosphere_rules
    
    def test_filter_by_tags(self):
        """Test filtering rules by tags."""
        conservation_rules = self.registry.list_rules(tags=["conservation"])
        assert len(conservation_rules) == 1
        assert "conservation_rule" in conservation_rules
        
        # Test multiple tag filtering
        cf_unit_rules = self.registry.list_rules(tags=["cf", "units"])
        assert len(cf_unit_rules) == 1
        assert "cf_rule" in cf_unit_rules
    
    def test_filter_by_multiple_criteria(self):
        """Test filtering by multiple criteria simultaneously."""
        # L1 rules in atmosphere domain
        l1_atm_rules = self.registry.list_rules(layer="L1", domains=["atmosphere"])
        assert len(l1_atm_rules) == 2
        assert "cf_rule" in l1_atm_rules
        assert "stochastic_rule" in l1_atm_rules
        
        # L2 rules with coupling tags
        l2_coupling_rules = self.registry.list_rules(layer="L2", tags=["coupling"])
        assert len(l2_coupling_rules) == 1
        assert "coupling_rule" in l2_coupling_rules
    
    def test_filter_enabled_only(self):
        """Test filtering by enabled status."""
        # Disable one rule
        cf_metadata = self.registry.get_metadata("cf_rule")
        cf_metadata.enabled = False
        
        # Test enabled only filter
        enabled_rules = self.registry.list_rules(enabled=True)
        assert len(enabled_rules) == 4
        assert "cf_rule" not in enabled_rules
        
        # Test all rules (including disabled)
        all_rules = self.registry.list_rules(enabled=None)
        assert len(all_rules) == 5
        assert "cf_rule" in all_rules


class TestDependencyResolution:
    """Test dependency resolution and topological sorting."""
    
    def setup_method(self):
        """Set up registry with dependent rules."""
        self.registry = ValidationRegistry()
    
    def test_simple_dependency_chain(self):
        """Test resolving a simple linear dependency chain."""
        def dummy_rule(context):
            return []
        
        # Create chain: A -> B -> C
        metadata_a = RuleMetadata(rule_id="rule_a", name="Rule A", description="Test rule A", layer="L1", domains=["test"], dependencies=[])
        metadata_b = RuleMetadata(rule_id="rule_b", name="Rule B", description="Test rule B", layer="L1", domains=["test"], dependencies=["rule_a"])
        metadata_c = RuleMetadata(rule_id="rule_c", name="Rule C", description="Test rule C", layer="L1", domains=["test"], dependencies=["rule_b"])
        
        self.registry.register_rule(dummy_rule, metadata_a)
        self.registry.register_rule(dummy_rule, metadata_b)  
        self.registry.register_rule(dummy_rule, metadata_c)
        
        resolved_order = self.registry.resolve_dependencies(["rule_c", "rule_a", "rule_b"])
        assert resolved_order == ["rule_a", "rule_b", "rule_c"]
    
    def test_complex_dependency_graph(self):
        """Test resolving complex dependency graphs."""
        def dummy_rule(context):
            return []
        
        # Create diamond dependency: A -> B,C -> D
        metadata_a = RuleMetadata(rule_id="rule_a", name="Rule A", description="Test rule A", layer="L1", domains=["test"], dependencies=[])
        metadata_b = RuleMetadata(rule_id="rule_b", name="Rule B", description="Test rule B", layer="L1", domains=["test"], dependencies=["rule_a"])
        metadata_c = RuleMetadata(rule_id="rule_c", name="Rule C", description="Test rule C", layer="L1", domains=["test"], dependencies=["rule_a"])
        metadata_d = RuleMetadata(rule_id="rule_d", name="Rule D", description="Test rule D", layer="L1", domains=["test"], dependencies=["rule_b", "rule_c"])
        
        for rule_id, metadata in [("rule_a", metadata_a), ("rule_b", metadata_b), 
                                 ("rule_c", metadata_c), ("rule_d", metadata_d)]:
            self.registry.register_rule(dummy_rule, metadata)
        
        resolved_order = self.registry.resolve_dependencies(["rule_d", "rule_c", "rule_b", "rule_a"])
        
        # Check that dependencies are respected
        a_index = resolved_order.index("rule_a")
        b_index = resolved_order.index("rule_b")
        c_index = resolved_order.index("rule_c")
        d_index = resolved_order.index("rule_d")
        
        assert a_index < b_index  # A before B
        assert a_index < c_index  # A before C
        assert b_index < d_index  # B before D
        assert c_index < d_index  # C before D
    
    def test_cyclic_dependency_detection(self):
        """Test detection of cyclic dependencies."""
        def dummy_rule(context):
            return []
        
        # Create cycle: A -> B -> C -> A
        metadata_a = RuleMetadata(rule_id="rule_a", name="Rule A", description="Test rule A", layer="L1", domains=["test"], dependencies=["rule_c"])
        metadata_b = RuleMetadata(rule_id="rule_b", name="Rule B", description="Test rule B", layer="L1", domains=["test"], dependencies=["rule_a"])
        metadata_c = RuleMetadata(rule_id="rule_c", name="Rule C", description="Test rule C", layer="L1", domains=["test"], dependencies=["rule_b"])
        
        for rule_id, metadata in [("rule_a", metadata_a), ("rule_b", metadata_b), ("rule_c", metadata_c)]:
            self.registry.register_rule(dummy_rule, metadata)
        
        with pytest.raises(ValueError, match="Circular dependency detected"):
            self.registry.resolve_dependencies(["rule_a", "rule_b", "rule_c"])
    
    def test_missing_dependency_handling(self):
        """Test handling of missing dependencies."""
        def dummy_rule(context):
            return []
        
        metadata = RuleMetadata(
            rule_id="rule_a",
            name="Rule A",
            description="Test rule A",
            layer="L1",
            domains=["test"],
            dependencies=["missing_rule"]
        )
        
        self.registry.register_rule(dummy_rule, metadata)
        
        with pytest.raises(ValueError, match="Dependency 'missing_rule' not found"):
            self.registry.resolve_dependencies(["rule_a"])


class TestRegistryIntegrity:
    """Test registry integrity and consistency checks."""
    
    def setup_method(self):
        """Set up registry for integrity testing."""
        self.registry = ValidationRegistry()
    
    def test_validate_registry_integrity(self):
        """Test registry integrity validation."""
        def dummy_rule(context):
            return []
        
        # Register valid rules
        metadata_a = RuleMetadata(rule_id="rule_a", name="Rule A", description="Test rule A", layer="L1", domains=["test"])
        metadata_b = RuleMetadata(rule_id="rule_b", name="Rule B", description="Test rule B", layer="L1", domains=["test"], dependencies=["rule_a"])
        
        self.registry.register_rule(dummy_rule, metadata_a)
        self.registry.register_rule(dummy_rule, metadata_b)
        
        integrity_result = self.registry.validate_registry_integrity()
        
        assert integrity_result["valid"] is True
        assert len(integrity_result["errors"]) == 0
        assert integrity_result["total_rules"] == 2
        assert integrity_result["rules_with_dependencies"] == 1
    
    def test_integrity_with_missing_dependencies(self):
        """Test integrity validation with missing dependencies."""
        def dummy_rule(context):
            return []
        
        # Register rule with missing dependency
        metadata = RuleMetadata(
            rule_id="rule_a",
            name="Rule A",
            description="Test rule A",
            layer="L1", 
            domains=["test"],
            dependencies=["missing_rule"]
        )
        
        self.registry.register_rule(dummy_rule, metadata)
        
        integrity_result = self.registry.validate_registry_integrity()
        
        assert integrity_result["valid"] is False
        assert len(integrity_result["errors"]) >= 1
        assert any("missing_rule" in error for error in integrity_result["errors"])
    
    def test_get_rule_statistics(self):
        """Test rule statistics generation."""
        def dummy_rule(context):
            return []
        
        # Register rules with different layers and domains
        rules_data = [
            ("rule_1", "L1", ["atmosphere"]),
            ("rule_2", "L1", ["biosphere"]),
            ("rule_3", "L2", ["atmosphere"]),
            ("rule_4", "L2", ["biosphere"])
        ]
        
        for rule_id, layer, domains in rules_data:
            metadata = RuleMetadata(rule_id=rule_id, name=rule_id, description=f"Test {rule_id}", layer=layer, domains=domains)
            self.registry.register_rule(dummy_rule, metadata)
        
        stats = self.registry.get_rule_statistics()
        
        assert stats["total_rules"] == 4
        assert stats["rules_by_layer"]["L1"] == 2
        assert stats["rules_by_layer"]["L2"] == 2
        assert "atmosphere" in stats["rules_by_domain"]
        assert "biosphere" in stats["rules_by_domain"]
        assert stats["enabled_rules"] == 4
        assert stats["disabled_rules"] == 0


class TestGlobalRegistry:
    """Test global registry singleton functionality."""

    def test_framework_public_registry_is_canonical_singleton(self, monkeypatch):
        """The framework package must not create a second registry owner."""
        from relic.validation import framework

        monkeypatch.setattr(
            registry_module,
            "_global_registry",
            ValidationRegistry(),
        )
        reset_global_registry()

        assert framework.validation_registry is get_global_registry()
    
    def test_global_registry_singleton(self):
        """Test that global registry is a singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2
        assert id(registry1) == id(registry2)
    
    def test_global_registry_persistence(self):
        """Test that global registry persists between calls."""
        def dummy_rule(context):
            return []
        
        metadata = RuleMetadata(rule_id="test_rule", name="Test Rule", description="Test rule", layer="L1", domains=["test"])
        
        registry1 = get_global_registry()
        registry1.register_rule(dummy_rule, metadata, overwrite=True)
        
        registry2 = get_global_registry()
        assert "test_rule" in registry2._rules
        assert registry2.get_rule("test_rule") is dummy_rule


class TestThreadSafety:
    """Test thread safety of registry operations."""
    
    def setup_method(self):
        """Set up for thread safety testing."""
        self.registry = ValidationRegistry()
        self.errors = []
    
    def test_concurrent_rule_registration(self):
        """Test concurrent rule registration from multiple threads."""
        def register_rules(thread_id):
            try:
                for i in range(10):
                    def dummy_rule(context):
                        return []
                    
                    rule_id = f"rule_{thread_id}_{i}"
                    metadata = RuleMetadata(
                        rule_id=rule_id,
                        name=f"Rule {thread_id}-{i}",
                        description=f"Test rule {thread_id}-{i}",
                        layer="L1",
                        domains=["test"]
                    )
                    
                    self.registry.register_rule(dummy_rule, metadata)
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
                    
            except Exception as e:
                self.errors.append(str(e))
        
        # Create and start multiple threads
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=register_rules, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"
        assert len(self.registry._rules) == 50  # 5 threads * 10 rules each
    
    def test_concurrent_rule_access(self):
        """Test concurrent rule access and filtering."""
        # Pre-register some rules
        for i in range(20):
            def dummy_rule(context):
                return []
            
            metadata = RuleMetadata(
                rule_id=f"rule_{i}",
                name=f"Rule {i}",
                description=f"Test rule {i}",
                layer="L1" if i % 2 == 0 else "L2",
                domains=["test"]
            )
            
            self.registry.register_rule(dummy_rule, metadata)
        
        def access_rules():
            try:
                # Perform various read operations
                all_rules = self.registry.list_rules()
                l1_rules = self.registry.list_rules(layer="L1")
                stats = self.registry.get_rule_statistics()
                
                # Basic consistency checks
                assert len(all_rules) == 20
                assert len(l1_rules) == 10
                assert stats["total_rules"] == 20
                
            except Exception as e:
                self.errors.append(str(e))
        
        # Create and start multiple reader threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=access_rules)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that no errors occurred
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"


if __name__ == "__main__":
    pytest.main([__file__])

