"""
Validation Rule Registry

This module provides a centralized registry for validation rules, enabling
dynamic rule registration, discovery, and metadata management across all
validation layers and domains.

Features:
    - Dynamic rule registration with metadata
    - Rule discovery by layer, domain, and tags
    - Dependency tracking and ordering
    - Rule versioning and overrides
    - Metadata validation and standardization

Mathematical Framework:
    Registry R contains rules r with metadata M(r):
    R = {(r, M(r)) : r in ValidationRules}
    Query(R, filters) = {r : (r, M(r)) in R  matches(M(r), filters)}
    
References:
    - Registry pattern: Gang of Four Design Patterns
    - Plugin architecture principles
"""

from typing import Dict, List, Set, Optional, Callable, Any, Union
from dataclasses import dataclass, field
import logging
import inspect

from .layers import LAYER_METADATA

logger = logging.getLogger(__name__)


@dataclass
class RuleMetadata:
    """
    Metadata for a validation rule.
    
    Contains all information needed to categorize, execute, and document
    a validation rule within the unified framework.
    
    Fields:
        rule_id: Unique identifier for the rule
        name: Human-readable name
        description: Detailed description of what the rule validates
        layer: Validation layer (L1/L2/L3/L4)
        domains: Applicable scientific domains
        tags: Categorization tags
        inputs_schema: Expected input parameters/schema
        outputs_schema: Expected output structure
        dependencies: Other rules that must run first
        cf_requirements: bounded CF variable-metadata requirements
        units_contract: Expected units for inputs/outputs
        references: Scientific literature references
        version: Rule version for consistency
        author: Rule author/maintainer
        enabled: Whether rule is currently active
    """
    rule_id: str
    name: str
    description: str
    layer: str
    domains: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Schema information
    inputs_schema: Optional[Dict[str, Any]] = None
    outputs_schema: Optional[Dict[str, Any]] = None
    
    # Dependencies and ordering
    dependencies: List[str] = field(default_factory=list)
    execution_order: int = 0
    
    # Scientific metadata
    cf_requirements: List[str] = field(default_factory=list)
    units_contract: Dict[str, str] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    
    # Version and maintenance
    version: str = "1.0.0"
    author: Optional[str] = None
    enabled: bool = True
    
    # Execution hints
    parallel_safe: bool = True
    requires_state_manager: bool = False
    required_data: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize metadata after creation."""
        # Normalize tags and domains to lowercase
        self.tags = [tag.lower() if isinstance(tag, str) else tag for tag in self.tags]
        self.domains = [domain.lower() if isinstance(domain, str) else domain for domain in self.domains]
        
        # Validate layer
        if self.layer not in LAYER_METADATA:
            raise ValueError(f"Invalid validation layer: {self.layer}")
        
        # Add domain-specific tags automatically
        for domain in self.domains:
            if domain not in self.tags:
                self.tags.append(domain)
    
    def matches_filter(self, **filters) -> bool:
        """Check if this rule metadata matches the given filters."""
        for key, value in filters.items():
            if key == 'layer' and self.layer != value:
                return False
            elif key == 'domains' and not any(d in self.domains for d in value):
                return False
            elif key == 'tags' and not any(t in self.tags for t in value):
                return False
            elif key == 'enabled' and value is not None and self.enabled != value:
                return False
            elif key not in ['layer', 'domains', 'tags', 'enabled'] and hasattr(self, key) and getattr(self, key) != value:
                return False
        return True


class ValidationRegistry:
    """
    Central registry for validation rules.
    
    Maintains a registry of all validation rules with their metadata,
    enabling dynamic discovery, execution ordering, and rule management.
    
    The registry supports:
    - Rule registration with comprehensive metadata
    - Query and filtering by multiple criteria
    - Dependency resolution and execution ordering
    - Rule versioning and overrides
    - Bulk operations and batch registration
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._rules: Dict[str, Callable] = {}
        self._metadata: Dict[str, RuleMetadata] = {}
        self._rule_groups: Dict[str, Set[str]] = {}  # Group name -> rule IDs
        self._dependency_cache: Optional[List[str]] = None
    
    def register_rule(self, 
                     rule_function: Callable,
                     metadata: RuleMetadata,
                     overwrite: bool = False) -> None:
        """
        Register a validation rule with metadata.
        
        Parameters:
            rule_function: The validation rule function
            metadata: Rule metadata
            overwrite: Whether to overwrite existing rules with same ID
            
        Raises:
            ValueError: If rule already exists and overwrite=False
        """
        rule_id = metadata.rule_id
        
        if rule_id in self._rules and not overwrite:
            raise ValueError(f"Rule '{rule_id}' already exists. Use overwrite=True to replace.")
        
        # Validate rule function signature
        self._validate_rule_function(rule_function, metadata)
        
        # Store rule and metadata
        self._rules[rule_id] = rule_function
        self._metadata[rule_id] = metadata
        
        # Update rule groups
        for tag in metadata.tags:
            if tag not in self._rule_groups:
                self._rule_groups[tag] = set()
            self._rule_groups[tag].add(rule_id)
        
        # Clear dependency cache
        self._dependency_cache = None
        
        logger.debug(f"Registered validation rule: {rule_id} (layer={metadata.layer})")
    
    def unregister_rule(self, rule_id: str) -> bool:
        """
        Unregister a validation rule.
        
        Parameters:
            rule_id: Rule identifier to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        if rule_id not in self._rules:
            return False
        
        # Get metadata before removal
        metadata = self._metadata[rule_id]
        
        # Remove from main registry
        del self._rules[rule_id]
        del self._metadata[rule_id]
        
        # Remove from rule groups
        for tag in metadata.tags:
            if tag in self._rule_groups:
                self._rule_groups[tag].discard(rule_id)
                if not self._rule_groups[tag]:  # Remove empty groups
                    del self._rule_groups[tag]
        
        # Clear dependency cache
        self._dependency_cache = None
        
        logger.debug(f"Unregistered validation rule: {rule_id}")
        return True
    
    def get_rule(self, rule_id: str) -> Optional[Callable]:
        """Get a rule function by ID."""
        return self._rules.get(rule_id)
    
    def get_metadata(self, rule_id: str) -> Optional[RuleMetadata]:
        """Get rule metadata by ID."""
        return self._metadata.get(rule_id)
    
    def list_rules(self, **filters) -> List[str]:
        """
        List rule IDs matching the given filters.
        
        Parameters:
            **filters: Filter criteria (layer, domains, tags, enabled, etc.)
            
        Returns:
            List of rule IDs that match all filter criteria
        """
        matching_rules = []
        
        for rule_id, metadata in self._metadata.items():
            if metadata.matches_filter(**filters):
                matching_rules.append(rule_id)
        
        return matching_rules
    
    def get_rules_by_layer(self, layer: str, enabled_only: bool = True) -> Dict[str, Callable]:
        """Get all rules for a specific validation layer."""
        filters = {'layer': layer}
        if enabled_only:
            filters['enabled'] = True
        
        rule_ids = self.list_rules(**filters)
        return {rule_id: self._rules[rule_id] for rule_id in rule_ids}
    
    def get_rules_by_domain(self, domains: Union[str, List[str]], enabled_only: bool = True) -> Dict[str, Callable]:
        """Get all rules applicable to specific domain(s)."""
        if isinstance(domains, str):
            domains = [domains]
        
        filters = {'domains': [d.lower() for d in domains]}
        if enabled_only:
            filters['enabled'] = True
        
        rule_ids = self.list_rules(**filters)
        return {rule_id: self._rules[rule_id] for rule_id in rule_ids}
    
    def get_rules_by_tags(self, tags: Union[str, List[str]], enabled_only: bool = True) -> Dict[str, Callable]:
        """Get all rules with specific tag(s)."""
        if isinstance(tags, str):
            tags = [tags]
        
        filters = {'tags': [t.lower() for t in tags]}
        if enabled_only:
            filters['enabled'] = True
        
        rule_ids = self.list_rules(**filters)
        return {rule_id: self._rules[rule_id] for rule_id in rule_ids}
    
    def resolve_dependencies(self, rule_ids: List[str]) -> List[str]:
        """
        Resolve rule dependencies and return execution order.
        
        Parameters:
            rule_ids: List of rule IDs to order
            
        Returns:
            List of rule IDs in dependency-resolved execution order
            
        Raises:
            ValueError: If circular dependencies are detected or dependencies are missing
        """
        # Check for missing dependencies first
        for rule_id in rule_ids:
            metadata = self._metadata.get(rule_id)
            if metadata:
                for dep in metadata.dependencies:
                    if dep not in self._rules:
                        raise ValueError(f"Dependency '{dep}' not found for rule '{rule_id}'")
        
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(rule_id: str):
            if rule_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving rule: {rule_id}")
            if rule_id in visited:
                return
            
            temp_visited.add(rule_id)
            
            # Visit dependencies first
            metadata = self._metadata.get(rule_id)
            if metadata:
                for dep in metadata.dependencies:
                    if dep in rule_ids:  # Only consider dependencies in our set
                        visit(dep)
            
            temp_visited.remove(rule_id)
            visited.add(rule_id)
            result.append(rule_id)
        
        for rule_id in rule_ids:
            if rule_id not in visited:
                visit(rule_id)
        
        return result
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a validation rule."""
        if rule_id in self._metadata:
            self._metadata[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a validation rule."""
        if rule_id in self._metadata:
            self._metadata[rule_id].enabled = False
            return True
        return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_rules = len(self._rules)
        enabled_rules = len([r for r in self._metadata.values() if r.enabled])
        
        # Count by layer
        layer_counts = {}
        for metadata in self._metadata.values():
            layer = metadata.layer
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        # Count by domain
        domain_counts = {}
        for metadata in self._metadata.values():
            for domain in metadata.domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "rules_by_layer": layer_counts,
            "rules_by_domain": domain_counts,
            "rule_groups": len(self._rule_groups)
        }
    
    def validate_registry_integrity(self) -> Dict[str, Any]:
        """
        Validate registry integrity and return structured result.
        
        Returns:
            Dictionary with validation results and statistics
        """
        issues = []
        
        # Check for missing dependencies
        for rule_id, metadata in self._metadata.items():
            for dep in metadata.dependencies:
                if dep not in self._rules:
                    issues.append(f"Rule '{rule_id}' depends on missing rule '{dep}'")
        
        # Check for circular dependencies
        try:
            all_rule_ids = list(self._rules.keys())
            self.resolve_dependencies(all_rule_ids)
        except ValueError as e:
            issues.append(str(e))
        
        # Note: Execution order checks removed - dependency resolution is more important
        # than strict execution order uniqueness
        
        # Count rules with dependencies
        rules_with_dependencies = len([m for m in self._metadata.values() if m.dependencies])
        
        return {
            "valid": len(issues) == 0,
            "errors": issues,
            "total_rules": len(self._rules),
            "rules_with_dependencies": rules_with_dependencies
        }
    
    def _validate_rule_function(self, rule_function: Callable, metadata: RuleMetadata) -> None:
        """Validate that rule function has correct signature."""
        signature = inspect.signature(rule_function)
        
        # Check that function accepts ValidationContext parameter
        params = list(signature.parameters.keys())
        if not params or 'context' not in params[0]:
            logger.warning(f"Rule {metadata.rule_id} may not have correct signature. Expected parameter 'context'.")
        
        # Store function metadata for execution hints
        if hasattr(rule_function, '_required_data'):
            metadata.required_data = getattr(rule_function, '_required_data')
        
        if hasattr(rule_function, '_fail_fast'):
            # This rule should cause fail-fast behavior in L2
            pass
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry configuration for serialization."""
        return {
            "rules": {
                rule_id: {
                    "metadata": {
                        "rule_id": meta.rule_id,
                        "name": meta.name,
                        "description": meta.description,
                        "layer": meta.layer,
                        "domains": meta.domains,
                        "tags": meta.tags,
                        "version": meta.version,
                        "enabled": meta.enabled
                    }
                }
                for rule_id, meta in self._metadata.items()
            },
            "statistics": self.get_rule_statistics()
        }
    
    def clear_registry(self) -> None:
        """Clear all rules from registry (for testing)."""
        self._rules.clear()
        self._metadata.clear()
        self._rule_groups.clear()
        self._dependency_cache = None
        logger.warning("Validation registry cleared")


# Global registry instance (singleton pattern)
_global_registry = None


def get_global_registry() -> ValidationRegistry:
    """Get the global validation registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ValidationRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = ValidationRegistry()
