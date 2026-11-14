"""
Validation Rules Collection

This package contains all validation rules organized by category and domain.
Rules are automatically registered with the global validation registry when
imported, enabling dynamic discovery and execution.

Rule Categories:
    - cross_domain: Cross-domain variable and unit consistency
    - stochastic_scores: Probabilistic scoring rules (CRPS, PIT, etc.)
    - parameters: Parameter specification and distribution validation
    - conservation: Mass and energy conservation laws
    - coupling_contracts: Inter-domain coupling interface validation

Usage:
    # Import all rules to register them
    from relic.validation.framework.rules import *
    
    # Or import specific rule categories
    from relic.validation.framework.rules.cross_domain import *
    from relic.validation.framework.rules.stochastic_scores import *

Architecture:
    Each rule module contains functions decorated with @validated_rule
    that are automatically registered in the global ValidationRegistry.
    Rules follow the pattern:
    
    @validated_rule("rule_id", "layer", domains=[], tags=[], ...)
    def validation_function(context: ValidationContext) -> ValidationResults:
        # Validation logic
        return [ValidationResultModel(...)]

References:
    - RELIC validation framework specification
    - Unified validation framework design document
"""

# Import all rule modules to ensure registration
from . import cross_domain
from . import stochastic_scores
from . import parameters
from . import conservation
from . import coupling_contracts
from . import dec_complex
from . import hodge
from . import iphs
from . import chain_map
from . import state_domain
from . import observation_diagnostic
from . import entropy_step
from . import port_power
from . import port_matching
from . import overflow_routing
from . import equivalence_certificate
from . import reaction_network
from . import stochastic_sdae

# Collect all exported validation rules
__all__ = []

def _reexport_module_symbols(module) -> None:
    """Re-export symbols declared in a rule submodule __all__."""
    if not hasattr(module, "__all__"):
        return
    for symbol in getattr(module, "__all__", []):
        if hasattr(module, symbol):
            globals()[symbol] = getattr(module, symbol)
            __all__.append(symbol)


_reexport_module_symbols(cross_domain)
_reexport_module_symbols(stochastic_scores)
_reexport_module_symbols(parameters)
_reexport_module_symbols(conservation)
_reexport_module_symbols(coupling_contracts)
_reexport_module_symbols(dec_complex)
_reexport_module_symbols(hodge)
_reexport_module_symbols(iphs)
_reexport_module_symbols(chain_map)
_reexport_module_symbols(state_domain)
_reexport_module_symbols(observation_diagnostic)
_reexport_module_symbols(entropy_step)
_reexport_module_symbols(port_power)
_reexport_module_symbols(port_matching)
_reexport_module_symbols(overflow_routing)
_reexport_module_symbols(equivalence_certificate)
_reexport_module_symbols(reaction_network)
_reexport_module_symbols(stochastic_sdae)

# Rule count for diagnostics
def get_registered_rule_count():
    """Get count of registered validation rules."""
    from ..registry import get_global_registry
    registry = get_global_registry()
    return len(registry._rules)

# Rule discovery utilities
def list_available_rules_by_category():
    """List available validation rules organized by category."""
    from ..registry import get_global_registry
    registry = get_global_registry()
    
    categories = {}
    for rule_id, metadata in registry._metadata.items():
        # Categorize using the fields owned by RuleMetadata.
        tags = set(metadata.tags)
        domains = set(metadata.domains)
        if "cross_domain" in tags or "cross_domain" in domains:
            category = 'cross_domain'
        elif "stochastic" in tags or "stochastic" in domains:
            category = 'stochastic_scores'
        elif "parameters" in tags or "parameters" in domains:
            category = 'parameters'
        elif "conservation" in tags or "conservation" in domains:
            category = 'conservation'
        elif "coupling" in tags or "coupling" in domains:
            category = 'coupling_contracts'
        else:
            category = 'other'
            
        if category not in categories:
            categories[category] = []
        categories[category].append({
            'rule_id': rule_id,
            'name': metadata.name,
            'layer': metadata.layer,
            'domains': metadata.domains
        })
    
    return categories
