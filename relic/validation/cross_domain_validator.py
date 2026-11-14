"""
Cross-Domain Variable and Unit Consistency Validator for RELIC

This module provides automated validation of variable names and units across
different domains (atmosphere, hydrosphere, pedosphere, biosphere, etc.) to
ensure consistency and prevent unit conversion errors.

The validator checks:
1. Variable name consistency across domain interfaces
2. Unit consistency for coupled variables
3. Mass/energy conservation at domain boundaries
4. Compliance with CF-1.8 conventions
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from torch import Tensor

from relic.core.state import ConservationSummary, StateManager
from .framework import (
    ValidationContext, ValidationReportModel
)
from .cross_domain_coupling_ops import (
    check_cnp_stoichiometry_impl,
    check_cross_domain_flux_consistency_impl,
    check_element_balance_impl,
    check_energy_conservation_impl,
    check_stress_factor_bounds_impl,
    check_water_conservation_impl,
    from_state_manager_summary_impl,
    validate_cnp_cycle_impl,
    validate_coupling_conservation_impl,
    validate_step_conservation_impl,
)
from .cross_domain_interface_ops import (
    validate_hydrosphere_interfaces_impl,
    validate_mass_conservation_impl,
    validate_variable_consistency_impl,
)
from .cross_domain_registry_ops import (
    infer_domain_from_name as infer_domain_from_name_impl,
    initialize_interface_contracts as initialize_interface_contracts_impl,
    initialize_standard_variables as initialize_standard_variables_impl,
    integrate_hydrosphere_mappings as integrate_hydrosphere_mappings_impl,
)
from .cross_domain_report_ops import (
    generate_validation_report_impl,
    run_unified_validation_impl,
    to_json_impl,
    to_markdown_impl,
    to_report_impl,
)


class DomainType(Enum):
    """Scientific domains in RELIC."""
    ATMOSPHERE = "atmosphere"
    HYDROSPHERE = "hydrosphere"
    PEDOSPHERE = "pedosphere"
    BIOSPHERE = "biosphere"
    ANTHROPOSPHERE = "anthroposphere"
    TRANSPORT = "transport"


@dataclass
class VariableSpec:
    """Specification for a scientific variable."""
    name: str
    units: str
    domain: DomainType
    cf_standard_name: Optional[str] = None
    bounds: Optional[Tuple[float, float]] = None
    description: Optional[str] = None
    aliases: List[str] = field(default_factory=list)


@dataclass
class InterfaceContract:
    """Contract for inter-domain coupling."""
    source_domain: DomainType
    target_domain: DomainType
    variables: List[VariableSpec]
    conversion_factors: Dict[str, float] = field(default_factory=dict)
    validation_rules: List[str] = field(default_factory=list)


# Import unified unit converter instead of local implementation
from relic.data.standards.unit_converter import unit_converter


class CrossDomainValidator:
    """
    Validates consistency of variables and units across scientific domains.
    
    This validator ensures that:
    1. Variable names follow consistent conventions
    2. Units are consistent at domain interfaces
    3. Conservation laws are maintained
    4. CF-1.8 conventions are followed
    """
    
    def __init__(self, state_manager: Optional[StateManager] = None):
        """
        Initialize the cross-domain validator.
        
        Parameters
        ----------
        state_manager : StateManager, optional
            State manager containing variable specifications
        """
        self.state_manager = state_manager
        self.ValidationContext = ValidationContext
        self.unit_converter = unit_converter
        self.variable_registry: Dict[str, VariableSpec] = {}
        self.interface_contracts: List[InterfaceContract] = []
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
        # Initialize standard variable specifications
        self._initialize_standard_variables()
        self._initialize_interface_contracts()
        self._integrate_hydrosphere_mappings()
    
    def _initialize_standard_variables(self):
        """
        Initialize registry from runtime schema and CF metadata only.

        This method intentionally avoids hardcoded variable defaults to keep
        StateManager schema as the primary source of truth.
        """
        initialize_standard_variables_impl(self, DomainType, VariableSpec)

    def _infer_domain_from_name(self, var_name: str) -> DomainType:
        """Infer scientific domain from variable name heuristics."""
        return infer_domain_from_name_impl(var_name, DomainType)
    
    def _initialize_interface_contracts(self):
        """Initialize standard interface contracts between domains."""
        initialize_interface_contracts_impl(
            self,
            DomainType,
            InterfaceContract,
            VariableSpec,
        )
    
    def _integrate_hydrosphere_mappings(self):
        """Integrate hydrosphere interface mappings into validator."""
        integrate_hydrosphere_mappings_impl(self, DomainType, VariableSpec)
    
    def register_variable(self, var_spec: VariableSpec) -> None:
        """Register a variable specification."""
        self.variable_registry[var_spec.name] = var_spec
        # Also register aliases
        for alias in var_spec.aliases:
            self.variable_registry[alias] = var_spec
    
    def register_interface_contract(self, contract: InterfaceContract) -> None:
        """Register an interface contract between domains."""
        self.interface_contracts.append(contract)

    @staticmethod
    def _framework_level_name(level: Any) -> str:
        """Normalize framework validation level to an uppercase string."""
        return str(getattr(level, "value", level)).upper()

    def _merge_framework_results(self, results: List[Any]) -> None:
        """
        Merge ValidationResultModel-like outputs into legacy error/warning lists.

        Parameters
        ----------
        results : List[Any]
            Validation results returned by rule functions.
        """
        for item in results:
            level_name = self._framework_level_name(getattr(item, "level", "INFO"))
            message = str(getattr(item, "message", ""))
            if not message:
                continue
            if level_name in {"ERROR", "CRITICAL"}:
                self.validation_errors.append(message)
            elif level_name == "WARNING":
                self.validation_warnings.append(message)
    
    def validate_variable_consistency(self, 
                                     source_vars: Dict[str, Tuple[Tensor, str]],
                                     target_vars: Dict[str, Tuple[Tensor, str]],
                                     source_domain: DomainType,
                                     target_domain: DomainType) -> bool:
        """
        Validate consistency between source and target variables.
        
        Parameters
        ----------
        source_vars : Dict[str, Tuple[Tensor, str]]
            Source variables with (value, unit) tuples
        target_vars : Dict[str, Tuple[Tensor, str]]
            Target variables with (value, unit) tuples
        source_domain : DomainType
            Source domain type
        target_domain : DomainType
            Target domain type
            
        Returns
        -------
        bool
            True if validation passes, False otherwise
        """
        return validate_variable_consistency_impl(
            self,
            source_vars,
            target_vars,
            source_domain,
            target_domain,
        )
    
    def validate_mass_conservation(self,
                                  inputs: Dict[str, Tuple[Tensor, str]],
                                  outputs: Dict[str, Tuple[Tensor, str]],
                                  storage_change: Optional[Tuple[Tensor, str]] = None,
                                  element: str = "water") -> bool:
        """
        Validate mass conservation across domain boundary.
        
        Parameters
        ----------
        inputs : Dict[str, Tuple[Tensor, str]]
            Input fluxes with units
        outputs : Dict[str, Tuple[Tensor, str]]
            Output fluxes with units
        storage_change : Tuple[Tensor, str], optional
            Change in storage with units
        element : str
            Element to check conservation for ("water", "carbon", "nitrogen", "phosphorus");
            the relative closure tolerance is
            validation.conservation.<element>_conservation_tolerance.
            
        Returns
        -------
        bool
            True if mass is conserved within tolerance
        """
        return validate_mass_conservation_impl(
            self,
            inputs,
            outputs,
            storage_change,
            element,
        )
    
    def validate_hydrosphere_interfaces(self, 
                                      functor_inputs: List[str], 
                                      functor_outputs: List[str],
                                      functor_name: str = "unnamed") -> bool:
        """
        Validate hydrosphere functor interface using CF-1.8 mappings.
        
        Parameters
        ----------
        functor_inputs : List[str]
            Input variable names from functor
        functor_outputs : List[str] 
            Output variable names from functor
        functor_name : str
            Name of functor for error reporting
            
        Returns
        -------
        bool
            True if the declared interface and bounded variable metadata are valid
        """
        return validate_hydrosphere_interfaces_impl(
            self,
            functor_inputs,
            functor_outputs,
            functor_name,
        )
    
    def validate_coupling_conservation(self, coupling_name: str, 
                                      input_state: Dict, output_fluxes: Dict) -> Dict[str, Any]:
        """Validate coupling conservation using extracted helper logic."""
        return validate_coupling_conservation_impl(
            self,
            coupling_name,
            input_state,
            output_fluxes,
        )
    
    def _check_water_conservation(self, coupling_name: str, 
                                 input_state: Dict, output_fluxes: Dict) -> Dict:
        """Check water mass conservation."""
        return check_water_conservation_impl(coupling_name, input_state, output_fluxes)
    
    def _check_energy_conservation(self, input_state: Dict, output_fluxes: Dict) -> Dict:
        """Check energy balance conservation."""
        return check_energy_conservation_impl(input_state, output_fluxes)
    
    def _check_stress_factor_bounds(self, output_fluxes: Dict) -> Dict:
        """Check that all stress factors are within [0, 1]."""
        return check_stress_factor_bounds_impl(output_fluxes)
    
    def validate_step_conservation(self, coupling_name: str, input_state: Dict,
                                 output_fluxes: Dict) -> Dict[str, Any]:
        """
        Unified step-level conservation validation for Simulator integration.
        
        This method provides a simplified interface for the Simulator to validate
        conservation at each time step, integrating with existing coupling validation.
        
        Parameters
        ----------
        coupling_name : str
            Name of the coupling to validate
        input_state : Dict
            Current state variables
        output_fluxes : Dict
            Computed fluxes from coupling. The water and energy budgets compare their relative
            residual with validation.conservation.water_conservation_tolerance and
            validation.conservation.energy_conservation_tolerance.
            
        Returns
        -------
        Dict[str, Any]
            Step validation results with unified format
        """
        return validate_step_conservation_impl(
            self,
            coupling_name,
            input_state,
            output_fluxes,
        )
    
    def from_state_manager_summary(self, summary: 'ConservationSummary') -> Dict[str, Any]:
        """
        Convert StateManager ConservationSummary to CrossDomainValidator format.
        
        This method bridges the StateManager conservation reporting with the
        CrossDomainValidator validation structure for unified reporting.
        
        Parameters
        ----------
        summary : ConservationSummary
            Conservation summary from StateManager
            
        Returns
        -------
        Dict[str, Any]
            Converted validation structure
        """
        return from_state_manager_summary_impl(summary)
    
    def validate_cnp_cycle(self, 
                          input_state: Dict, 
                          output_fluxes: Dict,
                          check_stoichiometry: bool = True) -> Dict[str, Any]:
        """
        Validate coupled C/N/P cycle conservation across domains.
        
        This method performs integrated validation of carbon, nitrogen, and phosphorus
        cycles to ensure mass conservation and chemical stoichiometry consistency.
        
        Mathematical Expressions:
        Mass balance: sum(inputs_i) = sum(outputs_i) + d(storage)/dt for each element
        Ratio checks: low * (1 - tol) <= X:Y <= high * (1 + tol) for C:N, C:P and N:P, with
        the ranges and tol read from validation/conservation.yaml
        (validation.conservation.stoichiometry.*); there is no per-call tolerance.
        
        Parameters
        ----------
        input_state : Dict
            Current state variables containing C/N/P pools
        output_fluxes : Dict
            Output fluxes for all three elements
        check_stoichiometry : bool
            Whether to validate C:N:P stoichiometric ratios
            
        Returns
        -------
        Dict[str, Any]
            Validation results with keys:
            - 'passed': bool, overall validation status
            - 'carbon_balance': Dict, carbon conservation results
            - 'nitrogen_balance': Dict, nitrogen conservation results  
            - 'phosphorus_balance': Dict, phosphorus conservation results
            - 'stoichiometry': Dict, C:N:P ratio validation results
            - 'cross_domain_fluxes': Dict, flux consistency across domains
            - 'violations': List[str], detailed violation messages
            
        References
        ----------
        - Redfield et al. (1963). The influence of organisms on seawater composition.
        - Sterner & Elser (2002). Ecological Stoichiometry: C:N:P ratios.
        """
        return validate_cnp_cycle_impl(
            self,
            input_state,
            output_fluxes,
            check_stoichiometry=check_stoichiometry,
        )
    
    def _check_element_balance(self, element: str, 
                               input_state: Dict, 
                               output_fluxes: Dict) -> Dict[str, Any]:
        """Check mass balance for a single element."""
        return check_element_balance_impl(element, input_state, output_fluxes)
    
    def _check_cnp_stoichiometry(self, input_state: Dict,
                                  output_fluxes: Dict) -> Dict[str, Any]:
        """Check C:N:P stoichiometric ratios."""
        return check_cnp_stoichiometry_impl(input_state, output_fluxes)
    
    def _check_cross_domain_flux_consistency(self, output_fluxes: Dict) -> Dict[str, Any]:
        """Check consistency of fluxes across domain boundaries."""
        return check_cross_domain_flux_consistency_impl(output_fluxes)
    
    def _find_contract(self, source: DomainType, target: DomainType) -> Optional[InterfaceContract]:
        """Find interface contract between two domains."""
        for contract in self.interface_contracts:
            if contract.source_domain == source and contract.target_domain == target:
                return contract
            # Check reverse direction
            if contract.source_domain == target and contract.target_domain == source:
                return contract
        return None
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        return generate_validation_report_impl(self)
    
    def to_report(self, include_metadata: bool = True) -> Optional['ValidationReportModel']:
        """
        Generate unified validation report using new framework.
        
        This method provides access to the new ValidationReportModel format
        while maintaining API continuity with existing code.
        
        Parameters
        ----------
        include_metadata : bool
            Whether to include execution metadata in report
            
        Returns
        -------
        ValidationReportModel or None
            Unified validation report, or None if framework not available
        """
        return to_report_impl(self, include_metadata=include_metadata)
    
    def to_json(self, indent: int = 2, include_metadata: bool = True) -> Optional[str]:
        """
        Export validation results to JSON format using unified framework.
        
        Parameters
        ----------
        indent : int
            JSON indentation level
        include_metadata : bool
            Whether to include execution metadata
            
        Returns
        -------
        str or None
            JSON representation, or None if framework not available
        """
        return to_json_impl(self, indent=indent, include_metadata=include_metadata)
    
    def to_markdown(self, include_details: bool = True) -> Optional[str]:
        """
        Export validation results to Markdown format using unified framework.
        
        Parameters
        ----------
        include_details : bool
            Whether to include detailed diagnostic information
            
        Returns
        -------
        str or None
            Markdown representation, or None if framework not available
        """
        return to_markdown_impl(self, include_details=include_details)
    
    def run_unified_validation(self, 
                              layers: Optional[List[str]] = None,
                              domains: Optional[List[str]] = None) -> Optional['ValidationReportModel']:
        """
        Run validation using unified framework instead of prior methods.
        
        This method provides direct access to the new validation engine
        for users who want to migrate to the unified framework.
        
        Parameters
        ----------
        layers : List[str], optional
            Validation layers to execute (default: ["L1", "L2"])
        domains : List[str], optional  
            Scientific domains to include (default: cross-domain relevant)
            
        Returns
        -------
        ValidationReportModel or None
            Unified validation report, or None if framework not available
        """
        return run_unified_validation_impl(self, layers=layers, domains=domains)
    
# Export main classes
__all__ = [
    "CrossDomainValidator",
    "VariableSpec",
    "InterfaceContract",
    "DomainType"
]

