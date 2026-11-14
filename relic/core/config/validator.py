"""
Configuration Validation System - Comprehensive Configuration Checking

This module provides comprehensive validation capabilities for RELIC
configurations including scientific parameter validation, physical
constraint checking, and consistency verification across different
configuration sections.

The validation system ensures that configurations are scientifically
sound, numerically stable, and internally consistent before simulation
execution begins.

Architecture:
    - Multi-level validation (syntactic, semantic, scientific)
    - Pluggable validation rules with priority system
    - Detailed error reporting with suggestions
    - Performance-optimized validation caching
    - Integration with scientific parameter databases

Validation Levels:
    1. Syntactic: Type checking, required fields, format validation
    2. Semantic: Cross-reference validation, consistency checking  
    3. Scientific: Physical bounds, conservation laws, stability conditions
    4. Performance: Computational efficiency warnings and optimizations

References:
    - Scientific parameter validation standards
    - Numerical stability analysis methods
    - Configuration validation design patterns
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Parameter management
from ...parameters import get_global_manager
from ...utils.time_constants import SECONDS_PER_DAY
from .models import ParameterValidationLevel as ConfigValidationLevel

# Import configuration models
from .models import DeviceType, IntegrationMethod, RELICConfig

logger = logging.getLogger(__name__)


class ConfigValidationStage(Enum):
    """Levels of configuration validation."""
    SYNTACTIC = "syntactic"      # Basic type and structure checking
    SEMANTIC = "semantic"        # Cross-reference and consistency checking
    SCIENTIFIC = "scientific"    # Physical constraints and scientific validity
    PERFORMANCE = "performance"  # Performance and efficiency recommendations


class ValidationSeverity(Enum):
    """Severity levels for validation messages."""
    INFO = "info"          # Informational messages
    WARNING = "warning"    # Non-critical issues
    ERROR = "error"        # Critical issues that prevent execution
    CRITICAL = "critical"  # Severe issues that could cause system failure


@dataclass
class ValidationMessage:
    """
    A single validation message with context and suggestions.
    
    Provides detailed information about validation issues including
    location, severity, description, and suggested fixes.
    """
    level: ConfigValidationStage
    severity: ValidationSeverity
    message: str
    location: str = ""
    parameter_name: str = ""
    current_value: Any = None
    suggested_value: Any = None
    suggested_action: str = ""
    reference_url: str = ""
    
    def __str__(self) -> str:
        """Format validation message for display."""
        parts = [f"[{self.severity.value.upper()}]"]
        if self.location:
            parts.append(f"({self.location})")
        parts.append(self.message)
        
        if self.current_value is not None:
            parts.append(f"Current: {self.current_value}")
        
        if self.suggested_value is not None:
            parts.append(f"Suggested: {self.suggested_value}")
            
        if self.suggested_action:
            parts.append(f"Action: {self.suggested_action}")
            
        return " ".join(parts)


@dataclass
class ConfigValidationResult:
    """
    Complete validation result with all messages and summary statistics.
    
    Aggregates all validation messages and provides summary information
    for quick assessment of configuration validity.
    """
    is_valid: bool = True
    messages: List[ValidationMessage] = field(default_factory=list)
    validation_time: float = 0.0
    
    # Summary statistics
    num_errors: int = 0
    num_warnings: int = 0
    num_info: int = 0
    num_critical: int = 0
    
    def __post_init__(self) -> None:
        """Compute summary statistics from messages."""
        self.update_statistics()
    
    def update_statistics(self) -> None:
        """Update summary statistics from messages."""
        self.num_errors = sum(1 for msg in self.messages if msg.severity == ValidationSeverity.ERROR)
        self.num_warnings = sum(1 for msg in self.messages if msg.severity == ValidationSeverity.WARNING)
        self.num_info = sum(1 for msg in self.messages if msg.severity == ValidationSeverity.INFO)
        self.num_critical = sum(1 for msg in self.messages if msg.severity == ValidationSeverity.CRITICAL)
        
        # Configuration is invalid if there are any errors or critical issues
        self.is_valid = (self.num_errors == 0 and self.num_critical == 0)
    
    def add_message(self, message: ValidationMessage) -> None:
        """Add a validation message and update statistics."""
        self.messages.append(message)
        self.update_statistics()
    
    def get_messages_by_severity(self, severity: ValidationSeverity) -> List[ValidationMessage]:
        """Get all messages of a specific severity level."""
        return [msg for msg in self.messages if msg.severity == severity]
    
    def get_messages_by_level(self, level: ConfigValidationStage) -> List[ValidationMessage]:
        """Get all messages from a specific validation level."""
        return [msg for msg in self.messages if msg.level == level]
    
    def summary(self) -> str:
        """Get a summary string of validation results."""
        if self.is_valid:
            return f" Configuration valid ({self.num_warnings} warnings, {self.num_info} info)"
        else:
            critical_str = f", {self.num_critical} critical" if self.num_critical > 0 else ""
            return f" Configuration invalid ({self.num_errors} errors{critical_str}, {self.num_warnings} warnings)"


class ValidationRule:
    """
    Base class for configuration validation rules.
    
    Validation rules are pluggable components that check specific
    aspects of the configuration and report any issues found.
    """
    
    def __init__(self, 
                 rule_id: str,
                 level: ConfigValidationStage,
                 description: str,
                 priority: int = 100) -> None:
        """
        Initialize validation rule.
        
        Parameters:
            rule_id : str
                Unique identifier for this rule
            level : ConfigValidationStage  
                Validation level this rule operates at
            description : str
                Human-readable description of what this rule checks
            priority : int
                Rule priority (lower = higher priority)
        """
        self.rule_id = rule_id
        self.level = level
        self.description = description
        self.priority = priority
    
    def validate(self, config: RELICConfig) -> List[ValidationMessage]:
        """
        Validate configuration against this rule.
        
        Parameters:
            config : RELICConfig
                Configuration object to validate
                
        Returns:
            List[ValidationMessage]: List of validation messages
        """
        raise NotImplementedError("Subclasses must implement validate method")


# ============================================================================
# Concrete Validation Rules
# ============================================================================

class TemperatureBoundsRule(ValidationRule):
    """Validate temperature-related parameters are within physical bounds."""
    
    def __init__(self) -> None:
        super().__init__(
            rule_id="temperature_bounds",
            level=ConfigValidationStage.SCIENTIFIC,
            description="Check temperature parameters are within physical bounds",
            priority=10
        )
    
    def validate(self, config: RELICConfig) -> List[ValidationMessage]:
        """Validate temperature bounds in configuration."""
        messages = []
        # The Celsius offset is a physical constant; a failed lookup fails this
        # rule (ConfigValidator records it as an ERROR) instead of skipping the
        # absolute-zero check.
        celsius_offset = get_global_manager().get(
            "relic.physical.constants.temperature.celsius_to_kelvin",
            units="K",
            expected_type=float,
            required=True,
        )
        abs_zero_c = -float(celsius_offset)
        
        # Check if temperature parameters exist in overrides
        if config.parameters and config.parameters.parameter_overrides:
            for param_name, value in config.parameters.parameter_overrides.items():
                if 'temperature' in param_name.lower() or 'temp' in param_name.lower():
                    # Check for absolute zero violations
                    if isinstance(value, (int, float)):
                        if 'kelvin' in param_name.lower() or '_k' in param_name.lower():
                            if value < 0:
                                messages.append(ValidationMessage(
                                    level=self.level,
                                    severity=ValidationSeverity.ERROR,
                                    message="Temperature cannot be negative in Kelvin",
                                    location="parameters.parameter_overrides",
                                    parameter_name=param_name,
                                    current_value=value,
                                    suggested_value=max(0, value),
                                    suggested_action="Ensure temperature is positive in Kelvin scale"
                                ))
                        elif 'celsius' in param_name.lower() or '_c' in param_name.lower():
                            if value < abs_zero_c:
                                messages.append(ValidationMessage(
                                    level=self.level,
                                    severity=ValidationSeverity.ERROR,
                                    message="Temperature below absolute zero",
                                    location="parameters.parameter_overrides",
                                    parameter_name=param_name,
                                    current_value=value,
                                    suggested_value=max(abs_zero_c, value),
                                    suggested_action=(
                                        f"Temperature cannot be below {abs_zero_c}degC"
                                    ),
                                ))
        
        return messages


class TimeStepStabilityRule(ValidationRule):
    """Validate time step stability for numerical integration."""
    
    def __init__(self) -> None:
        super().__init__(
            rule_id="timestep_stability",
            level=ConfigValidationStage.SCIENTIFIC,
            description="Check time step stability conditions",
            priority=20
        )
    
    def validate(self, config: RELICConfig) -> List[ValidationMessage]:
        """Validate time step stability conditions."""
        messages = []
        sim = config.simulation
        
        # CFL condition for spatial discretization
        if sim.spatial_resolution > 0:
            # Estimate maximum stable time step based on diffusion
            if config.physics.molecular_diffusion:
                # Diffusion stability: dt < dx^2/(2D)
                # Assume typical diffusivity D ~ 1e-5 m^2/s
                max_dt_diffusion = sim.spatial_resolution**2 / (2 * 1e-5)
                if sim.time_step > max_dt_diffusion:
                    messages.append(ValidationMessage(
                        level=self.level,
                        severity=ValidationSeverity.WARNING,
                        message="Time step may violate diffusion stability",
                        location="simulation.time_step",
                        current_value=sim.time_step,
                        suggested_value=max_dt_diffusion * 0.5,
                        suggested_action="Consider reducing time step for diffusion stability"
                    ))
            
            # Advection stability if enabled
            if config.physics.advection:
                # CFL condition: dt < dx/v
                # Assume typical velocity v ~ 1 m/s
                max_dt_advection = sim.spatial_resolution / 1.0
                if sim.time_step > max_dt_advection:
                    messages.append(ValidationMessage(
                        level=self.level,
                        severity=ValidationSeverity.WARNING,
                        message="Time step may violate CFL condition",
                        location="simulation.time_step",
                        current_value=sim.time_step,
                        suggested_value=max_dt_advection * 0.8,
                        suggested_action="Consider reducing time step for advection stability"
                    ))
        
        return messages


class ConservationConsistencyRule(ValidationRule):
    """Validate conservation law settings are consistent across modules."""
    
    def __init__(self) -> None:
        super().__init__(
            rule_id="conservation_consistency",
            level=ConfigValidationStage.SEMANTIC,
            description="Check conservation settings consistency",
            priority=30
        )
    
    def validate(self, config: RELICConfig) -> List[ValidationMessage]:
        """Validate conservation settings are consistent."""
        messages = []
        
        # Check carbon cycle consistency
        if config.physics.carbon_cycle:
            if not config.physics.photosynthesis_model or config.physics.photosynthesis_model == "none":
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.WARNING,
                    message="Carbon cycle enabled but photosynthesis disabled",
                    location="physics",
                    suggested_action="Enable photosynthesis model for carbon cycle"
                ))
            
            if not config.physics.respiration_model or config.physics.respiration_model == "none":
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.WARNING,
                    message="Carbon cycle enabled but respiration disabled",
                    location="physics",
                    suggested_action="Enable respiration model for carbon cycle"
                ))
        
        # Check nitrogen cycle dependencies
        if config.physics.nitrogen_cycle:
            if not config.physics.carbon_cycle:
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.ERROR,
                    message="Nitrogen cycle requires carbon cycle",
                    location="physics.nitrogen_cycle",
                    suggested_action="Enable carbon cycle for nitrogen cycling"
                ))
            
            if config.physics.decomposition_model == "first_order":
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.INFO,
                    message="Consider MIMICS or CENTURY for nitrogen cycle",
                    location="physics.decomposition_model",
                    suggested_action="Advanced decomposition models better represent N dynamics"
                ))
        
        # Check water balance consistency
        if config.physics.water_balance:
            if config.physics.evapotranspiration_model == "simple" and config.simulation.grid_size_z > 1:
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.WARNING,
                    message="Simple ET model may be insufficient for multi-layer grids",
                    location="physics.evapotranspiration_model",
                    suggested_action="Consider Penman-Monteith for multi-layer simulations"
                ))
        
        # Check strict conservation feature flag
        if config.feature_flags.strict_conservation_enabled:
            if config.parameters.validation_level == ConfigValidationLevel.NONE:
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.WARNING,
                    message="Strict conservation enabled but parameter validation disabled",
                    location="parameters.validation_level",
                    suggested_action="Enable parameter validation for conservation checking"
                ))
        
        return messages


class ParameterBoundsRule(ValidationRule):
    """Validate parameter values are within specified bounds."""
    
    def __init__(self) -> None:
        super().__init__(
            rule_id="parameter_bounds",
            level=ConfigValidationStage.SCIENTIFIC,
            description="Check parameter bound violations",
            priority=50
        )
    
    def validate(self, config: RELICConfig) -> List[ValidationMessage]:
        """Validate parameter bounds."""
        messages = []
        
        # Check simulation time bounds
        sim = config.simulation
        if sim.start_time < 0:
            messages.append(ValidationMessage(
                level=self.level,
                severity=ValidationSeverity.ERROR,
                message="Start time cannot be negative",
                location="simulation.start_time",
                current_value=sim.start_time,
                suggested_value=0.0
            ))
        
        if sim.time_step <= 0:
            messages.append(ValidationMessage(
                level=self.level,
                severity=ValidationSeverity.ERROR,
                message="Time step must be positive",
                location="simulation.time_step",
                current_value=sim.time_step,
                suggested_value=60.0
            ))
        
        # Check spatial bounds
        if sim.spatial_resolution <= 0:
            messages.append(ValidationMessage(
                level=self.level,
                severity=ValidationSeverity.ERROR,
                message="Spatial resolution must be positive",
                location="simulation.spatial_resolution",
                current_value=sim.spatial_resolution,
                suggested_value=1.0
            ))
        
        # Check optimization bounds
        if config.optimization.enable_optimization:
            if config.optimization.learning_rate <= 0 or config.optimization.learning_rate > 1:
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.WARNING,
                    message="Learning rate outside typical range",
                    location="optimization.learning_rate",
                    current_value=config.optimization.learning_rate,
                    suggested_value=0.01,
                    suggested_action="Use learning rate between 0 and 1"
                ))
        
        # Check stochastic bounds
        if config.stochastic.enable_monte_carlo:
            if config.stochastic.num_realizations < 10:
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.WARNING,
                    message="Very few Monte Carlo realizations",
                    location="stochastic.num_realizations",
                    current_value=config.stochastic.num_realizations,
                    suggested_value=100,
                    suggested_action="Increase realizations for robust uncertainty quantification"
                ))
        
        return messages


class MemoryUsageRule(ValidationRule):
    """Validate configuration for memory usage and performance."""
    
    def __init__(self) -> None:
        super().__init__(
            rule_id="memory_usage",
            level=ConfigValidationStage.PERFORMANCE,
            description="Check memory usage and performance implications",
            priority=100
        )
    
    def validate(self, config: RELICConfig) -> List[ValidationMessage]:
        """Validate memory usage implications."""
        messages = []
        sim = config.simulation
        
        # Estimate memory usage for grid
        grid_points = sim.grid_size_x * sim.grid_size_y * sim.grid_size_z
        # Assume ~100 variables, 8 bytes per float64
        estimated_memory_mb = (grid_points * 100 * 8) / (1024 * 1024)
        
        if estimated_memory_mb > 1000:  # > 1GB
            messages.append(ValidationMessage(
                level=self.level,
                severity=ValidationSeverity.WARNING,
                message=f"Large grid may require ~{estimated_memory_mb:.0f}MB memory",
                location="simulation",
                parameter_name="grid_size",
                current_value=f"{sim.grid_size_x}x{sim.grid_size_y}x{sim.grid_size_z}",
                suggested_action="Consider reducing grid size or using sparse representations"
            ))
        
        # Check distributed computing recommendations
        if grid_points > 1_000_000 and not config.computation.enable_distributed:
            messages.append(ValidationMessage(
                level=self.level,
                severity=ValidationSeverity.INFO,
                message="Large domain may benefit from distributed computing",
                location="computation.enable_distributed",
                current_value=False,
                suggested_value=True,
                suggested_action="Enable distributed computing for large domains"
            ))
        
        # Check GPU recommendations
        # A field set from YAML holds the enum value (use_enum_values=True) while an
        # unset field holds the enum member; the enum constructor accepts both.
        if DeviceType(config.computation.device) is DeviceType.CPU and grid_points > 100_000:
            messages.append(ValidationMessage(
                level=self.level,
                severity=ValidationSeverity.INFO,
                message="GPU acceleration recommended for large domains",
                location="computation.device",
                current_value="cpu",
                suggested_value="cuda",
                suggested_action="Consider GPU acceleration for better performance"
            ))
        
        # Check output frequency vs storage
        if config.data.output_frequency == "timestep":
            total_timesteps = (sim.end_time - sim.start_time) / sim.time_step
            output_size_gb = (estimated_memory_mb * total_timesteps) / 1024
            if output_size_gb > 10:
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.WARNING,
                    message=f"Output may require ~{output_size_gb:.1f}GB storage",
                    location="data.output_frequency",
                    current_value="timestep",
                    suggested_value="hourly",
                    suggested_action="Consider reducing output frequency to save storage"
                ))
        
        return messages


class IntegrationMethodconsistencyRule(ValidationRule):
    """Validate integration method consistency with physics."""
    
    def __init__(self, calendar_type: Optional[str] = None) -> None:
        super().__init__(
            rule_id="integration_consistency",
            level=ConfigValidationStage.SEMANTIC,
            description="Check integration method consistency",
            priority=60
        )
        from relic.coupling.scales.temporal.calendar import CalendarHandler

        self.calendar_type = (
            None
            if calendar_type is None
            else CalendarHandler.resolve_simulator_calendar(
                configured_calendar=calendar_type,
                runtime_calendar=None,
                owner_name="IntegrationMethodconsistencyRule",
            )
        )
        supported_year_days = (
            CalendarHandler.days_per_simulator_year("noleap"),
            CalendarHandler.days_per_simulator_year("360_day"),
        )
        self._minimum_supported_year_days = min(supported_year_days)
        self._calendar_year_days = (
            None
            if self.calendar_type is None
            else CalendarHandler.days_per_simulator_year(self.calendar_type)
        )
    
    def validate(self, config: RELICConfig) -> List[ValidationMessage]:
        """Validate integration method consistency."""
        messages = []
        sim = config.simulation
        
        # Check stiff system recommendations
        stiff_processes = []
        if config.physics.atmospheric_chemistry:
            stiff_processes.append("atmospheric_chemistry")
        if config.physics.soil_biogeochemistry:
            stiff_processes.append("soil_biogeochemistry")
        
        integration_method = IntegrationMethod(sim.integration_method)
        if stiff_processes and integration_method is IntegrationMethod.EULER:
            messages.append(ValidationMessage(
                level=self.level,
                severity=ValidationSeverity.WARNING,
                message="Stiff processes may require implicit methods",
                location="simulation.integration_method",
                current_value=integration_method.value,
                suggested_value=IntegrationMethod.CRANK_NICOLSON.value,
                suggested_action=f"Processes {stiff_processes} may be stiff, consider implicit methods"
            ))
        
        # References: CF Conventions 1.8 Section 4.4; RELIC CalendarHandler.
        # Mathematical expression: T = t_end - t_start; long(C) iff T > Y(C).
        # Parameter meanings: C is calendar_type and Y(C) is one calendar year.
        # Units: T and Y(C) are compared in ParameterManager-backed SI seconds.
        # Domain/constraints: C must be explicit when supported calendars can
        # produce different decisions; T is positive by SimulationConfig.
        # Sign convention: elapsed simulation time is positive forward time.
        duration_seconds = sim.end_time - sim.start_time
        seconds_per_day = float(SECONDS_PER_DAY)
        minimum_year_seconds = self._minimum_supported_year_days * seconds_per_day
        if duration_seconds > minimum_year_seconds:
            if self._calendar_year_days is None:
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        "Long-simulation validation requires explicit "
                        "calendar_type"
                    ),
                    location="simulation.calendar_type",
                    current_value=None,
                    suggested_action=(
                        "Pass calendar_type='noleap' or '360_day' to "
                        "ConfigValidator"
                    ),
                ))
            elif duration_seconds > self._calendar_year_days * seconds_per_day:
                messages.append(ValidationMessage(
                    level=self.level,
                    severity=ValidationSeverity.INFO,
                    message="Long simulation may benefit from adaptive time stepping",
                    location="runtime.adaptive_dt_enabled",
                    current_value=None,
                    suggested_value=True,
                    suggested_action=(
                        "Set the RELICSimulator runtime key adaptive_dt_enabled for efficiency"
                    )
                ))
        
        return messages


class ConfigValidator:
    """
    Main configuration validator with pluggable rule system.
    
    Orchestrates validation by applying registered rules in priority order
    and collecting results into comprehensive validation reports.
    """
    
    def __init__(self, calendar_type: Optional[str] = None) -> None:
        """Initialize validator with an optional explicit simulator calendar."""
        self.rules: List[ValidationRule] = []
        self.rule_cache: Dict[str, ConfigValidationResult] = {}
        self.validation_count = 0
        self.calendar_type = calendar_type
        
        # Load default validation rules
        self._load_default_rules()
    
    def _load_default_rules(self) -> None:
        """Load default validation rules for RELIC configurations."""
        # Add all concrete validation rules
        self.add_rule(TemperatureBoundsRule())
        self.add_rule(TimeStepStabilityRule())
        self.add_rule(ConservationConsistencyRule())
        self.add_rule(ParameterBoundsRule())
        self.add_rule(MemoryUsageRule())
        self.add_rule(
            IntegrationMethodconsistencyRule(calendar_type=self.calendar_type)
        )
        
        logger.debug(f"Loaded {len(self.rules)} default validation rules")
    
    def add_rule(self, rule: ValidationRule) -> None:
        """
        Add a validation rule to the validator.
        
        Parameters:
            rule : ValidationRule
                Rule to add to validation set
        """
        # Check for duplicate rule IDs
        existing_ids = [r.rule_id for r in self.rules]
        if rule.rule_id in existing_ids:
            raise ValueError(f"Validation rule {rule.rule_id} already exists")
        
        self.rules.append(rule)
        
        # Sort rules by priority
        self.rules.sort(key=lambda r: r.priority)
        
        logger.debug(f"Added validation rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a validation rule by ID.
        
        Parameters:
            rule_id : str
                ID of rule to remove
                
        Returns:
            bool: True if rule was removed, False if not found
        """
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        
        removed = len(self.rules) < initial_count
        if removed:
            logger.debug(f"Removed validation rule: {rule_id}")
        
        return removed
    
    def validate(self, 
                config: RELICConfig, 
                levels: Optional[List[ConfigValidationStage]] = None,
                use_cache: bool = True) -> ConfigValidationResult:
        """
        Validate configuration using registered rules.
        
        Parameters:
            config : RELICConfig
                Configuration object to validate
            levels : Optional[List[ConfigValidationStage]]
                Validation levels to run, all if None
            use_cache : bool
                Whether to use cached results for performance
                
        Returns:
            ConfigValidationResult: Complete validation results
        """
        start_time = time.time()
        
        self.validation_count += 1
        
        # Initialize result
        result = ConfigValidationResult()
        
        # Determine which levels to validate
        if levels is None:
            levels = list(ConfigValidationStage)
        
        # Apply rules in priority order
        for rule in self.rules:
            if rule.level in levels:
                try:
                    rule_messages = rule.validate(config)
                    for message in rule_messages:
                        result.add_message(message)
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                    # Add error message if rule fails
                    result.add_message(ValidationMessage(
                        level=rule.level,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation rule {rule.rule_id} failed: {str(e)}",
                        location="validator",
                        suggested_action="Check rule implementation or configuration"
                    ))
                    logger.error(f"Rule {rule.rule_id} failed: {e}")
        
        # Add timing information
        result.validation_time = time.time() - start_time
        
        logger.info(f"Validation completed in {result.validation_time:.3f}s - {result.summary()}")
        
        return result
    
    def validate_and_dump_schema(self, config: RELICConfig) -> Dict[str, Any]:
        """
        Validate configuration and export schema.
        
        Parameters:
            config : RELICConfig
                Configuration to validate
                
        Returns:
            Dict[str, Any]: Validation results and schema
        """
        validation_result = self.validate(config)
        schema = config.to_schema()
        
        return {
            "validation": {
                "is_valid": validation_result.is_valid,
                "summary": validation_result.summary(),
                "num_errors": validation_result.num_errors,
                "num_warnings": validation_result.num_warnings,
                "messages": [str(msg) for msg in validation_result.messages]
            },
            "schema": schema
        }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validator statistics and performance metrics."""
        return {
            'num_rules': len(self.rules),
            'validation_count': self.validation_count,
            'cache_size': len(self.rule_cache),
            'rules_by_level': {
                level.value: len([r for r in self.rules if r.level == level])
                for level in ConfigValidationStage
            },
            'rule_ids': [r.rule_id for r in self.rules]
        }
    
    def clear_cache(self) -> None:
        """Clear validation result cache."""
        self.rule_cache.clear()
        logger.debug("Validation cache cleared")


# Convenience functions for common validation tasks
def validate_configuration(
    config: RELICConfig,
    strict: bool = True,
    *,
    calendar_type: Optional[str] = None,
) -> ConfigValidationResult:
    """
    Convenience function to validate a configuration with default settings.
    
    Parameters:
        config : RELICConfig
            Configuration to validate
        strict : bool
            Whether to run all validation levels (True) or just basic checks
        calendar_type : Optional[str]
            Explicit simulator calendar for calendar-dependent rules
            
    Returns:
        ConfigValidationResult: Validation results
    """
    validator = ConfigValidator(calendar_type=calendar_type)
    
    if strict:
        levels = list(ConfigValidationStage)
    else:
        levels = [ConfigValidationStage.SYNTACTIC, ConfigValidationStage.SEMANTIC]
    
    return validator.validate(config, levels=levels)


def quick_validate(
    config: RELICConfig,
    *,
    calendar_type: Optional[str] = None,
) -> bool:
    """
    Quick validation that returns only pass/fail status.
    
    Parameters:
        config : RELICConfig
            Configuration to validate
        calendar_type : Optional[str]
            Explicit simulator calendar for calendar-dependent rules
            
    Returns:
        bool: True if configuration is valid
    """
    result = validate_configuration(
        config,
        strict=False,
        calendar_type=calendar_type,
    )
    return result.is_valid


def validate_and_export_schema(
    config: RELICConfig,
    output_path: Optional[Path] = None,
    *,
    calendar_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate configuration and optionally export schema.
    
    Parameters:
        config : RELICConfig
            Configuration to validate
        output_path : Optional[Path]
            Path to save schema JSON
        calendar_type : Optional[str]
            Explicit simulator calendar for calendar-dependent rules
            
    Returns:
        Dict[str, Any]: Validation results and schema
    """
    import json
    
    validator = ConfigValidator(calendar_type=calendar_type)
    result = validator.validate_and_dump_schema(config)
    
    if output_path:
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(result['schema'], f, indent=2)
        logger.info(f"Schema exported to {output_path}")
    
    return result


ValidationLevel = ConfigValidationStage
ValidationResult = ConfigValidationResult


# Export main validation classes and functions  
__all__ = [
    'ConfigValidator',
    'ValidationRule',
    'ConfigValidationResult',
    'ValidationResult',
    'ValidationMessage',
    'ConfigValidationStage',
    'ValidationLevel',
    'ValidationSeverity',
    'validate_configuration',
    'quick_validate',
    'validate_and_export_schema',
    # Concrete rules
    'TemperatureBoundsRule',
    'TimeStepStabilityRule',
    'ConservationConsistencyRule',
    'ParameterBoundsRule',
    'MemoryUsageRule',
    'IntegrationMethodconsistencyRule'
]

