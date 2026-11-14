"""
Unified Parameter Management System for RELIC

This package provides the authoritative parameter management system for RELIC,
integrating hierarchical configuration loading, scientific validation, unit
checking, PyTorch integration for learnable parameters, and comprehensive
parameter lifecycle management.

The system serves as the single source of truth for all model parameters,
providing advanced features for modern scientific computing.

Architecture Components:
    manager.py: Core UnifiedParameterManager with global instance management
    database/: Two-tier parameter database (relic/calibrated)
    validators/: Parameter validation and constraint checking

Key Features:
    - Hierarchical parameter loading with precedence rules
    - Scientific constraint validation with physical bounds
    - Unit consistency checking using CF-1.8 conventions
    - PyTorch nn.Parameter integration for optimization
    - Parameter database management with provenance tracking
    - Event-driven change notifications
    - Device migration and precision control
    - Snapshot/restore capabilities for debugging

Usage Examples:
    Basic parameter access:
        >>> import relic.parameters as params
        >>> params.get('biosphere.photosynthesis.vcmax25')
        60.0
        >>> params.set('biosphere.photosynthesis.vcmax25', 75.0)
    
    Parameter manager access:
        >>> manager = params.get_global_manager()
        >>> manager.list(namespace='biosphere.photosynthesis')
        ['biosphere.photosynthesis.vcmax25', 'biosphere.photosynthesis.jmax25', ...]
    
    PyTorch integration:
        >>> manager.to('cuda')  # Move learnable parameters to GPU
        >>> nn_params = manager.get_nn_parameters()
        >>> optimizer = torch.optim.Adam(nn_params.values())
    
    Configuration loading:
        >>> manager.database.load_all()  # Load from database
        >>> manager.snapshot()  # Create snapshot for debugging

Mathematical Foundation:
    Parameter space: Theta = {theta_1, theta_2, ..., theta} where theta in [a, b]
    Hierarchical resolution: P_final = P_default  P_calibrated  P_config  P_runtime
    Validation constraint: V(theta) =  C(theta) where C are physical/scientific constraints
    Update propagation: deltatheta -> {Functors dependent on theta}
    
References:
    - CF-1.8 Convention for Climate and Forecast Metadata
    - FAIR Data Principles for scientific parameter management
    - PyTorch Parameter and Module system documentation
    - Pydantic data validation and serialization
"""

# Import shared types first to avoid circular imports
from pathlib import Path

from .types import (
    ParameterCategory,
    ValidationLevel,
    ValidationContext,
    ParameterValue,
    ParameterDict,
    MetadataDict,
    ValidationErrors,
    ChangeCallback
)

# Import components that don't depend on manager
from .storage import (
    ParameterStorage,
    ParameterSnapshot,
    extract_namespace_hierarchy,
    merge_namespaces
)

from .validation import (
    ParameterValidator,
    ValidationType,
    validate_parameter_bounds,
    validate_parameter_type
)

from .serialization import (
    export,
    load,
    convert_format,
    serialize_tensor,
    deserialize_tensor,
    serialize_parameter_spec,
    deserialize_parameter_spec,
    ParameterSchema,
    SerializationFormat,
)

from .audit import (
    AuditIssueType,
    AuditSeverity,
    ParameterAuditIssue,
    ParameterAuditReport,
)

# Import core specs before manager to keep dependencies explicit
from .spec import ParameterSpec
from .database_manager import ParameterDatabase

# Import manager last to avoid circular imports
from .manager import UnifiedParameterManager
from .manager import (
    get_global_manager,
    reset_global_manager,
    get,
    set_value,
    list_parameters,
    BoundsPolicy,
    UnitValidationMode,
)

from .decorators import (
    managed_parameters,
    parameter,
    validated_parameter,
    learnable_parameter,
    ParameterDefinition
)

# Import database components
from .database import (
    DATABASE_TIERS,
    PARAMETER_NAMESPACES,
    get_database_root,
    validate_database_structure,
    create_database_structure
)

# Version information
__version__ = "1.0.0"
__author__ = "RELIC Parameter Management Team"

# Package configuration
DEFAULT_DATABASE_PATH = "relic/parameters/database"
SUPPORTED_FORMATS = ["yaml", "json", "toml"]
SCHEMA_VERSION = "1.0.0"

# Convenience functions for immediate use
def initialize_database() -> bool:
    """
    Initialize parameter database structure.
    
    Creates the three-tier database directory structure if it doesn't exist
    and validates the setup.
    
    Returns:
        bool: True if initialization successful
    """
    try:
        create_database_structure()
        validation_results = validate_database_structure()
        return all(validation_results.values())
    except (OSError, ValueError, RuntimeError):
        return False


def get_status() -> dict:
    """
    Get comprehensive parameter system status.
    
    Returns:
        dict: Status information including manager state, database info, etc.
    """
    manager = get_global_manager()
    manager_status = manager.get_status()
    
    # Add database validation status
    db_validation = validate_database_structure()
    manager_status['database_validation'] = db_validation
    
    return manager_status


def validate_system() -> dict:
    """
    Validate entire parameter system integrity.
    
    Returns:
        dict: Validation results and any issues found
    """
    results = {
        'manager_validation': [],
        'database_validation': {},
        'parameter_validation': [],
        'overall_status': 'unknown'
    }
    
    try:
        # Validate manager
        manager = get_global_manager()
        param_errors = manager.validate()
        results['parameter_validation'] = param_errors
        
        # Validate database structure
        results['database_validation'] = validate_database_structure()
        
        # Overall assessment
        has_param_errors = len(param_errors) > 0
        has_db_errors = not all(results['database_validation'].values())
        
        if not has_param_errors and not has_db_errors:
            results['overall_status'] = 'healthy'
        elif has_param_errors and not has_db_errors:
            results['overall_status'] = 'parameter_issues'
        elif not has_param_errors and has_db_errors:
            results['overall_status'] = 'database_issues'
        else:
            results['overall_status'] = 'multiple_issues'
            
    except (OSError, ValueError, RuntimeError) as e:
        results['manager_validation'].append(f"System validation failed: {e}")
        results['overall_status'] = 'validation_failed'
    
    return results


def reset_system() -> None:
    """
    Reset the entire parameter system to initial state.
    
    This function resets the global manager and clears all cached data.
    Use with caution as this will lose all runtime parameter changes.
    """
    reset_global_manager()
    # Clear any module-level caches if they exist
    

def load_configuration(config_path: str, merge: bool = True) -> dict:
    """
    Load parameter configuration from file.
    
    Parameters:
        config_path : str
            Path to configuration file (YAML/JSON/TOML)
        merge : bool
            Whether to merge with existing parameters (True) or replace (False)
            
    Returns:
        dict: Loading results and any errors
    """
    manager = get_global_manager()
    
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            return {'success': False, 'error': f'Configuration file not found: {config_path}'}
        
        # Clear existing parameters if not merging
        if not merge:
            reset_global_manager()
            manager = get_global_manager()
        
        # Load based on file extension
        if config_file.suffix.lower() == '.yaml':
            # Use database loading mechanism
            errors = manager.database.load_from_yaml(config_file)
        elif config_file.suffix.lower() == '.json':
            errors = manager.database.load_from_json(config_file)  
        else:
            return {'success': False, 'error': f'Unsupported file format: {config_file.suffix}'}
        
        return {
            'success': len(errors) == 0,
            'errors': dict(errors) if errors else {},
            'parameters_loaded': len(manager.parameters)
        }
        
    except (OSError, ValueError, RuntimeError) as e:
        return {'success': False, 'error': f'Configuration loading failed: {e}'}


def export_configuration(output_path: str, 
                        namespace: str = None,
                        format: str = 'yaml') -> dict:
    """
    Export current parameters to configuration file.
    
    Parameters:
        output_path : str
            Output file path
        namespace : str, optional
            Export only parameters under this namespace
        format : str
            Export format ('yaml', 'json', 'toml')
            
    Returns:
        dict: Export results
    """
    manager = get_global_manager()
    
    try:
        output_file = Path(output_path)
        
        # Export parameters
        manager.export(namespace=namespace, format=format, output_path=output_file)
        
        return {
            'success': True,
            'output_path': str(output_file),
            'parameters_exported': len(manager.list(namespace=namespace)),
            'format': format
        }
        
    except (OSError, ValueError, RuntimeError) as e:
        return {'success': False, 'error': f'Export failed: {e}'}


# Auto-initialize database structure on first import
import atexit

def _cleanup():
    """Cleanup function for parameter system."""
    # Perform any necessary cleanup

# Register cleanup function
atexit.register(_cleanup)

# Intentionally avoid import-time database initialization side effects.
# Call initialize_database() explicitly from CLI/bootstrap code if needed.

# Export main components for public API
__all__ = [
    # Core manager classes
    'UnifiedParameterManager',
    'ParameterSpec', 
    'ParameterDatabase',
    'ParameterCategory',
    'ValidationLevel',
    
    # Storage components
    'ParameterStorage',
    'ParameterSnapshot',
    'extract_namespace_hierarchy',
    'merge_namespaces',
    
    # Validation components
    'ParameterValidator',
    'ValidationContext',
    'ValidationType',
    'BoundsPolicy',
    'UnitValidationMode',
    'validate_parameter_bounds',
    'validate_parameter_type',
    
    # Serialization components
    'export',
    'load',
    'convert_format',
    'serialize_tensor',
    'deserialize_tensor',
    'serialize_parameter_spec',
    'deserialize_parameter_spec',
    'ParameterSchema',
    'SerializationFormat',
    
    # Decorator components
    'managed_parameters',
    'parameter',
    'validated_parameter',
    'learnable_parameter',
    'ParameterDefinition',
    
    # Global manager functions
    'get_global_manager',
    'reset_global_manager',
    
    # Convenience functions
    'get',
    'set_value',
    'list_parameters',
    
    # Database functions  
    'DATABASE_TIERS',
    'PARAMETER_NAMESPACES',
    'get_database_root',
    'create_database_structure',
    'validate_database_structure',
    
    # System management
    'initialize_database',
    'get_status',
    'validate_system',
    'reset_system',
    'load_configuration',
    'export_configuration',
    
    # Type definitions
    'ParameterValue',
    'ParameterDict',
    'MetadataDict',
    'ValidationErrors',
    'ChangeCallback',
    'AuditIssueType',
    'AuditSeverity',
    'ParameterAuditIssue',
    'ParameterAuditReport',
    
    # Constants
    'DEFAULT_DATABASE_PATH',
    'SUPPORTED_FORMATS',
    'SCHEMA_VERSION'
]
