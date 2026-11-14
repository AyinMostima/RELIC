"""
Unified error handling for coupling operations.

Provides consistent exception types and error messages
across all coupling modules.
"""

from typing import Optional, Any, Dict, Iterable


class CouplingError(Exception):
    """Base exception for all coupling errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize coupling error.
        
        Parameters:
            message : str
                Error message
            details : Optional[Dict[str, Any]]
                Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Format error as string."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConservationError(CouplingError):
    """
    Raised when conservation constraints are violated.
    
    Attributes:
        property_name : str
            The conserved property that was violated
        source_total : float
            Total in source
        target_total : float
            Total in target
        error : float
            Conservation error
        threshold : float
            Acceptable threshold
    """
    
    def __init__(
        self,
        property_name: str,
        source_total: float,
        target_total: float,
        error: float,
        threshold: float
    ):
        """Initialize conservation error."""
        self.property_name = property_name
        self.source_total = source_total
        self.target_total = target_total
        self.error = error
        self.threshold = threshold
        
        message = (
            f"Conservation violation for {property_name}: "
            f"error={error:.2e} exceeds threshold={threshold:.2e}"
        )
        
        details = {
            'property': property_name,
            'source_total': source_total,
            'target_total': target_total,
            'absolute_error': abs(target_total - source_total),
            'relative_error': error,
            'threshold': threshold
        }
        
        super().__init__(message, details)


class CFSyncError(CouplingError):
    """
    Raised when CF metadata synchronization fails.
    
    Attributes:
        attribute : str
            The CF attribute that failed
        reason : str
            Reason for failure
        metadata : Dict[str, Any]
            The metadata that failed
    """
    
    def __init__(
        self,
        attribute: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize CF sync error."""
        self.attribute = attribute
        self.reason = reason
        self.metadata = metadata or {}
        
        message = f"CF metadata sync failed for '{attribute}': {reason}"
        
        details = {
            'attribute': attribute,
            'reason': reason,
            'metadata_keys': list(metadata.keys()) if metadata else []
        }
        
        super().__init__(message, details)


class ScalingPolicyError(CouplingError):
    """
    Raised when scaling policy validation fails.
    
    Attributes:
        policy : str
            The policy that failed
        value : Any
            The invalid value
        expected : str
            Expected value or range
    """
    
    def __init__(
        self,
        policy: str,
        value: Any,
        expected: str
    ):
        """Initialize scaling policy error."""
        self.policy = policy
        self.value = value
        self.expected = expected
        
        message = (
            f"Scaling policy violation for '{policy}': "
            f"got {value}, expected {expected}"
        )
        
        details = {
            'policy': policy,
            'invalid_value': value,
            'expected': expected
        }
        
        super().__init__(message, details)


class ScaleHierarchyError(CouplingError):
    """
    Raised when scale hierarchy is invalid.
    
    Attributes:
        source_scale : str
            Source scale name
        target_scale : str
            Target scale name
        direction : str
            Scaling direction
        valid_hierarchy : list
            Valid scale hierarchy
    """
    
    def __init__(
        self,
        source_scale: str,
        target_scale: str,
        direction: str,
        valid_hierarchy: list[str]
    ):
        """Initialize scale hierarchy error."""
        self.source_scale = source_scale
        self.target_scale = target_scale
        self.direction = direction
        self.valid_hierarchy = valid_hierarchy
        
        message = (
            f"Invalid {direction}scaling: {source_scale} -> {target_scale}. "
            f"Valid hierarchy: {' -> '.join(valid_hierarchy)}"
        )
        
        details = {
            'source': source_scale,
            'target': target_scale,
            'direction': direction,
            'valid_hierarchy': valid_hierarchy
        }
        
        super().__init__(message, details)


class DataValidationError(CouplingError):
    """
    Raised when input data validation fails.
    
    Attributes:
        variable : str
            Variable name
        issue : str
            Validation issue
        data_info : Dict[str, Any]
            Information about the data
    """
    
    def __init__(
        self,
        variable: str,
        issue: str,
        data_info: Optional[Dict[str, Any]] = None
    ):
        """Initialize data validation error."""
        self.variable = variable
        self.issue = issue
        self.data_info = data_info or {}
        
        message = f"Data validation failed for '{variable}': {issue}"
        
        details = {
            'variable': variable,
            'issue': issue,
            **(data_info or {})
        }
        
        super().__init__(message, details)


class DeviceContextError(CouplingError):
    """
    Raised when device context is inconsistent.
    
    Attributes:
        expected_device : str
            Expected device
        actual_device : str
            Actual device
        tensor_name : str
            Name of tensor with wrong device
    """
    
    def __init__(
        self,
        expected_device: str,
        actual_device: str,
        tensor_name: str
    ):
        """Initialize device context error."""
        self.expected_device = expected_device
        self.actual_device = actual_device
        self.tensor_name = tensor_name
        
        message = (
            f"Device mismatch for '{tensor_name}': "
            f"expected {expected_device}, got {actual_device}"
        )
        
        details = {
            'expected': expected_device,
            'actual': actual_device,
            'tensor': tensor_name
        }
        
        super().__init__(message, details)


class AggregationError(CouplingError):
    """
    Raised when aggregation operation fails.
    
    Attributes:
        method : str
            Aggregation method
        variable : str
            Variable being aggregated
        reason : str
            Reason for failure
    """
    
    def __init__(
        self,
        method: str,
        variable: str,
        reason: str
    ):
        """Initialize aggregation error."""
        self.method = method
        self.variable = variable
        self.reason = reason
        
        message = (
            f"Aggregation failed for '{variable}' using {method}: {reason}"
        )
        
        details = {
            'method': method,
            'variable': variable,
            'reason': reason
        }
        
        super().__init__(message, details)


def format_error_report(errors: Iterable[Any]) -> str:
    """
    Format a list of errors into a report.
    
    Parameters:
        errors : list
            List of errors (Exception objects or strings)
            
    Returns:
        str: Formatted error report
    """
    if not errors:
        return "No errors reported."
    
    lines = ["=== Error Report ==="]
    
    for i, error in enumerate(errors, 1):
        if isinstance(error, CouplingError):
            lines.append(f"{i}. {error.message}")
            if error.details:
                for key, value in error.details.items():
                    lines.append(f"   - {key}: {value}")
        else:
            lines.append(f"{i}. {str(error)}")
    
    return "\n".join(lines)
