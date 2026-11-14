"""
Base class for PFT parameter groups.

This module provides a unified base class for all PFT parameter groups,
eliminating code duplication and ensuring consistent parameter loading
from the ParameterManager.
"""

from typing import Any, Dict, Optional
from dataclasses import InitVar, dataclass, field
import logging
from relic.parameters import UnifiedParameterManager

logger = logging.getLogger(__name__)


@dataclass
class PFTParameterBase:
    """
    Base class for PFT parameter groups.
    
    This class provides common functionality for all parameter groups,
    including unified parameter fetching from ParameterManager.
    """
    
    parameter_manager: InitVar[UnifiedParameterManager] = field(kw_only=True)

    # PFT namespace for parameter retrieval
    _pft_namespace: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self, parameter_manager: UnifiedParameterManager):
        """Initialize base class (called by dataclass mechanism)."""
        if parameter_manager is None:
            raise TypeError("parameter_manager is required")
        self._parameter_manager = parameter_manager

    def get_parameter_from_manager(self, param_name: str,
                                  required: bool = True,
                                  register_if_missing: bool = False,
                                  metadata: Optional[Dict] = None) -> Any:
        """
        Get a parameter value from ParameterManager using the correct API.

        Parameters
        ----------
        param_name : str
            Local parameter name (will be prefixed with namespace)
        required : bool
            Whether the parameter is required
        register_if_missing : bool
            Whether to register the parameter if it doesn't exist
        metadata : Optional[Dict]
            Metadata including units, description, references

        Returns
        -------
        Any
            Parameter value from ParameterManager
        """
        if not self._pft_namespace:
            raise ValueError("PFT namespace required for parameter retrieval")

        # The param_name already includes the full namespace path.
        value = self._parameter_manager.get(
            param_name,
            required=required,
            expected_type=None,
        )
        if value is None and register_if_missing:
            logger.warning(
                "Parameter %s not found, will use system default",
                param_name,
            )
        return value

    def fetch_all_parameters(self, param_definitions: Dict[str, Dict]) -> None:
        """
        Fetch all parameters based on definitions.
        
        Parameters
        ----------
        param_definitions : Dict[str, Dict]
            Dictionary mapping attribute names to their parameter definitions
            Each definition can have 'required' and 'enum_type' keys
        """
        for attr_name, definition in param_definitions.items():
            # Skip if already set
            if getattr(self, attr_name) is not None:
                continue
            
            # Get parameter from manager
            value = self.get_parameter_from_manager(
                param_name=attr_name,
                required=definition.get('required', True),
                register_if_missing=definition.get('register_if_missing', False)
            )
            
            # Handle enum conversion if needed
            if value is not None and 'enum_type' in definition:
                enum_class = definition['enum_type']
                if isinstance(value, str):
                    try:
                        value = enum_class(value)
                    except ValueError:
                        logger.warning(f"Invalid enum value '{value}' for {attr_name}")
            
            # Set the attribute
            setattr(self, attr_name, value)
    
    def validate_parameters(self) -> None:
        """
        Base validation method to be overridden by subclasses.
        """
        pass


