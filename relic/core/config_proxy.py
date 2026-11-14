"""
Config Proxy for Parameter Access

This module provides a proxy wrapper for configuration dictionaries to route
access through the UnifiedParameterManager and keep a single source of truth.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConfigProxy:
    """
    Proxy wrapper for configuration dictionary access.
    
    Intercepts direct config access attempts and redirects to the parameter
    manager for parameter retrieval.
    """
    
    def __init__(self, functor_instance: Any) -> None:
        """
        Initialize config proxy.
        
        Parameters:
            functor_instance : FunctorBase
                The functor instance this proxy belongs to
        """
        self._functor = functor_instance
        self._original_config = functor_instance._original_config if hasattr(functor_instance, '_original_config') else {}
        self._access_count = 0
        self._warned_keys: set[str] = set()
    
    def __getitem__(self, key: str) -> Any:
        """
        Intercept dictionary-style access.
        
        Parameters:
            key : str
                Configuration key being accessed
                
        Returns:
            Any: Configuration value
        """
        self._log_access(key, "dictionary")
        
        # Special handling for 'parameters' key to return another proxy
        if key == "parameters":
            return ParametersProxy(self._functor)

        # get_parameter_value resolves the manager and then this config (its
        # 'parameters' section and root); None means the key is absent or holds
        # None. Lookup failures (validation, units, bounds) propagate.
        value = self._functor.get_parameter_value(
            key,
        )
        if value is not None:
            return value
        return self._original_config.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Intercept get() method calls.
        
        Parameters:
            key : str
                Configuration key
            default : Any
                Default value if key not found
                
        Returns:
            Any: Configuration value
        """
        self._log_access(key, "get")
        
        # Special handling for 'parameters' key
        if key == "parameters":
            return ParametersProxy(self._functor)

        # Same resolution as __getitem__; the caller's default applies only to an
        # absent key, never to a failed lookup.
        value = self._functor.get_parameter_value(
            key,
        )
        if value is not None:
            return value
        return self._original_config.get(key, default)
    
    def _log_access(self, key: str, access_type: str) -> None:
        """
        Log configuration access attempt.
        
        Parameters:
            key : str
                Key being accessed
            access_type : str
                Type of access (get, dictionary, etc.)
        """
        self._access_count += 1
        
        # Only log once per key to avoid spam
        if key not in self._warned_keys:
            self._warned_keys.add(key)
            
            functor_name = self._functor.name if hasattr(self._functor, 'name') else 'unknown'
            logger.debug(
                "Config access via %s for key '%s' in functor '%s'",
                access_type,
                key,
                functor_name,
            )
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._original_config
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigProxy(functor={self._functor.name}, accesses={self._access_count})"

    def __eq__(self, other: Any) -> bool:
        """Support equality comparison with plain dictionaries."""
        if isinstance(other, ConfigProxy):
            return self._original_config == other._original_config
        if isinstance(other, dict):
            return self._original_config == other
        return False


class ParametersProxy:
    """
    Proxy for nested 'parameters' config access.
    
    Handles nested access of config.get("parameters", {}).get(...)
    """
    
    def __init__(self, functor_instance: Any) -> None:
        """
        Initialize parameters proxy.
        
        Parameters:
            functor_instance : FunctorBase
                The functor instance this proxy belongs to
        """
        self._functor = functor_instance
        self._original_params = {}
        if hasattr(functor_instance, '_original_config'):
            self._original_params = functor_instance._original_config.get('parameters', {})

    def _get_managed_value(self, key: str) -> Any:
        """Return a managed value without registering a missing parameter.

        None means absent; lookup failures propagate.
        """
        return self._functor.get_parameter_value(
            key,
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Intercept nested parameter access.
        
        Parameters:
            key : str
                Parameter name
            default : Any
                Default value or dict
                
        Returns:
            Any: Parameter value or another proxy for deeper nesting
        """
        # Check if this is the deeply nested pattern
        if isinstance(default, dict) and 'value' in default:
            # This is likely config.get("parameters", {}).get("param", {}).get("value", default)
            actual_default = default.get('value')
            value = self._get_managed_value(key)
            if value is None:
                original_value = self._original_params.get(key)
                if isinstance(original_value, dict) and 'value' in original_value:
                    value = original_value.get('value')
                elif original_value is not None:
                    value = original_value
            if value is None:
                value = actual_default
            return {'value': value}

        # Regular parameter access
        value = self._get_managed_value(key)
        if value is not None:
            return value
        return self._original_params.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)


class StrictConfigProxy(ConfigProxy):
    """
    Strict version of ConfigProxy that raises errors instead of warnings.
    
    Used when strict mode is enabled to enforce parameter manager access.
    """
    
    def _log_access(self, key: str, access_type: str) -> None:
        """
        Raise error on configuration access attempt.
        
        Parameters:
            key : str
                Key being accessed
            access_type : str
                Type of access
                
        Raises:
            RuntimeError: Always, to enforce parameter access rules
        """
        functor_name = self._functor.name if hasattr(self._functor, 'name') else 'unknown'
        
        raise RuntimeError(
            f"Direct config access not allowed in strict mode.\n"
            f"  Functor: {functor_name}\n"
            f"  Attempted: self.config.{access_type}('{key}')\n"
            f"  Required: self.get_parameter_value('{key}')\n"
            f"  Please update your code to use the parameter manager."
        )


__all__ = ['ConfigProxy', 'ParametersProxy', 'StrictConfigProxy']
