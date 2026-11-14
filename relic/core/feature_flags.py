"""
Global Feature Flag Manager for RELIC

This module provides a centralized system for managing feature flags
throughout the RELIC framework. Feature flags enable safe deployment
of new functionality with the ability to disable features if issues arise.

Key Principles:
1. Default Disabled: All advanced features default to False 
2. Baseline Behavior: Disabling features maintains baseline behavior
3. Environment Override: Flags can be overridden via environment variables
4. Centralized Management: Single source of truth for all feature states
5. Runtime Control: Features can be enabled/disabled at runtime
6. Declared Names Only: the FeatureFlagConfig fields are the flags; any other
   name raises ValueError instead of reading as disabled

Usage:
    from relic.core.feature_flags import feature_manager
    
    # Check if feature is enabled
    if feature_manager.is_enabled('full_richards_solver'):
        # Use advanced solver
        pass
    else:
        # Use simplified solver
        pass
        
    # Enable feature programmatically
    feature_manager.enable('o3_stress_enabled')
    
    # Disable feature
    feature_manager.disable('canopy_layers_enabled')

References:
    - Feature Flag best practices
    - Martin Fowler's Feature Toggle patterns
    - Environment-based configuration patterns
"""

from typing import Any, Callable, Dict, List, Optional, cast
from threading import Lock
from relic.core.config.models import FeatureFlagConfig


class FeatureFlagManager:
    """
    Centralized feature flag manager for RELIC.
    
    Provides thread-safe access to feature flags with caching,
    environment variable override, and runtime modification capabilities.
    """
    
    _instance: Optional['FeatureFlagManager'] = None
    _lock = Lock()
    
    def __new__(cls) -> "FeatureFlagManager":
        # Singleton pattern to ensure single source of truth
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if not hasattr(self, '_initialized'):
            self._config = FeatureFlagConfig()
            self._cache: Dict[str, bool] = {}
            self._callbacks: Dict[str, List[Callable[[str, bool], None]]] = {}
            self._initialized = True
    
    def _declared_flag(self, feature_name: str) -> str:
        """Return ``feature_name`` when it is a declared FeatureFlagConfig field.

        The declared fields are the only flags; a name that is not one of them
        (a setting such as ``conservation_checks``, a misspelling, or another
        attribute of the configuration model) is rejected, never read as off.

        Raises
        ------
        TypeError
            If ``feature_name`` is not a string.
        ValueError
            If ``feature_name`` is not a declared flag.
        """
        if not isinstance(feature_name, str):
            raise TypeError(
                f"Feature flag name must be a string, got {type(feature_name).__name__}"
            )
        declared = type(self._config).model_fields
        if feature_name not in declared:
            raise ValueError(
                f"Unknown feature flag '{feature_name}'; the declared flags "
                f"(FeatureFlagConfig fields) are: {', '.join(sorted(declared))}"
            )
        return feature_name

    def is_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled.

        Parameters
        ----------
        feature_name : str
            Name of a declared feature flag (a FeatureFlagConfig field)

        Returns
        -------
        bool
            True if the feature is enabled, False if it is disabled

        Raises
        ------
        ValueError
            If ``feature_name`` is not a declared flag.
        """
        flag = self._declared_flag(feature_name)
        if flag in self._cache:
            return self._cache[flag]
        enabled = bool(getattr(self._config, flag))
        self._cache[flag] = enabled
        return enabled

    def update_flag(self, feature_name: str, enabled: bool) -> None:
        """Set a feature flag state and refresh cache."""
        if enabled:
            self.enable(feature_name)
        else:
            self.disable(feature_name)
    
    def enable(self, feature_name: str) -> None:
        """
        Enable a feature flag.
        
        Parameters
        ----------
        feature_name : str
            Name of a declared feature flag; an undeclared name raises ValueError
        """
        flag = self._declared_flag(feature_name)
        with self._lock:
            setattr(self._config, flag, True)
            self._cache[flag] = True
            self._notify_callbacks(flag, True)
    
    def disable(self, feature_name: str) -> None:
        """
        Disable a feature flag.
        
        Parameters
        ----------
        feature_name : str
            Name of a declared feature flag; an undeclared name raises ValueError
        """
        flag = self._declared_flag(feature_name)
        with self._lock:
            setattr(self._config, flag, False)
            self._cache[flag] = False
            self._notify_callbacks(flag, False)
    
    def toggle(self, feature_name: str) -> bool:
        """
        Toggle a feature flag state.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature to toggle
            
        Returns
        -------
        bool
            New state of the feature
        """
        current_state = self.is_enabled(feature_name)
        if current_state:
            self.disable(feature_name)
        else:
            self.enable(feature_name)
        return not current_state
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all feature flags and their current states."""
        flags_raw: Dict[str, Any]
        if hasattr(self._config, "model_dump"):
            flags_raw = cast(Dict[str, Any], self._config.model_dump())
        else:
            flags_raw = cast(Dict[str, Any], self._config.dict())
        return {k: bool(v) for k, v in flags_raw.items()}
    
    def get_enabled_features(self) -> List[str]:
        """Get the declared feature flags that are currently enabled."""
        return [name for name, enabled in self.get_all_flags().items() if enabled]

    def get_disabled_features(self) -> List[str]:
        """Get the declared feature flags that are currently disabled."""
        return [name for name, enabled in self.get_all_flags().items() if not enabled]
    
    def reset_to_defaults(self) -> None:
        """Reset all feature flags to their default states."""
        with self._lock:
            self._config = FeatureFlagConfig()
            self._cache.clear()
    
    def load_from_environment(self) -> None:
        """Reload configuration from environment variables."""
        with self._lock:
            self._config = FeatureFlagConfig()
            self._cache.clear()
    
    def register_callback(
        self,
        feature_name: str,
        callback: Callable[[str, bool], None],
    ) -> None:
        """
        Register a callback to be called when feature state changes.
        
        Parameters
        ----------
        feature_name : str
            Name of a declared feature flag to watch; an undeclared name raises
            ValueError (its callback could never fire)
        callback : callable
            Function to call when feature state changes.
            Should accept (feature_name: str, enabled: bool) arguments.
        """
        flag = self._declared_flag(feature_name)
        if flag not in self._callbacks:
            self._callbacks[flag] = []
        self._callbacks[flag].append(callback)
    
    def _notify_callbacks(self, feature_name: str, enabled: bool) -> None:
        """Notify every registered callback of a feature state change.

        All callbacks run even when one fails; their failures are raised
        together afterwards (ExceptionGroup). The flag change itself has been
        applied by the caller.
        """
        errors: List[Exception] = []
        for callback in self._callbacks.get(feature_name, ()):
            try:
                callback(feature_name, enabled)
            except Exception as exc:  # noqa: BLE001 - collected and re-raised below
                errors.append(exc)
        if errors:
            raise ExceptionGroup(
                f"Feature flag '{feature_name}' listeners failed after it was set to {enabled}",
                errors,
            )
    
    def get_feature_status_report(self) -> str:
        """Generate a human-readable status report of all features."""
        lines = ["RELIC Feature Flag Status Report", "=" * 40]
        
        enabled = self.get_enabled_features()
        disabled = self.get_disabled_features()
        
        lines.append(f"\nEnabled Features ({len(enabled)}):")
        for feature in enabled:
            lines.append(f"  [+] {feature}")
            
        lines.append(f"\nDisabled Features ({len(disabled)}):")
        for feature in disabled:
            lines.append(f"  [-] {feature}")
            
        return "\n".join(lines)
    
    def validate_feature_consistency(self) -> List[str]:
        """
        Validate feature flag consistency and return any warnings.
        
        Returns
        -------
        List[str]
            List of warning messages for inconsistent feature combinations
        """
        warnings: List[str] = []
        flags = self.get_all_flags()
        
        # Check for potentially problematic combinations
        if flags.get('full_richards_solver') and not flags.get('advanced_numerical_solvers'):
            warnings.append(
                "full_richards_solver enabled but advanced_numerical_solvers disabled. "
                "Consider enabling advanced solvers for better stability."
            )
        
        if flags.get('strict_conservation_enabled') and not flags.get('canopy_interception_diagnostics'):
            warnings.append(
                "strict_conservation_enabled but canopy_interception_diagnostics disabled. "
                "Consider enabling diagnostics for complete mass balance checking."
            )
        
        if flags.get('o3_stress_enabled') and not flags.get('parameter_sensitivity_analysis'):
            warnings.append(
                "o3_stress_enabled but parameter_sensitivity_analysis disabled. "
                "Consider enabling sensitivity analysis to understand O3 impact."
            )
        
        return warnings


# Global singleton instance
feature_manager = FeatureFlagManager()


def require_feature(feature_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to require a feature flag to be enabled for function execution.
    
    Parameters
    ----------
    feature_name : str
        Name of the required feature flag
        
    Usage
    -----
    @require_feature('full_richards_solver')
    def advanced_solver_method(self, ...):
        # This method only runs if full_richards_solver is enabled
        pass
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not feature_manager.is_enabled(feature_name):
                raise RuntimeError(
                    f"Feature '{feature_name}' is disabled. "
                    f"Enable it to use {func.__name__}()"
                )
            return func(*args, **kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


def feature_enabled(feature_name: str) -> bool:
    """
    Convenience function to check if a feature is enabled.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature to check
        
    Returns
    -------
    bool
        True if enabled, False otherwise
    """
    return feature_manager.is_enabled(feature_name)
