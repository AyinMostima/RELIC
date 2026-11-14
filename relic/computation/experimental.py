"""
Experimental Features Management for RELIC Computation Layer

This module provides isolation and management of experimental features
with explicit enabling, safety controls, and explicit opt-in behavior.

Experimental features include:
- Advanced GPU kernels (Triton, CuPy)
- Machine learning acceleration
- Streaming computation
- Experimental algorithms

Features:
- Environment variable based feature flags
- Import protection with clear error messages
- Dependency checking and installation guidance
- Explicit selection of stable implementations when requested
- Performance monitoring and warnings

Usage:
    export RELIC_EXPERIMENTAL=true
    export RELIC_EXPERIMENTAL_TRITON_KERNELS=true
    
    from relic.computation.experimental import is_experimental_enabled
    if is_experimental_enabled('triton_kernels'):
        from relic.computation.experimental.triton import TritonPhotosynthesis

Architecture:
    - Feature isolation with import guards
    - Lazy loading of experimental dependencies
    - Clear documentation of risks and limitations
    - Integration with health check system
    - Windows/Conda support checks
"""

import os
import logging
from typing import Dict, List, Optional, Any, Callable
import importlib.util

logger = logging.getLogger(__name__)

# Global experimental feature registry
_EXPERIMENTAL_FEATURES = {
    'triton_kernels': {
        'description': 'Triton GPU kernels for scientific computing',
        'dependencies': ['triton'],
        'platforms': ['linux', 'darwin'],  # Windows support limited
        'risks': [
            'May cause system instability',
            'Limited Windows support',
            'Requires specific GPU architectures'
        ],
        'alternative': 'standard PyTorch kernels'
    },
    
    'cupy_acceleration': {
        'description': 'CuPy-based CUDA acceleration',
        'dependencies': ['cupy'],
        'platforms': ['linux', 'darwin', 'win32'],
        'risks': [
            'Memory management differences from PyTorch',
            'Potential device synchronization issues'
        ],
        'alternative': 'PyTorch CUDA operations'
    },
    
    'ml_acceleration': {
        'description': 'Machine learning based model acceleration',
        'dependencies': ['sklearn', 'xgboost'],
        'platforms': ['linux', 'darwin', 'win32'],
        'risks': [
            'Model accuracy may differ from physics-based approach',
            'Requires pre-trained models'
        ],
        'alternative': 'Original physics-based calculations'
    },
    
    'streaming': {
        'description': 'Streaming computation for large datasets',
        'dependencies': ['dask', 'zarr'],
        'platforms': ['linux', 'darwin', 'win32'],
        'risks': [
            'Memory usage patterns may differ',
            'Requires distributed computing setup'
        ],
        'alternative': 'Batch processing'
    },
    'flash_attention': {
        'description': 'Flash attention for memory-efficient attention',
        'dependencies': ['flash_attn'],
        'platforms': ['linux'],  # Very limited platform support
        'risks': [
            'Highly experimental',
            'Limited platform support',
            'May require specific hardware'
        ],
        'alternative': 'Standard attention mechanisms'
    }
}

# Cache for dependency checking results
_DEPENDENCY_CACHE = {}
_PLATFORM_INFO = None


def get_platform_info() -> Dict[str, Any]:
    """Get current platform information."""
    global _PLATFORM_INFO
    
    if _PLATFORM_INFO is None:
        import platform
        import sys
        
        _PLATFORM_INFO = {
            'system': platform.system().lower(),
            'platform': sys.platform,
            'machine': platform.machine(),
            'python_version': sys.version_info,
            'is_windows': sys.platform == 'win32',
            'is_linux': sys.platform.startswith('linux'),
            'is_macos': sys.platform == 'darwin'
        }
    
    return _PLATFORM_INFO


def check_dependency_available(dependency: str) -> Dict[str, Any]:
    """
    Check if a dependency is available and get version info.
    
    Parameters:
        dependency : str
            Package name to check
    
    Returns:
        Dict[str, Any]: Availability status and metadata
    """
    if dependency in _DEPENDENCY_CACHE:
        return _DEPENDENCY_CACHE[dependency]
    
    result = {
        'available': False,
        'version': None,
        'import_error': None,
        'installation_hint': None
    }
    
    # Optional-dependency probe: only a missing or unimportable package
    # (ImportError, including ModuleNotFoundError and DLL load failures) is
    # recorded as unavailable, with its message; every other error raised
    # while importing the package propagates.
    try:
        spec = importlib.util.find_spec(dependency)
        if spec is None:
            result['import_error'] = f"Package '{dependency}' not found"
        else:
            module = importlib.import_module(dependency)
            if hasattr(module, '__version__'):
                result['version'] = module.__version__
            result['available'] = True
    except ImportError as e:
        result['import_error'] = f"Error importing {dependency}: {e}"
    
    # Add installation hints
    if not result['available']:
        conda_packages = {
            'triton': 'conda install pytorch-triton-cuda -c pytorch',
            'cupy': 'conda install cupy -c conda-forge',
            'flash_attn': 'pip install flash-attn --no-build-isolation',
            'sklearn': 'conda install scikit-learn -c conda-forge',
            'xgboost': 'conda install xgboost -c conda-forge',
            'dask': 'conda install dask -c conda-forge',
            'zarr': 'conda install zarr -c conda-forge'
        }
        
        if dependency in conda_packages:
            result['installation_hint'] = conda_packages[dependency]
        else:
            result['installation_hint'] = f"pip install {dependency}"
    
    _DEPENDENCY_CACHE[dependency] = result
    return result


def is_experimental_enabled(feature_name: str) -> bool:
    """
    Check if an experimental feature is enabled via environment variables.
    
    Parameters:
        feature_name : str
            Name of experimental feature
    
    Returns:
        bool: True if feature is enabled
    """
    if feature_name not in _EXPERIMENTAL_FEATURES:
        # A misspelled feature name is an error, not a disabled feature.
        raise ValueError(
            f"Unknown experimental feature '{feature_name}'; known features: "
            f"{sorted(_EXPERIMENTAL_FEATURES)}"
        )
    
    # Check specific feature flag
    specific_flag = f"RELIC_EXPERIMENTAL_{feature_name.upper()}"
    if os.environ.get(specific_flag, "").lower() in ("true", "1", "yes", "on"):
        return True
    
    # Check global experimental flag
    global_flag = os.environ.get("RELIC_EXPERIMENTAL", "").lower()
    if global_flag in ("true", "1", "yes", "on"):
        return True
    
    # Check if explicitly disabled
    if os.environ.get(specific_flag, "").lower() in ("false", "0", "no", "off"):
        return False
    
    return False


def validate_experimental_feature(feature_name: str) -> Dict[str, Any]:
    """
    Validate that an experimental feature can be safely enabled.
    
    Parameters:
        feature_name : str
            Name of experimental feature
    
    Returns:
        Dict[str, Any]: Validation results
    """
    if feature_name not in _EXPERIMENTAL_FEATURES:
        return {
            'valid': False,
            'reason': f"Unknown experimental feature: {feature_name}",
            'recommendations': []
        }
    
    feature_info = _EXPERIMENTAL_FEATURES[feature_name]
    platform_info = get_platform_info()
    
    result = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'missing_deps': [],
        'recommendations': [],
        'platform_supported': True
    }
    
    # Check platform support
    if feature_info['platforms'] and platform_info['platform'] not in feature_info['platforms']:
        result['platform_supported'] = False
        result['errors'].append(
            f"Feature '{feature_name}' not supported on {platform_info['platform']}. "
            f"Supported platforms: {', '.join(feature_info['platforms'])}"
        )
        result['valid'] = False
    
    # Check dependencies
    for dep in feature_info['dependencies']:
        dep_status = check_dependency_available(dep)
        if not dep_status['available']:
            result['missing_deps'].append(dep)
            result['errors'].append(
                f"Required dependency '{dep}' not available: {dep_status['import_error']}"
            )
            if dep_status['installation_hint']:
                result['recommendations'].append(
                    f"Install {dep}: {dep_status['installation_hint']}"
                )
            result['valid'] = False
    
    # Add platform-specific warnings
    if platform_info['is_windows']:
        if feature_name in ['triton_kernels', 'flash_attention']:
            result['warnings'].append(
                f"Feature '{feature_name}' has limited Windows support"
            )
    
    # Add general warnings about experimental features
    result['warnings'].extend([
        f"Experimental feature: {feature_info['description']}",
        f"Alternative path: {feature_info['alternative']}"
    ])
    result['warnings'].extend(feature_info['risks'])
    
    return result


def get_experimental_status() -> Dict[str, Any]:
    """
    Get comprehensive status of all experimental features.
    
    Returns:
        Dict[str, Any]: Status of all experimental features
    """
    platform_info = get_platform_info()
    
    status = {
        'platform': platform_info,
        'global_experimental_enabled': os.environ.get("RELIC_EXPERIMENTAL", "").lower() in ("true", "1", "yes", "on"),
        'features': {}
    }
    
    for feature_name in _EXPERIMENTAL_FEATURES:
        feature_status = {
            'enabled': is_experimental_enabled(feature_name),
            'validation': validate_experimental_feature(feature_name),
            'description': _EXPERIMENTAL_FEATURES[feature_name]['description'],
            'dependencies': _EXPERIMENTAL_FEATURES[feature_name]['dependencies']
        }
        
        # Check dependency status
        feature_status['dependency_status'] = {}
        for dep in _EXPERIMENTAL_FEATURES[feature_name]['dependencies']:
            feature_status['dependency_status'][dep] = check_dependency_available(dep)
        
        status['features'][feature_name] = feature_status
    
    return status


def experimental_feature_guard(feature_name: str):
    """
    Decorator to guard experimental feature usage.
    
    Parameters:
        feature_name : str
            Name of experimental feature
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_experimental_enabled(feature_name):
                raise RuntimeError(
                    f"Experimental feature '{feature_name}' is not enabled. "
                    f"Set RELIC_EXPERIMENTAL_{feature_name.upper()}=true to enable."
                )
            
            validation = validate_experimental_feature(feature_name)
            if not validation['valid']:
                raise RuntimeError(
                    f"Experimental feature '{feature_name}' cannot be used: "
                    f"{'; '.join(validation['errors'])}"
                )
            
            # Log warnings
            for warning in validation['warnings']:
                logger.warning(f"Experimental feature '{feature_name}': {warning}")
            
            return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator


class ExperimentalFeatureManager:
    """
    Manager for experimental features with lifecycle management.
    """
    
    def __init__(self):
        self._enabled_features = set()
        self._feature_instances = {}
        self._initialization_callbacks = {}
        self._cleanup_callbacks = {}
    
    def enable_feature(self, feature_name: str, force: bool = False) -> bool:
        """
        Enable an experimental feature with validation.
        
        Parameters:
            feature_name : str
                Name of feature to enable
            force : bool
                Skip validation checks
        
        Returns:
            bool: True if successfully enabled
        """
        if not force:
            validation = validate_experimental_feature(feature_name)
            if not validation['valid']:
                raise RuntimeError(
                    f"Cannot enable feature '{feature_name}': {validation['errors']}"
                )
            for warning in validation['warnings']:
                logger.warning(f"Feature '{feature_name}': {warning}")
        
        self._enabled_features.add(feature_name)
        
        # Run initialization callback if available
        if feature_name in self._initialization_callbacks:
            try:
                self._initialization_callbacks[feature_name]()
                logger.info(f"Experimental feature '{feature_name}' initialized")
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                self._enabled_features.discard(feature_name)
                raise RuntimeError(
                    f"Failed to initialize feature '{feature_name}': {e}"
                ) from e
        
        logger.info(f"Experimental feature '{feature_name}' enabled")
        return True
    
    def disable_feature(self, feature_name: str) -> bool:
        """
        Disable an experimental feature with cleanup.
        
        Parameters:
            feature_name : str
                Name of feature to disable
        
        Returns:
            bool: True if successfully disabled
        """
        if feature_name not in self._enabled_features:
            return True
        
        # Run cleanup callback if available. A failing cleanup propagates and
        # the feature stays enabled, so it can be retried; it is never
        # reported as disabled.
        if feature_name in self._cleanup_callbacks:
            self._cleanup_callbacks[feature_name]()
            logger.info(f"Experimental feature '{feature_name}' cleaned up")
        
        self._enabled_features.discard(feature_name)
        self._feature_instances.pop(feature_name, None)
        
        logger.info(f"Experimental feature '{feature_name}' disabled")
        return True
    
    def is_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled in this manager."""
        return feature_name in self._enabled_features
    
    def get_feature(self, feature_name: str, factory: Optional[Callable] = None):
        """
        Get feature instance with lazy initialization.
        
        Parameters:
            feature_name : str
                Name of feature
            factory : Optional[Callable]
                Factory function to create feature instance
        
        Returns:
            Feature instance or None
        """
        if not self.is_enabled(feature_name):
            return None
        
        if feature_name not in self._feature_instances:
            if factory is not None:
                # A failing factory propagates; no None instance is returned.
                self._feature_instances[feature_name] = factory()
            else:
                return None
        
        return self._feature_instances.get(feature_name)
    
    def register_initialization_callback(self, feature_name: str, callback: Callable):
        """Register initialization callback for feature."""
        self._initialization_callbacks[feature_name] = callback
    
    def register_cleanup_callback(self, feature_name: str, callback: Callable):
        """Register cleanup callback for feature."""
        self._cleanup_callbacks[feature_name] = callback
    
    def get_enabled_features(self) -> List[str]:
        """Get list of currently enabled features."""
        return list(self._enabled_features)
    
    def cleanup_all(self):
        """Cleanup all enabled features.

        Every feature is attempted; features whose cleanup failed stay
        enabled and one aggregated RuntimeError names them (chained to the
        first failure).
        """
        failures = []
        for feature_name in sorted(self._enabled_features):
            try:
                self.disable_feature(feature_name)
            except Exception as error:  # noqa: BLE001 - teardown aggregation, re-raised below
                failures.append((feature_name, error))
        if failures:
            names = ", ".join(name for name, _ in failures)
            raise RuntimeError(
                f"Cleanup failed for experimental feature(s): {names}"
            ) from failures[0][1]


# Global feature manager
_feature_manager = ExperimentalFeatureManager()


def get_feature_manager() -> ExperimentalFeatureManager:
    """Get global experimental feature manager."""
    return _feature_manager


def safe_import_experimental(
    feature_name: str,
    module_name: str,
    alternative=None,
    allow_alternative: bool = False,
):
    """
    Safely import experimental module with explicit opt-in behavior.
    
    Parameters:
        feature_name : str
            Name of experimental feature
        module_name : str
            Module to import
        alternative : Any
            Alternative return value if import fails and allow_alternative is True
        allow_alternative : bool
            Whether returning the alternative value is allowed
    
    Returns:
        Imported module or alternative value if explicitly allowed
    """
    if not is_experimental_enabled(feature_name):
        msg = (
            f"Experimental feature '{feature_name}' is not enabled. "
            f"Set RELIC_EXPERIMENTAL_{feature_name.upper()}=true to enable."
        )
        if allow_alternative:
            logger.info(msg)
            return alternative
        raise RuntimeError(msg)
    
    validation = validate_experimental_feature(feature_name)
    if not validation['valid']:
        msg = (
            f"Experimental feature '{feature_name}' not available: "
            f"{'; '.join(validation['errors'])}"
        )
        if allow_alternative:
            logger.warning(msg)
            return alternative
        raise RuntimeError(msg)
    
    # Only a missing or unimportable module (ImportError) may be replaced by
    # the explicitly allowed alternative; any other import failure propagates.
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        msg = f"Failed to import experimental module '{module_name}': {e}"
        if allow_alternative:
            logger.warning(msg)
            return alternative
        raise RuntimeError(msg) from e
    logger.info(f"Successfully imported experimental module: {module_name}")
    return module


def print_experimental_status(verbose: bool = False):
    """
    Print status of experimental features.
    
    Parameters:
        verbose : bool
            Include detailed information
    """
    status = get_experimental_status()
    
    print("=== RELIC Experimental Features Status ===")
    print(f"Platform: {status['platform']['system']} ({status['platform']['platform']})")
    print(f"Global experimental flag: {'Enabled' if status['global_experimental_enabled'] else 'Disabled'}")
    print()
    
    enabled_count = sum(1 for f in status['features'].values() if f['enabled'])
    available_count = sum(1 for f in status['features'].values() if f['validation']['valid'])
    
    print(f"Features: {enabled_count} enabled, {available_count} available, {len(status['features'])} total")
    print()
    
    for feature_name, feature_info in status['features'].items():
        enabled = feature_info['enabled']
        valid = feature_info['validation']['valid']
        
        status_icon = "[OK]" if enabled and valid else "[AVAILABLE]" if valid else "[X]"
        status_text = "ENABLED" if enabled else "available" if valid else "unavailable"
        
        print(f"{status_icon} {feature_name}: {status_text}")
        if verbose:
            print(f"    Description: {feature_info['description']}")
            
            if feature_info['dependencies']:
                print("    Dependencies:")
                for dep, dep_status in feature_info['dependency_status'].items():
                    dep_icon = "[OK]" if dep_status['available'] else "[X]"
                    version_str = f" (v{dep_status['version']})" if dep_status['version'] else ""
                    print(f"      {dep_icon} {dep}{version_str}")
                    
                    if not dep_status['available'] and dep_status['installation_hint']:
                        print(f"        Install: {dep_status['installation_hint']}")
            
            validation = feature_info['validation']
            if validation['warnings']:
                print("    Warnings:")
                for warning in validation['warnings'][:3]:  # Limit to first 3
                    print(f"      [WARN] {warning}")
            
            if validation['errors']:
                print("    Errors:")
                for error in validation['errors']:
                    print(f"      [X] {error}")
            
            print()


# Auto-initialize experimental features based on environment variables
def _auto_initialize_experimental_features():
    """Initialize experimental features based on environment variables."""
    errors = []
    for feature_name in _EXPERIMENTAL_FEATURES:
        if is_experimental_enabled(feature_name):
            validation = validate_experimental_feature(feature_name)
            if validation['valid']:
                _feature_manager.enable_feature(feature_name, force=False)
            else:
                errors.append(
                    f"{feature_name}: {', '.join(validation['errors'])}"
                )
    if errors:
        raise RuntimeError(
            "Experimental feature auto-initialization failed: "
            + "; ".join(errors)
        )


# Initialize on import
_auto_initialize_experimental_features()

# Export main interface
__all__ = [
    'is_experimental_enabled',
    'validate_experimental_feature',
    'get_experimental_status',
    'experimental_feature_guard',
    'ExperimentalFeatureManager',
    'get_feature_manager',
    'safe_import_experimental',
    'print_experimental_status',
    'check_dependency_available'
]
