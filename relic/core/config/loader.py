"""
Configuration Loading System - Flexible Configuration File Management

This module provides comprehensive configuration loading capabilities
including YAML/JSON parsing, environment variable integration, 
hierarchical configuration merging, and template management.

The loader system supports multiple configuration sources and formats
with intelligent merging, validation, and error handling to provide
robust configuration management for RELIC simulations.

Architecture:
    - Multi-format configuration loading (YAML, JSON, TOML)
    - Hydra integration for structured configuration management
    - Environment variable override support
    - Hierarchical configuration composition  
    - Template-based configuration generation
    - Validation integration with automatic error reporting
    - Configuration versioning and schema evolution support

Loading Priority (highest to lowest):
    1. Command-line overrides
    2. Environment variables
    3. User configuration files
    4. System configuration files
    5. Default configuration templates

References:
    - Hydra documentation: https://hydra.cc/
    - YAML specification and best practices
    - Environment variable configuration patterns
    - Hierarchical configuration management strategies
"""

from typing import Dict, List, Any, Mapping, Optional, Tuple, Union, cast
from pathlib import Path
import os
import json
import logging
import hashlib
from copy import deepcopy

# Configuration models
from .models import RELICConfig
from .validator import ConfigValidator, ValidationSeverity

# Import Hydra for structured configuration
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
HYDRA_AVAILABLE = True

# Import YAML support
import yaml
YAML_AVAILABLE = True

# Import TOML support
import toml
TOML_AVAILABLE = True

# Import parameter manager for integration
# Delay import to avoid circular dependency
PARAMETER_MANAGER_AVAILABLE = False
_PARAMETER_MANAGER = None
_PARAMETER_SPEC = None
_PARAMETER_CATEGORY = None

def _lazy_import_parameter_system() -> None:
    """Lazily import parameter manager to avoid circular imports."""
    global PARAMETER_MANAGER_AVAILABLE, _PARAMETER_MANAGER, _PARAMETER_SPEC, _PARAMETER_CATEGORY
    
    if not PARAMETER_MANAGER_AVAILABLE:
        from ...parameters import ParameterSpec
        from ...parameters import get_global_manager
        from ...parameters.types import ParameterCategory
        _PARAMETER_MANAGER = get_global_manager
        _PARAMETER_SPEC = ParameterSpec
        _PARAMETER_CATEGORY = ParameterCategory
        PARAMETER_MANAGER_AVAILABLE = True

logger = logging.getLogger(__name__)

# Configuration file format support
SUPPORTED_FORMATS = ['.yaml', '.yml', '.json', '.toml']
DEFAULT_CONFIG_PATHS = [
    './relic_config.yaml',
    './config/relic_config.yaml', 
    './examples/configs/default.yaml',
    '~/.relic/config.yaml',
    '/etc/relic/config.yaml'
]

# Environment variable that names the configuration file.
CONFIG_PATH_ENV_VAR = "RELIC_CONFIG_PATH"


def resolve_config_source(
    config_path: Optional[Union[str, Path]] = None,
    *,
    env_var: str = CONFIG_PATH_ENV_VAR,
) -> Optional[Path]:
    """Return the configuration source by the one RELIC source rule.

    A named source (``config_path``, else a non-empty ``env_var`` environment
    variable) must exist, otherwise FileNotFoundError; it may be a file or a
    Hydra configuration directory. Without a named source the first existing
    file of ``DEFAULT_CONFIG_PATHS`` is returned, and None when none exists: the
    caller then uses the global default configuration (``RELICConfig``
    defaults). The environment and the candidates are read on every call.
    """
    named = config_path if config_path is not None else (os.environ.get(env_var) or None)
    if named is not None:
        source = Path(named).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"Configuration file not found: {source}")
        return source
    for candidate in DEFAULT_CONFIG_PATHS:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path
    return None


def configuration_environment(env_prefix: str = "RELIC") -> Tuple[Tuple[str, str], ...]:
    """The set environment variables that override RELICConfig fields, sorted."""
    field_paths = _environment_field_paths(RELICConfig, env_prefix)
    return tuple(
        (name, os.environ[name]) for name in sorted(field_paths) if name in os.environ
    )


def _environment_field_paths(model: Any, prefix: str) -> Dict[str, str]:
    """Map each environment variable name to the dotted RELICConfig field path it sets."""
    from pydantic import BaseModel

    paths: Dict[str, str] = {}

    def walk(current: Any, dotted: List[str]) -> None:
        for name, field in current.model_fields.items():
            annotation = field.annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                walk(annotation, [*dotted, name])
            else:
                parts = [*dotted, name]
                paths["_".join([prefix, *parts]).upper()] = ".".join(parts)

    walk(model, [])
    return paths


class ConfigurationError(Exception):
    """Exception raised for configuration loading errors."""


class HydraIntegration:
    """
    Hydra integration for structured configuration management.
    
    Provides seamless integration with Hydra for advanced configuration
    features including defaults composition, interpolation, and overrides.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize Hydra integration.
        
        Parameters:
            config_path : Optional[Path]
                Base configuration directory
        """
        self.config_path = config_path or Path("./config")
        self.initialized = False
        
        if HYDRA_AVAILABLE:
            self._setup_hydra()
    
    def _setup_hydra(self) -> None:
        """Set up Hydra configuration store and search paths.

        A failure here is a failure of the requested Hydra integration and
        propagates; the integration is never left silently uninitialized.
        """
        from .models import RELICConfig

        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        # Register configuration schemas with ConfigStore
        cs = ConfigStore.instance()

        # Register RELICConfig with a Hydra-compatible node.
        # RELICConfig is a Pydantic model, not a dataclass, so we register
        # a plain JSON-compatible defaults dict.
        config_node = RELICConfig().model_dump(mode="json")
        cs.store(name="relic_config", node=config_node)

        self.initialized = True
        logger.debug("Hydra integration initialized")

    def compose_config(self,
                       config_name: str,
                       overrides: Optional[List[str]] = None,
                       config_dir: Optional[Path] = None) -> Any:
        """
        Compose configuration using Hydra.

        Parameters:
            config_name : str
                Name of configuration to compose
            overrides : Optional[List[str]]
                List of override strings (e.g., "physics.carbon_cycle=true")
            config_dir : Optional[Path]
                Directory holding the primary config; defaults to this
                integration's ``config_path``.

        Returns:
            DictConfig: Composed configuration. Composition errors (missing
            directory or config, invalid overrides) propagate.
        """
        if not self.initialized:
            raise ConfigurationError("Hydra integration is not initialized")
        directory = Path(config_dir) if config_dir is not None else self.config_path
        with initialize_config_dir(config_dir=str(directory.absolute()), version_base="1.3"):
            return compose(config_name=config_name, overrides=overrides or [])
    
    def merge_with_defaults(self, 
                           config: Dict[str, Any],
                           defaults: List[str]) -> Dict[str, Any]:
        """
        Merge configuration with Hydra defaults.
        
        Parameters:
            config : Dict[str, Any]
                Base configuration
            defaults : List[str]
                List of default configurations to merge
                
        Returns:
            Dict[str, Any]: Merged configuration
        """
        if not self.initialized:
            raise ConfigurationError("Hydra integration is not initialized")

        # Convert to OmegaConf for merging; every requested default must compose.
        base_cfg = OmegaConf.create(config)
        for default in defaults:
            base_cfg = OmegaConf.merge(base_cfg, self.compose_config(default))

        # Convert back to dict through the ParameterManager bridge (the process-wide
        # manager caches the resolved mapping under its namespace).
        from ...parameters import get_global_manager

        return cast(Dict[str, Any], get_global_manager().load_omegaconf_config(
            base_cfg,
            namespace="core.config.merge",
        ))


class ConfigLoader:
    """
    Flexible configuration loader with multiple source support.
    
    Provides comprehensive configuration loading from files, environment
    variables, and programmatic sources with intelligent merging and
    validation integration.
    
    Key Features:
        - Multi-format support (YAML, JSON, TOML)
        - Hydra integration for structured configs
        - Environment variable integration with prefix support
        - Hierarchical configuration merging with precedence rules
        - Template-based configuration generation
        - Automatic validation integration
        - Configuration caching for performance
        - Error handling with detailed diagnostics
    """
    
    def __init__(self, 
                 env_prefix: str = "RELIC",
                 validate_on_load: bool = True,
                 cache_configs: bool = True,
                 use_hydra: bool = True,
                 parameter_manager: Optional[Any] = None):
        """
        Initialize configuration loader.
        
        Parameters:
            env_prefix : str
                Prefix for environment variable overrides
            validate_on_load : bool
                Whether to validate configurations after loading
            cache_configs : bool
                Whether to cache loaded configurations for performance
            use_hydra : bool
                Whether to use Hydra for configuration management
        """
        self.env_prefix = env_prefix
        self.validate_on_load = validate_on_load
        self.cache_configs = cache_configs
        self.use_hydra = use_hydra and HYDRA_AVAILABLE
        
        # Configuration cache
        self._config_cache: Dict[str, RELICConfig] = {}
        self._file_timestamps: Dict[str, float] = {}
        self._cache_hashes: Dict[str, str] = {}
        
        # Loading statistics
        self.load_count = 0
        self.cache_hits = 0
        
        # Hydra integration
        self.hydra: Optional[HydraIntegration]
        if self.use_hydra:
            self.hydra = HydraIntegration()
        else:
            self.hydra = None
        
        # Parameter manager integration (supports dependency injection)
        # Ensure parameter manager is loaded if available
        _lazy_import_parameter_system()
        self.integrate_with_parameter_manager = PARAMETER_MANAGER_AVAILABLE or (parameter_manager is not None)
        self._parameter_manager = parameter_manager
        
        # Validator
        self.validator = ConfigValidator() if validate_on_load else None
        
        logger.debug(f"ConfigLoader initialized with prefix '{env_prefix}'")
    
    def load_config(self,
                   config_path: Optional[Union[str, Path]] = None,
                   overrides: Optional[Union[List[str], Mapping[str, Any]]] = None,
                   env_overrides: bool = True) -> RELICConfig:
        """
        Load configuration with Hydra support and overrides.
        
        Parameters:
            config_path : Optional[Union[str, Path]]
                Path to configuration file or directory
            overrides : Optional[Union[List[str], Mapping[str, Any]]]
                Dotted overrides, as "key=value" strings (e.g., ["physics.carbon_cycle=false"])
                or as a mapping (e.g., {"simulation.time_step": 1800.0})
            env_overrides : bool
                Whether to apply environment variable overrides
                
        Returns:
            RELICConfig: Loaded and validated configuration
            
        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        self.load_count += 1
        if isinstance(overrides, Mapping):
            # JSON text round-trips the value type through the override parser.
            overrides = [f"{key}={json.dumps(value)}" for key, value in overrides.items()]

        # The one source rule: a named source that does not exist raises
        # FileNotFoundError here, outside the load-error wrapping below.
        source = resolve_config_source(config_path)
        try:
            # Check cache if enabled
            cache_key = self._get_cache_key(source, overrides, env_overrides)
            if self.cache_configs and cache_key in self._config_cache:
                if self._is_cache_valid(cache_key, source):
                    self.cache_hits += 1
                    logger.debug(f"Using cached configuration (hit rate: {self.cache_hits}/{self.load_count})")
                    return self._config_cache[cache_key]

            # Load base configuration
            if source is None:
                logger.info(
                    "Using the global default configuration (RELICConfig defaults): no "
                    "configuration file was named (argument or %s) and none of "
                    "DEFAULT_CONFIG_PATHS exists.",
                    CONFIG_PATH_ENV_VAR,
                )
                config_dict = RELICConfig().model_dump(mode="json", exclude={"created_at"})
                if overrides:
                    config_dict = self._apply_hydra_overrides(config_dict, overrides)
            elif source.is_dir() and self.use_hydra:
                # Use Hydra for directory-based configs
                config_dict = self._load_hydra_config(source, overrides)
            else:
                # Load single file
                config_dict = self._load_from_file(source)
                
                # Apply dotted overrides (never silently ignored)
                if overrides:
                    config_dict = self._apply_hydra_overrides(config_dict, overrides)
            
            # Apply environment variable overrides
            if env_overrides:
                env_config = self._load_from_environment()
                config_dict = self._merge_configurations(config_dict, env_config)
            
            # Create configuration object
            config = RELICConfig(**config_dict)
            
            # Load parameters to manager if enabled
            if self.integrate_with_parameter_manager:
                self._load_parameters_to_manager(config)
            
            # Validate if requested
            if self.validate_on_load and self.validator:
                validation_result = self.validator.validate(config)
                if not validation_result.is_valid:
                    error_messages = [
                        str(msg)
                        for msg in validation_result.get_messages_by_severity(ValidationSeverity.ERROR)
                    ]
                    raise ConfigurationError(
                        "Configuration validation failed:\n" + "\n".join(error_messages)
                    )
                elif validation_result.num_warnings > 0:
                    logger.warning(f"Configuration loaded with {validation_result.num_warnings} warnings")
            
            # Cache result if enabled
            if self.cache_configs:
                self._config_cache[cache_key] = config
                if source is not None and source.is_file():
                    self._file_timestamps[cache_key] = source.stat().st_mtime
                self._cache_hashes[cache_key] = self._compute_config_hash(config_dict)

            logger.info(
                "Configuration loaded successfully from %s",
                source if source is not None else "the global default configuration",
            )
            return config
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def save_config(self,
                   config: RELICConfig,
                   output_path: Union[str, Path],
                   format: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Parameters:
            config : RELICConfig
                Configuration to save
            output_path : Union[str, Path]
                Output file path
            format : Optional[str]
                Output format ('yaml', 'json', 'toml'), auto-detect if None
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format
        if format is None:
            format = output_path.suffix.lower().lstrip('.')
            if format not in ['yaml', 'yml', 'json', 'toml']:
                format = 'yaml'
        
        # Convert to dictionary
        config_dict = config.model_dump(exclude_unset=False, exclude_none=False)
        
        # Save based on format
        if format in ['yaml', 'yml']:
            self._save_yaml(config_dict, output_path)
        elif format == 'json':
            self._save_json(config_dict, output_path)
        elif format == 'toml':
            self._save_toml(config_dict, output_path)
        else:
            raise ConfigurationError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {output_path}")
    
    def export_schema(self, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Export configuration JSON schema.
        
        Parameters:
            output_path : Optional[Union[str, Path]]
                Path to save schema, returns dict if None
                
        Returns:
            Dict[str, Any]: JSON schema
        """
        # Create a default config to get schema
        default_config = RELICConfig()
        schema: Dict[str, Any] = default_config.to_schema()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(schema, f, indent=2)
            logger.info(f"Schema exported to {output_path}")
        
        return schema
    
    def save_resolved(self,
                     config: RELICConfig,
                     output_path: Union[str, Path]) -> None:
        """
        Save fully resolved configuration with all defaults expanded.
        
        Parameters:
            config : RELICConfig
                Configuration to save
            output_path : Union[str, Path]
                Output file path
        """
        # Save with all fields included
        config_dict = config.model_dump(exclude_unset=False, exclude_none=False, mode='json')
        
        output_path = Path(output_path)
        format = output_path.suffix.lower().lstrip('.')
        
        if format in ['yaml', 'yml']:
            self._save_yaml(config_dict, output_path)
        else:
            self._save_json(config_dict, output_path)
        
        logger.info(f"Resolved configuration saved to {output_path}")
    
    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------
    
    def _load_from_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from file based on format.
        
        Parameters:
            config_path : Path
                Path to configuration file
                
        Returns:
            Dict[str, Any]: Loaded configuration dictionary
        """
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        file_format = config_path.suffix.lower()
        
        if file_format in ['.yaml', '.yml']:
            return self._load_yaml(config_path)
        elif file_format == '.json':
            return self._load_json(config_path)
        elif file_format == '.toml':
            return self._load_toml(config_path)
        else:
            raise ConfigurationError(f"Unsupported configuration format: {file_format}")
    
    def _load_yaml(self, config_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not YAML_AVAILABLE:
            raise ConfigurationError("PyYAML not installed. Install with: pip install pyyaml")
        
        try:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    
    def _load_json(self, config_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                return cast(Dict[str, Any], json.load(f))
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {config_path}: {e}")
    
    def _load_toml(self, config_path: Path) -> Dict[str, Any]:
        """Load TOML configuration file."""
        if not TOML_AVAILABLE:
            raise ConfigurationError("toml not installed. Install with: pip install toml")
        
        try:
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                return toml.load(f)
        except toml.TomlDecodeError as e:
            raise ConfigurationError(f"Invalid TOML in {config_path}: {e}")
    
    def _save_yaml(self, config_dict: Dict[str, Any], output_path: Path) -> None:
        """Save configuration as YAML."""
        if not YAML_AVAILABLE:
            raise ConfigurationError("PyYAML not installed")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
    
    def _save_json(self, config_dict: Dict[str, Any], output_path: Path) -> None:
        """Save configuration as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    def _save_toml(self, config_dict: Dict[str, Any], output_path: Path) -> None:
        """Save configuration as TOML."""
        if not TOML_AVAILABLE:
            raise ConfigurationError("toml not installed")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            toml.dump(config_dict, f)
    
    def _load_hydra_config(self, 
                          config_dir: Path,
                          overrides: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load configuration using Hydra.
        
        Parameters:
            config_dir : Path
                Configuration directory
            overrides : Optional[List[str]]
                Hydra overrides
                
        Returns:
            Dict[str, Any]: Loaded configuration
        """
        if self.hydra is None:
            raise ConfigurationError(
                f"Configuration directory {config_dir} is composed by Hydra; this loader "
                "was created with use_hydra=False"
            )

        # Compose <config_dir>/config.yaml; composition errors propagate.
        cfg = self.hydra.compose_config("config", overrides, config_dir=config_dir)

        # Convert OmegaConf to dict
        return self._omegaconf_to_dict(cfg, namespace="core.config.compose")

    def _omegaconf_to_dict(self, cfg: Any, namespace: str) -> Dict[str, Any]:
        """Convert OmegaConf config to a plain dict, using ParameterManager when available."""
        _lazy_import_parameter_system()
        if self.integrate_with_parameter_manager:
            # The declared bridge converts and caches the mapping; its failures
            # propagate instead of switching to an uncached direct conversion.
            param_manager = self._parameter_manager
            if param_manager is None:
                from ...parameters import get_global_manager

                param_manager = get_global_manager()
            return cast(Dict[str, Any], param_manager.load_omegaconf_config(cfg, namespace=namespace))
        return cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    
    def _apply_hydra_overrides(self,
                              config_dict: Dict[str, Any],
                              overrides: List[str]) -> Dict[str, Any]:
        """
        Apply Hydra-style overrides to configuration.
        
        Parameters:
            config_dict : Dict[str, Any]
                Base configuration
            overrides : List[str]
                Override strings
                
        Returns:
            Dict[str, Any]: Updated configuration
        """
        if not HYDRA_AVAILABLE:
            raise ConfigurationError(
                f"Configuration overrides {list(overrides)} require OmegaConf/Hydra, which is not installed"
            )
        
        # Convert to OmegaConf
        cfg = OmegaConf.create(config_dict)
        
        # Parse and apply overrides
        for override in overrides:
            if '=' not in override:
                raise ConfigurationError(
                    f"Invalid configuration override {override!r}: expected 'dotted.key=value'"
                )
            
            key, value = override.split('=', 1)
            keys = key.split('.')
            
            # Navigate to parent and set value
            current = cfg
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Parse value by the override grammar: a JSON literal; else the words
            # true/false/null/none; else an int or float literal; else the text
            # itself (an unquoted string). ValueError (JSONDecodeError is one) is
            # the "not this literal form" signal of each step, nothing else.
            try:
                parsed_value = json.loads(value)
            except ValueError:
                if value.lower() == 'true':
                    parsed_value = True
                elif value.lower() == 'false':
                    parsed_value = False
                elif value.lower() == 'null' or value.lower() == 'none':
                    parsed_value = None
                else:
                    try:
                        parsed_value = int(value)
                    except ValueError:
                        try:
                            parsed_value = float(value)
                        except ValueError:
                            parsed_value = value
            
            current[keys[-1]] = parsed_value
        
        # Convert back to dict
        return self._omegaconf_to_dict(cfg, namespace="core.config.overrides")
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.
        
        Returns:
            Dict[str, Any]: Environment-based configuration overrides
        """
        env_config: Dict[str, Any] = {}
        # Each RELICConfig leaf field has one variable name: the prefix and the upper-case
        # field path joined by '_' (RELIC_PHYSICS_CARBON_CYCLE -> physics.carbon_cycle,
        # RELIC_DATA_IO_COMPRESSION_LEVEL -> data.io.compression_level). Other variables
        # sharing the prefix (RELIC_CONFIG_PATH, RELIC_SKIP_*, RELIC_ENABLE_*, ...) are
        # process switches owned elsewhere and are not configuration fields.
        field_paths = _environment_field_paths(RELICConfig, self.env_prefix)

        for key, value in os.environ.items():
            config_key = field_paths.get(key)
            if config_key is not None:
                # Parse value
                if value.lower() in ['true', 'false']:
                    parsed_value: Any = value.lower() == 'true'
                elif value.lower() in ['null', 'none']:
                    parsed_value = None
                else:
                    # int, else float, else the text itself; ValueError is the
                    # "not this literal form" signal of int() and float().
                    try:
                        parsed_value = int(value)
                    except ValueError:
                        try:
                            parsed_value = float(value)
                        except ValueError:
                            parsed_value = value

                # Set nested value
                keys = config_key.split('.')
                current = env_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = parsed_value
        
        if env_config:
            logger.debug(f"Loaded {len(env_config)} environment overrides")
        
        return env_config
    
    def _merge_configurations(self, 
                            base: Dict[str, Any], 
                            override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Parameters:
            base : Dict[str, Any]
                Base configuration dictionary
            override : Dict[str, Any]
                Override configuration dictionary
                
        Returns:
            Dict[str, Any]: Merged configuration
        """
        merged = deepcopy(base)
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                merged[key] = self._merge_configurations(merged[key], value)
            else:
                # Override value
                merged[key] = deepcopy(value)
        
        return merged
    
    def _get_cache_key(self,
                      config_path: Optional[Path],
                      overrides: Optional[List[str]],
                      env_overrides: bool) -> str:
        """Cache key over every input of a load: the source, the overrides and,
        when applied, the configuration environment variables."""
        key_parts = [str(config_path) if config_path is not None else "<global defaults>"]
        if overrides:
            key_parts.extend(sorted(overrides))
        if env_overrides:
            key_parts.extend(f"{name}={value}" for name, value in configuration_environment(self.env_prefix))
        return "|".join(key_parts)

    def _is_cache_valid(self, cache_key: str, config_path: Optional[Path]) -> bool:
        """Check if cached configuration is still valid."""
        if cache_key not in self._config_cache:
            return False

        # Check file modification time
        if config_path is not None and config_path.is_file() and cache_key in self._file_timestamps:
            try:
                current_mtime = config_path.stat().st_mtime
                cached_mtime = self._file_timestamps[cache_key]
                if current_mtime != cached_mtime:
                    return False
            except OSError:
                return False
        
        return True
    
    def _compute_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Compute hash of configuration for cache validation."""
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_parameters_to_manager(self, config: RELICConfig) -> None:
        """Load parameters from configuration to parameter manager."""
        # Ensure parameter manager is loaded
        _lazy_import_parameter_system()
        
        if not PARAMETER_MANAGER_AVAILABLE:
            # Allow DI usage even if module-level flag is False
            if self._parameter_manager is None:
                return
        
        param_manager = self._parameter_manager
        if param_manager is None:
            from ...parameters import get_global_manager
            param_manager = get_global_manager()

        overrides = config.parameters.parameter_overrides if config.parameters else {}
        if not overrides:
            return
        # An override must name a registered parameter: writing an unknown key would
        # create a value that no Functor reads, i.e. a setting that does nothing.
        unregistered = sorted(key for key in overrides if param_manager.get_parameter(key) is None)
        if unregistered:
            raise ConfigurationError(
                "parameters.parameter_overrides names parameters that are not registered "
                f"in the parameter database: {unregistered}"
            )
        for key, value in overrides.items():
            param_manager.set_value(key, value, validate=True, notify=False)
        logger.debug(f"Loaded {len(overrides)} parameter overrides")

    def set_parameter_manager(self, manager: Any) -> None:
        """Inject an external parameter manager instance for overrides ingestion."""
        self._parameter_manager = manager
        self.integrate_with_parameter_manager = True
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        self._file_timestamps.clear()
        self._cache_hashes.clear()
        logger.debug("Configuration cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics and performance metrics."""
        cache_hit_rate = (self.cache_hits / self.load_count * 100) if self.load_count > 0 else 0
        
        return {
            'load_count': self.load_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'cache_size': len(self._config_cache),
            'env_prefix': self.env_prefix,
            'validate_on_load': self.validate_on_load,
            'hydra_enabled': self.use_hydra,
            'parameter_manager_enabled': self.integrate_with_parameter_manager
        }


# Convenience functions for common configuration loading tasks
def load_config(config_path: Optional[Union[str, Path]] = None,
               overrides: Optional[List[str]] = None,
               validate: bool = True) -> RELICConfig:
    """
    Convenience function to load configuration with default settings.
    
    Parameters:
        config_path : Optional[Union[str, Path]]
            Path to configuration file
        overrides : Optional[List[str]]
            Hydra-style overrides
        validate : bool
            Whether to validate configuration
            
    Returns:
        RELICConfig: Loaded configuration object
    """
    loader = ConfigLoader(validate_on_load=validate)
    return loader.load_config(config_path, overrides)


def save_config(config: RELICConfig,
               output_path: Union[str, Path],
               format: Optional[str] = None) -> None:
    """
    Convenience function to save configuration.
    
    Parameters:
        config : RELICConfig
            Configuration to save
        output_path : Union[str, Path]
            Output path
        format : Optional[str]
            Output format
    """
    loader = ConfigLoader()
    loader.save_config(config, output_path, format)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.
    
    Parameters:
        *configs : Dict[str, Any]
            Configuration dictionaries to merge (later ones override earlier)
            
    Returns:
        Dict[str, Any]: Merged configuration
    """
    if not configs:
        return {}
    
    loader = ConfigLoader()
    merged = configs[0]
    
    for config in configs[1:]:
        merged = loader._merge_configurations(merged, config)
    
    return merged


def export_schema(output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Export configuration JSON schema.
    
    Parameters:
        output_path : Optional[Union[str, Path]]
            Path to save schema
            
    Returns:
        Dict[str, Any]: JSON schema
    """
    loader = ConfigLoader()
    return loader.export_schema(output_path)


# Export main classes and functions
__all__ = [
    'ConfigLoader',
    'ConfigurationError',
    'HydraIntegration',
    'load_config',
    'save_config',
    'merge_configs',
    'export_schema',
    'SUPPORTED_FORMATS',
    'DEFAULT_CONFIG_PATHS',
    'CONFIG_PATH_ENV_VAR',
    'resolve_config_source',
    'configuration_environment',
]
