"""
Configuration Loader for RELIC Computation Layer

This module provides unified configuration loading that integrates:
- Hydra structured configurations (YAML/OmegaConf)
- ParameterManager parameter database
- Environment variables and CLI overrides
- Factory instantiation helpers

Design Philosophy:
    - Unified Interface: Single load_config() entry point
    - Hierarchical Merge: Hydra + ParameterManager + Runtime overrides
    - Type Safety: Pydantic validation before instantiation
    - Environment Awareness: Auto-detection of device/precision
    - Lazy Instantiation: Load config first, instantiate on demand

Usage Example:
    >>> from relic.computation.config import load_computation_config
    >>>
    >>> # Load from Hydra config
    >>> config = load_computation_config("conf/operators/gradient.yaml")
    >>>
    >>> # Instantiate operator from config
    >>> operator = config.instantiate_operator("gradient")
    >>>
    >>> # Override parameters
    >>> config.set_parameter("gradient.dx", 50.0, units="m")
    >>> operator = config.instantiate_operator("gradient")

References:
    - Hydra Documentation: https://hydra.cc/
    - OmegaConf: https://omegaconf.readthedocs.io/
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field

import torch
from relic.core import device_policy

# Import Hydra/OmegaConf (required in the new computation config system)
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
HYDRA_AVAILABLE = True

# Import factories
from ..operators.factory import OperatorFactory
from ..operators.registry import OperatorRegistry
from ..solvers.factory import SolverFactory
from ..solvers.registry import SolverRegistry

# Import ParameterManager accessors (required in the new config system)
from ...parameters import get_global_manager, ParameterSpec, ParameterCategory
PARAMETER_MANAGER_AVAILABLE = True

logger = logging.getLogger(__name__)


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge dictionaries where override values take precedence."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _insert_nested(mapping: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Insert a dotted key into a nested mapping."""
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return
    cursor = mapping
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _extract_hydra_section(
    cfg: Any,
    section_name: str,
    param_manager: Optional[Any],
    namespace: str,
) -> Dict[str, Any]:
    """Extract one Hydra section, optionally via ParameterManager loader."""
    if not hasattr(cfg, section_name):
        return {}
    section = getattr(cfg, section_name)
    if param_manager is not None and hasattr(param_manager, "load_omegaconf_config"):
        return param_manager.load_omegaconf_config(section, namespace=namespace)
    container = OmegaConf.to_container(section, resolve=True)
    return container if isinstance(container, dict) else {}


def _load_hydra_config(
    config_path: Optional[Union[str, Path]],
    config_name: Optional[str],
    overrides: Optional[List[str]],
) -> tuple[Dict[str, Any], Dict[str, Any], torch.device, torch.dtype, bool]:
    """Load computation config from Hydra and return normalized pieces."""
    operators_config: Dict[str, Any] = {}
    solvers_config: Dict[str, Any] = {}
    device = torch.device("cpu")
    dtype = torch.float32
    validation_enabled = True

    if config_path is None:
        # No Hydra configuration requested: the dataclass defaults above apply.
        # Overrides address a Hydra composition and are rejected without one.
        if overrides:
            raise ValueError(
                f"Configuration overrides {list(overrides)} require a Hydra config_path"
            )
        return operators_config, solvers_config, device, dtype, validation_enabled

    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        config_dir = str(config_path.absolute()) if isinstance(config_path, Path) else str(config_path)
        initialize_config_dir(config_dir=config_dir)

        cfg = compose(
            config_name=config_name or "config",
            overrides=overrides or [],
        )
        param_manager = get_global_manager() if PARAMETER_MANAGER_AVAILABLE else None

        operators_config = _extract_hydra_section(
            cfg,
            "operators",
            param_manager,
            "computation.operators",
        )
        solvers_config = _extract_hydra_section(
            cfg,
            "solvers",
            param_manager,
            "computation.solvers",
        )

        if hasattr(cfg, "device"):
            device = torch.device(cfg.device)
        if hasattr(cfg, "dtype"):
            dtype = getattr(torch, cfg.dtype)
        if hasattr(cfg, "validation") and hasattr(cfg.validation, "enabled"):
            validation_enabled = cfg.validation.enabled

        logger.info("Loaded config from Hydra: %s/%s", config_dir, config_name)
    finally:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

    return operators_config, solvers_config, device, dtype, validation_enabled


def _namespace_defaults_from_manager(param_manager: Any, namespace: str) -> Dict[str, Any]:
    """Collect nested namespace defaults from ParameterManager APIs."""
    defaults: Dict[str, Any] = {}

    if hasattr(param_manager, "get_namespace_dict"):
        cached = param_manager.get_namespace_dict(namespace)
        if isinstance(cached, dict):
            defaults = _deep_merge_dict(defaults, cached)

    if hasattr(param_manager, "get_namespace_parameters"):
        values = param_manager.get_namespace_parameters(
            namespace=namespace,
            include_subnamespaces=True,
            return_specs=False,
        )
        if isinstance(values, dict):
            for full_key, value in values.items():
                if not isinstance(full_key, str):
                    continue
                short_key = (
                    full_key[len(namespace) + 1 :]
                    if full_key.startswith(f"{namespace}.")
                    else full_key
                )
                _insert_nested(defaults, short_key, value)

    return defaults


def _enrich_from_parameter_manager(
    operators_config: Dict[str, Any],
    solvers_config: Dict[str, Any],
    use_parameter_manager: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Merge ParameterManager defaults into Hydra/override configs."""
    if not (use_parameter_manager and PARAMETER_MANAGER_AVAILABLE):
        return operators_config, solvers_config

    # The enrichment was requested: manager failures propagate.
    param_manager = get_global_manager()
    operator_defaults = _namespace_defaults_from_manager(param_manager, "computation.operators")
    solver_defaults = _namespace_defaults_from_manager(param_manager, "computation.solvers")
    operators_merged = _deep_merge_dict(operator_defaults, operators_config)
    solvers_merged = _deep_merge_dict(solver_defaults, solvers_config)
    logger.debug("Enriched config from ParameterManager")
    return operators_merged, solvers_merged


def _apply_environment_overrides(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    """Apply RELIC_* environment variable overrides."""
    env_device = os.getenv("RELIC_DEVICE")
    if env_device:
        device = torch.device(env_device)
        logger.info("Device overridden from environment: %s", device)

    env_dtype = os.getenv("RELIC_DTYPE")
    if env_dtype:
        dtype = getattr(torch, env_dtype)
        logger.info("Dtype overridden from environment: %s", dtype)

    return device, dtype


@dataclass
class ComputationConfig:
    """
    Unified configuration for computation layer (operators and solvers).

    Attributes
    ----------
    operators : Dict[str, Any]
        Operator configurations
    solvers : Dict[str, Any]
        Solver configurations
    device : torch.device
        Default computation device
    dtype : torch.dtype
        Default data type
    validation_enabled : bool
        Whether to validate parameters
    parameter_namespace : str
        Base namespace for parameters
    """
    operators: Dict[str, Any] = field(default_factory=dict)
    solvers: Dict[str, Any] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32
    validation_enabled: bool = True
    parameter_namespace: str = "computation"

    def instantiate_operator(self, operator_key: str, **overrides: Any) -> Any:
        """
        Instantiate operator from config.

        Parameters
        ----------
        operator_key : str
            Operator identifier in config (e.g., "gradient")
        **overrides
            Runtime parameter overrides

        Returns
        -------
        OperatorBase
            Instantiated operator

        Examples
        --------
        >>> config = ComputationConfig(...)
        >>> op = config.instantiate_operator("gradient", dx=50.0)
        """
        if operator_key not in self.operators:
            raise KeyError(f"Operator '{operator_key}' not found in config")

        op_config = self.operators[operator_key].copy()

        # Apply overrides to a new 'config' mapping: the top-level copy above
        # still shares the stored one, and an in-place update would carry these
        # overrides into every later instantiation.
        op_config['config'] = {**op_config.get('config', {}), **overrides}

        # Set device and dtype if not specified
        if 'device' not in op_config:
            op_config['device'] = self.device
        if 'dtype' not in op_config:
            op_config['dtype'] = self.dtype

        # Use OperatorFactory
        return OperatorFactory.create_from_config(op_config)

    def instantiate_solver(
        self,
        solver_key: str,
        operator: Optional[Any] = None,
        **overrides: Any,
    ) -> Any:
        """
        Instantiate solver from config.

        Parameters
        ----------
        solver_key : str
            Solver identifier in config (e.g., "backward_euler")
        operator : Optional[Any]
            Operator instance for operator-based solvers
        **overrides
            Runtime parameter overrides

        Returns
        -------
        SolverBase
            Instantiated solver

        Examples
        --------
        >>> config = ComputationConfig(...)
        >>> solver = config.instantiate_solver("backward_euler", rtol=1e-8)
        """
        if solver_key not in self.solvers:
            raise KeyError(f"Solver '{solver_key}' not found in config")

        solver_config = self.solvers[solver_key].copy()

        # Apply overrides to a new 'config' mapping (see instantiate_operator).
        solver_config['config'] = {**solver_config.get('config', {}), **overrides}

        # Set device and dtype if not specified
        if 'device' not in solver_config:
            solver_config['device'] = self.device
        if 'dtype' not in solver_config:
            solver_config['dtype'] = self.dtype

        # Use SolverFactory
        return SolverFactory.create_from_config(solver_config, operator=operator)

    def get_parameter(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get parameter value from configuration or ParameterManager.

        Parameters
        ----------
        key : str
            Parameter key (dot-notation)
        default : Optional[Any]
            Default value if not found

        Returns
        -------
        Any
            Parameter value
        """
        # ParameterManager first; None means absent, and a failed lookup
        # propagates (an explicit default covers absence only).
        if PARAMETER_MANAGER_AVAILABLE:
            param_manager = get_global_manager()
            full_key = f"{self.parameter_namespace}.{key}"
            value = param_manager.get(full_key, default=None)
            if value is not None:
                return value

        if default is None and self.validation_enabled:
            raise KeyError(
                f"Required computation parameter '{self.parameter_namespace}.{key}' not found "
                "and no explicit default provided."
            )

        return default

    def set_parameter(self, key: str, value: Any, units: Optional[str] = None) -> None:
        """
        Set parameter value in ParameterManager.

        Parameters
        ----------
        key : str
            Parameter key (dot-notation)
        value : Any
            Parameter value
        units : Optional[str]
            Physical units of ``value``; required (no unit is assumed).

        Raises
        ------
        ValueError
            If ``units`` is missing. Manager failures propagate.
        """
        if not units:
            raise ValueError(
                f"set_parameter('{key}') requires units; no unit is assumed for a runtime override"
            )
        param_manager = get_global_manager()
        full_key = f"{self.parameter_namespace}.{key}"

        spec = ParameterSpec(
            key=full_key,
            value=value,
            units=units,
            dtype=str(type(value).__name__),
            # A runtime override declares no limits: explicitly unbounded.
            bounds=(-float("inf"), float("inf")),
            category=ParameterCategory.NUMERICAL_PARAMETER,
            description=f"Runtime override for {key}"
        )

        param_manager.set_parameter(spec, tier="calibrated")
        logger.info(f"Set parameter {full_key} = {value} {units}")


def load_computation_config(
    config_path: Optional[Union[str, Path]] = None,
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    use_parameter_manager: bool = True
) -> ComputationConfig:
    """
    Load unified computation configuration from multiple sources.

    Load priority (later overrides earlier):
    1. ParameterManager defaults
    2. Hydra YAML config
    3. Environment variables
    4. CLI overrides

    Parameters
    ----------
    config_path : Optional[Union[str, Path]]
        Path to Hydra config directory
    config_name : Optional[str]
        Name of config file (without .yaml extension)
    overrides : Optional[List[str]]
        CLI-style overrides (e.g., ["operators.gradient.config.dx=50.0"])
    use_parameter_manager : bool
        Whether to enrich config from ParameterManager

    Returns
    -------
    ComputationConfig
        Unified configuration object

    Examples
    --------
    >>> # Load from Hydra config
    >>> config = load_computation_config(
    ...     config_path="conf",
    ...     config_name="operators",
    ...     overrides=["operators.gradient.config.dx=50.0"]
    ... )
    >>>
    >>> # Load with ParameterManager defaults
    >>> config = load_computation_config(use_parameter_manager=True)
    >>>
    >>> # Instantiate operator
    >>> op = config.instantiate_operator("gradient")
    """
    (
        operators_config,
        solvers_config,
        device,
        dtype,
        validation_enabled,
    ) = _load_hydra_config(config_path, config_name, overrides)
    operators_config, solvers_config = _enrich_from_parameter_manager(
        operators_config,
        solvers_config,
        use_parameter_manager,
    )
    device, dtype = _apply_environment_overrides(device, dtype)

    # Create unified config
    config = ComputationConfig(
        operators=operators_config,
        solvers=solvers_config,
        device=device,
        dtype=dtype,
        validation_enabled=validation_enabled,
        parameter_namespace="computation"
    )

    return config


def auto_detect_device() -> torch.device:
    """
    Auto-detect optimal computation device.

    Returns
    -------
    torch.device
        Detected device (CUDA if available, else CPU)

    Examples
    --------
    >>> device = auto_detect_device()
    >>> print(device)
    device(type='cuda', index=0)
    """
    device = device_policy.resolve_device(module_name=__name__, default="auto")
    if device.type == "cuda":
        logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, using CPU")

    return device


def validate_config(config: ComputationConfig) -> List[str]:
    """
    Validate configured provider identities against runtime registries.

    This boundary validates that every named operator and solver has a real
    runtime provider. Provider-specific parameters are validated by the
    canonical Factory, Pydantic, and ParameterManager path during
    instantiation; this function does not maintain a second parameter schema.

    Parameters
    ----------
    config : ComputationConfig
        Configuration to validate

    Returns
    -------
    List[str]
        List of validation errors (empty if valid)

    Examples
    --------
    >>> config = load_computation_config(...)
    >>> errors = validate_config(config)
    >>> if errors:
    ...     print("Validation errors:", errors)
    """
    errors = []

    # Validate operators
    for op_key, op_config in config.operators.items():
        if 'operator_name' not in op_config:
            errors.append(f"Operator '{op_key}' missing 'operator_name' field")
            continue

        operator_name = op_config['operator_name']
        if not isinstance(operator_name, str) or not OperatorRegistry.is_registered(
            operator_name
        ):
            errors.append(
                f"Operator '{op_key}' references unregistered operator "
                f"'{operator_name}'"
            )

    # Validate solvers
    for solver_key, solver_config in config.solvers.items():
        if 'solver_name' not in solver_config:
            errors.append(f"Solver '{solver_key}' missing 'solver_name' field")
            continue

        solver_name = solver_config['solver_name']
        if not isinstance(solver_name, str) or not SolverRegistry.is_registered(
            solver_name
        ):
            errors.append(
                f"Solver '{solver_key}' references unregistered solver "
                f"'{solver_name}'"
            )

    return errors


# Export main functions
__all__ = [
    'ComputationConfig',
    'load_computation_config',
    'auto_detect_device',
    'validate_config'
]
