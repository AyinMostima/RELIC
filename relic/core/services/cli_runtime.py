from __future__ import annotations

import logging
import os
import sys
import warnings
import importlib
from typing import Any, Callable, List, Optional, Tuple

from relic.core import device_policy

def check_python_version(min_version: str) -> None:
    """Ensure Python version consistency."""
    version_info = sys.version_info
    required = tuple(map(int, min_version.split('.')))
    if version_info < required:
        print(f"Error: RELIC requires Python {min_version} or higher.")
        print(f"Current Python version: {version_info.major}.{version_info.minor}.{version_info.micro}")
        sys.exit(1)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on command line options."""
    if quiet:
        log_level = logging.WARNING
    elif verbose:
        log_level = logging.DEBUG
    else:
        # Unset RELIC_LOG_LEVEL means INFO (the documented CLI default); a set
        # value must name a logging level, otherwise it raises ValueError.
        env_level = os.getenv('RELIC_LOG_LEVEL', 'INFO').upper()
        log_level = logging.getLevelName(env_level)
        if not isinstance(log_level, int):
            raise ValueError(
                f"RELIC_LOG_LEVEL={env_level!r} is not a logging level "
                "(DEBUG, INFO, WARNING, ERROR, CRITICAL)."
            )

    logging.getLogger().setLevel(log_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)


def handle_warnings(no_warnings: bool = False) -> None:
    """Configure warning handling."""
    if no_warnings or os.getenv('RELIC_NO_WARNINGS', '').lower() in ('1', 'true', 'yes'):
        warnings.filterwarnings('ignore')
    else:
        warnings.filterwarnings('default', category=DeprecationWarning)
        warnings.filterwarnings('default', category=PendingDeprecationWarning)


def get_system_info_lines(
    cli_version: str,
    config_system_available: bool,
    load_runtime_config: Optional[Callable[[], Tuple[Any, Any, Any]]] = None,
) -> List[str]:
    """Build system and version information lines for CLI output."""
    import platform

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("RELIC System Information")
    lines.append("=" * 60)
    lines.append(f"CLI Version:        {cli_version}")
    lines.append(f"Python Version:     {sys.version}")
    lines.append(f"Platform:           {platform.platform()}")
    lines.append(f"Architecture:       {platform.architecture()[0]}")
    lines.append(f"Processor:          {platform.processor()}")

    if importlib.util.find_spec("torch") is not None:
        import torch
        lines.append(f"PyTorch Version:    {torch.__version__}")
        if device_policy.is_cuda_available():
            lines.append(f"CUDA Version:       {torch.version.cuda}")
            lines.append(f"CUDA Devices:       {torch.cuda.device_count()}")
    else:
        lines.append("PyTorch:            Not available")

    if importlib.util.find_spec("numpy") is not None:
        import numpy as np
        lines.append(f"NumPy Version:      {np.__version__}")
    else:
        lines.append("NumPy:              Not available")

    if config_system_available and load_runtime_config is not None:
        # load_runtime_config reports no file exactly when it used the global
        # default configuration (RELICConfig defaults); the log level shown is
        # the effective root level.
        _, _, config_path = load_runtime_config()
        config_file = (
            str(config_path)
            if config_path is not None
            else "global default configuration (RELICConfig defaults)"
        )
        log_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())
        lines.append(f"Config File:        {config_file}")
        lines.append(f"Log Level:          {log_level}")

    lines.append("=" * 60)
    return lines


def print_system_info(
    cli_version: str,
    config_system_available: bool,
    load_runtime_config: Optional[Callable[[], Tuple[Any, Any, Any]]] = None,
) -> None:
    """Print system and version information."""
    for line in get_system_info_lines(
        cli_version,
        config_system_available,
        load_runtime_config,
    ):
        print(line)


def get_experimental_features(
    env_vars: Optional[List[str]] = None,
) -> List[str]:
    """Return enabled experimental feature env vars."""
    # None means the documented set of experimental flags; an explicit list
    # (including an empty one) is used as given.
    experimental_vars = list(env_vars) if env_vars is not None else [
        'RELIC_EXPERIMENTAL_CLI',
        'RELIC_EXPERIMENTAL_WEB',
        'RELIC_EXPERIMENTAL_API',
        'RELIC_EXPERIMENTAL_STREAMING'
    ]

    active_experimental: List[str] = []
    for var in experimental_vars:
        if os.getenv(var, '').lower() in ('1', 'true', 'yes', 'on'):
            active_experimental.append(var)

    return active_experimental


def print_experimental_features(env_vars: Optional[List[str]] = None) -> None:
    """Print enabled experimental feature flags."""
    active_experimental = get_experimental_features(env_vars)
    if active_experimental:
        print("\nExperimental Features Enabled:")
        for var in active_experimental:
            print(f"  - {var}")
        print("Note: Experimental features may be unstable or change without notice.")
