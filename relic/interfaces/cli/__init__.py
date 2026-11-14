"""
RELIC CLI Interface

Unified command-line interface for all RELIC operations.

This module provides the main entry point and common utilities for
CLI operations across all subsystems.

Main entry points:
    main()          Main CLI entry point
    cli_entry_point()  Alternative entry point

Available submodules:
    main            Main CLI dispatcher
    utils           Common CLI utilities  
    healthcheck     System health diagnostics
    compute         Computational optimization
    params          Parameter management
    optimization    Optimization algorithms
    data            Data I/O operations
    stochastic      Stochastic processes

Usage:
    python -m relic.interfaces.cli.main <command>
    
    # Or programmatically:
    from relic.interfaces.cli.main import main
    sys.exit(main())
"""

from typing import Any

__all__ = ['main', 'cli_entry_point']


def main(*args: Any, **kwargs: Any) -> int:
    """
    Lazily dispatch to the CLI main entry point.

    Avoid importing ``relic.interfaces.cli.main`` at package import time so
    ``python -m relic.interfaces.cli.main`` does not trigger runpy warnings.
    """
    from .main import main as _main
    return _main(*args, **kwargs)


def cli_entry_point(*args: Any, **kwargs: Any) -> int:
    """
    Lazily dispatch to the alternative CLI entry point.
    """
    from .main import cli_entry_point as _cli_entry_point
    return _cli_entry_point(*args, **kwargs)
