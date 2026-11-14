"""
RELIC Interfaces Layer

Unified interface layer providing CLI, API, Web, and Notebook access to all
RELIC functionality. This layer provides consistent, user-friendly interfaces
to the underlying scientific computing capabilities.

Architecture:
    cli/        Command-line interface with subcommands
    api/        REST API using FastAPI
    web/        Web interface with visualization
    notebooks/  Jupyter notebook integration

Features:
    - Unified configuration management
    - Experimental feature isolation
    - Comprehensive error handling and validation
    - Multi-format output support

Usage:
    # CLI usage
    from relic.interfaces.cli import main
    
    # API usage
    from relic.interfaces.api import create_app
    
    # Web interface
    from relic.interfaces.web import create_web_app

Authors: Yuchong Long
"""

__version__ = "1.0.0"
