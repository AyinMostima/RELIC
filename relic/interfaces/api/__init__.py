"""
RELIC REST API Interface

FastAPI-based REST API providing unified access to all RELIC functionality
through HTTP endpoints with structured JSON responses.

Key Features:
    - RESTful design with OpenAPI documentation
    - Asynchronous request handling
    - Comprehensive error handling and validation
    - CORS support for web integration
    - Structured response format
    - Health monitoring and diagnostics

Main Components:
    server          FastAPI application server
    schemas         Request/response data models
    middleware      Custom middleware for logging, errors, security
    dependencies    Dependency injection for shared resources
    routes/         API endpoint modules organized by functionality

Usage:
    # Start API server programmatically
    from relic.interfaces.api.server import create_app, run_server
    
    app = create_app()
    run_server(host="127.0.0.1", port=8000)
    
    # Or from command line
    python -m relic.interfaces.api.server --host 127.0.0.1 --port 8000

API Documentation:
    When running, visit http://localhost:8000/docs for interactive documentation
"""

from .server import create_app, run_server, main

__all__ = ['create_app', 'run_server', 'main']
__version__ = "1.0.0"
