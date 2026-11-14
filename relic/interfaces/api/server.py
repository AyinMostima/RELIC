"""
RELIC REST API Server

FastAPI-based REST API server providing unified access to all RELIC
functionality through HTTP endpoints. This server provides structured
JSON interfaces for parameter management, bounded optimization component
discovery, data I/O operations, and computational acceleration.

Key Features:
    - RESTful JSON API with OpenAPI documentation
    - Asynchronous request handling for long-running operations
    - Comprehensive error handling and validation
    - CORS support for web integration
    - Optional authentication and rate limiting
    - Health monitoring and metrics
    - Structured response format with diagnostics

API Structure:
    /health/*       System health and diagnostics
    /health/metrics Prometheus metrics export
    /graph/*        Graph orchestration validation and compilation
    /data/*         Data I/O operations and streaming
    /compute/*      Computational optimization and benchmarking
    /optimization/* [N] runtime-unbound optimization capability contract
    /stochastic/*   Stochastic processes and uncertainty quantification
    /config/*       Configuration management

Response Format:
    All API responses follow a consistent structure with status,
    message, data, diagnostics, and optional file paths.

Authors: Yuchong Long
"""

import sys
import logging
import time
from typing import Dict, Any, Optional, Sequence
import uvicorn

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse

from .schemas import APIResponse
from .middleware import (
    APIKeyAuthMiddleware,
    RateLimitingMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    add_process_time_header,
    build_api_auth_config,
    handle_errors,
)
from ...core.services import check_api_dependencies, load_runtime_config
from ...core.services.api_runtime import initialize_api_state, shutdown_api_state
from ...monitoring import export_prometheus_metrics
from ...parameters import get_global_manager

# Import route modules
from .routes import (
    health,
    data,
    compute,
    optimize,
    stochastic,
    config as config_routes,
    graph as graph_routes,
)

logger = logging.getLogger(__name__)

# Parameter-database keys of the request rate limit.
RATE_LIMIT_REQUESTS_KEY = "interfaces.api.rate_limit.requests_per_minute"
RATE_LIMIT_WINDOW_KEY = "interfaces.api.rate_limit.window_seconds"

# API metadata
API_VERSION = "1.0.0"
API_TITLE = "RELIC REST API"
API_DESCRIPTION = """
RELIC (Plant-Atmosphere-Vegetation Modeling) REST API

This API provides programmatic access to all RELIC functionality including:
- Parameter management and validation
- Optimization algorithm components; execution is [N] runtime-unbound until a
  canonical GraphIR/ExecutionPlan/GraphExecutor and simulator lifecycle are bound
- Data I/O operations with bounded CF variable metadata validation
- Computational acceleration (GPU, JIT, distributed computing)
- Stochastic processes and uncertainty quantification
- System health monitoring and diagnostics

## Authentication

API key authentication is enabled by default for non-health endpoints.
Provide `X-API-Key: <key>` or `Authorization: Bearer <key>`.
Configure keys via `RELIC_API_KEY`, `RELIC_API_KEYS`, or `config.security.api_auth.api_keys`.

## Rate Limiting

Rate limiting is enabled with an in-memory policy.

## Response Format

All API responses follow this structure:
```json
{
  "status": "success|error|warning",
  "message": "Human-readable description",
  "data": {...},           // Response data
  "diagnostics": {...},    // Debug information
  "metadata": {
    "timestamp": "ISO datetime",
    "execution_time": 1.23,
    "version": "1.0.0"
  },
  "paths": {              // Optional file paths
    "output": "/path/to/result.nc",
    "logs": "/path/to/logs/"
  }
}
```

## Error Handling

Errors include detailed information for debugging:
- HTTP status codes follow standard conventions
- Error responses include error type, details, and suggestions
- Validation errors provide field-specific information
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    initialize_api_state(
        app,
        load_runtime_config,
        check_api_dependencies,
    )

    yield

    shutdown_api_state()

def create_app(
    config: Optional[Dict[str, Any]] = None,
    enable_cors: bool = True,
    enable_docs: bool = True,
    debug: bool = False,
    enable_auth: Optional[bool] = None,
    api_keys: Optional[Sequence[str]] = None,
    requests_per_minute: Optional[int] = None,
) -> FastAPI:
    """
    Create FastAPI application instance
    
    Args:
        config: Optional configuration dictionary
        enable_cors: Enable CORS middleware
        enable_docs: Enable OpenAPI documentation
        debug: Enable debug mode
        requests_per_minute: Request count per rate-limit window; None reads
            interfaces.api.rate_limit.requests_per_minute

    Returns:
        FastAPI application instance
    """
    # Create FastAPI app
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
        debug=debug
    )

    app.state.debug = debug
    
    # Add middleware
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    auth_config = build_api_auth_config(config, enable_auth=enable_auth, api_keys=api_keys)
    app.state.api_auth_config = auth_config

    app.add_middleware(RequestLoggingMiddleware, log_level="INFO")
    app.add_middleware(SecurityHeadersMiddleware)
    # The rate limit lives only in the parameter database
    # (interfaces.api.rate_limit.*); an explicit requests_per_minute argument
    # replaces the database request count for this app.
    parameter_manager = get_global_manager()
    if requests_per_minute is None:
        requests_per_minute = parameter_manager.get(
            RATE_LIMIT_REQUESTS_KEY, units="dimensionless", required=True
        )
    app.add_middleware(
        RateLimitingMiddleware,
        requests_per_minute=requests_per_minute,
        window_seconds=parameter_manager.get(RATE_LIMIT_WINDOW_KEY, units="s", required=True),
    )
    app.add_middleware(APIKeyAuthMiddleware, auth_config=auth_config)

    app.middleware("http")(add_process_time_header)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return await handle_errors(request, exc)

    # HTTP exceptions (route-raised and routing 404/405) and request
    # validation errors use the same APIResponse error body.
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return await handle_errors(request, exc)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        return await handle_errors(request, exc)

    # Store configuration
    if config:
        app.state.config = config
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(graph_routes.router, prefix="/graph", tags=["Graph"])
    app.include_router(data.router, prefix="/data", tags=["Data I/O"])
    app.include_router(compute.router, prefix="/compute", tags=["Compute"])
    app.include_router(optimize.router, prefix="/optimization", tags=["Optimization"])
    app.include_router(stochastic.router, prefix="/stochastic", tags=["Stochastic"])
    app.include_router(config_routes.router, prefix="/config", tags=["Configuration"])

    @app.get("/health/metrics", include_in_schema=False)
    async def metrics_endpoint() -> PlainTextResponse:
        """Export Prometheus-compatible metrics."""
        payload, content_type = export_prometheus_metrics()
        return PlainTextResponse(content=payload, media_type=content_type)
    
    # Root endpoint
    @app.get("/", response_model=APIResponse)
    async def root():
        """API root endpoint with basic information"""
        return APIResponse(
            status="success",
            message="RELIC REST API is running",
            data={
                "title": API_TITLE,
                "version": API_VERSION,
                "docs": "/docs" if enable_docs else None,
                "endpoints": {
                    "health": "/health",
                    "graph": "/graph",
                    "data": "/data",
                    "compute": "/compute", 
                    "optimization": "/optimization",
                    "stochastic": "/stochastic",
                    "config": "/config"
                },
                "capabilities": {
                    "optimization": {
                        "status": "[N]",
                        "runtime_ready": False,
                        "runtime_blocker": "runtime_unbound",
                    }
                }
            },
            metadata={
                "timestamp": time.time(),
                "uptime": time.time() - getattr(app.state, 'startup_time', time.time())
            }
        )
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=API_TITLE,
            version=API_VERSION,
            description=API_DESCRIPTION,
            routes=app.routes,
        )
        
        # Add custom info
        openapi_schema["info"]["x-logo"] = {
            "url": "/static/logo.png"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app

def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    config: Optional[Dict[str, Any]] = None,
    log_level: str = "info",
    reload: bool = False
):
    """
    Run the API server using uvicorn
    
    Args:
        host: Host address
        port: Port number
        config: Optional configuration
        log_level: Logging level
        reload: Enable auto-reload
    """
    # Create app
    app = create_app(
        config=config,
        debug=(log_level.lower() == "debug"),
        enable_cors=True,
        enable_docs=True
    )
    
    logger.info("Starting RELIC API server on %s:%s", host, port)
    logger.info("API Documentation: http://%s:%s/docs", host, port)
    logger.info("Health Check: http://%s:%s/health", host, port)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            reload=reload,
            reload_dirs=["relic/interfaces/api"] if reload else None
        )
        return True
    except (OSError, RuntimeError, ValueError) as e:
        # Entry-point boundary: socket/bind failures, event-loop state errors,
        # and invalid server settings are reported and main() exits non-zero.
        logger.error("Failed to start server: %s", e)
        return False

def main():
    """Main entry point for API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RELIC REST API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--no-docs", action="store_true", help="Disable API documentation")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        from ...core.services.web_runtime import load_interface_config
        try:
            config = load_interface_config(
                args.config,
                namespace="interfaces.api.server",
            )
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Failed to load config file %s: %s", args.config, exc)
            return 1
    
    # Run server
    success = run_server(
        host=args.host,
        port=args.port,
        config=config,
        log_level=args.log_level,
        reload=args.reload
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
