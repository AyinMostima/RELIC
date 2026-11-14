"""
RELIC Web Interface

FastAPI-based web interface providing user-friendly access to RELIC
functionality through HTML pages with basic interactivity.

Key Features:
    - System health monitoring dashboard
    - Data file inspection and browsing
    - Computation benchmarking results display
    - Configuration viewing and validation
    - Minimal JavaScript for basic interactivity
    - Responsive design for different screen sizes
    - Integration with REST API endpoints

Architecture:
    - FastAPI app with Jinja2 templates
    - Static resources (CSS, JS, images)
    - Reuse of API routes for data access
    - Optional enhanced features with experimental flag

Authors: Yuchong Long
"""

import os
import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from ...core.services.web_runtime import (
    resolve_web_experimental_flag,
    ensure_web_assets,
    build_health_payload,
    load_web_config,
)

# fastapi is required: the web app mounts the API app (relic.interfaces.api),
# so a missing fastapi raises the original ImportError at import.
from fastapi import FastAPI, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..api.server import create_app as create_api_app
from ...validation.healthcheck import STATUS_DEGRADED, STATUS_HEALTHY, HealthChecker

# Overall health statuses for which the /status endpoint answers HTTP 200.
_SERVING_HEALTH_STATUSES = frozenset({STATUS_HEALTHY, STATUS_DEGRADED})

logger = logging.getLogger(__name__)

# Get templates and static directories
CURRENT_DIR = Path(__file__).parent
TEMPLATES_DIR = CURRENT_DIR / "templates"
STATIC_DIR = CURRENT_DIR / "static"

def create_web_app(
    config: Optional[Dict[str, Any]] = None,
    enable_experimental: Optional[bool] = None,
    enable_cors: bool = True,
    debug: bool = False
) -> FastAPI:
    """
    Create FastAPI web application
    
    Args:
        config: Optional configuration dictionary
        enable_experimental: Enable experimental features
        enable_cors: Enable CORS middleware
        debug: Enable debug mode
        
    Returns:
        FastAPI application instance
    """
    # Check experimental features
    enable_experimental = resolve_web_experimental_flag(enable_experimental)
    
    # Create FastAPI app
    app = FastAPI(
        title="RELIC Web Interface",
        description="Web interface for RELIC biogeochemical modeling system",
        version="1.0.0",
        debug=debug
    )
    
    # Setup templates
    ensure_web_assets(TEMPLATES_DIR, STATIC_DIR)
    
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    
    # Store configuration
    if config:
        app.state.config = config
    app.state.experimental = enable_experimental
    # Health checks run with the web app's own configuration, passed
    # explicitly to the HealthChecker (an app without configuration checks
    # with an empty mapping).
    health_config: dict[str, Any] = dict(config) if config is not None else {}
    
    # Include API routes for data access
    api_app = create_api_app(config=config, enable_cors=False, enable_docs=False, debug=debug)
    app.mount("/api", api_app, name="api")
    
    # Web page routes
    @app.get("/", response_class=HTMLResponse)  # type: ignore[misc]
    async def home_page(request: Request) -> Any:
        """Home page with system overview"""
        try:
            health_data = build_health_payload(HealthChecker, health_config, detailed=False)

            context = {
                "request": request,
                "title": "RELIC System Overview",
                "health_status": health_data['overall_status'],
                "subsystems": health_data['subsystems'],
                "experimental_enabled": enable_experimental
            }
            
            return templates.TemplateResponse("home.html", context)
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            # Page boundary: a health check that could not be completed renders
            # the error page with HTTP 503 so it never reads as a successful page.
            logger.error(f"Error rendering home page: {e}")
            context = {
                "request": request,
                "title": "RELIC System Overview",
                "error": str(e),
                "experimental_enabled": enable_experimental
            }
            return templates.TemplateResponse("error.html", context, status_code=503)
    
    @app.get("/health", response_class=HTMLResponse)  # type: ignore[misc]
    async def health_page(request: Request) -> Any:
        """System health monitoring dashboard"""
        try:
            health_data = build_health_payload(HealthChecker, health_config, detailed=True)

            context = {
                "request": request,
                "title": "System Health Dashboard",
                "health_data": health_data,
                "experimental_enabled": enable_experimental
            }
            
            return templates.TemplateResponse("health.html", context)
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            # Page boundary: a health check that could not be completed renders
            # the error page with HTTP 503 so it never reads as a successful page.
            logger.error(f"Error rendering health page: {e}")
            context = {
                "request": request,
                "title": "System Health Dashboard",
                "error": str(e),
                "experimental_enabled": enable_experimental
            }
            return templates.TemplateResponse("error.html", context, status_code=503)
    
    @app.get("/data", response_class=HTMLResponse)  # type: ignore[misc]
    async def data_page(request: Request) -> Any:
        """Data file inspection and browsing"""
        context = {
            "request": request,
            "title": "Data File Manager",
            "experimental_enabled": enable_experimental
        }
        
        if enable_experimental:
            return templates.TemplateResponse("data.html", context)
        else:
            context["message"] = "Data management interface requires experimental features to be enabled"
            context["instruction"] = "Set RELIC_EXPERIMENTAL_WEB=1 to enable"
            return templates.TemplateResponse("feature_disabled.html", context)
    
    @app.get("/compute", response_class=HTMLResponse)  # type: ignore[misc]
    async def compute_page(request: Request) -> Any:
        """Computation benchmarking and monitoring"""
        context = {
            "request": request,
            "title": "Computation Monitor",
            "experimental_enabled": enable_experimental
        }
        
        return templates.TemplateResponse("compute.html", context)
    
    @app.get("/config", response_class=HTMLResponse)  # type: ignore[misc]
    async def config_page(request: Request) -> Any:
        """Configuration viewer and validator"""
        if enable_experimental:
            context = {
                "request": request,
                "title": "Configuration Manager",
                "experimental_enabled": enable_experimental
            }
            return templates.TemplateResponse("config.html", context)
        else:
            context = {
                "request": request,
                "title": "Configuration Manager",
                "message": "Configuration management interface requires experimental features",
                "instruction": "Set RELIC_EXPERIMENTAL_WEB=1 to enable",
                "experimental_enabled": enable_experimental
            }
            return templates.TemplateResponse("feature_disabled.html", context)
    
    # API endpoint for getting system status (used by JavaScript)
    @app.get("/status")  # type: ignore[misc]
    async def get_status() -> Any:
        """Get current system status for dynamic updates.

        HTTP 200 only when the overall status is HEALTHY or DEGRADED; any
        other status (UNHEALTHY, unavailable, error) answers 503 with the
        same payload, so an unhealthy system never reads as success.
        """
        try:
            payload = build_health_payload(HealthChecker, health_config, detailed=True)
            overall_status = str(payload["overall_status"]).upper()
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            # Status boundary: a health check that could not be completed is
            # reported as unavailable (HTTP 503) with its error.
            return JSONResponse(
                status_code=503,
                content={"error": str(e)}
            )
        http_status = 200 if overall_status in _SERVING_HEALTH_STATUSES else 503
        return JSONResponse(status_code=http_status, content=jsonable_encoder(payload))
    
    # Error handlers
    @app.exception_handler(404)  # type: ignore[misc]
    async def not_found_handler(request: Request, exc: HTTPException) -> Any:
        """Custom 404 page"""
        context = {
            "request": request,
            "title": "Page Not Found",
            "message": "The requested page could not be found",
            "experimental_enabled": enable_experimental
        }
        return templates.TemplateResponse("error.html", context, status_code=404)
    
    @app.exception_handler(500)  # type: ignore[misc]
    async def server_error_handler(request: Request, exc: HTTPException) -> Any:
        """Custom 500 page"""
        context = {
            "request": request,
            "title": "Internal Server Error",
            "message": "An internal server error occurred",
            "experimental_enabled": enable_experimental
        }
        return templates.TemplateResponse("error.html", context, status_code=500)
    
    return app

def run_web_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    config: Optional[Dict[str, Any]] = None,
    log_level: str = "info",
    reload: bool = False
) -> bool:
    """
    Run the web server using uvicorn
    
    Args:
        host: Host address
        port: Port number
        config: Optional configuration
        log_level: Logging level
        reload: Enable auto-reload
    """
    if importlib.util.find_spec("uvicorn") is None:
        logger.error("uvicorn not available. Install with: pip install uvicorn")
        return False
    uvicorn = importlib.import_module("uvicorn")
    
    # Create app
    app = create_web_app(
        config=config,
        debug=(log_level.lower() == "debug"),
        enable_cors=True
    )
    
    logger.info("Starting RELIC Web Interface on %s:%s", host, port)
    logger.info("Home Page: http://%s:%s/", host, port)
    logger.info("Health Dashboard: http://%s:%s/health", host, port)
    logger.info("API Docs: http://%s:%s/api/docs", host, port)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level.lower(),
            reload=reload,
            reload_dirs=["relic/interfaces/web"] if reload else None
        )
        return True
    except (OSError, RuntimeError, ValueError) as e:
        # Entry-point boundary: socket/bind failures, event-loop state errors,
        # and invalid server settings are reported and main() exits non-zero.
        logger.error("Failed to start web server: %s", e)
        return False

def main() -> int:
    """Main entry point for web server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RELIC Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--experimental", action="store_true", help="Enable experimental features")
    
    args = parser.parse_args()
    
    # Set experimental flag
    if args.experimental:
        os.environ['RELIC_EXPERIMENTAL_WEB'] = '1'
    
    # Load configuration if provided
    config = None
    if args.config:
        try:
            config = load_web_config(args.config)
        except (OSError, ValueError, RuntimeError, ModuleNotFoundError) as exc:
            logger.error("Failed to load config file %s: %s", args.config, exc)
            return 1
    
    # Run server
    success = run_web_server(
        host=args.host,
        port=args.port,
        config=config,
        log_level=args.log_level,
        reload=args.reload
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
