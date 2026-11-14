"""
API Middleware

Custom middleware for the RELIC REST API providing request/response processing,
error handling, logging, and performance monitoring.

Key Features:
    - Request timing and performance metrics
    - Structured error handling and response formatting
    - Request/response logging with sanitization
    - CORS handling for web integration
    - In-memory rate limiting with deterministic window policy
    - Security headers and validation

Authors: Yuchong Long
"""

import hmac
import json
import logging
import os
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Mapping, Optional, Sequence

from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from ...monitoring import record_api_auth_failure, record_api_exception, record_api_request
from .schemas import create_error_response

logger = logging.getLogger(__name__)

DEFAULT_AUTH_EXEMPT_PATH_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
)
# Request header that carries the API key unless the configuration names another.
DEFAULT_API_KEY_HEADER = "x-api-key"
# Rate-limit bucket of every request that has neither an X-Forwarded-For entry
# nor a transport client address: such requests share this one bucket.
MISSING_CLIENT_ADDRESS_BUCKET = "unknown"


def _as_bool(value: object, default: bool) -> bool:
    """Convert configuration values to booleans with a safe default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_keys(keys: Optional[Iterable[str]]) -> set[str]:
    """Normalize API keys by trimming blanks and discarding empty entries."""
    if keys is None:
        return set()
    normalized: set[str] = set()
    for key in keys:
        text = str(key).strip()
        if text:
            normalized.add(text)
    return normalized


@dataclass(frozen=True)
class APIAuthConfig:
    """Authentication policy for REST and WebSocket API endpoints."""

    enabled: bool
    api_keys: frozenset[str]
    exempt_path_prefixes: tuple[str, ...]
    header_name: str

    def should_skip(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.exempt_path_prefixes)


def build_api_auth_config(
    config: Optional[Mapping[str, object]] = None,
    *,
    enable_auth: Optional[bool] = None,
    api_keys: Optional[Sequence[str]] = None,
) -> APIAuthConfig:
    """
    Build API auth configuration from explicit args, config object, and env vars.

    Precedence (high to low):
    1. Function arguments
    2. ``config["security"]["api_auth"]``
    3. Environment variables
    4. Secure defaults
    """
    security_cfg = {}
    if isinstance(config, Mapping):
        security_cfg = dict(config.get("security", {}) or {})
    auth_cfg = dict(security_cfg.get("api_auth", {}) or {})

    env_enable = os.getenv("RELIC_API_AUTH_ENABLED")
    cfg_enable = auth_cfg.get("enabled")
    enabled = _as_bool(enable_auth, default=True) if enable_auth is not None else _as_bool(cfg_enable, default=True)
    if enable_auth is None and cfg_enable is None and env_enable is not None:
        enabled = _as_bool(env_enable, default=True)

    explicit_keys = _normalize_keys(api_keys)
    config_keys = _normalize_keys(auth_cfg.get("api_keys", []))
    env_key = os.getenv("RELIC_API_KEY", "").strip()
    env_keys = _normalize_keys(os.getenv("RELIC_API_KEYS", "").split(","))
    all_keys = explicit_keys or config_keys or env_keys
    if not all_keys and env_key:
        all_keys = {env_key}

    if enabled and not all_keys:
        # Fail-closed: no implicit development key fallback.
        logger.error(
            "API authentication enabled but no API key is configured. "
            "Configure RELIC_API_KEY/RELIC_API_KEYS or config.security.api_auth.api_keys."
        )

    prefixes = auth_cfg.get("exempt_path_prefixes")
    exempt_prefixes = tuple(prefixes) if isinstance(prefixes, (list, tuple)) and prefixes else DEFAULT_AUTH_EXEMPT_PATH_PREFIXES

    return APIAuthConfig(
        enabled=enabled,
        api_keys=frozenset(all_keys),
        exempt_path_prefixes=exempt_prefixes,
        header_name=str(auth_cfg.get("header_name", DEFAULT_API_KEY_HEADER)).lower(),
    )


def extract_api_key(
    headers: Mapping[str, str],
    *,
    header_name: str = DEFAULT_API_KEY_HEADER,
) -> Optional[str]:
    """Extract API key from `X-API-Key` or `Authorization: Bearer` headers."""
    direct_key = headers.get(header_name) or headers.get(header_name.title())
    if isinstance(direct_key, str) and direct_key.strip():
        return direct_key.strip()

    authorization = headers.get("authorization") or headers.get("Authorization")
    if not authorization:
        return None

    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2:
        return None

    scheme, token = parts[0].lower(), parts[1].strip()
    if scheme in {"bearer", "apikey"} and token:
        return token
    return None


def is_api_key_authorized(candidate_key: Optional[str], valid_keys: Iterable[str]) -> bool:
    """Constant-time API key comparison to avoid timing side channels."""
    if not candidate_key:
        return False
    for valid in valid_keys:
        if hmac.compare_digest(candidate_key, valid):
            return True
    return False


def is_websocket_request_authorized(
    *,
    headers: Mapping[str, str],
    query_params: Mapping[str, str],
    auth_config: APIAuthConfig,
) -> bool:
    """Authenticate websocket requests using header first, then query token."""
    if not auth_config.enabled:
        return True

    key = extract_api_key(headers, header_name=auth_config.header_name)
    if key is None:
        key = (query_params.get("api_key") or "").strip() or None
    return is_api_key_authorized(key, auth_config.api_keys)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Protect API endpoints using API key authentication."""

    def __init__(self, app, auth_config: APIAuthConfig):
        super().__init__(app)
        self.auth_config = auth_config

    async def dispatch(self, request: Request, call_next):
        if not self.auth_config.enabled or self.auth_config.should_skip(request.url.path):
            return await call_next(request)

        if not self.auth_config.api_keys:
            error_response = create_error_response(
                message="API authentication is enabled but no keys are configured.",
                error_type="AuthConfigurationError",
                suggestions=[
                    "Set RELIC_API_KEY or RELIC_API_KEYS.",
                    "Or configure config.security.api_auth.api_keys.",
                ],
            )
            return JSONResponse(status_code=503, content=error_response.model_dump())

        api_key = extract_api_key(request.headers, header_name=self.auth_config.header_name)
        if is_api_key_authorized(api_key, self.auth_config.api_keys):
            return await call_next(request)

        error_response = create_error_response(
            message="Authentication required or invalid API key.",
            error_type="AuthenticationError",
            suggestions=[
                "Provide a valid API key in header X-API-Key or Authorization: Bearer <key>.",
                "Use /health endpoints for unauthenticated readiness checks.",
            ],
        )
        record_api_auth_failure(request.url.path)
        return JSONResponse(
            status_code=401,
            content=error_response.model_dump(),
            headers={"WWW-Authenticate": "Bearer"},
        )

async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    
    # Call the next middleware/endpoint
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Timestamp"] = str(start_time)
    record_api_request(request.method, request.url.path, response.status_code, process_time)
    
    return response

async def handle_errors(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler with structured error responses"""
    # Get request details for logging
    client_host = getattr(request.client, 'host', 'unknown') if hasattr(request, 'client') else 'unknown'
    method = getattr(request, 'method', 'UNKNOWN')
    url = str(getattr(request, 'url', 'unknown'))
    
    # Log the error
    logger.error(f"API Error - {method} {url} from {client_host}: {exc}")
    record_api_exception(type(exc).__name__)
    
    # Handle different exception types. StarletteHTTPException covers the
    # route-raised FastAPI HTTPException and routing errors (404, 405).
    if isinstance(exc, StarletteHTTPException):
        return await _handle_http_exception(exc, request)
    elif isinstance(exc, RequestValidationError):
        return await _handle_request_validation_error(exc, request)
    elif isinstance(exc, ValueError):
        return await _handle_validation_error(exc, request)
    elif isinstance(exc, FileNotFoundError):
        return await _handle_file_not_found(exc, request)
    elif isinstance(exc, PermissionError):
        return await _handle_permission_error(exc, request)
    elif isinstance(exc, ImportError):
        return await _handle_import_error(exc, request)
    else:
        return await _handle_generic_error(exc, request)

async def _handle_http_exception(exc: StarletteHTTPException, request: Request) -> JSONResponse:
    """Handle HTTP exceptions with the structured APIResponse body.

    The status code and headers of the exception are kept. A structured
    ``detail`` mapping with a ``message`` keeps its ``type`` as the error type
    and its fields in ``details``; any other detail becomes the message.
    """
    detail = exc.detail
    if isinstance(detail, Mapping) and isinstance(detail.get("message"), str):
        message = detail["message"]
        error_type = str(detail.get("type", "HTTPException"))
        details = {"status_code": exc.status_code, **jsonable_encoder(dict(detail))}
    else:
        message = detail if isinstance(detail, str) else json.dumps(jsonable_encoder(detail))
        error_type = "HTTPException"
        details = {"status_code": exc.status_code}
    error_response = create_error_response(
        message=message,
        error_type=error_type,
        details=details,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
        headers=exc.headers,
    )


async def _handle_request_validation_error(
    exc: RequestValidationError,
    request: Request,
) -> JSONResponse:
    """Handle request-body/parameter validation errors (HTTP 422)."""
    error_response = create_error_response(
        message="Request validation failed",
        error_type="RequestValidationError",
        details={"errors": jsonable_encoder(exc.errors())},
        suggestions=["Check the request body and parameters against the API schema"],
    )
    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(),
    )

async def _handle_validation_error(exc: ValueError, request: Request) -> JSONResponse:
    """Handle validation errors"""
    error_response = create_error_response(
        message=f"Validation error: {str(exc)}",
        error_type="ValidationError",
        suggestions=[
            "Check input parameters and data types",
            "Refer to API documentation for valid values",
            "Use the /docs endpoint for interactive documentation"
        ]
    )
    
    return JSONResponse(
        status_code=400,
        content=error_response.model_dump()
    )

async def _handle_file_not_found(exc: FileNotFoundError, request: Request) -> JSONResponse:
    """Handle file not found errors"""
    error_response = create_error_response(
        message=f"File not found: {str(exc)}",
        error_type="FileNotFoundError",
        suggestions=[
            "Verify the file path is correct and accessible",
            "Check file permissions",
            "Ensure the file exists at the specified location"
        ]
    )
    
    return JSONResponse(
        status_code=404,
        content=error_response.model_dump()
    )

async def _handle_permission_error(exc: PermissionError, request: Request) -> JSONResponse:
    """Handle permission errors"""
    error_response = create_error_response(
        message=f"Permission denied: {str(exc)}",
        error_type="PermissionError",
        suggestions=[
            "Check file and directory permissions",
            "Ensure the API has read/write access to the specified paths",
            "Run with appropriate user permissions"
        ]
    )
    
    return JSONResponse(
        status_code=403,
        content=error_response.model_dump()
    )

async def _handle_import_error(exc: ImportError, request: Request) -> JSONResponse:
    """Handle import errors (missing dependencies)"""
    module_name = str(exc).split("'")[1] if "'" in str(exc) else "unknown module"
    
    error_response = create_error_response(
        message=f"Missing dependency: {str(exc)}",
        error_type="ImportError",
        details={"missing_module": module_name},
        suggestions=[
            f"Install the missing module: pip install {module_name}",
            "Check that all required dependencies are installed",
            "Refer to installation documentation for requirements"
        ]
    )
    
    return JSONResponse(
        status_code=503,  # Service Unavailable
        content=error_response.model_dump()
    )

async def _handle_generic_error(exc: Exception, request: Request) -> JSONResponse:
    """Handle generic/unexpected errors"""
    # Include traceback in debug mode
    include_traceback = getattr(request.app.state, 'debug', False) if hasattr(request, 'app') else False
    
    error_details = {
        "exception_type": type(exc).__name__,
        "exception_module": getattr(type(exc), '__module__', 'unknown')
    }
    
    if include_traceback:
        error_details["traceback"] = traceback.format_exc()
    
    error_response = create_error_response(
        message=f"Unexpected error: {str(exc)}",
        error_type="InternalServerError",
        details=error_details,
        suggestions=[
            "Report this error to the development team",
            "Include the request details and error information",
            "Check system logs for additional information"
        ]
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump()
    )

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses"""
    
    def __init__(self, app, log_level: str):
        super().__init__(app)
        level = logging.getLevelName(log_level.upper())
        if not isinstance(level, int):
            raise ValueError(f"Unknown request log level '{log_level.upper()}'")
        self.log_level = level
        self.sensitive_headers = {'authorization', 'api-key', 'x-api-key'}
        self.sensitive_params = {'password', 'token', 'api_key', 'secret'}
    
    async def dispatch(self, request: Request, call_next):
        """Process request and response with logging"""
        start_time = time.time()
        
        # Log request
        await self._log_request(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            await self._log_response(request, response, process_time)
            
            return response
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            # Log error
            process_time = time.time() - start_time
            self._log_error(request, exc, process_time)
            raise
    
    async def _log_request(self, request: Request):
        """Log incoming request"""
        if logger.isEnabledFor(self.log_level):
            # Sanitize headers
            headers = dict(request.headers)
            for key in list(headers.keys()):
                if key.lower() in self.sensitive_headers:
                    headers[key] = "[REDACTED]"
            
            # Get query parameters
            query_params = dict(request.query_params)
            for key in list(query_params.keys()):
                if key.lower() in self.sensitive_params:
                    query_params[key] = "[REDACTED]"
            
            log_data = {
                "method": request.method,
                "url": str(request.url),
                "headers": headers,
                "query_params": query_params,
                "client": getattr(request.client, 'host', 'unknown') if hasattr(request, 'client') else 'unknown'
            }
            
            logger.log(self.log_level, f"API Request: {json.dumps(log_data, default=str)}")
    
    async def _log_response(self, request: Request, response: Response, process_time: float):
        """Log outgoing response"""
        if logger.isEnabledFor(self.log_level):
            log_data = {
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "process_time": round(process_time, 4),
                "response_size": response.headers.get("content-length", "unknown")
            }
            
            logger.log(self.log_level, f"API Response: {json.dumps(log_data, default=str)}")
    
    def _log_error(self, request: Request, exc: Exception, process_time: float):
        """Log request error"""
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "error": str(exc),
            "error_type": type(exc).__name__,
            "process_time": round(process_time, 4)
        }
        
        logger.error(f"API Error: {json.dumps(log_data, default=str)}")

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )
        
        # Add API-specific headers
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Powered-By"] = "RELIC"
        
        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    In-memory sliding-window rate limiting middleware.

    Each client address may send ``requests_per_minute`` requests within any
    ``window_seconds`` window; further requests answer HTTP 429 (``/health``
    paths are exempt). The address is the first X-Forwarded-For entry, else the
    transport client host. Requests that carry neither are counted in one
    shared bucket, ``MISSING_CLIENT_ADDRESS_BUCKET``: together they get one
    client's allowance. This is the deterministic handling of a missing
    address, not an error.

    This limiter is process-local and intended for single-instance deployments.
    Distributed deployments should move state to Redis or a shared backend.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int,
        window_seconds: int,
    ):
        super().__init__(app)
        # Invalid settings are rejected, never clamped or rounded into range.
        for name, value in (("requests_per_minute", requests_per_minute), ("window_seconds", window_seconds)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Rate limit {name} must be an integer, got {value!r}")
            if value < 1:
                raise ValueError(f"Rate limit {name} must be at least 1, got {value}")
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.request_history: Dict[str, Deque[float]] = defaultdict(deque)
        self.cleanup_interval = self.window_seconds
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        # Check rate limit
        if not await self._check_rate_limit(client_ip):
            error_response = create_error_response(
                message="Rate limit exceeded",
                error_type="RateLimitError",
                suggestions=[
                    f"Limit: {self.requests_per_minute} requests per minute",
                    "Wait before making additional requests",
                    "Consider caching responses to reduce API calls"
                ]
            )
            return JSONResponse(
                status_code=429,
                content=error_response.model_dump(),
                headers={"Retry-After": str(self.window_seconds)},
            )

        return await call_next(request)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Return the rate-limit bucket key of a request.

        The first X-Forwarded-For entry, else the transport client host; a
        request with neither is counted in the shared
        ``MISSING_CLIENT_ADDRESS_BUCKET``.
        """
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            first_ip = forwarded.split(",", 1)[0].strip()
            if first_ip:
                return first_ip
        if hasattr(request, "client") and getattr(request.client, "host", None):
            return str(request.client.host)
        return MISSING_CLIENT_ADDRESS_BUCKET

    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limit"""
        current_time = time.time()

        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_history()

        # Get client history
        client_history = self.request_history[client_ip]
        cutoff_time = current_time - self.window_seconds
        while client_history and client_history[0] <= cutoff_time:
            client_history.popleft()

        # Check if under limit
        if len(client_history) >= self.requests_per_minute:
            return False

        # Add current request
        client_history.append(current_time)
        return True

    async def _cleanup_history(self):
        """Cleanup old request history"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        # Remove old entries for all clients
        for client_ip in list(self.request_history.keys()):
            client_history = self.request_history[client_ip]
            while client_history and client_history[0] <= cutoff_time:
                client_history.popleft()

            # Remove empty client histories
            if not client_history:
                del self.request_history[client_ip]

        self.last_cleanup = current_time
