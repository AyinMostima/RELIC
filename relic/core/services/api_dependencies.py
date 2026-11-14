"""API dependency checks shared by interface entry points."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, cast

logger = logging.getLogger(__name__)


def check_api_dependencies() -> Dict[str, Any]:
    """
    Report optional dependency availability for API usage.

    FastAPI itself is imported by the API package, so it is never reported
    here: a missing FastAPI fails the import of ``relic.interfaces.api``.

    Returns
    -------
    Dict[str, Dict[str, bool]]
        ``{"optional_dependencies": {name: available}}``.
    """
    optional_deps = {
        "numpy": "Scientific computing",
        "torch": "Deep learning and GPU acceleration",
        "pydantic": "Data validation",
    }

    available_optional: Dict[str, bool] = {}
    for dep in optional_deps:
        available_optional[dep] = importlib.util.find_spec(dep) is not None

    return {
        "optional_dependencies": available_optional,
    }


def get_config_from_request(request: Any = None) -> Dict[str, Any]:
    """
    Return the application configuration of a request, else the global configuration.

    The application configuration must be a mapping; the global configuration
    (a RELICConfig model) is returned as its ``model_dump()`` mapping. Any
    other shape raises TypeError instead of being replaced by an empty mapping.
    """
    if request and hasattr(request, "app") and hasattr(request.app, "state"):
        if hasattr(request.app.state, "config"):
            app_config = request.app.state.config
            if not isinstance(app_config, Mapping):
                raise TypeError(
                    "Application state config must be a mapping, "
                    f"got {type(app_config).__name__}."
                )
            return dict(app_config)

    from relic.core.config import get_global_config

    get_global_config_typed = cast(Callable[[], Any], get_global_config)
    global_config = get_global_config_typed()
    if isinstance(global_config, Mapping):
        return dict(global_config)
    if hasattr(global_config, "model_dump"):
        return cast(dict[str, Any], global_config.model_dump())
    raise TypeError(
        "Global configuration must be a mapping or a RELICConfig model, "
        f"got {type(global_config).__name__}."
    )


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a logger instance for API helpers."""
    return logging.getLogger(name)


def resolve_service(
    getter: Callable[[], Any],
    available: bool,
    error: str,
    label: str,
) -> Any:
    """Resolve a cached service or raise a runtime error."""
    if not available:
        raise RuntimeError(error or f"{label} not available")
    return getter()


def resolve_simulator(
    getter: Callable[..., Any],
    available: bool,
    error: str,
    config: Dict[str, Any],
) -> Any:
    """Resolve simulator service with a configuration payload."""
    if not available:
        raise RuntimeError(error or "Simulation system not available")
    return getter(config=config)


def get_computation_health() -> Dict[str, str]:
    """Return the computation layer health summary.

    relic.computation is RELIC's own package: an import failure or a missing
    ``healthcheck`` is a code defect and raises; a report that is not a mapping
    raises TypeError instead of being replaced by an error summary.
    """
    computation_module = importlib.import_module("relic.computation")
    health = computation_module.healthcheck(verbose=False)
    if not isinstance(health, Mapping):
        raise TypeError(
            "relic.computation.healthcheck must return a mapping, "
            f"got {type(health).__name__}."
        )
    return {str(key): str(value) for key, value in health.items()}


@asynccontextmanager
async def task_context(task_name: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async context manager for long-running tasks.
    """
    task_logger = get_logger(f"task.{task_name}")

    start_time = asyncio.get_event_loop().time()
    context = {
        "logger": task_logger,
        "start_time": start_time,
        "status": "running",
        "progress": 0.0,
        "messages": [],
    }

    task_logger.info(f"Task '{task_name}' started")

    try:
        yield context
        context["status"] = "completed"
        task_logger.info(f"Task '{task_name}' completed successfully")
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
        context["status"] = "failed"
        context["error"] = str(exc)
        task_logger.error(f"Task '{task_name}' failed: {exc}")
        raise
    finally:
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        task_logger.info(f"Task '{task_name}' finished in {duration:.2f} seconds")
