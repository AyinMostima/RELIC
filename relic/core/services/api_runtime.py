from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Tuple

logger = logging.getLogger(__name__)


def initialize_api_state(
    app: Any,
    load_runtime_config: Callable[[], Tuple[Dict[str, Any], Any, Any]],
    check_api_dependencies: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Initialize API application state and return dependency status."""
    logger.info("Starting RELIC API server...")

    deps = check_api_dependencies()
    for dep, available in deps["optional_dependencies"].items():
        status = "available" if available else "missing"
        logger.info("Optional dependency %s: %s", dep, status)

    # A configuration that cannot be loaded stops the startup (the error
    # propagates); the API never runs with a substituted empty configuration.
    config, validation_result, _ = load_runtime_config()
    app.state.config = config
    if not validation_result.is_valid:
        logger.warning("Configuration validation reported errors")
    else:
        logger.info("Configuration loaded successfully")

    app.state.startup_time = time.time()
    return deps


def shutdown_api_state() -> None:
    """Finalize API application state."""
    logger.info("Shutting down RELIC API server...")
