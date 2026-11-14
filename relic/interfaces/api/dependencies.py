"""
API Dependencies

Dependency injection for the RELIC REST API providing shared resources,
configuration access, logging, and component initialization.

Key Features:
    - Configuration management and access
    - Logging setup and access
    - Component initialization (parameter manager, simulator)
    - Async context management for long-running operations

Authors: Yuchong Long
"""

import logging
from typing import Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger(__name__)

from relic.core.services import api_dependencies as _core_api_deps

from relic.core.service_registry import (
    get_data_io_manager as _get_data_io_manager,
    get_parameter_manager as _get_parameter_manager,
    get_simulator as _get_simulator,
)
SERVICE_REGISTRY_AVAILABLE = True
_SERVICE_REGISTRY_ERROR = ""

def get_config(request: Request = None) -> Dict[str, Any]:
    """
    Get configuration from application state
    
    Args:
        request: FastAPI request object
        
    Returns:
        Configuration dictionary
    """
    return _core_api_deps.get_config_from_request(request)

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get logger instance with consistent formatting
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return _core_api_deps.get_logger(name)

async def get_parameter_manager():
    """
    Get parameter manager instance with caching
    
    Returns:
        UnifiedParameterManager instance
    """
    try:
        return _core_api_deps.resolve_service(
            _get_parameter_manager,
            SERVICE_REGISTRY_AVAILABLE,
            _SERVICE_REGISTRY_ERROR,
            "Parameter manager",
        )
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Parameter manager not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Parameter management system not available",
        )

async def get_simulator(request: Request):
    """
    Create the simulator session bound to the current API request.

    FastAPI caches this dependency within one request. The service registry
    creates a new canonical simulator for every distinct dependency call.

    Args:
        request: FastAPI request carrying the application configuration.
    
    Returns:
        RELICSimulator instance
    """
    try:
        config = get_config(request)
        return _core_api_deps.resolve_simulator(
            _get_simulator,
            SERVICE_REGISTRY_AVAILABLE,
            _SERVICE_REGISTRY_ERROR,
            config,
        )
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Simulator not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Simulation system not available",
        )

async def get_computation_health():
    """
    Get computation layer health status
    
    Returns:
        Health status dictionary
    """
    return _core_api_deps.get_computation_health()

async def get_data_io_manager():
    """
    Get data I/O manager instance
    
    Returns:
        Data I/O manager
    """
    try:
        return _core_api_deps.resolve_service(
            _get_data_io_manager,
            SERVICE_REGISTRY_AVAILABLE,
            _SERVICE_REGISTRY_ERROR,
            "Data I/O system",
        )
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"Data I/O system not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Data I/O system not available",
        )

@asynccontextmanager
async def task_context(task_name: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async context manager for long-running tasks
    
    Args:
        task_name: Name of the task for logging
        
    Yields:
        Task context dictionary with logging and status tracking
    """
    async with _core_api_deps.task_context(task_name) as context:
        yield context

# Dependency functions for use with FastAPI Depends()
ConfigDependency = Depends(get_config)
ParameterManagerDependency = Depends(get_parameter_manager)
SimulatorDependency = Depends(get_simulator)
DataIODependency = Depends(get_data_io_manager)
