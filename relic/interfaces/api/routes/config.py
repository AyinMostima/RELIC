"""
Configuration API routes.
"""

from __future__ import annotations

from typing import Any, Dict, List

import relic.core.config as core_config
from fastapi import APIRouter

from relic.core.config.validator import ValidationSeverity
from relic.core.services.runtime_config import load_runtime_config

from ..schemas import (
    APIResponse,
    ConfigValidationRequest,
    create_success_response,
)

router = APIRouter()


def _serialize_config(config_obj: Any) -> Dict[str, Any]:
    if isinstance(config_obj, dict):
        return config_obj
    if hasattr(config_obj, "model_dump"):
        return dict(config_obj.model_dump())
    if hasattr(config_obj, "dict"):
        return dict(config_obj.dict())
    return {"repr": repr(config_obj)}


def _serialize_validation_messages(messages: List[Any], severity: ValidationSeverity) -> List[str]:
    return [str(message) for message in messages if getattr(message, "severity", None) == severity]


@router.get("/", response_model=APIResponse)
async def config_info() -> APIResponse:
    # load_runtime_config always returns the ValidationResult of
    # validate_configuration; its fields are read directly (no defaults).
    config, validation_result, config_file = load_runtime_config()
    payload = {
        "status": "ready",
        "config_source": str(config_file) if config_file else "global_default",
        "is_valid": bool(validation_result.is_valid),
        "warnings": int(validation_result.num_warnings),
        "errors": int(validation_result.num_errors),
    }
    return create_success_response(
        message="Configuration endpoints are available.",
        data=payload,
    )


@router.get("/effective", response_model=APIResponse)
async def get_effective_config() -> APIResponse:
    config, validation_result, config_file = load_runtime_config()
    return create_success_response(
        message="Effective configuration loaded.",
        data={
            "config_source": str(config_file) if config_file else "global_default",
            "validation": {
                "is_valid": bool(validation_result.is_valid),
                "num_errors": int(validation_result.num_errors),
                "num_warnings": int(validation_result.num_warnings),
            },
            "config": _serialize_config(config),
        },
    )


@router.post("/validate", response_model=APIResponse)
async def validate_config(request: ConfigValidationRequest) -> APIResponse:
    # A configuration that cannot be built or validated raises; the app-level
    # error handler (middleware.handle_errors) returns the structured error
    # body with an HTTP error status (pydantic ValidationError -> 400).
    candidate = core_config.RELICConfig(**request.config)
    validation_result = core_config.validate_configuration(candidate, strict=request.strict)
    errors = _serialize_validation_messages(validation_result.messages, ValidationSeverity.ERROR)
    critical = _serialize_validation_messages(validation_result.messages, ValidationSeverity.CRITICAL)
    warnings = _serialize_validation_messages(validation_result.messages, ValidationSeverity.WARNING)
    info = _serialize_validation_messages(validation_result.messages, ValidationSeverity.INFO)

    return create_success_response(
        message="Configuration validation completed.",
        data={
            "is_valid": bool(validation_result.is_valid),
            "errors": errors + critical,
            "warnings": warnings,
            "info": info,
            "effective_config": _serialize_config(candidate),
        },
    )
