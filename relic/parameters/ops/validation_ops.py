"""Parameter validation helpers for UnifiedParameterManager."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..types import ValidationContext, ValidationLevel

logger = logging.getLogger(__name__)


def validate_parameters(manager: Any, keys: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Validate parameters against their specifications."""
    if manager.validation_level == ValidationLevel.NONE:
        return []

    errors: List[Dict[str, str]] = []
    target_keys = keys or list(manager.parameters.keys())

    context = ValidationContext(
        all_parameters=manager.storage.get_all(),
        strict_mode=(manager.validation_level == ValidationLevel.STRICT),
    )

    for key in target_keys:
        if key in manager.parameters:
            spec = manager.parameters[key]
            result = manager.validator.validate_value(
                key,
                spec.value,
                spec,
                manager.validation_level,
                context,
            )
            for error_msg in result.errors:
                errors.append({"parameter": key, "error": error_msg})


    return errors
