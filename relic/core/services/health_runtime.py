from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any, Optional, Type, cast

logger = logging.getLogger(__name__)


def _resolve_health_checker_class() -> Type[Any]:
    """Resolve HealthChecker lazily to keep monkeypatch behavior consistent."""
    module = importlib.import_module("relic.validation.healthcheck")
    checker_cls = cast(Optional[Type[Any]], getattr(module, "HealthChecker", None))
    if checker_cls is None:
        raise ImportError("HealthChecker is not available in relic.validation.healthcheck")
    return checker_cls


def _load_health_checker(config: Mapping[str, Any]) -> Any:
    """Instantiate HealthChecker with the caller's explicit configuration mapping.

    The caller always supplies the mapping (the API passes its application
    configuration); None or a non-mapping raises TypeError instead of being
    replaced by an empty configuration. The checker receives its own copy.
    """
    if not isinstance(config, Mapping):
        raise TypeError(
            "HealthChecker requires an explicit configuration mapping, "
            f"got {type(config).__name__}."
        )
    checker_cls = _resolve_health_checker_class()
    return checker_cls(dict(config))


async def get_overall_health_payload(
    config: Mapping[str, Any],
    detailed: bool = False,
    benchmark: bool = False,
    serialize_details: bool = True,
) -> Any:
    """Return overall health payload via HealthChecker."""
    checker = _load_health_checker(config)
    return await checker.check_all_async(
        detailed=detailed,
        benchmark=benchmark,
        serialize_details=serialize_details,
    )


async def get_subsystem_health_payload(
    subsystem: str,
    config: Mapping[str, Any],
    detailed: bool = True,
    serialize_details: bool = True,
) -> Any:
    """Return subsystem health payload via HealthChecker."""
    checker = _load_health_checker(config)
    return await checker.check_subsystem_async(
        subsystem,
        detailed=detailed,
        serialize_details=serialize_details,
    )


def run_system_benchmark_payload(
    config: Mapping[str, Any],
) -> Any:
    """Run system benchmark via HealthChecker."""
    checker = _load_health_checker(config)
    return checker.run_system_benchmark()
