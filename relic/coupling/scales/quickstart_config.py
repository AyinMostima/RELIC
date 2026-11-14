"""Quickstart configuration helpers for scale coupling."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from relic.parameters import get_global_manager


def build_quickstart_scale_config_impl(
    *,
    defaults_namespace: str,
    parameter_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Construct a minimal auto-scale configuration from the defaults namespace.

    This helper keeps scale-manager defaults centralized in the parameter
    database and avoids hard-coded runtime defaults.
    """
    pm = get_global_manager()
    if pm is None:
        raise RuntimeError("ParameterManager is required to build quickstart scale configuration.")

    quickstart: Dict[str, Any] = {}
    quickstart["mode"] = pm.get(
        "auto_scale_mode",
        namespace=defaults_namespace,
        units="dimensionless",
        expected_type=str,
        required=True,
    )
    quickstart["enforce_conservation"] = bool(
        pm.get(
            "auto_scale_enforce_conservation",
            namespace=defaults_namespace,
            units="dimensionless",
            expected_type=bool,
            required=True,
        )
    )
    cf_strict = pm.get(
        "auto_scale_cf_strict_mode",
        namespace=defaults_namespace,
        units="dimensionless",
        expected_type=bool,
        required=False,
    )
    quickstart["cf_strict_mode"] = bool(cf_strict) if cf_strict is not None else False
    # No conservation_tolerance entry: the ScaleManager ledger tolerance is
    # coupling.scales.defaults.precision.<mode>.tolerance of the mode above.

    optional_defaults: Tuple[Tuple[str, type], ...] = (
        ("relative_tolerance", float),
        ("absolute_tolerance", float),
        ("coverage_epsilon", float),
    )
    for key, expected in optional_defaults:
        value = pm.get(
            key,
            namespace=defaults_namespace,
            units="dimensionless",
            expected_type=expected,
            required=False,
        )
        if value is not None:
            quickstart[key] = float(value)

    return quickstart




