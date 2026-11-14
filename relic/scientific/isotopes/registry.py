"""Fail-closed stable-isotope assembly boundary.

Stable-isotope execution has no governed StateManager component ledger,
complete route coverage, or restart contract. This module therefore exposes
one public entry that validates candidate configuration data and rejects every
request before reading state, constructing runtime objects, or mutating the
simulator. It intentionally contains no domain registry, health taxonomy,
component factory, removal path, or conditional future runtime.
"""

from __future__ import annotations

from typing import Any, NoReturn

from .config import IsotopeConfig, get_default_isotope_config


class IsotopeValidationError(ValueError):
    """Raised when the unavailable isotope runtime is requested."""


def wire_isotope_tracker(
    simulator: Any,
    state_manager: Any,
    config: IsotopeConfig | None = None,
    device: Any = None,
    execution_group: str = "medium",
    enable_netcdf_output: bool = False,
) -> NoReturn:
    """Reject isotope assembly before runtime or state side effects.

    Candidate configuration is reconstructed from primitive data so unsafe
    Pydantic construction cannot enable an unimplemented runtime. The remaining
    arguments are accepted only to keep this single public request boundary;
    none are inspected while the capability is ``[N] runtime_unbound``.
    """
    _ = simulator, state_manager, device, execution_group

    if enable_netcdf_output:
        raise IsotopeValidationError(
            "[N] postcommit_writer_unbound: isotope NetCDF output is not "
            "bound to an atomic post-commit observer"
        )

    candidate = get_default_isotope_config() if config is None else config
    try:
        payload = candidate.model_dump(mode="python", round_trip=True)
        validated = IsotopeConfig.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - normalize public input failures
        raise IsotopeValidationError(
            f"[N] invalid isotope runtime config: {exc}"
        ) from exc

    if not validated.enabled:
        raise IsotopeValidationError("[N] isotope root runtime_unbound")

    raise IsotopeValidationError(
        "[N] isotope runtime_unbound: no governed StateManager component "
        "ledger, complete route coverage, or restart contract"
    )


__all__ = ["wire_isotope_tracker", "IsotopeValidationError"]
