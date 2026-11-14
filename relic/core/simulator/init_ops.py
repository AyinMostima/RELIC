"""
Initialization helpers for RELICSimulator.

This module extracts large setup blocks from ``main.py`` so the simulator class
remains focused on orchestration logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from relic.core import device_policy
from relic.utils import time_constants


def configure_calendar_settings(simulator: Any, logger: Any) -> None:
    """Validate the CF calendar identifier used by multi-rate execution."""
    from relic.coupling.external.common.base_adapter import (
        canonical_cf_calendar,
        parse_time_origin,
    )

    # References: CF Conventions 1.8, Section 4.4; existing adapter time contract.
    # Mathematical expression: C_runtime = canonical_CF(C_config);
    #   origin_valid = CF_decode_C_runtime(0; seconds since o) when o is supplied.
    # Parameter meanings: C_config is the original configured calendar,
    #   C_runtime its CF synonym, and o the existing configured time origin.
    # Units: calendar names are dimensionless; o is a calendar timestamp;
    #   the numeric reference offset is in seconds.
    # Domain/constraints: admit only the declared runtime calendar families;
    #   preserve original configuration, reuse origin validation, no conversion.
    # Sign convention: initialization introduces neither flux nor state advance.
    source_calendar = simulator._get_setting("calendar_type", "noleap")
    simulator._calendar_type = canonical_cf_calendar(source_calendar)
    supported_calendars = ("noleap", "360_day", "standard", "proleptic_gregorian")
    if simulator._calendar_type not in supported_calendars:
        raise ValueError(
            f"Unsupported calendar_type '{source_calendar}'. "
            f"Supported values are: {', '.join(supported_calendars)}."
        )
    # time_origin: the ISO 8601 date and time of t = 0 in calendar_type (for example
    # "2000-01-01T00:00:00"); optional (None) until an external adapter reads forcing at the
    # simulation time, which then raises without it. A value that is not a date and time of
    # the calendar raises here.
    simulator._time_origin = simulator._get_setting("time_origin", None)
    if simulator._time_origin is not None:
        parse_time_origin(simulator._time_origin, simulator._calendar_type)

    # Configurable multi-rate intervals (E1)
    # The medium group runs once per day unless configured (one day from the
    # time-constant SSOT, relic.utils.time_constants).
    simulator._medium_interval_seconds = float(
        simulator._get_setting("medium_interval_seconds", time_constants.SECONDS_PER_DAY)
    )
    simulator._slow_interval_seconds = simulator._get_setting("slow_interval_seconds", None)
    if simulator._slow_interval_seconds is not None:
        simulator._slow_interval_seconds = float(simulator._slow_interval_seconds)


def initialize_distributed_runtime(
    simulator: Any,
    logger: Any,
    has_distributed: Callable[[], bool],
    get_process_group_manager: Callable[[], Any],
    get_rank: Callable[[], int],
    get_world_size: Callable[[], int],
    is_distributed: Callable[[], bool],
) -> None:
    """Initialize the distributed runtime when ``enable_distributed`` is requested.

    Fail fast, like ``initialize_scale_manager``: a requested runtime that is
    not installed, a process group whose validation reports errors, a failure
    while setting the process group up, and a GPU ``local_rank`` without a
    visible CUDA device all raise; the simulator never switches distributed
    execution off silently. The device policy's documented fallback stays:
    without CUDA the ranks run on the CPU, and an explicit ``device: cpu`` is
    honored on every rank.
    """
    if not simulator.enable_distributed:
        return
    if not has_distributed():
        raise RuntimeError(
            "enable_distributed=True but the distributed runtime "
            "(relic.computation.distributed) is unavailable; install its "
            "dependencies or set enable_distributed=False."
        )
    try:
        process_group_manager = get_process_group_manager()
        validation_errors = process_group_manager.validate()
    except Exception as err:  # noqa: BLE001 - re-raised with the owner context
        raise RuntimeError(
            f"Distributed runtime initialization failed for enable_distributed=True: {err}"
        ) from err
    if validation_errors:
        raise RuntimeError(
            "enable_distributed=True but the distributed process group is invalid: "
            f"{validation_errors}"
        )
    simulator.process_group_manager = process_group_manager

    requested_device = str(simulator._get_setting("device", "auto")).strip().lower()
    if requested_device == "cpu":
        # GPU-first is the distributed default, not an override of an explicit
        # CPU request: every rank keeps the configured device so the ranks
        # share one device identity.
        logger.info(
            "Distributed mode: explicit device 'cpu' honored on rank %s",
            get_rank() if is_distributed() else 0,
        )
    elif device_policy.is_cuda_available():
        local_rank = simulator._get_setting(
            "local_rank",
            get_rank() if is_distributed() else 0,
        )
        device_count = torch.cuda.device_count()
        if isinstance(local_rank, bool) or not isinstance(local_rank, int) or not 0 <= local_rank < device_count:
            raise RuntimeError(
                f"enable_distributed=True requests GPU local_rank={local_rank!r}, but "
                f"{device_count} CUDA device(s) are visible; set local_rank to a visible "
                "device index or request device='cpu'."
            )
        simulator.device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(simulator.device)
        logger.info("Distributed mode: using GPU device %s", simulator.device)
    else:
        # Device policy: GPU first, CPU when CUDA is unavailable (documented
        # default, not a failure of an explicit request).
        logger.info("Distributed mode: using CPU (CUDA not available)")

    logger.info(
        "Distributed computing enabled: rank=%s, world_size=%s",
        get_rank(),
        get_world_size(),
    )


def initialize_scale_manager(
    simulator: Any,
    logger: Any,
    has_scale_manager: Callable[[], bool],
    get_global_manager: Callable[[], Any],
    parameter_namespace: Any,
    scaling_mode: Any,
    scale_manager_cls: Optional[type],
) -> None:
    """Construct the ScaleManager when ``enable_auto_scale`` is requested.

    Fail fast: a requested but unavailable ScaleManager, an invalid scaling
    mode, a failed parameter lookup, or a failed construction raises; the
    simulator never switches automatic scale coupling off silently. The
    lifecycle (initialize, configure, validate) runs later against the
    simulator's StateManager (``RELICSimulator._ensure_scale_manager_ready``).
    Values come from the ``auto_scale`` configuration first and then from the
    ``coupling.defaults`` parameters (``auto_scale_mode``,
    ``auto_scale_enforce_conservation``, ``auto_scale_cf_strict_mode``).
    """
    if not simulator.enable_auto_scale:
        return
    if not has_scale_manager() or scale_manager_cls is None:
        raise RuntimeError(
            "enable_auto_scale=True but the ScaleManager "
            "(relic.coupling.scales.scale_manager) is unavailable; install its "
            "dependencies or set enable_auto_scale=False."
        )

    parameter_manager = get_global_manager()
    auto_scale_cfg = simulator._get_setting("auto_scale", {}) or {}
    if not isinstance(auto_scale_cfg, Mapping):
        raise TypeError("auto_scale configuration must be a mapping")
    mode_candidate = auto_scale_cfg.get(
        "mode",
        simulator._get_setting("auto_scale_mode"),
    )
    enforce_candidate = auto_scale_cfg.get(
        "enforce_conservation",
        simulator._get_setting("auto_scale_enforce_conservation"),
    )
    cf_strict_candidate = auto_scale_cfg.get(
        "cf_strict_mode",
        simulator._get_setting("auto_scale_cf_strict_mode"),
    )
    if mode_candidate is None:
        mode_candidate = parameter_manager.get(
            "auto_scale_mode",
            namespace=parameter_namespace.COUPLING_DEFAULTS,
            expected_type=str,
            required=False,
        )
    if enforce_candidate is None:
        enforce_candidate = parameter_manager.get(
            "auto_scale_enforce_conservation",
            namespace=parameter_namespace.COUPLING_DEFAULTS,
            expected_type=bool,
            required=False,
        )
    if cf_strict_candidate is None:
        cf_strict_candidate = parameter_manager.get(
            "auto_scale_cf_strict_mode",
            namespace=parameter_namespace.COUPLING_DEFAULTS,
            expected_type=bool,
            required=False,
        )

    resolved_mode = mode_candidate or scaling_mode.CONSERVATIVE.value
    if isinstance(resolved_mode, scaling_mode):
        default_mode = resolved_mode
    else:
        try:
            default_mode = scaling_mode(str(resolved_mode))
        except ValueError as exc:
            raise ValueError(
                f"Invalid auto-scale mode {resolved_mode!r}; expected one of "
                f"{[member.value for member in scaling_mode]}."
            ) from exc

    cf_strict_mode_cfg = bool(cf_strict_candidate) if cf_strict_candidate is not None else False
    target_device = getattr(simulator, "device", torch.device("cpu"))
    try:
        simulator.scale_manager = scale_manager_cls(
            device=torch.device(target_device),
            default_mode=default_mode,
            enforce_conservation=enforce_candidate,
            cf_strict_mode=cf_strict_mode_cfg,
            parameter_manager=parameter_manager,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with the owner context
        raise RuntimeError(
            f"ScaleManager construction failed for enable_auto_scale=True: {exc}"
        ) from exc
    logger.info(
        "ScaleManager constructed for automatic multi-scale coupling "
        "(mode=%s, enforce_conservation=%s, cf_strict_mode=%s)",
        default_mode,
        enforce_candidate,
        cf_strict_mode_cfg,
    )


def initialize_diagnostic_export(
    simulator: Any,
    logger: Any,
    has_diagnostic_writer: bool,
    diagnostic_writer_cls: Optional[type],
) -> None:
    """Initialize optional diagnostics export writer and output path."""
    if not simulator.enable_diagnostic_export:
        return

    output_path = simulator._get_setting(
        "diagnostic_output_path",
        "build/diagnostics/diagnostics.nc",
    )
    simulator.diagnostic_output_path = Path(output_path) if output_path else None

    if has_diagnostic_writer and diagnostic_writer_cls is not None:
        simulator.diagnostic_writer = diagnostic_writer_cls()
        logger.info(
            "DiagnosticWriter initialized for export to: %s",
            simulator.diagnostic_output_path,
        )
    else:
        # The export was requested; an unavailable writer is not a reason to
        # switch it off silently.
        raise RuntimeError(
            "enable_diagnostic_export=True but DiagnosticWriter "
            "(relic.data.io.diagnostic_writer) is not available"
        )
