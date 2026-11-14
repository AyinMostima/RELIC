"""
Centralized device policy utilities for RELIC.

This module provides a single runtime policy for resolving CPU/GPU devices
across simulator, coupling, computation, and service modules.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

DeviceLike = Union[str, torch.device]

_GLOBAL_DEFAULT_DEVICE: Optional[torch.device] = None
# Override maps keep the requested device (syntax-checked, not yet resolved):
# CUDA availability, the CPU fallback and require_gpu are applied when a
# request selects the override, with that request's require_gpu.
_RUNTIME_OVERRIDES: Dict[str, DeviceLike] = {}
_ENV_OVERRIDE_CACHE_KEY: Optional[str] = None
_ENV_OVERRIDE_CACHE: Dict[str, DeviceLike] = {}


def is_cuda_available() -> bool:
    """Return CUDA availability as reported by torch.

    torch.cuda.is_available() returns False when no driver or device is
    present; an exception from it means a broken torch installation and is
    propagated rather than reported as a missing GPU.
    """
    return bool(torch.cuda.is_available())


def _parse_device(
    requested: DeviceLike,
    *,
    require_gpu: bool = False,
) -> torch.device:
    """Parse a device request with CUDA fallback semantics."""
    if isinstance(requested, torch.device):
        parsed = requested
    else:
        raw = str(requested).strip()
        lowered = raw.lower()
        if lowered in {"", "auto", "default"}:
            if is_cuda_available():
                return torch.device("cuda")
            if require_gpu:
                raise RuntimeError("Device 'auto' requires CUDA, but CUDA is not available.")
            return torch.device("cpu")
        parsed = torch.device(lowered)

    if parsed.type != "cuda":
        return parsed

    if not is_cuda_available():
        if require_gpu:
            raise RuntimeError(
                f"Device '{parsed}' requires CUDA, but CUDA is not available."
            )
        return torch.device("cpu")

    device_index = parsed.index
    if device_index is None:
        return parsed

    device_count = int(torch.cuda.device_count())
    if device_index < device_count:
        return parsed

    if require_gpu:
        raise RuntimeError(
            f"Device '{parsed}' is out of range. Available CUDA devices: {device_count}."
        )
    return torch.device("cuda:0") if device_count > 0 else torch.device("cpu")


def _infer_caller_module() -> str:
    """Infer caller module name to enable module-level overrides."""
    frame = inspect.currentframe()
    if frame is None:
        return "__unknown__"
    caller = frame.f_back
    while caller is not None:
        module_name = str(caller.f_globals.get("__name__", "")).strip()
        if module_name and module_name != __name__:
            return module_name
        caller = caller.f_back
    return "__unknown__"


def _requested_device(value: Any) -> DeviceLike:
    """Validate the syntax of one device request without resolving it.

    "", "auto" and "default" stay the policy keyword "auto"; any other text
    must name a torch device type (torch.device raises RuntimeError otherwise).
    """
    if isinstance(value, torch.device):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"", "auto", "default"}:
        return "auto"
    return torch.device(lowered)


def _requested_device_map(
    mapping: Optional[Mapping[str, Any]],
    *,
    source: str,
) -> Dict[str, DeviceLike]:
    """Parse a {module_or_functor_name: device} override map into requested devices.

    None means that no overrides are declared. An entry whose value is None
    declares no override for that name. Any other non-mapping value raises
    TypeError, and an unparseable device raises RuntimeError naming the source
    and key, instead of silently dropping the override. Availability and
    require_gpu are checked by _resolve_override when a request selects it.
    """
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"Device overrides from {source} must be a mapping of names to devices, "
            f"got {type(mapping).__name__}."
        )
    requested: Dict[str, DeviceLike] = {}
    for key, value in mapping.items():
        if value is None:
            continue
        text_key = str(key).strip()
        if not text_key:
            continue
        try:
            requested[text_key] = _requested_device(value)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Invalid device override '{text_key}' -> {value!r} from {source}: {exc}"
            ) from exc
    return requested


def _resolve_override(
    name: str,
    requested: DeviceLike,
    *,
    source: str,
    require_gpu: bool,
) -> torch.device:
    """Resolve one selected override under the policy of the current request."""
    try:
        return _parse_device(requested, require_gpu=require_gpu)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Invalid device override '{name}' -> {requested!r} from {source}: {exc}"
        ) from exc


def _load_env_overrides() -> Dict[str, DeviceLike]:
    """
    Load module-level overrides from RELIC_DEVICE_OVERRIDES_JSON.

    Example:
        {"relic.coupling.scales.temporal.core":"cpu","compute":"cuda:1"}

    Text that is not a JSON object raises ValueError. The requested devices are
    cached per raw text; they are resolved per request.
    """
    global _ENV_OVERRIDE_CACHE_KEY, _ENV_OVERRIDE_CACHE
    raw = str(os.getenv("RELIC_DEVICE_OVERRIDES_JSON", "")).strip()
    if raw == _ENV_OVERRIDE_CACHE_KEY:
        return dict(_ENV_OVERRIDE_CACHE)
    overrides: Dict[str, DeviceLike] = {}
    if raw:
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise ValueError(
                f"RELIC_DEVICE_OVERRIDES_JSON is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                "RELIC_DEVICE_OVERRIDES_JSON must be a JSON object mapping module "
                f"names to devices, got {type(payload).__name__}."
            )
        overrides = _requested_device_map(payload, source="RELIC_DEVICE_OVERRIDES_JSON")
    _ENV_OVERRIDE_CACHE_KEY = raw
    _ENV_OVERRIDE_CACHE = overrides
    return dict(overrides)


def _extract_config_device(config: Optional[Mapping[str, Any]]) -> Optional[Any]:
    if not isinstance(config, Mapping):
        return None
    if "device" in config:
        return config["device"]
    simulator_cfg = config.get("simulator")
    if isinstance(simulator_cfg, Mapping) and "device" in simulator_cfg:
        return simulator_cfg["device"]
    runtime_cfg = config.get("runtime")
    if isinstance(runtime_cfg, Mapping) and "device" in runtime_cfg:
        return runtime_cfg["device"]
    return None


def _extract_config_overrides(config: Optional[Mapping[str, Any]]) -> Dict[str, DeviceLike]:
    if not isinstance(config, Mapping):
        return {}
    merged: Dict[str, DeviceLike] = {}
    for key in ("device_overrides", "module_device_overrides"):
        merged.update(_requested_device_map(config.get(key), source=f"config[{key!r}]"))
    simulator_cfg = config.get("simulator")
    if isinstance(simulator_cfg, Mapping):
        for key in ("device_overrides", "module_device_overrides"):
            merged.update(
                _requested_device_map(
                    simulator_cfg.get(key),
                    source=f"config['simulator'][{key!r}]",
                )
            )
    runtime_cfg = config.get("runtime")
    if isinstance(runtime_cfg, Mapping):
        for key in ("device_overrides", "module_device_overrides"):
            merged.update(
                _requested_device_map(
                    runtime_cfg.get(key),
                    source=f"config['runtime'][{key!r}]",
                )
            )
    return merged


def set_default_device(
    device: Optional[DeviceLike],
    *,
    require_gpu: bool = False,
) -> None:
    """Set process-wide default device for modules using resolve_device()."""
    global _GLOBAL_DEFAULT_DEVICE
    if device is None:
        _GLOBAL_DEFAULT_DEVICE = None
        return
    _GLOBAL_DEFAULT_DEVICE = _parse_device(device, require_gpu=require_gpu)


def set_device_overrides(
    overrides: Mapping[str, DeviceLike],
    *,
    clear_existing: bool = False,
    require_gpu: bool = False,
) -> None:
    """Set process-wide module overrides for resolve_device().

    The requested devices are stored; each resolve_device() request resolves
    the selected override with its own require_gpu. require_gpu=True here also
    checks every entry against the current CUDA availability before storing.
    """
    global _RUNTIME_OVERRIDES
    # Check the whole map before changing the process-wide overrides, so a
    # rejected entry leaves the previous overrides in place.
    requested = _requested_device_map(overrides, source="set_device_overrides")
    if require_gpu:
        for name, device in requested.items():
            _resolve_override(name, device, source="set_device_overrides", require_gpu=True)
    if clear_existing:
        _RUNTIME_OVERRIDES = {}
    _RUNTIME_OVERRIDES.update(requested)


def get_device_overrides() -> Dict[str, torch.device]:
    """Return the runtime overrides resolved under the default policy (CUDA first, CPU fallback)."""
    return {
        name: _resolve_override(name, device, source="set_device_overrides", require_gpu=False)
        for name, device in _RUNTIME_OVERRIDES.items()
    }


def resolve_device(
    explicit_device: Optional[DeviceLike] = None,
    *,
    module_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    default: DeviceLike = "cpu",
    distributed: bool = False,
    require_gpu: bool = False,
) -> torch.device:
    """
    Resolve a device with policy order:
      1) explicit_device
      2) config module overrides
      3) runtime/env module overrides
      4) config global device
      5) process-wide default device
      6) RELIC_DEFAULT_DEVICE env var
      7) distributed default (cuda->cpu fallback) or provided default
    """
    if explicit_device is not None:
        return _parse_device(explicit_device, require_gpu=require_gpu)

    resolved_module = module_name or _infer_caller_module()
    short_name = resolved_module.rsplit(".", 1)[-1]

    override_sources = (
        ("config", _extract_config_overrides(config)),
        ("set_device_overrides", _RUNTIME_OVERRIDES),
        ("RELIC_DEVICE_OVERRIDES_JSON", _load_env_overrides()),
    )
    # Only the selected override is resolved, with this request's require_gpu.
    for name in (resolved_module, short_name):
        for source, overrides in override_sources:
            if name in overrides:
                return _resolve_override(
                    name,
                    overrides[name],
                    source=source,
                    require_gpu=require_gpu,
                )

    config_device = _extract_config_device(config)
    if config_device is not None:
        return _parse_device(config_device, require_gpu=require_gpu)

    if _GLOBAL_DEFAULT_DEVICE is not None:
        return _GLOBAL_DEFAULT_DEVICE

    env_default = str(os.getenv("RELIC_DEFAULT_DEVICE", "")).strip()
    if env_default:
        return _parse_device(env_default, require_gpu=require_gpu)

    if distributed:
        return torch.device("cuda") if is_cuda_available() else torch.device("cpu")

    return _parse_device(default, require_gpu=require_gpu)


def distributed_default_device(
    *,
    module_name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    require_gpu: bool = False,
) -> torch.device:
    """Resolve distributed default device: CUDA first, CPU fallback."""
    return resolve_device(
        module_name=module_name,
        config=config,
        default="auto",
        distributed=True,
        require_gpu=require_gpu,
    )


class DeviceFence:
    """Completion fence for device work already enqueued when it is recorded.

    A Python return from a kernel launch is not completion of the device
    work; the fence is the only owner-side evidence that the work enqueued
    on the recorded streams has finished (E01).

    References: NVIDIA CUDA C Programming Guide, asynchronous concurrent
    execution (streams and events); PyTorch CUDA semantics notes
    (stream-ordered execution, ``torch.cuda.Event.query``/``synchronize``).
    Mathematical expression: completed = AND_d query(e_d) over the events
    e_d recorded on the current stream of every CUDA device d; a fence over
    CPU devices only is vacuously complete.
    Parameter meanings: e_d event recorded at construction on the current
    stream of device d; query(e_d) is true once every operation enqueued on
    that stream before the record has completed.
    Units: none (booleans); ``timeout`` in wall-clock seconds.
    Domain/constraints: only work enqueued before the record on the
    recorded streams is covered; later work and other streams are not.
    Sign convention: not applicable.
    """

    __slots__ = ("devices", "_events", "recorded_at")

    def __init__(self, devices: Iterable[DeviceLike]) -> None:
        unique: list[torch.device] = []
        cuda_available = is_cuda_available()
        for candidate in devices:
            device = torch.device(candidate)
            if device.type != "cuda":
                continue
            # An index-less "cuda" device means the current device; the
            # fence records the exact device the event lives on.
            if device.index is None and cuda_available:
                device = torch.device("cuda", torch.cuda.current_device())
            if device not in unique:
                unique.append(device)
        self.devices: Tuple[torch.device, ...] = tuple(unique)
        self._events: Dict[torch.device, Any] = {}
        self.recorded_at = time.monotonic()
        if not self.devices or not is_cuda_available():
            return
        for device in self.devices:
            index = device.index if device.index is not None else torch.cuda.current_device()
            with torch.cuda.device(index):
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(index))
            self._events[device] = event

    @property
    def completed(self) -> bool:
        return all(event.query() for event in self._events.values())

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until every recorded event completed; False on timeout."""
        if timeout is None:
            for event in self._events.values():
                event.synchronize()
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        while not self.completed:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        return True

    def describe(self) -> Dict[str, Any]:
        return {
            "devices": [str(device) for device in self.devices],
            "events": len(self._events),
            "completed": self.completed,
        }


def record_device_fence(devices: Iterable[DeviceLike]) -> DeviceFence:
    """Record a completion fence on the current streams of the given devices."""
    return DeviceFence(devices)


__all__ = [
    "DeviceFence",
    "DeviceLike",
    "is_cuda_available",
    "record_device_fence",
    "resolve_device",
    "distributed_default_device",
    "set_default_device",
    "set_device_overrides",
    "get_device_overrides",
]
