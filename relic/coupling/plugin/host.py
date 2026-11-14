"""ctypes-based host driver for coupling plugins.

Lifecycle (E03): ``load`` binds the shared library and checks the ABI
version, ``retire`` refuses new native calls, ``drain`` waits for the calls
still executing inside the library, ``unload`` releases the library only
after the drain, and ``reload`` binds the same path again. A Python return
from a native entry point is the only completion signal the C ABI offers, so
the in-flight counter is the drain boundary; every transition is recorded in
a receipt that can be consumed by the extension loader.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

COUPLER_PLUGIN_ABI_VERSION = "1.0"
_STATE_UNLOADED = "unloaded"
_STATE_LOADED = "loaded"
_STATE_RETIRED = "retired"


class PluginABIError(RuntimeError):
    """The plugin declares an interface version the host cannot honor."""


class PluginLifecycleError(RuntimeError):
    """A native call or transition was requested in an inadmissible state."""


def _free_library(lib: ctypes.CDLL) -> None:
    """Release one ctypes library handle through the platform loader."""
    handle = getattr(lib, "_handle", None)
    if handle is None:
        raise PluginLifecycleError("Library handle is unavailable; cannot release it.")
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
        kernel32.FreeLibrary.restype = ctypes.c_int
        if not kernel32.FreeLibrary(ctypes.c_void_p(handle)):
            raise OSError(ctypes.get_last_error(), "FreeLibrary failed for the plugin handle")
        return
    libc = ctypes.CDLL(None)
    libc.dlclose.argtypes = [ctypes.c_void_p]
    libc.dlclose.restype = ctypes.c_int
    if libc.dlclose(ctypes.c_void_p(handle)) != 0:
        raise OSError("dlclose failed for the plugin handle")


class PluginHost:
    """Host-side loader and dispatcher for C ABI coupling plugins."""

    def __init__(self, path: Optional[str] = None, manifest_buffer_size: int = 65536) -> None:
        self.path: Optional[Path] = None
        self.manifest_buffer_size = int(manifest_buffer_size)
        self._lib: Optional[ctypes.CDLL] = None
        self._plugin_probe: Optional[Any] = None
        self._plugin_negotiate: Optional[Any] = None
        self._plugin_exchange: Optional[Any] = None
        self._state = _STATE_UNLOADED
        self._lock = threading.Condition()
        self._inflight = 0
        self._completed_calls = 0
        self._load_count = 0
        self._abi: Dict[str, Any] = {}
        self._receipts: list[Dict[str, Any]] = []
        if path is not None:
            self.load(path)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def loaded(self) -> bool:
        return self._state != _STATE_UNLOADED

    @property
    def inflight_calls(self) -> int:
        with self._lock:
            return self._inflight

    @property
    def completed_calls(self) -> int:
        with self._lock:
            return self._completed_calls

    @property
    def abi(self) -> Dict[str, Any]:
        return dict(self._abi)

    @property
    def receipts(self) -> tuple[Dict[str, Any], ...]:
        return tuple(dict(item) for item in self._receipts)

    def _record(self, action: str, **fields: Any) -> Dict[str, Any]:
        receipt = {
            "action": action,
            "path": None if self.path is None else str(self.path),
            "state": self._state,
            "load_count": self._load_count,
            "completed_calls": self._completed_calls,
            "inflight_calls": self._inflight,
            **fields,
        }
        self._receipts.append(receipt)
        return dict(receipt)

    # ------------------------------------------------------------------
    # Load / ABI
    # ------------------------------------------------------------------
    def load(self, path: str) -> Dict[str, Any]:
        """Load plugin shared library, resolve required symbols, check the ABI."""
        with self._lock:
            if self._state != _STATE_UNLOADED:
                raise PluginLifecycleError(
                    f"Plugin is already {self._state}; unload it before loading again."
                )
        plugin_path = Path(path).resolve()
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin library not found: {plugin_path}")

        lib = ctypes.cdll.LoadLibrary(str(plugin_path))
        missing = [name for name in ("plugin_probe", "plugin_negotiate", "plugin_exchange") if not hasattr(lib, name)]
        if missing:
            _free_library(lib)
            raise RuntimeError(f"Missing required plugin symbol(s): {', '.join(missing)}")

        probe = getattr(lib, "plugin_probe")
        probe.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
        probe.restype = ctypes.c_int

        negotiate = getattr(lib, "plugin_negotiate")
        negotiate.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
        negotiate.restype = ctypes.c_int

        exchange = getattr(lib, "plugin_exchange")
        exchange.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        exchange.restype = ctypes.c_int

        with self._lock:
            self.path = plugin_path
            self._lib = lib
            self._plugin_probe = probe
            self._plugin_negotiate = negotiate
            self._plugin_exchange = exchange
            self._state = _STATE_LOADED
            self._load_count += 1
            self._abi = {}
        # The ABI check is part of loading: a plugin whose declared interface
        # version the host cannot honor is released before any exchange.
        try:
            manifest = self.probe()
            self._abi = self._verify_abi(manifest)
        except BaseException:
            self._release_library()
            raise
        with self._lock:
            return self._record("load", abi=dict(self._abi))

    def _verify_abi(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        declared = str(manifest.get("interface_version", "")).strip()
        if declared != COUPLER_PLUGIN_ABI_VERSION:
            raise PluginABIError(
                f"Plugin '{manifest.get('name', self.path)}' declares interface_version "
                f"{declared!r}; this host implements {COUPLER_PLUGIN_ABI_VERSION!r}."
            )
        metadata = manifest.get("metadata", {})
        native = metadata.get("abi", {}) if isinstance(metadata, dict) else {}
        return {
            "interface_version": declared,
            "host_platform": platform.platform(),
            "host_python": sys.version.split()[0],
            "host_pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
            "native": dict(native) if isinstance(native, dict) else {},
        }

    # ------------------------------------------------------------------
    # Retire / drain / unload / reload
    # ------------------------------------------------------------------
    def retire(self) -> Dict[str, Any]:
        """Refuse new native calls; calls already inside the library continue."""
        with self._lock:
            if self._state == _STATE_UNLOADED:
                raise PluginLifecycleError("Plugin is not loaded; nothing to retire.")
            if self._state == _STATE_RETIRED:
                return self._record("retire", repeated=True)
            self._state = _STATE_RETIRED
            return self._record("retire", repeated=False)

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until no native call is in flight; True when drained."""
        with self._lock:
            return self._lock.wait_for(lambda: self._inflight == 0, timeout=timeout)

    def _release_library(self) -> None:
        with self._lock:
            lib = self._lib
            self._lib = None
            self._plugin_probe = None
            self._plugin_negotiate = None
            self._plugin_exchange = None
            self._state = _STATE_UNLOADED
        if lib is not None:
            _free_library(lib)

    def unload(self, *, drain_timeout: Optional[float] = None) -> Dict[str, Any]:
        """Retire, drain, and release the library; repeated unload is refused.

        References: Goebel, Sanfelice and Teel (2012) hybrid retirement
        ordering; POSIX dlclose / Win32 FreeLibrary contracts.
        Mathematical expression: release happens only when inflight = 0.
        Parameter meanings: drain_timeout bounds the wait for in-flight calls.
        Units: seconds.
        Domain/constraints: a timeout leaves the plugin retired (not
        unloaded) with an observable receipt so the release can be retried.
        Sign convention: not applicable.
        """
        with self._lock:
            if self._state == _STATE_UNLOADED:
                raise PluginLifecycleError("Plugin is already unloaded; repeated unload is refused.")
            if self._state == _STATE_LOADED:
                self._state = _STATE_RETIRED
                self._record("retire", repeated=False)
            started = time.perf_counter()
            drained = self._lock.wait_for(lambda: self._inflight == 0, timeout=drain_timeout)
            waited = time.perf_counter() - started
            if not drained:
                receipt = self._record("unload", released=False, drained=False, waited_seconds=waited)
                return receipt
        self._release_library()
        with self._lock:
            return self._record("unload", released=True, drained=True, waited_seconds=waited)

    def reload(self) -> Dict[str, Any]:
        """Unload (after drain) and load the same library path again."""
        path = self.path
        if path is None:
            raise PluginLifecycleError("Plugin has no path to reload.")
        if self._state != _STATE_UNLOADED:
            receipt = self.unload()
            if not receipt["released"]:
                raise PluginLifecycleError("Plugin could not be drained; reload refused.")
        return self.load(str(path))

    # ------------------------------------------------------------------
    # Native calls
    # ------------------------------------------------------------------
    def _enter_call(self, name: str) -> None:
        with self._lock:
            if self._state != _STATE_LOADED:
                raise PluginLifecycleError(
                    f"Plugin call '{name}' refused: plugin is {self._state}."
                )
            self._inflight += 1

    def _exit_call(self) -> None:
        with self._lock:
            self._inflight -= 1
            self._completed_calls += 1
            if self._inflight == 0:
                self._lock.notify_all()

    def probe(self) -> Dict[str, Any]:
        """Query plugin probe manifest."""
        self._enter_call("plugin_probe")
        try:
            return self._call_json_export(self._plugin_probe, None)
        finally:
            self._exit_call()

    def negotiate(self, host_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Negotiate host capabilities and return agreed manifest."""
        payload = json.dumps(host_capabilities, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self._enter_call("plugin_negotiate")
        try:
            agreed = self._call_json_export(self._plugin_negotiate, payload)
        finally:
            self._exit_call()
        self._verify_abi(agreed)
        return agreed

    def exchange(self, port_name: str, buffer: Any, timestamp: float) -> int:
        """
        Exchange a data buffer with plugin.

        Zero-copy paths:
        - numpy.ndarray: pointer from ``ndarray.ctypes.data``
        - torch.Tensor: pointer from ``tensor.data_ptr()``
        """
        ptr, size, keepalive = self._prepare_buffer(buffer)
        if size <= 0:
            raise ValueError("Buffer size must be positive.")
        self._enter_call("plugin_exchange")
        try:
            exchange_fn = self._plugin_exchange
            if exchange_fn is None:
                raise PluginLifecycleError("Plugin is not loaded.")
            rc = int(
                exchange_fn(
                    ctypes.c_void_p(0),
                    port_name.encode("utf-8"),
                    ptr,
                    ctypes.c_size_t(size),
                    ctypes.c_double(float(timestamp)),
                )
            )
        finally:
            self._exit_call()
        # Keep references alive for the duration of the call.
        _ = keepalive
        if rc != 0:
            raise RuntimeError(f"plugin_exchange failed for port '{port_name}' with code {rc}.")
        return rc

    def _call_json_export(self, fn: Any, input_payload: Optional[bytes]) -> Dict[str, Any]:
        if fn is None:
            raise PluginLifecycleError("Plugin is not loaded.")
        buffer = ctypes.create_string_buffer(self.manifest_buffer_size)
        if input_payload is None:
            rc = int(fn(buffer, ctypes.c_size_t(len(buffer))))
        else:
            rc = int(fn(ctypes.c_char_p(input_payload), buffer, ctypes.c_size_t(len(buffer))))
        if rc != 0:
            raise RuntimeError(f"Plugin manifest call failed with code {rc}.")
        text = buffer.value.decode("utf-8").strip()
        if not text:
            raise RuntimeError("Plugin returned empty manifest JSON.")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Plugin manifest payload must be a JSON object.")
        return data

    @staticmethod
    def _prepare_buffer(buffer: Any) -> Tuple[ctypes.c_void_p, int, Any]:
        if isinstance(buffer, np.ndarray):
            arr = buffer if buffer.flags["C_CONTIGUOUS"] else np.ascontiguousarray(buffer)
            return ctypes.c_void_p(int(arr.ctypes.data)), int(arr.nbytes), arr

        if isinstance(buffer, torch.Tensor):
            tensor = buffer if buffer.is_contiguous() else buffer.contiguous()
            nbytes = int(tensor.numel() * tensor.element_size())
            return ctypes.c_void_p(int(tensor.data_ptr())), nbytes, tensor

        if isinstance(buffer, memoryview):
            if buffer.readonly:
                data = bytearray(buffer.tobytes())
                c_buf = (ctypes.c_ubyte * len(data)).from_buffer(data)
                return ctypes.cast(c_buf, ctypes.c_void_p), len(data), (data, c_buf)
            c_buf = (ctypes.c_ubyte * buffer.nbytes).from_buffer(buffer)
            return ctypes.cast(c_buf, ctypes.c_void_p), int(buffer.nbytes), c_buf

        if isinstance(buffer, bytearray):
            c_buf = (ctypes.c_ubyte * len(buffer)).from_buffer(buffer)
            return ctypes.cast(c_buf, ctypes.c_void_p), len(buffer), c_buf

        raise TypeError(
            "Unsupported buffer type for plugin exchange. "
            "Use numpy.ndarray, torch.Tensor, memoryview, or bytearray."
        )


__all__ = ["COUPLER_PLUGIN_ABI_VERSION", "PluginABIError", "PluginHost", "PluginLifecycleError"]
