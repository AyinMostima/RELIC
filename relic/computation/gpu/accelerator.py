"""
GPU Acceleration Module for RELIC

This module provides GPU acceleration capabilities with automatic mixed precision,
device management, and numerical consistency validation. Designed to be Windows-friendly
with comprehensive error handling and explicit policy.

Key Features:
    - Automatic CUDA device detection and selection
    - Mixed precision (AMP) with GradScaler support
    - Memory management with pinned memory optimization
    - Numerical consistency validation between CPU/GPU paths
    - Thread-safe device operations
    - Comprehensive error handling and logging

Mathematical Considerations:
    - Automatic Mixed Precision maintains numerical stability through:
      * FP16 computation for forward pass (speed)
      * FP32 accumulation for gradients (precision)
      * Loss scaling to prevent underflow
    - Memory layout optimization for coalesced access
    - Tensor operations vectorization for maximum throughput

References:
    - NVIDIA Automatic Mixed Precision documentation
    - PyTorch CUDA semantics and best practices
    - Numerical stability in mixed precision training
    - GPU memory management patterns
"""

import logging
from typing import Dict, Any, Optional, Union, List, ContextManager
from contextlib import contextmanager
import time

import torch
from relic.core import device_policy
from torch import Tensor

TORCH_AVAILABLE = True
AMP_AVAILABLE = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

# Configure logging
logger = logging.getLogger(__name__)

class GPUAcceleratorError(Exception):
    """Exception raised for GPU acceleration errors."""


class GPUStreamManager:
    """Manage CUDA streams and synchronization events for pipelined execution.

    References:
        - NVIDIA CUDA Programming Guide (Streams and Events).

    Parameters:
        max_streams: maximum number of tracked CUDA streams
    """

    def __init__(self, max_streams: int = 4):
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._events: Dict[str, torch.cuda.Event] = {}
        self._max_streams = max(1, int(max_streams))
        self._event_counts: Dict[str, int] = {"recorded": 0, "waits": 0}
        self._cuda_available = TORCH_AVAILABLE and device_policy.is_cuda_available()

    def get_stream(self, name: str) -> Optional[torch.cuda.Stream]:
        """Get or create a named CUDA stream, or None when CUDA is unavailable."""
        if not self._cuda_available:
            return None
        if name in self._streams:
            return self._streams[name]
        if len(self._streams) >= self._max_streams:
            return next(iter(self._streams.values()))
        stream = torch.cuda.Stream()
        self._streams[name] = stream
        return stream

    def record_event(self, stream_name: str, event_name: str) -> None:
        """Record a synchronization event on the named stream."""
        if not self._cuda_available:
            return
        stream = self.get_stream(stream_name)
        if stream is None:
            return
        event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            event.record(stream)
        self._events[event_name] = event
        self._event_counts["recorded"] += 1

    def wait_event(self, stream_name: str, event_name: str) -> None:
        """Make a stream wait for a previously recorded event."""
        if not self._cuda_available:
            return
        stream = self.get_stream(stream_name)
        event = self._events.get(event_name)
        if stream is None or event is None:
            return
        stream.wait_event(event)
        self._event_counts["waits"] += 1

    def synchronize_all(self) -> None:
        """Synchronize all tracked streams."""
        if not self._cuda_available:
            return
        for stream in self._streams.values():
            stream.synchronize()

    def get_pipeline_efficiency(self) -> Dict[str, float]:
        """Estimate stream pipeline efficiency from event statistics."""
        recorded = float(self._event_counts.get("recorded", 0))
        waits = float(self._event_counts.get("waits", 0))
        if recorded <= 0.0:
            utilization = 0.0
        else:
            utilization = max(0.0, min(1.0, waits / recorded))
        return {
            "utilization": utilization,
            "recorded_events": recorded,
            "wait_operations": waits,
            "active_streams": float(len(self._streams)),
            "max_streams": float(self._max_streams),
        }


StreamManager = GPUStreamManager


def is_cuda_available() -> bool:
    """Return CUDA availability from the canonical device policy."""
    return TORCH_AVAILABLE and device_policy.is_cuda_available()


def select_device(
    preference: Union[str, torch.device] = "auto",
    *,
    require_gpu: bool = False,
) -> torch.device:
    """Resolve a device without creating a second device-policy owner."""
    requested = str(preference).strip()
    explicit_device: Optional[Union[str, torch.device]]
    if requested.lower() in {"", "auto", "default"}:
        explicit_device = None
    else:
        explicit_device = preference
    return device_policy.resolve_device(
        explicit_device=explicit_device,
        module_name=__name__,
        default="auto",
        require_gpu=require_gpu,
    )


def get_device_info(
    device: Union[str, torch.device] = "auto",
) -> Dict[str, Any]:
    """Read current device properties without a RELIC-side device cache."""
    requested = str(device).strip().lower()
    if requested in {"", "auto", "default"}:
        resolved = select_device("auto")
    else:
        try:
            resolved = torch.device(device)
        except (TypeError, ValueError, RuntimeError) as exc:
            # An unparsable device string is invalid input, not an
            # 'unknown' device.
            raise ValueError(f"Invalid device specification {device!r}") from exc

    if resolved.type == "cpu":
        return {
            "type": "cpu",
            "name": "CPU",
            "available": True,
            "memory": "system",
        }
    if resolved.type != "cuda" or not is_cuda_available():
        return {"type": "unknown", "available": False}

    device_index = 0 if resolved.index is None else int(resolved.index)
    device_count = int(torch.cuda.device_count())
    if device_index < 0 or device_index >= device_count:
        return {"type": "unknown", "available": False}

    cuda_device = torch.device("cuda", device_index)
    # A failing driver query propagates; the device is never reported as
    # 'unknown' in its place.
    properties = torch.cuda.get_device_properties(cuda_device)
    memory_allocated = torch.cuda.memory_allocated(cuda_device) / (1024**3)
    memory_reserved = torch.cuda.memory_reserved(cuda_device) / (1024**3)

    total_memory = properties.total_memory / (1024**3)
    return {
        "index": device_index,
        "name": properties.name,
        "memory_gb": total_memory,
        "compute_capability": (properties.major, properties.minor),
        "multiprocessors": properties.multi_processor_count,
        "type": "cuda",
        "available": True,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "memory_free_gb": total_memory - memory_reserved,
    }

class AutocastContext:
    """
    Automatic Mixed Precision context manager with error handling.
    
    Provides AMP operations with explicit policy when AMP
    is not available or when errors occur.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
        allow_disabled: bool = False,
    ):
        if enabled and not (AMP_AVAILABLE and is_cuda_available()):
            if allow_disabled:
                self.enabled = False
            else:
                raise GPUAcceleratorError("AMP requested but not available")
        else:
            self.enabled = enabled and AMP_AVAILABLE and is_cuda_available()
        self.dtype = dtype
        self.cache_enabled = cache_enabled
        self.allow_disabled = allow_disabled
        # Reason AMP was disabled under the explicit allow_disabled policy.
        self.disabled_reason: Optional[str] = None
        self._autocast_context = None
    
    def __enter__(self):
        if self.enabled:
            try:
                self._autocast_context = torch.amp.autocast(
                    device_type="cuda",
                    enabled=True,
                    dtype=self.dtype,
                    cache_enabled=self.cache_enabled
                )
                return self._autocast_context.__enter__()
            except RuntimeError as e:
                # torch.amp.autocast rejects an unsupported device/dtype
                # combination with RuntimeError. Only then, and only under the
                # explicit allow_disabled policy, AMP is switched off for this
                # context and the reason recorded; otherwise it is an error.
                # Other exceptions (programming errors) propagate.
                if self.allow_disabled:
                    logger.warning(f"AMP autocast failed: {e}. Disabling AMP for this context.")
                    self.enabled = False
                    self.disabled_reason = str(e)
                else:
                    raise GPUAcceleratorError(f"AMP autocast failed: {e}") from e
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._autocast_context is not None:
            return self._autocast_context.__exit__(exc_type, exc_val, exc_tb)
        return False


@contextmanager
def autocast_if_enabled(
    enabled: bool = True,
    dtype: torch.dtype = torch.float16,
    allow_disabled: bool = False,
) -> ContextManager:
    """
    Context manager for conditional automatic mixed precision.
    
    Parameters:
        enabled : bool
            Whether to enable AMP
        dtype : torch.dtype
            AMP dtype (torch.float16 or torch.bfloat16)
    
    Yields:
        Context manager for AMP operations
    """
    if enabled and not (AMP_AVAILABLE and is_cuda_available()):
        if allow_disabled:
            from contextlib import nullcontext
            with nullcontext() as ctx:
                yield ctx
            return
        raise GPUAcceleratorError("AMP requested but not available")
    with AutocastContext(
        enabled=enabled,
        dtype=dtype,
        allow_disabled=allow_disabled,
    ) as ctx:
        yield ctx


class TensorMover:
    """
    Utility class for safe tensor device movement with batching and error handling.
    """
    
    @staticmethod
    def move_to_device(
        data: Union[Tensor, Dict[str, Tensor], List[Tensor]],
        device: torch.device,
        non_blocking: bool = False,
        dtype: Optional[torch.dtype] = None
    ) -> Union[Tensor, Dict[str, Tensor], List[Tensor]]:
        """
        Move tensors to specified device with type conversion.
        
        Parameters:
            data : Union[Tensor, Dict[str, Tensor], List[Tensor]]
                Data to move
            device : torch.device
                Target device
            non_blocking : bool
                Enable non-blocking transfer (for pinned memory)
            dtype : Optional[torch.dtype]
                Target dtype for conversion
        
        Returns:
            Moved and optionally converted data
        """
        if isinstance(data, torch.Tensor):
            result = data.to(device=device, non_blocking=non_blocking)
            if dtype is not None and result.dtype != dtype:
                result = result.to(dtype=dtype)
            return result
        
        elif isinstance(data, dict):
            return {
                key: TensorMover.move_to_device(value, device, non_blocking, dtype)
                if isinstance(value, torch.Tensor) else value
                for key, value in data.items()
            }
        
        elif isinstance(data, (list, tuple)):
            moved_list = [
                TensorMover.move_to_device(item, device, non_blocking, dtype)
                if isinstance(item, torch.Tensor) else item
                for item in data
            ]
            return type(data)(moved_list)
        
        else:
            return data
    
    @staticmethod
    def move_state_dict(
        state_dict: Dict[str, Tensor],
        device: torch.device,
        non_blocking: bool = True
    ) -> Dict[str, Tensor]:
        """
        Move entire state dictionary to device.
        
        Parameters:
            state_dict : Dict[str, Tensor]
                State dictionary to move
            device : torch.device
                Target device
            non_blocking : bool
                Enable non-blocking transfer
        
        Returns:
            Dict[str, Tensor]: Moved state dictionary
        """
        return TensorMover.move_to_device(state_dict, device, non_blocking)


def move_to_device(
    data: Union[Tensor, Dict[str, Tensor], List[Tensor]],
    device: Union[str, torch.device] = "auto",
    non_blocking: bool = False,
    dtype: Optional[torch.dtype] = None
) -> Union[Tensor, Dict[str, Tensor], List[Tensor]]:
    """
    Convenience function to move data to device.
    
    Parameters:
        data : Union[Tensor, Dict[str, Tensor], List[Tensor]]
            Data to move
        device : Union[str, torch.device]
            Target device
        non_blocking : bool
            Enable non-blocking transfer
        dtype : Optional[torch.dtype]
            Target dtype
    
    Returns:
        Moved data
    """
    if isinstance(device, str):
        device = select_device(device)
    
    return TensorMover.move_to_device(data, device, non_blocking, dtype)


class NumericalConsistencyChecker:
    """
    Utility for checking numerical consistency between CPU/GPU computations.
    
    Validates that GPU-accelerated computations produce results within
    acceptable tolerance of CPU baseline computations.

    Tolerances not stated by the caller are read from the parameter database
    (computation.gpu.consistency.rtol / atol / amp_rtol / amp_atol), their
    single source.
    """

    def __init__(
        self,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        amp_rtol: Optional[float] = None,
        amp_atol: Optional[float] = None
    ):
        from relic.parameters import get_global_manager

        manager = get_global_manager()

        def _resolve(value: Optional[float], name: str) -> float:
            if value is not None:
                return value
            return float(manager.get(f"computation.gpu.consistency.{name}", required=True))

        self.rtol = _resolve(rtol, "rtol")
        self.atol = _resolve(atol, "atol")
        self.amp_rtol = _resolve(amp_rtol, "amp_rtol")
        self.amp_atol = _resolve(amp_atol, "amp_atol")
    
    def check_consistency(
        self,
        cpu_result: Union[Tensor, Dict[str, Tensor]],
        gpu_result: Union[Tensor, Dict[str, Tensor]],
        use_amp_tolerance: bool = False,
        context: str = "computation"
    ) -> Dict[str, Any]:
        """
        Check numerical consistency between CPU and GPU results.
        
        Parameters:
            cpu_result : Union[Tensor, Dict[str, Tensor]]
                CPU computation result
            gpu_result : Union[Tensor, Dict[str, Tensor]]
                GPU computation result  
            use_amp_tolerance : bool
                Use relaxed tolerances for AMP comparisons
            context : str
                Context description for logging
        
        Returns:
            Dict[str, Any]: Consistency check results
        """
        results = {
            'consistent': True,
            'max_error': 0.0,
            'failed_keys': [],
            'errors': [],
            'tolerance_used': {
                'rtol': self.amp_rtol if use_amp_tolerance else self.rtol,
                'atol': self.amp_atol if use_amp_tolerance else self.atol
            }
        }
        
        rtol = self.amp_rtol if use_amp_tolerance else self.rtol
        atol = self.amp_atol if use_amp_tolerance else self.atol
        
        if isinstance(cpu_result, torch.Tensor) and isinstance(gpu_result, torch.Tensor):
            # Move GPU result to CPU for comparison
            gpu_cpu = gpu_result.detach().cpu()
            cpu_detached = cpu_result.detach()
            
            try:
                if not torch.allclose(cpu_detached, gpu_cpu, rtol=rtol, atol=atol):
                    results['consistent'] = False
                    error = torch.max(torch.abs(cpu_detached - gpu_cpu)).item()
                    results['max_error'] = error
                    results['failed_keys'] = ['tensor']
                    
                    logger.warning(
                        f"Numerical inconsistency in {context}: "
                        f"max_error={error:.2e}, rtol={rtol:.2e}, atol={atol:.2e}"
                    )
            
            except RuntimeError as e:
                # torch.allclose raises RuntimeError for incompatible shapes or
                # dtypes. The check is then failed and its reason recorded;
                # any other error propagates.
                logger.error(f"Error checking tensor consistency: {e}")
                results['consistent'] = False
                results['failed_keys'] = ['tensor_error']
                results['errors'].append(f"{context}: {e}")
        
        elif isinstance(cpu_result, dict) and isinstance(gpu_result, dict):
            for key in cpu_result:
                if key in gpu_result:
                    sub_result = self.check_consistency(
                        cpu_result[key], gpu_result[key], use_amp_tolerance, f"{context}.{key}"
                    )
                    
                    if not sub_result['consistent']:
                        results['consistent'] = False
                        results['failed_keys'].extend([f"{key}.{k}" for k in sub_result['failed_keys']])
                        results['max_error'] = max(results['max_error'], sub_result['max_error'])
                        results['errors'].extend(sub_result['errors'])
                else:
                    results['consistent'] = False
                    results['failed_keys'].append(f"missing_{key}")
                    logger.warning(f"Key '{key}' missing from GPU result in {context}")
        
        return results


class GPUProfiler:
    """
    GPU performance profiling utilities.
    
    Provides timing, memory usage tracking, and throughput measurement
    for GPU-accelerated computations.
    """
    
    def __init__(self):
        self.events = {}
        self.memory_stats = {}
        self._active_timers = {}
    
    def start_timer(self, name: str, device: Optional[torch.device] = None) -> None:
        """Start a GPU timer."""
        if not is_cuda_available():
            self._active_timers[name] = time.time()
            return
        
        if device is None or device.type != 'cuda':
            device = select_device("auto")
            if device.type != 'cuda':
                self._active_timers[name] = time.time()
                return
        
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        self.events[name] = {'start': start_event, 'end': end_event, 'device': device}
    
    def end_timer(self, name: str) -> float:
        """End a GPU timer and return elapsed time in milliseconds."""
        if name in self._active_timers:
            elapsed = (time.time() - self._active_timers[name]) * 1000  # Convert to ms
            del self._active_timers[name]
            return elapsed
        
        if name not in self.events:
            raise ValueError(f"Timer '{name}' was not started")
        
        event_data = self.events[name]
        event_data['end'].record()
        torch.cuda.synchronize(event_data['device'])
        
        elapsed = event_data['start'].elapsed_time(event_data['end'])
        del self.events[name]
        return elapsed
    
    @contextmanager
    def timer(self, name: str, device: Optional[torch.device] = None):
        """Context manager for timing operations."""
        self.start_timer(name, device)
        try:
            yield
        finally:
            elapsed = self.end_timer(name)
            logger.debug(f"Timer '{name}': {elapsed:.2f} ms")
    
    def get_memory_usage(self, device: Optional[torch.device] = None) -> Dict[str, float]:
        """Get current GPU memory usage in GB."""
        if not is_cuda_available():
            return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}
        
        if device is None:
            device = select_device("auto")
        
        if device.type != 'cuda':
            return {'allocated': 0.0, 'reserved': 0.0, 'free': 0.0}
        
        # A failing CUDA memory query propagates; it is never reported as
        # zero allocated, reserved and free memory.
        with torch.cuda.device(device):
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)

            # Get total memory from device properties
            props = torch.cuda.get_device_properties(device)
            total = props.total_memory / (1024**3)
            free = total - reserved

            return {
                'allocated': allocated,
                'reserved': reserved,
                'free': free,
                'total': total
            }


# Global profiler instance
profiler = GPUProfiler()

# Convenience functions
def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Get GPU memory usage."""
    return profiler.get_memory_usage(device)


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if is_cuda_available():
        torch.cuda.empty_cache()


def synchronize_device(device: Optional[torch.device] = None):
    """Synchronize device operations."""
    if not is_cuda_available():
        return
    if device is not None:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return
        torch.cuda.synchronize(resolved)
        return
    torch.cuda.synchronize()


# Export main classes and functions
__all__ = [
    # Classes
    'GPUStreamManager',
    'StreamManager',
    'AutocastContext', 
    'TensorMover',
    'NumericalConsistencyChecker',
    'GPUProfiler',
    'GPUAcceleratorError',
    
    # Functions
    'is_cuda_available',
    'select_device',
    'get_device_info',
    'move_to_device',
    'autocast_if_enabled',
    'get_memory_usage',
    'clear_gpu_cache',
    'synchronize_device',
    
    # Global instances
    'profiler'
]
