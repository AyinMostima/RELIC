"""
Computational Optimization Layer for RELIC

This package exposes bounded computation utilities:
- device selection through the canonical device policy;
- explicit parallel helpers and backend availability checks;
- one stateless ``torch.compile`` generic callable adapter, ``maybe_compile``;
- local health probes and synthetic microbenchmark support.

The generic callable adapter is not scientific-kernel approval, an arbitrary
Python purity proof, or evidence of stateful, stochastic, input-mutating,
cross-backend, cross-device, release, or long-integration equivalence.
``validate_numerics=True`` remains ``[N]`` and rejects before compilation or
callable execution. Local probe success is scoped to the requested function,
backend, device, inputs, and current environment.

Usage Example:
    >>> from relic.computation import select_device, healthcheck
    >>> device = select_device("auto")
    >>> status = healthcheck(verbose=True)

References:
    - PyTorch documentation for GPU, torch.compile, and distributed computing
    - Best practices for scientific computing optimization
    - Windows-specific distributed computing limitations
"""

import warnings
from typing import Dict, Any, Optional, Callable, List, TYPE_CHECKING
import logging
import importlib
import importlib.util as importlib_util
from ..utils.imports import module_available

if TYPE_CHECKING:
    import torch
from relic.core import device_policy

# Configure logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__torch_min_version__ = "1.12.0"


def _check_dependencies() -> Dict[str, bool]:
    """
    Check availability of computational optimization dependencies.
    
    Returns:
        Dict[str, bool]: Availability status of each dependency
    """
    status = {}
    
    # Core PyTorch
    if not module_available("torch"):
        status['torch'] = False
        logger.error("PyTorch not available - computation optimization disabled")
        return status
    # PyTorch is installed (checked above); an import or version-probe
    # failure propagates with its cause and is never reported as an
    # unavailable PyTorch.
    torch = importlib.import_module("torch")
    status['torch'] = True
    status['torch_version'] = torch.__version__

    # Check minimum version
    version_mod = importlib.import_module("packaging.version")
    if version_mod.parse(torch.__version__) < version_mod.parse(__torch_min_version__):
        warnings.warn(f"PyTorch {torch.__version__} is older than minimum {__torch_min_version__}")
    
    # GPU availability
    status['cuda'] = device_policy.is_cuda_available()
    if status['cuda']:
        status['cuda_version'] = torch.version.cuda
        status['cudnn_version'] = torch.backends.cudnn.version()
        status['cuda_device_count'] = torch.cuda.device_count()
        
        # Check for CUDA devices
        cuda_devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cuda_devices.append({
                'name': props.name,
                'memory': props.total_memory // (1024**3),  # GB
                'compute_capability': f"{props.major}.{props.minor}"
            })
        status['cuda_devices'] = cuda_devices
    
    # Import-time checks report API presence only. Runtime readiness requires
    # an explicit health probe that executes the returned compiled callable.
    status['torch_jit'] = hasattr(torch, 'jit')
    status['torch_compile'] = hasattr(torch, 'compile')
    
    # Distributed computing
    status['distributed'] = torch.distributed.is_available()
    if status['distributed']:
        # Check available backends
        available_backends = []
        if torch.distributed.is_gloo_available():
            available_backends.append('gloo')
        if torch.distributed.is_nccl_available():
            available_backends.append('nccl')
        if torch.distributed.is_mpi_available():
            available_backends.append('mpi')
        status['distributed_backends'] = available_backends
    
    # Mixed precision
    if module_available("torch.cuda.amp"):
        status['amp'] = status['cuda']  # AMP requires CUDA
    else:
        status['amp'] = False
    
    # Parallel processing
    import multiprocessing
    status['parallel'] = True
    status['cpu_count'] = multiprocessing.cpu_count()
    
    # Experimental dependencies (optional)
    experimental_deps = {
        'triton': False,
        'cupy': False,
        'flash_attn': False
    }
    
    for dep in experimental_deps:
        if importlib_util.find_spec(dep) is not None:
            experimental_deps[dep] = True
    
    status['experimental'] = experimental_deps
    
    return status


# Global dependency status (computed once at import)
_DEPENDENCIES = _check_dependencies()


def is_available(feature: str) -> bool:
    """
    Check if a computational feature is available.
    
    Parameters:
        feature : str
            Feature name ('cuda', 'jit', 'compile', 'distributed', 'amp', 'parallel')
    
    Returns:
        bool: True if feature is available
    """
    normalized = str(feature).strip().lower()
    if normalized in {"compile", "torch_compile"}:
        # API presence is not runtime readiness. Call healthcheck with
        # runtime_probe=True before enabling compilation.
        return False
    return bool(_DEPENDENCIES.get(normalized, False))


# The experimental feature flags have one owner, relic.computation.experimental;
# the package re-exports it (the former copy here accepted any flag name).
from .experimental import is_experimental_enabled  # noqa: E402


def select_device(preference: str = "auto") -> "torch.device":
    """
    Select computation device with explicit policy.
    
    Parameters:
        preference : str
            Device preference ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
        torch.device: Selected device
    """
    if not _DEPENDENCIES.get('torch', False):
        raise RuntimeError("PyTorch not available")

    requested = str(preference).strip()
    lowered = requested.lower()
    require_gpu = lowered.startswith("cuda")
    try:
        return device_policy.resolve_device(
            explicit_device=requested or "auto",
            module_name=__name__,
            distributed=lowered in {"", "auto", "default"},
            require_gpu=require_gpu,
        )
    except RuntimeError as exc:
        if require_gpu:
            raise RuntimeError("CUDA requested but not available") from exc
        raise
    except (TypeError, ValueError, KeyError, AttributeError, ImportError, OSError) as exc:
        raise ValueError(f"Invalid device '{preference}'") from exc


def parallel_map(
    fn: Callable,
    items: List[Any],
    backend: str = "threads",
    max_workers: Optional[int] = None,
    allow_sequential: bool = False,
    computation_config: Any = None,
) -> List[Any]:
    """
    Parallel execution with explicit policy.
    
    Parameters:
        fn : Callable
            Function to apply to each item
        items : List[Any]
            Items to process
        backend : str
            Parallel backend ('threads', 'processes', 'auto')
        max_workers : Optional[int]
            Maximum number of workers
        allow_sequential : bool
            Whether sequential execution is allowed when parallel is unavailable
    
    Returns:
        List[Any]: Results in same order as input
    """
    from .parallel.executor import parallel_map as canonical_parallel_map

    runtime_kwargs = {
        "backend": backend,
        "max_workers": max_workers,
        "allow_sequential": allow_sequential,
    }
    if computation_config is not None:
        runtime_kwargs["computation_config"] = computation_config
    return canonical_parallel_map(fn, items, **runtime_kwargs)


def _compile_probe_status(
    *,
    requested_backend: str,
    requested_device: str,
) -> Dict[str, Any]:
    """Return the stable compile-runtime health schema without executing it."""
    return {
        "probe_scope": "non_scientific_single_function_current_environment",
        "scientific_kernel_ready": False,
        "scientific_equivalence": False,
        "api_available": bool(_DEPENDENCIES.get("torch_compile", False)),
        "probe_status": "not_run",
        "requested_backend": requested_backend,
        "effective_backend": None,
        "requested_device": requested_device,
        "effective_device": None,
        "callable_created": False,
        "first_call_attempted": False,
        "first_call_completed": False,
        "runtime_ready": None,
        "fallback_used": False,
        "blocker": None,
    }


def _probe_compile_runtime(
    *,
    backend: str = "inductor",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Construct, execute, and validate one non-scientific compile probe."""
    result = _compile_probe_status(
        requested_backend=backend,
        requested_device=device,
    )
    result["probe_status"] = "failed"
    result["runtime_ready"] = False
    if not result["api_available"]:
        result["blocker"] = {
            "phase": "api_check",
            "type": "RuntimeError",
            "message": "torch.compile API is unavailable",
        }
        return result

    torch_module = importlib.import_module("torch")
    try:
        effective_device = device_policy.resolve_device(
            explicit_device=device,
            module_name=__name__,
            distributed=False,
        )
        example = torch_module.tensor(
            [1.0, 2.0],
            dtype=torch_module.float32,
            device=effective_device,
        )

        def probe_function(value):
            return value * 2.0 + 1.0

        compiled = maybe_compile(
            probe_function,
            mode="compile",
            backend=backend,
            allow_uncompiled=False,
        )
        result["callable_created"] = True
        result["effective_device"] = str(effective_device)
    except Exception as exc:  # noqa: BLE001 - health ledger preserves blocker
        result["blocker"] = {
            "phase": "construction",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return result

    result["first_call_attempted"] = True
    try:
        output = compiled(example)
        if effective_device.type == "cuda":
            torch_module.cuda.synchronize(effective_device)
        result["first_call_completed"] = True
    except Exception as exc:  # noqa: BLE001 - health ledger preserves blocker
        result["blocker"] = {
            "phase": "first_call",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return result

    expected = probe_function(example)
    if (
        not isinstance(output, torch_module.Tensor)
        or output.shape != expected.shape
        or output.dtype != expected.dtype
        or not torch_module.equal(output, expected)
    ):
        result["blocker"] = {
            "phase": "result_validation",
            "type": "RuntimeError",
            "message": "compiled callable output did not match the eager probe",
        }
        return result

    result["probe_status"] = "passed"
    result["runtime_ready"] = True
    result["effective_backend"] = backend
    result["blocker"] = None
    return result


def healthcheck(
    verbose: bool = False,
    *,
    runtime_probe: bool = False,
    compile_backend: str = "inductor",
    compile_device: str = "cpu",
) -> Dict[str, Any]:
    """
    Comprehensive health check of computation capabilities.
    
    Parameters:
        verbose : bool
            Enable verbose output with detailed information
    
    Returns:
        Dict[str, Any]: Health check results
    """
    status = {
        'overall': 'healthy',
        'dependencies': _DEPENDENCIES.copy(),
        'capabilities': {
            'torch_compile': (
                _probe_compile_runtime(
                    backend=compile_backend,
                    device=compile_device,
                )
                if runtime_probe
                else _compile_probe_status(
                    requested_backend=compile_backend,
                    requested_device=compile_device,
                )
            )
        },
        'recommendations': [],
        'warnings': []
    }
    
    # Check critical dependencies
    if not _DEPENDENCIES.get('torch', False):
        status['overall'] = 'critical'
        status['recommendations'].append("Install PyTorch for computation optimization")
        return status
    
    # Check GPU availability
    if not _DEPENDENCIES.get('cuda', False):
        status['warnings'].append("CUDA not available - GPU acceleration disabled")
        status['recommendations'].append("Install CUDA-enabled PyTorch for GPU acceleration")
    elif verbose and 'cuda_devices' in _DEPENDENCIES:
        logger.info(f"Available CUDA devices: {_DEPENDENCIES['cuda_devices']}")
    
    # Check JIT compilation
    if not _DEPENDENCIES.get('torch_jit', False) and not _DEPENDENCIES.get('torch_compile', False):
        status['warnings'].append("JIT compilation not available")
        status['recommendations'].append("Update to PyTorch 2.0+ for torch.compile support")
    compile_status = status['capabilities']['torch_compile']
    if compile_status['probe_status'] == 'failed':
        status['overall'] = 'degraded'
        blocker = compile_status['blocker'] or {}
        status['warnings'].append(
            "torch.compile runtime probe failed during "
            f"{blocker.get('phase', 'unknown')}: "
            f"{blocker.get('message', 'unknown error')}"
        )
    
    # Check distributed computing
    if not _DEPENDENCIES.get('distributed', False):
        status['warnings'].append("Distributed computing not available")
    elif 'distributed_backends' in _DEPENDENCIES:
        backends = _DEPENDENCIES['distributed_backends']
        if 'nccl' not in backends and _DEPENDENCIES.get('cuda', False):
            status['warnings'].append("NCCL not available - multi-GPU performance may be limited")
    
    # Platform-specific checks
    import platform
    if platform.system() == "Windows":
        if 'nccl' in _DEPENDENCIES.get('distributed_backends', []):
            status['warnings'].append("NCCL on Windows may have limitations - prefer Gloo backend")
            status['recommendations'].append("Use Gloo backend for distributed computing on Windows")
    
    # Set overall status based on warnings
    if len(status['warnings']) > 3:
        status['overall'] = 'degraded'
    
    if verbose:
        # Print detailed status
        logger.info(f"Computation Health Check - Status: {status['overall']}")
        for warning in status['warnings']:
            logger.warning(warning)
        for recommendation in status['recommendations']:
            logger.info(f"Recommendation: {recommendation}")
    
    return status


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available computation devices.
    
    Returns:
        Dict[str, Any]: Device information
    """
    info = {'devices': []}
    
    if not _DEPENDENCIES.get('torch', False):
        return info
    
    
    # CPU info
    cpu_info = {
        'type': 'cpu',
        'name': 'CPU',
        'available': True,
        'memory': 'system_memory'  # Could query psutil if available
    }
    info['devices'].append(cpu_info)
    
    # CUDA devices
    if _DEPENDENCIES.get('cuda', False):
        for i, device_info in enumerate(_DEPENDENCIES.get('cuda_devices', [])):
            cuda_info = {
                'type': 'cuda',
                'index': i,
                'name': device_info['name'],
                'memory_gb': device_info['memory'],
                'compute_capability': device_info['compute_capability'],
                'available': True
            }
            info['devices'].append(cuda_info)
    
    return info


# Convenience re-exports for CLI/examples.
from .gpu import move_to_device
from .jit import maybe_compile

# Export main interface
__all__ = [
    # Core functions
    'is_available',
    'is_experimental_enabled',
    'select_device',
    'maybe_compile',
    'parallel_map',
    'healthcheck',
    'get_device_info',
    'move_to_device',

    # Status information
    '_DEPENDENCIES'
]
