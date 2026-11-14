"""
GPU Acceleration Submodule

Provides GPU acceleration capabilities for RELIC including:
- Automatic device selection and management
- Mixed precision computation (AMP)
- Memory optimization and profiling
- Numerical consistency validation
- Windows-friendly error handling

Main Components:
    - AutocastContext: Mixed precision context manager
    - TensorMover: Safe tensor device movement utilities
    - NumericalConsistencyChecker: CPU/GPU result validation
    - GPUProfiler: Performance monitoring and timing
"""

# Import main accelerator components
from .accelerator import (
    # Classes
    AutocastContext,
    TensorMover,
    NumericalConsistencyChecker,
    GPUProfiler,
    GPUAcceleratorError,

    # Functions
    is_cuda_available,
    select_device,
    get_device_info,
    move_to_device,
    autocast_if_enabled,
    get_memory_usage,
    clear_gpu_cache,
    synchronize_device,

    # Global instances
    profiler
)

GPU_AVAILABLE = True


# Health check function
def gpu_healthcheck() -> dict:
    """
    Perform GPU-specific health check.
    
    Returns:
        dict: Health status with GPU-specific information
    """
    status = {
        'gpu_available': GPU_AVAILABLE,
        'cuda_available': is_cuda_available(),
        'devices': [],
        'warnings': [],
        'recommendations': []
    }
    
    if not GPU_AVAILABLE:
        status['warnings'].append("GPU module not imported successfully")
        status['recommendations'].append("Check PyTorch installation")
        return status
    
    if not is_cuda_available():
        status['warnings'].append("CUDA not available")
        status['recommendations'].append("Install CUDA-enabled PyTorch")
        return status
    
    # Get device information. A failing CUDA query propagates; it is not
    # turned into a warning of an otherwise healthy report.
    import torch
    for i in range(torch.cuda.device_count()):
        device_info = get_device_info(f'cuda:{i}')
        status['devices'].append(device_info)

    if len(status['devices']) == 0:
        status['warnings'].append("No CUDA devices detected")
    
    # Check memory status
    memory_info = get_memory_usage()
    if memory_info['free'] < 1.0:  # Less than 1GB free
        status['warnings'].append("Low GPU memory available")
    
    return status


# Export interface
__all__ = [
    # Availability flag
    'GPU_AVAILABLE',
    
    # Classes  
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
    'gpu_healthcheck',
    
    # Global instances
    'profiler'
]
