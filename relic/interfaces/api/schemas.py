"""
RELIC API Data Models

Pydantic models for request/response validation in the RELIC REST API.
These models ensure consistent data structures, validation, and documentation
across all API endpoints.

Key Features:
    - Comprehensive input validation with type checking
    - Consistent response format across all endpoints
    - Automatic OpenAPI schema generation
    - Error handling with detailed diagnostics
    - Support for complex scientific data structures
    - CF-1.8 compliant metadata handling

Model Categories:
    - Base response models (APIResponse, ErrorDetail)
    - Health monitoring models (HealthStatus, BenchmarkResult)
    - Data I/O models (DataReadRequest, DataWriteRequest)
    - Compute models (ComputeRequest, DeviceInfo)
    - Optimization models (CalibrationRequest, AssimilationRequest)
    - Configuration models (ConfigValidation, ParameterSet)

Authors: Yuchong Long
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _api_data_root() -> Path:
    override = os.getenv("RELIC_API_DATA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    # relic/interfaces/api/schemas.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3]


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_safe_local_path(raw_path: str, *, must_exist: bool) -> str:
    """
    Validate path against traversal and workspace-escape attacks.

    Rules:
    - Relative paths are resolved under `RELIC_API_DATA_ROOT` (or project root).
    - Absolute paths are allowed only if they remain under the same root.
    - Optional existence check can be enabled for read-side operations.
    """
    if not isinstance(raw_path, str):
        raise TypeError("Path must be a string")
    trimmed = raw_path.strip()
    if not trimmed:
        raise ValueError("Path cannot be empty")

    root = _api_data_root()
    candidate = Path(trimmed).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not _is_subpath(candidate, root):
        raise ValueError(
            "Path traversal blocked: path must stay inside API data root "
            f"({root})."
        )
    if must_exist and not candidate.exists():
        raise ValueError(f"Path does not exist: {candidate}")
    return str(candidate)

# Base Response Models

class RequestModel(BaseModel):
    """Base of every API request body and its nested components.

    Unknown fields are rejected (HTTP 422 through the RequestValidationError
    handler): a misspelled or removed field is never silently dropped. A field
    that carries caller-defined keys is typed as a mapping (Dict[str, Any])
    at field level instead of opening the whole model.
    """

    model_config = ConfigDict(extra="forbid")


class APIResponse(BaseModel):
    """Standard API response format"""
    status: Literal["success", "error", "warning"] = Field(..., description="Response status")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    diagnostics: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic information")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Response metadata")
    paths: Optional[Dict[str, str]] = Field(default=None, description="File paths for outputs")
    
    model_config = ConfigDict(extra="allow")

class ErrorDetail(BaseModel):
    """Detailed error information"""
    type: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestions: Optional[List[str]] = Field(None, description="Suggested solutions")
    traceback: Optional[str] = Field(default=None, description="Stack trace for debugging")

class ValidationError(BaseModel):
    """Input validation error"""
    field: str = Field(..., description="Field name with error")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="Invalid value provided")
    valid_options: Optional[List[Any]] = Field(None, description="Valid options")

# Health and Monitoring Models

class HealthStatus(BaseModel):
    """System health status"""
    overall: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall health status")
    subsystems: Dict[str, Dict[str, Any]] = Field(..., description="Individual subsystem status")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    timestamp: float = Field(..., description="Check timestamp")

class DeviceInfo(BaseModel):
    """Device information"""
    name: str = Field(..., description="Device name")
    type: Literal["cpu", "cuda", "mps"] = Field(..., description="Device type")
    memory_total: Optional[float] = Field(None, description="Total memory in GB")
    memory_available: Optional[float] = Field(None, description="Available memory in GB")
    compute_capability: Optional[str] = Field(None, description="Compute capability")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional device properties")

class BenchmarkResult(BaseModel):
    """Performance benchmark results"""
    target: str = Field(..., description="Benchmark target function")
    device: str = Field(..., description="Device used for benchmark")
    compilation_modes: Dict[str, Dict[str, float]] = Field(..., description="Results by compilation mode")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Benchmark metadata")

# Data I/O Models

class DataReadRequest(RequestModel):
    """Data reading request"""
    path: str = Field(..., description="File path to read")
    variables: Optional[List[str]] = Field(None, description="Variables to read")
    format: Literal["auto", "netcdf4", "zarr", "csv", "hdf5"] = Field(
        "auto", description="Input format; \"auto\" detects it from the file"
    )
    chunks: Optional[Dict[str, int]] = Field(None, description="Chunk sizes for lazy loading")
    to_torch: bool = Field(False, description="Convert to PyTorch tensors")
    device: str = Field("cpu", description="Target device for tensors")
    dtype: str = Field("float32", description="Target tensor dtype")
    
    @field_validator('path', mode='before')
    def validate_path(cls, v: str):
        return validate_safe_local_path(v, must_exist=True)

class DataWriteRequest(RequestModel):
    """Data writing request"""
    input_path: Optional[str] = Field(None, description="Input file path")
    output_path: str = Field(..., description="Output file path")
    data: Optional[Dict[str, Any]] = Field(None, description="Data to write (alternative to input_path)")
    format: Literal["netcdf4", "zarr", "csv"] = Field("netcdf4", description="Output format")
    compression: Optional[Literal["none", "zlib", "gzip", "zstd"]] = Field("zstd", description="Compression method")
    compression_level: int = Field(4, description="Compression level (1-9)")
    validate_cf_variable_metadata: bool = Field(
        True,
        description="Validate the bounded single-variable standard-name/unit subset",
    )
    
    @field_validator('compression_level')
    def validate_compression_level(cls, v: int):
        if not 1 <= v <= 9:
            raise ValueError("Compression level must be between 1 and 9")
        return v

    @field_validator('input_path', mode='before')
    def validate_input_path(cls, v: Optional[str]):
        if v is None:
            return None
        return validate_safe_local_path(v, must_exist=True)

    @field_validator('output_path', mode='before')
    def validate_output_path(cls, v: str):
        return validate_safe_local_path(v, must_exist=False)

    @model_validator(mode='after')
    def validate_input_source(self):
        if not self.input_path and self.data is None:
            raise ValueError("Either input_path or data must be provided")
        return self

class DataConvertRequest(RequestModel):
    """Data format conversion request"""
    input_path: str = Field(..., description="Input file path")
    output_path: str = Field(..., description="Output file path")
    target_format: Literal["netcdf4", "zarr", "csv"] = Field(..., description="Target format")
    variables: Optional[List[str]] = Field(None, description="Variables to convert")
    compression: Literal["none", "zlib", "gzip", "zstd"] = Field("zstd", description="Compression method")
    preserve_metadata: bool = Field(True, description="Preserve original metadata")

    @field_validator('input_path', mode='before')
    def validate_input_path(cls, v: str):
        return validate_safe_local_path(v, must_exist=True)

    @field_validator('output_path', mode='before')
    def validate_output_path(cls, v: str):
        return validate_safe_local_path(v, must_exist=False)

# Compute Models

class ComputeHealthRequest(RequestModel):
    """Compute health check request"""
    detailed: bool = Field(False, description="Include detailed diagnostics")
    include_benchmark: bool = Field(False, description="Include basic performance benchmark")
    check_memory: bool = Field(True, description="Check memory usage")

class ComputeBenchmarkRequest(RequestModel):
    """Compute benchmark request"""
    target: Literal["photosynthesis", "transport", "decomposition"] = Field(..., description="Benchmark target")
    device: Literal["auto", "cpu", "cuda"] = Field("auto", description="Target device")
    batch_size: int = Field(64, description="Batch size for benchmark")
    n_points: int = Field(1000, description="Number of computation points")
    test_jit: bool = Field(True, description="Test JIT compilation")
    test_amp: bool = Field(False, description="Test automatic mixed precision")
    n_runs: int = Field(10, description="Number of benchmark runs")
    
    @field_validator('batch_size', 'n_points', 'n_runs')
    def validate_positive(cls, v: int):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

# Optimization Models

class ParameterBound(RequestModel):
    """Parameter bounds for optimization"""
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    units: Optional[str] = Field(None, description="Parameter units")
    description: Optional[str] = Field(None, description="Parameter description")

    @model_validator(mode='after')
    def check_bounds(self):
        if self.max < self.min:
            raise ValueError("Maximum must be greater than minimum")
        return self

class CalibrationRequest(RequestModel):
    """Parameter calibration request"""
    parameters: List[str] = Field(..., description="Parameter names to calibrate")
    parameter_bounds: Dict[str, ParameterBound] = Field(..., description="Parameter bounds")
    target_data: Optional[Dict[str, Any]] = Field(None, description="Target data for calibration")
    method: str = Field(..., description="Optimization method")
    max_iterations: int = Field(..., description="Maximum iterations")
    tolerance: float = Field(..., description="Convergence tolerance")
    parallel: bool = Field(False, description="Use parallel evaluation")
    
    @field_validator('max_iterations')
    def validate_max_iterations(cls, v: int):
        if v <= 0:
            raise ValueError("Maximum iterations must be positive")
        return v

class SensitivityRequest(RequestModel):
    """Sensitivity analysis request"""
    method: Literal["morris", "sobol"] = Field(..., description="Sensitivity analysis method")
    parameters: Dict[str, ParameterBound] = Field(..., description="Parameter ranges")
    n_samples: Optional[int] = Field(None, description="Number of samples (required for method 'sobol')")
    n_trajectories: Optional[int] = Field(None, description="Number of trajectories (required for method 'morris')")
    n_levels: Optional[int] = Field(None, description="Number of grid levels (required for method 'morris')")

    @field_validator('n_samples', 'n_trajectories', 'n_levels')
    def validate_sample_counts(cls, v: Optional[int]):
        if v is not None and v <= 0:
            raise ValueError("Sample counts must be positive")
        return v

    @model_validator(mode='after')
    def validate_method_parameters(self):
        # Each method states its own design size; nothing is filled in.
        required = {
            "morris": ("n_trajectories", "n_levels"),
            "sobol": ("n_samples",),
        }[self.method]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"Sensitivity method '{self.method}' requires {', '.join(missing)}"
            )
        return self

class UncertaintyRequest(RequestModel):
    """Uncertainty quantification request"""
    method: Literal["ensemble", "bootstrap"] = Field(..., description="UQ method")
    uncertain_parameters: Dict[str, Dict[str, Any]] = Field(..., description="Uncertain parameter definitions")
    n_samples: int = Field(..., description="Number of samples")
    sampling_strategy: Literal["random", "lhs", "sobol"] = Field(..., description="Sampling strategy")
    confidence_levels: List[float] = Field(..., description="Confidence levels for intervals")
    parallel: bool = Field(False, description="Use parallel evaluation")
    
    @field_validator('confidence_levels')
    def validate_confidence_levels(cls, v: List[float]):
        for level in v:
            if not 0 < level < 1:
                raise ValueError("Confidence levels must be between 0 and 1")
        return v

# Stochastic Models

class WeatherGenerationRequest(RequestModel):
    """Weather generation request"""
    duration_days: int = Field(..., description="Duration in days")
    n_realizations: int = Field(..., description="Number of realizations")
    location: Optional[Dict[str, float]] = Field(None, description="Location coordinates")
    seed: Optional[int] = Field(None, description="Random seed")
    variables: List[str] = Field(..., description="Variables to generate")
    output_format: Literal["netcdf4", "zarr", "csv"] = Field("netcdf4", description="Output format")

class SDERequest(RequestModel):
    """Bound continuous-SDE execution request on the simulator time basis.

    Drift and diffusion coefficients are owned by the ParameterManager
    database, so the request carries no per-run parameter overrides; unknown
    fields are rejected.
    """


    process: str = Field(
        ..., description="Bound continuous-SDE process name (see GET /stochastic/)"
    )
    initial_value: float = Field(
        ..., description="Initial state shared by all realizations, in the SCM state unit"
    )
    dt_seconds: float = Field(..., gt=0.0, description="Fixed simulator step [s]")
    n_steps: int = Field(..., ge=1, description="Number of fixed steps")
    n_realizations: int = Field(
        ..., ge=1, description="Batched realizations advanced by one Functor substream"
    )
    seed: int = Field(..., ge=0, description="Functor-owned Wiener substream seed")


class MarkedJumpRequest(RequestModel):
    """Bound marked-jump execution request on the simulator time basis.

    Counting and mark states start at zero; propensities and mark laws are
    owned by the bound Functor and the ParameterManager database, so the
    request carries no reaction declarations or parameter overrides.
    """


    process: str = Field(
        ..., description="Bound marked-jump process name (see GET /stochastic/)"
    )
    dt_seconds: float = Field(..., gt=0.0, description="Fixed simulator step [s]")
    n_steps: int = Field(..., ge=1, description="Number of fixed steps")
    n_realizations: int = Field(
        ..., ge=1, description="Batched realizations advanced by one Functor substream"
    )
    seed: int = Field(..., ge=0, description="Functor-owned event substream seed")

# Configuration Models

class ConfigValidationRequest(RequestModel):
    """Configuration validation request"""
    config: Dict[str, Any] = Field(..., description="Configuration to validate")
    strict: bool = Field(False, description="Strict validation mode")
    check_dependencies: bool = Field(True, description="Check dependency availability")

class ConfigValidationResult(BaseModel):
    """Configuration validation result"""
    is_valid: bool = Field(..., description="Validation result")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    effective_config: Dict[str, Any] = Field(..., description="Effective configuration after defaults")

class TaskStatus(BaseModel):
    """Asynchronous task status"""
    task_id: str = Field(..., description="Task identifier")
    status: Literal["pending", "running", "completed", "failed"] = Field(..., description="Task status")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    message: Optional[str] = Field(None, description="Status message")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result (when completed)")
    error: Optional[ErrorDetail] = Field(None, description="Error details (when failed)")
    created_at: datetime = Field(..., description="Task creation time")
    updated_at: datetime = Field(..., description="Last update time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

# Create response model factories
def create_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None,
    paths: Optional[Dict[str, str]] = None
) -> APIResponse:
    """Create a success response"""
    return APIResponse(
        status="success",
        message=message,
        data=data,
        metadata={
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        },
        paths=paths
    )

def create_error_response(
    message: str,
    error_type: str = "ProcessingError",
    details: Optional[Dict[str, Any]] = None,
    suggestions: Optional[List[str]] = None
) -> APIResponse:
    """Create an error response"""
    return APIResponse(
        status="error",
        message=message,
        data={
            "error": ErrorDetail(
                type=error_type,
                message=message,
                details=details,
                suggestions=suggestions
            )
        },
        metadata={
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    )
