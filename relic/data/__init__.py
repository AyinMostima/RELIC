"""
RELIC Data Processing Layer - Unified I/O and Standards

This package provides comprehensive data processing capabilities for the
Recomposable Ecosystem Library for Integrated Coupling (RELIC), including:
- High-level data I/O operations (NetCDF, Zarr, CSV, HDF5)
- Bounded CF standard-name and unit metadata validation
- CMIP6 standards support
- Data quality control and validation
- Data transformations and processing
- Experimental streaming capabilities (optional)

Architecture Components:
    - io/: High-level data input/output with format detection
    - standards/: CF-1.8 conventions and CMIP6 standards
    - quality/: Data quality control and validation (placeholder)
    - transformers/: Data transformation utilities (placeholder)
    - streaming/: Real-time data streaming (experimental, requires opt-in)

Design Principles:
    1. CF scope honesty: dataset-level conformance is not implemented
    2. Format Agnostic: Automatic format detection and conversion
    3. PyTorch Integration: Seamless tensor conversion for model input
    4. Performance Optimized: Lazy loading, chunking, compression
    5. Extensible: Plugin architecture for custom formats

Data Standards:
    - CF Conventions 1.8: http://cfconventions.org/
    - CMIP6 Data Request: https://www.earthsystemcog.org/projects/wip/CMIP6DataRequest
    - UDUNITS-2: Unit conversion and validation
    - NetCDF Climate and Forecast Metadata Conventions

Key Features:
    - Official single-variable CF standard-name/unit subset validation
    - Unit conversion and validation with UnitConverter
    - Time coordinate processing and calendar handling
    - Spatial grid operations and regridding support
    - Compression and optimization (zstd, lz4, blosc)
    - Health checking and diagnostics
    - PyTorch tensor integration

References:
    - Eaton, B., et al. (2020). NetCDF Climate and Forecast (CF) Metadata Conventions
    - CF Conventions 1.8: http://cfconventions.org/cf-conventions/cf-conventions.html
    - xarray documentation: http://xarray.pydata.org/en/stable/
    - PyTorch data loading: https://pytorch.org/docs/stable/data.html

Version: 1.0.0 (Phase 1 - Data Infrastructure)
"""

# Standard library imports
import importlib
import os
import logging

from relic.utils.imports import module_available

_LAZY_MODULES = {
    "io": "relic.data.io",
    "standards": "relic.data.standards",
    "unit_converter": "relic.data.standards.unit_converter",
}

_IO_PUBLIC_SYMBOLS = (
    "DataReader",
    "DataWriter",
    "DataIO",
    "ReadConfig",
    "WriteConfig",
    "ReadResult",
    "WriteResult",
    "IOPerformanceMetrics",
    "read",
    "write",
    "read_dataset",
    "read_tensors",
    "write_dataset",
    "write_tensors",
    "convert_format",
    "get_file_info",
    "default_io",
    "IOHealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    "SystemHealthReport",
    "io_healthcheck",
    "IOOperation",
    "DataFormat",
    "CompressionType",
)
_STANDARDS_PUBLIC_SYMBOLS = (
    "CFConventions",
    "CFAttributeSynchronizer",
)

_lazy_enabled = os.environ.get("RELIC_LAZY_IMPORT", "1").lower() in {"1", "true", "yes"}
logger = logging.getLogger(__name__)


# Why an optional relic.data submodule reads as absent (None): the cause of each absence.
OPTIONAL_IMPORT_ERRORS: dict = {}


def _import_optional_data_module(submodule: str):
    """Import an optional relic.data submodule.

    Two absences are declared: a placeholder module without its experimental opt-in
    (ExperimentalFeatureError) and a missing optional third-party dependency (ImportError); both
    return None and keep their cause in OPTIONAL_IMPORT_ERRORS. Any other import error of RELIC's
    own module propagates.
    """
    from ._placeholder import ExperimentalFeatureError

    full_module = f"relic.data.{submodule}"
    if not module_available(full_module):
        return None
    try:
        return importlib.import_module(f".{submodule}", __name__)
    except ExperimentalFeatureError as exc:
        OPTIONAL_IMPORT_ERRORS[submodule] = exc
        logger.info("Optional module %s disabled: %s", full_module, exc)
        return None
    except ImportError as exc:
        OPTIONAL_IMPORT_ERRORS[submodule] = exc
        logger.warning("Optional module %s unavailable: %s", full_module, exc)
        return None


def __getattr__(name: str):
    if name == "io":
        module = importlib.import_module(_LAZY_MODULES["io"])
        globals()["io"] = module
        return module
    if name in _IO_PUBLIC_SYMBOLS:
        module = importlib.import_module(_LAZY_MODULES["io"])
        globals()["io"] = module
        for symbol in _IO_PUBLIC_SYMBOLS:
            globals()[symbol] = getattr(module, symbol, None)
        return globals()[name]
    if name == "standards":
        module = importlib.import_module(_LAZY_MODULES["standards"])
        globals()["standards"] = module
        return module
    if name in _STANDARDS_PUBLIC_SYMBOLS:
        module = importlib.import_module(_LAZY_MODULES["standards"])
        globals()["standards"] = module
        for symbol in _STANDARDS_PUBLIC_SYMBOLS:
            globals()[symbol] = getattr(module, symbol, None)
        return globals()[name]
    if name in {"UnitConverter", "unit_converter"}:
        module = importlib.import_module(_LAZY_MODULES["unit_converter"])
        globals()["UnitConverter"] = getattr(module, "UnitConverter", None)
        globals()["unit_converter"] = getattr(module, "unit_converter", None)
        return globals()[name]
    raise AttributeError(f"module 'relic.data' has no attribute '{name}'")


def _get_symbol(name: str):
    """Resolve an exported symbol, triggering lazy loading when enabled."""
    value = globals().get(name, None)
    if value is not None:
        return value
    if _lazy_enabled:
        try:
            return __getattr__(name)
        except (ImportError, AttributeError):
            return None
    return None

# Import quality/transformers components (placeholder)
quality = _import_optional_data_module("quality")
transformers = _import_optional_data_module("transformers")

# Import streaming components (experimental - requires opt-in)
streaming = _import_optional_data_module("streaming")
if streaming is not None:
    STREAMING_ENABLED = bool(getattr(streaming, "STREAMING_ENABLED", False))
    StreamManager = getattr(streaming, "StreamManager", None)
    StreamConfig = getattr(streaming, "StreamConfig", None)
    StreamWindow = getattr(streaming, "StreamWindow", None)
    run_streaming = getattr(streaming, "run_streaming", None)
    streaming_healthcheck = getattr(streaming, "streaming_healthcheck", None)

    if not all(
        symbol is not None
        for symbol in (StreamManager, StreamConfig, StreamWindow, run_streaming, streaming_healthcheck)
    ):
        logger.warning("Optional module relic.data.streaming loaded but required symbols are incomplete")
else:
    STREAMING_ENABLED = False
    StreamManager = None
    StreamConfig = None
    StreamWindow = None
    run_streaming = None
    streaming_healthcheck = None

# Version and metadata
__version__ = "1.0.0"
__phase__ = "Phase 1 - Data Infrastructure"
__author__ = "Yuchong Long"
__license__ = "MIT"

# Helper functions for data operations
def check_cf_variable_metadata(metadata: dict):
    """
    Validate one variable's standard-name and unit metadata subset.

    Parameters:
        metadata : dict
            Metadata dictionary to validate

    Returns:
        CFVariableMetadataValidationResult with explicit bounded scope.
    """
    cf_conventions = _get_symbol("CFConventions")
    if cf_conventions is None:
        return False, ["CF conventions module not available"]
    return cf_conventions.validate_cf_compliance(metadata)

def standardize_variable_name(variable_name: str) -> str:
    """
    Resolve an exact official standard-name entry or alias.

    Parameters:
        variable_name : str
            Common variable name (e.g., 'temperature', 'gpp')

    Returns:
        str: CF-1.8 standard name
    """
    cf_conventions = _get_symbol("CFConventions")
    if cf_conventions is None:
        return variable_name
    return cf_conventions.get_standard_name(variable_name)

def get_standard_units(standard_name: str) -> str:
    """
    Get official canonical units for an exact standard name or alias.

    Parameters:
        standard_name : str
            CF-1.8 standard name

    Returns:
        str: CF-1.8 standard units
    """
    cf_conventions = _get_symbol("CFConventions")
    if cf_conventions is None:
        return ""
    return cf_conventions.get_standard_units(standard_name)

def validate_units(value, units: str) -> bool:
    """
    Validate only that a unit string is recognized by UDUNITS syntax.

    Parameters:
        value : Union[Tensor, float]
            Data value(s) to check
        units : str
            Units string to validate

    Returns:
        bool: True if units are valid
    """
    cf_conventions = _get_symbol("CFConventions")
    if cf_conventions is None:
        return False
    return cf_conventions.validate_units(value, units)

def enable_streaming():
    """
    Enable experimental streaming functionality.

    Sets environment variable and provides instructions for activation.
    Requires restart of Python session to take effect.
    """
    import os
    os.environ['RELIC_EXPERIMENTAL_STREAMING'] = '1'
    print("[INFO] Streaming functionality enabled.")
    print("[INFO] Please restart your Python session for changes to take effect.")
    print("[INFO] Required dependencies: pip install watchdog aiofiles zarr")

# Export all public components
__all__ = [
    # Subpackages
    'io',
    'standards',
    'quality',
    'transformers',
    'streaming',

    # I/O core classes
    'DataReader',
    'DataWriter',
    'DataIO',

    # I/O configuration
    'ReadConfig',
    'WriteConfig',

    # I/O results
    'ReadResult',
    'WriteResult',
    'IOPerformanceMetrics',

    # I/O convenience functions
    'read',
    'write',
    'read_dataset',
    'read_tensors',
    'write_dataset',
    'write_tensors',
    'convert_format',
    'get_file_info',
    'default_io',

    # I/O health checking
    'IOHealthChecker',
    'HealthStatus',
    'HealthCheckResult',
    'SystemHealthReport',
    'io_healthcheck',

    # I/O enums and types
    'IOOperation',
    'DataFormat',
    'CompressionType',

    # Standards
    'CFConventions',
    'CFAttributeSynchronizer',
    'UnitConverter',
    'unit_converter',

    # Streaming (experimental)
    'STREAMING_ENABLED',
    'StreamManager',
    'StreamConfig',
    'StreamWindow',
    'run_streaming',
    'streaming_healthcheck',

    # Helper functions
    'check_cf_variable_metadata',
    'standardize_variable_name',
    'get_standard_units',
    'validate_units',
    'enable_streaming',

    # Metadata
    '__version__',
    '__phase__',
    '__author__',
    '__license__'
]

# Informational messages
def print_data_capabilities():
    """Print available data processing capabilities."""
    io_module = _get_symbol("io")
    cf_conventions = _get_symbol("CFConventions")
    unit_converter_cls = _get_symbol("UnitConverter")
    print("RELIC Data Processing Capabilities:")
    print(f"  - I/O Operations: {'Available' if io_module is not None else 'Not Available'}")
    print(f"  - CF standard-name/unit subset: {'Available' if cf_conventions is not None else 'Not Available'}")
    print(f"  - Unit Conversion: {'Available' if unit_converter_cls is not None else 'Not Available'}")
    print(f"  - Quality Control: {'Available' if quality is not None else 'Placeholder'}")
    print(f"  - Transformers: {'Available' if transformers is not None else 'Placeholder'}")
    print(f"  - Streaming: {'Enabled' if STREAMING_ENABLED else 'Disabled (set RELIC_EXPERIMENTAL_STREAMING=1)'}")

    if io_module is not None:
        print("\n  Supported formats:")
        print("    - NetCDF4 (.nc)")
        print("    - Zarr (.zarr)")
        print("    - CSV (.csv)")
        print("    - HDF5 (.h5, .hdf5)")

    if cf_conventions is not None:
        print("\n  Standards metadata support:")
        print("    - single-variable CF standard-name/unit subset")
        print("    - UDUNITS-2 unit system")
        print("    - CMIP6 variable naming")

# Add to __all__
__all__.append('print_data_capabilities')
