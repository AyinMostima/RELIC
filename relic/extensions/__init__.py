"""
RELIC Extension Interface.

Non-invasive plugin system for adding custom Functors without modifying core
code in relic.core, relic.scientific, or relic.orchestration.
"""

from .discovery import ExtensionInfo, discover_extensions
from .hooks import (
    load_and_register_extensions,
    register_with_orchestration,
    unload_from_runtime,
)
from .loader import (
    LoadedFunctors,
    get_loaded_extensions,
    load_extensions,
    load_single_extension,
    unload_extension,
)
from .params import (
    load_extension_parameters,
    register_extension_parameter_source,
    save_extension_parameters,
)
from .protocol import ExtensionManifest, MANIFEST_FILENAME
from .validator import ValidationResult, validate_extension

__all__ = [
    "discover_extensions",
    "ExtensionInfo",
    "ExtensionManifest",
    "MANIFEST_FILENAME",
    "validate_extension",
    "ValidationResult",
    "load_extensions",
    "LoadedFunctors",
    "load_single_extension",
    "get_loaded_extensions",
    "unload_extension",
    "load_and_register_extensions",
    "register_with_orchestration",
    "unload_from_runtime",
    "load_extension_parameters",
    "save_extension_parameters",
    "register_extension_parameter_source",
]
