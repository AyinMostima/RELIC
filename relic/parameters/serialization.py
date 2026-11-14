"""
Parameter Serialization Component for RELIC

This module provides serialization and deserialization capabilities for
parameter management, supporting multiple formats (YAML, JSON, TOML) with
standardized schemas and format conversion.

It serves as the data interchange layer for the unified parameter management
system, enabling parameter persistence, configuration loading, and export.

Key Features:
    - Multi-format support (YAML, JSON, TOML)
    - Standardized parameter schemas
    - Format conversion and migration
    - Metadata preservation
    - Tensor serialization support

Design Patterns:
    - Strategy Pattern for format handlers
    - Adapter Pattern for format conversion
    - Factory Pattern for loader creation

References:
    - YAML 1.2 Specification
    - JSON Schema Draft 2020-12
    - TOML v1.0.0 Specification
"""

import json
import importlib
import importlib.util
from typing import Dict, Any, Optional, Union
from pathlib import Path
from enum import Enum
from datetime import datetime
import logging

if importlib.util.find_spec("yaml") is not None:
    yaml = importlib.import_module("yaml")
    YAML_AVAILABLE = True
else:
    yaml = None  # type: ignore[assignment]
    YAML_AVAILABLE = False

if importlib.util.find_spec("toml") is not None:
    toml = importlib.import_module("toml")
    TOML_AVAILABLE = True
else:
    toml = None  # type: ignore[assignment]
    TOML_AVAILABLE = False

import torch
from torch import Tensor

from .spec import ParameterSpec

logger = logging.getLogger(__name__)

# Type aliases
ParameterDict = Dict[str, Any]
SerializationResult = Union[str, bytes, Dict[str, Any]]


class SerializationFormat(Enum):
    """Supported serialization formats."""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    PYTHON = "python"  # Native Python dict


class ParameterSchema:
    """
    Standard parameter serialization schema.
    
    Defines the canonical structure for parameter data interchange,
    ensuring consistency across different formats and versions.
    
    Schema Structure:
        {
            "version": "1.0.0",
            "timestamp": "2024-01-15T10:30:00",
            "metadata": {...},
            "parameters": {
                "key": {
                    "value": ...,
                    "units": "...",
                    "bounds": [...],
                    "dtype": "...",
                    "category": "...",
                    ...
                }
            }
        }
    """
    
    VERSION = "1.0.0"
    
    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> bool:
        """
        Validate data against parameter schema.
        
        Parameters:
            data : Dict[str, Any]
                Data to validate
                
        Returns:
            bool: True if valid schema
        """
        required_keys = {"parameters"}
        if not all(key in data for key in required_keys):
            return False
        
        # Check parameter structure
        if not isinstance(data["parameters"], dict):
            return False
        
        for key, param_data in data["parameters"].items():
            if not isinstance(param_data, dict):
                return False
            if "value" not in param_data:
                return False
        
        return True
    
    @staticmethod
    def migrate_schema(data: Dict[str, Any], 
                       from_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Migrate data from old schema versions.
        
        Parameters:
            data : Dict[str, Any]
                Data in old schema
            from_version : str
                Source schema version
                
        Returns:
            Dict[str, Any]: Data in current schema
        """
        if not isinstance(data, dict):
            raise ValueError("Unsupported parameter schema: expected mapping data")

        migrated = dict(data)

        # Canonical schema path only.
        if "parameters" in migrated:
            if not isinstance(migrated["parameters"], dict):
                raise ValueError("Unsupported parameter schema: 'parameters' must be a map")
            if "version" not in migrated:
                migrated["version"] = ParameterSchema.VERSION
            return migrated
        raise ValueError(
            "Unsupported parameter schema: missing 'parameters' map. "
            "Legacy flat parameter maps are not supported."
        )


def serialize_tensor(tensor: Tensor) -> Dict[str, Any]:
    """
    Serialize PyTorch tensor to JSON-consistent format.
    
    Parameters:
        tensor : Tensor
            Tensor to serialize
            
    Returns:
        Dict[str, Any]: Serialized representation
        
    Examples:
        >>> t = torch.tensor([1.0, 2.0])
        >>> serialize_tensor(t)
        {'type': 'tensor', 'data': [1.0, 2.0], 'dtype': 'float32', 'shape': [2]}
    """
    return {
        "type": "tensor",
        "data": tensor.tolist(),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": list(tensor.shape),
        "requires_grad": tensor.requires_grad
    }


def deserialize_tensor(data: Dict[str, Any]) -> Tensor:
    """
    Deserialize tensor from JSON-consistent format.
    
    Parameters:
        data : Dict[str, Any]
            Serialized tensor data
            
    Returns:
        Tensor: Reconstructed tensor
        
    Examples:
        >>> data = {'type': 'tensor', 'data': [1.0, 2.0], 'dtype': 'float32',
        ...         'shape': [2], 'requires_grad': False}
        >>> deserialize_tensor(data)
        tensor([1., 2.])
    """
    dtype_name = data.get("dtype")
    if not isinstance(dtype_name, str):
        raise TypeError("Tensor dtype must be an explicit canonical name")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype) or str(dtype) != f"torch.{dtype_name}":
        raise ValueError(f"Unsupported Tensor dtype: {dtype_name}")
    shape = data.get("shape")
    if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Tensor shape must contain nonnegative integer dimensions")
    requires_grad = data.get("requires_grad")
    if type(requires_grad) is not bool:
        raise TypeError("Tensor requires_grad must be an explicit boolean")
    tensor = torch.tensor(data["data"], dtype=dtype)
    if list(tensor.shape) != shape:
        # Nested empty lists lose dimensions after the first zero-length axis.
        if (
            tensor.numel() == 0
            and len(shape) > tensor.ndim
            and shape[:tensor.ndim] == list(tensor.shape)
        ):
            tensor = tensor.reshape(shape)
        else:
            raise ValueError("Tensor data shape does not match its declared shape")
    if requires_grad:
        tensor.requires_grad_(True)
    
    return tensor


def serialize_parameter_spec(spec: 'ParameterSpec') -> Dict[str, Any]:
    """
    Serialize ParameterSpec to dictionary.
    
    Parameters:
        spec : ParameterSpec
            Parameter specification
            
    Returns:
        Dict[str, Any]: Serialized specification
    """
    return spec.to_dict()


def deserialize_parameter_spec(data: Dict[str, Any]) -> 'ParameterSpec':
    """
    Deserialize ParameterSpec from dictionary.
    
    Parameters:
        data : Dict[str, Any]
            Serialized specification
            
    Returns:
        ParameterSpec: Reconstructed specification
    """
    return ParameterSpec.from_dict(data)


# Format-specific loaders and savers

def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load parameters from YAML file.
    
    Parameters:
        path : Path
            Path to YAML file
            
    Returns:
        Dict[str, Any]: Loaded parameters
        
    Raises:
        ImportError: If PyYAML not available
        FileNotFoundError: If file doesn't exist
    """
    if not YAML_AVAILABLE or yaml is None:
        raise ImportError("PyYAML required for YAML format")
    
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = yaml.safe_load(f)
    return ParameterSchema.migrate_schema(data)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """
    Save parameters to YAML file.
    
    Parameters:
        data : Dict[str, Any]
            Parameters to save
        path : Path
            Output file path
            
    Raises:
        ImportError: If PyYAML not available
    """
    if not YAML_AVAILABLE or yaml is None:
        raise ImportError("PyYAML required for YAML format")
    
    payload = dict(data)
    if not ParameterSchema.validate_schema(payload):
        payload = ParameterSchema.migrate_schema(payload)
    if "timestamp" not in payload:
        payload["timestamp"] = datetime.now().isoformat()
    if "version" not in payload:
        payload["version"] = ParameterSchema.VERSION
    
    # Create parent directories if they don't exist
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load parameters from JSON file.
    
    Parameters:
        path : Path
            Path to JSON file
            
    Returns:
        Dict[str, Any]: Loaded parameters
    """
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return ParameterSchema.migrate_schema(data)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save parameters to JSON file.
    
    Parameters:
        data : Dict[str, Any]
            Parameters to save
        path : Path
            Output file path
    """
    payload = dict(data)
    if not ParameterSchema.validate_schema(payload):
        payload = ParameterSchema.migrate_schema(payload)
    if "timestamp" not in payload:
        payload["timestamp"] = datetime.now().isoformat()
    if "version" not in payload:
        payload["version"] = ParameterSchema.VERSION
    
    # Custom JSON encoder for special types
    class ParameterEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, Tensor):
                return serialize_tensor(o)
            if isinstance(o, set):
                return list(o)
            if isinstance(o, Enum):
                return o.value
            return super().default(o)
    
    # Create parent directories if they don't exist
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, cls=ParameterEncoder)


def load_toml(path: Path) -> Dict[str, Any]:
    """
    Load parameters from TOML file.
    
    Parameters:
        path : Path
            Path to TOML file
            
    Returns:
        Dict[str, Any]: Loaded parameters
        
    Raises:
        ImportError: If toml not available
    """
    if not TOML_AVAILABLE or toml is None:
        raise ImportError("toml required for TOML format")
    
    with open(path, 'r', encoding='utf-8-sig') as f:
        data = toml.load(f)
    return ParameterSchema.migrate_schema(data)


def save_toml(data: Dict[str, Any], path: Path) -> None:
    """
    Save parameters to TOML file.
    
    Parameters:
        data : Dict[str, Any]
            Parameters to save
        path : Path
            Output file path
            
    Raises:
        ImportError: If toml not available
    """
    if not TOML_AVAILABLE or toml is None:
        raise ImportError("toml required for TOML format")
    
    payload = dict(data)
    if not ParameterSchema.validate_schema(payload):
        payload = ParameterSchema.migrate_schema(payload)
    if "timestamp" not in payload:
        payload["timestamp"] = datetime.now().isoformat()
    if "version" not in payload:
        payload["version"] = ParameterSchema.VERSION
    
    # TOML has limitations, need to convert some types
    toml_data = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            toml_data[key] = {}
            for k, v in value.items():
                if isinstance(v, Tensor):
                    toml_data[key][k] = serialize_tensor(v)
                elif isinstance(v, set):
                    toml_data[key][k] = list(v)
                else:
                    toml_data[key][k] = v
        else:
            toml_data[key] = value
    
    # Create parent directories if they don't exist
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        toml.dump(toml_data, f)


# Main export function

def export(params: Union[Dict[str, Any], Dict[str, 'ParameterSpec']],
          format: str = "yaml",
          output_path: Optional[Path] = None) -> Optional[str]:
    """
    Export parameters to specified format.
    
    Parameters:
        params : Union[Dict[str, Any], Dict[str, ParameterSpec]]
            Parameters to export (values or specs)
        format : str
            Output format (yaml, json, toml, python)
        output_path : Optional[Path]
            Output file path (None for string return)
            
    Returns:
        Optional[str]: Serialized string if no output_path
        
    Examples:
        >>> export(params, format="yaml", output_path=Path("params.yaml"))
        >>> json_str = export(params, format="json")
    """
    # Convert ParameterSpecs to dicts if needed
    if params and isinstance(next(iter(params.values())), ParameterSpec):
        serialized = {
            key: serialize_parameter_spec(spec)
            for key, spec in params.items()
        }
    else:
        serialized = params
    
    # Build standard schema
    data = {
        "version": ParameterSchema.VERSION,
        "timestamp": datetime.now().isoformat(),
        "parameters": serialized
    }
    
    # Select format handler
    format_enum = SerializationFormat(format.lower())
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_enum == SerializationFormat.YAML:
            save_yaml(data, output_path)
        elif format_enum == SerializationFormat.JSON:
            save_json(data, output_path)
        elif format_enum == SerializationFormat.TOML:
            save_toml(data, output_path)
        else:
            raise ValueError(f"Unsupported format for file output: {format}")
        
        logger.info(f"Exported parameters to {output_path}")
        return None
    
    else:
        # Return as string
        if format_enum == SerializationFormat.YAML:
            if not YAML_AVAILABLE or yaml is None:
                raise ImportError("PyYAML required for YAML format")
            return yaml.dump(data, default_flow_style=False)
        
        elif format_enum == SerializationFormat.JSON:
            return json.dumps(data, indent=2)
        
        elif format_enum == SerializationFormat.TOML:
            if not TOML_AVAILABLE or toml is None:
                raise ImportError("toml required for TOML format")
            return toml.dumps(data)
        
        elif format_enum == SerializationFormat.PYTHON:
            return repr(data)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


def load(path: Path, format: Optional[str] = None) -> Dict[str, Any]:
    """
    Load parameters from file with auto-detection.
    
    Parameters:
        path : Path
            Input file path
        format : Optional[str]
            Format override (auto-detect from extension if None)
            
    Returns:
        Dict[str, Any]: Loaded parameters
        
    Examples:
        >>> params = load(Path("config.yaml"))
        >>> params = load(Path("params.json"), format="json")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    
    # Auto-detect format from extension
    if format is None:
        suffix = path.suffix.lower()
        format_map = {
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml"
        }
        format = format_map.get(suffix)
        
        if format is None:
            raise ValueError(f"Cannot determine format from extension: {suffix}")
    
    format_enum = SerializationFormat(format.lower())
    
    if format_enum == SerializationFormat.YAML:
        return load_yaml(path)
    elif format_enum == SerializationFormat.JSON:
        return load_json(path)
    elif format_enum == SerializationFormat.TOML:
        return load_toml(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def convert_format(input_path: Path,
                  output_path: Path,
                  input_format: Optional[str] = None,
                  output_format: Optional[str] = None) -> None:
    """
    Convert parameter file between formats.
    
    Parameters:
        input_path : Path
            Source file path
        output_path : Path
            Destination file path
        input_format : Optional[str]
            Source format (auto-detect if None)
        output_format : Optional[str]
            Target format (auto-detect if None)
            
    Examples:
        >>> convert_format(
        ...     Path("params.yaml"),
        ...     Path("params.json")
        ... )
    """
    # Load from input
    data = load(input_path, format=input_format)
    
    # Determine output format
    if output_format is None:
        suffix = output_path.suffix.lower()
        format_map = {
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml"
        }
        output_format = format_map.get(suffix)
        
        if output_format is None:
            raise ValueError(f"Cannot determine output format from extension: {suffix}")
    
    # Export to output using strict canonical schema.
    output_enum = SerializationFormat(output_format.lower())
    if output_enum == SerializationFormat.YAML:
        save_yaml(data, output_path)
    elif output_enum == SerializationFormat.JSON:
        save_json(data, output_path)
    elif output_enum == SerializationFormat.TOML:
        save_toml(data, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    logger.info(f"Converted {input_path} to {output_path}")


# Export main functions and constants
__all__ = [
    'export',
    'load',
    'convert_format',
    'serialize_parameter_spec',
    'deserialize_parameter_spec',
    'serialize_tensor',
    'deserialize_tensor',
    'load_yaml',
    'save_yaml',
    'load_json',
    'save_json',
    'load_toml',
    'save_toml',
    'YAML_AVAILABLE',
    'TOML_AVAILABLE',
    'SerializationFormat',
    'ParameterSchema'
]
