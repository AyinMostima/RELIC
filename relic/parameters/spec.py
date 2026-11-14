"""ParameterSpec definition for the RELIC parameter system.

This module isolates the ParameterSpec dataclass to keep the manager
implementation focused on orchestration rather than data modeling.

References:
    - CF-1.8 Convention for Climate and Forecast Metadata
    - FAIR data principles for parameter metadata
"""

from __future__ import annotations

import builtins
import logging
import math
from dataclasses import dataclass, field, fields
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple, Set
from collections.abc import Mapping, Sequence, Set as AbstractSet

import torch
from torch import Tensor

from .types import ParameterCategory, ParameterValue, MetadataDict, ValidationErrors
from .category_aliases import normalize_category
from relic.utils.tensor_ops import to_tensor as convert_to_tensor

logger = logging.getLogger(__name__)

_MISSING_PARAMETER_FIELD = object()
_REQUIRED_SCIENTIFIC_METADATA_FIELDS = (
    "key",
    "units",
    "dtype",
    "bounds",
    "category",
    "references",
    "provenance",
)

# dtype names whose values are not numbers (text, boolean switches, enumerations,
# lists, mappings and other structured payloads). Every other dtype, including an
# unrecognized one, is treated as numeric.
_NON_NUMERIC_DTYPES = frozenset({
    "bool",
    "boolean",
    "str",
    "string",
    "category",
    "enum",
    "list",
    "tuple",
    "dict",
    "mapping",
    "json",
    "object",
})

# dtype names of sequence-valued parameters, the ones that may declare one
# (minimum, maximum) pair per component (componentwise_bounds).
_SEQUENCE_DTYPES = frozenset({"list", "tuple"})


def _dtype_root(dtype: Any) -> str:
    """Normalize a declared dtype to its lower-case root name ('list[float]' -> 'list')."""

    text = "" if dtype is None else str(dtype).strip().lower()
    if text.startswith("<class '") and text.endswith("'>"):
        text = text[len("<class '"):-2]
    return text.split("[")[0]


def parameter_dtype_is_numeric(dtype: Any) -> bool:
    """Return whether a declared dtype is numeric (every dtype outside the non-numeric vocabulary)."""

    return _dtype_root(dtype) not in _NON_NUMERIC_DTYPES


def parameter_dtype_is_sequence(dtype: Any) -> bool:
    """Return whether a declared dtype is a list or tuple ('list[float]' counts as list)."""

    return _dtype_root(dtype) in _SEQUENCE_DTYPES


def _text_leaves(value: Any) -> List[Any]:
    """Return the text (str or bytes) scalars of a declared number, pair or nested list."""

    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _text_leaves(item)]
    return []


def parameter_bounds_error(bounds: Any, dtype: Any) -> Optional[str]:
    """Return why ``bounds`` is not a valid declaration for ``dtype``; None when it is valid.

    This is the one rule for the bounds field, read by ParameterSpec,
    has_complete_parameter_metadata, the structural parameter audit and the
    parameter YAML schema:

    - ``None`` states that no interval applies. It is valid only for a
      non-numeric dtype (text, boolean switch, enumeration, list, mapping).
    - Otherwise bounds is a ``(lower, upper)`` pair. For a numeric dtype both
      endpoints are declared; an unbounded side is written ``-inf`` or ``inf``,
      never ``None``.
    - The endpoints of a numeric dtype are numbers, never text. PyYAML follows
      YAML 1.1 and loads ``1.0e7`` or ``1e-6`` as text (``1.0e+7`` and ``1.0e-6``
      are floats); a text endpoint is rejected, not converted.
    """

    numeric = parameter_dtype_is_numeric(dtype)
    if bounds is None:
        if numeric:
            return (
                f"bounds is None but dtype {dtype!r} is numeric; a numeric parameter "
                "declares an interval, written (-inf, inf) when it is unbounded"
            )
        return None
    if (
        isinstance(bounds, (str, bytes))
        or not isinstance(bounds, Sequence)
        or len(bounds) != 2
    ):
        return f"bounds {bounds!r} is not a (lower, upper) pair"
    if numeric and any(endpoint is None for endpoint in bounds):
        return (
            f"numeric bounds {tuple(bounds)!r} leave a side undeclared; write -inf or inf "
            "for an unbounded side"
        )
    text = _text_leaves(tuple(bounds)) if numeric else []
    if text:
        return (
            f"numeric bounds {tuple(bounds)!r} state {text!r} as text, not numbers; write "
            "YAML numbers (1.0e+7, 1.0e-6, .inf)"
        )
    return None


def _is_real_number(value: Any) -> bool:
    """Return whether value is a real number; a boolean is a switch, not a number."""

    if isinstance(value, bool):
        return False
    # int and float are tested first; the abstract Real test is several times slower.
    return isinstance(value, (int, float)) or isinstance(value, Real)


def componentwise_bounds(bounds: Any) -> Optional[Tuple[Tuple[float, float], ...]]:
    """Return the per-component (minimum, maximum) pairs of a componentwise declaration.

    A list-valued parameter may declare one ``[minimum, maximum]`` pair of real
    numbers per component instead of a single interval, for example
    ``[[223.15, 273.15], [323.15, 373.15]]`` for a two-component ``[low, high]``
    value (ParameterSpec admits exactly two entries, hence two components). Every
    other declaration returns None: no bounds, a flat ``(lower, upper)`` interval,
    a pair of labels or text ranges, and nested entries that are not pairs of numbers.
    """

    if not isinstance(bounds, (list, tuple)) or not bounds:
        return None
    pairs = []
    for pair in bounds:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(_is_real_number(endpoint) for endpoint in pair)
        ):
            return None
        pairs.append((float(pair[0]), float(pair[1])))
    return tuple(pairs)


def componentwise_bounds_errors(
    value: Any,
    component_bounds: Sequence[Tuple[float, float]],
) -> List[str]:
    """Return why ``value`` violates one (minimum, maximum) pair per component; empty when valid.

    This is the one componentwise rule, read by ParameterSpec.validate_value
    (validated set_value, database loading, parameter audits) and by
    FunctorBase._validate_parameter_value (Functor reads and configuration
    candidates). A message names the offending component and carries no
    parameter name; the caller prefixes it.
    """

    # References: RELIC parameter database convention for list-valued
    #   parameters (one [minimum, maximum] pair per component, for example
    #   hydrosphere.numerical.temperature_bounds); RELIC fail-fast input policy.
    # Mathematical expression: valid iff len(v) = n and, for every i in
    #   0..n-1, v_i is a finite real number with lo_i <= v_i <= hi_i.
    # Parameter meanings: v the candidate value; v_i its component i;
    #   (lo_i, hi_i) the declared pair of component i; n the number of pairs.
    # Units: v_i, lo_i and hi_i share the declared units of the parameter.
    # Domain/constraints: v is a list or tuple; closed intervals; a boolean,
    #   NaN and +/-inf are no components; a pair that cannot order a component
    #   (NaN endpoint) rejects it; no component is clamped.
    # Sign convention: none (a domain check, not a flux).
    count = len(component_bounds)
    if not isinstance(value, (list, tuple)):
        return [
            f"takes a list or tuple of {count} numbers (one per declared bounds pair), "
            f"got {type(value).__name__}"
        ]
    if len(value) != count:
        return [f"has length {len(value)} but its bounds declare {count} components"]
    errors: List[str] = []
    for index, (component, (lower, upper)) in enumerate(zip(value, component_bounds)):
        if not _is_real_number(component):
            errors.append(
                f"component {index} must be a real number, "
                f"got {type(component).__name__} {component!r}"
            )
            continue
        number = float(component)
        if not math.isfinite(number):
            errors.append(f"component {index} is not finite: {number}")
        elif number < lower:
            errors.append(f"component {index} below minimum: {number} < {lower}")
        elif number > upper:
            errors.append(f"component {index} above maximum: {number} > {upper}")
        elif not lower <= number <= upper:
            errors.append(
                f"component {index} = {number} cannot be ordered against its "
                f"declared bounds [{lower}, {upper}]"
            )
    return errors


def _parameter_field(value: Any, key: str) -> Any:
    """Read one field from either ParameterSpec or its serialized mapping."""

    if isinstance(value, Mapping):
        return value.get(key, _MISSING_PARAMETER_FIELD)
    return getattr(value, key, _MISSING_PARAMETER_FIELD)


def _parameter_field_is_present(value: Any) -> bool:
    """Return whether one metadata field carries explicit evidence."""

    if value is _MISSING_PARAMETER_FIELD or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence, AbstractSet)):
        return len(value) > 0
    return True


def _metadata_field_is_declared(value: Any, field_name: str) -> bool:
    """Return whether one scientific metadata field is declared.

    bounds follows parameter_bounds_error: the field must exist, and None
    counts as declared only for a non-numeric dtype.
    """

    field_value = _parameter_field(value, field_name)
    if field_name == "bounds":
        if field_value is _MISSING_PARAMETER_FIELD:
            return False
        dtype = _parameter_field(value, "dtype")
        return parameter_bounds_error(
            field_value,
            None if dtype is _MISSING_PARAMETER_FIELD else dtype,
        ) is None
    return _parameter_field_is_present(field_value)


def has_complete_parameter_metadata(
    value: Any,
    *,
    require_value: bool = False,
) -> bool:
    """Return whether a parameter carries the canonical scientific metadata.

    This predicate is the shared evidence boundary for callers that must prove
    ParameterManager binding. It accepts both ``ParameterSpec`` and serialized
    mappings so certificate validation and runtime validation use one owner.
    ``require_value`` additionally requires a resolved runtime value.
    """

    if value is None:
        return False
    required_fields = _REQUIRED_SCIENTIFIC_METADATA_FIELDS
    if require_value:
        required_fields = (*required_fields, "value")
    return all(
        _metadata_field_is_declared(value, field_name)
        for field_name in required_fields
    )


@dataclass
class ParameterSpec(Mapping[str, Any]):
    """
    Comprehensive parameter specification with scientific validation.
    
    Contains all metadata and constraints needed to properly manage
    a scientific parameter including units, bounds, references, and
    validation rules following FAIR data principles.
    
    Attributes:
        key : str
            Unique dot-notation parameter key
        value : ParameterValue
            Current parameter value  
        units : str
            Physical units following CF-1.8 conventions (required)
        dtype : str
            Data type ('float32', 'float64', 'int32', etc.) (required)
        bounds : Optional[Tuple[float, float]]
            Physical bounds (min, max) (required). A numeric dtype declares an
            interval and writes an unbounded side as -inf or inf; None states
            that no interval applies and is valid only for a non-numeric dtype
            (see parameter_bounds_error)

        category : ParameterCategory
            Parameter classification (required)
        description : str
            Scientific description and context
        references : List[str]
            Literature citations (DOI, BibTeX)
        tags : Set[str]
            Classification tags ('learnable', 'sensitive', etc.)
        dimension : str
            Physical dimension for unit validation
        provenance : Dict[str, Any]
            Source tracking and version information
        requires_grad : bool
            Whether parameter needs gradients for optimization
        
    Mathematical Expression:
        Parameter constraint: theta in [theta_min, theta_max]  units(theta) = expected_units
        Validation function: V(theta) -> {valid, error_messages}
        
    References:
        - Wilkinson, M.D. et al. (2016). FAIR Guiding Principles for data management
        - Force11 Data Citation Synthesis Group (2014). Data citation recommendations
        - CF-1.8 Convention for Climate and Forecast Metadata
    """
    key: str
    value: ParameterValue
    # units, dtype and bounds have no universal value and no default. A numeric
    # parameter without limits declares bounds=(-inf, inf) explicitly; None states
    # that no interval applies and is accepted only for a non-numeric dtype
    # (parameter_bounds_error).
    units: str
    dtype: str
    bounds: Optional[Tuple[Any, Any]]
    # The classification of the parameter is declared, never assumed.
    category: ParameterCategory
    default: Optional[ParameterValue] = None

    description: str = ""
    references: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    dimension: str = ""
    provenance: MetadataDict = field(default_factory=dict)
    requires_grad: bool = False
    allowed_values: Optional[List[Any]] = None  # For enum/categorical parameters
    
    def __post_init__(self):
        """Validate parameter specification after initialization."""
        if not self.key:
            raise ValueError("Parameter key cannot be empty")
        if not self._is_valid_key(self.key):
            raise ValueError(f"Invalid parameter key format: {self.key}")

        if isinstance(self.category, str):
            self.category = normalize_category(self.category, logger=logger)
        elif not isinstance(self.category, ParameterCategory):
            raise TypeError(
                "Parameter category must be a ParameterCategory or canonical string"
            )
        
        # Convert tags to set if it's a list
        if isinstance(self.tags, list):
            self.tags = builtins.set(self.tags)
        if any(not isinstance(tag, str) for tag in self.tags):
            raise TypeError("Parameter tags must be strings")
        
        # Convert bounds to tuple if it's a list (for JSON deserialization consistency)
        if isinstance(self.bounds, list):
            if len(self.bounds) != 2:
                raise ValueError(f"Invalid bounds format: {self.bounds}")
            self.bounds = (self.bounds[0], self.bounds[1])

        # The one bounds rule judges the declared bounds before normalization, so a
        # text endpoint of a numeric dtype is rejected rather than converted.
        bounds_error = parameter_bounds_error(self.bounds, self.dtype)
        if bounds_error is not None:
            raise ValueError(f"Parameter {self.key}: {bounds_error}")
        # A numeric dtype states its value and default as numbers; text (for example
        # a YAML 1.1 token such as 1e-6) is rejected, never converted.
        if parameter_dtype_is_numeric(self.dtype):
            for field_name in ("value", "default"):
                text = _text_leaves(getattr(self, field_name))
                if text:
                    raise ValueError(
                        f"Parameter {self.key}: {field_name} states {text!r} as text, but dtype "
                        f"{self.dtype!r} is numeric; write a number (1.0e+7, 1.0e-6)"
                    )

        # Normalize a declared pair (numbers to float; categorical labels are kept).
        if self.bounds is not None:
            try:
                if len(self.bounds) != 2:
                    raise ValueError(f"Invalid bounds format: {self.bounds}")
                min_val, max_val = self.bounds

                # Skip validation for nested iterable bounds (used by array parameters)
                if isinstance(min_val, (list, tuple)) and isinstance(max_val, (list, tuple)):
                    pass
                else:
                    try:
                        min_numeric = float(min_val)
                        max_numeric = float(max_val)
                    except (TypeError, ValueError):
                        # Categorical bounds retain their declared labels.
                        self.bounds = (min_val, max_val)
                    else:
                        if min_numeric > max_numeric:
                            raise ValueError(
                                f"Invalid bounds: minimum ({min_numeric}) > maximum ({max_numeric})"
                            )
                        self.bounds = (min_numeric, max_numeric)
            except (IndexError, TypeError) as e:
                raise ValueError(f"Invalid bounds format: {self.bounds}, error: {e}")

        # Initialize provenance if empty
        if not self.provenance:
            self.provenance = {
                'source': 'user_defined',
                'created_at': None,
                'schema_version': '1.0.0'
            }
    @staticmethod
    def _is_valid_key(key: str) -> bool:
        """Check if parameter key follows dot-notation conventions."""
        if not key or not isinstance(key, str):
            return False
        
        # Must be dot-separated alphanumeric segments
        segments = key.split('.')
        for segment in segments:
            if not segment or not segment.replace('_', '').isalnum():
                return False
        
        return True
    
    def validate_value(self, value: ParameterValue) -> ValidationErrors:
        """
        Validate parameter value against specification constraints.
        
        Parameters:
            value : ParameterValue
                Value to validate
                
        Returns:
            ValidationErrors: List of validation error messages
        """
        errors = []
        dtype_root = _dtype_root(self.dtype)

        # Check for allowed values first (for enum/categorical parameters)
        if hasattr(self, "allowed_values") and self.allowed_values is not None:
            if isinstance(value, (list, tuple, builtins.set)):
                invalid_values = [item for item in value if item not in self.allowed_values]
                if invalid_values:
                    errors.append(
                        f"Parameter {self.key} values {invalid_values} not in allowed values: {self.allowed_values}"
                    )
            else:
                if value not in self.allowed_values:
                    errors.append(
                        f"Parameter {self.key} value '{value}' not in allowed values: {self.allowed_values}"
                    )
            return errors

        # A list-valued parameter that declares one (minimum, maximum) pair per
        # component is judged by the one componentwise rule, whatever the type of
        # the candidate; every other declaration follows the checks below.
        if dtype_root in _SEQUENCE_DTYPES:
            component_bounds = componentwise_bounds(self.bounds)
            if component_bounds is not None:
                return [
                    f"Parameter {self.key} {reason}"
                    for reason in componentwise_bounds_errors(value, component_bounds)
                ]
        
        # Treat configuration-like payloads as non-numeric when dtype is absent or declarative
        if isinstance(value, (dict, list, tuple)) and (dtype_root in _NON_NUMERIC_DTYPES or dtype_root in {"", "none"}):
            if value is None:
                errors.append(
                    f"Parameter {self.key} value is None but dtype '{self.dtype}' expects non-null configuration"
                )
            return errors
        
        # Check if this is a string/enum parameter (e.g., pathway)
        if isinstance(value, str):
            # For string parameters with string bounds (like ["C3", "C4"])
            if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
                min_val, max_val = self.bounds
                if isinstance(min_val, str) and isinstance(max_val, str):
                    # String bounds mean it's an enum - just check equality
                    if min_val == max_val and value != min_val:
                        errors.append(f"Parameter {self.key} must be '{min_val}', got '{value}'")
                    # Otherwise no validation for general string bounds
                    return errors
            # No numeric validation for strings
            return errors
        
        # Type checking for numeric parameters
        if isinstance(value, Tensor):
            # Check for NaN values
            if torch.isnan(value).any():
                errors.append(f"Parameter {self.key} contains NaN values")
            
            # Check for Inf values  
            if torch.isinf(value).any():
                errors.append(f"Parameter {self.key} contains Inf values")
            
            # For bounds checking, only process scalar tensors
            if value.numel() == 1:
                numeric_value = float(value.item())
            else:
                # Skip bounds checking for multi-element tensors
                return errors
        elif isinstance(value, (int, float)):
            numeric_value = float(value)
            # References: IEEE Std 754-2019, Sect. 5.11 (every ordered comparison
            #   with a NaN operand is false) and Sect. 6.1 (infinity arithmetic);
            #   RELIC fail-fast input policy.
            # Mathematical expression: valid(v) = isfinite(v) and lo <= v <= hi,
            #   with the bounds comparison below.
            # Parameter meanings: v the candidate value; (lo, hi) the declared
            #   bounds of the parameter (an infinite bound is an open side).
            # Units: v, lo and hi share the declared units of the parameter.
            # Domain/constraints: NaN has no position relative to the interval
            #   and +/-inf reports an upstream overflow; neither is a bounds
            #   violation, so a non-finite v is rejected before any bounds
            #   comparison, with finite or open bounds alike, as the
            #   componentwise rule rejects a non-finite component. The message
            #   carries the value (nan, inf or -inf), which register_parameter
            #   treats as a severe error.
            # Sign convention: none (a domain check, not a flux).
            if not math.isfinite(numeric_value):
                errors.append(f"Parameter {self.key} is not finite: {numeric_value}")
                return errors
        else:
            # Non-numeric configuration-like parameters (dict/list/object/etc.) should not be
            # forced through numeric validation based solely on their Python type.
            if dtype_root in _NON_NUMERIC_DTYPES:
                if value is None:
                    errors.append(
                        f"Parameter {self.key} value is None but dtype '{self.dtype}' expects non-null configuration"
                    )
                return errors
            errors.append(f"Parameter {self.key} must be numeric, got {type(value)}")
            return errors
        
        # Bounds checking for numeric values
        if self.bounds is not None:
            min_val, max_val = self.bounds
            if min_val is not None and numeric_value < min_val:
                errors.append(f"Parameter {self.key} below minimum: {numeric_value} < {min_val}")
            if max_val is not None and numeric_value > max_val:
                errors.append(f"Parameter {self.key} above maximum: {numeric_value} > {max_val}")
        
        # Sign and range constraints come from the declared bounds only; none is
        # inferred from substrings of the key (rate, flux, concentration, temperature, ...).
        
        return errors
    
    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def to_tensor(self, device: Optional[torch.device] = None) -> Tensor:
        """Convert parameter value to PyTorch tensor."""
        if isinstance(self.value, Tensor):
            tensor = self.value
        else:
            dtype_str = (self.dtype or "").lower()

            # Guard against non-numeric parameters (e.g., categorical/string)
            non_numeric_types = {"string", "str", "category", "enum", "object"}
            if isinstance(self.value, str) or dtype_str in non_numeric_types:
                raise TypeError(
                    f"Parameter {self.key} with dtype '{self.dtype}' is non-numeric "
                    "and cannot be converted to a tensor. Ensure requires_grad is False."
                )

            if dtype_str in {"bool", "boolean"}:
                tensor = convert_to_tensor(bool(self.value), dtype=torch.bool)
            elif dtype_str in {"int32", "int"}:
                tensor = convert_to_tensor(int(self.value), dtype=torch.int32)
            elif dtype_str in {"int64", "long"}:
                tensor = convert_to_tensor(int(self.value), dtype=torch.int64)
            elif dtype_str in {"float64", "double"}:
                tensor = convert_to_tensor(float(self.value), dtype=torch.float64)
            else:
                tensor = convert_to_tensor(self.value, dtype=torch.float32)

        if device is not None:
            tensor = convert_to_tensor(tensor, dtype=tensor.dtype, device=device)

        if self.requires_grad and not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)

        if tensor.is_floating_point():
            if tensor.requires_grad != self.requires_grad:
                # Never change the gradient flag of the stored value (a shallow copy
                # of this spec shares it): take a detached alias of the same storage
                # and set the flag on the alias. A stored tensor whose flag already
                # matches is returned unchanged, so a trainable leaf keeps receiving
                # its gradients; the governed trainable path is the manager's
                # nn.Parameter built from this tensor.
                tensor = tensor.detach()
                tensor.requires_grad_(self.requires_grad)
        elif self.requires_grad:
            raise TypeError(
                f"Parameter {self.key} uses non-floating dtype {tensor.dtype} but requires gradients."
            )

        return tensor
    
    def from_tensor(self, tensor: Tensor) -> None:
        """Update parameter value from PyTorch tensor.
        
        Parameters:
            tensor : Tensor
                PyTorch tensor to convert from
        """
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        
        # Extract scalar value
        if tensor.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape {tensor.shape}")
        
        # Update value
        self.value = float(tensor.item())
        
        # Update dtype based on tensor dtype
        if tensor.dtype == torch.float64:
            self.dtype = 'float64'
        elif tensor.dtype == torch.int32:
            self.dtype = 'int32'
        elif tensor.dtype == torch.int64:
            self.dtype = 'int64'
        else:
            self.dtype = 'float32'
        
        # Update requires_grad
        self.requires_grad = tensor.requires_grad
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Pydantic-style serialization method.
        
        This is an alias for to_dict() to maintain consistency
        with Pydantic v2 patterns.
        
        Returns:
            Dict[str, Any]: Serialized parameter specification
        """
        return self.to_dict()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter specification to dictionary for serialization."""
        # Handle tensor values properly
        def convert_tensor_value(value):
            if isinstance(value, Tensor):
                from .serialization import serialize_tensor
                return serialize_tensor(value)
            return value

        bounds_value = self.bounds
        if isinstance(bounds_value, tuple):
            bounds_value = list(bounds_value)
        
        return {
            'key': self.key,
            'value': convert_tensor_value(self.value),
            'default': convert_tensor_value(self.default),
            'units': self.units,
            'dtype': self.dtype,
            'bounds': bounds_value,
            'category': self.category.value,
            'description': self.description,
            'references': self.references,
            'tags': sorted(self.tags),
            'dimension': self.dimension,
            'provenance': self.provenance,
            'requires_grad': self.requires_grad,
            'allowed_values': (
                list(self.allowed_values) if self.allowed_values is not None else None
            ),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSpec':
        """Create ParameterSpec from dictionary."""
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Handle missing required value field
        if 'value' not in data:
            # Skip parameters without values (they might be configuration sections)
            raise ValueError("Parameter specification missing required 'value' field")

        # The parameter serializer owns Tensor wire encoding for both fields.
        for field_name in ("value", "default"):
            encoded = data.get(field_name)
            if isinstance(encoded, dict) and encoded.get("type") == "tensor":
                from .serialization import deserialize_tensor
                data[field_name] = deserialize_tensor(encoded)
        
        # Handle enum conversion with default for unknown categories
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = normalize_category(data['category'], logger=logger)
        
        # dtype is declared by the payload; it is never inferred from the value type.
        # A payload without 'dtype' fails at construction (required field).
        if 'dtype' in data:
            dtype_val = data.get('dtype')
            if isinstance(dtype_val, type):
                data['dtype'] = dtype_val.__name__
            elif isinstance(dtype_val, str):
                data['dtype'] = dtype_val.strip()
            if data['dtype'] in (None, "") or str(data['dtype']).lower() == "none":
                raise ValueError(f"Parameter '{data.get('key')}' declares an empty dtype")
            # A text value is passed through unchanged: construction rejects text in a
            # numeric entry instead of converting it.

        # Handle set conversion
        if 'tags' in data and isinstance(data['tags'], list):
            data['tags'] = builtins.set(data['tags'])
        
        # Handle bounds conversion (list to tuple)
        if 'bounds' in data and isinstance(data['bounds'], list):
            data['bounds'] = tuple(data['bounds'])
        
        # Normalize allowed_values for categorical parameters
        if 'allowed_values' in data:
            allowed = data['allowed_values']
            if allowed is None:
                data['allowed_values'] = None
            else:
                if isinstance(allowed, (AbstractSet, Sequence)) and not isinstance(allowed, (str, bytes)):
                    data['allowed_values'] = list(allowed)
                else:
                    data['allowed_values'] = [allowed]

        # Merge metadata into provenance
        provenance = data.get('provenance', {})
        if 'calibration_metadata' in data:
            provenance['calibration_metadata'] = data.pop('calibration_metadata')
        
        # Store additional fields in provenance instead of discarding them
        additional_fields = ['namespace', 'uncertainty', 'options']
        for field_item in additional_fields:
            if field_item in data:
                provenance[field_item] = data.pop(field_item)
        
        data['provenance'] = provenance
        
        # Filter out any unsupported fields
        valid_fields = {item.name for item in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
