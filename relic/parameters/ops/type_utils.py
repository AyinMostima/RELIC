from __future__ import annotations

import copy
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn


def format_expected_type(expected_type: Optional[Union[type, Tuple[type, ...]]]) -> str:
    """Format expected_type for cache keys and error messages."""
    if expected_type is None:
        return "any"
    if isinstance(expected_type, tuple):
        names = [t.__name__ for t in expected_type if hasattr(t, "__name__")]
        return "|".join(names) if names else "tuple"
    if hasattr(expected_type, "__name__"):
        return expected_type.__name__
    return str(expected_type)


def validate_type(value: Any, expected_type: type, key: str) -> Any:
    """
    Validate that value matches expected type.

    Parameters
    ----------
    value : Any
        Value to validate.
    expected_type : type
        Expected Python type.
    key : str
        Parameter key for error messages.

    Returns
    -------
    Any
        The validated value.
    """
    numeric_expected = expected_type in (float, int) or (
        isinstance(expected_type, tuple) and any(t in (float, int) for t in expected_type)
    )
    if isinstance(value, (torch.Tensor, nn.Parameter)) and numeric_expected:
        tensor_value = value.detach()
        if tensor_value.numel() == 1:
            scalar_value = tensor_value.item()
            if expected_type == int:
                return int(scalar_value)
            if expected_type == float:
                return float(scalar_value)
            if isinstance(expected_type, tuple):
                if int in expected_type and float not in expected_type:
                    return int(scalar_value)
                if float in expected_type:
                    return float(scalar_value)
                return scalar_value
            return scalar_value
    # An int where float is expected is widened (lossless numeric-tower
    # conversion, PEP 484); a bool is not a number here and no value is narrowed.
    float_expected = expected_type is float or (
        isinstance(expected_type, tuple) and float in expected_type and int not in expected_type
    )
    if float_expected and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, bool) and numeric_expected and bool not in (
        expected_type if isinstance(expected_type, tuple) else (expected_type,)
    ):
        raise TypeError(
            f"Parameter '{key}' has type bool, expected {format_expected_type(expected_type)}"
        )
    if not isinstance(value, expected_type):
        expected_name = format_expected_type(expected_type)
        raise TypeError(
            f"Parameter '{key}' has type {type(value).__name__}, expected {expected_name}"
        )
    return value


def caller_owned_value(value: Any) -> Any:
    """
    Return a value read from the manager as an object owned by the caller.

    The one copy rule of the parameter value reads (architecture overview
    Section 6.1): a Python container (dict, list, set, tuple) is copied at
    every depth, so changing the returned value changes neither the stored
    value, the database specification it was taken from, nor the
    query-result cache; a stored value changes only through the mutation
    owners (set_value, set_parameter, register_parameter, restore). A
    Tensor, a learnable nn.Parameter binding included, keeps its identity
    at any depth (Section 4.2); every other object is returned as it is.
    A container type is kept (copy.copy of the container, then its items).
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        owned = copy.copy(value)
        for item_key, item in value.items():
            owned[item_key] = caller_owned_value(item)
        return owned
    if isinstance(value, list):
        owned = copy.copy(value)
        owned[:] = [caller_owned_value(item) for item in value]
        return owned
    if isinstance(value, set):
        # Set members are hashable, so only the set itself is copied.
        return copy.copy(value)
    if isinstance(value, tuple):
        items = [caller_owned_value(item) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return tuple(items) if type(value) is tuple else type(value)(*items)
    return value


def unwrap_trainable_value(value: Any, expected_type: Optional[type]) -> Any:
    """Detach nn.Parameter or requires_grad tensor when returning raw values."""
    if expected_type is not None:
        if isinstance(expected_type, tuple):
            if any(t in (torch.Tensor, nn.Parameter) for t in expected_type):
                return value
        if expected_type in (torch.Tensor, nn.Parameter):
            return value
    if isinstance(value, nn.Parameter):
        tensor_value = value.detach()
        if tensor_value.numel() == 1:
            return tensor_value.item()
        return tensor_value
    if isinstance(value, torch.Tensor) and value.requires_grad:
        tensor_value = value.detach()
        if tensor_value.numel() == 1:
            return tensor_value.item()
        return tensor_value
    return value
