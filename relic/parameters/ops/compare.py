"""Comparison helpers for parameter values."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def values_differ(old_value: Any, new_value: Any) -> bool:
    """Robust equality checker that is safe for torch.Tensor inputs."""
    if isinstance(old_value, Tensor) and isinstance(new_value, Tensor):
        if old_value.shape != new_value.shape:
            return True
        if old_value.dtype != new_value.dtype:
            return True
        # Equality is a value comparison; the device a cached copy lives on is
        # not part of the parameter identity.
        if old_value.device != new_value.device:
            new_value = new_value.to(device=old_value.device)
        return not torch.equal(old_value, new_value)
    if isinstance(old_value, Tensor) or isinstance(new_value, Tensor):
        return True
    return bool(old_value != new_value)
