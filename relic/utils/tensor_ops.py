"""Shared tensor conversion utilities."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

__all__ = ["to_tensor"]


def to_tensor(
    value: Any,
    *,
    dtype: Optional[torch.dtype] = torch.float32,
    device: Optional[torch.device] = None,
    requires_grad: Optional[bool] = None,
) -> Tensor:
    """Convert value to ``torch.Tensor`` with explicit dtype/device behavior."""
    if isinstance(value, Tensor):
        tensor = value
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
    else:
        target_device = (
            device
            if device is not None
            else (torch.device("cpu"))
        )
        if dtype is None:
            tensor = torch.as_tensor(value, device=target_device)
        else:
            tensor = torch.as_tensor(value, dtype=dtype, device=target_device)

    if requires_grad is not None:
        if requires_grad and not tensor.is_floating_point():
            raise TypeError(
                f"Tensor with dtype {tensor.dtype} cannot set requires_grad=True"
            )
        if tensor.is_floating_point():
            tensor.requires_grad_(requires_grad)

    return tensor
