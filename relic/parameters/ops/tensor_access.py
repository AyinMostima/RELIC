"""Tensor access helpers for parameter manager."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor


def get_tensor(
    manager: Any,
    key: str,
    device: Optional[torch.device] = None,
) -> Optional[Tensor]:
    """Get parameter value as a PyTorch tensor."""
    spec = manager.get_parameter(key)
    if spec is None:
        return None

    if device is None:
        device = manager.device

    return spec.to_tensor(device)


def get_as_tensor(
    manager: Any,
    key: str,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return parameter tensor or raise if missing."""
    tensor = get_tensor(manager, key, device=device)
    if tensor is None:
        raise KeyError(f"Parameter '{key}' not found in registry")
    return tensor
