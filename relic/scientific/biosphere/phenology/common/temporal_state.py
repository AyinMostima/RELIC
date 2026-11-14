"""Temporal state utilities for phenology modules.

These lightweight containers provide rolling-window and accumulator helpers for
phenological diagnostics. They follow the statistical running-mean practices
outlined in Wilks (2011, Statistical Methods in the Atmospheric Sciences) and
maintain differentiability by storing values as torch.Tensor objects.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Union

import torch
from torch import Tensor
from relic.utils.tensor_ops import to_tensor as convert_to_tensor

Number = Union[float, int, Tensor]


class RollingWindowState:
    """Fixed-size rolling buffer that stores the most recent samples."""

    def __init__(
        self,
        window_size: int,
        feature_shape: Union[int, Iterable[int]] = 1,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        if isinstance(feature_shape, int):
            feature_shape = (feature_shape,)
        self.window_size = int(window_size)
        self._buffer = torch.zeros(
            (self.window_size, *feature_shape),
            dtype=dtype,
            device=device or (torch.device("cpu")),
        )
        self._count = 0

    @property
    def device(self) -> torch.device:
        return self._buffer.device

    def append(self, value: Number) -> None:
        """Append a new value, discarding the oldest sample when full."""
        tensor = convert_to_tensor(value, dtype=self._buffer.dtype, device=self.device)
        if tensor.shape != self._buffer.shape[1:]:
            tensor = tensor.reshape(self._buffer.shape[1:])
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=0)
        self._buffer[-1] = tensor
        if self._count < self.window_size:
            self._count += 1

    def to_tensor(self) -> Tensor:
        """Return the buffer ordered from oldest to newest sample."""
        if self._count < self.window_size:
            return self._buffer[-self._count :].clone()
        return self._buffer.clone()

    def clear(self) -> None:
        """Reset the rolling window contents."""
        self._buffer.zero_()
        self._count = 0


class TemporalAccumulator:
    """Numerically stable accumulator that tracks running sums and means."""

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        self._sum = torch.zeros(1, dtype=dtype, device=device or (torch.device("cpu")))
        self._count = torch.zeros(1, dtype=dtype, device=self._sum.device)

    def update(self, value: Number) -> Tensor:
        """Add a sample and return the updated mean."""
        tensor = convert_to_tensor(value, dtype=self._sum.dtype, device=self._sum.device)
        self._sum += tensor
        self._count += 1.0
        return self.mean()

    def mean(self) -> Tensor:
        """Return the running mean (zero when no samples exist)."""
        if bool((self._count <= 0).all()):
            return torch.zeros(1, dtype=self._sum.dtype, device=self._sum.device)
        return self._sum / self._count

    def total(self) -> Tensor:
        """Return the accumulated sum."""
        return self._sum.clone()

    def reset(self) -> None:
        self._sum.zero_()
        self._count.zero_()


class MultiVariableRollingWindow:
    """Dictionary of rolling windows for multi-variable diagnostics."""

    def __init__(
        self,
        variable_shapes: Dict[str, Union[int, Iterable[int]]],
        window_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        self._windows = {
            name: RollingWindowState(window_size, shape, dtype=dtype, device=device)
            for name, shape in variable_shapes.items()
        }

    def append(self, **samples: Number) -> None:
        for name, value in samples.items():
            if name not in self._windows:
                raise KeyError(f"Unknown variable '{name}' in MultiVariableRollingWindow")
            self._windows[name].append(value)

    def get_window(self, name: str) -> Tensor:
        if name not in self._windows:
            raise KeyError(f"Unknown variable '{name}' in MultiVariableRollingWindow")
        return self._windows[name].to_tensor()

    def clear(self) -> None:
        for window in self._windows.values():
            window.clear()


def create_rolling_buffer(
    window_size: int,
    feature_shape: Union[int, Iterable[int]] = 1,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> RollingWindowState:
    """Factory helper mirroring prior API for backwards consistency."""
    return RollingWindowState(window_size, feature_shape, dtype=dtype, device=device)

