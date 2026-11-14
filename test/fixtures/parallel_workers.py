"""Importable worker callables for real Windows process tests."""

from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


def parameter_record(alpha: int, beta: int) -> dict[str, int]:
    """Return a deterministic parameter-sweep value and the real worker PID."""
    return {
        "pid": os.getpid(),
        "value": alpha * 10 + beta,
    }


def ensemble_linear_model(inputs: dict[str, object]) -> dict[str, torch.Tensor]:
    """Evaluate a deterministic pure tensor member and expose its worker PID."""
    theta = torch.as_tensor(inputs["theta"])
    offset = torch.as_tensor(
        inputs.get("offset", 0.0),
        dtype=theta.dtype,
        device=theta.device,
    )
    return {
        "y": theta * 2.0 + offset,
        "worker_pid": torch.tensor(float(os.getpid()), dtype=torch.float64),
    }


class PIDTensorDataset(Dataset):
    """Return stable indices together with the real DataLoader worker PID."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(index, dtype=torch.int64),
            torch.tensor(os.getpid(), dtype=torch.int64),
        )
