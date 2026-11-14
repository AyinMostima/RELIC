# ruff: noqa: N999
"""Tests for Windows-safe surrogate DataLoader configuration."""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import TensorDataset

from relic.models.pytorch_surrogate import PyTorchSurrogate
from test.fixtures.parallel_workers import PIDTensorDataset


def _loader_owner(training_config: dict[str, object]) -> PyTorchSurrogate:
    owner = object.__new__(PyTorchSurrogate)
    object.__setattr__(owner, "training_config", dict(training_config))
    object.__setattr__(owner, "device", torch.device("cpu"))
    return owner


def test_zero_workers_omits_persistent_and_prefetch_options() -> None:
    owner = _loader_owner(
        {
            "shuffle_training": False,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "seed": 17,
        }
    )
    dataset = TensorDataset(torch.arange(4), torch.arange(4))

    train_loader, val_loader = owner._build_train_val_loaders(
        training_dataset=dataset,
        validation_dataset=dataset,
        batch_size=2,
    )

    assert train_loader.num_workers == 0
    assert train_loader.pin_memory is False
    assert train_loader.persistent_workers is False
    assert train_loader.prefetch_factor is None
    assert val_loader is not None
    assert val_loader.num_workers == 0
    assert owner._dataloader_execution_info["effective_num_workers"] == 0


def test_seeded_shuffle_is_reproducible() -> None:
    dataset = TensorDataset(torch.arange(12), torch.arange(12))

    def order() -> list[int]:
        owner = _loader_owner(
            {
                "shuffle_training": True,
                "num_workers": 0,
                # The owner is built without Functor initialization, so every
                # loader setting is declared (no YAML lookup is possible).
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "seed": 123,
            }
        )
        loader, _ = owner._build_train_val_loaders(
            training_dataset=dataset,
            validation_dataset=None,
            batch_size=3,
        )
        return [int(value) for batch, _ in loader for value in batch]

    assert order() == order()


@pytest.mark.serial
def test_real_windows_workers_use_picklable_initializer_and_report_child_pids() -> None:
    owner = _loader_owner(
        {
            "shuffle_training": False,
            "num_workers": 2,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "seed": 321,
        }
    )

    loader, _ = owner._build_train_val_loaders(
        training_dataset=PIDTensorDataset(8),
        validation_dataset=None,
        batch_size=2,
    )
    records = [
        (int(index), int(pid))
        for indices, pids in loader
        for index, pid in zip(indices, pids, strict=True)
    ]

    assert [index for index, _ in records] == list(range(8))
    worker_pids = {pid for _, pid in records}
    assert worker_pids
    assert os.getpid() not in worker_pids
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 2
    assert owner._dataloader_execution_info == {
        "requested_num_workers": 2,
        "effective_num_workers": 2,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "seed": 321,
    }
