"""Tests for computation benchmark tensor contracts."""

from __future__ import annotations

import torch

from relic.computation.benchmarks import (
    create_benchmark_function,
    create_test_inputs,
)


def test_transport_benchmark_inputs_and_outputs_preserve_batch_space_contract() -> None:
    inputs = create_test_inputs(
        "transport",
        batch_size=2,
        n_points=8,
        device=torch.device("cpu"),
    )

    assert inputs.shape == (2, 8, 4)
    assert torch.isfinite(inputs).all()
    for channel in (1, 2, 3):
        torch.testing.assert_close(
            inputs[:, :, channel],
            inputs[:, :1, channel].expand(-1, 8),
        )

    output = create_benchmark_function("transport")(inputs)

    assert output.shape == (2,)
    assert torch.isfinite(output).all()
