import math

import pytest
import torch

from relic.computation.benchmarks import (
    compute_dynamical_embedding,
    compute_mode_similarity_score,
    compute_spectral_signature,
    compute_structure_function,
)


def test_spectral_signature_sine_wave() -> None:
    dt = 0.1
    n_steps = 1024
    frequency = 0.25
    t = torch.arange(n_steps, dtype=torch.float32) * dt
    signal = torch.sin(2.0 * math.pi * frequency * t)

    result = compute_spectral_signature(signal, dt=dt)
    dominant = float(result["dominant_frequency"].item())

    assert dominant == pytest.approx(frequency, abs=0.02)


def test_structure_function_uniform() -> None:
    field = torch.ones(256, dtype=torch.float32) * 5.0
    result = compute_structure_function(field, max_lag=40, order=2)
    with pytest.raises(TypeError):
        compute_structure_function(field)  # lag range and order are declared
    assert torch.allclose(result["structure_function"], torch.zeros_like(result["structure_function"]))


def test_embedding_preserves_variance() -> None:
    t = torch.linspace(0.0, 100.0, steps=500)
    state_histories = {
        "x": torch.sin(t),
        "y": torch.cos(0.5 * t),
        "z": torch.sin(0.25 * t + 0.3),
    }
    embedding = compute_dynamical_embedding(state_histories, n_components=3)
    variance_sum = float(torch.sum(embedding["explained_variance_ratio"]).item())
    assert variance_sum <= 1.0 + 1e-6


def test_similarity_score_identity() -> None:
    t = torch.linspace(0.0, 20.0, steps=300)
    state_histories = {
        "x": torch.sin(t),
        "y": torch.cos(t),
    }
    embedding = compute_dynamical_embedding(state_histories, n_components=2)
    score = compute_mode_similarity_score(embedding, embedding)
    assert score == pytest.approx(1.0, abs=1e-6)
