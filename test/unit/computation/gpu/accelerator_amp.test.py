"""Tests for the canonical GPU AMP context helper."""

from __future__ import annotations

import warnings

import pytest
import torch

import relic.computation.gpu.accelerator as accelerator


def test_autocast_context_uses_modern_torch_amp_cuda_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_autocast(**kwargs):
        calls.append(dict(kwargs))
        return FakeContext()

    monkeypatch.setattr(accelerator, "is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)

    with accelerator.AutocastContext(
        enabled=True,
        dtype=torch.float16,
        cache_enabled=False,
    ):
        pass

    assert calls == [
        {
            "device_type": "cuda",
            "enabled": True,
            "dtype": torch.float16,
            "cache_enabled": False,
        }
    ]


def test_disabled_amp_does_not_enter_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.amp,
        "autocast",
        lambda **_kwargs: pytest.fail("disabled AMP must not enter autocast"),
    )

    with accelerator.AutocastContext(enabled=False):
        pass


@pytest.mark.gpu
def test_real_cuda_autocast_has_no_deprecated_api_warning() -> None:
    assert accelerator.is_cuda_available(), "CUDA is required for the real AMP test"

    left = torch.randn((32, 32), device="cuda", dtype=torch.float32)
    right = torch.randn((32, 32), device="cuda", dtype=torch.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with accelerator.AutocastContext(enabled=True, dtype=torch.float16):
            output = left @ right
        torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert not any(
        "torch.cuda.amp.autocast" in str(warning.message)
        for warning in caught
    )


def _failing_autocast(error: Exception):
    def autocast(**_kwargs):
        raise error

    return autocast


def test_autocast_failure_raises_unless_disabling_is_explicitly_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(accelerator, "is_cuda_available", lambda: True)
    monkeypatch.setattr(
        torch.amp, "autocast", _failing_autocast(RuntimeError("unsupported autocast"))
    )

    with pytest.raises(accelerator.GPUAcceleratorError, match="unsupported autocast"):
        with accelerator.AutocastContext(enabled=True):
            pass

    context = accelerator.AutocastContext(enabled=True, allow_disabled=True)
    with context:
        pass
    assert context.enabled is False
    assert context.disabled_reason == "unsupported autocast"


def test_autocast_programming_errors_are_never_turned_into_disabled_amp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(accelerator, "is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.amp, "autocast", _failing_autocast(TypeError("bad keyword")))

    with pytest.raises(TypeError, match="bad keyword"):
        with accelerator.AutocastContext(enabled=True, allow_disabled=True):
            pass
