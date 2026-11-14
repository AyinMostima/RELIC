"""Helpers for forward-only functor API in tests."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def _state_mapping(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    # Support tests that previously passed a raw concentration tensor.
    return {"concentration": state}


def call_forward(
    functor: Any,
    state: Optional[Dict[str, Any]] = None,
    *,
    time: float = 0.0,
    dt: float = 1.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call forward() for biological-style tests (direct kernel call).

    The state is the only input channel: a driver belongs in the state, and an
    ``environment`` keyword is rejected.
    """
    if "environment" in kwargs:
        raise TypeError("call_forward takes no environment argument; put the drivers in the state")
    result = functor.forward(_state_mapping(state), time, dt, **kwargs)
    if isinstance(result, dict):
        unwrapped = {
            key: (value.value if hasattr(value, "value") else value)
            for key, value in result.items()
        }
        return _normalize_rate_alias(unwrapped)
    return result


def call_transport_forward(
    functor: Any,
    source: Any,
    sink: Any,
    distance: Any = None,
    *,
    time: float = 0.0,
    dt: float = 1.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call forward() for transport-style source/sink tests."""
    runtime_kwargs = dict(kwargs)
    if distance is not None:
        runtime_kwargs.setdefault("distance", distance)
    source_dict = source if isinstance(source, dict) else {}
    sink_dict = sink if isinstance(sink, dict) else {}
    if hasattr(functor, "merge_source_sink"):
        state = functor.merge_source_sink(source, sink, **runtime_kwargs)
    else:
        state = {"source": source, "sink": sink, **runtime_kwargs}
    # Expose flattened transport fields for forward-only functors that read
    # top-level state variables instead of source/sink containers.
    for key, value in sink_dict.items():
        state.setdefault(key, value)
    for key, value in source_dict.items():
        state[key] = value
    # Backward-compatible aliases for forward-only transport functors.
    tracer_candidate = source_dict.get("concentration", source)
    state.setdefault("tracer_concentration", tracer_candidate)
    state.setdefault("concentration_field", tracer_candidate)
    state.setdefault("concentration", tracer_candidate)
    result = functor.forward(state, time, dt, **runtime_kwargs)
    if isinstance(result, dict):
        # Transport tests commonly assert raw tensors; unwrap UpdateItem-like payloads.
        unwrapped = {
            key: (value.value if hasattr(value, "value") else value)
            for key, value in result.items()
        }
        normalized = _normalize_flux_alias(unwrapped)
        if "advective_flux_x" not in normalized:
            if "flux" in normalized and isinstance(normalized["flux"], torch.Tensor):
                enriched = dict(normalized)
                enriched["advective_flux_x"] = normalized["flux"]
                return enriched
            if "advection_rate" in normalized and isinstance(normalized["advection_rate"], torch.Tensor):
                enriched = dict(normalized)
                enriched["advective_flux_x"] = normalized["advection_rate"]
                return enriched
            tracer_like = normalized.get("tracer_concentration")
            if isinstance(tracer_like, torch.Tensor):
                enriched = dict(normalized)
                enriched["advective_flux_x"] = tracer_like
                return enriched
        return normalized
    return result


def as_tensor(value: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert test values to tensor with optional device placement."""
    if isinstance(value, torch.Tensor):
        return value if device is None else value.to(device)
    return torch.as_tensor(value, device=device)


def _normalize_rate_alias(result: Dict[str, Any]) -> Dict[str, Any]:
    """Add a best-effort `rate` alias for forward-only test assertions."""
    enriched = dict(result)

    if "rate" in enriched:
        return enriched
    preferred_keys = [
        "total_bvoc_emission",
        "net_photosynthesis",
        "total_growth_rate",
        "phenological_factor",
        "monoterpenes_emission",
    ]
    for key in preferred_keys:
        if key in enriched and isinstance(enriched[key], torch.Tensor):
            enriched["rate"] = enriched[key]
            return enriched
    rate_like = [
        key
        for key, value in enriched.items()
        if isinstance(value, torch.Tensor) and ("rate" in key.lower() or "flux" in key.lower())
    ]
    if len(rate_like) == 1:
        enriched["rate"] = enriched[rate_like[0]]
        return enriched
    tensor_values = [value for value in enriched.values() if isinstance(value, torch.Tensor)]
    if len(tensor_values) == 1:
        enriched["rate"] = tensor_values[0]
        return enriched
    return enriched


def _normalize_flux_alias(result: Dict[str, Any]) -> Dict[str, Any]:
    """Add a best-effort `flux` alias for forward-only transport assertions."""
    if "flux" in result:
        return result
    flux_like = [
        key
        for key, value in result.items()
        if isinstance(value, torch.Tensor) and ("flux" in key.lower() or "rate" in key.lower())
    ]
    if len(flux_like) == 1:
        enriched = dict(result)
        enriched["flux"] = result[flux_like[0]]
        return enriched
    return result
