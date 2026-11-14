"""Time-index alignment helpers for tutorial case diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeCoverage:
    model_steps: int
    available_steps: int
    is_complete: bool


def _validate_datetime_index(frame: pd.DataFrame, *, frame_name: str) -> None:
    if frame.empty:
        raise ValueError(f"{frame_name} is empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{frame_name} must use a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError(f"{frame_name} contains duplicate timestamps")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{frame_name} timestamps are not monotonic")


def validate_model_time_coverage(
    model_state: pd.DataFrame,
    available_input: pd.DataFrame,
    *,
    allow_partial: bool,
) -> TimeCoverage:
    """Validate that model timestamps form the expected input-window prefix."""

    _validate_datetime_index(model_state, frame_name="model state")
    _validate_datetime_index(available_input, frame_name="forcing")

    model_steps = len(model_state)
    available_steps = len(available_input)
    if model_steps > available_steps:
        raise ValueError(
            f"model timeline has {model_steps} timestamps but forcing has only "
            f"{available_steps}"
        )

    expected_index = available_input.index[:model_steps]
    if not model_state.index.equals(expected_index):
        raise ValueError("model timeline does not match the forcing timeline prefix")

    is_complete = model_steps == available_steps
    if not is_complete and not allow_partial:
        raise ValueError(
            f"incomplete model timeline: {model_steps}/{available_steps} timestamps; "
            "rerun without --days or pass --allow-partial"
        )

    return TimeCoverage(
        model_steps=model_steps,
        available_steps=available_steps,
        is_complete=is_complete,
    )


def align_frame_to_reference_time(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    frame_name: str,
    require_complete: bool,
) -> pd.DataFrame:
    """Align a frame to the exact model timeline without extrapolation."""

    _validate_datetime_index(reference, frame_name="model state")
    _validate_datetime_index(frame, frame_name=frame_name)

    common_index = reference.index.intersection(frame.index)
    if common_index.empty:
        raise ValueError(f"{frame_name} does not overlap the model timeline")

    missing_timestamps = reference.index.difference(frame.index)
    if require_complete and not missing_timestamps.empty:
        raise ValueError(
            f"{frame_name} is missing {len(missing_timestamps)} required timestamps"
        )

    aligned = frame.reindex(reference.index)
    aligned.index.name = reference.index.name
    return aligned
