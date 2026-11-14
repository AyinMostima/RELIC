"""Unit tests for the streaming module (source adapters and window processor)."""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

import relic.data.streaming.stream as stream_module
from relic.data.streaming.stream import (
    KafkaStreamSource,
    SocketStreamSource,
    StreamConfig,
    StreamProcessor,
    StreamSourceType,
    StreamWindow,
)


def test_socket_source_constructs_and_parses_text_message():
    config = StreamConfig(source_type=StreamSourceType.SOCKET)
    source = SocketStreamSource(config)
    parsed = source._parse_message("hello stream")
    assert parsed["data"] == "hello stream"
    assert parsed["metadata"]["transport"] == "tcp"


def test_kafka_source_dependency_gate_behavior():
    config = StreamConfig(source_type=StreamSourceType.KAFKA)
    if not stream_module.STREAMING_DEPS.get("kafka", False):
        with pytest.raises(RuntimeError, match="requires kafka-python"):
            KafkaStreamSource(config)
    else:
        source = KafkaStreamSource(config)
        assert source._topic


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("min_data_fraction", 0.7), ("max_gap_duration", timedelta(hours=2))],
)
def test_stream_window_has_no_unconsumed_fields(field_name, value):
    """Coverage and gap limits are owned by QualityConfig parameters."""
    assert set(StreamWindow.__dataclass_fields__) == {
        "size",
        "step",
        "aggregation_method",
        "quality_checks",
    }
    with pytest.raises(TypeError, match=field_name):
        StreamWindow(size=timedelta(hours=1), **{field_name: value})


@pytest.mark.parametrize("method", ["median", "MEAN", "", "time_weighted_mean"])
def test_stream_window_rejects_unknown_aggregation_methods(method):
    with pytest.raises(ValueError, match="aggregation_method"):
        StreamWindow(size=timedelta(hours=1), aggregation_method=method)


@pytest.mark.parametrize(
    ("method", "expected"),
    [("mean", 281.5), ("sum", 1126.0), ("min", 280.0), ("max", 283.0)],
)
def test_window_reduction_uses_the_declared_method(method, expected):
    processor = _processor_with_window(
        [280.0, 281.0, 282.0, 283.0],
        quality_checks=False,
        aggregation_method=method,
    )

    result = asyncio.run(processor._process_window())

    assert float(result["data"]["tas"]) == pytest.approx(expected)
    assert result["data"]["tas"].attrs["cell_methods"] == f"time: {method}"


def _processor_with_window(values, *, quality_checks=True, aggregation_method="mean"):
    start = datetime(2000, 1, 1)
    times = np.array(
        [np.datetime64("2000-01-01T00:00") + np.timedelta64(h, "h") for h in range(len(values))]
    )
    dataset = xr.Dataset(
        {"tas": ("time", np.asarray(values, dtype="float64"), {"units": "K"})},
        coords={"time": times},
    )
    processor = StreamProcessor(
        StreamConfig(
            window=StreamWindow(
                size=timedelta(hours=len(values)),
                aggregation_method=aggregation_method,
                quality_checks=quality_checks,
            )
        )
    )
    processor._window_start = start
    processor._current_window.append({"data": dataset, "metadata": {"timestamp": start}})
    return processor


def test_window_quality_control_uses_the_quality_controller_api():
    """Four of ten hours valid grades POOR, which fails validation for aggregation."""
    nan = float("nan")
    processor = _processor_with_window([280.0] + [nan] * 6 + [281.0, 282.0, 283.0])

    with pytest.warns(UserWarning, match="insufficient coverage"):
        result = asyncio.run(processor._process_window())

    assert result["data"]["tas"].attrs["quality_flags"] == "validation_failed"


def test_window_quality_control_passes_a_complete_series():
    processor = _processor_with_window([280.0 + 0.1 * hour for hour in range(12)])

    result = asyncio.run(processor._process_window())

    assert "quality_flags" not in result["data"]["tas"].attrs
    assert float(result["data"]["tas"]) == pytest.approx(280.55)


def test_window_quality_control_errors_propagate(monkeypatch):
    processor = _processor_with_window([280.0 + 0.1 * hour for hour in range(12)])

    def fail(*_args, **_kwargs):
        raise RuntimeError("quality controller failed")

    monkeypatch.setattr(processor._quality_controller, "validate_for_operation", fail)

    with pytest.raises(RuntimeError, match="quality controller failed"):
        asyncio.run(processor._process_window())


def test_window_quality_control_can_be_disabled(monkeypatch):
    processor = _processor_with_window(
        [280.0 + 0.1 * hour for hour in range(12)],
        quality_checks=False,
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("quality control must not run when disabled")

    monkeypatch.setattr(processor._quality_controller, "validate_for_operation", fail)

    result = asyncio.run(processor._process_window())
    assert "quality_flags" not in result["data"]["tas"].attrs


def test_directory_source_and_sink_raise_instead_of_warning(tmp_path, monkeypatch):
    """A file the directory source cannot process and an item the sink cannot write raise."""
    from relic.data.streaming.stream import DirectoryStreamSource, StreamSink

    source = DirectoryStreamSource(StreamConfig(source_type=StreamSourceType.DIRECTORY, source_path=tmp_path))
    broken = tmp_path / "broken.nc"
    broken.write_bytes(b"not a netcdf file")
    with pytest.raises((OSError, ValueError, RuntimeError)):
        asyncio.run(source._process_new_file(broken))

    sink = StreamSink(StreamConfig(source_type=StreamSourceType.DIRECTORY, output_path=tmp_path / "out"))

    async def failing_append(item):
        raise OSError("disk full")

    monkeypatch.setattr(sink, "_append_to_current_file", failing_append)
    monkeypatch.setattr(sink, "_needs_new_file", lambda item: False)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(sink._write_item({"data": None, "metadata": {}}))
