"""Tests for API path traversal protections in schema validators."""

from __future__ import annotations

import pytest

from relic.interfaces.api.schemas import DataConvertRequest, DataReadRequest, DataWriteRequest


def test_data_read_request_blocks_path_traversal() -> None:
    with pytest.raises(ValueError, match="Path traversal blocked"):
        DataReadRequest(path="..\\..\\Windows\\System32\\drivers\\etc\\hosts")


def test_data_write_request_requires_output_inside_api_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RELIC_API_DATA_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="Path traversal blocked"):
        DataWriteRequest(
            input_path=str(tmp_path / "missing.nc"),
            output_path="C:\\Windows\\Temp\\out.nc",
            format="netcdf4",
        )


def test_data_convert_request_accepts_relative_paths_under_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RELIC_API_DATA_ROOT", str(tmp_path))
    source = tmp_path / "input.nc"
    source.write_text("dummy", encoding="utf-8")
    request = DataConvertRequest(
        input_path="input.nc",
        output_path="out.nc",
        target_format="csv",
    )
    assert request.input_path.endswith("input.nc")
    assert request.output_path.endswith("out.nc")
