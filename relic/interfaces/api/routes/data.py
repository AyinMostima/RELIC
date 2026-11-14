"""
Data I/O API routes.

These endpoints expose safe file inspection, read, write, and format conversion
operations through the shared `DataIO` adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from fastapi import APIRouter

from relic.coupling.external.common.io import DataFormat

from ..schemas import (
    APIResponse,
    DataConvertRequest,
    DataReadRequest,
    DataWriteRequest,
    create_success_response,
)

router = APIRouter()
_DATA_IO_SINGLETON = None


def _get_data_io():
    global _DATA_IO_SINGLETON
    if _DATA_IO_SINGLETON is None:
        from relic.data.io.adapter import DataIO

        _DATA_IO_SINGLETON = DataIO()
    return _DATA_IO_SINGLETON


def _summarize_payload(data: Any) -> Dict[str, Any]:
    """
    Build a compact schema-safe summary for API responses.
    """
    summary: Dict[str, Any] = {"kind": type(data).__name__}

    if isinstance(data, dict):
        variables: Dict[str, Any] = {}
        for name, value in data.items():
            item: Dict[str, Any] = {"type": type(value).__name__}
            if torch.is_tensor(value):
                item["shape"] = list(value.shape)
                item["dtype"] = str(value.dtype)
                item["device"] = str(value.device)
            elif isinstance(value, np.ndarray):
                item["shape"] = list(value.shape)
                item["dtype"] = str(value.dtype)
            variables[str(name)] = item
        summary["variables"] = variables
        summary["variable_count"] = len(variables)
        return summary

    # xarray Dataset/DataArray expose dims and dtypes. Dimension sizes must be
    # integers; a mapping that is not name -> size raises instead of being
    # copied into the response unconverted.
    dims = getattr(data, "dims", None)
    if dims is not None:
        summary["dims"] = {str(k): int(v) for k, v in dict(dims).items()}

    data_vars = getattr(data, "data_vars", None)
    if data_vars is not None:
        var_summary: Dict[str, Any] = {}
        for key, value in data_vars.items():
            var_summary[str(key)] = {
                "dtype": str(getattr(value, "dtype", "unknown")),
                "dims": [str(d) for d in getattr(value, "dims", ())],
            }
        summary["variables"] = var_summary
        summary["variable_count"] = len(var_summary)

    return summary


def _normalize_json_data(payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
    converted: Dict[str, np.ndarray] = {}
    for key, value in payload.items():
        converted[str(key)] = np.asarray(value)
    return converted


@router.get("/", response_model=APIResponse)
async def data_info() -> APIResponse:
    """
    Return supported data capabilities and API data root.
    """
    data_io = _get_data_io()
    formats = data_io.list_formats()
    return create_success_response(
        message="Data I/O endpoints are available.",
        data={
            "status": "ready",
            "capabilities": ["inspect", "read", "write", "convert"],
            "formats": formats,
        },
    )


# Route failures propagate to the app-level error handler
# (middleware.handle_errors), which returns the structured error body with an
# HTTP error status (ValueError 400, FileNotFoundError 404, ImportError 503,
# other errors 500).
@router.post("/inspect", response_model=APIResponse)
async def inspect_data_file(request: DataReadRequest) -> APIResponse:
    data_io = _get_data_io()
    info = data_io.get_file_info(request.path)
    return create_success_response(
        message="File inspection completed.",
        data=info,
    )


@router.post("/read", response_model=APIResponse)
async def read_data(request: DataReadRequest) -> APIResponse:
    data_io = _get_data_io()
    torch_dtype = getattr(torch, request.dtype, None)
    if not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype '{request.dtype}' requested for data read.")
    # Every request field reaches the reader explicitly; the boundary default
    # "auto" is passed as DataFormat.AUTO, never left to the reader's own default.
    kwargs: Dict[str, Any] = {
        "variables": request.variables,
        "to_torch": request.to_torch,
        "torch_device": request.device,
        "torch_dtype": torch_dtype,
        "format": DataFormat(request.format),
    }
    if request.chunks:
        kwargs["chunks"] = request.chunks

    payload = data_io.read(request.path, **kwargs)
    summary = _summarize_payload(payload)
    return create_success_response(
        message="Read operation completed.",
        data={
            "path": request.path,
            "summary": summary,
        },
    )


@router.post("/write", response_model=APIResponse)
async def write_data(request: DataWriteRequest) -> APIResponse:
    data_io = _get_data_io()
    output_path = Path(request.output_path)

    if request.input_path:
        source = data_io.read(request.input_path)
    elif request.data is not None:
        source = _normalize_json_data(request.data)
    else:
        raise ValueError("Either input_path or data must be provided.")

    # DataIO.write returns a WriteResult; its fields are read directly.
    result = data_io.write(
        source,
        output_path,
        format=request.format,
        compression=request.compression,
        compression_level=request.compression_level,
        cf_sync=request.validate_cf_variable_metadata,
    )
    return create_success_response(
        message="Write operation completed.",
        data={
            "output_path": str(output_path),
            "file_size": int(result.file_size),
            "compression_ratio": result.compression_ratio,
            "cf_validation_scope": "single_variable_standard_name_and_units",
            "dataset_conformance_checked": False,
            "cf_compliant": False,
        },
        paths={"output": str(output_path)},
    )


@router.post("/convert", response_model=APIResponse)
async def convert_data_format(request: DataConvertRequest) -> APIResponse:
    data_io = _get_data_io()
    output_path = Path(request.output_path)
    # DataIO.convert_format returns (ReadResult, WriteResult); fields are read directly.
    read_result, write_result = data_io.convert_format(
        source_path=request.input_path,
        target_path=request.output_path,
        target_format=request.target_format,
        preserve_metadata=request.preserve_metadata,
        compression=request.compression,
    )
    return create_success_response(
        message="Format conversion completed.",
        data={
            "input_path": request.input_path,
            "output_path": str(output_path),
            "target_format": request.target_format,
            "source_load_time_seconds": float(read_result.load_time),
            "write_time_seconds": float(write_result.write_time),
            "output_size": int(write_result.file_size),
        },
        paths={"output": str(output_path)},
    )
