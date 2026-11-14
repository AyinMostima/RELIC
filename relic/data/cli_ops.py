"""Data CLI workflow helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from relic.interfaces.cli.utils import split_names_csv
from relic.utils.imports import module_available

logger = logging.getLogger(__name__)


def _require_modules(module_names: Iterable[str], error_message: str) -> bool:
    missing = [name for name in module_names if not module_available(name)]
    if missing:
        print(f"{error_message}: missing {', '.join(missing)}")
        return False
    return True


def _resolve_data_format(value: Optional[str]):
    from relic.coupling.external.common.io import DataFormat

    if not value:
        return DataFormat.AUTO
    return DataFormat(value)


def _resolve_compression(value: Optional[str]):
    from relic.coupling.external.common.io import CompressionType

    if not value:
        return CompressionType.AUTO
    return CompressionType(value)


def _resolve_torch_dtype(value: Optional[str]):
    import torch

    if not value:
        return torch.float32
    if hasattr(torch, value):
        return getattr(torch, value)
    raise ValueError(f"Unsupported torch dtype: {value}")


def _print_dataset_info(dataset) -> None:
    print("Dataset summary:")
    try:
        print(dataset)
    except (TypeError, ValueError, RuntimeError) as exc:
        logger.warning("Dataset summary unavailable: %s", exc)
        print("Dataset summary unavailable")


class DataCLI:
    """Workflow coordinator for data CLI commands."""

    def run(self, args, config: Dict[str, Any]) -> bool:
        if not hasattr(args, "data_command") or not args.data_command:
            print("Error: data command requires a subcommand")
            print("Available commands: io, quality, stream")
            return False

        if args.data_command == "io":
            return self._run_io(args, config)
        if args.data_command == "quality":
            return self._run_quality(args, config)
        if args.data_command == "stream":
            return self._run_stream(args, config)

        logger.error("Unknown data command: %s", args.data_command)
        return False

    def _run_io(self, args, config: Dict[str, Any]) -> bool:
        if not hasattr(args, "io_command") or not args.io_command:
            print("Error: io command requires a subcommand")
            print("Available commands: read, write, convert, check")
            return False

        if args.io_command == "read":
            return self._run_io_read(args)
        if args.io_command == "write":
            return self._run_io_write(args)
        if args.io_command == "convert":
            return self._run_io_convert(args)
        if args.io_command == "check":
            return self._run_io_check(args)

        logger.error("Unknown io command: %s", args.io_command)
        return False

    def _run_io_read(self, args) -> bool:
        if not _require_modules(
            ["relic.data.io.reader"],
            "Error: data reader dependencies missing",
        ):
            return False
        from relic.data.io.reader import DataReader, ReadConfig

        try:
            read_config = ReadConfig(
                format=_resolve_data_format(getattr(args, "format", None)),
                variables=split_names_csv(getattr(args, "variables", None)) or None,
                to_torch=getattr(args, "to_torch", False),
                torch_device=args.device,
                torch_dtype=_resolve_torch_dtype(getattr(args, "dtype", None)),
            )
            reader = DataReader(read_config)
            result = reader.read_dataset(args.path)

            if getattr(args, "info_only", False):
                _print_dataset_info(result.data)
                return True

            if isinstance(result.data, dict):
                print("Loaded variables:")
                for name in result.data.keys():
                    print(f"  - {name}")
            else:
                _print_dataset_info(result.data)

            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")

            return True
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            print(f"Error reading data: {exc}")
            return False

    def _run_io_write(self, args) -> bool:
        if not _require_modules(
            ["relic.data.io.reader", "relic.data.io.writer"],
            "Error: data writer dependencies missing",
        ):
            return False
        from relic.data.io.reader import DataReader, ReadConfig
        from relic.data.io.writer import DataWriter, WriteConfig

        try:
            reader = DataReader(ReadConfig(format=_resolve_data_format("auto")))
            read_result = reader.read_dataset(args.input)

            write_config = WriteConfig(
                format=_resolve_data_format(getattr(args, "format", None)),
                compression=_resolve_compression(getattr(args, "compression", None)),
                compression_level=args.compression_level,
            )
            writer = DataWriter(write_config)
            writer.write_dataset(read_result.data, args.output)
            print(f"Wrote data to {args.output}")
            return True
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            print(f"Error writing data: {exc}")
            return False

    def _run_io_convert(self, args) -> bool:
        if not _require_modules(
            ["relic.data.io.reader", "relic.data.io.writer"],
            "Error: data conversion dependencies missing",
        ):
            return False
        from relic.data.io.reader import DataReader, ReadConfig
        from relic.data.io.writer import DataWriter, WriteConfig

        try:
            reader = DataReader(ReadConfig(format=_resolve_data_format("auto")))
            read_result = reader.read_dataset(args.input)

            write_config = WriteConfig(
                format=_resolve_data_format(getattr(args, "target_format", None)),
                compression=_resolve_compression(getattr(args, "compression", None)),
            )
            writer = DataWriter(write_config)
            writer.write_dataset(read_result.data, args.output)
            print(f"Converted {args.input} to {args.output}")
            return True
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            print(f"Error converting data: {exc}")
            return False

    def _run_io_check(self, args) -> bool:
        from relic.data.io.healthcheck import io_healthcheck

        try:
            detailed = getattr(args, "detailed", False)
            report = io_healthcheck(detailed=detailed)
            print(report)
            return True
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            print(f"Error running I/O healthcheck: {exc}")
            return False

    def _run_quality(self, args, config: Dict[str, Any]) -> bool:
        if not hasattr(args, "quality_command") or not args.quality_command:
            print("Error: quality command requires a subcommand")
            print("Available commands: validate, clean")
            return False

        if args.quality_command == "validate":
            return self._run_quality_validate(args)
        if args.quality_command == "clean":
            return self._run_quality_clean(args)

        logger.error("Unknown quality command: %s", args.quality_command)
        return False

    def _run_quality_validate(self, args) -> bool:
        if not _require_modules(
            ["relic.data.io.reader"],
            "Error: quality validation dependencies missing",
        ):
            return False
        from relic.data.io.reader import DataReader, ReadConfig

        try:
            read_config = ReadConfig(
                format=_resolve_data_format("auto"),
                cf_validate=getattr(
                    args,
                    "cf_variable_metadata_check",
                    False,
                ),
            )
            reader = DataReader(read_config)
            result = reader.read_dataset(args.input)

            if result.warnings:
                print("Validation warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
            else:
                print("Validation completed without warnings")

            if getattr(args, "output", None):
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"warnings": result.warnings}
                with output_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                print(f"Saved report to {args.output}")

            return True
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            print(f"Error validating data: {exc}")
            return False

    def _run_quality_clean(self, args) -> bool:
        print("Data cleaning is not implemented yet")
        return True

    def _run_stream(self, args, config: Dict[str, Any]) -> bool:
        if not hasattr(args, "stream_command") or not args.stream_command:
            print("Error: stream command requires a subcommand")
            print("Available commands: run, check")
            return False

        if args.stream_command == "run":
            return self._run_stream_run(args)
        if args.stream_command == "check":
            return self._run_stream_check(args)

        logger.error("Unknown stream command: %s", args.stream_command)
        return False

    def _run_stream_run(self, args) -> bool:
        # No streaming pipeline is wired to the CLI: report it as a failure
        # (non-zero exit) instead of a successful no-op.
        print("Error: data stream run is not implemented in the CLI")
        return False

    def _run_stream_check(self, args) -> bool:
        from relic.data.streaming.healthcheck import streaming_healthcheck_sync

        try:
            detailed = getattr(args, "detailed", False)
            result = streaming_healthcheck_sync(detailed=detailed)
            print(result)
            return True
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            print(f"Error running streaming healthcheck: {exc}")
            return False
