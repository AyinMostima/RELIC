"""
Data Streaming Module (Experimental)

Provides real-time data streaming capabilities for RELIC data processing.
This module implements asynchronous data streaming with windowed aggregation,
backpressure handling, and rolling output to various formats.

EXPERIMENTAL FEATURE - Controlled by feature flag RELIC_EXPERIMENTAL_STREAMING

This module enables real-time processing of scientific data streams including:
- File-based streaming (directory watching, new file detection)
- Network streaming (Socket, Kafka placeholders)
- Windowed temporal aggregation
- Rolling output to NetCDF/Zarr
- Quality control and gap filling

Architecture:
StreamSource ? StreamProcessor ? StreamSink
    ?             ?              ?
File Watch    Windowing      Rolling Write
Socket        Aggregation    Compression
Kafka         Quality QC     Validation

Mathematical Framework:
Stream processing maintains temporal consistency through:
- Windowed aggregation: W(t) = A(data[t-w:t], method)
- Temporal alignment: T_align = sync(T_stream, T_target, tolerance)
- Quality control: Q(data, window) = valid_fraction(data, thresholds)

References:
- Streaming data processing patterns
- Apache Kafka streaming concepts
- Real-time scientific data processing
- Backpressure and flow control
"""

import os
import asyncio
import importlib
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, AsyncIterator, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
from xarray.coding.times import encode_cf_datetime

from ...utils.imports import module_available

# Check feature flag
STREAMING_ENABLED = os.getenv('RELIC_EXPERIMENTAL_STREAMING', '0') == '1'

# Optional streaming dependencies
STREAMING_DEPS = {
    "watchdog": module_available("watchdog"),
    "aiofiles": module_available("aiofiles"),
    "zarr": module_available("zarr"),
    "kafka": module_available("kafka"),
}

if STREAMING_DEPS["watchdog"]:
    Observer = importlib.import_module("watchdog.observers").Observer
    FileSystemEventHandler = importlib.import_module("watchdog.events").FileSystemEventHandler
else:
    Observer = None

    class FileSystemEventHandler:  # type: ignore[no-redef]
        """Placeholder base class when watchdog is not installed."""


# Import existing modules
from ..io.adapter import DataIO
from ..standards.cf_conventions import CFConventions
from ...coupling.scales.temporal.core import TimeCoordinate
from ...coupling.scales.temporal.quality_control import QualityController

logger = logging.getLogger(__name__)


class StreamSourceType(Enum):
    """Types of streaming data sources."""
    DIRECTORY = "directory"                  # Directory watching for new files
    SOCKET = "socket"                        # TCP/UDP socket streaming
    KAFKA = "kafka"                          # Apache Kafka consumer
    HTTP = "http"                            # HTTP streaming endpoint
    MEMORY = "memory"                        # In-memory data queue


class StreamProcessingMode(Enum):
    """Stream processing modes."""
    REALTIME = "realtime"                    # Process immediately
    BATCH = "batch"                          # Batch processing with windows
    HYBRID = "hybrid"                        # Adaptive processing


# Window reductions over the time dimension; the key is also the CF
# cell_methods token written to each variable ("time: <key>").
_WINDOW_REDUCTIONS = {
    "mean": lambda data: data.mean(dim="time", keep_attrs=True),
    "sum": lambda data: data.sum(dim="time", keep_attrs=True),
    "min": lambda data: data.min(dim="time", keep_attrs=True),
    "max": lambda data: data.max(dim="time", keep_attrs=True),
}


@dataclass
class StreamWindow:
    """
    Configuration for streaming time windows.

    Defines temporal windowing strategy for aggregation and processing
    of streaming scientific data. ``aggregation_method`` must be one of
    ``mean``, ``sum``, ``min``, ``max``. Coverage and gap limits are not
    window fields: quality control reads them from the QualityConfig
    parameters (coupling.scales.temporal.coverage_threshold and
    coupling.scales.temporal.quality_control.*).
    """
    size: timedelta                          # Window size
    step: Optional[timedelta] = None         # Step size (None = size)
    aggregation_method: str = "mean"         # Reduction over the window
    quality_checks: bool = True              # Enable quality control

    def __post_init__(self) -> None:
        if self.aggregation_method not in _WINDOW_REDUCTIONS:
            raise ValueError(
                "aggregation_method must be one of "
                f"{sorted(_WINDOW_REDUCTIONS)}; received {self.aggregation_method!r}"
            )


@dataclass
class StreamConfig:
    """
    Configuration for data streaming operations.
    
    Controls source monitoring, processing windows, output format,
    and performance parameters for real-time data streaming.
    """
    # Source configuration
    source_type: StreamSourceType = StreamSourceType.DIRECTORY
    source_path: Optional[Path] = None       # Source path/URL
    file_pattern: str = "*.nc"               # File pattern for directory watching
    poll_interval: float = 5.0               # Polling interval in seconds
    
    # Processing configuration
    processing_mode: StreamProcessingMode = StreamProcessingMode.BATCH
    window: StreamWindow = field(default_factory=lambda: StreamWindow(timedelta(hours=1)))
    variables: Optional[List[str]] = None    # Variables to process
    
    # Output configuration
    output_path: Optional[Path] = None       # Output directory/file
    output_format: str = "zarr"              # Output format (zarr, netcdf)
    rolling_files: bool = True               # Create rolling output files
    file_duration: timedelta = timedelta(days=1)  # Duration per output file
    
    # Performance and reliability
    buffer_size: int = 1000                  # Buffer size for backpressure
    max_concurrent: int = 4                  # Max concurrent processing
    error_tolerance: float = 0.1             # Fraction of errors to tolerate
    retry_attempts: int = 3                  # Retry attempts for failures


class StreamSource:
    """
    Abstract base for streaming data sources.
    
    Provides asynchronous data streaming from various sources including
    file systems, network endpoints, and message queues with automatic
    error handling and reconnection.
    """
    
    def __init__(self, config: StreamConfig):
        """
        Initialize stream source.
        
        Parameters:
            config: Stream configuration
        """
        self.config = config
        self._running = False
        self._buffer = deque(maxlen=config.buffer_size)
    
    async def start(self) -> None:
        """Start the streaming source."""
        if not STREAMING_ENABLED:
            raise RuntimeError("Streaming not enabled. Set RELIC_EXPERIMENTAL_STREAMING=1")
        
        self._running = True
    
    async def stop(self) -> None:
        """Stop the streaming source."""
        self._running = False
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream data items asynchronously.
        
        Yields:
            Dictionary containing data and metadata
        """
        raise RuntimeError(
            f"{self.__class__.__name__}.stream() must be implemented by a concrete source."
        )
    
    def is_running(self) -> bool:
        """Check if source is running."""
        return self._running


class DirectoryStreamSource(StreamSource):
    """
    Directory-based streaming source.
    
    Monitors directory for new files and streams their content.
    Implements file watching with configurable polling intervals.
    """
    
    def __init__(self, config: StreamConfig):
        """Initialize directory stream source."""
        super().__init__(config)
        
        if not config.source_path:
            raise ValueError("source_path required for directory streaming")
        
        self.directory = Path(config.source_path)
        self._processed_files = set()
        self._file_handler = None
        self._observer = None
        
        # Initialize data I/O
        self._io = DataIO()
    
    async def start(self) -> None:
        """Start directory monitoring."""
        await super().start()
        
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")
        
        # Set up file system monitoring if watchdog available
        if STREAMING_DEPS['watchdog']:
            self._setup_file_watcher()
        else:
            raise RuntimeError(
                "Directory streaming requires watchdog. "
                "Install watchdog or disable directory streaming."
            )
    
    async def stop(self) -> None:
        """Stop directory monitoring."""
        await super().stop()
        
        if self._observer:
            self._observer.stop()
            self._observer.join()
    
    def _setup_file_watcher(self) -> None:
        """Set up file system watcher."""
        class NewFileHandler(FileSystemEventHandler):
            def __init__(self, source):
                self.source = source
            
            def on_created(self, event):
                if not event.is_directory:
                    file_path = Path(event.src_path)
                    if file_path.match(self.source.config.file_pattern):
                        asyncio.create_task(self.source._process_new_file(file_path))
        
        self._file_handler = NewFileHandler(self)
        self._observer = Observer()
        self._observer.schedule(self._file_handler, str(self.directory), recursive=False)
        self._observer.start()
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream data from directory files."""
        while self._running:
            # Check for new files
            await self._scan_for_new_files()
            
            # Yield buffered data
            while self._buffer and self._running:
                yield self._buffer.popleft()
            
            # Wait before next scan
            await asyncio.sleep(self.config.poll_interval)
    
    async def _scan_for_new_files(self) -> None:
        """Scan directory for new files."""
        try:
            pattern = self.config.file_pattern
            
            for file_path in self.directory.glob(pattern):
                if file_path not in self._processed_files:
                    await self._process_new_file(file_path)
        
        except (OSError, RuntimeError, ValueError) as e:
            warnings.warn(f"Error scanning directory: {e}")
    
    async def _process_new_file(self, file_path: Path) -> None:
        """Process newly detected file."""
        # Read file data
        data = self._io.read(file_path)

        # Create stream item
        stream_item = {
            'data': data,
            'metadata': {
                'source_file': str(file_path),
                'timestamp': datetime.now(),
                'file_size': file_path.stat().st_size
            }
        }

        # Add to buffer
        if len(self._buffer) < self.config.buffer_size:
            self._buffer.append(stream_item)
        else:
            warnings.warn(f"Buffer full, dropping data from {file_path}")

        # Mark as processed
        self._processed_files.add(file_path)



class SocketStreamSource(StreamSource):
    """
    Socket-based streaming source.

    Reads newline-delimited records from a TCP socket endpoint.
    Message payloads may be JSON or plain text.
    """
    
    def __init__(self, config: StreamConfig):
        """Initialize socket stream source."""
        super().__init__(config)
        endpoint = str(config.source_path).strip() if config.source_path else "127.0.0.1:9000"
        if ":" not in endpoint:
            raise ValueError(
                "Socket source_path must be in '<host>:<port>' format, "
                f"got '{endpoint}'."
            )
        host, port_text = endpoint.rsplit(":", 1)
        host = host.strip()
        if not host:
            raise ValueError("Socket host cannot be empty.")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError(f"Invalid socket port: '{port_text}'.") from exc
        if port <= 0 or port > 65535:
            raise ValueError(f"Socket port out of range: {port}")
        self._host = host
        self._port = port
        self._encoding = "utf-8"
        self._delimiter = b"\n"
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield items read from the socket endpoint."""
        if not self._running:
            await self.start()
        while self._running:
            if self._reader is None:
                await asyncio.sleep(self.config.poll_interval)
                continue
            try:
                raw = await asyncio.wait_for(
                    self._reader.readuntil(self._delimiter),
                    timeout=max(1.0, self.config.poll_interval * 2.0),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.IncompleteReadError:
                warnings.warn(
                    "Socket stream disconnected by peer; attempting reconnect."
                )
                await self._reconnect()
                continue
            except (ConnectionError, OSError) as exc:
                warnings.warn(f"Socket read failed: {exc}")
                await self._reconnect()
                continue

            line = raw.decode(self._encoding, errors="ignore").strip()
            if not line:
                continue
            yield self._parse_message(line)

    async def start(self) -> None:
        """Start socket streaming by opening a TCP connection."""
        await super().start()
        await self._connect()

    async def stop(self) -> None:
        """Stop socket streaming and close active connection."""
        await super().stop()
        await self._close_stream()

    async def _connect(self) -> None:
        """Open the socket connection."""
        try:
            self._reader, self._writer = await asyncio.open_connection(
                host=self._host,
                port=self._port,
            )
        except (ConnectionError, OSError) as exc:
            self._reader = None
            self._writer = None
            warnings.warn(
                f"Socket connection to {self._host}:{self._port} failed: {exc}"
            )

    async def _reconnect(self) -> None:
        """Reconnect after connection loss."""
        await self._close_stream()
        if not self._running:
            return
        await asyncio.sleep(self.config.poll_interval)
        await self._connect()

    async def _close_stream(self) -> None:
        """Close socket stream handles safely."""
        writer = self._writer
        self._reader = None
        self._writer = None
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                warnings.warn("Socket writer close completed with transport error.")

    def _parse_message(self, line: str) -> Dict[str, Any]:
        """Parse socket message as JSON when possible."""
        timestamp = datetime.now()
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = {"data": line, "metadata": {"format": "text"}}

        if isinstance(payload, dict):
            data = payload.get("data", payload)
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": metadata}
        else:
            data = payload
            metadata = {}
        metadata.setdefault("timestamp", timestamp)
        metadata.setdefault("source", f"socket://{self._host}:{self._port}")
        metadata.setdefault("transport", "tcp")
        return {"data": data, "metadata": metadata}


class KafkaStreamSource(StreamSource):
    """
    Kafka-based streaming source.

    Reads records from a Kafka topic and yields normalized stream items.
    """
    
    def __init__(self, config: StreamConfig):
        """Initialize Kafka stream source."""
        super().__init__(config)
        if not STREAMING_DEPS['kafka']:
            raise RuntimeError(
                "KafkaStreamSource requires kafka-python. "
                "Install kafka-python to enable Kafka streaming."
            )
        topic = str(config.source_path).strip() if config.source_path else ""
        if not topic:
            raise ValueError(
                "Kafka source_path must be set to the Kafka topic name."
            )
        self._topic = topic
        self._consumer = None
        self._consumer_iter = None
    
    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield items consumed from Kafka."""
        if not self._running:
            await self.start()
        while self._running:
            if self._consumer_iter is None:
                await asyncio.sleep(self.config.poll_interval)
                continue
            try:
                message = await asyncio.to_thread(next, self._consumer_iter)
            except StopIteration:
                await asyncio.sleep(self.config.poll_interval)
                continue
            except (RuntimeError, OSError, ValueError) as exc:
                warnings.warn(f"Kafka consumer poll failed: {exc}")
                await asyncio.sleep(self.config.poll_interval)
                continue

            yield self._normalize_kafka_message(message)

    async def start(self) -> None:
        """Create Kafka consumer and begin polling."""
        await super().start()
        kafka = importlib.import_module("kafka")
        bootstrap_servers = str(
            getattr(self.config, "bootstrap_servers", "localhost:9092")
        )
        consumer_group = str(getattr(self.config, "consumer_group", "relic-stream"))
        try:
            self._consumer = kafka.KafkaConsumer(
                self._topic,
                bootstrap_servers=bootstrap_servers,
                group_id=consumer_group,
                auto_offset_reset=str(
                    getattr(self.config, "auto_offset_reset", "latest")
                ),
                enable_auto_commit=True,
                consumer_timeout_ms=int(self.config.poll_interval * 1000.0),
            )
            self._consumer_iter = iter(self._consumer)
        except (RuntimeError, OSError, ValueError) as exc:
            self._consumer = None
            self._consumer_iter = None
            warnings.warn(f"Kafka consumer init failed: {exc}")

    async def stop(self) -> None:
        """Stop Kafka consumer."""
        await super().stop()
        if self._consumer is not None:
            try:
                self._consumer.close()
            except (RuntimeError, OSError, ValueError):
                warnings.warn("Kafka consumer close completed with transport error.")
        self._consumer = None
        self._consumer_iter = None

    def _normalize_kafka_message(self, message: Any) -> Dict[str, Any]:
        """Normalize Kafka message object into stream item schema."""
        value = getattr(message, "value", None)
        payload: Any = value
        if isinstance(value, (bytes, bytearray)):
            text = bytes(value).decode("utf-8", errors="ignore").strip()
            try:
                payload = json.loads(text) if text else None
            except json.JSONDecodeError:
                payload = text
        metadata = {
            "timestamp": datetime.now(),
            "topic": getattr(message, "topic", self._topic),
            "partition": getattr(message, "partition", None),
            "offset": getattr(message, "offset", None),
            "source": f"kafka://{getattr(message, 'topic', self._topic)}",
            "transport": "kafka",
        }
        if isinstance(payload, dict) and "data" in payload:
            data = payload.get("data")
            payload_meta = payload.get("metadata")
            if isinstance(payload_meta, dict):
                metadata.update(payload_meta)
        else:
            data = payload
        return {"data": data, "metadata": metadata}


class StreamProcessor:
    """
    Stream data processor with windowed aggregation.
    
    Processes streaming data through temporal windows with aggregation,
    quality control, and gap filling capabilities integrated with
    existing RELIC temporal processing components.
    """
    
    def __init__(self, config: StreamConfig):
        """
        Initialize stream processor.
        
        Parameters:
            config: Stream configuration
        """
        self.config = config
        
        # Initialize components
        self._quality_controller = QualityController()
        self._cf_conventions = CFConventions()
        
        # Window management
        self._current_window = deque()
        self._window_start = None
    
    async def process_stream(self, 
                           source: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """
        Process data stream through windowed aggregation.
        
        Parameters:
            source: Source data stream
            
        Yields:
            Processed data windows
        """
        async for item in source:
            # Add item to current window
            await self._add_to_window(item)
            
            # Check if window is complete
            if self._is_window_complete():
                processed_window = await self._process_window()
                if processed_window:
                    yield processed_window
                
                # Start new window
                self._advance_window()
    
    async def _add_to_window(self, item: Dict[str, Any]) -> None:
        """Add data item to current processing window."""
        if self._window_start is None:
            self._window_start = item['metadata']['timestamp']
        
        self._current_window.append(item)
    
    def _is_window_complete(self) -> bool:
        """Check if current window is complete."""
        if not self._current_window or self._window_start is None:
            return False
        
        # Check time-based completion
        latest_time = self._current_window[-1]['metadata']['timestamp']
        window_duration = latest_time - self._window_start
        
        return window_duration >= self.config.window.size
    
    async def _process_window(self) -> Optional[Dict[str, Any]]:
        """Process complete window and return aggregated result."""
        if not self._current_window:
            return None

        # Errors in combination, aggregation, or quality control propagate to
        # the stream consumer; a failed window is not dropped with a warning.
        datasets = []
        for item in self._current_window:
            if item['data'] is not None:
                datasets.append(item['data'])

        if not datasets:
            warnings.warn("No valid data in window")
            return None

        # Combine datasets if multiple
        if len(datasets) == 1:
            combined_data = datasets[0]
        else:
            # Concatenate along time dimension
            combined_data = xr.concat(datasets, dim='time')

        # Apply temporal aggregation
        aggregated_data = await self._aggregate_window_data(combined_data)

        # Quality control assesses the window time series, not the reduced value
        qc_data = await self._apply_quality_control(combined_data, aggregated_data)

        return {
            'data': qc_data,
            'metadata': {
                'window_start': self._window_start,
                'window_end': self._current_window[-1]['metadata']['timestamp'],
                'window_size': len(self._current_window),
                'processing_time': datetime.now()
            }
        }
    
    async def _aggregate_window_data(self, data: xr.Dataset) -> xr.Dataset:
        """Apply temporal aggregation to window data."""
        if 'time' not in data.dims:
            return data
        
        # Reduce the completed xarray window without inferring an undeclared
        # sampling frequency. StreamWindow validated the method, so the lookup
        # has no fallback.
        method = self.config.window.aggregation_method
        aggregated = _WINDOW_REDUCTIONS[method](data)

        # Add window time coordinate
        window_center = self._window_start + (self.config.window.size / 2)
        aggregated = aggregated.assign_coords(time=window_center)
        
        # Update CF metadata for aggregation
        for var_name in aggregated.data_vars:
            attrs = aggregated[var_name].attrs.copy()
            attrs['cell_methods'] = f"time: {method}"
            aggregated[var_name].attrs = attrs
        
        return aggregated
    
    @staticmethod
    def _window_time_coordinate(time: xr.DataArray) -> TimeCoordinate:
        """Build the TimeCoordinate of a window from its xarray time axis."""
        if np.issubdtype(time.dtype, np.number):
            units = time.attrs.get("units")
            calendar = time.attrs.get("calendar")
            if not units or not calendar:
                raise ValueError(
                    "a numeric window time axis needs CF 'units' and 'calendar' "
                    "attributes for quality control"
                )
            values = np.asarray(time.values, dtype="float64")
        else:
            encoded, units, calendar = encode_cf_datetime(time.values)
            values = np.asarray(encoded, dtype="float64")
        return TimeCoordinate(
            values=torch.as_tensor(values, dtype=torch.float64),
            units=str(units),
            calendar=str(calendar),
        )

    async def _apply_quality_control(
        self,
        window_data: xr.Dataset,
        aggregated: xr.Dataset,
    ) -> xr.Dataset:
        """Flag aggregated variables whose window series fails quality control.

        Each variable's window time series is assessed with
        ``QualityController.validate_for_operation(..., "aggregation")``
        (coverage, gaps, outliers, graded against the QC parameters). A
        variable that fails gets ``quality_flags = "validation_failed"`` on the
        aggregated output; one without a time axis gets ``"not_assessed"``.
        Errors from the quality controller propagate.
        """
        if not self.config.window.quality_checks:
            return aggregated
        if 'time' not in window_data.dims:
            raise ValueError(
                "quality_checks needs window data with a 'time' dimension; set "
                "StreamWindow.quality_checks=False for records without one"
            )
        time_coord = self._window_time_coordinate(window_data['time'])

        for var_name in aggregated.data_vars:
            series = window_data[var_name]
            attrs = aggregated[var_name].attrs.copy()
            if 'time' not in series.dims:
                attrs['quality_flags'] = 'not_assessed'
            else:
                values = torch.as_tensor(
                    np.asarray(series.transpose(..., 'time').values, dtype="float64")
                )
                if not self._quality_controller.validate_for_operation(
                    values,
                    time_coord,
                    'aggregation',
                ):
                    attrs['quality_flags'] = 'validation_failed'
            aggregated[var_name].attrs = attrs

        return aggregated
    
    def _advance_window(self) -> None:
        """Advance to next processing window."""
        # Clear current window
        self._current_window.clear()
        
        # Update window start time
        if self.config.window.step:
            self._window_start += self.config.window.step
        else:
            self._window_start += self.config.window.size


class StreamSink:
    """
    Stream data sink with rolling file output.
    
    Writes processed streaming data to rolling output files with
    compression, bounded CF metadata handling, and format optimization.
    """
    
    def __init__(self, config: StreamConfig):
        """
        Initialize stream sink.
        
        Parameters:
            config: Stream configuration
        """
        self.config = config
        
        if not config.output_path:
            raise ValueError("output_path required for stream sink")
        
        self.output_dir = Path(config.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data I/O
        self._io = DataIO()
        
        # Current output file management
        self._current_file = None
        self._current_file_start = None
        self._file_counter = 0
    
    async def write_stream(self, source: AsyncIterator[Dict[str, Any]]) -> None:
        """
        Write stream data to rolling output files.
        
        Parameters:
            source: Processed data stream
        """
        async for item in source:
            await self._write_item(item)
    
    async def _write_item(self, item: Dict[str, Any]) -> None:
        """Write single data item to output."""
        # Check if new output file needed
        if self._needs_new_file(item):
            await self._rotate_output_file()

        # Write data to current file
        await self._append_to_current_file(item)

    
    def _needs_new_file(self, item: Dict[str, Any]) -> bool:
        """Check if new output file is needed."""
        if self._current_file is None:
            return True
        
        if not self.config.rolling_files:
            return False
        
        # Check time-based rotation
        item_time = item['metadata'].get('processing_time', datetime.now())
        
        if self._current_file_start is None:
            return True
        
        time_since_start = item_time - self._current_file_start
        return time_since_start >= self.config.file_duration
    
    async def _rotate_output_file(self) -> None:
        """Create new output file and close current."""
        # Close current file if exists
        if self._current_file:
            # Placeholder for future persistent file-handle lifecycle.
            logger.debug("Rotating stream sink file from %s", self._current_file)
        
        # Generate new filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_suffix = 'zarr' if self.config.output_format == 'zarr' else 'nc'
        filename = f"stream_data_{timestamp}_{self._file_counter:04d}.{file_suffix}"
        
        self._current_file = self.output_dir / filename
        self._current_file_start = datetime.now()
        self._file_counter += 1
    
    async def _append_to_current_file(self, item: Dict[str, Any]) -> None:
        """Append data item to current output file."""
        data = item['data']
        if data is None:
            return
        
        # For simplicity, write each item as separate file
        # In production, would implement proper appending/concatenation
        if self.config.output_format == 'zarr':
            if STREAMING_DEPS['zarr']:
                # Write to Zarr format
                output_path = str(self._current_file).replace('.zarr', f'_{item["metadata"]["processing_time"].strftime("%H%M%S")}.zarr')
                self._io.write(data, output_path, format='zarr', compression='zstd')
            else:
                raise RuntimeError(
                    "Zarr output requested but dependency missing. "
                    "Install zarr or use NetCDF output format."
                )
        else:
            # Write to NetCDF format
            self._io.write(data, self._current_file, format='netcdf4', compression='zstd')
        


class DataStreamManager:
    """
    Complete streaming pipeline manager.
    
    Orchestrates the entire streaming pipeline from source to sink
    with error handling, monitoring, and graceful shutdown.    
    Mathematical Expression:
        Stream Pipeline: Source ? Processor ? Sink
        Throughput: T = min(T_source, T_processor, T_sink)
        
    References:
        - Streaming data processing patterns
        - Apache Kafka streaming concepts
        - Real-time scientific data processing
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        # Store initialization parameters
        self._init_config = config or StreamConfig()
        
        # Core attributes (initialized in _do_initialize)
        self.config = None
        self.source = None
        self.processor = None
        self.sink = None
        self._running = False
        self._tasks = []
        self.initialize()

    def initialize(self) -> None:
        """Initialize StreamManager components."""
        self._do_initialize()

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure StreamManager with provided settings."""
        self._do_configure(config)

    def validate(self) -> List[str]:
        """Validate StreamManager configuration."""
        return self._do_validate()

    def cleanup(self) -> None:
        """Clean up StreamManager resources."""
        self._do_cleanup()

    def _do_initialize(self) -> None:
        """Perform StreamManager-specific initialization."""
        if not STREAMING_ENABLED:
            warnings.warn(
                "Streaming functionality is experimental. "
                "Set RELIC_EXPERIMENTAL_STREAMING=1 to enable.",
                UserWarning
            )
        
        # Initialize configuration
        self.config = self._init_config
        
        # Initialize components
        if STREAMING_ENABLED:
            self.source = self._create_source()
            self.processor = StreamProcessor(self.config)
            self.sink = StreamSink(self.config) if self.config.output_path else None
        
        # Initialize state
        self._running = False
        self._tasks = []
    
    def _do_configure(self, config: Dict[str, Any]) -> None:
        """Configure StreamManager with provided settings."""
        settings = dict(config or {})
        # Update stream config with new settings
        if "source_path" in settings:
            self.config.source_path = Path(settings["source_path"])
        
        if "output_path" in settings:
            self.config.output_path = Path(settings["output_path"])
        
        if "window_size_hours" in settings:
            self.config.window.size = timedelta(hours=settings["window_size_hours"])
        
        if "file_pattern" in settings:
            self.config.file_pattern = settings["file_pattern"]
        
        # Recreate components if needed
        if STREAMING_ENABLED and hasattr(self, 'source'):
            self.source = self._create_source()
            self.processor = StreamProcessor(self.config)
            self.sink = StreamSink(self.config) if self.config.output_path else None
    
    def _do_validate(self) -> List[str]:
        """Validate StreamManager configuration."""
        errors = []
        
        if not STREAMING_ENABLED:
            errors.append("Streaming not enabled - set RELIC_EXPERIMENTAL_STREAMING=1")
        
        if self.config.source_type == StreamSourceType.DIRECTORY and not self.config.source_path:
            errors.append("source_path required for directory streaming")
        
        if self.config.source_path and not self.config.source_path.exists():
            errors.append(f"Source path does not exist: {self.config.source_path}")
        
        if self.config.output_path and self.config.output_path.exists() and not self.config.output_path.is_dir():
            errors.append(f"Output path exists but is not a directory: {self.config.output_path}")
        
        # Check optional dependencies
        missing_deps = []
        if self.config.source_type == StreamSourceType.DIRECTORY and not STREAMING_DEPS.get('watchdog'):
            missing_deps.append("watchdog (for directory watching)")
        
        if self.config.output_format == "zarr" and not STREAMING_DEPS.get('zarr'):
            missing_deps.append("zarr (for Zarr format output)")
        
        if self.config.source_type == StreamSourceType.KAFKA and not STREAMING_DEPS.get('kafka'):
            missing_deps.append("kafka-python (for Kafka streaming)")
        
        if missing_deps:
            errors.append(f"Missing optional dependencies: {', '.join(missing_deps)}")
        
        return errors
    
    def _do_cleanup(self) -> None:
        """Clean up StreamManager resources."""
        # Stop streaming if running
        if self._running and hasattr(self, 'stop'):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup for later
                    loop.create_task(self._async_cleanup())
                else:
                    loop.run_until_complete(self._async_cleanup())
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("StreamManager async cleanup failed, using synchronous fallback: %s", exc)
                self._running = False
                if hasattr(self, '_tasks'):
                    self._tasks.clear()
    
    async def _async_cleanup(self):
        """Asynchronous cleanup helper."""
        if hasattr(self, 'stop'):
            await self.stop()
    
    def _create_source(self) -> StreamSource:
        """Create appropriate stream source."""
        if self.config.source_type == StreamSourceType.DIRECTORY:
            return DirectoryStreamSource(self.config)
        elif self.config.source_type == StreamSourceType.SOCKET:
            return SocketStreamSource(self.config)
        elif self.config.source_type == StreamSourceType.KAFKA:
            return KafkaStreamSource(self.config)
        else:
            raise ValueError(f"Unsupported source type: {self.config.source_type}")
    
    async def start(self) -> None:
        """Start streaming pipeline."""
        if not STREAMING_ENABLED:
            raise RuntimeError(
                "Streaming is experimental. Set RELIC_EXPERIMENTAL_STREAMING=1 to enable.\n"
                "Note: Requires optional dependencies: watchdog, aiofiles, zarr"
            )
        
        self._running = True
        
        # Start source
        await self.source.start()
        
        # Create processing pipeline
        raw_stream = self.source.stream()
        processed_stream = self.processor.process_stream(raw_stream)
        
        # Start sink if configured
        if self.sink:
            sink_task = asyncio.create_task(self.sink.write_stream(processed_stream))
            self._tasks.append(sink_task)
        else:
            # Just consume processed stream
            async for item in processed_stream:
                logger.info("Processed item metadata: %s", item.get("metadata", {}))
    
    async def stop(self) -> None:
        """Stop streaming pipeline gracefully."""
        self._running = False
        
        # Stop source
        await self.source.stop()
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                continue
        
        self._tasks.clear()
    
    def is_running(self) -> bool:
        """Check if streaming pipeline is running."""
        return self._running


StreamManager = DataStreamManager


# Convenience function for quick streaming setup
async def run_streaming(source_path: Union[str, Path],
                       output_path: Optional[Union[str, Path]] = None,
                       window_hours: float = 1.0,
                       file_pattern: str = "*.nc") -> None:
    """
    Run streaming pipeline with default configuration.
    
    Parameters:
        source_path: Source directory to monitor
        output_path: Output directory for processed data
        window_hours: Processing window size in hours
        file_pattern: File pattern to monitor
        
    Example:
        # Monitor directory for new NetCDF files
        await run_streaming(
            source_path="./incoming_data",
            output_path="./processed_stream", 
            window_hours=0.5
        )
    """
    if not STREAMING_ENABLED:
        logger.warning(
            "Streaming is experimental. Set RELIC_EXPERIMENTAL_STREAMING=1 to enable."
        )
        return
    
    config = StreamConfig(
        source_type=StreamSourceType.DIRECTORY,
        source_path=Path(source_path),
        file_pattern=file_pattern,
        window=StreamWindow(
            size=timedelta(hours=window_hours),
            aggregation_method="mean"
        ),
        output_path=Path(output_path) if output_path else None,
        output_format="zarr"
    )
    
    manager = StreamManager(config)
    
    try:
        logger.info("Starting streaming pipeline...")
        logger.info("  Source: %s", source_path)
        logger.info("  Output: %s", output_path)
        logger.info("  Window: %.3f hours", window_hours)
        logger.info("Press Ctrl+C to stop.")
        
        await manager.start()
        
    except KeyboardInterrupt:
        logger.info("Stopping streaming pipeline...")
        await manager.stop()


# Export main classes and functions  
__all__ = [
    # Configuration classes
    'StreamSourceType',
    'StreamProcessingMode', 
    'StreamWindow',
    'StreamConfig',
    
    # Pipeline components
    'StreamSource',
    'DirectoryStreamSource',
    'SocketStreamSource', 
    'KafkaStreamSource',
    'StreamProcessor',
    'StreamSink',
    
    # Main manager
    'DataStreamManager',
    'StreamManager',
    
    # Convenience functions
    'run_streaming',
]


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        logger.error("Usage: python stream.py <source_directory> [output_directory]")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    asyncio.run(run_streaming(source_dir, output_dir))
