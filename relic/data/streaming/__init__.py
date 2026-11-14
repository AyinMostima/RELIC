"""
Data Streaming Module (Experimental)

Real-time data streaming capabilities for RELIC with asynchronous processing,
windowed aggregation, and rolling output to various formats.

EXPERIMENTAL FEATURE - Controlled by environment variable RELIC_EXPERIMENTAL_STREAMING

This module provides streaming data processing capabilities including:
- File-based streaming with directory monitoring
- Asynchronous data processing pipelines
- Windowed temporal aggregation
- Real-time quality control and validation
- Rolling output to NetCDF/Zarr formats
- Backpressure handling and error recovery

Core Components:
- StreamManager: Complete pipeline orchestration
- StreamSource: Data ingestion from various sources
- StreamProcessor: Windowed aggregation and processing
- StreamSink: Output to rolling files
- Health checking and diagnostics

Key Features:
- Asynchronous processing with asyncio
- Configurable time windows and aggregation
- Integration with RELIC data I/O layer
- CF-1.8 compliant output
- Comprehensive error handling
- Performance monitoring and optimization

Usage Requirements:
- Set environment variable RELIC_EXPERIMENTAL_STREAMING=1 to enable
- Optional dependencies: watchdog, aiofiles, zarr, kafka-python

Examples:
    # Enable streaming functionality
    import os
    os.environ['RELIC_EXPERIMENTAL_STREAMING'] = '1'
    
    # Basic streaming setup
    from relic.data.streaming import run_streaming
    import asyncio
    
    asyncio.run(run_streaming(
        source_path="./incoming_data",
        output_path="./processed_stream",
        window_hours=0.5
    ))
    
    # Advanced streaming configuration
    from relic.data.streaming import StreamManager, StreamConfig
    from relic.data.streaming import StreamWindow, StreamSourceType
    from datetime import timedelta
    
    config = StreamConfig(
        source_type=StreamSourceType.DIRECTORY,
        source_path="./data_input",
        window=StreamWindow(
            size=timedelta(minutes=30),
            aggregation_method="mean"
        ),
        output_path="./data_output",
        output_format="zarr"
    )
    
    manager = StreamManager(config)
    asyncio.run(manager.start())
    
    # Health check
    from relic.data.streaming import streaming_healthcheck
    
    status = asyncio.run(streaming_healthcheck())
    print(f"Streaming Status: {status['overall']}")

Installation:
    # Basic streaming (directory monitoring)
    pip install watchdog
    
    # Full streaming capabilities  
    pip install watchdog aiofiles zarr kafka-python

References:
    - Async programming with asyncio: https://docs.python.org/3/library/asyncio.html
    - Real-time data processing patterns
    - Apache Kafka streaming concepts
    - Scientific data streaming best practices
"""

import os

# Check if streaming is enabled
STREAMING_ENABLED = os.getenv('RELIC_EXPERIMENTAL_STREAMING', '0') == '1'

if STREAMING_ENABLED:
    # Import streaming components when enabled
    from .stream import (
        DataStreamManager,
        StreamManager,
        StreamSource,
        StreamProcessor, 
        StreamSink,
        DirectoryStreamSource,
        SocketStreamSource,
        KafkaStreamSource,
        StreamConfig,
        StreamWindow,
        StreamSourceType,
        StreamProcessingMode,
        run_streaming
    )
    
    from .healthcheck import (
        StreamingHealthChecker,
        StreamingPerformanceMetrics,
        streaming_healthcheck
    )

    _STREAMING_REEXPORTS = (
        DataStreamManager,
        StreamManager,
        StreamSource,
        StreamProcessor,
        StreamSink,
        DirectoryStreamSource,
        SocketStreamSource,
        KafkaStreamSource,
        StreamConfig,
        StreamWindow,
        StreamSourceType,
        StreamProcessingMode,
        run_streaming,
        StreamingHealthChecker,
        StreamingPerformanceMetrics,
        streaming_healthcheck,
    )
    del _STREAMING_REEXPORTS
    
    # Expose all streaming functionality
    __all__ = [
        # Core streaming classes
        'DataStreamManager',
        'StreamManager',
        'StreamSource',
        'StreamProcessor',
        'StreamSink',
        
        # Source implementations
        'DirectoryStreamSource',
        'SocketStreamSource', 
        'KafkaStreamSource',
        
        # Configuration classes
        'StreamConfig',
        'StreamWindow',
        
        # Enums
        'StreamSourceType',
        'StreamProcessingMode',
        
        # Convenience functions
        'run_streaming',
        
        # Health checking
        'StreamingHealthChecker',
        'StreamingPerformanceMetrics', 
        'streaming_healthcheck',
        
        # Status
        'STREAMING_ENABLED'
    ]

else:
    # Streaming disabled - provide minimal interface
    # Placeholder functions that explain how to enable
    def _streaming_disabled_message():
        """Return message explaining how to enable streaming."""
        return (
            "Streaming is disabled. Enable with:\n"
            "1. Set environment variable: RELIC_EXPERIMENTAL_STREAMING=1\n"
            "2. Install dependencies: pip install watchdog aiofiles zarr\n"
            "3. Restart Python session"
        )
    
    class StreamingDisabledError(RuntimeError):
        """Error raised when streaming functionality is accessed while disabled."""
        pass
    
    # Create placeholder classes that raise helpful errors
    class _StreamingPlaceholder:
        def __init__(self, *args, **kwargs):
            raise StreamingDisabledError(_streaming_disabled_message())
    
    # Assign placeholders to expected names
    StreamManager = _StreamingPlaceholder
    DataStreamManager = _StreamingPlaceholder
    StreamConfig = _StreamingPlaceholder
    StreamWindow = _StreamingPlaceholder
    
    async def run_streaming(*args, **kwargs):
        """Placeholder function for disabled streaming."""
        raise StreamingDisabledError(_streaming_disabled_message())
    
    async def streaming_healthcheck(*args, **kwargs):
        """Health check for disabled streaming."""
        return {
            'overall': 'disabled',
            'message': _streaming_disabled_message(),
            'streaming_enabled': False
        }
    
    __all__ = [
        'StreamManager',  # Placeholder that raises error
        'DataStreamManager',  # Placeholder that raises error
        'StreamConfig',   # Placeholder that raises error  
        'StreamWindow',   # Placeholder that raises error
        'run_streaming',  # Raises error with instructions
        'streaming_healthcheck',  # Returns disabled status
        'STREAMING_ENABLED'  # False
    ]

# Module metadata
__version__ = '1.0.0'
__author__ = 'Yuchong Long'
__description__ = 'Experimental streaming data processing for RELIC'
__experimental__ = True
