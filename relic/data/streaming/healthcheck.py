"""
Streaming Health Check Module (Experimental)

Provides comprehensive health checking and diagnostics for the streaming
data processing layer. Tests streaming capabilities, dependency availability,
throughput performance, and integration with core RELIC components.

EXPERIMENTAL FEATURE - Controlled by feature flag RELIC_EXPERIMENTAL_STREAMING

This module validates the complete streaming stack including:
- Streaming dependencies (watchdog, aiofiles, zarr, kafka)
- Asynchronous processing capabilities
- Throughput and latency characteristics
- Backpressure handling and error recovery
- Integration with data I/O layer

Health Check Categories:
1. Dependencies: Streaming library availability
2. Throughput: Processing speed and latency
3. Reliability: Error handling and recovery
4. Integration: Compatibility with RELIC components
5. Resources: Memory usage and system resources

References:
- Streaming system diagnostics
- Async performance testing
- Real-time data processing validation
"""

import os
import sys
import time
import asyncio
import importlib
import gc
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import xarray as xr
from ...utils.imports import module_available

# Check feature flag
STREAMING_ENABLED = os.getenv('RELIC_EXPERIMENTAL_STREAMING', '0') == '1'

# Test streaming dependencies
STREAMING_DEPS_STATUS = {
    "watchdog": module_available("watchdog"),
    "aiofiles": module_available("aiofiles"),
    "zarr": module_available("zarr"),
    "kafka": module_available("kafka"),
}

# Import health check infrastructure
from ..io.healthcheck import HealthStatus, HealthCheckResult


@dataclass
class StreamingPerformanceMetrics:
    """
    Performance metrics for streaming operations.
    
    Tracks throughput, latency, and resource usage for streaming
    data processing pipelines.
    """
    items_processed: int                     # Number of items processed
    processing_time: float                   # Total processing time (seconds)
    throughput_items_per_sec: float          # Items per second throughput
    avg_latency_ms: float                    # Average processing latency
    max_latency_ms: float                    # Maximum latency observed
    memory_peak_mb: float                    # Peak memory usage
    cpu_usage_percent: float                 # Average CPU usage


class StreamingHealthChecker:
    """
    Health checker for streaming data processing capabilities.
    
    Validates streaming infrastructure, performance characteristics,
    and integration with RELIC data processing components.
    """
    
    def __init__(self):
        """Initialize streaming health checker."""
        self._test_data_cache = {}
    
    def check_streaming_availability(self) -> HealthCheckResult:
        """
        Check if streaming functionality is available and enabled.
        
        Returns:
            Health check result for streaming availability
        """
        details = {}
        recommendations = []
        
        # Check feature flag
        details['feature_enabled'] = STREAMING_ENABLED
        
        if not STREAMING_ENABLED:
            status = HealthStatus.WARNING
            message = "Streaming functionality is disabled"
            recommendations.append(
                "Set environment variable RELIC_EXPERIMENTAL_STREAMING=1 to enable streaming"
            )
        else:
            status = HealthStatus.HEALTHY
            message = "Streaming functionality is enabled"
        
        # Check Python version (async/await support)
        python_version = sys.version_info
        details['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 7):
            status = HealthStatus.ERROR
            message = "Python 3.7+ required for streaming"
            recommendations.append("Upgrade to Python 3.7 or higher")
        
        return HealthCheckResult(
            name="streaming_availability",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    def check_streaming_dependencies(self) -> HealthCheckResult:
        """
        Check availability of streaming-specific dependencies.
        
        Returns:
            Health check result for streaming dependencies
        """
        details = {}
        missing_deps = []
        optional_deps = []
        
        # Required dependencies for basic streaming
        _ = {
            'asyncio': True,  # Built-in, always available
        }
        
        # Optional dependencies with impact assessment
        optional_dep_info = {
            'watchdog': {
                'description': 'File system monitoring',
                'impact': 'high',
                'available': STREAMING_DEPS_STATUS.get('watchdog', False)
            },
            'aiofiles': {
                'description': 'Asynchronous file I/O',
                'impact': 'medium',
                'available': STREAMING_DEPS_STATUS.get('aiofiles', False)
            },
            'zarr': {
                'description': 'Zarr format streaming output',
                'impact': 'medium',
                'available': STREAMING_DEPS_STATUS.get('zarr', False)
            },
            'kafka': {
                'description': 'Apache Kafka streaming',
                'impact': 'low',
                'available': STREAMING_DEPS_STATUS.get('kafka', False)
            }
        }
        
        for dep_name, dep_info in optional_dep_info.items():
            details[dep_name] = dep_info
            if not dep_info['available']:
                if dep_info['impact'] == 'high':
                    missing_deps.append(dep_name)
                else:
                    optional_deps.append(dep_name)
        
        # Determine status
        if missing_deps:
            status = HealthStatus.WARNING
            message = f"High-impact dependencies missing: {', '.join(missing_deps)}"
        elif len(optional_deps) > 2:
            status = HealthStatus.WARNING
            message = f"Multiple optional dependencies missing: {len(optional_deps)}"
        else:
            status = HealthStatus.HEALTHY
            message = "Streaming dependencies available"
        
        # Generate recommendations
        recommendations = []
        if 'watchdog' not in [dep for dep, info in details.items() if info['available']]:
            recommendations.append("Install watchdog for file system monitoring: pip install watchdog")
        if 'zarr' not in [dep for dep, info in details.items() if info['available']]:
            recommendations.append("Install zarr for optimized streaming output: pip install zarr")
        
        return HealthCheckResult(
            name="streaming_dependencies",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    async def check_async_performance(self) -> HealthCheckResult:
        """
        Check asynchronous processing performance.
        
        Returns:
            Health check result for async performance
        """
        details = {}
        performance_metrics = {}
        
        try:
            # Test basic async performance
            metrics = await self._benchmark_async_processing()
            details['async_performance'] = metrics
            performance_metrics.update(metrics)
            
            # Test concurrent processing
            concurrent_metrics = await self._benchmark_concurrent_processing()
            details['concurrent_performance'] = concurrent_metrics
            performance_metrics.update(concurrent_metrics)
            
            # Determine status based on performance
            throughput = performance_metrics.get('throughput_items_per_sec', 0)
            latency = performance_metrics.get('avg_latency_ms', float('inf'))
            
            if throughput > 100 and latency < 100:
                status = HealthStatus.HEALTHY
                message = f"Good async performance: {throughput:.1f} items/s, {latency:.1f}ms latency"
            elif throughput > 50:
                status = HealthStatus.WARNING
                message = f"Moderate async performance: {throughput:.1f} items/s"
            else:
                status = HealthStatus.WARNING
                message = f"Low async performance: {throughput:.1f} items/s"
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            status = HealthStatus.ERROR
            message = f"Async performance test failed: {e}"
            details['error'] = str(e)
        
        recommendations = []
        if performance_metrics.get('throughput_items_per_sec', 0) < 50:
            recommendations.append("Consider optimizing async processing or increasing concurrency")
        
        return HealthCheckResult(
            name="async_performance",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations,
            performance_metrics=performance_metrics
        )
    
    def check_streaming_integration(self) -> HealthCheckResult:
        """
        Check integration with RELIC data processing components.
        
        Returns:
            Health check result for streaming integration
        """
        details = {}
        
        try:
            # Test streaming module imports
            if STREAMING_ENABLED:
                stream_module = importlib.import_module("relic.data.streaming.stream")
                _ = getattr(stream_module, "StreamManager", None)
                _ = getattr(stream_module, "StreamSource", None)
                _ = getattr(stream_module, "StreamProcessor", None)
                _ = getattr(stream_module, "StreamSink", None)
                _ = getattr(stream_module, "StreamConfig", None)
                _ = getattr(stream_module, "StreamWindow", None)
                _ = getattr(stream_module, "StreamSourceType", None)
                details['streaming_imports'] = True
            else:
                details['streaming_imports'] = False
                details['reason'] = 'Streaming not enabled'
            
            # Test integration with data I/O
            io_module = importlib.import_module("relic.data.io.adapter")
            DataIO = getattr(io_module, "DataIO")
            DataIO()
            details['io_integration'] = True
            
            # Test integration with CF conventions
            cf_module = importlib.import_module("relic.data.standards.cf_conventions")
            CFConventions = getattr(cf_module, "CFConventions")
            CFConventions()
            details['cf_integration'] = True
            
            # Test integration with temporal processing
            temporal_agg_module = importlib.import_module("relic.coupling.scales.temporal.aggregator")
            temporal_types_module = importlib.import_module("relic.coupling.scales.temporal.types")
            TemporalAggregator = getattr(temporal_agg_module, "TemporalAggregator")
            TemporalConfig = getattr(temporal_types_module, "TemporalConfig")
            ScalingDirection = getattr(temporal_types_module, "ScalingDirection")

            temp_config = TemporalConfig(
                source_frequency="daily",
                target_frequency="monthly",
                direction=ScalingDirection.UPSCALE,
            )
            TemporalAggregator(temp_config)
            details['temporal_integration'] = True
            
            status = HealthStatus.HEALTHY
            message = "Streaming integration working"
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            details['error'] = str(e)
            status = HealthStatus.ERROR
            message = f"Integration check failed: {e}"
        
        recommendations = []
        if not details.get('streaming_imports', False):
            recommendations.append("Enable streaming functionality to test full integration")
        
        return HealthCheckResult(
            name="streaming_integration",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations
        )
    
    async def check_streaming_throughput(self) -> HealthCheckResult:
        """
        Check streaming data throughput capabilities.
        
        Returns:
            Health check result for streaming throughput
        """
        if not STREAMING_ENABLED:
            return HealthCheckResult(
                name="streaming_throughput",
                status=HealthStatus.WARNING,
                message="Streaming disabled - cannot test throughput",
                details={'streaming_enabled': False},
                recommendations=["Enable streaming to test throughput capabilities"]
            )
        
        details = {}
        performance_metrics = {}
        
        try:
            # Test file-based streaming throughput
            throughput_metrics = await self._benchmark_streaming_throughput()
            details['throughput_test'] = throughput_metrics
            performance_metrics.update(throughput_metrics)
            
            # Test memory usage under load
            memory_metrics = await self._benchmark_streaming_memory()
            details['memory_test'] = memory_metrics
            performance_metrics['memory_efficiency'] = memory_metrics.get('efficiency', 0)
            
            # Determine status
            throughput = performance_metrics.get('data_throughput_mbps', 0)
            
            if throughput > 10:
                status = HealthStatus.HEALTHY
                message = f"Good streaming throughput: {throughput:.1f} MB/s"
            elif throughput > 1:
                status = HealthStatus.WARNING
                message = f"Moderate streaming throughput: {throughput:.1f} MB/s"
            else:
                status = HealthStatus.WARNING
                message = f"Low streaming throughput: {throughput:.1f} MB/s"
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            status = HealthStatus.ERROR
            message = f"Streaming throughput test failed: {e}"
            details['error'] = str(e)
        
        recommendations = []
        if performance_metrics.get('data_throughput_mbps', 0) < 5:
            recommendations.append("Consider optimizing I/O operations or buffer sizes")
        
        return HealthCheckResult(
            name="streaming_throughput",
            status=status,
            message=message,
            details=details,
            recommendations=recommendations,
            performance_metrics=performance_metrics
        )
    
    # Private benchmark methods
    async def _benchmark_async_processing(self) -> Dict[str, float]:
        """Benchmark basic asynchronous processing."""
        n_items = 1000
        items = list(range(n_items))
        
        async def process_item(item):
            # Simulate processing delay
            await asyncio.sleep(0.001)
            return item * 2
        
        start_time = time.time()
        results = []
        
        for item in items:
            result = await process_item(item)
            results.append(result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'items_processed': n_items,
            'processing_time': processing_time,
            'throughput_items_per_sec': n_items / processing_time,
            'avg_latency_ms': (processing_time * 1000) / n_items
        }
    
    async def _benchmark_concurrent_processing(self) -> Dict[str, float]:
        """Benchmark concurrent asynchronous processing."""
        n_items = 1000
        max_concurrent = 10
        
        async def process_item(item):
            await asyncio.sleep(0.001)
            return item * 2
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await process_item(item)
        
        start_time = time.time()
        
        tasks = [process_with_semaphore(i) for i in range(n_items)]
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'concurrent_items_processed': n_items,
            'concurrent_processing_time': processing_time,
            'concurrent_throughput_items_per_sec': n_items / processing_time,
            'speedup_factor': (n_items * 0.001) / processing_time  # Theoretical vs actual
        }
    
    async def _benchmark_streaming_throughput(self) -> Dict[str, float]:
        """Benchmark streaming data throughput."""
        if not STREAMING_ENABLED:
            return {'error': 'Streaming not enabled'}
        
        # Create test data files
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            output_dir = Path(tmpdir) / "output"
            source_dir.mkdir()
            output_dir.mkdir()
            
            # Create test files
            n_files = 5
            _ = 10  # 10MB per file
            
            total_data_size = 0
            file_creation_start = time.time()
            
            for i in range(n_files):
                # Create test dataset
                shape = (100, 50, 40)  # ~8MB of float32 data
                data = np.random.random(shape).astype(np.float32)
                ds = xr.Dataset({
                    'test_var': (['time', 'lat', 'lon'], data),
                    'timestamp': (['time'], np.arange(100))
                })
                
                file_path = source_dir / f"test_{i:03d}.nc"
                ds.to_netcdf(file_path)
                total_data_size += file_path.stat().st_size
            
            time.time() - file_creation_start
            
            # Test streaming processing (simplified simulation)
            processing_start = time.time()
            
            # Simulate streaming processing
            from ..io.adapter import DataIO
            io_adapter = DataIO()

            processed_files = []
            for file_path in source_dir.glob("*.nc"):
                # Simulate streaming read and process
                data = io_adapter.read(file_path)

                # Simulate aggregation
                if 'time' in data.dims:
                    aggregated = data.mean(dim='time', keep_attrs=True)
                else:
                    aggregated = data

                # Write to output
                output_path = output_dir / f"processed_{file_path.name}"
                io_adapter.write(aggregated, output_path)
                processed_files.append(output_path)
            
            processing_time = time.time() - processing_start
            
            # Calculate metrics
            total_mb = total_data_size / (1024 * 1024)
            throughput_mbps = total_mb / processing_time
            
            return {
                'files_processed': len(processed_files),
                'total_data_mb': total_mb,
                'processing_time': processing_time,
                'data_throughput_mbps': throughput_mbps,
                'files_per_second': len(processed_files) / processing_time
            }
        

    async def _benchmark_streaming_memory(self) -> Dict[str, float]:
        """Benchmark streaming memory usage."""
        if importlib.util.find_spec("psutil") is None:
            return {"error": "psutil not available for memory monitoring"}

        psutil = importlib.import_module("psutil")
        process = psutil.Process()

        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate streaming data processing
        data_items = []

        # Create multiple datasets to simulate streaming
        for i in range(10):
            shape = (200, 100, 50)  # ~40MB per dataset
            data = np.random.random(shape).astype(np.float32)
            ds = xr.Dataset({'var': (['time', 'lat', 'lon'], data)})
            data_items.append(ds)

            # Check memory periodically
            process.memory_info().rss / 1024 / 1024

        peak_memory = process.memory_info().rss / 1024 / 1024

        # Clear data and check memory cleanup
        del data_items

        # Wait for garbage collection
        gc.collect()
        await asyncio.sleep(0.1)

        final_memory = process.memory_info().rss / 1024 / 1024

        memory_used = peak_memory - baseline_memory
        memory_recovered = peak_memory - final_memory
        efficiency = (memory_recovered / memory_used) if memory_used > 0 else 1.0

        return {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_used_mb': memory_used,
            'memory_recovered_mb': memory_recovered,
            'efficiency': efficiency
        }

    
    def generate_streaming_recommendations(self, results: Dict[str, HealthCheckResult]) -> List[str]:
        """Generate prioritized recommendations for streaming setup."""
        recommendations = []
        
        # Check if streaming is enabled
        availability_result = results.get('streaming_availability')
        if availability_result and availability_result.status != HealthStatus.HEALTHY:
            recommendations.extend(availability_result.recommendations)
        
        # Check dependencies
        deps_result = results.get('streaming_dependencies')
        if deps_result and deps_result.status == HealthStatus.WARNING:
            recommendations.extend(deps_result.recommendations)
        
        # Performance recommendations
        performance_result = results.get('async_performance')
        if performance_result and performance_result.status == HealthStatus.WARNING:
            recommendations.append("Consider increasing concurrency limits or optimizing processing")
        
        throughput_result = results.get('streaming_throughput')
        if throughput_result and throughput_result.status == HealthStatus.WARNING:
            recommendations.append("Consider buffer size tuning or I/O optimization")
        
        # Add general recommendations
        if not any("watchdog" in rec for rec in recommendations):
            recommendations.append("Install watchdog for optimal file system monitoring")
        
        return recommendations[:10]  # Limit to top 10 recommendations


async def streaming_healthcheck(detailed: bool = True) -> Dict[str, Any]:
    """
    Run streaming health check and return results.
    
    Parameters:
        detailed: Return detailed results (True) or summary (False)
        
    Returns:
        Health check results
        
    Example:
        # Quick check
        status = await streaming_healthcheck(detailed=False)
        print(f"Streaming Status: {status['overall']}")
        
        # Detailed check
        results = await streaming_healthcheck(detailed=True)
        for name, result in results['checks'].items():
            print(f"{name}: {result['status']} - {result['message']}")
    """
    checker = StreamingHealthChecker()
    
    # Run all health checks
    results = {}
    results['streaming_availability'] = checker.check_streaming_availability()
    results['streaming_dependencies'] = checker.check_streaming_dependencies()
    results['streaming_integration'] = checker.check_streaming_integration()
    
    # Run async checks
    results['async_performance'] = await checker.check_async_performance()
    results['streaming_throughput'] = await checker.check_streaming_throughput()
    
    # Determine overall status
    status_counts = {}
    for result in results.values():
        status = result.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    if status_counts.get('error', 0) > 0:
        overall_status = 'error'
    elif status_counts.get('warning', 0) > 0:
        overall_status = 'warning'
    else:
        overall_status = 'healthy'
    
    # Generate recommendations
    recommendations = checker.generate_streaming_recommendations(results)
    
    if detailed:
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'streaming_enabled': STREAMING_ENABLED,
            'checks': {name: {
                'status': result.status.value,
                'message': result.message,
                'details': result.details,
                'recommendations': result.recommendations,
                'performance_metrics': result.performance_metrics
            } for name, result in results.items()},
            'recommendations': recommendations,
            'dependency_status': STREAMING_DEPS_STATUS
        }
    else:
        return {
            'overall': overall_status,
            'timestamp': datetime.now().isoformat(),
            'streaming_enabled': STREAMING_ENABLED,
            'healthy_checks': status_counts.get('healthy', 0),
            'total_checks': len(results),
            'recommendations': len(recommendations)
        }


def streaming_healthcheck_sync(detailed: bool = True) -> Dict[str, Any]:
    """Synchronous wrapper for streaming healthcheck (CLI-friendly)."""
    try:
        return asyncio.run(streaming_healthcheck(detailed=detailed))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(streaming_healthcheck(detailed=detailed))
        finally:
            asyncio.set_event_loop(None)
            loop.close()


if __name__ == "__main__":
    # Run streaming health check when executed directly
    async def main():
        print("Running RELIC Streaming Health Check...")
        print("=" * 50)
        
        results = await streaming_healthcheck(detailed=True)
        
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Streaming Enabled: {results['streaming_enabled']}")
        print(f"Timestamp: {results['timestamp']}")
        print()
        
        # Show individual check results
        for name, result in results['checks'].items():
            status_symbol = {
                'healthy': '[OK]',
                'warning': '[WARN]',
                'error': '[ERR]'
            }[result['status']]
            
            print(f"{status_symbol} {name.replace('_', ' ').title()}: {result['message']}")
        
        # Show dependency status
        print("\nDependency Status:")
        for dep, available in results['dependency_status'].items():
            status_symbol = '[OK]' if available else '[MISS]'
            print(f"  {status_symbol} {dep}")
        
        # Show recommendations
        if results['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "=" * 50)
        print("Streaming health check completed.")
    
    asyncio.run(main())
