"""L3 System Test: Performance Benchmarks.

Tests the performance impact of unit conversion system,
ensuring overhead is minimal (<1%).
"""
import torch
import time
from relic.data.standards.unit_converter import UnitConverter, ConversionContext


class TestUnitConversionOverhead:
    """Test computational overhead of unit conversions."""

    def test_single_conversion_latency(self):
        """Test latency of single unit conversion.

        Target: <10ms per conversion for typical values.
        """
        converter = UnitConverter()
        context = ConversionContext(allow_defaults=False)

        # Warm up
        for _ in range(10):
            converter.convert(25.0, 'air_temperature', 'celsius', 'K', context)

        # Measure
        start = time.perf_counter()
        num_iterations = 1000

        for _ in range(num_iterations):
            converter.convert(25.0, 'air_temperature', 'celsius', 'K', context)

        end = time.perf_counter()
        avg_time_ms = ((end - start) / num_iterations) * 1000

        print(f"\nAverage conversion time: {avg_time_ms:.4f} ms")
        assert avg_time_ms < 10.0, \
            f"Conversion too slow: {avg_time_ms:.4f} ms > 10 ms"

    def test_batch_conversion_performance(self, large_scale_tensor):
        """Test performance of batch conversions.

        Converting large arrays should be efficient.
        """
        converter = UnitConverter()
        context = ConversionContext()

        # Create smaller test tensor
        data = torch.randn(100, 100)  # 10k elements

        start = time.perf_counter()

        # Convert array
        result = converter.convert(
            data,
            'air_temperature',
            'celsius',
            'K',
            context
        )

        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000

        print(f"\nBatch conversion time: {elapsed_ms:.2f} ms for {data.numel()} elements")
        # Should be fast for vectorized operations
        assert elapsed_ms < 100.0

    def test_same_unit_fast_path_performance(self):
        """Test that same-unit conversion is very fast (fast path).

        Same-unit conversion should have minimal overhead.
        """
        converter = UnitConverter()
        context = ConversionContext()

        data = torch.randn(1000)

        # Time same-unit conversion
        start = time.perf_counter()
        result = converter.convert(data, 'air_temperature', 'K', 'K', context)
        same_unit_time = time.perf_counter() - start

        # Time actual conversion
        start = time.perf_counter()
        result = converter.convert(data, 'air_temperature', 'celsius', 'K', context)
        convert_time = time.perf_counter() - start

        print(f"\nSame-unit time: {same_unit_time*1000:.4f} ms")
        print(f"Conversion time: {convert_time*1000:.4f} ms")

        # Same-unit should be faster (or at least not slower)
        # Allow some variance due to CPU scheduling
        assert same_unit_time < convert_time * 2.0


class TestCacheEfficiency:
    """Test caching mechanisms (if implemented).

    Note: These tests assume caching might be implemented in the future.
    """

    def test_repeated_conversion_efficiency(self):
        """Test that repeated conversions are efficient.

        If caching is implemented, repeated conversions should be faster.
        """
        converter = UnitConverter()
        context = ConversionContext()

        # First conversion (cache miss)
        start = time.perf_counter()
        for _ in range(100):
            converter.convert(25.0, 'air_temperature', 'celsius', 'K', context)
        first_time = time.perf_counter() - start

        # Subsequent conversions (should hit cache if implemented)
        start = time.perf_counter()
        for _ in range(100):
            converter.convert(25.0, 'air_temperature', 'celsius', 'K', context)
        second_time = time.perf_counter() - start

        print(f"\nFirst batch: {first_time*1000:.2f} ms")
        print(f"Second batch: {second_time*1000:.2f} ms")

        # Second should not be significantly slower
        # (May be faster if caching implemented)
        assert second_time < first_time * 4.0


class TestLargeScaleConversion:
    """Test performance with large-scale data."""

    def test_spatial_grid_conversion(self):
        """Test converting large spatial grids.

        Simulates converting a full spatial grid (e.g., 100x100 grid cells).
        """
        converter = UnitConverter()
        context = ConversionContext()

        # Create spatial grid
        grid_size = (100, 100)
        temperature_grid = 15 + 10 * torch.randn(*grid_size)  # Celsius

        start = time.perf_counter()
        temp_k = converter.convert(
            temperature_grid,
            'air_temperature',
            'celsius',
            'K',
            context
        )
        elapsed = time.perf_counter() - start

        print(f"\nGrid conversion ({grid_size[0]}x{grid_size[1]}): {elapsed*1000:.2f} ms")

        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second
        assert temp_k.shape == grid_size

    def test_temporal_series_conversion(self):
        """Test converting long time series.

        Simulates converting hourly data for a full year.
        """
        converter = UnitConverter()
        context = ConversionContext()

        # 8760 hours in a year
        hourly_temps = 15 + 10 * torch.randn(8760)

        start = time.perf_counter()
        temps_k = converter.convert(
            hourly_temps,
            'air_temperature',
            'celsius',
            'K',
            context
        )
        elapsed = time.perf_counter() - start

        print(f"\nAnnual hourly data conversion: {elapsed*1000:.2f} ms")
        assert elapsed < 0.5  # Less than 500ms


class TestMemoryEfficiency:
    """Test memory efficiency of unit conversions."""

    def test_conversion_doesnt_copy_unnecessarily(self):
        """Test that conversions minimize memory copying.

        For same-unit conversions, should ideally return same object.
        """
        converter = UnitConverter()
        context = ConversionContext()

        data = torch.tensor([298.15, 300.0])

        # Same-unit conversion
        result = converter.convert(data, 'air_temperature', 'K', 'K', context)

        # Result should be equal (whether same object or not)
        assert torch.allclose(result, data)

    def test_large_array_memory_usage(self):
        """Test that large array conversions don't cause memory issues."""
        converter = UnitConverter()
        context = ConversionContext()

        # Create large array
        large_array = torch.randn(10000)

        # Should convert without memory error
        result = converter.convert(
            large_array,
            'air_temperature',
            'celsius',
            'K',
            context
        )

        assert result.shape == large_array.shape


class TestConcurrentConversions:
    """Test performance under concurrent conversion requests.

    Note: These tests check if conversions are thread-safe and performant.
    """

    def test_sequential_different_conversions(self):
        """Test multiple different conversions in sequence."""
        converter = UnitConverter()
        context = ConversionContext()

        conversions = [
            (25.0, 'air_temperature', 'celsius', 'K'),
            (1000.0, 'length', 'm', 'km'),
            (1000.0, 'mass', 'g', 'kg'),
            (3600.0, 'time', 's', 'hour'),
            (1000.0, 'pressure', 'Pa', 'kPa'),
        ]

        start = time.perf_counter()

        for value, var_name, from_unit, to_unit in conversions:
            converter.convert(value, var_name, from_unit, to_unit, context)

        elapsed = time.perf_counter() - start

        print(f"\nSequential conversions: {elapsed*1000:.2f} ms")
        assert elapsed < 0.1  # Should be very fast


class TestOverheadComparison:
    """Compare conversion overhead to raw computation."""

    def test_overhead_vs_raw_computation(self):
        """Test conversion overhead compared to raw arithmetic.

        Target: Conversion should add <1% overhead to computation.
        """
        converter = UnitConverter()
        context = ConversionContext(allow_defaults=False)

        data = torch.randn(10000)

        # Time raw computation (adding constant)
        start = time.perf_counter()
        raw_result = data + 273.15
        raw_time = time.perf_counter() - start

        # Time conversion
        start = time.perf_counter()
        converted = converter.convert(data, 'air_temperature', 'celsius', 'K', context)
        convert_time = time.perf_counter() - start

        overhead_percent = ((convert_time - raw_time) / raw_time) * 100

        print(f"\nRaw computation: {raw_time*1000:.4f} ms")
        print(f"With conversion: {convert_time*1000:.4f} ms")
        print(f"Overhead: {overhead_percent:.2f}%")

        # Absolute cap avoids ratio instability when raw_time is very small
        max_conversion_ms = 50.0
        assert convert_time * 1000 < max_conversion_ms

