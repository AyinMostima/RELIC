"""
Cache Diagnosis Test for UnifiedParameterManager

This test diagnoses why cache hit rate is only 1.04% and identifies optimization opportunities.
"""

import sys
import time
from pathlib import Path

# Add relic to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.parameters import get_global_manager


def test_cache_key_variability():
    """Test how cache keys vary with different call patterns."""
    print("\n[TEST 1] Cache Key Variability Diagnosis")
    print("=" * 60)

    manager = get_global_manager()

    # Clear cache to start fresh
    if hasattr(manager, '_parameter_cache'):
        manager._parameter_cache._cache.clear()

    # Test 1: Same parameter, no units, no type
    print("\nTest 1.1: Same key, no units, no type (should hit cache)")
    for i in range(5):
        val = manager.get('biosphere.photosynthesis.vcmax25')
        print(f"  Call {i+1}: {val}")

    stats1 = manager._parameter_cache.get_stats() if manager._parameter_cache else {}
    print(f"  Cache stats: hits={stats1.get('hits', 0)}, misses={stats1.get('misses', 0)}, "
          f"hit_rate={stats1.get('hit_rate', 0):.2f}%")

    # Clear cache
    manager._parameter_cache._cache.clear()

    # Test 2: Same parameter, different units
    print("\nTest 1.2: Same key, different units (will miss cache)")
    manager.get('biosphere.photosynthesis.vcmax25')  # no units
    manager.get('biosphere.photosynthesis.vcmax25', units='umol CO2 m-2 s-1')  # with units
    manager.get('biosphere.photosynthesis.vcmax25')  # no units again

    stats2 = manager._parameter_cache.get_stats() if manager._parameter_cache else {}
    print(f"  Cache stats: hits={stats2.get('hits', 0)}, misses={stats2.get('misses', 0)}, "
          f"hit_rate={stats2.get('hit_rate', 0):.2f}%")

    # Clear cache
    manager._parameter_cache._cache.clear()

    # Test 3: Same parameter, different expected_type
    print("\nTest 1.3: Same key, different expected_type (will miss cache)")
    manager.get('biosphere.photosynthesis.vcmax25')  # no type
    manager.get('biosphere.photosynthesis.vcmax25', expected_type=float)  # with type
    manager.get('biosphere.photosynthesis.vcmax25')  # no type again

    stats3 = manager._parameter_cache.get_stats() if manager._parameter_cache else {}
    print(f"  Cache stats: hits={stats3.get('hits', 0)}, misses={stats3.get('misses', 0)}, "
          f"hit_rate={stats3.get('hit_rate', 0):.2f}%")


def test_realistic_access_pattern():
    """Test realistic Functor access pattern."""
    print("\n[TEST 2] Realistic Functor Access Pattern")
    print("=" * 60)

    manager = get_global_manager()

    # Clear cache
    if hasattr(manager, '_parameter_cache'):
        manager._parameter_cache._cache.clear()

    # Simulate Functor forward() calls - typically access same parameters repeatedly
    print("\nSimulating 100 forward() calls accessing 5 common parameters...")

    common_params = [
        'biosphere.photosynthesis.vcmax25',
        'pedosphere.respiration.q10',
        'core.simulation.time_step',
        'atmosphere.constants.R_over_Cp',
        'hydrosphere.soil.field_capacity'
    ]

    start = time.perf_counter()
    for iteration in range(100):
        for param_key in common_params:
            # In real Functors, parameters are accessed with same signature repeatedly
            _ = manager.get(param_key)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    per_call = elapsed_ms / (100 * len(common_params))

    stats = manager._parameter_cache.get_stats() if manager._parameter_cache else {}
    print(f"\nResults:")
    print(f"  Total calls: {100 * len(common_params)}")
    print(f"  Time: {elapsed_ms:.2f}ms ({per_call:.4f}ms per call)")
    print(f"  Cache hits: {stats.get('hits', 0)}")
    print(f"  Cache misses: {stats.get('misses', 0)}")
    print(f"  Hit rate: {stats.get('hit_rate', 0):.2f}%")
    print(f"  Cache size: {stats.get('size', 0)}/{stats.get('maxsize', 0)}")

    # This should have high hit rate (>95%) for optimal performance
    if stats.get('hit_rate', 0) < 95:
        print(f"\n  [WARNING] Hit rate {stats.get('hit_rate', 0):.2f}% is below 95% target!")
        print(f"  [WARNING] This indicates cache is not working effectively.")


def test_cache_with_units_conversion():
    """Test cache behavior with unit conversion."""
    print("\n[TEST 3] Cache with Unit Conversion")
    print("=" * 60)

    manager = get_global_manager()

    # Clear cache
    if hasattr(manager, '_parameter_cache'):
        manager._parameter_cache._cache.clear()

    print("\nAccessing temperature parameter with different units...")

    # Access with different units
    for i in range(5):
        val1 = manager.get('atmosphere.temperature.reference', units='K')
        val2 = manager.get('atmosphere.temperature.reference', units='celsius')
        val3 = manager.get('atmosphere.temperature.reference')  # raw units
        print(f"  Iteration {i+1}: K={val1}, C={val2}, raw={val3}")

    stats = manager._parameter_cache.get_stats() if manager._parameter_cache else {}
    print(f"\nCache stats:")
    print(f"  Hits: {stats.get('hits', 0)}")
    print(f"  Misses: {stats.get('misses', 0)}")
    print(f"  Hit rate: {stats.get('hit_rate', 0):.2f}%")

    # Expected: Each unit variant creates a separate cache entry
    # But repeated access with same units should hit cache
    expected_hits = 4 * 3  # 4 iterations (after first) * 3 unit variants
    print(f"  Expected hits: ~{expected_hits} (if caching works correctly)")


def test_cache_warmup_benefit():
    """Test performance benefit of cache warmup."""
    print("\n[TEST 4] Cache Warmup Performance Benefit")
    print("=" * 60)

    manager = get_global_manager()

    # Test without warmup
    manager._parameter_cache._cache.clear()

    common_params = [
        'biosphere.photosynthesis.vcmax25',
        'pedosphere.respiration.q10',
        'core.simulation.time_step'
    ]

    print("\nTest 4.1: Cold cache (no warmup)")
    start = time.perf_counter()
    for _ in range(1000):
        for param in common_params:
            _ = manager.get(param)
    cold_time = (time.perf_counter() - start) * 1000

    cold_stats = manager._parameter_cache.get_stats()
    print(f"  Time: {cold_time:.2f}ms")
    print(f"  Hit rate: {cold_stats.get('hit_rate', 0):.2f}%")

    # Test with warmup
    manager._parameter_cache._cache.clear()

    print("\nTest 4.2: Warm cache (pre-loaded)")
    # Pre-load parameters
    for param in common_params:
        _ = manager.get(param)

    # Clear stats but keep cache
    if hasattr(manager._parameter_cache, '_hits'):
        manager._parameter_cache._hits = 0
        manager._parameter_cache._misses = 0

    start = time.perf_counter()
    for _ in range(1000):
        for param in common_params:
            _ = manager.get(param)
    warm_time = (time.perf_counter() - start) * 1000

    warm_stats = manager._parameter_cache.get_stats()
    print(f"  Time: {warm_time:.2f}ms")
    print(f"  Hit rate: {warm_stats.get('hit_rate', 0):.2f}%")

    speedup = cold_time / warm_time if warm_time > 0 else 0
    print(f"\nSpeedup with warmup: {speedup:.2f}x")
    print(f"Time saved: {cold_time - warm_time:.2f}ms")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" Parameter Cache Diagnosis Test Suite")
    print("=" * 70)

    test_cache_key_variability()
    test_realistic_access_pattern()
    test_cache_with_units_conversion()
    test_cache_warmup_benefit()

    print("\n" + "=" * 70)
    print(" Diagnosis Complete")
    print("=" * 70)

