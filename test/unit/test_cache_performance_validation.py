"""
Cache Performance Validation Test

Validates that P5 optimizations (parameter loading, warmup, caching) work correctly.
"""

import sys
import time
from pathlib import Path

# Add relic to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.parameters import get_global_manager
import pytest


@pytest.fixture(autouse=True)
def _fresh_global_parameter_manager(fresh_global_parameter_manager):
    """Every test here starts from a fresh process-wide ParameterManager; the
    manager that was global before is put back after each test."""
    yield


def test_parameter_loading():
    """Test that parameters are loaded correctly during initialization."""
    print("\n[TEST] Parameter Loading Verification")
    print("=" * 60)

    # Reset to get fresh instance
    manager = get_global_manager()

    # Check that database cache is populated
    print(f"Database cache size: {len(manager.database.cache)}")
    print(f"Manager parameters size: {len(manager.parameters)}")
    print(f"Storage has items: {len(manager.storage._store) if hasattr(manager.storage, '_store') else 'N/A'}")

    # Try to get a known parameter
    test_params = [
        'biosphere.photosynthesis.vcmax25',
        'core.simulation.time_step',
        'atmosphere.constants.R_over_Cp'
    ]

    print("\nTesting parameter retrieval:")
    for key in test_params:
        value = manager.get(key)
        exists = manager.exists(key)
        print(f"  {key}: value={value}, exists={exists}")

    # Check if any parameters loaded
    if len(manager.database.cache) > 0:
        print(f"\n[OK] Database has {len(manager.database.cache)} parameters loaded")
        # Show first 5 keys
        sample_keys = list(manager.database.cache.keys())[:5]
        print(f"Sample keys: {sample_keys}")
    else:
        print("\n[ERROR] Database cache is empty!")

    return len(manager.database.cache) > 0


def test_cache_warmup():
    """Test that cache warmup executes correctly."""
    print("\n[TEST] Cache Warmup Verification")
    print("=" * 60)

    manager = get_global_manager()

    # Get cache stats after warmup (should happen during init)
    stats = manager.get_cache_stats()

    print(f"Cache enabled: {stats['cache_enabled']}")
    print(f"Parameter cache size: {stats['parameter_cache'].get('size', 0)}")
    print(f"Parameter cache maxsize: {stats['parameter_cache'].get('maxsize', 0)}")
    print(f"Parameter cache hits: {stats['parameter_cache'].get('hits', 0)}")
    print(f"Parameter cache misses: {stats['parameter_cache'].get('misses', 0)}")

    # Manual warmup to verify method works
    print("\nManually triggering warmup...")
    warmup_stats = manager.warmup_cache()
    print(f"Warmup cached: {warmup_stats['cached_count']}/{warmup_stats['attempted_count']}")
    print(f"Warmup failed: {warmup_stats['failed_count']}")

    if warmup_stats['failed_count'] > 0:
        print(f"Failed keys (first 5): {warmup_stats['failed_keys'][:5]}")

    return warmup_stats['cached_count'] > 0


def test_cache_hit_rate():
    """Test cache hit rate with repeated access."""
    print("\n[TEST] Cache Hit Rate Verification")
    print("=" * 60)

    manager = get_global_manager()

    # Clear cache stats (keep cache content)
    if hasattr(manager._parameter_cache, '_hits'):
        manager._parameter_cache._hits = 0
        manager._parameter_cache._misses = 0

    # Access same parameter 100 times
    test_key = 'biosphere.photosynthesis.vcmax25'

    print(f"Accessing '{test_key}' 100 times...")

    start = time.perf_counter()
    for _ in range(100):
        value = manager.get(test_key)
    elapsed = (time.perf_counter() - start) * 1000

    stats = manager.get_cache_stats()
    param_stats = stats.get('parameter_cache', {})

    hits = param_stats.get('hits', 0)
    misses = param_stats.get('misses', 0)
    hit_rate = param_stats.get('hit_rate', 0) * 100

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}ms ({elapsed/100:.4f}ms per call)")
    print(f"  Cache hits: {hits}")
    print(f"  Cache misses: {misses}")
    print(f"  Hit rate: {hit_rate:.2f}%")

    # After warmup and repeated access, hit rate should be >90%
    if hit_rate > 90:
        print(f"\n[OK] Cache hit rate {hit_rate:.2f}% > 90%")
        success = True
    else:
        print(f"\n[WARNING] Cache hit rate {hit_rate:.2f}% < 90%")
        success = False

    return success, hit_rate, elapsed/100


def test_performance_target():
    """Test that performance target is met: 1000 get() < 1ms."""
    print("\n[TEST] Performance Target Verification")
    print("=" * 60)

    manager = get_global_manager()

    # Reset cache stats
    if hasattr(manager._parameter_cache, '_hits'):
        manager._parameter_cache._hits = 0
        manager._parameter_cache._misses = 0

    # Simulate realistic access pattern: 5 parameters x 200 times
    # Use parameters that are actually in warmup list for fair test
    hot_params = [
        'biosphere.photosynthesis.kc25',
        'biosphere.photosynthesis.ko25',
        'biosphere.constants.gas_constant',
        'biosphere.respiration.rd_q10',
        'standard_atmosphere.standard_temperature'
    ]

    print(f"Simulating 1000 get() calls (5 params x 200 times)...")

    start = time.perf_counter()
    for _ in range(200):
        for param in hot_params:
            _ = manager.get(param)
    elapsed_ms = (time.perf_counter() - start) * 1000

    stats = manager.get_cache_stats()
    param_stats = stats.get('parameter_cache', {})
    hit_rate = param_stats.get('hit_rate', 0) * 100

    print(f"\nResults:")
    print(f"  Total time: {elapsed_ms:.2f}ms")
    print(f"  Per-call time: {elapsed_ms/1000:.4f}ms")
    print(f"  Cache hit rate: {hit_rate:.2f}%")
    print(f"  Target: < 1.0ms total")

    if elapsed_ms < 1.0:
        print(f"\n[OK] Performance target met: {elapsed_ms:.2f}ms < 1.0ms")
        success = True
    else:
        print(f"\n[FAILED] Performance target not met: {elapsed_ms:.2f}ms >= 1.0ms")
        success = False

    return success, elapsed_ms, hit_rate


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" P5 Cache Performance Validation Test Suite")
    print("=" * 70)

    results = {}

    # Test 1: Parameter loading
    results['loading'] = test_parameter_loading()

    # Test 2: Cache warmup
    results['warmup'] = test_cache_warmup()

    # Test 3: Cache hit rate
    results['hit_rate'], hit_rate, _ = test_cache_hit_rate()

    # Test 4: Performance target
    results['performance'], perf_time, perf_hit_rate = test_performance_target()

    # Summary
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)
    print(f"Parameter Loading:    {'[OK]' if results['loading'] else '[FAILED]'}")
    print(f"Cache Warmup:         {'[OK]' if results['warmup'] else '[FAILED]'}")
    print(f"Cache Hit Rate:       {'[OK]' if results['hit_rate'] else '[FAILED]'} ({hit_rate:.1f}%)")
    print(f"Performance Target:   {'[OK]' if results['performance'] else '[FAILED]'} ({perf_time:.2f}ms)")
    print("=" * 70)

    all_passed = all(results.values())
    if all_passed:
        print("\n[SUCCESS] All P5 optimizations verified!")
    else:
        print("\n[NEEDS WORK] Some optimizations need improvement")
        failed = [k for k, v in results.items() if not v]
        print(f"Failed tests: {failed}")

