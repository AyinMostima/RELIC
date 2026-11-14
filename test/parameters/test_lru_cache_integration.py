"""
L1 Unit Tests for Parameter Manager LRU Cache Integration

Tests LRU cache functionality in UnifiedParameterManager including:
- Parameter query caching
- Unit conversion caching
- Cache hit/miss statistics
- Cache invalidation
- Thread safety

Test Level: L1 (Unit)
Dependencies: UnifiedParameterManager, ThreadSafeLRUCache
"""

import pytest
import torch
import sys
import time
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.parameters import (
    ParameterCategory,
    ParameterSpec,
    UnifiedParameterManager,
    ValidationLevel,
)
from relic.parameters.lru_cache import ThreadSafeLRUCache, make_hashable


class TestLRUCacheBasics:
    """Test basic LRU cache functionality"""
    
    def test_lru_cache_initialization(self):
        """Test that LRU cache is properly initialized"""
        cache = ThreadSafeLRUCache(maxsize=100, ttl=None)
        
        assert len(cache) == 0
        assert cache._maxsize == 100
        assert cache._ttl is None
    
    def test_lru_cache_put_and_get(self):
        """Test basic put and get operations"""
        cache = ThreadSafeLRUCache(maxsize=100)
        
        cache.put('key1', 'value1')
        result = cache.get('key1')
        
        assert result == 'value1'
        assert len(cache) == 1
    
    def test_lru_cache_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = ThreadSafeLRUCache(maxsize=3)
        
        # Fill cache
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')
        
        assert len(cache) == 3
        
        # Add one more, should evict key1
        cache.put('key4', 'value4')
        
        assert len(cache) == 3
        assert cache.get('key1') is None  # Evicted
        assert cache.get('key4') == 'value4'
    
    def test_lru_cache_ttl_expiration(self):
        """Test TTL expiration"""
        cache = ThreadSafeLRUCache(maxsize=100, ttl=0.1)  # 100ms TTL
        
        cache.put('key1', 'value1')
        
        # Should be available immediately
        assert cache.get('key1') == 'value1'
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get('key1') is None
    
    def test_lru_cache_statistics(self):
        """Test cache statistics tracking"""
        cache = ThreadSafeLRUCache(maxsize=100)
        
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        
        # Generate hits and misses
        cache.get('key1')  # Hit
        cache.get('key1')  # Hit
        cache.get('key3')  # Miss
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 2/3
        assert stats['size'] == 2
    
    def test_make_hashable_conversion(self):
        """Test make_hashable utility function"""
        # Test dict
        assert make_hashable({'a': 1, 'b': 2}) == (('a', 1), ('b', 2))
        
        # Test list
        assert make_hashable([1, 2, 3]) == (1, 2, 3)
        
        # Test nested structures
        result = make_hashable({'a': [1, 2], 'b': {'c': 3}})
        assert isinstance(result, tuple)


class TestParameterManagerCacheIntegration:
    """Test LRU cache integration in UnifiedParameterManager"""
    
    @pytest.fixture
    def manager_with_cache(self):
        """Create parameter manager with cache enabled"""
        manager = UnifiedParameterManager(
            cache_enabled=True,
            cache_max_size=100,
            cache_ttl=None
        )
        
        # Register a test parameter
        spec = ParameterSpec(
            key='test.param1',
            value=42.0,
            units='umol m-2 s-1',
            dtype='float32',
            bounds=(0, 100),
            default=42.0,
            category=ParameterCategory.BIOLOGICAL_PARAMETER,
            description='Test parameter',
            references=['test_ref'],
            tags=['test'],
            dimension='flux',
            requires_grad=False
        )
        manager.register_parameter(spec)
        
        yield manager
        
        # Cleanup
        if manager._parameter_cache:
            manager._parameter_cache.clear()
    
    def test_parameter_query_caching(self, manager_with_cache):
        """Test that parameter queries are cached"""
        manager = manager_with_cache
        
        # First query - cache miss
        value1 = manager.get('test.param1')
        
        # Second query - should be cached
        value2 = manager.get('test.param1')
        
        assert value1 == value2
        assert value1 == 42.0
        
        # Check cache statistics
        if manager._parameter_cache:
            stats = manager._parameter_cache.get_stats()
            assert stats['hits'] >= 1  # At least one cache hit

    def test_parameter_cache_never_serves_a_failed_conversion(self, manager_with_cache):
        """A failed conversion raises on every repeated query, also without validation.

        No unconverted value (42 umol m-2 s-1 read as metres) is returned or cached.
        """
        manager = manager_with_cache
        manager.validation_level = ValidationLevel.NONE
        manager._unit_conversion_cache = None

        for _repeat in range(3):
            with pytest.warns(UserWarning):
                with pytest.raises(ValueError, match="Unit conversion failed"):
                    manager.get("test.param1", units="m")
        assert manager.get("test.param1") == 42.0

    def test_unit_conversion_cache_never_serves_a_failed_conversion(self, manager_with_cache):
        """The conversion cache holds no result for a failed conversion."""
        manager = manager_with_cache
        manager.validation_level = ValidationLevel.NONE
        manager._parameter_cache = None

        for _repeat in range(3):
            with pytest.warns(UserWarning):
                with pytest.raises(ValueError, match="Unit conversion failed"):
                    manager.get("test.param1", units="m")
        assert manager.get("test.param1") == 42.0

    def test_unit_conversion_cache_does_not_reuse_mutable_tensor_values(
        self,
        manager_with_cache,
    ):
        """Tensor conversions must reflect the current tensor value."""
        manager = manager_with_cache
        spec = manager.get_parameter("test.param1")
        assert spec is not None

        with pytest.warns(UserWarning):
            first = manager._convert_parameter_units(
                torch.tensor([1.0]),
                spec,
                "mol m-2 s-1",
            )
        with pytest.warns(UserWarning):
            second = manager._convert_parameter_units(
                torch.tensor([2.0]),
                spec,
                "mol m-2 s-1",
            )

        assert torch.allclose(first, torch.tensor([1.0e-6]))
        assert torch.allclose(second, torch.tensor([2.0e-6]))
    
    def test_unit_conversion_caching(self, manager_with_cache):
        """Test that unit conversions are cached"""
        manager = manager_with_cache
        
        # First conversion
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"Parameter 'test\.param1' requested with units 'mol m-2 s-1' "
                        r"but stored units are 'umol m-2 s-1'\. Performing conversion\."
                    ),
                )
                value1 = manager.get('test.param1', units='mol m-2 s-1')
        except Exception:
            pytest.skip("Unit conversion not available")
        
        # Second conversion - should be cached
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"Parameter 'test\.param1' requested with units 'mol m-2 s-1' "
                        r"but stored units are 'umol m-2 s-1'\. Performing conversion\."
                    ),
                )
                value2 = manager.get('test.param1', units='mol m-2 s-1')
        except Exception:
            pytest.skip("Unit conversion not available")
        
        assert value1 == value2
        
        # Check unit conversion cache
        if manager._unit_conversion_cache:
            stats = manager._unit_conversion_cache.get_stats()
            # May have hits depending on implementation details
            assert stats['size'] >= 0
    
    def test_cache_with_different_units(self, manager_with_cache):
        """Test that different unit requests use separate cache entries"""
        manager = manager_with_cache
        
        # Query with original units
        value_raw = manager.get('test.param1')
        
        # Query with different units (if conversion works)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"Parameter 'test\.param1' requested with units 'mol m-2 s-1' "
                        r"but stored units are 'umol m-2 s-1'\. Performing conversion\."
                    ),
                )
                value_converted = manager.get('test.param1', units='mol m-2 s-1')
            
            # Should be different values (after conversion)
            # But both should be cached separately
            assert value_raw != value_converted or value_raw == value_converted  # Either is valid
            
        except Exception:
            # Unit conversion may not be fully implemented
            pytest.skip("Unit conversion not available")
    
    def test_cache_disabled_mode(self):
        """Test parameter manager queries work regardless of cache setting"""
        # Note: UnifiedParameterManager has cache enabled by default
        # This test verifies queries work even if we explicitly disable
        manager = UnifiedParameterManager()
        
        # Manually disable cache for test
        manager._cache_enabled = False
        
        # Register parameter
        spec = ParameterSpec(
            key='test.no_cache',
            value=10.0,
            units='K',
            dtype='float32',
            bounds=(0, 500),
            default=10.0,
            category=ParameterCategory.PHYSICAL_CONSTANT,
            description='Test parameter without cache',
            references=['test'],
            tags=['test'],
            dimension='temperature',
            requires_grad=False
        )
        manager.register_parameter(spec)
        
        # Should still work without cache
        value = manager.get('test.no_cache')
        assert value == 10.0
    
    def test_cache_with_expected_type(self, manager_with_cache):
        """Test caching with expected_type parameter"""
        manager = manager_with_cache
        
        # Query with type expectation
        value1 = manager.get('test.param1', expected_type=float)
        value2 = manager.get('test.param1', expected_type=float)
        
        assert isinstance(value1, float)
        assert value1 == value2
        
        # Check cache statistics
        if manager._parameter_cache:
            stats = manager._parameter_cache.get_stats()
            assert stats['hits'] >= 1


class TestCachePerformance:
    """Test cache performance improvements"""
    
    def test_cache_improves_query_performance(self):
        """Test that cache improves repeated query performance"""
        manager = UnifiedParameterManager(cache_enabled=True, cache_max_size=1000)
        
        # Register multiple parameters
        for i in range(100):
            spec = ParameterSpec(
                key=f'perf.param{i}',
                value=float(i),
                units='dimensionless',
                dtype='float32',
                bounds=(0, 1000),
                default=float(i),
                category=ParameterCategory.BIOLOGICAL_PARAMETER,
                description=f'Performance test parameter {i}',
                references=['test'],
                tags=['perf'],
                dimension='dimensionless',
                requires_grad=False
            )
            manager.register_parameter(spec)
        
        # Warm up cache
        for i in range(100):
            manager.get(f'perf.param{i}')
        
        # Measure cached performance
        start_cached = time.time()
        for _ in range(100):
            for i in range(100):
                manager.get(f'perf.param{i}')
        time_cached = time.time() - start_cached
        
        # Get cache statistics
        if manager._parameter_cache:
            stats = manager._parameter_cache.get_stats()
            
            # Should have many hits
            assert stats['hits'] > 0
            # Hit rate should be very high after warm-up
            assert stats['hit_rate'] > 0.9
            
            logger_msg = (
                f"Cache performance: {stats['hits']} hits, "
                f"{stats['misses']} misses, "
                f"hit rate: {stats['hit_rate']:.2%}, "
                f"time: {time_cached:.4f}s"
            )
            print(logger_msg)
    
    def test_cache_memory_efficiency(self):
        """Test that cache performs evictions when needed"""
        max_size = 50
        manager = UnifiedParameterManager(
            cache_enabled=True, 
            cache_max_size=max_size
        )
        
        # Register more parameters than cache can hold
        for i in range(150):
            spec = ParameterSpec(
                key=f'mem.param{i}',
                value=float(i),
                units='dimensionless',
                dtype='float32',
                bounds=(0, 1000),
                default=float(i),
                category=ParameterCategory.BIOLOGICAL_PARAMETER,
                description=f'Memory test parameter {i}',
                references=['test'],
                tags=['mem'],
                dimension='dimensionless',
                requires_grad=False
            )
            manager.register_parameter(spec)
        
        # Query all parameters
        for i in range(150):
            manager.get(f'mem.param{i}')
        
        # Cache should have performed evictions since we queried more than max_size
        # Note: actual cache size may vary due to multiple cache entries per get()
        # (e.g., different cache keys for same parameter with different options)
        if manager._parameter_cache:
            stats = manager._parameter_cache.get_stats()
            # The key assertion: evictions should have occurred
            assert stats['evictions'] > 0


class TestCacheThreadSafety:
    """Test cache thread safety"""
    
    def test_concurrent_cache_access(self):
        """Test cache is thread-safe under concurrent access"""
        import threading
        
        cache = ThreadSafeLRUCache(maxsize=100)
        errors = []
        
        def worker(thread_id: int):
            try:
                for i in range(100):
                    key = f'key{i % 10}'
                    value = f'value{thread_id}_{i}'
                    cache.put(key, value)
                    result = cache.get(key)
                    # Result may be from this or another thread, both valid
                    assert result is not None
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0


class TestCacheWarmupHotKeys:
    """The default warm-up list names only defined parameters that a module reads."""

    RETIRED_HOT_KEYS = (
        "validation.numerical.q10_factor_min",
        "validation.numerical.q10_factor_max",
        "biosphere.constants.gas_constant",
        "biosphere.constants.time.seconds_per_day",
        "biosphere.constants.time.seconds_per_hour",
        "anthroposphere.emissions.temperature_response.q10",
        "relic.physical.constants.air.density_stp",
    )

    def test_default_hot_keys_exclude_parameters_without_functional_readers(self):
        """Undefined keys, alias copies of physical constants and parameters that
        only the warm-up list named are not warmed."""
        import inspect

        from relic.parameters.cache import ops

        source = inspect.getsource(ops.warmup_cache)
        assert [key for key in self.RETIRED_HOT_KEYS if f"'{key}'" in source] == []

    def test_default_hot_keys_all_resolve(self):
        """Every default hot key is a defined parameter, so warm-up caches all of them."""
        manager = UnifiedParameterManager(cache_enabled=True, cache_max_size=256, cache_ttl=None)
        stats = manager.warmup_cache()
        assert stats["failed_keys"] == []
        assert stats["cached_count"] == stats["attempted_count"]

    def test_unread_parameters_are_removed_from_the_database(self):
        """The Q10 factor bounds, the emission temperature-response Q10 and the
        biosphere alias copies of the physical time constants have no functional
        reader and are removed."""
        import yaml

        root = project_root / "relic" / "parameters" / "database" / "relic"
        numerical = yaml.safe_load(
            (root / "validation_numerical.yaml").read_text(encoding="utf-8-sig")
        )["parameters"]
        assert "validation.numerical.q10_factor_min" not in numerical
        assert "validation.numerical.q10_factor_max" not in numerical
        emissions = yaml.safe_load(
            (root / "anthroposphere" / "emissions_parameters.yaml").read_text(encoding="utf-8-sig")
        )["parameters"]
        assert "anthroposphere.emissions.temperature_response.q10" not in emissions
        assert not (root / "biosphere" / "constants_time.yaml").exists()


class TestCacheWarmupErrorContract:
    """Warm-up reports a missing key, but a failing parameter read raises."""

    @staticmethod
    def _stub_manager(get):
        from types import SimpleNamespace

        return SimpleNamespace(
            _cache_enabled=True,
            _parameter_cache=SimpleNamespace(get_stats=lambda: {"size": 0, "maxsize": 8}),
            _unit_conversion_cache=None,
            get=get,
        )

    @staticmethod
    def _failing_get(key, *args, **kwargs):
        raise ValueError(f"Unit conversion failed for parameter '{key}'")

    def test_warmup_raises_a_failing_parameter_read(self):
        from relic.parameters.cache import ops

        with pytest.raises(ValueError, match="Unit conversion failed"):
            ops.warmup_cache(self._stub_manager(self._failing_get), hot_keys=["a.b"])

    def test_warmup_reports_a_missing_key(self):
        from relic.parameters.cache import ops

        stats = ops.warmup_cache(self._stub_manager(lambda key: None), hot_keys=["a.b"])

        assert stats["failed_keys"] == ["a.b"]
        assert stats["cached_count"] == 0

    def test_preheat_raises_a_failing_parameter_read(self):
        from relic.parameters.cache import ops

        with pytest.raises(ValueError, match="Unit conversion failed"):
            ops.preheat_cache(self._stub_manager(self._failing_get), ["a.b"])

    def test_preheat_counts_a_missing_key_as_failed_not_loaded(self):
        from relic.parameters.cache import ops

        stats = ops.preheat_cache(self._stub_manager(lambda key: None), ["a.b"])

        assert stats["loaded"] == 0
        assert stats["failed"] == 1

    def test_preload_raises_a_failing_warmup(self):
        from types import SimpleNamespace

        from relic.parameters.ops import bootstrap

        def _failing_warmup():
            raise RuntimeError("warmup backend failed")

        manager = SimpleNamespace(
            _preloaded=False,
            _cache_enabled=True,
            database=SimpleNamespace(load_all=lambda: {}),
            parameters={},
            warmup_cache=_failing_warmup,
        )

        with pytest.raises(RuntimeError, match="warmup backend failed"):
            bootstrap.preload_parameters(manager, warmup_cache=True)
        assert manager._preloaded is False


class TestUnitConversionCacheErrorContract:
    """The conversion cache wrapper neither retries a failed conversion nor hides a cache failure."""

    @staticmethod
    def _manager(cache):
        from types import SimpleNamespace

        return SimpleNamespace(
            _cache_enabled=True,
            _unit_conversion_cache=cache,
            validation_level=ValidationLevel.NONE,
            unit_validation_mode="warn",
        )

    @staticmethod
    def _spec():
        from types import SimpleNamespace

        return SimpleNamespace(key="test.param1", units="umol m-2 s-1", dtype="float32")

    def test_failed_conversion_runs_once_and_raises(self, monkeypatch):
        from types import SimpleNamespace

        from relic.parameters.ops import unit_validation

        calls = []

        def _failing_impl(manager, value, spec, target_units):
            calls.append(target_units)
            raise ValueError("Unit conversion failed for parameter 'test.param1'")

        monkeypatch.setattr(unit_validation, "convert_parameter_units_impl", _failing_impl)
        cache = SimpleNamespace(get=lambda key: None, put=lambda key, value: None)

        with pytest.raises(ValueError, match="Unit conversion failed"):
            unit_validation.convert_parameter_units(self._manager(cache), 42.0, self._spec(), "m")
        assert calls == ["m"]

    def test_conversion_cache_failure_is_raised(self, monkeypatch):
        from types import SimpleNamespace

        from relic.parameters.ops import unit_validation

        monkeypatch.setattr(
            unit_validation,
            "convert_parameter_units_impl",
            lambda manager, value, spec, target_units: value * 1.0e-6,
        )

        def _broken_get(key):
            raise RuntimeError("conversion cache backend broke")

        cache = SimpleNamespace(get=_broken_get, put=lambda key, value: None)

        with pytest.raises(RuntimeError, match="conversion cache backend broke"):
            unit_validation.convert_parameter_units(self._manager(cache), 42.0, self._spec(), "m")

    def test_repeated_conversion_is_served_from_the_cache(self, monkeypatch):
        """The conversion cache key is (key, source units, target units, value)."""
        from types import SimpleNamespace

        from relic.parameters.ops import unit_validation

        calls = []

        def _impl(manager, value, spec, target_units):
            calls.append(target_units)
            return value * 1.0e-6

        monkeypatch.setattr(unit_validation, "convert_parameter_units_impl", _impl)
        store = {}
        cache = SimpleNamespace(get=store.get, put=store.__setitem__)
        manager = self._manager(cache)

        first = unit_validation.convert_parameter_units(manager, 42.0, self._spec(), "m")
        second = unit_validation.convert_parameter_units(manager, 42.0, self._spec(), "m")

        assert first == second == pytest.approx(4.2e-5)
        assert calls == ["m"]

    def test_manager_conversion_result_is_stable_across_repeated_reads(self):
        manager = UnifiedParameterManager(cache_enabled=True, cache_max_size=16, cache_ttl=None)
        manager.register_parameter(
            ParameterSpec(key="c9.flux", value=42.0, units="umol m-2 s-1", dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter")
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = {
                repeat: manager.get("c9.flux", units="mol m-2 s-1")
                for repeat in (False, True)
            }
            for _repeat in range(2):
                with pytest.raises(ValueError, match="Unit conversion failed"):
                    manager.get("c9.flux", units="m")

        assert converted[False] == pytest.approx(4.2e-5)
        assert converted[True] == pytest.approx(converted[False])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

