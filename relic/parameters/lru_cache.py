"""
Thread-safe LRU cache implementation for parameter management.

This module provides a high-performance LRU (Least Recently Used) cache
with TTL (Time-To-Live) support for caching parameter queries and conversions.

References:
    - LRU Cache algorithm: https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU
    - Python collections.OrderedDict: https://docs.python.org/3/library/collections.html#collections.OrderedDict
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Hashable, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and optional expiration time."""
    value: Any
    timestamp: float
    ttl: Optional[float] = None  # Time-to-live in seconds

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


class ThreadSafeLRUCache:
    """
    Thread-safe LRU cache with TTL support.

    This cache uses an OrderedDict to maintain insertion order and implements
    the LRU eviction policy when the cache reaches its maximum size.

    Parameters:
        maxsize : int
            Maximum number of entries in the cache (default: 1000)
        ttl : Optional[float]
            Time-to-live for cache entries in seconds (default: None, no expiration)

    Attributes:
        hits : int
            Number of cache hits
        misses : int
            Number of cache misses
        evictions : int
            Number of entries evicted due to size limit
        expirations : int
            Number of entries expired due to TTL

    Examples:
        >>> cache = ThreadSafeLRUCache(maxsize=100, ttl=60.0)
        >>> cache.put('key1', 'value1')
        >>> value = cache.get('key1')
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(self, maxsize: int = 1000, ttl: Optional[float] = None):
        """Initialize LRU cache with optional TTL."""
        self._cache: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._maxsize = maxsize
        self._ttl = ttl

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: Hashable) -> Optional[Any]:
        """
        Get value from cache.

        Parameters:
            key : Hashable
                Cache key

        Returns:
            Optional[Any]: Cached value or None if not found or expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                self._expirations += 1
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: Hashable, value: Any, ttl: Optional[float] = None) -> None:
        """
        Put value into cache.

        Parameters:
            key : Hashable
                Cache key
            value : Any
                Value to cache
            ttl : Optional[float]
                Override default TTL for this entry
        """
        with self._lock:
            # Use entry-specific TTL or default
            entry_ttl = ttl if ttl is not None else self._ttl

            # If key exists, update it
            if key in self._cache:
                self._cache[key] = CacheEntry(value, time.time(), entry_ttl)
                self._cache.move_to_end(key)
                return

            # Evict oldest entry if at capacity
            if len(self._cache) >= self._maxsize:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1

            # Add new entry
            self._cache[key] = CacheEntry(value, time.time(), entry_ttl)

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def invalidate(self, key: Hashable) -> bool:
        """
        Invalidate a specific cache entry.

        Parameters:
            key : Hashable
                Cache key to invalidate

        Returns:
            bool: True if key was found and removed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Dictionary containing cache statistics
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Cache hit rate (0.0 to 1.0)
                - size: Current number of entries
                - maxsize: Maximum cache size
                - evictions: Number of evictions
                - expirations: Number of expirations
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'size': len(self._cache),
                'maxsize': self._maxsize,
                'evictions': self._evictions,
                'expirations': self._expirations,
            }

    def __len__(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: Hashable) -> bool:
        """Check if key is in cache (without updating LRU order)."""
        with self._lock:
            return key in self._cache


def make_hashable(obj: Any) -> Hashable:
    """
    Convert an object to a hashable type for cache keys.

    Parameters:
        obj : Any
            Object to convert

    Returns:
        Hashable: Hashable representation of the object

    Examples:
        >>> make_hashable({'a': 1, 'b': [2, 3]})
        (('a', 1), ('b', (2, 3)))
        >>> make_hashable([1, 2, 3])
        (1, 2, 3)
    """
    if isinstance(obj, Hashable):
        return obj
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(make_hashable(item) for item in obj)
    else:
        # For unhashable types like torch.Tensor, use id() as a last-resort key
        # This won't work across different objects with same values
        # but prevents errors
        return f"<unhashable:{type(obj).__name__}:{id(obj)}>"
