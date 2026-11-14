"""Cache helper functions for ParameterManager."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def warmup_cache(manager: Any, hot_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Pre-load frequently accessed parameters into LRU cache.
    """
    if not manager._cache_enabled or manager._parameter_cache is None:
        logger.warning("Cache not enabled, warmup skipped")
        return {'cached_count': 0, 'cache_enabled': False}

    if hot_keys is None:
        hot_keys = [
            'biosphere.photosynthesis.kc25',
            'biosphere.photosynthesis.ko25',
            'biosphere.photosynthesis.gamma_star25',
            'biosphere.photosynthesis.ha_kc',
            'biosphere.photosynthesis.ha_ko',
            'biosphere.photosynthesis.ha_gamma_star',
            'anthroposphere.land_use.vegetation.primary_forest.vcmax25',
            'anthroposphere.land_use.vegetation.primary_forest.jmax25',
            'anthroposphere.land_use.vegetation.cropland_c3.vcmax25',
            'anthroposphere.land_use.vegetation.cropland_c3.jmax25',
            'biosphere.respiration.rd_q10',
            'biosphere.respiration.maintenance.rm_leaf_25',
            'relic.physical.constants.universal.gas_constant',
            'relic.physical.constants.universal.stefan_boltzmann_constant',
            'relic.physical.constants.temperature.standard_temperature',
            'relic.physical.constants.pressure.standard_atmosphere',
        ]

    cached_count = 0
    failed_keys: List[str] = []

    logger.info("Warming cache with %s hot parameters...", len(hot_keys))

    # A hot key that is not defined (get returns None) is reported in
    # failed_keys; an error raised while reading a defined parameter
    # (validation, unit conversion, backend) propagates.
    for key in hot_keys:
        value = manager.get(key)
        if value is not None:
            cached_count += 1
            logger.debug("Cached parameter: %s = %s", key, value)
        else:
            failed_keys.append(key)
            logger.debug("Parameter not found for caching: %s", key)

    cache_stats = manager._parameter_cache.get_stats() if manager._parameter_cache else {}

    warmup_stats = {
        'cached_count': cached_count,
        'attempted_count': len(hot_keys),
        'failed_count': len(failed_keys),
        'failed_keys': failed_keys,
        'cache_size': cache_stats.get('size', 0),
        'cache_maxsize': cache_stats.get('maxsize', 0),
        'cache_enabled': True,
    }

    logger.info(
        "Cache warmup complete: %s/%s parameters cached, %s failed",
        cached_count,
        len(hot_keys),
        len(failed_keys),
    )

    if failed_keys:
        logger.debug("Failed to cache parameters: %s", failed_keys)

    return warmup_stats


def get_cache_stats(manager: Any) -> Dict[str, Any]:
    """Get comprehensive cache statistics."""
    if not manager._cache_enabled:
        return {
            'cache_enabled': False,
            'parameter_cache': {},
            'unit_conversion_cache': {},
            'most_accessed': []
        }

    stats: Dict[str, Any] = {'cache_enabled': True}

    if manager._parameter_cache:
        stats['parameter_cache'] = manager._parameter_cache.get_stats()
    else:
        stats['parameter_cache'] = {}

    if manager._unit_conversion_cache:
        stats['unit_conversion_cache'] = manager._unit_conversion_cache.get_stats()
    else:
        stats['unit_conversion_cache'] = {}

    if manager._access_history:
        access_counts: Dict[str, int] = {}
        for key in manager._access_history:
            access_counts[key] = access_counts.get(key, 0) + 1
        stats['most_accessed'] = sorted(
            access_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
    else:
        stats['most_accessed'] = []

    return stats


def cache_stats(manager: Any) -> Dict[str, Any]:
    """Return LRU cache statistics for quick access."""
    if not manager._cache_enabled:
        return {
            'cache_enabled': False,
            'parameter_cache': None,
            'unit_conversion_cache': None
        }

    return {
        'cache_enabled': True,
        'parameter_cache': manager._parameter_cache.get_stats() if manager._parameter_cache else None,
        'unit_conversion_cache': manager._unit_conversion_cache.get_stats() if manager._unit_conversion_cache else None
    }


def clear_cache(manager: Any) -> None:
    """Clear all parameter caches."""
    if manager._cache_enabled:
        if manager._parameter_cache:
            manager._parameter_cache.clear()
        if manager._unit_conversion_cache:
            manager._unit_conversion_cache.clear()
        logger.info("Parameter caches cleared")


def invalidate_cache_key(manager: Any, key: str) -> bool:
    """Invalidate all cache entries for a specific parameter key."""
    if not manager._cache_enabled:
        return False

    invalidated = False

    if manager._parameter_cache:
        manager._parameter_cache.clear()
        invalidated = True

    if manager._unit_conversion_cache:
        manager._unit_conversion_cache.clear()
        invalidated = True

    logger.debug("Invalidated cache for parameter: %s", key)
    return invalidated


def preheat_cache(manager: Any, parameter_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Pre-load common parameters into cache."""
    if not manager._cache_enabled:
        logger.warning("Cache is disabled, preheat_cache has no effect")
        return {'loaded': 0, 'failed': 0, 'cache_size': 0}

    default_common_parameters = [
        'time_step',
        'biosphere.photosynthesis.vcmax25',
        'biosphere.photosynthesis.jmax25',
        'biosphere.photosynthesis.rd25',
        'biosphere.photosynthesis.kc25',
        'biosphere.photosynthesis.ko25',
        'biosphere.photosynthesis.theta',
        'biosphere.photosynthesis.alpha',
        'pedosphere.decomposition.q10',
        'pedosphere.decomposition.base_rate',
        'atmosphere.radiation.solar_constant',
        'atmosphere.constants.stefan_boltzmann',
        'atmosphere.constants.von_karman',
        'hydrosphere.soil.field_capacity',
        'hydrosphere.soil.wilting_point',
    ]

    keys_to_load = parameter_keys or default_common_parameters

    loaded = 0
    failed = 0

    logger.info("Preheating cache with %s parameters...", len(keys_to_load))

    # manager.get returns None for an undefined key, which counts as failed;
    # an error raised while reading a defined parameter propagates.
    for key in keys_to_load:
        if manager.get(key) is None:
            logger.debug("Parameter %s not found during preheat", key)
            failed += 1
        else:
            loaded += 1

    stats = cache_stats(manager)
    cache_size = stats['parameter_cache']['size'] if stats['parameter_cache'] else 0

    logger.info(
        "Cache preheat complete: %s loaded, %s failed, cache size: %s",
        loaded,
        failed,
        cache_size,
    )

    return {
        'loaded': loaded,
        'failed': failed,
        'cache_size': cache_size
    }
