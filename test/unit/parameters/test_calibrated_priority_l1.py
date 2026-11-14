"""L1 tests verifying calibrated tier precedence in the parameter system."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import pytest

from relic.parameters import get_global_manager, reset_global_manager
from relic.scientific.biosphere.bvoc.emission_factors import (
    CompoundCategory,
    EmissionFactorDatabase,
    PlantFunctionalType,
)


@pytest.fixture
def fresh_manager(fresh_global_parameter_manager):
    """Provide a fresh ParameterManager instance for each test (the previous one is
    put back afterwards)."""
    previous_skip_calibrated = os.environ.get("RELIC_SKIP_CALIBRATED")
    os.environ["RELIC_SKIP_CALIBRATED"] = "0"
    manager = get_global_manager(preload_all=False, warmup_cache=False)
    yield manager
    if previous_skip_calibrated is None:
        os.environ.pop("RELIC_SKIP_CALIBRATED", None)
    else:
        os.environ["RELIC_SKIP_CALIBRATED"] = previous_skip_calibrated


def _expected_calibrated_value(manager) -> float:
    merged = manager.database.load_all()
    spec = merged["biosphere.bvoc.broadleaf_deciduous_temperate.isoprene"]
    return float(spec.value)


def _reset_caches(manager) -> None:
    manager.parameters.clear()
    manager.storage.clear()
    manager.database.cache.clear()
    manager.database.cache_dirty = True


def test_calibrated_tier_overrides_base_on_lazy_load(fresh_manager):
    """Calibrated BVOC factor should override base values when lazily loaded."""
    expected_value = _expected_calibrated_value(fresh_manager)
    assert expected_value != pytest.approx(70.0, rel=1e-6)

    _reset_caches(fresh_manager)
    value = fresh_manager.get(
        "biosphere.bvoc.broadleaf_deciduous_temperate.isoprene",
        units="ug g-1 DW h-1",
        required=True,
    )
    assert value == pytest.approx(expected_value)


def test_emission_factor_database_uses_calibrated_values_by_default(fresh_manager):
    """EmissionFactorDatabase should surface calibrated factors without manual defaults."""
    expected_value = _expected_calibrated_value(fresh_manager)
    _reset_caches(fresh_manager)

    database = EmissionFactorDatabase()
    entry = database.get_emission_factor(
        PlantFunctionalType.BROADLEAF_DECIDUOUS_TEMPERATE,
        CompoundCategory.ISOPRENE,
    )
    assert entry.value == pytest.approx(expected_value)


def test_skip_calibrated_keeps_lazy_and_preloaded_native_values_identical(
    monkeypatch, fresh_global_parameter_manager,
):
    """Native-only selection must not depend on parameter load order."""
    monkeypatch.setenv("RELIC_SKIP_CALIBRATED", "1")
    key = "biosphere.photosynthesis.vcmax25"

    preloaded_manager = get_global_manager(
        preload_all=True,
        warmup_cache=False,
    )
    expected_native = preloaded_manager.get(
        key,
        units="umol CO2 m-2 s-1",
        required=True,
    )

    # A second fresh manager, loaded lazily (the fixture puts the original back).
    reset_global_manager()
    lazy_manager = get_global_manager(
        preload_all=False,
        warmup_cache=False,
    )
    actual_lazy = lazy_manager.get(
        key,
        units="umol CO2 m-2 s-1",
        required=True,
    )

    assert actual_lazy == pytest.approx(expected_native)

