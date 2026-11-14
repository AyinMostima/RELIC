import pytest
import torch

from relic.coupling.core.policy import ScalingMode
import relic.coupling.scales.alias_utils as alias_utils
import relic.coupling.scales.scale_manager as scale_manager_module
import relic.coupling.scales.scale_strategies.manager_hierarchy_ops as hierarchy_ops
import relic.coupling.scales.temporal_strategies as temporal_strategies_module
from relic.coupling.scales.scale_manager import ScaleManager
from relic.coupling.scales.types import ScalingStrategy
from relic.coupling.scales.topology_builder import TopologyBuilder
from relic.core.state import StateManager
from relic.parameters import get_global_manager


def _initialized_scale_manager():
    manager = ScaleManager(
        parameter_manager=get_global_manager(),
        state_manager=StateManager(),
    )
    manager.initialize()
    return manager


def test_flux_metadata_missing_raises_descriptive_error():
    manager = _initialized_scale_manager()
    with pytest.raises(KeyError) as excinfo:
        manager.get_variable_metadata("evapotranspiration_flux_missing", scale="patch")
    assert "coupling.scales.patch.variables" in str(excinfo.value)


def test_param_missing_reports_namespace():
    manager = _initialized_scale_manager()
    with pytest.raises(RuntimeError) as excinfo:
        manager._get_param(
            "nonexistent_param_for_test",
            namespace="coupling.scales.test_guardrail",
            required=True,
            validate=False,
        )
    assert "coupling.scales.test_guardrail" in str(excinfo.value)


def test_scale_manager_temporal_alias_getter_surface_is_absent():
    assert not hasattr(ScaleManager, "_get_temporal_alias_aggregator")
    assert not hasattr(scale_manager_module, "get_temporal_alias_aggregator_impl")


def test_manager_hierarchy_temporal_alias_getter_impl_is_absent():
    assert not hasattr(hierarchy_ops, "get_temporal_alias_aggregator_impl")


def test_temporal_strategy_alias_getter_stub_is_absent():
    assert not hasattr(
        temporal_strategies_module.TemporalScalingStrategy,
        "get_temporal_alias_aggregator",
    )


def test_temporal_alias_fails_closed_without_explicit_period_contract():
    manager = _initialized_scale_manager()
    metadata = {}
    var_metadata = {
        "air_temperature": {
            "variable_type": "intensive",
            "units": "K",
        }
    }

    with pytest.raises(RuntimeError, match="temporal_alias_periodization_unbound"):
        manager._handle_alias_aggregation(
            data_dict={"air_temperature": torch.ones(30)},
            source_scale="daily",
            target_scale="monthly",
            variables=["air_temperature"],
            strategy=ScalingStrategy.AREA_WEIGHTED_MEAN,
            mode=ScalingMode.CONSERVATIVE,
            metadata=metadata,
            var_metadata=var_metadata,
        )

    assert metadata == {}
    assert var_metadata["air_temperature"]["units"] == "K"
    assert not hasattr(manager, "_temporal_alias_aggregators")


def test_scale_manager_has_no_generic_year_length_surface():
    """Calendar-specific owners replace the unused generic year-length API."""
    assert not hasattr(ScaleManager, "_get_days_per_year")
    assert not hasattr(hierarchy_ops, "get_days_per_year_impl")


def test_temporal_alias_orphan_helpers_are_absent():
    for name in (
        "summarize_temporal_quality",
        "reduce_temporal_alias_tensor",
        "apply_temporal_flux_scaling",
        "build_temporal_alias_time_vectors",
    ):
        assert not hasattr(alias_utils, name)


def test_hierarchy_config_reads_yaml_not_defaults():
    pm = get_global_manager()
    manager = _initialized_scale_manager()
    config = manager.topology_builder.ensure_hierarchy_config()

    expected_patch_units = pm.get(
        "patch.units_per_hillslope",
        namespace="coupling.scales.hierarchy",
        expected_type=int,
    )
    expected_patch_area = pm.get(
        "patch.default_area",
        namespace="coupling.scales.hierarchy",
        expected_type=float,
    )
    assert config["patch_units_per_hillslope"] == expected_patch_units
    assert pytest.approx(config["patch_area"], rel=1e-6) == expected_patch_area


def test_topology_builder_missing_param_reports_path():
    hierarchy_values = {
        "coupling.scales.hierarchy.levels": ["individual", "patch", "hillslope", "basin"],
        "coupling.scales.hierarchy.default_source_level": "patch",
        "coupling.scales.hierarchy.default_target_level": "basin",
        "coupling.scales.hierarchy.hillslope.units_per_basin": 2,
        "coupling.scales.hierarchy.basin.units_per_landscape": 1,
        "coupling.scales.hierarchy.landscape.units_per_regional": 1,
        "coupling.scales.hierarchy.individual.units_per_patch": 1,
        "coupling.scales.hierarchy.patch.default_area": 100.0,
        "coupling.scales.hierarchy.area_variation_factor": 0.1,
        "coupling.scales.hierarchy.extensive_variable_strategy": "conservative",
        "coupling.scales.hierarchy.intensive_variable_strategy": "area_weighted_mean",
        "coupling.scales.hierarchy.categorical_variable_strategy": "nearest",
    }

    class _StubParameterManager:
        def get(self, key, namespace=None, **_):
            full_key = f"{namespace}.{key}" if namespace and not key.startswith(namespace) else key
            if full_key not in hierarchy_values:
                raise KeyError(full_key)
            return hierarchy_values[full_key]

    pm = _StubParameterManager()

    def _get_param(key, namespace=None, **kwargs):
        return pm.get(key, namespace=namespace)

    builder = TopologyBuilder(parameter_manager=pm, get_param=_get_param)
    with pytest.raises(RuntimeError) as excinfo:
        builder.ensure_hierarchy_config()
    assert "coupling.scales.hierarchy.patch.units_per_hillslope" in str(excinfo.value)
