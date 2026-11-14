from relic.parameters import get_global_manager

import pytest


def test_soil_vegetation_parameter_keys_exist():
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    keys = [
        "coupling.scales.soil_vegetation.distribution_type",
        "coupling.scales.soil_vegetation.root_depth_max",
        "coupling.scales.soil_vegetation.shape_parameter",
        "coupling.scales.soil_vegetation.linear_distribution_coeff",
        "coupling.scales.soil_vegetation.beta_q_parameter",
        "coupling.scales.soil_vegetation.wilting_point_potential",
        "coupling.scales.soil_vegetation.anaerobic_point",
        "coupling.scales.soil_vegetation.optimal_range",
    ]
    for key in keys:
        val = pm.get(key)
        assert val is not None, f"Missing parameter {key}"

    removed_keys = [
        "coupling.scales.soil_vegetation.stress_function_type",
        "coupling.scales.soil_vegetation.field_capacity_pressure",
        "coupling.scales.soil_vegetation.severe_water_stress_threshold",
        "coupling.scales.soil_vegetation.minimum_soil_moisture",
        "coupling.scales.soil_vegetation.fine_root_fraction",
    ]
    for key in removed_keys:
        with pytest.raises(KeyError):
            pm.get(key, required=True)


def test_water_carbon_parameter_keys_exist():
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    keys = [
        "coupling.scales.water_carbon.wilting_point_volumetric",
        "coupling.scales.water_carbon.field_capacity_volumetric",
        "coupling.scales.water_carbon.stress_function_type",
    ]
    for key in keys:
        val = pm.get(key)
        assert val is not None, f"Missing parameter {key}"


def test_scale_parameter_keys_without_a_reader_are_removed():
    """coupling.scales.temporal.aggregation_strategy/preserve_totals and
    coupling.scales.conservative.preserve_positivity had no reader in relic (the spatial
    aggregation_strategy/preserve_totals keys read by SpatialAggregation stay)."""
    pm = get_global_manager(preload_all=False, warmup_cache=False)
    for key in (
        "coupling.scales.temporal.aggregation_strategy",
        "coupling.scales.temporal.preserve_totals",
        "coupling.scales.conservative.preserve_positivity",
    ):
        with pytest.raises(KeyError):
            pm.get(key, required=True)
    for key in ("coupling.scales.spatial.aggregation_strategy", "coupling.scales.spatial.preserve_totals"):
        assert pm.get(key) is not None, key
