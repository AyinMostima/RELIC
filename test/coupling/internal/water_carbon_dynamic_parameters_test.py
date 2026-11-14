from dataclasses import replace

import torch

import relic.coupling.internal.water_carbon as water_carbon_module
from relic.coupling.internal.water_carbon import WaterCarbonCoupling
from relic.parameters import get_global_manager
from relic.parameters.namespaces import ParameterNamespace


def test_water_carbon_parameters_follow_injected_manager(monkeypatch):
    """Ensure WaterCarbonCoupling reads live values from the injected ParameterManager."""
    pm = get_global_manager()
    key = "coupling.scales.water_carbon.wilting_point_volumetric"
    original_spec = pm.get_parameter(key)
    assert original_spec is not None
    monkeypatch.setattr(pm.database, "set_parameter", lambda spec, tier: None)

    functor = WaterCarbonCoupling(
        name="water_carbon_dynamic_params",
        config={},
        parameter_manager=pm,
    )

    original = functor.wilting_point_volumetric.clone()
    new_value = original + 0.02

    pm.set_parameter(
        replace(original_spec, value=float(new_value.item())),
        tier="calibrated",
    )

    try:
        updated = functor.wilting_point_volumetric
        assert functor.parameter_manager is pm
        assert not torch.allclose(original, updated)
        assert torch.isclose(updated, new_value, atol=1e-6)
    finally:
        pm.set_parameter(
            original_spec,
            tier="calibrated",
        )


def test_water_carbon_uses_physical_constant_ssot(monkeypatch):
    """Physical constants must bypass the domain ParameterManager path."""

    class GuardedWaterCarbonCoupling(WaterCarbonCoupling):
        def get_parameter_value(self, name, *args, **kwargs):
            if kwargs.get("namespace") == ParameterNamespace.PHYSICAL_CONSTANTS:
                raise AssertionError("Physical constants must use get_physical_constant")
            return super().get_parameter_value(name, *args, **kwargs)

    monkeypatch.setattr(
        water_carbon_module,
        "get_physical_constant",
        lambda name: 273.15,
        raising=False,
    )

    functor = GuardedWaterCarbonCoupling(
        name="water_carbon_physical_constant_ssot",
        config={},
    )
    functor._refresh_cached_parameters()

    assert functor._t_freeze == 273.15
