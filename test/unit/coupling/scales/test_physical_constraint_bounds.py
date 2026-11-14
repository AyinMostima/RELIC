import torch

from relic.coupling.scales import physical_constraint_manager as constraint_module
from relic.coupling.scales.physical_constraint_manager import PhysicalConstraintManager
from relic.parameters import get_global_manager


class _BoundsParameterManager:
    def __init__(self, upper_bound: float) -> None:
        self.upper_bound = upper_bound

    def get_parameter(self, key):
        # Parameter metadata (the allowed constraint methods) comes from the parameter database.
        return get_global_manager().get_parameter(key)

    def get(self, key, *, namespace, **_kwargs):
        if key == "variables" and namespace == "coupling.scales.defaults":
            return {
                "soil_temperature": {
                    "bounds": [0.0, self.upper_bound]
                }
            }
        return None


def test_scale_variable_bounds_clamp_via_parameter_manager():
    """Bounds from coupling.scales.<scale>.variables must clamp data and record source."""
    pm = get_global_manager()
    manager = PhysicalConstraintManager(parameter_manager=pm)
    manager.initialize()

    # precipitation_flux declares [0, 0.52] kg m-2 s-1 (the WMO greatest 1-min rainfall, 31.2 mm).
    data = torch.tensor([0.0, 0.6], dtype=torch.float32)
    constrained = manager.apply_constraints(
        data, variable_name="precipitation_flux", scale="defaults"
    )

    assert torch.allclose(
        constrained, torch.tensor([0.0, 0.52], dtype=torch.float32), atol=1e-6
    )
    assert manager._last_bounds_source is not None
    assert (
        "coupling.scales.defaults.variables.precipitation_flux.bounds"
        in manager._last_bounds_source
    )


def test_injected_parameter_manager_remains_the_constraint_owner(monkeypatch):
    """Initialization must not replace the explicitly injected parameter owner."""
    injected_manager = _BoundsParameterManager(upper_bound=1.0)
    global_manager = _BoundsParameterManager(upper_bound=2.0)
    monkeypatch.setattr(
        constraint_module, "get_global_manager", lambda: global_manager
    )

    manager = PhysicalConstraintManager(
        parameter_manager=injected_manager,
        method="clip",
        preserve_gradients=True,
        device=torch.device("cpu"),
    )
    manager.initialize()
    constrained = manager.apply_constraints(
        torch.tensor([1.5], dtype=torch.float32),
        variable_name="soil_temperature",
    )

    assert manager.parameter_manager is injected_manager
    assert torch.equal(constrained, torch.tensor([1.0], dtype=torch.float32))


def test_leaf_conductance_catalog_preserves_medlyn_values_without_a_hidden_upper_cap():
    """Numerical admissibility above 2 is not a physiological observation claim."""
    manager = PhysicalConstraintManager(parameter_manager=get_global_manager(), device=torch.device("cpu"))
    manager.initialize()
    values = torch.tensor([-0.1, 0.01, 2.5, 6.0], dtype=torch.float64, requires_grad=True)
    result = manager.apply_constraints(values, variable_name="stomatal_conductance")
    torch.testing.assert_close(result, torch.tensor([0.0, 0.01, 2.5, 6.0], dtype=torch.float64), rtol=0, atol=0)
    gradient, = torch.autograd.grad(result.sum(), values)
    torch.testing.assert_close(gradient[1:], torch.ones(3, dtype=torch.float64))
    assert manager._last_bounds_source == "coupling.scales.defaults.variables.stomatal_conductance.bounds"
