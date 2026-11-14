"""SoilWaterRetention: soil water potential of the effective soil layer.

The process binds the shared van Genuchten inversion of
relic.utils.soil_hydraulics to the parameter system and converts the pressure
head to a potential. Expected values are computed from the owner values read at
run time, so the cases do not depend on any particular retention default.
"""

from __future__ import annotations

import pytest
import torch

from relic.contracts.contract_loader import load_contract_registry
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific import FUNCTOR_REGISTRY
from relic.utils.physical_constants import get_physical_constant
from relic.utils.soil_hydraulics import van_genuchten_pressure_head

_NAMESPACE = "hydrosphere.soil_moisture."


def _functor():
    from relic.scientific.hydrosphere.soil_moisture.retention import (
        SoilWaterRetention,
    )

    return SoilWaterRetention()


def _owner_values():
    manager = get_global_manager()
    return {
        name: float(manager.get(_NAMESPACE + name))
        for name in ("alpha_vg", "n_vg", "residual_water_content", "porosity")
    }


def _theta(fractions, dtype):
    # theta = theta_r + s (theta_s - theta_r) for effective saturations s [1];
    # theta, theta_r and theta_s [m3 m-3].
    owner = _owner_values()
    span = owner["porosity"] - owner["residual_water_content"]
    return torch.tensor(
        [[owner["residual_water_content"] + s * span] for s in fractions],
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_potential_is_the_shared_retention_law_in_megapascal(dtype):
    # References: van Genuchten (1980), Eqs. (2), (3), (21); Hillel (1998).
    # Mathematical expression: psi = rho_w g h(theta) with
    #   h = -(Se^(-1/m) - 1)^(1/n) / alpha, m = 1 - 1/n.
    # Parameter meanings: theta water content, Se effective saturation, alpha
    #   and n retention shape parameters, rho_w water density, g gravity.
    # Units: theta [m3 m-3]; h [m]; rho_w [kg m-3]; g [m s-2]; psi [MPa]
    #   (1 MPa = 1e6 Pa, SI prefix).
    # Domain/constraints: theta_r < theta <= theta_s; the governed call returns
    #   one REPLACE item and advances no state.
    # Sign convention: psi <= 0, zero at saturation.
    owner = _owner_values()
    theta = _theta([0.2, 0.4, 0.6, 0.8], dtype)
    functor = _functor()

    updates = functor({"soil_moisture": theta}, 0.0, 1800.0)

    item = updates["soil_water_potential"]
    assert isinstance(item, UpdateItem)
    assert item.mode == UpdateMode.REPLACE and item.unit == "MPa"
    head = van_genuchten_pressure_head(
        theta,
        owner["alpha_vg"],
        owner["n_vg"],
        owner["residual_water_content"],
        owner["porosity"],
    )
    expected = (
        head
        * float(get_physical_constant("gravity_acceleration"))
        * float(get_physical_constant("water_density"))
        / 1.0e6
    )
    assert item.value.dtype == dtype and item.value.shape == theta.shape
    assert torch.allclose(item.value, expected, rtol=1.0e-5, atol=0.0)
    assert bool((item.value < 0).all())
    assert bool((item.value[1:] > item.value[:-1]).all())


def test_potential_vanishes_at_saturation():
    porosity = get_global_manager().get(_NAMESPACE + "porosity")
    theta = torch.as_tensor(porosity, dtype=torch.float32).detach().reshape(1, 1)
    potential = _functor()({"soil_moisture": theta}, 0.0, 1800.0)[
        "soil_water_potential"
    ].value
    assert float(potential.detach()) == 0.0


def test_potential_is_differentiable_in_the_water_content():
    theta = _theta([0.3, 0.7], torch.float64).requires_grad_(True)
    potential = _functor()({"soil_moisture": theta}, 0.0, 1800.0)[
        "soil_water_potential"
    ].value
    (gradient,) = torch.autograd.grad(potential.sum(), theta)
    assert bool(torch.isfinite(gradient).all()) and bool((gradient > 0).all())


@pytest.mark.parametrize("fraction", [-0.05, 1.05])
def test_water_content_outside_the_retention_domain_is_refused(fraction):
    functor = _functor()
    with pytest.raises(Exception, match="residual_water_content, porosity"):
        functor({"soil_moisture": _theta([fraction], torch.float32)}, 0.0, 1800.0)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "'soil_moisture' not found"),
        ({"soil_moisture": 0.3}, "soil_moisture must be a torch.Tensor"),
        ({"soil_moisture": torch.tensor([[1]])}, "soil_moisture must be a floating"),
        (
            {"soil_moisture": torch.tensor([[float("nan")]])},
            "soil_moisture must contain only finite values",
        ),
    ],
)
def test_invalid_soil_moisture_input_is_refused(state, message):
    functor = _functor()
    with pytest.raises(Exception, match=message):
        functor(state, 0.0, 1800.0)


def test_process_is_registered_and_is_the_contract_writer():
    functor = _functor()
    assert FUNCTOR_REGISTRY["SoilWaterRetention"] == (
        type(functor).__module__ + ":" + type(functor).__name__
    )
    assert functor.metadata.input_variables == ["soil_moisture"]
    assert functor.metadata.output_variables == ["soil_water_potential"]
    registry = load_contract_registry()
    assert registry.can_write("soil_water_potential", functor.name)
    assert not registry.is_incrementable("soil_water_potential")



def test_unselected_soil_provider_keeps_existing_dependency_contract():
    functor = _functor()
    assert functor.get_dependencies() == []
    functor.set_dependencies(["legacy_theta_writer"])
    assert functor.get_dependencies() == ["legacy_theta_writer"]


@pytest.mark.parametrize("provider", ["bucket_soil_moisture", "richards_soil_moisture", "soil_water_transport"])
@pytest.mark.parametrize("lagged", [False, True], ids=["required", "lagged"])
def test_selected_soil_provider_keeps_declared_category(provider, lagged):
    from relic.core.dependency_graph import normalize_dependency_declaration
    from relic.scientific.hydrosphere.soil_moisture.retention import SoilWaterRetention
    functor = SoilWaterRetention(config={
        "soil_moisture_provider": provider, "soil_moisture_lagged": lagged,
    })
    initial = normalize_dependency_declaration(functor.get_dependencies(), owner=functor.name)
    assert set(initial.required) == (set() if lagged else {provider})
    assert set(initial.lagged) == ({provider} if lagged else set())
    functor.set_dependencies([provider, "independent_provider"])
    declared = normalize_dependency_declaration(functor.get_dependencies(), owner=functor.name)
    assert set(declared.lagged) == ({provider} if lagged else set())
    assert set(declared.required) == ({"independent_provider"} if lagged else {provider, "independent_provider"})
    assert set(declared.optional) == set()
    assert functor.metadata.input_variables == ["soil_moisture"]
    assert functor.metadata.output_variables == ["soil_water_potential"]


@pytest.mark.parametrize("selection", [
    {"soil_moisture_provider": True},
    {"soil_moisture_provider": 5},
    {"soil_moisture_provider": ""},
    {"soil_moisture_provider": " "},
    {"soil_moisture_provider": "soil_water_retention"},
    {"soil_moisture_provider": " bucket_soil_moisture "},
    {"soil_moisture_provider": "bucket_soil_moisture", "soil_moisture_lagged": 1},
    {"soil_moisture_provider": "bucket_soil_moisture", "soil_moisture_lagged": "true"},
    {"soil_moisture_provider": "bucket_soil_moisture", "soil_moisture_lagged": None},
    {"soil_moisture_lagged": True},
], ids=["bool_provider", "number_provider", "empty_provider", "blank_provider", "self_provider",
        "untrimmed_provider", "number_lag", "text_lag", "null_lag", "lag_without_provider"])
def test_invalid_soil_provider_or_lag_selection_rejected(selection):
    from relic.scientific.hydrosphere.soil_moisture.retention import SoilWaterRetention
    with pytest.raises((TypeError, ValueError), match="soil_moisture"):
        SoilWaterRetention(config=selection)


def test_source_declaration_does_not_change_theta_or_potential():
    from relic.scientific.hydrosphere.soil_moisture.retention import SoilWaterRetention
    theta = _theta([0.3, 0.7], torch.float64).requires_grad_(True)
    original = theta.detach().clone()
    reference = _functor()({"soil_moisture": theta}, 0.0, 1800.0)["soil_water_potential"]
    for provider, lagged in (("bucket_soil_moisture", False), ("bucket_soil_moisture", True),
                             ("richards_soil_moisture", True)):
        functor = SoilWaterRetention(config={
            "soil_moisture_provider": provider, "soil_moisture_lagged": lagged,
        })
        item = functor({"soil_moisture": theta}, 0.0, 1800.0)["soil_water_potential"]
        assert torch.equal(item.value, reference.value)
        assert item.value.dtype == theta.dtype and item.value.device == theta.device
        assert item.value.shape == theta.shape and item.value.requires_grad
        assert item.mode == reference.mode and item.unit == "MPa"
        assert item.per_timestep is False
        assert torch.equal(theta.detach(), original)


def _retention_source_simulator(provider, lagged):
    from relic.core.simulator import RELICSimulator
    from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
    from relic.scientific.hydrosphere.soil_moisture.retention import SoilWaterRetention
    simulator = RELICSimulator({
        "device": "cpu", "time_step": 60.0, "medium_interval_seconds": 120.0,
        "strict_cross_group_dependencies": True, "enable_auto_scale": False,
        "strict_scale_detection": True, "auto_report": False,
    })
    bucket = BucketSoilMoisture(config={"water_input_source": "precipitation_flux"})
    retention = SoilWaterRetention(config={
        "soil_moisture_provider": provider, "soil_moisture_lagged": lagged,
    })
    simulator.register_functor(bucket, execution_group="medium")
    return simulator, retention


@pytest.mark.parametrize("provider,lagged,accepted", [
    ("bucket_soil_moisture", True, True),
    ("bucket_soil_moisture", False, False),
    ("richards_soil_moisture", True, False),
], ids=["matching_lag", "required_cross_group", "wrong_provider_lag"])
def test_strict_registration_checks_matching_lagged_soil_provider(provider, lagged, accepted):
    simulator, retention = _retention_source_simulator(provider, lagged)
    if accepted:
        simulator.register_functor(retention, execution_group="fast")
        assert "soil_water_retention" in simulator.functors
    else:
        with pytest.raises(RuntimeError, match="Unacknowledged cross-group"):
            simulator.register_functor(retention, execution_group="fast")
        assert "soil_water_retention" not in simulator.functors


def test_lag_acknowledgement_preserves_actual_read_phase():
    # References: architecture_overview.md Sections4.3-4.4.1; Manabe(1969);
    # van Genuchten(1980), existing bucket and retention owners.
    # Mathematical expression: psi_fast,k=R(theta_committed_before_fast,k);
    # the medium bucket commits theta_new after that boundary's fast group.
    # Parameter meanings: theta is the actual committed water content; R the
    # existing retention map; k a native fast step.
    # Units: theta[m3 m-3], psi[MPa], precipitation[kg m-2 s-1], time/dt[s].
    # Domain/constraints: synthetic one-cell CPU fixed-step fixture; no new
    # scientific law, value cache, scheduling rule or physical-delay correction.
    # Sign convention: positive precipitation enters the existing bucket;
    # the diagnostic adds no water source, sink or state advance.
    manager = get_global_manager()
    theta0 = torch.full((1, 1), 0.5 * (
        float(manager.get(_NAMESPACE + "wilting_point"))
        + float(manager.get(_NAMESPACE + "field_capacity"))
    ), dtype=torch.float32)
    simulator, retention = _retention_source_simulator("bucket_soil_moisture", True)
    simulator.register_functor(retention, execution_group="fast")
    simulator.initialize_state({
        "soil_moisture": theta0.clone(),
        "precipitation_flux": torch.full_like(theta0, 0.001),
        "evapotranspiration_flux": torch.zeros_like(theta0),
    })
    expected0 = _functor()({"soil_moisture": theta0}, 0.0, 60.0)["soil_water_potential"].value
    simulator.step(60.0)
    assert torch.equal(simulator.state_manager.get_state("soil_water_potential", clone=True), expected0)
    simulator.step(60.0)
    theta1 = simulator.state_manager.get_state("soil_moisture", clone=True)
    assert not torch.equal(theta1, theta0)
    assert torch.equal(simulator.state_manager.get_state("soil_water_potential", clone=True), expected0)
    expected1 = _functor()({"soil_moisture": theta1}, 120.0, 60.0)["soil_water_potential"].value
    simulator.step(60.0)
    assert torch.equal(simulator.state_manager.get_state("soil_water_potential", clone=True), expected1)
