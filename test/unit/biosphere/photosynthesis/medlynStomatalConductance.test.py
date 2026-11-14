"""Conditional Medlyn conductance: equation, support, domain and differentiation."""

import importlib
import math

import pytest
import torch

from relic.core.functor import BiologicalProcessFunctor, ConflictResolutionPolicy
from relic.core.functor.functor_exceptions import ExecutionError
from relic.core.state import UpdateMode
from relic.parameters import get_global_manager
from relic.scientific import get_functor_path


@pytest.fixture
def model():
    module = importlib.import_module("relic.scientific.biosphere.photosynthesis.stomatal")
    return module.MedlynStomatalConductance(
        config={"surface_conditions": "leaf_surface_equals_canopy_air"}
    )


def state(leaf=10.0, lai=2.0, vpd=1000.0, co2=400.0, ppfd=500.0):
    # Independent Beer-Lambert support oracle, not the production helper.
    k = float(get_global_manager().get("biosphere.photosynthesis.canopy_light_extinction", required=True))
    effective = -math.expm1(-k * lai) / k
    return {key: torch.tensor(value, dtype=torch.float64) for key, value in {
        "net_photosynthesis": leaf * effective,
        "leaf_area_index": lai,
        "vapor_pressure_deficit": vpd,
        "atmospheric_co2_ppm": co2,
        "ppfd": ppfd,
    }.items()}


def test_positive_equation_units_governed_call_and_no_state_mutation(model):
    inputs = state()
    original = {key: value.clone() for key, value in inputs.items()}
    output = model(inputs, time=0.0, dt=1800.0)
    expected = 0.01 + 1.6 * (1 + 4.0 / math.sqrt(1.0)) * 10.0 / 400.0
    torch.testing.assert_close(output["stomatal_conductance"].value, torch.tensor(expected, dtype=torch.float64), rtol=1e-7, atol=0)
    assert output["stomatal_conductance"].mode == UpdateMode.REPLACE
    assert output["stomatal_conductance"].unit == "mol m-2 s-1"
    assert output["stomatal_conductance"].per_timestep is False
    for key in original:
        torch.testing.assert_close(inputs[key], original[key], rtol=0, atol=0)
    assert isinstance(model, BiologicalProcessFunctor)
    assert model.metadata.conflict_resolution_policy == ConflictResolutionPolicy.EXCLUSIVE
    assert get_functor_path("medlyn_stomatal_conductance").endswith("stomatal:MedlynStomatalConductance")


@pytest.mark.parametrize("leaf,ppfd,dark,nonpositive", [(-2., 500., False, True), (0., 500., False, True), (10., 0., True, False), (-2., 0., True, False)])
def test_residual_branches_are_explicit_and_signed_assimilation_survives(model, leaf, ppfd, dark, nonpositive):
    inputs = state(leaf=leaf, ppfd=ppfd)
    outputs = model(inputs, 0.0, 1800.0)
    assert float(outputs["stomatal_conductance"].value) == pytest.approx(0.01)
    assert bool(outputs["diagnostics.medlyn.dark_residual"]) is dark
    assert bool(outputs["diagnostics.medlyn.nonpositive_assimilation_residual"]) is nonpositive
    assert float(outputs["diagnostics.medlyn.leaf_net_assimilation"]) == pytest.approx(leaf)


@pytest.mark.parametrize("vpd", [-100., 0., 25., 50., 1000.])
def test_disclosed_vpd_floor_and_no_maximum_conductance_clipping(model, vpd):
    outputs = model(state(leaf=100., vpd=vpd), 0., 1.)
    expected = 0.01 + 1.6 * (1 + 4 / math.sqrt(max(vpd, 50.) / 1000)) * 100 / 400
    assert float(outputs["stomatal_conductance"].value) == pytest.approx(expected, rel=1e-7)
    assert bool(outputs["diagnostics.medlyn.vpd_floor_active"]) is (vpd < 50.)
    assert float(outputs["stomatal_conductance"].value) > 1.0


def test_leaf_support_is_invariant_across_lai_and_dt(model):
    values = [model(state(lai=lai), 0., dt)["stomatal_conductance"].value
              for lai, dt in [(0.5, 1.), (2., 1800.), (6., 3600.)]]
    for value in values[1:]:
        torch.testing.assert_close(value, values[0], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("key,value", [("leaf_area_index", 0.), ("leaf_area_index", -1.), ("leaf_area_index", float("nan")), ("net_photosynthesis", float("inf")), ("vapor_pressure_deficit", float("nan")), ("atmospheric_co2_ppm", 0.), ("atmospheric_co2_ppm", -1.), ("atmospheric_co2_ppm", float("inf")), ("ppfd", float("nan"))])
def test_invalid_required_inputs_raise_instead_of_defaulting(model, key, value):
    inputs = state()
    inputs[key] = torch.tensor(value, dtype=torch.float64)
    with pytest.raises(ValueError, match=key):
        model.forward(inputs, 0., 1.)


@pytest.mark.parametrize("key", ["net_photosynthesis", "leaf_area_index", "vapor_pressure_deficit", "atmospheric_co2_ppm", "ppfd"])
def test_every_input_is_required(model, key):
    inputs = state()
    inputs.pop(key)
    with pytest.raises(KeyError, match=key):
        model.forward(inputs, 0., 1.)


def test_condition_is_not_implicitly_claimed(model):
    for config in ({}, {"surface_conditions": "measured_leaf_surface"}):
        with pytest.raises(ValueError, match="surface_conditions"):
            type(model)(config=config)
    assert "leaf_surface_equals_canopy_air" in model.metadata.description


def test_leaf_input_and_parameter_gradients_are_preserved(model, monkeypatch):
    original = model.get_parameter_value
    g0 = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    g1 = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
    k = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    def parameters(key, **kwargs):
        assert kwargs["required"] is True
        if key == "coupling.scales.biosphere_atmosphere.g0":
            return g0
        if key == "coupling.scales.biosphere_atmosphere.g1":
            assert kwargs.get("units") == "kPa^0.5"
            return g1
        if key == "biosphere.photosynthesis.canopy_light_extinction":
            return k
        return original(key, **kwargs)

    monkeypatch.setattr(model, "get_parameter_value", parameters)
    inputs = state()
    for value in inputs.values():
        value.requires_grad_(True)
    output = model(inputs, 0., 1800.)["stomatal_conductance"].value
    targets = [inputs[key] for key in ("net_photosynthesis", "leaf_area_index", "vapor_pressure_deficit", "atmospheric_co2_ppm")] + [g0, g1, k]
    grads = torch.autograd.grad(output, targets)
    assert all(bool(torch.isfinite(grad)) for grad in grads)
    assert grads[0] > 0 and grads[1] < 0 and grads[2] < 0 and grads[3] < 0
    assert grads[4] == 1 and grads[5] > 0 and grads[6] > 0


def test_vectorized_branches_preserve_shape_and_declared_diagnostics(model):
    inputs = {key: torch.stack([state(leaf=leaf, ppfd=light)[key] for leaf, light in [(10., 500.), (-2., 500.), (10., 0.)]]) for key in state()}
    output = model(inputs, 0., 1800.)
    assert output["stomatal_conductance"].value.shape == (3,)
    torch.testing.assert_close(output["diagnostics.medlyn.dark_residual"], torch.tensor([False, False, True]))
    torch.testing.assert_close(output["diagnostics.medlyn.nonpositive_assimilation_residual"], torch.tensor([False, True, False]))
    assert set(output) == set(model.metadata.output_variables)


def test_non_scalar_input_shapes_cannot_create_cross_patch_broadcasts(model):
    inputs = state()
    inputs["net_photosynthesis"] = torch.tensor([10., 20.], dtype=torch.float64)
    inputs["leaf_area_index"] = torch.tensor([[1.], [2.]], dtype=torch.float64)
    with pytest.raises(ExecutionError, match="shape|support") as error:
        model(inputs, 0., 1800.)
    assert isinstance(error.value.__cause__, ValueError)


def test_scalars_expand_to_one_common_support_for_every_diagnostic(model):
    inputs = state()
    inputs["net_photosynthesis"] = torch.tensor([10., -2.], dtype=torch.float64)
    outputs = model(inputs, 0., 1800.)
    for name, value in outputs.items():
        tensor = value.value if name == "stomatal_conductance" else value
        assert tensor.shape == (2,)
    inputs = state()
    inputs["leaf_area_index"] = torch.tensor([1., 2.], dtype=torch.float64)
    outputs = model(inputs, 0., 1800.)
    assert all((value.value if name == "stomatal_conductance" else value).shape == (2,)
               for name, value in outputs.items())


def test_nonfinite_leaf_inversion_is_not_hidden_by_a_residual_branch(model):
    inputs = state()
    inputs["net_photosynthesis"] = torch.tensor(-1e308, dtype=torch.float64)
    inputs["leaf_area_index"] = torch.tensor(0.01, dtype=torch.float64)
    with pytest.raises(ExecutionError, match="leaf|assimilation") as error:
        model(inputs, 0., 1800.)
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("ground,ppfd", [(-1e308, 500.), (1e308, 0.)])
def test_inactive_formula_cannot_poison_residual_branch_gradients(model, ground, ppfd):
    inputs = state(ppfd=ppfd)
    inputs["net_photosynthesis"] = torch.tensor(ground, dtype=torch.float64)
    targets = [inputs[key].requires_grad_() for key in
               ("net_photosynthesis", "leaf_area_index", "vapor_pressure_deficit", "atmospheric_co2_ppm")]
    outputs = model(inputs, 0., 1.)
    assert torch.isfinite(outputs["diagnostics.medlyn.leaf_net_assimilation"])
    gradients = torch.autograd.grad(outputs["stomatal_conductance"].value, targets)
    assert all(bool(torch.isfinite(gradient)) and gradient == 0 for gradient in gradients)


def test_representable_active_result_does_not_overflow_before_co2_division(model):
    inputs = state()
    inputs["net_photosynthesis"] = torch.tensor(1e308, dtype=torch.float64)
    outputs = model(inputs, 0., 1.)
    leaf = outputs["diagnostics.medlyn.leaf_net_assimilation"]
    assert torch.isfinite(outputs["stomatal_conductance"].value)
    torch.testing.assert_close(outputs["stomatal_conductance"].value / leaf,
                               torch.tensor(0.02, dtype=torch.float64), rtol=1e-7, atol=0)


def test_injected_manager_owns_the_experiment_slope_and_gradient(model):
    from copy import deepcopy

    from relic.parameters import UnifiedParameterManager

    manager = UnifiedParameterManager()
    key = "coupling.scales.biosphere_atmosphere.g1"
    spec = deepcopy(manager.get_parameter(key))
    spec.value = spec.default = 4.45
    spec.dtype = "float64"
    spec.requires_grad = True
    manager.register_parameter(spec)
    conditional = type(model)(config={"surface_conditions": "leaf_surface_equals_canopy_air"},
                              parameter_manager=manager)
    output = conditional(state(), 0., 1800.)["stomatal_conductance"].value
    assert float(output.detach()) == pytest.approx(0.01 + 1.6 * 5.45 * 10 / 400, rel=1e-7)
    output.backward()
    assert float(manager.nn_parameters[key].grad) == pytest.approx(1.6 * 10 / 400, rel=1e-7)
    assert float(get_global_manager().get(key, required=True)) == pytest.approx(4.0)
    assert conditional.get_parameter_manager() is manager


def test_state_manager_commit_and_single_writer_exclusive_registration(model):
    from relic.core.simulator import RELICSimulator
    from relic.core.state import StateManager
    from relic.scientific.biosphere.stress.ozone_stress import OzoneStress

    state_manager = StateManager(device="cpu")
    state_manager.set_auto_report_enabled(False)
    state_manager.initialize_state(state())
    updates = model(state_manager.get_state(), 0.0, 1800.0)
    committed = state_manager.apply_updates(
        updates,
        dt=1800.0,
        update_sources={key: model.name for key in updates},
    )
    expected = 0.01 + 1.6 * (1 + 4.0 / math.sqrt(1.0)) * 10.0 / 400.0
    torch.testing.assert_close(
        committed["stomatal_conductance"],
        torch.tensor(expected, dtype=torch.float64),
        rtol=1e-7,
        atol=0,
    )
    diagnostics = state_manager.get_diagnostics()["current"]
    for key in model.metadata.output_variables:
        if key.startswith("diagnostics."):
            assert key in diagnostics
        else:
            assert key in committed

    # Downstream OzoneStress consumes committed stomatal_conductance without fallback.
    committed["o3_mixing_ratio"] = torch.tensor(60e-9, dtype=torch.float64)
    ozone = OzoneStress()
    ozone_out = ozone(committed, 0.0, 1800.0)
    assert float(ozone_out["ozone_uptake"].value.detach()) > 0.0
    assert 0.0 < float(ozone_out["ozone_stress_factor"].value.detach()) <= 1.0

    # Simulator enforces exclusive single-writer ownership of stomatal_conductance.
    sim = RELICSimulator(config={"strict_dependencies": True, "device": "cpu"})
    sim.register_functor(model)
    duplicate = type(model)(
        name="duplicate_stomatal_writer",
        config={"surface_conditions": "leaf_surface_equals_canopy_air"},
    )
    with pytest.raises(ValueError, match="stomatal_conductance"):
        sim.register_functor(duplicate)

