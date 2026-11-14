import inspect

import pytest
import torch

import relic.scientific.hydrosphere.transport.vertical_percolation as module
from relic.core.device_policy import is_cuda_available
from relic.core.state import UpdateItem
from relic.parameters.ops.nn_params import relocate_binding
from relic.scientific.hydrosphere.transport.vertical_percolation import (
    VerticalPercolationTransport,
)


EXPECTED_PARAMETER_KEYS = {
    "hydrosphere.transport.vertical_percolation.enable_temperature_effects",
    "hydrosphere.transport.vertical_percolation.hydraulic_model",
    "hydrosphere.soil_moisture.porosity",
    "hydrosphere.physical.viscosity_temperature_factor",
    "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
    "hydrosphere.soil_moisture.residual_water_content",
    "hydrosphere.soil_moisture.field_capacity",
    "hydrosphere.soil_moisture.wilting_point",
    "hydrosphere.soil_moisture.alpha_vg",
    "hydrosphere.soil_moisture.n_vg",
    "hydrosphere.soil_moisture.root_zone_depth",
    "hydrosphere.transport.vg_l",
    "hydrosphere.transport.bc_lambda",
    "hydrosphere.transport.ch_b_parameter",
    "hydrosphere.transport.drainage_time_constant",
    "hydrosphere.transport.soil_layer_thickness",
    "hydrosphere.numerical.temperature_bounds",
    "hydrosphere.numerical.saturation_epsilon",
    "hydrosphere.numerical.effective_saturation_bounds",
}


def test_vertical_percolation_has_one_parameter_access_path():
    source = inspect.getsource(module)

    assert "self.get_parameter_value(" in source
    assert "get_global_manager" not in source
    assert "_require_manager_param" not in source
    assert "manager.get(" not in source
    assert ".data =" not in source


def test_vertical_percolation_declares_exact_runtime_parameter_keys():
    functor = VerticalPercolationTransport(name="vp_parameter_contract")

    assert set(functor.required_parameters()) == EXPECTED_PARAMETER_KEYS
    assert not any(
        key.startswith("relic.physical.constants")
        for key in functor.required_parameters()
    )


def test_vertical_percolation_uses_physical_constant_ssot():
    source = inspect.getsource(module)

    assert "get_physical_constant('water_density')" in source
    assert "get_physical_constant('reference_temperature')" in source


def test_vertical_percolation_has_no_runtime_parameter_or_enable_mutators():
    assert not hasattr(VerticalPercolationTransport, "set_soil_parameters")
    assert not hasattr(VerticalPercolationTransport, "set_hydraulic_model")
    assert not hasattr(VerticalPercolationTransport, "enable_transport")


def test_saturated_water_content_is_the_formal_soil_porosity():
    """theta_s is hydrosphere.soil_moisture.porosity (as in Richards), not a default."""
    functor = VerticalPercolationTransport(name="vp_theta_s")
    manager = functor.get_parameter_manager()
    assert functor.theta_s_value == pytest.approx(
        float(manager.get("hydrosphere.soil_moisture.porosity", required=True))
    )


_HYDRAULIC_MODEL_KEY = "hydrosphere.transport.vertical_percolation.hydraulic_model"
_KS_KEY = "hydrosphere.soil_moisture.saturated_hydraulic_conductivity"
_FIELD_CAPACITY_KEY = "hydrosphere.soil_moisture.field_capacity"
_ALPHA_VG_KEY = "hydrosphere.soil_moisture.alpha_vg"
_N_VG_KEY = "hydrosphere.soil_moisture.n_vg"
_VG_L_KEY = "hydrosphere.transport.vg_l"
_BC_LAMBDA_KEY = "hydrosphere.transport.bc_lambda"
_CH_B_KEY = "hydrosphere.transport.ch_b_parameter"
_TAU_DRAIN_KEY = "hydrosphere.transport.drainage_time_constant"
HELD_BINDING_KEYS = (
    _KS_KEY,
    _FIELD_CAPACITY_KEY,
    _ALPHA_VG_KEY,
    _N_VG_KEY,
    _VG_L_KEY,
    _BC_LAMBDA_KEY,
    _CH_B_KEY,
    _TAU_DRAIN_KEY,
)


def _percolation_tensors(result) -> dict:
    """Every tensor of a percolation update, keyed by output name."""
    tensors = {}
    for name, update in result.items():
        value = update.value if isinstance(update, UpdateItem) else update
        members = value.items() if isinstance(value, dict) else [(None, value)]
        for member, item in members:
            if isinstance(item, torch.Tensor):
                tensors[name if member is None else f"{name}:{member}"] = item
    return tensors


def _draining_state() -> dict:
    """Soil wetter than field capacity at the reference temperature."""
    return {
        "soil_moisture": torch.tensor(0.40, dtype=torch.float32),
        "soil_temperature": torch.tensor(298.15, dtype=torch.float32),
    }


def _select_hydraulic_model(manager, hydraulic_model: str) -> None:
    spec = manager.get_parameter(_HYDRAULIC_MODEL_KEY)
    manager.set_value(
        _HYDRAULIC_MODEL_KEY,
        hydraulic_model,
        units=spec.units,
        validate=True,
        notify=False,
    )


@pytest.mark.parametrize(
    "hydraulic_model, key, value",
    [
        ("van_genuchten_mualem", _TAU_DRAIN_KEY, 3960.0),
        ("van_genuchten_mualem", _N_VG_KEY, 2.079),
        ("brooks_corey", _BC_LAMBDA_KEY, 0.22),
        ("clapp_hornberger", _CH_B_KEY, 5.929),
    ],
    ids=["drainage_time_constant", "n_vg", "bc_lambda", "ch_b_parameter"],
)
def test_vertical_percolation_follows_a_changed_binding(
    restore_global_parameter_manager,
    hydraulic_model,
    key,
    value,
):
    """Every value the drainage law reads from a learnable binding is read at
    use, so an instance built before the binding changed computes like one
    built afterwards."""
    # References: van Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898;
    #   Mualem (1976), Water Resour. Res. 12(3), 513-522; Brooks and Corey
    #   (1964), Hydrology Paper 3, Colorado State University; Clapp and
    #   Hornberger (1978), Water Resour. Res. 14(4), 601-604.
    # Mathematical expression: m = 1 - 1/n;
    #   Brooks-Corey K/Ks = Se^((2 + 3 lambda_p)/lambda_p);
    #   Clapp-Hornberger K/Ks = Se^(2 b + 3);
    #   F_drain = max(theta - theta_fc, 0) / tau_drain (K/Ks) rho_w dz.
    # Parameter meanings: n and m van Genuchten shape exponents, lambda_p the
    #   pore-size index, b the Clapp-Hornberger exponent, Se effective
    #   saturation, K and Ks unsaturated and saturated conductivity, theta and
    #   theta_fc water content and field capacity, tau_drain the drainage time
    #   constant, rho_w water density, dz layer thickness, F_drain drainage.
    # Units: n, m, lambda_p, b, Se, K/Ks [1]; theta, theta_fc [m3 m-3];
    #   tau_drain [s]; rho_w [kg m-3]; dz [m]; F_drain [kg m-2 s-1].
    # Domain/constraints: the new values lie inside the ParameterManager
    #   bounds, so none is clamped; theta > theta_fc; the soil temperature is
    #   the reference temperature, so the viscosity factor is exactly 1.
    # Sign convention: F_drain is a positive downward export.
    manager = restore_global_parameter_manager
    _select_hydraulic_model(manager, hydraulic_model)
    model = VerticalPercolationTransport(name="vp_built_before_the_change")
    manager.set_value(key, value, units=manager.get_parameter(key).units, validate=True)
    fresh = VerticalPercolationTransport(name="vp_built_after_the_change")

    expected = _percolation_tensors(fresh(_draining_state(), time=0.0, dt=3600.0))
    actual = _percolation_tensors(model(_draining_state(), time=0.0, dt=3600.0))
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )

    binding = manager.get_trainable_parameters()[key]
    current = float(binding.detach())
    assert current == pytest.approx(value)
    assert model.held_manager_parameters()[key] is binding

    saturation = actual[
        "diagnostics.vertical_percolation.effective_saturation"
    ].detach().cpu()
    relative_conductivity = actual[
        "diagnostics.vertical_percolation.relative_hydraulic_conductivity"
    ].detach().cpu()
    if key == _N_VG_KEY:
        torch.testing.assert_close(
            model.vg_m.detach(), 1.0 - 1.0 / binding.detach(), rtol=0.0, atol=0.0
        )
    elif key == _BC_LAMBDA_KEY:
        torch.testing.assert_close(
            relative_conductivity,
            saturation ** ((2.0 + 3.0 * current) / current),
            rtol=1.0e-5,
            atol=0.0,
        )
    elif key == _CH_B_KEY:
        torch.testing.assert_close(
            relative_conductivity,
            saturation ** (2.0 * current + 3.0),
            rtol=1.0e-5,
            atol=0.0,
        )
    else:
        excess = actual[
            "diagnostics.vertical_percolation.field_capacity_excess"
        ].detach().cpu()
        torch.testing.assert_close(
            actual["drainage_flux"].detach().cpu(),
            excess
            / current
            * relative_conductivity
            * model.water_density
            * model.soil_layer_thickness,
            rtol=1.0e-6,
            atol=0.0,
        )


def test_vertical_percolation_supports_repeated_backward_through_one_instance():
    """No autograd graph is carried from one call into the next: the instance
    stores no tensor derived from a binding, a second backward through it
    works and it gives the gradients of the first."""
    model = VerticalPercolationTransport(name="vp_repeated_backward")
    stored_graphs = sorted(
        name
        for name, attribute in vars(model).items()
        if isinstance(attribute, torch.Tensor) and attribute.grad_fn is not None
    )
    assert stored_graphs == []

    held = model.held_manager_parameters()
    assert set(HELD_BINDING_KEYS) <= set(held)
    differentiated = (_KS_KEY, _FIELD_CAPACITY_KEY, _N_VG_KEY, _VG_L_KEY)
    gradients = []
    try:
        for _ in range(2):
            for binding in held.values():
                binding.grad = None
            update = model(_draining_state(), time=0.0, dt=3600.0)
            total = (
                update["drainage_flux"].value.sum()
                + update[
                    "diagnostics.vertical_percolation.hydraulic_conductivity"
                ].value.sum()
            )
            total.backward()
            gradients.append(
                {key: held[key].grad.detach().cpu().clone() for key in differentiated}
            )
    finally:
        for binding in held.values():
            binding.grad = None

    for key in differentiated:
        assert torch.isfinite(gradients[0][key]).all(), key
        assert bool((gradients[0][key] != 0).all()), key
        torch.testing.assert_close(
            gradients[1][key], gradients[0][key], rtol=0.0, atol=0.0, msg=key
        )


@pytest.mark.skipif(
    not is_cuda_available(), reason="Held-binding relocation requires a CUDA device"
)
def test_vertical_percolation_holds_its_bindings_wherever_they_sit(
    restore_global_parameter_manager,
):
    """A held ParameterManager binding has one device, that of the latest
    registration of any Functor holding it. An instance built while its
    bindings sit on CUDA still holds them by reference, stores no converted
    copy, computes a CPU state on the CPU with unchanged values, and follows a
    later change of a binding."""
    manager = restore_global_parameter_manager
    bindings = manager.get_trainable_parameters()
    devices = {key: bindings[key].device for key in HELD_BINDING_KEYS}
    state = {
        "soil_moisture": torch.tensor([0.36, 0.40, 0.42], dtype=torch.float32),
        "soil_temperature": torch.tensor([288.15, 298.15, 303.15], dtype=torch.float32),
    }
    try:
        for key in HELD_BINDING_KEYS:
            relocate_binding(manager, key, "cpu")
        reference = VerticalPercolationTransport(name="vp_reference_binding_device")
        expected = _percolation_tensors(reference(dict(state), time=0.0, dt=3600.0))

        for key in HELD_BINDING_KEYS:
            relocate_binding(manager, key, "cuda")
        model = VerticalPercolationTransport(name="vp_built_after_relocation")
        held = model.held_manager_parameters()
        stored_graphs = sorted(
            name
            for name, attribute in vars(model).items()
            if isinstance(attribute, torch.Tensor) and attribute.grad_fn is not None
        )
        actual = _percolation_tensors(model(dict(state), time=0.0, dt=3600.0))

        spec = manager.get_parameter(_KS_KEY)
        manager.set_value(_KS_KEY, 1.1e-5, units=spec.units, validate=True)
        fresh = VerticalPercolationTransport(name="vp_built_after_the_change")
        changed_expected = _percolation_tensors(fresh(dict(state), time=0.0, dt=3600.0))
        changed_actual = _percolation_tensors(model(dict(state), time=0.0, dt=3600.0))
    finally:
        for key, device in devices.items():
            relocate_binding(manager, key, device)

    for key in HELD_BINDING_KEYS:
        assert held[key] is bindings[key], key
    assert stored_graphs == []
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        assert tensor.device.type == "cpu", name
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )
    conductivity = "diagnostics.vertical_percolation.hydraulic_conductivity"
    assert not torch.equal(changed_actual[conductivity], actual[conductivity])
    for name, tensor in changed_actual.items():
        torch.testing.assert_close(
            tensor, changed_expected[name], rtol=0.0, atol=0.0, msg=name
        )


@pytest.mark.parametrize(
    "hydraulic_model, key",
    [
        ("van_genuchten_mualem", _TAU_DRAIN_KEY),
        ("brooks_corey", _BC_LAMBDA_KEY),
        ("clapp_hornberger", _CH_B_KEY),
    ],
    ids=["drainage_time_constant", "bc_lambda", "ch_b_parameter"],
)
def test_vertical_percolation_gradient_reaches_time_constant_and_shape_exponents(
    restore_global_parameter_manager,
    hydraulic_model,
    key,
):
    """The drainage time constant and the Brooks-Corey and Clapp-Hornberger
    shape parameters are learnable bindings: the gradient of the drainage flux
    lands on the ParameterManager binding, equals the analytic derivative and
    is the same on a second backward through the same instance."""
    # References: Brooks and Corey (1964), Hydrology Paper 3, Colorado State
    #   University; Clapp and Hornberger (1978), Water Resour. Res. 14(4),
    #   601-604; Baydin et al. (2018), J. Mach. Learn. Res. 18(153), 1-43
    #   (reverse-mode differentiation).
    # Mathematical expression:
    #   F = max(theta - theta_fc, 0) / tau (K/Ks) rho_w dz;
    #   dF/dtau = -F / tau;
    #   Brooks-Corey K/Ks = Se^(2/lambda_p + 3):
    #     dF/dlambda_p = -2 F ln(Se) / lambda_p^2;
    #   Clapp-Hornberger K/Ks = Se^(2 b + 3): dF/db = 2 F ln(Se).
    # Parameter meanings: F drainage flux, theta and theta_fc water content
    #   and field capacity, tau the drainage time constant, K/Ks the relative
    #   conductivity, Se effective saturation, lambda_p the pore-size index, b
    #   the Clapp-Hornberger exponent, rho_w water density, dz layer thickness.
    # Units: F [kg m-2 s-1]; theta, theta_fc [m3 m-3]; tau [s]; rho_w [kg m-3];
    #   dz [m]; Se, K/Ks, lambda_p, b [1]; dF/dtau [kg m-2 s-2].
    # Domain/constraints: theta > theta_fc and 0 < Se < 1 inside the saturation
    #   bounds, so no clamp is active; the soil temperature is the reference
    #   temperature, so the viscosity factor is exactly 1.
    # Sign convention: F is a positive downward export; a longer tau and a
    #   larger b reduce it, a larger lambda_p increases it (ln(Se) < 0).
    manager = restore_global_parameter_manager
    _select_hydraulic_model(manager, hydraulic_model)
    model = VerticalPercolationTransport(name="vp_scalar_parameter_gradient")
    binding = manager.get_trainable_parameters()[key]
    held = model.held_manager_parameters()
    assert held[key] is binding

    gradients = []
    try:
        for _ in range(2):
            for other in held.values():
                other.grad = None
            update = model(_draining_state(), time=0.0, dt=3600.0)
            update["drainage_flux"].value.sum().backward()
            assert binding.grad is not None, key
            gradients.append(binding.grad.detach().cpu().clone())
    finally:
        for other in held.values():
            other.grad = None

    flux = update["drainage_flux"].value.detach().cpu().to(torch.float64)
    saturation = update[
        "diagnostics.vertical_percolation.effective_saturation"
    ].value.detach().cpu().to(torch.float64)
    current = float(binding.detach())
    if key == _TAU_DRAIN_KEY:
        expected = -flux / current
    elif key == _BC_LAMBDA_KEY:
        expected = -2.0 * flux * torch.log(saturation) / current**2
    else:
        expected = 2.0 * flux * torch.log(saturation)
    assert float(flux) > 0.0
    assert 0.0 < float(saturation) < 1.0
    torch.testing.assert_close(
        gradients[0].to(torch.float64).reshape(()),
        expected.reshape(()),
        rtol=1.0e-5,
        atol=0.0,
    )
    torch.testing.assert_close(gradients[1], gradients[0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "hydraulic_model, key, value",
    [
        ("brooks_corey", _BC_LAMBDA_KEY, 0.137),
        ("clapp_hornberger", _CH_B_KEY, 7.31),
    ],
    ids=["bc_lambda", "ch_b_parameter"],
)
def test_vertical_percolation_power_closures_are_rounded_once(
    restore_global_parameter_manager,
    hydraulic_model,
    key,
    value,
):
    """The Brooks-Corey and Clapp-Hornberger relative conductivities are the
    float64 power of the saturation, rounded once to the state dtype: a
    float32 state gives the float64 result to within float32 rounding of the
    product and quotient with Ks, for steep and flat exponents."""
    # References: Brooks and Corey (1964); Clapp and Hornberger (1978);
    #   Goldberg (1991), ACM Comput. Surv. 23(1), 5-48 (rounding error of
    #   floating-point operations).
    # Mathematical expression: K/Ks = Se^e, e = 2/lambda_p + 3 or e = 2 b + 3;
    #   a relative error delta of e changes Se^e by the factor
    #   exp(e ln(Se) delta), about 1 + e ln(Se) delta.
    # Parameter meanings: Se effective saturation, e the closure exponent,
    #   lambda_p the pore-size index, b the Clapp-Hornberger exponent, delta
    #   the relative rounding error of e.
    # Units: Se, e, lambda_p, b, K/Ks, delta [1].
    # Domain/constraints: 0 < Se < 1 inside the saturation bounds; the
    #   parameter values lie inside the ParameterManager bounds and give a
    #   steep exponent (about 17.6) that float32 does not represent exactly;
    #   rtol 4e-7 is about three float32 spacings (one rounding each for the
    #   power, the product with Ks and the quotient by Ks).
    # Sign convention: none (a dimensionless ratio in (0, 1)).
    manager = restore_global_parameter_manager
    _select_hydraulic_model(manager, hydraulic_model)
    manager.set_value(key, value, units=manager.get_parameter(key).units, validate=True)
    model = VerticalPercolationTransport(name="vp_power_closure")
    binding = manager.get_trainable_parameters()[key]
    assert float(binding.detach()) == pytest.approx(value)
    theta = torch.linspace(
        model.theta_r_value + 0.02, model.theta_s_value - 0.002, 41, dtype=torch.float32
    )
    state = {
        "soil_moisture": theta,
        "soil_temperature": torch.full_like(theta, 298.15),
    }
    with torch.no_grad():
        update = model(state, time=0.0, dt=3600.0)
    saturation = update[
        "diagnostics.vertical_percolation.effective_saturation"
    ].value.cpu()
    relative_conductivity = update[
        "diagnostics.vertical_percolation.relative_hydraulic_conductivity"
    ].value.cpu()
    current = float(binding.detach())
    exponent = (2.0 + 3.0 * current) / current if key == _BC_LAMBDA_KEY else 2.0 * current + 3.0
    expected = torch.pow(saturation.to(torch.float64), exponent)

    assert relative_conductivity.dtype == torch.float32
    torch.testing.assert_close(
        relative_conductivity.to(torch.float64), expected, rtol=4.0e-7, atol=1.0e-38
    )
