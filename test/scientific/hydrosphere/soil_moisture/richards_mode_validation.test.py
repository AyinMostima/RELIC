import warnings
from pathlib import Path

import pytest
import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.core.device_policy import is_cuda_available
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import ParameterSpec, get_global_manager
from relic.parameters.ops.nn_params import relocate_binding
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.utils.physical_constants import get_physical_constant


@pytest.fixture
def parameter_manager_snapshot():
    manager = get_global_manager()
    snapshot = manager.snapshot()
    yield manager
    manager.restore(snapshot)


def _minimal_state() -> dict:
    return {
        "soil_moisture": torch.full((10,), 0.3, dtype=torch.float32),
        "infiltration_flux": torch.tensor(0.0, dtype=torch.float32),
        "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float32),
    }


def test_richards_obsolete_compute_mode_does_not_select_runtime_path(parameter_manager_snapshot):
    manager = parameter_manager_snapshot
    manager.register_parameter(ParameterSpec(
        key="hydrosphere.soil_moisture.richards_compute_mode",
        value="simplified",
        units="dimensionless",
        dtype="str",
        description="Removed Richards compute mode supplied by a stale configuration", bounds=(-float("inf"), float("inf")), category="model_parameter",
    ))

    model = RichardsSoilMoisture(name="richards_removed_mode_key")
    result = model.forward(_minimal_state(), time=0.0, dt=60.0)
    diagnostics = result["diagnostics.soil_moisture_richards"]

    assert model.runtime_path == "canonical_iphs_mixed_form_richards"
    assert "runtime_path" not in diagnostics
    assert "compute_mode" not in diagnostics
    assert not hasattr(model, "compute_mode")
    assert not hasattr(model, "enable_simplified_mode")
    assert not hasattr(model, "_compute_simplified_solution")


@pytest.mark.parametrize("obsolete_mode", ["full", "simplified", "invalid_mode"])
def test_richards_has_one_canonical_runtime_under_obsolete_mode_values(
    parameter_manager_snapshot,
    obsolete_mode,
):
    manager = parameter_manager_snapshot
    reference = RichardsSoilMoisture(name=f"richards_reference_{obsolete_mode}")
    reference_result = reference.forward(_minimal_state(), time=0.0, dt=60.0)
    manager.register_parameter(ParameterSpec(
        key="hydrosphere.soil_moisture.richards_compute_mode",
        value=obsolete_mode,
        units="dimensionless",
        dtype="str",
        description="Removed Richards compute mode supplied by a stale configuration", bounds=(-float("inf"), float("inf")), category="model_parameter",
    ))

    model = RichardsSoilMoisture(name=f"richards_canonical_{obsolete_mode}")
    result = model.forward(_minimal_state(), time=0.0, dt=60.0)
    assert isinstance(result, dict)
    assert "diagnostics.soil_moisture_richards" in result
    assert model.runtime_path == "canonical_iphs_mixed_form_richards"
    # The obsolete mode value selects no other runtime: the update is unchanged.
    assert torch.equal(
        result["soil_moisture"].value, reference_result["soil_moisture"].value
    )


def test_richards_exposes_relative_moisture_as_namespaced_observation():
    """Richards reports soil wetness without claiming plant water stress.

    References:
    - van Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898.
    - Feddes et al. (1978), Simulation of Field Water Use and Crop Yield.

    Mathematical expression:
    - f_rel = clamp((theta_bar - theta_wp) / (theta_fc - theta_wp), 0, 1)

    Parameter meanings:
    - theta_bar: profile-mean volumetric water content [m3 m-3].
    - theta_fc: diagnostic field-capacity threshold [m3 m-3].
    - theta_wp: diagnostic wilting threshold [m3 m-3].
    - f_rel: field-capacity-normalized relative moisture observation [1].
    """
    model = RichardsSoilMoisture(name="richards_relative_moisture_contract")
    result = model.forward(_minimal_state(), time=0.0, dt=60.0)

    expected_outputs = {
        "soil_moisture",
        "diagnostics.soil_moisture_richards.soil_moisture_change_rate",
        "diagnostics.soil_moisture_richards.water_flux_profile",
        "soil_matric_potential",
        "diagnostics.soil_moisture_richards.relative_moisture",
        "drainage_flux",
        "diagnostics.soil_moisture_richards",
    }
    assert set(result) == expected_outputs
    assert set(model.metadata.output_variables) == expected_outputs
    assert "water_stress_factor" not in result

    update_contract = {
        "soil_moisture": "m3 m-3",
        "diagnostics.soil_moisture_richards.soil_moisture_change_rate": "m3 m-3 s-1",
        "diagnostics.soil_moisture_richards.water_flux_profile": "kg m-2 s-1",
        "soil_matric_potential": "Pa",
        "drainage_flux": "kg m-2 s-1",
        "diagnostics.soil_moisture_richards.relative_moisture": "dimensionless",
    }
    for output_name, expected_unit in update_contract.items():
        update = result[output_name]
        assert isinstance(update, UpdateItem)
        assert update.mode is UpdateMode.REPLACE
        assert update.per_timestep is False
        assert update.unit == expected_unit
        assert update.source_functor == model.name

    relative_moisture = result[
        "diagnostics.soil_moisture_richards.relative_moisture"
    ]
    assert isinstance(relative_moisture, UpdateItem)
    assert relative_moisture.mode is UpdateMode.REPLACE
    assert relative_moisture.per_timestep is False
    assert relative_moisture.unit == "dimensionless"
    assert relative_moisture.source_functor == model.name

    diagnostics = result["diagnostics.soil_moisture_richards"]

    # References: van Genuchten (1980) and Celia et al. (1990).
    # Mathematical expression: psi_m = rho_w * g * h(theta).
    # Parameter meanings: h is pressure head [m], rho_w is water density
    # [kg m-3], g is gravitational acceleration [m s-2], and psi_m is
    # matric pressure [Pa]. This read-only profile is an observation and
    # therefore has no water source or sink route.
    matric_potential = result["soil_matric_potential"]
    pressure_head = diagnostics["pressure_head"]
    expected_matric_potential = (
        pressure_head
        * get_physical_constant("gravity_acceleration")
        * get_physical_constant("water_density")
    )
    assert matric_potential.value.shape == pressure_head.shape
    torch.testing.assert_close(matric_potential.value, expected_matric_potential)
    assert torch.isfinite(matric_potential.value).all()
    assert torch.all(matric_potential.value <= 0.0)
    assert "matric_potential" not in result
    assert "matric_potential" not in diagnostics
    assert "soil_water_potential" not in result

    average_moisture = diagnostics["average_soil_moisture"]
    field_capacity = model.field_capacity_fraction * model.theta_saturated
    wilting_point = model.wilting_point_fraction * model.theta_residual
    expected_relative_moisture = torch.clamp(
        (average_moisture - wilting_point) / (field_capacity - wilting_point),
        min=0.0,
        max=1.0,
    )
    assert torch.allclose(relative_moisture.value, expected_relative_moisture)
    assert torch.isfinite(relative_moisture.value).all()


def test_richards_open_boundary_uses_explicit_water_ledger_without_base_warning():
    """The open Richards column reports its own finite water-balance residual.

    References:
    - Celia et al. (1990), Water Resources Research 26, 1483-1496.

    Mathematical expression:
    - epsilon_m = abs(Delta W - (q_top - q_bottom) dt) / W_old

    Parameter meanings:
    - W: water storage depth [m].
    - q_top and q_bottom: signed boundary water fluxes [m s-1].
    - epsilon_m: relative open-boundary ledger residual [1].
    """
    model = RichardsSoilMoisture(name="richards_open_water_ledger")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = model(_minimal_state(), time=0.0, dt=60.0)

    user_warnings = [
        item for item in captured if issubclass(item.category, UserWarning)
    ]
    assert not user_warnings
    assert set(model.metadata.conservation_laws) == {"mass"}
    diagnostics = result["diagnostics.soil_moisture_richards"]
    residual = diagnostics["mass_conservation_error"]
    assert torch.isfinite(residual)
    assert residual >= 0
    assert model.check_conservation_laws(_minimal_state(), result) == {
        "mass": pytest.approx(float(residual.item()))
    }

    diagnostic_key = "diagnostics.soil_moisture_richards"
    perturbed = dict(result)
    perturbed_diagnostics = dict(diagnostics)
    perturbed_diagnostics["mass_conservation_error"] = torch.tensor(-0.125)
    perturbed[diagnostic_key] = perturbed_diagnostics
    assert model.check_conservation_laws(_minimal_state(), perturbed) == {
        "mass": pytest.approx(0.125)
    }

    missing_diagnostics = dict(result)
    missing_diagnostics.pop(diagnostic_key)
    with pytest.raises(KeyError, match="Missing conservation diagnostics"):
        model.check_conservation_laws(_minimal_state(), missing_diagnostics)

    missing_residual = dict(result)
    missing_residual[diagnostic_key] = {
        key: value
        for key, value in diagnostics.items()
        if key != "mass_conservation_error"
    }
    with pytest.raises(KeyError, match="mass_conservation_error"):
        model.check_conservation_laws(_minimal_state(), missing_residual)

    nonfinite = dict(result)
    nonfinite_diagnostics = dict(diagnostics)
    nonfinite_diagnostics["mass_conservation_error"] = torch.tensor(float("nan"))
    nonfinite[diagnostic_key] = nonfinite_diagnostics
    with pytest.raises(ValueError, match="Richards mass residual must be finite"):
        model.check_conservation_laws(_minimal_state(), nonfinite)
    assert {port.name for port in model.ports()} == {
        "soil_infiltration_boundary",
        "soil_evapotranspiration_boundary",
        "soil_capillary_boundary",
        "richards_drainage_flux",
    }
    infiltration_port = next(
        port for port in model.ports() if port.flow_var == "infiltration_flux"
    )
    assert infiltration_port.flow_space.units == "kg m-2 s-1"
    assert infiltration_port.flow_space.complex_id == "surface_soil_water_interface"
    assert infiltration_port.flow_space.physical_domain == "surface_soil_interface"
    assert infiltration_port.positive_power_convention == "into_system"
    evapotranspiration_port = next(
        port
        for port in model.ports()
        if port.flow_var == "evapotranspiration_flux"
    )
    assert evapotranspiration_port.flow_space.units == "kg m-2 s-1"
    assert (
        evapotranspiration_port.flow_space.complex_id
        == "surface_soil_water_interface"
    )
    assert (
        evapotranspiration_port.flow_space.physical_domain
        == "surface_soil_interface"
    )
    assert evapotranspiration_port.positive_power_convention == "out_of_system"


@pytest.mark.parametrize(
    "state, missing_name",
    [
        (
            {
                "soil_moisture": torch.full((10,), 0.3),
                "evapotranspiration_flux": torch.tensor(0.0),
                "surface_water_flux": torch.tensor(0.0),
            },
            "infiltration_flux",
        ),
        (
            {
                "soil_moisture": torch.full((10,), 0.3),
                "infiltration_flux": torch.tensor(0.0),
                "surface_water_flux": torch.tensor(0.0),
            },
            "evapotranspiration_flux",
        ),
    ],
)
def test_richards_rejects_old_surface_flux_and_missing_canonical_inputs(
    state,
    missing_name,
):
    """Old surface flux cannot replace either canonical mass boundary input."""
    with pytest.raises(ValueError, match=missing_name):
        RichardsSoilMoisture().forward(state, time=0.0, dt=60.0)


@pytest.mark.parametrize(
    "soil_moisture",
    [
        torch.tensor(0.3),
        torch.full((2,), 0.3),
        torch.full((2, 5), 0.3),
        [0.3] * 10,
    ],
)
def test_richards_requires_an_exact_tensor_profile(soil_moisture):
    """The state owner cannot infer, average, or tensorize a soil profile."""
    state = {
        "soil_moisture": soil_moisture,
        "infiltration_flux": torch.tensor(0.0),
        "evapotranspiration_flux": torch.tensor(0.0),
    }
    with pytest.raises(ValueError, match="one-dimensional Tensor profile"):
        RichardsSoilMoisture().forward(state, time=0.0, dt=60.0)


@pytest.mark.parametrize("key", ["infiltration_flux", "evapotranspiration_flux"])
def test_richards_requires_tensor_boundary_fluxes(key):
    """Boundary forcing cannot be silently tensorized from Python values."""
    state = _minimal_state()
    state[key] = 0.0
    with pytest.raises(ValueError, match=f"{key} must be a scalar Tensor"):
        RichardsSoilMoisture().forward(state, time=0.0, dt=60.0)


def test_richards_matric_potential_has_one_canonical_scm_contract():
    """The derived Pa profile has one non-incrementable writer contract."""
    project_root = Path(__file__).resolve().parents[4]
    contract_path = (
        project_root
        / "relic"
        / "contracts"
        / "variables"
        / "hydrosphere"
        / "matric_potential.yaml"
    )
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(contract_path)

    contract = registry.get_contract("soil_matric_potential")
    assert contract is not None
    assert registry.get_contract("matric_potential") is None
    assert not registry.is_alias("matric_potential")
    assert contract.kind == "diagnostic"
    assert contract.udunits == "Pa"
    assert contract.owner == "richards_soil_moisture"
    assert contract.incrementable is False
    assert contract.aliases == []
    assert contract.writers == ["richards_soil_moisture"]
    assert registry.can_write(
        "soil_matric_potential",
        "richards_soil_moisture",
    )
    assert not registry.can_write("soil_matric_potential", "unauthorized_writer")


def test_richards_routes_matric_potential_through_state_manager_with_provenance():
    """The canonical Pa profile reaches StateManager through one writer."""
    simulator = RELICSimulator(
        {
            "conservation_checks": False,
            "strict_dependencies": False,
            "strict_scale_detection": False,
            "device": "cpu",
            "time_step": 60.0,
        }
    )
    simulator.initialize_state(
        {
            "soil_moisture": torch.full((10,), 0.3, dtype=torch.float32),
            "infiltration_flux": torch.tensor(0.0, dtype=torch.float32),
            "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float32),
        }
    )
    model = RichardsSoilMoisture(name="richards_soil_moisture")
    simulator.register_functor(model)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        updates = simulator.step(dt=60.0)

    state = simulator.state_manager.get_state()
    actual = state["soil_matric_potential"]
    pressure_head = model._compute_pressure_head(state["soil_moisture"])
    expected = (
        pressure_head
        * get_physical_constant("gravity_acceleration")
        * get_physical_constant("water_density")
    )
    routed_update = updates["soil_matric_potential"]
    if isinstance(routed_update, UpdateItem):
        routed_update = routed_update.value

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(routed_update, expected)
    provenance = simulator.get_variable_provenance("soil_matric_potential")
    assert provenance
    assert {item["functor"] for item in provenance} == {
        "richards_soil_moisture"
    }
    assert {item["mode"] for item in provenance} == {"UpdateMode.REPLACE"}
    assert str(simulator.get_effective_device("richards_soil_moisture")) == "cpu"


def test_richards_diagnostics_are_numeric_and_profiles_are_plain_tensors():
    """Stored diagnostics are real numbers or real tensors: the configuration-derived
    solver strings are not diagnostics (the solver is the configured
    enable_newton_raphson; runtime_path is the Functor attribute), and the layer
    profiles are plain tensors of the profile shape."""
    model = RichardsSoilMoisture(name="richards_numeric_diagnostics")
    result = model.forward(_minimal_state(), time=0.0, dt=60.0)
    diagnostics = result["diagnostics.soil_moisture_richards"]

    assert "runtime_path" not in diagnostics
    assert "nonlinear_solver" not in diagnostics
    for key, value in diagnostics.items():
        assert isinstance(value, (torch.Tensor, bool, int, float)), key
        if isinstance(value, torch.Tensor):
            assert not value.is_complex(), key
    for key in ("pressure_head", "hydraulic_conductivity"):
        assert type(diagnostics[key]) is torch.Tensor
        assert diagnostics[key].shape == (model.soil_layers,)
    assert type(diagnostics["vertical_flux"]) is torch.Tensor
    assert diagnostics["vertical_flux"].shape == (model.soil_layers + 1,)
    torch.testing.assert_close(
        diagnostics["vertical_flux"][1:],
        result["diagnostics.soil_moisture_richards.water_flux_profile"].value,
    )


def _result_tensors(result) -> dict:
    """Every tensor of a Richards update, keyed by output and diagnostic name."""
    tensors = {}
    for name, update in result.items():
        value = update.value if isinstance(update, UpdateItem) else update
        members = value.items() if isinstance(value, dict) else [(None, value)]
        for member, item in members:
            if isinstance(item, torch.Tensor):
                tensors[name if member is None else f"{name}:{member}"] = item
    return tensors


@pytest.mark.skipif(
    not is_cuda_available(), reason="Held-binding relocation requires a CUDA device"
)
@pytest.mark.parametrize("built", ["before_relocation", "after_relocation"])
@pytest.mark.parametrize("newton", [False, True], ids=["picard", "newton"])
@pytest.mark.parametrize(
    "head_boundaries", [False, True], ids=["flux_boundaries", "head_boundaries"]
)
def test_richards_computes_on_the_state_device_wherever_its_bindings_sit(
    parameter_manager_snapshot,
    built,
    newton,
    head_boundaries,
):
    """A held ParameterManager binding has one device, that of the latest
    registration of any Functor holding it. The kernel computes on the device
    of its state: after another simulator registered a Richards process on
    CUDA, a CPU state gives the unchanged update with every tensor on the CPU."""
    manager = parameter_manager_snapshot
    manager.set_value(
        "hydrosphere.soil_moisture.richards_enable_newton_raphson",
        newton,
        units="dimensionless",
        validate=True,
        notify=False,
    )

    def build(name: str) -> RichardsSoilMoisture:
        model = RichardsSoilMoisture(name=name)
        if head_boundaries:
            model.set_boundary_conditions(
                top={"type": "prescribed_head", "value": -0.1},
                bottom={"type": "prescribed_head", "value": -2.0},
            )
        return model

    holder = RichardsSoilMoisture(name="richards_soil_moisture")
    bindings = holder.held_manager_parameters()
    assert bindings
    devices = {key: binding.device for key, binding in bindings.items()}
    try:
        for key in bindings:
            relocate_binding(manager, key, "cpu")
        reference = build("richards_reference_binding_device")
        expected = _result_tensors(reference(_minimal_state(), time=0.0, dt=60.0))
        early = build("richards_built_before_relocation")

        RELICSimulator({"device": "cuda"}).register_functor(holder)
        assert all(binding.device.type == "cuda" for binding in bindings.values())
        model = (
            early
            if built == "before_relocation"
            else build("richards_built_after_relocation")
        )
        actual = _result_tensors(model(_minimal_state(), time=0.0, dt=60.0))
    finally:
        for key, device in devices.items():
            relocate_binding(manager, key, device)

    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        assert tensor.device.type == "cpu", name
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )


def _graded_state() -> dict:
    """A drying-upward profile with infiltration, so every hydraulic term is active."""
    return {
        "soil_moisture": torch.linspace(0.22, 0.34, 10, dtype=torch.float32),
        "infiltration_flux": torch.tensor(1.0e-4, dtype=torch.float32),
        "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float32),
    }


@pytest.mark.parametrize(
    "key, units, value",
    [
        ("hydrosphere.soil_moisture.n_vg", "dimensionless", 1.2),
        ("hydrosphere.soil_moisture.soil_depth", "m", 2.0),
    ],
    ids=["n_vg", "soil_depth"],
)
def test_richards_derived_hydraulic_values_follow_a_changed_binding(
    parameter_manager_snapshot,
    key,
    units,
    value,
):
    """The retention exponent and the layer thickness are derived from the held
    bindings at use, so an instance built before a binding changed computes
    with the same consistent values as one built afterwards."""
    # References: van Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898;
    #   Mualem (1976), Water Resour. Res. 12(3), 513-522; Celia et al. (1990),
    #   Water Resour. Res. 26(7), 1483-1496 (uniform node spacing).
    # Mathematical expression: m = 1 - 1/n; dz = L / N;
    #   Se = (theta - theta_r) / (theta_s - theta_r);
    #   h(theta) = -[Se^(-1/m) - 1]^(1/n) / alpha;
    #   K(theta) = Ks Se^l [1 - (1 - Se^(1/m))^m]^2.
    # Parameter meanings: n and m retention shape exponents, alpha the inverse
    #   air-entry scale, theta_r and theta_s residual and saturated water
    #   content, Ks saturated conductivity, l tortuosity, L profile depth, N
    #   the number of layers, dz the layer thickness.
    # Units: theta, theta_r, theta_s [m3 m-3]; h, L, dz [m]; alpha [m-1];
    #   K, Ks [m s-1]; n, m, l, Se, N [1].
    # Domain/constraints: n > 1; theta_r < theta < theta_s; the new values lie
    #   inside the ParameterManager bounds, so no value is clamped.
    # Sign convention: h <= 0 in unsaturated soil; K >= 0; depth positive downward.
    manager = parameter_manager_snapshot
    model = RichardsSoilMoisture(name="richards_built_before_the_change")
    manager.set_value(key, value, units=units, validate=True)
    fresh = RichardsSoilMoisture(name="richards_built_after_the_change")

    n_vg = model.n_vg.detach()
    depth = model.total_soil_depth.detach()
    changed = n_vg if key.endswith("n_vg") else depth
    assert float(changed) == pytest.approx(value)
    torch.testing.assert_close(
        model.m_vg.detach(), 1.0 - 1.0 / n_vg, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        model.layer_thickness.detach(), depth / model.soil_layers, rtol=0.0, atol=0.0
    )

    theta = torch.linspace(0.10, 0.40, 10, dtype=torch.float32)
    for kernel in (
        "_compute_pressure_head",
        "_compute_hydraulic_conductivity",
        "_compute_dh_dtheta",
        "_compute_dk_dtheta",
    ):
        torch.testing.assert_close(
            getattr(model, kernel)(theta),
            getattr(fresh, kernel)(theta),
            rtol=0.0,
            atol=0.0,
            msg=kernel,
        )
    theta_residual = model.theta_residual.to(theta.device)
    theta_saturated = model.theta_saturated.to(theta.device)
    exponent_n = n_vg.to(theta.device)
    exponent_m = 1.0 - 1.0 / exponent_n
    saturation = (theta - theta_residual) / (theta_saturated - theta_residual)
    expected_head = -(
        (saturation.pow(-1.0 / exponent_m) - 1.0).pow(1.0 / exponent_n)
    ) / model.alpha_vg.detach().to(theta.device)
    torch.testing.assert_close(model._compute_pressure_head(theta), expected_head)

    expected = _result_tensors(fresh(_graded_state(), time=0.0, dt=60.0))
    actual = _result_tensors(model(_graded_state(), time=0.0, dt=60.0))
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )


def test_richards_supports_repeated_backward_through_one_instance():
    """No autograd graph is carried from one call into the next: the instance
    stores no tensor derived from a binding, a second backward through it
    works and it gives the gradients of the first."""
    model = RichardsSoilMoisture(name="richards_repeated_backward")
    stored_graphs = sorted(
        name
        for name, attribute in vars(model).items()
        if isinstance(attribute, torch.Tensor) and attribute.grad_fn is not None
    )
    assert stored_graphs == []

    held = model.held_manager_parameters()
    differentiated = (
        "hydrosphere.soil_moisture.n_vg",
        "hydrosphere.soil_moisture.soil_depth",
    )
    gradients = []
    try:
        for _ in range(2):
            for binding in held.values():
                binding.grad = None
            update = model(_graded_state(), time=0.0, dt=60.0)
            update["soil_moisture"].value.sum().backward()
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
