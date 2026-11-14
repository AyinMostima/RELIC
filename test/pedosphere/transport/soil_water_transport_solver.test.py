"""L1 checks for SoilWaterTransportFunctor Richards solver outputs."""

import inspect
import warnings
from pathlib import Path

import pytest
import torch

from relic.contracts.contract_loader import ContractRegistry
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.transport.SoilWaterTransportFunctor import (
    SoilWaterTransportFunctor,
)


EXPECTED_OUTPUTS = {
    "soil_moisture",
    "diagnostics.soil_water_transport.clipping_residual",
    "diagnostics.soil_water_transport.water_balance_residual",
    "diagnostics.soil_water_transport.darcy_flux",
    "diagnostics.soil_water_transport.richards_iterations",
}

EXPECTED_INPUTS = {
    "soil_moisture",
    "pressure_head",
    "infiltration_flux",
    "evapotranspiration_flux",
}


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _value(output: dict[str, UpdateItem], key: str) -> torch.Tensor:
    item = output[key]
    assert isinstance(item, UpdateItem)
    return torch.as_tensor(item.value)


def _canonical_state(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "soil_moisture": theta,
        "infiltration_flux": theta.new_zeros(()),
        "evapotranspiration_flux": theta.new_zeros(()),
    }


def _interior_moisture(functor: SoilWaterTransportFunctor) -> float:
    lower = _as_float(functor.wilting_point)
    upper = _as_float(functor.porosity)
    return lower + 0.5 * (upper - lower)


def test_richards_solver_outputs_explicit_updates_and_closed_contract() -> None:
    """Ensure the Richards step exposes one full state and audited observations."""
    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))

    wilting = _as_float(functor.wilting_point)
    porosity = _as_float(functor.porosity)
    upper = min(_interior_moisture(functor), porosity - 0.01)
    theta = torch.linspace(wilting + 0.01, upper, steps=5)

    state = _canonical_state(theta)

    output = functor.forward(state=state, time=0.0, dt=60.0)

    assert set(output) == EXPECTED_OUTPUTS
    assert set(functor.metadata.output_variables) == EXPECTED_OUTPUTS
    assert "diagnostics.soil_water_non1d_approximation_used" not in output

    assert all(isinstance(item, UpdateItem) for item in output.values())
    assert "soil_moisture_richards" not in output
    state_update = output["soil_moisture"]
    assert state_update.mode is UpdateMode.REPLACE
    assert state_update.per_timestep is False
    assert state_update.unit == "m3 m-3"
    assert state_update.source_functor == "soil_water_transport"

    for key in EXPECTED_OUTPUTS - {"soil_moisture"}:
        assert output[key].mode is UpdateMode.REPLACE
        assert output[key].source_functor == "soil_water_transport"

    assert output["diagnostics.soil_water_transport.darcy_flux"].unit == "m s-1"
    assert output["diagnostics.soil_water_transport.darcy_flux"].per_timestep is False
    assert output["diagnostics.soil_water_transport.clipping_residual"].per_timestep is True
    assert output["diagnostics.soil_water_transport.water_balance_residual"].per_timestep is True

    clipping_residual = _value(
        output, "diagnostics.soil_water_transport.clipping_residual"
    )
    theta_updated = _value(output, "soil_moisture")
    soil_delta = theta_updated - theta
    assert torch.all(torch.isfinite(theta_updated))
    assert torch.all(theta_updated >= wilting)
    assert torch.all(theta_updated <= porosity)
    assert torch.all(torch.isfinite(clipping_residual))

    assert torch.all(torch.isfinite(soil_delta))

    iterations = _value(
        output, "diagnostics.soil_water_transport.richards_iterations"
    )
    assert float(iterations) >= 1.0
    assert "_last_richards_iterations" not in vars(functor)

    darcy_flux = _value(output, "diagnostics.soil_water_transport.darcy_flux")
    balance_residual = _value(
        output, "diagnostics.soil_water_transport.water_balance_residual"
    )
    dz = _as_float(functor.grid_spacing[0])
    assert darcy_flux.shape == (theta.numel() + 1,)
    assert torch.allclose(darcy_flux[0], theta.new_zeros(()))
    conductivity_updated = functor._compute_hydraulic_conductivity(theta_updated)
    gravity_gradient = torch.as_tensor(
        functor.gravity_head_gradient,
        device=theta.device,
        dtype=theta.dtype,
    )
    expected_bottom_flux = conductivity_updated[-1] * gravity_gradient
    assert torch.allclose(darcy_flux[-1], expected_bottom_flux)
    expected_residual = torch.sum(soil_delta) * dz + darcy_flux[-1] * 60.0
    assert torch.allclose(balance_residual, expected_residual, rtol=0.0, atol=0.0)
    assert abs(float(balance_residual)) < 1.0e-7


def test_clipping_and_overflow_are_routed_into_explicit_residuals() -> None:
    """Bounds projection must expose both the clipped amount and water residual."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    porosity = _as_float(functor.porosity)
    theta = torch.full((3,), _interior_moisture(functor))
    raw_theta = torch.full_like(theta, porosity + 0.05)

    functor._check_richards_stability = (
        lambda dt, _theta, _pressure, _conductivity: dt
    )
    functor._solve_richards_equation = (
        lambda _theta, _pressure, _conductivity, _source, _dt: (raw_theta, 1)
    )
    functor._compute_water_flux = (
        lambda pressure, conductivity, surface_flux: torch.zeros(
            pressure.numel() + 1,
            dtype=pressure.dtype,
            device=pressure.device,
        )
    )

    output = functor.forward(
        state=_canonical_state(theta),
        time=0.0,
        dt=60.0,
    )

    clipped = _value(output, "soil_moisture")
    clipping_residual = _value(
        output, "diagnostics.soil_water_transport.clipping_residual"
    )
    water_residual = _value(
        output, "diagnostics.soil_water_transport.water_balance_residual"
    )
    state_delta = clipped - theta

    assert torch.allclose(clipped, torch.full_like(theta, porosity))
    assert torch.allclose(clipping_residual, raw_theta - clipped)
    assert torch.allclose(state_delta, clipped - theta)
    expected_water_residual = torch.sum(clipped - theta) * _as_float(
        functor.grid_spacing[0]
    )
    assert torch.allclose(water_residual, expected_water_residual)


def test_metadata_declares_only_runtime_inputs() -> None:
    """Hydraulic conductivity is internal and must not be a stale public input."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))

    assert set(functor.metadata.input_variables) == EXPECTED_INPUTS
    assert set(functor.optional_input_variables) == {"pressure_head"}
    assert "infiltration_rate" not in functor.metadata.input_variables
    assert "evapotranspiration_rate" not in functor.metadata.input_variables
    assert functor.get_dependencies() == {
        "required": [],
        "optional": [],
        "lagged": [
            "surface_runoff",
            "penman_monteith_et",
            "priestley_taylor_et",
        ],
    }


def test_boundary_input_scms_route_to_soil_water_transport() -> None:
    """Optional effort and required flux ports must have canonical SCM routes."""

    registry = ContractRegistry(strict=True)
    contract_dir = Path("relic/contracts/variables/hydrosphere")
    for name in (
        "pressure_head",
        "infiltration_flux",
        "evapotranspiration_flux",
    ):
        path = contract_dir / f"{name}.yaml"
        assert "REVIEW_NEEDED" not in path.read_text(encoding="utf-8")
        registry.load_contract_file(path)

    pressure = registry.get_contract("pressure_head")
    assert pressure is not None
    assert pressure.kind == "forcing"
    assert pressure.udunits == "m"
    assert pressure.owner is None
    assert pressure.incrementable is False
    assert pressure.writers == []
    assert pressure.flux_out == ["soil_water_transport"]

    infiltration = registry.get_contract("infiltration_flux")
    assert infiltration is not None
    assert set(infiltration.flux_out) == {
        "richards_soil_moisture",
        "soil_water_transport",
    }

    evapotranspiration = registry.get_contract("evapotranspiration_flux")
    assert evapotranspiration is not None
    assert set(evapotranspiration.flux_out) == {
        "bucket_soil_moisture",
        "richards_soil_moisture",
        "soil_water_transport",
    }


def test_soil_moisture_scm_declares_mutually_exclusive_full_state_writers() -> None:
    """The canonical SCM must authorize every selected full-state closure."""

    contract_path = Path("relic/contracts/variables/hydrosphere/soil_moisture.yaml")
    assert "REVIEW_NEEDED" not in contract_path.read_text(encoding="utf-8")

    registry = ContractRegistry(strict=True)
    registry.load_contract_file(contract_path)
    contract = registry.get_contract("soil_moisture")

    assert contract is not None
    assert contract.owner is None
    assert contract.udunits == "m3 m-3"
    assert contract.incrementable is False
    assert contract.increment_writers == []
    assert contract.tendency_writers == [
        "bucket_soil_moisture",
        "richards_soil_moisture",
        "soil_water_transport",
        "observation_state_feedback",
    ]
    for writer in contract.tendency_writers:
        assert registry.can_write("soil_moisture", writer)
        assert not registry.can_increment("soil_moisture", writer)
    assert contract.raw["provenance"]["runtime_role"] == (
        "mutually_exclusive_soil_water_storage"
    )


def test_real_simulator_commits_full_state_with_water_ledger_provenance() -> None:
    """The simulator must commit the exact full state without false warnings."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    wilting = _as_float(functor.wilting_point)
    porosity = _as_float(functor.porosity)
    theta = torch.linspace(
        wilting + 0.01,
        min(_interior_moisture(functor), porosity - 0.01),
        steps=5,
    )
    state = _canonical_state(theta)
    direct = functor.forward(state=state, time=0.0, dt=60.0)

    simulator = RELICSimulator(
        {
            "conservation_checks": False,
            "strict_dependencies": False,
            "device": "cpu",
            "time_step": 60.0,
        }
    )
    simulator.initialize_state(state)
    simulator.register_functor(functor)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulator.step(dt=60.0)

    final_theta = simulator.state_manager.get_state()["soil_moisture"]
    direct_state = _value(direct, "soil_moisture")
    assert torch.allclose(final_theta, direct_state, rtol=0.0, atol=0.0)
    assert caught == []

    provenance = simulator.get_variable_provenance("soil_moisture")
    assert {record["functor"] for record in provenance} == {"soil_water_transport"}
    assert {record["role"] for record in provenance} == {
        "writer",
        "final_writer",
    }


@pytest.mark.parametrize("missing", ["infiltration_flux", "evapotranspiration_flux"])
def test_boundary_mass_fluxes_are_required(missing: str) -> None:
    """Open-system water ports must never be supplied by hidden zero defaults."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    theta = torch.full((3,), _interior_moisture(functor))
    state = _canonical_state(theta)
    state.pop(missing)

    with pytest.raises(ValueError, match=missing):
        functor.forward(state=state, time=0.0, dt=60.0)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("soil_moisture", torch.tensor([0.3, float("nan"), 0.3]), "finite"),
        ("soil_moisture", torch.tensor([0.3]), "at least two"),
        ("infiltration_flux", torch.tensor(float("inf")), "finite"),
        ("evapotranspiration_flux", torch.tensor(-1.0e-6), "non-negative"),
        ("infiltration_flux", torch.zeros(2), "scalar"),
    ],
)
def test_runtime_inputs_fail_fast_on_invalid_domain(
    key: str,
    value: torch.Tensor,
    message: str,
) -> None:
    """Richards inputs must be finite and match their declared physical domains."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    theta = torch.full((3,), _interior_moisture(functor))
    state = _canonical_state(theta)
    state[key] = value

    with pytest.raises(ValueError, match=message):
        functor.forward(state=state, time=0.0, dt=60.0)


def test_pressure_head_is_optional_but_validated_when_supplied() -> None:
    """A supplied hydraulic effort must be finite and conform to the storage grid."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    theta = torch.full((3,), _interior_moisture(functor))
    state = _canonical_state(theta)
    state["pressure_head"] = torch.tensor([-1.0, float("nan"), -1.0])

    with pytest.raises(ValueError, match="pressure_head.*finite"):
        functor.forward(state=state, time=0.0, dt=60.0)


def test_nonconverged_picard_step_fails_without_state_update() -> None:
    """A failed nonlinear solve must not emit a state increment."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    theta = torch.full((3,), _interior_moisture(functor))
    functor.max_richards_iterations = 1
    object.__setattr__(
        functor._richards_solver,
        "_solve_linearized_system_picard",
        lambda **kwargs: (kwargs["theta_iter"] + 1.0e-3, False),
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        functor.forward(state=_canonical_state(theta), time=0.0, dt=60.0)


def test_nonfinite_picard_solution_fails_without_state_update() -> None:
    """A non-finite numerical state must not reach the StateManager contract."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    theta = torch.full((3,), _interior_moisture(functor))
    functor._check_richards_stability = (
        lambda dt, _theta, _pressure, _conductivity: dt
    )
    functor._solve_richards_equation = (
        lambda _theta, _pressure, _conductivity, _source, _dt: (
            torch.full_like(theta, float("nan")),
            1,
        )
    )

    with pytest.raises(RuntimeError, match="finite"):
        functor.forward(state=_canonical_state(theta), time=0.0, dt=60.0)


def test_unstable_requested_timestep_fails_instead_of_partial_advancement() -> None:
    """A Functor update must represent the full requested dt or fail explicitly."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    theta = torch.full((3,), _interior_moisture(functor))
    functor._check_richards_stability = (
        lambda dt, _theta, _pressure, _conductivity: dt / 2.0
    )

    with pytest.raises(RuntimeError, match="stable timestep"):
        functor.forward(state=_canonical_state(theta), time=0.0, dt=60.0)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("soil_model", "brooks_corey"), ("soil_model", "campbell"), ("spatial_dims", 2)],
)
def test_single_valued_selectors_are_not_constructor_keywords(keyword: str, value: object) -> None:
    """The one-dimensional van Genuchten-Mualem column is the only production law: the
    constructor offers no selector keyword for it, and an unsupported request is
    rejected (not ignored) before initialization."""
    import inspect

    parameters = inspect.signature(SoilWaterTransportFunctor.__init__).parameters
    assert keyword not in parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    assert SoilWaterTransportFunctor.SPATIAL_DIMS == 1
    with pytest.raises(TypeError):
        SoilWaterTransportFunctor(**{keyword: value})


@pytest.mark.parametrize("dt", [0.0, -1.0, torch.tensor(0.0)])
def test_nonpositive_timestep_fails_fast(dt: torch.Tensor | float) -> None:
    """Richards state advancement requires a positive finite scalar time step."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    theta = torch.full((3,), _interior_moisture(functor))

    with pytest.raises(ValueError, match="dt must be a positive finite scalar"):
        functor.forward(state=_canonical_state(theta), time=0.0, dt=dt)

def test_infiltration_gradient_survives_the_richards_solve() -> None:
    """The returned soil moisture must stay differentiable with respect to the surface mass flux."""
    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    wilting = _as_float(functor.wilting_point)
    porosity = _as_float(functor.porosity)
    upper = min(_interior_moisture(functor), porosity - 0.01)
    theta = torch.linspace(wilting + 0.01, upper, steps=5, dtype=torch.float64)
    dt = 60.0
    infiltration = torch.tensor(1.0e-5, dtype=torch.float64, requires_grad=True)  # kg m-2 s-1

    def _run(flux_value: torch.Tensor) -> torch.Tensor:
        state = {
            "soil_moisture": theta,
            "infiltration_flux": flux_value,
            "evapotranspiration_flux": theta.new_zeros(()),
        }
        output = functor(state, time=0.0, dt=dt)
        return _value(output, "soil_moisture")

    theta_new = _run(infiltration)
    total = theta_new.sum()
    (gradient,) = torch.autograd.grad(total, infiltration, allow_unused=True)
    assert gradient is not None, "infiltration_flux was detached from the returned soil moisture"
    assert torch.isfinite(gradient)

    # References: Richards (1931) water balance; Celia et al. (1990).
    # Mathematical expression: sum_i theta_i dz = ... + q_surface dt with q_surface = F/rho_w,
    #   so d(sum_i theta_i)/dF = dt / (rho_w dz) when no saturation clipping occurs.
    # Parameter meanings: F surface mass flux, rho_w water density, dz layer thickness, dt step.
    # Units: F [kg m-2 s-1]; rho_w [kg m-3]; dz [m]; dt [s]; gradient [m3 m-3 per kg m-2 s-1].
    # Domain/constraints: interior moisture profile, zero clipping residual.
    # Sign convention: infiltration positive downward increases stored water.
    rho_w = _as_float(functor.water_density) if hasattr(functor, "water_density") else None
    dz = _as_float(functor.grid_spacing[0])
    eps = 1.0e-7
    with torch.no_grad():
        plus = _run(torch.tensor(1.0e-5 + eps, dtype=torch.float64)).sum()
        minus = _run(torch.tensor(1.0e-5 - eps, dtype=torch.float64)).sum()
    finite_difference = float((plus - minus) / (2.0 * eps))
    assert abs(float(gradient) - finite_difference) <= 1.0e-3 * abs(finite_difference)
    if rho_w is not None:
        expected = dt / (rho_w * dz)
        assert abs(float(gradient) - expected) <= 1.0e-3 * expected


def test_no_generic_diffusion_fallback_is_constructed() -> None:
    """The canonical 1D Richards runtime must not own a second diffusion solver."""

    functor = SoilWaterTransportFunctor(device=torch.device("cpu"))
    source = inspect.getsource(SoilWaterTransportFunctor)

    assert not hasattr(functor, "_diffusion_operator")
    for forbidden_token in (
        "OperatorFactory",
        "BoundaryConditionManager",
        "_diffusion_operator",
        "(theta / porosity) ** 3",
        "return torch.zeros_like(pressure_head)",
        "_last_non1d_richards_approximation_used",
    ):
        assert forbidden_token not in source
