import dataclasses
import math
import warnings
from pathlib import Path

import pytest
import torch

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.contracts.contract_loader import ContractRegistry
from relic.core.functor import IPHSDECFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.hydrosphere.runoff.subsurface import SubsurfaceRunoff


def _unwrap_update(value):
    return value.value if isinstance(value, UpdateItem) else value


def _expected_rate(mass_flux, water_density):
    flux = torch.as_tensor(mass_flux, dtype=torch.float32)
    rho_w = torch.as_tensor(water_density, dtype=torch.float32)
    return (flux / rho_w) * 1000.0


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


def _build_state(**overrides):
    base = {
        "soil_moisture": torch.tensor(0.4, dtype=torch.float32),
        "drainage_flux": torch.tensor(0.0, dtype=torch.float32),
        "groundwater_storage": torch.tensor(120.0, dtype=torch.float32),
    }
    base.update(overrides)
    return base


def test_subsurface_runoff_uses_canonical_drainage_flux():
    model = SubsurfaceRunoff(name="test_subsurface")
    drainage_flux = torch.tensor(2.0e-6, dtype=torch.float32)
    state = _build_state(drainage_flux=drainage_flux)

    result = model.forward(state, time=0.0, dt=3600.0)
    drainage_rate = _unwrap_update(
        result["diagnostics.subsurface_runoff.drainage_rate_mm_s"]
    )
    expected = _expected_rate(drainage_flux, model.water_density)
    assert torch.allclose(drainage_rate, expected, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("old_key", ["drainage_input", "subsurface_drainage"])
def test_subsurface_runoff_rejects_noncanonical_drainage_aliases(old_key):
    model = SubsurfaceRunoff(name="test_subsurface")
    state = _build_state()
    state.pop("drainage_flux")
    state[old_key] = torch.tensor(1.0e-6, dtype=torch.float32)

    with pytest.raises(ValueError, match="drainage_flux"):
        model.forward(state, time=0.0, dt=3600.0)


@pytest.mark.parametrize("missing_key", ["soil_moisture", "drainage_flux", "groundwater_storage"])
def test_subsurface_runoff_requires_all_canonical_inputs(missing_key):
    model = SubsurfaceRunoff(name="test_subsurface")
    state = _build_state()
    state.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        model.forward(state, time=0.0, dt=3600.0)


def test_subsurface_runoff_dependencies_match_noncompeting_state_providers():
    """Every mutually exclusive drainage writer may precede this consumer."""
    model = SubsurfaceRunoff(name="subsurface_dependency_contract")

    assert model.get_dependencies() == {
        "required": [],
        "optional": [
            "bucket_soil_moisture",
            "richards_soil_moisture",
            "vertical_percolation_transport",
        ],
    }
    registry = ContractRegistry(strict=True)
    registry.load_contract_file(
        Path("relic/contracts/variables/hydrosphere/drainage_flux.yaml")
    )
    contract = registry.get_contract("drainage_flux")
    assert contract is not None
    assert contract.flux_out == [
        "groundwater_dynamics",
        "subsurface_runoff",
    ]


def test_subsurface_runoff_forward_returns_finite_flux():
    model = SubsurfaceRunoff(name="test_subsurface")
    fluxes = model.forward(_build_state(), time=0.0, dt=3600.0)
    flux = _unwrap_update(fluxes["subsurface_runoff_flux"])

    assert torch.isfinite(flux).all()
    assert torch.all(flux >= 0)


def test_subsurface_runoff_outputs_match_metadata_and_have_explicit_semantics():
    model = SubsurfaceRunoff(name="subsurface_contract")
    result = model.forward(_build_state(), time=0.0, dt=3600.0)

    expected_outputs = {
        "subsurface_runoff_flux",
        "groundwater_storage",
        "diagnostics.subsurface_runoff.baseflow_flux",
        "diagnostics.subsurface_runoff.interflow_flux",
        "diagnostics.subsurface_runoff.storage_fraction",
        "diagnostics.subsurface_runoff.water_table_depth",
        "diagnostics.subsurface_runoff.recession_time_constant_days",
        "diagnostics.subsurface_runoff.drainage_rate_mm_s",
        "diagnostics.subsurface_runoff.total_runoff_rate_mm_s",
        "diagnostics.subsurface_runoff.storage_capacity_overflow",
        "diagnostics.subsurface_runoff.outflow_limiter_residual",
        "diagnostics.subsurface_runoff.water_table_depth_clipping_residual",
        "diagnostics.subsurface_runoff.water_balance_residual",
    }
    assert set(result) == expected_outputs
    assert set(model.metadata.output_variables) == expected_outputs
    assert set(model.metadata.conservation_laws) == {"mass"}

    expected_units = {
        "subsurface_runoff_flux": "kg m-2 s-1",
        "groundwater_storage": "mm",
        "diagnostics.subsurface_runoff.baseflow_flux": "kg m-2 s-1",
        "diagnostics.subsurface_runoff.interflow_flux": "kg m-2 s-1",
        "diagnostics.subsurface_runoff.storage_fraction": "dimensionless",
        "diagnostics.subsurface_runoff.water_table_depth": "m",
        "diagnostics.subsurface_runoff.recession_time_constant_days": "d",
        "diagnostics.subsurface_runoff.drainage_rate_mm_s": "mm s-1",
        "diagnostics.subsurface_runoff.total_runoff_rate_mm_s": "mm s-1",
        "diagnostics.subsurface_runoff.storage_capacity_overflow": "mm",
        "diagnostics.subsurface_runoff.outflow_limiter_residual": "mm",
        "diagnostics.subsurface_runoff.water_table_depth_clipping_residual": "m",
        "diagnostics.subsurface_runoff.water_balance_residual": "mm",
    }
    for key, item in result.items():
        assert isinstance(item, UpdateItem), key
        assert item.mode == UpdateMode.REPLACE, key
        assert item.unit == expected_units[key], key
        assert item.source_functor == model.name, key
    assert result["subsurface_runoff_flux"].per_timestep is False
    assert result["groundwater_storage"].per_timestep is True


def test_subsurface_runoff_open_water_balance_is_exact():
    model = SubsurfaceRunoff(name="subsurface_balance")
    dt = 3600.0
    initial_storage = torch.tensor(99.0, dtype=torch.float32)
    drainage_flux = torch.tensor(1.0e-3, dtype=torch.float32)
    result = model.forward(
        _build_state(
            groundwater_storage=initial_storage,
            drainage_flux=drainage_flux,
        ),
        time=0.0,
        dt=dt,
    )

    residual = result["diagnostics.subsurface_runoff.water_balance_residual"].value
    final_storage = result["groundwater_storage"].value
    runoff_flux = result["subsurface_runoff_flux"].value
    overflow = result[
        "diagnostics.subsurface_runoff.storage_capacity_overflow"
    ].value
    input_depth = _expected_rate(drainage_flux, model.water_density) * dt
    runoff_depth = _expected_rate(runoff_flux, model.water_density) * dt
    reconstructed = initial_storage + input_depth - runoff_depth

    assert runoff_depth >= overflow
    assert torch.allclose(final_storage, reconstructed, rtol=0.0, atol=2.0e-5)
    assert torch.allclose(residual, torch.zeros_like(residual), rtol=0.0, atol=2.0e-5)

    expected_residual = float(torch.max(torch.abs(residual)).item())
    conservation = model.check_conservation_laws({}, result)
    assert conservation == {"mass": pytest.approx(expected_residual)}
    assert conservation["mass"] <= 2.0e-5

    diagnostic_key = "diagnostics.subsurface_runoff.water_balance_residual"
    perturbed = dict(result)
    perturbed[diagnostic_key] = torch.tensor(-0.125)
    assert model.check_conservation_laws({}, perturbed) == {
        "mass": pytest.approx(0.125)
    }

    missing = dict(result)
    missing.pop(diagnostic_key)
    with pytest.raises(KeyError, match="Missing conservation diagnostic"):
        model.check_conservation_laws({}, missing)

    wrong_type = dict(result)
    wrong_type[diagnostic_key] = 0.0
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        model.check_conservation_laws({}, wrong_type)

    empty = dict(result)
    empty[diagnostic_key] = torch.empty(0)
    with pytest.raises(ValueError, match="must be non-empty"):
        model.check_conservation_laws({}, empty)

    nonfinite = dict(result)
    nonfinite[diagnostic_key] = torch.tensor(float("inf"))
    with pytest.raises(ValueError, match="must be finite"):
        model.check_conservation_laws({}, nonfinite)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("soil_moisture", torch.tensor(float("nan"))),
        ("drainage_flux", torch.tensor(float("inf"))),
        ("drainage_flux", torch.tensor(-1.0e-6)),
        ("groundwater_storage", torch.tensor(float("nan"))),
        ("groundwater_storage", torch.tensor(-1.0)),
    ],
)
def test_subsurface_runoff_rejects_nonfinite_or_negative_inputs(key, value):
    model = SubsurfaceRunoff(name="subsurface_fail_fast")
    with pytest.raises(ValueError, match=key):
        model.forward(_build_state(**{key: value}), time=0.0, dt=3600.0)


class TestSubsurfaceRunoffPhase8:
    """Test Phase 8 IPHS contract."""

    def test_phase8_iphs_protocol(self):
        """Test structural IPHS hooks."""
        functor = SubsurfaceRunoff()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.runoff.transmissivity" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "groundwater_storage",
            "soil_moisture",
            "subsurface_runoff_flux",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {"subsurface_runoff_boundary_flux"}

    def test_phase8_certificate_is_valid(self):
        """Test Phase 8 certificate completeness."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_runoff_subsurface_subsurface_runoff.yaml")
        )
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == (
            "relic.scientific.hydrosphere.runoff.subsurface.SubsurfaceRunoff"
        )
        assert certificate.canonical_role == "transport"

    def test_residual_diagnostics_are_routed(self):
        """Test projection residual diagnostics are exposed."""
        functor = SubsurfaceRunoff()
        result = functor.forward(
            _build_state(
                drainage_flux=torch.tensor(2.0e-6, dtype=torch.float32),
                groundwater_storage=torch.tensor(400.0, dtype=torch.float32),
            ),
            time=0.0,
            dt=3600.0,
        )

        diagnostic_keys = {
            "diagnostics.subsurface_runoff.storage_capacity_overflow",
            "diagnostics.subsurface_runoff.outflow_limiter_residual",
            "diagnostics.subsurface_runoff.water_table_depth_clipping_residual",
            "diagnostics.subsurface_runoff.water_balance_residual",
        }
        for key in diagnostic_keys:
            assert key in result
            assert torch.isfinite(result[key].value).all()
        assert torch.all(
            result["diagnostics.subsurface_runoff.storage_capacity_overflow"].value
            >= 0.0
        )
        assert torch.all(
            result["diagnostics.subsurface_runoff.outflow_limiter_residual"].value
            >= 0.0
        )
        assert torch.all(
            result[
                "diagnostics.subsurface_runoff.water_table_depth_clipping_residual"
            ].value
            >= 0.0
        )


def test_topmodel_baseflow_uses_the_catchment_mean_exp_minus_lambda_scale():
    model = SubsurfaceRunoff(name="subsurface_topmodel_kernel")
    storage = torch.tensor([50.0, 150.0], dtype=torch.float32)
    theta = torch.tensor([0.2, 0.4], dtype=torch.float32)
    actual = model._compute_topmodel_baseflow(storage, theta, 3600.0)
    # References: Beven and Kirkby (1979); Beven (1997).
    # Mathematical expression: Q_b = 1000*T0*exp(-lambda)*exp(-S_def/m)*f_theta with
    #   S_def = max(S_def_max - S, 0), m = c_m*S_def_max, f_theta = clip(theta/theta_ref, f_min, f_max).
    # Parameter meanings: T0 transmissivity, lambda = mean ln(a/tan beta) with a in metres.
    # Units: T0 [m2 s-1]; exp(-lambda) [m-1]; 1000 converts m to mm; Q_b [mm s-1].
    # Domain/constraints: the former 1/lambda form is not dimensionally a depth rate.
    # Sign convention: nonnegative outward lateral export.
    t0 = model._to_scalar(model.transmissivity)
    lam = model._to_scalar(model.topographic_index)
    d_max = model._to_scalar(model.topmodel_max_deficit)
    m_scale = model._to_scalar(model.topmodel_m_scaling_factor) * d_max
    deficit = torch.clamp(d_max - storage.double(), min=0.0)
    f_theta = torch.clamp(
        theta.double() / model._to_scalar(model.topmodel_moisture_reference),
        min=model._to_scalar(model.topmodel_moisture_factor_min),
        max=model._to_scalar(model.topmodel_moisture_factor_max),
    )
    expected = 1000.0 * t0 * math.exp(-lam) * torch.exp(-deficit / m_scale) * f_theta
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1.0e-5, atol=0.0)
    former = 1000.0 * t0 / lam * torch.exp(-deficit / m_scale) * f_theta
    assert not torch.allclose(actual.double(), former, rtol=1.0e-3, atol=0.0)


def test_interflow_is_a_moisture_gated_drainage_of_the_groundwater_store():
    model = SubsurfaceRunoff(name="subsurface_interflow_kernel")
    theta = torch.tensor([0.25, 0.4, 0.4, 0.4], dtype=torch.float32)
    storage = torch.tensor([100.0, 0.0, 100.0, 200.0], dtype=torch.float32)
    actual = model._compute_interflow(theta, storage, 3600.0)
    # References: Beven and Kirkby (1979); Brutsaert and Nieber (1977).
    # Mathematical expression: Q_i = k_i*c_i*max(0, theta - theta_t)^n*S.
    # Parameter meanings: k_i [s-1], c_i and n [1], theta_t threshold, S store depth [mm].
    # Units: Q_i [mm s-1] because the s-1 coefficient multiplies the mm store.
    # Domain/constraints: zero below threshold and zero for an empty store.
    # Sign convention: nonnegative outward export.
    k_i = model._to_scalar(model.interflow_coeff_s)
    c_i = model._to_scalar(model.interflow_rate_scaling_factor)
    n = model._to_scalar(model.interflow_exponent)
    theta_t = model._to_scalar(model.threshold_moisture)
    excess = torch.clamp(theta.double() - theta_t, min=0.0)
    expected = k_i * c_i * excess ** n * storage.double()
    torch.testing.assert_close(actual.double(), expected, rtol=1.0e-5, atol=0.0)
    detached = actual.detach()
    assert float(detached[0]) == 0.0
    assert float(detached[1]) == 0.0
    assert float(detached[2]) > 0.0
    assert abs(float(detached[3]) - 2.0 * float(detached[2])) <= 1.0e-6 * float(detached[3])


def test_hodge_mobility_carries_the_topmodel_length_scale_and_orphan_length_is_gone():
    model = SubsurfaceRunoff(name="subsurface_hodge_scale")
    hodge = model.build_hodge_contract(n_dofs=2, strict=True)
    expected = (
        model._to_scalar(model.water_density)
        * model._to_scalar(model.transmissivity)
        * math.exp(-model._to_scalar(model.topographic_index))
    )
    torch.testing.assert_close(
        torch.diag(hodge.matrix).double(),
        torch.full((2,), expected, dtype=torch.float64),
        rtol=1.0e-5,
        atol=0.0,
    )
    assert hodge.codomain.units == "kg m-2 s-1"
    assert hodge.metric_keys == ("hydrosphere.runoff.topographic_index_mean",)
    assert "hydrosphere.runoff.topographic_index_mean" in hodge.parameter_keys
    assert "hydrosphere.runoff.subsurface_flux_default_distance" not in model.required_parameters()
    manager = get_global_manager()
    assert manager.get("hydrosphere.runoff.subsurface_flux_default_distance", required=False) is None


def test_governed_topmodel_forward_uses_the_corrected_baseflow_scale():
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        manager.set_value(
            "hydrosphere.runoff.baseflow_method",
            "topmodel",
            units="dimensionless",
            validate=True,
            notify=False,
        )
        model = SubsurfaceRunoff(name="subsurface_topmodel_governed")
        assert model.baseflow_method == "topmodel"
        dt = 60.0
        # Keep the store below hydrosphere.runoff.storage_capacity (100 mm) so the
        # capacity projection does not change the deficit seen by the kernel.
        initial_storage = torch.tensor(80.0, dtype=torch.float32)
        # The governed call must not report float32 rounding of the 80 mm store as a mass violation.
        warnings.simplefilter("error", UserWarning)
        result = model(
            _build_state(
                groundwater_storage=initial_storage,
                soil_moisture=torch.tensor(0.4, dtype=torch.float32),
                drainage_flux=torch.tensor(0.0, dtype=torch.float32),
            ),
            time=0.0,
            dt=dt,
        )
        baseflow_mass = result["diagnostics.subsurface_runoff.baseflow_flux"].value
        interflow_mass = result["diagnostics.subsurface_runoff.interflow_flux"].value
        kernel_baseflow = model._compute_topmodel_baseflow(
            initial_storage, torch.tensor(0.4, dtype=torch.float32), dt
        )
        kernel_interflow = model._compute_interflow(
            torch.tensor(0.4, dtype=torch.float32), initial_storage, dt
        )
        candidate_depth = float(((kernel_baseflow + kernel_interflow) * dt).detach())
        assert candidate_depth < float(initial_storage)
        rho_w = model._to_scalar(model.water_density)
        torch.testing.assert_close(
            baseflow_mass.double(), kernel_baseflow.double() / 1000.0 * rho_w, rtol=1.0e-5, atol=0.0,
        )
        torch.testing.assert_close(
            interflow_mass.double(), kernel_interflow.double() / 1000.0 * rho_w, rtol=1.0e-5, atol=0.0,
        )
        residual = result["diagnostics.subsurface_runoff.water_balance_residual"].value
        assert float(torch.max(torch.abs(residual.detach()))) <= 2.0e-5
    finally:
        warnings.resetwarnings()
        manager.restore(snapshot)

def test_governed_call_reports_a_relative_water_closure_without_float32_false_warnings():
    """The mass hook must compare a dimensionless closure, not float32 rounding of an 80 mm store."""
    model = SubsurfaceRunoff(name="subsurface_relative_closure")
    dt = 60.0
    state = _build_state(
        groundwater_storage=torch.tensor(80.0, dtype=torch.float32),
        soil_moisture=torch.tensor(0.4, dtype=torch.float32),
        drainage_flux=torch.tensor(0.0, dtype=torch.float32),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = model(state, time=0.0, dt=dt)
    violations = model.check_conservation_laws(state, result)
    assert set(violations) == {"mass"}
    residual = result["diagnostics.subsurface_runoff.water_balance_residual"].value
    storage = result["groundwater_storage"].value
    # References: Higham (2002), relative error; Beven and Kirkby (1979).
    # Mathematical expression: r_S = |epsilon_S| / |S1| when S1 > 0.
    # Parameter meanings: epsilon_S the ledger residual, S1 the returned storage.
    # Units: epsilon_S, S1 [mm]; r_S [1].
    # Domain/constraints: float32 runtime state; drainage zero.
    # Sign convention: r_S >= 0.
    expected = float(torch.max(torch.abs(residual.detach())).double() / torch.max(torch.abs(storage.detach())).double())
    assert violations["mass"] == pytest.approx(expected, rel=1.0e-6, abs=1.0e-12)
    assert violations["mass"] <= 1.0e-6
    tampered = dict(result)
    tampered["diagnostics.subsurface_runoff.water_balance_residual"] = dataclasses.replace(
        result["diagnostics.subsurface_runoff.water_balance_residual"],
        value=torch.tensor(1.0, dtype=torch.float32),
    )
    assert model.check_conservation_laws(state, tampered)["mass"] == pytest.approx(
        1.0 / float(torch.max(torch.abs(storage.detach()))), rel=1.0e-6
    )


_RETURN_FLOW_FRACTION_KEY = "hydrosphere.runoff.return_flow_fraction"


def _subsurface_tensors(result) -> dict:
    """Every tensor of a subsurface-runoff update, keyed by output name."""
    tensors = {}
    for name, update in result.items():
        value = update.value if isinstance(update, UpdateItem) else update
        members = value.items() if isinstance(value, dict) else [(None, value)]
        for member, item in members:
            if isinstance(item, torch.Tensor):
                tensors[name if member is None else f"{name}:{member}"] = item
    return tensors


def _interflow_state() -> dict:
    """Moisture above the interflow threshold and a partly filled store."""
    return _build_state(
        soil_moisture=torch.tensor(0.4, dtype=torch.float32),
        drainage_flux=torch.tensor(2.0e-6, dtype=torch.float32),
        groundwater_storage=torch.tensor(50.0, dtype=torch.float32),
    )


def test_subsurface_interflow_coefficient_follows_a_changed_binding(
    restore_global_parameter_manager,
):
    """The interflow coefficient is derived from the held return-flow binding
    at use, so an instance built before the binding changed computes like one
    built afterwards, and the binding is held by reference."""
    # References: Beven and Kirkby (1979), Hydrol. Sci. Bull. 24(1), 43-69;
    #   Brutsaert and Nieber (1977), Water Resour. Res. 13(3), 637-643.
    # Mathematical expression: k_d = f_r c_s; k_i = k_d / s_day;
    #   Q_i = k_i c_i max(theta - theta_t, 0)^n S.
    # Parameter meanings: f_r return-flow fraction
    #   (hydrosphere.runoff.return_flow_fraction), c_s interflow scaling
    #   factor, s_day seconds per day, k_d and k_i the interflow coefficient per
    #   day and per second, c_i rate multiplier, theta soil moisture, theta_t
    #   interflow threshold, n exponent, S groundwater storage.
    # Units: f_r, c_s, c_i, n [1]; k_d [d-1]; k_i [s-1]; s_day [s d-1];
    #   theta, theta_t [m3 m-3]; S [mm]; Q_i [mm s-1].
    # Domain/constraints: the new f_r lies inside the ParameterManager bounds,
    #   so it is not clamped; theta > theta_t; dt = 10 s keeps the outflow
    #   below the store, so the storage limiter is inactive.
    # Sign convention: Q_i is a nonnegative outward lateral export.
    manager = restore_global_parameter_manager
    model = SubsurfaceRunoff(name="subsurface_built_before_the_change")
    spec = manager.get_parameter(_RETURN_FLOW_FRACTION_KEY)
    manager.set_value(_RETURN_FLOW_FRACTION_KEY, 0.33, units=spec.units, validate=True)
    fresh = SubsurfaceRunoff(name="subsurface_built_after_the_change")

    expected = _subsurface_tensors(fresh(_interflow_state(), time=0.0, dt=10.0))
    actual = _subsurface_tensors(model(_interflow_state(), time=0.0, dt=10.0))
    assert set(actual) == set(expected)
    for name, tensor in actual.items():
        torch.testing.assert_close(
            tensor, expected[name], rtol=0.0, atol=0.0, msg=name
        )

    binding = manager.get_trainable_parameters()[_RETURN_FLOW_FRACTION_KEY]
    assert float(binding.detach()) == pytest.approx(0.33)
    assert model.held_manager_parameters()[_RETURN_FLOW_FRACTION_KEY] is binding
    expected_coefficient = binding.detach() * model.interflow_scaling_factor
    torch.testing.assert_close(
        model.interflow_coefficient.detach(), expected_coefficient, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        model.interflow_coeff_s.detach(),
        expected_coefficient / model.time_conversion,
        rtol=0.0,
        atol=0.0,
    )


def test_subsurface_supports_repeated_backward_through_one_instance():
    """No autograd graph is carried from one call into the next: the instance
    stores no tensor derived from a binding, a second backward through it
    works and it gives the gradient of the first."""
    model = SubsurfaceRunoff(name="subsurface_repeated_backward")
    stored_graphs = sorted(
        name
        for name, attribute in vars(model).items()
        if isinstance(attribute, torch.Tensor) and attribute.grad_fn is not None
    )
    assert stored_graphs == []

    held = model.held_manager_parameters()
    gradients = []
    try:
        for _ in range(2):
            for binding in held.values():
                binding.grad = None
            update = model(_interflow_state(), time=0.0, dt=10.0)
            update["subsurface_runoff_flux"].value.sum().backward()
            gradients.append(
                held[_RETURN_FLOW_FRACTION_KEY].grad.detach().cpu().clone()
            )
    finally:
        for binding in held.values():
            binding.grad = None

    assert torch.isfinite(gradients[0]).all()
    assert bool((gradients[0] != 0).all())
    torch.testing.assert_close(gradients[1], gradients[0], rtol=0.0, atol=0.0)
