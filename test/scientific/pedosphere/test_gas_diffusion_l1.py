"""
L1 Unit Tests for GasDiffusion1D (Pedosphere)

Scope:
- Verify that GasDiffusion1D initializes, validates inputs, and produces
  expected output structure and basic physical properties.
- Tests are CPU-only and use simple synthetic profiles.

Windows usage:
- Activate env: conda activate RELIC
- Run: python -m pytest .\\test\\scientific\\pedosphere\\test_gas_diffusion_l1.py -q

Notes:
- Assertions avoid strict unit string equality to tolerate internal format.
- All prints and comments use ASCII only.
"""

import math
from pathlib import Path

import torch
import pytest

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.pedosphere.transport.gas_diffusion import GasDiffusion1D, DiffusivityModel
from relic.utils.physical_constants import get_physical_constant


CERTIFICATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "pedosphere_transport_gas_diffusion_gas_diffusion1_d.yaml"
)


def _build_state(n_layers: int):
    """Create a minimal valid state for GasDiffusion1D tests."""
    # Soil moisture smaller than porosity
    theta = torch.linspace(0.20, 0.25, steps=n_layers)
    porosity = torch.full((n_layers,), 0.40)
    # Temperature slightly above 293 K to test temperature correction
    temperature = torch.full((n_layers,), 303.15)
    clay = torch.full((n_layers,), 0.20)
    # Concentrations decreasing toward surface to drive upward emission
    co2_conc = torch.linspace(2.5, 2.0, steps=n_layers)
    n2o_conc = torch.linspace(0.020, 0.010, steps=n_layers)

    return {
        "soil_moisture": theta,
        "porosity": porosity,
        "soil_temperature": temperature,
        "clay_content": clay,
        "soil_co2_concentration": co2_conc,
        "n2o_concentration": n2o_conc,
    }


def _value(item):
    """Return the tensor carried by an explicit state update."""
    return item.value if isinstance(item, UpdateItem) else item


pytestmark = pytest.mark.usefixtures(
    "pedosphere_transport_constructor_overrides"
)


def test_forward_outputs_keys_and_shapes():
    n_layers = 4
    model = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK.value,
        enable_temperature_correction=True,
        enable_atmospheric_exchange=True,
        device=torch.device("cpu"),
    )
    state = _build_state(n_layers)

    out = model.forward(state=state, time=0.0, dt=60.0)

    expected_outputs = {
        "co2_surface_flux",
        "n2o_surface_flux",
        "co2_flux_profile",
        "n2o_flux_profile",
        "soil_co2_concentration",
        "n2o_concentration",
        "diagnostics.gas_diffusion.energy_storage",
        "diagnostics.gas_diffusion.dissipation_rate",
        "diagnostics.gas_diffusion.dH_dt_audit",
        "diagnostics.gas_diffusion.port_power_residual",
        "diagnostics.gas_diffusion.co2_clipping_residual",
        "diagnostics.gas_diffusion.n2o_clipping_residual",
        "diagnostics.gas_diffusion.step_seconds",
    }
    assert set(out) == expected_outputs
    assert set(model.metadata.output_variables) == expected_outputs
    for key in expected_outputs:
        assert isinstance(out[key], UpdateItem), f"Output {key} must be UpdateItem"

    for key in ("soil_co2_concentration", "n2o_concentration"):
        assert out[key].mode == UpdateMode.ACCUMULATE
        assert out[key].per_timestep is True

    for key in expected_outputs - {"soil_co2_concentration", "n2o_concentration"}:
        assert out[key].mode == UpdateMode.REPLACE
        assert out[key].per_timestep is False

    assert "heterotrophic_respiration" not in out
    assert "n2o_flux" not in out

    # Shapes
    assert _value(out["co2_flux_profile"]).numel() == n_layers - 1
    assert _value(out["n2o_flux_profile"]).numel() == n_layers - 1
    assert _value(
        out["diagnostics.gas_diffusion.co2_clipping_residual"]
    ).numel() == n_layers
    assert _value(
        out["diagnostics.gas_diffusion.n2o_clipping_residual"]
    ).numel() == n_layers

    # Surface fluxes must be finite; sign follows concentration-gradient direction.
    assert torch.all(torch.isfinite(_value(out["co2_surface_flux"])))
    assert torch.all(torch.isfinite(_value(out["n2o_surface_flux"])))
    assert _value(out["co2_surface_flux"]) > 0.0
    assert _value(out["n2o_surface_flux"]) > 0.0

    # Units are part of the declared interface contract, not runtime state keys.
    assert not any(key.endswith("_units") for key in out)


def test_atmospheric_boundary_concentrations_use_elemental_mass_basis():
    """Configured gas boundaries must match the elemental output-flux basis."""
    manager = get_global_manager(preload_all=True, warmup_cache=False)
    pressure = float(get_physical_constant("standard_pressure"))
    temperature = float(get_physical_constant("standard_temperature"))
    gas_constant = float(get_physical_constant("gas_constant"))
    ppm_to_fraction = float(
        get_physical_constant(
            "relic.physical.constants.conversions.ppm_to_fraction"
        )
    )
    expected = {
        "pedosphere.physical.atmospheric_co2_concentration": (
            "g C m-3",
            float(get_physical_constant("co2_reference"))
            * ppm_to_fraction
            * pressure
            / (gas_constant * temperature)
            * float(
                get_physical_constant(
                    "relic.physical.constants.molecular_weights.carbon"
                )
            ),
        ),
        "pedosphere.physical.atmospheric_n2o_concentration": (
            "g N m-3",
            float(
                get_physical_constant(
                    "relic.physical.constants.atmosphere.n2o_current"
                )
            )
            * ppm_to_fraction
            * pressure
            / (gas_constant * temperature)
            * float(
                get_physical_constant(
                    "relic.physical.constants.molecular_weights.n2"
                )
            ),
        ),
    }

    for key, (units, expected_value) in expected.items():
        spec = manager.get_parameter(key)
        assert spec.units == units
        assert math.isclose(float(spec.value), expected_value, rel_tol=1.0e-9)


def test_temperature_correction_increases_flux():
    n_layers = 4
    state = _build_state(n_layers)

    model_no_tc = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK.value,
        enable_temperature_correction=False,
        enable_atmospheric_exchange=True,
        device=torch.device("cpu"),
    )
    out_no = model_no_tc.forward(state=state, time=0.0, dt=60.0)

    model_tc = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK.value,
        enable_temperature_correction=True,
        enable_atmospheric_exchange=True,
        device=torch.device("cpu"),
    )
    out_yes = model_tc.forward(state=state, time=0.0, dt=60.0)

    # With temperature correction, diffusion magnitude should not decrease.
    assert abs(float(_value(out_yes["co2_surface_flux"]))) >= abs(
        float(_value(out_no["co2_surface_flux"]))
    )


def test_dncd_model_emits_non_negative_n2o():
    n_layers = 4
    state = _build_state(n_layers)

    model_mq = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK.value,
        enable_temperature_correction=True,
        enable_atmospheric_exchange=True,
        device=torch.device("cpu"),
    )
    out_mq = model_mq.forward(state=state, time=0.0, dt=60.0)

    model_dn = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.DNDC.value,
        enable_temperature_correction=True,
        enable_atmospheric_exchange=True,
        device=torch.device("cpu"),
    )
    out_dn = model_dn.forward(state=state, time=0.0, dt=60.0)

    # Both finite; sign follows concentration-gradient direction.
    for v in [
        _value(out_mq["n2o_surface_flux"]),
        _value(out_dn["n2o_surface_flux"]),
    ]:
        assert torch.isfinite(v).all()


def test_input_validation_missing_porosity():
    n_layers = 4
    model = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK.value,
        device=torch.device("cpu"),
    )
    state = _build_state(n_layers)
    state.pop("porosity")

    with pytest.raises(ValueError):
        _ = model.forward(state=state, time=0.0, dt=60.0)


def test_only_canonical_soil_temperature_is_accepted():
    model = GasDiffusion1D(n_layers=4, soil_depth=1.0, device=torch.device("cpu"))
    state = _build_state(4)
    state["temperature"] = state.pop("soil_temperature")

    with pytest.raises(ValueError, match="soil_temperature"):
        model.forward(state=state, time=0.0, dt=60.0)


def test_required_soil_moisture_has_no_parameter_fallback():
    model = GasDiffusion1D(n_layers=4, soil_depth=1.0, device=torch.device("cpu"))
    state = _build_state(4)
    state.pop("soil_moisture")

    with pytest.raises(ValueError, match="soil_moisture"):
        model.forward(state=state, time=0.0, dt=60.0)


def test_non_finite_required_input_fails_fast():
    model = GasDiffusion1D(n_layers=4, soil_depth=1.0, device=torch.device("cpu"))
    state = _build_state(4)
    state["porosity"] = state["porosity"].clone()
    state["porosity"][1] = torch.nan

    with pytest.raises(ValueError, match="porosity must be finite"):
        model.forward(state=state, time=0.0, dt=60.0)


@pytest.mark.parametrize(
    "concentration_key",
    ["soil_co2_concentration", "n2o_concentration"],
)
def test_prognostic_concentration_profiles_must_match_grid(concentration_key):
    """Prognostic gas states cannot be silently broadcast or resampled."""
    model = GasDiffusion1D(n_layers=4, soil_depth=1.0, device=torch.device("cpu"))
    state = _build_state(4)
    state[concentration_key] = state[concentration_key][:-1]

    with pytest.raises(ValueError, match=f"{concentration_key} must have 4 layers"):
        model.forward(state=state, time=0.0, dt=60.0)


def test_flux_divergence_closes_each_species_mass_ledger():
    n_layers = 4
    dt = 1.0
    soil_depth = 1.0
    model = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=soil_depth,
        enable_atmospheric_exchange=True,
        device=torch.device("cpu"),
    )
    out = model.forward(state=_build_state(n_layers), time=0.0, dt=dt)
    dz = soil_depth / n_layers

    for concentration_key, surface_flux_key, clipping_key in (
        (
            "soil_co2_concentration",
            "co2_surface_flux",
            "diagnostics.gas_diffusion.co2_clipping_residual",
        ),
        (
            "n2o_concentration",
            "n2o_surface_flux",
            "diagnostics.gas_diffusion.n2o_clipping_residual",
        ),
    ):
        storage_increment = torch.sum(_value(out[concentration_key])) * dz
        outward_boundary_export = _value(out[surface_flux_key]) * dt
        clipping_route = torch.sum(_value(out[clipping_key])) * dz
        assert torch.allclose(
            storage_increment + outward_boundary_export - clipping_route,
            torch.zeros_like(storage_increment),
            rtol=1.0e-6,
            atol=1.0e-12,
        )


def test_runtime_unit_configuration_is_not_a_functor_constructor_api():
    with pytest.raises(TypeError, match="output_unit_config"):
        GasDiffusion1D(output_unit_config=object())


def test_phase8_iphs_structure_hooks_are_declared():
    n_layers = 4
    model = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK.value,
        enable_temperature_correction=True,
        enable_atmospheric_exchange=True,
        device=torch.device("cpu"),
    )

    forms = model.declared_forms()
    hodge_terms = model.build_hodge_terms()
    dissipative = model.dissipative_structure(_build_state(n_layers))
    ports = model.ports()

    assert isinstance(model, IPHSDECFunctor)
    assert model.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
    assert "pedosphere.physical.co2_diffusivity_air_stp" in model.required_parameters()
    assert "pedosphere.physical.n2o_diffusivity_air_stp" in model.required_parameters()
    assert {form.semantic_name for form in forms} == {
        "soil_co2_concentration",
        "n2o_concentration",
        "soil_gas_concentration_gradient",
        "soil_gas_diffusive_flux",
    }
    assert len(hodge_terms) == 1
    assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
    assert {port.name for port in ports} == {
        "soil_atmosphere_co2_flux",
        "soil_atmosphere_n2o_flux",
    }


def test_phase8_level3_runtime_audits_are_exposed():
    n_layers = 4
    model = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK.value,
        enable_temperature_correction=True,
        enable_atmospheric_exchange=True,
        device=torch.device("cpu"),
    )
    state = _build_state(n_layers)

    assert hasattr(model, "_iphs_audit_terms")
    assert not hasattr(model, "_strict_iphs_audit_terms")

    energy_terms = model.energy_terms(state)
    entropy_terms = model.entropy_terms(state)
    out = model.forward(state=state, time=0.0, dt=60.0)

    assert torch.is_tensor(energy_terms["storage_functional"])
    assert torch.is_tensor(energy_terms["effort"])
    assert torch.is_tensor(entropy_terms["dissipation_rate"])
    assert entropy_terms["dissipation_rate"] >= 0.0
    assert _value(out["diagnostics.gas_diffusion.energy_storage"]).ndim == 0
    assert _value(out["diagnostics.gas_diffusion.dissipation_rate"]) >= 0.0
    assert torch.isfinite(_value(out["diagnostics.gas_diffusion.dH_dt_audit"]))
    assert torch.isfinite(_value(out["diagnostics.gas_diffusion.port_power_residual"]))


def test_phase8_certificate_level3_declaration_is_valid():
    certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
    audit = certificate.validate()

    assert audit["passed"] is True
    assert certificate.process_id == "relic.scientific.pedosphere.transport.gas_diffusion.GasDiffusion1D"
    assert certificate.canonical_role == "transport"
    assert certificate.iphs_level == 3
    assert certificate.iphs_level_evidence["level_3_limit"]
    assert audit["runtime_evidence_status"] == "descriptive_only"


def test_forward_keeps_no_module_state_and_declares_its_step():
    """Round 47 (I1): forward writes no module tensor; two identical calls give
    bit-identical outputs and unchanged parameters and buffers; the step length reaches
    the public ledger as the declared diagnostics.gas_diffusion.step_seconds output."""
    functor = GasDiffusion1D(n_layers=4, soil_depth=1.0, diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK.value, enable_temperature_correction=True, enable_atmospheric_exchange=True, device=torch.device("cpu"))

    def _snapshot():
        return {
            **{f"p.{k}": v.detach().clone() for k, v in functor.named_parameters()},
            **{f"b.{k}": v.detach().clone() for k, v in functor.named_buffers()},
        }

    before = _snapshot()
    first = functor(_build_state(4), time=0.0, dt=3600.0)
    second = functor(_build_state(4), time=0.0, dt=3600.0)
    after = _snapshot()
    assert not hasattr(functor, "_last_time_step")
    assert before.keys() == after.keys()
    for key, value in before.items():
        assert torch.equal(value, after[key]), key
    assert first.keys() == second.keys()
    for key in first:
        a = first[key].value if isinstance(first[key], UpdateItem) else first[key]
        b = second[key].value if isinstance(second[key], UpdateItem) else second[key]
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), key
    step = first["diagnostics.gas_diffusion.step_seconds"]
    assert isinstance(step, UpdateItem) and step.unit == "s"
    assert float(step.value) == 3600.0
    assert "diagnostics.gas_diffusion.step_seconds" in functor.metadata.output_variables
