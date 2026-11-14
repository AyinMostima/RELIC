"""Numerical tests for the canonical soil solute transport closure.

References
----------
Simunek et al. (2008), Vadose Zone Journal 7, 587-600.
HYDRUS-1D Version 4.16 Technical Manual.
Bear (1972), Dynamics of Fluids in Porous Media.

Mathematical expression
-----------------------
The tested law is B dC/dt = -div(J), with B = theta + rho_b Kd and
J = q C_upwind - K grad(C).
"""

import pytest
import torch

from relic.core.functor import ProcessCategory
from relic.core.state import UpdateItem
from relic.parameters import get_global_manager
from relic.scientific.pedosphere.transport.solute_advection_diffusion import (
    SoluteAdvectionDiffusion1D,
)
from relic.utils.physical_constants import get_physical_constant


CANONICAL_INPUTS = {
    "soil_water_content",
    "darcy_flux",
    "no3_concentration",
    "doc_concentration",
    "bulk_density",
    "porosity",
    "soil_temperature",
    "surface_no3_concentration",
    "surface_doc_concentration",
}


@pytest.fixture
def functor() -> SoluteAdvectionDiffusion1D:
    """Create a five-layer transport closure."""

    return SoluteAdvectionDiffusion1D(
        n_layers=5,
        soil_depth=1.0,
        cfl_number=0.5,
        max_timestep=1800.0,
    )


@pytest.fixture
def basic_state() -> dict[str, torch.Tensor]:
    """Create a finite canonical state."""

    return {
        "soil_water_content": torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10]),
        "darcy_flux": torch.full((5,), 1.0e-6),
        "no3_concentration": torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0]),
        "doc_concentration": torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0]),
        "bulk_density": torch.tensor([1300.0, 1350.0, 1400.0, 1450.0, 1500.0]),
        "porosity": torch.tensor([0.45, 0.42, 0.40, 0.38, 0.35]),
        "soil_temperature": torch.tensor([288.15, 287.15, 286.15, 285.15, 284.15]),
        "surface_no3_concentration": torch.tensor(10.0),
        "surface_doc_concentration": torch.tensor(50.0),
    }


class TestConfigurationAndContracts:
    """Test parameter binding, grid buffers, and formal declarations."""

    def test_initialization_uses_parameter_manager(self):
        instance = SoluteAdvectionDiffusion1D()
        manager = get_global_manager()
        assert instance.n_layers == int(
            manager.get("pedosphere.grid.n_layers", required=True)
        )
        assert instance.soil_depth == pytest.approx(
            float(manager.get("pedosphere.grid.soil_depth", required=True))
        )
        assert instance.cfl_number == pytest.approx(
            float(manager.get("pedosphere.transport.cfl_number", required=True))
        )
        assert instance.metadata.category is ProcessCategory.TRANSPORT_MECHANISM
        assert "Canonical IPHS:" in instance.metadata.mathematical_expression
        assert "B_i C_i" in SoluteAdvectionDiffusion1D.forward.__doc__

    def test_uniform_grid_is_registered(self):
        instance = SoluteAdvectionDiffusion1D(
            n_layers=3,
            soil_depth=1.5,
            layer_thickness_ratio=1.0,
        )
        torch.testing.assert_close(
            instance.layer_thickness,
            torch.full((3,), 0.5, dtype=instance.layer_thickness.dtype),
        )
        buffers = dict(instance.named_buffers())
        assert {"layer_thickness", "layer_centers", "layer_interfaces"} <= set(buffers)

    def test_geometric_grid_preserves_depth_and_migrates_dtype(self):
        instance = SoluteAdvectionDiffusion1D(
            n_layers=3,
            soil_depth=1.0,
            layer_thickness_ratio=2.0,
        )
        assert instance.layer_thickness[0] < instance.layer_thickness[1]
        assert instance.layer_thickness[1] < instance.layer_thickness[2]
        torch.testing.assert_close(
            torch.sum(instance.layer_thickness),
            torch.tensor(1.0, dtype=instance.layer_thickness.dtype),
        )
        instance = instance.to(dtype=torch.float64)
        assert instance.layer_thickness.dtype is torch.float64
        assert instance.layer_centers.dtype is torch.float64
        assert instance.layer_interfaces.dtype is torch.float64

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_layers": 0}, "n_layers"),
            ({"soil_depth": 0.0}, "soil_depth"),
            ({"cfl_number": 0.0}, "cfl_number"),
            ({"cfl_number": 1.1}, "cfl_number"),
            ({"max_timestep": 0.0}, "max_timestep"),
            ({"layer_thickness_ratio": 0.0}, "layer_thickness_ratio"),
        ],
    )
    def test_invalid_configuration_fails_fast(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SoluteAdvectionDiffusion1D(**kwargs)

    def test_declared_runtime_surface_is_exact(self, functor):
        assert functor.get_dependencies() == {"required": [], "optional": []}
        assert set(functor.get_required_inputs()) == CANONICAL_INPUTS
        assert set(functor.metadata.input_variables) == CANONICAL_INPUTS

    def test_output_specifications_use_element_units(self, functor):
        outputs = functor.get_output_specifications()
        assert outputs["nitrogen_leaching_flux"].units == "g N m-2 s-1"
        assert outputs["carbon_leaching_flux"].units == "g C m-2 s-1"
        assert outputs["no3_concentration"].units == "g N m-3"
        assert outputs["doc_concentration"].units == "g C m-3"


class TestValidation:
    """Test fail-fast validation without aliases or hidden defaults."""

    def test_valid_state_passes(self, functor, basic_state):
        assert functor.validate_inputs(basic_state) == []

    def test_missing_and_non_tensor_inputs_fail(self, functor, basic_state):
        missing = dict(basic_state)
        missing.pop("soil_temperature")
        assert any("soil_temperature" in item for item in functor.validate_inputs(missing))

        wrong_type = dict(basic_state)
        wrong_type["surface_no3_concentration"] = 10.0
        assert any("torch.Tensor" in item for item in functor.validate_inputs(wrong_type))

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("soil_water_content", torch.full((5,), 0.5), "must not exceed"),
            ("darcy_flux", torch.full((5,), -1.0e-6), "non-negative"),
            ("no3_concentration", torch.full((5,), -1.0), "non-negative"),
            ("bulk_density", torch.zeros(5), "positive"),
            ("soil_temperature", torch.full((5,), float("nan")), "finite"),
            ("surface_doc_concentration", torch.tensor([1.0, 2.0]), "scalar"),
        ],
    )
    def test_invalid_domains_fail(self, functor, basic_state, key, value, message):
        state = dict(basic_state)
        state[key] = value
        assert any(message in item for item in functor.validate_inputs(state))

    def test_layer_extraction_rejects_broadcast_and_truncation(self, functor):
        layers = torch.arange(5, dtype=torch.float32)
        torch.testing.assert_close(functor._extract_layer_values(layers), layers)
        with pytest.raises(ValueError, match="exactly 5"):
            functor._extract_layer_values(torch.tensor(0.3), "theta")
        with pytest.raises(ValueError, match="exactly 5"):
            functor._extract_layer_values(torch.arange(6), "theta")


class TestConservativeNumerics:
    """Test storage, dispersion, fluxes, stability, and autograd."""

    def test_storage_capacity_without_sorption_is_theta(self, functor):
        functor.enable_sorption = False
        theta = torch.full((5,), 0.3)
        density = torch.full((5,), 1400.0)
        torch.testing.assert_close(
            functor._storage_capacity(theta, density, "NO3"),
            theta,
        )

    def test_storage_capacity_applies_si_conversion(self, functor):
        theta = torch.full((5,), 0.3, dtype=torch.float64)
        density = torch.full((5,), 1400.0, dtype=torch.float64)
        kd_value = functor.get_parameter_value(
            "nitrate_distribution_coefficient",
            namespace="pedosphere.transport",
            required=True,
        )
        kd = float(kd_value.detach())
        conversion = float(
            get_physical_constant(
                "relic.physical.constants.conversions.liter_to_cubic_meter"
            )
        )
        torch.testing.assert_close(
            functor._storage_capacity(theta, density, "NO3"),
            theta + density * kd * conversion,
        )

    def test_dispersion_is_positive_and_species_specific(self, functor, basic_state):
        args = (
            basic_state["soil_water_content"],
            basic_state["darcy_flux"],
            basic_state["soil_temperature"],
            basic_state["porosity"],
        )
        no3 = functor._calculate_dispersion_coefficient(*args, "NO3")
        doc = functor._calculate_dispersion_coefficient(*args, "DOC")
        assert torch.all(no3 > 0.0)
        assert torch.all(doc > 0.0)
        assert torch.all(no3 > doc)

    def test_dispersion_responds_to_temperature_and_porosity(self, functor):
        theta = torch.full((5,), 0.2)
        q = torch.zeros(5)
        temperature = torch.full((5,), 288.15)
        high_phi = torch.full((5,), 0.50)
        low_phi = torch.full((5,), 0.35)
        reference = functor._calculate_dispersion_coefficient(
            theta, q, temperature, high_phi, "NO3"
        )
        warmer = functor._calculate_dispersion_coefficient(
            theta, q, temperature + 10.0, high_phi, "NO3"
        )
        lower_porosity = functor._calculate_dispersion_coefficient(
            theta, q, temperature, low_phi, "NO3"
        )
        assert torch.all(warmer > reference)
        assert torch.all(lower_porosity > reference)

    def test_face_fluxes_match_open_boundaries(self, functor):
        concentration = torch.full((5,), 5.0)
        q = torch.full((5,), 1.0e-6)
        fluxes = functor._face_fluxes(
            concentration,
            q,
            torch.zeros(5),
            torch.tensor(7.0),
        )
        torch.testing.assert_close(fluxes[0], torch.tensor(7.0e-6))
        torch.testing.assert_close(fluxes[-1], torch.tensor(5.0e-6))
        torch.testing.assert_close(fluxes[1:-1], torch.full((4,), 5.0e-6))

    def test_internal_dispersion_follows_negative_gradient(self, functor):
        fluxes = functor._face_fluxes(
            torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]),
            torch.zeros(5),
            torch.full((5,), 1.0e-6),
            torch.tensor(0.0),
        )
        assert fluxes[1] > 0.0
        assert fluxes[0] == 0.0
        assert fluxes[-1] == 0.0

    def test_stable_timestep_decreases_for_faster_flow(self, functor):
        capacity = torch.full((5,), 0.3)
        dispersion = torch.full((5,), 1.0e-7)
        slow = functor._calculate_stable_timestep(
            capacity, torch.full((5,), 1.0e-6), dispersion, 3600.0
        )
        fast = functor._calculate_stable_timestep(
            capacity, torch.full((5,), 1.0e-3), dispersion, 3600.0
        )
        assert 0.0 < fast < slow <= functor.max_timestep

    def test_zero_flow_uniform_profiles_are_unchanged(self, functor, basic_state):
        state = dict(basic_state)
        state["darcy_flux"] = torch.zeros(5)
        state["no3_concentration"] = torch.full((5,), 5.0)
        state["doc_concentration"] = torch.full((5,), 10.0)
        result = functor.forward(state, time=0.0, dt=1800.0)
        torch.testing.assert_close(result["no3_concentration"].value, state["no3_concentration"])
        torch.testing.assert_close(result["doc_concentration"].value, state["doc_concentration"])
        assert result["nitrogen_leaching_flux"].value == 0.0
        assert result["carbon_leaching_flux"].value == 0.0

    def test_closed_internal_diffusion_conserves_mass(self, functor, basic_state):
        state = dict(basic_state)
        state["darcy_flux"] = torch.zeros(5)
        state["surface_no3_concentration"] = torch.tensor(0.0)
        state["surface_doc_concentration"] = torch.tensor(0.0)
        result = functor.forward(state, time=0.0, dt=1800.0)
        assert abs(float(result[
            "diagnostics.solute_transport.nitrogen_mass_balance_residual"
        ].value)) < 1.0e-6
        assert abs(float(result[
            "diagnostics.solute_transport.carbon_mass_balance_residual"
        ].value)) < 1.0e-6

    def test_forward_preserves_autograd(self, functor, basic_state):
        state = dict(basic_state)
        state["no3_concentration"] = basic_state["no3_concentration"].clone().requires_grad_()
        state["doc_concentration"] = basic_state["doc_concentration"].clone().requires_grad_()
        result = functor.forward(state, time=0.0, dt=300.0)
        loss = (
            result["no3_concentration"].value.sum()
            + result["doc_concentration"].value.sum()
            + result["nitrogen_leaching_flux"].value
            + result["carbon_leaching_flux"].value
        )
        loss.backward()
        assert torch.all(torch.isfinite(state["no3_concentration"].grad))
        assert torch.all(torch.isfinite(state["doc_concentration"].grad))

    @pytest.mark.parametrize("dt", [60.0, 300.0, 1800.0])
    def test_forward_is_finite_across_timesteps(self, functor, basic_state, dt):
        result = functor.forward(basic_state, time=0.0, dt=dt)
        assert all(isinstance(value, UpdateItem) for value in result.values())
        assert all(torch.all(torch.isfinite(value.value)) for value in result.values())

    def test_high_flow_substeps_remain_non_negative(self, functor, basic_state):
        state = dict(basic_state)
        state["darcy_flux"] = torch.full((5,), 1.0e-4)
        result = functor.forward(state, time=0.0, dt=1800.0)
        assert torch.all(result["no3_concentration"].value >= 0.0)
        assert torch.all(result["doc_concentration"].value >= 0.0)

    def test_surface_diagnostics_match_cauchy_flux(self, functor, basic_state):
        result = functor.forward(basic_state, time=0.0, dt=300.0)
        torch.testing.assert_close(
            result[
                "diagnostics.solute_transport.nitrogen_surface_input_flux"
            ].value,
            basic_state["darcy_flux"][0]
            * basic_state["surface_no3_concentration"],
        )
        torch.testing.assert_close(
            result[
                "diagnostics.solute_transport.carbon_surface_input_flux"
            ].value,
            basic_state["darcy_flux"][0]
            * basic_state["surface_doc_concentration"],
        )
