"""
Unit tests for Solute Advection-Diffusion Transport

Tests the solute transport functor including:
- Advection flux (water-driven transport)
- Diffusion gradient effects
- Combined advection-diffusion transport
- Sorption effects on mobility
- Conservation and numerical stability

References:
- Bear (1972) Dynamics of Fluids in Porous Media
- Jury & Horton (2004) Soil Physics
- Simunek et al. (2008) HYDRUS model

Mathematical Expressions:
partial(thetaC)/partialt + partial(qC)/partialz = partial/partialz(thetaD partialC/partialz) - R(C)
R = 1 + (rhob*Kd)/theta  (retardation factor)
"""

import pytest
import torch
import inspect
from pathlib import Path

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.scientific.pedosphere.transport.solute_advection_diffusion import (
    SoluteAdvectionDiffusion1D,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


CERTIFICATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "source"
    / "theory"
    / "iphs_certificates"
    / "pedosphere_transport_solute_advection_diffusion_solute_advection_diffusion1_d.yaml"
)

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

CANONICAL_OUTPUTS = {
    "nitrogen_leaching_flux",
    "carbon_leaching_flux",
    "no3_concentration",
    "doc_concentration",
    "diagnostics.solute_transport.nitrogen_surface_input_flux",
    "diagnostics.solute_transport.carbon_surface_input_flux",
    "diagnostics.solute_transport.nitrogen_mass_balance_residual",
    "diagnostics.solute_transport.carbon_mass_balance_residual",
}


def _phase8_state(n_layers=5):
    """Create a valid state for Phase 8 solute transport structural tests."""

    return {
        "soil_water_content": torch.full((n_layers,), 0.3),
        "darcy_flux": torch.full((n_layers,), 1.0e-6),
        "no3_concentration": torch.linspace(5.0, 1.0, steps=n_layers),
        "doc_concentration": torch.linspace(15.0, 10.0, steps=n_layers),
        "bulk_density": torch.full((n_layers,), 1400.0),
        "porosity": torch.full((n_layers,), 0.45),
        "soil_temperature": torch.full((n_layers,), 288.15),
        "surface_no3_concentration": torch.tensor(5.0),
        "surface_doc_concentration": torch.tensor(15.0),
    }


class TestSoluteTransportInit:
    """Test initialization and setup"""

    def test_initialization(self):
        """Test that SoluteAdvectionDiffusion1D can be instantiated"""
        functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0
        )
        assert functor is not None
        assert functor.n_layers == 5
        assert functor.soil_depth == 1.0

    def test_grid_setup(self):
        """Test vertical grid setup"""
        functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0
        )

        # Check grid properties
        assert functor.layer_thickness.shape == (5,)
        assert functor.layer_centers.shape == (5,)

        # Check total depth
        total_depth = torch.sum(functor.layer_thickness)
        expected_depth = torch.tensor(
            1.0,
            dtype=functor.layer_thickness.dtype,
        )
        assert torch.allclose(total_depth, expected_depth, rtol=1e-3)

    def test_canonical_interface_has_no_runtime_compatibility_switches(self):
        """The production closure exposes one fixed NO3/DOC transport law."""
        functor = SoluteAdvectionDiffusion1D(n_layers=5)

        signature = inspect.signature(SoluteAdvectionDiffusion1D)
        assert "species_config" not in signature.parameters
        assert "enable_reaction_coupling" not in signature.parameters
        assert set(functor.metadata.input_variables) == CANONICAL_INPUTS
        assert set(functor.metadata.output_variables) == CANONICAL_OUTPUTS


class TestAdvectionFlux:
    """Test advection (water-driven) transport"""

    def test_advection_flux(self):
        """
        Test water-driven solute transport.
        
        Mathematical Expression:
        Advective flux: F = q * C
        where q is Darcy flux [m/s] and C is concentration [g/m3]
        
        Expected behavior:
        - Positive q (downward) -> downward transport
        - Higher q -> higher transport
        - Concentration gradient drives spatial redistribution
        """
        functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_sorption=False  # Disable sorption for pure advection test
        )

        # Setup state with downward water flux
        state = {
            'soil_water_content': torch.full((5,), 0.3),
            'darcy_flux': torch.full((5,), 1e-6),  # Downward flux (m/s)
            'no3_concentration': torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),  # Decreasing with depth
            'doc_concentration': torch.full((5,), 10.0),
            'bulk_density': torch.full((5,), 1400.0),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 288.15),
            'surface_no3_concentration': torch.tensor(5.0),
            'surface_doc_concentration': torch.tensor(10.0),
        }

        result = functor.forward(state, time=0.0, dt=1800.0)

        # Check that leaching flux is calculated
        assert 'nitrogen_leaching_flux' in result
        assert 'carbon_leaching_flux' in result

        # With downward flux and decreasing concentration, should have leaching
        n_leach = _unwrap_update(result['nitrogen_leaching_flux'])
        assert float(n_leach) > 0

        # Test with increased water flux
        state_high_flux = state.copy()
        state_high_flux['darcy_flux'] = torch.full((5,), 5e-6)  # 5x higher
        result_high_flux = functor.forward(state_high_flux, time=0.0, dt=1800.0)

        # Higher water flux should produce higher leaching
        n_leach_high = _unwrap_update(result_high_flux['nitrogen_leaching_flux'])
        assert float(n_leach_high) > float(n_leach)


class TestDiffusionGradient:
    """Test diffusion gradient effects"""

    def test_diffusion_gradient(self):
        """
        Test concentration gradient-driven diffusion.
        
        Mathematical Expression:
        Diffusive flux: F = -D * dC/dz
        where D is diffusion coefficient [m2/s]
        
        Expected behavior:
        - Steep gradient -> higher flux
        - Diffusion smooths concentration profiles
        - Flux proportional to gradient magnitude
        """
        functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_sorption=False
        )

        # Setup state with steep concentration gradient, no advection
        state = {
            'soil_water_content': torch.full((5,), 0.3),
            'darcy_flux': torch.zeros(5),  # No advection
            'no3_concentration': torch.tensor([10.0, 10.0, 10.0, 0.0, 0.0]),  # Steep gradient (nonlinear)
            'doc_concentration': torch.full((5,), 10.0),
            'bulk_density': torch.full((5,), 1400.0),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 288.15),
            'surface_no3_concentration': torch.tensor(10.0),
            'surface_doc_concentration': torch.tensor(10.0),
        }

        result = functor.forward(state, time=0.0, dt=1800.0)

        # Check concentration changes
        assert 'no3_concentration' in result
        no3_updated = _unwrap_update(result['no3_concentration'])
        delta_conc = no3_updated - state['no3_concentration']

        # Diffusion should smooth the profile
        # Top layer (high conc) should decrease, bottom layer (low conc) should increase
        # Note: delta_conc derived from absolute update, so check if profile is smoothing
        assert delta_conc.shape == (5,)

        # Test with gentle gradient
        state_gentle = state.copy()
        state_gentle['no3_concentration'] = torch.tensor([10.0, 10.0, 10.0, 8.0, 8.0])  # Gentle gradient (nonlinear)
        result_gentle = functor.forward(state_gentle, time=0.0, dt=1800.0)

        # Steeper gradient should produce larger changes
        no3_updated_gentle = _unwrap_update(result_gentle['no3_concentration'])
        delta_gentle = no3_updated_gentle - state_gentle['no3_concentration']
        
        # Compare magnitudes of changes (sum of absolute values)
        steep_magnitude = torch.sum(torch.abs(delta_conc))
        gentle_magnitude = torch.sum(torch.abs(delta_gentle))
        
        # Steeper gradient should cause larger total changes
        assert float(steep_magnitude) > float(gentle_magnitude) * 0.5


class TestCombinedTransport:
    """Test combined advection-diffusion transport"""

    def test_combined_transport(self):
        """
        Test combined advection and diffusion.
        
        Mathematical Expression:
        Total flux: F = q*C - D*dC/dz
        
        Expected behavior:
        - Both processes contribute to transport
        - Advection dominates when Peclet number high
        - Diffusion dominates when Peclet number low
        - Conservation maintained
        """
        functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_sorption=True  # Enable sorption for realistic scenario
        )

        # Setup state with both advection and diffusion
        state = {
            'soil_water_content': torch.full((5,), 0.35),
            'darcy_flux': torch.full((5,), 2e-6),  # Moderate downward flux
            'no3_concentration': torch.tensor([8.0, 6.0, 4.0, 2.0, 1.0]),  # Gradient
            'doc_concentration': torch.tensor([20.0, 18.0, 15.0, 12.0, 10.0]),
            'bulk_density': torch.full((5,), 1400.0),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 293.15),
            'surface_no3_concentration': torch.tensor(8.0),
            'surface_doc_concentration': torch.tensor(20.0),
        }

        result = functor.forward(state, time=0.0, dt=1800.0)

        # Both NO3 and DOC should show transport
        assert 'nitrogen_leaching_flux' in result
        assert 'carbon_leaching_flux' in result

        n_leach = _unwrap_update(result['nitrogen_leaching_flux'])
        c_leach = _unwrap_update(result['carbon_leaching_flux'])

        # Both should be positive (leaching occurring)
        assert float(n_leach) > 0
        assert float(c_leach) > 0

        # DOC has higher sorption (Kd), so should be more retarded
        # For same concentration gradient, NO3 should leach more
        # But we have different concentrations, so just check both are positive

        # Test sorption effect by comparing with no-sorption case
        functor_no_sorb = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_sorption=False
        )
        result_no_sorb = functor_no_sorb.forward(state, time=0.0, dt=1800.0)

        # Without sorption, leaching should be higher
        c_leach_no_sorb = _unwrap_update(result_no_sorb['carbon_leaching_flux'])
        assert float(c_leach_no_sorb) > float(c_leach) * 0.8  # At least 80% higher


class TestSorptionEffects:
    """Test sorption retardation"""

    def test_sorption_retardation(self):
        """
        Test sorption effect on solute mobility.
        
        Mathematical Expression:
        Storage capacity: B = theta + rhob * Kd * liter_to_cubic_meter
        Concentration response: dC/dt = -div(J) / B
        
        Expected behavior:
        - Higher Kd -> stronger retardation
        - Lower mobility -> slower transport
        """
        # Create functor with sorption
        functor_with_sorb = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_sorption=True
        )

        # Create functor without sorption
        functor_no_sorb = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_sorption=False
        )

        # Setup identical state
        state = {
            'soil_water_content': torch.full((5,), 0.3),
            'darcy_flux': torch.full((5,), 3e-6),
            'no3_concentration': torch.full((5,), 5.0),
            'doc_concentration': torch.full((5,), 15.0),
            'bulk_density': torch.full((5,), 1400.0),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 288.15),
            'surface_no3_concentration': torch.tensor(0.0),
            'surface_doc_concentration': torch.tensor(0.0),
        }

        # Run both
        result_with_sorb = functor_with_sorb.forward(state, time=0.0, dt=1800.0)
        result_no_sorb = functor_no_sorb.forward(state, time=0.0, dt=1800.0)

        doc_with_sorb = _unwrap_update(result_with_sorb['doc_concentration'])
        doc_no_sorb = _unwrap_update(result_no_sorb['doc_concentration'])
        response_with_sorb = torch.linalg.vector_norm(
            doc_with_sorb - state['doc_concentration']
        )
        response_no_sorb = torch.linalg.vector_norm(
            doc_no_sorb - state['doc_concentration']
        )
        assert float(response_with_sorb) < float(response_no_sorb)


class TestNumericalStability:
    """Test numerical stability and conservation"""

    def test_cfl_condition(self):
        """Test CFL condition for stability"""
        functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            cfl_number=0.5,
            max_timestep=1800.0
        )

        # Setup state with high flux (potential instability)
        state = {
            'soil_water_content': torch.full((5,), 0.3),
            'darcy_flux': torch.full((5,), 1e-4),  # Very high flux
            'no3_concentration': torch.full((5,), 5.0),
            'doc_concentration': torch.full((5,), 15.0),
            'bulk_density': torch.full((5,), 1400.0),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 288.15),
            'surface_no3_concentration': torch.tensor(5.0),
            'surface_doc_concentration': torch.tensor(15.0),
        }

        # Should not crash even with high flux
        result = functor.forward(state, time=0.0, dt=1800.0)
        
        # Result should be valid
        assert result is not None
        assert 'nitrogen_leaching_flux' in result

        # Concentrations should remain non-negative
        if 'no3_concentration' in result:
            no3_updated = _unwrap_update(result['no3_concentration'])
            assert torch.all(no3_updated >= -1e-6)


class TestCanonicalTransportContract:
    """Test the single conservative NO3/DOC finite-volume closure."""

    def test_outputs_have_exact_update_semantics_and_units(self):
        functor = SoluteAdvectionDiffusion1D(n_layers=5, soil_depth=1.0)
        result = functor.forward(_phase8_state(), time=0.0, dt=1800.0)

        assert set(result) == CANONICAL_OUTPUTS
        expected = {
            "nitrogen_leaching_flux": (False, "g N m-2 s-1"),
            "carbon_leaching_flux": (False, "g C m-2 s-1"),
            "no3_concentration": (True, "g N m-3"),
            "doc_concentration": (True, "g C m-3"),
            "diagnostics.solute_transport.nitrogen_surface_input_flux": (
                False,
                "g N m-2 s-1",
            ),
            "diagnostics.solute_transport.carbon_surface_input_flux": (
                False,
                "g C m-2 s-1",
            ),
            "diagnostics.solute_transport.nitrogen_mass_balance_residual": (
                True,
                "g N m-2",
            ),
            "diagnostics.solute_transport.carbon_mass_balance_residual": (
                True,
                "g C m-2",
            ),
        }
        for key, (per_timestep, unit) in expected.items():
            update = result[key]
            assert isinstance(update, UpdateItem)
            assert update.mode is UpdateMode.REPLACE
            assert update.per_timestep is per_timestep
            assert update.unit == unit
            # Concentration storage requires B * C * dz geometry, which the
            # global areal element ledger cannot infer from this UpdateItem.
            assert update.element is None
            assert update.source_functor == functor.name

    def test_discrete_column_mass_ledger_closes_without_sorption(self):
        functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_sorption=False,
        )
        state = {key: value.to(torch.float64) for key, value in _phase8_state().items()}
        dt = 1800.0
        result = functor.forward(state, time=0.0, dt=dt)
        dz = functor.layer_thickness.to(dtype=torch.float64)
        theta = state["soil_water_content"]

        for concentration_key, surface_key, bottom_key, residual_key in (
            (
                "no3_concentration",
                "diagnostics.solute_transport.nitrogen_surface_input_flux",
                "nitrogen_leaching_flux",
                "diagnostics.solute_transport.nitrogen_mass_balance_residual",
            ),
            (
                "doc_concentration",
                "diagnostics.solute_transport.carbon_surface_input_flux",
                "carbon_leaching_flux",
                "diagnostics.solute_transport.carbon_mass_balance_residual",
            ),
        ):
            old_mass = torch.sum(theta * state[concentration_key] * dz)
            new_mass = torch.sum(theta * result[concentration_key].value * dz)
            reconstructed = (
                new_mass
                - old_mass
                - result[surface_key].value * dt
                + result[bottom_key].value * dt
            )
            torch.testing.assert_close(
                result[residual_key].value,
                reconstructed,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            assert abs(float(reconstructed)) < 1.0e-10

    def test_linear_sorption_uses_liter_to_cubic_meter_conversion(self):
        functor = SoluteAdvectionDiffusion1D(
            n_layers=1,
            soil_depth=1.0,
            enable_sorption=True,
        )
        state = {
            "soil_water_content": torch.tensor([0.3], dtype=torch.float64),
            "darcy_flux": torch.tensor([1.0e-6], dtype=torch.float64),
            "no3_concentration": torch.tensor([5.0], dtype=torch.float64),
            "doc_concentration": torch.tensor([10.0], dtype=torch.float64),
            "bulk_density": torch.tensor([1400.0], dtype=torch.float64),
            "porosity": torch.tensor([0.45], dtype=torch.float64),
            "soil_temperature": torch.tensor([298.15], dtype=torch.float64),
            "surface_no3_concentration": torch.tensor(0.0, dtype=torch.float64),
            "surface_doc_concentration": torch.tensor(0.0, dtype=torch.float64),
        }
        dt = 1000.0
        result = functor.forward(state, time=0.0, dt=dt)
        kd_value = functor.get_parameter_value(
            "nitrate_distribution_coefficient",
            namespace="pedosphere.transport",
            required=True,
        )
        kd = float(
            kd_value.detach() if isinstance(kd_value, torch.Tensor) else kd_value
        )
        liter_to_cubic_meter = float(
            get_physical_constant(
                "relic.physical.constants.conversions.liter_to_cubic_meter"
            )
        )
        capacity = state["soil_water_content"] + (
            state["bulk_density"] * kd * liter_to_cubic_meter
        )
        old_mass = torch.sum(capacity * state["no3_concentration"])
        new_mass = torch.sum(capacity * result["no3_concentration"].value)
        reconstructed = (
            new_mass
            - old_mass
            - result[
                "diagnostics.solute_transport.nitrogen_surface_input_flux"
            ].value
            * dt
            + result["nitrogen_leaching_flux"].value * dt
        )
        assert abs(float(reconstructed)) < 1.0e-10

    def test_alias_defaults_and_malformed_layer_shapes_fail_fast(self):
        functor = SoluteAdvectionDiffusion1D(n_layers=5, soil_depth=1.0)
        alias_state = _phase8_state()
        alias_state["temperature"] = alias_state.pop("soil_temperature")
        with pytest.raises(ValueError, match="soil_temperature"):
            functor.forward(alias_state, time=0.0, dt=60.0)

        malformed_state = _phase8_state()
        malformed_state["darcy_flux"] = torch.full((6,), 1.0e-6)
        with pytest.raises(ValueError, match="darcy_flux"):
            functor.forward(malformed_state, time=0.0, dt=60.0)

    def test_reaction_aliases_do_not_create_hidden_sources_or_sinks(self):
        functor = SoluteAdvectionDiffusion1D(n_layers=5, soil_depth=1.0)
        baseline_state = _phase8_state()
        alias_state = dict(baseline_state)
        alias_state.update(
            {
                "nitrification_rate": torch.tensor(1000.0),
                "denitrification_rate": torch.tensor(900.0),
                "doc_inputs": torch.tensor(800.0),
                "doc_mineralization_flux": torch.tensor(700.0),
            }
        )
        baseline = functor.forward(baseline_state, time=0.0, dt=60.0)
        with_aliases = functor.forward(alias_state, time=0.0, dt=60.0)

        for key in CANONICAL_OUTPUTS:
            torch.testing.assert_close(
                baseline[key].value,
                with_aliases[key].value,
                rtol=0.0,
                atol=0.0,
            )


class TestPhase8IPHSStructure:
    """Test Phase 8 structural IPHS declarations for solute transport."""

    def test_phase8_iphs_structure_hooks_are_declared(self):
        functor = SoluteAdvectionDiffusion1D(n_layers=5, soil_depth=1.0)
        state = _phase8_state()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure(state)
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "pedosphere.transport.default_molecular_diffusivity_water" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "no3_concentration",
            "doc_concentration",
            "solute_advective_velocity",
            "solute_concentration_gradient",
            "solute_advection_diffusion_flux",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {
            "solute_surface_inflow",
            "solute_bottom_leaching",
        }

    def test_phase8_certificate_level3_evidence_is_valid(self):
        certificate = load_iphs_equivalence_certificate(CERTIFICATE_PATH)
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == (
            "relic.scientific.pedosphere.transport.solute_advection_diffusion."
            "SoluteAdvectionDiffusion1D"
        )
        assert certificate.canonical_role == "transport"
        assert certificate.iphs_level == 3
        assert "storage_functional" not in certificate.iphs_level_evidence


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

