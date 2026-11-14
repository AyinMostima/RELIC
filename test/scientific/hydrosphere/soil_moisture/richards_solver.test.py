"""
L1 Unit Tests for Richards Equation Solver

IMPORTANT NOTE:
--------------
The Richards equation implementation uses one canonical Phase 8 IPHS runtime
path. Legacy approximations are not callable from production code.

Test Strategy:
--------------
- TEST the canonical mixed-form Richards runtime path
- TEST basic hydraulic functions (Van Genuchten-Mualem models)
- PASS tests that work with the canonical runtime

Tests in this file have been adjusted to match the current implementation status.
"""

import pytest
import torch
import sys
import os
from pathlib import Path
from types import SimpleNamespace

# Add relic to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
from relic.core.functor import IPHSDECFunctor
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


def _get_richards_diagnostics(result):
    diagnostics = result.get("diagnostics.soil_moisture_richards", {})
    if isinstance(diagnostics, dict):
        return diagnostics
    return {}


def _certificate_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "docs"
        / "source"
        / "theory"
        / "iphs_certificates"
        / name
    )


class TestRichardsFullSolution:
    """Test suite for Richards equation complete numerical solution."""
    
    @pytest.fixture
    def parameter_overrides(self):
        """Parameter overrides for Richards model tests."""
        return {
            "hydrosphere.soil_moisture.layer_count": (5, "dimensionless"),
            "hydrosphere.soil_moisture.soil_depth": (1.0, "m"),
            "hydrosphere.soil_moisture.residual_water_content": (0.05, "m3 m-3"),
            "hydrosphere.soil_moisture.porosity": (0.45, "m3 m-3"),
            "hydrosphere.soil_moisture.alpha_vg": (5.0, "m-1"),
            "hydrosphere.soil_moisture.n_vg": (1.5, "dimensionless"),
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (1e-5, "m s-1"),
            "hydrosphere.soil_moisture.tortuosity_parameter": (0.5, "dimensionless"),
        }

    @pytest.fixture
    def parameter_manager_snapshot(self):
        """Snapshot and restore ParameterManager state."""
        manager = get_global_manager()
        snapshot = manager.snapshot()
        yield manager
        manager.restore(snapshot)
    
    @pytest.fixture
    def richards_model(self, parameter_manager_snapshot, parameter_overrides):
        """Create Richards model instance."""
        manager = parameter_manager_snapshot
        for key, (value, units) in parameter_overrides.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)
        return RichardsSoilMoisture(name="test_richards")
    
    def test_model_initialization(self, richards_model):
        """Test model initializes correctly."""
        assert richards_model.soil_layers == 5
        assert richards_model.total_soil_depth == 1.0
        assert richards_model.layer_thickness == 0.2
        assert richards_model.runtime_path == "canonical_iphs_mixed_form_richards"
        assert not hasattr(richards_model, "enable_simplified_mode")
        assert richards_model.theta_residual == 0.05
        assert richards_model.theta_saturated == 0.45
        assert set(richards_model.metadata.input_variables) == {
            "soil_moisture",
            "infiltration_flux",
            "evapotranspiration_flux",
            "capillary_rise_flux",
        }
        assert richards_model.optional_input_variables == {"capillary_rise_flux"}
        assert set(richards_model.get_dependencies()["optional"]) == {
            "surface_runoff",
            "penman_monteith_et",
            "priestley_taylor_et",
            "capillary_rise_transport",
        }

    def test_top_boundary_requires_explicit_head(
        self, parameter_manager_snapshot, parameter_overrides
    ):
        """Ensure a prescribed head is explicit in the boundary contract."""
        manager = parameter_manager_snapshot
        for key, (value, units) in parameter_overrides.items():
            manager.set_value(key, value, units=units, validate=True, notify=False)
        model = RichardsSoilMoisture(name="test_boundary_default")
        model.set_boundary_conditions(
            top={"type": "prescribed_head", "value": -0.5},
            bottom={"type": "free_drainage"},
        )
        theta = torch.full((model.soil_layers,), 0.25)
        h = model._compute_pressure_head(theta)
        k = model._compute_hydraulic_conductivity(theta)
        residual = model._numerical_solver._compute_newton_residual(
            theta, theta, h, k, dt=3600.0, surface_flux=0.0
        )
        theta_prescribed = model._compute_theta_from_head(torch.tensor(-0.5))
        assert torch.allclose(residual[0], theta[0] - theta_prescribed)
        
    def test_hydraulic_functions(self, richards_model):
        """Test Van Genuchten-Mualem hydraulic functions."""
        # Test moisture range
        theta_test = torch.linspace(0.1, 0.4, 10)
        
        # Pressure head calculation
        h = richards_model._compute_pressure_head(theta_test)
        assert torch.all(h <= 0), "Pressure heads should be negative for unsaturated conditions"
        assert torch.all(torch.isfinite(h)), "Pressure heads should be finite"
        
        # Hydraulic conductivity calculation
        k = richards_model._compute_hydraulic_conductivity(theta_test)
        assert torch.all(k > 0), "Hydraulic conductivity should be positive"
        assert torch.all(k <= richards_model.k_saturated), "K should not exceed K_sat"
        assert torch.all(torch.isfinite(k)), "Hydraulic conductivity should be finite"
        
    def test_adaptive_timestep(self, richards_model):
        """Test adaptive time stepping for stability."""
        theta = torch.tensor([0.2, 0.25, 0.3, 0.35, 0.4])
        dt_requested = 3600.0  # 1 hour

        dt_adaptive = richards_model._compute_adaptive_timestep(
            theta,
            dt_requested,
            cfl_limit=richards_model.richards_cfl_coefficient,
        )

        assert dt_adaptive > 0, "Adaptive time step should be positive"
        assert dt_adaptive <= dt_requested, "Adaptive time step should not exceed requested"
        
    def test_thomas_algorithm(self, richards_model):
        """Test Thomas algorithm for tridiagonal systems."""
        n = 5

        # Create a simple tridiagonal system with known solution
        a = torch.tensor([0.0, -1.0, -1.0, -1.0, -1.0])  # Lower diagonal
        b = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])       # Main diagonal
        c = torch.tensor([-1.0, -1.0, -1.0, -1.0, 0.0])   # Upper diagonal
        d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0])       # RHS

        solver_result = richards_model.thomas_solver.solve(a, b, c, d)
        x = solver_result.solution

        assert len(x) == n, "Solution should have correct length"
        assert torch.all(torch.isfinite(x)), "Solution should be finite"

        # Verify solution by matrix multiplication (Ax = d)
        residual = torch.zeros(n)
        residual[0] = b[0] * x[0] + c[0] * x[1] - d[0]
        for i in range(1, n-1):
            residual[i] = a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] - d[i]
        residual[n-1] = a[n-1] * x[n-2] + b[n-1] * x[n-1] - d[n-1]

        assert torch.max(torch.abs(residual)) < 1e-6, "Thomas algorithm solution should be accurate"
        
    def test_mass_conservation_check(self, richards_model):
        """Test mass conservation checking against the discrete water ledger."""
        theta_old = torch.tensor([0.2, 0.25, 0.3, 0.25, 0.2])
        theta_new = torch.tensor([0.22, 0.26, 0.31, 0.26, 0.21])

        surface_flux = 1e-6  # m/s
        bottom_flux = 0.5e-6  # m/s
        dt = 3600.0  # s

        error = richards_model._check_mass_conservation(
            theta_old, theta_new, surface_flux, bottom_flux, dt
        )

        expected = torch.abs(
            torch.sum(theta_new - theta_old) * richards_model.layer_thickness
            - (surface_flux - bottom_flux) * dt
        ) / (
            torch.sum(theta_old) * richards_model.layer_thickness
        )
        assert isinstance(error, torch.Tensor), "Mass conservation error should be a tensor"
        assert torch.isfinite(error), "Mass conservation error should be finite"
        assert error >= 0, "Mass conservation error should be non-negative"
        assert torch.allclose(
            error,
            expected,
            rtol=1e-7,
            atol=0.0,
        ), "Mass ledger should match the discrete equation"
        
    def test_full_solution_basic(self, richards_model):
        """Test basic functionality of full Richards solution."""
        # Initial state
        initial_moisture = 0.25  # Uniform initial condition
        state = {
            "soil_moisture": torch.full(
                (richards_model.soil_layers,), initial_moisture
            ),
            "infiltration_flux": torch.tensor(1e-6 * 1000.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }

        time = 0.0
        dt = 1800.0  # 30 minutes

        # Run full solution
        result = richards_model._compute_full_solution(state, time, dt)

        # Check output structure
        required_keys = [
            "soil_moisture",
            "diagnostics.soil_moisture_richards.soil_moisture_change_rate",
            "drainage_flux",
            "diagnostics.soil_moisture_richards",
        ]
        for key in required_keys:
            assert key in result, f"Missing required output: {key}"

        # Check physical constraints
        moisture_change = _unwrap_update(
            result["diagnostics.soil_moisture_richards.soil_moisture_change_rate"]
        )
        assert torch.all(torch.isfinite(moisture_change)), "Moisture changes should be finite"

        # Check solver convergence
        diagnostics = _get_richards_diagnostics(result)
        assert diagnostics["solver_converged"], "Solver should converge for basic test"

        # Check mass conservation
        mass_error = diagnostics["mass_conservation_error"]
        assert mass_error < 0.1, f"Mass conservation error too large: {mass_error:.2e}"
        
    def test_boundary_conditions(self, richards_model):
        """Test different boundary condition types."""
        initial_moisture = 0.2
        state = {
            "soil_moisture": torch.full(
                (richards_model.soil_layers,), initial_moisture
            ),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }

        # Test prescribed flux boundary
        richards_model.set_boundary_conditions(
            top={"type": "prescribed_flux", "value": 1e-6},
            bottom={"type": "free_drainage"}
        )

        result = richards_model._compute_full_solution(state, 0.0, 1800.0)
        diagnostics = _get_richards_diagnostics(result)
        assert diagnostics["solver_converged"], "Should converge with prescribed flux BC"

        # Test prescribed head boundary
        richards_model.set_boundary_conditions(
            top={"type": "prescribed_head", "value": -0.1},
            bottom={"type": "prescribed_flux", "value": 0.0}
        )

        result = richards_model._compute_full_solution(state, 0.0, 1800.0)
        diagnostics = _get_richards_diagnostics(result)
        assert diagnostics["solver_converged"], "Should converge with prescribed head BC"

    def test_capillary_boundary_requires_selected_finite_mass_flux(
        self, richards_model
    ):
        """Reject missing, invalid, or silently ignored capillary inputs."""
        base_state = {
            "soil_moisture": torch.full((richards_model.soil_layers,), 0.25),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }
        richards_model.set_boundary_conditions(bottom={"type": "capillary_flux"})
        with pytest.raises(ValueError, match="requires canonical"):
            richards_model.forward(base_state, 0.0, 60.0)

        for invalid_flux in (
            -1.0e-6,
            torch.tensor(-1.0e-6),
            torch.tensor(float("nan")),
            torch.ones(2),
        ):
            invalid_state = {**base_state, "capillary_rise_flux": invalid_flux}
            with pytest.raises(ValueError, match="capillary_rise_flux"):
                richards_model.forward(invalid_state, 0.0, 60.0)

        richards_model.set_boundary_conditions(bottom={"type": "free_drainage"})
        unselected_state = {
            **base_state,
            "capillary_rise_flux": torch.tensor(1.0e-6),
        }
        with pytest.raises(ValueError, match="explicit 'capillary_flux'"):
            richards_model.forward(unselected_state, 0.0, 60.0)

    def test_free_drainage_bottom_residual_uses_downward_positive_flux(self, richards_model):
        """A unit-gradient lower boundary has q_bottom = K_bottom > 0.

        References:
        - Celia et al. (1990), Water Resources Research 26, 1483-1496.

        Mathematical expression:
        - q = -K(dh/dz - 1); dh/dz = 0 implies q_bottom = K_bottom.

        Parameter meanings:
        - q_bottom: downward-positive lower boundary flux [m s-1].
        - K_bottom: lower-layer hydraulic conductivity [m s-1].
        """
        richards_model.set_boundary_conditions(bottom={"type": "free_drainage"})
        theta = torch.full((richards_model.soil_layers,), 0.30)
        h = richards_model._compute_pressure_head(theta)
        k = richards_model._compute_hydraulic_conductivity(theta)

        residual = richards_model._numerical_solver._compute_newton_residual(
            theta=theta,
            theta_old=theta,
            h_iter=h,
            k_iter=k,
            dt=1.0,
            surface_flux=float(k[0].detach()),
        )

        assert k[-1] > 0
        k_interface_above = (
            richards_model.harmonic_mean_factor
            * k[-2]
            * k[-1]
            / (k[-2] + k[-1])
        )
        expected_bottom_residual = (
            -k_interface_above + k[-1]
        ) / richards_model.layer_thickness
        assert torch.allclose(
            residual[-1],
            expected_bottom_residual,
            rtol=1e-6,
            atol=1e-12,
        )
        
    @staticmethod
    def _top_face_conductivity(model, monkeypatch, k_pair, requires_grad=False):
        """Recover K_face(0, 1) from the top-cell residual of prescribed K.

        References: Celia, Bouloutas and Zarba (1990), Water Resour. Res.
        26(7):1483-1496 (mixed-form cell balance).
        Mathematical expression: with theta = theta_old, uniform h, dt = 1 s
        and zero top flux, R_0 = K_face(0, 1) / dz, so K_face = R_0 dz.
        Parameter meanings: K_face interface conductivity, R_0 top-cell
        residual, dz layer thickness.
        Units: K [m s-1]; R_0 [m3 m-3]; dz [m]; dt [s].
        Domain/constraints: float64 profile; K_pair = (K_0, K_1), the other
        layers carry K_1.
        Sign convention: downward-positive q = -K_face (dh/dz - 1).
        """
        n = model.soil_layers
        k_profile = torch.full((n,), k_pair[1], dtype=torch.float64)
        k_profile[0] = k_pair[0]
        if requires_grad:
            k_profile.requires_grad_(True)
        model.set_boundary_conditions(
            top={"type": "prescribed_flux", "value": 0.0},
            bottom={"type": "free_drainage"},
        )
        monkeypatch.setattr(
            model, "_compute_hydraulic_conductivity", lambda theta: k_profile * 1.0
        )
        monkeypatch.setattr(
            model, "_compute_pressure_head", lambda theta: torch.zeros_like(theta)
        )
        theta = torch.full((n,), 0.30, dtype=torch.float64)
        residual = model._numerical_solver._compute_newton_residual(
            theta=theta,
            theta_old=theta,
            h_iter=torch.zeros_like(theta),
            k_iter=k_profile,
            dt=1.0,
            surface_flux=0.0,
        )
        dz = float(torch.as_tensor(model.layer_thickness).detach())
        return residual[0] * dz, k_profile

    @pytest.mark.parametrize(
        "k_pair, expected",
        [
            ((1.0e-12, 1.0e-12), 1.0e-12),
            ((1.0e-5, 1.0e-12), 2.0 * 1.0e-5 * 1.0e-12 / (1.0e-5 + 1.0e-12)),
            ((1.0e-12, 1.0e-5), 2.0 * 1.0e-5 * 1.0e-12 / (1.0e-5 + 1.0e-12)),
            ((1.0e-20, 1.0e-20), 1.0e-20),
            ((2.5e-7, 2.5e-7), 2.5e-7),
        ],
    )
    def test_residual_interface_conductivity_is_the_harmonic_mean(
        self, richards_model, monkeypatch, k_pair, expected
    ):
        """K_face = 2 K_i K_j / (K_i + K_j) at every conductivity magnitude.

        References: Belfort, Younes, Fahs and Lehmann (2013), J. Hydrol.
        505:202-217, Table 2 (harmonic interblock mean).
        Mathematical expression: K_face = 2 K_i K_j / (K_i + K_j); equal
        neighbours give K_face = K. No term in m s-1 is added to the sum.
        Parameter meanings: K_i, K_j conductivities of the two cells.
        Units: K [m s-1].
        Domain/constraints: dry-end van Genuchten-Mualem values reach
        1e-12 m s-1 and below; tolerance is relative 1e-12 in float64.
        Sign convention: K >= 0 is a magnitude.
        """
        face, _ = self._top_face_conductivity(richards_model, monkeypatch, k_pair)
        assert torch.isfinite(face)
        assert float(face.detach()) == pytest.approx(expected, rel=1.0e-12, abs=0.0)

    @pytest.mark.parametrize("k_pair", [(0.0, 0.0), (0.0, 1.0e-6), (1.0e-6, 0.0)])
    def test_residual_interface_conductivity_vanishes_with_a_dry_neighbour(
        self, richards_model, monkeypatch, k_pair
    ):
        """A zero neighbour gives K_face = 0 (the harmonic-mean limit) and the
        0/0 case K_i = K_j = 0 gives K_face = 0 with finite gradients.

        References: Belfort et al. (2013), J. Hydrol. 505:202-217, Table 2.
        Mathematical expression: lim 2 K_i K_j / (K_i + K_j) = 0 when either
        K -> 0; K_face(0, 0) := 0.
        Parameter meanings: K_i, K_j cell conductivities.
        Units: K [m s-1]; dK_face/dK [1].
        Domain/constraints: K >= 0; gradients must be finite for the exact
        autograd Newton Jacobian.
        Sign convention: K >= 0 is a magnitude.
        """
        face, k_profile = self._top_face_conductivity(
            richards_model, monkeypatch, k_pair, requires_grad=True
        )
        assert float(face.detach()) == 0.0
        (gradient,) = torch.autograd.grad(face, k_profile)
        assert torch.all(torch.isfinite(gradient))

    def test_picard_matrix_uses_the_harmonic_interface_conductivity(
        self, richards_model, monkeypatch
    ):
        """The modified-Picard matrix carries the same K_face as the residual.

        References: Celia, Bouloutas and Zarba (1990), Water Resour. Res.
        26(7):1483-1496; Belfort et al. (2013), J. Hydrol. 505:202-217, Table 2.
        Mathematical expression: c_0 = -dt K_face(0, 1) / dz^2 with
        K_face = 2 K_0 K_1 / (K_0 + K_1).
        Parameter meanings: c_0 upper diagonal of the top row, dt time step,
        dz layer thickness.
        Units: c_0 [m-1]; K [m s-1]; dt [s]; dz [m].
        Domain/constraints: equal neighbours at 1e-12 m s-1, float64 state.
        Sign convention: off-diagonal entries of A_K are nonpositive.
        """
        richards_model.set_boundary_conditions(
            top={"type": "prescribed_flux", "value": 0.0},
            bottom={"type": "free_drainage"},
        )
        n = richards_model.soil_layers
        k_profile = torch.full((n,), 1.0e-12, dtype=torch.float64)
        theta = torch.full((n,), 0.30, dtype=torch.float64)
        captured = {}

        def recording_solve(a, b, c, d):
            captured["c"] = c.detach().clone()
            return SimpleNamespace(solution=torch.zeros_like(d))

        monkeypatch.setattr(richards_model.thomas_solver, "solve", recording_solve)
        richards_model._numerical_solver._solve_linearized_system_picard(
            theta,
            theta,
            richards_model._compute_pressure_head(theta),
            k_profile,
            dt=1.0,
            surface_flux=0.0,
        )
        dz = torch.as_tensor(richards_model.layer_thickness).detach()
        face = -captured["c"][0] * float(dz * dz)
        assert float(face) == pytest.approx(1.0e-12, rel=1.0e-12, abs=0.0)

    def test_infiltration_capacity(self, richards_model):
        """Test infiltration capacity calculation."""
        surface_theta = 0.3
        surface_k = 1e-5
        
        capacity = richards_model._compute_infiltration_capacity(surface_theta, surface_k)
        
        assert capacity > 0, "Infiltration capacity should be positive"
        assert capacity >= surface_k, "Capacity should be at least equal to conductivity"
        
    def test_theta_from_head_conversion(self, richards_model):
        """Test moisture content calculation from pressure head."""
        # Test saturated conditions
        theta_sat = richards_model._compute_theta_from_head(0.0)
        assert abs(theta_sat - richards_model.theta_saturated) < 1e-6
        
        # Test unsaturated conditions
        theta_unsat = richards_model._compute_theta_from_head(-1.0)
        assert richards_model.theta_residual < theta_unsat < richards_model.theta_saturated
        
        # Test dry conditions
        theta_dry = richards_model._compute_theta_from_head(-10.0)
        assert theta_dry > richards_model.theta_residual
        
    def test_solver_robustness(self, richards_model):
        """Test solver robustness with challenging conditions."""
        # Test with very dry initial conditions
        dry_state = {
            "soil_moisture": torch.full((richards_model.soil_layers,), 0.06),
            "infiltration_flux": torch.tensor(5e-6 * 1000.0),
            "evapotranspiration_flux": torch.tensor(0.0),
        }

        result = richards_model._compute_full_solution(dry_state, 0.0, 3600.0)
        assert torch.all(
            torch.isfinite(
                _unwrap_update(
                    result[
                        "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
                    ]
                )
            )
        )

        # Test with wet initial conditions
        wet_state = {
            "soil_moisture": torch.full((richards_model.soil_layers,), 0.42),
            "infiltration_flux": torch.tensor(0.0),
            "evapotranspiration_flux": torch.tensor(2e-6 * 1000.0),
        }

        result = richards_model._compute_full_solution(
            wet_state,
            0.0,
            3600.0,
        )
        assert torch.all(
            torch.isfinite(
                _unwrap_update(
                    result[
                        "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
                    ]
                )
            )
        )
        
    def test_physical_consistency(self, richards_model):
        """Test physical consistency of solutions."""
        state = {
            "soil_moisture": torch.tensor(
                [0.15, 0.20, 0.25, 0.30, 0.35],
                dtype=torch.float32,
            ),
            "infiltration_flux": torch.tensor(
                2e-6 * 1000.0, dtype=torch.float32
            ),
            "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float32),
        }

        result = richards_model._compute_full_solution(state, 0.0, 1800.0)

        # Check that moisture stays within physical bounds
        new_moisture = (
            state["soil_moisture"]
            + _unwrap_update(
                result[
                    "diagnostics.soil_moisture_richards.soil_moisture_change_rate"
                ]
            )
            * 1800.0
        )
        full_state = _unwrap_update(result["soil_moisture"])
        assert full_state.shape == state["soil_moisture"].shape
        assert torch.allclose(full_state, new_moisture, rtol=1e-6, atol=1e-7)
        assert torch.all(new_moisture >= richards_model.theta_residual - 1e-6)
        assert torch.all(new_moisture <= richards_model.theta_saturated + 1e-6)

        # Check that fluxes have reasonable magnitudes
        diagnostics = _get_richards_diagnostics(result)
        fluxes = _unwrap_update(diagnostics["vertical_flux"]) / 1000.0
        assert torch.all(torch.abs(fluxes) < 1e-3), "Fluxes should have reasonable magnitude"


class TestRichardsPhase8:
    """Test Phase 8 IPHS contract."""

    def test_phase8_iphs_protocol(self):
        """Test structural IPHS hooks."""
        functor = RichardsSoilMoisture()

        forms = functor.declared_forms()
        hodge_terms = functor.build_hodge_terms()
        dissipative = functor.dissipative_structure({})
        ports = functor.ports()

        assert isinstance(functor, IPHSDECFunctor)
        assert functor.PHASE8_RUNTIME_REPLACEMENT_STATUS == "canonical_iphs_runtime"
        assert "hydrosphere.soil_moisture.alpha_vg" in functor.required_parameters()
        assert {form.semantic_name for form in forms} == {
            "soil_moisture",
            "pressure_head",
            "water_flux_profile",
        }
        assert len(hodge_terms) == 1
        assert torch.all(torch.linalg.eigvalsh(dissipative["M"]) >= 0.0)
        assert {port.name for port in ports} == {
            "soil_infiltration_boundary",
            "soil_evapotranspiration_boundary",
            "soil_capillary_boundary",
            "richards_drainage_flux",
        }

    def test_phase8_certificate_is_valid(self):
        """Test Phase 8 certificate completeness."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_soil_moisture_richards_richards_soil_moisture.yaml")
        )
        functor = RichardsSoilMoisture()
        audit = certificate.validate()

        assert audit["passed"] is True
        assert certificate.process_id == (
            "relic.scientific.hydrosphere.soil_moisture.richards.RichardsSoilMoisture"
        )
        assert certificate.canonical_role == "transport"
        assert tuple(certificate.input_variables) == tuple(
            functor.metadata.input_variables
        )
        assert tuple(certificate.output_variables) == tuple(
            functor.metadata.output_variables
        )
        assert set(certificate.parameter_keys) == set(functor.required_parameters())
        assert set(certificate.composition_ports) == {
            port.name for port in functor.ports()
        }
        assert certificate.interface_map["forward_signature"] == (
            "forward(self, state, time, dt)"
        )
        assert set(certificate.interface_map["legacy_inputs"]) == set(
            functor.metadata.input_variables
        )
        assert set(certificate.interface_map["iphs_inputs"]) == set(
            functor.metadata.input_variables
        )
        assert set(certificate.interface_map["legacy_outputs"]) == set(
            functor.metadata.output_variables
        )
        assert set(certificate.interface_map["iphs_outputs"]) == set(
            functor.metadata.output_variables
        )
        assert set(certificate.parameter_map) == set(functor.required_parameters())
        assert set(certificate.parameter_management_map) == set(
            functor.required_parameters()
        )

    def test_residual_diagnostics_are_routed(self):
        """Test projection residual diagnostics are exposed."""
        functor = RichardsSoilMoisture()
        result = functor.forward(
            {
                "soil_moisture": torch.full(
                    (functor.soil_layers,), 0.3, dtype=torch.float32
                ),
                "infiltration_flux": torch.tensor(1.0e-6, dtype=torch.float32),
                "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float32),
            },
            time=0.0,
            dt=60.0,
        )

        diagnostics = result["diagnostics.soil_moisture_richards"]
        assert "moisture_clipping_residual" in diagnostics
        assert torch.isfinite(diagnostics["moisture_clipping_residual"]).all()

    def test_certificate_unit_map_names_consumed_or_emitted_variables(self):
        """Every unit_map key is a declared input, a returned output, a sub-key path
        of the returned diagnostics.soil_moisture_richards bundle, or a declared
        form. The Mualem conductivity K(theta) [m s-1] leaves the solver only inside
        the bundle; the bare hydraulic_conductivity contract is retired."""
        certificate = load_iphs_equivalence_certificate(
            _certificate_path("hydrosphere_soil_moisture_richards_richards_soil_moisture.yaml")
        )
        functor = RichardsSoilMoisture()
        result = functor.forward(
            {
                "soil_moisture": torch.full(
                    (functor.soil_layers,), 0.3, dtype=torch.float32
                ),
                "infiltration_flux": torch.tensor(1.0e-6, dtype=torch.float32),
                "evapotranspiration_flux": torch.tensor(0.0, dtype=torch.float32),
            },
            time=0.0,
            dt=60.0,
        )

        bundle = "diagnostics.soil_moisture_richards"
        named = (
            set(functor.metadata.input_variables)
            | set(result)
            | {f"{bundle}.{key}" for key in result[bundle]}
            | {form.semantic_name for form in functor.declared_forms()}
        )
        assert sorted(set(certificate.unit_map) - named) == []
        assert certificate.unit_map[f"{bundle}.hydraulic_conductivity"] == "m s-1"


if __name__ == "__main__":
    # Run basic smoke test
    manager = get_global_manager()
    snapshot = manager.snapshot()
    try:
        manager.set_value("hydrosphere.soil_moisture.layer_count", 3, units="dimensionless")
        manager.set_value("hydrosphere.soil_moisture.soil_depth", 0.5, units="m")
        model = RichardsSoilMoisture("test")
    finally:
        manager.restore(snapshot)
    
    state = {
        "soil_moisture": torch.full((model.soil_layers,), 0.2),
        "infiltration_flux": torch.tensor(1e-6 * 1000.0),
        "evapotranspiration_flux": torch.tensor(0.0),
    }
    
    result = model._compute_full_solution(state, 0.0, 1800.0)
    diagnostics = result["diagnostics.soil_moisture_richards"]
    print("Richards full solution test passed")
    print(f"Solver converged: {diagnostics['solver_converged']}")
    print(f"Mass conservation error: {diagnostics['mass_conservation_error']:.2e}")
    print(f"Iterations: {diagnostics['solver_iterations']}")



def test_richards_transport_label_and_state_dependent_diffusivity():
    """The Richards equation is diffusive, but D(theta) = K dh/dtheta is state dependent and
    computed in the solver, so no scalar diffusivity is read and diffusivity is None."""
    functor = RichardsSoilMoisture()
    assert functor.transport_type == "diffusion"
    assert functor.diffusivity is None
    assert "hydrosphere.transport.soil_moisture_diffusivity" not in functor.required_parameters()
    assert "hydrosphere.transport.soil_moisture_diffusivity" not in functor.metadata.parameters


# References: Balsamo, G. et al. (2008), ECMWF Technical Memorandum 563, Table 1
#   (van Genuchten soil parameters of the six HTESSEL texture classes).
# Mathematical expression: theta(h) = theta_r + (theta_sat - theta_r)
#   / (1 + (alpha |h|)^n)^m and K = K_sat Se^l [1 - (1 - Se^(1/m))^m]^2, m = 1 - 1/n.
# Parameter meanings: alpha inverse length scale of the retention curve, l Mualem
#   pore-connectivity exponent, n shape parameter, K_sat saturated hydraulic
#   conductivity; tuple order (alpha, l, n, K_sat).
# Units: alpha [m-1], l [1], n [1], K_sat [m s-1] (the table prints 1e-6 m s-1).
# Domain/constraints: the six texture classes of the source table; alpha > 0, n > 1.
# Sign convention: not applicable (parameter values; l may be negative).
ECMWF_TM563_TABLE_1_SOIL_CLASSES = {
    "coarse": (3.83, 1.250, 1.38, 6.94e-6),
    "medium": (3.14, -2.342, 1.28, 1.16e-6),
    "medium_fine": (0.83, -0.588, 1.25, 0.26e-6),
    "fine": (3.67, -1.977, 1.10, 2.87e-6),
    "very_fine": (2.65, 2.500, 1.10, 1.74e-6),
    "organic": (1.30, 0.400, 1.20, 0.93e-6),
}
VAN_GENUCHTEN_CLASS_KEYS = (
    ("hydrosphere.soil_moisture.alpha_vg", "m-1"),
    ("hydrosphere.soil_moisture.tortuosity_parameter", "dimensionless"),
    ("hydrosphere.soil_moisture.n_vg", "dimensionless"),
    ("hydrosphere.soil_moisture.saturated_hydraulic_conductivity", "m s-1"),
)


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)


@pytest.mark.parametrize("soil_class", sorted(ECMWF_TM563_TABLE_1_SOIL_CLASSES))
def test_owner_bounds_admit_the_published_van_genuchten_classes(
    soil_class, restore_global_parameter_manager,
):
    """The owner bounds of alpha, the Mualem exponent, n and K_sat admit every texture
    class of the published table in its published unit: a class value is stored as
    requested, not moved to a bound, and the Richards process is built with it."""
    manager = restore_global_parameter_manager
    values = ECMWF_TM563_TABLE_1_SOIL_CLASSES[soil_class]
    for (key, units), value in zip(VAN_GENUCHTEN_CLASS_KEYS, values):
        lower, upper = manager.get_parameter(key).bounds
        assert lower <= value <= upper, (key, value, (lower, upper))
        manager.set_value(key, value, units=units, validate=True, notify=False)
        assert _as_float(manager.get(key)) == pytest.approx(value, rel=1.0e-6)
    functor = RichardsSoilMoisture(name="published_texture_class")
    alpha, exponent, shape, conductivity = values
    assert _as_float(functor.alpha_vg) == pytest.approx(alpha, rel=1.0e-6)
    assert _as_float(functor.mualem_tortuosity) == pytest.approx(exponent, rel=1.0e-6)
    assert _as_float(functor.n_vg) == pytest.approx(shape, rel=1.0e-6)
    assert _as_float(functor.k_saturated) == pytest.approx(conductivity, rel=1.0e-6)


# References: Carsel, R.F. and Parrish, R.S. (1988), Water Resour. Res. 24:755-769,
#   class means as tabulated by van Genuchten, M.Th., Leij, F.J. and Yates, S.R.
#   (1991), The RETC Code, EPA/600/2-91/065, Table 4 (sandy loam).
# Mathematical expression: alpha [m-1] = 100 * alpha [cm-1].
# Parameter meanings: alpha and n of the van Genuchten retention curve of the sandy
#   loam texture class, the class of the owner defaults of alpha and n.
# Units: alpha [cm-1] in the source table and [m-1] in the owner; n [1].
# Domain/constraints: one texture class; the owner unit of alpha is the inverse of
#   the pressure-head unit [m] in h = -(Se^(-1/m) - 1)^(1/n) / alpha.
# Sign convention: not applicable.
SANDY_LOAM_ALPHA_PER_CM = 0.075
SANDY_LOAM_N = 1.89


def test_default_alpha_is_the_sandy_loam_class_value_in_the_owner_unit(
    restore_global_parameter_manager,
):
    """The owner default of alpha is the sandy loam class value in the owner unit m-1,
    and the class value written in its published unit cm-1 is accepted and stored as
    the same number."""
    manager = restore_global_parameter_manager
    key = "hydrosphere.soil_moisture.alpha_vg"
    spec = manager.get_parameter(key)
    expected = 100.0 * SANDY_LOAM_ALPHA_PER_CM
    assert spec.units == "m-1"
    assert _as_float(manager.get("hydrosphere.soil_moisture.n_vg")) == pytest.approx(
        SANDY_LOAM_N, rel=1.0e-6
    )
    assert _as_float(spec.default) == pytest.approx(expected, rel=1.0e-6)
    assert _as_float(manager.get(key)) == pytest.approx(expected, rel=1.0e-6)
    manager.set_value(
        key, SANDY_LOAM_ALPHA_PER_CM, units="cm-1", validate=True, notify=False
    )
    assert _as_float(manager.get(key)) == pytest.approx(expected, rel=1.0e-6)
