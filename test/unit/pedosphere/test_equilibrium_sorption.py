"""L1 tests for the equilibrium sorption DAE Functor (index-1, algebraic elimination).

References:
    - Jury and Horton (2004), Soil Physics, ch. 7 (Freundlich sorption).
    - Hairer and Wanner (1996), Solving ODEs II, ch. VI (index-1 DAEs).
"""

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.transport.equilibrium_sorption import EquilibriumSorptionFunctor

TOTAL = "equilibrium_total_solute_mass"
DISSOLVED = "equilibrium_dissolved_concentration"
SORBED = "equilibrium_sorbed_concentration"
# The volumetric liquid soil water content is the prognostic soil_moisture state [m3 m-3]
# (the former volume_fraction_of_water_in_soil forcing was the same quantity).
THETA = "soil_moisture"
DIAG = "diagnostics.equilibrium_sorption."


def _functor(**overrides):
    config = {"parameters": overrides} if overrides else None
    return EquilibriumSorptionFunctor(name="equilibrium_sorption", config=config)


# Soil dry bulk density of the test column: the bulk_density forcing input [kg m-3].
BULK_DENSITY = 1300.0


def _coefficients(functor):
    kf = float(functor.get_parameter_value("equilibrium_sorption_freundlich_coefficient", required=True).detach())
    n = float(functor.get_parameter_value("equilibrium_sorption_freundlich_exponent", required=True).detach())
    lam = float(functor.get_parameter_value("equilibrium_sorption_dissolved_decay_rate", required=True).detach())
    return kf, n, lam, BULK_DENSITY


def _consistent_state(functor, dissolved, theta):
    kf, n, _, rho = _coefficients(functor)
    c = torch.tensor(dissolved, dtype=torch.float64)
    s = kf * c**n
    y = theta * c + rho * s
    return {
        TOTAL: y,
        DISSOLVED: c,
        SORBED: s,
        THETA: torch.full_like(c, theta),
        "bulk_density": torch.full_like(c, BULK_DENSITY),
    }


def _apply(functor, state, dt, time=0.0):
    out = functor.forward(state, time, dt)
    new = dict(state)
    new[TOTAL] = state[TOTAL] + out[TOTAL].value
    new[DISSOLVED] = out[DISSOLVED].value
    new[SORBED] = out[SORBED].value
    return new, out


class TestContract:
    def test_metadata_and_constraint_declaration(self):
        functor = _functor()
        assert functor.metadata.input_variables == [TOTAL, DISSOLVED, SORBED, THETA, "bulk_density"]
        assert functor.metadata.output_variables[:3] == [TOTAL, DISSOLVED, SORBED]
        assert all(name.startswith(DIAG) for name in functor.metadata.output_variables[3:])
        assert functor.constraint_category == "descriptor_dae"
        assert functor.STATE_VARIABLES == (TOTAL, DISSOLVED, SORBED)
        assert functor.ALGEBRAIC_VARIABLES == (DISSOLVED, SORBED)
        assert functor.get_dependencies() == []
        assert "bulk_density" in functor.get_required_inputs()


class TestIndexOneStep:
    def test_linear_isotherm_matches_the_closed_form_implicit_step(self):
        # References: Jury and Horton (2004); Hairer and Wanner (1996).
        # Mathematical expression: n = 1 gives y' = y / (1 + dt lambda theta / R),
        #   R = theta + rho_b K_f, with C' = y' / R and S' = K_f C'.
        # Parameter meanings: R the retardation denominator, lambda decay rate.
        # Units: y, C kg m-3; S kg kg-1; theta 1; rho_b kg m-3; K_f m3 kg-1.
        # Domain/constraints: consistent initial triple; finite positive dt.
        # Sign convention: degradation removes mass from y and C.
        functor = _functor(equilibrium_sorption_freundlich_exponent=1.0)
        kf, n, lam, rho = _coefficients(functor)
        theta = 0.3
        state = _consistent_state(functor, [2.0, 0.5], theta)
        dt = 3600.0
        new, out = _apply(functor, state, dt)
        retardation = theta + rho * kf
        expected_y = state[TOTAL] / (1.0 + dt * lam * theta / retardation)
        torch.testing.assert_close(new[TOTAL], expected_y, rtol=1e-9, atol=0.0)
        torch.testing.assert_close(new[DISSOLVED], expected_y / retardation, rtol=1e-9, atol=0.0)
        torch.testing.assert_close(new[SORBED], kf * expected_y / retardation, rtol=1e-9, atol=0.0)
        assert out[TOTAL].mode is UpdateMode.ACCUMULATE and out[TOTAL].unit == "kg m-3"
        assert out[DISSOLVED].mode is UpdateMode.REPLACE and out[DISSOLVED].unit == "kg m-3"
        assert out[SORBED].mode is UpdateMode.REPLACE and out[SORBED].unit == "kg kg-1"

    def test_nonlinear_isotherm_step_satisfies_the_constraint_and_the_mass_ledger(self):
        functor = _functor()
        kf, n, lam, rho = _coefficients(functor)
        theta = 0.25
        state = _consistent_state(functor, [3.0, 1.0, 0.1], theta)
        dt = 7200.0
        new, out = _apply(functor, state, dt)
        constraint = theta * new[DISSOLVED] + rho * new[SORBED] - new[TOTAL]
        assert float(constraint.abs().max()) < 1e-9 * float(new[TOTAL].max())
        torch.testing.assert_close(new[SORBED], kf * new[DISSOLVED] ** n, rtol=1e-12, atol=0.0)
        # Degraded mass equals dt lambda theta C' exactly (implicit dissolved decay).
        torch.testing.assert_close(state[TOTAL] - new[TOTAL], dt * lam * theta * new[DISSOLVED], rtol=1e-12, atol=0.0)
        torch.testing.assert_close(out[DIAG + "degradation_flux"].value, lam * theta * new[DISSOLVED])
        assert bool((new[DISSOLVED] < state[DISSOLVED]).all())
        assert float(out[DIAG + "constraint_residual"].value.max()) < 1e-9 * float(new[TOTAL].max())
        assert int(out[DIAG + "solver_iterations"].value.max()) >= 1

    def test_inconsistent_initial_values_are_rejected_not_tolerated(self):
        functor = _functor()
        state = _consistent_state(functor, [2.0], 0.3)
        state[SORBED] = state[SORBED] * 1.5
        with pytest.raises(ValueError, match="consistent"):
            functor.forward(state, 0.0, 3600.0)
        state = _consistent_state(functor, [2.0], 0.3)
        state[TOTAL] = state[TOTAL] + 0.1
        with pytest.raises(ValueError, match="consistent"):
            functor.forward(state, 0.0, 3600.0)

    def test_zero_mass_stays_zero_without_solving(self):
        functor = _functor()
        state = _consistent_state(functor, [0.0], 0.3)
        new, out = _apply(functor, state, 3600.0)
        assert float(new[TOTAL]) == 0.0 and float(new[DISSOLVED]) == 0.0 and float(new[SORBED]) == 0.0
        assert int(out[DIAG + "solver_iterations"].value) == 0

    def test_solver_failure_is_reported_not_hidden(self, monkeypatch):
        functor = _functor()
        state = _consistent_state(functor, [2.0], 0.3)
        solver = functor._root_solver(state[TOTAL])
        monkeypatch.setattr(solver, "max_iterations", 1)
        with pytest.raises(RuntimeError, match="did not converge"):
            functor.forward(state, 0.0, 3600.0)

    def test_gradient_flows_through_the_implicit_solve(self):
        kf = torch.tensor(2.0e-3, dtype=torch.float64, requires_grad=True)
        functor = _functor(equilibrium_sorption_freundlich_coefficient=kf)
        state = _consistent_state(functor, [2.0], 0.3)
        out = functor.forward(state, 0.0, 3600.0)
        (grad,) = torch.autograd.grad(out[DISSOLVED].value.sum(), (kf,))
        # Stronger sorption lowers the dissolved concentration.
        assert torch.isfinite(grad) and float(grad) < 0.0

    @pytest.mark.parametrize(
        "case", ["missing_theta", "negative_total", "nonfinite", "bad_theta", "bad_dt", "shape"]
    )
    def test_domain_rejections(self, case):
        functor = _functor()
        state = _consistent_state(functor, [2.0, 1.0], 0.3)
        dt = 3600.0
        if case == "missing_theta":
            state.pop(THETA)
        elif case == "negative_total":
            state[TOTAL] = -state[TOTAL]
        elif case == "nonfinite":
            state[DISSOLVED] = torch.tensor([float("nan"), 1.0], dtype=torch.float64)
        elif case == "bad_theta":
            state[THETA] = torch.tensor([0.0, 0.3], dtype=torch.float64)
        elif case == "bad_dt":
            dt = -1.0
        elif case == "shape":
            state[SORBED] = state[SORBED][:1]
        with pytest.raises((ValueError, TypeError)):
            functor.forward(state, 0.0, dt)


class TestGovernedRuntime:
    @staticmethod
    def _simulator():
        from relic.core.simulator import RELICSimulator

        functor = _functor()
        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False, "enable_auto_scale": False})
        simulator.initialize_state(_consistent_state(functor, [2.0, 0.5], 0.3))
        simulator.register_functor(functor, execution_group="fast")
        return simulator, functor

    def test_governed_steps_keep_the_algebraic_states_consistent(self):
        simulator, functor = self._simulator()
        kf, n, lam, rho = _coefficients(functor)
        for _ in range(5):
            simulator.step(dt=3600.0)
        y = simulator.state_manager.get_state(TOTAL)
        c = simulator.state_manager.get_state(DISSOLVED)
        s = simulator.state_manager.get_state(SORBED)
        assert float((0.3 * c + rho * s - y).abs().max()) < 1e-9 * float(y.max())
        torch.testing.assert_close(s, kf * c**n, rtol=1e-12, atol=0.0)
        assert simulator.step_count == 5

    def test_checkpoint_restart_restores_a_consistent_triple_and_continues(self, tmp_path):
        uninterrupted, _ = self._simulator()
        for _ in range(6):
            uninterrupted.step(dt=3600.0)
        source, _ = self._simulator()
        for _ in range(3):
            source.step(dt=3600.0)
        checkpoint = tmp_path / "sorption.pt"
        source.create_checkpoint(checkpoint)
        restored, _ = self._simulator()
        restored.restore_checkpoint(checkpoint)
        for _ in range(3):
            restored.step(dt=3600.0)
        for name in (TOTAL, DISSOLVED, SORBED):
            torch.testing.assert_close(
                restored.state_manager.get_state(name),
                uninterrupted.state_manager.get_state(name),
                rtol=0.0,
                atol=0.0,
            )

# Round 35: the bulk density is the forcing input bulk_density of its contract
# (soil_sorption_geometry_input), not the parameter bulk_density_default.
def test_bulk_density_is_a_required_input_not_a_default():
    functor = _functor()
    assert "bulk_density" in functor.get_required_inputs()
    assert "pedosphere.physical.bulk_density_default" not in functor.metadata.parameters
    state = _consistent_state(functor, [0.2], 0.3)
    del state["bulk_density"]
    with pytest.raises(ValueError, match="bulk_density"):
        functor.forward(state, 0.0, 3600.0)


def test_config_first_source_order_is_the_mixin_class_declaration():
    """Round 47: explicit config overrides precede the ParameterManager through the
    NamespaceDefaultMixin class declaration, not through a get_parameter_value override."""
    from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin

    for functor_class in (EquilibriumSorptionFunctor,):
        assert issubclass(functor_class, NamespaceDefaultMixin)
        assert functor_class.parameter_lookup_source_order == ("config", "manager")
        assert "get_parameter_value" not in functor_class.__dict__
