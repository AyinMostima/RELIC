"""L1 tests for the moving-domain (ALE) column diffusion Functor.

References:
    - Thomas and Lombard (1979), AIAA J. 17, 1030-1037 (GCL).
    - Donea et al. (2004), ALE methods, Encyclopedia of Computational Mechanics.
    - Crank (1975), The Mathematics of Diffusion.
"""

import pytest
import torch

from relic.computation.operators.functional import _compute_diffusion_rate_and_boundary_flux_1d
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.transport.moving_domain_diffusion import (
    MovingDomainDiffusionFunctor,
)

CONC = "moving_domain_solute_concentration"
LENGTH = "moving_domain_length"
VELOCITY = "moving_domain_boundary_velocity"
BOUNDARY_CONC = "moving_boundary_solute_concentration"
DIAG = "diagnostics.moving_domain."


def _functor(n_cells=None, **overrides):
    # The cell count is defined by the concentration state; n_cells is
    # accepted by the test helper only for readability.
    config = {"parameters": overrides} if overrides else None
    return MovingDomainDiffusionFunctor(name="moving_domain_diffusion", config=config)


def _state(conc, length, velocity, boundary_conc):
    return {
        CONC: torch.tensor(conc, dtype=torch.float64),
        LENGTH: torch.tensor(length, dtype=torch.float64),
        VELOCITY: torch.tensor(velocity, dtype=torch.float64),
        BOUNDARY_CONC: torch.tensor(boundary_conc, dtype=torch.float64),
    }


def _apply(functor, state, dt, time=0.0):
    out = functor.forward(state, time, dt)
    new = dict(state)
    new[CONC] = state[CONC] + out[CONC].value
    new[LENGTH] = state[LENGTH] + out[LENGTH].value
    return new, out


class TestContract:
    def test_metadata_and_geometry_declaration(self):
        functor = _functor(n_cells=4)
        assert functor.metadata.input_variables == [CONC, LENGTH, VELOCITY, BOUNDARY_CONC]
        assert functor.metadata.output_variables[:2] == [CONC, LENGTH]
        assert all(name.startswith(DIAG) for name in functor.metadata.output_variables[2:])
        assert functor.geometry_category == "ale"
        assert functor.STATE_VARIABLES == (CONC, LENGTH)
        assert functor.get_dependencies() == []

    @pytest.mark.parametrize("conc", [[1.0], [[1.0, 1.0], [1.0, 1.0]]])
    def test_columns_need_at_least_two_cells_in_one_dimension(self, conc):
        functor = _functor()
        with pytest.raises(ValueError, match="at least two cells"):
            functor.forward(_state(conc, 1.0, 0.0, 0.0), 0.0, 10.0)


class TestGeometricConservation:
    def test_constant_field_is_preserved_under_pure_stretching(self):
        # References: Thomas and Lombard (1979): discrete GCL.
        # Mathematical expression: V_i(t+dt) = V_i(t) + dt (w_{i+1/2} - w_{i-1/2});
        #   with F = 0 and c_face = c the constant field is exact.
        # Parameter meanings: w_b boundary velocity, N cells, L length.
        # Units: L m; w_b m s-1; c kg m-3.
        # Domain/constraints: mesh Courant |w_b| dt / dx <= courant_max.
        # Sign convention: w_b > 0 grows the column.
        functor = _functor(n_cells=6)
        state = _state([3.0] * 6, 1.0, 1.0e-3, 3.0)
        for step in range(5):
            state, out = _apply(functor, state, 60.0, time=step * 60.0)
            torch.testing.assert_close(state[CONC], torch.full((6,), 3.0, dtype=torch.float64), rtol=0.0, atol=1e-13)
            assert abs(float(out[DIAG + "gcl_residual"].value)) < 1e-15
            assert abs(float(out[DIAG + "mass_balance_residual"].value)) < 1e-12
        assert float(state[LENGTH]) == pytest.approx(1.0 + 5 * 60.0 * 1.0e-3)

    def test_growth_with_clean_boundary_dilutes_exactly(self):
        # Pure stretching with zero-concentration entrainment and no diffusion
        # gradient conserves the column mass: c(t) = c0 L0 / L(t).
        functor = _functor(n_cells=5)
        state = _state([2.0] * 5, 0.5, 2.0e-3, 0.0)
        mass0 = float(state[CONC].sum()) * float(state[LENGTH]) / 5
        for step in range(4):
            state, out = _apply(functor, state, 30.0, time=step * 30.0)
            assert abs(float(out[DIAG + "mass_balance_residual"].value)) < 1e-12
        length = float(state[LENGTH])
        mass = float(state[CONC].sum()) * length / 5
        assert mass == pytest.approx(mass0, rel=1e-12)
        # The column mean dilutes exactly as c0 L0 / L; the clean fluid enters
        # at the moving bottom face and is advected upward relative to the
        # stretching mesh, so the profile is monotone with the fixed-top cell
        # still at c0 and the bottom cell diluted.
        assert float(state[CONC].mean()) == pytest.approx(2.0 * 0.5 / length, rel=1e-12)
        assert float(state[CONC][0]) == pytest.approx(2.0)
        assert bool((state[CONC][1:] <= state[CONC][:-1] + 1e-15).all())
        assert float(state[CONC][-1]) < 2.0

    def test_boundary_entrainment_is_the_only_mass_source(self):
        functor = _functor(n_cells=4)
        state = _state([1.0, 2.0, 3.0, 4.0], 2.0, 5.0e-3, 7.0)
        length0 = float(state[LENGTH])
        mass0 = float(state[CONC].sum()) * length0 / 4
        state, out = _apply(functor, state, 80.0)
        mass1 = float(state[CONC].sum()) * float(state[LENGTH]) / 4
        swept = float(out[DIAG + "boundary_mass_flux"].value) * 80.0
        assert float(out[DIAG + "boundary_mass_flux"].value) == pytest.approx(7.0 * 5.0e-3)
        assert mass1 - mass0 == pytest.approx(swept, rel=1e-12)

    def test_shrinking_column_releases_interior_concentration(self):
        functor = _functor(n_cells=4)
        state = _state([1.0, 2.0, 3.0, 4.0], 2.0, -5.0e-3, 100.0)
        mass0 = float(state[CONC].sum()) * 2.0 / 4
        state, out = _apply(functor, state, 80.0)
        # Released fluid carries the bottom-cell concentration, not the outside value.
        assert float(out[DIAG + "boundary_mass_flux"].value) == pytest.approx(4.0 * -5.0e-3)
        mass1 = float(state[CONC].sum()) * float(state[LENGTH]) / 4
        assert mass1 - mass0 == pytest.approx(-4.0 * 5.0e-3 * 80.0, rel=1e-12)
        assert float(state[LENGTH]) == pytest.approx(1.6)

    def test_fixed_domain_reduces_to_the_canonical_finite_volume_diffusion(self):
        functor = _functor(n_cells=6, moving_domain_diffusivity=1.0e-6)
        conc = torch.tensor([0.0, 0.0, 5.0, 5.0, 0.0, 0.0], dtype=torch.float64)
        state = _state(conc.tolist(), 0.6, 0.0, 0.0)
        dt = 100.0
        _, out = _apply(functor, state, dt)
        rate, _ = _compute_diffusion_rate_and_boundary_flux_1d(
            conc, torch.full_like(conc, 1.0e-6), 0.1, "zero"
        )
        torch.testing.assert_close(out[CONC].value, rate * dt, rtol=1e-12, atol=0.0)
        assert float(out[LENGTH].value) == 0.0
        assert float(out[DIAG + "geometry_power"].value) == 0.0

    def test_geometry_power_matches_the_hodge_time_derivative(self):
        functor = _functor(n_cells=4)
        state = _state([1.0, 2.0, 3.0, 4.0], 2.0, 5.0e-3, 0.0)
        _, out = _apply(functor, state, 10.0)
        # -0.5 c^T dot(star) c with dot(star) = diag(w_b / N).
        expected = -0.5 * float((state[CONC] ** 2).sum()) * 5.0e-3 / 4
        assert float(out[DIAG + "geometry_power"].value) == pytest.approx(expected)

    def test_update_contract_and_units(self):
        functor = _functor(n_cells=3)
        _, out = _apply(functor, _state([1.0, 1.0, 1.0], 1.0, 1.0e-3, 1.0), 10.0)
        assert out[CONC].mode is UpdateMode.ACCUMULATE and out[CONC].unit == "kg m-3"
        assert out[LENGTH].mode is UpdateMode.ACCUMULATE and out[LENGTH].unit == "m"
        assert isinstance(out[DIAG + "gcl_residual"], UpdateItem)
        assert out[DIAG + "boundary_mass_flux"].unit == "kg m-2 s-1"
        assert out[DIAG + "mass_balance_residual"].unit == "kg m-2"
        assert out[DIAG + "gcl_residual"].unit == "m"
        assert out[DIAG + "geometry_power"].unit == "kg2 m-5 s-1"

    def test_gradient_flows_to_the_diffusivity(self):
        diffusivity = torch.tensor(1.0e-6, dtype=torch.float64, requires_grad=True)
        functor = _functor(n_cells=4, moving_domain_diffusivity=diffusivity)
        _, out = _apply(functor, _state([0.0, 4.0, 0.0, 0.0], 0.4, 1.0e-3, 0.0), 50.0)
        (grad,) = torch.autograd.grad(out[CONC].value[0], (diffusivity,))
        assert torch.isfinite(grad) and float(grad) > 0.0

    @pytest.mark.parametrize(
        "case",
        ["mesh_courant", "diffusion_cfl", "nonpositive_length", "negative_conc", "missing_velocity", "bad_dt"],
    )
    def test_domain_rejections(self, case):
        functor = _functor(n_cells=4, moving_domain_diffusivity=1.0e-6)
        state = _state([1.0, 1.0, 1.0, 1.0], 0.4, 1.0e-3, 1.0)
        dt = 10.0
        if case == "mesh_courant":
            state[VELOCITY] = torch.tensor(0.05, dtype=torch.float64)  # |w| dt / dx = 5 > courant_max
        elif case == "diffusion_cfl":
            dt = 1.0e4  # D dt / dx^2 = 1 > cfl_max
        elif case == "nonpositive_length":
            state[LENGTH] = torch.tensor(0.0, dtype=torch.float64)
        elif case == "negative_conc":
            state[CONC] = torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        elif case == "missing_velocity":
            state.pop(VELOCITY)
        elif case == "bad_dt":
            dt = 0.0
        with pytest.raises((ValueError, TypeError)):
            functor.forward(state, 0.0, dt)


class TestGovernedRuntime:
    @staticmethod
    def _simulator(velocity):
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False, "enable_auto_scale": False})
        simulator.initialize_state(_state([1.0, 2.0, 3.0, 4.0], 2.0, velocity, 0.5))
        functor = _functor(n_cells=4)
        simulator.register_functor(functor, execution_group="fast")
        return simulator, functor

    def test_governed_steps_keep_the_column_ledger_closed(self):
        simulator, _ = self._simulator(5.0e-3)
        mass0 = float(simulator.state_manager.get_state(CONC).sum()) * 2.0 / 4
        swept = 0.0
        for _ in range(6):
            outputs = simulator.step(dt=50.0)
            swept += float(outputs[DIAG + "boundary_mass_flux"]) * 50.0
        length = float(simulator.state_manager.get_state(LENGTH))
        assert length == pytest.approx(2.0 + 6 * 50.0 * 5.0e-3)
        mass = float(simulator.state_manager.get_state(CONC).sum()) * length / 4
        assert mass - mass0 == pytest.approx(swept, rel=1e-10)
        assert simulator.step_count == 6

    def test_checkpoint_restart_continues_the_geometry(self, tmp_path):
        uninterrupted, _ = self._simulator(5.0e-3)
        for _ in range(6):
            uninterrupted.step(dt=50.0)
        source, _ = self._simulator(5.0e-3)
        for _ in range(3):
            source.step(dt=50.0)
        checkpoint = tmp_path / "moving.pt"
        source.create_checkpoint(checkpoint)
        restored, _ = self._simulator(5.0e-3)
        restored.restore_checkpoint(checkpoint)
        for _ in range(3):
            restored.step(dt=50.0)
        for name in (CONC, LENGTH):
            torch.testing.assert_close(
                restored.state_manager.get_state(name),
                uninterrupted.state_manager.get_state(name),
                rtol=0.0,
                atol=0.0,
            )

# Round 35: the diffusivity attribute is the live ParameterManager object that forward
# reads, so a calibration update moves both together (no float copy).
def test_diffusivity_attribute_is_the_live_parameter(restore_global_parameter_manager):
    from relic.parameters import get_global_manager

    functor = _functor()
    live = functor.get_parameter_value("moving_domain_diffusivity", required=True)
    assert functor.diffusivity is live
    manager = get_global_manager()
    manager.set_value(
        "pedosphere.transport.moving_domain_diffusivity", 2.0e-9, units="m2 s-1",
        validate=True, notify=False,
    )
    assert float(functor.diffusivity.detach()) == pytest.approx(2.0e-9)
    assert float(
        functor.get_parameter_value("moving_domain_diffusivity", required=True).detach()
    ) == pytest.approx(2.0e-9)


def test_config_first_source_order_is_the_mixin_class_declaration():
    """Round 47: explicit config overrides precede the ParameterManager through the
    NamespaceDefaultMixin class declaration, not through a get_parameter_value override."""
    from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin

    for functor_class in (MovingDomainDiffusionFunctor,):
        assert issubclass(functor_class, NamespaceDefaultMixin)
        assert functor_class.parameter_lookup_source_order == ("config", "manager")
        assert "get_parameter_value" not in functor_class.__dict__
