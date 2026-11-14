"""L1 tests for the two-region (mobile-immobile) memory exchange Functor.

References:
    - van Genuchten and Wierenga (1976), Soil Sci. Soc. Am. J. 40, 473-480.
    - Haggerty and Gorelick (1995), Water Resour. Res. 31, 2383-2400.
    - Revised RELIC foundations, memory/Volterra semantics (exponential-kernel
      passive realization).
"""

import math

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.transport.two_region_mass_transfer import (
    TwoRegionMassTransferFunctor,
)

MOBILE = "mobile_solute_concentration"
IMMOBILE = "immobile_solute_concentration"
DIAG = "diagnostics.two_region_exchange."


def _functor(**overrides):
    config = {"parameters": overrides} if overrides else None
    return TwoRegionMassTransferFunctor(name="two_region_mass_transfer", config=config)


def _coefficients(functor):
    alpha = float(functor.get_parameter_value("two_region_mass_transfer_coefficient", required=True).detach())
    theta_m = float(functor.get_parameter_value("two_region_mobile_water_content", required=True).detach())
    theta_im = float(functor.get_parameter_value("two_region_immobile_water_content", required=True).detach())
    return alpha, theta_m, theta_im


def _state(mobile, immobile):
    return {
        MOBILE: torch.tensor(mobile, dtype=torch.float64),
        IMMOBILE: torch.tensor(immobile, dtype=torch.float64),
    }


class TestContract:
    def test_metadata_and_memory_declaration(self):
        functor = _functor()
        assert functor.metadata.input_variables == [MOBILE, IMMOBILE]
        assert functor.metadata.output_variables[:2] == [MOBILE, IMMOBILE]
        assert all(name.startswith(DIAG) for name in functor.metadata.output_variables[2:])
        assert functor.get_dependencies() == []
        assert functor.memory_category == "memory"
        assert functor.transport_type == "interphase_mass_transfer"
        assert functor.STATE_VARIABLES == (MOBILE, IMMOBILE)
        assert functor.HIDDEN_STATE_VARIABLE == IMMOBILE
        assert functor.EFFORT_STATE_VARIABLE == MOBILE
        assert "mass" in functor.metadata.conservation_laws

    def test_declared_iphs_structure_hooks(self):
        functor = _functor()
        alpha, _, _ = _coefficients(functor)
        state = _state([1.0], [0.0])
        structure = functor.dissipative_structure(state)
        assert structure["state_vars"] == (MOBILE, IMMOBILE)
        torch.testing.assert_close(
            structure["R"], alpha * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
        )
        assert torch.count_nonzero(structure["J"]) == 0
        eigenvalues = torch.linalg.eigvalsh(structure["R"])
        assert float(eigenvalues.min()) >= -1e-15 and float(eigenvalues.max()) > 0.0
        assert functor.reversible_structure(state) is None
        assert "storage" in functor.energy_terms(state)
        assert "exchange_dissipation" in functor.entropy_terms(state)

    def test_unknown_override_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown two-region parameter override"):
            _functor(unknown=1.0)


class TestExactExchange:
    def test_step_matches_the_closed_form_two_region_solution(self):
        # References: van Genuchten and Wierenga (1976).
        # Mathematical expression: m = theta_m c_m + theta_im c_im conserved;
        #   d = c_m - c_im decays as exp(-kappa dt), kappa = alpha (theta_m + theta_im)/(theta_m theta_im).
        # Parameter meanings: alpha mass transfer coefficient, theta water contents.
        # Units: c kg m-3; theta m3 m-3; alpha s-1; dt s.
        # Domain/constraints: nonnegative concentrations; finite positive dt.
        # Sign convention: mass moves from the higher to the lower concentration.
        functor = _functor()
        alpha, theta_m, theta_im = _coefficients(functor)
        dt = 3600.0
        cm0, cim0 = 2.0, 0.5
        result = functor.forward(_state([cm0], [cim0]), 0.0, dt)
        for name in (MOBILE, IMMOBILE):
            item = result[name]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.unit == "kg m-3"
        kappa = alpha * (theta_m + theta_im) / (theta_m * theta_im)
        total = theta_m * cm0 + theta_im * cim0
        d1 = (cm0 - cim0) * math.exp(-kappa * dt)
        cm1 = (total + theta_im * d1) / (theta_m + theta_im)
        cim1 = (total - theta_m * d1) / (theta_m + theta_im)
        assert float(result[MOBILE].value) == pytest.approx(cm1 - cm0, rel=1e-12)
        assert float(result[IMMOBILE].value) == pytest.approx(cim1 - cim0, rel=1e-12)
        # Closed ledger: theta_m Delta c_m + theta_im Delta c_im = 0.
        assert abs(float(result[DIAG + "memory_residual"].value)) < 1e-12
        assert float(result[DIAG + "mass_transfer_rate"].value) == pytest.approx(alpha * (cm0 - cim0))
        assert float(result[DIAG + "storage_dissipation"].value) == pytest.approx(-alpha * (cm0 - cim0) ** 2)

    def test_two_steps_equal_one_double_step(self):
        # The exact exponential update is a semigroup: two steps of dt equal one step of 2 dt.
        functor = _functor()
        state = _state([1.0, 3.0], [0.0, 1.0])
        first = functor.forward(state, 0.0, 1800.0)
        mid = {name: state[name] + first[name].value for name in (MOBILE, IMMOBILE)}
        second = functor.forward(mid, 1800.0, 1800.0)
        double = functor.forward(state, 0.0, 3600.0)
        for name in (MOBILE, IMMOBILE):
            torch.testing.assert_close(
                state[name] + first[name].value + second[name].value,
                state[name] + double[name].value,
                rtol=1e-12,
                atol=0.0,
            )

    def test_extended_storage_is_nonincreasing_and_memory_power_can_return_energy(self):
        # H = 0.5 (theta_m c_m^2 + theta_im c_im^2) never increases; the
        # instantaneous power delivered to the mobile state may be positive
        # when the immobile region has stored more (legitimate energy return).
        functor = _functor()
        alpha, theta_m, theta_im = _coefficients(functor)
        state = _state([0.5], [2.0])
        result = functor.forward(state, 0.0, 7200.0)
        cm1 = float(state[MOBILE] + result[MOBILE].value)
        cim1 = float(state[IMMOBILE] + result[IMMOBILE].value)
        storage0 = 0.5 * (theta_m * 0.5**2 + theta_im * 2.0**2)
        storage1 = 0.5 * (theta_m * cm1**2 + theta_im * cim1**2)
        assert storage1 < storage0
        # Mobile state receives mass (positive increment) from the immobile memory.
        assert float(result[MOBILE].value) > 0.0
        assert float(result[DIAG + "mass_transfer_rate"].value) < 0.0

    def test_volterra_kernel_realization_matches_the_hidden_state(self):
        # Eliminating c_im gives the exponential memory kernel
        # c_im(t) = e^{-omega t} c_im(0) + int_0^t omega e^{-omega (t-s)} c_m(s) ds,
        # omega = alpha / theta_im. Reconstruct c_im from the mobile history
        # with a fine trapezoidal convolution and compare with the hidden state.
        functor = _functor()
        alpha, theta_m, theta_im = _coefficients(functor)
        omega = alpha / theta_im
        dt = 60.0
        n = 400
        state = _state([1.0], [0.0])
        mobile_history = [1.0]
        for step in range(n):
            out = functor.forward(state, step * dt, dt)
            state = {name: state[name] + out[name].value for name in (MOBILE, IMMOBILE)}
            mobile_history.append(float(state[MOBILE]))
        t = n * dt
        # Trapezoidal convolution of the recorded mobile history.
        integral = 0.0
        for k in range(n):
            s0, s1 = k * dt, (k + 1) * dt
            f0 = omega * math.exp(-omega * (t - s0)) * mobile_history[k]
            f1 = omega * math.exp(-omega * (t - s1)) * mobile_history[k + 1]
            integral += 0.5 * (f0 + f1) * dt
        assert float(state[IMMOBILE]) == pytest.approx(integral, rel=2e-4)

    def test_gradients_flow_to_the_mass_transfer_coefficient(self):
        alpha = torch.tensor(2.0e-5, dtype=torch.float64, requires_grad=True)
        functor = _functor(two_region_mass_transfer_coefficient=alpha)
        result = functor.forward(_state([2.0], [0.5]), 0.0, 3600.0)
        (grad,) = torch.autograd.grad(result[MOBILE].value.sum(), (alpha,))
        assert torch.isfinite(grad) and float(grad) < 0.0

    @pytest.mark.parametrize(
        "case", ["missing_mobile", "missing_immobile", "negative", "nonfinite", "bad_dt", "non_tensor", "shape"]
    )
    def test_domain_rejections(self, case):
        functor = _functor()
        state = _state([1.0, 1.0], [0.5, 0.5])
        dt = 3600.0
        if case == "missing_mobile":
            state.pop(MOBILE)
        elif case == "missing_immobile":
            state.pop(IMMOBILE)
        elif case == "negative":
            state[MOBILE] = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        elif case == "nonfinite":
            state[IMMOBILE] = torch.tensor([float("nan"), 0.5], dtype=torch.float64)
        elif case == "bad_dt":
            dt = -1.0
        elif case == "non_tensor":
            state[MOBILE] = 1.0
        elif case == "shape":
            state[IMMOBILE] = torch.tensor([0.5], dtype=torch.float64)
        with pytest.raises((ValueError, TypeError)):
            functor.forward(state, 0.0, dt)


class TestGovernedRuntime:
    @staticmethod
    def _simulator():
        from relic.core.simulator import RELICSimulator

        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False, "enable_auto_scale": False})
        simulator.initialize_state(_state([2.0, 1.0], [0.0, 0.0]))
        functor = _functor()
        simulator.register_functor(functor, execution_group="fast")
        return simulator, functor

    def test_governed_steps_conserve_the_two_region_mass_and_track_the_closed_form(self):
        simulator, functor = self._simulator()
        alpha, theta_m, theta_im = _coefficients(functor)
        dt = 1800.0
        for _ in range(6):
            simulator.step(dt=dt)
        cm = simulator.state_manager.get_state(MOBILE)
        cim = simulator.state_manager.get_state(IMMOBILE)
        total = theta_m * cm + theta_im * cim
        torch.testing.assert_close(total, torch.tensor([theta_m * 2.0, theta_m * 1.0], dtype=torch.float64), rtol=1e-12, atol=0.0)
        kappa = alpha * (theta_m + theta_im) / (theta_m * theta_im)
        expected_difference = torch.tensor([2.0, 1.0], dtype=torch.float64) * math.exp(-kappa * 6 * dt)
        torch.testing.assert_close(cm - cim, expected_difference, rtol=1e-10, atol=0.0)
        assert simulator.step_count == 6

    def test_checkpoint_restart_continues_the_memory_state(self, tmp_path):
        dt = 1800.0
        uninterrupted, _ = self._simulator()
        for _ in range(6):
            uninterrupted.step(dt=dt)
        source, _ = self._simulator()
        for _ in range(3):
            source.step(dt=dt)
        checkpoint = tmp_path / "two_region.pt"
        source.create_checkpoint(checkpoint)
        restored, _ = self._simulator()
        restored.restore_checkpoint(checkpoint)
        for _ in range(3):
            restored.step(dt=dt)
        for name in (MOBILE, IMMOBILE):
            torch.testing.assert_close(
                restored.state_manager.get_state(name),
                uninterrupted.state_manager.get_state(name),
                rtol=0.0,
                atol=0.0,
            )


def test_config_first_source_order_is_the_mixin_class_declaration():
    """Round 47: explicit config overrides precede the ParameterManager through the
    NamespaceDefaultMixin class declaration, not through a get_parameter_value override."""
    from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin

    for functor_class in (TwoRegionMassTransferFunctor,):
        assert issubclass(functor_class, NamespaceDefaultMixin)
        assert functor_class.parameter_lookup_source_order == ("config", "manager")
        assert "get_parameter_value" not in functor_class.__dict__
