"""G12-EXERGY: relative storage Lyapunov contract on an affine ledger leaf.

References: BIHSS theorem "exergy Lyapunov stability" and its counterexample;
Willems (1972); Khalil (2002) Theorem 4.1; Crank (1975) for the Fickian
consumer.
Mathematical expression: V = F(x) - F(x*) - grad F(x*)^T (x - x*);
dot V = (grad F(x) - grad F(x*))^T x_dot; premises grad F(x*) = C^T lambda and
grad^2 F(x*) > 0 on ker C.
Parameter meanings: F storage, C ledger rows, x* candidate equilibrium.
Units: F in energy-like units; the Fickian storage is 0.5 c^T c.
Domain/constraints: affine leaves only; curved constraints are refused by
scope; the counterexample must be refused, not accepted.
Sign convention: dot V = -D <= 0 along ledger-preserving trajectories when
the premises hold.
"""

from __future__ import annotations

import pytest
import torch

from relic.computation.operators.relative_storage import (
    certify_relative_storage,
    feedback_power_ledger,
    relative_storage,
    relative_storage_rate,
)
from relic.scientific.transport.diffusion import FickianDiffusion

F64 = torch.float64


class TestOracles:
    def test_quadratic_storage_on_a_mass_leaf_is_a_lyapunov_candidate(self):
        k = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=F64))

        def storage(x):
            return 0.5 * x @ k @ x

        ledger = torch.ones(1, 3, dtype=F64)
        # The constrained stationary point on the leaf sum(x) = 6 minimizes F:
        # grad F = K x = lambda 1 -> x = lambda K^-1 1.
        lam = 6.0 / float(torch.sum(1.0 / torch.diagonal(k)))
        x_star = lam / torch.diagonal(k)
        certificate = certify_relative_storage(storage, x_star, ledger=ledger)
        assert certificate.verdict == "lyapunov_candidate"
        assert certificate.multiplier == pytest.approx((lam,))
        assert certificate.hessian_min_eigenvalue_on_leaf > 0.0
        # V >= 0 on the leaf and dot V = -D < 0 for the gradient flow projected on the leaf.
        x = x_star + torch.tensor([0.3, -0.1, -0.2], dtype=F64)
        assert float(relative_storage(storage, x, x_star)) > 0.0
        grad = k @ x
        projected = grad - ledger.reshape(-1) * (grad.sum() / 3.0)
        assert float(relative_storage_rate(storage, x, -projected, x_star)) < 0.0

    def test_bihss_counterexample_is_refused_and_its_full_rate_is_positive(self):
        # z = x - 1, F = 2 + z + z^2/2, z' = -z^2: F'' = 1 and dot F <= 0, but
        # grad F(x*) = 1 != 0, so V = z^2/2 is not a Lyapunov function.
        def storage(x):
            z = x - 1.0
            return (2.0 + z + 0.5 * z * z).sum()

        x_star = torch.tensor([1.0], dtype=F64)
        certificate = certify_relative_storage(storage, x_star)
        assert certificate.stationary is False
        assert certificate.verdict.startswith("refused: x* is not a constrained stationary point")
        z = torch.tensor([-0.2], dtype=F64)
        x = x_star + z
        x_dot = -z * z
        assert float(relative_storage(storage, x, x_star)) == pytest.approx(0.5 * 0.04)
        # Full chain rule keeps the linear term: dot V = -z^3 > 0 for z < 0.
        assert float(relative_storage_rate(storage, x, x_dot, x_star)) == pytest.approx(0.008)
        # Dropping the linear term would wrongly report dot F = -(1 + z) z^2 < 0.
        assert float(-(1.0 + z) * z * z) < 0.0

    def test_indefinite_hessian_on_the_leaf_is_refused(self):
        def storage(x):
            return 0.5 * (x[0] ** 2 - x[1] ** 2)

        certificate = certify_relative_storage(storage, torch.zeros(2, dtype=F64))
        assert certificate.stationary is True and certificate.definite_on_leaf is False
        assert certificate.verdict.startswith("refused: Hessian")

    def test_feedback_power_ledger_closes_and_refuses_negative_dissipation(self):
        closed, residual = feedback_power_ledger(
            torch.tensor(-0.3, dtype=F64), torch.tensor(0.2, dtype=F64), torch.tensor(0.5, dtype=F64), torch.tensor(0.0, dtype=F64)
        )
        assert closed and residual == pytest.approx(0.0)
        with pytest.raises(ValueError, match="nonnegative"):
            feedback_power_ledger(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(0.0))


class TestFickianConsumer:
    """Closed 1D Fickian diffusion: H = 0.5 c^T c, ledger C = 1^T (mass)."""

    def _functor(self):
        return FickianDiffusion(
            name="fickian_storage_consumer",
            config={"parameters": {"diffusivity": 1e-5}},
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            time_integration="explicit",
        )

    def test_uniform_state_is_the_constrained_stationary_point_and_storage_decreases_along_the_real_step(self):
        functor = self._functor()
        concentration = torch.ones(1, 20, dtype=F64) * 50.0
        concentration[0, 10] = 100.0
        state = {"tracer_concentration": concentration.clone(), "source_term": torch.zeros_like(concentration)}
        ledger = torch.ones(1, 20, dtype=F64)
        x = concentration.reshape(-1)
        x_star = torch.full_like(x, float(x.mean()))

        def storage(c):
            return 0.5 * (c * c).sum()

        certificate = certify_relative_storage(storage, x_star, ledger=ledger)
        assert certificate.verdict == "lyapunov_candidate"
        assert certificate.multiplier == pytest.approx((float(x.mean()),))
        # The functor's own storage functional agrees with H = 0.5 c^T c.
        assert float(functor.energy_terms(state)["storage_functional"]) == pytest.approx(float(storage(x)))
        dt = 0.01
        outputs = functor.forward(state, time=0.0, dt=dt)
        delta = outputs["tracer_concentration"].value.reshape(-1)
        assert float(delta.sum()) == pytest.approx(0.0, abs=1e-9)  # ledger preserved
        x_next = x + delta
        v_before = float(relative_storage(storage, x, x_star))
        v_after = float(relative_storage(storage, x_next, x_star))
        assert v_before > v_after > 0.0
        # dot V from the actual increment equals the negative dissipation the
        # functor reports, to explicit-Euler order.
        rate = float(relative_storage_rate(storage, x, delta / dt, x_star))
        dissipation = float(functor.entropy_terms(state)["dissipation_rate"])
        assert dissipation > 0.0
        assert rate == pytest.approx(-dissipation, rel=1e-2)
        closed, _ = feedback_power_ledger(torch.tensor(rate, dtype=F64), torch.tensor(0.0, dtype=F64), torch.tensor(dissipation, dtype=F64), torch.tensor(0.0, dtype=F64), tolerance=1e-2)
        assert closed


def _validation_iphs_entry(key):
    from pathlib import Path

    import yaml

    path = (
        Path(__file__).resolve().parents[4]
        / "relic" / "parameters" / "database" / "relic" / "validation" / "iphs.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][key]


class TestLeafBasisRankGuard:
    """The ker C rank cut is a dtype roundoff guard: sigma_i > eps max(m, n) sigma_max.

    A tiny but exact ledger row keeps its rank (the former absolute 1e-12 cut
    dropped it), and duplicate rows count once in float32 and float64.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_tiny_exact_ledger_keeps_its_rank(self, dtype):
        from relic.computation.operators.relative_storage import _leaf_basis

        ledger = 1.0e-14 * torch.tensor([[1.0, 1.0]], dtype=dtype)
        basis = _leaf_basis(ledger, 2, dtype)
        assert tuple(basis.shape) == (2, 1)
        assert float(torch.max(torch.abs(ledger @ basis))) <= 4.0 * float(torch.finfo(dtype).eps) * 1.0e-14

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_duplicate_ledger_rows_count_once(self, dtype):
        from relic.computation.operators.relative_storage import _leaf_basis

        ledger = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=dtype)
        basis = _leaf_basis(ledger, 3, dtype)
        assert tuple(basis.shape) == (3, 2)

    def test_tiny_ledger_leaf_certifies_the_constrained_storage(self):
        # F = 0.5 (x0 - x1)^2 has Hessian eigenvalue 2 on ker C = span(1, -1) and 0 along (1, 1).
        def storage(x):
            return 0.5 * (x[0] - x[1]) ** 2

        certificate = certify_relative_storage(
            storage, torch.zeros(2, dtype=F64), ledger=1.0e-14 * torch.tensor([[1.0, 1.0]], dtype=F64),
            tolerance=1.0e-9,
        )
        assert certificate.definite_on_leaf is True
        assert certificate.verdict == "lyapunov_candidate"


class TestToleranceParameters:
    """Nonzero default tolerances resolve from ParameterManager; explicit values override.

    Mathematical expression: F(x) = 0.5 x.x + 5e-10 x_0 has grad F(0) = (5e-10, 0),
    so x* = 0 is stationary only for a tolerance >= 5e-10; the power ledger
    dot F = 1 + 5e-10, y^T u = 1, D = Phi = 0 has residual 5e-10 on scale 1.
    """

    STORAGE_KEY = "validation.iphs.relative_storage_tolerance"
    LEDGER_KEY = "validation.iphs.feedback_power_ledger_rtol"
    METADATA = ("value", "units", "dtype", "bounds", "category", "description",
                "references", "tags", "requires_grad", "provenance")

    @staticmethod
    def _storage(x):
        return 0.5 * torch.dot(x, x) + 5.0e-10 * x[0]

    @staticmethod
    def _ledger(**kwargs):
        return feedback_power_ledger(
            torch.tensor(1.0 + 5.0e-10, dtype=F64), torch.tensor(1.0, dtype=F64),
            torch.tensor(0.0, dtype=F64), torch.tensor(0.0, dtype=F64), **kwargs,
        )

    def test_defaults_resolve_from_the_parameter_database(self):
        from relic.parameters import get_global_manager

        for key in (self.STORAGE_KEY, self.LEDGER_KEY):
            entry = _validation_iphs_entry(key)
            for field_name in self.METADATA:
                assert field_name in entry, (key, field_name)
            assert get_global_manager().get(key, required=True) == pytest.approx(float(entry["value"]))
        certificate = certify_relative_storage(self._storage, torch.zeros(2, dtype=F64))
        assert certificate.verdict == "lyapunov_candidate"
        assert certificate.tolerance == pytest.approx(float(_validation_iphs_entry(self.STORAGE_KEY)["value"]))
        refused = certify_relative_storage(self._storage, torch.zeros(2, dtype=F64), tolerance=1.0e-10)
        assert refused.stationary is False
        assert self._ledger()[0] is True
        assert self._ledger(tolerance=1.0e-10)[0] is False

    def test_changing_the_parameters_changes_the_checks(self):
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        originals = {key: manager.get(key, required=True) for key in (self.STORAGE_KEY, self.LEDGER_KEY)}
        try:
            manager.set_value(self.STORAGE_KEY, 1.0e-10)
            assert certify_relative_storage(self._storage, torch.zeros(2, dtype=F64)).stationary is False
            manager.set_value(self.LEDGER_KEY, 1.0e-10)
            assert self._ledger()[0] is False
        finally:
            for key, value in originals.items():
                manager.set_value(key, value)
        assert {key: manager.get(key, required=True) for key in originals} == originals

    GRADIENT_FLOOR_KEY = "validation.iphs.relative_storage_gradient_scale_floor"
    LEDGER_FLOOR_KEY = "validation.iphs.feedback_power_ledger_scale_floor"

    def test_scale_floors_resolve_from_the_parameter_database(self):
        # References: Hairer, Norsett and Wanner (1993) Sect. II.4 (mixed
        #   absolute/relative tolerance); both floors are convention values.
        # Mathematical expression: ||grad F(x*)|| <= tol max(floor_g, ||grad F(x*)||)
        #   and |res| <= tol max(floor_p, |dot F|, |y^T u|, |D|, |Phi|).
        # Parameter meanings: F(x) = 0.5e-3 x.x at x* = (1, 0) gives ||grad F|| = 1e-3;
        #   the ledger dot F = 1e-3, y^T u = D = Phi = 0 gives |res| = 1e-3; tol = 1e-2.
        # Units: normalized storage and power of the declared basis.
        # Domain/constraints: floor 1.0 -> absolute test 1e-2 (both pass); floor
        #   1e-6 -> relative test 1e-5 (both refused).
        # Sign convention: only magnitudes are compared.
        from relic.parameters import get_global_manager

        def storage(x):
            return 0.5e-3 * torch.dot(x, x)

        x_star = torch.tensor([1.0, 0.0], dtype=F64)
        zero = torch.tensor(0.0, dtype=F64)
        rate = torch.tensor(1.0e-3, dtype=F64)
        manager = get_global_manager()
        keys = (self.GRADIENT_FLOOR_KEY, self.LEDGER_FLOOR_KEY)
        for key in keys:
            entry = _validation_iphs_entry(key)
            for field_name in self.METADATA:
                assert field_name in entry, (key, field_name)
            assert manager.get(key, required=True) == 1.0
        assert certify_relative_storage(storage, x_star, tolerance=1.0e-2).stationary is True
        assert feedback_power_ledger(rate, zero, zero, zero, tolerance=1.0e-2)[0] is True
        originals = {key: manager.get(key, required=True) for key in keys}
        try:
            for key in keys:
                manager.set_value(key, 1.0e-6)
            assert certify_relative_storage(storage, x_star, tolerance=1.0e-2).stationary is False
            assert feedback_power_ledger(rate, zero, zero, zero, tolerance=1.0e-2)[0] is False
        finally:
            for key, value in originals.items():
                manager.set_value(key, value)
        assert {key: manager.get(key, required=True) for key in keys} == originals
