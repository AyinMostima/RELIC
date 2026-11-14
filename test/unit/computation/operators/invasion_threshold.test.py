"""G12-THRESHOLD: resource-activity invasion threshold on a finite matrix family.

References: BIHSS conclusion C5 and appendix lemma (Metzler threshold);
Berman and Plemmons (1994); van den Driessche and Watmough (2002).
Mathematical expression: R_act = rho(Delta^-1 G(R0)); sign(R_act - 1) =
sign(s(G - Delta)); trajectories of x' = (G - Delta) x decay iff R_act < 1.
Parameter meanings: G nonnegative growth operator, Delta nonsingular
M-matrix loss operator, R0 = Lambda_R^-1 I_R resource background.
Units: rates s-1; R_act dimensionless.
Domain/constraints: premises are checked and refused; R_act = 1 is critical.
Sign convention: R_act > 1 invasion, < 1 decay.
"""

from __future__ import annotations

import pytest
import torch

from relic.computation.operators.invasion_threshold import (
    activity_threshold,
    is_nonnegative,
    is_nonsingular_m_matrix,
    is_z_matrix,
    resource_background,
    spectral_bound,
    spectral_radius,
)
from relic.scientific.pedosphere.decomposition.microbial_enzyme import MicrobialEnzymeDecomposition

F64 = torch.float64


def _trajectory_norm(growth, loss, x0, horizon):
    return float(torch.linalg.norm(torch.linalg.matrix_exp((growth - loss) * horizon) @ x0))


class TestMatrixPremises:
    def test_z_and_m_matrix_predicates(self):
        delta = torch.tensor([[1.0, -2.0], [0.0, 1.0]], dtype=F64)
        assert is_z_matrix(delta) and is_nonsingular_m_matrix(delta)
        assert not is_z_matrix(torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=F64))
        # Z-matrix with a nonpositive eigenvalue is not a nonsingular M-matrix.
        assert not is_nonsingular_m_matrix(torch.tensor([[1.0, -2.0], [-2.0, 1.0]], dtype=F64))
        assert not is_nonsingular_m_matrix(torch.zeros(2, 2, dtype=F64))
        assert is_nonnegative(torch.tensor([[0.0, 0.1]], dtype=F64))

    def test_resource_background_solves_the_positive_diagonal_loss(self):
        background = resource_background(torch.diag(torch.tensor([2.0, 4.0], dtype=F64)), torch.tensor([1.0, 2.0], dtype=F64))
        assert background.tolist() == pytest.approx([0.5, 0.5])


class TestThresholdOracles:
    def test_bihss_counterexample_matrices_give_r_act_below_one_and_a_negative_spectral_bound(self):
        delta = torch.tensor([[1.0, -2.0], [0.0, 1.0]], dtype=F64)
        growth = torch.tensor([[0.05, 0.05], [0.10, 0.10]], dtype=F64)
        certificate = activity_threshold(growth, delta)
        assert certificate.activity_threshold == pytest.approx(0.35, abs=1e-12)
        assert certificate.spectral_bound < 0.0 and certificate.verdict == "decay"
        # The superseded left-multiplication step is invalid: K v < v holds
        # for v = (1, 1) but G v < Delta v does not.
        v = torch.ones(2, dtype=F64)
        k = torch.linalg.solve(delta, growth)
        assert bool(torch.all(k @ v < v))
        assert not bool(torch.all(growth @ v < delta @ v))
        # Two-sided trajectory evidence: the activity-free background is stable.
        x0 = torch.tensor([1e-3, 1e-3], dtype=F64)
        assert _trajectory_norm(growth, delta, x0, 50.0) < float(torch.linalg.norm(x0))

    def test_analytic_two_by_two_invasion(self):
        delta = torch.diag(torch.tensor([1.0, 2.0], dtype=F64))
        growth = torch.tensor([[0.0, 3.0], [1.0, 0.0]], dtype=F64)
        certificate = activity_threshold(growth, delta)
        # K = Delta^-1 G = [[0, 3], [0.5, 0]] has rho = sqrt(1.5).
        assert certificate.activity_threshold == pytest.approx(1.5 ** 0.5)
        assert certificate.verdict == "invasion" and certificate.spectral_bound > 0.0
        assert spectral_bound(growth - delta) == pytest.approx(certificate.spectral_bound)
        x0 = torch.tensor([1e-3, 1e-3], dtype=F64)
        assert _trajectory_norm(growth, delta, x0, 20.0) > float(torch.linalg.norm(x0))

    def test_growth_callable_of_the_resource_background(self):
        loss = torch.diag(torch.tensor([1.0, 1.0], dtype=F64))
        certificate = activity_threshold(
            lambda r0: torch.diag(r0), torch.eye(2, dtype=F64),
            resource_loss=torch.diag(torch.tensor([2.0, 0.5], dtype=F64)),
            resource_supply=torch.tensor([1.0, 1.0], dtype=F64),
        )
        assert certificate.resource_background == pytest.approx((0.5, 2.0))
        assert certificate.activity_threshold == pytest.approx(2.0) and certificate.verdict == "invasion"
        assert spectral_radius(loss) == pytest.approx(1.0)

    def test_critical_case_is_reported_without_a_stability_verdict(self):
        certificate = activity_threshold(torch.eye(2, dtype=F64), torch.eye(2, dtype=F64))
        assert certificate.activity_threshold == pytest.approx(1.0)
        assert certificate.verdict == "critical" and certificate.hyperbolic is False
        assert abs(certificate.spectral_bound) <= 1e-12

    def test_premise_violations_are_refused(self):
        with pytest.raises(ValueError, match="nonnegative"):
            activity_threshold(torch.tensor([[-0.1, 0.0], [0.0, 0.1]], dtype=F64), torch.eye(2, dtype=F64))
        with pytest.raises(ValueError, match="nonsingular M-matrix"):
            activity_threshold(torch.eye(2, dtype=F64), torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=F64))
        with pytest.raises(ValueError, match="same shape"):
            activity_threshold(torch.eye(3, dtype=F64), torch.eye(2, dtype=F64))
        with pytest.raises(ValueError, match="requires resource_loss"):
            activity_threshold(lambda r: torch.eye(2, dtype=F64), torch.eye(2, dtype=F64))


class TestMicrobialEnzymeConsumer:
    """The activity-free linearization of the microbial-enzyme decomposition."""

    def _rates(self, functor, substrate: float, biomass: float):
        state = {
            "metabolic_litter_c": torch.tensor([substrate], dtype=F64),
            "structural_litter_c": torch.tensor([substrate], dtype=F64),
            "available_som_c": torch.tensor([substrate], dtype=F64),
            "microbial_biomass_c": torch.tensor([biomass], dtype=F64),
            "soil_temperature": torch.tensor([293.15], dtype=F64),
            "soil_moisture": torch.tensor([0.3], dtype=F64),
        }
        # A short step keeps the explicit update inside the declared biomass bounds.
        outputs = functor.forward(state, time=0.0, dt=1.0e-3)
        growth = float(outputs["diagnostics.microbial_enzyme.microbial_growth_rate"].value.reshape(-1)[0])
        death = float(outputs["diagnostics.microbial_enzyme.microbial_death_rate"].value.reshape(-1)[0])
        return growth, death

    def test_threshold_predicts_the_sign_of_the_functor_net_change_near_the_activity_free_state(self):
        functor = MicrobialEnzymeDecomposition(name="microbial_threshold_consumer", config={})
        # Just above the smallest biomass the functor admits (declared
        # bounds): the activity-free neighbourhood where both rates are
        # proportional to MIC.
        biomass = 10.0 * float(functor.microbial_params["min_biomass"])
        assert biomass > 0.0
        verdicts = {}
        for substrate in (1e-3, 50.0):
            growth, death = self._rates(functor, substrate, biomass)
            # Near MIC -> 0 both rates are linear in MIC: G = growth / MIC,
            # Delta = death / MIC (one activity variable).
            g = torch.tensor([[growth / biomass]], dtype=F64)
            delta = torch.tensor([[death / biomass]], dtype=F64)
            certificate = activity_threshold(g, delta)
            verdicts[substrate] = certificate.verdict
            assert certificate.activity_threshold == pytest.approx(growth / death)
            assert (growth > death) == (certificate.verdict == "invasion")
        assert verdicts[1e-3] == "decay"
        assert verdicts[50.0] == "invasion"


def _validation_iphs_entry(key):
    from pathlib import Path

    import yaml

    path = (
        Path(__file__).resolve().parents[4]
        / "relic" / "parameters" / "database" / "relic" / "validation" / "iphs.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][key]


class TestToleranceParameters:
    """Nonzero default tolerances resolve from ParameterManager; explicit values override.

    Mathematical expression: Delta = [[1, 5e-13], [0, 1]] has a positive off-diagonal
    entry of 5e-13 (a Z-matrix only within the sign slack); G = diag(1 + 5e-10, 0.5)
    with Delta = I gives R_act = 1 + 5e-10 and s(G - Delta) = 5e-10.
    """

    M_MATRIX_KEY = "validation.iphs.m_matrix_sign_tolerance"
    CRITICAL_BAND_KEY = "validation.iphs.invasion_threshold_critical_band"
    METADATA = ("value", "units", "dtype", "bounds", "category", "description",
                "references", "tags", "requires_grad", "provenance")

    @staticmethod
    def _perturbed_loss():
        return torch.tensor([[1.0, 5.0e-13], [0.0, 1.0]], dtype=F64)

    @staticmethod
    def _near_critical():
        return torch.diag(torch.tensor([1.0 + 5.0e-10, 0.5], dtype=F64)), torch.eye(2, dtype=F64)

    def test_defaults_resolve_from_the_parameter_database(self):
        from relic.parameters import get_global_manager

        for key in (self.M_MATRIX_KEY, self.CRITICAL_BAND_KEY):
            entry = _validation_iphs_entry(key)
            for field_name in self.METADATA:
                assert field_name in entry, (key, field_name)
            assert get_global_manager().get(key, required=True) == pytest.approx(float(entry["value"]))
        assert is_nonsingular_m_matrix(self._perturbed_loss()) is True
        assert is_nonsingular_m_matrix(self._perturbed_loss(), tolerance=1.0e-14) is False
        growth, loss = self._near_critical()
        certificate = activity_threshold(growth, loss)
        assert certificate.verdict == "critical"
        assert certificate.tolerance == pytest.approx(
            float(_validation_iphs_entry(self.CRITICAL_BAND_KEY)["value"])
        )
        assert activity_threshold(growth, loss, tolerance=1.0e-10).verdict == "invasion"

    def test_changing_the_parameters_changes_the_checks(self):
        from relic.parameters import get_global_manager

        manager = get_global_manager()
        originals = {key: manager.get(key, required=True) for key in (self.M_MATRIX_KEY, self.CRITICAL_BAND_KEY)}
        growth, loss = self._near_critical()
        try:
            manager.set_value(self.M_MATRIX_KEY, 1.0e-14)
            assert is_nonsingular_m_matrix(self._perturbed_loss()) is False
            manager.set_value(self.CRITICAL_BAND_KEY, 1.0e-10)
            assert activity_threshold(growth, loss).verdict == "invasion"
        finally:
            for key, value in originals.items():
                manager.set_value(key, value)
        assert {key: manager.get(key, required=True) for key in originals} == originals
