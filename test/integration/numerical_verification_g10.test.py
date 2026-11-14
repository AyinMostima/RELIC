"""G10: applicable numerical verification, coupling consistency, and outputs.

Independent oracles in priority order: analytic solutions and conservation
identities first, then the corrected A04 event-time / reset sensitivity
formulas against autograd and central finite differences, then the real
governed simulator runs (dt sensitivity, long-run finiteness and domain,
single-commit provenance of every coupled increment, and a fixed-time
management event whose parameter sensitivity is checked by finite
differences through the governed chain).

References: Parton et al. (1987) SSSAJ 51 (first-order kinetics);
Hairer, Norsett and Wanner (1993) Solving ODEs I (explicit Euler global
error O(dt)); Goebel, Sanfelice and Teel (2012) Hybrid Dynamical Systems
(event-time sensitivity and saltation); RELIC manuscript section 21 (A04).
Mathematical expression (Euler order): e(dt) = |C_dt(T) - C0 exp(-k T)|,
e(dt) / e(dt/2) -> 2 as dt -> 0.
Mathematical expression (A04): tau_theta = -(g_x S_minus + g_theta) /
(g_x f_minus + g_t); S_plus = R_x S_minus + R_theta +
(R_x f_minus + R_t - f_plus) tau_theta.
Parameter meanings: g the guard, f_minus / f_plus the pre- and post-event
vector fields, R the reset map, S the state sensitivity to the parameter
theta, tau the event time.
Units: decomposition pools g C m-2 and rates s-1; the hybrid oracle is
dimensionless; the fertilizer sensitivity is kg N per kg product.
Domain/constraints: transversal events only (g_x f_minus + g_t != 0);
grazing (vanishing denominator) is excluded and not claimed.
Sign convention: as in the manuscript; sensitivities are total
derivatives of the terminal state with respect to theta.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.scientific.anthroposphere.management.fertilization import (
    FertilizationManagement,
    FertilizerEvent,
    FertilizerType,
)
from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
from relic.utils.physical_constants import get_physical_constant

REPO_ROOT = Path(__file__).resolve().parents[2]
SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))
INGESTED = "first_order_decomposition.first_order_decomposition."


def _decomposition_simulator(dt_hint: float) -> tuple[RELICSimulator, FirstOrderDecomposition]:
    f64 = torch.float64
    sim = RELICSimulator({
        "device": "cpu",
        "conservation_checks": False,
        "strict_dependencies": False,
        "adaptive_dt_enabled": False,
        "time_step": dt_hint,
    })
    sim.initialize_state({
        "metabolic_litter_c": torch.tensor(100.0, dtype=f64),
        "structural_litter_c": torch.tensor(200.0, dtype=f64),
        "fast_som_c": torch.tensor(500.0, dtype=f64),
        "slow_som_c": torch.tensor(1000.0, dtype=f64),
        "passive_som_c": torch.tensor(2000.0, dtype=f64),
        "soil_temperature": torch.tensor(293.15, dtype=f64),
        "soil_moisture": torch.tensor(0.3, dtype=f64),
        "metabolic_litter_input": torch.tensor(0.0, dtype=f64),
        "structural_litter_input": torch.tensor(0.0, dtype=f64),
        "lignin_n_ratio": torch.tensor(0.3, dtype=f64),
        "clay_content": torch.tensor(0.25, dtype=f64),
        "carbon_decomposition_rate": torch.tensor(0.0, dtype=f64),
        "heterotrophic_respiration": torch.tensor(0.0, dtype=f64),
    })
    functor = FirstOrderDecomposition(name="first_order_decomposition", config={})
    sim.register_functor(functor, execution_group="fast")
    return sim, functor


def _run_metabolic(dt: float, horizon: float) -> tuple[float, float]:
    sim, _ = _decomposition_simulator(dt)
    steps = int(round(horizon / dt))
    for _ in range(steps):
        sim.step(dt=dt)
    env = float(torch.as_tensor(sim.state_manager.get_diagnostics()["current"][INGESTED + "environmental_modifier"]).item())
    return float(sim.state_manager.get_state("metabolic_litter_c").item()), env


class TestTimeStepSensitivity:
    def test_explicit_euler_error_halves_with_the_time_step_against_the_analytic_decay(self):
        horizon = 24.0 * 3600.0
        k_year = float(get_physical_constant("seconds_per_year"))
        from relic.parameters import get_global_manager

        k_met = float(get_global_manager().get("pedosphere.decomposition.k_metabolic", required=True)) / k_year
        coarse, env = _run_metabolic(3600.0, horizon)
        fine, env_fine = _run_metabolic(1800.0, horizon)
        assert env == pytest.approx(env_fine, rel=1e-12)
        analytic = 100.0 * math.exp(-k_met * env * horizon)
        error_coarse = abs(coarse - analytic)
        error_fine = abs(fine - analytic)
        assert 0.0 < error_fine < error_coarse
        # First-order global error: the ratio approaches 2 (here within 5%).
        assert error_coarse / error_fine == pytest.approx(2.0, rel=0.05)
        assert error_coarse / analytic < 1e-5

    def test_long_run_stays_finite_inside_the_state_domain(self):
        sim, _ = _decomposition_simulator(3600.0)
        pools = ("metabolic_litter_c", "structural_litter_c", "fast_som_c", "slow_som_c", "passive_som_c")
        start = {k: float(sim.state_manager.get_state(k).item()) for k in pools}
        cumulative = 0.0
        for _ in range(240):
            sim.step(dt=3600.0)
            cumulative += float(sim.state_manager.get_state("heterotrophic_respiration").item()) * 3600.0
        end = {k: float(sim.state_manager.get_state(k).item()) for k in pools}
        assert all(math.isfinite(v) and v >= 0.0 for v in end.values())
        assert sum(end[k] - start[k] for k in pools) + cumulative == pytest.approx(0.0, abs=1e-8)
        assert end["metabolic_litter_c"] < start["metabolic_litter_c"]


def _hybrid_terminal_state(theta: torch.Tensor, *, f_minus: float, f_plus: float, threshold: float, horizon: float, reset_shift: float = 0.0) -> torch.Tensor:
    """x' = f_minus until x = threshold, reset x+ = x- + reset_shift * theta, then x' = f_plus.

    The event time is the differentiable root of the guard along the exact
    pre-event flow (implicit differentiation is automatic because the root
    is expressed in closed form).
    """
    event_time = (threshold - theta) / f_minus
    x_event = theta + f_minus * event_time
    x_plus = x_event + reset_shift * theta
    return x_plus + f_plus * (horizon - event_time)


def _saltation_formula(*, g_x, g_theta, g_t, f_minus, f_plus, r_x, r_theta, r_t, s_minus):
    tau_theta = -(g_x * s_minus + g_theta) / (g_x * f_minus + g_t)
    return r_x * s_minus + r_theta + (r_x * f_minus + r_t - f_plus) * tau_theta, tau_theta


class TestEventTimeSensitivityOracle:
    def test_counterexample_of_the_manuscript_gives_two_by_every_route(self):
        # x(0) = theta, event at x = 1, identity reset, f- = 1, f+ = 2, T = 2.
        theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        terminal = _hybrid_terminal_state(theta, f_minus=1.0, f_plus=2.0, threshold=1.0, horizon=2.0)
        (autograd,) = torch.autograd.grad(terminal, theta)
        assert float(autograd) == pytest.approx(2.0, abs=1e-12)
        formula, tau_theta = _saltation_formula(g_x=1.0, g_theta=0.0, g_t=0.0, f_minus=1.0, f_plus=2.0, r_x=1.0, r_theta=0.0, r_t=0.0, s_minus=1.0)
        assert tau_theta == pytest.approx(-1.0)
        assert formula == pytest.approx(2.0)
        # Post-event sensitivity propagates unchanged (f+ is constant), so the
        # terminal sensitivity equals S_plus.
        h = 1e-6
        plus = _hybrid_terminal_state(torch.tensor(0.3 + h, dtype=torch.float64), f_minus=1.0, f_plus=2.0, threshold=1.0, horizon=2.0)
        minus = _hybrid_terminal_state(torch.tensor(0.3 - h, dtype=torch.float64), f_minus=1.0, f_plus=2.0, threshold=1.0, horizon=2.0)
        assert float((plus - minus) / (2 * h)) == pytest.approx(2.0, abs=1e-7)
        # The superseded expression (no event-time term) misses the jump
        # contribution and cannot reproduce the terminal sensitivity.
        superseded = 1.0 * 1.0 + 0.0 + (1.0 * 1.0 - 2.0) * 0.0
        assert superseded == pytest.approx(1.0) and superseded != pytest.approx(2.0)

    def test_reset_parameter_dependence_and_fixed_time_event(self):
        # R_theta != 0: reset x+ = x- + theta.
        theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        terminal = _hybrid_terminal_state(theta, f_minus=1.0, f_plus=2.0, threshold=1.0, horizon=2.0, reset_shift=1.0)
        (autograd,) = torch.autograd.grad(terminal, theta)
        formula, _ = _saltation_formula(g_x=1.0, g_theta=0.0, g_t=0.0, f_minus=1.0, f_plus=2.0, r_x=1.0, r_theta=1.0, r_t=0.0, s_minus=1.0)
        assert float(autograd) == pytest.approx(formula, abs=1e-12) == pytest.approx(3.0)
        # Fixed-time event g = t - t_e: tau_theta = 0 and S_plus = R_x S_minus + R_theta.
        formula_fixed, tau_fixed = _saltation_formula(g_x=0.0, g_theta=0.0, g_t=1.0, f_minus=1.0, f_plus=2.0, r_x=1.0, r_theta=0.5, r_t=0.0, s_minus=1.0)
        assert tau_fixed == 0.0 and formula_fixed == pytest.approx(1.5)

    def test_grazing_denominator_is_refused_not_extrapolated(self):
        with pytest.raises(ZeroDivisionError):
            _saltation_formula(g_x=1.0, g_theta=0.0, g_t=-1.0, f_minus=1.0, f_plus=2.0, r_x=1.0, r_theta=0.0, r_t=0.0, s_minus=1.0)


class TestGovernedEventSensitivity:
    def _applied_nitrogen(self, product_mass: float) -> float:
        event = FertilizerEvent(fertilizer_type=FertilizerType.UREA, product_mass=product_mass, application_date=100)
        manager = FertilizationManagement(config={"calendar_type": "noleap"}, events=(event,))
        sim = RELICSimulator({"time_step": SECONDS_PER_DAY, "calendar_type": "noleap", "conservation_checks": False, "device": "cpu"})
        sim.initialize_state({
            "available_phosphorus": torch.tensor([0.0]),
            "fertilizer_n_flux": torch.tensor([0.0]),
            "fertilizer_p_flux": torch.tensor([0.0]),
        })
        sim.register_functor(manager, execution_group="fast")
        total = 0.0
        for _ in range(120):
            sim.step(dt=SECONDS_PER_DAY)
            total += float(sim.state_manager.get_state("fertilizer_n_flux").reshape(-1)[0]) * SECONDS_PER_DAY
        return total

    def test_fixed_time_event_sensitivity_matches_the_nitrogen_fraction_by_finite_differences(self):
        # tau_theta = 0 for a calendar event; S_plus = R_theta = f_N.
        base = self._applied_nitrogen(0.01)
        h = 1e-4
        finite_difference = (self._applied_nitrogen(0.01 + h) - self._applied_nitrogen(0.01 - h)) / (2 * h)
        assert base == pytest.approx(0.01 * 0.46, rel=1e-6)
        assert finite_difference == pytest.approx(0.46, rel=1e-6)


class TestCouplingConsistency:
    def test_every_coupled_increment_is_committed_exactly_once_per_step(self):
        spec = importlib.util.spec_from_file_location(
            "relic_budget_chain",
            REPO_ROOT / "test" / "integration" / "pedosphere" / "budget_ledger_governed_multistep.test.py",
        )
        assert spec is not None and spec.loader is not None
        chain = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chain)
        sim = chain._build(chain._initial_state())
        sim.step(dt=chain.DT)
        for key in ("nh4_nitrogen", "no3_nitrogen", "metabolic_litter_c", "organic_nitrogen"):
            sources = [item["functor"] for item in sim.get_variable_provenance(key) if item.get("role") == "increment_source"]
            assert sources, key
            assert len(sources) == len(set(sources)), (key, sources)
