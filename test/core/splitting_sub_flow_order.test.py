"""CR-A-04 T11: the declared sub-flow accuracy of a Functor equals its measured own order.

Design 20261019_cr_a04_design_01, D5 and T11: steps 9-13, reaction, two-region, bucket and snow sub-flows. D5: no
production Functor declared FunctorMetadata.sub_flow_accuracy, so every real Strang
composition reported its order as unverified. A declaration is a claim about the
local order p of one governed call over [t, t + dt]; it is accepted here only when
the global error of that Functor alone, run through governed calls (RELICSimulator
.step -> execution group -> FunctorBase.__call__ -> StateManager commit) with dt
halved over a fixed horizon, falls by the factor 2^p that the declaration predicts.
For a single fast member the splitting report gives q = p_1 (no splitting error), so
the reported order of the governed composition is the expectation the measurement is
checked against; an undeclared Functor reports "unverified" and cannot pass.
An EXACT sub-flow is checked against an independent matrix exponential across
the dt ladder and by the governed semigroup law, with rounding-level errors;
no finite-order ratio is fitted to those errors.

Both reaction kernels advance x_dot = N_h r_h(x) by one explicit Euler step per call (the
call's Michaelis-Menten rate at the call-start pool, drivers held), with the donor
limit R_exec = max(min(R, S / dt), 0). The limit is inactive exactly when
R(S) dt < S, i.e. Vmax F dt < Km + S. Where it binds the call empties the pool: the
local error includes the exact pool remaining after the call. The tested coarse
levels both exhaust the numerical pool and show an equal-error plateau; this is
a finite-step characterization, not loss of asymptotic convergence as dt -> 0.

References: Hairer, Norsett and Wanner (1993) Solving Ordinary Differential
Equations I, 2nd ed., Sect. II.3 (explicit Euler converges with order 1); Hairer,
Lubich and Wanner (2006) Geometric Numerical Integration, 2nd ed., Sect. II.4-II.5
(order of compositions with order-p sub-flows); Michaelis and Menten (1913);
Prosser (1989) Adv. Microb. Physiol. 30:125-181; Parton et al. (1996) GBC 10:401-412;
Li et al. (1992) JGR 97:9759-9776.
Mathematical expression: dS/dt = -Veff S / (Km + S) with Veff = Vmax F constant;
exact S(t) from S + Km ln S = S0 + Km ln S0 - Veff t; global error e(dt) = S_dt(T) -
S(T); observed ratio rho = e(dt) / e(dt / 2) -> 2^p; registered band 2^p +- 0.3
over the two finest pairs (design section 5: 1.7-2.3 for p = 1).
Parameter meanings: S the ammonium or nitrate donor pool; Km its half-saturation; Vmax the potential rate; F the product of
the environmental factors at the constant drivers; Veff = R0 (Km + S0) / S0 from the
governed rate R0 of a 1-s call (limiter inactive); T the horizon.
Units: S, Km [g N m-2]; R, Vmax, Veff [g N m-2 s-1]; t, dt, T [s]; rho [1].
Domain/constraints: CPU, float64 states (test/conftest.py); constant drivers; the
smooth cases keep Vmax F dt < Km + S at every call (checked per call); the
limiter-active cases bind the limit only on the coarse levels.
Sign convention: R and Veff are positive consumption magnitudes; e < 0 means the
scheme under-predicts the remaining pool.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Tuple

import pytest
import torch

from relic.core.functor import SubFlowAccuracy
from relic.core.simulator import RELICSimulator
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess
from relic.scientific.pedosphere.transport.two_region_mass_transfer import TwoRegionMassTransferFunctor
from relic.scientific.hydrosphere.soil_moisture.bucket import BucketSoilMoisture
from relic.scientific.hydrosphere.snow.snowpack import SnowPack
from relic.data.standards.unit_converter import unit_converter

# Registered half width of the observed-ratio band around 2^p (design section 5).
_RATIO_HALF_WIDTH = 0.3
# Order-1 band used by the declaration-independent limiter characterizations.
_ORDER_1_BAND = (2.0 - _RATIO_HALF_WIDTH, 2.0 + _RATIO_HALF_WIDTH)
_LIMITER_KEY = "nitrification.nitrification.substrate_limiter_residual"


@dataclass(frozen=True)
class _Case:
    """One governed dt-halving experiment of a single Functor."""

    factory: Callable[[], object]
    pool: str
    rate: str
    initial: float
    horizon: float
    ladder: Tuple[float, ...]
    drivers: Callable[[object], Dict[str, float]]


@dataclass(frozen=True)
class _Level:
    dt: float
    final: float
    starts: Tuple[float, ...]  # pool at the start of every call [g N m-2]
    releases: Tuple[float, ...]  # realized release R_exec dt of every call [g N m-2]
    limiter: Tuple[float, ...]  # stored limiter residual of every call (nitrification)
    report: Mapping[str, object]
    plan_accuracy: Tuple[str, ...]


@dataclass(frozen=True)
class _Ladder:
    km: float
    veff: float
    exact: float
    levels: Tuple[_Level, ...]

    def errors(self) -> List[float]:
        return [level.final - self.exact for level in self.levels]

    def ratios(self) -> List[float]:
        errors = self.errors()
        return [coarse / fine for coarse, fine in zip(errors[:-1], errors[1:])]


def _scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu().item())
    return float(value)


def _nitrification_drivers(functor) -> Dict[str, float]:
    """Reference temperature, aerobic soil, optimum pH: every factor equals 1."""
    return {
        "soil_temperature": _scalar(functor.temperature_reference),
        "water_filled_pore_space": 0.5 * _scalar(functor.wfps_optimum),
        "soil_ph": _scalar(functor.ph_optimum),
    }


def _denitrification_drivers(functor) -> Dict[str, float]:
    """Reference temperature, optimum pH, half-way through the anaerobic ramp."""
    return {
        "soil_temperature": _scalar(functor.temperature_reference),
        "water_filled_pore_space": _scalar(functor.wfps_threshold)
        + 0.5 * _scalar(functor.anaerobic_transition_width),
        "soil_ph": _scalar(functor.ph_optimum),
        "labile_carbon": 1.0,
    }


def _km(functor) -> float:
    return _scalar(functor.km_nh4 if isinstance(functor, NitrificationProcess) else functor.km_no3)


def _governed_level(case: _Case, dt: float, horizon: float) -> _Level:
    simulator = RELICSimulator({"device": "cpu", "time_step": dt})
    functor = case.factory()
    state = {case.pool: torch.tensor(case.initial, dtype=torch.float64)}
    if case.pool == "nh4_nitrogen":
        state["no3_nitrogen"] = torch.tensor(0.0, dtype=torch.float64)
    state.update({key: torch.tensor(value, dtype=torch.float64) for key, value in case.drivers(functor).items()})
    simulator.initialize_state(state)
    simulator.register_functor(functor, execution_group="fast")
    report = simulator.get_splitting_order_report()
    plan_accuracy = tuple(
        str(stage.stage_metadata["sub_flow_accuracy"]) for stage in simulator.get_composition_plan().stages
    )
    manager = simulator.state_manager
    starts: List[float] = []
    releases: List[float] = []
    limiter: List[float] = []
    steps = int(round(horizon / dt))
    assert steps * dt == pytest.approx(horizon, rel=0.0, abs=1.0e-9)
    for _ in range(steps):
        starts.append(_scalar(manager.get_state(case.pool)))
        simulator.step(dt)
        releases.append(_scalar(manager.get_state(case.rate)) * dt)
        current = manager.get_diagnostics()["current"]
        if _LIMITER_KEY in current:
            limiter.append(_scalar(current[_LIMITER_KEY]))
    return _Level(
        dt=dt,
        final=_scalar(manager.get_state(case.pool)),
        starts=tuple(starts),
        releases=tuple(releases),
        limiter=tuple(limiter),
        report=report,
        plan_accuracy=plan_accuracy,
    )


def _implicit_michaelis_menten(s0: float, km: float, veff: float, horizon: float) -> float:
    """Solve the independent constant-driver substrate oracle.

    References: Michaelis and Menten (1913); Hairer et al. (1993), Sect. II.3.
    Mathematical expression: S + Km*ln(S) = S0 + Km*ln(S0) - Veff*T.
    Parameter meanings: S/S0 are final/initial donor nitrogen; Km is half saturation;
    Veff is the environment-adjusted maximum rate; T is the horizon.
    Units: S/S0/Km [g N m-2]; Veff [g N m-2 s-1]; T [s].
    Domain/constraints: S0, Km, Veff, T > 0; the root lies in (0, S0).
    Sign convention: positive Veff depletes the donor; the root remains positive.
    """
    target = s0 + km * math.log(s0) - veff * horizon
    lower, upper = 1.0e-300, s0
    for _ in range(400):
        middle = 0.5 * (lower + upper)
        if middle + km * math.log(middle) - target < 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


@functools.lru_cache(maxsize=None)
def _ladder(name: str) -> _Ladder:
    case = _CASES[name]
    km = _km(case.factory())
    # The kinetic rate at S0 from one governed 1-s call (Veff * 1 s << Km: no limit).
    probe = _governed_level(case, 1.0, 1.0)
    veff = probe.releases[0] / 1.0 * (km + case.initial) / case.initial
    exact = _implicit_michaelis_menten(case.initial, km, veff, case.horizon)
    levels = tuple(_governed_level(case, dt, case.horizon) for dt in case.ladder)
    return _Ladder(km=km, veff=veff, exact=exact, levels=levels)


def _limiter_inactive(ladder: _Ladder, level: _Level) -> bool:
    """A-priori condition Veff dt < Km + S at every call start, and the governed outputs agree."""
    a_priori = all(ladder.veff * level.dt < ladder.km + start for start in level.starts)
    released_less = all(release < start for release, start in zip(level.releases, level.starts))
    residual_zero = all(value == 0.0 for value in level.limiter)
    return a_priori and released_less and residual_zero


def _limiter_bound_somewhere(level: _Level) -> bool:
    """Some call released the whole pool present at its start (to rounding)."""
    emptied = any(
        start > 0.0 and abs(release - start) <= 1.0e-12 * start
        for release, start in zip(level.releases, level.starts)
    )
    residual = any(value > 0.0 for value in level.limiter) if level.limiter else True
    return emptied and residual


_CASES: Dict[str, _Case] = {
    # S0 = 2 Km, every factor 1 (Veff = Vmax, Km / Veff = 2500 s); S(T) about Km / 2.
    "nitrification_smooth": _Case(
        factory=NitrificationProcess,
        pool="nh4_nitrogen",
        rate="nitrification_flux",
        initial=0.01,
        horizon=7200.0,
        ladder=(1800.0, 900.0, 450.0, 225.0, 112.5),
        drivers=_nitrification_drivers,
    ),
    # S0 = 0.4 Km: the limit binds for dt >= 3600 s (Vmax dt > Km + S0), not below.
    "nitrification_limiter": _Case(
        factory=NitrificationProcess,
        pool="nh4_nitrogen",
        rate="nitrification_flux",
        initial=0.002,
        horizon=7200.0,
        ladder=(7200.0, 3600.0, 1800.0, 900.0, 450.0),
        drivers=_nitrification_drivers,
    ),

    # S0 = 2 Km, anaerobic factor 0.5 (Km / Veff = 2800 s).
    "denitrification_smooth": _Case(
        factory=DenitrificationProcess,
        pool="no3_nitrogen",
        rate="denitrification_flux",
        initial=0.004,
        horizon=7200.0,
        ladder=(1800.0, 900.0, 450.0, 225.0, 112.5),
        drivers=_denitrification_drivers,
    ),
    # S0 = 0.5 Km: the limit binds for dt >= 3600 s (first or second call), not below.
    "denitrification_limiter": _Case(
        factory=DenitrificationProcess,
        pool="no3_nitrogen",
        rate="denitrification_flux",
        initial=0.001,
        horizon=7200.0,
        ladder=(7200.0, 3600.0, 1800.0, 900.0, 450.0),
        drivers=_denitrification_drivers,
    ),
}


def _assert_declared_order_1(functor, ladder_name: str) -> None:
    metadata = functor.metadata
    assert metadata.sub_flow_accuracy is SubFlowAccuracy.ORDER_1, (
        f"{functor.name} declares {metadata.sub_flow_accuracy!r}; its kernel is one explicit "
        "Euler step per call (order 1)"
    )
    assert metadata.to_dict()["sub_flow_accuracy"] == "order_1"
    level = _ladder(ladder_name).levels[0]
    assert level.report["sub_flow_accuracy"] == {functor.name: "order_1"}
    assert level.report["effective_order"] == 1  # n = 1: q = p_1
    assert level.plan_accuracy == ("order_1",)


def _assert_own_order_matches_the_declaration(ladder_name: str) -> None:
    ladder = _ladder(ladder_name)
    for level in ladder.levels:
        assert _limiter_inactive(ladder, level), f"donor limit active at dt = {level.dt}"
    reported = ladder.levels[0].report["effective_order"]
    assert isinstance(reported, int) and not isinstance(reported, bool), (
        f"the governed composition reports order {reported!r}; a measured order can only "
        "confirm a declared one"
    )
    expected = 2.0 ** reported
    errors = ladder.errors()
    assert all(abs(fine) < abs(coarse) for coarse, fine in zip(errors[:-1], errors[1:])), errors
    # The finest error lies far above the float64 floor of the pool.
    assert abs(errors[-1]) > 1.0e6 * torch.finfo(torch.float64).eps * ladder.exact
    for ratio in ladder.ratios()[-2:]:
        assert expected - _RATIO_HALF_WIDTH <= ratio <= expected + _RATIO_HALF_WIDTH, (
            f"observed ratio {ratio:.4f} outside {expected} +- {_RATIO_HALF_WIDTH}; "
            f"errors {errors}"
        )


def _assert_limit_degrades_where_it_binds(ladder_name: str, bound_levels: int) -> None:
    ladder = _ladder(ladder_name)
    bound, free = ladder.levels[:bound_levels], ladder.levels[bound_levels:]
    for level in bound:
        assert _limiter_bound_somewhere(level), f"limit not active at dt = {level.dt}"
        # The pool is exhausted, so the error is the whole exact remaining pool.
        assert level.final <= 1.0e-12 * ladder.levels[0].starts[0]
        assert level.final - ladder.exact == pytest.approx(-ladder.exact, rel=1.0e-9)
    for level in free:
        assert _limiter_inactive(ladder, level), f"donor limit active at dt = {level.dt}"
    ratios = ladder.ratios()
    # Halving dt inside the bound range does not reduce the error (rho = 1): the
    # first-order claim does not hold where the limit binds.
    for ratio in ratios[: bound_levels - 1]:
        assert ratio == pytest.approx(1.0, abs=1.0e-9)
        assert not (_ORDER_1_BAND[0] <= ratio <= _ORDER_1_BAND[1])
    # Once the limit is inactive the explicit Euler order 1 is observed again.
    for ratio in ratios[-2:]:
        assert _ORDER_1_BAND[0] <= ratio <= _ORDER_1_BAND[1], ratios


# --------------------------------------------------------------------------
# Step 9: nitrification (NH4 -> NO3 + N2O).
# --------------------------------------------------------------------------


def test_nitrification_declares_an_order_1_sub_flow():
    _assert_declared_order_1(NitrificationProcess(), "nitrification_smooth")


def test_nitrification_own_order_by_dt_halving_matches_the_declared_order():
    _assert_own_order_matches_the_declaration("nitrification_smooth")


def test_nitrification_donor_limit_degrades_the_observed_order_where_it_binds():
    _assert_limit_degrades_where_it_binds("nitrification_limiter", bound_levels=2)


# --------------------------------------------------------------------------
# Step 10: denitrification (NO3 -> N2O + N2) and the P-N pair.
# --------------------------------------------------------------------------


def test_denitrification_declares_an_order_1_sub_flow():
    _assert_declared_order_1(DenitrificationProcess(), "denitrification_smooth")


def test_denitrification_own_order_by_dt_halving_matches_the_declared_order():
    _assert_own_order_matches_the_declaration("denitrification_smooth")


def test_denitrification_donor_limit_degrades_the_observed_order_where_it_binds():
    _assert_limit_degrades_where_it_binds("denitrification_limiter", bound_levels=2)


def test_pn_pair_is_admitted_under_require_declared_strang_with_reported_order_1():
    """Problem P-N (design section 5): S3 admits the pair once both members declare.

    Both members are stateful (they share no3_nitrogen, a state-coupled transfer that
    S1 admits); with ORDER_1 sub-flows the Strang composition reports
    q = min(2, 1) = 1 for one call per sub-flow, and every stage of the exported
    IntegrationPlan carries the declaration.
    """
    simulator = RELICSimulator(
        {
            "device": "cpu",
            "time_step": 3600.0,
            "splitting_mode": "strang",
            "splitting_policy": "require_declared",
        }
    )
    simulator.initialize_state(
        {
            "nh4_nitrogen": torch.tensor(0.2, dtype=torch.float64),
            "no3_nitrogen": torch.tensor(0.002, dtype=torch.float64),
            "soil_temperature": torch.tensor(293.15, dtype=torch.float64),
            "water_filled_pore_space": torch.tensor(0.9, dtype=torch.float64),
            "soil_ph": torch.tensor(7.25, dtype=torch.float64),
            "labile_carbon": torch.tensor(1.0, dtype=torch.float64),
        }
    )
    simulator.register_functor(NitrificationProcess(), execution_group="fast")
    simulator.register_functor(DenitrificationProcess(), execution_group="fast")
    report = simulator.get_splitting_order_report()
    assert report["splitting_policy"] == "require_declared"
    assert report["member_class"] == {"nitrification": "stateful", "denitrification": "stateful"}
    assert report["refused_or_flagged_edges"] == ()
    assert report["sub_flow_accuracy"] == {"nitrification": "order_1", "denitrification": "order_1"}
    assert report["splitting_order_limit"] == 2
    assert report["effective_order"] == 1
    plan = simulator.get_composition_plan()
    assert [stage.operator_kind for stage in plan.stages] == ["nitrification", "denitrification", "nitrification"]
    assert {stage.stage_metadata["sub_flow_accuracy"] for stage in plan.stages} == {"order_1"}


# --------------------------------------------------------------------------
# Step 11: exact two-region mobile-immobile exchange.
# --------------------------------------------------------------------------


def _two_region_simulator(policy="warn"):
    simulator = RELICSimulator(
        {
            "device": "cpu",
            "time_step": 3600.0,
            "splitting_mode": "strang",
            "splitting_policy": policy,
            "enable_auto_scale": False,
        }
    )
    simulator.initialize_state(
        {
            TwoRegionMassTransferFunctor.MOBILE_VARIABLE: torch.tensor(
                [2.0, 0.5, 0.0], dtype=torch.float64
            ),
            TwoRegionMassTransferFunctor.IMMOBILE_VARIABLE: torch.tensor(
                [0.1, 0.5, 3.0], dtype=torch.float64
            ),
        }
    )
    functor = TwoRegionMassTransferFunctor()
    simulator.register_functor(functor, execution_group="fast")
    return simulator, functor


def _two_region_run(step_lengths, policy="warn"):
    simulator, functor = _two_region_simulator(policy)
    mobile, immobile = functor.STATE_VARIABLES
    state_manager = simulator.state_manager
    initial = torch.stack(
        [state_manager.get_state(mobile).clone(), state_manager.get_state(immobile).clone()]
    )
    alpha, theta_m, theta_im = (
        functor.get_parameter_value(name, required=True).detach().to(dtype=torch.float64)
        for name in (
            "two_region_mass_transfer_coefficient",
            "two_region_mobile_water_content",
            "two_region_immobile_water_content",
        )
    )
    # References: van Genuchten and Wierenga (1976); Haggerty and Gorelick (1995).
    # Mathematical expression: c_dot = A c, A = [[-alpha/theta_m, alpha/theta_m],
    #   [alpha/theta_im, -alpha/theta_im]]; oracle(T) = exp(T A) c(0).
    # Parameter meanings: c = (c_m, c_im); alpha is the exchange coefficient;
    #   theta_m and theta_im are the two water contents; T is the elapsed time.
    # Units: c [kg m-3]; alpha and A [s-1]; theta [m3 m-3]; T [s].
    # Domain/constraints: fixed positive coefficients and float64 states;
    #   a matrix exponential is an independent oracle for the scalar-exp kernel.
    # Sign convention: positive alpha transfers mass down c_m - c_im;
    #   theta_m c_m + theta_im c_im is invariant.
    operator = torch.stack(
        [torch.stack([-alpha / theta_m, alpha / theta_m]),
         torch.stack([alpha / theta_im, -alpha / theta_im])]
    )
    horizon = sum(step_lengths)
    exact = torch.linalg.matrix_exp(horizon * operator) @ initial
    total0 = theta_m * initial[0] + theta_im * initial[1]
    residuals = []
    report = simulator.get_splitting_order_report()
    plan_accuracy = tuple(
        stage.stage_metadata["sub_flow_accuracy"]
        for stage in simulator.get_composition_plan().stages
    )
    for dt in step_lengths:
        simulator.step(dt)
        total = theta_m * state_manager.get_state(mobile) + theta_im * state_manager.get_state(immobile)
        residuals.append(float((total - total0).abs().max().detach()))
    final = torch.stack([state_manager.get_state(mobile), state_manager.get_state(immobile)])
    return {
        "final": final,
        "exact": exact,
        "max_abs_error": float((final - exact).abs().max().detach()),
        "max_abs_mass_residual": max(residuals),
        "report": report,
        "plan_accuracy": plan_accuracy,
        "steps": simulator.step_count,
        "horizon": horizon,
    }


@functools.lru_cache(maxsize=1)
def _two_region_ladder():
    return tuple(_two_region_run((7200.0 / count,) * count) for count in (1, 2, 4, 8, 16))


def test_two_region_declares_an_exact_sub_flow():
    functor = TwoRegionMassTransferFunctor()
    assert functor.metadata.sub_flow_accuracy is SubFlowAccuracy.EXACT
    assert functor.metadata.to_dict()["sub_flow_accuracy"] == "exact"


def test_two_region_governed_dt_halving_stays_at_roundoff():
    # References: van Genuchten and Wierenga (1976); Hairer et al. (2006), II.4-II.5.
    # Mathematical expression: Phi_h^n c(0) = exp(n h A) c(0), for n h = T;
    #   the observed error is rounding only, so no finite-order ratio is fitted.
    # Parameter meanings: Phi_h is one governed call, A the exchange generator,
    #   n the step count, h the step duration, T the common horizon.
    # Units: c [kg m-3]; A [s-1]; h and T [s]; n [1].
    # Domain/constraints: constant coefficients; h halves over five levels;
    #   the 1e-12 bound is the existing two-region exact-solution acceptance.
    # Sign convention: errors are magnitudes; theta-weighted mass stays constant.
    for result in _two_region_ladder():
        torch.testing.assert_close(result["final"], result["exact"], rtol=1e-12, atol=1e-12)
        assert result["max_abs_error"] <= 1e-12
        assert result["max_abs_mass_residual"] <= 1e-12
        assert result["report"]["effective_order"] == "exact"
        assert result["report"]["splitting_order_limit"] is None
        assert result["plan_accuracy"] == ("exact",)


def test_two_region_governed_partitions_obey_exact_semigroup():
    coarse = _two_region_run((7200.0,))
    partitioned = _two_region_run((111.0, 222.0, 333.0, 1444.0, 5090.0))
    assert coarse["horizon"] == partitioned["horizon"] == 7200.0
    torch.testing.assert_close(coarse["final"], partitioned["final"], rtol=1e-12, atol=1e-12)
    assert partitioned["max_abs_mass_residual"] <= 1e-12


def test_two_region_exact_is_admitted_under_require_declared_strang():
    result = _two_region_run((3600.0, 3600.0), policy="require_declared")
    assert result["steps"] == 2
    assert result["report"]["member_class"] == {"two_region_mass_transfer": "stateful"}
    assert result["report"]["sub_flow_accuracy"] == {"two_region_mass_transfer": "exact"}
    assert result["report"]["effective_order"] == "exact"
    assert result["plan_accuracy"] == ("exact",)
    torch.testing.assert_close(result["final"], result["exact"], rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# Step 12: bucket drainage in a smooth domain and at the drainage cap.
# --------------------------------------------------------------------------


def _bucket_parameters(functor):
    values = {name: _scalar(getattr(functor, name)) for name in (
        "soil_depth", "field_capacity", "wilting_point", "saturation",
        "k_sat", "drainage_factor", "drainage_exponent", "water_density",
    )}
    assert values["drainage_factor"] > 0.0, "The order witness requires nonzero drainage"
    assert values["drainage_exponent"] >= 1.0
    return values


def _bucket_exact(theta0, k, exponent, elapsed):
    # References: Manabe (1969); Rodriguez-Iturbe and Porporato (2004).
    # Mathematical expression: theta_dot=-k*theta^b;
    #   theta(t)=[theta0^(1-b)+(b-1)*k*t]^(1/(1-b)) for b != 1,
    #   and theta(t)=theta0*exp(-k*t) for b=1.
    # Parameter meanings: k=Ks*fD/(Z*theta_sat^b), b is the drainage exponent;
    #   Ks is saturated conductivity, fD the drainage factor, Z the soil depth,
    #   theta_sat the porosity, theta0 initial water content and t elapsed time.
    # Units: theta/theta0 [m3 m-3]; k [s-1]; Ks [m s-1]; Z [m]; t [s]; fD/b [1].
    # Domain/constraints: P=ET=0, k>0, b>=1; the returned value remains above
    #   field capacity for the registered horizons, with no runoff or projection.
    # Sign convention: positive drainage lowers theta; this is the donor solution.
    if exponent == 1.0:
        return theta0 * math.exp(-k * elapsed)
    return (theta0 ** (1.0 - exponent) + (exponent - 1.0) * k * elapsed) ** (1.0 / (1.0 - exponent))


def _bucket_horizon(theta0, target, k, exponent):
    # References: Manabe (1969); the separated power-law drainage equation.
    # Mathematical expression: T=(theta_t^(1-b)-theta0^(1-b))/((b-1)*k)
    #   for b!=1, and T=log(theta0/theta_t)/k for b=1.
    # Parameter meanings: theta0/theta_t are initial/target water content,
    #   k the drainage coefficient and b the power-law exponent.
    # Units: theta0/theta_t [m3 m-3]; k [s-1]; b [1]; T [s].
    # Domain/constraints: 0<theta_t<theta0, k>0 and b>=1.
    # Sign convention: T is the positive time for drainage to lower theta.
    if exponent == 1.0:
        return math.log(theta0 / target) / k
    return (target ** (1.0 - exponent) - theta0 ** (1.0 - exponent)) / ((exponent - 1.0) * k)


def _bucket_level(initial, horizon, count):
    functor = BucketSoilMoisture()
    params = _bucket_parameters(functor)
    dt = horizon / count
    simulator = RELICSimulator(
        {"device": "cpu", "time_step": dt, "splitting_mode": "strang", "enable_auto_scale": False}
    )
    state = {name: torch.tensor(0.0, dtype=torch.float64) for name in functor.metadata.input_variables}
    state["soil_moisture"] = torch.tensor(initial, dtype=torch.float64)
    simulator.initialize_state(state)
    simulator.register_functor(functor, execution_group="fast")
    report = simulator.get_splitting_order_report()
    plan_accuracy = tuple(stage.stage_metadata["sub_flow_accuracy"] for stage in simulator.get_composition_plan().stages)
    manager = simulator.state_manager
    depth, density, field_capacity = params["soil_depth"], params["water_density"], params["field_capacity"]
    # References: Manabe (1969); the existing bucket power-law drainage.
    # Mathematical expression: k=Ks*fD/(Z*theta_sat^b), theta_dot=-k*theta^b.
    # Parameter meanings: Ks is saturated conductivity, fD its scale factor,
    #   Z=depth, theta_sat=porosity, b=drainage_exponent, theta=soil moisture.
    # Units: Ks [m s-1]; Z [m]; theta/theta_sat [m3 m-3]; fD/b [1]; k [s-1].
    # Domain/constraints: fixed positive Ks/Z/theta_sat, fD>0, b>=1.
    # Sign convention: positive k produces loss of soil water.
    k = params["k_sat"] * params["drainage_factor"] / (depth * params["saturation"] ** params["drainage_exponent"])
    starts, margins, residuals, runoff = [], [], [], []
    for _ in range(count):
        before = _scalar(manager.get_state("soil_moisture"))
        starts.append(before)
        # References: Manabe (1969); implemented power-law drainage contract.
        # Mathematical expression: h*k*theta^b < theta-theta_fc is the exact
        #   inactive-cap condition of the current explicit drainage call.
        # Parameter meanings: h=dt, theta is call-start water content, theta_fc
        #   field capacity; k and b are the fixed drainage coefficients.
        # Units: both sides and their margin [m3 m-3]; h [s]; k [s-1]; b [1].
        # Domain/constraints: zero precipitation and ET; strict positive margin
        #   with a rounding allowance identifies the smooth witness, not a zero
        #   comparison of the post-correction diagnostic.
        # Sign convention: positive margin means the drainage cap is inactive.
        margins.append(before - field_capacity - dt * k * before ** params["drainage_exponent"])
        simulator.step(dt)
        after = _scalar(manager.get_state("soil_moisture"))
        drainage = _scalar(manager.get_state("drainage_flux"))
        surface_runoff = _scalar(manager.get_state("surface_runoff_flux"))
        # References: Manabe (1969), open bucket water balance.
        # Mathematical expression: R=(theta_after-theta_before)*Z*rho+(D+Q)*h.
        # Parameter meanings: Z=depth, rho=water density, D=drainage,
        #   Q=surface runoff, h=dt; R is the unforced water ledger residual.
        # Units: theta [m3 m-3]; Z [m]; rho [kg m-3]; D/Q [kg m-2 s-1];
        #   h [s]; R [kg m-2].
        # Domain/constraints: prescribed P=ET=0; use actual committed state
        #   and reported rates without absorbing the residual in the oracle.
        # Sign convention: storage gain and exported water enter R positively.
        residuals.append((after - before) * depth * density + (drainage + surface_runoff) * dt)
        runoff.append(surface_runoff)
    return {"dt": dt, "count": count, "final": _scalar(manager.get_state("soil_moisture")),
            "starts": starts, "cap_margins": margins, "water_residuals": residuals,
            "runoff": runoff, "report": report, "plan_accuracy": plan_accuracy}


@functools.lru_cache(maxsize=2)
def _bucket_ladder(kind):
    parameters = _bucket_parameters(BucketSoilMoisture())
    fc, sat = parameters["field_capacity"], parameters["saturation"]
    # References: the power-law drainage solution; Hairer et al. (1993), II.3.
    # Mathematical expression: theta0=fc+0.75*(sat-fc); theta_t=fc+0.25*(sat-fc);
    #   T_smooth=T(theta0,theta_t); T_cap=(T(theta0,fc)+T_E)/2,
    #   T_E=(theta0-fc)/(k*theta0^b); e_h=theta_h-theta(T); ratio=e_h/e_(h/2).
    # Parameter meanings: fc/sat are field capacity/porosity; theta0/theta_t
    #   initial/target moisture, k the fixed drainage coefficient, b its power,
    #   T the analytic arrival time, T_E the one-step Euler cap time, h a step.
    # Units: theta/fc/sat [m3 m-3]; T/T_E/h [s]; k [s-1]; b/ratio [1].
    # Domain/constraints: the registered nonzero-drainage witness and fixed
    #   total horizon; actual cap margins are checked at every required level.
    # Sign convention: e_h<0 is excessive drainage by explicit Euler.
    initial = fc + 0.75 * (sat - fc)
    exponent = parameters["drainage_exponent"]
    k = parameters["k_sat"] * parameters["drainage_factor"] / (parameters["soil_depth"] * sat ** exponent)
    if kind == "smooth":
        target = fc + 0.25 * (sat - fc)
        horizon = _bucket_horizon(initial, target, k, exponent)
        counts = (4, 8, 16, 32, 64)
    elif kind == "cap":
        hit_time = _bucket_horizon(initial, fc, k, exponent)
        euler_cap_time = (initial - fc) / (k * initial ** exponent)
        assert euler_cap_time < hit_time
        horizon = 0.5 * (euler_cap_time + hit_time)
        counts = (1, 2, 4, 8, 16)
    else:
        raise ValueError("Unknown bucket T11 case")
    exact = _bucket_exact(initial, k, exponent, horizon)
    assert exact > fc
    levels = tuple(_bucket_level(initial, horizon, count) for count in counts)
    errors = [level["final"] - exact for level in levels]
    ratios = [coarse / fine for coarse, fine in zip(errors[:-1], errors[1:])]
    return {"parameters": parameters, "initial": initial, "horizon": horizon,
            "exact": exact, "levels": levels, "errors": errors, "ratios": ratios}


def test_bucket_declares_an_order_1_sub_flow():
    functor = BucketSoilMoisture()
    assert functor.metadata.sub_flow_accuracy is SubFlowAccuracy.ORDER_1
    assert functor.metadata.to_dict()["sub_flow_accuracy"] == "order_1"
    level = _bucket_ladder("smooth")["levels"][0]
    assert level["report"]["sub_flow_accuracy"] == {functor.name: "order_1"}
    assert level["report"]["effective_order"] == 1
    assert level["plan_accuracy"] == ("order_1",)


def test_bucket_smooth_drainage_order_matches_governed_report():
    ladder = _bucket_ladder("smooth")
    params = ladder["parameters"]
    allowance = 64.0 * torch.finfo(torch.float64).eps
    for level in ladder["levels"]:
        assert min(level["cap_margins"]) > allowance
        assert all(params["field_capacity"] < value < params["saturation"] for value in level["starts"])
        assert max(map(abs, level["water_residuals"])) <= 1.0e-9
        assert all(value == 0.0 for value in level["runoff"])
    errors = ladder["errors"]
    assert all(error < 0.0 for error in errors)
    assert all(abs(fine) < abs(coarse) for coarse, fine in zip(errors[:-1], errors[1:]))
    assert abs(errors[-1]) > 100.0 * torch.finfo(torch.float64).eps * ladder["exact"]
    reported = ladder["levels"][0]["report"]["effective_order"]
    assert reported == 1
    expected = 2.0 ** reported
    assert all(expected - _RATIO_HALF_WIDTH <= ratio <= expected + _RATIO_HALF_WIDTH
               for ratio in ladder["ratios"][-2:])


def test_bucket_coarse_drainage_cap_is_a_finite_step_characterization():
    ladder = _bucket_ladder("cap")
    fc = ladder["parameters"]["field_capacity"]
    coarse = ladder["levels"][0]
    assert coarse["cap_margins"][0] < 0.0
    assert coarse["final"] == pytest.approx(fc, rel=0.0, abs=1e-12)
    assert ladder["exact"] > fc
    for level in ladder["levels"][-3:]:
        assert min(level["cap_margins"]) > 64.0 * torch.finfo(torch.float64).eps
        assert max(map(abs, level["water_residuals"])) <= 1.0e-9
    assert all(_ORDER_1_BAND[0] <= ratio <= _ORDER_1_BAND[1] for ratio in ladder["ratios"][-2:])


# --------------------------------------------------------------------------
# Step 13: snow storage is affine inside a branch, not an exact global flow.
# --------------------------------------------------------------------------


def _snowpack_level(kind, intervals):
    functor = SnowPack()
    horizon = sum(intervals)
    melt_threshold = _scalar(functor.melt_temperature_threshold)
    partition_threshold = _scalar(functor.rain_snow_temperature_threshold)
    # References: Hock (2003), Eq. 1; Anderson (1976); the existing SnowPack law.
    # Mathematical expression: M0=DDF*max(T-Tm,0)/s_day; U converts depth/mass
    #   rates; for replenishment P>0, E0=2P, S0=P*H/4 and M0=0.
    # Parameter meanings: M0/E0 are potential melt/sublimation, P snowfall,
    #   S0 starting SWE, H the horizon, T/Tm air/melt temperature, DDF melt
    #   factor, s_day seconds per day and U the canonical UnitConverter.
    # Units: S0 [mm]; P/M0/E0 [mm s-1]; external P/E [kg m-2 s-1];
    #   T/Tm [K]; H/s_day [s]; DDF [mm K-1 day-1].
    # Domain/constraints: all drivers are constant; cold replenishment has
    #   no melt, warm cases have no snowfall, and all prescribed rates >=0.
    # Sign convention: P enters storage; M0 and E0 are outward demands.
    temperature = max(melt_threshold, partition_threshold) + 3.0
    potential_melt = _scalar(functor.degree_day_factor) * (temperature - melt_threshold) / _scalar(functor.seconds_per_day)
    precipitation, sublimation, initial = 0.0, 1.0e-5, 40.0
    if kind == "replenishment":
        temperature = min(melt_threshold, partition_threshold) - 3.0
        precipitation, sublimation, potential_melt = 1.0e-3, 2.0e-3, 0.0
        snowfall = _scalar(unit_converter.convert(precipitation, "precipitation_flux", "kg m-2 s-1", "mm s-1"))
        initial = snowfall * horizon / 4.0
    elif kind == "priority":
        initial = potential_melt * horizon
        sublimation = _scalar(unit_converter.convert(potential_melt, "sublimation_flux", "mm s-1", "kg m-2 s-1"))
    elif kind != "affine":
        raise ValueError("Unknown snowpack T11 case")
    potential_sublimation = _scalar(unit_converter.convert(sublimation, "sublimation_flux", "kg m-2 s-1", "mm s-1"))
    snowfall = _scalar(unit_converter.convert(precipitation, "precipitation_flux", "kg m-2 s-1", "mm s-1")) if kind == "replenishment" else 0.0
    simulator = RELICSimulator({"device": "cpu", "time_step": intervals[0], "splitting_mode": "strang", "enable_auto_scale": False})
    simulator.initialize_state({name: torch.tensor(value, dtype=torch.float64) for name, value in {
        "precipitation_flux": precipitation, "air_temperature": temperature,
        "snow_water_equivalent": initial, "sublimation_flux": sublimation,
    }.items()})
    simulator.register_functor(functor, execution_group="fast")
    report = simulator.get_splitting_order_report()
    plan_accuracy = tuple(stage.stage_metadata["sub_flow_accuracy"] for stage in simulator.get_composition_plan().stages)
    steps = []
    for dt in intervals:
        before = _scalar(simulator.state_manager.get_state("snow_water_equivalent"))
        simulator.step(dt)
        stored = simulator.state_manager.get_diagnostics()["current"]
        diagnostics = {name: stored[f"{functor.name}.snow_pack.{name}"] for name in (
            "melt_rate_mm_s", "sublimation_rate_mm_s", "melt_capacity_residual_mm_s",
            "sublimation_capacity_residual_mm_s", "swe_nonnegative_projection_residual_mm",
        )}
        after = _scalar(simulator.state_manager.get_state("snow_water_equivalent"))
        melt = _scalar(diagnostics["melt_rate_mm_s"])
        sublimated = _scalar(diagnostics["sublimation_rate_mm_s"])
        # References: Anderson (1976); SnowPack's one-commit water balance.
        # Mathematical expression: R=S1-S0-(P-M-E)*h; margins S0-M0*h and
        #   S0-(M0+E0)*h test the two sequential unconstrained demands.
        # Parameter meanings: S0/S1 starting/committed SWE, P snowfall,
        #   M/E realized melt/sublimation, M0/E0 demands, h the call length.
        # Units: S0/S1/R/margins [mm]; P/M/E/M0/E0 [mm s-1]; h [s].
        # Domain/constraints: use the committed state and actual returned
        #   diagnostics; a positive margin excludes its capacity limiter.
        # Sign convention: R>0 is excess storage; M and E export water.
        steps.append({"dt": dt, "before": before, "after": after,
                      "melt": melt, "sublimation": sublimated,
                      "melt_margin": before - potential_melt * dt,
                      "total_margin": before - (potential_melt + potential_sublimation) * dt,
                      "water_residual": after - before - (snowfall - melt - sublimated) * dt,
                      "melt_cap": _scalar(diagnostics["melt_capacity_residual_mm_s"]),
                      "sublimation_cap": _scalar(diagnostics["sublimation_capacity_residual_mm_s"]),
                      "projection": _scalar(diagnostics["swe_nonnegative_projection_residual_mm"])})
    return {"initial": initial, "horizon": horizon, "snowfall": snowfall,
            "potential_melt": potential_melt, "potential_sublimation": potential_sublimation,
            "final": _scalar(simulator.state_manager.get_state("snow_water_equivalent")),
            "steps": steps, "report": report, "plan_accuracy": plan_accuracy}


@functools.lru_cache(maxsize=3)
def _snowpack_ladder(kind):
    return tuple(_snowpack_level(kind, (3600.0 / count,) * count) for count in (4, 8, 16, 32, 64))


def test_snowpack_declares_conservative_order_1_without_claiming_exactness():
    functor = SnowPack()
    assert functor.metadata.sub_flow_accuracy is SubFlowAccuracy.ORDER_1
    assert functor.metadata.to_dict()["sub_flow_accuracy"] == "order_1"
    level = _snowpack_ladder("affine")[0]
    assert level["report"]["sub_flow_accuracy"] == {functor.name: "order_1"}
    assert level["report"]["effective_order"] == 1
    assert level["plan_accuracy"] == ("order_1",)


def test_snowpack_fixed_driver_interior_is_affine_to_roundoff():
    # References: Hock (2003), Eq. 1; Anderson (1976).
    # Mathematical expression: S(H)=S0+(P-M0-E0)*H while no cap is active.
    # Parameter meanings: S0/S starting/final SWE, P snowfall, M0/E0 fixed
    #   melt/sublimation demand and H the shared time horizon.
    # Units: S0/S [mm]; P/M0/E0 [mm s-1]; H [s].
    # Domain/constraints: positive storage throughout; fixed forcing;
    #   rounding-level errors must not be fitted to a finite order ratio.
    # Sign convention: positive P increases SWE; M0/E0 decrease it.
    for level in (*_snowpack_ladder("affine"), _snowpack_level("affine", (111.0, 222.0, 333.0, 2934.0))):
        exact = level["initial"] + (level["snowfall"] - level["potential_melt"] - level["potential_sublimation"]) * level["horizon"]
        assert level["final"] == pytest.approx(exact, rel=0.0, abs=1e-10)
        for step in level["steps"]:
            assert step["total_margin"] > 0.0
            assert step["melt_cap"] == step["sublimation_cap"] == step["projection"] == 0.0
            assert abs(step["water_residual"]) <= 1e-10


def test_snowpack_replenishment_boundary_has_nonzero_first_order_error():
    # References: Anderson (1976); Hairer et al. (2006), splitting of flows.
    # Mathematical expression: S_next=max(S-E0*h,0)+P*h, E0=2P;
    #   S_exact(H)=max(S0+(P-E0)*H,0)=0; after depletion S_h(H)=P*h.
    # Parameter meanings: S/S0/S_next are SWE, P snowfall, E0 potential
    #   sublimation, h the numerical step and H the common final time.
    # Units: S [mm]; P/E0 [mm s-1]; h/H [s]; error ratio [1].
    # Domain/constraints: constant cold forcing, no melt, S0=P*H/4;
    #   the registered grids deplete before H, so errors are nonzero.
    # Sign convention: error>0 is snow retained by end-of-step replenishment.
    ladder = _snowpack_ladder("replenishment")
    errors = []
    for level in ladder:
        assert level["final"] == pytest.approx(level["snowfall"] * level["steps"][0]["dt"], rel=0.0, abs=1e-12)
        assert level["final"] > 100.0 * torch.finfo(torch.float64).eps
        assert any(step["sublimation_cap"] > 0.0 for step in level["steps"])
        assert all(abs(step["water_residual"]) <= 1e-10 for step in level["steps"])
        errors.append(level["final"])
    ratios = [coarse / fine for coarse, fine in zip(errors[:-1], errors[1:])]
    assert all(_ORDER_1_BAND[0] <= ratio <= _ORDER_1_BAND[1] for ratio in ratios[-2:])


def test_snowpack_melt_sublimation_priority_is_not_an_exact_flux_semigroup():
    # References: Anderson (1976); existing SnowPack sequential donor limits.
    # Mathematical expression: S0=M0*H, E0=M0, P=0. One call releases
    #   M0*H as melt and zero sublimation; two half calls release M0*H/2
    #   as each flux. Both finish at S=0 and close the water budget.
    # Parameter meanings: M0/E0 potential melt/sublimation, S/S0 final/start
    #   SWE, P snowfall and H the common horizon.
    # Units: S/S0 [mm]; M0/E0/P [mm s-1]; H [s].
    # Domain/constraints: fixed warm forcing and simultaneous equal demands;
    #   compare integrated actual fluxes, not merely identical final storage.
    # Sign convention: realized melt and sublimation are positive exports.
    whole = _snowpack_level("priority", (3600.0,))
    halves = _snowpack_level("priority", (1800.0, 1800.0))
    whole_melt = sum(step["melt"] * step["dt"] for step in whole["steps"])
    halves_melt = sum(step["melt"] * step["dt"] for step in halves["steps"])
    whole_sub = sum(step["sublimation"] * step["dt"] for step in whole["steps"])
    halves_sub = sum(step["sublimation"] * step["dt"] for step in halves["steps"])
    assert whole["final"] == halves["final"] == pytest.approx(0.0, abs=1e-12)
    assert whole_melt == pytest.approx(whole["initial"], abs=1e-12)
    assert whole_sub == pytest.approx(0.0, abs=1e-12)
    assert halves_melt == halves_sub == pytest.approx(whole["initial"] / 2.0, abs=1e-12)
    assert abs(whole_melt - halves_melt) > 1e-3
    assert all(abs(step["water_residual"]) <= 1e-10 for level in (whole, halves) for step in level["steps"])
