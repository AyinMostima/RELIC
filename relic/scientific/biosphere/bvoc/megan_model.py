"""Reduced instantaneous MEGAN BVOC closure for RELIC.

This module implements a ParameterManager-backed, instantaneous subset of the
MEGAN2.1 response equations. It is not the full MEGAN canopy-history model:
leaf-age, soil-moisture history, canopy radiation profiles, and bidirectional
exchange remain outside this functor. The public ports are canonical RELIC
StateManager ports and all atmospheric species fluxes are molar. With the
``temperature_history`` option the 24 h and 240 h temperature means T24 and
T240 of Eqs. (9)-(10) are declared state that this functor reads at the start
of its call and advances afterwards (item 8); without it E_opt and T_opt stay
at the standard history T24 = T240 = T_s.

References
----------
- Guenther, A., et al. (2012). The Model of Emissions of Gases and Aerosols from 
  Nature version 2.1 (MEGAN2.1): an extended and updated framework for modeling 
  biogenic emissions. Geoscientific Model Development 5, 1471-1492
- Guenther, A.B., et al. (2006). Estimates of global terrestrial isoprene emissions 
  using MEGAN (Model of Emissions of Gases and Aerosols from Nature). 
  Atmospheric Chemistry and Physics 6, 3181-3210
- Monson, R.K., et al. (2007). Isoprene emission from terrestrial ecosystems in
  response to global change: minding the gap between models and observations.
  Philosophical Transactions of the Royal Society A 365, 1677-1695
- Monson, R.K., et al. (2012). Modeling the isoprene emission rate from leaves.
  New Phytologist 195, 541-559
- Niinemets, U., et al. (2010). The leaf-level emission factor of volatile 
  isoprenoids: caveats, model algorithms, response shapes and scaling. 
  Biogeosciences 7, 1809-1832

Mathematical Expressions
-------------------------
1. RELIC reduced dry-mass rate:
   rho_f = LAI k_f and
   E_i = epsilon_i gamma_T,i gamma_P,i rho_f / seconds_per_hour
2. Light activity factor: gamma_P = alpha CL PPFD / sqrt(1 + alpha^2 PPFD^2)
3. Temperature factor (Eq. 7): gamma_T,i = (1 - LDF_i) gamma_T,LIF,i
   + LDF_i gamma_T,LDF,i with gamma_T,LDF,i = E_opt,i CT2 exp(CT1,i x) /
   (CT2 - CT1,i (1 - exp(CT2 x))), x = 1/T_opt - 1/T over R (Eq. 8), and
   gamma_T,LIF,i = exp(beta_i (T - T_s)) (Eq. 11); one Table 4 row per class,
   LDF = 1 for isoprene
4. (Eq. 11 is the light-independent part of item 3)
5. CO2 factor: gamma_CO2 = ISmax - ISmax Ci^h / (Cstar^h + Ci^h)
6. Molar species flux: F_i = E_i 1e-6 / MW_i
7. Equivalent carbon allocation: F_C,alloc =
   convert(Sum_i(F_i MW_i f_C,i), g C to kg C)
8. Temperature history (option temperature_history): T_opt = T_opt,s +
   0.6 (T240 - T_s,h) (Eq. 9), E_opt,i = C_eo,i exp(0.05 (T24 - T_s,h))
   exp(0.05 (T240 - T_s,h)) (Eq. 10), T_s,h = 297 K; the running means
   advance as X_new = (1 - w) X + w T, w = min(dt / tau, 1), tau = 24 h and
   240 h

Units
-----
- E_i: Internal mass emission rate [ug m-2 s-1]
- F_i: Public species flux [mol m-2 s-1]
- F_C,alloc: Equivalent carbon allocation [kg C m-2 s-1]
- epsilon: Base emission factor [ug g-1 DW h-1]
- gamma_*: Activity factors [dimensionless]
- k_f: Effective dry-leaf mass per canonical RELIC LAI leaf area [g DW m-2]
- rho_f: LAI-derived dry-leaf mass per ground area [g DW m-2]
- Q: PPFD [umol photons m-2 s-1]
- T: Temperature [K]
- A_n: Net photosynthesis [umol CO2 m-2 s-1]

CO2 parameters are ``ismax``, ``c_star``, ``hill_exponent``, and
``intercellular_to_ambient_ratio`` under ``biosphere.bvoc.co2_inhibition``.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import torch
from torch import Tensor
from enum import Enum

from ....utils.physical_constants import get_physical_constant

# Import from core framework
from ....core.functor.biological_bridge import BiologicalProcessFunctor
from ....core.abstractions import FluxValue
from ....core.state.types import UpdateItem, UpdateMode
from ....computation.operators.iphs import advance_reaction_network
from ....data.standards.unit_converter import unit_converter
from ...atmosphere import daylight_quadrature


def megan_light_dependent_activity(ppfd: Tensor, alpha: Tensor, cl: Tensor) -> Tensor:
    """Light-dependent MEGAN activity factor gamma_P_LDF.

    This is the single production implementation shared by ``MEGANModel`` and the
    BVOC response helper (``relic.scientific.biosphere.bvoc.responses``).
    """
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eq. (4); the
    #   ParameterManager scale C_L stands for the history coefficient C_P of Eqs. (5)-(6),
    #   which RELIC's reduced instantaneous closure does not evaluate.
    # Mathematical expression: gamma_P_LDF = alpha * C_L * PPFD / sqrt(1 + alpha^2 PPFD^2).
    # Parameter meanings: PPFD is the photosynthetic photon flux density, alpha the PPFD
    #   response slope and C_L the light scaling coefficient.
    # Units: PPFD [umol photons m-2 s-1]; alpha [m2 s umol-1]; C_L and gamma_P_LDF [1].
    # Domain/constraints: PPFD must be finite and non-negative (validated by the callers);
    #   alpha and C_L are positive ParameterManager values; no clamp is applied.
    # Sign convention: gamma_P_LDF is a non-negative multiplier of the emission magnitude.
    return alpha * cl * ppfd / torch.sqrt(1.0 + alpha.square() * ppfd.square())


def megan_light_activity(ppfd: Tensor, alpha: Tensor, cl: Tensor, ldf: Tensor) -> Tensor:
    """MEGAN2.1 light activity factor gamma_P of a partly light-dependent class.

    This is the single production implementation of Eq. (3), shared by
    ``MEGANModel`` and the BVOC response helper.
    """
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eq. (3) and
    #   Table 4 (light-dependent fraction LDF of each emission class).
    # Mathematical expression: gamma_P = (1 - LDF) + LDF * gamma_P_LDF, with gamma_P_LDF
    #   the Eq. (4) activity evaluated by megan_light_dependent_activity.
    # Parameter meanings: LDF is the light-dependent fraction of the compound class;
    #   PPFD, alpha and C_L are those of megan_light_dependent_activity.
    # Units: PPFD [umol photons m-2 s-1]; alpha [m2 s umol-1]; C_L, LDF, gamma_P [1].
    # Domain/constraints: LDF in [0, 1] by its ParameterManager bounds; PPFD finite and
    #   non-negative (validated by the callers); no clamp is applied.
    # Sign convention: gamma_P is a non-negative multiplier of the emission magnitude.
    return (1.0 - ldf) + ldf * megan_light_dependent_activity(ppfd, alpha, cl)


def megan_ldf_temperature_activity(
    temperature: Tensor,
    eopt: Tensor,
    topt: Tensor,
    ct1: Tensor,
    ct2: Tensor,
    gas_constant: Tensor,
) -> Tensor:
    """Light-dependent MEGAN temperature activity factor gamma_T_LDF."""
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eq. (8), with
    #   E_opt and T_opt of Eqs. (9)-(10) at the fixed standard history or from the
    #   carried T24/T240 history (megan_optimum_history).
    #   Goldberg (1991), ACM Computing Surveys 23, 5-48: avoid cancellation of rounded
    #   reciprocals by evaluating the same expression before reciprocal rounding.
    # Mathematical expression: x = (T - T_opt) / (T T_opt R) = (1/T_opt - 1/T) / R and
    #   gamma_T_LDF = E_opt C_T2 exp(C_T1 x) / (C_T2 - C_T1 (1 - exp(C_T2 x))).
    # Parameter meanings: T is leaf temperature, T_opt and E_opt the optimum temperature
    #   and activity, C_T1 and C_T2 the activation and deactivation energies, R the molar
    #   gas constant; x is the inverse-temperature difference scaled by R.
    # Units: T, T_opt [K]; x [mol J-1]; C_T1, C_T2 [J mol-1]; R [J mol-1 K-1];
    #   E_opt, gamma_T_LDF [1].
    # Domain/constraints: T must be finite and positive (validated by the callers); a
    #   non-positive denominator raises ValueError; no clamp is applied.
    # Sign convention: gamma_T_LDF is a non-negative multiplier of the emission magnitude.
    x = (temperature - topt) / (temperature * topt * gas_constant)
    denominator = ct2 - ct1 * (1.0 - torch.exp(ct2 * x))
    if torch.any(denominator <= 0.0):
        raise ValueError("MEGAN2.1 Eq. (8) temperature response denominator is non-positive")
    return eopt * ct2 * torch.exp(ct1 * x) / denominator


def megan_lif_temperature_activity(
    temperature: Tensor,
    beta: Tensor,
    standard_temperature: Tensor,
) -> Tensor:
    """Light-independent MEGAN temperature activity factor gamma_T_LIF."""
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eq. (11), the
    #   exponential response of Guenther et al. (1993).
    # Mathematical expression: gamma_T_LIF = exp(beta (T - T_s)).
    # Parameter meanings: T is leaf temperature, beta the compound-class temperature
    #   coefficient and T_s the standard temperature.
    # Units: T, T_s [K]; beta [K-1]; gamma_T_LIF [1].
    # Domain/constraints: T must be finite (validated by the callers); no clamp is applied.
    #   T_s must equal the reference temperature of the emission factor that gamma_T_LIF
    #   multiplies: RELIC pairs 303.15 K (30 degC) with its dry-mass emission factors,
    #   whereas the MEGAN2.1 T_s = 297 K of Guenther et al. (2012) Eqs. (9)-(11) belongs to
    #   the MEGAN2.1 area-based emission factors; the two pairs must not be mixed.
    # Sign convention: gamma_T_LIF is a positive multiplier of the emission magnitude.
    return torch.exp(beta * (temperature - standard_temperature))


def megan_optimum_history(
    temperature_24h: Tensor,
    temperature_240h: Tensor,
    topt_standard: Tensor,
    history_standard_temperature: Tensor,
    topt_slope: Tensor,
    eopt_coefficient_24h: Tensor,
    eopt_coefficient_240h: Tensor,
) -> Tuple[Tensor, Tensor]:
    """T_opt and the class-independent E_opt factor of the MEGAN2.1 temperature history.

    Returns ``(T_opt, h)`` with E_opt,i = C_eo,i h for every Table 4 row i.
    """
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, p. 1476,
    #   Eqs. (9)-(10) (T_s = 297 K; T24 and T240 the average leaf temperatures of the
    #   past 24 h and 240 h).
    # Mathematical expression: T_opt = T_opt,s + a (T240 - T_s,h) (Eq. 9);
    #   h = exp(c24 (T24 - T_s,h)) exp(c240 (T240 - T_s,h)), so that
    #   E_opt,i = C_eo,i h (Eq. 10).
    # Parameter meanings: T24 and T240 are the 24 h and 240 h temperature means of
    #   the temperature MEGAN reads, T_opt,s = 313 K the Eq. (9) intercept (the T_opt
    #   of the standard history), T_s,h = 297 K the standard leaf temperature of
    #   Eqs. (9)-(10), a = 0.6 the Eq. (9) slope, c24 = c240 = 0.05 K-1 the Eq. (10)
    #   coefficients.
    # Units: T24, T240, T_opt,s, T_s,h, T_opt [K]; a, h [1]; c24, c240 [K-1].
    # Domain/constraints: finite T24 and T240 (validated by the caller); at
    #   T24 = T240 = T_s,h the result is exactly (T_opt,s, 1), the fixed standard
    #   history; no clamp is applied.
    # Sign convention: a warmer history raises T_opt and h; h is a positive multiplier.
    topt = topt_standard + topt_slope * (temperature_240h - history_standard_temperature)
    factor = torch.exp(
        eopt_coefficient_24h * (temperature_24h - history_standard_temperature)
    ) * torch.exp(
        eopt_coefficient_240h * (temperature_240h - history_standard_temperature)
    )
    return topt, factor


def megan_temperature_history_update(
    previous_mean: Tensor,
    temperature: Tensor,
    dt: Tensor,
    time_scale: Tensor,
) -> Tensor:
    """Advance one MEGAN2.1 running temperature mean (T24 or T240) by one step dt."""
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, p. 1476 (T24
    #   and T240 the average leaf temperatures of the past 24 h and 240 h); the running
    #   mean (accumulation type "runmean") of the Community Land Model that carries
    #   t_veg24 and t_veg240 for MEGAN2.1 (Oleson et al. 2010, NCAR/TN-478+STR, CLM4
    #   accumulMod: accum = ((N - 1) accum + field) / N, N = period / dt).
    # Mathematical expression: X_new = (1 - w) X + w T with w = min(dt / tau, 1), the
    #   explicit-Euler step of dX/dt = (T - X) / tau and, for N = tau / dt >= 1, the
    #   CLM form ((N - 1) X + T) / N.
    # Parameter meanings: X is the running mean at the start of the step, T the
    #   temperature of the step, dt the step length, tau the averaging time scale
    #   (24 h for T24, 240 h for T240) and w the weight of the step.
    # Units: X, T, X_new [K]; dt, tau [s]; w [1].
    # Domain/constraints: dt >= 0 finite and tau > 0 (validated by the caller); w is
    #   limited to 1, so a step at least as long as tau gives X_new = T exactly (at
    #   dt = 86400 s, T24 becomes the mean of the step just completed); X_new lies
    #   between X and T.
    # Sign convention: none; X relaxes towards T.
    weight = torch.clamp(dt / time_scale, max=1.0)
    return (1.0 - weight) * previous_mean + weight * temperature


@dataclass(frozen=True)
class MeganTable4Row:
    """ParameterManager keys (namespace ``biosphere.bvoc.megan``) of the Guenther et al.
    (2012) Table 4 row that parameterises the temperature activity of a compound class.

    ``ldf`` is None for a fully light-dependent row (LDF = 1): its Eq. (7) reduces to
    Eq. (8) and needs no Eq. (11) coefficient ``beta``. ``ct1`` and ``eopt`` are the
    Eq. (8) C_T1 and C_eo (E_opt at the fixed standard history T24 = T240 = T_s).
    ``global_emission`` is the Table 6 global emission of a component of a lumped class,
    whose activity factors are emission-weighted mixtures of the component factors.
    """

    row: str
    ldf: Optional[str]
    beta: Optional[str]
    ct1: str
    eopt: str
    global_emission: Optional[str] = None


def _ovoc_component(name: str, row: str) -> MeganTable4Row:
    prefix = f"ovoc.{name}"
    return MeganTable4Row(
        row,
        f"{prefix}.ldf",
        f"{prefix}.beta",
        f"{prefix}.ct1",
        f"{prefix}.eopt",
        f"{prefix}.global_emission",
    )


# Table 4 rows behind each compound class; every coefficient of a row comes from that one
# row. T_opt (Eq. 9) and C_T2 are class independent (biosphere.bvoc.megan.topt
# and biosphere.bvoc.megan.ct2). The lumped OVOC class is methanol, acetone and acetaldehyde (the
# bidirectional VOC class of Guenther et al. 2012 Table 1), weighted by their Table 6
# global emissions.
MEGAN_TABLE4_ROWS: Dict[str, Tuple[MeganTable4Row, ...]] = {
    "isoprene": (MeganTable4Row("Isoprene", None, None, "ct1_isoprene", "eopt_isoprene"),),
    "monoterpenes": (
        MeganTable4Row(
            "Other Monoterpenes",
            "ldf_monoterpenes",
            "beta_monoterpenes",
            "ct1_monoterpenes",
            "eopt_monoterpenes",
        ),
    ),
    "sesquiterpenes": (
        MeganTable4Row(
            "Other Sesquiterpenes",
            "ldf_sesquiterpenes",
            "beta_sesquiterpenes",
            "ct1_sesquiterpenes",
            "eopt_sesquiterpenes",
        ),
    ),
    "ovocs": (
        _ovoc_component("methanol", "Methanol"),
        _ovoc_component("acetone", "Acetone"),
        _ovoc_component("acetaldehyde", "Bidirectional VOC"),
    ),
}

# lookup(key under biosphere.bvoc.megan, units) -> coefficient in the driver dtype/device.
MeganLookup = Callable[[str, str], Tensor]


def megan_temperature_activity(
    temperature: Tensor,
    ldf: Optional[Tensor],
    beta: Optional[Tensor],
    standard_temperature: Optional[Tensor],
    eopt: Tensor,
    topt: Tensor,
    ct1: Tensor,
    ct2: Tensor,
    gas_constant: Tensor,
    eopt_history_factor: Optional[Tensor] = None,
) -> Tensor:
    """MEGAN2.1 temperature activity factor gamma_T,i, Guenther et al. (2012) Eq. (7).

    This is the single production implementation shared by ``MEGANModel`` and the
    BVOC response helper. ``ldf=None`` marks a fully light-dependent class (LDF = 1).
    """
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eq. (7) with
    #   Eqs. (8) and (11) and the coefficients of one Table 4 row.
    # Mathematical expression: gamma_T = (1 - LDF) gamma_T_LIF + LDF gamma_T_LDF, with
    #   gamma_T_LIF = exp(beta (T - T_s)) (megan_lif_temperature_activity) and
    #   gamma_T_LDF = E_opt C_T2 exp(C_T1 x) / (C_T2 - C_T1 (1 - exp(C_T2 x))),
    #   x = (1/T_opt - 1/T) / R (megan_ldf_temperature_activity); gamma_T = gamma_T_LDF
    #   when LDF = 1.
    # Parameter meanings: LDF is the light-dependent fraction, beta the Eq. (11)
    #   temperature coefficient, T_s the standard temperature of the emission factor,
    #   E_opt the optimum activity at the fixed standard history (Table 4 C_eo), times
    #   the Eq. (10) history factor h of megan_optimum_history when one is given, T_opt
    #   the optimum temperature, C_T1 and C_T2 the activation and deactivation energies
    #   and R the molar gas constant.
    # Units: T, T_s, T_opt [K]; beta [K-1]; C_T1, C_T2 [J mol-1]; R [J mol-1 K-1];
    #   LDF, E_opt, gamma_T [1].
    # Domain/constraints: T finite and positive (validated by the callers); LDF in
    #   [0, 1] by its ParameterManager bounds; beta and T_s are required when LDF is
    #   given (ValueError otherwise); a non-positive Eq. (8) denominator raises; no clamp
    #   is applied.
    # Sign convention: gamma_T is a non-negative multiplier of the emission magnitude.
    if eopt_history_factor is not None:
        eopt = eopt * eopt_history_factor
    light_dependent = megan_ldf_temperature_activity(
        temperature, eopt, topt, ct1, ct2, gas_constant
    )
    if ldf is None:
        return light_dependent
    if beta is None or standard_temperature is None:
        raise ValueError(
            "A partly light-dependent MEGAN2.1 class needs its Eq. (11) beta and T_s"
        )
    light_independent = megan_lif_temperature_activity(
        temperature, beta, standard_temperature
    )
    return (1.0 - ldf) * light_independent + ldf * light_dependent


def _megan_class_rows(compound: str) -> Tuple[MeganTable4Row, ...]:
    if compound not in MEGAN_TABLE4_ROWS:
        raise ValueError(
            "No MEGAN2.1 Table 4 row (light-dependent fraction and temperature "
            f"coefficients) is registered for compound class '{compound}'; registered "
            f"classes: {sorted(MEGAN_TABLE4_ROWS)}."
        )
    return MEGAN_TABLE4_ROWS[compound]


def _megan_mixture(
    rows: Tuple[MeganTable4Row, ...],
    activities: List[Tensor],
    lookup: MeganLookup,
) -> Tensor:
    """Emission-weighted mixture sum_j w_j a_j of the row activities (one row: a_1)."""
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Table 6 (global
    #   annual emissions for 2000) and Table 1 (compound classes).
    # Mathematical expression: a_class = sum_j w_j a_j, w_j = E_j / sum_k E_k; a_class = a_1
    #   for a single-row class.
    # Parameter meanings: a_j is an activity factor of component j, E_j its global annual
    #   emission and w_j its emission share at standard conditions.
    # Units: E_j [g yr-1]; w_j, a_j, a_class [1].
    # Domain/constraints: E_j > 0 by the ParameterManager bounds, so sum_j w_j = 1; no
    #   clamp is applied.
    # Sign convention: a_class is a non-negative multiplier of the emission magnitude.
    if len(rows) == 1:
        return activities[0]
    emissions = [lookup(cast(str, row.global_emission), "g yr-1") for row in rows]
    total = emissions[0]
    for emission in emissions[1:]:
        total = total + emission
    mixture = emissions[0] / total * activities[0]
    for emission, activity in zip(emissions[1:], activities[1:]):
        mixture = mixture + emission / total * activity
    return mixture


def megan_class_light_activity(
    compound: str,
    ppfd: Tensor,
    alpha: Tensor,
    cl: Tensor,
    lookup: MeganLookup,
) -> Tensor:
    """Light activity gamma_P of a compound class (shared by MEGANModel and the helper).

    Eq. (4) for a fully light-dependent row, Eq. (3) otherwise; a lumped class takes the
    emission-weighted mixture of its component rows.
    """
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (3)-(4) and
    #   Tables 4 and 6.
    # Mathematical expression: gamma_P,j = (1 - LDF_j) + LDF_j gamma_P_LDF (gamma_P_LDF when
    #   LDF_j = 1); gamma_P = sum_j w_j gamma_P,j.
    # Parameter meanings: LDF_j is the light-dependent fraction of Table 4 row j, w_j the
    #   Table 6 emission share of component j; alpha and C_L as in Eq. (4).
    # Units: PPFD [umol photons m-2 s-1]; alpha [m2 s umol-1]; C_L, LDF_j, w_j, gamma_P [1].
    # Domain/constraints: finite PPFD >= 0 (validated by the callers); an unregistered
    #   class raises ValueError; no clamp is applied.
    # Sign convention: gamma_P is a non-negative multiplier of the emission magnitude.
    rows = _megan_class_rows(compound)
    activities = [
        megan_light_dependent_activity(ppfd, alpha, cl)
        if row.ldf is None
        else megan_light_activity(ppfd, alpha, cl, lookup(row.ldf, "dimensionless"))
        for row in rows
    ]
    return _megan_mixture(rows, activities, lookup)


def megan_class_temperature_activity(
    compound: str,
    temperature: Tensor,
    standard_temperature: Tensor,
    topt: Tensor,
    ct2: Tensor,
    gas_constant: Tensor,
    lookup: MeganLookup,
    eopt_history_factor: Optional[Tensor] = None,
) -> Tensor:
    """Temperature activity gamma_T of a compound class (shared by MEGANModel and the helper).

    Eq. (7) of each Table 4 row through megan_temperature_activity; a lumped class takes
    the emission-weighted mixture of its component rows.
    """
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (7), (8),
    #   (11) and Tables 4 and 6.
    # Mathematical expression: gamma_T,j = (1 - LDF_j) exp(beta_j (T - T_s)) + LDF_j E_opt,j
    #   C_T2 exp(C_T1,j x) / (C_T2 - C_T1,j (1 - exp(C_T2 x))), x = (1/T_opt - 1/T)/R;
    #   gamma_T = sum_j w_j gamma_T,j.
    # Parameter meanings: beta_j, LDF_j, C_T1,j and E_opt,j (Table 4 C_eo) come from row j;
    #   T_opt and C_T2 are class independent; w_j is the Table 6 emission share.
    # Units: T, T_s, T_opt [K]; beta_j [K-1]; C_T1,j, C_T2 [J mol-1]; R [J mol-1 K-1];
    #   LDF_j, E_opt,j, w_j, gamma_T [1].
    # Domain/constraints: finite T > 0 (validated by the callers); an unregistered class
    #   raises ValueError; the Eq. (8) denominator must be positive; no clamp is applied.
    # Sign convention: gamma_T is a non-negative multiplier of the emission magnitude.
    rows = _megan_class_rows(compound)
    activities = []
    for row in rows:
        activities.append(
            megan_temperature_activity(
                temperature,
                None if row.ldf is None else lookup(row.ldf, "dimensionless"),
                None if row.beta is None else lookup(row.beta, "K-1"),
                standard_temperature,
                lookup(row.eopt, "dimensionless"),
                topt,
                lookup(row.ct1, "J mol-1"),
                ct2,
                gas_constant,
                eopt_history_factor=eopt_history_factor,
            )
        )
    return _megan_mixture(rows, activities, lookup)


class CouplingStrategy(Enum):
    """BVOC-photosynthesis coupling strategies."""
    SIMULTANEOUS = "simultaneous"
    SEQUENTIAL = "sequential"
    INDEPENDENT = "independent"


class EmissionPathway(Enum):
    """BVOC emission pathways."""
    STANDARD = "standard"          # Reduced dry-mass emission-potential closure
    PHOTOSYNTHESIS = "photosynthesis"  # Photosynthesis-coupled emissions
    HYBRID = "hybrid"              # Combination of both


class MEGANModel(BiologicalProcessFunctor):
    """BiologicalProcess bridge for a reduced instantaneous MEGAN closure.

    The standard pathway evaluates compound emission potentials with explicit
    temperature, PPFD, LAI-derived foliar mass, CO2, and optional upstream stress
    factors. The photosynthesis pathway is an alternative empirical carbon
    allocation closure. No pathway silently substitutes for another.
    """
    SUPPORTED_COMPOUNDS = (
        "isoprene",
        "monoterpenes",
        "sesquiterpenes",
        "ovocs",
    )
    # Declared meaning and [database units] of every ParameterManager key and physical
    # constant this closure reads (the PFT emission factors are added per instance).
    PARAMETER_TEXTS = {
        "biosphere.bvoc.megan.hybrid_weight": (
            "Weight blending the photosynthesis-coupled and empirical emission "
            "pathways [dimensionless]"
        ),
        "biosphere.bvoc.carbon_fraction.isoprene": (
            "Carbon mass fraction of isoprene (C5H8) [dimensionless]"
        ),
        "biosphere.bvoc.carbon_fraction.monoterpenes": (
            "Carbon mass fraction of monoterpenes (C10H16) [dimensionless]"
        ),
        "biosphere.bvoc.carbon_fraction.sesquiterpenes": (
            "Carbon mass fraction of sesquiterpenes (C15H24) [dimensionless]"
        ),
        "biosphere.bvoc.carbon_fraction.ovocs": (
            "Carbon mass fraction of the oxygenated VOC mixture [dimensionless]"
        ),
        "biosphere.bvoc.validation.temperature.min_operational": (
            "Lowest valid canopy temperature of the MEGAN inputs [K]"
        ),
        "biosphere.bvoc.validation.temperature.max_operational": (
            "Highest valid canopy temperature of the MEGAN inputs [K]"
        ),
        "biosphere.bvoc.emission_factors.isoprene": (
            "Isoprene emission factor at standard conditions [ug g-1 DW h-1]"
        ),
        "biosphere.bvoc.emission_factors.monoterpenes": (
            "Monoterpene emission factor at standard conditions [ug g-1 DW h-1]"
        ),
        "biosphere.bvoc.emission_factors.sesquiterpenes": (
            "Sesquiterpene emission factor at standard conditions [ug g-1 DW h-1]"
        ),
        "biosphere.bvoc.emission_factors.ovocs": (
            "Oxygenated VOC emission factor at standard conditions [ug g-1 DW h-1]"
        ),
        "biosphere.bvoc.megan.alpha_light": (
            "PPFD response slope alpha of the MEGAN2.1 light activity [m2 s umol-1]"
        ),
        "biosphere.bvoc.megan.cl_light": (
            "Light scaling coefficient C_L of the MEGAN2.1 light activity [dimensionless]"
        ),
        "biosphere.bvoc.megan.ldf_monoterpenes": (
            "Light-dependent fraction LDF of the monoterpene class [dimensionless]"
        ),
        "biosphere.bvoc.megan.ldf_sesquiterpenes": (
            "Light-dependent fraction LDF of sesquiterpenes [dimensionless]"
        ),
        "biosphere.bvoc.megan.topt": (
            "Optimum temperature T_opt of the Eq. (8) temperature activity, Eq. (9) at the "
            "standard history T240 = T_s (313 K); class independent [K]"
        ),
        "biosphere.bvoc.t_standard": (
            "Standard temperature T_s of the light-independent temperature response [K]"
        ),
        "biosphere.bvoc.megan.eopt_isoprene": (
            "Optimum activity E_opt (Table 4 C_eo) of the isoprene Eq. (8) temperature "
            "activity at the fixed standard history T24 = T240 = T_s [dimensionless]"
        ),
        "biosphere.bvoc.megan.ct1_isoprene": (
            "Activation energy C_T1 of the isoprene Eq. (8) temperature activity [J mol-1]"
        ),
        "biosphere.bvoc.megan.ct2": (
            "Deactivation energy C_T2 of the Eq. (8) temperature activity, class "
            "independent [J mol-1]"
        ),
        "biosphere.bvoc.megan.ct1_monoterpenes": (
            "Activation energy C_T1 of the monoterpene Eq. (8) temperature activity [J mol-1]"
        ),
        "biosphere.bvoc.megan.eopt_monoterpenes": (
            "Optimum activity E_opt (Table 4 C_eo) of the monoterpene Eq. (8) temperature "
            "activity [dimensionless]"
        ),
        "biosphere.bvoc.megan.ct1_sesquiterpenes": (
            "Activation energy C_T1 of the sesquiterpene Eq. (8) temperature activity "
            "[J mol-1]"
        ),
        "biosphere.bvoc.megan.eopt_sesquiterpenes": (
            "Optimum activity E_opt (Table 4 C_eo) of the sesquiterpene Eq. (8) "
            "temperature activity [dimensionless]"
        ),
        "biosphere.bvoc.photosynthetic_coupling.f_carbon_isoprene": (
            "Fraction of gross photosynthesis allocated to isoprene [dimensionless]"
        ),
        "biosphere.bvoc.photosynthetic_coupling.f_carbon_monoterpenes": (
            "Fraction of gross photosynthesis allocated to monoterpenes [dimensionless]"
        ),
        "biosphere.bvoc.photosynthetic_coupling.f_carbon_sesquiterpenes": (
            "Fraction of gross photosynthesis allocated to sesquiterpenes [dimensionless]"
        ),
        "biosphere.bvoc.photosynthetic_coupling.f_carbon_ovocs": (
            "Fraction of gross photosynthesis allocated to oxygenated VOCs [dimensionless]"
        ),
        "relic.physical.constants.molecular_weights.isoprene": (
            "Molecular weight of isoprene (C5H8) [g mol-1]"
        ),
        "biosphere.bvoc.molecular_weights.monoterpenes": (
            "Mean molecular weight of monoterpenes (C10H16) [g mol-1]"
        ),
        "biosphere.bvoc.molecular_weights.sesquiterpenes": (
            "Mean molecular weight of sesquiterpenes (C15H24) [g mol-1]"
        ),
        "biosphere.bvoc.molecular_weights.ovocs": (
            "Mean molecular weight of the oxygenated VOC mixture [g mol-1]"
        ),
        "biosphere.bvoc.carbon_atoms.isoprene": (
            "Carbon atoms per isoprene molecule [dimensionless]"
        ),
        "biosphere.bvoc.carbon_atoms.monoterpenes": (
            "Carbon atoms per monoterpene molecule [dimensionless]"
        ),
        "biosphere.bvoc.carbon_atoms.sesquiterpenes": (
            "Carbon atoms per sesquiterpene molecule [dimensionless]"
        ),
        "biosphere.bvoc.carbon_atoms.ovocs": (
            "Mean carbon atoms per oxygenated VOC molecule [dimensionless]"
        ),
        "biosphere.bvoc.foliar_density_factor": (
            "Dry-leaf mass per unit leaf area of the emission scaling [g DW m-2]"
        ),
        "biosphere.bvoc.megan.beta_monoterpenes": (
            "Temperature coefficient beta of the monoterpene Eq. (11) activity [K-1]"
        ),
        "biosphere.bvoc.megan.beta_sesquiterpenes": (
            "Temperature coefficient beta of the sesquiterpene Eq. (11) activity [K-1]"
        ),
        "biosphere.bvoc.megan.ovoc.methanol.global_emission": (
            "Table 6 global emission of the OVOC component methanol [g yr-1]"
        ),
        "biosphere.bvoc.megan.ovoc.methanol.beta": (
            "Temperature coefficient beta of the OVOC component methanol [K-1]"
        ),
        "biosphere.bvoc.megan.ovoc.methanol.ldf": (
            "Light-dependent fraction LDF of the OVOC component methanol [dimensionless]"
        ),
        "biosphere.bvoc.megan.ovoc.methanol.ct1": (
            "Activation energy C_T1 of the OVOC component methanol [J mol-1]"
        ),
        "biosphere.bvoc.megan.ovoc.methanol.eopt": (
            "Optimum activity E_opt (Table 4 C_eo) of the OVOC component methanol "
            "[dimensionless]"
        ),
        "biosphere.bvoc.megan.ovoc.acetone.global_emission": (
            "Table 6 global emission of the OVOC component acetone [g yr-1]"
        ),
        "biosphere.bvoc.megan.ovoc.acetone.beta": (
            "Temperature coefficient beta of the OVOC component acetone [K-1]"
        ),
        "biosphere.bvoc.megan.ovoc.acetone.ldf": (
            "Light-dependent fraction LDF of the OVOC component acetone [dimensionless]"
        ),
        "biosphere.bvoc.megan.ovoc.acetone.ct1": (
            "Activation energy C_T1 of the OVOC component acetone [J mol-1]"
        ),
        "biosphere.bvoc.megan.ovoc.acetone.eopt": (
            "Optimum activity E_opt (Table 4 C_eo) of the OVOC component acetone "
            "[dimensionless]"
        ),
        "biosphere.bvoc.megan.ovoc.acetaldehyde.global_emission": (
            "Table 6 global emission of the OVOC component acetaldehyde [g yr-1]"
        ),
        "biosphere.bvoc.megan.ovoc.acetaldehyde.beta": (
            "Temperature coefficient beta of the OVOC component acetaldehyde [K-1]"
        ),
        "biosphere.bvoc.megan.ovoc.acetaldehyde.ldf": (
            "Light-dependent fraction LDF of the OVOC component acetaldehyde [dimensionless]"
        ),
        "biosphere.bvoc.megan.ovoc.acetaldehyde.ct1": (
            "Activation energy C_T1 of the OVOC component acetaldehyde [J mol-1]"
        ),
        "biosphere.bvoc.megan.ovoc.acetaldehyde.eopt": (
            "Optimum activity E_opt (Table 4 C_eo) of the OVOC component acetaldehyde "
            "[dimensionless]"
        ),
        "biosphere.bvoc.co2_inhibition.ismax": (
            "Maximum isoprene CO2 activity factor I_smax [dimensionless]"
        ),
        "biosphere.bvoc.co2_inhibition.c_star": (
            "CO2 scaling concentration C* of the isoprene CO2 response [ppm]"
        ),
        "biosphere.bvoc.co2_inhibition.hill_exponent": (
            "Hill exponent h of the isoprene CO2 response [dimensionless]"
        ),
        "biosphere.bvoc.co2_inhibition.intercellular_to_ambient_ratio": (
            "Intercellular to ambient CO2 ratio of the CO2 response [dimensionless]"
        ),
        "validation.numerical.carbon_flux_min": (
            "Lower bound of the coupled carbon flux [umol m-2 s-1]"
        ),
        "validation.numerical.epsilon": (
            "Denominator floor of the emission normalization [dimensionless]"
        ),
        "relic.physical.constants.conversions.micrograms_to_grams": (
            "Micrograms to grams conversion [g ug-1]"
        ),
        "relic.physical.constants.universal.gas_constant": "Universal gas constant [J mol-1 K-1]",
        "relic.physical.constants.time.seconds_per_hour": "Seconds per hour [s]",
        "validation.conservation.conservation_tolerance": (
            "Tolerance of the direct-forward carbon-boundary check [dimensionless]"
        ),
        "validation.conservation.strict_conservation": (
            "Switch that makes a direct-forward conservation violation raise [dimensionless]"
        ),
    }
    PFT_EMISSION_FACTOR_KEYS = {
        "isoprene": "ef_isoprene",
        "monoterpenes": "ef_monoterpene",
        "sesquiterpenes": "ef_sesquiterpene",
        "ovocs": "ef_other_voc",
    }
    PHOTOSYNTHESIS_CARBON_FLOOR_RESIDUAL = (
        "diagnostics.bvoc.megan.photosynthesis_carbon_floor_residual"
    )
    # Stress controls of f_stress in their product order: label of the
    # stress_factors selection -> (state variable [1], provider Functor name).
    # References: Salvucci and Crafts-Brandner (2004) heat and Hatfield and
    #   Prueger (2015) cold thresholds, as implemented by ThermalStress
    #   (relic/scientific/biosphere/stress/thermal_stress.py); no law is added here.
    # Mathematical expression: heat f_hot = max(0, 1 + k_hot (T - T_hot)) for
    #   T > T_hot, 0 for T > T_crit, else 1; cold f_cold = max(0, 1 + k_cold
    #   (T_cold - T)) for T < T_cold, 0 for T < T_freeze, else 1.
    # Parameter meanings: owner biosphere.stress.thermal, T_hot heat threshold
    #   (308.15 K), k_hot heat slope (-0.1 K-1), T_crit critical high temperature
    #   (318.15 K), T_cold cold threshold (278.15 K), k_cold cold slope
    #   (-0.15 K-1), T_freeze freezing damage temperature (271.15 K); T is the air
    #   temperature the provider instance reads.
    # Units: T and thresholds [K]; slopes [K-1]; f_hot, f_cold [1].
    # Domain/constraints: heat and cold are read as declared controls in [0, 1];
    #   when heat reads the daily maximum and cold the daily minimum air
    #   temperature they come from two ThermalStress instances, so the cold
    #   provider is the second instance thermal_stress_cold (graph node names
    #   are unique).
    # Sign convention: 1 is unstressed; a smaller factor lowers the emission.
    STRESS_FACTORS = {
        "water": ("water_stress_factor", "water_stress"),
        "temperature": ("temperature_stress_factor", "thermal_stress"),
        "ozone": ("ozone_stress_factor", "ozone_stress"),
        "heat": ("heat_stress_factor", "thermal_stress"),
        "cold": ("cold_stress_factor", "thermal_stress_cold"),
    }
    # Controls declared by use_stress_functors without a stress_factors key.
    DEFAULT_STRESS_FACTORS = ("water", "temperature", "ozone")
    # Declared state of the temperature_history option: T24 and T240 of Guenther et al.
    #   (2012) Eqs. (9)-(10) [K], read at the start of the call (a {state: ...} binding)
    #   and replaced after it by megan_temperature_history_update; the owner of both
    #   variables is this functor (relic/contracts/variables/biosphere).
    TEMPERATURE_HISTORY_STATES = (
        "air_temperature_24h_mean",
        "air_temperature_240h_mean",
    )
    # Declared temperature input T of gamma_T (Eqs. 7-8 and 11) and, with
    # temperature_history, of T24 and T240; config["temperature_source"] selects it and an
    # absent key keeps "air_temperature" (big-leaf T_leaf = T_air, the former interface):
    #   "air_temperature": the near-surface air temperature forcing [K];
    #   "leaf_temperature": the leaf temperature [K] of the leaf energy balance
    #     (relic.scientific.biosphere.photosynthesis.leaf_energy_balance), the leaf
    #     temperature that MEGAN2.1 defines its response and history on (Guenther et al.
    #     2012, Sect. 2.2); the history states are then LEAF_TEMPERATURE_HISTORY_STATES.
    # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (7)-(11).
    # Units: T, T24, T240 [K].
    # Domain/constraints: one source per instance; a graph binds only same-named ports, so
    #   the selected name is the declared input.
    TEMPERATURE_SOURCES = ("air_temperature", "leaf_temperature")
    # Declared temporal supports; config["temporal_support"], and an absent key keeps
    #   "instantaneous" (the activity factors once on the supplied inputs):
    #   "daylight_gauss": daily step only; gamma_P gamma_T (and the whole emission law)
    #     at the daylight Gauss samples of relic.scientific.atmosphere.daylight_quadrature
    #     and averaged with the sample weights (the night sample keeps the
    #     light-independent fraction): ppfd is the 24-hour mean redistributed by the solar
    #     elevation; the temperature is the SWAT+ Eq. 1:1.3.1 course of the daily mean,
    #     maximum and minimum air temperature, or the per-sample leaf temperature
    #     leaf_temperature_daylight_samples (temperature_source leaf_temperature); the
    #     net photosynthesis (hybrid and photosynthesis pathways) and the ozone factor are
    #     read per sample (DAYLIGHT_SAMPLE_INPUTS); T24 and T240 advance with the daily
    #     mean temperature (the air input, or the weighted mean of the leaf samples).
    TEMPORAL_SUPPORTS = ("instantaneous", daylight_quadrature.DAYLIGHT_GAUSS)
    DAYLIGHT_INPUTS = ("latitude", "day_of_year")
    DAYLIGHT_AIR_TEMPERATURE_INPUTS = ("air_temperature_max", "air_temperature_min")
    DAYLIGHT_SAMPLE_INPUTS = {
        "net_photosynthesis": "net_photosynthesis_daylight_samples",
        "ozone_stress_factor": "ozone_stress_factor_daylight_samples",
        "leaf_temperature": "leaf_temperature_daylight_samples",
    }
    LEAF_TEMPERATURE_HISTORY_STATES = (
        "leaf_temperature_24h_mean",
        "leaf_temperature_240h_mean",
    )
    TEMPERATURE_HISTORY_DIAGNOSTICS = (
        "diagnostics.bvoc.megan.optimum_temperature",
        "diagnostics.bvoc.megan.optimum_activity_history_factor",
    )
    TEMPERATURE_HISTORY_PARAMETER_TEXTS = {
        "biosphere.bvoc.megan.history_standard_temperature": (
            "Standard leaf temperature T_s of the Eq. (9)-(10) temperature history [K]"
        ),
        "biosphere.bvoc.megan.topt_history_slope": (
            "Slope of T_opt on T240 in Eq. (9) [dimensionless]"
        ),
        "biosphere.bvoc.megan.eopt_history_coefficient_24h": (
            "Coefficient of T24 in the Eq. (10) optimum activity [K-1]"
        ),
        "biosphere.bvoc.megan.eopt_history_coefficient_240h": (
            "Coefficient of T240 in the Eq. (10) optimum activity [K-1]"
        ),
        "biosphere.bvoc.megan.history_time_scale_24h": (
            "Averaging time scale of the running mean T24 [s]"
        ),
        "biosphere.bvoc.megan.history_time_scale_240h": (
            "Averaging time scale of the running mean T240 [s]"
        ),
    }

    CANONICAL_OUTPUTS = (
        "isoprene_emission",
        "monoterpenes_emission",
        "sesquiterpenes_emission",
        "ovocs_emission",
        "total_bvoc_emission",
        "bvoc_carbon_allocation_flux",
        "diagnostics.bvoc.megan.environmental_stress_factor",
        PHOTOSYNTHESIS_CARBON_FLOOR_RESIDUAL,
        "diagnostics.bvoc.megan.co2_inhibition_factor",
        "diagnostics.bvoc.megan.isoprene.gamma_temperature",
        "diagnostics.bvoc.megan.monoterpenes.gamma_temperature",
        "diagnostics.bvoc.megan.sesquiterpenes.gamma_temperature",
        "diagnostics.bvoc.megan.ovocs.gamma_temperature",
        "diagnostics.bvoc.megan.isoprene.gamma_light",
        "diagnostics.bvoc.megan.monoterpenes.gamma_light",
        "diagnostics.bvoc.megan.sesquiterpenes.gamma_light",
        "diagnostics.bvoc.megan.ovocs.gamma_light",
    )
    
    def __init__(
        self,
        name: str = "megan_bvoc_model",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        coupling_strategy: CouplingStrategy = CouplingStrategy.SEQUENTIAL,
        emission_pathway: EmissionPathway = EmissionPathway.HYBRID,
        validate_inputs: bool = True
    ):
        """
        Initialize unified MEGAN model.
        
        Parameters
        ----------
        name : str, optional
            Functor name for identification
        config : Dict[str, Any], optional
            Configuration parameters
        device : torch.device, optional
            Computation device
        coupling_strategy : CouplingStrategy, optional
            How to couple with photosynthesis
        emission_pathway : EmissionPathway, optional
            Primary emission calculation pathway
        validate_inputs : bool, optional
            Enable input validation
        
        Notes
        -----
        Scientific parameters are sourced from the ParameterManager/YAML database.
        The config dict is reserved for feature toggles (e.g., enabled compounds,
        coupling modes) and is not used to override emission factors or response
        coefficients, preventing silent hardcoding. use_stress_functors enables
        the stress product; stress_factors (a non-empty subset of water,
        temperature, ozone, heat and cold) declares which controls the
        composition supplies, and water, temperature and ozone are declared
        without it.
        """
        conf = dict(config or {})
        supported_compounds = list(self.SUPPORTED_COMPOUNDS)
        configured_compounds = conf.get("enabled_compounds", supported_compounds)
        if not isinstance(configured_compounds, (list, tuple)):
            raise TypeError("enabled_compounds must be a non-empty list or tuple")
        enabled_compounds = list(configured_compounds)
        if not enabled_compounds:
            raise ValueError("enabled_compounds must not be empty")
        if len(set(enabled_compounds)) != len(enabled_compounds):
            raise ValueError("enabled_compounds must not contain duplicates")
        unsupported = [
            compound
            for compound in enabled_compounds
            if compound not in supported_compounds
        ]
        if unsupported:
            raise ValueError(
                "enabled_compounds contains unsupported values: "
                + ", ".join(unsupported)
            )

        use_stress_functors = conf.get("use_stress_functors", False)
        if not isinstance(use_stress_functors, bool):
            raise TypeError("use_stress_functors must be a bool")
        # The stress controls a composition supplies are declared, not inferred
        # from the state: every declared control is a required input with its
        # provider as a required dependency, and an undeclared control is no
        # term of f_stress. Without the key the enabled path declares
        # DEFAULT_STRESS_FACTORS (water, temperature and ozone).
        configured_factors = conf.get("stress_factors")
        if configured_factors is None:
            stress_factors = self.DEFAULT_STRESS_FACTORS if use_stress_functors else ()
        else:
            if not use_stress_functors:
                raise ValueError("stress_factors requires use_stress_functors")
            if not isinstance(configured_factors, (list, tuple)) or not configured_factors:
                raise TypeError("stress_factors must be a non-empty list or tuple")
            if len(set(configured_factors)) != len(configured_factors):
                raise ValueError("stress_factors must not contain duplicates")
            unsupported_factors = [
                factor
                for factor in configured_factors
                if factor not in self.STRESS_FACTORS
            ]
            if unsupported_factors:
                raise ValueError(
                    "stress_factors contains unsupported values: "
                    + ", ".join(str(factor) for factor in unsupported_factors)
                )
            stress_factors = tuple(
                factor
                for factor in self.STRESS_FACTORS
                if factor in configured_factors
            )

        # temperature_history declares T24 and T240 (TEMPERATURE_HISTORY_STATES) as
        # required inputs and replaced outputs; without it Eqs. (9)-(10) stay at the
        # fixed standard history and no history port exists.
        temperature_history = conf.get("temperature_history", False)
        if not isinstance(temperature_history, bool):
            raise TypeError("temperature_history must be a bool")
        # temperature_source names the declared temperature input (TEMPERATURE_SOURCES);
        # with "leaf_temperature" the history states are the leaf-temperature means.
        temperature_source = conf.get("temperature_source", "air_temperature")
        if temperature_source not in self.TEMPERATURE_SOURCES:
            raise ValueError(
                "temperature_source must be one of "
                f"{list(self.TEMPERATURE_SOURCES)}, got {temperature_source!r}"
            )
        temperature_history_states = (
            self.LEAF_TEMPERATURE_HISTORY_STATES
            if temperature_source == "leaf_temperature"
            else self.TEMPERATURE_HISTORY_STATES
        )

        ppfd_source = conf.get("ppfd_source", "atmospheric_radiation")
        if not isinstance(ppfd_source, str) or ppfd_source not in ("atmospheric_radiation", "state"):
            raise ValueError("ppfd_source must be 'atmospheric_radiation' or 'state'")

        default_pft = conf.get("default_pft")
        if default_pft is not None:
            if not isinstance(default_pft, str) or not default_pft.strip():
                raise TypeError("default_pft must be a non-empty canonical label")
            default_pft = default_pft.strip()

        # The literal default lists are intentional: Phase 8 statically audits
        # the default Hybrid interface without importing scientific modules.
        input_variables = [
            "air_temperature",
            "ppfd",
            "leaf_area_index",
            "atmospheric_co2_ppm",
            "net_photosynthesis",
        ]
        if emission_pathway == EmissionPathway.STANDARD:
            input_variables = input_variables[:-1]
        input_variables.extend(
            self.STRESS_FACTORS[factor][0] for factor in stress_factors
        )
        if temperature_source != "air_temperature":
            input_variables[0] = temperature_source
        if temperature_history:
            input_variables.extend(temperature_history_states)
        temporal_support = conf.get("temporal_support", "instantaneous")
        if temporal_support not in self.TEMPORAL_SUPPORTS:
            raise ValueError(
                "MEGANModel temporal_support must be one of "
                f"{list(self.TEMPORAL_SUPPORTS)}, got {temporal_support!r}"
            )
        daylight_rule: Optional[Tuple[int, str]] = None
        daylight_sample_inputs: Dict[str, str] = {}
        if temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            daylight_rule = daylight_quadrature.daylight_config(conf, "MEGANModel")
            daylight_sample_inputs = {
                law: port for law, port in self.DAYLIGHT_SAMPLE_INPUTS.items()
                if law in input_variables
            }
            input_variables = [
                self.DAYLIGHT_SAMPLE_INPUTS.get(key, key) for key in input_variables
            ]
            input_variables.extend(self.DAYLIGHT_INPUTS)
            if temperature_source == "air_temperature":
                input_variables.extend(self.DAYLIGHT_AIR_TEMPERATURE_INPUTS)
        else:
            daylight_quadrature.reject_daylight_keys(conf, "MEGANModel")

        output_variables = list(self.CANONICAL_OUTPUTS)
        if enabled_compounds != supported_compounds:
            output_variables = [
                *(f"{compound}_emission" for compound in enabled_compounds),
                "total_bvoc_emission",
                "bvoc_carbon_allocation_flux",
                "diagnostics.bvoc.megan.environmental_stress_factor",
                *(
                    ["diagnostics.bvoc.megan.co2_inhibition_factor"]
                    if "isoprene" in enabled_compounds
                    else []
                ),
                *(
                    f"diagnostics.bvoc.megan.{compound}.gamma_temperature"
                    for compound in enabled_compounds
                ),
                *(
                    f"diagnostics.bvoc.megan.{compound}.gamma_light"
                    for compound in enabled_compounds
                ),
            ]
        if emission_pathway == EmissionPathway.STANDARD:
            if self.PHOTOSYNTHESIS_CARBON_FLOOR_RESIDUAL in output_variables:
                output_variables.remove(self.PHOTOSYNTHESIS_CARBON_FLOOR_RESIDUAL)
        elif self.PHOTOSYNTHESIS_CARBON_FLOOR_RESIDUAL not in output_variables:
            output_variables.append(self.PHOTOSYNTHESIS_CARBON_FLOOR_RESIDUAL)
        if temperature_history:
            output_variables.extend(temperature_history_states)
            output_variables.extend(self.TEMPERATURE_HISTORY_DIAGNOSTICS)
        history_description = (
            "E_opt and T_opt of Eq. (8) follow the carried history: T_opt = 313 K + "
            "0.6 (T240 - 297 K) and E_opt = C_eo exp(0.05 (T24 - 297 K)) "
            "exp(0.05 (T240 - 297 K)) (Guenther et al. 2012 Eqs. 9-10), with T24 and "
            f"T240 the declared states {temperature_history_states[0]} and "
            f"{temperature_history_states[1]} read at the start of the call and advanced by "
            "X_new = (1 - w) X + w T, w = min(dt / tau, 1), tau = 24 h and 240 h"
            if temperature_history
            else "E_opt and T_opt of Eq. (8) "
            "are fixed at the standard "
            "history T24 = T240 = T_s = 297 K (Guenther et al. 2012 Eqs. 9-10: "
            "E_opt = C_eo, T_opt = 313 K), since no 24 h or 240 h leaf-temperature "
            "history is carried"
        )
        history_expression = (
            "; T_opt = T_opt,s + a (T240 - T_s,h), E_opt,i = C_eo,i exp(c24 (T24 - "
            "T_s,h)) exp(c240 (T240 - T_s,h)) (Eqs. 9-10); T24, T240 <- (1 - w) X + "
            "w T, w = min(dt / tau, 1)"
            if temperature_history
            else ""
        )
        
        # Create metadata for FunctorBase
        from ....core.functor import FunctorMetadata, ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Reduced instantaneous MEGAN BVOC emission closure; the lumped OVOC "
                "class mixes each activity factor over its components (sum_j w_j "
                "gamma_X,j, Table 6 emission shares) and multiplies the mixed factors, "
                "neglecting their covariance (relative error of E_OVOC in darkness "
                "+12.7% at 0 degC, +5.0% at 20 degC and -10.3% at 40 degC; over "
                "0-40 degC at most 0.65% for PPFD >= 500 and at most 0.2% for PPFD "
                "750-2000 umol m-2 s-1 (0.22% at 2500)); "
                + history_description
                + (
                    "; T is the declared leaf_temperature of the leaf energy balance "
                    "(temperature_source leaf_temperature), not the air temperature"
                    if temperature_source == "leaf_temperature"
                    else ""
                )
            ),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            parameters=cast(
                Dict[str, str],
                {
                    **self.PARAMETER_TEXTS,
                    **(
                        self.TEMPERATURE_HISTORY_PARAMETER_TEXTS
                        if temperature_history
                        else {}
                    ),
                    **(
                        {
                            "biosphere.pft."
                            f"{default_pft}.bvoc."
                            f"{self.PFT_EMISSION_FACTOR_KEYS[compound]}": (
                                f"Emission factor of {compound} for the PFT {default_pft} "
                                "[ug g-1 h-1]"
                            )
                            for compound in enabled_compounds
                        }
                        if default_pft is not None
                        else {}
                    ),
                },
            ),

            mathematical_expression=(
                "rho_f = LAI x k_f; E_i = epsilon_i x gamma_L,i x "
                "gamma_T,i x rho_f x gamma_CO2,i x f_stress / "
                "seconds_per_hour; gamma_L,i = (1 - LDF_i) + LDF_i x "
                "gamma_P_LDF; gamma_T,i = (1 - LDF_i) exp(beta_i (T - T_s)) + "
                "LDF_i gamma_T,LDF,i (Eq. 7, one Table 4 row per class); lumped "
                "OVOC: gamma_X = sum_j w_j gamma_X,j over methanol, acetone and "
                "acetaldehyde with Table 6 emission shares w_j, and E_OVOC uses the "
                "product of the mixed factors instead of sum_j w_j gamma_L,j gamma_T,j "
                "(neglected covariance: in darkness +12.7% at 0 degC, +5.0% at 20 degC, "
                "-10.3% at 40 degC; over 0-40 degC at most 0.65% for PPFD >= 500 and "
                "at most 0.2% for PPFD 750-2000 umol m-2 s-1 (0.22% at 2500)); "
                "F_C,alloc = convert(sum_i(F_i MW_i f_C,i), "
                "g C m-2 s-1 to kg C m-2 s-1)"
                + history_expression
            ),
            references=[
                "Guenther et al. (2012). The Model of Emissions of Gases and Aerosols from Nature version 2.1 (MEGAN2.1): an extended and updated framework for modeling biogenic emissions. Geosci. Model Dev., 5, 1471-1492.",
                "Guenther et al. (2006). Estimates of global terrestrial isoprene emissions using MEGAN (Model of Emissions of Gases and Aerosols from Nature). Atmos. Chem. Phys., 6, 3181-3210.",
                "Niinemets et al. (2011). Estimations of isoprenoid emission capacity from enclosure studies: measurements, data processing, quality and standardized measurement protocols. Biogeosciences, 8, 2209-2246."
            ],
            input_variables=input_variables,
            output_variables=output_variables,
            # This reduced closure exposes an open species-to-carbon mapping;
            # it does not own a closed whole-plant carbon stock.
            conservation_laws=["carbon_boundary"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        declared_parameters = dict(metadata.parameters)
        declared_references = list(metadata.references)
        
        # Initialize BiologicalProcess
        resolved_device = device if device is not None else torch.device("cpu")

        super().__init__(
            name=name,
            config=conf,
            metadata=metadata,
            device=resolved_device,
        )

        # The bridge exposes generic response helpers. This process declares
        # only the parameters and references used by its canonical runtime.
        metadata.parameters = declared_parameters
        metadata.references = declared_references
        self.default_pft = default_pft
            
        self.coupling_strategy = coupling_strategy
        self.use_stress_functors = use_stress_functors
        self.stress_factors = stress_factors
        self.temperature_history = temperature_history
        self.temperature_source = temperature_source
        self.temperature_history_states = temperature_history_states
        self.ppfd_source = ppfd_source
        self.emission_pathway = emission_pathway
        self.validate_inputs = validate_inputs
        self.temporal_support = temporal_support
        self._daylight_rule = daylight_rule
        self._daylight_sample_inputs = daylight_sample_inputs

        # Compound tracking
        self.supported_compounds = supported_compounds
        self.enabled_compounds = enabled_compounds

        self._initialize_parameter_buffers(conf)
        self.to(self.device)

    def _initialize_parameter_buffers(self, config: Dict[str, Any]) -> None:
        """Initialize frequently used scalar parameters as device-aware buffers."""

        def _as_tensor(value: Union[Tensor, float], units: Optional[str] = None) -> Tensor:
            tensor = (
                value.detach().to(device=self.device, dtype=torch.float32)
                if isinstance(value, Tensor)
                else torch.tensor(float(value), dtype=torch.float32, device=self.device)
            )
            if tensor.numel() != 1:
                raise ValueError(
                    f"Parameter expected to be scalar tensor, got shape {tuple(tensor.shape)} with units {units}"
                )
            return tensor

        def _hold(attribute: str, value: Union[Tensor, float], units: Optional[str] = None) -> None:
            # References: architecture SSOT (docs/source/architecture_overview.md)
            #   Sections 4.2 and 4.4.2, learnable parameter ownership.
            # Mathematical expression: theta_held = theta_manager (identity), else
            #   theta_held = float32 scalar copy of the resolved value.
            # Parameter meanings: theta is one scalar MEGAN coefficient resolved by
            #   get_parameter_value; units are those of the calling key.
            # Domain/constraints: a learnable ParameterManager binding is held by
            #   reference (FunctorBase.__setattr__ stores it as a plain attribute), so
            #   gradients of every consumer reach the one manager-owned theta and the
            #   binding stays visible to held_manager_parameters() and checkpoints.
            #   A copy made here would freeze the construction value and cut the
            #   autograd path. Every other value keeps the former float32 buffer.
            # Sign convention: none; values are passed unchanged.
            if self.manager_binding_key(value) is not None:
                if value.numel() != 1:
                    raise ValueError(
                        f"Parameter expected to be scalar tensor, got shape {tuple(value.shape)} with units {units}"
                    )
                setattr(self, attribute, value)
                return
            self.register_buffer(attribute, _as_tensor(value, units=units))

        self.hybrid_coupling_weight = float(
            self.get_parameter_value(
                "hybrid_weight",
                namespace="biosphere.bvoc.megan", required=True)
        )
        # References: BIPM SI Brochure, 9th edition.
        # Mathematical expression: m[g] = m[ug] * k_ug_to_g.
        # Parameter meanings: k_ug_to_g is the exact SI mass conversion [g ug-1].
        micro_to_gram = get_physical_constant(
            "relic.physical.constants.conversions.micrograms_to_grams",
            as_tensor=True,
            device=self.device,
        )
        self.register_buffer(
            "micrograms_to_grams",
            _as_tensor(micro_to_gram, units="g ug-1"),
        )
        self._carbon_fraction_buffers: Dict[str, str] = {}
        for compound in self.supported_compounds:
            param_name = f"carbon_fraction.{compound}"
            fraction_value = self.get_parameter_value(
                param_name,
                namespace="biosphere.bvoc",
                required=True,
            )
            buffer_name = f"{compound}_carbon_fraction"
            _hold(buffer_name, fraction_value, units="dimensionless")
            self._carbon_fraction_buffers[compound] = buffer_name

        # References: CODATA 2018 molar gas constant.
        # Mathematical expression: x = (1 / Topt - 1 / T) / R.
        # Parameter meanings: R is the molar gas constant [J mol-1 K-1].
        gas_constant_value = get_physical_constant(
            "gas_constant", as_tensor=True, device=self.device
        )
        self.register_buffer(
            "gas_constant",
            _as_tensor(gas_constant_value, units="J mol-1 K-1"),
        )
        seconds_per_hour_value = get_physical_constant(
            "seconds_per_hour", as_tensor=True, device=self.device
        )
        self.register_buffer(
            "seconds_per_hour",
            _as_tensor(seconds_per_hour_value, units="s h-1"),
        )

        min_temp_value = self.get_parameter_value(
            "temperature.min_operational",
            namespace="biosphere.bvoc.validation", required=True)
        max_temp_value = self.get_parameter_value(
            "temperature.max_operational",
            namespace="biosphere.bvoc.validation", required=True)
        self.register_buffer(
            "min_operational_temperature",
            _as_tensor(min_temp_value, units="K"),
        )
        self.register_buffer(
            "max_operational_temperature",
            _as_tensor(max_temp_value, units="K"),
        )

        # Emission factors (ug g-1 DW h-1)
        iso_ef_value = self.get_parameter_value(
            "emission_factors.isoprene",
            namespace="biosphere.bvoc", required=True)
        _hold("isoprene_ef", iso_ef_value, units="ug g-1 DW h-1")
        mono_ef_value = self.get_parameter_value(
            "emission_factors.monoterpenes",
            namespace="biosphere.bvoc", required=True)
        _hold("monoterpenes_ef", mono_ef_value, units="ug g-1 DW h-1")
        sesq_ef_value = self.get_parameter_value(
            "emission_factors.sesquiterpenes",
            namespace="biosphere.bvoc", required=True)
        _hold("sesquiterpenes_ef", sesq_ef_value, units="ug g-1 DW h-1")
        ovoc_ef_value = self.get_parameter_value(
            "emission_factors.ovocs",
            namespace="biosphere.bvoc", required=True)
        _hold("ovocs_ef", ovoc_ef_value, units="ug g-1 DW h-1")

        self.emission_factors = {
            'isoprene': self.isoprene_ef,
            'monoterpenes': self.monoterpenes_ef,
            'sesquiterpenes': self.sesquiterpenes_ef,
            'ovocs': self.ovocs_ef,
        }
        self._pft_emission_factor_buffers: Dict[str, str] = {}
        if self.default_pft is not None:
            for compound in self.enabled_compounds:
                # References: Guenther et al. (2012), PFT emission potentials.
                # Mathematical expression: epsilon_i = epsilon_i(PFT).
                # Parameter meanings: epsilon_i is the configured PFT emission
                # potential [ug g-1 DW h-1], resolved once at construction.
                pft_parameter_key = self.PFT_EMISSION_FACTOR_KEYS[compound]
                pft_value = self.get_parameter_value(
                    pft_parameter_key,
                    namespace=f"biosphere.pft.{self.default_pft}.bvoc",
                    required=True,
                    source_order=("manager",),
                )
                buffer_name = f"pft_{compound}_emission_factor"
                _hold(buffer_name, pft_value, units="ug g-1 DW h-1")
                self._pft_emission_factor_buffers[compound] = buffer_name

        # Light response parameters
        alpha_value = self.get_parameter_value(
            "alpha_light",
            namespace="biosphere.bvoc.megan", required=True)
        _hold("alpha_light", alpha_value)
        cl_value = self.get_parameter_value(
            "cl_light",
            namespace="biosphere.bvoc.megan", required=True)
        _hold("cl_light", cl_value)
        # Table 4 keys read from registered buffers (the tutorial case promotes these
        # isoprene buffers to trainable parameters); every other Table 4 coefficient is
        # read through get_parameter_value by the one lookup of both activities.
        self._table4_buffers: Dict[str, str] = {
            "eopt_isoprene": "eopt_isoprene",
            "ct1_isoprene": "ct1_iso",
        }

        # Temperature references
        topt_value = self.get_parameter_value(
            "topt",
            namespace="biosphere.bvoc.megan", required=True)
        _hold("topt", topt_value, units="K")

        temp_std_value = self.get_parameter_value(
            "t_standard",
            namespace="biosphere.bvoc", required=True)
        _hold("temp_std", temp_std_value, units="K")

        eopt_value = self.get_parameter_value(
            "eopt_isoprene",
            namespace="biosphere.bvoc.megan", required=True)
        _hold("eopt_isoprene", eopt_value, units="dimensionless")

        ct1_value = self.get_parameter_value(
            "ct1_isoprene",
            namespace="biosphere.bvoc.megan", required=True)
        _hold("ct1_iso", ct1_value, units="J mol-1")

        ct2_value = self.get_parameter_value(
            "ct2",
            namespace="biosphere.bvoc.megan", required=True)
        _hold("ct2", ct2_value, units="J mol-1")

        # References: Guenther et al. (2012) Eqs. (9)-(10); Oleson et al. (2010) CLM4
        #   running means of MEGAN2.1 (megan_temperature_history_update).
        # Parameter meanings: T_s,h standard leaf temperature [K], a the Eq. (9) slope
        #   [1], c24 and c240 the Eq. (10) coefficients [K-1], tau24 and tau240 the
        #   averaging time scales [s]; read only by the temperature_history option.
        if self.temperature_history:
            for attribute, units in (
                ("history_standard_temperature", "K"),
                ("topt_history_slope", "dimensionless"),
                ("eopt_history_coefficient_24h", "K-1"),
                ("eopt_history_coefficient_240h", "K-1"),
                ("history_time_scale_24h", "s"),
                ("history_time_scale_240h", "s"),
            ):
                history_value = self.get_parameter_value(
                    attribute,
                    namespace="biosphere.bvoc.megan", required=True)
                _hold(attribute, history_value, units=units)

        # Photosynthesis coupling fractions
        coupling_param_map: Dict[str, str] = {
            'isoprene': "photosynthetic_coupling.f_carbon_isoprene",
            'monoterpenes': "photosynthetic_coupling.f_carbon_monoterpenes",
            'sesquiterpenes': "photosynthetic_coupling.f_carbon_sesquiterpenes",
            'ovocs': "photosynthetic_coupling.f_carbon_ovocs",
        }
        default_coupling: Dict[str, float] = {}
        for compound, param_key in coupling_param_map.items():
            base_value = self.get_parameter_value(
                param_key,
                namespace="biosphere.bvoc", required=True)
            default_coupling[compound] = float(
                self._scalar_from_value(base_value) if isinstance(base_value, Tensor) else base_value
            )

        self.molecular_weights: Dict[str, float] = {}
        self.carbon_atoms: Dict[str, float] = {}

        for compound in self.supported_compounds:
            # Carbon allocation fractions
            coupling_key = f"f_carbon_{compound}"
            override_value = default_coupling[compound]
            if isinstance(override_value, Tensor):
                value_tensor = override_value.to(device=self.device, dtype=torch.float32)
            else:
                value_tensor = torch.tensor(float(override_value), dtype=torch.float32, device=self.device)
            self.register_buffer(coupling_key, value_tensor)

            # References: IUPAC/CIAAW atomic weights and Guenther et al. (2012).
            # Expression: E_mol = E_mass / M_compound.
            # Parameters: M_compound is molar mass [g mol-1].
            if compound == "isoprene":
                mw_value = float(
                    get_physical_constant(
                        "relic.physical.constants.molecular_weights.isoprene"
                    )
                )
            else:
                mw_value = float(
                    self.get_parameter_value(
                        f"molecular_weights.{compound}",
                        namespace="biosphere.bvoc",
                        required=True,
                    )
                )
            carbon_value = float(
                self.get_parameter_value(
                    f"carbon_atoms.{compound}",
                    namespace="biosphere.bvoc", required=True)
            )
            self.molecular_weights[compound] = mw_value
            self.carbon_atoms[compound] = carbon_value
    
    def get_dependencies(self) -> List[str]:
        """
        Get required functor dependencies for execution ordering.
        
        Returns the list of functor names that must be executed before
        this functor during each time step. This follows FunctorBase semantics
        where dependencies are other functors, not state variables.
        
        Returns
        -------
        list
            Required functor names to execute before this one
        """
        # The selected PPFD provider changes wiring only; both declarations
        # consume the same required state field through the single forward law.
        deps = ["atmospheric_radiation"] if self.ppfd_source == "atmospheric_radiation" else []

        if self.emission_pathway in (
            EmissionPathway.PHOTOSYNTHESIS,
            EmissionPathway.HYBRID,
        ):
            deps.append("farquhar_photosynthesis")

        deps.extend(self.STRESS_FACTORS[factor][1] for factor in self.stress_factors)

        return self._merge_declared_dependencies(deps)
    
    def get_provides(self) -> List[str]:
        """
        Get output variables provided.
        
        Returns
        -------
        list
            Output variable names
        """
        return list(self.metadata.output_variables)

    @staticmethod
    def _scalar_from_value(value: Any, reducer: str = "first") -> float:
        """Convert tensor/scalar-like values to Python float in non-gradient paths."""
        if isinstance(value, Tensor):
            flat = value.detach().reshape(-1)
            if flat.numel() == 0:
                raise ValueError("Cannot convert an empty tensor to a scalar")
            if reducer == "mean":
                return float(flat.mean().tolist())
            return float(flat[0].tolist())
        return float(value)

    def _resolve_forward_inputs(
        self,
        mapped_state: Dict[str, Any],
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor]:
        # The declared temperature input (temperature_source): the air temperature or,
        # with "leaf_temperature", the leaf temperature of the leaf energy balance [K].
        air_temperature = self._extract_tensor(
            mapped_state, self.temperature_source
        )
        ppfd = self._extract_tensor(mapped_state, "ppfd")
        leaf_area_index = self._extract_tensor(
            mapped_state, "leaf_area_index"
        )
        atmospheric_co2_ppm = self._extract_tensor(
            mapped_state, "atmospheric_co2_ppm"
        )

        # References: Guenther et al. (2012) for response-factor heritage and
        # Niinemets et al. (2010) for leaf-mass emission-potential scaling.
        # Mathematical expression: RELIC reduced transform rho_f = LAI * k_f.
        # Parameter meanings: LAI is canonical leaf area [m2 m-2], k_f is the
        # effective dry-leaf mass per canonical LAI leaf area [g DW m-2], and
        # rho_f is dry-leaf mass per ground area [g DW m-2]. This transform is
        # not a full PFT-resolved MEGAN2.1 canopy integration.
        foliar_density_factor = self.get_parameter_value(
            "foliar_density_factor",
            namespace="biosphere.bvoc",
            source_order=("manager",),
            required=True,
        )
        foliar_density = leaf_area_index * self._ensure_tensor(
            foliar_density_factor, dtype=torch.float32
        ).to(dtype=leaf_area_index.dtype, device=leaf_area_index.device)

        # Only the declared stress controls are read, keyed by their state
        # variable in the product order of STRESS_FACTORS.
        stress_controls = {
            self.STRESS_FACTORS[factor][0]: self._extract_tensor(
                mapped_state, self.STRESS_FACTORS[factor][0]
            )
            for factor in self.stress_factors
        }

        return (
            air_temperature,
            ppfd,
            foliar_density,
            stress_controls,
            atmospheric_co2_ppm,
        )

    def _compute_compound_emission(
        self,
        compound: str,
        available_photosynthesis_carbon: Optional[Tensor],
        temperature: Tensor,
        ppfd: Tensor,
        foliar_density: Tensor,
        environmental_stress: Tensor,
        atmospheric_co2_ppm: Tensor,
        optimum_history: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        # References: Guenther et al. (2012); Monson et al. (2012), existing reduced coupling
        #   choices.
        # Mathematical expression: E_base=w*E_coupled+(1-w)*E_standard in hybrid mode;
        #   E=E_base*gamma_CO2*f_env, with gamma_CO2=1 for non-isoprene.
        # Parameter meanings: w is hybrid_coupling_weight; f_env is the product of the declared
        #   stress controls.
        # Units: E [ug m-2 s-1]; w,gamma_CO2,f_env [1].
        # Domain/constraints: Coupled and hybrid selections require projected photosynthesis
        #   carbon; each pathway is selected by the existing constructor, not fallback on a failed
        #   calculation.
        # Sign convention: All multipliers modify the emission magnitude; no plant stock is debited
        #   by this helper.
        if self.emission_pathway == EmissionPathway.PHOTOSYNTHESIS:
            if available_photosynthesis_carbon is None:
                raise RuntimeError(
                    "Photosynthesis-coupled MEGAN emission requires projected "
                    "photosynthesis carbon."
                )
            emission_rate = self._compute_photosynthesis_coupled_emission(
                compound,
                available_photosynthesis_carbon,
                temperature,
                optimum_history=optimum_history,
            )
        elif self.emission_pathway == EmissionPathway.HYBRID:
            if available_photosynthesis_carbon is None:
                raise RuntimeError(
                    "Hybrid MEGAN emission requires projected photosynthesis carbon."
                )
            standard_rate = self._compute_standard_emission(
                compound, temperature, ppfd, foliar_density,
                optimum_history=optimum_history,
            )
            coupled_rate = self._compute_photosynthesis_coupled_emission(
                compound,
                available_photosynthesis_carbon,
                temperature,
                optimum_history=optimum_history,
            )
            weight_coupled = self.hybrid_coupling_weight
            emission_rate = weight_coupled * coupled_rate + (1 - weight_coupled) * standard_rate
        else:
            emission_rate = self._compute_standard_emission(
                compound, temperature, ppfd, foliar_density,
                optimum_history=optimum_history,
            )

        co2_factor: Optional[Tensor] = None
        if compound == 'isoprene':
            co2_factor = self._compute_co2_inhibition_factor(atmospheric_co2_ppm)
            emission_rate = emission_rate * co2_factor

        emission_rate = emission_rate * environmental_stress
        temperature_factor = self._get_temperature_factor(
            compound, temperature, optimum_history
        )
        light_factor = self._get_light_factor(compound, ppfd)
        return emission_rate, co2_factor, temperature_factor, light_factor

    def _finalize_forward_results(
        self,
        emission_results: Dict[str, Tensor],
        state: Dict[str, Tensor],
        environmental_stress: Tensor,
    ) -> Dict[str, FluxValue]:
        # References: Guenther et al. (2012); SI prefix conversion and declared compound molecular
        #   composition.
        # Mathematical expression: F_i=E_i*ug_to_g/MW_i; F_total=sum_i(F_i);
        #   F_C=convert(sum_i(F_i*MW_i*fC_i),g C to kg C).
        # Parameter meanings: MW_i and fC_i are compound molecular weight and carbon mass fraction
        #   from the existing parameter owner.
        # Units: F_i,F_total [mol m-2 s-1]; F_C [kg C m-2 s-1].
        # Domain/constraints: Sum only enabled compounds; identity output maps and a sum row
        #   preserve these ports but do not prove closed plant/species stoichiometry or strict
        #   IPHS.
        # Sign convention: Species emissions and carbon allocation are positive outward magnitudes.
        molar_fluxes: List[Tensor] = []
        carbon_flux_terms: List[Tensor] = []
        for compound in self.SUPPORTED_COMPOUNDS:
            if compound not in self.enabled_compounds:
                continue
            emission_key = f"{compound}_emission"
            mass_flux = emission_results[emission_key]
            molecular_weight = torch.as_tensor(
                self.molecular_weights[compound],
                dtype=torch.float32,
                device=mass_flux.device,
            ).to(dtype=mass_flux.dtype)
            conversion = cast(Tensor, self.micrograms_to_grams).to(
                dtype=mass_flux.dtype, device=mass_flux.device
            )
            molar_flux = mass_flux * conversion / molecular_weight
            emission_results[emission_key] = molar_flux
            molar_fluxes.append(molar_flux)
            carbon_fraction = self._get_carbon_fraction(compound).to(
                dtype=mass_flux.dtype, device=mass_flux.device
            )
            carbon_flux_terms.append(
                molar_flux * molecular_weight * carbon_fraction
            )

        total_emission = torch.stack(molar_fluxes).sum(dim=0)
        total_carbon_flux_g = torch.stack(carbon_flux_terms).sum(dim=0)
        carbon_allocation_flux = unit_converter.convert(
            total_carbon_flux_g,
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        )
        emission_results["total_bvoc_emission"] = total_emission
        emission_results["bvoc_carbon_allocation_flux"] = (
            carbon_allocation_flux
        )
        emission_results[
            "diagnostics.bvoc.megan.environmental_stress_factor"
        ] = environmental_stress

        compound_species = tuple(
            f"{compound}_emission"
            for compound in self.SUPPORTED_COMPOUNDS
            if f"{compound}_emission" in emission_results
        )
        if compound_species:
            reactions = compound_species
            matrix_rows = []
            for index, _key in enumerate(compound_species):
                row = [0.0] * len(reactions)
                row[index] = 1.0
                matrix_rows.append(row)
            total_row = [1.0] * len(compound_species)
            species = compound_species + ("total_bvoc_emission",)
            # References: Guenther et al. (2012); Sindelarova et al. (2022); Grmela and Ottinger (1997).
            # Mathematical expression: x_dot = N_h r_h for compound emissions,
            # total BVOC emission, and the equivalent carbon-allocation proxy.
            # Parameter meanings: N_h maps compound emission channels to typed
            # BVOC output cochains; r_h stores ParameterManager-backed MEGAN rates.
            emission_network = advance_reaction_network(
                name=f"{self.name}.bvoc_emission_reaction_network",
                species=species,
                reactions=reactions,
                stoichiometric_N=torch.as_tensor(
                    matrix_rows + [total_row],
                    device=total_emission.device,
                    dtype=total_emission.dtype,
                ),
                rates=tuple(emission_results[key] for key in compound_species),
                open_system_route="port",
            )
            for compound in self.SUPPORTED_COMPOUNDS:
                emission_key = f"{compound}_emission"
                if emission_key not in emission_results:
                    continue
                emission_results[emission_key] = emission_network.species_rate(
                    emission_key
                )
            emission_results["total_bvoc_emission"] = (
                emission_network.species_rate("total_bvoc_emission")
            )

        carbon_network = advance_reaction_network(
            name=f"{self.name}.bvoc_carbon_allocation_network",
            species=("bvoc_carbon_allocation_flux",),
            reactions=("bvoc_carbon_allocation",),
            stoichiometric_N=torch.ones(
                (1, 1),
                device=carbon_allocation_flux.device,
                dtype=carbon_allocation_flux.dtype,
            ),
            rates=(carbon_allocation_flux,),
            open_system_route="port",
        )
        emission_results[
            "bvoc_carbon_allocation_flux"
        ] = carbon_network.species_rate(
            "bvoc_carbon_allocation_flux"
        )
        self._enforce_forward_conservation(state, emission_results)

        wrapped_results: Dict[str, FluxValue] = {}
        for key in self.CANONICAL_OUTPUTS:
            if key not in emission_results:
                continue
            wrapped_results[key] = UpdateItem(
                value=emission_results[key].to(self.device),
                mode=UpdateMode.REPLACE,
                unit=self._canonical_output_unit(key),
                per_timestep=False,
                source_functor=self.name,
            )
        return wrapped_results

    @staticmethod
    def _canonical_output_unit(key: str) -> str:
        """Return the exact unit for a retained canonical output."""
        if key == "bvoc_carbon_allocation_flux":
            return "kg C m-2 s-1"
        if key == MEGANModel.PHOTOSYNTHESIS_CARBON_FLOOR_RESIDUAL:
            return "umol C m-2 s-1"
        if key.endswith("_emission"):
            return "mol m-2 s-1"
        return "dimensionless"
    
    def forward(
        self,
        state: Dict[str, Tensor],
        time: float,
        dt: float,
        **kwargs: Any
    ) -> Dict[str, FluxValue]:
        """Dispatch to the declared temporal support (instantaneous or daylight_gauss)."""
        # References: Guenther et al. (2012); Goudriaan (1986);
        #   relic.scientific.atmosphere.daylight_quadrature.
        # Mathematical expression: instantaneous: _forward_law on the inputs;
        #   daylight_gauss: E = sum_k W_k E_k (_forward_daylight_gauss).
        # Parameter meanings: W_k the daylight sample weights; E_k the law at sample k.
        # Units: as _forward_law.
        # Domain/constraints: daylight_gauss requires dt = seconds_per_day.
        # Sign convention: emissions are positive outward magnitudes.
        if self.temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            return self._forward_daylight_gauss(state, time, dt)
        return self._forward_law(state, time, dt, **kwargs)

    def _forward_daylight_gauss(
        self,
        state: Dict[str, Tensor],
        time: float,
        dt: float,
    ) -> Dict[str, FluxValue]:
        """Daily means of the emission law over the daylight Gauss samples."""
        # References: Guenther et al. (2012), Eqs. (2)-(11); Goudriaan (1986);
        #   Goudriaan and van Laar (1994); daylight_quadrature (FAO-56 geometry, SWAT+
        #   Eq. 1:1.3.1 course).
        # Mathematical expression: E = sum_k W_k E(T_k, Q_k, ...); W_i = (N/24) w_i,
        #   W_night = 1 - N/24, Q_night = 0, so the night term keeps the light-independent
        #   fraction (1 - LDF) gamma_T(T_night) of each class; X_new = (1 - w) X + w T_d
        #   for T24 and T240 with T_d the daily mean temperature (the air_temperature input,
        #   or sum_k W_k T_leaf,k).
        # Parameter meanings: T_k the sample air or leaf temperature; Q_k the 24-hour mean
        #   ppfd redistributed by the solar elevation; LAI, CO2 and the daily stress
        #   factors held constant; the ozone factor and the net photosynthesis per sample.
        # Units: E [mol m-2 s-1]; T [K]; Q [umol photons m-2 s-1].
        # Domain/constraints: dt = seconds_per_day; every sample passes the input checks
        #   of the law; the history states are read once (start of the day).
        # Sign convention: emissions are positive outward magnitudes.
        daylight_quadrature.require_daily_step(dt, "MEGANModel")
        node_count, shortwave_shape = cast(Tuple[int, str], self._daylight_rule)
        ports = set(self._daylight_sample_inputs.values())
        values = {
            key: self._extract_tensor(state, key)
            for key in self.metadata.input_variables
            if key not in ports
        }
        sample_inputs = {
            law: self._extract_tensor(state, port)
            for law, port in self._daylight_sample_inputs.items()
        }
        air_source = self.temperature_source == "air_temperature"
        samples = daylight_quadrature.daylight_sample_state(
            values, sample_inputs, node_count=node_count, shortwave_shape=shortwave_shape,
            flux_keys=("ppfd",), temperature=air_source, dtype=values["ppfd"].dtype)
        sample_outputs = self._forward_law(samples.state, time, dt)
        outputs = daylight_quadrature.daily_outputs(sample_outputs, samples.support)
        if self.temperature_history:
            if air_source:
                daily_temperature = values["air_temperature"]
            else:
                daily_temperature = samples.support.daily_mean(samples.state["leaf_temperature"])
            history = {key: values[key] for key in self.temperature_history_states}
            shape = torch.broadcast_shapes(
                tuple(daily_temperature.shape), *(tuple(v.shape) for v in history.values()))
            daily_temperature = daily_temperature.expand(shape)
            history = {key: value.expand(shape) for key, value in history.items()}
            outputs.update(self._temperature_history_updates(
                daily_temperature, history, self._optimum_history(history, daily_temperature), dt))
        return outputs

    def _forward_law(
        # References:
        # - Guenther et al. (2012), Model of Emissions of Gases and Aerosols from Nature.
        # - Sindelarova et al. (2022), High-resolution global biogenic emission inventory.
        # Mathematical expression:
        # - rho_f = LAI * k_f
        # - E_i = epsilon_i * gamma_T,i * gamma_L,i * rho_f /
        #   seconds_per_hour * f_stress * f_CO2,i
        # Parameter meanings:
        # - epsilon_i: RELIC dry-leaf-mass emission potential [ug g-1 DW h-1]
        # - gamma_T,i, gamma_L,i: temperature and light response factors [-]
        # - k_f: effective dry-leaf mass per canonical LAI leaf area [g DW m-2]
        # - rho_f: LAI-derived dry-leaf mass per ground area [g DW m-2]
        # - f_stress: product of the declared stress controls [-]
        # - f_CO2,i: CO2 inhibition factor for isoprene [-]
        self,
        state: Dict[str, Tensor],
        time: float,
        dt: float,
        **kwargs: Any
    ) -> Dict[str, FluxValue]:
        """
        Forward pass through MEGAN model.
        
        Parameters
        ----------
        state : Dict[str, Tensor]
            Canonical inputs are air_temperature [K], ppfd
            [umol photons m-2 s-1], leaf_area_index [m2 m-2], and
            atmospheric_co2_ppm [ppm]. Photosynthesis and hybrid pathways also
            require net_photosynthesis [umol C m-2 s-1]. When stress coupling
            is enabled, each declared stress control (stress_factors; without
            it water_stress_factor, temperature_stress_factor and
            ozone_stress_factor) is a required dimensionless control.
        time : float
            Current time [s]
        dt : float
            Time step [s]
            
        Returns
        -------
        Dict[str, FluxValue]
            Explicit REPLACE updates for molar species fluxes, carbon boundary
            export, and namespaced diagnostics.
        """
        _ = (time, dt, kwargs)
        mapped_state = dict(state)
        (
            temperature,
            ppfd,
            foliar_density,
            stress_controls,
            atmospheric_co2_ppm,
        ) = self._resolve_forward_inputs(mapped_state)
        leaf_area_index = self._extract_tensor(
            mapped_state, "leaf_area_index"
        )

        net_photosynthesis: Optional[Tensor] = None
        if self.emission_pathway in (
            EmissionPathway.PHOTOSYNTHESIS,
            EmissionPathway.HYBRID,
        ):
            net_photosynthesis = self._extract_tensor(
                mapped_state, "net_photosynthesis"
            )
        # The committed T24 and T240 at the start of this call (temperature_history):
        # the history through the previous step, never the temperature of this step.
        temperature_history: Dict[str, Tensor] = (
            {
                key: self._extract_tensor(mapped_state, key)
                for key in self.temperature_history_states
            }
            if self.temperature_history
            else {}
        )
        if self.validate_inputs:
            self._validate_inputs(
                air_temperature=temperature,
                ppfd=ppfd,
                leaf_area_index=leaf_area_index,
                atmospheric_co2_ppm=atmospheric_co2_ppm,
                net_photosynthesis=net_photosynthesis,
                stress_factors=stress_controls,
                temperature_history=temperature_history,
            )
        input_dtypes = {
            temperature.dtype,
            ppfd.dtype,
            leaf_area_index.dtype,
            atmospheric_co2_ppm.dtype,
            *(control.dtype for control in stress_controls.values()),
            *((net_photosynthesis.dtype,) if net_photosynthesis is not None else ()),
            *(value.dtype for value in temperature_history.values()),
        }
        if len(input_dtypes) != 1:
            raise TypeError(
                "All MEGAN canonical input tensors must have identical floating dtypes."
            )

        available_photosynthesis_carbon: Optional[Tensor] = None
        photosynthesis_carbon_floor_residual: Optional[Tensor] = None
        if net_photosynthesis is not None:
            (
                available_photosynthesis_carbon,
                photosynthesis_carbon_floor_residual,
            ) = self._project_photosynthesis_carbon(net_photosynthesis)

        # References: Guenther et al. (2012), activity-factor product of the reduced
        #   closure; stress controls of relic/scientific/biosphere/stress.
        # Mathematical expression: f_stress = prod_{k in S} f_k over the declared set S
        #   of stress controls, and f_stress = 1 for the empty set.
        # Parameter meanings: f_k is a declared control read from the state (water,
        #   temperature, ozone, heat or cold); S is the stress_factors selection.
        # Units: f_k, f_stress [1].
        # Domain/constraints: each f_k is finite and within [0, 1] (validated above);
        #   a control outside S is not read.
        # Sign convention: f_stress scales the emission magnitude; 1 is unstressed.
        environmental_stress = torch.ones_like(temperature)
        for factor in stress_controls.values():
            environmental_stress = environmental_stress * factor

        optimum_history = (
            self._optimum_history(temperature_history, temperature)
            if self.temperature_history
            else None
        )

        emission_results: Dict[str, Tensor] = {}
        for compound in self.SUPPORTED_COMPOUNDS:
            if compound not in self.enabled_compounds:
                continue
            (
                compound_emission,
                co2_factor,
                temperature_factor,
                light_factor,
            ) = self._compute_compound_emission(
                compound=compound,
                available_photosynthesis_carbon=available_photosynthesis_carbon,
                temperature=temperature,
                ppfd=ppfd,
                foliar_density=foliar_density,
                environmental_stress=environmental_stress,
                atmospheric_co2_ppm=atmospheric_co2_ppm,
                optimum_history=optimum_history,
            )
            emission_results[f'{compound}_emission'] = compound_emission
            emission_results[
                f"diagnostics.bvoc.megan.{compound}.gamma_temperature"
            ] = temperature_factor
            emission_results[
                f"diagnostics.bvoc.megan.{compound}.gamma_light"
            ] = light_factor
            if co2_factor is not None:
                emission_results[
                    "diagnostics.bvoc.megan.co2_inhibition_factor"
                ] = co2_factor

        if photosynthesis_carbon_floor_residual is not None:
            emission_results[self.PHOTOSYNTHESIS_CARBON_FLOOR_RESIDUAL] = (
                photosynthesis_carbon_floor_residual
            )

        results = self._finalize_forward_results(
            emission_results=emission_results,
            state=mapped_state,
            environmental_stress=environmental_stress,
        )
        if optimum_history is not None:
            results.update(
                self._temperature_history_updates(
                    temperature, temperature_history, optimum_history, dt
                )
            )
        return results

    def _optimum_history(
        self,
        temperature_history: Dict[str, Tensor],
        temperature: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """T_opt and the E_opt factor h of Eqs. (9)-(10) from the committed T24 and T240."""
        # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (9)-(10).
        # Mathematical expression: T_opt = T_opt,s + a (T240 - T_s,h); h = exp(c24 (T24 -
        #   T_s,h)) exp(c240 (T240 - T_s,h)) (megan_optimum_history).
        # Parameter meanings: T_opt,s is biosphere.bvoc.megan.topt, T_s,h, a, c24 and c240
        #   the history parameters held at construction.
        # Units: temperatures [K]; a, h [1]; c24, c240 [K-1].
        # Domain/constraints: coefficients are cast to the dtype and device of T; no clamp.
        # Sign convention: a warmer history raises T_opt and h.
        def held(attribute: str) -> Tensor:
            return cast(Tensor, getattr(self, attribute)).to(
                dtype=temperature.dtype, device=temperature.device
            )

        return megan_optimum_history(
            temperature_history[self.temperature_history_states[0]],
            temperature_history[self.temperature_history_states[1]],
            held("topt"),
            held("history_standard_temperature"),
            held("topt_history_slope"),
            held("eopt_history_coefficient_24h"),
            held("eopt_history_coefficient_240h"),
        )

    def _temperature_history_updates(
        self,
        temperature: Tensor,
        temperature_history: Dict[str, Tensor],
        optimum_history: Tuple[Tensor, Tensor],
        dt: Any,
    ) -> Dict[str, FluxValue]:
        """Advanced T24 and T240 (REPLACE state updates) and the history diagnostics."""
        # References: Guenther et al. (2012) p. 1476 (T24, T240 the past 24 h and 240 h
        #   means); Oleson et al. (2010) CLM4 running mean (megan_temperature_history_update).
        # Mathematical expression: X_new = (1 - w) X + w T, w = min(dt / tau, 1), for
        #   (X, tau) = (T24, tau24) and (T240, tau240); T is the temperature of this call.
        # Parameter meanings: dt is the length of this call [s]; tau24 and tau240 the held
        #   averaging time scales [s]; T_opt and h are the values used by this call.
        # Units: X, T, T_opt [K]; dt, tau [s]; w, h [1].
        # Domain/constraints: dt finite and >= 0 (ValueError otherwise); the written
        #   values are the history of the next call (one-step lag by construction).
        # Sign convention: none; X relaxes towards T.
        dt_tensor = torch.as_tensor(dt, dtype=temperature.dtype, device=temperature.device)
        if not bool(torch.isfinite(dt_tensor).all()) or bool(torch.any(dt_tensor < 0.0)):
            raise ValueError("MEGAN temperature history needs a finite dt >= 0")
        updates: Dict[str, FluxValue] = {}
        for key, attribute in zip(
            self.temperature_history_states,
            ("history_time_scale_24h", "history_time_scale_240h"),
        ):
            time_scale = cast(Tensor, getattr(self, attribute)).to(
                dtype=temperature.dtype, device=temperature.device
            )
            updated = megan_temperature_history_update(
                temperature_history[key], temperature, dt_tensor, time_scale
            )
            updates[key] = UpdateItem(
                value=updated.to(self.device),
                mode=UpdateMode.REPLACE,
                unit="K",
                per_timestep=False,
                source_functor=self.name,
            )
        for key, value, unit in zip(
            self.TEMPERATURE_HISTORY_DIAGNOSTICS, optimum_history, ("K", "dimensionless")
        ):
            updates[key] = UpdateItem(
                value=value.to(self.device),
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=False,
                source_functor=self.name,
            )
        return updates

    def check_conservation_laws(
        self,
        input_state: Dict[str, Tensor],
        output_updates: Dict[str, FluxValue],
    ) -> Dict[str, float]:
        """Audit the open mapping from enabled BVOC emissions to carbon.

        References:
            - Guenther et al. (2012), MEGAN2.1 compound emission framework.
            - Niinemets et al. (2010), leaf-level isoprenoid emission factor.
        Mathematical expression:
            R_C = F_C,alloc
                  - convert(sum_i(F_i MW_i f_C,i), g C to kg C).
        Parameter meanings:
            F_i is the enabled compound molar flux [mol m-2 s-1], MW_i is its
            ParameterManager-backed molecular weight [g mol-1], and f_C,i is
            its ParameterManager-backed carbon mass fraction [1].
        Open-system route:
            The positive carbon allocation leaves the vegetation ledger and
            enabled species use their declared atmosphere or boundary ports.
            No closed whole-plant carbon-stock conservation is claimed.
        """
        # References: Guenther et al. (2012); Niinemets et al. (2010).
        # Mathematical expression: r=F_C_alloc-convert(sum_i(F_i*MW_i*fC_i),g C to kg C);
        #   report=max(abs(r)).
        # Parameter meanings: F_i are enabled compound molar ports; F_C_alloc is their
        #   carbon-equivalent debit.
        # Units: Residual [kg C m-2 s-1]; MW [g mol-1]; fC [1].
        # Domain/constraints: Required nonempty finite floating outputs (float32 or float64) of
        #   identical dtype and shape; coefficients use the registered float32 parameter basis
        #   promoted to the ledger dtype.
        # Sign convention: Nonnegative reported magnitude audits an open carbon mapping, not
        #   whole-plant stock closure.

        del input_state

        def require_tensor(key: str) -> Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"MEGAN conservation output '{key}' must be a torch.Tensor."
                )
            if value.numel() == 0:
                raise ValueError(
                    f"MEGAN conservation output '{key}' must be non-empty."
                )
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError(
                    f"MEGAN conservation output '{key}' dtype must be float32 or float64."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"MEGAN conservation output '{key}' must be finite."
                )
            return value.to(device=self.device)

        allocation = require_tensor("bvoc_carbon_allocation_flux")
        emissions = {
            compound: require_tensor(f"{compound}_emission")
            for compound in self.enabled_compounds
        }
        ledger_values = (allocation, *emissions.values())
        if len({tuple(value.shape) for value in ledger_values}) != 1:
            raise ValueError(
                "MEGAN conservation outputs must have identical shapes."
            )
        if len({value.dtype for value in ledger_values}) != 1:
            raise TypeError(
                "MEGAN conservation outputs must have identical dtypes."
            )
        ledger_dtype = allocation.dtype

        carbon_terms = []
        for compound, molar_flux in emissions.items():
            molecular_weight = torch.as_tensor(
                self.molecular_weights[compound],
                device=self.device,
                dtype=torch.float32,
            ).to(dtype=ledger_dtype)
            carbon_fraction = self._get_carbon_fraction(compound).to(
                device=self.device,
                dtype=ledger_dtype,
            )
            carbon_terms.append(
                molar_flux * molecular_weight * carbon_fraction
            )
        expected_carbon = unit_converter.convert(
            torch.stack(carbon_terms).sum(dim=0),
            variable_name="carbon_flux",
            source_units="g C m-2 s-1",
            target_units="kg C m-2 s-1",
        ).to(dtype=ledger_dtype)
        residual = allocation - expected_carbon
        return {
            "carbon_boundary": float(torch.max(torch.abs(residual)).detach().item())
        }

    def _get_carbon_fraction(self, compound: str) -> Tensor:
        """Return per-compound carbon mass fraction tensor."""
        buffer_name = self._carbon_fraction_buffers.get(compound)
        if not isinstance(buffer_name, str):
            raise KeyError(f"No carbon fraction is registered for '{compound}'")
        buffer_value = getattr(self, buffer_name)
        if not isinstance(buffer_value, Tensor):
            raise TypeError(f"Carbon fraction for '{compound}' is not a tensor")
        return buffer_value
    
    def _compute_standard_emission(
        self,
        compound: str,
        temperature: Tensor,
        ppfd: Tensor,
        foliar_density: Tensor,
        optimum_history: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """
        Compute the reduced RELIC dry-mass-normalized emission rate.
        
        References
        ----------
        Guenther et al. (2012), Eq. (1)-(4), reduced instantaneous form.

        Mathematical expression
        -----------------------
        rho_f = LAI * k_f.
        E_i = epsilon_i * gamma_T,i * gamma_P,i * rho_f / seconds_per_hour.

        Parameter meanings and units
        ----------------------------
        epsilon_i is a RELIC dry-leaf-mass emission potential
        [ug g-1 DW h-1], gamma terms are dimensionless, k_f is the effective
        dry-leaf mass per canonical LAI leaf area [g DW m-2], rho_f is the
        resulting dry-leaf mass per ground area [g DW m-2], and E_i is the
        internal species mass flux [ug m-2 s-1]. The epsilon_i and k_f product
        is not separately identifiable from flux alone. Carbon leaves by the
        public atmospheric species and carbon-export ports.
        """
        # References: Guenther et al. (2012); Niinemets et al. (2010).
        # Mathematical expression: E_i=epsilon_i*gamma_T_i*gamma_P_i*rho_f/seconds_per_hour.
        # Parameter meanings: epsilon_i is the selected PFT or default dry-leaf emission factor;
        #   rho_f is the canonical LAI-derived dry-leaf density.
        # Units: epsilon [ug g-1 DW h-1]; rho_f [g DW m-2]; E_i [ug m-2 s-1].
        # Domain/constraints: This explicit RELIC dry-mass bridge is not the full ground-area MEGAN
        #   canopy-history law; enabled compounds and their coefficient contracts apply.
        # Sign convention: E_i is a nonnegative atmospheric emission magnitude.
        buffer_name = self._pft_emission_factor_buffers.get(compound)
        if buffer_name is None:
            emission_factor_tensor = self.emission_factors[compound]
        else:
            emission_factor_tensor = getattr(self, buffer_name)
            if not isinstance(emission_factor_tensor, Tensor):
                raise TypeError(
                    f"PFT emission factor for '{compound}' is not a tensor"
                )
        emission_factor_tensor = emission_factor_tensor.to(
            dtype=temperature.dtype, device=temperature.device
        )
        
        # Temperature response
        temp_factor = self._get_temperature_factor(compound, temperature, optimum_history)
        
        # Light response (compound-specific)
        light_factor = self._get_light_factor(compound, ppfd)
        
        # Calculate the reduced mass-normalized emission rate.
        # [ug g-1 DW h-1] x [g DW m-2] / [s h-1] = [ug m-2 s-1]
        seconds_per_hour = cast(Tensor, self.seconds_per_hour).to(
            dtype=temperature.dtype, device=temperature.device
        )
        emission_rate = (
            emission_factor_tensor
            * temp_factor
            * light_factor
            * foliar_density
            / seconds_per_hour
        )
        return emission_rate

    def _project_photosynthesis_carbon(
        self,
        net_assimilation: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Project net assimilation onto available carbon and expose the residual.

        References
        ----------
        Guenther et al. (2012), MEGAN2.1 emission activity framework.
        Monson et al. (2012), carbon allocation constraints on isoprenoids.

        Mathematical expression
        -----------------------
        A_available = max(A_n, A_min);
        R_floor = A_available - A_n >= 0.

        Parameter meanings and units
        ----------------------------
        A_n is net assimilation [umol C m-2 s-1], A_min is the
        ParameterManager-backed carbon_flux_min [umol C m-2 s-1], and
        R_floor is the explicitly routed lower-bound projection residual in
        the same units.
        """
        # References: Existing RELIC carbon availability projection; Guenther et al. (2012)
        #   coupling context.
        # Mathematical expression: A_available=max(A_net,A_floor); residual=A_available-A_net.
        # Parameter meanings: A_floor is validation.numerical.carbon_flux_min; the residual records
        #   the exact projection amount.
        # Units: All three terms [umol CO2 m-2 s-1].
        # Domain/constraints: A_floor must be one finite scalar; this projection is numerical
        #   availability accounting, not production of new plant carbon.
        # Sign convention: A positive residual discloses a raised availability value; it is not a
        #   physical balancing input.

        carbon_flux_min = self.get_parameter_value(
            "carbon_flux_min",
            namespace="validation.numerical",
            required=True,
        )
        floor = torch.as_tensor(
            carbon_flux_min,
            device=net_assimilation.device,
            dtype=net_assimilation.dtype,
        )
        if floor.numel() != 1 or not bool(torch.isfinite(floor).all()):
            raise ValueError("carbon_flux_min must be one finite scalar")
        floor = floor.reshape(())
        available_carbon = torch.maximum(net_assimilation, floor)
        floor_residual = available_carbon - net_assimilation
        return available_carbon, floor_residual
    
    def _compute_photosynthesis_coupled_emission(
        self, 
        compound: str, 
        available_carbon: Tensor,
        temperature: Tensor,
        optimum_history: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """
        Compute photosynthesis-coupled BVOC emission.
        
        References
        ----------
        Monson et al. (2012), carbon-allocation constraint on isoprenoid emission.

        Mathematical expression
        -----------------------
        E_i = max(A_n, 0) * f_i * MW_i / n_C,i * gamma_T,i.

        Parameter meanings and units
        ----------------------------
        A_n is net assimilation [umol C m-2 s-1], f_i is allocated carbon
        fraction, MW_i is [g mol-1], n_C,i is carbon atoms per molecule, and
        E_i is [ug compound m-2 s-1]. The equality follows because umol times
        g mol-1 equals ug.
        """
        # Get compound parameters dynamically from parameter manager
        allocation = getattr(self, f"f_carbon_{compound}")
        if not isinstance(allocation, Tensor):
            raise TypeError(f"Carbon allocation for '{compound}' is not a tensor")
        carbon_fraction = allocation.to(
            dtype=temperature.dtype, device=temperature.device
        )
        molecular_weight = torch.as_tensor(
            self.molecular_weights[compound],
            dtype=torch.float32,
            device=temperature.device,
        ).to(dtype=temperature.dtype)
        carbon_number = self.carbon_atoms[compound]
        
        # Carbon allocation to BVOC synthesis
        allocated_carbon = available_carbon * carbon_fraction
        
        # Convert carbon flux to BVOC mass flux
        # umol CO2 m-2 s-1 -> ug BVOC m-2 s-1
        bvoc_emission = allocated_carbon * (molecular_weight / carbon_number)
        
        # Apply temperature modifier
        temp_factor = self._get_temperature_factor(compound, temperature, optimum_history)
        
        return bvoc_emission * temp_factor
    
    def _table4_value(self, key: str, units: str, like: Tensor) -> Tensor:
        """Resolve one Table 4 or Table 6 coefficient in the dtype and device of ``like``."""
        buffer_name = self._table4_buffers.get(key)
        if buffer_name is not None:
            value = getattr(self, buffer_name)
        else:
            value = self.get_parameter_value(
                key, namespace="biosphere.bvoc.megan", units=units, required=True
            )
        if isinstance(value, Tensor):
            return value.to(dtype=like.dtype, device=like.device)
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    def _get_temperature_factor(
        self,
        compound: str,
        temperature: Tensor,
        optimum_history: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Evaluate the declared reduced MEGAN temperature activity factor.

        References
        ----------
        Guenther et al. (2012), Eqs. (7), (8), and (11) and Table 4.

        Mathematical expression
        -----------------------
        ``gamma_T = (1 - LDF) exp(beta (T - T_s)) + LDF Eopt CT2 exp(CT1 x) /
        (CT2 - CT1 (1 - exp(CT2 x)))`` with ``x = (1/Topt - 1/T)/R`` and the
        coefficients of the class's Table 4 row (LDF = 1 for isoprene). The lumped
        OVOC class is the Table 6 emission-weighted mixture of its component rows.

        Parameter meanings and units
        ----------------------------
        T, T_s and Topt are [K], CT1 and CT2 are [J mol-1], R is
        [J mol-1 K-1], beta is [K-1], and each returned factor is dimensionless.
        ``optimum_history`` is ``(T_opt, h)`` of Eqs. (9)-(10) from the carried
        history (megan_optimum_history); None keeps the fixed standard history
        (T_opt = biosphere.bvoc.megan.topt, E_opt = C_eo).
        """
        # References: Guenther et al. (2012) Geosci. Model Dev. 5, 1471-1492, Eqs. (7),
        #   (8), (11) and Table 4.
        # Mathematical expression: gamma_T = (1 - LDF) exp(beta (T - T_s)) + LDF gamma_T_LDF,
        #   gamma_T_LDF = Eopt CT2 exp(CT1 x) / (CT2 - CT1 (1 - exp(CT2 x))),
        #   x = (1/Topt - 1/T)/R, evaluated by megan_class_temperature_activity; the lumped
        #   OVOC class takes sum_j w_j gamma_T,j over its component rows.
        # Parameter meanings: LDF, beta, CT1 and Eopt come from the Table 4 row
        #   (MEGAN_TABLE4_ROWS); w_j are the Table 6 emission shares; Topt and CT2 are
        #   class independent; R is the physical-constant gas constant and T_s the
        #   standard temperature of the emission factors.
        # Units: T, Topt, T_s [K]; CT1, CT2 [J mol-1]; R [J mol-1 K-1]; beta [K-1];
        #   LDF, Eopt, gamma_T [1].
        # Domain/constraints: supported compound and positive physical temperature; the
        #   Eq. (8) denominator must be positive; no clamp is applied.
        # Sign convention: Activity is nonnegative in its physical domain and creates no
        #   independent source.
        if compound not in self.supported_compounds:
            raise ValueError(f"Unsupported compound '{compound}'")
        temperature_tensor = temperature.to(device=self.device)
        standard_temperature = cast(Tensor, self.temp_std).to(
            dtype=temperature_tensor.dtype, device=temperature_tensor.device
        )
        if optimum_history is None:
            topt = cast(Tensor, self.topt).to(
                dtype=temperature_tensor.dtype, device=temperature_tensor.device
            )
            eopt_history_factor: Optional[Tensor] = None
        else:
            topt, eopt_history_factor = optimum_history
        return megan_class_temperature_activity(
            compound,
            temperature_tensor,
            standard_temperature,
            topt,
            cast(Tensor, self.ct2).to(
                dtype=temperature_tensor.dtype, device=temperature_tensor.device
            ),
            cast(Tensor, self.gas_constant).to(
                dtype=temperature_tensor.dtype, device=temperature_tensor.device
            ),
            lambda key, units: self._table4_value(key, units, temperature_tensor),
            eopt_history_factor=eopt_history_factor,
        )

    def _get_light_factor(self, compound: str, ppfd: Tensor) -> Tensor:
        """Evaluate the reduced MEGAN PPFD activity factor.

        References
        ----------
        Guenther et al. (2012), Eqs. (3) and (4) and Table 4.

        Mathematical expression
        -----------------------
        gamma_P = (1 - LDF) + LDF * alpha * CL * PPFD / sqrt(1 + alpha^2 * PPFD^2).

        Parameter meanings and units
        ----------------------------
        alpha is the PPFD response slope [m2 s umol-1], CL and LDF are
        dimensionless, and PPFD is [umol photons m-2 s-1]. Isoprene is fully light
        dependent (LDF = 1); monoterpenes and sesquiterpenes use the LDF of their Table 4
        row, and the lumped OVOC class is the Table 6 emission-weighted mixture of its
        methanol, acetone and acetaldehyde rows.
        """
        # References: Guenther et al. (2012), Eqs. 3 and 4 and Tables 4 and 6.
        # Mathematical expression: gamma_P,j=(1-LDF_j)+LDF_j*alpha*CL*Q/sqrt(1+alpha^2*Q^2);
        #   LDF=1 for isoprene (Eq. 4 alone); gamma_P = sum_j w_j gamma_P,j.
        # Parameter meanings: Q is PPFD; alpha the PPFD response slope; CL the existing
        #   light-response coefficient; LDF_j the light-dependent fraction of Table 4 row j;
        #   w_j the Table 6 emission share of component j.
        # Units: Q [umol photons m-2 s-1]; alpha [m2 s umol-1]; CL,LDF_j,w_j,gamma_P [1].
        # Domain/constraints: Supported compound and nonnegative physical PPFD; this reduced
        #   selection does not model full canopy radiation or temporal light history. The
        #   law is the shared module-level megan_class_light_activity.
        # Sign convention: Nonnegative light activity multiplies the emission magnitude.
        if compound not in self.supported_compounds:
            raise ValueError(f"Unsupported compound '{compound}'")
        alpha = cast(Tensor, self.alpha_light).to(dtype=ppfd.dtype, device=ppfd.device)
        cl_factor = cast(Tensor, self.cl_light).to(dtype=ppfd.dtype, device=ppfd.device)
        return megan_class_light_activity(
            compound,
            ppfd,
            alpha,
            cl_factor,
            lambda key, units: self._table4_value(key, units, ppfd),
        )

    def _compute_co2_inhibition_factor(self, atmospheric_co2_ppm: Tensor) -> Tensor:
        """Evaluate the Guenther et al. (2012) isoprene CO2 response.

        References
        ----------
        Guenther et al. (2012), Eq. (14); Heald et al. (2009); Wilkinson et al.
        (2009).

        Mathematical expression
        -----------------------
        Ci = r_Ci * Ca and gamma_CO2 = ISmax - ISmax * Ci^h /
        (Cstar^h + Ci^h).

        Parameter meanings and units
        ----------------------------
        Ca and Ci are ambient and estimated intercellular CO2 [ppm], r_Ci is
        dimensionless, Cstar is [ppm], h is the Hill exponent, and ISmax and
        gamma_CO2 are dimensionless.
        """
        # References: Guenther et al. (2012), Eq. 14; Heald et al. (2009); Wilkinson et al. (2009).
        # Mathematical expression: Ci=r_Ci*Ca; gamma_CO2=ISmax-ISmax*Ci^h/(Cstar^h+Ci^h).
        # Parameter meanings: r_Ci is the fixed intercellular-to-ambient ratio, h the Hill
        #   exponent, and Cstar the response concentration scale.
        # Units: Ca,Ci,Cstar [ppm]; ISmax,r_Ci,h,gamma_CO2 [1].
        # Domain/constraints: Finite nonnegative Ca and valid positive response parameters; this is
        #   a proxy Ci relation rather than a stomatal solution.
        # Sign convention: Increasing CO2 suppresses the isoprene multiplier within the declared
        #   parameter domain.
        parameter_names = (
            "ismax",
            "c_star",
            "hill_exponent",
            "intercellular_to_ambient_ratio",
        )
        parameter_values = {
            name: self._ensure_tensor(
                self.get_parameter_value(
                    name, namespace="biosphere.bvoc.co2_inhibition", required=True
                ),
                dtype=torch.float32,
            ).to(dtype=atmospheric_co2_ppm.dtype, device=atmospheric_co2_ppm.device)
            for name in parameter_names
        }
        intercellular_co2 = (
            parameter_values["intercellular_to_ambient_ratio"]
            * atmospheric_co2_ppm
        )
        hill = parameter_values["hill_exponent"]
        intercellular_power = intercellular_co2.pow(hill)
        c_star_power = parameter_values["c_star"].pow(hill)
        ismax = parameter_values["ismax"]
        return ismax - ismax * intercellular_power / (
            c_star_power + intercellular_power
        )
    
    def _extract_tensor(
        self, 
        state: Dict[str, Any], 
        key: str,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Extract a required canonical input and move it to the functor device."""
        if key not in state:
            raise ValueError(f"Required input '{key}' not found in state")
        value = state[key]
        
        if not isinstance(value, Tensor):
            target_dtype = torch.float32 if dtype is None else dtype
            if target_dtype not in (torch.float32, torch.float64):
                raise TypeError(
                    f"Input '{key}' dtype must be torch.float32 or torch.float64, got {target_dtype}"
                )
            value = torch.tensor(value, dtype=target_dtype, device=self.device)
        else:
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError(
                    f"Input '{key}' dtype must be torch.float32 or torch.float64, got {value.dtype}"
                )
            if dtype is not None and value.dtype != dtype:
                raise TypeError(
                    f"Input '{key}' dtype {value.dtype} does not match expected {dtype}"
                )
            value = value.to(device=self.device, dtype=value.dtype)
            
        return value
    
    def _validate_inputs(
        self,
        air_temperature: Tensor,
        ppfd: Tensor,
        leaf_area_index: Tensor,
        atmospheric_co2_ppm: Tensor,
        net_photosynthesis: Optional[Tensor],
        stress_factors: Dict[str, Tensor],
        temperature_history: Optional[Dict[str, Tensor]] = None,
    ) -> None:
        """Fail fast on non-finite or out-of-domain canonical inputs."""

        def require_finite(name: str, value: Tensor) -> None:
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")

        # air_temperature holds the declared temperature input (temperature_source); the
        # messages name that input.
        temperature_name = self.temperature_source
        require_finite(temperature_name, air_temperature)
        min_temperature = cast(Tensor, self.min_operational_temperature).to(
            dtype=air_temperature.dtype, device=air_temperature.device
        )
        max_temperature = cast(Tensor, self.max_operational_temperature).to(
            dtype=air_temperature.dtype, device=air_temperature.device
        )
        if torch.any(
            (air_temperature < min_temperature)
            | (air_temperature > max_temperature)
        ):
            raise ValueError(f"{temperature_name} is outside the configured operational range")

        require_finite("ppfd", ppfd)
        if torch.any(ppfd < 0.0):
            raise ValueError("ppfd must be non-negative")

        require_finite("leaf_area_index", leaf_area_index)
        if torch.any(leaf_area_index < 0.0):
            raise ValueError("leaf_area_index must be non-negative")

        require_finite("atmospheric_co2_ppm", atmospheric_co2_ppm)
        if torch.any(atmospheric_co2_ppm <= 0.0):
            raise ValueError("atmospheric_co2_ppm must be positive")

        if net_photosynthesis is not None:
            require_finite("net_photosynthesis", net_photosynthesis)

        for name, factor in stress_factors.items():
            require_finite(name, factor)
            if torch.any((factor < 0.0) | (factor > 1.0)):
                raise ValueError(f"{name} must be within [0, 1]")

        # T24 and T240 are means of the validated air temperature, so they share its
        # operational range.
        for name, history in (temperature_history or {}).items():
            require_finite(name, history)
            if torch.any((history < min_temperature) | (history > max_temperature)):
                raise ValueError(f"{name} is outside the configured operational range")


# Export main classes
__all__ = [
    'MEGANModel',
    'CouplingStrategy',
    'EmissionPathway'
]

