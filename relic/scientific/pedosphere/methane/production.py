"""Methane production functor blending the iMAPLE law with a calibrated Rh-based branch."""

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import StateDict
from relic.computation.operators.iphs import advance_reaction_network
from relic.scientific.pedosphere.api_contract import PedosphereContractMixin
from relic.utils.physical_constants import get_physical_constant

class MethaneProductionFunctor(PedosphereContractMixin, BiologicalProcessFunctor):
    """Compute wetland methane production from an iMAPLE branch and a calibrated branch.

    Scientific references:
        - Yue et al. (2024) doi:10.5194/gmd-17-4621-2024, Eqs. 34-38
        - Zhu et al. (2014) doi:10.5194/gmd-7-981-2014, Eqs. 2-6
        - Wania et al. (2010) doi:10.5194/gmd-3-565-2010 (release ratio r)

    Governing equations (ASCII form):
        P_iMAPLE = Rh * r * f_T(Ts) * f_pH(pH) * f_wet
        f_T(Ts) = 0                      if Ts < Tmin or Ts > Tmax
                = v_t**x_t * exp(x_t * (1 - v_t))   otherwise
            where v_t = (Tmax - Ts) / (Tmax - Topt)
                  Q10(Ts) = clip(r_b * Q_b**((Ts - T_base) / T_interval),
                                 Q10_min, Q10_max)
                  a_t = 1 + A_t / (log(Q10(Ts)) * (Tmax - Topt))
                  x_t = (log(Q10(Ts)) * (Tmax - Topt))**2
                        * (1 + sqrt(a_t))**2 / N_t
        f_pH(pH) = 0                                    outside [pH_low, pH_high]
                 = ((pH - pH_low) / (pH_opt - pH_low))
                   * ((pH_high - pH) / (pH_high - pH_opt))**a_pH otherwise
            where a_pH = (pH_high - pH_opt) / (pH_opt - pH_low)
        f_wet = wetland inundation fraction (continuous, no activation floor)

        f_Eh(Eh) = clip((Eh_0 - Eh) / (Eh_0 - Eh_1), 0, 1)
        P_cal = Rh * r * f_T(Ts) * f_Eh(Eh) * A

        Hybrid production: P = (1 - w) * P_iMAPLE + w * P_cal

    Parameter meanings:
        Rh: heterotrophic respiration flux [g C m-2 s-1]
        r: methane release ratio [-]
        r_b (q10): Q10 at the base temperature, the base Q10 3.0 of Zhu et al. (2014) [-]
        Q_b: rise of Q10 per T_interval above T_base (Yue et al. 2024 Eq. 39) [-]
        T_base: base temperature of Eq. 39 (25 C) [K]
        Q10_min/Q10_max: clip of Q10(Ts), the observed 1.7-16 range of Zhu et al. (2014) [-]
        T_interval: fixed Q10 temperature interval from physical constants [K]
        Tmin/Topt/Tmax: temperature response thresholds [K]
        A_t: iMAPLE temperature-shape coefficient [K]
        N_t: iMAPLE squared-temperature normalization [K2]
        pH_low/pH_opt/pH_high: pH response limits and optimum [-]
        f_wet: wetland inundation fraction [-]
        Eh_1/Eh_0: full-activity (-200 mV) and zero-activity (-100 mV) redox
            potentials of the piecewise-linear factor of Zhu et al. (2014) [mV]
        A: DNDC95 microbial activity index, an activity fraction in [0, 1] [-]
        w: calibrated-branch weight provided by the scale functor [-]

    Notes:
        P_iMAPLE follows Yue et al. (2024) Eq. 34 with the TRIPLEX-GHG temperature
        and pH factors (Zhu et al. 2014, Eqs. 3-6) and the temperature-dependent Q10
        of Yue et al. (2024) Eq. 39 (r_b = 3.0, Q_b = 1.33, T_base = 25 C), clipped to
        the observed 1.7-16 range of Zhu et al. (2014) as in the released iMAPLE code.
        The exponent x_t is the published Zhu et al. (2014) Eq. 5; the released iMAPLE
        Fortran evaluates it as ln(Q10)^2 + 2.05 by operator precedence, which is not
        reproduced.
        P_cal is an Rh-based empirical RELIC expression with DNDC-type Q10 and
        redox controls. It is not the DNDC substrate-kinetic methanogenesis model
        (acetate and H2/CO2 Michaelis-Menten kinetics; Fumoto et al. 2008,
        doi:10.1111/j.1365-2486.2007.01475.x; van Bodegom and Scholten 2001,
        doi:10.1016/S0016-7037(01)00563-4), which is not implemented. Its redox
        factor is the piecewise-linear form of Zhu et al. (2014) Sect. 2.2.1,
        after Zhang et al. (2002).
        Fluxes are represented on a CH4-C basis [g C m-2 s-1]. If CH4 mass is
        needed for downstream coupling, use: F_CH4 = F_CH4_C * (M_CH4 / M_C),
        where M_CH4 and M_C are methane and carbon molar masses from the
        physical constants database.
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(
        self,
        name: str = "pedosphere_methane_production",
        device: Optional[torch.device] = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Compute methane production by blending the iMAPLE branch with an"
                " Rh-based empirical branch that applies DNDC-type Q10 and redox controls"
                " (not the DNDC substrate-kinetic model)"
            ),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "P = (1 - w) * Rh * r * f_T(Ts) * f_pH(pH) * f_wet"
                " + w * Rh * r * f_T(Ts) * f_Eh(Eh) * A;"
                " Q10(Ts) = clip(r_b * Q_b^((Ts - T_base)/10), Q10_min, Q10_max)"
            ),
            references=[
                "Yue et al. (2024) doi:10.5194/gmd-17-4621-2024",
                "Zhu et al. (2014) doi:10.5194/gmd-7-981-2014",
                "Wania et al. (2010) doi:10.5194/gmd-3-565-2010",
                "Zhang et al. (2002) doi:10.1029/2001GB001838",
            ],
            parameters={
                "pedosphere.methane.production.release_ratio": "Methane to CO2 release ratio [-]",
                "pedosphere.methane.production.q10": "Q10 at the base temperature, r_b of Yue et al. (2024) Eq. 39 [-]",
                "pedosphere.methane.production.q10_temperature_sensitivity": (
                    "Rise of the production Q10 per 10 K, Q_b of Yue et al. (2024) Eq. 39 [-]"
                ),
                "pedosphere.methane.production.q10_base_temperature": (
                    "Base temperature of Yue et al. (2024) Eq. 39 [K]"
                ),
                "pedosphere.methane.production.q10_min": "Lower clip of the production Q10 [-]",
                "pedosphere.methane.production.q10_max": "Upper clip of the production Q10 [-]",
                "pedosphere.methane.production.temperature_min": "Lower activation threshold [K]",
                "pedosphere.methane.production.temperature_optimum": "Optimal temperature [K]",
                "pedosphere.methane.production.temperature_max": "Upper activation threshold [K]",
                "pedosphere.methane.production.ph_optimum": "Optimal soil pH [-]",
                "pedosphere.methane.production.ph_lower_limit": "Lower soil pH limit [-]",
                "pedosphere.methane.production.ph_upper_limit": "Upper soil pH limit [-]",
                "pedosphere.methane.production.calibrated.eh_full_activity": "Redox potential at and below which the calibrated-branch redox factor is 1 [mV]",
                "pedosphere.methane.production.calibrated.eh_zero_activity": "Redox potential at and above which the calibrated-branch redox factor is 0 [mV]",
                "pedosphere.numerical.minimum_temperature_span": "Minimum temperature span for response calculations [K]",
                "pedosphere.numerical.minimum_dimensionless": "Minimum dimensionless value for methane numerical denominators [-]",
            },
            input_variables=[
                "soil_temperature",
                "soil_ph",
                "heterotrophic_respiration",
                "redox_potential",
                "land_surface.wetland_fraction",
                "microbial_activity_index",
                "methane_gating_weight",
            ],
            output_variables=[
                "ch4_production_rate",
                "ch4_storage_tendency_prod",
                "diagnostics.methane_production.imaple_rate",
                "diagnostics.methane_production.calibrated_rate",
            ],
            conservation_laws=["mass"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
        )
        super().__init__(name=name, metadata=metadata, device=device)

        # Core iMAPLE parameter set
        self.release_ratio = self.get_parameter_value(
            "release_ratio",
            namespace="pedosphere.methane.production", required=True)
        self.q10 = self.get_parameter_value(
            "q10",
            namespace="pedosphere.methane.production", required=True)
        self.q10_sensitivity = self.get_parameter_value(
            "q10_temperature_sensitivity",
            namespace="pedosphere.methane.production", required=True)
        self.q10_base_temperature = self.get_parameter_value(
            "q10_base_temperature",
            namespace="pedosphere.methane.production", required=True)
        self.q10_min = self.get_parameter_value(
            "q10_min",
            namespace="pedosphere.methane.production", required=True)
        self.q10_max = self.get_parameter_value(
            "q10_max",
            namespace="pedosphere.methane.production", required=True)
        if not bool(torch.all(self.q10_min > 1.0)):
            raise ValueError(
                "pedosphere.methane.production.q10_min must exceed 1: ln(Q10) enters the "
                "Zhu et al. (2014) exponent as a divisor."
            )
        if not bool(torch.all(self.q10_max > self.q10_min)):
            raise ValueError(
                "pedosphere.methane.production.q10_max must exceed q10_min."
            )
        self.q10_interval = get_physical_constant(
            "relic.physical.constants.temperature.q10_interval"
        )
        self.temperature_shape_at_coefficient = 4.0 * self.q10_interval
        self.temperature_shape_normalization = (2.0 * self.q10_interval) ** 2
        self.temperature_min = self.get_parameter_value(
            "temperature_min",
            namespace="pedosphere.methane.production", required=True)
        self.temperature_optimum = self.get_parameter_value(
            "temperature_optimum",
            namespace="pedosphere.methane.production", required=True)
        self.temperature_max = self.get_parameter_value(
            "temperature_max",
            namespace="pedosphere.methane.production", required=True)
        self.ph_optimum = self.get_parameter_value(
            "ph_optimum",
            namespace="pedosphere.methane.production", required=True)
        self.ph_lower_limit = self.get_parameter_value(
            "ph_lower_limit",
            namespace="pedosphere.methane.production", required=True)
        self.ph_upper_limit = self.get_parameter_value(
            "ph_upper_limit",
            namespace="pedosphere.methane.production", required=True)
        self.minimum_temperature_span = self.get_parameter_value(
            "minimum_temperature_span",
            namespace="pedosphere.numerical", required=True)
        self.minimum_dimensionless = self.get_parameter_value(
            "minimum_dimensionless",
            namespace="pedosphere.numerical", required=True)
        # Calibrated-branch parameters
        self.eh_full_activity = self.get_parameter_value(
            "calibrated.eh_full_activity",
            namespace="pedosphere.methane.production", required=True)
        self.eh_zero_activity = self.get_parameter_value(
            "calibrated.eh_zero_activity",
            namespace="pedosphere.methane.production", required=True)
        if not bool(torch.all(self.eh_zero_activity > self.eh_full_activity)):
            raise ValueError(
                "pedosphere.methane.production.calibrated.eh_zero_activity must exceed "
                "calibrated.eh_full_activity."
            )

        # Production depends on gating weights from scale functor
        self.add_dependency("pedosphere_methane_scale")

    def forward(
        # References: Yue et al. (2024), doi:10.5194/gmd-17-4621-2024, Eq. 34 (P_i);
        #   Zhu et al. (2014), doi:10.5194/gmd-7-981-2014, Eqs. 2-6 and Sect. 2.2.1 (f_Eh);
        #   P_c is a RELIC convention (Rh-based empirical branch with DNDC-type
        #   controls, not the DNDC substrate-kinetic model, which is not implemented).
        # Mathematical expression: P_i=Rh*r*f_T*f_pH*f_wet;
        #   P_c=Rh*r*f_T*f_Eh*A with the same f_T as P_i;
        #   P=max((1-w)*P_i+w*P_c,0).
        # Parameter meanings: w is the injected calibrated-branch weight; Rh is
        #   heterotrophic respiration; r is the CH4:CO2 release ratio; f_wet is the
        #   continuous wetland inundation fraction; A the microbial activity index.
        # Units: P_i,P_c,P,Rh [g C m-2 s-1]; T [K]; r,w,A,f_* [1].
        # Domain/constraints: Both candidates are evaluated before blending; required
        #   forcing is finite and fractions lie in [0,1]; no activation floor applies.
        # Sign convention: Production is positive source; the matching +P tendency is
        #   a port alias, not a second input. MethaneBudgetCoupler alone advances storage.
        self,
        state: StateDict,
        time: float,
        dt: Tensor | float,
        **kwargs,
    ) -> Dict[str, UpdateItem]:
        _ = dt  # retained for interface consistency; production returns rate tendencies

        soil_temperature = self._require_state_tensor(state, "soil_temperature")
        soil_potential_h = self._require_state_tensor(state, "soil_ph")
        rh_flux = self._require_state_tensor(
            state,
            "heterotrophic_respiration",
            lower_bound=0.0,
        )
        wetland_fraction = self._require_state_tensor(
            state,
            "land_surface.wetland_fraction",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        gating_weight = self._require_state_tensor(
            state,
            "methane_gating_weight",
            lower_bound=0.0,
            upper_bound=1.0,
        )

        # References: Zhu et al. (2014), doi:10.5194/gmd-7-981-2014, Eqs. 3-5
        #   (the TRIPLEX-GHG factor adopted by Yue et al. (2024) Eqs. 35-38) with the
        #   temperature-dependent Q10 of Yue et al. (2024) Eq. 39.
        # Equation: f_T(Ts) = vt^xt * exp(xt * (1 - vt)) within [Tmin, Tmax].
        # Parameters: Tmin, Topt, Tmax [K], Q10(Ts) [-], q10_interval [K].
        temperature_factor_opt = self._optimal_temperature_response(soil_temperature)
        # References: Zhu et al. (2014), doi:10.5194/gmd-7-981-2014, Eq. 6.
        # Equation: f_pH = L * U^a within [pH_low, pH_high], otherwise zero.
        # Parameters: pH_low, pH_opt, pH_high [-].
        ph_factor = self._ph_factor(soil_potential_h)
        # References: Yue et al. (2024), doi:10.5194/gmd-17-4621-2024, Eq. 34.
        # Mathematical expression: f_wet = wetland_fraction;
        #   P_iMAPLE = Rh * r * f_T(Ts) * f_pH(pH) * f_wet.
        # Parameter meanings: Rh is heterotrophic respiration; r the CH4:CO2 release
        #   ratio; f_T and f_pH the temperature and pH factors; f_wet the wetland
        #   inundation fraction, used continuously (no activation floor).
        # Units: Rh, P_iMAPLE [g C m-2 s-1]; r, f_T, f_pH, f_wet [1].
        # Domain/constraints: Rh >= 0 and f_wet in [0, 1] are enforced on input;
        #   the product is clamped nonnegative.
        # Sign convention: P_iMAPLE is a positive methane-carbon source.
        wetland_factor = wetland_fraction
        imaple_flux = (
            rh_flux
            * self.release_ratio
            * temperature_factor_opt
            * ph_factor
            * wetland_factor
        )
        imaple_flux = torch.clamp(imaple_flux, min=0.0)

        # Calibrated-branch drivers
        redox_potential = self._require_state_tensor(state, "redox_potential")
        microbial_activity = self._require_state_tensor(
            state,
            "microbial_activity_index",
            lower_bound=0.0,
            upper_bound=1.0,
        )

        # References: Zhu et al. (2014) Sect. 2.2.1 after Zhang et al. (2002); see _redox_gate.
        # Equation: f_Eh(Eh) = clip((Eh_0 - Eh) / (Eh_0 - Eh_1), 0, 1)
        # Parameters: Eh_1 = -200 mV (full activity), Eh_0 = -100 mV (zero activity)
        redox_gate = self._redox_gate(redox_potential)
        # References: Zhu et al. (2014), doi:10.5194/gmd-7-981-2014, Eqs. 3-5 for f_T
        #   (the same peak factor as the iMAPLE branch, evaluated once above); the
        #   DNDC95 reference code (Dndc_main.cpp, Dndc_CH4.cpp) for the microbial
        #   activity index A, which multiplies methanogenesis directly and never
        #   exceeds 1; the Rh-based product is a RELIC convention with DNDC-type
        #   controls, not the DNDC substrate-kinetic methanogenesis model.
        # Mathematical expression: P_cal = Rh * r * f_T(Ts) * f_Eh(Eh) * A.
        # Parameter meanings: Rh heterotrophic respiration; r the CH4:CO2 release
        #   ratio; f_T the Zhu et al. (2014) peak temperature factor (maximum 1 at
        #   Topt); f_Eh the piecewise-linear redox factor; A the microbial activity
        #   index, an activity fraction.
        # Units: Rh, P_cal [g C m-2 s-1]; r, f_T, f_Eh, A [1].
        # Domain/constraints: A in [0, 1] is enforced on input; every factor lies
        #   in [0, 1], so P_cal <= r * Rh <= Rh without truncation.
        # Sign convention: P_cal is a positive methane-carbon source.
        calibrated_flux = (
            rh_flux
            * self.release_ratio
            * temperature_factor_opt
            * redox_gate
            * microbial_activity
        )
        calibrated_flux = torch.clamp(calibrated_flux, min=0.0)

        # References: RELIC convention (hybrid blend; not part of Yue et al. (2024)).
        # Mathematical expression: P = max((1 - w) * P_iMAPLE + w * P_cal, 0).
        # Parameter meanings: w is the calibrated-branch weight from
        #   pedosphere_methane_scale.
        # Units: P, P_iMAPLE, P_cal [g C m-2 s-1]; w [1].
        # Domain/constraints: w in [0, 1] is enforced on input.
        # Sign convention: P is a positive methane-carbon source.
        production_rate = (self._tensor(1.0) - gating_weight) * imaple_flux + gating_weight * calibrated_flux
        production_rate = torch.clamp(production_rate, min=0.0)

        # References: Yue et al. (2024); Feinberg (2019).
        # Mathematical expression: x_dot = N_h r_h for methane production and storage tendency.
        # Parameter meanings: N_h maps production-rate channels to diagnostic and storage
        # tendencies; r_h contains ParameterManager-backed iMAPLE and calibrated-branch rates.
        production_network = advance_reaction_network(
            name="methane_production_network",
            species=(
                "ch4_production_rate",
                "ch4_storage_tendency_prod",
                "diagnostics.methane_production.imaple_rate",
                "diagnostics.methane_production.calibrated_rate",
            ),
            reactions=(
                "combined_methane_production",
                "storage_production_tendency",
                "imaple_methane_production",
                "calibrated_methane_production",
            ),
            stoichiometric_N=torch.eye(
                4,
                device=production_rate.device,
                dtype=production_rate.dtype,
            ),
            rates=(
                production_rate,
                production_rate,
                imaple_flux,
                calibrated_flux,
            ),
            **self._iphs_reaction_conservation_kwargs(open_system_route="port"),
        )
        production_rate = production_network.species_rate("ch4_production_rate")
        storage_tendency = production_network.species_rate("ch4_storage_tendency_prod")
        imaple_flux = production_network.species_rate(
            "diagnostics.methane_production.imaple_rate"
        )
        calibrated_flux = production_network.species_rate(
            "diagnostics.methane_production.calibrated_rate"
        )

        updates: Dict[str, UpdateItem] = {}
        updates["ch4_production_rate"] = UpdateItem(
            value=production_rate,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            element="C",
            source_functor=self.name,
        )
        # References: Yue et al. (2024) doi:10.5194/gmd-17-4621-2024
        # Equation: dS/dt = P - O - T; here tendency contribution from production is +P
        # Parameters: P is ch4_production_rate [g C m-2 s-1]. This output is
        # an internal interconnection consumed by the budget state owner, not
        # a second carbon boundary input.
        updates["ch4_storage_tendency_prod"] = UpdateItem(
            value=storage_tendency,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )
        updates["diagnostics.methane_production.imaple_rate"] = UpdateItem(
            value=imaple_flux,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )
        updates["diagnostics.methane_production.calibrated_rate"] = UpdateItem(
            value=calibrated_flux,
            mode=UpdateMode.REPLACE,
            unit="g C m-2 s-1",
            per_timestep=False,
            source_functor=self.name,
        )

        self._last_updates = updates
        return updates

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _scalar_from_value(value: Any) -> float:
        """Convert numeric values to Python float without tensor.item/cpu APIs."""
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    def _tensor(self, value: Any) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, device=self.device)

    def _require_state_tensor(
        self,
        state: StateDict,
        key: str,
        *,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> Tensor:
        """Return one required finite tensor and enforce its declared domain."""
        if key not in state:
            raise KeyError(f"Required methane production input '{key}' is missing.")
        value = state[key]
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Methane production input '{key}' must be a torch.Tensor."
            )
        tensor = value.to(self.device)
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(
                f"Methane production input '{key}' must contain only finite values."
            )
        if lower_bound is not None and bool(torch.any(tensor < lower_bound)):
            raise ValueError(
                f"Methane production input '{key}' must be >= {lower_bound}."
            )
        if upper_bound is not None and bool(torch.any(tensor > upper_bound)):
            raise ValueError(
                f"Methane production input '{key}' must be <= {upper_bound}."
            )
        return tensor

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: Dict[str, Tensor],
    ) -> Dict[str, float]:
        """
        Check methane production closure against its storage tendency contribution.

        References:
            - Yue et al. (2024) doi:10.5194/gmd-17-4621-2024
        Mathematical expression:
            violation["mass"] = |dS_prod_dt - P|
        Parameter meanings:
            - ch4_production_rate: methane source flux on a carbon basis [g C m-2 s-1]
            - ch4_storage_tendency_prod: production contribution to storage tendency [g C m-2 s-1]
        """
        if "mass" not in self.metadata.conservation_laws:
            return {}

        production_rate = output_updates.get("ch4_production_rate")
        tendency = output_updates.get("ch4_storage_tendency_prod")
        if production_rate is None or tendency is None:
            return {}

        production_value = self._scalar_from_value(production_rate)
        tendency_value = self._scalar_from_value(tendency)
        return {"mass": abs(tendency_value - production_value)}

    def _temperature_dependent_q10(self, temperature: Tensor) -> Tensor:
        # References: Yue et al. (2024), doi:10.5194/gmd-17-4621-2024, Eq. 39 (r_b = 3.0,
        #   Q_b = 1.33, T_base = 25 C for production); Zhu et al. (2014),
        #   doi:10.5194/gmd-7-981-2014, Sect. 2.3 and Table 2 (observed production Q10
        #   1.7-16); the released iMAPLE code (ch4flux_wetland.f, CH4prod) applies that clip.
        # Mathematical expression: Q10(Ts) = clip(r_b * Q_b^((Ts - T_base)/dT10),
        #   Q10_min, Q10_max).
        # Parameter meanings: r_b the Q10 at T_base (q10); Q_b the rise per dT10
        #   (q10_temperature_sensitivity); T_base (q10_base_temperature); dT10 the
        #   physical-SSOT Q10 interval; Q10_min, Q10_max the clip (q10_min, q10_max).
        # Units: Ts, T_base, dT10 [K]; r_b, Q_b, Q10, Q10_min, Q10_max [1].
        # Domain/constraints: 1 < Q10_min < Q10_max (checked at construction), so
        #   ln(Q10) > 0 in the Zhu et al. (2014) exponent; evaluated elementwise.
        # Sign convention: Q10 >= Q10_min > 1; it only shapes the temperature factor.
        exponent = (temperature - self.q10_base_temperature) / self.q10_interval
        q10 = self.q10 * torch.pow(self.q10_sensitivity, exponent)
        # The clip bounds follow the device and dtype of the evaluated Q10.
        return torch.clamp(
            q10,
            min=torch.as_tensor(self.q10_min, dtype=q10.dtype, device=q10.device),
            max=torch.as_tensor(self.q10_max, dtype=q10.dtype, device=q10.device),
        )

    def _optimal_temperature_response(self, temperature: Tensor) -> Tensor:
        # References: Yue et al. (2024), doi:10.5194/gmd-17-4621-2024, Eqs. 35-39;
        #   Zhu et al. (2014), doi:10.5194/gmd-7-981-2014, Eqs. 3-5.
        # Mathematical expression: d=max(Tmax-Topt,d_min); L=log(Q10(T))*d;
        #   a=1+4*dT10/L; x=L^2*(1+sqrt(a))^2/(2*dT10)^2;
        #   v=(Tmax-T)/d; f_T=v^x*exp(x*(1-v)) on [Tmin,Tmax], otherwise zero.
        # Parameter meanings: Tmin,Topt,Tmax are lower, optimum and upper temperatures;
        #   dT10 is the physical-SSOT Q10 temperature interval; Q10(T) is the clipped
        #   temperature-dependent Q10 of Yue et al. (2024) Eq. 39 (_temperature_dependent_q10).
        #   x is the published Zhu et al. (2014) Eq. 5 exponent, not the ln(Q10)^2 + 2.05
        #   that the released iMAPLE Fortran evaluates by operator precedence.
        # Units: temperatures,d,d_min,dT10,L [K]; a,x,v,Q10,f_T [1].
        # Domain/constraints: Q10(T) >= Q10_min > 1 gives a real positive exponent;
        #   f_T = 0 outside [Tmin, Tmax]; with the default parameters f_T rises on
        #   [Tmin, Topt], equals 1 at Topt and falls on [Topt, Tmax].
        # Sign convention: The multiplier scales a nonnegative production rate only.
        Tmin = self.temperature_min
        Topt = self.temperature_optimum
        Tmax = self.temperature_max
        span = torch.clamp(Tmax - Topt, min=self.minimum_temperature_span)
        log_q10_span = torch.log(self._temperature_dependent_q10(temperature)) * span
        a_t = self._tensor(1.0) + self.temperature_shape_at_coefficient / log_q10_span
        x_t = (
            log_q10_span**2
            * (self._tensor(1.0) + torch.sqrt(a_t)) ** 2
            / self.temperature_shape_normalization
        )
        v_t = (Tmax - temperature) / span
        response = torch.zeros_like(temperature, device=self.device)
        active = (temperature >= Tmin) & (temperature <= Tmax)
        if torch.any(active):
            v_active = torch.clamp(v_t[active], min=self._tensor(0.0))
            x_active = x_t[active]
            term = torch.pow(v_active, x_active) * torch.exp(
                x_active * (self._tensor(1.0) - v_active)
            )
            response = response.clone()
            response[active] = term
        return torch.clamp(response, min=self._tensor(0.0))

    def _ph_factor(self, soil_ph: Tensor) -> Tensor:
        # References: Zhu et al. (2014), doi:10.5194/gmd-7-981-2014, Eq. 6.
        # Mathematical expression: L=max(pHopt-pHlow,eps); U=max(pHhigh-pHopt,eps);
        #   f_pH=((pH-pHlow)/L)*((pHhigh-pH)/U)^(U/L) within [pHlow,pHhigh], else 0.
        # Parameter meanings: pHlow,pHopt,pHhigh define the response support and optimum;
        #   eps is the existing minimum_dimensionless denominator safeguard.
        # Units: pH coordinates, spans, exponent and response [1].
        # Domain/constraints: Physically ordered pH limits; active ratios are clipped
        #   nonnegative before fractional exponentiation.
        # Sign convention: This nonnegative response changes production magnitude only.
        lower_span = torch.clamp(
            self.ph_optimum - self.ph_lower_limit,
            min=self.minimum_dimensionless,
        )
        upper_span = torch.clamp(
            self.ph_upper_limit - self.ph_optimum,
            min=self.minimum_dimensionless,
        )
        exponent = upper_span / lower_span
        response = torch.zeros_like(soil_ph, device=self.device)
        active = (soil_ph >= self.ph_lower_limit) & (
            soil_ph <= self.ph_upper_limit
        )
        if torch.any(active):
            active_ph = soil_ph[active]
            lower_ratio = torch.clamp(
                (active_ph - self.ph_lower_limit) / lower_span,
                min=self._tensor(0.0),
            )
            upper_ratio = torch.clamp(
                (self.ph_upper_limit - active_ph) / upper_span,
                min=self._tensor(0.0),
            )
            response = response.clone()
            response[active] = lower_ratio * torch.pow(upper_ratio, exponent)
        return response

    def _redox_gate(self, redox_potential: Tensor) -> Tensor:
        # References: Zhu et al. (2014), doi:10.5194/gmd-7-981-2014, Sect. 2.2.1,
        #   generalized from Zhang et al. (2002), doi:10.1029/2001GB001838.
        # Mathematical expression: f_Eh = 1 for Eh <= Eh_1; (Eh_0 - Eh)/(Eh_0 - Eh_1)
        #   for Eh_1 < Eh < Eh_0; 0 for Eh >= Eh_0, i.e. clip((Eh_0-Eh)/(Eh_0-Eh_1),0,1).
        # Parameter meanings: Eh is soil redox potential; Eh_1 = -200 mV is the
        #   full-activity potential and Eh_0 = -100 mV the zero-activity potential.
        # Units: Eh, Eh_1, Eh_0 [mV]; f_Eh [1].
        # Domain/constraints: Eh_0 > Eh_1 is checked at construction; the factor is
        #   continuous and piecewise linear on the whole real line.
        # Sign convention: Higher redox potential suppresses methane production.
        span = self.eh_zero_activity - self.eh_full_activity
        gate = (self.eh_zero_activity - redox_potential) / span
        return torch.clamp(gate, min=self._tensor(0.0), max=self._tensor(1.0))


__all__ = ["MethaneProductionFunctor"]
