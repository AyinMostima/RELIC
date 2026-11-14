"""
SnowPack Mass Balance Module for RELIC

This module implements snow water equivalent (SWE) mass balance with a
temperature index (degree day) melt scheme and precipitation partitioning.

References
----------
- Anderson, E.A. (1976). A point energy and mass balance model of a snow cover.
  NOAA Technical Report NWS 19.
- Hock, R. (2003). Temperature index melt modeling in mountain areas.
  Journal of Hydrology, 282:104-115.
- RHESSys snowpack_daily_F.c implementation (degree day and SWE update).
- Biome-BGC snowmelt.c (temperature index melt and sublimation handling).

Mathematical Expressions
------------------------
SWE mass balance:
    dSWE/dt = P_snow - M - E_sub

Degree day melt:
    M = DDF * max(0, T_air - T_melt) / seconds_per_day

Snow depth estimate:
    References: Anderson (1976), NOAA NWS 19; CF Standard Name Table v94
        (lwe_thickness_of_surface_snow_amount, surface_snow_thickness).
    Mathematical expression: snow_depth = SWE_m * (rho_w / rho_snow), with
        SWE_m = UnitConverter(SWE; mm -> m).
    Parameter meanings: SWE is the snow water equivalent, SWE_m the same liquid-water
        depth in metres, rho_w the liquid water density (physical-constant SSOT),
        rho_snow the bulk snow density (hydrosphere.snow.snow_density).
    Units: SWE [mm]; SWE_m [m]; rho_w and rho_snow [kg m-3]; snow_depth [m].
    Domain/constraints: SWE >= 0 and rho_snow > 0, so snow_depth >= 0; the mm -> m
        step is the same-dimension UDUNITS ratio of the unit SSOT (no literal factor).
    Sign convention: snow_depth is a non-negative thickness measured upward from the
        ground surface.
"""

from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch import Tensor

from ....computation.operators.dec.forms import CochainSpaceSpec
from ....computation.operators.dec.hodge import HodgeStarSpec
from ....computation.operators.iphs import PortSpecIPHS
from ....core.abstractions import StateDict, FluxDict
from ....core.functor import (
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    SubFlowAccuracy,
    VariableKind,
    TemporalScale,
)
from ....core.state import UpdateItem, UpdateMode
from ....data.standards.unit_converter import unit_converter
from ....utils.physical_constants import get_physical_constant
from ..interface_mapping import HydrosphereInterfaceMixin

numeric_types = (float, int, Tensor)


class SnowPack(HydrosphereInterfaceMixin, PhysicalFieldFunctor):
    """
    SnowPack SWE Mass Balance Model.

    This functor partitions precipitation into snow and rain, computes
    degree day melt, applies sublimation losses, and updates snow water
    equivalent (SWE) and snow depth.

    Parameters
    ----------
    name : str
        Functor identifier.
    config : dict
        Optional configuration overrides.

    Returns
    -------
    FluxDict
        Update dictionary containing SWE, snow depth, melt flux, rain flux,
        snowfall flux, and diagnostics.

    References
    ----------
    - Anderson, E.A. (1976). NOAA Tech. Rep. NWS 19.
    - Hock, R. (2003). J. Hydrol. 282:104-115.
    - RHESSys snowpack_daily_F.c.
    - Biome-BGC snowmelt.c.

    Scientific mathematical expressions
    -----------------------------------
    - SWE update: SWE_new = max(0, SWE_old + (P_snow - M - E_sub) * dt)
    - Melt: M = DDF * max(0, T_air - T_melt) / seconds_per_day
    """

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(self, name: str = "snow_pack", config: Optional[Dict] = None) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Snow water equivalent mass balance with degree day melt",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "dSWE/dt = P_snow - M - E_sub; "
                "M = DDF * max(0, T_air - T_melt) / seconds_per_day; "
                "dSWE/dt = star_SWE m_dot"
            ),
            references=[
                "Anderson, E.A. (1976). NOAA Tech. Rep. NWS 19",
                "Hock, R. (2003). J. Hydrol. 282:104-115",
                "RHESSys snowpack_daily_F.c implementation",
                "Biome-BGC snowmelt.c implementation",
            ],
            parameters={
                "hydrosphere.snow.degree_day_factor": (
                    "Degree day melt factor for snow [mm K-1 day-1]"
                ),
                "hydrosphere.snow.temperature_threshold_warm": (
                    "Temperature threshold for switching from cold to warm albedo decay (0 degC = "
                    "273.15 K) [K]"
                ),
                "hydrosphere.snow.rain_snow_temperature_threshold": (
                    "Temperature threshold for rain snow partitioning [K]"
                ),
                "hydrosphere.snow.snow_density": (
                    "Bulk snow density for SWE to depth conversion [kg m-3]"
                ),
            },
            input_variables=[
                "precipitation_flux",
                "air_temperature",
                "snow_water_equivalent",
                "sublimation_flux",
            ],
            output_variables=[
                "snow_water_equivalent",
                "snow_depth",
                "snow_melt_flux",
                "rain_flux",
                "snowfall",
                "diagnostics.snow_pack",
            ],
            conservation_laws=["mass"],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["snow_albedo", "surface_runoff", "energy_balance"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            # References: Hock (2003), Eq. 1; Anderson (1976);
            #   Hairer, Lubich and Wanner (2006), composition of sub-flows.
            # Mathematical expression: M=min(M0,S/h);
            #   E=min(E0,max(S-M*h,0)/h); S1=max(S+(P-M-E)*h,0).
            #   The held-input interior is affine-exact, but with M0=0,
            #   E0=2*P and S0=P*H/4, the depleted update retains S_h(H)=P*h.
            #   Its nonzero boundary error is first order, not global EXACT.
            # Parameter meanings: S/S1 are starting/updated SWE; P snowfall;
            #   M0/E0 potential melt/sublimation; M/E accepted withdrawals;
            #   h the call length and H the fixed observation horizon.
            # Units: S/S1 [mm]; P/M/E/M0/E0 [mm s-1]; h/H [s].
            # Domain/constraints: nonnegative SWE and forcing, h>0; each
            #   withdrawal uses starting inventory before snowfall is added.
            #   No uniform smooth-order claim covers capacity or phase
            #   boundaries. The S1 snowpack-to-bucket refusal is unchanged.
            # Sign convention: P enters storage; M/E leave it; positive
            #   retained SWE is the boundary error against an exhausted store.
            sub_flow_accuracy=SubFlowAccuracy.ORDER_1,
        )

        super().__init__(name=name, config=config or {}, metadata=metadata)
        # All four inputs are required: a missing value is an error, never a state default.

        self.field_type = "snow_pack"
        self.spatial_dimensions = 0
        self.field_units = "mm"
        self._unit_converter = unit_converter

        self.degree_day_factor = self.get_parameter_value(
            "degree_day_factor",
            namespace="hydrosphere.snow", required=True)
        self.melt_temperature_threshold = self.get_parameter_value(
            "temperature_threshold_warm",
            namespace="hydrosphere.snow", required=True)
        self.rain_snow_temperature_threshold = self.get_parameter_value(
            "rain_snow_temperature_threshold",
            namespace="hydrosphere.snow", required=True)
        self.snow_density = self.get_parameter_value(
            "snow_density",
            namespace="hydrosphere.snow", required=True)
        self.seconds_per_day = get_physical_constant("seconds_per_day")
        self.water_density = get_physical_constant("water_density")

    def get_dependencies(self) -> DependencyDeclaration:
        return self._merge_declared_dependencies({"required": [], "optional": [], "lagged": []})

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve structural audit size without changing runtime tensor shape."""

        if n_dofs is not None:
            value = int(n_dofs)
        elif complex_bundle is not None and hasattr(complex_bundle, "n_dofs"):
            value = int(getattr(complex_bundle, "n_dofs"))
        else:
            value = 1
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for snowpack mass balance.

        References:
          - Anderson (1976), Hock (2003), and Hirani (2003).
        Mathematical expression:
          dSWE/dt = P_snow - M - E_sub, with all mass fluxes as port flows.
        Parameter meanings:
          SWE is snow water equivalent [mm]; P_snow, M, and E_sub are mass
          fluxes [kg m-2 s-1].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_snowpack",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="mm",
                extensivity="density",
                physical_domain="cryosphere",
                semantic_name="snow_water_equivalent",
                orientation_convention="storage_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_snowpack",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="snowfall_mass_flux",
                orientation_convention="into_snowpack_positive",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_snowpack",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="snow_melt_mass_flux",
                orientation_convention="out_of_snowpack_positive",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_snowpack",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="sublimation_mass_flux",
                orientation_convention="out_of_snowpack_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "hydrosphere.snow.degree_day_factor",
            "hydrosphere.snow.temperature_threshold_warm",
            "hydrosphere.snow.rain_snow_temperature_threshold",
            "hydrosphere.snow.snow_density",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive mass-to-SWE conversion Hodge contract.

        References:
          - Anderson (1976), Hock (2003), and van der Schaft and Maschke (1995).
        Mathematical expression:
          star_SWE = diag(1000 / rho_w), mapping kg m-2 s-1 to mm s-1.
        Parameter meanings:
          rho_w is relic.physical.constants.water.density_liquid [kg m-3].
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        # The factor is UnitConverter's single CF liquid-water-equivalent rule for the
        # declared snowpack net mass flux: 1 kg m-2 s-1 -> 1000 / rho_w mm s-1.
        conversion = float(
            self._unit_converter.convert(1.0, "snowpack_net_mass_flux", "kg m-2 s-1", "mm s-1")
        )
        values = torch.full(
            (n_dofs,),
            conversion,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        domain = CochainSpaceSpec(
            complex_id="hydrosphere_snowpack",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="snowpack_net_mass_flux",
            orientation_convention="into_snowpack_positive",
            boundary_role="port",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_snowpack",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="mm s-1",
            extensivity="density",
            physical_domain="cryosphere",
            semantic_name="snow_water_equivalent_rate",
            orientation_convention="storage_positive",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_snowpack_mass_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=("relic.physical.constants.water.density_liquid",),
            metric_keys=("hydrosphere.snow.snow_density",),
            material_law="snowpack_mass_to_swe_depth",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build snowpack Hodge terms for IPHS assembly."""

        return (
            self.build_hodge_contract(
                n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
                strict=True,
            ),
        )

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return storage and degree-day energy proxy terms for audits."""

        return {
            "snow_water_storage": "SWE",
            "degree_day_melt_proxy": "DDF * max(0, T_air - T_melt)",
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return nonnegative melt and sublimation dissipation statements."""

        return {
            "melt_dissipation": "sigma_m = M * X_m >= 0",
            "sublimation_dissipation": "sigma_sub = E_sub * X_sub >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for temperature-index snowpack balance."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive snowpack mass conversion mobility.

        References:
          - Anderson (1976), Hock (2003), and van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = star_SWE, sigma = m_dot.T M_h m_dot >= 0.
        Parameter meanings:
          m_dot is the net snowpack mass flux [kg m-2 s-1].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"snow_water_equivalent[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare snow accumulation and melt release mass ports.

        References:
          - Anderson (1976), Hock (2003), and van der Schaft and Maschke (1995).
        Mathematical expression:
          P_port = e_h f_h, with snowfall into the snowpack and melt out.
        Parameter meanings:
          f_h is a mass flux [kg m-2 s-1] and e_h is a water potential head [m].
        """

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_snowpack",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="snow_water_potential",
            orientation_convention="storage_positive",
            boundary_role="port",
        )
        accumulation_flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_snowpack",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="snowfall_mass_flux",
            orientation_convention="into_snowpack_positive",
            boundary_role="port",
        )
        melt_flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_snowpack",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="snow_melt_mass_flux",
            orientation_convention="out_of_snowpack_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="snowpack_accumulation_port",
                effort_var="snow_water_potential",
                flow_var="snowfall",
                effort_space=effort_space,
                flow_space=accumulation_flow_space,
                power_units="kg m-1 s-1",
                positive_power_convention="into_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="snowpack_melt_release_port",
                effort_var="snow_water_potential",
                flow_var="snow_melt_flux",
                effort_space=effort_space,
                flow_space=melt_flow_space,
                power_units="kg m-1 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
        )

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Mapping[str, Any]:
        """Run the canonical calculation through the existing Functor lifecycle."""

        return self.forward(dict(state), time, dt)

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Check SnowPack storage balance using explicit Phase 8 residuals.

        References:
          - Anderson (1976), Hock (2003), and RELIC conservation architecture.
        Mathematical expression:
          violation = sum(abs(r_SWE)), where r_SWE is the routed nonnegative
          storage projection residual [mm].
        Parameter meanings:
          input_state and output_updates are the FunctorBase conservation API;
          the residual is computed in forward from the same SWE balance terms.
        """

        residual = getattr(self, "_last_snowpack_mass_balance_residual_mm", None)
        if residual is None:
            return {"mass": 0.0}
        residual_tensor = self._ensure_tensor(residual)
        mass_value = torch.abs(residual_tensor).detach().sum().tolist()
        return {"mass": float(mass_value)}

    def _to_float(self, value: Union[float, int, Tensor]) -> float:
        if isinstance(value, Tensor):
            return float(value.detach().reshape(-1)[0].tolist())
        return float(value)

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """
        Execute snow mass balance over the time step.

        Parameters
        ----------
        state : StateDict
            Current model state.
        time : float
            Current simulation time [s].
        dt : float
            Time step [s].

        Returns
        -------
        FluxDict
            Updates for SWE, snow depth, snow melt flux, rain flux, and snowfall.
        """
        # References:
        #   Anderson (1976); Hock (2003), temperature-index melt.
        # Mathematical expression:
        #   Ps=P if T<=T_partition, otherwise 0; Pr=P-Ps;
        #   Mraw=DDF*max(T-Tm,0)/seconds_per_day; M=min(Mraw,max(S0,0)/dt);
        #   E=min(Eraw,max(S0-M*dt,0)/dt); S1=max(S0+(Ps-M-E)*dt,0);
        #   depth=U(S1; mm->m)*(rho_water/rho_snow).
        # Parameter meanings:
        #   S is SWE; Ps/Pr are snowfall/rain depth rates; M and E are actual melt
        #   and sublimation after sequential availability limits; U(.; mm->m) is the
        #   UnitConverter same-dimension conversion of the unit SSOT.
        # Units:
        #   S [mm], Ps/Pr/M/E [mm s-1]; external fluxes [kg m-2 s-1];
        #   DDF [mm K-1 day-1], T [K], dt [s], depth [m], density [kg m-3].
        # Domain/constraints:
        #   Positive dt; melt then sublimation are limited by starting SWE, so fresh
        #   snow enters at this step end. Report rejected demand and nonnegative
        #   projection; these diagnostics do not add water.
        # Sign convention:
        #   Snowfall increases SWE; melt and sublimation remove it; rain bypasses SWE.
        missing = [name for name in self.get_required_inputs() if state.get(name) is None]
        if missing:
            raise ValueError(
                "SnowPack requires the inputs " + ", ".join(missing)
                + "; no state default is substituted"
            )

        precipitation_flux = self._ensure_tensor(state["precipitation_flux"])
        air_temperature = self._ensure_tensor(state["air_temperature"])
        swe = self._ensure_tensor(state["snow_water_equivalent"])
        sublimation_flux = self._ensure_tensor(state["sublimation_flux"])

        dt_tensor = self._ensure_tensor(dt, dtype=swe.dtype)
        seconds_per_day = self._ensure_tensor(self.seconds_per_day, dtype=swe.dtype)
        degree_day_factor = self._ensure_tensor(self.degree_day_factor, dtype=swe.dtype)
        melt_threshold = self._ensure_tensor(self.melt_temperature_threshold, dtype=swe.dtype)
        rain_snow_threshold = self._ensure_tensor(self.rain_snow_temperature_threshold, dtype=swe.dtype)
        water_density_tensor = self._ensure_tensor(self.water_density, dtype=swe.dtype)
        snow_density_tensor = self._ensure_tensor(self.snow_density, dtype=swe.dtype)

        # References: Anderson 1976; RHESSys snowpack_daily_F.c.
        # Formula: f_snow = 1 if T_air <= T_rain_snow else 0.
        # Parameter meanings: T_air [K], T_rain_snow [K].
        snow_fraction = torch.where(air_temperature <= rain_snow_threshold, 1.0, 0.0)

        # References: Anderson 1976; CF-1.8 water depth conversion.
        # Formula: P_mm_s = (P_mass / rho_w) * 1000, evaluated by UnitConverter's single CF
        #   liquid-water-equivalent rule (rho_w = water.density_liquid).
        # Parameter meanings: P_mass [kg m-2 s-1], rho_w [kg m-3].
        precip_mm_s = self._unit_converter.convert(
            precipitation_flux, "precipitation_flux", "kg m-2 s-1", "mm s-1"
        )
        snowfall_mm_s = precip_mm_s * snow_fraction
        rain_mm_s = precip_mm_s - snowfall_mm_s

        # References: Hock 2003; Biome-BGC snowmelt.c.
        # Formula: M_mm_s = DDF * max(0, T_air - T_melt) / seconds_per_day.
        # Parameter meanings: DDF [mm K-1 day-1], T_air [K], T_melt [K], seconds_per_day [s].
        temp_excess = torch.clamp(air_temperature - melt_threshold, min=0.0)
        raw_melt_mm_s = degree_day_factor * temp_excess / seconds_per_day

        # References: Anderson 1976; RHESSys snowpack_daily_F.c.
        # Formula: E_sub_mm_s = (E_sub_mass / rho_w) * 1000.
        # Parameter meanings: E_sub_mass [kg m-2 s-1], rho_w [kg m-3].
        raw_sublimation_mm_s = self._unit_converter.convert(
            sublimation_flux, "sublimation_flux", "kg m-2 s-1", "mm s-1"
        )

        # References: Anderson 1976; Hock 2003.
        # Formula: SWE_new = max(0, SWE_old + (P_snow - M - E_sub) * dt).
        # Parameter meanings: SWE [mm], P_snow [mm s-1], M [mm s-1], E_sub [mm s-1], dt [s].
        if torch.any(dt_tensor <= 0):
            raise ValueError("SnowPack requires positive dt.")
        # References: Anderson 1976; RELIC Phase 8 overflow routing.
        # Mathematical expression: r_M = max(M_raw - SWE/dt, 0), r_E = max(E_raw - (SWE-M*dt)/dt, 0).
        # Parameter meanings: r_M and r_E are melt and sublimation capacity residual rates [mm s-1].
        available_melt = torch.clamp(swe, min=0.0) / dt_tensor
        melt_capacity_residual_mm_s = torch.clamp(raw_melt_mm_s - available_melt, min=0.0)
        melt_mm_s = torch.minimum(raw_melt_mm_s, available_melt)
        available_after_melt = torch.clamp(swe - melt_mm_s * dt_tensor, min=0.0)
        sublimation_capacity = available_after_melt / dt_tensor
        sublimation_capacity_residual_mm_s = torch.clamp(
            raw_sublimation_mm_s - sublimation_capacity,
            min=0.0,
        )
        sublimation_mm_s = torch.minimum(raw_sublimation_mm_s, sublimation_capacity)
        raw_swe_new = swe + (snowfall_mm_s - melt_mm_s - sublimation_mm_s) * dt_tensor
        swe_nonnegative_projection_residual_mm = torch.clamp(-raw_swe_new, min=0.0)
        swe_new = torch.clamp(raw_swe_new, min=0.0)
        self._last_snowpack_mass_balance_residual_mm = swe_nonnegative_projection_residual_mm

        # References: Anderson (1976), NOAA Technical Report NWS 19; CF Standard Name
        #   Table v94 (lwe_thickness_of_surface_snow_amount, surface_snow_thickness).
        # Mathematical expression: d = SWE_m * rho_w / rho_snow, with
        #   SWE_m = U(SWE_mm; mm -> m) taken from the unit SSOT (UnitConverter).
        # Parameter meanings: SWE_mm is the updated snow water equivalent, SWE_m the
        #   same liquid-water depth in metres, rho_w the liquid water density
        #   (physical-constant SSOT), rho_snow the bulk snow density
        #   (hydrosphere.snow.snow_density), d the snow depth.
        # Units: SWE_mm [mm]; SWE_m [m]; rho_w, rho_snow [kg m-3]; d [m].
        # Domain/constraints: SWE_mm >= 0 (projected above); rho_snow > 0 by its
        #   parameter bounds, so d >= 0 and d = 0 exactly when SWE = 0.
        # Sign convention: d is a non-negative thickness measured upward from the
        #   ground surface.
        swe_new_m = self._unit_converter.convert(swe_new, "snow_water_equivalent", "mm", "m")
        snow_depth = swe_new_m * (water_density_tensor / snow_density_tensor)

        snow_melt_flux = self._unit_converter.convert(melt_mm_s, "snow_melt_flux", "mm s-1", "kg m-2 s-1")
        rain_flux = self._unit_converter.convert(rain_mm_s, "rain_flux", "mm s-1", "kg m-2 s-1")
        snowfall_flux = self._unit_converter.convert(snowfall_mm_s, "snowfall_flux", "mm s-1", "kg m-2 s-1")

        diagnostics = {
            "snow_fraction": snow_fraction,
            "melt_rate_mm_s": melt_mm_s,
            "sublimation_rate_mm_s": sublimation_mm_s,
            "swe_before": swe,
            "swe_after": swe_new,
            "melt_capacity_residual_mm_s": melt_capacity_residual_mm_s,
            "sublimation_capacity_residual_mm_s": sublimation_capacity_residual_mm_s,
            "swe_nonnegative_projection_residual_mm": swe_nonnegative_projection_residual_mm,
        }

        # Each state output is one REPLACE item in the unit its SCM contract declares
        # (snow_water_equivalent mm, snow_depth m, the three boundary fluxes
        # kg m-2 s-1), so StateManager checks the unit and writer at commit.
        def _replace(value: Tensor, unit: str) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=False,
                source_functor=self.name,
            )

        return {
            "snow_water_equivalent": _replace(swe_new, "mm"),
            "snow_depth": _replace(snow_depth, "m"),
            "snow_melt_flux": _replace(snow_melt_flux, "kg m-2 s-1"),
            "rain_flux": _replace(rain_flux, "kg m-2 s-1"),
            "snowfall": _replace(snowfall_flux, "kg m-2 s-1"),
            "diagnostics.snow_pack": diagnostics,
        }
