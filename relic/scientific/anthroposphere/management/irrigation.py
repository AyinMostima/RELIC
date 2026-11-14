"""
Irrigation Management Module

This module implements irrigation management practices including scheduling,
water application, and efficiency calculations for agricultural systems.

References:
    - Allen, R.G., et al. (1998). Crop evapotranspiration - Guidelines for computing crop water requirements. FAO Irrigation and drainage paper 56.
    - Doorenbos, J. & Pruitt, W.O. (1977). Crop water requirements. FAO irrigation and drainage paper 24.
    - Hanks, R.J. (1974). Model for predicting plant yield as influenced by water use. Agronomy Journal 66(5), 660-665.

Mathematical Expression:
    M_net = max((theta_fc - theta) Z_r rho_w - F_P dt, 0)
    M_gross,i = s min(M_net,i / eta, F_max dt), s = min(1, Q / D),
    D = h sum_i min(M_net,i / eta, F_max dt), Q' = Q - s D
    M_delivered = eta M_gross

    where theta is volumetric soil moisture [m3 m-3], Z_r is root-zone
    depth [m], rho_w is water density [kg m-3], F_P and F_max are water
    mass fluxes [kg m-2 s-1], eta is application efficiency [1], and the
    M terms are areal water masses [kg m-2].

    Precipitation is taken as fully effective (P_e = F_P). This is a
    simplification: FAO effective-precipitation methods (USDA Soil Conservation
    Service; FAO/AGLW dependable rainfall) give P_e <= P because part of the rain
    runs off or percolates below the root zone, so M_net underestimates the FAO
    net irrigation requirement.

Assumptions:
    - Water depth and mass conversions use UnitConverter and the physical constants SSOT.
"""

import torch
from typing import Any, Dict, Mapping, Optional, Union
import logging

from relic.utils.physical_constants import get_physical_constant
from relic.data.standards.unit_converter import ConversionContext, unit_converter

from relic.core.functor import (
    FunctorMetadata,
    FluxDict,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
    StateDict,
)
from relic.core.functor import TransportMechanismFunctor
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.unified_response_functions import (
    moisture_response_linear
)

logger = logging.getLogger(__name__)

class IrrigationManagement(TransportMechanismFunctor):
    """
    Irrigation Management Functor
    
    Implements irrigation scheduling, water application, and abstraction
    accounting for agricultural water management.
    
    Runtime inputs:
        irrigation_type: Canonical integer system selector in StateManager
        soil_moisture: Volumetric root-zone soil moisture [m3 m-3]
        precipitation_flux: External water mass flux [kg m-2 s-1]

    ParameterManager coefficients:
        System efficiencies, root-zone depth, application-rate limit, moisture
        threshold, annual source-water quota, and conservation tolerance.
    """

    TRANSPORT_TYPE = "managed_application"
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    # irrigation_type code -> name of the application efficiency of that system
    # under anthroposphere.management.irrigation (the codes of the
    # irrigation_type contract, 1 to 4).
    SYSTEM_EFFICIENCY_NAMES = {
        1: "surface_efficiency",
        2: "sprinkler_efficiency",
        3: "drip_efficiency",
        4: "subsurface_efficiency",
    }
    
    def __init__(self, name: str = "irrigation_management", config: Optional[Dict] = None):
        """
        Initialize irrigation management functor
        
        Args:
            name: Functor name
            config: Configuration dictionary
        """
        if name != "irrigation_management":
            raise ValueError(
                "IrrigationManagement requires canonical runtime name "
                "'irrigation_management' to match its SCM writer contract."
            )

        # Set up metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Irrigation management and water application for agriculture",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "Canonical IPHS:\n"
                "  owner: irrigation_management\n"
                "  role: managed water boundary port with explicit source and "
                "delivered flows\n"
                "  state: no prognostic water state is owned; the management "
                "allocation irrigation_remaining_quota is owned StateManager state "
                "(Q' = Q - s D, REPLACE); no module buffer\n"
                "  port closure: "
                "M_net=max((theta_fc-theta)*Z_r*rho_w-precipitation_flux*dt,0); "
                "M_gross=s*min(M_net/eta,F_max*dt), s=min(1,Q/D) (equal-percentage quota share); "
                "M_delivered=eta*M_gross\n"
                "  simplification: effective precipitation P_e = precipitation_flux; "
                "FAO methods give P_e <= P, so M_net underestimates the net "
                "irrigation requirement\n"
                "  open-system route: abstraction is the managed source withdrawal "
                "and delivered irrigation enters the root zone\n"
                "  classification: level 2 non-strict phenomenological wrapper; "
                "no strict H/e/J/M or GENERIC degeneracy claim"
            ),
            references=[
                "Allen et al. (1998) FAO-56 Crop evapotranspiration",
                "Doorenbos & Pruitt (1977) FAO-24 Crop water requirements",
                "Hanks (1974) DOI:10.2134/agronj1974.00021962006600050017x"
            ],
            parameters={
                "anthroposphere.management.irrigation.surface_efficiency": (
                    "Surface-system application efficiency [dimensionless]"
                ),
                "anthroposphere.management.irrigation.sprinkler_efficiency": (
                    "Sprinkler-system application efficiency [dimensionless]"
                ),
                "anthroposphere.management.irrigation.drip_efficiency": (
                    "Drip-system application efficiency [dimensionless]"
                ),
                "anthroposphere.management.irrigation.subsurface_efficiency": (
                    "Subsurface-system application efficiency [dimensionless]"
                ),
                "anthroposphere.management.irrigation.moisture_threshold": (
                    "Irrigation trigger as a fraction of the plant-available water "
                    "range [dimensionless]"
                ),
                "anthroposphere.management.irrigation.root_zone_depth": (
                    "Effective root-zone depth [m]"
                ),
                "anthroposphere.management.irrigation.max_application_rate": (
                    "Maximum water application rate [mm hr-1]"
                ),
                "anthroposphere.management.irrigation.conservation_tolerance": (
                    "Water-port residual tolerance [kg m-2 s-1]"
                ),
            },
            input_variables=[
                "soil_moisture",
                "precipitation_flux",
                "land_use_type",
                "irrigation_type",
                "field_capacity",
                "wilting_point",
                "irrigation_remaining_quota",
            ],
            output_variables=[
                "irrigation_flux",
                "abstraction_flux",
                "irrigation_remaining_quota",
                "diagnostics.irrigation.applied_water",
                "diagnostics.irrigation.effective_efficiency",
                "diagnostics.irrigation.moisture_response",
                "diagnostics.irrigation.water_balance_residual",
            ],
            conservation_laws=["mass"],  # Water mass balance
            spatial_dimensions=2,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            coupling_interfaces=["hydrosphere", "pedosphere"]
        )
        
        # Namespaces for parameter lookups (must be set before super().__init__)
        self._irrigation_namespace = "anthroposphere.management.irrigation"
        # Initialize the TransportMechanismFunctor bridge and FunctorBase lifecycle.
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata
        )
        
        # The application efficiencies of the four irrigation systems are the
        # learnable parameters of this process: ParameterManager bindings held
        # by reference under their irrigation_type code.
        self._system_efficiencies: Dict[int, Any] = self._held_system_efficiencies()
        
        # The Functor holds no accounting tensor: the remaining source-water quota is
        # the StateManager state irrigation_remaining_quota it owns (initial value
        # from the scenario initial state), and the delivered depth is a per-step
        # diagnostic.

    @staticmethod
    def _scalar_from_value(value: Union[torch.Tensor, float, int]) -> float:
        """Convert tensor-like parameter values to float."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                detached = value.detach().reshape(-1)
                values = detached.tolist()
                if not values:
                    raise ValueError("Expected non-empty scalar tensor for parameter conversion.")
                return float(values[0])
            raise ValueError("Expected scalar tensor for parameter conversion.")
        return float(value)

    @staticmethod
    def _to_bool(value: Union[torch.Tensor, bool, int, float]) -> bool:
        """Convert tensor-like values to bool without scalar extraction sync."""
        if isinstance(value, torch.Tensor):
            detached = value.detach().reshape(-1)
            values = detached.tolist()
            return bool(values[0]) if values else False
        return bool(value)

    @staticmethod
    def _reshape_output(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure outputs align with column-vector expectations."""
        if tensor.ndim == 1:
            return tensor.reshape(-1, 1)
        if tensor.ndim == 2 and tensor.shape[0] == 1 and tensor.shape[1] > 1:
            return tensor.transpose(0, 1).contiguous()
        return tensor

    def _get_irrigation_param(self, name: str) -> Any:
        """Fetch irrigation parameter from the unified parameter database."""
        return self.get_parameter_value(
            name,
            namespace=self._irrigation_namespace,
            source_order=("manager",), required=True)

    def _as_param_tensor(self, value: Any, reference: torch.Tensor) -> torch.Tensor:
        """Cast parameter value to reference tensor dtype and device."""
        if isinstance(value, str):
            value = float(value)
        if isinstance(value, (int, float)):
            return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if value is None:
            raise ValueError("Required irrigation parameter value must not be None")
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)

    def _kg_m2_to_mm_factor(self, reference: torch.Tensor) -> torch.Tensor:
        """Convert areal mass [kg m-2] to water depth [mm] using SSOT water density."""
        # References: BIPM (2019); RELIC liquid-water density SSOT.
        # Mathematical expression: h_mm = m_A (1000 / rho_w).
        # Parameter meanings: m_A is areal water mass, rho_w density, and h_mm water depth.
        # Units: m_A [kg m-2], rho_w [kg m-3], conversion factor [mm m2 kg-1], h_mm [mm].
        # Domain/constraints: reference is floating point and rho_w is finite and strictly positive.
        # Sign convention: positive areal water mass maps to positive water depth.
        water_density = self._as_param_tensor(
            get_physical_constant("water_density"),
            reference,
        )
        if (
            water_density.numel() != 1
            or not bool(torch.isfinite(water_density).all().item())
            or bool(torch.any(water_density <= 0.0).item())
        ):
            raise ValueError("water_density must be a finite positive scalar")
        metres_per_mass = torch.reciprocal(water_density)
        converted = unit_converter.convert(
            metres_per_mass,
            "generic",
            "m",
            "mm",
            ConversionContext(),
        )
        return torch.as_tensor(
            converted,
            device=reference.device,
            dtype=reference.dtype,
        )

    def get_metadata(self) -> FunctorMetadata:
        """Return functor metadata (backward-consistent accessor)."""
        return self.metadata

    def _standard_moisture_response(
        self,
        soil_moisture: torch.Tensor,
        method: str = "linear",
        params: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute soil moisture response modifier.

        References: Allen et al. (1998) FAO-56 water stress factor.
        Expression: f(theta) = clip((theta - theta_min) / (theta_max - theta_min), 0, 1)
        Parameters:
            theta: volumetric soil moisture [m3 m-3]
            theta_min: wilting point [m3 m-3]
            theta_max: field capacity [m3 m-3]
        """
        # References: Allen et al. (1998), FAO-56 water-stress response.
        # Mathematical expression: f_theta=clip((theta-theta_min)/(theta_max-theta_min),0,1).
        # Parameter meanings: theta is moisture, theta_min wilting point, and theta_max field capacity.
        # Units: theta, theta_min, and theta_max [m3 m-3]; f_theta [1].
        # Domain/constraints: inputs are finite, 0 <= theta <= 1, and theta_max > theta_min.
        # Sign convention: f_theta increases from 0 at wilting to 1 at field capacity.
        params = params or {}
        theta_min = params.get("min", 0.0)
        theta_max = params.get("max", 1.0)
        if method != "linear":
            raise ValueError(f"Unsupported moisture response method '{method}'")
        return moisture_response_linear(
            soil_moisture,
            wilting_point=theta_min,
            field_capacity=theta_max,
        )

    @staticmethod
    def _require_state_tensor(
        state: Dict[str, torch.Tensor],
        key: str,
        *,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        integer: bool = False,
    ) -> torch.Tensor:
        """Return one required finite canonical input without aliases or defaults."""
        if key not in state:
            raise KeyError(f"Missing required irrigation input '{key}'")
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Irrigation input '{key}' must be a torch.Tensor")
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise ValueError(f"Irrigation input '{key}' must contain only finite values")
        if integer and value.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(f"Irrigation input '{key}' must use an integer dtype")
        if minimum is not None and bool(torch.any(value < minimum).item()):
            raise ValueError(
                f"Irrigation input '{key}' must be greater than or equal to {minimum}"
            )
        if maximum is not None and bool(torch.any(value > maximum).item()):
            raise ValueError(
                f"Irrigation input '{key}' must be less than or equal to {maximum}"
            )
        return value

    def _held_system_efficiencies(self) -> Dict[int, Any]:
        """Resolve the system application efficiencies this Functor holds."""
        # References: Allen et al. (1998), FAO-56; FAO, Determination of the
        #   irrigation schedule for a field (gross depth = net depth / field
        #   application efficiency); ParameterManager binding ownership,
        #   docs/source/architecture_overview.md Sections 4.2, 4.4.2 and 4.4.3.
        # Mathematical expression: eta_i = eta_sys(c_i), c_i in {1, 2, 3, 4}.
        # Parameter meanings: c_i is the irrigation_type code of cell i (1
        #   surface, 2 sprinkler, 3 drip, 4 subsurface), eta_sys(c) the
        #   application efficiency of that system, eta_i the efficiency of the
        #   cell in M_gross = min(M_net / eta, F_max dt) and
        #   M_delivered = eta M_gross.
        # Units: eta_sys, eta_i [1]; c_i [1] (categorical code).
        # Domain/constraints: 0 < eta_sys <= 1 (checked in forward). The four
        #   efficiencies are learnable ParameterManager bindings and the only
        #   parameters of this process the gradient of a call reaches through
        #   eta; they are held by reference, so registration relocates them to
        #   the device of this Functor and a durable checkpoint saves them in
        #   parameter_bindings. Every valid irrigation_type selects one of
        #   them, so there is no generic efficiency. The moisture threshold
        #   enters through a discrete trigger and is no learnable parameter: it
        #   is read from the manager at every call and not held.
        # Sign convention: none (dimensionless efficiencies in (0, 1]).
        return {
            code: self._get_irrigation_param(name)
            for code, name in self.SYSTEM_EFFICIENCY_NAMES.items()
        }

    
    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """
        Calculate irrigation fluxes and water abstraction
        
        Args:
            state: Current state dictionary
            time: Current simulation time [s]
            dt: Time step [s]
            **kwargs: Runtime keywords; the inputs come from the state only
            
        Returns:
            Dictionary with irrigation_flux and abstraction_flux
        """
        # References: Allen et al. (1998); FAO gross-irrigation efficiency guidance.
        # Mathematical expression: M_net=max((theta_fc-theta)Z rho-Pdt,0); M_gross=s min(M_net/eta,F_max dt) with s=min(1,Q/D); M_delivered=eta M_gross.
        # Parameter meanings: theta terms are moisture, Z root depth, rho density, P precipitation, and eta efficiency.
        # Units: M terms [kg m-2], P/F outputs [kg m-2 s-1], Z [m], rho [kg m-3], dt [s].
        # Domain/constraints: time is finite, dt > 0, theta_fc > theta_wp, eta in (0,1], and quota >= 0.
        # Sign convention: abstraction and delivered irrigation are positive into their declared boundary routes.
        try:
            time_value = self._scalar_from_value(time)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("time must be finite and non-negative") from exc
        if isinstance(time, bool) or not bool(
            torch.isfinite(torch.as_tensor(time_value)).item()
        ) or time_value < 0.0:
            raise ValueError("time must be finite and non-negative")
        try:
            dt_value = self._scalar_from_value(dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dt must be finite and greater than zero") from exc
        if isinstance(dt, bool) or not bool(
            torch.isfinite(torch.as_tensor(dt_value)).item()
        ) or dt_value <= 0.0:
            raise ValueError("dt must be finite and greater than zero")

        soil_moisture = self._require_state_tensor(
            state,
            "soil_moisture",
            minimum=0.0,
            maximum=1.0,
        )
        precipitation_flux = self._require_state_tensor(
            state,
            "precipitation_flux",
            minimum=0.0,
        )
        land_use_type = self._require_state_tensor(
            state,
            "land_use_type",
            minimum=0.0,
            integer=True,
        )
        irrigation_type_idx = self._require_state_tensor(
            state,
            "irrigation_type",
            minimum=1.0,
            maximum=4.0,
            integer=True,
        )

        # Initialize output fluxes
        irrigation_flux = torch.zeros_like(soil_moisture)
        abstraction_flux = torch.zeros_like(soil_moisture)
        
        # Only irrigate cropland (land_use_type == 1)
        is_cropland = (land_use_type == 1)
        
        # Get field capacity and wilting point
        field_capacity = self._require_state_tensor(
            state,
            "field_capacity",
            minimum=0.0,
            maximum=1.0,
        )
        wilting_point = self._require_state_tensor(
            state,
            "wilting_point",
            minimum=0.0,
            maximum=1.0,
        )
        canonical_shape = soil_moisture.shape
        for key, tensor in (
            ("precipitation_flux", precipitation_flux),
            ("land_use_type", land_use_type),
            ("irrigation_type", irrigation_type_idx),
            ("field_capacity", field_capacity),
            ("wilting_point", wilting_point),
        ):
            if tensor.shape != canonical_shape:
                raise ValueError(
                    f"Irrigation input '{key}' shape {tuple(tensor.shape)} does not "
                    f"match soil_moisture shape {tuple(canonical_shape)}"
                )
        if bool(torch.any(field_capacity <= wilting_point).item()):
            raise ValueError("field_capacity must be greater than wilting_point")
        
        # References: Allen et al. (1998), FAO-56, Eqs. 82 and 83 (total and
        #   readily available soil water).
        # Mathematical expression: theta_trigger = theta_wp + f (theta_fc - theta_wp);
        #   u = 1 if theta < theta_trigger on cropland, else 0.
        # Parameter meanings: f is the moisture threshold, the fraction of the
        #   plant-available water range below which irrigation starts; theta_wp
        #   and theta_fc are wilting point and field capacity; theta is the
        #   root-zone soil moisture; u the irrigation trigger.
        # Units: f, u [1]; theta, theta_wp, theta_fc, theta_trigger [m3 m-3].
        # Domain/constraints: theta_fc > theta_wp (checked above). u is a step
        #   function, so no gradient reaches f: it is an event parameter, read
        #   from the manager at every call, and declared not learnable.
        # Sign convention: none (a trigger; the triggered fluxes are nonnegative).
        threshold = self._get_irrigation_param("moisture_threshold")
        trigger_moisture = wilting_point + threshold * (field_capacity - wilting_point)
        needs_irrigation = (soil_moisture < trigger_moisture) & is_cropland
        
        # References: RELIC time-space contract (StateManager-owned state).
        # Mathematical expression: Q is read from the state; Q' = Q - A_used is returned.
        # Parameter meanings: Q the remaining source-water quota, A_used the domain
        #   abstraction of the step (sum of the cell abstraction depths).
        # Units: Q, Q', A_used [mm] (source-water depth summed over the cells).
        # Domain/constraints: Q is one finite non-negative value (a domain quota); a
        #   missing value raises; its initial value comes from the scenario state.
        # Sign convention: Q decreases by the abstracted depth.
        remaining_quota = self._require_state_tensor(
            state,
            "irrigation_remaining_quota",
            minimum=0.0,
        )
        if remaining_quota.numel() != 1:
            raise ValueError("irrigation_remaining_quota must be one domain value [mm]")
        remaining_quota = remaining_quota.reshape(()).to(
            device=soil_moisture.device,
            dtype=soil_moisture.dtype,
        )
        next_quota = remaining_quota.clone()
        applied_water_mm = torch.zeros((), dtype=soil_moisture.dtype, device=soil_moisture.device)
        has_quota = self._to_bool(remaining_quota > 0.0)

        # The efficiency of a cell is the application efficiency of its
        # irrigation system (eta_i = eta_sys(c_i), see
        # _held_system_efficiencies). The manager is read at every call and
        # returns the held binding of a learnable efficiency, so the value is
        # current and the gradient of the ports reaches it; the selection has no
        # default, because the input contract admits the codes 1 to 4 only.
        system_efficiencies = torch.stack(
            [
                self._as_param_tensor(
                    self._get_irrigation_param(self.SYSTEM_EFFICIENCY_NAMES[code]),
                    soil_moisture,
                ).reshape(())
                for code in sorted(self.SYSTEM_EFFICIENCY_NAMES)
            ]
        )
        efficiency = system_efficiencies[
            irrigation_type_idx.to(device=soil_moisture.device, dtype=torch.long) - 1
        ]
        if not bool(torch.all(torch.isfinite(efficiency)).item()):
            raise ValueError("Irrigation efficiency parameters must be finite")
        if bool(torch.any((efficiency <= 0.0) | (efficiency > 1.0)).item()):
            raise ValueError("Irrigation efficiency parameters must be in (0, 1]")
        
        # Apply irrigation where needed and quota available
        has_need = self._to_bool(torch.any(needs_irrigation))
        if has_need and has_quota:
            # References: FAO irrigation scheduling guidance, FAO-56 root-zone
            # water balance, and CF-1.8 water-flux units.
            # Expression: D_net = max((theta_fc-theta) Z_r rho_w - P dt, 0),
            # D_gross = D_net / eta, D_delivered = eta D_gross.
            # Parameters: theta values [m3 m-3], Z_r [m], rho_w [kg m-3],
            # P [kg m-2 s-1], eta [1], and dt [s].
            # Open route: abstraction is the managed source boundary and
            # delivered irrigation is the root-zone water input.
            # Simplification: all precipitation counts as effective (P_e = P);
            # FAO effective-precipitation methods (USDA-SCS; FAO/AGLW) give
            # P_e <= P, so D_net underestimates the FAO net requirement.
            root_zone_depth = self._as_param_tensor(
                self._get_irrigation_param("root_zone_depth"),
                soil_moisture,
            )
            water_density_value = float(get_physical_constant("water_density"))
            water_density = soil_moisture.new_tensor(water_density_value)
            water_deficit = (
                torch.clamp(field_capacity - soil_moisture, min=0.0)
                * root_zone_depth
                * water_density
            )

            max_rate_mm_hr = self._as_param_tensor(
                self._get_irrigation_param("max_application_rate"),
                soil_moisture,
            )
            # The maximum application rate is a liquid-water depth rate of
            # irrigation_flux; the converter's liquid-water-equivalent rule uses
            # the physical-constant water density (CF "lwe" convention).
            max_application_flux = unit_converter.convert(
                max_rate_mm_hr,
                "irrigation_flux",
                "mm hr-1",
                "kg m-2 s-1",
            )
            dt_tensor = torch.as_tensor(
                dt_value,
                dtype=soil_moisture.dtype,
                device=soil_moisture.device,
            )
            max_application_tensor = (
                torch.ones_like(water_deficit) * max_application_flux * dt_tensor
            )

            # References: FAO irrigation application efficiency guidance.
            # Expression: D_gross = D_net / eta and D_delivered = eta * D_gross.
            # Parameters: D_net root-zone deficit [mm], D_gross abstraction [mm],
            # eta application efficiency [-], and precipitation flux times dt
            # [kg m-2].
            # Open-system route: abstraction is the source boundary and delivered
            # irrigation is the soil-water input; the difference is explicit loss.
            precip_amount = precipitation_flux * dt_tensor
            net_requirement = torch.clamp(water_deficit - precip_amount, min=0.0)
            abstraction_amount = torch.minimum(
                net_requirement / efficiency,
                max_application_tensor,
            )

            # Apply only where needed
            abstraction_amount = abstraction_amount * needs_irrigation.float()

            # References: Yates, D., Sieber, J., Purkey, D. and Huber-Lee, A. (2005),
            #   WEAP21 - a demand-, priority-, and preference-driven water planning
            #   model, Part 1, Water International 30(4), doi:10.1080/02508060508691893
            #   (demand sites of one priority are supplied at an equal percentage of
            #   their demand).
            # Mathematical expression: D = sum_i A_i h; s = min(1, Q / D) (s = 1 when
            #   D = 0); A_i' = s A_i; A_used = sum_i A_i' h = min(D, Q).
            # Parameter meanings: A_i the gross abstraction demand of cell i, h the
            #   mass-to-depth factor, D the domain demand, Q the remaining quota, s the
            #   common supply fraction of all irrigated cells (one priority).
            # Units: A_i, A_i' [kg m-2]; h [mm m2 kg-1]; D, Q, A_used [mm]; s [1].
            # Domain/constraints: 0 <= s <= 1; the abstraction uses exactly min(D, Q),
            #   so no overdraft is discarded; Q' = Q - A_used >= 0.
            # Sign convention: abstraction and quota use are non-negative.
            kg_m2_to_mm = self._kg_m2_to_mm_factor(soil_moisture)
            demand_mm = abstraction_amount.sum() * kg_m2_to_mm
            supply_fraction = torch.where(
                demand_mm > remaining_quota,
                remaining_quota / demand_mm,
                torch.ones_like(demand_mm),
            )
            abstraction_amount = abstraction_amount * supply_fraction
            irrigation_amount = abstraction_amount * efficiency

            # Convert delivered and source-boundary amounts to fluxes.
            irrigation_flux = irrigation_amount / dt_tensor
            abstraction_flux = abstraction_amount / dt_tensor

            # Quota ledger: A_used + Q' = Q. When the demand exceeds the quota the
            # quota is used up (Q' = 0) and A_used equals Q up to rounding, which is
            # checked against the declared conservation tolerance on the depth basis.
            used_mm = abstraction_amount.sum() * kg_m2_to_mm
            applied_water_mm = irrigation_amount.sum() * kg_m2_to_mm
            quota_exhausted = demand_mm > remaining_quota
            next_quota = torch.where(
                quota_exhausted,
                torch.zeros_like(remaining_quota),
                remaining_quota - used_mm,
            )
            ledger_residual = torch.abs(used_mm + next_quota - remaining_quota)
            ledger_tolerance = self._ensure_tensor(
                self._get_irrigation_param("conservation_tolerance"),
                dtype=soil_moisture.dtype,
            ).to(soil_moisture.device) * dt_tensor * kg_m2_to_mm
            if bool((ledger_residual > torch.clamp(
                1.0e-6 * remaining_quota, min=ledger_tolerance
            )).item()):
                raise RuntimeError(
                    "Irrigation quota ledger does not close: "
                    f"|A_used + Q' - Q| = {float(ledger_residual)} mm"
                )
        
        # Water balance conservation check
        irrigation_flux = self._reshape_output(irrigation_flux)
        abstraction_flux = self._reshape_output(abstraction_flux)
        _ = irrigation_flux * dt_value  # kg/m2
        
        # Check conservation under the same accounting basis:
        # delivered irrigation should equal abstraction adjusted by efficiency.
        conservation_error = torch.zeros(1, dtype=irrigation_flux.dtype, device=irrigation_flux.device)
        has_irrigation_flux = self._to_bool(irrigation_flux.sum() > 0)
        if has_irrigation_flux:
            tolerance = self._ensure_tensor(
                self._get_irrigation_param("conservation_tolerance"),
                dtype=irrigation_flux.dtype
            ).to(irrigation_flux.device)
            delivered_from_abstraction = abstraction_flux * efficiency
            conservation_error = torch.abs(irrigation_flux - delivered_from_abstraction).mean()
            if conservation_error > tolerance:
                logger.warning(
                    "Water accounting imbalance: %.6f kg/m2/s",
                    self._scalar_from_value(conservation_error),
                )
        
        # Use unified moisture response for efficiency adjustment
        moisture_factor = self._standard_moisture_response(
            soil_moisture,
            method='linear',
            params={
                "min": wilting_point.detach(),
                "max": field_capacity.detach(),
            }
        )
        # Report the same system efficiency used by the water-port ledger.
        effective_efficiency = efficiency
        
        # Prepare output dictionary
        outputs: FluxDict = {
            "irrigation_flux": UpdateItem(
                value=irrigation_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "abstraction_flux": UpdateItem(
                value=abstraction_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "irrigation_remaining_quota": UpdateItem(
                value=next_quota,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="mm",
                source_functor=self.name,
            ),
            "diagnostics.irrigation.applied_water": UpdateItem(
                value=applied_water_mm.to(
                    dtype=irrigation_flux.dtype,
                    device=irrigation_flux.device,
                ).reshape(1, 1),
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="mm",
                source_functor=self.name,
            ),
            "diagnostics.irrigation.effective_efficiency": UpdateItem(
                value=effective_efficiency,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="1",
                source_functor=self.name,
            ),
            "diagnostics.irrigation.moisture_response": UpdateItem(
                value=moisture_factor,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="1",
                source_functor=self.name,
            ),
            "diagnostics.irrigation.water_balance_residual": UpdateItem(
                value=(
                    conservation_error
                    if has_irrigation_flux
                    else torch.zeros_like(conservation_error)
                ),
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
        }
        
        return outputs

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Return the explicit open-boundary water-ledger residual.

        References: Allen et al. (1998), FAO-56 irrigation water balance.
        Expression: R_w = abs(F_irrigation - eta * F_abstraction).
        Parameters: F_irrigation and F_abstraction [kg m-2 s-1], eta [1].
        Open-system route: abstraction is the source withdrawal and irrigation
        is the delivered root-zone boundary flux; their difference is loss.
        """
        # References: Allen et al. (1998), FAO-56 irrigation water balance.
        # Mathematical expression: R_w=F_irrigation-eta F_abstraction; e_w=max|R_w|.
        # Parameter meanings: F_irrigation is delivery, F_abstraction withdrawal, eta efficiency, and e_w error.
        # Units: F terms, R_w, and e_w [kg m-2 s-1]; eta [1].
        # Domain/constraints: the canonical residual diagnostic is present, floating, finite, and non-empty.
        # Sign convention: positive R_w is excess delivery; returned e_w is a non-negative magnitude.
        del input_state
        residual_key = "diagnostics.irrigation.water_balance_residual"
        if residual_key not in output_updates:
            raise KeyError(f"Missing conservation diagnostic '{residual_key}'")
        residual = output_updates[residual_key]
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, torch.Tensor):
            raise TypeError(
                f"Conservation diagnostic '{residual_key}' must be a Tensor"
            )
        if residual.numel() == 0:
            raise ValueError(
                f"Conservation diagnostic '{residual_key}' must be non-empty"
            )
        if not torch.is_floating_point(residual):
            raise TypeError(
                f"Conservation diagnostic '{residual_key}' must be floating point"
            )
        if not bool(torch.isfinite(residual).all().item()):
            raise ValueError(
                f"Conservation diagnostic '{residual_key}' must be finite"
            )
        magnitude = residual.abs().amax()
        return {"mass": float(magnitude.detach().item())}
