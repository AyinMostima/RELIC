"""
Grazing Management Module

This module implements livestock grazing management including vegetation
consumption, trampling effects, and nutrient cycling through dung and urine.

References:
    - Conant, R.T., et al. (2001). Grassland management and conversion into grassland: Effects on soil carbon. Ecological Applications 11, 343-355.
    - Milchunas, D.G. & Lauenroth, W.K. (1993). Quantitative effects of grazing on vegetation and soils over a global range of environments. Ecological Monographs 63, 327-366.
    - McNaughton, S.J. (1985). Ecology of a grazing ecosystem: The Serengeti. Ecological Monographs 55, 259-294.
    - Howland, B., et al. (2016). Partitioning global land degradation by grazing versus climate. Global Change Biology 22, 1560-1571.

Mathematical Expression:
    Grazing intake rate:
    I = a * W^b * D * Q * (1 - exp(-k * B))
    
    where:
    - I is daily intake rate [kg DM d?^1]
    - W is animal body weight [kg]
    - D is stocking density [animals ha?^1]
    - Q is forage quality factor [1]
    - B is biomass availability [kg DM ha?^1]
    - a, b, k are species-specific parameters
    
    Open livestock matter boundary:
    C_intake = C_dung + C_livestock_export
    N_intake = N_dung + N_urine + N_livestock_export
"""

import math

import torch
from typing import Any, Dict, List, Mapping, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.functor import BiologicalProcessFunctor
from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin
from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS, advance_reaction_network
from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant
from .event_scheduler import resolve_day_of_year

class LivestockType(Enum):
    """Types of grazing livestock"""
    CATTLE = "cattle"           # Beef and dairy cattle
    SHEEP = "sheep"             # Sheep
    GOAT = "goat"              # Goats
    HORSE = "horse"            # Horses
    BUFFALO = "buffalo"        # Water buffalo


class GrazingSystem(Enum):
    """Grazing management systems"""
    CONTINUOUS = "continuous"    # Continuous grazing
    ROTATIONAL = "rotational"    # Rotational grazing
    STRIP = "strip"             # Strip grazing
    CELL = "cell"               # Cell grazing (high intensity, short duration)
    SEASONAL = "seasonal"        # Seasonal grazing
    REST_ROTATION = "rest_rotation"  # Rest-rotation system


class GrazingIntensity(Enum):
    """Grazing intensity levels"""
    LIGHT = "light"             # Low stocking rate
    MODERATE = "moderate"       # Moderate stocking rate
    HEAVY = "heavy"             # High stocking rate
    VERY_HEAVY = "very_heavy"   # Overgrazing level


@dataclass
class GrazingEvent:
    """
    Data class for grazing management events
    
    Attributes:
        livestock_type: Type of livestock
        system: Grazing system type
        intensity: Grazing intensity
        start_date: Start day of grazing period
        duration: Duration of grazing [days]
        stocking_rate: Number of animals per hectare [animals/ha]
        body_weight: Average animal body weight [kg]
        supplement_rate: Supplemental feed rate [kg DM/animal day-1]
    """
    livestock_type: LivestockType
    system: GrazingSystem
    intensity: GrazingIntensity
    start_date: int | float  # Day of year; runtime validation requires an integer value
    duration: float  # days
    stocking_rate: float  # animals/ha
    body_weight: float  # kg
    supplement_rate: float  # kg DM/animal day-1


class GrazingManagement(NamespaceDefaultMixin, BiologicalProcessFunctor):
    """
    Grazing Management Functor
    
    Implements livestock grazing effects on vegetation including consumption,
    trampling, and nutrient cycling through excreta deposition.
    
    Parameters:
        grazing_schedule: List of grazing events
        intake_coefficient: Intake rate coefficient [kg/animal day-1]
        digestibility: Feed digestibility [1]
        n_retention: Nitrogen retention in animal [1]
        p_retention: Phosphorus retention in animal [1]
        trampling_factor: Vegetation damage from trampling [1]
        selectivity_factor: Grazing selectivity effect [1]
    """
    
    parameter_lookup_namespace = "anthroposphere.management.grazing"
    parameter_lookup_source_order = ("manager",)
    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"

    def __init__(self, name: str = "grazing_management", config: Optional[Dict] = None):
        """
        Initialize grazing management functor
        
        Args:
            name: Functor name
            config: Configuration dictionary
        """
        # Set up metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Livestock grazing management and vegetation-animal interactions",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "Canonical IPHS: x_dot = N_h r_h + G_h u_grazing, r_h = r_grazing; "
                "I = a * W^b * D * Q * (1 - exp(-k * B)); "
                "C_intake = C_dung + C_livestock_export; "
                "N_intake = N_dung + N_urine + N_livestock_export"
            ),
            references=[
                "Conant et al. (2001) DOI:10.1890/1051-0761(2001)011[0343:GMACIG]2.0.CO;2",
                "Milchunas & Lauenroth (1993) DOI:10.2307/2937150",
                "McNaughton (1985) DOI:10.2307/1942578",
                "Howland et al. (2016) DOI:10.1111/gcb.13199"
            ],
            parameters={
                "anthroposphere.management.clamp_min": (
                    "Lower bound of the management response factors [dimensionless]"
                ),
                "anthroposphere.management.clamp_max": (
                    "Upper bound of the management response factors [dimensionless]"
                ),
                "anthroposphere.management.grazing.base_efficiency": (
                    "Baseline efficiency scalar of grazing management [dimensionless]"
                ),
                "anthroposphere.management.grazing.digestibility_coefficient": (
                    "Feed digestibility coefficient [dimensionless]"
                ),
                "anthroposphere.management.grazing.selectivity_factor": (
                    "Grazing selectivity effect [dimensionless]"
                ),
                "anthroposphere.management.grazing.max_utilization": (
                    "Maximum allowable fraction of aboveground biomass removed by grazing "
                    "[dimensionless]"
                ),
                "anthroposphere.management.grazing.forage_carbon_fraction": (
                    "Forage carbon mass fraction [dimensionless]"
                ),
                "anthroposphere.management.grazing.functional_response_scalar": (
                    "Scalar k for intake functional response I = 1 - exp(-k * biomass_kg_ha) [ha "
                    "kg-1]"
                ),
                "anthroposphere.management.grazing.n_retention": (
                    "Nitrogen retention in animal [dimensionless]"
                ),
                "anthroposphere.management.grazing.dung_partition": (
                    "Fraction of excreted nitrogen routed to dung [dimensionless]"
                ),
                "anthroposphere.management.grazing.urine_partition": (
                    "Fraction of excreted nitrogen routed to urine [dimensionless]"
                ),
                "anthroposphere.management.grazing.trampling_factor": (
                    "Vegetation damage from trampling [dimensionless]"
                ),
                "anthroposphere.management.grazing.trampling_moisture_threshold": (
                    "Soil moisture threshold where trampling damage increases [dimensionless]"
                ),
                "anthroposphere.management.grazing.trampling_moisture_scale": (
                    "Moisture scaling factor for trampling damage [dimensionless]"
                ),
                "anthroposphere.management.grazing.trampling_stocking_normalization": (
                    "Stocking rate normalization for trampling response [animals ha-1]"
                ),
                "anthroposphere.management.grazing.livestock.*": (
                    "Digestibility, intake rate, metabolism factor, N and P content and trampling "
                    "impact per livestock type [units per key]"
                ),
                "anthroposphere.management.grazing.default_schedule.*": (
                    "Default spring and winter grazing events (start date, duration, stocking "
                    "rate, body weight, supplement rate) [units per key]"
                ),
            },
            input_variables=[
                "aboveground_biomass",
                "vegetation_quality",
                "day_of_year",
                "soil_moisture",
            ],
            output_variables=[
                "aboveground_biomass",
                "grazing_flux",
                "dung_c_flux",
                "dung_n_flux",
                "urine_n_flux",
                "livestock_carbon_intake_flux",
                "livestock_nitrogen_intake_flux",
                "livestock_carbon_export_flux",
                "livestock_nitrogen_export_flux",
                "trampling_damage",
                "grazing_active",
                "current_stocking_rate",
                "utilization_rate",
                "cumulative_intake",
                "days_grazed",
                "diagnostics.grazing_pressure",
                "diagnostics.grazing_management.canonical_iphs_runtime",
            ],
            conservation_laws=["grazing_matter_boundary"],
            spatial_dimensions=2,
            temporal_scale=TemporalScale.DAY,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            coupling_interfaces=[
                "biosphere",
                "pedosphere",
                "grazing_dung_boundary",
                "grazing_urine_boundary",
            ],
        )
        
        # Default configuration; numeric grazing parameters are managed in the parameter database
        default_config = {"dependencies": []}
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        # Initialize BiologicalProcessFunctor base class (includes FunctorBase initialization)
        super().__init__(
            name=name,
            config=default_config,
            metadata=metadata,
        )
        configured_calendar = default_config.get("calendar_type")
        self._configured_calendar_type = (
            None
            if configured_calendar is None
            else str(configured_calendar)
        )
        self.process_name = "grazing"
        
        # Initialize grazing schedule
        self.grazing_events: List[GrazingEvent] = []
        self._initialize_default_schedule()
        
        self.runtime_path = "canonical_iphs_grazing_biomass_return_port"

    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Resolve structural audit size for grazing matter channels."""

        value = 4 if n_dofs is None else int(n_dofs)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare cochain spaces for the grazing biomass-return port map.

        References:
          - Milchunas and Lauenroth (1993), grazing effects on vegetation.
          - Conant et al. (2001), grassland management and soil carbon.
          - Ramirez and Le Gorrec (2022), irreversible port-Hamiltonian systems.
        Mathematical expression:
          x_dot = N_h r_h + G_h u_grazing, with r_h = r_grazing and
          I = a W^b D Q (1 - exp(-k B)).
        Parameter meanings:
          x is grazable biomass [kg DM m^-2], r_h contains removal and
          excreta return fluxes [kg m^-2 s^-1], and u_grazing is the stocking
          management port input.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="anthroposphere_grazing",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="kg m^-2",
                extensivity="density",
                physical_domain="biosphere",
                semantic_name="grazing_biomass_pool",
                orientation_convention="positive_vegetation_storage",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_grazing",
                degree=0,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m^-2 s^-1",
                extensivity="density",
                physical_domain="pedosphere",
                semantic_name="grazing_return_flux",
                orientation_convention="positive_to_soil_return",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_grazing",
                degree=0,
                primality="dual",
                n_dofs=1,
                units="animals ha^-1",
                extensivity="intensive",
                physical_domain="anthroposphere",
                semantic_name="grazing_stocking_port",
                orientation_convention="positive_stocking_pressure",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager namespaces used by runtime and structure."""

        return (
            "anthroposphere.management.grazing.*",
            "anthroposphere.management.grazing.livestock.*",
            "anthroposphere.management.*",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive grazing removal-return mobility Hodge contract.

        References:
          - Hodgson (1990), grazing management functional response.
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical expression:
          H_grazing = diag(w_i / sum_j w_j), w_i >= 0.
        Parameter meanings:
          w_i are ParameterManager-backed utilization, digestibility,
          retention, and excreta partition coefficients.
        """
        # References: Hodgson (1990); van der Schaft and Maschke (1995).
        # Mathematical expression: H_g = diag(w_i / sum_j w_j), with w_i = abs(theta_i).
        # Parameter meanings: w_i are grazing partition weights and n_dofs is support size.
        # Units: weights and matrix entries [1]; domain [kg m-2]; codomain [kg m-2 s-1].
        # Domain/constraints: n_dofs >= 1; weights are finite, non-negative, and have positive sum.
        # Sign convention: positive mobility maps storage toward positive removal or return flows.

        n_dofs = self._structural_n_dofs(n_dofs)
        keys = (
            "max_utilization",
            "digestibility_coefficient",
            "n_retention",
            "forage_carbon_fraction",
        )
        values = [abs(float(self.require_scalar(key, namespace="anthroposphere.management.grazing"))) for key in keys]
        if len(values) < n_dofs:
            values.extend([values[-1] if values else 1.0] * (n_dofs - len(values)))
        weights = torch.as_tensor(
            values[:n_dofs],
            device=getattr(self, "device", torch.device("cpu")),
            dtype=torch.float32,
        )
        total = torch.sum(weights)
        if total <= 0:
            raise ValueError("Grazing mobility weights require positive support.")
        matrix = torch.diag(weights / total)
        domain = CochainSpaceSpec(
            complex_id="anthroposphere_grazing",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="kg m^-2",
            extensivity="density",
            physical_domain="biosphere",
            semantic_name="grazing_biomass_pool",
            orientation_convention="positive_vegetation_storage",
            boundary_role="interior",
        )
        codomain = CochainSpaceSpec(
            complex_id="anthroposphere_grazing",
            degree=0,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m^-2 s^-1",
            extensivity="density",
            physical_domain="pedosphere",
            semantic_name="grazing_return_flux",
            orientation_convention="positive_to_soil_return",
            boundary_role="interior",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_removal_return_mobility_hodge",
            domain=domain,
            codomain=codomain,
            matrix=matrix,
            parameter_keys=tuple(f"anthroposphere.management.grazing.{key}" for key in keys),
            metric_keys=("validation.conservation.conservation_tolerance",),
            material_law="grazing_biomass_return_partition_mobility",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build Hodge terms for IPHS structural assembly."""

        _ = complex_bundle, resolved_parameters
        return (self.build_hodge_contract(strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no independent energy storage for this open grazing port."""

        _ = state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the non-negative grazing dissipation audit term."""

        _ = state
        return {
            "grazing_removal_return_entropy": "sigma = z.T H_grazing z >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the declared grazing reaction-network metadata."""
        # References: Milchunas and Lauenroth (1993); Ramirez and Le Gorrec (2022).
        # Mathematical expression: x_dot = N_h r_h + G_h u_g, with r_h = r_grazing.
        # Parameter meanings: x is the channel state, r_h the grazing rates, and u_g stocking control.
        # Units: channel units are [kg m-2], [1], [kg element m-2 s-1], or [animals ha-1].
        # Domain/constraints: only declared state channels and ports enter this structural metadata.
        # Sign convention: removal/export is outward-positive and dung/urine return is soil-inward-positive.

        _ = state, complex_bundle
        return {
            "state_vars": (
                "aboveground_biomass",
                "vegetation_quality",
                "grazing_flux",
                "dung_c_flux",
                "urine_n_flux",
            ),
            "reaction_map": "x_dot = N_h r_h + G_h u_grazing, r_h = r_grazing",
            "mass_conserving_partition": True,
            "ports": self.ports(),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive grazing removal-return mobility operator."""
        # References: Hodgson (1990); Ramirez and Le Gorrec (2022).
        # Mathematical expression: J = 0 and M = H_g, with z.T M z >= 0.
        # Parameter meanings: M is grazing mobility and z is a compatible effort vector.
        # Units: M entries [1]; connected stock and flow spaces are [kg m-2] and [kg m-2 s-1].
        # Domain/constraints: H_g is square, finite, normalized, and positive semidefinite.
        # Sign convention: positive dual flow follows the declared removal or soil-return orientation.

        _ = state, complex_bundle
        hodge = self.build_hodge_contract(strict=True)
        matrix = hodge.matrix
        return {
            "state_vars": tuple(f"grazing_biomass_pool[{idx}]" for idx in range(int(matrix.shape[0]))),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the grazing stocking management port."""

        effort_space = CochainSpaceSpec(
            complex_id="anthroposphere_grazing",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="animals ha^-1",
            extensivity="intensive",
            physical_domain="anthroposphere",
            semantic_name="grazing_stocking_effort",
            orientation_convention="positive_stocking_pressure",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="anthroposphere_grazing",
            degree=0,
            primality="dual",
            n_dofs=1,
            units="kg m^-2 s^-1",
            extensivity="density",
            physical_domain="biosphere",
            semantic_name="grazing_biomass_removal_flux",
            orientation_convention="positive_biomass_removal",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="grazing_biomass_return_port",
                effort_var="grazing_stocking_effort",
                flow_var="grazing_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="kg m^-2 s^-1",
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
        """Run the canonical IPHS grazing biomass-return runtime."""

        return self.forward(dict(state), time=time, dt=dt)

    def _seconds_per_day_value(self) -> float:
        """Resolve the physical day length through the physical constants SSOT."""

        return float(
            get_physical_constant(
                "relic.physical.constants.time.seconds_per_day"
            )
        )

    def _attach_canonical_runtime_diagnostic(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Attach the Phase 8 canonical runtime diagnostic flag."""

        device = getattr(self, "device", torch.device("cpu"))
        dtype = torch.float32
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                dtype = value.dtype if torch.is_floating_point(value) else torch.float32
                break
        outputs["diagnostics.grazing_management.canonical_iphs_runtime"] = torch.ones(
            (), device=device, dtype=dtype
        )
        return outputs
    
    def _initialize_default_schedule(self) -> None:
        """Initialize default grazing schedule using parameters"""
        # Try to load spring schedule parameters
        spring_start = self.require_scalar(
            "default_schedule.spring.start_date",
            namespace="anthroposphere.management.grazing",
        )
        spring_duration = self.require_scalar(
            "default_schedule.spring.duration",
            namespace="anthroposphere.management.grazing",
        )
        spring_stocking = self.require_scalar(
            "default_schedule.spring.stocking_rate",
            namespace="anthroposphere.management.grazing",
        )
        spring_weight = self.require_scalar(
            "default_schedule.spring.body_weight",
            namespace="anthroposphere.management.grazing",
        )
        spring_supp = self.require_scalar(
            "default_schedule.spring.supplement_rate",
            namespace="anthroposphere.management.grazing",
        )

        # Spring/summer grazing period
        self.grazing_events.append(
            GrazingEvent(
                livestock_type=LivestockType.CATTLE,
                system=GrazingSystem.ROTATIONAL,
                intensity=GrazingIntensity.MODERATE,
                start_date=spring_start,
                duration=float(spring_duration),
                stocking_rate=float(spring_stocking),
                body_weight=float(spring_weight),
                supplement_rate=float(spring_supp)
            )
        )
        
        # Try to load winter schedule parameters
        winter_start = self.require_scalar(
            "default_schedule.winter.start_date",
            namespace="anthroposphere.management.grazing",
        )
        winter_duration = self.require_scalar(
            "default_schedule.winter.duration",
            namespace="anthroposphere.management.grazing",
        )
        winter_stocking = self.require_scalar(
            "default_schedule.winter.stocking_rate",
            namespace="anthroposphere.management.grazing",
        )
        winter_weight = self.require_scalar(
            "default_schedule.winter.body_weight",
            namespace="anthroposphere.management.grazing",
        )
        winter_supp = self.require_scalar(
            "default_schedule.winter.supplement_rate",
            namespace="anthroposphere.management.grazing",
        )

        # Winter feeding/light grazing
        self.grazing_events.append(
            GrazingEvent(
                livestock_type=LivestockType.CATTLE,
                system=GrazingSystem.CONTINUOUS,
                intensity=GrazingIntensity.LIGHT,
                start_date=winter_start,
                duration=float(winter_duration),
                stocking_rate=float(winter_stocking),
                body_weight=float(winter_weight),
                supplement_rate=float(winter_supp)
            )
        )
    
    def add_grazing_event(self, event: GrazingEvent) -> None:
        """
        Add a grazing event to schedule
        
        Args:
            event: GrazingEvent defining grazing details
        """
        self.grazing_events.append(event)
    
    def _get_livestock_characteristics(self, livestock_type: LivestockType) -> Dict[str, float]:
        """
        Get feeding characteristics for livestock types
        
        Args:
            livestock_type: Type of livestock
            
        Returns:
            Dictionary with intake parameters, digestibility, etc.
        """
        # References: Hodgson (1990); IPCC (2019); Greenwood and McKenzie (2001).
        # Mathematical expression: theta_l = PM[grazing.livestock.livestock_type.*].
        # Parameter meanings: theta_l contains intake, digestibility, elemental fractions, and impacts.
        # Units: intake [% body weight day-1]; N/P [kg kg-1]; other returned factors [1].
        # Domain/constraints: livestock_type is an enum and ParameterManager validates finite bounds.
        # Sign convention: all returned values are non-negative magnitudes and assign no flux direction.
        livestock_name = livestock_type.value
        
        intake_rate = self.require_scalar(
            f"grazing.livestock.{livestock_name}.intake_rate",
            namespace="anthroposphere.management",
        )
        digestibility = self.require_scalar(
            f"grazing.livestock.{livestock_name}.digestibility",
            namespace="anthroposphere.management",
        )
        trampling_impact = self.require_scalar(
            f"grazing.livestock.{livestock_name}.trampling_impact",
            namespace="anthroposphere.management",
        )
        
        # N/P content defaults (can be parameterized similarly if needed)
        n_content = self.require_scalar(
            f"grazing.livestock.{livestock_name}.n_content",
            namespace="anthroposphere.management",
        )
        p_content = self.require_scalar(
            f"grazing.livestock.{livestock_name}.p_content",
            namespace="anthroposphere.management",
        )
        metabolism_factor = self.require_scalar(
            f"grazing.livestock.{livestock_name}.metabolism_factor",
            namespace="anthroposphere.management",
        )
        
        return {
            "intake_rate": float(intake_rate),
            "digestibility": float(digestibility),
            "n_content": float(n_content),
            "p_content": float(p_content),
            "metabolism_factor": float(metabolism_factor),
            "trampling_impact": float(trampling_impact)
        }
    
    def _calculate_daily_intake(self,
                              livestock_chars: Dict[str, float],
                              body_weight: float,
                              stocking_rate: float | torch.Tensor,
                              biomass_availability: torch.Tensor,
                              vegetation_quality: torch.Tensor) -> torch.Tensor:
        """
        Calculate daily intake per unit area
        
        Args:
            livestock_chars: Livestock characteristics
            body_weight: Average animal body weight [kg]
            stocking_rate: Stocking rate [animals/ha]
            biomass_availability: Available biomass [kg DM/m^2]
            vegetation_quality: Canonical forage quality factor [1]
            
        Returns:
            Daily intake rate [kg DM/m^2 day-1]
        """
        # References: Hodgson (1990); Milchunas and Lauenroth (1993).
        # Mathematical expression: I = W(a/100) Q s_Q [1-exp(-k 10000 B)] D/10000.
        # Parameter meanings: W is body mass, a intake percent, Q quality, B biomass, and D stocking.
        # Units: W [kg animal-1], k [ha kg-1], B [kg DM m-2], D [animals ha-1], I [kg DM m-2 day-1].
        # Domain/constraints: W, B, D, and k are non-negative; Q and s_Q are bounded factors.
        # Sign convention: I >= 0 is an outward vegetation dry-matter removal magnitude.
        # Base intake rate (% of body weight)
        intake_rate_pct = livestock_chars["intake_rate"] / 100.0
        
        # Daily intake per animal [kg DM/animal day-1]
        intake_per_animal = body_weight * intake_rate_pct
        
        # Adjust for forage quality
        selectivity_factor = self.require_scalar(
            "selectivity_factor",
            namespace="anthroposphere.management.grazing",
        )
        selectivity_tensor = self._ensure_tensor(
            selectivity_factor,
            dtype=vegetation_quality.dtype,
        ).to(vegetation_quality.device)
        quality_factor = vegetation_quality * selectivity_tensor
        intake_per_animal *= quality_factor
        
        # Biomass availability constraint (functional response)
        biomass_kg_ha = biomass_availability * 10000  # Convert m^2 to ha
        functional_response = self.require_scalar(
            "functional_response_scalar",
            namespace="anthroposphere.management.grazing",
        )
        functional_response_tensor = self._ensure_tensor(
            functional_response, dtype=biomass_kg_ha.dtype
        ).to(biomass_kg_ha.device)
        # Functional response: I = 1 - exp(-k * biomass) (Hodgson 1990)
        availability_factor = 1.0 - torch.exp(-functional_response_tensor * biomass_kg_ha)
        intake_per_animal *= availability_factor
        
        # Total intake per unit area [kg DM/m^2 day-1]
        stocking_rate_m2 = stocking_rate / 10000  # animals/m^2
        total_intake = intake_per_animal * stocking_rate_m2
        
        return total_intake
    
    def _calculate_excreta_return(self,
                                 intake_rate: torch.Tensor,
                                 livestock_chars: Dict[str, float]) -> Tuple[
                                     torch.Tensor,
                                     torch.Tensor,
                                     torch.Tensor,
                                     torch.Tensor,
                                     torch.Tensor,
                                 ]:
        """
        Partition forage carbon and nitrogen across explicit boundary ports.

        References:
            DNDC grazing and manure modules partition daily intake carbon and
            nitrogen into dung, urine, animal retention, respiration, and
            enteric emissions before downstream temperature responses.
        Mathematical expression:
            C_in = I * f_C
            C_dung = I * (1 - d) * f_C
            C_export = C_in - C_dung
            N_in = I * f_N
            N_dung + N_urine = N_in * (1 - f_ret)
            N_export = N_in * f_ret
        Parameter meanings:
            I is forage dry-matter intake [kg DM m-2 day-1]; f_C and f_N are
            forage carbon and nitrogen mass fractions [1]; d is digestibility
            [1]; f_ret is livestock nitrogen retention [1].
        
        Args:
            intake_rate: Daily intake rate [kg DM/m^2 day-1]
            livestock_chars: Livestock characteristics
        Returns:
            Dung C, dung N, urine N, carbon intake, and nitrogen intake rates
            [kg element m-2 day-1]. Export ports are closed from these terms
            after the common per-second conversion.
        """
        # References: IPCC (2019); DNDC grazing and manure partition formulation.
        # Mathematical expression: C_dung=I(1-d)f_C; N_dung+N_urine=If_N(1-f_ret).
        # Parameter meanings: I is intake; d digestibility; f_C/f_N contents; f_ret N retention.
        # Units: I [kg DM m-2 day-1]; carbon and nitrogen outputs [kg element m-2 day-1].
        # Domain/constraints: fractions lie in [0,1] and dung/urine weights have positive support.
        # Sign convention: intake/export is outward-positive and dung/urine return is soil-inward-positive.
        fraction_values: Dict[str, float] = {}
        for fraction_name in ("digestibility", "n_content"):
            raw_fraction = livestock_chars[fraction_name]
            if isinstance(raw_fraction, bool):
                raise ValueError(
                    f"{fraction_name} must be a finite mass fraction within [0, 1]"
                )
            try:
                fraction = float(raw_fraction)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{fraction_name} must be a finite mass fraction within [0, 1]"
                ) from exc
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"{fraction_name} must be a finite mass fraction within [0, 1]"
                )
            fraction_values[fraction_name] = fraction

        digestibility = fraction_values["digestibility"]
        n_content = fraction_values["n_content"]
        n_retention_param = self.require_scalar(
            "n_retention",
            namespace="anthroposphere.management.grazing",
        )
        n_retention = self._ensure_tensor(n_retention_param, dtype=intake_rate.dtype).to(intake_rate.device)
        
        # Fresh excreta generation is an intake partition. Temperature acts on
        # downstream decomposition or volatilization, not this mass split.
        dung_dm = intake_rate * (1.0 - digestibility)

        forage_carbon_fraction_param = self.require_scalar(
            "forage_carbon_fraction",
            namespace="anthroposphere.management.grazing",
        )
        forage_carbon_fraction = self._ensure_tensor(
            forage_carbon_fraction_param, dtype=intake_rate.dtype
        ).to(intake_rate.device)
        carbon_intake = intake_rate * forage_carbon_fraction
        dung_c = dung_dm * forage_carbon_fraction
        nitrogen_intake = intake_rate * n_content
        nitrogen_excreted = nitrogen_intake * (1.0 - n_retention)
        
        urine_partition_param = self.require_scalar(
            "urine_partition",
            namespace="anthroposphere.management.grazing",
        )
        dung_partition_param = self.require_scalar(
            "dung_partition",
            namespace="anthroposphere.management.grazing",
        )
        urine_partition = self._ensure_tensor(
            urine_partition_param, dtype=intake_rate.dtype
        ).to(intake_rate.device)
        dung_partition = self._ensure_tensor(
            dung_partition_param, dtype=intake_rate.dtype
        ).to(intake_rate.device)
        partition_weights = torch.stack((urine_partition, dung_partition))
        partition_sum = urine_partition + dung_partition
        if (
            not bool(torch.isfinite(partition_weights).all().item())
            or bool(torch.any(partition_weights < 0.0).item())
            or not bool(torch.gt(partition_sum, 0.0).item())
        ):
            raise ValueError(
                "Grazing excreta partition weights must be finite, "
                "non-negative, and have positive support"
            )
        urine_fraction = urine_partition / partition_sum
        dung_fraction = dung_partition / partition_sum
        urine_n = nitrogen_excreted * urine_fraction
        dung_n = nitrogen_excreted * dung_fraction
        return (
            dung_c,
            dung_n,
            urine_n,
            carbon_intake,
            nitrogen_intake,
        )
    
    def _calculate_trampling_damage(self,
                                  livestock_chars: Dict[str, float],
                                  stocking_rate: float | torch.Tensor,
                                  soil_moisture: torch.Tensor) -> torch.Tensor:
        """
        Calculate vegetation damage from trampling
        
        Args:
            livestock_chars: Livestock characteristics
            stocking_rate: Stocking rate [animals/ha]
            soil_moisture: Soil moisture content [m^3/m^3]
            
        Returns:
            Trampling damage factor [1]
        """
        # References: Greenwood and McKenzie (2001); Conant et al. (2001).
        # Mathematical expression: D_t=clamp(f_t i_t D/D_ref [1+s max(theta-theta_0,0)],L,U).
        # Parameter meanings: f_t is base damage, i_t impact, D stocking, and theta soil moisture.
        # Units: D and D_ref [animals ha-1]; theta, f_t, i_t, s, L, U, and D_t [1].
        # Domain/constraints: D >= 0, D_ref > 0, theta in [0,1], and ParameterManager bounds apply.
        # Sign convention: D_t >= 0 is damage severity, not a signed biomass flux.
        base_trampling_param = self.require_scalar(
            "trampling_factor",
            namespace="anthroposphere.management.grazing",
        )
        base_trampling = self._ensure_tensor(base_trampling_param, dtype=soil_moisture.dtype).to(soil_moisture.device)
        trampling_impact = livestock_chars["trampling_impact"]
        
        # Stocking rate effect
        stocking_normalization_param = self.require_scalar(
            "trampling_stocking_normalization",
            namespace="anthroposphere.management.grazing",
        )
        stocking_normalization = self._ensure_tensor(
            stocking_normalization_param, dtype=soil_moisture.dtype
        ).to(soil_moisture.device)
        stocking_effect = stocking_rate / stocking_normalization  # Normalized stocking intensity
        
        # Soil moisture effect (more damage when wet)
        moisture_scale_param = self.require_scalar(
            "trampling_moisture_scale",
            namespace="anthroposphere.management.grazing",
        )
        moisture_threshold_param = self.require_scalar(
            "trampling_moisture_threshold",
            namespace="anthroposphere.management.grazing",
        )
        moisture_scale = self._ensure_tensor(
            moisture_scale_param, dtype=soil_moisture.dtype
        ).to(soil_moisture.device)
        moisture_threshold = self._ensure_tensor(
            moisture_threshold_param, dtype=soil_moisture.dtype
        ).to(soil_moisture.device)
        # Trampling moisture response: D = 1 + scale * max(0, theta - theta_threshold)
        moisture_effect = 1.0 + moisture_scale * (soil_moisture - moisture_threshold).clamp(min=0.0)
        
        # Combined trampling damage
        damage = base_trampling * trampling_impact * stocking_effect * moisture_effect
        
        clamp_min_value = self.require_scalar(
            "clamp_min",
            namespace="anthroposphere.management",
        )
        clamp_max_value = self.require_scalar(
            "clamp_max",
            namespace="anthroposphere.management",
        )
        clamp_min_tensor = self._ensure_tensor(clamp_min_value, dtype=damage.dtype).to(damage.device)
        clamp_max_tensor = self._ensure_tensor(clamp_max_value, dtype=damage.dtype).to(damage.device)
        return torch.clamp(damage, min=clamp_min_tensor, max=clamp_max_tensor)

    def _calculate_grazing_pressure(
        self,
        stocking_rate: Any,
        vegetation_quality: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the effective stocking pressure diagnostic.

        References:
            Hodgson (1990); Milchunas and Lauenroth (1993).
        Mathematical expression:
            P_g = D_stock * Q_forage.
        Parameter meanings:
            D_stock is livestock stocking density [animals ha^-1].
            Q_forage is the dimensionless forage quality factor [1].
            P_g is effective grazing pressure [animals ha^-1].
        Open-system route:
            P_g is an observation diagnostic only; biomass removal and
            excreta matter use explicit grazing and return flux ports.
        """
        # References: Hodgson (1990); Milchunas and Lauenroth (1993).
        # Mathematical expression: P_g = D_stock Q_forage.
        # Parameter meanings: D_stock is the active event stocking rate, Q_forage quality,
        #   and P_g pressure.
        # Units: D_stock and P_g [animals ha-1]; Q_forage [1].
        # Domain/constraints: D_stock is finite and non-negative; Q_forage is finite in [0,1].
        # Sign convention: P_g >= 0 is an observation magnitude and carries no matter-flow sign.

        density = torch.as_tensor(
            stocking_rate,
            dtype=vegetation_quality.dtype,
            device=vegetation_quality.device,
        )
        if not torch.isfinite(density).all():
            raise ValueError("stocking_rate must contain only finite values")
        if torch.any(density < 0):
            raise ValueError("stocking_rate must be non-negative")
        if not torch.isfinite(vegetation_quality).all():
            raise ValueError("vegetation_quality must contain only finite values")
        if torch.any((vegetation_quality < 0) | (vegetation_quality > 1)):
            raise ValueError("vegetation_quality must be within [0, 1]")
        return density * vegetation_quality
    
    def _check_grazing_schedule(
        self,
        day_of_year: int,
        *,
        calendar_type: str,
    ) -> Optional[GrazingEvent]:
        """Check if grazing should be active today.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, time-window
            and calendar contract.
        Mathematical expression:
            delta_C(d,s) = (d-s) mod Y(C); active iff
            0 <= delta_C < duration <= Y(C).
        Parameter meanings:
            d is current day [day], s is event start [day], Y(C) is calendar
            year length [day], and duration is event duration [day].

        Args:
            day_of_year: Current day of the configured simulator year
            calendar_type: Existing simulator calendar identifier

        Returns:
            Active GrazingEvent if grazing should occur, None otherwise
        """
        # References: CF Conventions 1.8 Section 4.4; RELIC calendar contract.
        # Mathematical expression: delta_C=(d-s) mod Y(C); active iff 0 <= delta_C < duration.
        # Parameter meanings: d is current day, s start day, Y year length, and duration event span.
        # Units: d, s, delta_C, duration, and Y [day].
        # Domain/constraints: calendar days are valid and 0 <= duration <= Y(C).
        # Sign convention: elapsed time advances positively with wraparound at the year boundary.
        days_per_year = CalendarHandler.days_per_simulator_year(calendar_type)
        for event in self.grazing_events:
            start_day = CalendarHandler.validate_simulator_day(
                event.start_date,
                calendar_type,
            )
            if isinstance(event.duration, bool):
                raise ValueError(
                    "Grazing event duration must be finite and non-negative."
                )
            try:
                duration = torch.as_tensor(event.duration).detach()
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    "Grazing event duration must be finite and non-negative."
                ) from exc
            if (
                duration.ndim != 0
                or duration.dtype == torch.bool
                or duration.is_complex()
                or not bool(torch.isfinite(duration).item())
                or bool(torch.lt(duration, 0.0).item())
            ):
                raise ValueError(
                    "Grazing event duration must be finite and non-negative."
                )
            if bool(torch.gt(duration, float(days_per_year)).item()):
                raise ValueError(
                    "Grazing event duration must be no greater than the calendar year."
                )
            elapsed_days = torch.remainder(
                torch.as_tensor(
                    day_of_year - start_day,
                    dtype=torch.float64,
                    device=duration.device,
                ),
                days_per_year,
            )
            if bool(torch.lt(elapsed_days, duration).item()):
                return event

        return None

    def _assemble_grazing_reaction_outputs(
        self,
        *,
        grazing_flux: torch.Tensor,
        dung_c_flux: torch.Tensor,
        dung_n_flux: torch.Tensor,
        urine_n_flux: torch.Tensor,
        livestock_carbon_intake_flux: torch.Tensor,
        livestock_carbon_export_flux: torch.Tensor,
        livestock_nitrogen_intake_flux: torch.Tensor,
        livestock_nitrogen_export_flux: torch.Tensor,
        trampling_damage: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Assemble grazing outputs through the explicit IPHS reaction map.

        References:
          - Conant et al. (2001), grazing impacts on soil carbon.
          - Milchunas and Lauenroth (1993), vegetation and soil grazing effects.
          - Rao and Esposito (2016), reaction-network thermodynamics.
        Mathematical expression:
          y_dot = N_h r_h, with intake mapped to explicit dry-matter,
          carbon, nitrogen, excreta, livestock-export, and trampling ports.
        Parameter meanings:
          grazing_flux: livestock intake/removal rate [kg m^-2 s^-1].
          dung and urine fluxes: returned matter rates [kg m^-2 s^-1].
          trampling_damage: dimensionless diagnostic response rate.
        """
        # References: Conant et al. (2001); Rao and Esposito (2016).
        # Mathematical expression: y_dot = N_h r_h with N_h = I for declared open port channels.
        # Parameter meanings: r_h contains removal, C/N return/export, and trampling channels.
        # Units: matter channels [kg element m-2 s-1]; trampling channel [1].
        # Domain/constraints: rates share shape, dtype, and device and use an explicit open-system route.
        # Sign convention: each positive channel follows its declared outward or soil-return orientation.

        rate_names = (
            "grazing_flux",
            "dung_c_flux",
            "dung_n_flux",
            "urine_n_flux",
            "livestock_carbon_intake_flux",
            "livestock_carbon_export_flux",
            "livestock_nitrogen_intake_flux",
            "livestock_nitrogen_export_flux",
            "trampling_damage",
        )
        rates = (
            grazing_flux,
            dung_c_flux,
            dung_n_flux,
            urine_n_flux,
            livestock_carbon_intake_flux,
            livestock_carbon_export_flux,
            livestock_nitrogen_intake_flux,
            livestock_nitrogen_export_flux,
            trampling_damage,
        )
        grazing_network = advance_reaction_network(
            name=f"{self.name}_grazing_biomass_return_network",
            species=rate_names,
            reactions=tuple(f"{name}_route" for name in rate_names),
            stoichiometric_N=torch.eye(
                len(rate_names),
                device=grazing_flux.device,
                dtype=grazing_flux.dtype,
            ),
            rates=torch.stack(
                tuple(
                    rate.to(device=grazing_flux.device, dtype=grazing_flux.dtype)
                    for rate in rates
                ),
                dim=0,
            ),
            open_system_route="port",
        )
        return {name: grazing_network.species_rate(name) for name in rate_names}

    @staticmethod
    def _require_grazing_state(
        state: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate the three required scientific state ports."""

        names = (
            "aboveground_biomass",
            "vegetation_quality",
            "soil_moisture",
        )
        values: Dict[str, torch.Tensor] = {}
        for name in names:
            if name not in state:
                raise KeyError(name)
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a Tensor")
            if value.numel() == 0:
                raise ValueError(f"{name} must not be empty")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} must be floating point")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values")
            values[name] = value

        reference = values["aboveground_biomass"]
        for name in names[1:]:
            value = values[name]
            if value.shape != reference.shape:
                raise ValueError(f"{name} shape must match aboveground_biomass")
            if value.dtype != reference.dtype:
                raise ValueError(f"{name} dtype must match aboveground_biomass")
            if value.device != reference.device:
                raise ValueError(f"{name} device must match aboveground_biomass")

        if bool(torch.any(reference < 0).item()):
            raise ValueError("aboveground_biomass must be non-negative")
        quality = values["vegetation_quality"]
        if bool(torch.any((quality < 0) | (quality > 1)).item()):
            raise ValueError("vegetation_quality must be within [0, 1]")
        moisture = values["soil_moisture"]
        if bool(torch.any((moisture < 0) | (moisture > 1)).item()):
            raise ValueError("soil_moisture must be within [0, 1]")

        return reference, quality, moisture

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        time: float,
        dt: float,
        calendar_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate grazing effects on vegetation and nutrient cycling.

        References:
            Milchunas and Lauenroth (1993); Hodgson (1990); CF-1.8.
        Mathematical expression:
            x_dot = N_h r_h + G_h u_grazing, with grazing removal and
            excreta return rates active inside the declared calendar window.
        Parameter meanings:
            x is grazable biomass [kg DM m-2], r_h contains removal and
            return fluxes [kg m-2 s-1], dt is step length [s], and C is the
            explicit simulator calendar used to evaluate the event window.
        """
        # References: Milchunas and Lauenroth (1993); Hodgson (1990); CF-1.8.
        # Mathematical expression: F_remove=min(I,B U_max,B t_day/dt)/t_day; DeltaB=-F_remove dt.
        # Parameter meanings: I is daily intake, B biomass, U_max utilization, and dt step length.
        # Units: B and DeltaB [kg DM m-2]; F_remove [kg DM m-2 s-1]; dt and t_day [s].
        # Domain/constraints: dt > 0; required state ports are finite, aligned, and physically bounded.
        # Sign convention: removal/export fluxes are positive, biomass update negative, and returns positive.
        dt_value = float(dt)
        if not torch.isfinite(torch.as_tensor(dt_value)) or dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")
        local_state = dict(state)

        # Required state ports are fail-fast; no ParameterManager state value
        # may silently create vegetation or soil-water storage.
        (
            aboveground_biomass,
            vegetation_quality,
            soil_moisture,
        ) = self._require_grazing_state(
            local_state
        )
        day_input = (
            local_state["day_of_year"]
            if "day_of_year" in local_state
            else kwargs.get("day_of_year")
        )
        calendar_type = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self._configured_calendar_type,
            runtime_calendar=calendar_type,
            owner_name=self.__class__.__name__,
        )
        day_of_year = resolve_day_of_year(
            time,
            day_input,
            calendar_type=calendar_type,
        )
        
        seconds_per_day = self._seconds_per_day_value()

        # Initialize output fluxes
        grazing_flux = torch.zeros_like(aboveground_biomass)
        dung_c_flux = torch.zeros_like(aboveground_biomass)
        dung_n_flux = torch.zeros_like(aboveground_biomass)
        urine_n_flux = torch.zeros_like(aboveground_biomass)
        livestock_carbon_intake_flux = torch.zeros_like(aboveground_biomass)
        livestock_carbon_export_flux = torch.zeros_like(aboveground_biomass)
        livestock_nitrogen_intake_flux = torch.zeros_like(aboveground_biomass)
        livestock_nitrogen_export_flux = torch.zeros_like(aboveground_biomass)
        trampling_damage = torch.zeros_like(aboveground_biomass)
        grazing_pressure = torch.zeros_like(aboveground_biomass)
        grazing_active = torch.zeros_like(aboveground_biomass)
        current_stocking_rate = torch.zeros_like(aboveground_biomass)
        utilization_rate = torch.zeros_like(aboveground_biomass)
        
        # Check for active grazing
        active_event = self._check_grazing_schedule(
            day_of_year,
            calendar_type=calendar_type,
        )
        
        if active_event is not None:
            grazing_active = torch.ones_like(aboveground_biomass)
            # References: Li et al. (DNDC), Dndc_farming.cpp grazing(): the animal
            #   heads of grazing event oz are part of that event; Hodgson (1990).
            # Mathematical expression: D = D_event while the event is active, else no
            #   grazing (D enters I = a W^b D Q (1 - exp(-k B))).
            # Parameter meanings: D_event is the stocking rate of the active grazing
            #   event, the herd whose livestock type and body weight the intake uses.
            # Units: D, D_event [animals ha-1].
            # Domain/constraints: D_event finite and non-negative (checked in
            #   _calculate_grazing_pressure); it is the only stocking source, a state
            #   entry named animal_density is not read.
            # Sign convention: D >= 0 is a count per area.
            stocking_rate = active_event.stocking_rate
            current_stocking_rate = (
                torch.ones_like(aboveground_biomass)
                * torch.as_tensor(
                    stocking_rate,
                    dtype=aboveground_biomass.dtype,
                    device=aboveground_biomass.device,
                )
            )

            # References: Hodgson (1990); Milchunas and Lauenroth (1993).
            # Formula: P_g = D_stock * Q_forage.
            # Parameters: D_stock [animals ha^-1], Q_forage [1], P_g [animals ha^-1].
            # Open route: diagnostic only; matter is routed through the grazing port.
            grazing_pressure = self._calculate_grazing_pressure(
                stocking_rate,
                vegetation_quality,
            )
            
            # Get livestock characteristics
            livestock_chars = self._get_livestock_characteristics(active_event.livestock_type)
            
            # Calculate daily intake
            daily_intake = self._calculate_daily_intake(
                livestock_chars,
                active_event.body_weight,
                stocking_rate,
                aboveground_biomass,
                vegetation_quality
            )
            
            # Apply utilization constraint
            max_utilization_param = self.require_scalar(
                "max_utilization",
                namespace="anthroposphere.management.grazing",
            )
            max_utilization = self._ensure_tensor(max_utilization_param, dtype=aboveground_biomass.dtype).to(aboveground_biomass.device)
            max_consumption = aboveground_biomass * max_utilization
            storage_limited_daily_intake = (
                aboveground_biomass * seconds_per_day / dt_value
            )
            daily_intake = torch.minimum(
                daily_intake,
                torch.minimum(max_consumption, storage_limited_daily_intake),
            )
            
            # Partition fresh intake across explicit C and N boundary ports.
            (
                dung_c,
                dung_n,
                urine_n,
                carbon_intake,
                nitrogen_intake,
            ) = self._calculate_excreta_return(
                daily_intake,
                livestock_chars,
            )
            
            # Calculate trampling damage
            trampling = self._calculate_trampling_damage(
                livestock_chars, stocking_rate, soil_moisture
            )
            
            # Set output fluxes (convert daily rates to per-second)
            grazing_flux = daily_intake / seconds_per_day  # kg DM/m^2/s
            dung_c_flux = dung_c / seconds_per_day         # kg C/m^2/s
            # Dung N is the fecal organic-N boundary routed through the
            # litterfall interface mapper. Urine N remains a separate mineral-N
            # boundary and is never mixed into organic_nitrogen_input.
            dung_n_flux = dung_n / seconds_per_day         # kg N/m^2/s
            urine_n_flux = urine_n / seconds_per_day       # kg N/m^2/s
            livestock_carbon_intake_flux = carbon_intake / seconds_per_day
            livestock_carbon_export_flux = (
                livestock_carbon_intake_flux - dung_c_flux
            )
            livestock_nitrogen_intake_flux = nitrogen_intake / seconds_per_day
            livestock_nitrogen_export_flux = (
                livestock_nitrogen_intake_flux - dung_n_flux - urine_n_flux
            )
            trampling_damage = trampling
            utilization_rate = (grazing_flux * dt_value) / aboveground_biomass.clamp_min(
                torch.finfo(aboveground_biomass.dtype).tiny
            )

        step_intake = grazing_flux * dt_value
        days_grazed_update = grazing_active * (dt_value / seconds_per_day)
        
        # References: Conant et al. (2001); Milchunas and Lauenroth (1993).
        # Formula: y_dot = N_h r_h maps intake to removal, returns, and diagnostics.
        # Parameters: r_h entries are ParameterManager-backed intake and excreta closures.
        reaction_outputs = self._assemble_grazing_reaction_outputs(
            grazing_flux=grazing_flux,
            dung_c_flux=dung_c_flux,
            dung_n_flux=dung_n_flux,
            urine_n_flux=urine_n_flux,
            livestock_carbon_intake_flux=livestock_carbon_intake_flux,
            livestock_carbon_export_flux=livestock_carbon_export_flux,
            livestock_nitrogen_intake_flux=livestock_nitrogen_intake_flux,
            livestock_nitrogen_export_flux=livestock_nitrogen_export_flux,
            trampling_damage=trampling_damage,
        )

        # Prepare output dictionary
        outputs = {
            "aboveground_biomass": UpdateItem(
                value=-step_intake,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="kg m-2",
                source_functor=self.name,
            ),
            "grazing_flux": UpdateItem(
                value=reaction_outputs["grazing_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "dung_c_flux": UpdateItem(
                value=reaction_outputs["dung_c_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg C m-2 s-1",
                source_functor=self.name,
            ),
            "dung_n_flux": UpdateItem(
                value=reaction_outputs["dung_n_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg N m-2 s-1",
                source_functor=self.name,
            ),
            "urine_n_flux": UpdateItem(
                value=reaction_outputs["urine_n_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg N m-2 s-1",
                source_functor=self.name,
            ),
            "livestock_carbon_intake_flux": UpdateItem(
                value=reaction_outputs["livestock_carbon_intake_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg C m-2 s-1",
                source_functor=self.name,
            ),
            "livestock_nitrogen_intake_flux": UpdateItem(
                value=reaction_outputs["livestock_nitrogen_intake_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg N m-2 s-1",
                source_functor=self.name,
            ),
            "livestock_carbon_export_flux": UpdateItem(
                value=reaction_outputs["livestock_carbon_export_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg C m-2 s-1",
                source_functor=self.name,
            ),
            "livestock_nitrogen_export_flux": UpdateItem(
                value=reaction_outputs["livestock_nitrogen_export_flux"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg N m-2 s-1",
                source_functor=self.name,
            ),
            "trampling_damage": UpdateItem(
                value=reaction_outputs["trampling_damage"],
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="1",
                source_functor=self.name,
            ),
            "grazing_active": UpdateItem(
                value=grazing_active,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="1",
                source_functor=self.name,
            ),
            "current_stocking_rate": UpdateItem(
                value=current_stocking_rate,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="animals ha-1",
                source_functor=self.name,
            ),
            "utilization_rate": UpdateItem(
                value=utilization_rate,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="1",
                source_functor=self.name,
            ),
            "cumulative_intake": UpdateItem(
                value=step_intake,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="kg m-2",
                source_functor=self.name,
            ),
            "days_grazed": UpdateItem(
                value=days_grazed_update,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="day",
                source_functor=self.name,
            ),
            "diagnostics.grazing_pressure": grazing_pressure,
        }

        return self._attach_canonical_runtime_diagnostic(outputs)

    def check_conservation_laws(
        self,
        input_state: Mapping[str, Any],
        output_updates: Mapping[str, Any],
    ) -> Dict[str, float]:
        """Return the normalized dry-matter, carbon, and nitrogen residual.

        References:
            DNDC grazing and manure modules; Conant et al. (2001).
        Mathematical expression:
            R_DM = DeltaB + I_step
            R_C = C_in - C_dung - C_export
            R_N = N_in - N_dung - N_urine - N_export
            R = max(abs(R_i) / max(abs(input_i), eps)).
        Parameter meanings:
            DeltaB and I_step are integrated dry matter [kg m-2]. Carbon and
            nitrogen terms are boundary fluxes [kg element m-2 s-1].
        """
        # References: DNDC grazing/manure modules; Conant et al. (2001).
        # Mathematical expression: error=max(|R_DM|/S_DM,|R_C|/S_C,|R_N|/S_N).
        # Parameter meanings: R terms are ledger residuals and S terms are matching intake scales.
        # Units: residuals retain their ledger units; each ratio and returned boundary error are [1].
        # Domain/constraints: required UpdateItems have exact modes, units, shape, dtype, device, and finiteness.
        # Sign convention: raw signs follow input-minus-output ledgers; returned error is non-negative.

        if "aboveground_biomass" not in input_state:
            raise KeyError("aboveground_biomass")
        state_biomass = input_state["aboveground_biomass"]
        if not isinstance(state_biomass, torch.Tensor):
            raise TypeError("aboveground_biomass input must be a Tensor")
        if state_biomass.numel() == 0:
            raise ValueError("aboveground_biomass input must not be empty")
        if not torch.is_floating_point(state_biomass):
            raise TypeError("aboveground_biomass input must be floating point")
        if not bool(torch.isfinite(state_biomass).all().item()):
            raise ValueError("aboveground_biomass input must be finite")

        specifications = {
            "aboveground_biomass": (
                UpdateMode.ACCUMULATE,
                True,
                "kg m-2",
            ),
            "cumulative_intake": (
                UpdateMode.ACCUMULATE,
                True,
                "kg m-2",
            ),
            "livestock_carbon_intake_flux": (
                UpdateMode.REPLACE,
                False,
                "kg C m-2 s-1",
            ),
            "dung_c_flux": (
                UpdateMode.REPLACE,
                False,
                "kg C m-2 s-1",
            ),
            "livestock_carbon_export_flux": (
                UpdateMode.REPLACE,
                False,
                "kg C m-2 s-1",
            ),
            "livestock_nitrogen_intake_flux": (
                UpdateMode.REPLACE,
                False,
                "kg N m-2 s-1",
            ),
            "dung_n_flux": (
                UpdateMode.REPLACE,
                False,
                "kg N m-2 s-1",
            ),
            "urine_n_flux": (
                UpdateMode.REPLACE,
                False,
                "kg N m-2 s-1",
            ),
            "livestock_nitrogen_export_flux": (
                UpdateMode.REPLACE,
                False,
                "kg N m-2 s-1",
            ),
        }
        values: Dict[str, torch.Tensor] = {}
        for name, (mode, per_timestep, unit) in specifications.items():
            if name not in output_updates:
                raise KeyError(name)
            item = output_updates[name]
            if not isinstance(item, UpdateItem):
                raise TypeError(f"{name} must be an UpdateItem")
            if (
                item.mode is not mode
                or item.per_timestep is not per_timestep
                or item.unit != unit
            ):
                raise ValueError(f"{name} has incompatible update semantics")
            value = item.value
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} value must be a Tensor")
            if value.numel() == 0:
                raise ValueError(f"{name} value must not be empty")
            if not torch.is_floating_point(value):
                raise TypeError(f"{name} value must be floating point")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} value must be finite")
            if value.shape != state_biomass.shape:
                raise ValueError(f"{name} shape must match aboveground_biomass")
            if value.dtype != state_biomass.dtype:
                raise ValueError(f"{name} dtype must match aboveground_biomass")
            if value.device != state_biomass.device:
                raise ValueError(f"{name} device must match aboveground_biomass")
            values[name] = value

        dry_matter_residual = (
            values["aboveground_biomass"] + values["cumulative_intake"]
        )
        carbon_residual = (
            values["livestock_carbon_intake_flux"]
            - values["dung_c_flux"]
            - values["livestock_carbon_export_flux"]
        )
        nitrogen_residual = (
            values["livestock_nitrogen_intake_flux"]
            - values["dung_n_flux"]
            - values["urine_n_flux"]
            - values["livestock_nitrogen_export_flux"]
        )
        epsilon = torch.finfo(state_biomass.dtype).eps
        normalized = (
            torch.abs(dry_matter_residual)
            / values["cumulative_intake"].abs().clamp_min(epsilon),
            torch.abs(carbon_residual)
            / values["livestock_carbon_intake_flux"].abs().clamp_min(epsilon),
            torch.abs(nitrogen_residual)
            / values["livestock_nitrogen_intake_flux"].abs().clamp_min(epsilon),
        )
        magnitude = torch.stack(tuple(item.amax() for item in normalized)).amax()
        return {"grazing_matter_boundary": float(magnitude.detach().item())}

# Export all public classes and functions
__all__ = [
    'GrazingManagement',
    'LivestockType',
    'GrazingSystem',
    'GrazingIntensity',
    'GrazingEvent',
]
