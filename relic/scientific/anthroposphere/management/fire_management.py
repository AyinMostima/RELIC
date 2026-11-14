"""
Fire Management Module

This module implements fire management practices including prescribed burning,
fire suppression, and wildfire effects on vegetation and soil carbon.

References:
    - Bond, W.J. and Keeley, J.E. (2005). Fire as a global herbivore.
      Trends in Ecology and Evolution 20, 387-394.
    - Certini, G. (2005). Effects of fire on properties of forest soils.
      Oecologia 143, 1-10.
    - Seiler, W. and Crutzen, P.J. (1980). Biomass burning carbon fluxes.
      Climatic Change 2, 207-247.
    - Andreae, M.O. and Merlet, P. (2001). Trace gas emissions from biomass burning.
      Global Biogeochemical Cycles 15, 955-966.
    - van der Werf, G.R., et al. (2017). Global fire emissions 1997-2016.
      Earth System Science Data 9, 697-720.

Mathematical Expression:
    C_combusted = B_fuel * CC * CE * (1 - M_factor)
    Fuel_burned = C_combusted / f_C
    E_species = Fuel_burned * EF_species

Parameter meanings:
    - B_fuel: fuel load [kg DM m-2]
    - CC: combustion completeness [1]
    - CE: combustion efficiency [1]
    - M_factor: moisture reduction factor [1]
    - f_C: biomass carbon fraction [kg C kg-1 DM]
    - EF_species: emission factor [kg species kg-1 DM]
"""
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS, advance_reaction_network
from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    FunctorSubstreamMixin,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.core.functor.functor_bridge_utils import NamespaceDefaultMixin
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant

from .event_scheduler import resolve_day_of_year

logger = logging.getLogger(__name__)

class FireType(Enum):
    """Types of fire events"""
    PRESCRIBED = 'prescribed'
    WILDFIRE = 'wildfire'
    AGRICULTURAL = 'agricultural'
    SLASH = 'slash'
    SUPPRESSED = 'suppressed'

class FireIntensity(Enum):
    """Fire intensity levels"""
    LOW = 'low'
    MODERATE = 'moderate'
    HIGH = 'high'
    EXTREME = 'extreme'

class VegetationType(Enum):
    """Vegetation types for fire behavior"""
    GRASSLAND = 'grassland'
    SHRUBLAND = 'shrubland'
    FOREST_BOREAL = 'forest_boreal'
    FOREST_TEMPERATE = 'forest_temperate'
    FOREST_TROPICAL = 'forest_tropical'
    AGRICULTURAL = 'agricultural'
    PEAT = 'peat'

@dataclass
class FireEvent:
    """
    Data class for fire management events
    
    Attributes:
        fire_type: Type of fire event
        intensity: Fire intensity level
        vegetation_type: Vegetation type burned
        date: Day of year for fire event
        burned_fraction: Fraction of area burned [1]
        fuel_moisture: Fuel moisture content [1]
        wind_speed: Wind speed during fire [m/s]
        temperature: Air temperature during fire [K]
        suppression_effectiveness: Fire suppression effectiveness [1]
    """
    fire_type: FireType
    intensity: FireIntensity
    vegetation_type: VegetationType
    date: int
    burned_fraction: float
    fuel_moisture: float
    wind_speed: float
    temperature: float
    suppression_effectiveness: float

class FireManagement(FunctorSubstreamMixin, NamespaceDefaultMixin, BiologicalProcessFunctor):
    """
    Fire Management Functor
    
    Implements fire management practices including prescribed burning,
    wildfire effects, and emissions from biomass burning.
    
    Parameters:
        fire_schedule: List of fire management events
        combustion_completeness: Fraction of fuel consumed [1]
        combustion_efficiency: Combustion efficiency [1]
        emission_factors: Emission factors for different species
        fire_return_interval: Average fire return interval [years]
        suppression_capacity: Fire suppression capacity [1]
    """

    parameter_lookup_namespace = "anthroposphere.management.fire"
    parameter_lookup_source_order = ("manager",)
    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    # The substream extra state (seed, draw_count, generator_state, wiener_pending)
    # is extended by the fire schedule state; version 3 replaced the private
    # generator state of version 2.
    EXTRA_STATE_FORMAT = "relic.fire_management.extra_state"
    EXTRA_STATE_VERSION = 3
    SUBSTREAM_LABEL = "FireManagement"
    FIRE_EXTRA_STATE_KEYS = (
        "fire_events",
        "executed_schedule_occurrences",
    )

    @staticmethod
    def _scalar_from_value(value: Any) -> float:
        """Convert numeric values to Python float without tensor.item/cpu APIs."""
        if isinstance(value, torch.Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    def _require_scalar(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        source_order: Optional[Tuple[str, ...]] = None,
    ) -> float:
        value = self.get_parameter_value(name, namespace=namespace, source_order=source_order, required=True)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar for '{name}'")
            return self._scalar_from_value(value)
        return self._scalar_from_value(value)

    def __init__(self, name: str='fire_management', config: Optional[Dict]=None):
        """
        Initialize fire management functor
        
        Args:
            name: Functor name
            config: Configuration dictionary. ``seed`` (required) seeds the
                Functor-owned substream of the wildfire ignition draw: a
                non-negative integer below 2**63 (validate_substream_seed).
        """
        config_dict: Dict[str, Any] = dict(config) if config else {}
        if "random_seed" in config_dict:
            raise ValueError(
                f"{name}: config key 'random_seed' was removed; use 'seed' (the "
                "Functor substream seed, a non-negative integer below 2**63)."
            )
        seed = self.validate_substream_seed(config_dict.pop("seed", None))
        metadata = FunctorMetadata(
            name=name,
            version='1.0.0',
            description='Fire management including prescribed burning and wildfire effects',
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                'Canonical IPHS: x_dot = N_h r_h + G_h u_fire, r_h = r_fire; '
                'C_combusted = B_fuel * CC * CE * (1 - M_factor); '
                'Fuel_burned = C_combusted / f_C; '
                'E_species = Fuel_burned * EF_species / Delta_t_event; '
                'Delta_t_event = dt of the executing step, so E_species * dt = Fuel_burned * EF_species; '
                'f_burn,total = 1 - product_i(1 - f_burn,i); '
                'F_C,in = F_C,out = C_combusted / Delta_t_event; '
                'wildfire ignition: u < p = s (1 - M) (w / w_ref), u ~ U(0, 1) from the '
                'Functor substream (config seed in [0, 2**63))'
            ),
            references=[
                'Bond & Keeley (2005) DOI:10.1016/j.tree.2005.04.025',
                'Certini (2005) DOI:10.1007/s00442-004-1788-8',
                'Seiler & Crutzen (1980) DOI:10.1007/BF00137988',
                'Andreae & Merlet (2001) DOI:10.1029/2000GB001382',
                'van der Werf et al. (2017) DOI:10.5194/essd-9-697-2017',
            ],
            parameters={
                'anthroposphere.management.temperature': (
                    'Reference air temperature of the moisture-factor temperature '
                    'effect and air temperature of the default fire schedule events [K]'
                ),
                'anthroposphere.management.clamp_min': (
                    'Lower bound of the management response factors [dimensionless]'
                ),
                'anthroposphere.management.clamp_max': (
                    'Upper bound of the management response factors [dimensionless]'
                ),
                'anthroposphere.management.fire.base_efficiency': (
                    'Baseline efficiency scalar of fire management [dimensionless]'
                ),
                'anthroposphere.management.fire.combustion_completeness': (
                    'Fraction of the fuel consumed by a fire [dimensionless]'
                ),
                'anthroposphere.management.fire.combustion_efficiency': (
                    'Combustion efficiency of the fire [dimensionless]'
                ),
                'anthroposphere.management.fire.emission_factor_co2': (
                    'CO2 emission factor per burned dry matter [g kg-1 DM]'
                ),
                'anthroposphere.management.fire.emission_factor_co': (
                    'CO emission factor per burned dry matter [g kg-1 DM]'
                ),
                'anthroposphere.management.fire.emission_factor_ch4': (
                    'CH4 emission factor per burned dry matter [g kg-1 DM]'
                ),
                'anthroposphere.management.fire.ash_fraction': (
                    'Fraction of combusted biomass deposited as ash [dimensionless]'
                ),
                'anthroposphere.management.fire.biomass_carbon_fraction': (
                    'Carbon fraction of dry biomass [dimensionless]'
                ),
                'anthroposphere.management.fire.fire_frequency': (
                    'Mean annual fire frequency of the scheduler [year-1]'
                ),
                'anthroposphere.management.fire.fire_probability_scale': (
                    'Scale factor for wildfire ignition probability [dimensionless]'
                ),
                'anthroposphere.management.fire.moisture_reduction_coefficient': (
                    'Moisture reduction coefficient for fire spread calculations [dimensionless]'
                ),
                'anthroposphere.management.fire.moisture_temp_base_weight': (
                    'Base weight for moisture temperature coupling [dimensionless]'
                ),
                'anthroposphere.management.fire.moisture_temp_scale_weight': (
                    'Temperature scaling weight for moisture coupling [dimensionless]'
                ),
                'anthroposphere.management.fire.temperature_scale_range': (
                    'Temperature normalization range for moisture scaling [K]'
                ),
                'anthroposphere.management.fire.wind_speed_threshold': (
                    'Wind speed threshold affecting fire spread [m s-1]'
                ),
                'anthroposphere.management.fire.schedule_window_days': (
                    'Window size in days for matching scheduled fire events [day]'
                ),
                'anthroposphere.management.fire.default_schedule.*': (
                    'Default prescribed-fire and wildfire events (date, burned fraction, fuel '
                    'moisture, wind speed, suppression effectiveness) [units per key]'
                ),
                'anthroposphere.management.fire.intensity_factor.*': (
                    'Fire intensity multiplier per intensity class (low, moderate, high, extreme) '
                    '[dimensionless]'
                ),
                'anthroposphere.management.fire.vegetation_properties.*': (
                    'Fuel load factor, combustion completeness, aboveground and litter fractions '
                    'and CO2, CO, CH4, N2O emission factors per vegetation type [units per key]'
                ),
            },
            input_variables=[
                'aboveground_biomass',
                'litter_mass',
                'fuel_moisture',
                'day_of_year',
                'days_since_fire',
                'temperature',
                'fuel_load',
                'burn_fraction',
            ],
            output_variables=[
                'burned_fraction',
                'co2_emission',
                'co_emission',
                'fire_ch4_emission',
                'n2o_emission',
                'ash_deposition',
                'days_since_fire',
                'fire_fuel_carbon_input',
                'carbon_loss',
                'combustion_flux',
                'diagnostics.fire.events_today',
                'diagnostics.fire.event_support_seconds',
                'diagnostics.fire.carbon_availability_residual',
                'diagnostics.fire_management.canonical_iphs_runtime',
            ],
            conservation_laws=['mass'],
            spatial_dimensions=2,
            temporal_scale=TemporalScale.DAY,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            coupling_interfaces=['biosphere', 'atmosphere', 'pedosphere'],
        )
        default_config = {'dependencies': []}
        default_config.update(config_dict)
        super().__init__(name=name, config=default_config, metadata=metadata)
        # One Functor-owned stream (FunctorSubstreamMixin) for the wildfire ignition
        # draw; the simulator snapshots it with the module extra state.
        self._initialize_substream(seed)
        self.optional_input_variables = {
            "day_of_year",
            "fuel_load",
            "burn_fraction",
        }
        configured_calendar = default_config.get("calendar_type")
        self._configured_calendar_type = (
            None
            if configured_calendar is None
            else str(configured_calendar)
        )
        if self._configured_calendar_type is not None:
            CalendarHandler.days_per_simulator_year(
                self._configured_calendar_type
            )
        self.fire_events: List[FireEvent] = []
        self._executed_schedule_occurrences: set[str] = set()
        self._fire_param_defaults: List[str] = []
        self._initialize_default_schedule()
        # The Functor holds no tensor state: days_since_fire is the StateManager state
        # it owns (read from the state, returned as a REPLACE update); the executed
        # schedule occurrences and the substream position are module extra state.
        self.runtime_path = "canonical_iphs_fire_combustion_port"

    def _validate_fire_event_date(self, value: Any) -> int:
        """Validate a scheduled date without duplicating calendar arithmetic."""

        if self._configured_calendar_type is not None:
            try:
                return CalendarHandler.validate_simulator_day(
                    value,
                    self._configured_calendar_type,
                )
            except ValueError as exc:
                raise ValueError(
                    "Fire event date must be a finite in-domain integer for "
                    f"calendar_type={self._configured_calendar_type}."
                ) from exc
        message = (
            "Fire event date must be a positive finite integer numeric scalar "
            "until a simulator calendar is bound."
        )
        if isinstance(value, bool):
            raise ValueError(message)
        try:
            scalar = (
                value.detach()
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(message) from exc
        if scalar.ndim != 0 or scalar.dtype == torch.bool or scalar.is_complex():
            raise ValueError(message)
        scalar = scalar.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(scalar).item()):
            raise ValueError(message)
        if not bool(torch.eq(scalar, torch.trunc(scalar)).item()):
            raise ValueError(message)
        if not bool((scalar >= 1.0).item()):
            raise ValueError(
                message
            )
        return int(scalar.item())

    def _event_to_extra_state(self, event: FireEvent) -> Dict[str, Any]:
        """Serialize one scheduled event using weights-only-safe values."""

        return {
            "fire_type": event.fire_type.value,
            "intensity": event.intensity.value,
            "vegetation_type": event.vegetation_type.value,
            "date": self._validate_fire_event_date(event.date),
            "burned_fraction": float(event.burned_fraction),
            "fuel_moisture": float(event.fuel_moisture),
            "wind_speed": float(event.wind_speed),
            "temperature": float(event.temperature),
            "suppression_effectiveness": float(
                event.suppression_effectiveness
            ),
        }

    @staticmethod
    def _finite_extra_state_float(name: str, value: Any) -> float:
        """Validate and normalize one finite numeric extra-state value."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Fire extra-state field '{name}' must be numeric.")
        result = float(value)
        result_tensor = torch.as_tensor(result, dtype=torch.float64)
        if not torch.all(torch.isfinite(result_tensor)):
            raise ValueError(f"Fire extra-state field '{name}' must be finite.")
        return result

    @staticmethod
    def _validated_occurrence_keys(value: Any) -> set[str]:
        """Validate persisted schedule occurrence identifiers atomically."""

        if not isinstance(value, list):
            raise TypeError(
                "Fire executed schedule occurrences must be stored as a list."
            )
        restored: set[str] = set()
        for entry in value:
            if not isinstance(entry, str):
                raise TypeError(
                    "Each fire schedule occurrence identifier must be text."
                )
            parts = entry.split(":")
            if len(parts) != 2:
                raise ValueError(
                    "Fire schedule occurrence identifiers must use index:year."
                )
            try:
                event_index = int(parts[0])
                int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    "Fire schedule occurrence identifiers must contain integers."
                ) from exc
            if event_index < 0 or entry in restored:
                raise ValueError(
                    "Fire schedule occurrence identifiers must be unique and non-negative."
                )
            restored.add(entry)
        return restored

    def _event_from_extra_state(self, payload: Any) -> FireEvent:
        """Validate and restore one scheduled event."""

        if not isinstance(payload, Mapping):
            raise TypeError("Each fire event extra-state entry must be a mapping.")
        expected_keys = {
            "fire_type",
            "intensity",
            "vegetation_type",
            "date",
            "burned_fraction",
            "fuel_moisture",
            "wind_speed",
            "temperature",
            "suppression_effectiveness",
        }
        if set(payload) != expected_keys:
            raise ValueError(
                "Fire event extra-state keys do not match the required schema."
            )
        date = self._validate_fire_event_date(payload["date"])
        burned_fraction = self._finite_extra_state_float(
            "burned_fraction",
            payload["burned_fraction"],
        )
        fuel_moisture = self._finite_extra_state_float(
            "fuel_moisture",
            payload["fuel_moisture"],
        )
        wind_speed = self._finite_extra_state_float(
            "wind_speed",
            payload["wind_speed"],
        )
        temperature = self._finite_extra_state_float(
            "temperature",
            payload["temperature"],
        )
        suppression_effectiveness = self._finite_extra_state_float(
            "suppression_effectiveness",
            payload["suppression_effectiveness"],
        )
        if not 0.0 <= burned_fraction <= 1.0:
            raise ValueError("Fire event burned_fraction must be within [0, 1].")
        if not 0.0 <= fuel_moisture <= 1.0:
            raise ValueError("Fire event fuel_moisture must be within [0, 1].")
        if wind_speed < 0.0:
            raise ValueError("Fire event wind_speed must be non-negative.")
        if temperature <= 0.0:
            raise ValueError("Fire event temperature must be positive.")
        if not 0.0 <= suppression_effectiveness <= 1.0:
            raise ValueError(
                "Fire event suppression_effectiveness must be within [0, 1]."
            )
        try:
            fire_type = FireType(payload["fire_type"])
            intensity = FireIntensity(payload["intensity"])
            vegetation_type = VegetationType(payload["vegetation_type"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Fire event enum value is invalid.") from exc
        return FireEvent(
            fire_type=fire_type,
            intensity=intensity,
            vegetation_type=vegetation_type,
            date=date,
            burned_fraction=burned_fraction,
            fuel_moisture=fuel_moisture,
            wind_speed=wind_speed,
            temperature=temperature,
            suppression_effectiveness=suppression_effectiveness,
        )

    def get_extra_state(self) -> Dict[str, Any]:
        """Return the substream position and the fire schedule state for restart."""

        extra_state = super().get_extra_state()
        extra_state.update({
            "fire_events": [
                self._event_to_extra_state(event)
                for event in self.fire_events
            ],
            "executed_schedule_occurrences": sorted(
                self._executed_schedule_occurrences
            ),
        })
        return extra_state

    def set_extra_state(self, state: Any) -> None:
        """Restore module-owned non-tensor state after strict validation."""

        if not isinstance(state, Mapping):
            raise TypeError("Fire extra state must be a mapping.")
        missing = [key for key in self.FIRE_EXTRA_STATE_KEYS if key not in state]
        if missing:
            raise ValueError(
                "Fire extra-state keys do not match the required schema: "
                f"missing {missing}."
            )
        substream_state = {
            key: value
            for key, value in state.items()
            if key not in self.FIRE_EXTRA_STATE_KEYS
        }
        event_payloads = state["fire_events"]
        if not isinstance(event_payloads, list):
            raise TypeError("Fire event extra state must be a list.")
        restored_events = [
            self._event_from_extra_state(payload)
            for payload in event_payloads
        ]
        restored_occurrences = self._validated_occurrence_keys(
            state["executed_schedule_occurrences"]
        )

        # The substream schema (format, version, seed, draw_count, generator
        # state, pending increments) is validated by the substream owner.
        super().set_extra_state(substream_state)
        self.fire_events = restored_events
        self._executed_schedule_occurrences = restored_occurrences

    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Resolve structural audit size for fire reaction channels."""

        value = 5 if n_dofs is None else int(n_dofs)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare cochain spaces for the fire combustion reaction-port map.

        References:
          - Seiler and Crutzen (1980), biomass burning carbon fluxes.
          - Andreae and Merlet (2001), biomass burning emission factors.
          - Ramirez and Le Gorrec (2022), irreversible port-Hamiltonian systems.
        Mathematical expression:
          x_dot = N_h r_h + G_h u_fire, with r_h = r_fire and
          C_combusted = B_fuel * CC * CE * (1 - M_factor).
        Parameter meanings:
          x is combustible fuel carbon [kg C m^-2], r_h contains
          combustion and emission fluxes [kg m^-2 s^-1], and u_fire is the
          management or wildfire ignition port input.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="anthroposphere_fire",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="kg m^-2",
                extensivity="density",
                physical_domain="biosphere",
                semantic_name="fire_fuel_pool",
                orientation_convention="positive_fuel_storage",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_fire",
                degree=0,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m^-2 s^-1",
                extensivity="density",
                physical_domain="atmosphere",
                semantic_name="fire_combustion_flux",
                orientation_convention="positive_to_atmosphere",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_fire",
                degree=0,
                primality="dual",
                n_dofs=1,
                units="1",
                extensivity="intensive",
                physical_domain="anthroposphere",
                semantic_name="fire_management_port",
                orientation_convention="positive_ignition_or_suppression",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager namespaces used by runtime and structure."""

        return (
            "anthroposphere.management.fire.*",
            "anthroposphere.management.fire.vegetation_properties.*",
            "anthroposphere.management.*",
            "validation.conservation.conservation_tolerance",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive fire combustion mobility Hodge contract.

        References:
          - Andreae and Merlet (2001), biomass burning emission factors.
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical expression:
          H_fire = diag(w_i / sum_j w_j), w_i >= 0.
        Parameter meanings:
          w_i are ParameterManager-backed combustion completeness,
          combustion efficiency, emission factors, and ash partition weights.
        """

        n_dofs = self._structural_n_dofs(n_dofs)
        keys = (
            "combustion_completeness",
            "combustion_efficiency",
            "emission_factor_co2",
            "emission_factor_co",
            "emission_factor_ch4",
        )
        device = getattr(self, "device", torch.device("cpu"))
        values = [
            torch.abs(
                torch.as_tensor(
                    self._safe_fire_param(
                        key,
                        namespace="anthroposphere.management.fire",
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            )
            for key in keys
        ]
        if len(values) < n_dofs:
            fill_value = values[-1] if values else torch.ones((), device=device)
            values.extend(fill_value.clone() for _ in range(n_dofs - len(values)))
        weights = torch.stack(values[:n_dofs])
        total = torch.sum(weights)
        if total <= 0:
            raise ValueError("Fire combustion weights require positive support.")
        matrix = torch.diag(weights / total)
        domain = CochainSpaceSpec(
            complex_id="anthroposphere_fire",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="kg m^-2",
            extensivity="density",
            physical_domain="biosphere",
            semantic_name="fire_fuel_pool",
            orientation_convention="positive_fuel_storage",
            boundary_role="interior",
        )
        codomain = CochainSpaceSpec(
            complex_id="anthroposphere_fire",
            degree=0,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m^-2 s^-1",
            extensivity="density",
            physical_domain="atmosphere",
            semantic_name="fire_combustion_flux",
            orientation_convention="positive_to_atmosphere",
            boundary_role="interior",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_combustion_mobility_hodge",
            domain=domain,
            codomain=codomain,
            matrix=matrix,
            parameter_keys=tuple(f"anthroposphere.management.fire.{key}" for key in keys),
            metric_keys=("validation.conservation.conservation_tolerance",),
            material_law="fire_combustion_partition_mobility",
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
        """Return no independent energy storage for this open combustion port."""

        _ = state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the non-negative fire combustion entropy audit term."""

        _ = state
        return {
            "fire_combustion_entropy": "sigma = z.T H_fire z >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the declared fire reaction-network metadata."""

        _ = state, complex_bundle
        return {
            "state_vars": (
                "aboveground_biomass",
                "litter_mass",
                "fuel_moisture",
                "burned_fraction",
            ),
            "reaction_map": "x_dot = N_h r_h + G_h u_fire, r_h = r_fire",
            "mass_conserving_partition": True,
            "ports": self.ports(),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive fire combustion mobility operator."""

        _ = state, complex_bundle
        hodge = self.build_hodge_contract(strict=True)
        matrix = hodge.matrix
        return {
            "state_vars": tuple(f"fire_fuel_pool[{idx}]" for idx in range(int(matrix.shape[0]))),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the fire ignition and suppression management port."""

        effort_space = CochainSpaceSpec(
            complex_id="anthroposphere_fire",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="1",
            extensivity="intensive",
            physical_domain="anthroposphere",
            semantic_name="fire_management_effort",
            orientation_convention="positive_fire_control",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="anthroposphere_fire",
            degree=0,
            primality="dual",
            n_dofs=1,
            units="kg m^-2 s^-1",
            extensivity="density",
            physical_domain="atmosphere",
            semantic_name="fire_combustion_flux",
            orientation_convention="positive_to_atmosphere",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="fire_combustion_management_port",
                effort_var="fire_management_effort",
                flow_var="combustion_flux",
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
        """Run the canonical IPHS fire reaction-port runtime."""

        return self.forward(dict(state), time=time, dt=dt)

    def _seconds_per_day_value(self) -> float:
        """Resolve the physical day length through the physical constants SSOT."""

        return float(
            get_physical_constant(
                "relic.physical.constants.time.seconds_per_day"
            )
        )

    def _attach_canonical_runtime_diagnostic(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Attach the Phase 8 canonical runtime diagnostic flag."""

        device = getattr(self, "device", torch.device("cpu"))
        dtype = torch.float32
        for value in outputs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                dtype = value.dtype if torch.is_floating_point(value) else torch.float32
                break
        outputs["diagnostics.fire_management.canonical_iphs_runtime"] = torch.ones(
            (), device=device, dtype=dtype
        )
        return outputs

    def _safe_fire_param(self, key: str, namespace: str) -> Any:
        """Retrieve fire parameter from ParameterManager."""
        return self.get_parameter_value(key, namespace=namespace, source_order=('manager',), required=True)

    def _initialize_default_schedule(self) -> None:
        """Initialize default fire management schedule using parameters"""
        prescribed_date = self._safe_fire_param(
            'default_schedule.prescribed.date',
            namespace='anthroposphere.management.fire',
        )
        prescribed_fraction = self._safe_fire_param(
            'default_schedule.prescribed.burned_fraction',
            namespace='anthroposphere.management.fire',
        )
        prescribed_moisture = self._safe_fire_param(
            'default_schedule.prescribed.fuel_moisture',
            namespace='anthroposphere.management.fire',
        )
        prescribed_wind = self._safe_fire_param(
            'default_schedule.prescribed.wind_speed',
            namespace='anthroposphere.management.fire',
        )
        prescribed_suppression = self._safe_fire_param(
            'default_schedule.prescribed.suppression_effectiveness',
            namespace='anthroposphere.management.fire',
        )
        default_temperature = self._require_scalar(
            'temperature',
            namespace='anthroposphere.management',
            source_order=('manager',),
        )
        self.fire_events.append(
            FireEvent(
                fire_type=FireType.PRESCRIBED,
                intensity=FireIntensity.LOW,
                vegetation_type=VegetationType.GRASSLAND,
                date=self._validate_fire_event_date(prescribed_date),
                burned_fraction=float(prescribed_fraction),
                fuel_moisture=float(prescribed_moisture),
                wind_speed=float(prescribed_wind),
                temperature=float(default_temperature),
                suppression_effectiveness=float(prescribed_suppression),
            )
        )
        wildfire_date = self._safe_fire_param(
            'default_schedule.wildfire.date',
            namespace='anthroposphere.management.fire',
        )
        wildfire_fraction = self._safe_fire_param(
            'default_schedule.wildfire.burned_fraction',
            namespace='anthroposphere.management.fire',
        )
        wildfire_moisture = self._safe_fire_param(
            'default_schedule.wildfire.fuel_moisture',
            namespace='anthroposphere.management.fire',
        )
        wildfire_wind = self._safe_fire_param(
            'default_schedule.wildfire.wind_speed',
            namespace='anthroposphere.management.fire',
        )
        wildfire_suppression = self._safe_fire_param(
            'default_schedule.wildfire.suppression_effectiveness',
            namespace='anthroposphere.management.fire',
        )
        self.fire_events.append(
            FireEvent(
                fire_type=FireType.WILDFIRE,
                intensity=FireIntensity.MODERATE,
                vegetation_type=VegetationType.FOREST_TEMPERATE,
                date=self._validate_fire_event_date(wildfire_date),
                burned_fraction=float(wildfire_fraction),
                fuel_moisture=float(wildfire_moisture),
                wind_speed=float(wildfire_wind),
                temperature=float(default_temperature),
                suppression_effectiveness=float(wildfire_suppression),
            )
        )

    def add_fire_event(self, event: FireEvent) -> None:
        """
        Add a fire event to management schedule
        
        Args:
            event: FireEvent defining fire details
        """
        if not isinstance(event, FireEvent):
            raise TypeError("event must be a FireEvent instance.")
        event.date = self._validate_fire_event_date(event.date)
        self.fire_events.append(event)

    def _get_vegetation_fire_properties(
        self,
        vegetation_type: VegetationType,
    ) -> Dict[str, Any]:
        """
        Get fire behavior properties for vegetation types
        
        Args:
            vegetation_type: Type of vegetation
            
        Returns:
            Dictionary with combustion properties and emission factors
        """
        veg_name = vegetation_type.value
        fuel_load_factor = self._safe_fire_param(
            f'fire.vegetation_properties.{veg_name}.fuel_load_factor',
            namespace='anthroposphere.management',
        )
        combustion_completeness = self._safe_fire_param(
            f'fire.vegetation_properties.{veg_name}.combustion_completeness',
            namespace='anthroposphere.management',
        )
        aboveground_fraction = self._safe_fire_param(
            f'fire.vegetation_properties.{veg_name}.aboveground_fraction',
            namespace='anthroposphere.management',
        )
        litter_fraction = self._safe_fire_param(
            f'fire.vegetation_properties.{veg_name}.litter_fraction',
            namespace='anthroposphere.management',
        )
        co2_ef = self._safe_fire_param(
            f'fire.vegetation_properties.{veg_name}.co2_ef',
            namespace='anthroposphere.management',
        )
        co_ef = self._safe_fire_param(
            f'fire.vegetation_properties.{veg_name}.co_ef',
            namespace='anthroposphere.management',
        )
        ch4_ef = self._safe_fire_param(
            f'fire.vegetation_properties.{veg_name}.ch4_ef',
            namespace='anthroposphere.management',
        )
        n2o_ef = self._safe_fire_param(
            f'fire.vegetation_properties.{veg_name}.n2o_ef',
            namespace='anthroposphere.management',
        )
        return {
            'fuel_load_factor': fuel_load_factor,
            'combustion_completeness': combustion_completeness,
            'aboveground_fraction': aboveground_fraction,
            'litter_fraction': litter_fraction,
            'co2_ef': co2_ef,
            'co_ef': co_ef,
            'ch4_ef': ch4_ef,
            'n2o_ef': n2o_ef,
        }

    def _calculate_fuel_load(
        self,
        aboveground_biomass: torch.Tensor,
        litter_mass: torch.Tensor,
        vegetation_type: VegetationType,
    ) -> torch.Tensor:
        """
        Calculate available fuel load for combustion
        
        Args:
            aboveground_biomass: Aboveground biomass [kg DM/m^2]
            litter_mass: Litter mass [kg DM/m^2]
            vegetation_type: Vegetation type
            
        Returns:
            Available fuel load [kg DM/m^2]
        """
        props = self._get_vegetation_fire_properties(vegetation_type)
        fuel_factor = torch.as_tensor(
            props['fuel_load_factor'],
            dtype=aboveground_biomass.dtype,
            device=aboveground_biomass.device,
        )
        aboveground_fraction = torch.as_tensor(
            props['aboveground_fraction'],
            dtype=aboveground_biomass.dtype,
            device=aboveground_biomass.device,
        )
        litter_fraction = torch.as_tensor(
            props['litter_fraction'],
            dtype=aboveground_biomass.dtype,
            device=aboveground_biomass.device,
        )
        available_fuel = aboveground_biomass * aboveground_fraction + litter_mass * litter_fraction
        return available_fuel * fuel_factor

    def _calculate_moisture_factor(
        self,
        fuel_moisture: torch.Tensor,
        temperature: torch.Tensor,
        event_fuel_moisture: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Calculate moisture reduction factor for combustion
        
        Args:
            fuel_moisture: Fuel moisture content [1]
            temperature: Air temperature [K]
            event_fuel_moisture: Event-specific fuel moisture override [1]
            
        Returns:
            Moisture reduction factor [1]
        """
        if event_fuel_moisture is not None:
            effective_moisture = torch.as_tensor(
                event_fuel_moisture,
                device=temperature.device,
                dtype=temperature.dtype,
            )
        else:
            effective_moisture = fuel_moisture
        reference_temperature = self.get_parameter_value(
            'temperature',
            namespace='anthroposphere.management',
            source_order=('manager',),
            required=True,
        )
        clamp_min_value = self.get_parameter_value(
            'clamp_min',
            namespace='anthroposphere.management',
            source_order=('manager',),
            required=True,
        )
        clamp_max_value = self.get_parameter_value(
            'clamp_max',
            namespace='anthroposphere.management',
            source_order=('manager',),
            required=True,
        )
        reference_temperature_tensor = self._ensure_tensor(
            reference_temperature,
            dtype=temperature.dtype,
        ).to(temperature.device)
        clamp_min_tensor = self._ensure_tensor(
            clamp_min_value,
            dtype=temperature.dtype,
        ).to(temperature.device)
        clamp_max_tensor = self._ensure_tensor(
            clamp_max_value,
            dtype=temperature.dtype,
        ).to(temperature.device)
        temperature_scale = self._safe_fire_param(
            'temperature_scale_range',
            namespace='anthroposphere.management.fire',
        )
        temperature_scale_tensor = self._ensure_tensor(
            temperature_scale,
            dtype=temperature.dtype,
        ).to(temperature.device)
        temp_effect = torch.clamp(
            (temperature - reference_temperature_tensor) / temperature_scale_tensor,
            min=clamp_min_tensor,
            max=clamp_max_tensor,
        )
        moisture_coefficient = self.get_parameter_value(
            'moisture_reduction_coefficient',
            namespace='anthroposphere.management.fire',
            source_order=('manager',),
            required=True,
        )
        moisture_coefficient_tensor = self._ensure_tensor(
            moisture_coefficient,
            dtype=temperature.dtype,
        ).to(temperature.device)
        moisture_effect = torch.clamp(
            1.0 - moisture_coefficient_tensor * effective_moisture,
            min=clamp_min_tensor,
            max=clamp_max_tensor,
        )
        moisture_temp_base_weight = self._safe_fire_param(
            'moisture_temp_base_weight',
            namespace='anthroposphere.management.fire',
        )
        moisture_temp_scale_weight = self._safe_fire_param(
            'moisture_temp_scale_weight',
            namespace='anthroposphere.management.fire',
        )
        base_weight_tensor = self._ensure_tensor(
            moisture_temp_base_weight,
            dtype=temperature.dtype,
        ).to(temperature.device)
        scale_weight_tensor = self._ensure_tensor(
            moisture_temp_scale_weight,
            dtype=temperature.dtype,
        ).to(temperature.device)
        combined_factor = moisture_effect * (base_weight_tensor + scale_weight_tensor * temp_effect)
        return combined_factor

    def _calculate_combustion_completeness(
        self,
        fire_intensity: FireIntensity,
        vegetation_props: Dict[str, Any],
        moisture_factor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate combustion completeness based on fire conditions
        
        Args:
            fire_intensity: Fire intensity level
            vegetation_props: Vegetation fire properties
            moisture_factor: Moisture reduction factor
            
        Returns:
            Combustion completeness [1]
        """
        base_cc = torch.as_tensor(
            vegetation_props['combustion_completeness'],
            dtype=moisture_factor.dtype,
            device=moisture_factor.device,
        )
        intensity_factors = {
            FireIntensity.LOW: self._safe_fire_param(
                'intensity_factor.low',
                namespace='anthroposphere.management.fire',
            ),
            FireIntensity.MODERATE: self._safe_fire_param(
                'intensity_factor.moderate',
                namespace='anthroposphere.management.fire',
            ),
            FireIntensity.HIGH: self._safe_fire_param(
                'intensity_factor.high',
                namespace='anthroposphere.management.fire',
            ),
            FireIntensity.EXTREME: self._safe_fire_param(
                'intensity_factor.extreme',
                namespace='anthroposphere.management.fire',
            ),
        }
        default_intensity = intensity_factors.get(FireIntensity.MODERATE)
        intensity_factor = intensity_factors.get(fire_intensity, default_intensity)
        intensity_factor_tensor = self._ensure_tensor(
            intensity_factor,
            dtype=moisture_factor.dtype,
        ).to(moisture_factor.device)
        combustion_completeness = base_cc * intensity_factor_tensor * moisture_factor
        clamp_min_value = self.get_parameter_value(
            'clamp_min',
            namespace='anthroposphere.management',
            source_order=('manager',),
            required=True,
        )
        clamp_max_value = self.get_parameter_value(
            'clamp_max',
            namespace='anthroposphere.management',
            source_order=('manager',),
            required=True,
        )
        clamp_min_tensor = self._ensure_tensor(
            clamp_min_value,
            dtype=moisture_factor.dtype,
        ).to(moisture_factor.device)
        clamp_max_tensor = self._ensure_tensor(
            clamp_max_value,
            dtype=moisture_factor.dtype,
        ).to(moisture_factor.device)
        combustion_completeness = torch.clamp(combustion_completeness, min=clamp_min_tensor, max=clamp_max_tensor)
        return combustion_completeness

    def _calculate_emissions(
        self,
        carbon_combusted: torch.Tensor,
        vegetation_props: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate emissions from biomass burning
        
        Args:
            carbon_combusted: Carbon combusted [kg C/m^2]
            vegetation_props: Vegetation fire properties
            
        Returns:
            Dictionary with emission fluxes [kg/m^2]
        """
        emissions = {}
        biomass_carbon_fraction = self._safe_fire_param(
            'biomass_carbon_fraction',
            namespace='anthroposphere.management.fire',
        )
        carbon_fraction_tensor = self._ensure_tensor(
            biomass_carbon_fraction,
            dtype=carbon_combusted.dtype,
        ).to(carbon_combusted.device)
        fuel_burned_dm = carbon_combusted / carbon_fraction_tensor
        emission_scale = torch.as_tensor(
            1000.0,
            dtype=carbon_combusted.dtype,
            device=carbon_combusted.device,
        )
        co2_ef = torch.as_tensor(
            vegetation_props['co2_ef'],
            dtype=carbon_combusted.dtype,
            device=carbon_combusted.device,
        ) / emission_scale
        emissions['co2_emission'] = fuel_burned_dm * co2_ef
        co_ef = torch.as_tensor(
            vegetation_props['co_ef'],
            dtype=carbon_combusted.dtype,
            device=carbon_combusted.device,
        ) / emission_scale
        emissions['co_emission'] = fuel_burned_dm * co_ef
        ch4_ef = torch.as_tensor(
            vegetation_props['ch4_ef'],
            dtype=carbon_combusted.dtype,
            device=carbon_combusted.device,
        ) / emission_scale
        emissions['fire_ch4_emission'] = fuel_burned_dm * ch4_ef
        n2o_ef = torch.as_tensor(
            vegetation_props['n2o_ef'],
            dtype=carbon_combusted.dtype,
            device=carbon_combusted.device,
        ) / emission_scale
        emissions['n2o_emission'] = fuel_burned_dm * n2o_ef
        return emissions

    @staticmethod
    def _schedule_occurrence_key(
        *,
        event_index: int,
        event_day: int,
        day_of_year: int,
        time: Any,
        days_per_year: int,
        seconds_per_day: float,
    ) -> str:
        """Return the closest annual schedule occurrence identifier.

        References:
          - CF Conventions 1.8, Section 4.4, calendar-aware time axes.
        Mathematical expression:
          y=floor(t/(Y Delta_t_day)); the closest circular occurrence uses
          y-1 or y+1 when |d-event_day| exceeds Y/2.
        Parameter meanings:
          t is simulator time [s], Y is calendar days per year [day], and
          Delta_t_day is the physical day length [s day-1].
        """

        time_tensor = torch.as_tensor(time, dtype=torch.float64)
        if time_tensor.numel() != 1 or not torch.all(torch.isfinite(time_tensor)):
            raise ValueError("Fire schedule time must be one finite scalar.")
        if torch.any(time_tensor < 0.0):
            raise ValueError("Fire schedule time must be non-negative.")
        seconds_per_year = torch.as_tensor(
            seconds_per_day * days_per_year,
            dtype=torch.float64,
        )
        year_tensor = torch.floor(time_tensor.reshape(()) / seconds_per_year)
        occurrence_year = int(FireManagement._scalar_from_value(year_tensor))
        direct_distance = abs(day_of_year - event_day)
        if direct_distance > days_per_year / 2.0:
            if day_of_year < event_day:
                occurrence_year -= 1
            else:
                occurrence_year += 1
        return f"{event_index}:{occurrence_year}"

    def _check_fire_schedule(
        self,
        day_of_year: int,
        calendar_type: str,
        time: Any,
        seconds_per_day: float,
        force_fire: bool = False,
    ) -> List[Tuple[str, FireEvent]]:
        """Check which fire events should occur today.

        References:
            CF Conventions 1.8, Section 4.4; revised foundations, time-window
            and calendar contract.
        Mathematical expression:
            d_C(a,b) = min(|a-b|, Y(C)-|a-b|); active iff d_C <= w.
        Parameter meanings:
            a is current day [day], b is event day [day], Y(C) is calendar
            year length [day], and w is the schedule window [day].
        
        Args:
            day_of_year: Current day of the configured simulator year
            calendar_type: Existing simulator calendar identifier
            force_fire: If True, force wildfire events to trigger regardless of probability
            
        Returns:
            Occurrence identifiers and fire events to execute today
        """
        events_today = []
        schedule_window_days = int(
            self._safe_fire_param(
                'schedule_window_days',
                namespace='anthroposphere.management.fire',
            )
        )
        fire_probability_scale = self._safe_fire_param(
            'fire_probability_scale',
            namespace='anthroposphere.management.fire',
        )
        wind_speed_norm = self._safe_fire_param(
            'wind_speed_threshold',
            namespace='anthroposphere.management.fire',
        )
        day_of_year = CalendarHandler.validate_simulator_day(
            day_of_year,
            calendar_type,
        )
        days_per_year = CalendarHandler.days_per_simulator_year(calendar_type)
        for event_index, event in enumerate(self.fire_events):
            try:
                event_day = CalendarHandler.validate_simulator_day(
                    event.date,
                    calendar_type,
                )
            except ValueError as exc:
                raise ValueError(
                    "Fire event date must be a finite in-domain integer for "
                    f"calendar_type={calendar_type}."
                ) from exc
            direct_distance = abs(day_of_year - event_day)
            circular_distance = min(
                direct_distance,
                days_per_year - direct_distance,
            )
            if circular_distance <= schedule_window_days:
                occurrence_key = self._schedule_occurrence_key(
                    event_index=event_index,
                    event_day=event_day,
                    day_of_year=day_of_year,
                    time=time,
                    days_per_year=days_per_year,
                    seconds_per_day=seconds_per_day,
                )
                if occurrence_key in self._executed_schedule_occurrences:
                    continue
                if event.fire_type == FireType.WILDFIRE:
                    if force_fire:
                        events_today.append((occurrence_key, event))
                    else:
                        # References: Bond and Keeley (2005); RELIC fire management
                        #   theory page (wildfire ignition); Kloeden and Platen (1992),
                        #   ch. 10 (one Functor-owned stream per process).
                        # Mathematical expression: p = s (1 - M) (w / w_ref);
                        #   ignite when u < p, u ~ U(0, 1) from the Functor substream.
                        # Parameter meanings: s the ignition probability scale, M the
                        #   event fuel moisture, w the event wind speed, w_ref the wind
                        #   normalisation, u one uniform draw per unexecuted wildfire
                        #   occurrence in the schedule window.
                        # Units: s, M, p, u [1]; w, w_ref [m s-1].
                        # Domain/constraints: p > 1 always ignites and p <= 0 never; each
                        #   draw advances draw_count, and the stream position is part
                        #   of the module extra state (rollback, checkpoint, CRN).
                        # Sign convention: not applicable (event trigger).
                        fire_probability = (
                            fire_probability_scale
                            * (1.0 - event.fuel_moisture)
                            * (event.wind_speed / wind_speed_norm)
                        )
                        ignition_draw = torch.rand(
                            1,
                            generator=self.generator,
                            device=self.generator.device,
                        )
                        self.draw_count += 1
                        if ignition_draw < fire_probability:
                            events_today.append((occurrence_key, event))
                else:
                    events_today.append((occurrence_key, event))
        return events_today

    def _assemble_fire_reaction_outputs(
        self,
        *,
        combustion_flux: torch.Tensor,
        co2_emission: torch.Tensor,
        co_emission: torch.Tensor,
        fire_ch4_emission: torch.Tensor,
        n2o_emission: torch.Tensor,
        ash_deposition: torch.Tensor,
        carbon_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Assemble fire outputs through the explicit IPHS reaction map.

        References:
          - Seiler and Crutzen (1980), biomass-burning carbon fluxes.
          - Andreae and Merlet (2001), biomass-burning emission factors.
          - Rao and Esposito (2016), reaction-network thermodynamics.
        Mathematical expression:
          y_dot = N_h r_h, where r_h contains combustion, emission, ash,
          and carbon-loss closure rates.
        Parameter meanings:
          combustion_flux: burned fuel or carbon reaction rate [kg m^-2 s^-1].
          emission terms: open atmospheric reaction-port rates [kg m^-2 s^-1].
          ash_deposition and carbon_loss: pedosphere and diagnostic output
          rates from the same combustion reaction closure.
        """

        fire_network = advance_reaction_network(
            name=f"{self.name}_fire_combustion_reaction_port_network",
            species=(
                "combustion_flux",
                "co2_emission",
                "co_emission",
                "fire_ch4_emission",
                "n2o_emission",
                "ash_deposition",
                "carbon_loss",
            ),
            reactions=(
                "fuel_combustion",
                "co2_emission",
                "co_emission",
                "fire_ch4_emission",
                "n2o_emission",
                "ash_partition",
                "carbon_loss",
            ),
            stoichiometric_N=torch.eye(
                7,
                device=combustion_flux.device,
                dtype=combustion_flux.dtype,
            ),
            rates=torch.stack(
                (
                    combustion_flux,
                    co2_emission.to(device=combustion_flux.device, dtype=combustion_flux.dtype),
                    co_emission.to(device=combustion_flux.device, dtype=combustion_flux.dtype),
                    fire_ch4_emission.to(device=combustion_flux.device, dtype=combustion_flux.dtype),
                    n2o_emission.to(device=combustion_flux.device, dtype=combustion_flux.dtype),
                    ash_deposition.to(device=combustion_flux.device, dtype=combustion_flux.dtype),
                    carbon_loss.to(device=combustion_flux.device, dtype=combustion_flux.dtype),
                ),
                dim=0,
            ),
            open_system_route="port",
        )
        return {
            "combustion_flux": fire_network.species_rate("combustion_flux"),
            "co2_emission": fire_network.species_rate("co2_emission"),
            "co_emission": fire_network.species_rate("co_emission"),
            "fire_ch4_emission": fire_network.species_rate("fire_ch4_emission"),
            "n2o_emission": fire_network.species_rate("n2o_emission"),
            "ash_deposition": fire_network.species_rate("ash_deposition"),
            "carbon_loss": fire_network.species_rate("carbon_loss"),
        }

    def _carbon_availability_residual(
        self,
        *,
        co2_emission: torch.Tensor,
        co_emission: torch.Tensor,
        fire_ch4_emission: torch.Tensor,
        carbon_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Return carbon emitted beyond the combusted-carbon boundary.

        References:
          - Seiler and Crutzen (1980), biomass-burning carbon budgets.
          - Andreae and Merlet (2001), dry-matter emission factors.
        Mathematical expression:
          epsilon_C = relu(F_CO2 M_C/M_CO2 + F_CO M_C/M_CO
                           + F_CH4 M_C/M_CH4 - F_C,burned).
        Parameter meanings:
          F_species is species mass flux [kg species m-2 s-1], M_species is
          its SSOT molar mass [g mol-1], and F_C,burned is the open carbon
          boundary flux [kg C m-2 s-1].
        """

        dtype = carbon_loss.dtype
        device = carbon_loss.device

        def _molar_mass(name: str) -> torch.Tensor:
            return torch.as_tensor(
                get_physical_constant(
                    f"relic.physical.constants.molecular_weights.{name}"
                ),
                dtype=dtype,
                device=device,
            )

        carbon_mass = _molar_mass("carbon")
        co2_mass = _molar_mass("co2")
        ch4_mass = _molar_mass("ch4")
        co_mass = carbon_mass + 0.5 * _molar_mass("o2")
        emitted_carbon = (
            co2_emission.to(device=device, dtype=dtype)
            * carbon_mass
            / co2_mass
            + co_emission.to(device=device, dtype=dtype)
            * carbon_mass
            / co_mass
            + fire_ch4_emission.to(device=device, dtype=dtype)
            * carbon_mass
            / ch4_mass
        )
        return torch.relu(emitted_carbon - carbon_loss)

    def forward(
        self,
        state: Dict[str, torch.Tensor],
        time: float,
        dt: float,
        calendar_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calculate fire effects and emissions.

        References:
            Seiler and Crutzen (1980); Andreae and Merlet (2001); CF-1.8.
        Mathematical expression:
            C_burned = B_fuel CC CE (1 - M_factor),
            E_species = C_burned EF_species / (f_C Delta_t_event),
            Delta_t_event = dt of the step that executes the calendar
            occurrence (so E_species dt = C_burned EF_species / f_C), and
            d = floor(t / seconds_per_day) mod Y(C) + 1.
        Parameter meanings:
            B_fuel is available biomass [kg m-2], CC and CE are combustion
            fractions [1], M_factor is moisture limitation [1], EF_species is
            an emission factor [kg species kg-1 dry matter], Delta_t_event is
            the executing step length [s] (each occurrence executes once, so
            hourly, daily, and multi-day steps release the same event mass),
            and C is the explicit simulator calendar.
        """
        # References: Seiler and Crutzen (1980); Andreae and Merlet (2001); Bond and Keeley (2005);
        #   van der Werf et al. (2017); CF-1.8 calendars.
        # Mathematical expression: d = floor(t / seconds_per_day) mod Y(C) + 1; for each
        #   occurrence executing today C_burned = B_fuel CC CE (1 - M_factor) f_burn;
        #   E_s = C_burned EF_s / f_C; rates F = M / dt with Delta_t_event = dt;
        #   x_dot = N_h r_h through advance_reaction_network.
        # Parameter meanings: B_fuel available fuel, CC combustion completeness, CE combustion
        #   efficiency, M_factor moisture limitation, f_burn effective burn fraction, EF_s the
        #   species emission factor, f_C the biomass carbon fraction, dt the executing step.
        # Units: B_fuel, C_burned [kg m-2]; EF_s [kg species kg-1 DM]; emission, ash,
        #   combustion and carbon-loss rates [kg m-2 s-1]; burned_fraction [1];
        #   days_since_fire [day]; event_support_seconds [s].
        # Domain/constraints: each calendar occurrence executes once; rate times dt equals the
        #   event mass for any step length; the calendar must be explicit; wildfire ignition is a
        #   seeded random draw.
        # Sign convention: emissions and fuel carbon loss are positive out of the fuel pool.
        # aboveground_biomass and litter_mass [kg m-2], fuel_moisture [1] and the air
        # temperature [K] are declared inputs with no ParameterManager default: a
        # missing value raises, as in the governed call. Each is checked against its
        # physical domain, the contract bounds: non-negative biomass, litter and fuel
        # moisture (live fuel exceeds 1) and a non-negative absolute temperature; a
        # value outside raises (no clipping).
        aboveground_biomass = self._get_state_variable(
            state,
            'aboveground_biomass',
            validate_range=(0.0, float("inf")),
        )
        litter_mass = self._get_state_variable(
            state,
            'litter_mass',
            validate_range=(0.0, float("inf")),
        )
        temperature = self._get_state_variable(
            state,
            'temperature',
            validate_range=(0.0, float("inf")),
        )
        fuel_moisture = self._get_state_variable(
            state,
            'fuel_moisture',
            validate_range=(0.0, float("inf")),
        )
        calendar_type = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self._configured_calendar_type,
            runtime_calendar=calendar_type,
            owner_name=self.__class__.__name__,
        )
        state_day = state.get("day_of_year")
        runtime_day = kwargs.get("day_of_year")
        if state_day is not None and runtime_day is not None:
            normalized_state_day = CalendarHandler.validate_simulator_day(
                state_day,
                calendar_type,
            )
            normalized_runtime_day = CalendarHandler.validate_simulator_day(
                runtime_day,
                calendar_type,
            )
            if normalized_state_day != normalized_runtime_day:
                raise ValueError(
                    "Conflicting day_of_year values were supplied by state "
                    "and runtime arguments."
                )
        day_input = state_day if state_day is not None else runtime_day
        day_of_year = resolve_day_of_year(
            time,
            day_input,
            calendar_type=calendar_type,
        )
        force_fire = kwargs.get('force_fire', False)
        burned_fraction = torch.zeros_like(aboveground_biomass)
        co2_emission = torch.zeros_like(aboveground_biomass)
        co_emission = torch.zeros_like(aboveground_biomass)
        fire_ch4_emission = torch.zeros_like(aboveground_biomass)
        n2o_emission = torch.zeros_like(aboveground_biomass)
        ash_deposition = torch.zeros_like(aboveground_biomass)
        combustion_flux = torch.zeros_like(aboveground_biomass)
        carbon_loss = torch.zeros_like(aboveground_biomass)
        seconds_per_day = self._seconds_per_day_value()
        dt_tensor = torch.as_tensor(
            dt,
            dtype=aboveground_biomass.dtype,
            device=aboveground_biomass.device,
        )
        if dt_tensor.numel() != 1 or not torch.all(torch.isfinite(dt_tensor)):
            raise ValueError("Fire dt must be one finite scalar.")
        if torch.any(dt_tensor <= 0.0):
            raise ValueError("Fire dt must be positive.")
        # References: RELIC time-space contract (StateManager-owned state); Bond and
        #   Keeley (2005) (time since fire as the fire-regime clock).
        # Mathematical expression: D' = D + dt / t_day when no occurrence executes in
        #   the step, D' = 0 when one does.
        # Parameter meanings: D is days_since_fire read from the state, D' the value
        #   returned as a REPLACE update, dt the step length, t_day the day length.
        # Units: D, D' [day]; dt, t_day [s].
        # Domain/constraints: D is a declared input owned by fire_management, finite
        #   and non-negative; a missing value raises (no initial-value parameter stands
        #   in for it); the Functor holds no copy of it.
        # Sign convention: D' >= 0 counts elapsed days forward.
        days_since_fire = self._get_state_variable(
            state,
            'days_since_fire',
            validate_range=(0.0, float("inf")),
        )
        next_days_since_fire = days_since_fire + dt_tensor.reshape(()) / torch.as_tensor(
            seconds_per_day,
            dtype=days_since_fire.dtype,
            device=days_since_fire.device,
        )
        scheduled_occurrences = self._check_fire_schedule(
            day_of_year,
            calendar_type=calendar_type,
            time=time,
            seconds_per_day=seconds_per_day,
            force_fire=force_fire,
        )
        executed_occurrence_keys: List[str] = []
        for occurrence_key, event in scheduled_occurrences:
            veg_props = self._get_vegetation_fire_properties(event.vegetation_type)
            # The optional overrides fuel_load [kg DM m-2] and burn_fraction [1] are
            # checked against their contract domains (finite; fuel_load >= 0;
            # 0 <= burn_fraction <= 1). A value outside is an input error and raises;
            # it is not clipped.
            if "fuel_load" in state:
                fuel_load = torch.as_tensor(
                    state["fuel_load"],
                    dtype=aboveground_biomass.dtype,
                    device=aboveground_biomass.device,
                )
                if not bool(torch.all(torch.isfinite(fuel_load))) or bool(
                    torch.any(fuel_load < 0.0)
                ):
                    raise ValueError(
                        f"{self.name}: fuel_load [kg DM m-2] must be finite and non-negative."
                    )
            else:
                fuel_load = self._calculate_fuel_load(
                    aboveground_biomass,
                    litter_mass,
                    event.vegetation_type,
                )
            target_moisture = event.fuel_moisture
            if event.fire_type == FireType.WILDFIRE:
                target_moisture = None
            moisture_factor = self._calculate_moisture_factor(fuel_moisture, temperature, event_fuel_moisture=target_moisture)
            combustion_completeness = self._calculate_combustion_completeness(event.intensity, veg_props, moisture_factor)
            if "burn_fraction" in state:
                effective_burn_fraction = torch.as_tensor(
                    state["burn_fraction"],
                    dtype=fuel_load.dtype,
                    device=fuel_load.device,
                )
                if not bool(torch.all(torch.isfinite(effective_burn_fraction))) or bool(
                    torch.any((effective_burn_fraction < 0.0) | (effective_burn_fraction > 1.0))
                ):
                    raise ValueError(
                        f"{self.name}: burn_fraction [1] must be finite and lie in [0, 1]."
                    )
            else:
                effective_burn_fraction = torch.as_tensor(
                    event.burned_fraction * (1.0 - event.suppression_effectiveness),
                    dtype=fuel_load.dtype,
                    device=fuel_load.device,
                )
            # References: Bond and Keeley (2005); van der Werf et al. (2017).
            # Formula: f_i,increment = clamp(f_i, 0, 1) * (1 - f_burn,total).
            # Parameters: f_i is event burn fraction and f_burn,total is the
            # already burned fraction in the current accepted step.
            available_fraction = torch.clamp(1.0 - burned_fraction, min=0.0)
            event_burn_fraction = (
                torch.clamp(effective_burn_fraction, min=0.0, max=1.0)
                * available_fraction
            )
            carbon_content = torch.as_tensor(
                self._safe_fire_param(
                    'biomass_carbon_fraction',
                    namespace='anthroposphere.management.fire',
                ),
                dtype=fuel_load.dtype,
                device=fuel_load.device,
            )
            fuel_combusted = (
                fuel_load * combustion_completeness * event_burn_fraction
            )
            carbon_combusted = fuel_combusted * carbon_content
            emissions = self._calculate_emissions(carbon_combusted, veg_props)
            burned_fraction = burned_fraction + event_burn_fraction
            combustion_flux += fuel_combusted
            carbon_loss += carbon_combusted
            co2_emission += emissions['co2_emission']
            co_emission += emissions['co_emission']
            fire_ch4_emission += emissions['fire_ch4_emission']
            n2o_emission += emissions['n2o_emission']
            ash_fraction = torch.as_tensor(
                self._safe_fire_param(
                    'ash_fraction',
                    namespace='anthroposphere.management.fire',
                ),
                dtype=fuel_load.dtype,
                device=fuel_load.device,
            )
            ash_deposition += (
                fuel_load
                * combustion_completeness
                * event_burn_fraction
                * ash_fraction
            )
            next_days_since_fire = torch.zeros_like(next_days_since_fire)
            executed_occurrence_keys.append(occurrence_key)
        # References: Seiler and Crutzen (1980); Andreae and Merlet (2001); RELIC time-space contract.
        # Mathematical expression: F_s = M_s / Delta_t_event with Delta_t_event = dt of the executing
        #   step, so F_s * dt = M_s exactly; F_s = 0 when no occurrence executes.
        # Parameter meanings: M_s is the event-integrated species, ash, combusted-fuel or carbon mass
        #   accumulated above for the occurrences executing in this step; dt is the step length.
        # Units: M_s [kg m-2] (kg C m-2 for carbon terms); dt, Delta_t_event [s]; F_s [kg m-2 s-1].
        # Domain/constraints: dt > 0 (validated above); each calendar occurrence executes exactly once,
        #   so a fixed daily support would under-integrate sub-daily steps and double-integrate
        #   multi-day steps; the released mass is independent of dt.
        # Sign convention: emissions, ash deposition and fuel carbon loss are positive out of the fuel pool.
        event_support_seconds = torch.zeros(
            (),
            dtype=aboveground_biomass.dtype,
            device=aboveground_biomass.device,
        )
        if scheduled_occurrences:
            event_support_seconds = dt_tensor.reshape(()).clone()
            co2_emission = co2_emission / event_support_seconds
            co_emission = co_emission / event_support_seconds
            fire_ch4_emission = fire_ch4_emission / event_support_seconds
            n2o_emission = n2o_emission / event_support_seconds
            ash_deposition = ash_deposition / event_support_seconds
            combustion_flux = combustion_flux / event_support_seconds
            carbon_loss = carbon_loss / event_support_seconds
        # References: Seiler and Crutzen (1980); Andreae and Merlet (2001).
        # Formula: y_dot = N_h r_h for combustion, atmospheric emissions, ash, and carbon loss.
        # Parameters: r_h entries use ParameterManager-backed combustion and emission coefficients.
        reaction_outputs = self._assemble_fire_reaction_outputs(
            combustion_flux=combustion_flux,
            co2_emission=co2_emission,
            co_emission=co_emission,
            fire_ch4_emission=fire_ch4_emission,
            n2o_emission=n2o_emission,
            ash_deposition=ash_deposition,
            carbon_loss=carbon_loss,
        )
        carbon_residual = self._carbon_availability_residual(
            co2_emission=reaction_outputs["co2_emission"],
            co_emission=reaction_outputs["co_emission"],
            fire_ch4_emission=reaction_outputs["fire_ch4_emission"],
            carbon_loss=reaction_outputs["carbon_loss"],
        )
        # Schedule bookkeeping (module extra state, snapshotted with the accepted
        # step): an executed occurrence does not execute again.
        self._executed_schedule_occurrences.update(executed_occurrence_keys)

        raw_outputs = {
            'burned_fraction': burned_fraction,
            'co2_emission': reaction_outputs["co2_emission"],
            'co_emission': reaction_outputs["co_emission"],
            'fire_ch4_emission': reaction_outputs["fire_ch4_emission"],
            'n2o_emission': reaction_outputs["n2o_emission"],
            'ash_deposition': reaction_outputs["ash_deposition"],
            'days_since_fire': next_days_since_fire,
            'fire_fuel_carbon_input': reaction_outputs["carbon_loss"].clone(),
            'carbon_loss': reaction_outputs["carbon_loss"],
            'combustion_flux': reaction_outputs["combustion_flux"],
            'diagnostics.fire.events_today': torch.as_tensor(
                len(scheduled_occurrences),
                dtype=aboveground_biomass.dtype,
                device=aboveground_biomass.device,
            ),
            'diagnostics.fire.event_support_seconds': event_support_seconds,
            'diagnostics.fire.carbon_availability_residual': carbon_residual,
        }
        raw_outputs = self._attach_canonical_runtime_diagnostic(raw_outputs)
        units = {
            'burned_fraction': '1',
            'co2_emission': 'kg m-2 s-1',
            'co_emission': 'kg m-2 s-1',
            'fire_ch4_emission': 'kg m-2 s-1',
            'n2o_emission': 'kg m-2 s-1',
            'ash_deposition': 'kg m-2 s-1',
            'days_since_fire': 'day',
            'fire_fuel_carbon_input': 'kg C m-2 s-1',
            'carbon_loss': 'kg C m-2 s-1',
            'combustion_flux': 'kg m-2 s-1',
            'diagnostics.fire.events_today': '1',
            'diagnostics.fire.event_support_seconds': 's',
            'diagnostics.fire.carbon_availability_residual': 'kg C m-2 s-1',
            'diagnostics.fire_management.canonical_iphs_runtime': '1',
        }
        carbon_fields = {'fire_fuel_carbon_input', 'carbon_loss'}
        return {
            name: UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit=units[name],
                element="carbon" if name in carbon_fields else None,
                source_functor=self.name,
            )
            for name, value in raw_outputs.items()
        }

    def check_conservation_laws(
        self,
        input_state: Dict[str, torch.Tensor],
        output_updates: Dict[str, Any],
    ) -> Dict[str, float]:
        """Return the executable fire carbon-availability residual.

        References:
          - Seiler and Crutzen (1980), biomass-burning carbon budgets.
        Mathematical expression:
          epsilon_C = max(0, F_C,emitted - F_C,burned).
        Parameter meanings:
          F_C,emitted is carbon in CO2, CO, and CH4 [kg C m-2 s-1], and
          F_C,burned is the declared open carbon-loss port [kg C m-2 s-1].
        """

        del input_state
        residual_key = "diagnostics.fire.carbon_availability_residual"
        if residual_key not in output_updates:
            raise KeyError(f"Missing conservation diagnostic '{residual_key}'")
        residual = output_updates[residual_key]
        if isinstance(residual, UpdateItem):
            residual = residual.value
        residual_tensor = torch.as_tensor(residual)
        if residual_tensor.numel() == 0 or not torch.all(
            torch.isfinite(residual_tensor)
        ):
            raise ValueError(
                "Fire conservation residual must be finite and non-empty."
            )
        magnitude = torch.max(torch.abs(residual_tensor))
        return {"mass": self._scalar_from_value(magnitude)}

