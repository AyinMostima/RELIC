"""Canonical fertilization management boundary.

References:
    - FAO (2000). Fertilizers and Their Use, 4th edition.
    - E3SM/ELM Flow of Agriculture Nitrogen design documentation.
    - CLM/CTSM FERT_TO_SMINN and NFERTILIZATION history fields.
    - BIPM (2019). The International System of Units, 9th edition.

Mathematical expressions:
    M_N = A_product * f_N
    chi_P = 2 M_P / (2 M_P + 2.5 M_O2)
    M_P = A_product * f_P2O5 * chi_P
    Delta M_X = sum_e M_X,e * overlap_e / duration_e
    F_X = Delta M_X / dt

Parameter meanings:
    A_product is fertilizer product mass [kg product m-2].
    f_N is elemental nitrogen mass fraction [kg N kg-product-1].
    f_P2O5 is labelled P2O5 mass fraction [kg P2O5 kg-product-1].
    overlap_e is timestep overlap with the event window [s].
    duration_e is the declared application duration [s].

Open-system route:
    Nitrogen leaves the anthroposphere through fertilizer_n_flux and is routed
    by the existing nitrogen-cycling owner. Phosphorus enters through
    fertilizer_p_flux and the matching available_phosphorus increment in the
    same StateManager transaction. This module owns no soil reaction kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.coupling.scales.temporal.calendar import CalendarHandler
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    TransportMechanismFunctor,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant

from .event_scheduler import resolve_day_of_year


class FertilizerType(Enum):
    """Configured fertilizer product types."""

    UREA = "urea"
    AMMONIUM_NITRATE = "ammonium_nitrate"
    AMMONIUM_SULFATE = "ammonium_sulfate"
    SUPERPHOSPHATE = "superphosphate"
    POTASH = "potash"
    NPK_COMPOUND = "npk_compound"
    ORGANIC = "organic"
    SLOW_RELEASE = "slow_release"


class ApplicationTiming(Enum):
    """Descriptive timing labels for management configuration."""

    PLANTING = "planting"
    SIDE_DRESS = "side_dress"
    SPLIT_APPLICATION = "split"
    FALL = "fall"
    SPRING = "spring"
    FOLIAR = "foliar"


class ApplicationMethod(Enum):
    """Descriptive application methods without hidden efficiency factors."""

    BROADCAST = "broadcast"
    BAND = "band"
    INCORPORATION = "incorporation"
    INJECTION = "injection"
    FERTIGATION = "fertigation"
    FOLIAR_SPRAY = "foliar_spray"


@dataclass(frozen=True)
class FertilizerEvent:
    """One recurring annual fertilizer application window.

    Parameters:
        fertilizer_type: Product composition selected from ParameterManager.
        product_mass: Product mass applied per event [kg product m-2].
        application_date: Calendar day of year [day].
        application_duration: Optional application window [s]. A missing value
            uses the physical seconds-per-day constant.
        method: Descriptive management method. It does not silently discard
            mass through an empirical efficiency factor.
    """

    fertilizer_type: FertilizerType
    product_mass: float | torch.Tensor
    application_date: int
    application_duration: Optional[float] = None
    method: ApplicationMethod = ApplicationMethod.BROADCAST


class FertilizationManagement(TransportMechanismFunctor):
    """Map scheduled product mass to canonical nitrogen and phosphorus ports."""

    TRANSPORT_TYPE = "managed_application"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    PARAMETER_NAMESPACE = "anthroposphere.management.fertilization"

    DOMAIN_PARAMETER_KEYS = (
        "anthroposphere.management.fertilization.spring_urea_product_mass",
        "anthroposphere.management.fertilization.spring_day",
        "anthroposphere.management.fertilization.side_dress_ammonium_nitrate_product_mass",
        "anthroposphere.management.fertilization.side_dress_day",
        "anthroposphere.management.fertilization.spring_superphosphate_product_mass",
        "anthroposphere.management.fertilization.phosphorus_day",
        "anthroposphere.management.fertilization.urea.n_content",
        "anthroposphere.management.fertilization.ammonium_nitrate.n_content",
        "anthroposphere.management.fertilization.ammonium_sulfate.n_content",
        "anthroposphere.management.fertilization.superphosphate.p2o5_mass_fraction",
    )

    PHYSICAL_CONSTANT_KEYS = (
        "relic.physical.constants.time.seconds_per_day",
        "relic.physical.constants.conversions.kg_to_g",
        "relic.physical.constants.molecular_weights.phosphorus",
        "relic.physical.constants.molecular_weights.o2",
    )

    _COMPOSITION_PARAMETER_KEYS = {
        FertilizerType.UREA: ("urea.n_content", None),
        FertilizerType.AMMONIUM_NITRATE: (
            "ammonium_nitrate.n_content",
            None,
        ),
        FertilizerType.AMMONIUM_SULFATE: (
            "ammonium_sulfate.n_content",
            None,
        ),
        FertilizerType.SUPERPHOSPHATE: (
            None,
            "superphosphate.p2o5_mass_fraction",
        ),
    }

    def __init__(
        self,
        name: str = "fertilization_management",
        config: Optional[Dict[str, Any]] = None,
        events: Optional[Sequence[FertilizerEvent]] = None,
    ) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="2.0.0",
            description=(
                "Managed elemental nitrogen and phosphorus boundary transfer"
            ),
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "Delta M_N = sum_e(A_e f_N,e omega_e/tau_e); "
                "F_N = Delta M_N/dt; "
                "Delta P_available = F_P dt"
            ),
            references=[
                "FAO (2000) Fertilizers and Their Use, 4th edition",
                "E3SM/ELM Flow of Agriculture Nitrogen design documentation",
                "CLM/CTSM FERT_TO_SMINN and NFERTILIZATION history fields",
                "BIPM (2019) SI Brochure, 9th edition",
                "NIST and IUPAC standard atomic weights",
            ],
            parameters={key: "Required ParameterManager entry" for key in self.DOMAIN_PARAMETER_KEYS},
            input_variables=["day_of_year"],
            output_variables=[
                "fertilizer_n_flux",
                "fertilizer_p_flux",
                "available_phosphorus",
                "diagnostics.fertilization.nitrogen_boundary_residual",
                "diagnostics.fertilization.phosphorus_boundary_residual",
                "diagnostics.fertilization.canonical_iphs_runtime",
            ],
            conservation_laws=["nitrogen_boundary", "phosphorus_boundary"],
            iphs_terms=(
                "Delta M_N = sum_e(A_e f_N,e omega_e/tau_e)",
                "F_N = Delta M_N/dt",
                "Delta P_available = F_P dt",
            ),
            structural_role=("managed_matter_boundary",),
            spatial_dimensions=2,
            temporal_scale=TemporalScale.DAY,
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            coupling_interfaces=["pedosphere"],
        )
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
        )
        self.parameter_namespace = self.PARAMETER_NAMESPACE
        self.runtime_path = "canonical_iphs_fertilization_boundary_port"
        self.optional_input_variables = {"day_of_year"}
        configured_calendar = (config or {}).get("calendar_type")
        if configured_calendar is None:
            self._configured_calendar_type = None
        else:
            self._configured_calendar_type = str(configured_calendar)
            CalendarHandler.days_per_simulator_year(
                self._configured_calendar_type
            )

        self._seconds_per_day = float(get_physical_constant("seconds_per_day"))
        self._kg_to_g = float(
            get_physical_constant(
                "relic.physical.constants.conversions.kg_to_g"
            )
        )
        self._molar_mass_p = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.phosphorus"
            )
        )
        self._molar_mass_o2 = float(
            get_physical_constant(
                "relic.physical.constants.molecular_weights.o2"
            )
        )

        configured_events = (
            tuple(self._build_default_schedule())
            if events is None
            else tuple(events)
        )
        self._validate_schedule(
            configured_events,
            calendar_type=self._configured_calendar_type,
        )
        self.fertilizer_events = configured_events

    def required_parameters(self) -> tuple[str, ...]:
        """Return exact domain ParameterManager keys used by this process."""

        return self.DOMAIN_PARAMETER_KEYS

    def physical_constant_keys(self) -> tuple[str, ...]:
        """Return physical constants kept outside the domain parameter map."""

        return self.PHYSICAL_CONSTANT_KEYS

    @staticmethod
    def _scalar(value: Any, *, name: str) -> float:
        """Convert one scalar-like value without accepting arrays silently."""

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{name} must be scalar")
            value = value.detach().reshape(-1)[0].tolist()
        result = float(value)
        if not bool(torch.isfinite(torch.as_tensor(result)).item()):
            raise ValueError(f"{name} must be finite")
        return result

    def _parameter_scalar(self, name: str) -> float:
        value = self.get_parameter_value(
            name,
            namespace=self.PARAMETER_NAMESPACE,
            source_order=("config", "manager"),
            required=True,
        )
        return self._scalar(value, name=name)

    @staticmethod
    def _positive_integral_day(value: Any, *, name: str) -> int:
        """Validate event schema without inferring a simulator calendar."""

        message = f"{name} must be a finite positive integer numeric scalar"
        if isinstance(value, bool):
            raise ValueError(message)
        try:
            scalar = torch.as_tensor(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(message) from exc
        if scalar.numel() != 1 or scalar.dtype == torch.bool or scalar.is_complex():
            raise ValueError(message)
        scalar = scalar.detach().reshape(()).to(dtype=torch.float64)
        valid = torch.logical_and(
            torch.isfinite(scalar),
            torch.logical_and(
                torch.eq(scalar, torch.trunc(scalar)),
                torch.ge(scalar, 1.0),
            ),
        )
        if not bool(valid.item()):
            raise ValueError(message)
        return int(scalar.item())

    def _parameter_day(self, name: str) -> int:
        return self._positive_integral_day(
            self._parameter_scalar(name),
            name=name,
        )

    def _build_default_schedule(self) -> tuple[FertilizerEvent, ...]:
        """Build the three ParameterManager-backed mineral fertilizer events."""

        return (
            FertilizerEvent(
                fertilizer_type=FertilizerType.UREA,
                product_mass=self._parameter_scalar(
                    "spring_urea_product_mass"
                ),
                application_date=self._parameter_day("spring_day"),
                method=ApplicationMethod.BROADCAST,
            ),
            FertilizerEvent(
                fertilizer_type=FertilizerType.AMMONIUM_NITRATE,
                product_mass=self._parameter_scalar(
                    "side_dress_ammonium_nitrate_product_mass"
                ),
                application_date=self._parameter_day("side_dress_day"),
                method=ApplicationMethod.BAND,
            ),
            FertilizerEvent(
                fertilizer_type=FertilizerType.SUPERPHOSPHATE,
                product_mass=self._parameter_scalar(
                    "spring_superphosphate_product_mass"
                ),
                application_date=self._parameter_day("phosphorus_day"),
                method=ApplicationMethod.INCORPORATION,
            ),
        )

    def _event_duration(self, event: FertilizerEvent) -> float:
        # References: FAO (2000); CF Conventions 1.8 calendar coordinates.
        # Mathematical expression: tau_e = t_day if tau_config is None, otherwise tau_config.
        # Parameter meanings: tau_e is event duration and t_day is the physical day length.
        # Units: tau_e and t_day [s].
        # Domain/constraints: tau_e is finite and 0 < tau_e <= t_day.
        # Sign convention: duration is a strictly positive elapsed-time magnitude.
        duration = (
            self._seconds_per_day
            if event.application_duration is None
            else self._scalar(
                event.application_duration,
                name="application_duration",
            )
        )
        if duration <= 0.0 or duration > self._seconds_per_day:
            raise ValueError(
                "application_duration must be positive and no longer than one day"
            )
        return duration

    def _validate_event(
        self,
        event: FertilizerEvent,
        *,
        calendar_type: Optional[str] = None,
    ) -> None:
        if not isinstance(event, FertilizerEvent):
            raise TypeError("fertilizer schedule entries must be FertilizerEvent")
        try:
            product_mass = self._scalar(event.product_mass, name="product_mass")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "product_mass must be finite and positive"
            ) from exc
        if product_mass <= 0.0:
            raise ValueError("product_mass must be finite and positive")
        application_date = self._positive_integral_day(
            event.application_date,
            name="application_date",
        )
        if calendar_type is not None:
            days_per_year = CalendarHandler.days_per_simulator_year(
                calendar_type
            )
            try:
                CalendarHandler.validate_simulator_day(
                    application_date,
                    calendar_type,
                )
            except ValueError as exc:
                raise ValueError(
                    f"application_date must be in [1, {days_per_year}]"
                ) from exc
        if not isinstance(event.method, ApplicationMethod):
            raise TypeError("method must be an ApplicationMethod")
        self._event_duration(event)
        self.get_fertilizer_properties(event.fertilizer_type)

    def _validate_schedule(
        self,
        events: Sequence[FertilizerEvent],
        *,
        calendar_type: Optional[str] = None,
    ) -> None:
        for event in events:
            self._validate_event(event, calendar_type=calendar_type)

    def add_fertilizer_event(self, event: FertilizerEvent) -> None:
        """Add one validated event before simulator execution."""

        self._validate_event(
            event,
            calendar_type=self._configured_calendar_type,
        )
        self.fertilizer_events = (*self.fertilizer_events, event)

    def _p2o5_to_elemental_p_fraction(self) -> float:
        """Convert P2O5 mass to elemental phosphorus mass.

        References:
            - NIST and IUPAC standard atomic weights.
        Mathematical expression:
            chi_P = 2 M_P / (2 M_P + 2.5 M_O2).
        Parameter meanings:
            M_P is phosphorus molar mass [g mol-1].
            M_O2 is molecular oxygen molar mass [g mol-1].
        Units and route:
            chi_P is dimensionless and preserves phosphorus mass basis.
        """
        # References: NIST and IUPAC standard atomic weights; BIPM (2019).
        # Mathematical expression: chi_P = 2 M_P / (2 M_P + 2.5 M_O2).
        # Parameter meanings: M_P and M_O2 are canonical phosphorus and oxygen molar masses.
        # Units: M_P and M_O2 [g mol-1]; chi_P [1].
        # Domain/constraints: molar masses are finite and positive, so the denominator is positive.
        # Sign convention: chi_P > 0 converts product P2O5 mass to positive elemental-P mass.

        numerator = 2.0 * self._molar_mass_p
        denominator = numerator + 2.5 * self._molar_mass_o2
        return numerator / denominator

    def get_fertilizer_properties(
        self,
        fertilizer_type: FertilizerType,
    ) -> Dict[str, float]:
        """Resolve elemental product composition from ParameterManager."""
        # References: FAO (2000); NIST and IUPAC standard atomic weights.
        # Mathematical expression: f_N=PM[type.n_content]; f_P=f_P2O5 chi_P.
        # Parameter meanings: f_N/f_P2O5 are product fractions and chi_P converts P2O5 to P.
        # Units: f_N, f_P2O5, chi_P, and elemental f_P [1].
        # Domain/constraints: product type has a canonical route and returned fractions lie in [0,1].
        # Sign convention: fractions are non-negative mass shares and assign no boundary direction.

        if fertilizer_type not in self._COMPOSITION_PARAMETER_KEYS:
            name = (
                fertilizer_type.value
                if isinstance(fertilizer_type, FertilizerType)
                else str(fertilizer_type)
            )
            raise ValueError(
                f"Fertilizer product '{name}' has no complete canonical elemental route"
            )

        nitrogen_key, p2o5_key = self._COMPOSITION_PARAMETER_KEYS[
            fertilizer_type
        ]
        nitrogen_fraction = (
            0.0
            if nitrogen_key is None
            else self._parameter_scalar(nitrogen_key)
        )
        p2o5_fraction = (
            0.0 if p2o5_key is None else self._parameter_scalar(p2o5_key)
        )
        elemental_p_fraction = (
            p2o5_fraction * self._p2o5_to_elemental_p_fraction()
        )

        for name, value in (
            ("nitrogen_mass_fraction", nitrogen_fraction),
            ("elemental_phosphorus_mass_fraction", elemental_p_fraction),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        return {
            "nitrogen_mass_fraction": nitrogen_fraction,
            "elemental_phosphorus_mass_fraction": elemental_p_fraction,
        }

    def _event_overlap_seconds(
        self,
        event: FertilizerEvent,
        *,
        time: float,
        dt: float,
        explicit_day: Optional[int],
        calendar_type: str,
    ) -> float:
        """Return exact overlap of a step and one annual event window.

        References:
            - CF-1.8 calendar coordinates.
        Mathematical expression:
            omega = max(0, min(t+dt, b) - max(t, a)).
        Parameter meanings:
            [a,b) is the recurring event window [s].
            [t,t+dt) is the simulator step [s].
        Units and route:
            omega is seconds and only scales the declared boundary mass.
        """
        # References: CF Conventions 1.8; RELIC explicit-calendar contract.
        # Mathematical expression: omega=max(0,min(t+dt,b)-max(t,a)) over recurring [a,b).
        # Parameter meanings: t/dt define the step, a is event start, b=a+tau, and omega overlap.
        # Units: t, dt, a, b, tau, and omega [s].
        # Domain/constraints: dt > 0, 0 < tau <= t_day, and event day is valid for the calendar.
        # Sign convention: omega >= 0 measures forward-time overlap and never reverses application.

        duration = self._event_duration(event)
        if explicit_day is not None:
            if explicit_day != event.application_date:
                return 0.0
            phase = time % self._seconds_per_day
            return max(
                0.0,
                min(phase + dt, duration) - max(phase, 0.0),
            )

        step_start = time
        step_end = time + dt
        event_offset = (event.application_date - 1) * self._seconds_per_day
        seconds_per_year = (
            CalendarHandler.days_per_simulator_year(calendar_type)
            * self._seconds_per_day
        )
        step_bounds = torch.as_tensor(
            (step_start, step_end),
            dtype=torch.float64,
        )
        year_indices = torch.div(
            step_bounds,
            seconds_per_year,
            rounding_mode="floor",
        ).to(dtype=torch.int64)
        first_year = int(year_indices[0].item()) - 1
        last_year = int(year_indices[1].item()) + 1
        overlap = 0.0
        for year_index in range(first_year, last_year + 1):
            event_start = year_index * seconds_per_year + event_offset
            event_end = event_start + duration
            overlap += max(
                0.0,
                min(step_end, event_end) - max(step_start, event_start),
            )
        return overlap

    def _event_element_masses(
        self,
        event: FertilizerEvent,
        overlap: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return applied elemental N and P mass for one timestep.

        References:
            - FAO (2000), fertilizer product composition.
        Mathematical expression:
            Delta M_X = A_product f_X overlap / duration.
        Parameter meanings:
            A_product is product mass [kg product m-2].
            f_X is elemental mass fraction [kg X kg-product-1].
        Units and route:
            Returned values are kg N m-2 and kg P m-2 boundary masses.
        """
        # References: FAO (2000); BIPM (2019) SI mass conventions.
        # Mathematical expression: Delta M_X = A_product f_X omega / tau.
        # Parameter meanings: A_product is product mass, f_X elemental fraction, and omega/tau active share.
        # Units: A_product [kg product m-2]; Delta M_N [kg N m-2]; Delta M_P [kg P m-2].
        # Domain/constraints: A_product > 0, 0 <= omega <= tau, and 0 <= f_X <= 1.
        # Sign convention: positive mass enters the pedosphere through the declared matter port.

        properties = self.get_fertilizer_properties(event.fertilizer_type)
        active_fraction = overlap / self._event_duration(event)
        product_mass = torch.as_tensor(
            event.product_mass,
            device=self.device,
        )
        return (
            product_mass
            * properties["nitrogen_mass_fraction"]
            * active_fraction,
            product_mass
            * properties["elemental_phosphorus_mass_fraction"]
            * active_fraction,
        )

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare the two typed matter-boundary flow cochains."""

        return (
            CochainSpaceSpec(
                complex_id="anthroposphere_fertilization_boundary",
                degree=0,
                primality="dual",
                n_dofs=1,
                units="kg N m^-2 s^-1",
                extensivity="density",
                physical_domain="anthroposphere_pedosphere_interface",
                semantic_name="fertilization_nitrogen_boundary_flow",
                orientation_convention="positive_into_pedosphere",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="anthroposphere_fertilization_boundary",
                degree=0,
                primality="dual",
                n_dofs=1,
                units="g P m^-2 s^-1",
                extensivity="density",
                physical_domain="anthroposphere_pedosphere_interface",
                semantic_name="fertilization_phosphorus_boundary_flow",
                orientation_convention="positive_into_pedosphere",
                boundary_role="port",
            ),
        )

    @staticmethod
    def _effort_space(semantic_name: str) -> CochainSpaceSpec:
        return CochainSpaceSpec(
            complex_id="anthroposphere_fertilization_boundary",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="1",
            extensivity="intensive",
            physical_domain="anthroposphere_pedosphere_interface",
            semantic_name=semantic_name,
            orientation_convention="positive_application_potential",
            boundary_role="port",
        )

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare nitrogen and phosphorus matter input ports."""

        n_flow, p_flow = self.declared_forms()
        return (
            PortSpecIPHS(
                name="fertilization_nitrogen_input_port",
                effort_var="fertilization_nitrogen_application_potential",
                flow_var="fertilizer_n_flux",
                effort_space=self._effort_space(
                    "fertilization_nitrogen_application_potential"
                ),
                flow_space=n_flow,
                power_units="kg N m^-2 s^-1",
                positive_power_convention="into_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="fertilization_phosphorus_input_port",
                effort_var="fertilization_phosphorus_application_potential",
                flow_var="fertilizer_p_flux",
                effort_space=self._effort_space(
                    "fertilization_phosphorus_application_potential"
                ),
                flow_space=p_flow,
                power_units="g P m^-2 s^-1",
                positive_power_convention="into_system",
                port_kind="matter",
            ),
        )

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[()]:
        """Return no Hodge term because a boundary schedule is not a reaction."""

        del complex_bundle, resolved_parameters
        return ()

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no internal energy storage for the management boundary."""

        del state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no entropy claim for the non-strict boundary wrapper."""

        del state
        return {}

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Expose only the real port interconnection and no internal J operator."""
        # References: van der Schaft and Maschke (1995); RELIC fertilization certificate.
        # Mathematical expression: x_internal=empty and J=0_(0x0); x_dot=G_h u_fertilizer.
        # Parameter meanings: G_h is the declared port map and u_fertilizer is scheduled input.
        # Units: empty J has no internal unit; N/P port flows retain their declared flux units.
        # Domain/constraints: no internal state or reversible operator is claimed for this boundary.
        # Sign convention: positive port flow is into the pedosphere system.

        del state, complex_bundle
        empty = torch.zeros((0, 0), device=self.device)
        return {"state_vars": (), "J": empty, "ports": self.ports()}

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Expose no internal mobility for the non-reaction boundary."""
        # References: Ramirez and Le Gorrec (2022); RELIC fertilization certificate.
        # Mathematical expression: x_internal=empty and M=0_(0x0).
        # Parameter meanings: M would be internal mobility; only external scheduled ports are present.
        # Units: empty M has no internal unit; connected N/P flows retain declared flux units.
        # Domain/constraints: no reaction, dissipation, or entropy-production claim is introduced.
        # Sign convention: the zero operator adds no signed internal source or sink.

        del state, complex_bundle
        empty = torch.zeros((0, 0), device=self.device)
        return {"state_vars": (), "M": empty, "ports": self.ports()}

    def forward(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
        calendar_type: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Apply scheduled elemental mass through the canonical boundary ports.

        References:
            - FAO (2000), fertilizer application and composition.
            - CLM/ELM fertilizer-to-mineral-nitrogen boundary convention.
        Mathematical expressions:
            Delta M_X = sum_e A_e f_X,e omega_e / tau_e.
            F_X = Delta M_X / dt.
            Delta P_available = F_P dt.
        Parameter meanings:
            dt is timestep [s], A_e is product mass [kg product m-2], and
            omega_e/tau_e is the active fraction of event e [1].
        Units and open-system route:
            fertilizer_n_flux is kg N m-2 s-1 to nitrogen cycling.
            fertilizer_p_flux is g P m-2 s-1 and is paired with the integrated
            available_phosphorus increment in the phosphorus.total ledger.
        """

        # References: FAO (2000); CLM/ELM fertilizer boundary convention.
        # Mathematical expression: DeltaM_X=sum_e(A_e f_X,e omega_e/tau_e); F_X=DeltaM_X/dt.
        # Parameter meanings: A_e is product mass, f_X,e element fraction, and omega_e overlap.
        # Units: F_N [kg N m-2 s-1]; F_P [g P m-2 s-1]; DeltaP [g P m-2].
        # Domain/constraints: time >= 0, dt > 0, event windows are valid, and fractions lie in [0,1].
        # Sign convention: N/P fluxes are positive into soil and available-P accumulation is positive.
        time_value = self._scalar(time, name="time")
        if time_value < 0.0:
            raise ValueError("time must be finite and non-negative")
        try:
            dt_value = self._scalar(dt, name="dt")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("dt must be finite and positive") from exc
        if dt_value <= 0.0:
            raise ValueError("dt must be finite and positive")

        calendar_type = CalendarHandler.resolve_simulator_calendar(
            configured_calendar=self._configured_calendar_type,
            runtime_calendar=calendar_type,
            owner_name=self.__class__.__name__,
        )
        self._validate_schedule(
            self.fertilizer_events,
            calendar_type=calendar_type,
        )

        state_day = state["day_of_year"] if "day_of_year" in state else None
        keyword_day = kwargs["day_of_year"] if "day_of_year" in kwargs else None
        if state_day is not None and keyword_day is not None:
            state_day_value = resolve_day_of_year(
                time_value,
                state_day,
                calendar_type=calendar_type,
            )
            keyword_day_value = resolve_day_of_year(
                time_value,
                keyword_day,
                calendar_type=calendar_type,
            )
            if state_day_value != keyword_day_value:
                raise ValueError("Conflicting day_of_year inputs")
        day_input = state_day if state_day is not None else keyword_day
        explicit_day = (
            None
            if day_input is None
            else resolve_day_of_year(
                time_value,
                day_input,
                calendar_type=calendar_type,
            )
        )

        default_dtype = torch.get_default_dtype()
        nitrogen_mass_kg = torch.zeros(
            (),
            dtype=default_dtype,
            device=self.device,
        )
        phosphorus_mass_kg = torch.zeros(
            (),
            dtype=default_dtype,
            device=self.device,
        )
        for event in self.fertilizer_events:
            overlap = self._event_overlap_seconds(
                event,
                time=time_value,
                dt=dt_value,
                explicit_day=explicit_day,
                calendar_type=calendar_type,
            )
            event_n, event_p = self._event_element_masses(event, overlap)
            nitrogen_mass_kg = nitrogen_mass_kg + event_n
            phosphorus_mass_kg = phosphorus_mass_kg + event_p

        nitrogen_mass = nitrogen_mass_kg
        phosphorus_mass = phosphorus_mass_kg * self._kg_to_g
        nitrogen_dt = torch.as_tensor(
            dt_value,
            dtype=nitrogen_mass.dtype,
            device=self.device,
        )
        phosphorus_dt = torch.as_tensor(
            dt_value,
            dtype=phosphorus_mass.dtype,
            device=self.device,
        )
        nitrogen_flux = nitrogen_mass / nitrogen_dt
        phosphorus_flux = phosphorus_mass / phosphorus_dt
        nitrogen_residual = nitrogen_mass - nitrogen_flux * nitrogen_dt
        phosphorus_residual = phosphorus_mass - phosphorus_flux * phosphorus_dt

        return {
            "fertilizer_n_flux": UpdateItem(
                value=nitrogen_flux,
                mode=UpdateMode.REPLACE,
                unit="kg N m-2 s-1",
                per_timestep=False,
                element=None,
                source_functor=self.name,
            ),
            "fertilizer_p_flux": UpdateItem(
                value=phosphorus_flux,
                mode=UpdateMode.REPLACE,
                unit="g P m-2 s-1",
                per_timestep=False,
                element=None,
                source_functor=self.name,
            ),
            "available_phosphorus": UpdateItem(
                value=phosphorus_mass,
                mode=UpdateMode.ACCUMULATE,
                unit="g P m-2",
                per_timestep=True,
                element="P",
                source_functor=self.name,
            ),
            "diagnostics.fertilization.nitrogen_boundary_residual": nitrogen_residual,
            "diagnostics.fertilization.phosphorus_boundary_residual": phosphorus_residual,
            "diagnostics.fertilization.canonical_iphs_runtime": torch.ones_like(
                nitrogen_mass
            ),
        }

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Dict[str, Any]:
        """Execute the single canonical forward path."""

        return self.forward(state, time, dt)

    def check_conservation_laws(
        self,
        input_state: Mapping[str, Any],
        output_updates: Mapping[str, Any],
    ) -> Dict[str, float]:
        """Return exact local N and P boundary residual magnitudes."""
        # References: CLM/ELM fertilizer boundary convention; BIPM (2019).
        # Mathematical expression: R_N=DeltaM_N-F_N dt; R_P=DeltaM_P-F_P dt; e_X=max|R_X|.
        # Parameter meanings: DeltaM_X is applied step mass, F_X flux, dt step length, and e_X error.
        # Units: e_N [kg N m-2] and e_P [g P m-2].
        # Domain/constraints: both canonical diagnostics are present, finite, and non-empty.
        # Sign convention: raw positive residual is mass minus integrated flux; e_X is non-negative.

        del input_state
        result: Dict[str, float] = {}
        for diagnostic, label in (
            (
                "diagnostics.fertilization.nitrogen_boundary_residual",
                "nitrogen_boundary",
            ),
            (
                "diagnostics.fertilization.phosphorus_boundary_residual",
                "phosphorus_boundary",
            ),
        ):
            if diagnostic not in output_updates:
                raise KeyError(f"Missing conservation diagnostic '{diagnostic}'")
            value = output_updates[diagnostic]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Conservation diagnostic '{diagnostic}' must be a Tensor"
                )
            if value.numel() == 0:
                raise ValueError(
                    f"Conservation diagnostic '{diagnostic}' must be non-empty"
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"Conservation diagnostic '{diagnostic}' must be floating point"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(
                    f"Conservation diagnostic '{diagnostic}' must be finite"
                )
            result[label] = float(value.abs().amax().detach().item())
        return result


__all__ = [
    "ApplicationMethod",
    "ApplicationTiming",
    "FertilizationManagement",
    "FertilizerEvent",
    "FertilizerType",
]
